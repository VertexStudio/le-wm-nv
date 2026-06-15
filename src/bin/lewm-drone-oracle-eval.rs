use std::{
    fs,
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, ensure};
use candle::{D, DType, IndexOp, Tensor};
use clap::Parser;
use le_wm_nv::{
    checkpoint,
    data::drone_racing::{
        DRONE_ACTION_DIM, DRONE_OBSERVATION_DIM, DroneBatchConfig, DroneNormalization,
        DroneRacingDataset, RunningStats, shuffle,
    },
    models::world_model::{ObservationEncoderConfig, WorldModel, WorldModelConfig},
    runtime::{DTypeSpec, DeviceSpec},
};
use serde::Serialize;

#[derive(Parser, Debug)]
struct Args {
    /// Run directory containing final.safetensors, model-config.json, and normalization.json.
    #[arg(long)]
    model_dir: PathBuf,

    /// Imported drone dataset directory containing data.h5 and metadata.json.
    #[arg(long)]
    dataset_dir: Option<PathBuf>,

    /// Override checkpoint path. Defaults to <model-dir>/final.safetensors.
    #[arg(long)]
    weights: Option<PathBuf>,

    /// Override model config path. Defaults to <model-dir>/model-config.json.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Override normalization path. Defaults to <model-dir>/normalization.json.
    #[arg(long)]
    normalization: Option<PathBuf>,

    #[arg(long, default_value_t = DeviceSpec::Cuda(0))]
    device: DeviceSpec,

    #[arg(long, default_value_t = DTypeSpec::F32)]
    dtype: DTypeSpec,

    /// Comma-separated rollout horizons in dataset steps.
    #[arg(long, default_value = "1,5,10,25,50")]
    horizons: String,

    /// Number of valid windows to evaluate. Use 0 for all valid rows.
    #[arg(long, default_value_t = 1024)]
    rows: usize,

    /// Number of windows per GPU eval chunk.
    #[arg(long, default_value_t = 64)]
    chunk_size: usize,

    /// Dataset row source: all, train, or eval.
    #[arg(long, default_value = "all")]
    row_source: String,

    /// Deterministic row/noise seed.
    #[arg(long, default_value_t = 7)]
    seed: u64,

    /// Noise std in normalized action units for the noisy-expert action stream.
    #[arg(long, default_value_t = 0.5)]
    noise_std: f32,

    /// Optional JSON report path. If omitted, writes target/drone-oracle-eval/<stamp>.json.
    #[arg(long)]
    json_out: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let started = Instant::now();

    let dataset_dir = args.dataset_dir.clone().unwrap_or_else(default_dataset_dir);
    let weights = args
        .weights
        .clone()
        .unwrap_or_else(|| args.model_dir.join("final.safetensors"));
    let config_path = args
        .config
        .clone()
        .unwrap_or_else(|| args.model_dir.join("model-config.json"));
    let normalization_path = args
        .normalization
        .clone()
        .unwrap_or_else(|| args.model_dir.join("normalization.json"));

    let model_cfg: WorldModelConfig = read_json(&config_path)?;
    validate_model_config(&model_cfg)?;
    let history_size = model_cfg.history_size;
    let horizons = parse_horizons(&args.horizons)?;
    let max_horizon = *horizons.iter().max().context("horizons cannot be empty")?;
    let sequence_steps = history_size
        .checked_add(max_horizon)
        .context("history_size + max_horizon overflowed")?;
    let action_steps = sequence_steps - 1;

    let batch_cfg = DroneBatchConfig {
        batch_size: args.chunk_size,
        sequence_steps,
        normalize_observations: true,
        normalize_actions: true,
    };
    let dataset = DroneRacingDataset::open(&dataset_dir, batch_cfg)?;
    let run_normalization: DroneNormalization = read_json(&normalization_path)?;
    ensure_normalization_matches(&run_normalization, &dataset.metadata().normalization)?;

    let mut rows = rows_for_source(&dataset, &args.row_source)?;
    shuffle(&mut rows, args.seed);
    if args.rows > 0 {
        rows.truncate(args.rows.min(rows.len()));
    }
    ensure!(!rows.is_empty(), "no rows selected for oracle eval");
    let mut shuffled_rows = rows.clone();
    shuffle(&mut shuffled_rows, args.seed ^ 0xA5A5_5A5A_DEAD_BEEF);

    let device = args.device.resolve()?;
    let dtype = args.dtype.dtype();
    ensure!(
        dtype == DType::F32,
        "drone oracle eval currently requires f32"
    );
    let vb = checkpoint::var_builder_from_path(&weights, dtype, &device)
        .with_context(|| format!("failed to load weights {}", weights.display()))?;
    let model = WorldModel::new(model_cfg, vb)?;

    let action_bounds = normalized_action_bounds(&dataset.metadata().normalization.action)?;
    let mut variant_names = vec![
        "expert".to_string(),
        "mean_action".to_string(),
        "shuffled_actions".to_string(),
        "expert_plus_noise".to_string(),
    ];
    for name in ["drop_roll", "drop_pitch", "drop_throttle", "drop_yaw"] {
        variant_names.push(name.to_string());
    }
    let mut variant_accums = variant_names
        .iter()
        .map(|name| VariantAccum::new(name.clone(), &horizons))
        .collect::<Vec<_>>();
    let mut one_step_accum = VariantAccum::new("expert_one_step".to_string(), &horizons);
    let mut total_rollout_sec = 0.0f64;
    let mut total_encode_sec = 0.0f64;
    let mut total_one_step_sec = 0.0f64;

    for (chunk_idx, chunk) in rows.chunks(args.chunk_size).enumerate() {
        let shuffled_chunk =
            &shuffled_rows[chunk_idx * args.chunk_size..chunk_idx * args.chunk_size + chunk.len()];
        let batch = dataset.batch(chunk, dtype, &device)?;
        let shuffled_batch = dataset.batch(shuffled_chunk, dtype, &device)?;

        let encode_start = Instant::now();
        let actual_emb = model.encode_vector(&batch.observations)?;
        device.synchronize()?;
        total_encode_sec += encode_start.elapsed().as_secs_f64();

        let actions = batch.actions.narrow(1, 0, action_steps)?;
        let shuffled_actions = shuffled_batch.actions.narrow(1, 0, action_steps)?;
        let variants = action_variants(
            &actions,
            &shuffled_actions,
            args.noise_std,
            args.seed,
            chunk_idx,
            &action_bounds,
            dtype,
            &device,
        )?;
        ensure!(
            variants.len() == variant_accums.len(),
            "variant count mismatch"
        );
        let variant_refs = variants.iter().collect::<Vec<_>>();
        let action_candidates = Tensor::stack(&variant_refs, 1)?;

        let rollout_start = Instant::now();
        let history_emb = actual_emb.narrow(1, 0, history_size)?.unsqueeze(1)?;
        let history_emb = history_emb.broadcast_as((
            chunk.len(),
            variants.len(),
            history_size,
            actual_emb.dim(2)?,
        ))?;
        let rollout = model.rollout_embeddings_with_history(
            &history_emb,
            &action_candidates,
            history_size,
        )?;
        device.synchronize()?;
        total_rollout_sec += rollout_start.elapsed().as_secs_f64();

        for (variant_idx, accum) in variant_accums.iter_mut().enumerate() {
            for &horizon in &horizons {
                let pred = rollout.i((.., variant_idx, history_size + horizon - 1, ..))?;
                let actual = actual_emb.i((.., history_size + horizon - 1, ..))?;
                accum.accumulate(horizon, &pred, &actual)?;
            }
        }

        let one_step_start = Instant::now();
        let one_step =
            one_step_predictions(&model, &actual_emb, &actions, history_size, max_horizon)?;
        device.synchronize()?;
        total_one_step_sec += one_step_start.elapsed().as_secs_f64();
        for &horizon in &horizons {
            let pred = one_step.i((.., horizon - 1, ..))?;
            let actual = actual_emb.i((.., history_size + horizon - 1, ..))?;
            one_step_accum.accumulate(horizon, &pred, &actual)?;
        }
    }

    device.synchronize()?;
    let elapsed_sec = started.elapsed().as_secs_f64();
    let variants = variant_accums
        .into_iter()
        .map(VariantAccum::finish)
        .collect::<Vec<_>>();
    let one_step = one_step_accum.finish();
    let report = OracleReport {
        model_dir: args.model_dir.clone(),
        dataset_dir,
        weights,
        config: config_path,
        normalization: normalization_path,
        device: args.device.to_string(),
        dtype: args.dtype.to_string(),
        row_source: args.row_source.clone(),
        rows_evaluated: rows.len(),
        history_size,
        max_horizon,
        horizons: horizons.clone(),
        chunk_size: args.chunk_size,
        noise_std: args.noise_std,
        elapsed_sec,
        encode_sec: total_encode_sec,
        rollout_sec: total_rollout_sec,
        one_step_sec: total_one_step_sec,
        windows_per_sec: rows.len() as f64 / elapsed_sec.max(1e-9),
        variant_rollouts_per_sec: (rows.len() * variants.len()) as f64 / elapsed_sec.max(1e-9),
        one_step,
        variants,
    };

    print_report(&report);
    let json_out = args.json_out.unwrap_or_else(default_json_path);
    write_pretty_json(&json_out, &report)?;
    println!("json={}", json_out.display());
    Ok(())
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    ensure!(
        args.chunk_size > 0,
        "--chunk-size must be greater than zero"
    );
    ensure!(
        args.noise_std.is_finite() && args.noise_std >= 0.0,
        "--noise-std must be finite and non-negative"
    );
    Ok(())
}

fn validate_model_config(cfg: &WorldModelConfig) -> anyhow::Result<()> {
    cfg.validate()?;
    ensure!(
        cfg.action_encoder.input_dim == DRONE_ACTION_DIM,
        "model action dim {} does not match drone action dim {DRONE_ACTION_DIM}",
        cfg.action_encoder.input_dim
    );
    match &cfg.observation_encoder {
        ObservationEncoderConfig::VectorMlp(vector) => ensure!(
            vector.input_dim == DRONE_OBSERVATION_DIM,
            "model observation dim {} does not match drone observation dim {DRONE_OBSERVATION_DIM}",
            vector.input_dim
        ),
        ObservationEncoderConfig::ImageVit { .. } => {
            anyhow::bail!("lewm-drone-oracle-eval requires vector observations")
        }
    }
    Ok(())
}

fn default_dataset_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".stable_worldmodel")
        .join("le-wm-nv-data")
        .join("drone-racing-autonomous-100hz-pose12")
}

fn default_json_path() -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    PathBuf::from("target")
        .join("drone-oracle-eval")
        .join(format!("oracle-{stamp}.json"))
}

fn read_json<T>(path: &Path) -> anyhow::Result<T>
where
    T: serde::de::DeserializeOwned,
{
    serde_json::from_str(
        &fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))
}

fn write_pretty_json<T>(path: &Path, value: &T) -> anyhow::Result<()>
where
    T: Serialize,
{
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let bytes = serde_json::to_vec_pretty(value)?;
    fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))
}

fn parse_horizons(value: &str) -> anyhow::Result<Vec<usize>> {
    let mut horizons = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>()
                .with_context(|| format!("invalid horizon `{part}`"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    horizons.sort_unstable();
    horizons.dedup();
    ensure!(!horizons.is_empty(), "--horizons cannot be empty");
    ensure!(
        horizons.iter().all(|horizon| *horizon > 0),
        "--horizons must be positive"
    );
    Ok(horizons)
}

fn rows_for_source(dataset: &DroneRacingDataset, source: &str) -> anyhow::Result<Vec<usize>> {
    match source {
        "all" => Ok(dataset.valid_rows().to_vec()),
        "train" => Ok(dataset.train_rows()),
        "eval" => Ok(dataset.eval_rows()),
        other => anyhow::bail!("unsupported --row-source `{other}`; expected all, train, or eval"),
    }
}

fn ensure_normalization_matches(
    run: &DroneNormalization,
    dataset: &DroneNormalization,
) -> anyhow::Result<()> {
    ensure_stats_match("observation", &run.observation, &dataset.observation)?;
    ensure_stats_match("action", &run.action, &dataset.action)
}

fn ensure_stats_match(
    name: &str,
    run: &RunningStats,
    dataset: &RunningStats,
) -> anyhow::Result<()> {
    ensure!(
        run.mean.len() == dataset.mean.len() && run.std.len() == dataset.std.len(),
        "{name} normalization dimension mismatch: run mean/std={}/{} dataset mean/std={}/{}",
        run.mean.len(),
        run.std.len(),
        dataset.mean.len(),
        dataset.std.len()
    );
    let max_mean = max_abs_delta(&run.mean, &dataset.mean);
    let max_std = max_abs_delta(&run.std, &dataset.std);
    ensure!(
        max_mean <= 1e-5 && max_std <= 1e-5,
        "{name} normalization mismatch: max_mean_delta={max_mean:.3e} max_std_delta={max_std:.3e}"
    );
    Ok(())
}

fn max_abs_delta(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn normalized_action_bounds(stats: &RunningStats) -> anyhow::Result<ActionBoundsNorm> {
    ensure!(
        stats.mean.len() == DRONE_ACTION_DIM && stats.std.len() == DRONE_ACTION_DIM,
        "action stats dim mismatch"
    );
    let raw_low = [-1.0, -1.0, 0.0, -1.0];
    let raw_high = [1.0, 1.0, 1.0, 1.0];
    let mut low = [0.0f32; DRONE_ACTION_DIM];
    let mut high = [0.0f32; DRONE_ACTION_DIM];
    for idx in 0..DRONE_ACTION_DIM {
        let std = stats.std[idx].max(1e-6);
        low[idx] = (raw_low[idx] - stats.mean[idx]) / std;
        high[idx] = (raw_high[idx] - stats.mean[idx]) / std;
    }
    Ok(ActionBoundsNorm { low, high })
}

fn action_variants(
    expert: &Tensor,
    shuffled: &Tensor,
    noise_std: f32,
    seed: u64,
    chunk_idx: usize,
    bounds: &ActionBoundsNorm,
    dtype: DType,
    device: &candle::Device,
) -> anyhow::Result<Vec<Tensor>> {
    let dims = expert.dims3()?;
    ensure!(
        shuffled.dims() == expert.dims(),
        "shuffled actions shape {:?} does not match expert {:?}",
        shuffled.shape(),
        expert.shape()
    );
    let mut variants = Vec::with_capacity(8);
    variants.push(expert.clone());
    variants.push(Tensor::zeros(dims, dtype, device)?);
    variants.push(shuffled.clone());
    variants.push(noisy_actions(
        expert,
        noise_std,
        seed ^ ((chunk_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        bounds,
        dtype,
        device,
    )?);
    for channel in 0..DRONE_ACTION_DIM {
        variants.push(drop_action_channel(expert, channel, dtype, device)?);
    }
    Ok(variants)
}

fn drop_action_channel(
    actions: &Tensor,
    channel: usize,
    dtype: DType,
    device: &candle::Device,
) -> anyhow::Result<Tensor> {
    let mut values = actions.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    for sample in &mut values {
        for step in sample {
            step[channel] = 0.0;
        }
    }
    tensor_from_action_values(values, dtype, device)
}

fn noisy_actions(
    actions: &Tensor,
    noise_std: f32,
    seed: u64,
    bounds: &ActionBoundsNorm,
    dtype: DType,
    device: &candle::Device,
) -> anyhow::Result<Tensor> {
    let mut rng = SplitMix64::new(seed);
    let mut values = actions.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    for sample in &mut values {
        for step in sample {
            for (channel, value) in step.iter_mut().enumerate() {
                let noise = rng.next_f32_signed() * noise_std;
                *value = (*value + noise).clamp(bounds.low[channel], bounds.high[channel]);
            }
        }
    }
    tensor_from_action_values(values, dtype, device)
}

fn tensor_from_action_values(
    values: Vec<Vec<Vec<f32>>>,
    dtype: DType,
    device: &candle::Device,
) -> anyhow::Result<Tensor> {
    let batch = values.len();
    let steps = values.first().map(Vec::len).unwrap_or(0);
    let mut flat = Vec::with_capacity(batch * steps * DRONE_ACTION_DIM);
    for sample in values {
        ensure!(
            sample.len() == steps,
            "ragged action sample: {} vs {steps}",
            sample.len()
        );
        for step in sample {
            ensure!(
                step.len() == DRONE_ACTION_DIM,
                "action dim {} does not match {DRONE_ACTION_DIM}",
                step.len()
            );
            flat.extend(step);
        }
    }
    Ok(Tensor::from_vec(flat, (batch, steps, DRONE_ACTION_DIM), device)?.to_dtype(dtype)?)
}

fn one_step_predictions(
    model: &WorldModel,
    actual_emb: &Tensor,
    actions: &Tensor,
    history_size: usize,
    max_horizon: usize,
) -> anyhow::Result<Tensor> {
    let (batch, _, dim) = actual_emb.dims3()?;
    let action_dim = actions.dim(2)?;
    let mut emb_windows = Vec::with_capacity(max_horizon);
    let mut action_windows = Vec::with_capacity(max_horizon);
    for step in 0..max_horizon {
        emb_windows.push(actual_emb.narrow(1, step, history_size)?);
        action_windows.push(actions.narrow(1, step, history_size)?);
    }
    let emb_refs = emb_windows.iter().collect::<Vec<_>>();
    let action_refs = action_windows.iter().collect::<Vec<_>>();
    let emb = Tensor::stack(&emb_refs, 1)?.reshape((batch * max_horizon, history_size, dim))?;
    let action =
        Tensor::stack(&action_refs, 1)?.reshape((batch * max_horizon, history_size, action_dim))?;
    let pred = model.predict(&emb, &action)?;
    pred.i((.., history_size - 1, ..))?
        .reshape((batch, max_horizon, dim))
        .map_err(Into::into)
}

fn pair_stats(pred: &Tensor, actual: &Tensor) -> anyhow::Result<PairStats> {
    let diff = (pred - actual)?;
    let sq = diff.sqr()?;
    let sum_sq = sq.sum(D::Minus1)?;
    let l2 = sum_sq.sqrt()?;
    let dim = pred.dim(pred.dims().len() - 1)? as f64;
    let mse = (&sum_sq / dim)?;
    let dot = (pred * actual)?.sum(D::Minus1)?;
    let pred_norm = pred.sqr()?.sum(D::Minus1)?.sqrt()?;
    let actual_norm = actual.sqr()?.sum(D::Minus1)?.sqrt()?;
    let denom = (pred_norm * actual_norm)?;
    let cosine = (dot / denom.clamp(1e-12f32, f32::INFINITY)?)?;
    Ok(PairStats {
        mse: mse.to_dtype(DType::F32)?.to_vec1::<f32>()?,
        l2: l2.to_dtype(DType::F32)?.to_vec1::<f32>()?,
        cosine: cosine.to_dtype(DType::F32)?.to_vec1::<f32>()?,
    })
}

fn print_report(report: &OracleReport) {
    println!(
        "oracle_eval model={} rows={} source={} history={} horizons={:?} elapsed={:.3}s windows_per_sec={:.1} variant_rollouts_per_sec={:.1}",
        report.model_dir.display(),
        report.rows_evaluated,
        report.row_source,
        report.history_size,
        report.horizons,
        report.elapsed_sec,
        report.windows_per_sec,
        report.variant_rollouts_per_sec,
    );
    println!(
        "timing encode={:.3}s rollout={:.3}s one_step={:.3}s",
        report.encode_sec, report.rollout_sec, report.one_step_sec
    );
    println!("one_step:");
    print_variant(&report.one_step);
    println!("autoregressive variants:");
    for variant in &report.variants {
        print_variant(variant);
    }
}

fn print_variant(variant: &VariantReport) {
    println!("  {}", variant.name);
    println!(
        "    {:>8} {:>10} {:>10} {:>10} {:>10}",
        "horizon", "mean_l2", "mean_mse", "mean_cos", "max_l2"
    );
    for metric in &variant.horizons {
        println!(
            "    {:>8} {:>10.4} {:>10.6} {:>10.4} {:>10.4}",
            metric.horizon, metric.mean_l2, metric.mean_mse, metric.mean_cosine, metric.max_l2,
        );
    }
}

#[derive(Debug, Clone, Copy)]
struct ActionBoundsNorm {
    low: [f32; DRONE_ACTION_DIM],
    high: [f32; DRONE_ACTION_DIM],
}

struct PairStats {
    mse: Vec<f32>,
    l2: Vec<f32>,
    cosine: Vec<f32>,
}

#[derive(Debug)]
struct VariantAccum {
    name: String,
    horizons: Vec<HorizonAccum>,
}

impl VariantAccum {
    fn new(name: String, horizons: &[usize]) -> Self {
        Self {
            name,
            horizons: horizons.iter().copied().map(HorizonAccum::new).collect(),
        }
    }

    fn accumulate(&mut self, horizon: usize, pred: &Tensor, actual: &Tensor) -> anyhow::Result<()> {
        let stats = pair_stats(pred, actual)?;
        let accum = self
            .horizons
            .iter_mut()
            .find(|item| item.horizon == horizon)
            .with_context(|| format!("missing horizon accumulator {horizon}"))?;
        accum.push(&stats);
        Ok(())
    }

    fn finish(self) -> VariantReport {
        VariantReport {
            name: self.name,
            horizons: self
                .horizons
                .into_iter()
                .map(HorizonAccum::finish)
                .collect(),
        }
    }
}

#[derive(Debug)]
struct HorizonAccum {
    horizon: usize,
    count: usize,
    sum_mse: f64,
    sum_l2: f64,
    sum_cosine: f64,
    max_l2: f32,
}

impl HorizonAccum {
    fn new(horizon: usize) -> Self {
        Self {
            horizon,
            count: 0,
            sum_mse: 0.0,
            sum_l2: 0.0,
            sum_cosine: 0.0,
            max_l2: 0.0,
        }
    }

    fn push(&mut self, stats: &PairStats) {
        for ((&mse, &l2), &cosine) in stats.mse.iter().zip(&stats.l2).zip(&stats.cosine) {
            self.count += 1;
            self.sum_mse += f64::from(mse);
            self.sum_l2 += f64::from(l2);
            self.sum_cosine += f64::from(cosine);
            self.max_l2 = self.max_l2.max(l2);
        }
    }

    fn finish(self) -> HorizonMetric {
        let count = self.count.max(1) as f64;
        HorizonMetric {
            horizon: self.horizon,
            count: self.count,
            mean_mse: (self.sum_mse / count) as f32,
            mean_l2: (self.sum_l2 / count) as f32,
            mean_cosine: (self.sum_cosine / count) as f32,
            max_l2: self.max_l2,
        }
    }
}

#[derive(Debug, Serialize)]
struct OracleReport {
    model_dir: PathBuf,
    dataset_dir: PathBuf,
    weights: PathBuf,
    config: PathBuf,
    normalization: PathBuf,
    device: String,
    dtype: String,
    row_source: String,
    rows_evaluated: usize,
    history_size: usize,
    max_horizon: usize,
    horizons: Vec<usize>,
    chunk_size: usize,
    noise_std: f32,
    elapsed_sec: f64,
    encode_sec: f64,
    rollout_sec: f64,
    one_step_sec: f64,
    windows_per_sec: f64,
    variant_rollouts_per_sec: f64,
    one_step: VariantReport,
    variants: Vec<VariantReport>,
}

#[derive(Debug, Serialize)]
struct VariantReport {
    name: String,
    horizons: Vec<HorizonMetric>,
}

#[derive(Debug, Serialize)]
struct HorizonMetric {
    horizon: usize,
    count: usize,
    mean_mse: f32,
    mean_l2: f32,
    mean_cosine: f32,
    max_l2: f32,
}

#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_f32_signed(&mut self) -> f32 {
        let value = (self.next_u64() >> 40) as f32 / ((1u64 << 24) as f32);
        value * 2.0 - 1.0
    }
}
