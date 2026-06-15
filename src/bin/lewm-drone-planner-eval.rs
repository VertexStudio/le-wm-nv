use std::{
    fs,
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, ensure};
use candle::{DType, Device as CandleDevice, IndexOp, Tensor};
use clap::Parser;
use le_wm_nv::{
    checkpoint,
    data::drone_racing::{
        DRONE_ACTION_DIM, DRONE_OBSERVATION_DIM, DroneBatchConfig, DroneNormalization,
        DroneRacingDataset, RunningStats, shuffle,
    },
    models::world_model::{ObservationEncoderConfig, WorldModel, WorldModelConfig},
    planner::{ActionBounds, CandidateScorer, IcemConfig, IcemPlanner},
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

    /// Comma-separated future target horizons in dataset steps.
    #[arg(long, default_value = "5,10,25")]
    horizons: String,

    /// Number of valid windows to evaluate. Use 0 for all valid rows.
    #[arg(long, default_value_t = 128)]
    rows: usize,

    /// Number of windows per batched planner call.
    #[arg(long, default_value_t = 16)]
    chunk_size: usize,

    /// Dataset row source: all, train, or eval.
    #[arg(long, default_value = "all")]
    row_source: String,

    #[arg(long, default_value_t = 64)]
    planner_samples: usize,

    #[arg(long, default_value_t = 8)]
    planner_elites: usize,

    #[arg(long, default_value_t = 2)]
    planner_iterations: usize,

    #[arg(long, default_value_t = 0.35)]
    planner_init_std: f32,

    #[arg(long, default_value_t = 0.005)]
    planner_min_std: f32,

    #[arg(long, default_value_t = 7)]
    seed: u64,

    /// Optional JSON report path. If omitted, writes target/drone-planner-eval/<stamp>.json.
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

    let batch_cfg = DroneBatchConfig {
        batch_size: args.chunk_size,
        sequence_steps,
        normalize_observations: true,
        normalize_actions: false,
    };
    let dataset = DroneRacingDataset::open(&dataset_dir, batch_cfg)?;
    let run_normalization: DroneNormalization = read_json(&normalization_path)?;
    ensure_normalization_matches(&run_normalization, &dataset.metadata().normalization)?;

    let mut rows = rows_for_source(&dataset, &args.row_source)?;
    shuffle(&mut rows, args.seed);
    if args.rows > 0 {
        rows.truncate(args.rows.min(rows.len()));
    }
    ensure!(!rows.is_empty(), "no rows selected for planner eval");
    let mut shuffled_rows = rows.clone();
    shuffle(&mut shuffled_rows, args.seed ^ 0xBADC_0FFE_E0DD_F00D);

    let device = args.device.resolve()?;
    let dtype = args.dtype.dtype();
    ensure!(
        dtype == DType::F32,
        "drone planner eval currently requires f32"
    );
    let vb = checkpoint::var_builder_from_path(&weights, dtype, &device)
        .with_context(|| format!("failed to load weights {}", weights.display()))?;
    let model = WorldModel::new(model_cfg, vb)?;

    let action_mean = Tensor::from_vec(
        dataset.metadata().normalization.action.mean.clone(),
        (DRONE_ACTION_DIM,),
        &device,
    )?
    .to_dtype(dtype)?
    .reshape((1, 1, 1, DRONE_ACTION_DIM))?;
    let action_std = Tensor::from_vec(
        dataset.metadata().normalization.action.std.clone(),
        (DRONE_ACTION_DIM,),
        &device,
    )?
    .to_dtype(dtype)?
    .reshape((1, 1, 1, DRONE_ACTION_DIM))?;
    let mean_action_raw = dataset.metadata().normalization.action.mean.clone();

    let mut horizon_accums = horizons
        .iter()
        .copied()
        .map(HorizonAccum::new)
        .collect::<Vec<_>>();
    let mut total_plan_sec = 0.0f64;
    let mut total_encode_sec = 0.0f64;

    for &horizon in &horizons {
        for (chunk_idx, chunk) in rows.chunks(args.chunk_size).enumerate() {
            let shuffled_chunk = &shuffled_rows
                [chunk_idx * args.chunk_size..chunk_idx * args.chunk_size + chunk.len()];
            let batch = dataset.batch(chunk, dtype, &device)?;
            let shuffled_batch = dataset.batch(shuffled_chunk, dtype, &device)?;

            let encode_started = Instant::now();
            let actual_emb = model.encode_vector(&batch.observations)?;
            device.synchronize()?;
            total_encode_sec += encode_started.elapsed().as_secs_f64();

            let history_emb = actual_emb.narrow(1, 0, history_size)?;
            let target_emb = actual_emb.i((.., history_size + horizon - 1, ..))?;
            let prefix = batch
                .actions
                .narrow(1, 0, history_size.saturating_sub(1))?
                .unsqueeze(1)?;
            let expert_sequence = batch.actions.narrow(1, history_size - 1, horizon)?;
            let shuffled_sequence = shuffled_batch
                .actions
                .narrow(1, history_size - 1, horizon)?;
            let mean_sequence =
                mean_action_sequence(chunk.len(), horizon, &mean_action_raw, dtype, &device)?;
            let first_expert = expert_sequence.i((.., 0, ..))?;

            let scorer = OfflineScorer {
                model: &model,
                device: &device,
                dtype,
                history_emb: &history_emb,
                target_emb: &target_emb,
                action_mean: &action_mean,
                action_std: &action_std,
                action_prefix: &prefix,
            };

            let expert_scores = score_single_sequence(&scorer, &expert_sequence)?;
            let mean_scores = score_single_sequence(&scorer, &mean_sequence)?;
            let shuffled_scores = score_single_sequence(&scorer, &shuffled_sequence)?;

            let mut cfg = IcemConfig::new(
                horizon,
                args.planner_samples,
                args.planner_elites,
                DRONE_ACTION_DIM,
            );
            cfg.iterations = args.planner_iterations;
            cfg.keep_elites = args.planner_elites;
            cfg.init_std = args.planner_init_std;
            cfg.min_std = args.planner_min_std;
            cfg.seed = Some(args.seed ^ ((horizon as u64) << 32) ^ chunk_idx as u64);
            cfg.action_bounds = full_action_bounds();
            let mut planner = IcemPlanner::new(cfg);
            let warm_start = first_expert.unsqueeze(1)?.broadcast_as((
                chunk.len(),
                horizon,
                DRONE_ACTION_DIM,
            ))?;
            planner.set_warm_start_sequence(warm_start);

            let plan_started = Instant::now();
            let result = planner.plan_device(&scorer)?;
            device.synchronize()?;
            total_plan_sec += plan_started.elapsed().as_secs_f64();

            let selected_scores = score_single_sequence(&scorer, &result.sequence)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            let candidate_best_scores = best_scores_per_row(&result.scores)?;
            let expert_scores = expert_scores.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let mean_scores = mean_scores.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let shuffled_scores = shuffled_scores.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let action_error = action_l2_per_row(&result.first_action, &first_expert)?;
            let saturation = saturation_per_row(&result.first_action)?;

            let accum = horizon_accums
                .iter_mut()
                .find(|item| item.horizon == horizon)
                .expect("missing horizon accumulator");
            for idx in 0..selected_scores.len() {
                accum.push(RowPlannerMetric {
                    selected_score: selected_scores[idx],
                    candidate_best_score: candidate_best_scores[idx],
                    expert_score: expert_scores[idx],
                    mean_score: mean_scores[idx],
                    shuffled_score: shuffled_scores[idx],
                    first_action_l2: action_error[idx],
                    saturated: saturation[idx],
                    plan_ms: result.elapsed.as_secs_f32() * 1000.0,
                });
            }
        }
    }

    device.synchronize()?;
    let elapsed_sec = started.elapsed().as_secs_f64();
    let horizon_reports = horizon_accums
        .into_iter()
        .map(HorizonAccum::finish)
        .collect::<Vec<_>>();
    let report = PlannerEvalReport {
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
        chunk_size: args.chunk_size,
        planner_samples: args.planner_samples,
        planner_elites: args.planner_elites,
        planner_iterations: args.planner_iterations,
        planner_init_std: args.planner_init_std,
        planner_min_std: args.planner_min_std,
        elapsed_sec,
        encode_sec: total_encode_sec,
        plan_sec: total_plan_sec,
        plan_batches_per_sec: (report_batch_count(rows.len(), args.chunk_size)
            * horizon_reports.len()) as f64
            / total_plan_sec.max(1e-9),
        horizons: horizon_reports,
    };

    print_report(&report);
    let json_out = args.json_out.unwrap_or_else(default_json_path);
    write_pretty_json(&json_out, &report)?;
    println!("json={}", json_out.display());
    Ok(())
}

struct OfflineScorer<'a> {
    model: &'a WorldModel,
    device: &'a CandleDevice,
    dtype: DType,
    history_emb: &'a Tensor,
    target_emb: &'a Tensor,
    action_mean: &'a Tensor,
    action_std: &'a Tensor,
    action_prefix: &'a Tensor,
}

impl CandidateScorer for OfflineScorer<'_> {
    fn device(&self) -> &CandleDevice {
        self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn batch_size(&self) -> Option<usize> {
        self.history_emb.dims().first().copied()
    }

    fn score_candidates(&self, action_candidates: &Tensor) -> candle::Result<Tensor> {
        let action_candidates = action_candidates
            .to_device(self.device)?
            .to_dtype(self.dtype)?;
        let (batch, samples, _, _) = action_candidates.dims4()?;
        let prefix_len = self.action_prefix.dim(2)?;
        let prefix =
            self.action_prefix
                .broadcast_as((batch, samples, prefix_len, DRONE_ACTION_DIM))?;
        let actions = Tensor::cat(&[&prefix, &action_candidates], 2)?;
        let normalized_actions = actions
            .broadcast_sub(self.action_mean)?
            .broadcast_div(self.action_std)?;
        let (_, history, dim) = self.history_emb.dims3()?;
        let emb_init = self
            .history_emb
            .unsqueeze(1)?
            .broadcast_as((batch, samples, history, dim))?;
        let rollout =
            self.model
                .rollout_embeddings_with_history(&emb_init, &normalized_actions, history)?;
        self.model.goal_cost(&rollout, self.target_emb)
    }
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    ensure!(
        args.chunk_size > 0,
        "--chunk-size must be greater than zero"
    );
    ensure!(
        args.planner_samples > 0,
        "--planner-samples must be greater than zero"
    );
    ensure!(
        args.planner_elites >= 2 && args.planner_elites <= args.planner_samples,
        "--planner-elites must be in [2, planner_samples]"
    );
    ensure!(
        args.planner_iterations > 0,
        "--planner-iterations must be greater than zero"
    );
    ensure!(
        args.planner_init_std.is_finite() && args.planner_init_std > 0.0,
        "--planner-init-std must be finite and positive"
    );
    ensure!(
        args.planner_min_std.is_finite() && args.planner_min_std >= 0.0,
        "--planner-min-std must be finite and non-negative"
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
            anyhow::bail!("lewm-drone-planner-eval requires vector observations")
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
        .join("drone-planner-eval")
        .join(format!("planner-{stamp}.json"))
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

fn full_action_bounds() -> ActionBounds {
    ActionBounds {
        low: vec![-1.0, -1.0, 0.0, -1.0],
        high: vec![1.0, 1.0, 1.0, 1.0],
    }
}

fn mean_action_sequence(
    batch: usize,
    horizon: usize,
    mean_action: &[f32],
    dtype: DType,
    device: &CandleDevice,
) -> anyhow::Result<Tensor> {
    ensure!(
        mean_action.len() == DRONE_ACTION_DIM,
        "mean action dim mismatch"
    );
    let mut values = Vec::with_capacity(batch * horizon * DRONE_ACTION_DIM);
    for _ in 0..batch * horizon {
        values.extend_from_slice(mean_action);
    }
    Ok(Tensor::from_vec(values, (batch, horizon, DRONE_ACTION_DIM), device)?.to_dtype(dtype)?)
}

fn score_single_sequence(scorer: &OfflineScorer<'_>, sequence: &Tensor) -> anyhow::Result<Tensor> {
    Ok(scorer
        .score_candidates(&sequence.unsqueeze(1)?)?
        .squeeze(1)?)
}

fn best_scores_per_row(scores: &Tensor) -> anyhow::Result<Vec<f32>> {
    let rows = scores.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    Ok(rows
        .into_iter()
        .map(|row| row.into_iter().fold(f32::INFINITY, f32::min))
        .collect())
}

fn action_l2_per_row(pred: &Tensor, actual: &Tensor) -> anyhow::Result<Vec<f32>> {
    let pred = pred.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let actual = actual.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    ensure!(
        pred.len() == actual.len(),
        "action row count mismatch {} vs {}",
        pred.len(),
        actual.len()
    );
    let mut out = Vec::with_capacity(pred.len());
    for (pred, actual) in pred.iter().zip(actual.iter()) {
        let sum_sq = pred
            .iter()
            .zip(actual)
            .map(|(p, a)| {
                let delta = p - a;
                delta * delta
            })
            .sum::<f32>();
        out.push(sum_sq.sqrt());
    }
    Ok(out)
}

fn saturation_per_row(actions: &Tensor) -> anyhow::Result<Vec<bool>> {
    let actions = actions.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    Ok(actions
        .into_iter()
        .map(|row| {
            row[0].abs() >= 0.98
                || row[1].abs() >= 0.98
                || row[2] <= 0.02
                || row[2] >= 0.98
                || row[3].abs() >= 0.98
        })
        .collect())
}

fn report_batch_count(rows: usize, chunk_size: usize) -> usize {
    rows.div_ceil(chunk_size)
}

fn print_report(report: &PlannerEvalReport) {
    println!(
        "planner_eval model={} rows={} source={} history={} horizons={:?} elapsed={:.3}s plan_sec={:.3}s plan_batches_per_sec={:.2}",
        report.model_dir.display(),
        report.rows_evaluated,
        report.row_source,
        report.history_size,
        report
            .horizons
            .iter()
            .map(|item| item.horizon)
            .collect::<Vec<_>>(),
        report.elapsed_sec,
        report.plan_sec,
        report.plan_batches_per_sec,
    );
    println!(
        "{:>8} {:>7} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>9} {:>10}",
        "horizon",
        "count",
        "selected",
        "cand_best",
        "expert",
        "mean_act",
        "shuffled",
        "act_l2",
        "sat_pct",
        "plan_ms",
    );
    for metric in &report.horizons {
        println!(
            "{:>8} {:>7} {:>11.4} {:>11.4} {:>11.4} {:>11.4} {:>11.4} {:>11.4} {:>8.1}% {:>10.3}",
            metric.horizon,
            metric.count,
            metric.mean_selected_score,
            metric.mean_candidate_best_score,
            metric.mean_expert_score,
            metric.mean_mean_action_score,
            metric.mean_shuffled_score,
            metric.mean_first_action_l2,
            metric.saturation_rate * 100.0,
            metric.mean_plan_ms,
        );
    }
}

#[derive(Debug, Clone, Copy)]
struct RowPlannerMetric {
    selected_score: f32,
    candidate_best_score: f32,
    expert_score: f32,
    mean_score: f32,
    shuffled_score: f32,
    first_action_l2: f32,
    saturated: bool,
    plan_ms: f32,
}

#[derive(Debug)]
struct HorizonAccum {
    horizon: usize,
    count: usize,
    selected_score: f64,
    candidate_best_score: f64,
    expert_score: f64,
    mean_score: f64,
    shuffled_score: f64,
    first_action_l2: f64,
    saturated: usize,
    plan_ms: f64,
}

impl HorizonAccum {
    fn new(horizon: usize) -> Self {
        Self {
            horizon,
            count: 0,
            selected_score: 0.0,
            candidate_best_score: 0.0,
            expert_score: 0.0,
            mean_score: 0.0,
            shuffled_score: 0.0,
            first_action_l2: 0.0,
            saturated: 0,
            plan_ms: 0.0,
        }
    }

    fn push(&mut self, row: RowPlannerMetric) {
        self.count += 1;
        self.selected_score += f64::from(row.selected_score);
        self.candidate_best_score += f64::from(row.candidate_best_score);
        self.expert_score += f64::from(row.expert_score);
        self.mean_score += f64::from(row.mean_score);
        self.shuffled_score += f64::from(row.shuffled_score);
        self.first_action_l2 += f64::from(row.first_action_l2);
        self.saturated += usize::from(row.saturated);
        self.plan_ms += f64::from(row.plan_ms);
    }

    fn finish(self) -> HorizonPlannerMetric {
        let count = self.count.max(1) as f64;
        HorizonPlannerMetric {
            horizon: self.horizon,
            count: self.count,
            mean_selected_score: (self.selected_score / count) as f32,
            mean_candidate_best_score: (self.candidate_best_score / count) as f32,
            mean_expert_score: (self.expert_score / count) as f32,
            mean_mean_action_score: (self.mean_score / count) as f32,
            mean_shuffled_score: (self.shuffled_score / count) as f32,
            mean_first_action_l2: (self.first_action_l2 / count) as f32,
            saturation_rate: self.saturated as f32 / self.count.max(1) as f32,
            mean_plan_ms: (self.plan_ms / count) as f32,
        }
    }
}

#[derive(Debug, Serialize)]
struct PlannerEvalReport {
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
    horizons: Vec<HorizonPlannerMetric>,
    chunk_size: usize,
    planner_samples: usize,
    planner_elites: usize,
    planner_iterations: usize,
    planner_init_std: f32,
    planner_min_std: f32,
    elapsed_sec: f64,
    encode_sec: f64,
    plan_sec: f64,
    plan_batches_per_sec: f64,
}

#[derive(Debug, Serialize)]
struct HorizonPlannerMetric {
    horizon: usize,
    count: usize,
    mean_selected_score: f32,
    mean_candidate_best_score: f32,
    mean_expert_score: f32,
    mean_mean_action_score: f32,
    mean_shuffled_score: f32,
    mean_first_action_l2: f32,
    saturation_rate: f32,
    mean_plan_ms: f32,
}
