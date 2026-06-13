use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, ensure};
use candle::{DType, Device, IndexOp, Tensor};
use clap::Parser;
use le_wm_nv::{
    checkpoint,
    data::drone_racing::{
        DRONE_ACTION_DIM, DRONE_STATE_DELTA_DIM, DroneBatchConfig, DroneRacingDataset, RunningStats,
    },
    models::world_model::{WorldModel, WorldModelConfig},
    runtime::{DTypeSpec, DeviceSpec},
};
use serde::Serialize;

#[derive(Parser, Debug)]
struct Args {
    /// Imported drone dataset directory containing data.h5 and metadata.json.
    #[arg(long)]
    dataset_dir: Option<PathBuf>,

    /// Trained weights.
    #[arg(long)]
    weights: Option<PathBuf>,

    /// WorldModel config JSON.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Output benchmark JSON.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Dataset row used as current history start. Defaults to first eval row.
    #[arg(long)]
    row: Option<usize>,

    #[arg(long, default_value_t = DeviceSpec::Cuda(0))]
    device: DeviceSpec,

    #[arg(long, default_value_t = DTypeSpec::F32)]
    dtype: DTypeSpec,

    #[arg(long, default_value_t = 8)]
    history_steps: usize,

    /// Number of future action steps appended after the history action prefix.
    #[arg(long, default_value_t = 40)]
    horizon: usize,

    /// Number of candidate action sequences in the fixed-shape rollout batch.
    #[arg(long, default_value_t = 528)]
    samples: usize,

    #[arg(long, default_value_t = 2)]
    warmup: usize,

    #[arg(long, default_value_t = 10)]
    iterations: usize,

    /// Deterministic action perturbation scale around the dataset action mean.
    #[arg(long, default_value_t = 0.35)]
    action_scale: f32,

    #[arg(long)]
    no_observation_normalize: bool,

    #[arg(long)]
    no_action_normalize: bool,

    #[arg(long)]
    no_target_normalize: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    validate_args(&args)?;

    let dataset_dir = args.dataset_dir.clone().unwrap_or_else(default_dataset_dir);
    let weights = args.weights.clone().unwrap_or_else(default_weights);
    let config = args.config.clone().unwrap_or_else(default_config);
    let output = args.output.clone().unwrap_or_else(default_output);
    let dtype = args.dtype.dtype();
    ensure!(
        dtype == DType::F32,
        "drone rollout benchmark requires --dtype f32"
    );

    let device = args.device.resolve()?;
    ensure!(device.is_cuda(), "drone rollout benchmark requires CUDA");

    let setup_started = Instant::now();
    let batch_cfg = DroneBatchConfig {
        batch_size: 1,
        sequence_steps: args.history_steps.max(2),
        normalize_observations: !args.no_observation_normalize,
        normalize_actions: !args.no_action_normalize,
        normalize_targets: !args.no_target_normalize,
    };
    let dataset = DroneRacingDataset::open(&dataset_dir, batch_cfg)?;
    let row = args
        .row
        .unwrap_or_else(|| dataset.eval_rows().first().copied().unwrap_or(0));
    let cfg: WorldModelConfig = serde_json::from_str(
        &fs::read_to_string(&config)
            .with_context(|| format!("failed to read {}", config.display()))?,
    )
    .with_context(|| format!("failed to parse {}", config.display()))?;
    let model_history_size = cfg.history_size;
    let vb = checkpoint::var_builder_from_path(&weights, dtype, &device)
        .with_context(|| format!("failed to load {}", weights.display()))?;
    let model = WorldModel::new(cfg, vb)?;
    let setup_sec = setup_started.elapsed().as_secs_f64();

    let prep_started = Instant::now();
    let history = dataset.batch(&[row], dtype, &device)?;
    let emb = model.encode_vector(&history.observations)?;
    let action_prefix = history_action_prefix(&history.actions, args.history_steps)?;
    let prepared = prepare_rollout_tensors(
        &emb,
        &action_prefix,
        args.samples,
        args.horizon,
        args.action_scale,
        &dataset.metadata().normalization.action,
        !args.no_action_normalize,
        dtype,
        &device,
    )?;
    let target_stats = target_stats_tensors(
        &dataset.metadata().normalization.target_delta,
        !args.no_target_normalize,
        dtype,
        &device,
    )?;
    device.synchronize()?;
    let prep_sec = prep_started.elapsed().as_secs_f64();

    for _ in 0..args.warmup {
        let rollout = model.rollout_embeddings_with_history(
            &prepared.emb_init,
            &prepared.actions,
            prepared.history,
        )?;
        let pred = predict_future_state_deltas(
            &model,
            &rollout,
            prepared.samples,
            prepared.emb_dim,
            prepared.history,
            args.horizon,
        )?;
        let future = denormalize_deltas(&pred, &target_stats)?;
        device.synchronize()?;
        let _ = future.sum_all()?.to_scalar::<f32>()?;
    }

    let mut timings = Vec::with_capacity(args.iterations);
    for iter in 0..args.iterations {
        device.synchronize()?;
        let total_started = Instant::now();

        let (rollout, rollout_sec) = timed(&device, || {
            model.rollout_embeddings_with_history(
                &prepared.emb_init,
                &prepared.actions,
                prepared.history,
            )
        })?;
        let (pred, state_head_sec) = timed(&device, || {
            predict_future_state_deltas(
                &model,
                &rollout,
                prepared.samples,
                prepared.emb_dim,
                prepared.history,
                args.horizon,
            )
        })?;
        let (future, denorm_slice_sec) =
            timed(&device, || denormalize_deltas(&pred, &target_stats))?;
        let checksum_started = Instant::now();
        let checksum = future.sum_all()?.to_scalar::<f32>()?;
        let checksum_sec = checksum_started.elapsed().as_secs_f64();
        let total_sec = total_started.elapsed().as_secs_f64();

        timings.push(IterationTiming {
            iter,
            rollout_sec,
            state_head_sec,
            denorm_slice_sec,
            checksum_sec,
            total_sec,
            checksum,
        });
    }

    let report = BenchmarkReport {
        dataset_dir,
        weights,
        config,
        row,
        sample_rate_hz: dataset.metadata().sample_rate_hz,
        history_steps: args.history_steps,
        model_history_size,
        horizon: args.horizon,
        samples: args.samples,
        warmup: args.warmup,
        iterations: args.iterations,
        action_scale: args.action_scale,
        observation_normalized: !args.no_observation_normalize,
        action_normalized: !args.no_action_normalize,
        target_normalized: !args.no_target_normalize,
        setup_sec,
        prep_sec,
        rollout_stats: TimingStats::from_samples(timings.iter().map(|t| t.rollout_sec)),
        state_head_stats: TimingStats::from_samples(timings.iter().map(|t| t.state_head_sec)),
        denorm_slice_stats: TimingStats::from_samples(timings.iter().map(|t| t.denorm_slice_sec)),
        checksum_stats: TimingStats::from_samples(timings.iter().map(|t| t.checksum_sec)),
        total_stats: TimingStats::from_samples(timings.iter().map(|t| t.total_sec)),
        candidate_rollouts_per_sec_mean: args.samples as f64
            / TimingStats::from_samples(timings.iter().map(|t| t.total_sec)).mean_sec,
        state_steps_per_sec_mean: (args.samples * args.horizon) as f64
            / TimingStats::from_samples(timings.iter().map(|t| t.total_sec)).mean_sec,
        timings,
    };
    write_pretty_json(&output, &report)?;

    println!(
        "drone rollout bench row={} samples={} horizon={} history={} iterations={}",
        row, args.samples, args.horizon, args.history_steps, args.iterations
    );
    println!(
        "total mean={:.6}s p50={:.6}s p90={:.6}s min={:.6}s max={:.6}s",
        report.total_stats.mean_sec,
        report.total_stats.p50_sec,
        report.total_stats.p90_sec,
        report.total_stats.min_sec,
        report.total_stats.max_sec
    );
    println!(
        "rollout mean={:.6}s state_head mean={:.6}s denorm_slice mean={:.6}s",
        report.rollout_stats.mean_sec,
        report.state_head_stats.mean_sec,
        report.denorm_slice_stats.mean_sec
    );
    println!(
        "candidate_rollouts_per_sec={:.1} state_steps_per_sec={:.1}",
        report.candidate_rollouts_per_sec_mean, report.state_steps_per_sec_mean
    );
    println!("wrote {}", output.display());
    Ok(())
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    ensure!(
        args.history_steps >= 2,
        "--history-steps must be at least two"
    );
    ensure!(args.horizon > 0, "--horizon must be greater than zero");
    ensure!(args.samples > 0, "--samples must be greater than zero");
    ensure!(
        args.iterations > 0,
        "--iterations must be greater than zero"
    );
    ensure!(
        args.action_scale >= 0.0,
        "--action-scale must be zero or greater"
    );
    Ok(())
}

fn timed<T>(device: &Device, f: impl FnOnce() -> candle::Result<T>) -> anyhow::Result<(T, f64)> {
    device.synchronize()?;
    let started = Instant::now();
    let value = f()?;
    device.synchronize()?;
    Ok((value, started.elapsed().as_secs_f64()))
}

fn prepare_rollout_tensors(
    emb: &Tensor,
    action_prefix: &Tensor,
    samples: usize,
    horizon: usize,
    action_scale: f32,
    action_stats: &RunningStats,
    action_normalized: bool,
    dtype: DType,
    device: &Device,
) -> anyhow::Result<PreparedTensors> {
    let baseline = baseline_action(action_stats)?;
    let action_values = deterministic_action_batch(samples, horizon, action_scale, baseline);
    let action_sequences = Tensor::from_vec(
        action_values,
        (1, samples, horizon, DRONE_ACTION_DIM),
        device,
    )?
    .to_dtype(dtype)?;
    let model_future_actions = if action_normalized {
        let action_mean = Tensor::from_vec(
            action_stats.mean.clone(),
            (1, 1, 1, DRONE_ACTION_DIM),
            device,
        )?
        .to_dtype(dtype)?;
        let action_std = Tensor::from_vec(
            action_stats
                .std
                .iter()
                .map(|value| value.max(1e-6))
                .collect::<Vec<_>>(),
            (1, 1, 1, DRONE_ACTION_DIM),
            device,
        )?
        .to_dtype(dtype)?;
        action_sequences
            .broadcast_sub(&action_mean)?
            .broadcast_div(&action_std)?
    } else {
        action_sequences
    };

    let (_, history, emb_dim) = emb.dims3()?;
    let (_, prefix_len, prefix_dim) = action_prefix.dims3()?;
    ensure!(
        prefix_len + 1 == history && prefix_dim == DRONE_ACTION_DIM,
        "action prefix shape {:?} does not match history={history}",
        action_prefix.shape()
    );
    let emb_init = emb
        .unsqueeze(1)?
        .broadcast_as((1, samples, history, emb_dim))?;
    let prefix =
        action_prefix
            .unsqueeze(1)?
            .broadcast_as((1, samples, prefix_len, DRONE_ACTION_DIM))?;
    let actions = Tensor::cat(&[&prefix, &model_future_actions], 2)?;
    Ok(PreparedTensors {
        emb_init,
        actions,
        samples,
        history,
        emb_dim,
    })
}

fn target_stats_tensors(
    target_stats: &RunningStats,
    target_normalized: bool,
    dtype: DType,
    device: &Device,
) -> anyhow::Result<TargetStatsTensors> {
    if !target_normalized {
        return Ok(TargetStatsTensors {
            mean: None,
            std: None,
        });
    }
    Ok(TargetStatsTensors {
        mean: Some(
            Tensor::from_vec(
                target_stats.mean.clone(),
                (1, 1, DRONE_STATE_DELTA_DIM),
                device,
            )?
            .to_dtype(dtype)?,
        ),
        std: Some(
            Tensor::from_vec(
                target_stats
                    .std
                    .iter()
                    .map(|value| value.max(1e-6))
                    .collect::<Vec<_>>(),
                (1, 1, DRONE_STATE_DELTA_DIM),
                device,
            )?
            .to_dtype(dtype)?,
        ),
    })
}

fn predict_future_state_deltas(
    model: &WorldModel,
    rollout: &Tensor,
    samples: usize,
    emb_dim: usize,
    history: usize,
    horizon: usize,
) -> candle::Result<Tensor> {
    let future = rollout
        .i((0, .., history..history + horizon, ..))?
        .contiguous()?;
    let flat = future.reshape((samples, horizon, emb_dim))?;
    model.predict_state_deltas_from_embeddings(&flat)
}

fn denormalize_deltas(pred: &Tensor, target_stats: &TargetStatsTensors) -> candle::Result<Tensor> {
    match (&target_stats.mean, &target_stats.std) {
        (Some(mean), Some(std)) => pred.broadcast_mul(std)?.broadcast_add(mean),
        _ => Ok(pred.clone()),
    }
}

fn deterministic_action_batch(
    samples: usize,
    horizon: usize,
    scale: f32,
    baseline: [f32; DRONE_ACTION_DIM],
) -> Vec<f32> {
    let mut rng = XorShift64::new(0x9e37_79b9_7f4a_7c15);
    let mut values = Vec::with_capacity(samples * horizon * DRONE_ACTION_DIM);
    for sample in 0..samples {
        for step in 0..horizon {
            let step_phase = step as f32 / horizon.max(1) as f32;
            for dim in 0..DRONE_ACTION_DIM {
                let wave = ((sample as f32 * 0.137) + (step_phase * 6.2831855) + dim as f32).sin();
                let noise = rng.next_f32() * 2.0 - 1.0;
                let value = if sample == 0 {
                    baseline[dim]
                } else {
                    baseline[dim] + scale * (0.70 * noise + 0.30 * wave)
                };
                values.push(clamp_action(dim, value));
            }
        }
    }
    values
}

fn clamp_action(dim: usize, value: f32) -> f32 {
    let (low, high) = match dim {
        2 => (0.0, 1.0),
        _ => (-1.0, 1.0),
    };
    value.clamp(low, high)
}

fn baseline_action(stats: &RunningStats) -> anyhow::Result<[f32; DRONE_ACTION_DIM]> {
    ensure!(
        stats.mean.len() == DRONE_ACTION_DIM,
        "action mean length {} does not match action dim {DRONE_ACTION_DIM}",
        stats.mean.len()
    );
    Ok([stats.mean[0], stats.mean[1], stats.mean[2], stats.mean[3]])
}

fn history_action_prefix(history_actions: &Tensor, history_steps: usize) -> anyhow::Result<Tensor> {
    ensure!(
        history_steps >= 2,
        "history action prefix requires at least two history steps"
    );
    let (batch, time, action_dim) = history_actions.dims3()?;
    ensure!(batch == 1, "history action prefix expects batch=1");
    ensure!(
        time >= history_steps,
        "history action tensor has time={time}, expected at least {history_steps}"
    );
    ensure!(
        action_dim == DRONE_ACTION_DIM,
        "history action_dim {action_dim} does not match expected {DRONE_ACTION_DIM}"
    );
    Ok(history_actions
        .i((0, 0..history_steps - 1, ..))?
        .unsqueeze(0)?)
}

fn default_dataset_dir() -> PathBuf {
    home_dir()
        .join(".stable_worldmodel")
        .join("le-wm-nv-data")
        .join("drone-racing-autonomous-100hz")
}

fn default_run_dir() -> PathBuf {
    home_dir()
        .join(".stable_worldmodel")
        .join("le-wm-nv-runs")
        .join("drone-state-lewm-all-data-20260612-235255")
}

fn default_weights() -> PathBuf {
    default_run_dir().join("final.safetensors")
}

fn default_config() -> PathBuf {
    default_run_dir().join("model-config.json")
}

fn default_output() -> PathBuf {
    PathBuf::from("target")
        .join("bench")
        .join("drone-lewm-rollout-h40-s528.json")
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn write_pretty_json<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(value)?;
    fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))
}

struct PreparedTensors {
    emb_init: Tensor,
    actions: Tensor,
    samples: usize,
    history: usize,
    emb_dim: usize,
}

struct TargetStatsTensors {
    mean: Option<Tensor>,
    std: Option<Tensor>,
}

#[derive(Debug, Clone, Copy)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_f32(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        let bits = (x >> 40) as u32;
        bits as f32 / 16_777_215.0
    }
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    dataset_dir: PathBuf,
    weights: PathBuf,
    config: PathBuf,
    row: usize,
    sample_rate_hz: usize,
    history_steps: usize,
    model_history_size: usize,
    horizon: usize,
    samples: usize,
    warmup: usize,
    iterations: usize,
    action_scale: f32,
    observation_normalized: bool,
    action_normalized: bool,
    target_normalized: bool,
    setup_sec: f64,
    prep_sec: f64,
    rollout_stats: TimingStats,
    state_head_stats: TimingStats,
    denorm_slice_stats: TimingStats,
    checksum_stats: TimingStats,
    total_stats: TimingStats,
    candidate_rollouts_per_sec_mean: f64,
    state_steps_per_sec_mean: f64,
    timings: Vec<IterationTiming>,
}

#[derive(Debug, Serialize)]
struct IterationTiming {
    iter: usize,
    rollout_sec: f64,
    state_head_sec: f64,
    denorm_slice_sec: f64,
    checksum_sec: f64,
    total_sec: f64,
    checksum: f32,
}

#[derive(Debug, Serialize)]
struct TimingStats {
    mean_sec: f64,
    min_sec: f64,
    p50_sec: f64,
    p90_sec: f64,
    max_sec: f64,
}

impl TimingStats {
    fn from_samples(samples: impl IntoIterator<Item = f64>) -> Self {
        let mut values = samples.into_iter().collect::<Vec<_>>();
        values.sort_by(|a, b| a.total_cmp(b));
        let mean_sec = values.iter().sum::<f64>() / values.len() as f64;
        Self {
            mean_sec,
            min_sec: values[0],
            p50_sec: percentile(&values, 0.50),
            p90_sec: percentile(&values, 0.90),
            max_sec: values[values.len() - 1],
        }
    }
}

fn percentile(values: &[f64], p: f64) -> f64 {
    let idx = ((values.len() - 1) as f64 * p).round() as usize;
    values[idx.min(values.len() - 1)]
}
