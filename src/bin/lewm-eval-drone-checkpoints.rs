use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, ensure};
use candle::{DType, IndexOp, Tensor};
use clap::Parser;
use le_wm_nv::{
    checkpoint,
    data::drone_racing::{DRONE_ACTION_DIM, DroneBatchConfig, DroneFrame, DroneRacingDataset},
    drone_eval::{
        HorizonErrorSummary, frame_error, history_action_prefix, integrate_future_deltas,
        rollout_deltas, summarize_errors,
    },
    models::world_model::{
        VectorLossScalars, VectorLossWeights, WorldModel, WorldModelConfig, vector_batch_loss,
    },
    runtime::{DTypeSpec, DeviceSpec},
};
use serde::Serialize;
use serde_json::Value;

#[derive(Parser, Debug)]
struct Args {
    /// Training run directory containing checkpoint-step-*.safetensors.
    #[arg(long)]
    run_dir: Option<PathBuf>,

    /// Imported drone dataset directory containing data.h5 and metadata.json.
    #[arg(long)]
    dataset_dir: Option<PathBuf>,

    /// WorldModel config JSON. Defaults to run_dir/model-config.json.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Output JSON report.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Evaluate only these checkpoint steps. Comma-separated values are accepted.
    #[arg(long, value_delimiter = ',')]
    steps: Vec<usize>,

    /// Include final.safetensors as the final training step.
    #[arg(long, default_value_t = true)]
    include_final: bool,

    #[arg(long, default_value_t = DeviceSpec::Cuda(0))]
    device: DeviceSpec,

    #[arg(long, default_value_t = DTypeSpec::F32)]
    dtype: DTypeSpec,

    #[arg(long, default_value_t = 8)]
    history_steps: usize,

    /// Maximum replay horizon in model steps.
    #[arg(long, default_value_t = 100)]
    horizon_steps: usize,

    /// Horizons to summarize from the replay. Comma-separated values are accepted.
    #[arg(long, value_delimiter = ',', default_values_t = [20usize, 40, 100])]
    report_horizons: Vec<usize>,

    #[arg(long, default_value_t = 256)]
    batch_size: usize,

    #[arg(long, default_value_t = 16)]
    max_batches: usize,

    /// Dataset row used for autoregressive replay. Defaults to highest-motion eval row.
    #[arg(long)]
    replay_row: Option<usize>,

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

    let run_dir = args.run_dir.clone().unwrap_or_else(default_run_dir);
    let dataset_dir = args.dataset_dir.clone().unwrap_or_else(default_dataset_dir);
    let config = args
        .config
        .clone()
        .unwrap_or_else(|| run_dir.join("model-config.json"));
    let output = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from("target/drone-eval/checkpoint-curve.json"));

    let cfg: WorldModelConfig = serde_json::from_str(
        &fs::read_to_string(&config)
            .with_context(|| format!("failed to read {}", config.display()))?,
    )
    .with_context(|| format!("failed to parse {}", config.display()))?;
    let sequence_steps = cfg.predictor.num_frames;
    let batch_cfg = DroneBatchConfig {
        batch_size: args.batch_size,
        sequence_steps,
        normalize_observations: !args.no_observation_normalize,
        normalize_actions: !args.no_action_normalize,
        normalize_targets: !args.no_target_normalize,
    };
    let dataset = DroneRacingDataset::open(&dataset_dir, batch_cfg)?;
    let eval_rows = dataset.eval_rows();
    ensure!(!eval_rows.is_empty(), "dataset has no eval rows");
    let replay_row = match args.replay_row {
        Some(row) => {
            ensure!(
                replay_row_has_horizon(&dataset, row, args.history_steps, args.horizon_steps)?,
                "--replay-row {row} does not have {} future steps in one episode",
                args.horizon_steps
            );
            row
        }
        None => select_replay_row(&dataset, &eval_rows, args.history_steps, args.horizon_steps)?,
    };
    let replay_start = dataset.frame(replay_row + args.history_steps - 1)?;
    let checkpoints = discover_checkpoints(&run_dir, &args.steps, args.include_final)?;
    ensure!(!checkpoints.is_empty(), "no checkpoints selected");
    let elapsed_by_step = training_elapsed_by_step(&run_dir)?;
    let device = args.device.resolve()?;
    ensure!(device.is_cuda(), "checkpoint curve requires CUDA");
    let dtype = args.dtype.dtype();
    if dtype != DType::F32 {
        anyhow::bail!("checkpoint curve currently requires --dtype f32");
    }

    let mut results = Vec::with_capacity(checkpoints.len());
    for checkpoint in checkpoints {
        let started = Instant::now();
        let vb = checkpoint::var_builder_from_path(&checkpoint.path, dtype, &device)
            .with_context(|| format!("failed to load {}", checkpoint.path.display()))?;
        let model = WorldModel::new(cfg.clone(), vb)?;
        let eval_loss = evaluate_batches(
            &model,
            &dataset,
            &eval_rows,
            args.max_batches,
            dtype,
            &device,
        )?;
        let replay = evaluate_replay(
            &model,
            &dataset,
            replay_row,
            args.history_steps,
            args.horizon_steps,
            &args.report_horizons,
            dtype,
            &device,
            !args.no_action_normalize,
            !args.no_target_normalize,
        )?;
        let eval_elapsed_sec = started.elapsed().as_secs_f64();
        println!(
            "checkpoint step={} elapsed_train={:.2}s loss={:.6} h{}pos_rms={:.4} eval_sec={:.2}",
            checkpoint.step,
            elapsed_by_step
                .get(&checkpoint.step)
                .copied()
                .unwrap_or_default(),
            eval_loss.state_prediction,
            args.report_horizons
                .iter()
                .copied()
                .max()
                .unwrap_or(args.horizon_steps),
            replay
                .summaries
                .last()
                .map(|summary| summary.error.position_rms_m)
                .unwrap_or_default(),
            eval_elapsed_sec
        );
        results.push(CheckpointResult {
            step: checkpoint.step,
            elapsed_train_sec: elapsed_by_step.get(&checkpoint.step).copied(),
            weights: checkpoint.path,
            eval_elapsed_sec,
            mean_loss: eval_loss,
            replay,
        });
    }

    let report = CheckpointCurveReport {
        run_dir,
        dataset_dir,
        config,
        output: output.clone(),
        eval_rows: eval_rows.len(),
        eval_batches: args
            .max_batches
            .min(eval_rows.len().div_ceil(args.batch_size)),
        batch_size: args.batch_size,
        history_steps: args.history_steps,
        horizon_steps: args.horizon_steps,
        report_horizons: args.report_horizons.clone(),
        sample_rate_hz: dataset.metadata().sample_rate_hz,
        replay_row,
        replay_start,
        checkpoints: results,
    };
    write_pretty_json(&output, &report)?;
    println!("wrote {}", output.display());
    Ok(())
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    ensure!(
        args.history_steps >= 2,
        "--history-steps must be at least two"
    );
    ensure!(
        args.horizon_steps > 0,
        "--horizon-steps must be greater than zero"
    );
    ensure!(
        args.batch_size > 0,
        "--batch-size must be greater than zero"
    );
    ensure!(
        args.max_batches > 0,
        "--max-batches must be greater than zero"
    );
    ensure!(
        !args.report_horizons.is_empty(),
        "--report-horizons must not be empty"
    );
    ensure!(
        args.report_horizons
            .iter()
            .all(|horizon| *horizon > 0 && *horizon <= args.horizon_steps),
        "all --report-horizons entries must be in 1..=horizon_steps"
    );
    Ok(())
}

fn evaluate_batches(
    model: &WorldModel,
    dataset: &DroneRacingDataset,
    eval_rows: &[usize],
    max_batches: usize,
    dtype: DType,
    device: &candle::Device,
) -> anyhow::Result<VectorLossScalars> {
    let loss_weights = VectorLossWeights {
        state_prediction: 1.0,
        temporal_alignment: 0.0,
        std: 0.0,
        std_t: 0.0,
        covariance: 0.0,
        covariance_t: 0.0,
        temporal_straightening: 0.0,
    };
    let mut total = LossTotals::default();
    let mut batches = 0usize;
    for chunk in eval_rows
        .chunks(dataset.config().batch_size)
        .take(max_batches)
    {
        let batch = dataset.batch(chunk, dtype, device)?;
        let loss = vector_batch_loss(
            model,
            &batch.observations,
            &batch.actions,
            &batch.target_deltas,
            loss_weights,
        )?;
        total.push(&VectorLossScalars::from_loss(&loss)?);
        batches += 1;
    }
    ensure!(batches > 0, "no eval batches were run");
    Ok(total.mean(batches))
}

fn evaluate_replay(
    model: &WorldModel,
    dataset: &DroneRacingDataset,
    row: usize,
    history_steps: usize,
    horizon_steps: usize,
    report_horizons: &[usize],
    dtype: DType,
    device: &candle::Device,
    action_normalized: bool,
    target_normalized: bool,
) -> anyhow::Result<ReplayCurveResult> {
    let batch = dataset.batch(&[row], dtype, device)?;
    let emb_all = model.encode_vector(&batch.observations)?;
    let emb = emb_all.i((.., 0..history_steps, ..))?;
    let action_prefix = history_action_prefix(&batch.actions, history_steps)?;
    let current = dataset.frame(row + history_steps - 1)?;
    let future_actions =
        future_action_tensor(dataset, row, history_steps, horizon_steps, dtype, device)?;
    let started = Instant::now();
    let deltas = rollout_deltas(
        model,
        &emb,
        &action_prefix,
        &future_actions,
        &dataset.metadata().normalization.action,
        &dataset.metadata().normalization.target_delta,
        action_normalized,
        target_normalized,
        dtype,
        device,
    )?;
    let rollout_elapsed_sec = started.elapsed().as_secs_f64();
    let delta_values = deltas.i(0)?.flatten_all()?.to_vec1::<f32>()?;
    let predicted = integrate_future_deltas(&current, &delta_values, horizon_steps)?;
    let actual = (0..=horizon_steps)
        .map(|offset| dataset.frame(row + history_steps - 1 + offset))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let future_errors = (1..=horizon_steps)
        .map(|idx| frame_error(&actual[idx], &predicted[idx]))
        .collect::<Vec<_>>();
    let summaries = report_horizons
        .iter()
        .copied()
        .map(|horizon| HorizonReplaySummary {
            horizon_steps: horizon,
            error: summarize_errors(&future_errors[..horizon], dataset.metadata().sample_rate_hz),
        })
        .collect();
    Ok(ReplayCurveResult {
        rollout_elapsed_sec,
        summaries,
    })
}

fn future_action_tensor(
    dataset: &DroneRacingDataset,
    row: usize,
    history_steps: usize,
    horizon_steps: usize,
    dtype: DType,
    device: &candle::Device,
) -> anyhow::Result<Tensor> {
    let mut values = Vec::with_capacity(horizon_steps * DRONE_ACTION_DIM);
    let start = row + history_steps - 1;
    for step in 0..horizon_steps {
        values.extend_from_slice(&dataset.frame(start + step)?.channels_norm);
    }
    Ok(
        Tensor::from_vec(values, (1, 1, horizon_steps, DRONE_ACTION_DIM), device)?
            .to_dtype(dtype)?,
    )
}

fn select_replay_row(
    dataset: &DroneRacingDataset,
    rows: &[usize],
    history_steps: usize,
    horizon_steps: usize,
) -> anyhow::Result<usize> {
    rows.iter()
        .copied()
        .filter_map(|row| {
            replay_row_has_horizon(dataset, row, history_steps, horizon_steps)
                .ok()
                .filter(|valid| *valid)
                .map(|_| row)
        })
        .map(|row| {
            Ok((
                row,
                replay_path_length(dataset, row, history_steps, horizon_steps)?,
            ))
        })
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
        .map(|(row, _)| row)
        .context("failed to select replay row")
}

fn replay_row_has_horizon(
    dataset: &DroneRacingDataset,
    row: usize,
    history_steps: usize,
    horizon_steps: usize,
) -> anyhow::Result<bool> {
    let start = dataset.frame(row)?;
    let end_row = row + history_steps - 1 + horizon_steps;
    if end_row >= dataset.metadata().rows {
        return Ok(false);
    }
    Ok(dataset.frame(end_row)?.episode_idx == start.episode_idx)
}

fn replay_path_length(
    dataset: &DroneRacingDataset,
    row: usize,
    history_steps: usize,
    horizon_steps: usize,
) -> anyhow::Result<f32> {
    let start = row + history_steps - 1;
    let mut total = 0.0f32;
    let mut prev = dataset.frame(start)?.pos_world;
    for idx in 1..=horizon_steps {
        let current = dataset.frame(start + idx)?.pos_world;
        total +=
            le_wm_nv::data::drone_racing::norm3(le_wm_nv::data::drone_racing::sub3(current, prev));
        prev = current;
    }
    Ok(total)
}

fn discover_checkpoints(
    run_dir: &Path,
    selected_steps: &[usize],
    include_final: bool,
) -> anyhow::Result<Vec<CheckpointInfo>> {
    let selected = selected_steps.iter().copied().collect::<BTreeSet<_>>();
    let mut out = Vec::new();
    for entry in
        fs::read_dir(run_dir).with_context(|| format!("failed to read {}", run_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        let Some(step) = parse_checkpoint_step(name) else {
            continue;
        };
        if selected.is_empty() || selected.contains(&step) {
            out.push(CheckpointInfo { step, path });
        }
    }
    if include_final {
        let final_path = run_dir.join("final.safetensors");
        if final_path.exists() {
            let step = training_final_step(run_dir)?.unwrap_or_else(|| {
                out.iter()
                    .map(|checkpoint| checkpoint.step)
                    .max()
                    .unwrap_or_default()
            });
            if selected.is_empty() || selected.contains(&step) {
                out.push(CheckpointInfo {
                    step,
                    path: final_path,
                });
            }
        }
    }
    out.sort_by_key(|checkpoint| checkpoint.step);
    out.dedup_by(|lhs, rhs| lhs.step == rhs.step && lhs.path == rhs.path);
    Ok(out)
}

fn parse_checkpoint_step(name: &str) -> Option<usize> {
    name.strip_prefix("checkpoint-step-")?
        .strip_suffix(".safetensors")?
        .parse()
        .ok()
}

fn training_elapsed_by_step(run_dir: &Path) -> anyhow::Result<BTreeMap<usize, f64>> {
    let path = run_dir.join("metrics.jsonl");
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let mut out = BTreeMap::new();
    for line in fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?
        .lines()
    {
        let value: Value = serde_json::from_str(line)?;
        let Some(step) = value.get("step").and_then(Value::as_u64) else {
            continue;
        };
        let Some(elapsed) = value.get("elapsed_sec").and_then(Value::as_f64) else {
            continue;
        };
        out.insert(step as usize, elapsed);
    }
    Ok(out)
}

fn training_final_step(run_dir: &Path) -> anyhow::Result<Option<usize>> {
    let path = run_dir.join("training-state.json");
    if !path.exists() {
        return Ok(None);
    }
    let value: Value = serde_json::from_str(
        &fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?,
    )?;
    Ok(value
        .get("global_step")
        .and_then(Value::as_u64)
        .map(|step| step as usize))
}

fn write_pretty_json<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(value)?;
    fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))
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

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

#[derive(Debug)]
struct CheckpointInfo {
    step: usize,
    path: PathBuf,
}

#[derive(Default)]
struct LossTotals {
    total: f64,
    state_prediction: f64,
    temporal_alignment: f64,
    std: f64,
    std_t: f64,
    covariance: f64,
    covariance_t: f64,
    temporal_straightening: f64,
}

impl LossTotals {
    fn push(&mut self, loss: &VectorLossScalars) {
        self.total += f64::from(loss.total);
        self.state_prediction += f64::from(loss.state_prediction);
        self.temporal_alignment += f64::from(loss.temporal_alignment);
        self.std += f64::from(loss.std);
        self.std_t += f64::from(loss.std_t);
        self.covariance += f64::from(loss.covariance);
        self.covariance_t += f64::from(loss.covariance_t);
        self.temporal_straightening += f64::from(loss.temporal_straightening);
    }

    fn mean(self, n: usize) -> VectorLossScalars {
        let n = n as f64;
        VectorLossScalars {
            total: (self.total / n) as f32,
            state_prediction: (self.state_prediction / n) as f32,
            temporal_alignment: (self.temporal_alignment / n) as f32,
            std: (self.std / n) as f32,
            std_t: (self.std_t / n) as f32,
            covariance: (self.covariance / n) as f32,
            covariance_t: (self.covariance_t / n) as f32,
            temporal_straightening: (self.temporal_straightening / n) as f32,
        }
    }
}

#[derive(Debug, Serialize)]
struct CheckpointCurveReport {
    run_dir: PathBuf,
    dataset_dir: PathBuf,
    config: PathBuf,
    output: PathBuf,
    eval_rows: usize,
    eval_batches: usize,
    batch_size: usize,
    history_steps: usize,
    horizon_steps: usize,
    report_horizons: Vec<usize>,
    sample_rate_hz: usize,
    replay_row: usize,
    replay_start: DroneFrame,
    checkpoints: Vec<CheckpointResult>,
}

#[derive(Debug, Serialize)]
struct CheckpointResult {
    step: usize,
    elapsed_train_sec: Option<f64>,
    weights: PathBuf,
    eval_elapsed_sec: f64,
    mean_loss: VectorLossScalars,
    replay: ReplayCurveResult,
}

#[derive(Debug, Serialize)]
struct ReplayCurveResult {
    rollout_elapsed_sec: f64,
    summaries: Vec<HorizonReplaySummary>,
}

#[derive(Debug, Serialize)]
struct HorizonReplaySummary {
    horizon_steps: usize,
    error: HorizonErrorSummary,
}
