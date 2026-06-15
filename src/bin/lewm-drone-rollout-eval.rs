use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, ensure};
use candle::{DType, Tensor};
use clap::Parser;
use le_wm_nv::{
    checkpoint,
    data::drone_racing::{
        DRONE_ACTION_DIM, DRONE_OBSERVATION_DIM, DroneBatchConfig, DroneNormalization,
        DroneRacingDataset, RunningStats,
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

    /// First dataset row predicted by the rollout. Previous history rows are used as context.
    #[arg(long, default_value_t = 1020)]
    start_row: usize,

    /// Number of future rows to compare.
    #[arg(long, default_value_t = 300)]
    horizon_steps: usize,

    /// Print per-step metrics every N rollout steps.
    #[arg(long, default_value_t = 25)]
    report_every: usize,

    /// Optional JSON report path.
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
    model_cfg.validate()?;
    ensure!(
        model_cfg.action_encoder.input_dim == DRONE_ACTION_DIM,
        "model action dim {} does not match drone action dim {DRONE_ACTION_DIM}",
        model_cfg.action_encoder.input_dim
    );
    match &model_cfg.observation_encoder {
        ObservationEncoderConfig::VectorMlp(cfg) => ensure!(
            cfg.input_dim == DRONE_OBSERVATION_DIM,
            "model observation dim {} does not match drone observation dim {DRONE_OBSERVATION_DIM}",
            cfg.input_dim
        ),
        ObservationEncoderConfig::ImageVit { .. } => {
            anyhow::bail!("lewm-drone-rollout-eval requires a vector-observation WorldModel")
        }
    }

    let history_size = model_cfg.history_size;
    ensure!(
        args.start_row >= history_size,
        "--start-row {} must be at least model history_size {}",
        args.start_row,
        history_size
    );
    let history_start = args.start_row - history_size;
    let sequence_steps = history_size
        .checked_add(args.horizon_steps)
        .context("history_size + horizon_steps overflowed")?;
    let batch_cfg = DroneBatchConfig {
        batch_size: 1,
        sequence_steps,
        normalize_observations: true,
        normalize_actions: true,
    };
    let dataset = DroneRacingDataset::open(&dataset_dir, batch_cfg)?;
    let run_normalization: DroneNormalization = read_json(&normalization_path)?;
    ensure_normalization_matches(&run_normalization, &dataset.metadata().normalization)?;
    ensure_same_episode(&dataset, history_start, sequence_steps)?;

    let device = args.device.resolve()?;
    let dtype = args.dtype.dtype();
    let vb = checkpoint::var_builder_from_path(&weights, dtype, &device)
        .with_context(|| format!("failed to load weights {}", weights.display()))?;
    let model = WorldModel::new(model_cfg, vb)?;

    let batch = dataset.batch(&[history_start], dtype, &device)?;
    let actual_emb = model.encode_vector(&batch.observations)?;
    let action_steps = sequence_steps - 1;
    let actions = batch.actions.narrow(1, 0, action_steps)?;
    let history_emb = actual_emb.narrow(1, 0, history_size)?.unsqueeze(1)?;
    let action_candidates = actions.unsqueeze(1)?;
    let rollout =
        model.rollout_embeddings_with_history(&history_emb, &action_candidates, history_size)?;

    let rollout_flat = rollout.squeeze(0)?.squeeze(0)?;
    let actual_flat = actual_emb.squeeze(0)?;
    let autoreg_pred = rollout_flat.narrow(0, history_size, args.horizon_steps)?;
    let actual_future = actual_flat.narrow(0, history_size, args.horizon_steps)?;

    let one_step_pred = sliding_one_step_predictions(
        &model,
        &actual_flat,
        &actions.squeeze(0)?,
        history_size,
        args.horizon_steps,
    )?;

    let autoreg_steps = step_metrics(&autoreg_pred, &actual_future, args.start_row)?;
    let one_step_steps = step_metrics(&one_step_pred, &actual_future, args.start_row)?;
    let autoreg = aggregate(&autoreg_steps);
    let one_step = aggregate(&one_step_steps);
    device.synchronize()?;

    let report = RolloutReport {
        model_dir: args.model_dir.clone(),
        dataset_dir: dataset_dir.clone(),
        weights,
        config: config_path,
        normalization: normalization_path,
        device: args.device.to_string(),
        dtype: args.dtype.to_string(),
        history_size,
        start_row: args.start_row,
        history_start_row: history_start,
        horizon_steps: args.horizon_steps,
        predicted_row_start: args.start_row,
        predicted_row_end_inclusive: args.start_row + args.horizon_steps - 1,
        elapsed_sec: started.elapsed().as_secs_f64(),
        one_step,
        autoregressive: autoreg,
        samples: sample_steps(&one_step_steps, &autoreg_steps, args.report_every),
    };

    print_report(&report);
    if let Some(path) = args.json_out.as_ref() {
        write_pretty_json(path, &report)?;
        println!("json={}", path.display());
    }
    Ok(())
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    ensure!(
        args.horizon_steps > 0,
        "--horizon-steps must be greater than zero"
    );
    ensure!(
        args.report_every > 0,
        "--report-every must be greater than zero"
    );
    Ok(())
}

fn default_dataset_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".stable_worldmodel")
        .join("le-wm-nv-data")
        .join("drone-racing-autonomous-100hz-pose16")
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
    let bytes = serde_json::to_vec_pretty(value)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))
}

fn ensure_normalization_matches(
    run: &DroneNormalization,
    dataset: &DroneNormalization,
) -> anyhow::Result<()> {
    ensure_stats_match("observation", &run.observation, &dataset.observation)?;
    ensure_stats_match("action", &run.action, &dataset.action)?;
    Ok(())
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

fn ensure_same_episode(
    dataset: &DroneRacingDataset,
    history_start: usize,
    sequence_steps: usize,
) -> anyhow::Result<()> {
    let first = dataset.frame(history_start)?;
    for offset in 1..sequence_steps {
        let frame = dataset.frame(history_start + offset)?;
        ensure!(
            frame.episode_idx == first.episode_idx,
            "sequence crosses episode boundary at row {}",
            history_start + offset
        );
        ensure!(
            frame.step_idx == first.step_idx + offset as i64,
            "sequence has a step gap at row {}",
            history_start + offset
        );
    }
    Ok(())
}

fn step_metrics(
    pred: &Tensor,
    actual: &Tensor,
    start_row: usize,
) -> anyhow::Result<Vec<StepMetric>> {
    let pred = pred.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let actual = actual.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    ensure!(
        pred.len() == actual.len(),
        "prediction/actual time mismatch: {} vs {}",
        pred.len(),
        actual.len()
    );
    let mut out = Vec::with_capacity(pred.len());
    for (idx, (pred_row, actual_row)) in pred.iter().zip(actual.iter()).enumerate() {
        ensure!(
            pred_row.len() == actual_row.len(),
            "prediction/actual dim mismatch at step {idx}: {} vs {}",
            pred_row.len(),
            actual_row.len()
        );
        out.push(metric_for_pair(start_row + idx, pred_row, actual_row));
    }
    Ok(out)
}

fn sliding_one_step_predictions(
    model: &WorldModel,
    actual_emb: &Tensor,
    actions: &Tensor,
    history_size: usize,
    horizon_steps: usize,
) -> anyhow::Result<Tensor> {
    let mut emb_windows = Vec::with_capacity(horizon_steps);
    let mut action_windows = Vec::with_capacity(horizon_steps);
    for step in 0..horizon_steps {
        emb_windows.push(actual_emb.narrow(0, step, history_size)?);
        action_windows.push(actions.narrow(0, step, history_size)?);
    }
    let emb_refs = emb_windows.iter().collect::<Vec<_>>();
    let action_refs = action_windows.iter().collect::<Vec<_>>();
    let emb = Tensor::stack(&emb_refs, 0)?;
    let actions = Tensor::stack(&action_refs, 0)?;
    let pred = model.predict(&emb, &actions)?;
    Ok(pred.narrow(1, history_size - 1, 1)?.squeeze(1)?)
}

fn metric_for_pair(row: usize, pred: &[f32], actual: &[f32]) -> StepMetric {
    let mut sq = 0.0f64;
    let mut dot = 0.0f64;
    let mut pred_norm_sq = 0.0f64;
    let mut actual_norm_sq = 0.0f64;
    for (p, a) in pred.iter().zip(actual.iter()) {
        let p = f64::from(*p);
        let a = f64::from(*a);
        let d = p - a;
        sq += d * d;
        dot += p * a;
        pred_norm_sq += p * p;
        actual_norm_sq += a * a;
    }
    let dim = pred.len().max(1) as f64;
    let mse = (sq / dim) as f32;
    let l2 = sq.sqrt() as f32;
    let denom = (pred_norm_sq.sqrt() * actual_norm_sq.sqrt()).max(1e-12);
    StepMetric {
        row,
        mse,
        rmse: mse.sqrt(),
        l2,
        cosine: (dot / denom) as f32,
        pred_norm: pred_norm_sq.sqrt() as f32,
        actual_norm: actual_norm_sq.sqrt() as f32,
    }
}

fn aggregate(steps: &[StepMetric]) -> AggregateMetric {
    let len = steps.len().max(1) as f32;
    let mean_mse = steps.iter().map(|s| s.mse).sum::<f32>() / len;
    let mean_rmse = steps.iter().map(|s| s.rmse).sum::<f32>() / len;
    let mean_l2 = steps.iter().map(|s| s.l2).sum::<f32>() / len;
    let mean_cosine = steps.iter().map(|s| s.cosine).sum::<f32>() / len;
    let final_step = steps.last().cloned().unwrap_or_default();
    let max_l2 = steps.iter().map(|s| s.l2).fold(0.0, f32::max);
    AggregateMetric {
        mean_mse,
        mean_rmse,
        mean_l2,
        mean_cosine,
        final_mse: final_step.mse,
        final_rmse: final_step.rmse,
        final_l2: final_step.l2,
        final_cosine: final_step.cosine,
        max_l2,
    }
}

fn sample_steps(
    one_step: &[StepMetric],
    autoreg: &[StepMetric],
    report_every: usize,
) -> Vec<SampleRow> {
    let mut rows = Vec::new();
    for idx in 0..autoreg.len() {
        if idx == 0 || (idx + 1) % report_every == 0 || idx + 1 == autoreg.len() {
            rows.push(SampleRow {
                step: idx + 1,
                row: autoreg[idx].row,
                one_step_l2: one_step[idx].l2,
                one_step_cosine: one_step[idx].cosine,
                autoreg_l2: autoreg[idx].l2,
                autoreg_cosine: autoreg[idx].cosine,
            });
        }
    }
    rows
}

fn print_report(report: &RolloutReport) {
    println!(
        "rollout_eval model={} dataset={} history={} start_row={} horizon={} rows={}..{} elapsed={:.3}s",
        report.model_dir.display(),
        report.dataset_dir.display(),
        report.history_size,
        report.start_row,
        report.horizon_steps,
        report.predicted_row_start,
        report.predicted_row_end_inclusive,
        report.elapsed_sec,
    );
    println!(
        "one_step mean_mse={:.6e} mean_l2={:.4} mean_cos={:.4} final_l2={:.4} final_cos={:.4} max_l2={:.4}",
        report.one_step.mean_mse,
        report.one_step.mean_l2,
        report.one_step.mean_cosine,
        report.one_step.final_l2,
        report.one_step.final_cosine,
        report.one_step.max_l2,
    );
    println!(
        "autoregressive mean_mse={:.6e} mean_l2={:.4} mean_cos={:.4} final_l2={:.4} final_cos={:.4} max_l2={:.4}",
        report.autoregressive.mean_mse,
        report.autoregressive.mean_l2,
        report.autoregressive.mean_cosine,
        report.autoregressive.final_l2,
        report.autoregressive.final_cosine,
        report.autoregressive.max_l2,
    );
    println!(
        "{:<8} {:<8} {:>12} {:>12} {:>12} {:>12}",
        "step", "row", "one_l2", "one_cos", "ar_l2", "ar_cos"
    );
    for sample in &report.samples {
        println!(
            "{:<8} {:<8} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
            sample.step,
            sample.row,
            sample.one_step_l2,
            sample.one_step_cosine,
            sample.autoreg_l2,
            sample.autoreg_cosine,
        );
    }
}

#[derive(Debug, Clone, Default, Serialize)]
struct StepMetric {
    row: usize,
    mse: f32,
    rmse: f32,
    l2: f32,
    cosine: f32,
    pred_norm: f32,
    actual_norm: f32,
}

#[derive(Debug, Clone, Serialize)]
struct AggregateMetric {
    mean_mse: f32,
    mean_rmse: f32,
    mean_l2: f32,
    mean_cosine: f32,
    final_mse: f32,
    final_rmse: f32,
    final_l2: f32,
    final_cosine: f32,
    max_l2: f32,
}

#[derive(Debug, Clone, Serialize)]
struct SampleRow {
    step: usize,
    row: usize,
    one_step_l2: f32,
    one_step_cosine: f32,
    autoreg_l2: f32,
    autoreg_cosine: f32,
}

#[derive(Debug, Serialize)]
struct RolloutReport {
    model_dir: PathBuf,
    dataset_dir: PathBuf,
    weights: PathBuf,
    config: PathBuf,
    normalization: PathBuf,
    device: String,
    dtype: String,
    history_size: usize,
    start_row: usize,
    history_start_row: usize,
    horizon_steps: usize,
    predicted_row_start: usize,
    predicted_row_end_inclusive: usize,
    elapsed_sec: f64,
    one_step: AggregateMetric,
    autoregressive: AggregateMetric,
    samples: Vec<SampleRow>,
}
