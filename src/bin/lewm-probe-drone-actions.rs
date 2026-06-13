use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, ensure};
use candle::{DType, IndexOp, Tensor};
use clap::Parser;
use le_wm_nv::{
    checkpoint,
    data::drone_racing::{
        DRONE_ACTION_DIM, DRONE_STATE_DELTA_DIM, DroneBatchConfig, DroneFrame, DroneRacingDataset,
        RunningStats, add3, mat3_from_rotvec, mat3_mul, mat3_mul_vec3, norm3, sub3,
    },
    models::world_model::{WorldModel, WorldModelConfig},
    runtime::{DTypeSpec, DeviceSpec},
};
use serde::Serialize;

const ACTION_NAMES: [&str; DRONE_ACTION_DIM] = ["roll", "pitch", "throttle", "yaw"];

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

    /// Output sensitivity JSON.
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

    #[arg(long, default_value_t = 80)]
    horizon: usize,

    /// Number of values per action dimension. Values span the valid action range.
    #[arg(long, default_value_t = 9)]
    sweep_steps: usize,

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
    let device = args.device.resolve()?;
    ensure!(
        device.is_cuda(),
        "drone action sensitivity requires a CUDA device"
    );
    let dtype = args.dtype.dtype();
    if dtype != DType::F32 {
        anyhow::bail!("drone action sensitivity currently requires --dtype f32");
    }
    let vb = checkpoint::var_builder_from_path(&weights, dtype, &device)
        .with_context(|| format!("failed to load {}", weights.display()))?;
    let model = WorldModel::new(cfg, vb)?;
    let history = dataset.batch(&[row], dtype, &device)?;
    let emb = model.encode_vector(&history.observations)?;
    let action_prefix = history_action_prefix(&history.actions, args.history_steps)?;
    let current = dataset.frame(row + args.history_steps - 1)?;
    let baseline_action = baseline_action(&dataset.metadata().normalization.action)?;
    let cases = build_cases(baseline_action, args.sweep_steps);

    let started = Instant::now();
    let deltas = rollout_case_deltas(
        &model,
        &emb,
        &action_prefix,
        &cases,
        args.horizon,
        &dataset.metadata().normalization.action,
        &dataset.metadata().normalization.target_delta,
        !args.no_action_normalize,
        !args.no_target_normalize,
        dtype,
        &device,
    )?;
    let rollout_elapsed_sec = started.elapsed().as_secs_f64();
    let case_results = integrate_cases(&current, &cases, &deltas, args.horizon)?;
    let dimensions = summarize_dimensions(&case_results, args.sweep_steps);
    let report = SensitivityReport {
        dataset_dir,
        weights,
        config,
        row,
        history_steps: args.history_steps,
        horizon: args.horizon,
        sweep_steps: args.sweep_steps,
        sample_rate_hz: dataset.metadata().sample_rate_hz,
        baseline_action,
        current,
        rollout_elapsed_sec,
        cases: case_results,
        dimensions,
    };
    write_pretty_json(&output, &report)?;

    println!(
        "action sensitivity row={} cases={} horizon={} rollout_sec={:.3}",
        row,
        report.cases.len(),
        args.horizon,
        rollout_elapsed_sec
    );
    for dim in &report.dimensions {
        println!(
            "{} span_pos_m={:.4} span_z_m={:.4} span_speed_mps={:.4}",
            dim.action_name, dim.final_pos_span_m, dim.final_z_span_m, dim.final_speed_span_mps
        );
    }
    println!("wrote {}", output.display());
    Ok(())
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    ensure!(
        args.history_steps >= 2,
        "--history-steps must be at least two"
    );
    ensure!(args.horizon > 0, "--horizon must be greater than zero");
    ensure!(args.sweep_steps >= 2, "--sweep-steps must be at least two");
    Ok(())
}

fn rollout_case_deltas(
    model: &WorldModel,
    emb: &Tensor,
    action_prefix: &Tensor,
    cases: &[ActionCase],
    horizon: usize,
    action_stats: &RunningStats,
    target_stats: &RunningStats,
    action_normalized: bool,
    target_normalized: bool,
    dtype: DType,
    device: &candle::Device,
) -> anyhow::Result<Vec<f32>> {
    let cases_len = cases.len();
    let mut action_values = Vec::with_capacity(cases_len * horizon * DRONE_ACTION_DIM);
    for case in cases {
        for _ in 0..horizon {
            action_values.extend_from_slice(&case.action);
        }
    }
    let action_sequences = Tensor::from_vec(
        action_values,
        (1, cases_len, horizon, DRONE_ACTION_DIM),
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
        .broadcast_as((1, cases_len, history, emb_dim))?;
    let prefix =
        action_prefix
            .unsqueeze(1)?
            .broadcast_as((1, cases_len, prefix_len, DRONE_ACTION_DIM))?;
    let model_actions = Tensor::cat(&[&prefix, &model_future_actions], 2)?;
    let rollout = model.rollout_embeddings_with_history(&emb_init, &model_actions, history)?;
    let rollout_time = rollout.dim(2)?;
    let pred = model.predict_state_deltas_from_embeddings(&rollout.reshape((
        cases_len,
        rollout_time,
        emb_dim,
    ))?)?;
    let deltas = if target_normalized {
        let target_mean = Tensor::from_vec(
            target_stats.mean.clone(),
            (1, 1, DRONE_STATE_DELTA_DIM),
            device,
        )?
        .to_dtype(dtype)?;
        let target_std = Tensor::from_vec(
            target_stats
                .std
                .iter()
                .map(|value| value.max(1e-6))
                .collect::<Vec<_>>(),
            (1, 1, DRONE_STATE_DELTA_DIM),
            device,
        )?
        .to_dtype(dtype)?;
        pred.broadcast_mul(&target_std)?
            .broadcast_add(&target_mean)?
    } else {
        pred
    };
    let deltas = deltas.i((.., history..history + horizon, ..))?;
    Ok(deltas.flatten_all()?.to_vec1::<f32>()?)
}

fn integrate_cases(
    current: &DroneFrame,
    cases: &[ActionCase],
    deltas: &[f32],
    horizon: usize,
) -> anyhow::Result<Vec<CaseResult>> {
    let expected = cases.len() * horizon * DRONE_STATE_DELTA_DIM;
    ensure!(
        deltas.len() == expected,
        "delta buffer has {}, expected {expected}",
        deltas.len()
    );
    let mut out = Vec::with_capacity(cases.len());
    for (case_idx, case) in cases.iter().enumerate() {
        let mut frame = current.clone();
        let mut path_length_m = 0.0f32;
        for step in 0..horizon {
            let offset = (case_idx * horizon + step) * DRONE_STATE_DELTA_DIM;
            let delta = array13(&deltas[offset..offset + DRONE_STATE_DELTA_DIM]);
            let next = apply_delta(&frame, &delta);
            path_length_m += norm3(sub3(next.pos_world, frame.pos_world));
            frame = next;
        }
        let displacement_world = sub3(frame.pos_world, current.pos_world);
        out.push(CaseResult {
            name: case.name.clone(),
            action_dim: case.action_dim,
            action_name: case.action_name.clone(),
            action_value: case.action_value,
            action: case.action,
            final_pos_world: frame.pos_world,
            displacement_world,
            final_lin_vel_body: frame.lin_vel_body,
            final_speed_mps: norm3(frame.lin_vel_body),
            net_distance_m: norm3(displacement_world),
            path_length_m,
            altitude_delta_m: frame.pos_world[2] - current.pos_world[2],
        });
    }
    Ok(out)
}

fn summarize_dimensions(cases: &[CaseResult], sweep_steps: usize) -> Vec<DimensionSummary> {
    let mut summaries = Vec::with_capacity(DRONE_ACTION_DIM);
    for dim in 0..DRONE_ACTION_DIM {
        let values = cases
            .iter()
            .filter(|case| case.action_dim == Some(dim))
            .collect::<Vec<_>>();
        if values.len() != sweep_steps {
            continue;
        }
        let low = values.first().unwrap();
        let high = values.last().unwrap();
        let final_pos_delta = sub3(high.final_pos_world, low.final_pos_world);
        let action_delta = high.action_value - low.action_value;
        let derivative = if action_delta.abs() > 1e-6 {
            [
                final_pos_delta[0] / action_delta,
                final_pos_delta[1] / action_delta,
                final_pos_delta[2] / action_delta,
            ]
        } else {
            [0.0, 0.0, 0.0]
        };
        summaries.push(DimensionSummary {
            action_dim: dim,
            action_name: ACTION_NAMES[dim].to_string(),
            low_value: low.action_value,
            high_value: high.action_value,
            final_pos_span_m: norm3(final_pos_delta),
            final_z_span_m: (high.final_pos_world[2] - low.final_pos_world[2]).abs(),
            final_speed_span_mps: (high.final_speed_mps - low.final_speed_mps).abs(),
            endpoint_derivative_pos_world_per_action: derivative,
        });
    }
    summaries
}

fn build_cases(baseline: [f32; DRONE_ACTION_DIM], sweep_steps: usize) -> Vec<ActionCase> {
    let mut cases = Vec::with_capacity(1 + DRONE_ACTION_DIM * sweep_steps);
    cases.push(ActionCase {
        name: "baseline".to_string(),
        action_dim: None,
        action_name: "baseline".to_string(),
        action_value: 0.0,
        action: baseline,
    });
    for dim in 0..DRONE_ACTION_DIM {
        let (low, high) = action_bounds(dim);
        for idx in 0..sweep_steps {
            let t = idx as f32 / (sweep_steps - 1) as f32;
            let value = low + (high - low) * t;
            let mut action = baseline;
            action[dim] = value;
            cases.push(ActionCase {
                name: format!("{}={value:.3}", ACTION_NAMES[dim]),
                action_dim: Some(dim),
                action_name: ACTION_NAMES[dim].to_string(),
                action_value: value,
                action,
            });
        }
    }
    cases
}

fn action_bounds(dim: usize) -> (f32, f32) {
    match dim {
        2 => (0.0, 1.0),
        _ => (-1.0, 1.0),
    }
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

fn apply_delta(frame: &DroneFrame, delta: &[f32; DRONE_STATE_DELTA_DIM]) -> DroneFrame {
    let delta_pos_body = [delta[0], delta[1], delta[2]];
    let delta_rot_body = [delta[3], delta[4], delta[5]];
    let delta_pos_world = mat3_mul_vec3(frame.rotmat_world_from_body, delta_pos_body);
    let delta_rot = mat3_from_rotvec(delta_rot_body);
    let next_rot = mat3_mul(frame.rotmat_world_from_body, delta_rot);
    DroneFrame {
        pos_world: add3(frame.pos_world, delta_pos_world),
        rotmat_world_from_body: next_rot,
        lin_vel_body: [delta[6], delta[7], delta[8]],
        ang_vel_body: [delta[9], delta[10], delta[11]],
        vbat: frame.vbat + delta[12],
        ..frame.clone()
    }
}

fn array13(values: &[f32]) -> [f32; DRONE_STATE_DELTA_DIM] {
    let mut out = [0.0; DRONE_STATE_DELTA_DIM];
    out.copy_from_slice(values);
    out
}

fn default_dataset_dir() -> PathBuf {
    home_dir()
        .join(".stable_worldmodel")
        .join("le-wm-nv-data")
        .join("drone-racing-autonomous-100hz")
}

fn default_weights() -> PathBuf {
    home_dir()
        .join(".stable_worldmodel")
        .join("le-wm-nv-runs")
        .join("drone-state-lewm-autonomous-100hz")
        .join("latest.safetensors")
}

fn default_config() -> PathBuf {
    home_dir()
        .join(".stable_worldmodel")
        .join("le-wm-nv-runs")
        .join("drone-state-lewm-autonomous-100hz")
        .join("model-config.json")
}

fn default_output() -> PathBuf {
    home_dir()
        .join(".stable_worldmodel")
        .join("le-wm-nv-reports")
        .join("drone-state-lewm-autonomous-100hz")
        .join("action-sensitivity.json")
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

#[derive(Debug)]
struct ActionCase {
    name: String,
    action_dim: Option<usize>,
    action_name: String,
    action_value: f32,
    action: [f32; DRONE_ACTION_DIM],
}

#[derive(Debug, Serialize)]
struct SensitivityReport {
    dataset_dir: PathBuf,
    weights: PathBuf,
    config: PathBuf,
    row: usize,
    history_steps: usize,
    horizon: usize,
    sweep_steps: usize,
    sample_rate_hz: usize,
    baseline_action: [f32; DRONE_ACTION_DIM],
    current: DroneFrame,
    rollout_elapsed_sec: f64,
    cases: Vec<CaseResult>,
    dimensions: Vec<DimensionSummary>,
}

#[derive(Debug, Serialize)]
struct CaseResult {
    name: String,
    action_dim: Option<usize>,
    action_name: String,
    action_value: f32,
    action: [f32; DRONE_ACTION_DIM],
    final_pos_world: [f32; 3],
    displacement_world: [f32; 3],
    final_lin_vel_body: [f32; 3],
    final_speed_mps: f32,
    net_distance_m: f32,
    path_length_m: f32,
    altitude_delta_m: f32,
}

#[derive(Debug, Serialize)]
struct DimensionSummary {
    action_dim: usize,
    action_name: String,
    low_value: f32,
    high_value: f32,
    final_pos_span_m: f32,
    final_z_span_m: f32,
    final_speed_span_mps: f32,
    endpoint_derivative_pos_world_per_action: [f32; 3],
}
