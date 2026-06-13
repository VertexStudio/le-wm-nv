use std::{
    fs,
    path::{Path, PathBuf},
    sync::OnceLock,
    time::Instant,
};

use anyhow::{Context, ensure};
use candle::{
    CudaStorage, D, DType, IndexOp, Storage, Tensor,
    cuda_backend::{
        WrapErr,
        cudarc::{
            driver::{LaunchConfig, PushKernelArg},
            nvrtc,
        },
    },
    op::BackpropOp,
};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use le_wm_nv::{
    data::drone_racing::{
        DRONE_ACTION_DIM, DRONE_STATE_DELTA_DIM, DroneBatchConfig, DroneFrame, DroneRacingDataset,
        FlightGates, GateSequenceFile, GateSpec, RunningStats, add3, mat3_from_rotvec, mat3_mul,
        mat3_mul_vec3, norm3, sub3,
    },
    models::world_model::{WorldModel, WorldModelConfig},
    planner::{ActionBounds, CandidateScorer, CemConfig, CemPlanner},
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

    /// Gate JSON. Defaults to dataset_dir/gates.json.
    #[arg(long)]
    gates: Option<PathBuf>,

    /// Output plan JSON.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Dataset row used as current history start. Defaults to first eval row.
    #[arg(long)]
    row: Option<usize>,

    /// Gate index inside the selected episode gate list.
    #[arg(long, default_value_t = 0)]
    gate_index: usize,

    #[arg(long, default_value_t = DeviceSpec::Cuda(0))]
    device: DeviceSpec,

    #[arg(long, default_value_t = DTypeSpec::F32)]
    dtype: DTypeSpec,

    #[arg(long, default_value_t = 8)]
    history_steps: usize,

    #[arg(long, default_value_t = 50)]
    horizon: usize,

    #[arg(long, default_value_t = 512)]
    samples: usize,

    #[arg(long, default_value_t = 64)]
    elites: usize,

    #[arg(long, default_value_t = 4)]
    iterations: usize,

    #[arg(long, default_value_t = 7)]
    seed: u64,

    /// If set, run receding-horizon planning for this many model steps over the gate loop.
    #[arg(long)]
    loop_steps: Option<usize>,

    /// Number of actions to execute before replanning in loop mode.
    #[arg(long, default_value_t = 5)]
    control_stride: usize,

    /// Stop loop mode after this many completed laps. Zero disables the lap stop.
    #[arg(long, default_value_t = 1)]
    laps: usize,

    /// Weight for previewing the next gate after the current target gate.
    #[arg(long, default_value_t = 0.15)]
    next_gate_weight: f64,

    /// Minimum allowed altitude in meters for planner scoring.
    #[arg(long, default_value_t = 0.15)]
    min_altitude: f64,

    /// Maximum preferred body-frame speed in m/s before soft penalty.
    #[arg(long, default_value_t = 25.0)]
    max_speed: f64,

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
    let gates_path = args
        .gates
        .clone()
        .unwrap_or_else(|| dataset_dir.join("gates.json"));
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
    let gates = read_gates(&gates_path)?;
    let row_episode = dataset.frame(row)?.episode_idx;
    let gate = select_gate(&gates, row_episode, args.gate_index)?;
    let next_gate = select_next_gate(&gates, row_episode, args.gate_index).ok();
    let cfg: WorldModelConfig = serde_json::from_str(
        &fs::read_to_string(&config)
            .with_context(|| format!("failed to read {}", config.display()))?,
    )
    .with_context(|| format!("failed to parse {}", config.display()))?;
    let device = args.device.resolve()?;
    let dtype = args.dtype.dtype();
    if dtype != DType::F32 {
        anyhow::bail!("drone gate planning currently requires --dtype f32");
    }
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
    let model = WorldModel::new(cfg, vb)?;
    varmap
        .load(&weights)
        .with_context(|| format!("failed to load {}", weights.display()))?;
    let history = dataset.batch(&[row], dtype, &device)?;
    let emb = model.encode_vector(&history.observations)?;
    let current = dataset.frame(row + args.history_steps - 1)?;
    if let Some(loop_steps) = args.loop_steps {
        let flight = select_flight(&gates, current.episode_idx)?;
        let report = run_gate_loop(
            &args, &dataset, &model, emb, current, flight, loop_steps, dtype, &device,
        )?;
        write_pretty_json(&output, &report)?;
        println!(
            "planned loop row={} episode={} frames={} output={}",
            row,
            report.episode_idx,
            report.frames.len(),
            output.display()
        );
        return Ok(());
    }

    let scorer = DroneGateScorer::new(
        &model,
        emb,
        current,
        gate,
        next_gate,
        dataset.metadata().normalization.action.clone(),
        dataset.metadata().normalization.target_delta.clone(),
        !args.no_action_normalize,
        !args.no_target_normalize,
        device.clone(),
        dtype,
        args.history_steps,
        args.next_gate_weight,
        args.min_altitude,
        args.max_speed,
    )?;
    let planner = CemPlanner::new(cem_config(&args));
    let result = planner.plan_device(&scorer)?;
    let sequence = result.sequence.flatten_all()?.to_vec1::<f32>()?;
    let score_summary = ScoreSummary::from_tensor(&result.scores)?;
    let scores = result.scores.flatten_all()?.to_vec1::<f32>()?;
    let best_indices = best_indices_from_tensor(&result.best_indices)?;
    let report = PlanReport {
        dataset_dir,
        weights,
        config,
        row,
        gate: scorer.gate.clone(),
        horizon: args.horizon.max(args.history_steps),
        samples: args.samples,
        elites: args.elites,
        iterations_completed: result.iterations_completed,
        best_indices,
        score_summary,
        best_sequence: sequence
            .chunks_exact(DRONE_ACTION_DIM)
            .map(|chunk| [chunk[0], chunk[1], chunk[2], chunk[3]])
            .collect(),
        scores,
    };
    write_pretty_json(&output, &report)?;
    println!(
        "planned row={} gate={} output={}",
        row,
        report.gate.name,
        output.display()
    );
    Ok(())
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    ensure!(
        args.history_steps > 0,
        "--history-steps must be greater than zero"
    );
    ensure!(args.horizon > 0, "--horizon must be greater than zero");
    ensure!(args.samples > 0, "--samples must be greater than zero");
    ensure!(args.elites >= 2, "--elites must be at least two");
    ensure!(args.elites <= args.samples, "--elites must be <= --samples");
    ensure!(
        args.control_stride > 0,
        "--control-stride must be greater than zero"
    );
    ensure!(
        args.control_stride <= args.horizon,
        "--control-stride must be <= --horizon"
    );
    ensure!(
        args.next_gate_weight.is_finite() && args.next_gate_weight >= 0.0,
        "--next-gate-weight must be finite and non-negative"
    );
    ensure!(
        args.min_altitude.is_finite(),
        "--min-altitude must be finite"
    );
    ensure!(
        args.max_speed.is_finite() && args.max_speed > 0.0,
        "--max-speed must be finite and greater than zero"
    );
    Ok(())
}

fn cem_config(args: &Args) -> CemConfig {
    let mut cfg = CemConfig::new(
        args.horizon.max(args.history_steps),
        args.samples,
        args.elites,
        DRONE_ACTION_DIM,
    );
    cfg.iterations = args.iterations;
    cfg.seed = Some(args.seed);
    cfg.action_bounds = ActionBounds {
        low: vec![-1.0, -1.0, 0.0, -1.0],
        high: vec![1.0, 1.0, 1.0, 1.0],
    };
    cfg.init_std = 0.5;
    cfg.min_std = 0.02;
    cfg
}

fn run_gate_loop(
    args: &Args,
    dataset: &DroneRacingDataset,
    model: &WorldModel,
    mut emb: Tensor,
    mut current: DroneFrame,
    flight: &FlightGates,
    loop_steps: usize,
    dtype: DType,
    device: &candle::Device,
) -> anyhow::Result<LoopPlanReport> {
    ensure!(!flight.gates.is_empty(), "selected flight has no gates");
    let mut gate_index = args.gate_index % flight.gates.len();
    let mut completed_laps = 0usize;
    let mut executed_steps = 0usize;
    let mut frames = vec![current.clone()];
    let mut actions = Vec::new();
    let mut replans = Vec::new();
    let planner = CemPlanner::new(cem_config(args));

    while executed_steps < loop_steps {
        let gate = flight.gates[gate_index].clone();
        let next_gate = next_gate_in_flight(flight, gate_index).cloned();
        let scorer = DroneGateScorer::new(
            model,
            emb.clone(),
            current.clone(),
            gate.clone(),
            next_gate,
            dataset.metadata().normalization.action.clone(),
            dataset.metadata().normalization.target_delta.clone(),
            !args.no_action_normalize,
            !args.no_target_normalize,
            device.clone(),
            dtype,
            args.history_steps,
            args.next_gate_weight,
            args.min_altitude,
            args.max_speed,
        )?;
        let replan_started = Instant::now();
        let result = planner.plan_device(&scorer)?;
        let plan_elapsed_sec = replan_started.elapsed().as_secs_f64();
        let score_summary = ScoreSummary::from_tensor(&result.scores)?;
        let stride = args
            .control_stride
            .min(result.sequence.dim(1)?)
            .min(loop_steps - executed_steps);
        let previous = current.clone();
        let advance_started = Instant::now();
        let advance = advance_with_lewm(
            model,
            emb,
            current,
            &result.sequence,
            stride,
            &dataset.metadata().normalization.action,
            &dataset.metadata().normalization.target_delta,
            !args.no_action_normalize,
            !args.no_target_normalize,
            args.history_steps,
            dtype,
            device,
        )?;
        let advance_elapsed_sec = advance_started.elapsed().as_secs_f64();
        emb = advance.emb;
        current = advance.current;
        frames.extend(advance.frames);
        actions.extend(advance.actions);
        executed_steps += stride;
        let passed = gate_passed(previous.pos_world, current.pos_world, &gate);
        replans.push(ReplanStep {
            executed_steps,
            gate_index,
            gate_name: gate.name.clone(),
            passed_gate: passed,
            iterations_completed: result.iterations_completed,
            best_indices: best_indices_from_tensor(&result.best_indices)?,
            score_summary,
            planner_elapsed_sec: plan_elapsed_sec,
            model_advance_elapsed_sec: advance_elapsed_sec,
            candidate_sequences_scored: args.samples * result.iterations_completed,
            candidate_sequences_per_sec: throughput(
                args.samples * result.iterations_completed,
                plan_elapsed_sec,
            ),
        });
        if passed {
            gate_index += 1;
            if gate_index == flight.gates.len() {
                gate_index = 0;
                completed_laps += 1;
                if args.laps > 0 && completed_laps >= args.laps {
                    break;
                }
            }
        }
    }

    let total_planner_elapsed_sec = replans
        .iter()
        .map(|replan| replan.planner_elapsed_sec)
        .sum::<f64>();
    let total_candidate_sequences = replans
        .iter()
        .map(|replan| replan.candidate_sequences_scored)
        .sum::<usize>();
    Ok(LoopPlanReport {
        mode: "gate_loop".to_string(),
        dataset_dir: dataset.root().to_path_buf(),
        weights: args.weights.clone().unwrap_or_else(default_weights),
        config: args.config.clone().unwrap_or_else(default_config),
        episode_idx: flight.episode_idx,
        flight: flight.flight.clone(),
        horizon: args.horizon.max(args.history_steps),
        samples: args.samples,
        elites: args.elites,
        iterations: args.iterations,
        control_stride: args.control_stride,
        requested_loop_steps: loop_steps,
        executed_steps,
        completed_laps,
        next_gate_index: gate_index,
        total_replans: replans.len(),
        total_planner_elapsed_sec,
        candidate_sequences_scored: total_candidate_sequences,
        candidate_sequences_per_sec: throughput(
            total_candidate_sequences,
            total_planner_elapsed_sec,
        ),
        sample_rate_hz: dataset.metadata().sample_rate_hz,
        actual: frames.clone(),
        predicted: frames.clone(),
        baseline: frames.clone(),
        gates: GateSequenceFile {
            flights: vec![flight.clone()],
        },
        gate_loop: flight.gates.clone(),
        frames,
        actions,
        replans,
    })
}

struct AdvanceResult {
    emb: Tensor,
    current: DroneFrame,
    frames: Vec<DroneFrame>,
    actions: Vec<[f32; DRONE_ACTION_DIM]>,
}

fn advance_with_lewm(
    model: &WorldModel,
    emb: Tensor,
    mut current: DroneFrame,
    action_sequence: &Tensor,
    stride: usize,
    action_stats: &RunningStats,
    target_stats: &RunningStats,
    action_normalized: bool,
    target_normalized: bool,
    history_steps: usize,
    dtype: DType,
    device: &candle::Device,
) -> anyhow::Result<AdvanceResult> {
    ensure!(stride > 0, "cannot advance zero steps");
    let (batch, horizon, action_dim) = action_sequence.dims3()?;
    ensure!(batch == 1, "advance expects a single planned batch");
    ensure!(
        action_dim == DRONE_ACTION_DIM,
        "advance expected action_dim {}, got {action_dim}",
        DRONE_ACTION_DIM
    );
    ensure!(horizon >= stride, "action sequence shorter than stride");
    let action_mean =
        Tensor::from_vec(action_stats.mean.clone(), (1, 1, DRONE_ACTION_DIM), device)?
            .to_dtype(dtype)?;
    let action_std = Tensor::from_vec(
        action_stats
            .std
            .iter()
            .map(|value| value.max(1e-6))
            .collect::<Vec<_>>(),
        (1, 1, DRONE_ACTION_DIM),
        device,
    )?
    .to_dtype(dtype)?;
    let model_action_sequence = if action_normalized {
        action_sequence
            .broadcast_sub(&action_mean)?
            .broadcast_div(&action_std)?
    } else {
        action_sequence.clone()
    };
    let actions = model_action_sequence.unsqueeze(1)?;
    let emb_init = emb.unsqueeze(1)?;
    let rollout = model.rollout_embeddings_with_history(&emb_init, &actions, history_steps)?;
    let (_, _, rollout_time, emb_dim) = rollout.dims4()?;
    ensure!(
        stride + history_steps <= rollout_time,
        "rollout_time {rollout_time} too short for stride {stride} and history {history_steps}"
    );
    let pred = model.predict_state_deltas_from_embeddings(&rollout.reshape((
        1,
        rollout_time,
        emb_dim,
    ))?)?;
    let pred_values = pred
        .i((0, history_steps..history_steps + stride, ..))?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let action_values = action_sequence
        .i((0, 0..stride, ..))?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let actions = action_values
        .chunks_exact(DRONE_ACTION_DIM)
        .map(|chunk| [chunk[0], chunk[1], chunk[2], chunk[3]])
        .collect::<Vec<_>>();
    let mut frames = Vec::with_capacity(stride);
    for step in 0..stride {
        let delta = denormalized_delta(
            &pred_values[step * DRONE_STATE_DELTA_DIM..(step + 1) * DRONE_STATE_DELTA_DIM],
            target_stats,
            target_normalized,
        );
        current = apply_delta(&current, &delta);
        current.row = current.row.saturating_add(1);
        current.step_idx += 1;
        frames.push(current.clone());
    }
    let emb = rollout
        .i((0, 0, stride..stride + history_steps, ..))?
        .reshape((1, history_steps, emb_dim))?;
    Ok(AdvanceResult {
        emb,
        current,
        frames,
        actions,
    })
}

fn denormalized_delta(
    values: &[f32],
    stats: &RunningStats,
    normalized: bool,
) -> [f32; DRONE_STATE_DELTA_DIM] {
    let mut out = [0f32; DRONE_STATE_DELTA_DIM];
    for idx in 0..DRONE_STATE_DELTA_DIM {
        out[idx] = if normalized {
            values[idx] * stats.std[idx] + stats.mean[idx]
        } else {
            values[idx]
        };
    }
    out
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

fn gate_passed(prev_pos: [f32; 3], current_pos: [f32; 3], gate: &GateSpec) -> bool {
    let prev_rel = sub3(prev_pos, gate.center);
    let current_rel = sub3(current_pos, gate.center);
    let prev_plane = dot3(prev_rel, gate.normal);
    let current_plane = dot3(current_rel, gate.normal);
    let crossed = prev_plane.signum() != current_plane.signum()
        || prev_plane.abs().min(current_plane.abs()) < 0.25;
    let lateral = dot3(current_rel, gate.right).abs();
    let vertical = dot3(current_rel, gate.up).abs();
    let inside = lateral <= gate.half_width * 1.25 && vertical <= gate.half_height * 1.25;
    crossed && inside || norm3(current_rel) < gate.half_width.max(gate.half_height)
}

fn dot3(lhs: [f32; 3], rhs: [f32; 3]) -> f32 {
    lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
}

fn throughput(count: usize, elapsed_sec: f64) -> f64 {
    if elapsed_sec > 0.0 {
        count as f64 / elapsed_sec
    } else {
        0.0
    }
}

#[derive(Debug, Clone, Serialize)]
struct ScoreSummary {
    best: f32,
    mean: f32,
    worst: f32,
}

impl ScoreSummary {
    fn from_tensor(scores: &Tensor) -> candle::Result<Self> {
        Ok(Self {
            best: scores.min_all()?.to_scalar::<f32>()?,
            mean: scores.mean_all()?.to_scalar::<f32>()?,
            worst: scores.max_all()?.to_scalar::<f32>()?,
        })
    }
}

fn best_indices_from_tensor(indices: &Tensor) -> candle::Result<Vec<usize>> {
    let rows = indices.to_vec2::<u32>()?;
    let mut best_indices = Vec::with_capacity(rows.len());
    for (batch_idx, row) in rows.iter().enumerate() {
        let Some(&best_idx) = row.first() else {
            candle::bail!("best index row {batch_idx} is empty");
        };
        best_indices.push(best_idx as usize);
    }
    Ok(best_indices)
}

struct DroneGateScorer<'a> {
    model: &'a WorldModel,
    emb: Tensor,
    gate: GateSpec,
    current_pos: Tensor,
    current_rot: Tensor,
    start_plane: Tensor,
    action_mean: Tensor,
    action_std: Tensor,
    target_mean: Tensor,
    target_std: Tensor,
    gate_center: Tensor,
    gate_normal: Tensor,
    gate_right: Tensor,
    gate_up: Tensor,
    next_gate_center: Option<Tensor>,
    action_normalized: bool,
    target_normalized: bool,
    device: candle::Device,
    dtype: DType,
    history_steps: usize,
    next_gate_weight: f64,
    min_altitude: f64,
    max_speed: f64,
}

impl CandidateScorer for DroneGateScorer<'_> {
    fn device(&self) -> &candle::Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn batch_size(&self) -> Option<usize> {
        Some(1)
    }

    fn score_candidates(&self, action_candidates: &Tensor) -> candle::Result<Tensor> {
        let dims = action_candidates.dims();
        if dims.len() != 4 {
            candle::bail!(
                "drone scorer expects [batch, samples, horizon, action_dim], got {:?}",
                action_candidates.shape()
            );
        }
        let (batch, samples, _horizon, action_dim) = (dims[0], dims[1], dims[2], dims[3]);
        if batch != 1 || action_dim != DRONE_ACTION_DIM {
            candle::bail!(
                "drone scorer expects batch=1 action_dim={}, got {:?}",
                DRONE_ACTION_DIM,
                action_candidates.shape()
            );
        }
        let model_actions = if self.action_normalized {
            action_candidates
                .broadcast_sub(&self.action_mean)?
                .broadcast_div(&self.action_std)?
        } else {
            action_candidates.clone()
        };
        let (_, history, emb_dim) = self.emb.dims3()?;
        let emb_init = self
            .emb
            .unsqueeze(1)?
            .broadcast_as((1, samples, history, emb_dim))?;
        let rollout = self.model.rollout_embeddings_with_history(
            &emb_init,
            &model_actions,
            self.history_steps,
        )?;
        let rollout_time = rollout.dim(2)?;
        let pred = self
            .model
            .predict_state_deltas_from_embeddings(&rollout.reshape((
                samples,
                rollout_time,
                emb_dim,
            ))?)?;
        let deltas = if self.target_normalized {
            pred.broadcast_mul(&self.target_std)?
                .broadcast_add(&self.target_mean)?
        } else {
            pred
        };
        if let Some(scores) =
            self.score_candidates_cuda(action_candidates, &deltas, rollout_time, samples)?
        {
            return scores.reshape((1, samples));
        }
        let gate_scores = self.score_rollout(&deltas, rollout_time, samples)?;
        let action_effort =
            (action_candidates.sqr()?.sum(D::Minus1)?.sum(D::Minus1)? * 1e-3)?.squeeze(0)?;
        let action_smoothness = if action_candidates.dim(2)? > 1 {
            let t = action_candidates.dim(2)?;
            let current = action_candidates.i((.., .., 1..t, ..))?;
            let previous = action_candidates.i((.., .., 0..t - 1, ..))?;
            (current
                .broadcast_sub(&previous)?
                .sqr()?
                .sum(D::Minus1)?
                .sum(D::Minus1)?
                * 2e-3)?
                .squeeze(0)?
        } else {
            Tensor::zeros((samples,), self.dtype, &self.device)?
        };
        ((gate_scores + action_effort)? + action_smoothness)?.reshape((1, samples))
    }
}

impl DroneGateScorer<'_> {
    fn new<'a>(
        model: &'a WorldModel,
        emb: Tensor,
        current: DroneFrame,
        gate: GateSpec,
        next_gate: Option<GateSpec>,
        action_stats: RunningStats,
        target_stats: RunningStats,
        action_normalized: bool,
        target_normalized: bool,
        device: candle::Device,
        dtype: DType,
        history_steps: usize,
        next_gate_weight: f64,
        min_altitude: f64,
        max_speed: f64,
    ) -> candle::Result<DroneGateScorer<'a>> {
        let action_mean =
            Tensor::from_vec(action_stats.mean, (1, 1, 1, DRONE_ACTION_DIM), &device)?
                .to_dtype(dtype)?;
        let action_std = Tensor::from_vec(
            action_stats
                .std
                .into_iter()
                .map(|value| value.max(1e-6))
                .collect::<Vec<_>>(),
            (1, 1, 1, DRONE_ACTION_DIM),
            &device,
        )?
        .to_dtype(dtype)?;
        let target_mean =
            Tensor::from_vec(target_stats.mean, (1, 1, DRONE_STATE_DELTA_DIM), &device)?
                .to_dtype(dtype)?;
        let target_std = Tensor::from_vec(
            target_stats
                .std
                .into_iter()
                .map(|value| value.max(1e-6))
                .collect::<Vec<_>>(),
            (1, 1, DRONE_STATE_DELTA_DIM),
            &device,
        )?
        .to_dtype(dtype)?;
        let current_pos =
            Tensor::from_vec(current.pos_world.to_vec(), (1, 3), &device)?.to_dtype(dtype)?;
        let gate_center =
            Tensor::from_vec(gate.center.to_vec(), (1, 3), &device)?.to_dtype(dtype)?;
        let gate_normal =
            Tensor::from_vec(gate.normal.to_vec(), (1, 3), &device)?.to_dtype(dtype)?;
        let start_plane = dot_last(&current_pos.broadcast_sub(&gate_center)?, &gate_normal)?;
        let next_gate_center = next_gate
            .as_ref()
            .map(|gate| Tensor::from_vec(gate.center.to_vec(), (1, 3), &device)?.to_dtype(dtype))
            .transpose()?;
        Ok(DroneGateScorer {
            model,
            emb,
            current_pos,
            current_rot: Tensor::from_vec(
                current.rotmat_world_from_body.to_vec(),
                (1, 9),
                &device,
            )?
            .to_dtype(dtype)?,
            start_plane,
            gate_center,
            gate_normal,
            gate_right: Tensor::from_vec(gate.right.to_vec(), (1, 3), &device)?.to_dtype(dtype)?,
            gate_up: Tensor::from_vec(gate.up.to_vec(), (1, 3), &device)?.to_dtype(dtype)?,
            next_gate_center,
            gate,
            action_mean,
            action_std,
            target_mean,
            target_std,
            action_normalized,
            target_normalized,
            device,
            dtype,
            history_steps,
            next_gate_weight,
            min_altitude,
            max_speed,
        })
    }

    fn score_rollout(
        &self,
        deltas: &Tensor,
        rollout_time: usize,
        samples: usize,
    ) -> candle::Result<Tensor> {
        let mut pos = self.current_pos.broadcast_as((samples, 3))?;
        let mut rot = self.current_rot.broadcast_as((samples, 9))?;
        let mut best =
            Tensor::full(f32::INFINITY, (samples,), &self.device)?.to_dtype(self.dtype)?;
        let mut best_next =
            Tensor::full(f32::INFINITY, (samples,), &self.device)?.to_dtype(self.dtype)?;
        let mut prev_plane = self.start_plane.broadcast_as((samples,))?;
        let start = self.history_steps.min(rollout_time.saturating_sub(1));
        for step in start..rollout_time {
            let delta = deltas.i((.., step, ..))?;
            let delta_pos_body = delta.i((.., 0..3))?;
            let delta_rot_body = delta.i((.., 3..6))?;
            let lin_vel_body = delta.i((.., 6..9))?;
            let delta_pos_world = batched_mat3_mul_vec3(&rot, &delta_pos_body)?;
            pos = (pos + delta_pos_world)?;
            let delta_rot = batched_rotvec_to_mat3(&delta_rot_body)?;
            rot = batched_mat3_mul(&rot, &delta_rot)?;

            let rel = pos.broadcast_sub(&self.gate_center)?;
            let signed_plane = dot_last(&rel, &self.gate_normal)?;
            let plane = signed_plane.abs()?;
            let lateral =
                (dot_last(&rel, &self.gate_right)?.abs()? - self.gate.half_width as f64)?.relu()?;
            let vertical =
                (dot_last(&rel, &self.gate_up)?.abs()? - self.gate.half_height as f64)?.relu()?;
            let progress = rel.sqr()?.sum(D::Minus1)?.sqrt()?;
            let same_side = prev_plane.broadcast_mul(&signed_plane)?.relu()?;
            let altitude_low = ((pos.i((.., 2))? * -1.0)? + self.min_altitude)?.relu()?;
            let speed = lin_vel_body.sqr()?.sum(D::Minus1)?.sqrt()?;
            let speed_excess = (speed - self.max_speed)?.relu()?;
            let mut cost = plane.broadcast_add(&(lateral.sqr()? * 40.0)?)?;
            cost = cost.broadcast_add(&(vertical.sqr()? * 40.0)?)?;
            cost = cost.broadcast_add(&(progress * 0.03)?)?;
            cost = cost.broadcast_add(&(same_side * 0.02)?)?;
            cost = cost.broadcast_add(&(altitude_low.sqr()? * 50.0)?)?;
            cost = cost.broadcast_add(&(speed_excess.sqr()? * 0.02)?)?;
            best = best.broadcast_minimum(&cost)?;
            if let Some(next_center) = &self.next_gate_center {
                let next_rel = pos.broadcast_sub(next_center)?;
                let next_dist = next_rel.sqr()?.sum(D::Minus1)?.sqrt()?;
                best_next = best_next.broadcast_minimum(&next_dist)?;
            }
            prev_plane = signed_plane;
        }
        if self.next_gate_center.is_some() && self.next_gate_weight > 0.0 {
            best + (best_next * self.next_gate_weight)?
        } else {
            Ok(best)
        }
    }

    fn score_candidates_cuda(
        &self,
        action_candidates: &Tensor,
        deltas: &Tensor,
        rollout_time: usize,
        samples: usize,
    ) -> candle::Result<Option<Tensor>> {
        if !deltas.device().is_cuda() || deltas.dtype() != DType::F32 {
            return Ok(None);
        }
        if !action_candidates.device().is_cuda() || action_candidates.dtype() != DType::F32 {
            return Ok(None);
        }
        let (_, _, horizon, action_dim) = action_candidates.dims4()?;
        if action_dim != DRONE_ACTION_DIM {
            candle::bail!(
                "drone CUDA scorer expects action_dim={}, got {action_dim}",
                DRONE_ACTION_DIM
            );
        }
        let start_step = self.history_steps.min(rollout_time.saturating_sub(1));
        let has_next_gate = self.next_gate_center.is_some() && self.next_gate_weight > 0.0;
        let next_gate_center = self.next_gate_center.as_ref().unwrap_or(&self.gate_center);

        macro_rules! cuda_f32_view {
            ($tensor:expr, $name:literal, $contig:ident, $storage:ident, $cuda_storage:ident, $slice:ident, $view:ident) => {
                let $contig = $tensor.contiguous()?;
                let ($storage, layout) = $contig.storage_and_layout();
                let Storage::Cuda($cuda_storage) = &*$storage else {
                    return Ok(None);
                };
                let $slice = $cuda_storage.as_cuda_slice::<f32>()?;
                let Some((start, end)) = layout.contiguous_offsets() else {
                    candle::bail!(concat!(
                        $name,
                        " tensor must be contiguous for CUDA scoring"
                    ));
                };
                let $view = $slice.slice(start..end);
            };
        }

        cuda_f32_view!(
            deltas,
            "deltas",
            deltas_contig,
            deltas_storage,
            deltas_cuda_storage,
            deltas_slice,
            deltas_view
        );
        cuda_f32_view!(
            action_candidates,
            "action candidates",
            actions_contig,
            actions_storage,
            actions_cuda_storage,
            actions_slice,
            actions_view
        );
        cuda_f32_view!(
            &self.current_pos,
            "current position",
            current_pos_contig,
            current_pos_storage,
            current_pos_cuda_storage,
            current_pos_slice,
            current_pos_view
        );
        cuda_f32_view!(
            &self.current_rot,
            "current rotation",
            current_rot_contig,
            current_rot_storage,
            current_rot_cuda_storage,
            current_rot_slice,
            current_rot_view
        );
        cuda_f32_view!(
            &self.start_plane,
            "start plane",
            start_plane_contig,
            start_plane_storage,
            start_plane_cuda_storage,
            start_plane_slice,
            start_plane_view
        );
        cuda_f32_view!(
            &self.gate_center,
            "gate center",
            gate_center_contig,
            gate_center_storage,
            gate_center_cuda_storage,
            gate_center_slice,
            gate_center_view
        );
        cuda_f32_view!(
            &self.gate_normal,
            "gate normal",
            gate_normal_contig,
            gate_normal_storage,
            gate_normal_cuda_storage,
            gate_normal_slice,
            gate_normal_view
        );
        cuda_f32_view!(
            &self.gate_right,
            "gate right",
            gate_right_contig,
            gate_right_storage,
            gate_right_cuda_storage,
            gate_right_slice,
            gate_right_view
        );
        cuda_f32_view!(
            &self.gate_up,
            "gate up",
            gate_up_contig,
            gate_up_storage,
            gate_up_cuda_storage,
            gate_up_slice,
            gate_up_view
        );
        cuda_f32_view!(
            next_gate_center,
            "next gate center",
            next_gate_contig,
            next_gate_storage,
            next_gate_cuda_storage,
            next_gate_slice,
            next_gate_view
        );

        let cuda = deltas_cuda_storage.device.clone();
        let mut output = unsafe { cuda.alloc::<f32>(samples)? };
        let ptx = cached_drone_gate_score_ptx()?;
        let func =
            cuda.get_or_load_custom_func("swm_drone_gate_score_f32", "swm_drone_gate_score", ptx)?;
        let block_dim = 128u32;
        let grid_dim = ((samples as u32 + block_dim - 1) / block_dim, 1, 1);
        let cfg = LaunchConfig {
            grid_dim,
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let samples_arg = samples as u32;
        let rollout_time_arg = rollout_time as u32;
        let horizon_arg = horizon as u32;
        let delta_dim_arg = DRONE_STATE_DELTA_DIM as u32;
        let start_step_arg = start_step as u32;
        let has_next_gate_arg = u32::from(has_next_gate);
        let half_width_arg = self.gate.half_width;
        let half_height_arg = self.gate.half_height;
        let next_gate_weight_arg = self.next_gate_weight as f32;
        let min_altitude_arg = self.min_altitude as f32;
        let max_speed_arg = self.max_speed as f32;
        let stream = cuda.cuda_stream();
        let mut builder = stream.launch_builder(&func);
        builder.arg(&deltas_view);
        builder.arg(&actions_view);
        builder.arg(&current_pos_view);
        builder.arg(&current_rot_view);
        builder.arg(&start_plane_view);
        builder.arg(&gate_center_view);
        builder.arg(&gate_normal_view);
        builder.arg(&gate_right_view);
        builder.arg(&gate_up_view);
        builder.arg(&next_gate_view);
        builder.arg(&mut output);
        builder.arg(&samples_arg);
        builder.arg(&rollout_time_arg);
        builder.arg(&horizon_arg);
        builder.arg(&delta_dim_arg);
        builder.arg(&start_step_arg);
        builder.arg(&has_next_gate_arg);
        builder.arg(&half_width_arg);
        builder.arg(&half_height_arg);
        builder.arg(&next_gate_weight_arg);
        builder.arg(&min_altitude_arg);
        builder.arg(&max_speed_arg);
        unsafe { builder.launch(cfg) }.w()?;

        let storage = CudaStorage::wrap_cuda_slice(output, cuda);
        Ok(Some(Tensor::from_storage(
            Storage::Cuda(storage),
            (samples,),
            BackpropOp::none(),
            false,
        )))
    }
}

fn dot_last(lhs: &Tensor, rhs: &Tensor) -> candle::Result<Tensor> {
    lhs.broadcast_mul(rhs)?.sum(D::Minus1)
}

fn batched_mat3_mul_vec3(rot: &Tensor, v: &Tensor) -> candle::Result<Tensor> {
    let x0 = (rot.i((.., 0))? * v.i((.., 0))?)?;
    let x1 = (rot.i((.., 1))? * v.i((.., 1))?)?;
    let x2 = (rot.i((.., 2))? * v.i((.., 2))?)?;
    let x = ((x0 + x1)? + x2)?;
    let y0 = (rot.i((.., 3))? * v.i((.., 0))?)?;
    let y1 = (rot.i((.., 4))? * v.i((.., 1))?)?;
    let y2 = (rot.i((.., 5))? * v.i((.., 2))?)?;
    let y = ((y0 + y1)? + y2)?;
    let z0 = (rot.i((.., 6))? * v.i((.., 0))?)?;
    let z1 = (rot.i((.., 7))? * v.i((.., 1))?)?;
    let z2 = (rot.i((.., 8))? * v.i((.., 2))?)?;
    let z = ((z0 + z1)? + z2)?;
    Tensor::stack(&[x, y, z], 1)
}

fn batched_mat3_mul(lhs: &Tensor, rhs: &Tensor) -> candle::Result<Tensor> {
    let mut values = Vec::with_capacity(9);
    for row in 0..3 {
        for col in 0..3 {
            let v0 = (lhs.i((.., row * 3))? * rhs.i((.., col))?)?;
            let v1 = (lhs.i((.., row * 3 + 1))? * rhs.i((.., 3 + col))?)?;
            let v2 = (lhs.i((.., row * 3 + 2))? * rhs.i((.., 6 + col))?)?;
            let value = ((v0 + v1)? + v2)?;
            values.push(value);
        }
    }
    let refs = values.iter().collect::<Vec<_>>();
    Tensor::stack(&refs, 1)
}

fn batched_rotvec_to_mat3(rotvec: &Tensor) -> candle::Result<Tensor> {
    let samples = rotvec.dim(0)?;
    let theta = (rotvec.sqr()?.sum(D::Minus1)? + 1e-12)?.sqrt()?;
    let x = (rotvec.i((.., 0))? / &theta)?;
    let y = (rotvec.i((.., 1))? / &theta)?;
    let z = (rotvec.i((.., 2))? / &theta)?;
    let cos = theta.cos()?;
    let sin = theta.sin()?;
    let one = Tensor::ones((samples,), rotvec.dtype(), rotvec.device())?;
    let one_minus_cos = (one - &cos)?;

    let xx = (&x * &x)?;
    let yy = (&y * &y)?;
    let zz = (&z * &z)?;
    let xy = (&x * &y)?;
    let xz = (&x * &z)?;
    let yz = (&y * &z)?;
    let xs = (&x * &sin)?;
    let ys = (&y * &sin)?;
    let zs = (&z * &sin)?;

    let m00 = (&cos + (xx * &one_minus_cos)?)?;
    let m01 = ((xy.clone() * &one_minus_cos)? - &zs)?;
    let m02 = ((xz.clone() * &one_minus_cos)? + &ys)?;
    let m10 = ((xy * &one_minus_cos)? + &zs)?;
    let m11 = (&cos + (yy * &one_minus_cos)?)?;
    let m12 = ((yz.clone() * &one_minus_cos)? - &xs)?;
    let m20 = ((xz * &one_minus_cos)? - &ys)?;
    let m21 = ((yz * &one_minus_cos)? + &xs)?;
    let m22 = (cos + (zz * &one_minus_cos)?)?;
    Tensor::stack(&[m00, m01, m02, m10, m11, m12, m20, m21, m22], 1)
}

static DRONE_GATE_SCORE_PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();

fn cached_drone_gate_score_ptx() -> candle::Result<&'static str> {
    let cached = DRONE_GATE_SCORE_PTX.get_or_init(|| {
        nvrtc::safe::compile_ptx_with_opts(
            DRONE_GATE_SCORE_CUDA,
            nvrtc::CompileOptions {
                use_fast_math: Some(true),
                ..Default::default()
            },
        )
        .map(|ptx| ptx.to_src())
        .map_err(|err| err.to_string())
    });
    match cached {
        Ok(ptx) => Ok(ptx.as_str()),
        Err(err) => candle::bail!("drone gate score NVRTC compile failed: {err}"),
    }
}

const DRONE_GATE_SCORE_CUDA: &str = r#"
extern "C" __global__ void swm_drone_gate_score_f32(
    const float* __restrict__ deltas,
    const float* __restrict__ actions,
    const float* __restrict__ current_pos,
    const float* __restrict__ current_rot,
    const float* __restrict__ start_plane,
    const float* __restrict__ gate_center,
    const float* __restrict__ gate_normal,
    const float* __restrict__ gate_right,
    const float* __restrict__ gate_up,
    const float* __restrict__ next_gate_center,
    float* __restrict__ output,
    unsigned int samples,
    unsigned int rollout_time,
    unsigned int horizon,
    unsigned int delta_dim,
    unsigned int start_step,
    unsigned int has_next_gate,
    float half_width,
    float half_height,
    float next_gate_weight,
    float min_altitude,
    float max_speed
) {
    const unsigned int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= samples) {
        return;
    }

    float action_effort = 0.0f;
    float action_smoothness = 0.0f;
    float prev_action[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const unsigned long long action_base =
        static_cast<unsigned long long>(sample) * horizon * 4ULL;
    for (unsigned int step = 0; step < horizon; ++step) {
        const unsigned long long offset = action_base + step * 4ULL;
        float action[4] = {
            actions[offset + 0],
            actions[offset + 1],
            actions[offset + 2],
            actions[offset + 3],
        };
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            action_effort += action[i] * action[i];
            if (step > 0) {
                const float diff = action[i] - prev_action[i];
                action_smoothness += diff * diff;
            }
            prev_action[i] = action[i];
        }
    }

    float pos_x = current_pos[0];
    float pos_y = current_pos[1];
    float pos_z = current_pos[2];
    float rot[9];
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        rot[i] = current_rot[i];
    }

    float best = 3.402823466e+38F;
    float best_next = 3.402823466e+38F;
    float previous_plane = start_plane[0];
    const unsigned int first_step = start_step < rollout_time ? start_step : rollout_time - 1;

    for (unsigned int step = first_step; step < rollout_time; ++step) {
        const unsigned long long delta_offset =
            (static_cast<unsigned long long>(sample) * rollout_time + step) * delta_dim;
        const float dp_x = deltas[delta_offset + 0];
        const float dp_y = deltas[delta_offset + 1];
        const float dp_z = deltas[delta_offset + 2];
        const float rv_x = deltas[delta_offset + 3];
        const float rv_y = deltas[delta_offset + 4];
        const float rv_z = deltas[delta_offset + 5];
        const float lv_x = deltas[delta_offset + 6];
        const float lv_y = deltas[delta_offset + 7];
        const float lv_z = deltas[delta_offset + 8];

        const float world_x = rot[0] * dp_x + rot[1] * dp_y + rot[2] * dp_z;
        const float world_y = rot[3] * dp_x + rot[4] * dp_y + rot[5] * dp_z;
        const float world_z = rot[6] * dp_x + rot[7] * dp_y + rot[8] * dp_z;
        pos_x += world_x;
        pos_y += world_y;
        pos_z += world_z;

        const float theta = sqrtf(rv_x * rv_x + rv_y * rv_y + rv_z * rv_z + 1.0e-12f);
        const float axis_x = rv_x / theta;
        const float axis_y = rv_y / theta;
        const float axis_z = rv_z / theta;
        const float c = cosf(theta);
        const float s = sinf(theta);
        const float one_minus_c = 1.0f - c;
        const float xx = axis_x * axis_x;
        const float yy = axis_y * axis_y;
        const float zz = axis_z * axis_z;
        const float xy = axis_x * axis_y;
        const float xz = axis_x * axis_z;
        const float yz = axis_y * axis_z;
        const float xs = axis_x * s;
        const float ys = axis_y * s;
        const float zs = axis_z * s;
        const float d00 = c + xx * one_minus_c;
        const float d01 = xy * one_minus_c - zs;
        const float d02 = xz * one_minus_c + ys;
        const float d10 = xy * one_minus_c + zs;
        const float d11 = c + yy * one_minus_c;
        const float d12 = yz * one_minus_c - xs;
        const float d20 = xz * one_minus_c - ys;
        const float d21 = yz * one_minus_c + xs;
        const float d22 = c + zz * one_minus_c;

        const float n00 = rot[0] * d00 + rot[1] * d10 + rot[2] * d20;
        const float n01 = rot[0] * d01 + rot[1] * d11 + rot[2] * d21;
        const float n02 = rot[0] * d02 + rot[1] * d12 + rot[2] * d22;
        const float n10 = rot[3] * d00 + rot[4] * d10 + rot[5] * d20;
        const float n11 = rot[3] * d01 + rot[4] * d11 + rot[5] * d21;
        const float n12 = rot[3] * d02 + rot[4] * d12 + rot[5] * d22;
        const float n20 = rot[6] * d00 + rot[7] * d10 + rot[8] * d20;
        const float n21 = rot[6] * d01 + rot[7] * d11 + rot[8] * d21;
        const float n22 = rot[6] * d02 + rot[7] * d12 + rot[8] * d22;
        rot[0] = n00;
        rot[1] = n01;
        rot[2] = n02;
        rot[3] = n10;
        rot[4] = n11;
        rot[5] = n12;
        rot[6] = n20;
        rot[7] = n21;
        rot[8] = n22;

        const float rel_x = pos_x - gate_center[0];
        const float rel_y = pos_y - gate_center[1];
        const float rel_z = pos_z - gate_center[2];
        const float signed_plane =
            rel_x * gate_normal[0] + rel_y * gate_normal[1] + rel_z * gate_normal[2];
        const float plane = fabsf(signed_plane);
        const float lateral_axis =
            rel_x * gate_right[0] + rel_y * gate_right[1] + rel_z * gate_right[2];
        const float vertical_axis =
            rel_x * gate_up[0] + rel_y * gate_up[1] + rel_z * gate_up[2];
        const float lateral = fmaxf(fabsf(lateral_axis) - half_width, 0.0f);
        const float vertical = fmaxf(fabsf(vertical_axis) - half_height, 0.0f);
        const float progress = sqrtf(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z);
        const float same_side = fmaxf(previous_plane * signed_plane, 0.0f);
        const float altitude_low = fmaxf(-pos_z + min_altitude, 0.0f);
        const float speed = sqrtf(lv_x * lv_x + lv_y * lv_y + lv_z * lv_z);
        const float speed_excess = fmaxf(speed - max_speed, 0.0f);
        const float cost =
            plane +
            lateral * lateral * 40.0f +
            vertical * vertical * 40.0f +
            progress * 0.03f +
            same_side * 0.02f +
            altitude_low * altitude_low * 50.0f +
            speed_excess * speed_excess * 0.02f;
        best = fminf(best, cost);

        if (has_next_gate != 0U) {
            const float next_x = pos_x - next_gate_center[0];
            const float next_y = pos_y - next_gate_center[1];
            const float next_z = pos_z - next_gate_center[2];
            const float next_dist = sqrtf(next_x * next_x + next_y * next_y + next_z * next_z);
            best_next = fminf(best_next, next_dist);
        }
        previous_plane = signed_plane;
    }

    float score = best + action_effort * 1.0e-3f + action_smoothness * 2.0e-3f;
    if (has_next_gate != 0U) {
        score += best_next * next_gate_weight;
    }
    output[sample] = score;
}
"#;

fn read_gates(path: &Path) -> anyhow::Result<GateSequenceFile> {
    serde_json::from_str(
        &fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))
}

fn select_gate(
    gates: &GateSequenceFile,
    episode: i64,
    gate_index: usize,
) -> anyhow::Result<GateSpec> {
    let flight = select_flight(gates, episode)?;
    flight
        .gates
        .get(gate_index)
        .cloned()
        .with_context(|| format!("flight {} has no gate at index {gate_index}", flight.flight))
}

fn select_flight(gates: &GateSequenceFile, episode: i64) -> anyhow::Result<&FlightGates> {
    gates
        .flights
        .iter()
        .find(|flight| flight.episode_idx == episode)
        .or_else(|| gates.flights.first())
        .context("gate file does not contain any flights")
}

fn select_next_gate(
    gates: &GateSequenceFile,
    episode: i64,
    gate_index: usize,
) -> anyhow::Result<GateSpec> {
    let flight = select_flight(gates, episode)?;
    next_gate_in_flight(flight, gate_index)
        .cloned()
        .with_context(|| {
            format!(
                "flight {} has no gate after index {gate_index}",
                flight.flight
            )
        })
}

fn next_gate_in_flight(flight: &FlightGates, gate_index: usize) -> Option<&GateSpec> {
    if flight.gates.is_empty() {
        None
    } else {
        flight.gates.get((gate_index + 1) % flight.gates.len())
    }
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
        .join("gate-plan.json")
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

#[derive(Debug, Serialize)]
struct PlanReport {
    dataset_dir: PathBuf,
    weights: PathBuf,
    config: PathBuf,
    row: usize,
    gate: GateSpec,
    horizon: usize,
    samples: usize,
    elites: usize,
    iterations_completed: usize,
    best_indices: Vec<usize>,
    score_summary: ScoreSummary,
    best_sequence: Vec<[f32; 4]>,
    scores: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct LoopPlanReport {
    mode: String,
    dataset_dir: PathBuf,
    weights: PathBuf,
    config: PathBuf,
    episode_idx: i64,
    flight: String,
    horizon: usize,
    samples: usize,
    elites: usize,
    iterations: usize,
    control_stride: usize,
    requested_loop_steps: usize,
    executed_steps: usize,
    completed_laps: usize,
    next_gate_index: usize,
    total_replans: usize,
    total_planner_elapsed_sec: f64,
    candidate_sequences_scored: usize,
    candidate_sequences_per_sec: f64,
    sample_rate_hz: usize,
    actual: Vec<DroneFrame>,
    predicted: Vec<DroneFrame>,
    baseline: Vec<DroneFrame>,
    gates: GateSequenceFile,
    gate_loop: Vec<GateSpec>,
    frames: Vec<DroneFrame>,
    actions: Vec<[f32; DRONE_ACTION_DIM]>,
    replans: Vec<ReplanStep>,
}

#[derive(Debug, Serialize)]
struct ReplanStep {
    executed_steps: usize,
    gate_index: usize,
    gate_name: String,
    passed_gate: bool,
    iterations_completed: usize,
    best_indices: Vec<usize>,
    score_summary: ScoreSummary,
    planner_elapsed_sec: f64,
    model_advance_elapsed_sec: f64,
    candidate_sequences_scored: usize,
    candidate_sequences_per_sec: f64,
}
