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
use candle_nn::{Init, ParamsAdamW, VarMap};
use clap::{Parser, ValueEnum};
use le_wm_nv::{
    checkpoint,
    data::drone_racing::{
        DRONE_ACTION_DIM, DRONE_STATE_DELTA_DIM, DroneBatchConfig, DroneFrame, DroneRacingDataset,
        FlightGates, GateSequenceFile, GateSpec, RunningStats, add3, mat3_from_rotvec, mat3_mul,
        mat3_mul_vec3, norm3, sub3,
    },
    models::world_model::{WorldModel, WorldModelConfig},
    optim::StatefulAdamW,
    planner::{ActionBounds, CandidateScorer, CemConfig, CemPlanner, IcemConfig, IcemPlanner},
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

    #[arg(long, default_value_t = 40)]
    horizon: usize,

    /// Planner used to choose future action sequences.
    #[arg(long, value_enum, default_value_t = PlannerKind::Icem)]
    planner: PlannerKind,

    /// Sampling planner candidate sequences per iteration.
    #[arg(long, default_value_t = 512)]
    samples: usize,

    /// Sampling planner elite sequences per iteration.
    #[arg(long, default_value_t = 64)]
    elites: usize,

    /// iCEM elite sequences carried into the next iteration.
    #[arg(long, default_value_t = 16)]
    keep_elites: usize,

    /// Sampling planner refinement iterations.
    #[arg(long, default_value_t = 4)]
    iterations: usize,

    /// Sampling planner RNG seed.
    #[arg(long, default_value_t = 7)]
    seed: u64,

    /// Gradient planner Adam steps per replan.
    #[arg(long, default_value_t = 16)]
    grad_steps: usize,

    /// Gradient planner learning rate for action sequence optimization.
    #[arg(long, default_value_t = 0.001)]
    grad_lr: f64,

    /// Gradient planner AdamW weight decay on actions.
    #[arg(long, default_value_t = 0.0)]
    grad_weight_decay: f64,

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

    /// Distance in meters to place the moving carrot ahead on the current route segment.
    #[arg(long, default_value_t = 1.5)]
    carrot_lookahead: f64,

    /// Weight for the route-segment carrot/path objective.
    #[arg(long, default_value_t = 1.0)]
    path_weight: f64,

    /// Weight for scheduled route progress toward the active gate.
    #[arg(long, default_value_t = 0.8)]
    progress_weight: f64,

    /// Weight for the secondary gate-plane/inside-gate objective.
    #[arg(long, default_value_t = 0.08)]
    gate_weight: f64,

    /// Weight for the terminal rollout pose reaching the active gate center.
    #[arg(long, default_value_t = 1.5)]
    terminal_gate_weight: f64,

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
enum PlannerKind {
    Icem,
    Cem,
    Gradient,
}

const LEGACY_ICEM_HORIZON: usize = 50;
const LEGACY_ICEM_SAMPLES: usize = 512;
const LEGACY_ICEM_KEEP_ELITES: usize = 16;
const LEGACY_ICEM_ITERATIONS: usize = 4;
const LEGACY_ICEM_CONTROL_STRIDE: usize = 5;

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
    let vb = checkpoint::var_builder_from_path(&weights, dtype, &device)
        .with_context(|| format!("failed to load {}", weights.display()))?;
    let model = WorldModel::new(cfg, vb)?;
    let history = dataset.batch(&[row], dtype, &device)?;
    let emb = model.encode_vector(&history.observations)?;
    let action_prefix = history_action_prefix(&history.actions, args.history_steps)?;
    let current = dataset.frame(row + args.history_steps - 1)?;
    if let Some(loop_steps) = args.loop_steps {
        let flight = select_flight(&gates, current.episode_idx)?;
        let report = run_gate_loop(
            &args,
            &dataset,
            &model,
            emb,
            action_prefix,
            current,
            flight,
            loop_steps,
            dtype,
            &device,
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

    let path_anchor = current.pos_world;
    let mut planner_state = DronePlannerState::new(
        args.planner,
        &dataset.metadata().normalization.action.mean,
        &args,
        dtype,
        &device,
    )?;
    let plan = planner_state.plan(
        &args,
        &model,
        &emb,
        &action_prefix,
        &current,
        path_anchor,
        &gate,
        next_gate.as_ref(),
        &dataset.metadata().normalization.action,
        &dataset.metadata().normalization.target_delta,
        !args.no_action_normalize,
        !args.no_target_normalize,
        dtype,
        &device,
    )?;
    let sequence = plan.sequence.flatten_all()?.to_vec1::<f32>()?;
    let score_summary = plan.score_summary.clone();
    let report = PlanReport {
        dataset_dir,
        weights,
        config,
        row,
        gate: gate.clone(),
        carrot_lookahead: args.carrot_lookahead,
        path_weight: args.path_weight,
        progress_weight: args.progress_weight,
        gate_weight: args.gate_weight,
        terminal_gate_weight: args.terminal_gate_weight,
        next_gate_weight: args.next_gate_weight,
        path_anchor,
        carrot: moving_carrot_point(path_anchor, &gate, path_anchor, args.carrot_lookahead),
        horizon: args.horizon,
        planner: args.planner,
        samples: args.samples,
        elites: args.elites,
        keep_elites: args.keep_elites,
        iterations: args.iterations,
        grad_steps: args.grad_steps,
        grad_lr: args.grad_lr,
        planner_budget: planner_budget_summary(&args),
        planner_evals: plan.planner_evals,
        planner_elapsed_sec: plan.planner_elapsed_sec,
        planner_evals_per_sec: throughput(plan.planner_evals, plan.planner_elapsed_sec),
        score_summary,
        best_sequence: sequence
            .chunks_exact(DRONE_ACTION_DIM)
            .map(|chunk| [chunk[0], chunk[1], chunk[2], chunk[3]])
            .collect(),
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
        args.history_steps >= 2,
        "--history-steps must be at least two for action-prefix rollout"
    );
    ensure!(args.horizon > 0, "--horizon must be greater than zero");
    ensure!(
        args.control_stride > 0,
        "--control-stride must be greater than zero"
    );
    ensure!(
        args.control_stride <= args.horizon,
        "--control-stride must be <= --horizon"
    );
    match args.planner {
        PlannerKind::Icem | PlannerKind::Cem => {
            ensure!(args.samples > 0, "--samples must be greater than zero");
            ensure!(args.elites >= 2, "--elites must be at least two");
            ensure!(args.elites <= args.samples, "--elites must be <= --samples");
            ensure!(
                args.iterations > 0,
                "--iterations must be greater than zero"
            );
            ensure!(
                args.keep_elites <= args.elites,
                "--keep-elites must be <= --elites"
            );
        }
        PlannerKind::Gradient => {
            ensure!(
                args.grad_steps > 0,
                "--grad-steps must be greater than zero"
            );
            ensure!(
                args.grad_lr.is_finite() && args.grad_lr > 0.0,
                "--grad-lr must be finite and greater than zero"
            );
            ensure!(
                args.grad_weight_decay.is_finite() && args.grad_weight_decay >= 0.0,
                "--grad-weight-decay must be finite and non-negative"
            );
        }
    }
    ensure!(
        args.next_gate_weight.is_finite() && args.next_gate_weight >= 0.0,
        "--next-gate-weight must be finite and non-negative"
    );
    ensure!(
        args.carrot_lookahead.is_finite() && args.carrot_lookahead > 0.0,
        "--carrot-lookahead must be finite and greater than zero"
    );
    ensure!(
        args.path_weight.is_finite() && args.path_weight >= 0.0,
        "--path-weight must be finite and non-negative"
    );
    ensure!(
        args.progress_weight.is_finite() && args.progress_weight >= 0.0,
        "--progress-weight must be finite and non-negative"
    );
    ensure!(
        args.gate_weight.is_finite() && args.gate_weight >= 0.0,
        "--gate-weight must be finite and non-negative"
    );
    ensure!(
        args.terminal_gate_weight.is_finite() && args.terminal_gate_weight >= 0.0,
        "--terminal-gate-weight must be finite and non-negative"
    );
    ensure!(
        args.path_weight > 0.0
            || args.progress_weight > 0.0
            || args.gate_weight > 0.0
            || args.terminal_gate_weight > 0.0
            || args.next_gate_weight > 0.0,
        "at least one objective weight must be positive"
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

struct DronePlanOutcome {
    planner: PlannerKind,
    sequence: Tensor,
    score_summary: ScoreSummary,
    planner_evals: usize,
    planner_elapsed_sec: f64,
    initial_score: Option<f32>,
}

enum DronePlannerState {
    Icem(IcemPlanner),
    Cem(CemPlanner),
    Gradient(GradientPlannerState),
}

impl DronePlannerState {
    fn new(
        kind: PlannerKind,
        action_mean: &[f32],
        args: &Args,
        dtype: DType,
        device: &candle::Device,
    ) -> anyhow::Result<Self> {
        match kind {
            PlannerKind::Icem => Ok(Self::Icem(IcemPlanner::new(icem_config(args)))),
            PlannerKind::Cem => Ok(Self::Cem(CemPlanner::new(cem_config(args)))),
            PlannerKind::Gradient => Ok(Self::Gradient(GradientPlannerState::new(
                action_mean,
                args.horizon,
                dtype,
                device,
            )?)),
        }
    }

    fn plan(
        &mut self,
        args: &Args,
        model: &WorldModel,
        emb: &Tensor,
        action_prefix: &Tensor,
        current: &DroneFrame,
        path_anchor: [f32; 3],
        gate: &GateSpec,
        next_gate: Option<&GateSpec>,
        action_stats: &RunningStats,
        target_stats: &RunningStats,
        action_normalized: bool,
        target_normalized: bool,
        dtype: DType,
        device: &candle::Device,
    ) -> anyhow::Result<DronePlanOutcome> {
        match self {
            Self::Icem(planner) => sample_plan(
                PlannerKind::Icem,
                planner,
                args,
                model,
                emb,
                action_prefix,
                current,
                gate,
                next_gate,
                action_stats,
                target_stats,
                action_normalized,
                target_normalized,
                dtype,
                device,
            ),
            Self::Cem(planner) => sample_plan(
                PlannerKind::Cem,
                planner,
                args,
                model,
                emb,
                action_prefix,
                current,
                gate,
                next_gate,
                action_stats,
                target_stats,
                action_normalized,
                target_normalized,
                dtype,
                device,
            ),
            Self::Gradient(planner) => planner.plan(
                args,
                model,
                emb,
                action_prefix,
                current,
                path_anchor,
                gate,
                next_gate,
                action_stats,
                target_stats,
                action_normalized,
                target_normalized,
                dtype,
                device,
            ),
        }
    }
}

trait SamplingPlanner {
    fn plan_device(
        &mut self,
        scorer: &DroneGateScorer<'_>,
    ) -> candle::Result<le_wm_nv::planner::PlanDeviceResult>;
}

impl SamplingPlanner for IcemPlanner {
    fn plan_device(
        &mut self,
        scorer: &DroneGateScorer<'_>,
    ) -> candle::Result<le_wm_nv::planner::PlanDeviceResult> {
        IcemPlanner::plan_device(self, scorer)
    }
}

impl SamplingPlanner for CemPlanner {
    fn plan_device(
        &mut self,
        scorer: &DroneGateScorer<'_>,
    ) -> candle::Result<le_wm_nv::planner::PlanDeviceResult> {
        CemPlanner::plan_device(self, scorer)
    }
}

fn sample_plan<P: SamplingPlanner>(
    planner_kind: PlannerKind,
    planner: &mut P,
    args: &Args,
    model: &WorldModel,
    emb: &Tensor,
    action_prefix: &Tensor,
    current: &DroneFrame,
    gate: &GateSpec,
    next_gate: Option<&GateSpec>,
    action_stats: &RunningStats,
    target_stats: &RunningStats,
    action_normalized: bool,
    target_normalized: bool,
    dtype: DType,
    device: &candle::Device,
) -> anyhow::Result<DronePlanOutcome> {
    let scorer = DroneGateScorer::new(
        model,
        emb.clone(),
        action_prefix.clone(),
        current.clone(),
        gate.clone(),
        next_gate.cloned(),
        action_stats.clone(),
        target_stats.clone(),
        action_normalized,
        target_normalized,
        device.clone(),
        dtype,
        args.next_gate_weight,
        args.min_altitude,
        args.max_speed,
    )?;
    let started = Instant::now();
    let result = planner.plan_device(&scorer)?;
    let planner_elapsed_sec = started.elapsed().as_secs_f64();
    let score_summary = ScoreSummary::from_tensor(&result.scores)?;
    Ok(DronePlanOutcome {
        planner: planner_kind,
        sequence: result.sequence,
        score_summary,
        planner_evals: sampling_evals(args, planner_kind, result.iterations_completed),
        planner_elapsed_sec,
        initial_score: None,
    })
}

fn sampling_evals(args: &Args, planner: PlannerKind, iterations_completed: usize) -> usize {
    match planner {
        PlannerKind::Icem => {
            if iterations_completed == 0 {
                0
            } else {
                args.samples + (iterations_completed - 1) * (args.samples + args.keep_elites)
            }
        }
        PlannerKind::Cem => args.samples * iterations_completed,
        PlannerKind::Gradient => args.grad_steps,
    }
}

fn sampling_evals_budget(args: &Args) -> usize {
    match args.planner {
        PlannerKind::Icem => {
            args.samples + (args.iterations - 1) * (args.samples + args.keep_elites)
        }
        PlannerKind::Cem => args.samples * args.iterations,
        PlannerKind::Gradient => args.grad_steps,
    }
}

fn legacy_icem_evals_per_replan() -> usize {
    LEGACY_ICEM_SAMPLES
        + (LEGACY_ICEM_ITERATIONS - 1) * (LEGACY_ICEM_SAMPLES + LEGACY_ICEM_KEEP_ELITES)
}

fn planner_budget_summary(args: &Args) -> PlannerBudgetSummary {
    let current = sampling_evals_budget(args);
    let legacy = legacy_icem_evals_per_replan();
    PlannerBudgetSummary {
        current_per_replan_evals: current,
        legacy_icem_per_replan_evals: legacy,
        current_per_replan_model_steps: current * args.horizon,
        legacy_icem_per_replan_model_steps: legacy * LEGACY_ICEM_HORIZON,
        current_horizon: args.horizon,
        legacy_icem_horizon: LEGACY_ICEM_HORIZON,
        current_control_stride: args.control_stride,
        legacy_icem_control_stride: LEGACY_ICEM_CONTROL_STRIDE,
        current_samples: args.samples,
        legacy_icem_samples: LEGACY_ICEM_SAMPLES,
        current_keep_elites: args.keep_elites,
        legacy_icem_keep_elites: LEGACY_ICEM_KEEP_ELITES,
        current_iterations: args.iterations,
        legacy_icem_iterations: LEGACY_ICEM_ITERATIONS,
        per_replan_eval_ratio_vs_legacy_icem: ratio(current, legacy),
        per_replan_eval_reduction_pct_vs_legacy_icem: reduction_pct(current, legacy),
        per_replan_model_step_ratio_vs_legacy_icem: ratio(
            current * args.horizon,
            legacy * LEGACY_ICEM_HORIZON,
        ),
        per_replan_model_step_reduction_pct_vs_legacy_icem: reduction_pct(
            current * args.horizon,
            legacy * LEGACY_ICEM_HORIZON,
        ),
    }
}

fn loop_planner_benchmark(
    args: &Args,
    requested_loop_steps: usize,
    executed_steps: usize,
    replans: &[ReplanStep],
    total_planner_elapsed_sec: f64,
    total_planner_evals: usize,
) -> LoopPlannerBenchmark {
    let current_expected_replans = ceil_div(requested_loop_steps, args.control_stride);
    let legacy_expected_replans = ceil_div(requested_loop_steps, LEGACY_ICEM_CONTROL_STRIDE);
    let current_budget_total_evals = current_expected_replans * sampling_evals_budget(args);
    let legacy_icem_budget_total_evals = legacy_expected_replans * legacy_icem_evals_per_replan();
    let current_budget_total_model_steps = current_budget_total_evals * args.horizon;
    let legacy_icem_budget_total_model_steps = legacy_icem_budget_total_evals * LEGACY_ICEM_HORIZON;
    LoopPlannerBenchmark {
        current_expected_replans,
        legacy_icem_expected_replans: legacy_expected_replans,
        current_budget_total_evals,
        legacy_icem_budget_total_evals,
        current_budget_total_model_steps,
        legacy_icem_budget_total_model_steps,
        budget_eval_ratio_vs_legacy_icem: ratio(
            current_budget_total_evals,
            legacy_icem_budget_total_evals,
        ),
        budget_eval_reduction_pct_vs_legacy_icem: reduction_pct(
            current_budget_total_evals,
            legacy_icem_budget_total_evals,
        ),
        budget_model_step_ratio_vs_legacy_icem: ratio(
            current_budget_total_model_steps,
            legacy_icem_budget_total_model_steps,
        ),
        budget_model_step_reduction_pct_vs_legacy_icem: reduction_pct(
            current_budget_total_model_steps,
            legacy_icem_budget_total_model_steps,
        ),
        actual_total_planner_elapsed_sec: total_planner_elapsed_sec,
        actual_total_planner_evals: total_planner_evals,
        actual_planner_evals_per_sec: throughput(total_planner_evals, total_planner_elapsed_sec),
        actual_planner_evals_per_executed_step: ratio(total_planner_evals, executed_steps),
        actual_planner_ms_per_executed_step: if executed_steps > 0 {
            total_planner_elapsed_sec * 1000.0 / executed_steps as f64
        } else {
            0.0
        },
        planner_elapsed: timing_summary_secs(
            replans.iter().map(|replan| replan.planner_elapsed_sec),
        ),
        model_advance_elapsed: timing_summary_secs(
            replans
                .iter()
                .map(|replan| replan.model_advance_elapsed_sec),
        ),
    }
}

fn ceil_div(numerator: usize, denominator: usize) -> usize {
    if numerator == 0 {
        0
    } else {
        ((numerator - 1) / denominator) + 1
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator > 0 {
        numerator as f64 / denominator as f64
    } else {
        0.0
    }
}

fn reduction_pct(current: usize, baseline: usize) -> f64 {
    if baseline > 0 {
        (1.0 - current as f64 / baseline as f64) * 100.0
    } else {
        0.0
    }
}

fn timing_summary_secs(values: impl Iterator<Item = f64>) -> TimingSummary {
    let mut values = values.filter(|value| value.is_finite()).collect::<Vec<_>>();
    if values.is_empty() {
        return TimingSummary {
            count: 0,
            total_sec: 0.0,
            mean_sec: 0.0,
            p50_sec: 0.0,
            p90_sec: 0.0,
            min_sec: 0.0,
            max_sec: 0.0,
        };
    }

    values.sort_by(|a, b| a.total_cmp(b));
    let total_sec = values.iter().sum::<f64>();
    let count = values.len();
    TimingSummary {
        count,
        total_sec,
        mean_sec: total_sec / count as f64,
        p50_sec: percentile_sorted(&values, 0.50),
        p90_sec: percentile_sorted(&values, 0.90),
        min_sec: values[0],
        max_sec: values[count - 1],
    }
}

fn percentile_sorted(values: &[f64], percentile: f64) -> f64 {
    let idx = ((values.len() as f64 * percentile).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1);
    values[idx]
}

fn cem_config(args: &Args) -> CemConfig {
    let mut cfg = CemConfig::new(args.horizon, args.samples, args.elites, DRONE_ACTION_DIM);
    cfg.iterations = args.iterations;
    cfg.seed = Some(args.seed);
    cfg.action_bounds = drone_action_bounds();
    cfg.init_std = 0.5;
    cfg.min_std = 0.02;
    cfg
}

fn icem_config(args: &Args) -> IcemConfig {
    let mut cfg = IcemConfig::new(args.horizon, args.samples, args.elites, DRONE_ACTION_DIM);
    cfg.keep_elites = args.keep_elites;
    cfg.iterations = args.iterations;
    cfg.seed = Some(args.seed);
    cfg.action_bounds = drone_action_bounds();
    cfg.init_std = 0.5;
    cfg.min_std = 0.02;
    cfg
}

fn drone_action_bounds() -> ActionBounds {
    ActionBounds {
        low: vec![-1.0, -1.0, 0.0, -1.0],
        high: vec![1.0, 1.0, 1.0, 1.0],
    }
}

struct GradientPlannerState {
    warm_start: Tensor,
}

impl GradientPlannerState {
    fn new(
        action_mean: &[f32],
        horizon: usize,
        dtype: DType,
        device: &candle::Device,
    ) -> anyhow::Result<Self> {
        ensure!(
            action_mean.len() == DRONE_ACTION_DIM,
            "action mean length {} does not match action_dim {DRONE_ACTION_DIM}",
            action_mean.len()
        );
        let mut sequence = Vec::with_capacity(horizon * DRONE_ACTION_DIM);
        for _ in 0..horizon {
            sequence.extend_from_slice(action_mean);
        }
        let warm_start =
            Tensor::from_vec(sequence, (1, horizon, DRONE_ACTION_DIM), device)?.to_dtype(dtype)?;
        Ok(Self { warm_start })
    }

    fn plan(
        &mut self,
        args: &Args,
        model: &WorldModel,
        emb: &Tensor,
        action_prefix: &Tensor,
        current: &DroneFrame,
        path_anchor: [f32; 3],
        gate: &GateSpec,
        next_gate: Option<&GateSpec>,
        action_stats: &RunningStats,
        target_stats: &RunningStats,
        action_normalized: bool,
        target_normalized: bool,
        dtype: DType,
        device: &candle::Device,
    ) -> anyhow::Result<DronePlanOutcome> {
        let mut action_vars = VarMap::new();
        let raw_action_sequence = action_vars.get(
            (1, args.horizon, DRONE_ACTION_DIM),
            "actions",
            Init::Const(0.0),
            dtype,
            device,
        )?;
        action_vars.set_one("actions", &raw_from_drone_actions_tensor(&self.warm_start)?)?;
        let mut optimizer = StatefulAdamW::new_from_varmap(
            &action_vars,
            ParamsAdamW {
                lr: args.grad_lr,
                weight_decay: args.grad_weight_decay,
                ..ParamsAdamW::default()
            },
        )?;
        let mut initial_score = None;
        for step in 0..args.grad_steps {
            let bounded_actions =
                drone_actions_from_raw_tensor(&raw_action_sequence)?.contiguous()?;
            let loss = gradient_plan_cost(
                model,
                emb,
                action_prefix,
                &bounded_actions,
                current,
                path_anchor,
                gate,
                next_gate,
                action_stats,
                target_stats,
                action_normalized,
                target_normalized,
                args,
                dtype,
                device,
            )?;
            let loss_scalar = loss.to_scalar::<f32>()?;
            if step == 0 {
                initial_score = Some(loss_scalar);
            }
            optimizer.backward_step(&loss)?;
        }

        let final_actions = drone_actions_from_raw_tensor(&raw_action_sequence)?.contiguous()?;
        let final_loss = gradient_plan_cost(
            model,
            emb,
            action_prefix,
            &final_actions,
            current,
            path_anchor,
            gate,
            next_gate,
            action_stats,
            target_stats,
            action_normalized,
            target_normalized,
            args,
            dtype,
            device,
        )?;
        let final_loss = final_loss.to_scalar::<f32>()?;
        let sequence = final_actions.detach().contiguous()?;
        self.warm_start = shift_sequence_for_warm_start_local(&sequence)?;
        Ok(DronePlanOutcome {
            planner: PlannerKind::Gradient,
            sequence,
            score_summary: ScoreSummary {
                best: final_loss,
                mean: final_loss,
                worst: initial_score.unwrap_or(final_loss),
            },
            planner_evals: args.grad_steps,
            planner_elapsed_sec: 0.0,
            initial_score,
        })
    }
}

fn shift_sequence_for_warm_start_local(sequence: &Tensor) -> candle::Result<Tensor> {
    let (_, horizon, _) = sequence.dims3()?;
    if horizon == 1 {
        return Ok(sequence.clone());
    }
    let tail = sequence.narrow(1, 1, horizon - 1)?;
    let last = sequence.narrow(1, horizon - 1, 1)?;
    Tensor::cat(&[&tail, &last], 1)
}

fn drone_actions_from_raw_tensor(raw: &Tensor) -> candle::Result<Tensor> {
    let roll = raw.i((.., .., 0..1))?.tanh()?;
    let pitch = raw.i((.., .., 1..2))?.tanh()?;
    let throttle = sigmoid_tensor(&raw.i((.., .., 2..3))?)?;
    let yaw = raw.i((.., .., 3..4))?.tanh()?;
    Tensor::cat(&[&roll, &pitch, &throttle, &yaw], 2)
}

fn raw_from_drone_actions_tensor(actions: &Tensor) -> candle::Result<Tensor> {
    let roll = atanh_tensor(&actions.i((.., .., 0..1))?.clamp(-0.999, 0.999)?)?;
    let pitch = atanh_tensor(&actions.i((.., .., 1..2))?.clamp(-0.999, 0.999)?)?;
    let throttle = logit_tensor(&actions.i((.., .., 2..3))?.clamp(0.001, 0.999)?)?;
    let yaw = atanh_tensor(&actions.i((.., .., 3..4))?.clamp(-0.999, 0.999)?)?;
    Tensor::cat(&[&roll, &pitch, &throttle, &yaw], 2)
}

fn sigmoid_tensor(value: &Tensor) -> candle::Result<Tensor> {
    (value.neg()?.exp()? + 1.0)?.recip()
}

fn logit_tensor(value: &Tensor) -> candle::Result<Tensor> {
    (value / &(value.neg()? + 1.0)?)?.log()
}

fn atanh_tensor(value: &Tensor) -> candle::Result<Tensor> {
    ((value + 1.0)? / &(value.neg()? + 1.0)?)?.log()? * 0.5
}

fn gradient_plan_cost(
    model: &WorldModel,
    emb: &Tensor,
    action_prefix: &Tensor,
    action_sequence: &Tensor,
    current: &DroneFrame,
    path_anchor: [f32; 3],
    gate: &GateSpec,
    next_gate: Option<&GateSpec>,
    action_stats: &RunningStats,
    target_stats: &RunningStats,
    action_normalized: bool,
    target_normalized: bool,
    args: &Args,
    dtype: DType,
    device: &candle::Device,
) -> candle::Result<Tensor> {
    let (batch, horizon, action_dim) = action_sequence.dims3()?;
    if batch != 1 || action_dim != DRONE_ACTION_DIM {
        candle::bail!(
            "gradient planner action sequence expects [1, horizon, {}], got {:?}",
            DRONE_ACTION_DIM,
            action_sequence.shape()
        );
    }
    let (_, history, emb_dim) = emb.dims3()?;
    let (_, prefix_len, prefix_dim) = action_prefix.dims3()?;
    if prefix_len + 1 != history || prefix_dim != DRONE_ACTION_DIM {
        candle::bail!(
            "gradient planner action prefix must be [1, {}, {}], got {:?}",
            history - 1,
            DRONE_ACTION_DIM,
            action_prefix.shape()
        );
    }

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
    let model_future_actions = if action_normalized {
        action_sequence
            .broadcast_sub(&action_mean)?
            .broadcast_div(&action_std)?
    } else {
        action_sequence.clone()
    };
    let model_actions = Tensor::cat(
        &[
            &action_prefix.unsqueeze(1)?,
            &model_future_actions.unsqueeze(1)?,
        ],
        2,
    )?;
    let rollout =
        model.rollout_embeddings_with_history(&emb.unsqueeze(1)?, &model_actions, history)?;
    let rollout_time = rollout.dim(2)?;
    let pred = model.predict_state_deltas_from_embeddings(&rollout.reshape((
        1,
        rollout_time,
        emb_dim,
    ))?)?;
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
    let deltas = if target_normalized {
        pred.broadcast_mul(&target_std)?
            .broadcast_add(&target_mean)?
    } else {
        pred
    };
    let deltas = deltas.i((0, history..history + horizon, ..))?;

    let mut pos = tensor_vec3(current.pos_world, dtype, device)?;
    let mut rot = Tensor::from_vec(current.rotmat_world_from_body.to_vec(), (3, 3), device)?
        .to_dtype(dtype)?;
    let anchor = tensor_vec3(path_anchor, dtype, device)?;
    let gate_center = tensor_vec3(gate.center, dtype, device)?;
    let gate_normal = tensor_vec3(gate.normal, dtype, device)?;
    let gate_right = tensor_vec3(gate.right, dtype, device)?;
    let gate_up = tensor_vec3(gate.up, dtype, device)?;
    let next_gate_center = next_gate
        .map(|gate| tensor_vec3(gate.center, dtype, device))
        .transpose()?;
    let segment = (&gate_center - &anchor)?;
    let segment_len2 = (tensor_dot3(&segment, &segment)? + 1e-6)?;
    let segment_cpu = sub3(gate.center, path_anchor);
    let segment_len2_cpu = dot3(segment_cpu, segment_cpu).max(1e-6);
    let segment_len_cpu = segment_len2_cpu.sqrt().max(1e-3);
    let current_t_cpu = (dot3(sub3(current.pos_world, path_anchor), segment_cpu)
        / segment_len2_cpu)
        .clamp(0.0, 1.0);
    let lookahead_t_cpu = (args.carrot_lookahead as f32 / segment_len_cpu).min(1.0);

    let mut total = Tensor::new(0f32, device)?.to_dtype(dtype)?;
    let mut terminal_path = Tensor::new(0f32, device)?.to_dtype(dtype)?;
    for step in 0..horizon {
        let delta = deltas.i(step)?;
        let dp = delta.i(0..3)?;
        let rv = delta.i(3..6)?;
        let lv = delta.i(6..9)?;
        let world_dp = tensor_mat3_vec3(&rot, &dp)?;
        pos = (&pos + &world_dp)?;
        let delta_rot = tensor_rotmat_from_rotvec(&rv)?;
        rot = rot.matmul(&delta_rot)?;

        let step_frac = (step + 1) as f32 / horizon as f32;
        let carrot_t = (current_t_cpu + lookahead_t_cpu * step_frac).min(1.0) as f64;
        let progress_t = (current_t_cpu + (1.0 - current_t_cpu) * step_frac).min(1.0) as f64;
        let carrot = (&anchor + &(&segment * carrot_t)?)?;
        let carrot_error = tensor_norm_sq(&(&pos - &carrot)?)?;
        let path_rel = (&pos - &anchor)?;
        let path_t = (tensor_dot3(&path_rel, &segment)? / &segment_len2)?.clamp(0.0, 1.0)?;
        let segment_at_t = segment.broadcast_mul(&path_t.reshape((1,))?)?;
        let closest = (&anchor + &segment_at_t)?;
        let track_error = tensor_norm_sq(&(&pos - &closest)?)?;
        let progress_error = ((&path_t - progress_t)?.sqr()? * segment_len2_cpu as f64)?;
        let rel = (&pos - &gate_center)?;
        let plane = tensor_dot3(&rel, &gate_normal)?.sqr()?;
        let lateral = (tensor_dot3(&rel, &gate_right)?.abs()? - gate.half_width as f64)?
            .relu()?
            .sqr()?;
        let vertical = (tensor_dot3(&rel, &gate_up)?.abs()? - gate.half_height as f64)?
            .relu()?
            .sqr()?;
        let altitude_low = (pos.i(2)?.neg()? + args.min_altitude)?.relu()?.sqr()?;
        let speed_excess = (tensor_norm_sq(&lv)?.sqrt()? - args.max_speed)?
            .relu()?
            .sqr()?;
        let mut step_cost = (&carrot_error * args.path_weight)?;
        step_cost = (step_cost + (&track_error * (args.path_weight * 0.35))?)?;
        step_cost = (step_cost + (&progress_error * args.progress_weight)?)?;
        step_cost = (step_cost + (&plane * args.gate_weight)?)?;
        let lateral_term = (&lateral * (args.gate_weight * 4.0))?;
        step_cost = (&step_cost + &lateral_term)?;
        let vertical_term = (&vertical * (args.gate_weight * 4.0))?;
        step_cost = (&step_cost + &vertical_term)?;
        let altitude_term = (altitude_low * 50.0)?;
        step_cost = (&step_cost + &altitude_term)?;
        let speed_term = (speed_excess * 0.02)?;
        step_cost = (&step_cost + &speed_term)?;
        if let Some(next_center) = next_gate_center.as_ref() {
            let next_error = tensor_norm_sq(&(&pos - next_center)?)?;
            let next_term = (next_error * (args.next_gate_weight * 0.02))?;
            step_cost = (&step_cost + &next_term)?;
        }
        let weight = 1.0 / (1.0 + step as f64 * 0.04);
        total = (total + (&step_cost * weight)?)?;
        terminal_path = carrot_error;
    }
    total = (total / horizon as f64)?;
    total = (total + (terminal_path * 0.25)?)?;
    let terminal_gate = tensor_norm_sq(&(&pos - &gate_center)?)?;
    total = (total + (terminal_gate * args.terminal_gate_weight)?)?;

    let action_effort = action_sequence.sqr()?.mean_all()?;
    total = (total + (action_effort * 1e-3)?)?;
    if horizon > 1 {
        let tail = action_sequence.narrow(1, 1, horizon - 1)?;
        let head = action_sequence.narrow(1, 0, horizon - 1)?;
        let smoothness = (tail - head)?.sqr()?.mean_all()?;
        total = (total + (smoothness * 2e-3)?)?;
    }
    Ok(total)
}

fn tensor_vec3(value: [f32; 3], dtype: DType, device: &candle::Device) -> candle::Result<Tensor> {
    Tensor::from_vec(value.to_vec(), (3,), device)?.to_dtype(dtype)
}

fn tensor_dot3(lhs: &Tensor, rhs: &Tensor) -> candle::Result<Tensor> {
    lhs.broadcast_mul(rhs)?.sum_all()
}

fn tensor_norm_sq(value: &Tensor) -> candle::Result<Tensor> {
    value.sqr()?.sum_all()
}

fn tensor_mat3_vec3(mat: &Tensor, vec: &Tensor) -> candle::Result<Tensor> {
    mat.matmul(&vec.reshape((3, 1))?)?.reshape((3,))
}

fn tensor_rotmat_from_rotvec(rv: &Tensor) -> candle::Result<Tensor> {
    let theta = (tensor_norm_sq(rv)? + 1e-12)?.sqrt()?;
    let axis = rv.broadcast_div(&theta.reshape((1,))?)?;
    let x = axis.i(0)?;
    let y = axis.i(1)?;
    let z = axis.i(2)?;
    let c = theta.cos()?;
    let s = theta.sin()?;
    let one_minus_c = (c.neg()? + 1.0)?;
    let xx = (&x * &x)?;
    let yy = (&y * &y)?;
    let zz = (&z * &z)?;
    let xy = (&x * &y)?;
    let xz = (&x * &z)?;
    let yz = (&y * &z)?;
    let xs = (&x * &s)?;
    let ys = (&y * &s)?;
    let zs = (&z * &s)?;
    let d00 = (&c + &(&xx * &one_minus_c)?)?;
    let d01 = (&(&xy * &one_minus_c)? - &zs)?;
    let d02 = (&(&xz * &one_minus_c)? + &ys)?;
    let d10 = (&(&xy * &one_minus_c)? + &zs)?;
    let d11 = (&c + &(&yy * &one_minus_c)?)?;
    let d12 = (&(&yz * &one_minus_c)? - &xs)?;
    let d20 = (&(&xz * &one_minus_c)? - &ys)?;
    let d21 = (&(&yz * &one_minus_c)? + &xs)?;
    let d22 = (&c + &(&zz * &one_minus_c)?)?;
    Tensor::stack(&[&d00, &d01, &d02, &d10, &d11, &d12, &d20, &d21, &d22], 0)?.reshape((3, 3))
}

struct DroneGateScorer<'a> {
    model: &'a WorldModel,
    emb: Tensor,
    action_prefix: Tensor,
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
        let future_actions = if self.action_normalized {
            action_candidates
                .broadcast_sub(&self.action_mean)?
                .broadcast_div(&self.action_std)?
        } else {
            action_candidates.clone()
        };
        let (_, history, emb_dim) = self.emb.dims3()?;
        let (_, prefix_len, prefix_dim) = self.action_prefix.dims3()?;
        if prefix_len + 1 != history || prefix_dim != DRONE_ACTION_DIM {
            candle::bail!(
                "drone scorer action prefix must be [1, {}, {}], got {:?}",
                history - 1,
                DRONE_ACTION_DIM,
                self.action_prefix.shape()
            );
        }
        let emb_init = self
            .emb
            .unsqueeze(1)?
            .broadcast_as((1, samples, history, emb_dim))?;
        let prefix = self.action_prefix.unsqueeze(1)?.broadcast_as((
            1,
            samples,
            prefix_len,
            DRONE_ACTION_DIM,
        ))?;
        let model_actions = Tensor::cat(&[&prefix, &future_actions], 2)?;
        let rollout =
            self.model
                .rollout_embeddings_with_history(&emb_init, &model_actions, history)?;
        let future_emb = rollout
            .i((0, .., history..history + _horizon, ..))?
            .contiguous()?;
        let pred = self.model.predict_state_deltas_from_embeddings(
            &future_emb.reshape((samples, _horizon, emb_dim))?,
        )?;
        let deltas = if self.target_normalized {
            pred.broadcast_mul(&self.target_std)?
                .broadcast_add(&self.target_mean)?
        } else {
            pred
        };
        self.score_candidates_cuda(action_candidates, &deltas, _horizon, samples)?
            .reshape((1, samples))
    }
}

impl DroneGateScorer<'_> {
    fn new<'a>(
        model: &'a WorldModel,
        emb: Tensor,
        action_prefix: Tensor,
        current: DroneFrame,
        gate: GateSpec,
        next_gate: Option<GateSpec>,
        action_stats: RunningStats,
        target_stats: RunningStats,
        action_normalized: bool,
        target_normalized: bool,
        device: candle::Device,
        dtype: DType,
        next_gate_weight: f64,
        min_altitude: f64,
        max_speed: f64,
    ) -> candle::Result<DroneGateScorer<'a>> {
        ensure_cuda_f32(&device, dtype)?;
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
            action_prefix,
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
            next_gate_weight,
            min_altitude,
            max_speed,
        })
    }

    fn score_candidates_cuda(
        &self,
        action_candidates: &Tensor,
        deltas: &Tensor,
        rollout_time: usize,
        samples: usize,
    ) -> candle::Result<Tensor> {
        if !deltas.device().is_cuda()
            || deltas.dtype() != DType::F32
            || !action_candidates.device().is_cuda()
            || action_candidates.dtype() != DType::F32
        {
            candle::bail!("drone sampling planner requires CUDA f32 tensors for scoring");
        }
        let (_, _, horizon, action_dim) = action_candidates.dims4()?;
        if action_dim != DRONE_ACTION_DIM {
            candle::bail!(
                "drone CUDA scorer expects action_dim={}, got {action_dim}",
                DRONE_ACTION_DIM
            );
        }
        let start_step = 0usize;
        let has_next_gate = self.next_gate_center.is_some() && self.next_gate_weight > 0.0;
        let next_gate_center = self.next_gate_center.as_ref().unwrap_or(&self.gate_center);

        macro_rules! cuda_f32_view {
            ($tensor:expr, $name:literal, $contig:ident, $storage:ident, $cuda_storage:ident, $slice:ident, $view:ident) => {
                let $contig = $tensor.contiguous()?;
                let ($storage, layout) = $contig.storage_and_layout();
                let Storage::Cuda($cuda_storage) = &*$storage else {
                    candle::bail!(concat!($name, " tensor must be CUDA for drone scoring"));
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
        Ok(Tensor::from_storage(
            Storage::Cuda(storage),
            (samples,),
            BackpropOp::none(),
            false,
        ))
    }
}

fn ensure_cuda_f32(device: &candle::Device, dtype: DType) -> candle::Result<()> {
    if !device.is_cuda() || dtype != DType::F32 {
        candle::bail!("drone sampling planner requires --device cuda and --dtype f32");
    }
    Ok(())
}

fn dot_last(lhs: &Tensor, rhs: &Tensor) -> candle::Result<Tensor> {
    lhs.broadcast_mul(rhs)?.sum(D::Minus1)
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

fn run_gate_loop(
    args: &Args,
    dataset: &DroneRacingDataset,
    model: &WorldModel,
    mut emb: Tensor,
    mut action_prefix: Tensor,
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
    let mut path_anchor = current.pos_world;
    let mut planner_state = DronePlannerState::new(
        args.planner,
        &dataset.metadata().normalization.action.mean,
        args,
        dtype,
        device,
    )?;

    while executed_steps < loop_steps {
        let gate = flight.gates[gate_index].clone();
        let next_gate = next_gate_in_flight(flight, gate_index).cloned();
        let replan_started = Instant::now();
        let plan = planner_state.plan(
            args,
            model,
            &emb,
            &action_prefix,
            &current,
            path_anchor,
            &gate,
            next_gate.as_ref(),
            &dataset.metadata().normalization.action,
            &dataset.metadata().normalization.target_delta,
            !args.no_action_normalize,
            !args.no_target_normalize,
            dtype,
            device,
        )?;
        let plan_elapsed_sec = replan_started.elapsed().as_secs_f64();
        let stride = args
            .control_stride
            .min(plan.sequence.dim(1)?)
            .min(loop_steps - executed_steps);
        let previous = current.clone();
        let advance_started = Instant::now();
        let advance = advance_with_lewm(
            model,
            emb,
            action_prefix,
            current,
            &plan.sequence,
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
        action_prefix = advance.action_prefix;
        current = advance.current;
        let carrot = moving_carrot_point(
            path_anchor,
            &gate,
            previous.pos_world,
            args.carrot_lookahead,
        );
        frames.extend(advance.frames);
        actions.extend(advance.actions);
        executed_steps += stride;
        let passed = gate_passed(previous.pos_world, current.pos_world, &gate);
        replans.push(ReplanStep {
            executed_steps,
            gate_index,
            gate_name: gate.name.clone(),
            passed_gate: passed,
            path_anchor,
            carrot,
            initial_score: plan.initial_score,
            planner: plan.planner,
            planner_evals: plan.planner_evals,
            score_summary: plan.score_summary,
            planner_elapsed_sec: plan_elapsed_sec,
            model_advance_elapsed_sec: advance_elapsed_sec,
            planner_evals_per_sec: throughput(plan.planner_evals, plan_elapsed_sec),
        });
        if passed {
            path_anchor = gate.center;
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
    let total_planner_evals = replans
        .iter()
        .map(|replan| replan.planner_evals)
        .sum::<usize>();
    let planner_budget = planner_budget_summary(args);
    let planner_benchmark = loop_planner_benchmark(
        args,
        loop_steps,
        executed_steps,
        &replans,
        total_planner_elapsed_sec,
        total_planner_evals,
    );
    Ok(LoopPlanReport {
        dataset_dir: dataset.root().to_path_buf(),
        weights: args.weights.clone().unwrap_or_else(default_weights),
        config: args.config.clone().unwrap_or_else(default_config),
        episode_idx: flight.episode_idx,
        flight: flight.flight.clone(),
        carrot_lookahead: args.carrot_lookahead,
        path_weight: args.path_weight,
        progress_weight: args.progress_weight,
        gate_weight: args.gate_weight,
        terminal_gate_weight: args.terminal_gate_weight,
        next_gate_weight: args.next_gate_weight,
        horizon: args.horizon,
        planner: args.planner,
        samples: args.samples,
        elites: args.elites,
        keep_elites: args.keep_elites,
        iterations: args.iterations,
        grad_steps: args.grad_steps,
        grad_lr: args.grad_lr,
        grad_weight_decay: args.grad_weight_decay,
        control_stride: args.control_stride,
        requested_loop_steps: loop_steps,
        executed_steps,
        completed_laps,
        next_gate_index: gate_index,
        total_replans: replans.len(),
        total_planner_elapsed_sec,
        total_planner_evals,
        planner_evals_per_sec: throughput(total_planner_evals, total_planner_elapsed_sec),
        planner_budget,
        planner_benchmark,
        sample_rate_hz: dataset.metadata().sample_rate_hz,
        gate_loop: flight.gates.clone(),
        frames,
        actions,
        replans,
    })
}

struct AdvanceResult {
    emb: Tensor,
    action_prefix: Tensor,
    current: DroneFrame,
    frames: Vec<DroneFrame>,
    actions: Vec<[f32; DRONE_ACTION_DIM]>,
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

fn advance_with_lewm(
    model: &WorldModel,
    emb: Tensor,
    action_prefix: Tensor,
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
    let prefix_dims = action_prefix.dims();
    ensure!(
        prefix_dims == [1, history_steps - 1, DRONE_ACTION_DIM],
        "action prefix must have shape [1, {}, {}], got {:?}",
        history_steps - 1,
        DRONE_ACTION_DIM,
        action_prefix.shape()
    );
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
    let action_prefix = action_prefix.unsqueeze(1)?;
    let future_actions = model_action_sequence.unsqueeze(1)?;
    let model_actions = Tensor::cat(&[&action_prefix, &future_actions], 2)?;
    let emb_init = emb.unsqueeze(1)?;
    let rollout =
        model.rollout_embeddings_with_history(&emb_init, &model_actions, history_steps)?;
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
    let executed_actions = action_values
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
    let action_prefix = model_actions
        .i((0, 0, stride..stride + history_steps - 1, ..))?
        .unsqueeze(0)?;
    Ok(AdvanceResult {
        emb,
        action_prefix,
        current,
        frames,
        actions: executed_actions,
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

fn scale3(value: [f32; 3], scale: f32) -> [f32; 3] {
    [value[0] * scale, value[1] * scale, value[2] * scale]
}

fn moving_carrot_point(
    path_anchor: [f32; 3],
    gate: &GateSpec,
    current_pos: [f32; 3],
    lookahead_m: f64,
) -> [f32; 3] {
    let segment = sub3(gate.center, path_anchor);
    let segment_len2 = dot3(segment, segment).max(1e-6);
    let segment_len = segment_len2.sqrt().max(1e-3);
    let current_t = (dot3(sub3(current_pos, path_anchor), segment) / segment_len2).clamp(0.0, 1.0);
    let carrot_t = (current_t + lookahead_m as f32 / segment_len).min(1.0);
    add3(path_anchor, scale3(segment, carrot_t))
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
    carrot_lookahead: f64,
    path_weight: f64,
    progress_weight: f64,
    gate_weight: f64,
    terminal_gate_weight: f64,
    next_gate_weight: f64,
    path_anchor: [f32; 3],
    carrot: [f32; 3],
    horizon: usize,
    planner: PlannerKind,
    samples: usize,
    elites: usize,
    keep_elites: usize,
    iterations: usize,
    grad_steps: usize,
    grad_lr: f64,
    planner_budget: PlannerBudgetSummary,
    planner_evals: usize,
    planner_elapsed_sec: f64,
    planner_evals_per_sec: f64,
    score_summary: ScoreSummary,
    best_sequence: Vec<[f32; 4]>,
}

#[derive(Debug, Serialize)]
struct LoopPlanReport {
    dataset_dir: PathBuf,
    weights: PathBuf,
    config: PathBuf,
    episode_idx: i64,
    flight: String,
    carrot_lookahead: f64,
    path_weight: f64,
    progress_weight: f64,
    gate_weight: f64,
    terminal_gate_weight: f64,
    next_gate_weight: f64,
    horizon: usize,
    planner: PlannerKind,
    samples: usize,
    elites: usize,
    keep_elites: usize,
    iterations: usize,
    grad_steps: usize,
    grad_lr: f64,
    grad_weight_decay: f64,
    control_stride: usize,
    requested_loop_steps: usize,
    executed_steps: usize,
    completed_laps: usize,
    next_gate_index: usize,
    total_replans: usize,
    total_planner_elapsed_sec: f64,
    total_planner_evals: usize,
    planner_evals_per_sec: f64,
    planner_budget: PlannerBudgetSummary,
    planner_benchmark: LoopPlannerBenchmark,
    sample_rate_hz: usize,
    gate_loop: Vec<GateSpec>,
    frames: Vec<DroneFrame>,
    actions: Vec<[f32; DRONE_ACTION_DIM]>,
    replans: Vec<ReplanStep>,
}

#[derive(Debug, Serialize)]
struct PlannerBudgetSummary {
    current_per_replan_evals: usize,
    legacy_icem_per_replan_evals: usize,
    current_per_replan_model_steps: usize,
    legacy_icem_per_replan_model_steps: usize,
    current_horizon: usize,
    legacy_icem_horizon: usize,
    current_control_stride: usize,
    legacy_icem_control_stride: usize,
    current_samples: usize,
    legacy_icem_samples: usize,
    current_keep_elites: usize,
    legacy_icem_keep_elites: usize,
    current_iterations: usize,
    legacy_icem_iterations: usize,
    per_replan_eval_ratio_vs_legacy_icem: f64,
    per_replan_eval_reduction_pct_vs_legacy_icem: f64,
    per_replan_model_step_ratio_vs_legacy_icem: f64,
    per_replan_model_step_reduction_pct_vs_legacy_icem: f64,
}

#[derive(Debug, Serialize)]
struct LoopPlannerBenchmark {
    current_expected_replans: usize,
    legacy_icem_expected_replans: usize,
    current_budget_total_evals: usize,
    legacy_icem_budget_total_evals: usize,
    current_budget_total_model_steps: usize,
    legacy_icem_budget_total_model_steps: usize,
    budget_eval_ratio_vs_legacy_icem: f64,
    budget_eval_reduction_pct_vs_legacy_icem: f64,
    budget_model_step_ratio_vs_legacy_icem: f64,
    budget_model_step_reduction_pct_vs_legacy_icem: f64,
    actual_total_planner_elapsed_sec: f64,
    actual_total_planner_evals: usize,
    actual_planner_evals_per_sec: f64,
    actual_planner_evals_per_executed_step: f64,
    actual_planner_ms_per_executed_step: f64,
    planner_elapsed: TimingSummary,
    model_advance_elapsed: TimingSummary,
}

#[derive(Debug, Serialize)]
struct TimingSummary {
    count: usize,
    total_sec: f64,
    mean_sec: f64,
    p50_sec: f64,
    p90_sec: f64,
    min_sec: f64,
    max_sec: f64,
}

#[derive(Debug, Serialize)]
struct ReplanStep {
    executed_steps: usize,
    gate_index: usize,
    gate_name: String,
    passed_gate: bool,
    path_anchor: [f32; 3],
    carrot: [f32; 3],
    initial_score: Option<f32>,
    planner: PlannerKind,
    planner_evals: usize,
    score_summary: ScoreSummary,
    planner_elapsed_sec: f64,
    model_advance_elapsed_sec: f64,
    planner_evals_per_sec: f64,
}
