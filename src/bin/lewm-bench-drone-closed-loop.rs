use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, ensure};
use candle::{D, DType, IndexOp, Tensor};
use clap::Parser;
use le_wm_nv::{
    checkpoint,
    data::drone_racing::{
        DRONE_ACTION_DIM, DRONE_STATE_DELTA_DIM, DroneBatchConfig, DroneFrame, DroneRacingDataset,
        RunningStats, dot3, mat3_t_mul_vec3, norm3, scale3, sub3,
    },
    drone_eval::{baseline_action, drone_action_bounds, history_action_prefix, rollout_one_step},
    models::world_model::{WorldModel, WorldModelConfig},
    planner::{CandidateScorer, IcemConfig, IcemPlanner},
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

    /// Output JSON report.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Dataset row used as the initial history start.
    #[arg(long)]
    row: Option<usize>,

    #[arg(long, default_value_t = DeviceSpec::Cuda(0))]
    device: DeviceSpec,

    #[arg(long, default_value_t = DTypeSpec::F32)]
    dtype: DTypeSpec,

    #[arg(long, default_value_t = 8)]
    history_steps: usize,

    #[arg(long, default_value_t = 40)]
    horizon: usize,

    #[arg(long, default_value_t = 128)]
    samples: usize,

    #[arg(long, default_value_t = 32)]
    elites: usize,

    #[arg(long, default_value_t = 8)]
    keep_elites: usize,

    #[arg(long, default_value_t = 2)]
    iterations: usize,

    #[arg(long, default_value_t = 40)]
    loop_steps: usize,

    /// Desired local body-axis displacement over one planner horizon.
    #[arg(long, default_value_t = 0.75)]
    target_distance_m: f32,

    /// Desired local yaw change over one planner horizon.
    #[arg(long, default_value_t = 0.35)]
    target_yaw_rad: f32,

    #[arg(long, default_value_t = 11)]
    seed: u64,

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
    let output = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from("target/drone-eval/closed-loop.json"));
    let device = args.device.resolve()?;
    ensure!(
        device.is_cuda(),
        "closed-loop drone benchmark requires CUDA"
    );
    let dtype = args.dtype.dtype();
    if dtype != DType::F32 {
        anyhow::bail!("closed-loop drone benchmark currently requires --dtype f32");
    }

    let cfg: WorldModelConfig = serde_json::from_str(
        &fs::read_to_string(&config)
            .with_context(|| format!("failed to read {}", config.display()))?,
    )
    .with_context(|| format!("failed to parse {}", config.display()))?;
    let batch_cfg = DroneBatchConfig {
        batch_size: 1,
        sequence_steps: cfg.predictor.num_frames,
        normalize_observations: !args.no_observation_normalize,
        normalize_actions: !args.no_action_normalize,
        normalize_targets: !args.no_target_normalize,
    };
    let dataset = DroneRacingDataset::open(&dataset_dir, batch_cfg)?;
    let row = args.row.unwrap_or(40847);
    ensure!(
        row + args.history_steps < dataset.metadata().rows,
        "row {row} does not have enough history"
    );
    let vb = checkpoint::var_builder_from_path(&weights, dtype, &device)
        .with_context(|| format!("failed to load {}", weights.display()))?;
    let model = WorldModel::new(cfg, vb)?;
    let action_trim = baseline_action(&dataset.metadata().normalization.action)?;
    let tasks = default_tasks(args.target_distance_m, args.target_yaw_rad);
    let mut task_results = Vec::with_capacity(tasks.len());
    let started = Instant::now();

    for (task_idx, task) in tasks.iter().enumerate() {
        let result = run_task(
            &args,
            task,
            task_idx,
            &dataset,
            &model,
            row,
            action_trim,
            dtype,
            &device,
        )?;
        println!(
            "task={} final_body=({:.3},{:.3},{:.3}) path={:.3} planner_ms/step={:.2}",
            result.name,
            result.net_displacement_body[0],
            result.net_displacement_body[1],
            result.net_displacement_body[2],
            result.path_length_m,
            result.mean_planner_ms
        );
        task_results.push(result);
    }

    let report = ClosedLoopReport {
        dataset_dir,
        weights,
        config,
        output: output.clone(),
        row,
        history_steps: args.history_steps,
        horizon: args.horizon,
        samples: args.samples,
        elites: args.elites,
        keep_elites: args.keep_elites,
        iterations: args.iterations,
        loop_steps: args.loop_steps,
        sample_rate_hz: dataset.metadata().sample_rate_hz,
        benchmark_elapsed_sec: started.elapsed().as_secs_f64(),
        tasks: task_results,
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
    ensure!(args.horizon > 0, "--horizon must be greater than zero");
    ensure!(args.samples > 0, "--samples must be greater than zero");
    ensure!(args.elites >= 2, "--elites must be at least two");
    ensure!(args.elites <= args.samples, "--elites must be <= --samples");
    ensure!(
        args.keep_elites <= args.elites,
        "--keep-elites must be <= --elites"
    );
    ensure!(
        args.iterations > 0,
        "--iterations must be greater than zero"
    );
    ensure!(
        args.loop_steps > 0,
        "--loop-steps must be greater than zero"
    );
    Ok(())
}

fn run_task(
    args: &Args,
    task: &TaskSpec,
    task_idx: usize,
    dataset: &DroneRacingDataset,
    model: &WorldModel,
    row: usize,
    action_trim: [f32; DRONE_ACTION_DIM],
    dtype: DType,
    device: &candle::Device,
) -> anyhow::Result<TaskResult> {
    let mut cfg = IcemConfig::new(args.horizon, args.samples, args.elites, DRONE_ACTION_DIM);
    cfg.keep_elites = args.keep_elites;
    cfg.iterations = args.iterations;
    cfg.action_bounds = drone_action_bounds();
    cfg.init_std = 0.35;
    cfg.min_std = 0.02;
    cfg.seed = Some(args.seed + task_idx as u64 * 1_003);
    let mut planner = IcemPlanner::new(cfg);
    let mut context = initial_context(
        dataset,
        model,
        row,
        args.history_steps,
        action_trim,
        dtype,
        device,
    )?;
    let initial = context.current.clone();
    let mut frames = Vec::with_capacity(args.loop_steps + 1);
    let mut step_records = Vec::with_capacity(args.loop_steps);
    frames.push(context.current.clone());
    let mut total_planner_sec = 0.0f64;
    let mut total_iterations = 0usize;
    let mut total_action = [0.0f32; DRONE_ACTION_DIM];

    for step in 0..args.loop_steps {
        let scorer = ShortHorizonScorer::new(
            model,
            context.emb.clone(),
            context.action_prefix.clone(),
            task.target_pos_body,
            task.target_rot_body,
            dataset.metadata().normalization.action.clone(),
            dataset.metadata().normalization.target_delta.clone(),
            !args.no_action_normalize,
            !args.no_target_normalize,
            dtype,
            device.clone(),
        )?;
        let plan = planner.plan_device(&scorer)?;
        let action_values = plan.first_action.flatten_all()?.to_vec1::<f32>()?;
        let action = [
            action_values[0].clamp(-1.0, 1.0),
            action_values[1].clamp(-1.0, 1.0),
            action_values[2].clamp(0.0, 1.0),
            action_values[3].clamp(-1.0, 1.0),
        ];
        let (next, emb, action_prefix) = rollout_one_step(
            model,
            &context.emb,
            &context.action_prefix,
            &context.current,
            action,
            &dataset.metadata().normalization.action,
            &dataset.metadata().normalization.target_delta,
            !args.no_action_normalize,
            !args.no_target_normalize,
            dtype,
            device,
        )?;
        total_planner_sec += plan.elapsed.as_secs_f64();
        total_iterations += plan.iterations_completed;
        for idx in 0..DRONE_ACTION_DIM {
            total_action[idx] += action[idx].abs();
        }
        step_records.push(StepRecord {
            step,
            action,
            planner_ms: plan.elapsed.as_secs_f64() * 1000.0,
            iterations_completed: plan.iterations_completed,
            pos_world: next.pos_world,
            lin_vel_body: next.lin_vel_body,
            speed_mps: norm3(next.lin_vel_body),
        });
        context = SimContext {
            current: next.clone(),
            emb,
            action_prefix,
        };
        frames.push(next);
    }

    let final_frame = context.current.clone();
    let displacement_world = sub3(final_frame.pos_world, initial.pos_world);
    let displacement_body = mat3_t_mul_vec3(initial.rotmat_world_from_body, displacement_world);
    let path_length_m = path_length(&frames);
    let desired_body = scale3(
        task.target_pos_body,
        args.loop_steps as f32 / args.horizon as f32,
    );
    let desired_norm = norm3(desired_body);
    let progress_along_target_m = if desired_norm > 1e-6 {
        dot3(displacement_body, desired_body) / desired_norm
    } else {
        0.0
    };
    let cross_track_m = if desired_norm > 1e-6 {
        let target_unit = scale3(desired_body, 1.0 / desired_norm);
        norm3(sub3(
            displacement_body,
            scale3(target_unit, dot3(displacement_body, target_unit)),
        ))
    } else {
        norm3(displacement_body)
    };
    let mean_action_abs = [
        total_action[0] / args.loop_steps as f32,
        total_action[1] / args.loop_steps as f32,
        total_action[2] / args.loop_steps as f32,
        total_action[3] / args.loop_steps as f32,
    ];

    Ok(TaskResult {
        name: task.name.to_string(),
        target_pos_body_per_horizon: task.target_pos_body,
        target_rot_body_per_horizon: task.target_rot_body,
        expected_body_displacement: desired_body,
        initial,
        final_frame,
        net_displacement_world: displacement_world,
        net_displacement_body: displacement_body,
        progress_along_target_m,
        cross_track_m,
        path_length_m,
        mean_planner_ms: (total_planner_sec * 1000.0 / args.loop_steps as f64) as f32,
        replans_per_sec: (args.loop_steps as f64 / total_planner_sec.max(1e-9)) as f32,
        mean_iterations: total_iterations as f32 / args.loop_steps as f32,
        mean_action_abs,
        steps: step_records,
    })
}

fn initial_context(
    dataset: &DroneRacingDataset,
    model: &WorldModel,
    row: usize,
    history_steps: usize,
    action_trim: [f32; DRONE_ACTION_DIM],
    dtype: DType,
    device: &candle::Device,
) -> anyhow::Result<SimContext> {
    let history = dataset.batch(&[row], dtype, device)?;
    let emb_all = model.encode_vector(&history.observations)?;
    let emb = emb_all.i((.., 0..history_steps, ..))?;
    let action_prefix = history_action_prefix(&history.actions, history_steps)?;
    let mut current = dataset.frame(row + history_steps - 1)?;
    current.channels_norm = action_trim;
    Ok(SimContext {
        current,
        emb,
        action_prefix,
    })
}

struct ShortHorizonScorer<'a> {
    model: &'a WorldModel,
    emb: Tensor,
    action_prefix: Tensor,
    target_pos: Tensor,
    target_rot: Tensor,
    action_mean: Tensor,
    action_std: Tensor,
    target_mean: Tensor,
    target_std: Tensor,
    action_normalized: bool,
    target_normalized: bool,
    dtype: DType,
    device: candle::Device,
    position_weight: f64,
    rotation_weight: f64,
    velocity_weight: f64,
    terminal_weight: f64,
    action_weight: f64,
    smoothness_weight: f64,
}

impl<'a> ShortHorizonScorer<'a> {
    fn new(
        model: &'a WorldModel,
        emb: Tensor,
        action_prefix: Tensor,
        target_pos_body: [f32; 3],
        target_rot_body: [f32; 3],
        action_stats: RunningStats,
        target_stats: RunningStats,
        action_normalized: bool,
        target_normalized: bool,
        dtype: DType,
        device: candle::Device,
    ) -> candle::Result<Self> {
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
        Ok(Self {
            model,
            emb,
            action_prefix,
            target_pos: Tensor::from_vec(target_pos_body.to_vec(), (1, 3), &device)?
                .to_dtype(dtype)?,
            target_rot: Tensor::from_vec(target_rot_body.to_vec(), (1, 3), &device)?
                .to_dtype(dtype)?,
            action_mean,
            action_std,
            target_mean,
            target_std,
            action_normalized,
            target_normalized,
            dtype,
            device,
            position_weight: 1.0,
            rotation_weight: 0.35,
            velocity_weight: 0.015,
            terminal_weight: 1.5,
            action_weight: 2e-3,
            smoothness_weight: 4e-3,
        })
    }
}

impl CandidateScorer for ShortHorizonScorer<'_> {
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
        let (batch, samples, horizon, action_dim) = action_candidates.dims4()?;
        if batch != 1 || action_dim != DRONE_ACTION_DIM {
            candle::bail!(
                "closed-loop scorer expects [1, samples, horizon, {}], got {:?}",
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
                "action prefix must be [1, {}, {}], got {:?}",
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
            .i((0, .., history..history + horizon, ..))?
            .contiguous()?;
        let pred = self.model.predict_state_deltas_from_embeddings(
            &future_emb.reshape((samples, horizon, emb_dim))?,
        )?;
        let deltas = if self.target_normalized {
            pred.broadcast_mul(&self.target_std)?
                .broadcast_add(&self.target_mean)?
        } else {
            pred
        };

        let mut pos = Tensor::zeros((samples, 3), self.dtype, &self.device)?;
        let mut rot = Tensor::zeros((samples, 3), self.dtype, &self.device)?;
        let mut total = Tensor::zeros((samples,), self.dtype, &self.device)?;
        let mut terminal_pos_err = Tensor::zeros((samples,), self.dtype, &self.device)?;
        let mut terminal_rot_err = Tensor::zeros((samples,), self.dtype, &self.device)?;
        for step in 0..horizon {
            let delta = deltas.i((.., step, ..))?;
            let dp = delta.i((.., 0..3))?;
            let dr = delta.i((.., 3..6))?;
            let lv = delta.i((.., 6..9))?;
            pos = (&pos + &dp)?;
            rot = (&rot + &dr)?;
            let frac = (step + 1) as f64 / horizon as f64;
            let target_pos = (&self.target_pos * frac)?;
            let target_rot = (&self.target_rot * frac)?;
            let pos_err = pos.broadcast_sub(&target_pos)?.sqr()?.sum(D::Minus1)?;
            let rot_err = rot.broadcast_sub(&target_rot)?.sqr()?.sum(D::Minus1)?;
            let vel_err = lv.sqr()?.sum(D::Minus1)?;
            let step_cost = (&pos_err * self.position_weight)?;
            let step_cost = (&step_cost + &(rot_err.clone() * self.rotation_weight)?)?;
            let step_cost = (&step_cost + &(vel_err * self.velocity_weight)?)?;
            let weight = 1.0 / (1.0 + step as f64 * 0.03);
            total = (&total + &(step_cost * weight)?)?;
            terminal_pos_err = pos_err;
            terminal_rot_err = rot_err;
        }
        total = (total / horizon as f64)?;
        total = (&total + &(terminal_pos_err * self.terminal_weight)?)?;
        total = (&total + &(terminal_rot_err * (self.terminal_weight * self.rotation_weight))?)?;

        let action_flat = action_candidates
            .i(0)?
            .reshape((samples, horizon * action_dim))?;
        let action_effort = (action_flat.sqr()?.sum(D::Minus1)? / (horizon * action_dim) as f64)?;
        total = (&total + &(action_effort * self.action_weight)?)?;
        if horizon > 1 {
            let tail = action_candidates.i((0, .., 1..horizon, ..))?;
            let head = action_candidates.i((0, .., 0..horizon - 1, ..))?;
            let smooth = ((tail - head)?
                .reshape((samples, (horizon - 1) * action_dim))?
                .sqr()?
                .sum(D::Minus1)?
                / ((horizon - 1) * action_dim) as f64)?;
            total = (&total + &(smooth * self.smoothness_weight)?)?;
        }
        total.reshape((1, samples))
    }
}

fn default_tasks(distance: f32, yaw: f32) -> Vec<TaskSpec> {
    vec![
        TaskSpec {
            name: "hold",
            target_pos_body: [0.0, 0.0, 0.0],
            target_rot_body: [0.0, 0.0, 0.0],
        },
        TaskSpec {
            name: "body_x",
            target_pos_body: [distance, 0.0, 0.0],
            target_rot_body: [0.0, 0.0, 0.0],
        },
        TaskSpec {
            name: "body_y",
            target_pos_body: [0.0, distance, 0.0],
            target_rot_body: [0.0, 0.0, 0.0],
        },
        TaskSpec {
            name: "body_z",
            target_pos_body: [0.0, 0.0, distance],
            target_rot_body: [0.0, 0.0, 0.0],
        },
        TaskSpec {
            name: "yaw_z",
            target_pos_body: [0.0, 0.0, 0.0],
            target_rot_body: [0.0, 0.0, yaw],
        },
    ]
}

fn path_length(frames: &[DroneFrame]) -> f32 {
    frames
        .windows(2)
        .map(|pair| norm3(sub3(pair[1].pos_world, pair[0].pos_world)))
        .sum()
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

fn default_weights() -> PathBuf {
    default_run_dir().join("final.safetensors")
}

fn default_config() -> PathBuf {
    default_run_dir().join("model-config.json")
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

struct SimContext {
    current: DroneFrame,
    emb: Tensor,
    action_prefix: Tensor,
}

struct TaskSpec {
    name: &'static str,
    target_pos_body: [f32; 3],
    target_rot_body: [f32; 3],
}

#[derive(Debug, Serialize)]
struct ClosedLoopReport {
    dataset_dir: PathBuf,
    weights: PathBuf,
    config: PathBuf,
    output: PathBuf,
    row: usize,
    history_steps: usize,
    horizon: usize,
    samples: usize,
    elites: usize,
    keep_elites: usize,
    iterations: usize,
    loop_steps: usize,
    sample_rate_hz: usize,
    benchmark_elapsed_sec: f64,
    tasks: Vec<TaskResult>,
}

#[derive(Debug, Serialize)]
struct TaskResult {
    name: String,
    target_pos_body_per_horizon: [f32; 3],
    target_rot_body_per_horizon: [f32; 3],
    expected_body_displacement: [f32; 3],
    initial: DroneFrame,
    final_frame: DroneFrame,
    net_displacement_world: [f32; 3],
    net_displacement_body: [f32; 3],
    progress_along_target_m: f32,
    cross_track_m: f32,
    path_length_m: f32,
    mean_planner_ms: f32,
    replans_per_sec: f32,
    mean_iterations: f32,
    mean_action_abs: [f32; DRONE_ACTION_DIM],
    steps: Vec<StepRecord>,
}

#[derive(Debug, Serialize)]
struct StepRecord {
    step: usize,
    action: [f32; DRONE_ACTION_DIM],
    planner_ms: f64,
    iterations_completed: usize,
    pos_world: [f32; 3],
    lin_vel_body: [f32; 3],
    speed_mps: f32,
}
