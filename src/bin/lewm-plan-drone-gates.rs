use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, ensure};
use candle::{D, DType, IndexOp, Tensor};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use le_wm_nv::{
    data::drone_racing::{
        DRONE_ACTION_DIM, DRONE_STATE_DELTA_DIM, DroneBatchConfig, DroneFrame, DroneRacingDataset,
        GateSequenceFile, GateSpec, RunningStats,
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
    let gate = select_gate(&gates, dataset.frame(row)?.episode_idx, args.gate_index)?;
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
    let scorer = DroneGateScorer::new(
        &model,
        emb,
        current,
        gate,
        dataset.metadata().normalization.action.clone(),
        dataset.metadata().normalization.target_delta.clone(),
        !args.no_action_normalize,
        !args.no_target_normalize,
        device.clone(),
        dtype,
        args.history_steps,
    )?;
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
    let planner = CemPlanner::new(cfg);
    let result = planner.plan(&scorer)?;
    let sequence = result.sequence.flatten_all()?.to_vec1::<f32>()?;
    let scores = result.scores.flatten_all()?.to_vec1::<f32>()?;
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
        best_indices: result.best_indices,
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
    Ok(())
}

struct DroneGateScorer<'a> {
    model: &'a WorldModel,
    emb: Tensor,
    gate: GateSpec,
    current_pos: Tensor,
    current_rot: Tensor,
    action_mean: Tensor,
    action_std: Tensor,
    target_mean: Tensor,
    target_std: Tensor,
    gate_center: Tensor,
    gate_normal: Tensor,
    gate_right: Tensor,
    gate_up: Tensor,
    action_normalized: bool,
    target_normalized: bool,
    device: candle::Device,
    dtype: DType,
    history_steps: usize,
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
        let gate_scores = self.score_rollout(&deltas, rollout_time, samples)?;
        let action_effort =
            (action_candidates.sqr()?.sum(D::Minus1)?.sum(D::Minus1)? * 1e-3)?.squeeze(0)?;
        (gate_scores + action_effort)?.reshape((1, samples))
    }
}

impl DroneGateScorer<'_> {
    fn new<'a>(
        model: &'a WorldModel,
        emb: Tensor,
        current: DroneFrame,
        gate: GateSpec,
        action_stats: RunningStats,
        target_stats: RunningStats,
        action_normalized: bool,
        target_normalized: bool,
        device: candle::Device,
        dtype: DType,
        history_steps: usize,
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
        Ok(DroneGateScorer {
            model,
            emb,
            current_pos: Tensor::from_vec(current.pos_world.to_vec(), (1, 3), &device)?
                .to_dtype(dtype)?,
            current_rot: Tensor::from_vec(
                current.rotmat_world_from_body.to_vec(),
                (1, 9),
                &device,
            )?
            .to_dtype(dtype)?,
            gate_center: Tensor::from_vec(gate.center.to_vec(), (1, 3), &device)?
                .to_dtype(dtype)?,
            gate_normal: Tensor::from_vec(gate.normal.to_vec(), (1, 3), &device)?
                .to_dtype(dtype)?,
            gate_right: Tensor::from_vec(gate.right.to_vec(), (1, 3), &device)?.to_dtype(dtype)?,
            gate_up: Tensor::from_vec(gate.up.to_vec(), (1, 3), &device)?.to_dtype(dtype)?,
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
        let start = self.history_steps.min(rollout_time.saturating_sub(1));
        for step in start..rollout_time {
            let delta = deltas.i((.., step, ..))?;
            let delta_pos_body = delta.i((.., 0..3))?;
            let delta_rot_body = delta.i((.., 3..6))?;
            let delta_pos_world = batched_mat3_mul_vec3(&rot, &delta_pos_body)?;
            pos = (pos + delta_pos_world)?;
            let delta_rot = batched_rotvec_to_mat3(&delta_rot_body)?;
            rot = batched_mat3_mul(&rot, &delta_rot)?;

            let rel = pos.broadcast_sub(&self.gate_center)?;
            let plane = dot_last(&rel, &self.gate_normal)?.abs()?;
            let lateral =
                (dot_last(&rel, &self.gate_right)?.abs()? - self.gate.half_width as f64)?.relu()?;
            let vertical =
                (dot_last(&rel, &self.gate_up)?.abs()? - self.gate.half_height as f64)?.relu()?;
            let progress = rel.sqr()?.sum(D::Minus1)?.sqrt()?;
            let cost = (((plane + (lateral.sqr()? * 25.0)?)? + (vertical.sqr()? * 25.0)?)?
                + (progress * 0.05)?)?;
            best = best.broadcast_minimum(&cost)?;
        }
        Ok(best)
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
    let flight = gates
        .flights
        .iter()
        .find(|flight| flight.episode_idx == episode)
        .or_else(|| gates.flights.first())
        .context("gate file does not contain any flights")?;
    flight
        .gates
        .get(gate_index)
        .cloned()
        .with_context(|| format!("flight {} has no gate at index {gate_index}", flight.flight))
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
    best_sequence: Vec<[f32; 4]>,
    scores: Vec<f32>,
}
