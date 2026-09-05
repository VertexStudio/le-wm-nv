use std::{collections::BTreeSet, fs, path::PathBuf, time::Instant};

use anyhow::{Context, ensure};
use candle::{DType, Tensor};
use clap::{Parser, ValueEnum};
use le_wm_nv::{
    checkpoint::var_builder_from_path,
    data::skyjepa::{SkyJepaDroneDataset, SkyJepaSplitBy, skyjepa_artifact_fingerprint},
    models::skyjepa::{
        SkyJepaConfig, SkyJepaModel, SkyJepaProber,
        checkpoint::{SkyJepaCheckpoint, file_sha256},
        integrate_metric_rollout_inference, skyjepa_latent_rollout,
    },
    runtime::DeviceSpec,
};
use serde::Serialize;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum EvalSplit {
    Validation,
    Test,
    /// Evaluate every valid window of an external dataset, never training episodes.
    All,
}

#[derive(Debug, Parser)]
#[command(about = "Evaluate SkyJEPA latent and metric open-loop rollouts by horizon")]
struct Args {
    #[arg(long)]
    dataset_dir: Option<PathBuf>,

    #[arg(long)]
    checkpoint_dir: Option<PathBuf>,

    #[arg(long)]
    output: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = EvalSplit::Test)]
    split: EvalSplit,

    #[arg(long, default_value_t = 512)]
    batch_size: usize,

    /// Recursive evaluation horizon. Network weights are horizon-independent;
    /// the paper evaluates beyond the 20-step training rollout.
    #[arg(long, default_value_t = 60)]
    rollout_steps: usize,

    /// Zero evaluates the complete split.
    #[arg(long, default_value_t = 0)]
    max_batches: usize,

    #[arg(long, default_value_t = DeviceSpec::Cuda(0))]
    device: DeviceSpec,
}

#[derive(Debug, Serialize)]
struct HorizonMetrics {
    step: usize,
    time_seconds: f64,
    latent_rmse: f64,
    position_rmse_m: f64,
    position_vector_rmse_m: f64,
    velocity_rmse_mps: f64,
    attitude_rmse_deg: f64,
    angular_velocity_rmse_radps: f64,
}

#[derive(Debug, Serialize)]
struct BaselineHorizonMetrics {
    step: usize,
    time_seconds: f64,
    position_rmse_m: f64,
    position_vector_rmse_m: f64,
    velocity_rmse_mps: f64,
    attitude_rmse_deg: f64,
    angular_velocity_rmse_radps: f64,
}

#[derive(Debug, Serialize)]
struct EvaluationReport {
    report_version: u32,
    executable_sha256: String,
    configuration: serde_json::Value,
    domain_distribution: Option<String>,
    split: String,
    split_by: SkyJepaSplitBy,
    scope: String,
    dataset_artifact_sha256: String,
    checkpoint_sha256: String,
    available_windows: usize,
    complete_population: bool,
    evaluated_episode_ids: Vec<i64>,
    evaluated_domain_ids: Vec<i64>,
    evaluated_domain_fingerprints: Vec<String>,
    training_domain_overlap: usize,
    windows: usize,
    batches: usize,
    model_rate_hz: usize,
    elapsed_seconds: f64,
    normalization_source: String,
    skyjepa: Vec<HorizonMetrics>,
    constant_velocity_baseline: Vec<BaselineHorizonMetrics>,
    kinematic_baseline: Vec<BaselineHorizonMetrics>,
}

#[derive(Debug)]
struct Accumulator {
    count: usize,
    latent_sq: Vec<f64>,
    metric: MetricAccumulator,
}

#[derive(Debug)]
struct MetricAccumulator {
    count: usize,
    position_sq: Vec<f64>,
    velocity_sq: Vec<f64>,
    attitude_sq: Vec<f64>,
    omega_sq: Vec<f64>,
}

impl Accumulator {
    fn new(horizon: usize) -> Self {
        Self {
            count: 0,
            latent_sq: vec![0.0; horizon],
            metric: MetricAccumulator::new(horizon),
        }
    }

    fn push(
        &mut self,
        predicted_latents: &[Vec<Vec<f32>>],
        target_latents: &[Vec<Vec<f32>>],
        predicted_states: &[Vec<Vec<f32>>],
        target_states: &[Vec<Vec<f32>>],
    ) {
        self.count += predicted_states.len();
        for batch in 0..predicted_states.len() {
            for step in 0..predicted_states[batch].len() {
                self.latent_sq[step] += squared_error(
                    &predicted_latents[batch][step],
                    &target_latents[batch][step],
                );
            }
        }
        self.metric.push(predicted_states, target_states);
    }

    fn finish(self, cfg: &SkyJepaConfig, model_rate_hz: usize) -> Vec<HorizonMetrics> {
        let metric = self.metric.finish(model_rate_hz);
        metric
            .into_iter()
            .enumerate()
            .map(|(step, metric)| HorizonMetrics {
                step: metric.step,
                time_seconds: metric.time_seconds,
                latent_rmse: (self.latent_sq[step] / (self.count * cfg.latent_dim) as f64).sqrt(),
                position_rmse_m: metric.position_rmse_m,
                position_vector_rmse_m: metric.position_vector_rmse_m,
                velocity_rmse_mps: metric.velocity_rmse_mps,
                attitude_rmse_deg: metric.attitude_rmse_deg,
                angular_velocity_rmse_radps: metric.angular_velocity_rmse_radps,
            })
            .collect()
    }
}

impl MetricAccumulator {
    fn new(horizon: usize) -> Self {
        Self {
            count: 0,
            position_sq: vec![0.0; horizon],
            velocity_sq: vec![0.0; horizon],
            attitude_sq: vec![0.0; horizon],
            omega_sq: vec![0.0; horizon],
        }
    }

    fn push(&mut self, predicted_states: &[Vec<Vec<f32>>], target_states: &[Vec<Vec<f32>>]) {
        self.count += predicted_states.len();
        for batch in 0..predicted_states.len() {
            for step in 0..predicted_states[batch].len() {
                self.position_sq[step] += squared_error(
                    &predicted_states[batch][step][0..3],
                    &target_states[batch][step][0..3],
                );
                self.velocity_sq[step] += squared_error(
                    &predicted_states[batch][step][3..6],
                    &target_states[batch][step][3..6],
                );
                self.omega_sq[step] += squared_error(
                    &predicted_states[batch][step][15..18],
                    &target_states[batch][step][15..18],
                );
                let angle = rotation_distance(
                    &predicted_states[batch][step][6..15],
                    &target_states[batch][step][6..15],
                );
                self.attitude_sq[step] += angle * angle;
            }
        }
    }

    fn finish(self, model_rate_hz: usize) -> Vec<BaselineHorizonMetrics> {
        (0..self.position_sq.len())
            .map(|step| BaselineHorizonMetrics {
                step: step + 1,
                time_seconds: (step + 1) as f64 / model_rate_hz as f64,
                position_rmse_m: (self.position_sq[step] / (self.count * 3) as f64).sqrt(),
                position_vector_rmse_m: (self.position_sq[step] / self.count as f64).sqrt(),
                velocity_rmse_mps: (self.velocity_sq[step] / (self.count * 3) as f64).sqrt(),
                attitude_rmse_deg: (self.attitude_sq[step] / self.count as f64).sqrt() * 180.0
                    / std::f64::consts::PI,
                angular_velocity_rmse_radps: (self.omega_sq[step] / (self.count * 3) as f64).sqrt(),
            })
            .collect()
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    ensure!(args.batch_size > 0, "batch_size must be positive");
    ensure!(args.rollout_steps > 0, "rollout_steps must be positive");
    let started = Instant::now();
    let dataset_dir = args.dataset_dir.unwrap_or_else(default_dataset_dir);
    let checkpoint_dir = args.checkpoint_dir.unwrap_or_else(default_checkpoint_dir);
    let checkpoint = SkyJepaCheckpoint::load(&checkpoint_dir)?;
    let mut model_cfg = checkpoint.contract.model.clone();
    let prober_cfg = checkpoint
        .contract
        .prober
        .clone()
        .context("evaluation requires a trained prober")?;
    let mut dataset_cfg = checkpoint.contract.dataset;
    ensure!(
        dataset_cfg.history_steps == model_cfg.history_steps
            && dataset_cfg.rollout_steps == model_cfg.rollout_steps,
        "checkpoint dataset/model sequence dimensions disagree"
    );
    model_cfg.rollout_steps = args.rollout_steps;
    dataset_cfg.batch_size = args.batch_size;
    dataset_cfg.rollout_steps = args.rollout_steps;
    let normalization = checkpoint.contract.normalization.clone();
    let dataset = SkyJepaDroneDataset::open_with_normalization(
        &dataset_dir,
        dataset_cfg,
        Some(normalization),
    )?;
    let mut rows = match args.split {
        EvalSplit::Validation => dataset.validation_rows(),
        EvalSplit::Test => dataset.test_rows(),
        EvalSplit::All => dataset.all_rows(),
    };
    ensure!(!rows.is_empty(), "selected SkyJEPA split has no windows");
    let available_windows = rows.len();
    if args.max_batches > 0 {
        rows.truncate(args.max_batches.saturating_mul(args.batch_size));
    }
    let evaluated_episode_ids = dataset.episode_ids_for_rows(&rows);
    let evaluated_domain_ids = dataset.domain_ids_for_rows(&rows);
    let evaluated_domain_fingerprints = dataset.domain_fingerprints_for_rows(&rows)?;
    let dataset_artifact_sha256 = skyjepa_artifact_fingerprint(&dataset_dir)?;
    let mut training_domains = BTreeSet::new();
    let mut same_artifact = false;
    for provenance in [&checkpoint.provenance, checkpoint.latent_provenance()] {
        if provenance["dataset_artifact_sha256"].as_str() == Some(&dataset_artifact_sha256) {
            same_artifact = true;
            let training = provenance["training_episodes"]
                .as_array()
                .context("checkpoint lacks training episode provenance")?;
            let training = training
                .iter()
                .filter_map(|id| id.as_i64())
                .collect::<BTreeSet<_>>();
            ensure!(
                evaluated_episode_ids
                    .iter()
                    .all(|id| !training.contains(id)),
                "evaluation population overlaps training episodes; choose a held-out split or an external dataset"
            );
        }
        if let Some(domains) = provenance["training_domain_fingerprints"].as_array() {
            training_domains.extend(
                domains
                    .iter()
                    .filter_map(|value| value.as_str())
                    .map(str::to_owned),
            );
        }
    }
    let training_domain_overlap = evaluated_domain_fingerprints
        .iter()
        .filter(|hash| training_domains.contains(*hash))
        .count();
    let scope = if evaluated_domain_fingerprints.is_empty() || training_domains.is_empty() {
        "unknown_domain_overlap"
    } else {
        match (same_artifact, training_domain_overlap == 0) {
            (true, true) => "held_out_domains",
            (true, false) => "held_out_episodes",
            (false, true) => "external_unseen_domains",
            (false, false) => "external_shared_domains",
        }
    }
    .to_owned();
    let device = args.device.resolve()?;
    let model = SkyJepaModel::new(
        model_cfg.clone(),
        var_builder_from_path(
            &checkpoint.latent_path(&checkpoint_dir),
            DType::F32,
            &device,
        )?,
    )?;
    let prober = SkyJepaProber::new(
        prober_cfg,
        var_builder_from_path(
            &checkpoint.prober_path(&checkpoint_dir)?,
            DType::F32,
            &device,
        )?,
    )?;

    let mut accumulator = Accumulator::new(model_cfg.rollout_steps);
    let mut constant_velocity = MetricAccumulator::new(model_cfg.rollout_steps);
    let mut kinematic = MetricAccumulator::new(model_cfg.rollout_steps);
    let mut batches = 0usize;
    for rows in rows.chunks(args.batch_size) {
        if args.max_batches > 0 && batches >= args.max_batches {
            break;
        }
        let batch = dataset.batch(rows, DType::F32, &device)?;
        let rollout = skyjepa_latent_rollout(&model, &batch.states, &batch.actions)?;
        let initial_state = batch
            .metric_states
            .narrow(1, model_cfg.history_steps - 1, 1)?
            .squeeze(1)?;
        let future_actions =
            batch
                .metric_actions
                .narrow(1, model_cfg.history_steps - 1, model_cfg.rollout_steps)?;
        let transition_dt =
            batch
                .transition_dt
                .narrow(1, model_cfg.history_steps - 1, model_cfg.rollout_steps)?;
        let target_states =
            batch
                .metric_states
                .narrow(1, model_cfg.history_steps, model_cfg.rollout_steps)?;
        let predicted_states = prober.predict_metric_rollout(
            &initial_state,
            &future_actions,
            &transition_dt,
            &rollout.predicted_latents,
        )?;
        let constant_velocity_states = constant_velocity_rollout(&initial_state, &transition_dt)?;
        let batch_size = initial_state.dim(0)?;
        let residual = Tensor::zeros(
            (batch_size, model_cfg.rollout_steps, 3),
            DType::F32,
            &device,
        )?;
        let angular_map = Tensor::zeros(
            (batch_size, model_cfg.rollout_steps, 3, 4),
            DType::F32,
            &device,
        )?;
        let kinematic_states = integrate_metric_rollout_inference(
            &initial_state,
            &future_actions,
            &transition_dt,
            &residual,
            &angular_map,
            prober.config().kinematics,
        )?;
        let predicted_latents = rollout.predicted_latents.to_vec3::<f32>()?;
        let target_latents = rollout.target_latents.to_vec3::<f32>()?;
        let predicted_states = predicted_states.to_vec3::<f32>()?;
        let target_states = target_states.to_vec3::<f32>()?;
        let constant_velocity_states = constant_velocity_states.to_vec3::<f32>()?;
        let kinematic_states = kinematic_states.to_vec3::<f32>()?;
        accumulator.push(
            &predicted_latents,
            &target_latents,
            &predicted_states,
            &target_states,
        );
        constant_velocity.push(&constant_velocity_states, &target_states);
        kinematic.push(&kinematic_states, &target_states);
        batches += 1;
    }
    ensure!(accumulator.count > 0, "evaluation processed zero windows");
    let report = EvaluationReport {
        report_version: 2,
        executable_sha256: file_sha256(&std::env::current_exe()?)?,
        configuration: serde_json::json!({"argv":std::env::args().collect::<Vec<_>>(),
            "dataset":dataset_cfg,"model":model_cfg,"prober":prober.config()}),
        domain_distribution: serde_json::from_str::<serde_json::Value>(&fs::read_to_string(
            dataset_dir.join("metadata.json"),
        )?)?["domain_distribution"]
            .as_str()
            .map(str::to_owned),
        split: match args.split {
            EvalSplit::Validation => "validation",
            EvalSplit::Test => "test",
            EvalSplit::All => "all",
        }
        .to_string(),
        split_by: dataset_cfg.split_by,
        scope,
        dataset_artifact_sha256,
        checkpoint_sha256: checkpoint.fingerprint()?,
        available_windows,
        complete_population: rows.len() == available_windows,
        evaluated_episode_ids,
        evaluated_domain_ids,
        evaluated_domain_fingerprints,
        training_domain_overlap,
        windows: accumulator.count,
        batches,
        model_rate_hz: dataset_cfg.model_rate_hz,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        normalization_source: checkpoint_dir.join("checkpoint.json").display().to_string(),
        skyjepa: accumulator.finish(&model_cfg, dataset_cfg.model_rate_hz),
        constant_velocity_baseline: constant_velocity.finish(dataset_cfg.model_rate_hz),
        kinematic_baseline: kinematic.finish(dataset_cfg.model_rate_hz),
    };
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = args.output {
        fs::write(&path, &json).with_context(|| format!("failed to write {}", path.display()))?;
    }
    println!("{json}");
    Ok(())
}

fn constant_velocity_rollout(
    initial_state: &Tensor,
    transition_dt: &Tensor,
) -> candle::Result<Tensor> {
    let (batch, state_dim) = initial_state.dims2()?;
    let (dt_batch, steps) = transition_dt.dims2()?;
    if state_dim != 18 || dt_batch != batch {
        candle::bail!(
            "constant-velocity baseline shape mismatch: state={:?} dt={:?}",
            initial_state.shape(),
            transition_dt.shape()
        );
    }
    let initial = initial_state.to_vec2::<f32>()?;
    let dt = transition_dt.to_vec2::<f32>()?;
    let mut values = Vec::with_capacity(batch * steps * state_dim);
    for row in 0..batch {
        let mut state: [f32; 18] = initial[row]
            .as_slice()
            .try_into()
            .expect("validated state dimension");
        for step in 0..steps {
            for axis in 0..3 {
                state[axis] += state[3 + axis] * dt[row][step];
            }
            values.extend_from_slice(&state);
        }
    }
    Tensor::from_vec(values, (batch, steps, state_dim), initial_state.device())?
        .to_dtype(initial_state.dtype())
}

fn squared_error(lhs: &[f32], rhs: &[f32]) -> f64 {
    lhs.iter()
        .zip(rhs)
        .map(|(lhs, rhs)| {
            let error = f64::from(*lhs) - f64::from(*rhs);
            error * error
        })
        .sum()
}

fn rotation_distance(lhs: &[f32], rhs: &[f32]) -> f64 {
    let frobenius_inner = lhs
        .iter()
        .zip(rhs)
        .map(|(lhs, rhs)| f64::from(*lhs) * f64::from(*rhs))
        .sum::<f64>();
    ((frobenius_inner - 1.0) * 0.5).clamp(-1.0, 1.0).acos()
}

fn default_dataset_dir() -> PathBuf {
    user_home()
        .join(".stable_worldmodel")
        .join("le-wm-nv-data")
        .join("skyjepa-domain-randomized-20hz")
}

fn default_checkpoint_dir() -> PathBuf {
    user_home()
        .join(".stable_worldmodel")
        .join("le-wm-nv-runs")
        .join("skyjepa-drone-state18-20hz")
}

fn user_home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}
