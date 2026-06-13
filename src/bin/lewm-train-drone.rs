use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, ensure};
use candle::DType;
use candle_nn::{ParamsAdamW, VarBuilder, VarMap};
use clap::Parser;
use le_wm_nv::{
    data::drone_racing::{
        DRONE_ACTION_DIM, DRONE_OBSERVATION_DIM, DRONE_STATE_DELTA_DIM, DroneBatchConfig,
        DroneRacingDataset, DroneRacingMetadata, epoch_seed,
    },
    models::world_model::{
        VectorLossScalars, VectorLossWeights, WorldModel, WorldModelConfig, vector_batch_loss,
    },
    optim::StatefulAdamW,
    runtime::{DTypeSpec, DeviceSpec},
};
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
struct Args {
    /// Imported drone dataset directory containing data.h5 and metadata.json.
    #[arg(long)]
    dataset_dir: Option<PathBuf>,

    /// Output directory for checkpoints, metadata, and metrics.
    #[arg(long)]
    output_dir: Option<PathBuf>,

    /// Repo-native WorldModel config JSON. If omitted, use the default vector drone config.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Optional weights-only initialization checkpoint.
    #[arg(long)]
    init_safetensors: Option<PathBuf>,

    /// Resume weights, AdamW state, and sampler position from a previous output directory.
    #[arg(long)]
    resume_dir: Option<PathBuf>,

    #[arg(long, default_value_t = DeviceSpec::Cuda(0))]
    device: DeviceSpec,

    #[arg(long, default_value_t = DTypeSpec::F32)]
    dtype: DTypeSpec,

    #[arg(long, default_value_t = 100)]
    epochs: usize,

    #[arg(long)]
    max_steps: Option<usize>,

    /// Train on every valid sliding window in the imported dataset.
    #[arg(long)]
    train_all_data: bool,

    #[arg(long, default_value_t = 256)]
    batch_size: usize,

    #[arg(long, default_value_t = 8)]
    history_steps: usize,

    #[arg(long, default_value_t = 50)]
    horizon_steps: usize,

    #[arg(long, default_value_t = 7)]
    seed: u64,

    #[arg(long)]
    no_observation_normalize: bool,

    #[arg(long)]
    no_action_normalize: bool,

    #[arg(long)]
    no_target_normalize: bool,

    #[arg(long, default_value_t = 1e-4)]
    lr: f64,

    #[arg(long, default_value_t = 0.01)]
    weight_decay: f64,

    #[arg(long, default_value_t = 10)]
    log_every: usize,

    #[arg(long, default_value_t = 1000)]
    save_every: usize,

    #[arg(long, default_value_t = 1.0)]
    state_prediction_weight: f64,

    #[arg(long, default_value_t = 0.1)]
    temporal_alignment_weight: f64,

    #[arg(long, default_value_t = 0.1)]
    std_weight: f64,

    #[arg(long, default_value_t = 0.1)]
    std_t_weight: f64,

    #[arg(long, default_value_t = 0.1)]
    covariance_weight: f64,

    #[arg(long, default_value_t = 0.1)]
    covariance_t_weight: f64,

    #[arg(long, default_value_t = 0.1)]
    temporal_straightening_weight: f64,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let started_at_unix = unix_seconds();
    let started = Instant::now();
    let dataset_dir = args.dataset_dir.clone().unwrap_or_else(default_dataset_dir);
    let output_dir = args.output_dir.clone().unwrap_or_else(default_output_dir);
    let sequence_steps = args
        .history_steps
        .checked_add(args.horizon_steps)
        .and_then(|v| v.checked_add(1))
        .context("history_steps + horizon_steps + 1 overflowed")?;
    let batch_cfg = DroneBatchConfig {
        batch_size: args.batch_size,
        sequence_steps,
        normalize_observations: !args.no_observation_normalize,
        normalize_actions: !args.no_action_normalize,
        normalize_targets: !args.no_target_normalize,
    };
    let dataset = DroneRacingDataset::open(&dataset_dir, batch_cfg)?;
    let train_rows = training_rows(&dataset, args.train_all_data);
    let row_source = if args.train_all_data {
        "all_valid_rows"
    } else {
        "metadata_train_episodes"
    };
    let batches_per_epoch = train_rows.len() / args.batch_size;
    ensure!(
        batches_per_epoch > 0,
        "train rows {} are fewer than batch_size {}",
        train_rows.len(),
        args.batch_size
    );
    let run = RunSettings::from_args(&args, &dataset_dir, &output_dir, sequence_steps);
    let resume_state = match args.resume_dir.as_ref() {
        Some(dir) => Some(load_resume_state(dir)?),
        None => None,
    };
    if let Some(state) = resume_state.as_ref() {
        ensure_resume_compatible(&run, &state.run)?;
    }
    let device = args.device.resolve()?;
    let dtype = args.dtype.dtype();
    if dtype != DType::F32 {
        anyhow::bail!("drone training currently requires --dtype f32");
    }

    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    let cfg = load_or_default_config(args.config.as_ref(), args.history_steps, sequence_steps)?;
    cfg.validate()?;
    write_pretty_json(&output_dir.join("dataset-summary.json"), dataset.metadata())?;
    write_pretty_json(&output_dir.join("model-config.json"), &cfg)?;
    write_pretty_json(
        &output_dir.join("normalization.json"),
        &dataset.metadata().normalization,
    )?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
    let model = WorldModel::new(cfg.clone(), vb)?;
    if let Some(dir) = args.resume_dir.as_ref() {
        let path = dir.join("latest.safetensors");
        varmap
            .load(&path)
            .with_context(|| format!("failed to load {}", path.display()))?;
    } else if let Some(path) = args.init_safetensors.as_ref() {
        varmap
            .load(path)
            .with_context(|| format!("failed to load {}", path.display()))?;
    }
    let params = ParamsAdamW {
        lr: args.lr,
        weight_decay: args.weight_decay,
        ..ParamsAdamW::default()
    };
    let mut optimizer = StatefulAdamW::new_from_varmap(&varmap, params)?;
    let start_step = if let Some(state) = resume_state.as_ref() {
        let dir = args
            .resume_dir
            .as_ref()
            .expect("resume state requires resume dir");
        let path = dir.join("optimizer.safetensors");
        optimizer
            .load_state(&path, state.global_step)
            .with_context(|| format!("failed to load {}", path.display()))?;
        state.global_step
    } else {
        0
    };
    let loss_weights = loss_weights(&args);
    let metrics_path = output_dir.join("metrics.jsonl");
    let metrics_file = File::create(&metrics_path)
        .with_context(|| format!("failed to create {}", metrics_path.display()))?;
    let mut metrics = BufWriter::new(metrics_file);

    println!(
        "dataset={} rows={} train_windows={} row_source={} batches_per_epoch={} sequence_steps={}",
        dataset.root().display(),
        dataset.metadata().rows,
        train_rows.len(),
        row_source,
        batches_per_epoch,
        sequence_steps,
    );
    println!(
        "training output_dir={} epochs={} batch_size={} lr={:.3e} weight_decay={:.3e}",
        output_dir.display(),
        args.epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
    );

    let epoch_steps = args
        .epochs
        .checked_mul(batches_per_epoch)
        .context("--epochs * batches_per_epoch overflowed")?;
    let target_steps = args
        .max_steps
        .map(|max_steps| max_steps.min(epoch_steps))
        .unwrap_or(epoch_steps);
    ensure!(
        start_step < target_steps,
        "resume start_step {start_step} is already at or beyond target_steps {target_steps}"
    );

    let mut global_step = start_step;
    let mut last_loss = None;
    let mut last_epoch = start_step / batches_per_epoch;
    let mut last_batch_index = start_step % batches_per_epoch;
    let mut cached_epoch = None;
    let mut cached_rows = Vec::new();
    for step_index in start_step..target_steps {
        let epoch = step_index / batches_per_epoch;
        let batch_index = step_index % batches_per_epoch;
        if cached_epoch != Some(epoch) {
            cached_rows = dataset.shuffled_rows(&train_rows, epoch_seed(args.seed, epoch));
            cached_epoch = Some(epoch);
        }
        let row_start = batch_index * args.batch_size;
        let row_end = row_start + args.batch_size;
        let batch = dataset.batch(&cached_rows[row_start..row_end], dtype, &device)?;
        let loss = vector_batch_loss(
            &model,
            &batch.observations,
            &batch.actions,
            &batch.target_deltas,
            loss_weights,
        )?;
        let scalars = VectorLossScalars::from_loss(&loss)?;
        ensure_finite_loss(step_index + 1, scalars.total)?;
        optimizer.backward_step(&loss.total_loss)?;
        global_step = step_index + 1;
        last_epoch = epoch;
        last_batch_index = batch_index;
        last_loss = Some(scalars.clone());

        if global_step == 1 || global_step % args.log_every == 0 {
            print_loss(global_step, epoch, batch_index, &scalars);
            write_json_line(
                &mut metrics,
                &MetricsRow {
                    kind: "train",
                    step: global_step,
                    epoch,
                    batch_index,
                    rows: batch.meta.rows,
                    episode_idx: batch.meta.episode_idx,
                    step_idx: batch.meta.step_idx,
                    lr: args.lr,
                    weight_decay: args.weight_decay,
                    elapsed_sec: started.elapsed().as_secs_f64(),
                    loss: scalars.clone(),
                },
            )?;
        }

        if args.save_every > 0 && global_step % args.save_every == 0 {
            save_checkpoint(
                &output_dir,
                &varmap,
                &optimizer,
                started_at_unix,
                &run,
                dataset.metadata(),
                &cfg,
                global_step,
                epoch,
                batch_index,
                batches_per_epoch,
                last_loss.as_ref(),
            )?;
        }
    }

    ensure!(global_step > 0, "no optimizer steps were run");
    let final_checkpoint = output_dir.join("final.safetensors");
    let final_optimizer = output_dir.join("final-optimizer.safetensors");
    save_weights(&varmap, &final_checkpoint)?;
    save_optimizer(&optimizer, &final_optimizer)?;
    save_weights(&varmap, &output_dir.join("latest.safetensors"))?;
    save_optimizer(&optimizer, &output_dir.join("optimizer.safetensors"))?;
    let state = TrainingState::new(
        started_at_unix,
        &run,
        dataset.metadata(),
        &cfg,
        global_step,
        last_epoch,
        last_batch_index,
        batches_per_epoch,
        Some(&final_checkpoint),
        Some(&final_optimizer),
        last_loss.as_ref(),
    );
    write_pretty_json(&output_dir.join("training-state.json"), &state)?;
    metrics.flush()?;
    println!(
        "saved={} latest={} steps={} elapsed_sec={:.3}",
        final_checkpoint.display(),
        output_dir.join("latest.safetensors").display(),
        global_step,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    ensure!(args.epochs > 0, "--epochs must be greater than zero");
    ensure!(
        args.batch_size > 0,
        "--batch-size must be greater than zero"
    );
    ensure!(
        args.history_steps > 0,
        "--history-steps must be greater than zero"
    );
    ensure!(
        args.horizon_steps > 0,
        "--horizon-steps must be greater than zero"
    );
    ensure!(args.log_every > 0, "--log-every must be greater than zero");
    if let Some(max_steps) = args.max_steps {
        ensure!(max_steps > 0, "--max-steps must be greater than zero");
    }
    ensure!(
        !(args.resume_dir.is_some() && args.init_safetensors.is_some()),
        "--resume-dir and --init-safetensors are mutually exclusive"
    );
    ensure!(
        args.lr.is_finite() && args.lr > 0.0,
        "--lr must be finite and greater than zero"
    );
    ensure!(
        args.weight_decay.is_finite() && args.weight_decay >= 0.0,
        "--weight-decay must be finite and non-negative"
    );
    Ok(())
}

fn default_dataset_dir() -> PathBuf {
    home_dir()
        .join(".stable_worldmodel")
        .join("le-wm-nv-data")
        .join("drone-racing-autonomous-100hz")
}

fn default_output_dir() -> PathBuf {
    home_dir()
        .join(".stable_worldmodel")
        .join("le-wm-nv-runs")
        .join("drone-state-lewm-autonomous-100hz")
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn load_or_default_config(
    path: Option<&PathBuf>,
    history_steps: usize,
    sequence_steps: usize,
) -> anyhow::Result<WorldModelConfig> {
    let mut cfg = match path {
        Some(path) => serde_json::from_str(
            &fs::read_to_string(path)
                .with_context(|| format!("failed to read {}", path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", path.display()))?,
        None => WorldModelConfig::vector_drone_default(
            DRONE_OBSERVATION_DIM,
            DRONE_ACTION_DIM,
            DRONE_STATE_DELTA_DIM,
        ),
    };
    cfg.history_size = history_steps;
    cfg.predictor.num_frames = sequence_steps;
    Ok(cfg)
}

fn training_rows(dataset: &DroneRacingDataset, train_all_data: bool) -> Vec<usize> {
    if train_all_data {
        dataset.valid_rows().to_vec()
    } else {
        dataset.train_rows()
    }
}

fn loss_weights(args: &Args) -> VectorLossWeights {
    VectorLossWeights {
        state_prediction: args.state_prediction_weight,
        temporal_alignment: args.temporal_alignment_weight,
        std: args.std_weight,
        std_t: args.std_t_weight,
        covariance: args.covariance_weight,
        covariance_t: args.covariance_t_weight,
        temporal_straightening: args.temporal_straightening_weight,
    }
}

fn ensure_finite_loss(step: usize, value: f32) -> anyhow::Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        anyhow::bail!("loss at step {step} is not finite: {value}")
    }
}

fn print_loss(step: usize, epoch: usize, batch_index: usize, loss: &VectorLossScalars) {
    println!(
        "step={} epoch={} batch={} total={:.8e} state_prediction={:.8e} temp_align={:.8e} std={:.8e} std_t={:.8e} cov={:.8e} cov_t={:.8e} temporal_straightening={:.8e}",
        step,
        epoch,
        batch_index,
        loss.total,
        loss.state_prediction,
        loss.temporal_alignment,
        loss.std,
        loss.std_t,
        loss.covariance,
        loss.covariance_t,
        loss.temporal_straightening,
    );
}

fn save_checkpoint(
    output_dir: &Path,
    varmap: &VarMap,
    optimizer: &StatefulAdamW,
    started_at_unix: u64,
    run: &RunSettings,
    dataset: &DroneRacingMetadata,
    cfg: &WorldModelConfig,
    global_step: usize,
    epoch: usize,
    batch_index: usize,
    batches_per_epoch: usize,
    last_loss: Option<&VectorLossScalars>,
) -> anyhow::Result<()> {
    let checkpoint = output_dir.join(format!("checkpoint-step-{global_step:08}.safetensors"));
    let optimizer_checkpoint =
        output_dir.join(format!("optimizer-step-{global_step:08}.safetensors"));
    save_weights(varmap, &checkpoint)?;
    save_optimizer(optimizer, &optimizer_checkpoint)?;
    save_weights(varmap, &output_dir.join("latest.safetensors"))?;
    save_optimizer(optimizer, &output_dir.join("optimizer.safetensors"))?;
    let state = TrainingState::new(
        started_at_unix,
        run,
        dataset,
        cfg,
        global_step,
        epoch,
        batch_index,
        batches_per_epoch,
        Some(&checkpoint),
        Some(&optimizer_checkpoint),
        last_loss,
    );
    write_pretty_json(&output_dir.join("training-state.json"), &state)
}

fn save_weights(varmap: &VarMap, path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    varmap
        .save(path)
        .with_context(|| format!("failed to save {}", path.display()))
}

fn save_optimizer(optimizer: &StatefulAdamW, path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    optimizer
        .save_state(path)
        .with_context(|| format!("failed to save {}", path.display()))
}

fn write_pretty_json<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(value)?;
    fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))
}

fn write_json_line<W: Write, T: Serialize>(writer: &mut W, value: &T) -> anyhow::Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn load_resume_state(dir: &Path) -> anyhow::Result<SavedTrainingState> {
    let path = dir.join("training-state.json");
    let json =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&json).with_context(|| format!("failed to parse {}", path.display()))
}

fn ensure_resume_compatible(current: &RunSettings, saved: &RunSettings) -> anyhow::Result<()> {
    ensure_eq("dataset_dir", &current.dataset_dir, &saved.dataset_dir)?;
    ensure_eq("config", &current.config, &saved.config)?;
    ensure_eq("device", &current.device, &saved.device)?;
    ensure_eq("dtype", &current.dtype, &saved.dtype)?;
    ensure_eq("batch_size", &current.batch_size, &saved.batch_size)?;
    ensure_eq(
        "train_all_data",
        &current.train_all_data,
        &saved.train_all_data,
    )?;
    ensure_eq(
        "history_steps",
        &current.history_steps,
        &saved.history_steps,
    )?;
    ensure_eq(
        "horizon_steps",
        &current.horizon_steps,
        &saved.horizon_steps,
    )?;
    ensure_eq(
        "sequence_steps",
        &current.sequence_steps,
        &saved.sequence_steps,
    )?;
    ensure_eq("seed", &current.seed, &saved.seed)?;
    ensure_eq(
        "normalization",
        &current.normalization,
        &saved.normalization,
    )?;
    ensure_eq("lr", &current.lr, &saved.lr)?;
    ensure_eq("weight_decay", &current.weight_decay, &saved.weight_decay)?;
    Ok(())
}

fn ensure_eq<T>(name: &str, current: &T, saved: &T) -> anyhow::Result<()>
where
    T: std::fmt::Debug + PartialEq,
{
    ensure!(
        current == saved,
        "resume setting `{name}` mismatch: current={current:?} saved={saved:?}"
    );
    Ok(())
}

#[derive(Debug, Deserialize)]
struct SavedTrainingState {
    global_step: usize,
    run: RunSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct RunSettings {
    dataset_dir: PathBuf,
    output_dir: PathBuf,
    config: Option<PathBuf>,
    init_safetensors: Option<PathBuf>,
    device: String,
    dtype: String,
    epochs: usize,
    max_steps: Option<usize>,
    #[serde(default)]
    train_all_data: bool,
    batch_size: usize,
    history_steps: usize,
    horizon_steps: usize,
    sequence_steps: usize,
    seed: u64,
    normalization: NormalizationFlags,
    lr: f64,
    weight_decay: f64,
    log_every: usize,
    save_every: usize,
    loss_weights: SerializableLossWeights,
}

impl RunSettings {
    fn from_args(
        args: &Args,
        dataset_dir: &Path,
        output_dir: &Path,
        sequence_steps: usize,
    ) -> Self {
        Self {
            dataset_dir: dataset_dir.to_path_buf(),
            output_dir: output_dir.to_path_buf(),
            config: args.config.clone(),
            init_safetensors: args.init_safetensors.clone(),
            device: args.device.to_string(),
            dtype: args.dtype.to_string(),
            epochs: args.epochs,
            max_steps: args.max_steps,
            train_all_data: args.train_all_data,
            batch_size: args.batch_size,
            history_steps: args.history_steps,
            horizon_steps: args.horizon_steps,
            sequence_steps,
            seed: args.seed,
            normalization: NormalizationFlags {
                observations: !args.no_observation_normalize,
                actions: !args.no_action_normalize,
                targets: !args.no_target_normalize,
            },
            lr: args.lr,
            weight_decay: args.weight_decay,
            log_every: args.log_every,
            save_every: args.save_every,
            loss_weights: SerializableLossWeights {
                state_prediction: args.state_prediction_weight,
                temporal_alignment: args.temporal_alignment_weight,
                std: args.std_weight,
                std_t: args.std_t_weight,
                covariance: args.covariance_weight,
                covariance_t: args.covariance_t_weight,
                temporal_straightening: args.temporal_straightening_weight,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct NormalizationFlags {
    observations: bool,
    actions: bool,
    targets: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct SerializableLossWeights {
    state_prediction: f64,
    temporal_alignment: f64,
    std: f64,
    std_t: f64,
    covariance: f64,
    covariance_t: f64,
    temporal_straightening: f64,
}

#[derive(Debug, Serialize)]
struct MetricsRow {
    kind: &'static str,
    step: usize,
    epoch: usize,
    batch_index: usize,
    rows: Vec<usize>,
    episode_idx: Vec<i64>,
    step_idx: Vec<i64>,
    lr: f64,
    weight_decay: f64,
    elapsed_sec: f64,
    loss: VectorLossScalars,
}

#[derive(Debug, Serialize)]
struct TrainingState<'a> {
    started_at_unix: u64,
    global_step: usize,
    epoch: usize,
    batch_index: usize,
    batches_per_epoch: usize,
    latest_checkpoint: Option<PathBuf>,
    latest_optimizer_checkpoint: Option<PathBuf>,
    last_loss: Option<&'a VectorLossScalars>,
    run: &'a RunSettings,
    dataset: &'a DroneRacingMetadata,
    model_config: &'a WorldModelConfig,
    optimizer_state: &'static str,
}

impl<'a> TrainingState<'a> {
    fn new(
        started_at_unix: u64,
        run: &'a RunSettings,
        dataset: &'a DroneRacingMetadata,
        model_config: &'a WorldModelConfig,
        global_step: usize,
        epoch: usize,
        batch_index: usize,
        batches_per_epoch: usize,
        latest_checkpoint: Option<&Path>,
        latest_optimizer_checkpoint: Option<&Path>,
        last_loss: Option<&'a VectorLossScalars>,
    ) -> Self {
        Self {
            started_at_unix,
            global_step,
            epoch,
            batch_index,
            batches_per_epoch,
            latest_checkpoint: latest_checkpoint.map(Path::to_path_buf),
            latest_optimizer_checkpoint: latest_optimizer_checkpoint.map(Path::to_path_buf),
            last_loss,
            run,
            dataset,
            model_config,
            optimizer_state: "serialized; resume with --resume-dir",
        }
    }
}
