use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, ensure};
use candle::{DType, Tensor};
use candle_nn::{ParamsAdamW, VarBuilder, VarMap};
use clap::Parser;
use hdf5::filters::blosc_set_nthreads;
use le_wm_nv::{
    data::pusht::{PushTBatchConfig, PushTDataset, PushTDatasetSummary},
    models::lewm::{LeWm, LeWmBatchLoss, LeWmConfig, LeWmLossWeights, batch_loss},
    optim::StatefulAdamW,
    runtime::{DTypeSpec, DeviceSpec},
};
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
struct Args {
    /// PushT HDF5 dataset, usually ~/.stable_worldmodel/pusht_expert_train.h5.
    #[arg(long)]
    dataset_h5: Option<PathBuf>,

    /// Output directory for checkpoints, metadata, and metrics.
    #[arg(long)]
    output_dir: PathBuf,

    /// stable-worldmodel LeWM config JSON. If omitted, infer PushT action dim and use tiny Patch14.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Optional trainable initialization or weights-only resume checkpoint.
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

    /// Stop after this many optimizer steps, useful for smoke tests.
    #[arg(long)]
    max_steps: Option<usize>,

    #[arg(long, default_value_t = 64)]
    batch_size: usize,

    #[arg(long, default_value_t = 3)]
    history_size: usize,

    #[arg(long, default_value_t = 5)]
    action_block: usize,

    #[arg(long, default_value_t = 224)]
    image_size: usize,

    #[arg(long, default_value_t = 7)]
    seed: u64,

    #[arg(long)]
    no_action_normalize: bool,

    /// Number of Blosc threads used by native HDF5 filter decoding. Zero leaves the default.
    #[arg(long, default_value_t = 0)]
    blosc_threads: u8,

    #[arg(long, default_value_t = 1e-4)]
    lr: f64,

    #[arg(long, default_value_t = 0.01)]
    weight_decay: f64,

    #[arg(long, default_value_t = 10)]
    log_every: usize,

    /// Save checkpoint-step-*.safetensors every N optimizer steps. Zero disables periodic saves.
    #[arg(long, default_value_t = 1000)]
    save_every: usize,

    #[arg(long, default_value_t = 1.0)]
    prediction_weight: f64,

    #[arg(long, default_value_t = 1.0)]
    temporal_alignment_weight: f64,

    #[arg(long, default_value_t = 1.0)]
    std_weight: f64,

    #[arg(long, default_value_t = 1.0)]
    std_t_weight: f64,

    #[arg(long, default_value_t = 1.0)]
    covariance_weight: f64,

    #[arg(long, default_value_t = 1.0)]
    covariance_t_weight: f64,

    #[arg(long, default_value_t = 1.0)]
    temporal_straightening_weight: f64,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    validate_args(&args)?;

    if args.blosc_threads > 0 {
        let previous = blosc_set_nthreads(args.blosc_threads);
        println!(
            "blosc_threads={} previous_blosc_threads={previous}",
            args.blosc_threads
        );
    }

    let started_at_unix = unix_seconds();
    let started = Instant::now();
    let dataset_path = args
        .dataset_h5
        .clone()
        .unwrap_or_else(default_pusht_dataset_path);
    let batch_cfg = PushTBatchConfig {
        batch_size: args.batch_size,
        history_size: args.history_size,
        action_block: args.action_block,
        image_size: args.image_size,
        normalize_actions: !args.no_action_normalize,
    };
    let dataset = PushTDataset::open(&dataset_path, batch_cfg)?;
    let dataset_summary = dataset.summary();
    let batches_per_epoch = dataset.valid_rows().len() / args.batch_size;
    ensure!(
        batches_per_epoch > 0,
        "valid PushT rows {} are fewer than batch_size {}",
        dataset.valid_rows().len(),
        args.batch_size
    );
    let run = RunSettings::from_args(&args, &dataset_path);
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
        anyhow::bail!("LeWM training currently requires --dtype f32");
    }

    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("failed to create {}", args.output_dir.display()))?;

    let cfg = load_or_default_config(
        args.config.as_ref(),
        dataset.model_action_dim(),
        dataset.history_size(),
        dataset.image_size(),
    )?;
    validate_model_config(&cfg, &dataset)?;
    write_pretty_json(
        &args.output_dir.join("dataset-summary.json"),
        &dataset_summary,
    )?;
    write_pretty_json(&args.output_dir.join("model-config.json"), &cfg)?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
    let model = LeWm::new(cfg.clone(), vb)?;
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
    let metrics_path = args.output_dir.join("metrics.jsonl");
    let metrics_file = File::create(&metrics_path)
        .with_context(|| format!("failed to create {}", metrics_path.display()))?;
    let mut metrics = BufWriter::new(metrics_file);

    println!(
        "dataset={} rows={} valid_rows={} batches_per_epoch={} raw_action_dim={} model_action_dim={} image={}x{}",
        dataset_path.display(),
        dataset_summary.rows,
        dataset_summary.valid_rows,
        batches_per_epoch,
        dataset.raw_action_dim(),
        dataset.model_action_dim(),
        dataset_summary.pixel_height,
        dataset_summary.pixel_width,
    );
    println!(
        "training output_dir={} epochs={} batch_size={} lr={:.3e} weight_decay={:.3e}",
        args.output_dir.display(),
        args.epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
    );
    if let Some(dir) = args.resume_dir.as_ref() {
        println!(
            "resume_dir={} start_step={} optimizer_step={}",
            dir.display(),
            start_step,
            optimizer.step_t()
        );
    }

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
            cached_rows = dataset.shuffled_valid_rows(epoch_seed(args.seed, epoch));
            cached_epoch = Some(epoch);
        }
        let row_start = batch_index * args.batch_size;
        let row_end = row_start + args.batch_size;
        let data = dataset.batch(&cached_rows[row_start..row_end], dtype, &device)?;
        let loss = batch_loss(&model, &data.pixels, &data.actions, loss_weights)?;
        let scalars = loss_scalars(&loss)?;
        ensure_finite_loss(step_index + 1, scalars.total)?;
        optimizer.backward_step(&loss.total_loss)?;
        global_step = step_index + 1;
        last_epoch = epoch;
        last_batch_index = batch_index;
        last_loss = Some(scalars.clone());

        if global_step == 1 || global_step % args.log_every == 0 {
            print_loss(global_step, epoch, batch_index, &scalars);
            let row = MetricsRow {
                kind: "train",
                step: global_step,
                epoch,
                batch_index,
                rows: data.meta.rows,
                episode_idx: data.meta.episode_idx,
                step_idx: data.meta.step_idx,
                lr: args.lr,
                weight_decay: args.weight_decay,
                elapsed_sec: started.elapsed().as_secs_f64(),
                loss: scalars.clone(),
            };
            write_json_line(&mut metrics, &row)?;
        }

        if args.save_every > 0 && global_step % args.save_every == 0 {
            let checkpoint = args
                .output_dir
                .join(format!("checkpoint-step-{global_step:08}.safetensors"));
            let optimizer_checkpoint = args
                .output_dir
                .join(format!("optimizer-step-{global_step:08}.safetensors"));
            save_weights(&varmap, &checkpoint)?;
            save_optimizer(&optimizer, &optimizer_checkpoint)?;
            save_weights(&varmap, &args.output_dir.join("latest.safetensors"))?;
            save_optimizer(&optimizer, &args.output_dir.join("optimizer.safetensors"))?;
            let state = TrainingState::new(
                started_at_unix,
                &run,
                &dataset_summary,
                &cfg,
                global_step,
                epoch,
                batch_index,
                batches_per_epoch,
                Some(&checkpoint),
                Some(&optimizer_checkpoint),
                last_loss.as_ref(),
            );
            write_pretty_json(&args.output_dir.join("training-state.json"), &state)?;
        }
    }

    ensure!(global_step > 0, "no optimizer steps were run");
    let final_checkpoint = args.output_dir.join("final.safetensors");
    let final_optimizer = args.output_dir.join("final-optimizer.safetensors");
    save_weights(&varmap, &final_checkpoint)?;
    save_optimizer(&optimizer, &final_optimizer)?;
    save_weights(&varmap, &args.output_dir.join("latest.safetensors"))?;
    save_optimizer(&optimizer, &args.output_dir.join("optimizer.safetensors"))?;
    let state = TrainingState::new(
        started_at_unix,
        &run,
        &dataset_summary,
        &cfg,
        global_step,
        last_epoch,
        last_batch_index,
        batches_per_epoch,
        Some(&final_checkpoint),
        Some(&final_optimizer),
        last_loss.as_ref(),
    );
    write_pretty_json(&args.output_dir.join("training-state.json"), &state)?;
    metrics.flush()?;
    println!(
        "saved={} latest={} steps={} elapsed_sec={:.3}",
        final_checkpoint.display(),
        args.output_dir.join("latest.safetensors").display(),
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
        args.history_size > 0,
        "--history-size must be greater than zero"
    );
    ensure!(
        args.action_block > 0,
        "--action-block must be greater than zero"
    );
    ensure!(
        args.image_size > 0,
        "--image-size must be greater than zero"
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
    for (name, value) in [
        ("--prediction-weight", args.prediction_weight),
        (
            "--temporal-alignment-weight",
            args.temporal_alignment_weight,
        ),
        ("--std-weight", args.std_weight),
        ("--std-t-weight", args.std_t_weight),
        ("--covariance-weight", args.covariance_weight),
        ("--covariance-t-weight", args.covariance_t_weight),
        (
            "--temporal-straightening-weight",
            args.temporal_straightening_weight,
        ),
    ] {
        ensure!(
            value.is_finite() && value >= 0.0,
            "{name} must be finite and non-negative"
        );
    }
    Ok(())
}

fn default_pusht_dataset_path() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".stable_worldmodel")
        .join("pusht_expert_train.h5")
}

fn epoch_seed(seed: u64, epoch: usize) -> u64 {
    seed.wrapping_add((epoch as u64).wrapping_mul(0x9E3779B97F4A7C15))
}

fn load_resume_state(dir: &Path) -> anyhow::Result<SavedTrainingState> {
    let path = dir.join("training-state.json");
    let json =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&json).with_context(|| format!("failed to parse {}", path.display()))
}

fn ensure_resume_compatible(current: &RunSettings, saved: &RunSettings) -> anyhow::Result<()> {
    ensure_eq("dataset_h5", &current.dataset_h5, &saved.dataset_h5)?;
    ensure_eq("config", &current.config, &saved.config)?;
    ensure_eq("device", &current.device, &saved.device)?;
    ensure_eq("dtype", &current.dtype, &saved.dtype)?;
    ensure_eq("batch_size", &current.batch_size, &saved.batch_size)?;
    ensure_eq("history_size", &current.history_size, &saved.history_size)?;
    ensure_eq("action_block", &current.action_block, &saved.action_block)?;
    ensure_eq("image_size", &current.image_size, &saved.image_size)?;
    ensure_eq("seed", &current.seed, &saved.seed)?;
    ensure_eq(
        "normalize_actions",
        &current.normalize_actions,
        &saved.normalize_actions,
    )?;
    ensure_eq("lr", &current.lr, &saved.lr)?;
    ensure_eq("weight_decay", &current.weight_decay, &saved.weight_decay)?;
    ensure_eq("loss_weights", &current.loss_weights, &saved.loss_weights)?;
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

fn load_or_default_config(
    path: Option<&PathBuf>,
    action_dim: usize,
    history_size: usize,
    image_size: usize,
) -> anyhow::Result<LeWmConfig> {
    match path {
        Some(path) => load_config(path),
        None => {
            let mut cfg = LeWmConfig::tiny_patch14_224(action_dim);
            cfg.history_size = history_size;
            cfg.predictor.num_frames = history_size;
            cfg.encoder.image_size = image_size;
            Ok(cfg)
        }
    }
}

fn load_config(path: &PathBuf) -> anyhow::Result<LeWmConfig> {
    let json = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    match LeWmConfig::from_stable_worldmodel_json_str(&json) {
        Ok(cfg) => Ok(cfg),
        Err(stable_err) => serde_json::from_str(&json).with_context(|| {
            format!(
                "failed to parse {} as stable-worldmodel or repo-native LeWM config; stable parse error: {stable_err}",
                path.display()
            )
        }),
    }
}

fn validate_model_config(cfg: &LeWmConfig, dataset: &PushTDataset) -> anyhow::Result<()> {
    ensure!(
        cfg.action_encoder.input_dim == dataset.model_action_dim(),
        "config action_dim {} does not match PushT model action_dim {}",
        cfg.action_encoder.input_dim,
        dataset.model_action_dim()
    );
    ensure!(
        cfg.history_size == dataset.history_size(),
        "config history_size {} does not match requested history_size {}",
        cfg.history_size,
        dataset.history_size()
    );
    ensure!(
        cfg.predictor.num_frames == dataset.history_size(),
        "config predictor.num_frames {} does not match requested history_size {}",
        cfg.predictor.num_frames,
        dataset.history_size()
    );
    ensure!(
        cfg.encoder.image_size == dataset.image_size(),
        "config image_size {} does not match requested image_size {}",
        cfg.encoder.image_size,
        dataset.image_size()
    );
    Ok(())
}

fn loss_weights(args: &Args) -> LeWmLossWeights {
    LeWmLossWeights {
        prediction: args.prediction_weight,
        temporal_alignment: args.temporal_alignment_weight,
        std: args.std_weight,
        std_t: args.std_t_weight,
        covariance: args.covariance_weight,
        covariance_t: args.covariance_t_weight,
        temporal_straightening: args.temporal_straightening_weight,
    }
}

fn loss_scalars(loss: &LeWmBatchLoss) -> anyhow::Result<LossScalars> {
    Ok(LossScalars {
        total: scalar(&loss.total_loss)?,
        prediction: scalar(&loss.prediction_loss)?,
        temporal_alignment: scalar(&loss.temporal_alignment_loss)?,
        std: scalar(&loss.std_loss)?,
        std_t: scalar(&loss.std_t_loss)?,
        covariance: scalar(&loss.covariance_loss)?,
        covariance_t: scalar(&loss.covariance_t_loss)?,
        temporal_straightening: scalar(&loss.temporal_straightening_loss)?,
    })
}

fn scalar(tensor: &Tensor) -> anyhow::Result<f32> {
    tensor.to_scalar::<f32>().map_err(Into::into)
}

fn ensure_finite_loss(step: usize, value: f32) -> anyhow::Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        anyhow::bail!("loss at step {step} is not finite: {value}")
    }
}

fn print_loss(step: usize, epoch: usize, batch_index: usize, loss: &LossScalars) {
    println!(
        "step={} epoch={} batch={} total={:.8e} prediction={:.8e} temp_align={:.8e} std={:.8e} std_t={:.8e} cov={:.8e} cov_t={:.8e} temporal_straightening={:.8e}",
        step,
        epoch,
        batch_index,
        loss.total,
        loss.prediction,
        loss.temporal_alignment,
        loss.std,
        loss.std_t,
        loss.covariance,
        loss.covariance_t,
        loss.temporal_straightening,
    );
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

#[derive(Debug, Clone, Serialize)]
struct LossScalars {
    total: f32,
    prediction: f32,
    temporal_alignment: f32,
    std: f32,
    std_t: f32,
    covariance: f32,
    covariance_t: f32,
    temporal_straightening: f32,
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
    loss: LossScalars,
}

#[derive(Debug, Deserialize)]
struct SavedTrainingState {
    global_step: usize,
    run: RunSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunSettings {
    dataset_h5: PathBuf,
    config: Option<PathBuf>,
    init_safetensors: Option<PathBuf>,
    device: String,
    dtype: String,
    epochs: usize,
    max_steps: Option<usize>,
    batch_size: usize,
    history_size: usize,
    action_block: usize,
    image_size: usize,
    seed: u64,
    normalize_actions: bool,
    blosc_threads: u8,
    lr: f64,
    weight_decay: f64,
    log_every: usize,
    save_every: usize,
    loss_weights: SerializableLossWeights,
}

impl RunSettings {
    fn from_args(args: &Args, dataset_h5: &Path) -> Self {
        Self {
            dataset_h5: dataset_h5.to_path_buf(),
            config: args.config.clone(),
            init_safetensors: args.init_safetensors.clone(),
            device: args.device.to_string(),
            dtype: args.dtype.to_string(),
            epochs: args.epochs,
            max_steps: args.max_steps,
            batch_size: args.batch_size,
            history_size: args.history_size,
            action_block: args.action_block,
            image_size: args.image_size,
            seed: args.seed,
            normalize_actions: !args.no_action_normalize,
            blosc_threads: args.blosc_threads,
            lr: args.lr,
            weight_decay: args.weight_decay,
            log_every: args.log_every,
            save_every: args.save_every,
            loss_weights: SerializableLossWeights {
                prediction: args.prediction_weight,
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
struct SerializableLossWeights {
    prediction: f64,
    temporal_alignment: f64,
    std: f64,
    std_t: f64,
    covariance: f64,
    covariance_t: f64,
    temporal_straightening: f64,
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
    last_loss: Option<&'a LossScalars>,
    run: &'a RunSettings,
    dataset: &'a PushTDatasetSummary,
    model_config: &'a LeWmConfig,
    optimizer_state: &'static str,
}

impl<'a> TrainingState<'a> {
    fn new(
        started_at_unix: u64,
        run: &'a RunSettings,
        dataset: &'a PushTDatasetSummary,
        model_config: &'a LeWmConfig,
        global_step: usize,
        epoch: usize,
        batch_index: usize,
        batches_per_epoch: usize,
        latest_checkpoint: Option<&Path>,
        latest_optimizer_checkpoint: Option<&Path>,
        last_loss: Option<&'a LossScalars>,
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
