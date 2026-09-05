use std::{
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, ensure};
use candle::DType;
use candle_nn::{ParamsAdamW, VarBuilder, VarMap};
use clap::{Parser, ValueEnum};
use le_wm_nv::{
    data::{
        drone_racing::epoch_seed,
        skyjepa::{
            SkyJepaActionSpace, SkyJepaDatasetConfig, SkyJepaDroneDataset, SkyJepaNormalization,
            skyjepa_artifact_fingerprint,
        },
    },
    models::skyjepa::{
        SkyJepaConfig, SkyJepaLossConfig, SkyJepaLossScalars, SkyJepaModel, SkyJepaProber,
        SkyJepaProberConfig, SkyJepaProberLossScalars,
        checkpoint::{ModelContract, SkyJepaCheckpoint, TrainingSnapshot},
        skyjepa_batch_loss_with_config, skyjepa_latent_rollout, skyjepa_prober_loss,
    },
    optim::StatefulAdamW,
    runtime::DeviceSpec,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum TrainingStage {
    Both,
    Latent,
    Prober,
}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ActionSpaceArg {
    RotorForces,
    BodyRatesThrottle,
}

impl From<ActionSpaceArg> for SkyJepaActionSpace {
    fn from(value: ActionSpaceArg) -> Self {
        match value {
            ActionSpaceArg::RotorForces => Self::RotorForces,
            ActionSpaceArg::BodyRatesThrottle => Self::BodyRatesThrottle,
        }
    }
}

#[derive(Parser, Debug)]
#[command(about = "Train the native accelerated SkyJEPA latent model and physics prober")]
struct Args {
    #[arg(long)]
    dataset_dir: Option<PathBuf>,

    #[arg(long)]
    output_dir: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = TrainingStage::Both)]
    stage: TrainingStage,

    /// Continue the selected stage from its latest weights, optimizer, and
    /// deterministic global step in output-dir.
    #[arg(long)]
    resume: bool,

    /// Permit a fresh stage to replace pre-existing artifacts for that stage.
    #[arg(long)]
    overwrite: bool,

    /// Parent checkpoint PACKAGE DIRECTORY for --stage prober. Defaults to output-dir.
    #[arg(long)]
    latent_checkpoint: Option<PathBuf>,

    /// Audit JSON produced by lewm-audit-skyjepa. Defaults to
    /// dataset-dir/audit.json.
    #[arg(long)]
    audit_report: Option<PathBuf>,

    /// Intended only for tiny developer smoke artifacts.
    #[arg(long)]
    skip_audit: bool,

    #[arg(long, default_value_t = DeviceSpec::Cuda(0))]
    device: DeviceSpec,

    #[arg(long, default_value_t = 50)]
    latent_epochs: usize,

    /// The SkyJEPA paper omits this value; 50 is an explicit local assumption.
    #[arg(long, default_value_t = 50)]
    prober_epochs: usize,

    #[arg(long, default_value_t = 2048)]
    batch_size: usize,

    #[arg(long, default_value_t = 20)]
    model_rate_hz: usize,

    /// Rotor forces are the canonical SkyJEPA action model. Select
    /// body-rates-throttle only for the legacy LeWM racing import.
    #[arg(long, value_enum, default_value_t = ActionSpaceArg::RotorForces)]
    action_space: ActionSpaceArg,

    #[arg(long, default_value_t = 7)]
    seed: u64,

    /// Absolute global-step target for the latent stage.
    #[arg(long)]
    latent_max_steps: Option<usize>,

    /// Absolute global-step target for the prober stage.
    #[arg(long)]
    prober_max_steps: Option<usize>,

    #[arg(long, default_value_t = 5e-3)]
    max_lr: f64,

    #[arg(long, default_value_t = 1e-4)]
    min_lr: f64,

    #[arg(long, default_value_t = 4000)]
    warmup_steps: usize,

    #[arg(long, default_value_t = 20000)]
    cosine_steps: usize,

    #[arg(long, default_value_t = 1e-5)]
    weight_decay: f64,

    #[arg(long, default_value_t = 0.5)]
    grad_clip: f64,

    #[arg(long, default_value_t = 0.2)]
    hover_throttle: f64,

    #[arg(long, default_value_t = 1.3)]
    mass: f64,

    #[arg(long, default_value_t = 10)]
    log_every: usize,

    #[arg(long, default_value_t = 1000)]
    save_every: usize,

    /// Limit validation work per epoch. Zero evaluates every batch.
    #[arg(long, default_value_t = 8)]
    validation_batches: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AuditSummary {
    passed: bool,
    artifact_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunManifest {
    format_version: u32,
    created_at_unix: u64,
    git_commit: Option<String>,
    dataset_dir: PathBuf,
    dataset_artifact_sha256: Option<String>,
    audit_report: Option<PathBuf>,
    audit_skipped: bool,
    device: String,
    action_space: SkyJepaActionSpace,
    batch_size: usize,
    model_rate_hz: usize,
    seed: u64,
    max_lr: f64,
    min_lr: f64,
    warmup_steps: usize,
    cosine_steps: usize,
    weight_decay: f64,
    grad_clip: f64,
    model_config: SkyJepaConfig,
    dataset_config: SkyJepaDatasetConfig,
    normalization: SkyJepaNormalization,
    prober_config: SkyJepaProberConfig,
    parent_latent_identity: Option<String>,
    loss_config: serde_json::Value,
    validation_batches: usize,
    stage_epochs: usize,
    stage_max_steps: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StageProgress {
    format_version: u32,
    stage: TrainingStage,
    dataset_dir: PathBuf,
    batch_size: usize,
    epochs: usize,
    max_steps: Option<usize>,
    seed: u64,
    batches_per_epoch: usize,
    global_step: usize,
    best_validation: Option<f64>,
    best_epoch: Option<usize>,
    best_step: Option<usize>,
    completed_requested_steps: bool,
    updated_at_unix: u64,
}

#[derive(Debug, Clone, Serialize)]
struct LatentValidation {
    prediction_loss: f64,
    latent_std_min: f64,
    latent_std_max: f64,
    active_latent_dims: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let started = Instant::now();
    let dataset_dir = absolute_path(&args.dataset_dir.clone().unwrap_or_else(default_dataset_dir))?;
    let output_dir =
        absolute_output_path(&args.output_dir.clone().unwrap_or_else(default_output_dir))?;
    prepare_output_dir(&args, &output_dir)?;
    let (audit_path, _audit) = load_audit(&args, &dataset_dir)?;

    let model_cfg = SkyJepaConfig::paper_derived();
    let dataset_cfg = SkyJepaDatasetConfig {
        batch_size: args.batch_size,
        history_steps: model_cfg.history_steps,
        rollout_steps: model_cfg.rollout_steps,
        model_rate_hz: args.model_rate_hz,
        normalize_states: true,
        normalize_actions: true,
        action_space: args.action_space.into(),
    };
    // Verify the parent before writing any run metadata or interpreting inputs.
    let parent_root = args
        .latent_checkpoint
        .clone()
        .unwrap_or_else(|| output_dir.clone());
    let parent = if args.stage == TrainingStage::Prober {
        let package = SkyJepaCheckpoint::load(&parent_root)?;
        ensure!(
            package.contract.model == model_cfg
                && package.contract.dataset.action_space == dataset_cfg.action_space
                && package.contract.dataset.model_rate_hz == dataset_cfg.model_rate_hz,
            "parent latent checkpoint is incompatible with requested model/data contract"
        );
        Some(package)
    } else {
        None
    };
    let dataset = SkyJepaDroneDataset::open_with_normalization(
        &dataset_dir,
        dataset_cfg,
        parent
            .as_ref()
            .map(|package| package.contract.normalization.clone()),
    )?;
    let train_rows = dataset.train_rows();
    let validation_rows = dataset.validation_rows();
    ensure!(
        train_rows.len() >= 2,
        "SkyJEPA training split contains fewer than two windows"
    );
    ensure!(
        !validation_rows.is_empty(),
        "SkyJEPA validation split contains no windows"
    );
    let mut prober_cfg = SkyJepaProberConfig::paper_derived(model_cfg.latent_dim);
    prober_cfg.kinematics.mass = args.mass;
    prober_cfg.kinematics.hover_throttle = args.hover_throttle;
    prober_cfg.kinematics.action_space = dataset_cfg.action_space;
    let mut manifest = RunManifest {
        format_version: 2,
        created_at_unix: unix_seconds(),
        git_commit: git_commit(),
        dataset_dir: dataset_dir.clone(),
        dataset_artifact_sha256: Some(skyjepa_artifact_fingerprint(&dataset_dir)?),
        audit_report: audit_path,
        audit_skipped: args.skip_audit,
        device: args.device.to_string(),
        action_space: dataset_cfg.action_space,
        batch_size: args.batch_size,
        model_rate_hz: args.model_rate_hz,
        seed: args.seed,
        max_lr: args.max_lr,
        min_lr: args.min_lr,
        warmup_steps: args.warmup_steps,
        cosine_steps: args.cosine_steps,
        weight_decay: args.weight_decay,
        grad_clip: args.grad_clip,
        model_config: model_cfg.clone(),
        dataset_config: dataset_cfg,
        normalization: dataset.normalization().clone(),
        prober_config: prober_cfg.clone(),
        parent_latent_identity: parent
            .as_ref()
            .map(|package| package.latent_identity())
            .transpose()?,
        loss_config: json!({"latent_mse_weight":1.0,"sigreg_weight":0.02,
            "sigreg_knots":17,"sigreg_projections":64,"prober_objective":"state18_elementwise_mse",
            "randomness_version":1,"dtype":"f32","optimizer":"adamw",
            "beta1":0.9,"beta2":0.999,"epsilon":1e-8}),
        validation_batches: args.validation_batches,
        stage_epochs: if args.stage == TrainingStage::Prober {
            args.prober_epochs
        } else {
            args.latent_epochs
        },
        stage_max_steps: if args.stage == TrainingStage::Prober {
            args.prober_max_steps
        } else {
            args.latent_max_steps
        },
    };
    let first_stage = if args.stage == TrainingStage::Prober {
        "prober"
    } else {
        "latent"
    };
    ensure_or_write_manifest(
        &output_dir,
        first_stage,
        &manifest,
        args.resume,
        args.overwrite,
    )?;
    if args.resume {
        let snapshot = TrainingSnapshot::load(&output_dir, first_stage)?;
        let snapshot_contract: RunManifest =
            serde_json::from_value(snapshot.manifest.training_contract.clone())?;
        ensure!(
            manifest_compatibility(&snapshot_contract)? == manifest_compatibility(&manifest)?,
            "snapshot training contract disagrees with resume settings"
        );
        let batches = batch_count(train_rows.len(), args.batch_size);
        let progress = load_or_create_progress(
            &args,
            args.stage,
            &dataset_dir,
            batches,
            manifest.stage_epochs,
            manifest.stage_max_steps,
            &output_dir,
        )?;
        ensure!(
            progress.global_step
                <= requested_steps(manifest.stage_epochs, batches, manifest.stage_max_steps)?,
            "stage is past its requested target"
        );
        manifest = read_json(&output_dir.join(format!("{first_stage}-run-manifest.json")))?;
    }
    if args.overwrite {
        clear_stage_artifacts(&output_dir, first_stage)?;
        if args.stage == TrainingStage::Both {
            clear_stage_artifacts(&output_dir, "prober")?;
        }
    }
    if !args.resume {
        atomic_write_json(&output_dir.join("episode-splits.json"), dataset.splits())?;
    }
    let mut contract = ModelContract {
        model: model_cfg.clone(),
        dataset: dataset_cfg,
        normalization: dataset.normalization().clone(),
        prober: None,
    };

    let device = args.device.resolve()?;
    device.set_seed(args.seed)?;
    let dtype = DType::F32;
    let mut latent_vars = VarMap::new();
    let model = SkyJepaModel::new(
        model_cfg.clone(),
        VarBuilder::from_varmap(&latent_vars, dtype, &device),
    )?;
    println!(
        "skyjepa dataset={} train_windows={} validation_windows={} source_hz={} model_hz={} stride={} latent_params={} device={:?}",
        dataset.root().display(),
        train_rows.len(),
        validation_rows.len(),
        dataset.source_rate_hz(),
        args.model_rate_hz,
        dataset.source_stride(),
        parameter_count(&latent_vars),
        device.location()
    );

    if args.stage == TrainingStage::Prober {
        let checkpoint = parent.as_ref().unwrap().latent_path(&parent_root);
        latent_vars
            .load(&checkpoint)
            .with_context(|| format!("failed to load {}", checkpoint.display()))?;
        let packaged_checkpoint = output_dir.join("latent.safetensors");
        if checkpoint != packaged_checkpoint {
            atomic_copy(&checkpoint, &packaged_checkpoint)?;
        }
    } else {
        if args.resume {
            let checkpoint = TrainingSnapshot::load(&output_dir, "latent")?.weights_path();
            latent_vars
                .load(&checkpoint)
                .with_context(|| format!("failed to load {}", checkpoint.display()))?;
        }
        let mut metrics = metrics_writer(&output_dir.join("latent-metrics.jsonl"), args.resume)?;
        train_latent(
            &args,
            &dataset,
            &train_rows,
            &validation_rows,
            &model,
            &latent_vars,
            &output_dir,
            &dataset_dir,
            &mut metrics,
            started,
        )?;
        SkyJepaCheckpoint::publish(
            &output_dir,
            contract.clone(),
            &output_dir.join("latent.safetensors"),
            None,
            serde_json::to_value(&manifest)?,
        )?;
    }

    if args.stage != TrainingStage::Latent {
        if args.stage == TrainingStage::Both {
            latent_vars.load(output_dir.join("latent.safetensors"))?;
        }
        if args.stage == TrainingStage::Both {
            manifest.parent_latent_identity =
                Some(SkyJepaCheckpoint::load(&output_dir)?.latent_identity()?);
            manifest.stage_epochs = args.prober_epochs;
            manifest.stage_max_steps = args.prober_max_steps;
            ensure_or_write_manifest(&output_dir, "prober", &manifest, false, args.overwrite)?;
        }
        contract.prober = Some(prober_cfg.clone());
        device.set_seed(args.seed ^ 0x5052_4f42_4552_5f53)?;
        let mut prober_vars = VarMap::new();
        let prober = SkyJepaProber::new(
            prober_cfg,
            VarBuilder::from_varmap(&prober_vars, dtype, &device),
        )?;
        if args.resume {
            let checkpoint = TrainingSnapshot::load(&output_dir, "prober")?.weights_path();
            prober_vars
                .load(&checkpoint)
                .with_context(|| format!("failed to load {}", checkpoint.display()))?;
        }
        println!("skyjepa prober_params={}", parameter_count(&prober_vars));
        let mut metrics = metrics_writer(&output_dir.join("prober-metrics.jsonl"), args.resume)?;
        train_prober(
            &args,
            &dataset,
            &train_rows,
            &validation_rows,
            &model,
            &prober,
            &prober_vars,
            &output_dir,
            &dataset_dir,
            &mut metrics,
            started,
        )?;
        SkyJepaCheckpoint::publish(
            &output_dir,
            contract,
            &output_dir.join("latent.safetensors"),
            Some(&output_dir.join("prober.safetensors")),
            serde_json::to_value(&manifest)?,
        )?;
    }

    println!(
        "skyjepa requested stages complete output={} elapsed_sec={:.3}",
        output_dir.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn train_latent(
    args: &Args,
    dataset: &SkyJepaDroneDataset,
    train_rows: &[usize],
    validation_rows: &[usize],
    model: &SkyJepaModel,
    vars: &VarMap,
    output_dir: &Path,
    dataset_dir: &Path,
    metrics: &mut BufWriter<File>,
    started: Instant,
) -> anyhow::Result<()> {
    let batches_per_epoch = batch_count(train_rows.len(), args.batch_size);
    let requested_steps =
        requested_steps(args.latent_epochs, batches_per_epoch, args.latent_max_steps)?;
    let mut progress = load_or_create_progress(
        args,
        TrainingStage::Latent,
        dataset_dir,
        batches_per_epoch,
        args.latent_epochs,
        args.latent_max_steps,
        output_dir,
    )?;
    ensure!(
        progress.global_step <= requested_steps,
        "latent resume step {} is already at requested target {requested_steps}",
        progress.global_step
    );
    let mut optimizer = StatefulAdamW::new_from_varmap(
        vars,
        ParamsAdamW {
            lr: args.max_lr,
            weight_decay: args.weight_decay,
            ..ParamsAdamW::default()
        },
    )?;
    if args.resume {
        optimizer.load_state(
            TrainingSnapshot::load(output_dir, "latent")?.optimizer_path(),
            progress.global_step,
        )?;
    }
    let loss_cfg = SkyJepaLossConfig::default();
    let mut cached_epoch = None;
    let mut shuffled = Vec::new();
    for step_index in progress.global_step..requested_steps {
        let epoch = step_index / batches_per_epoch;
        let batch_index = step_index % batches_per_epoch;
        if cached_epoch != Some(epoch) {
            shuffled = dataset.shuffled_rows(train_rows, epoch_seed(args.seed, epoch));
            cached_epoch = Some(epoch);
        }
        let rows = batch_rows(&shuffled, batch_index, args.batch_size);
        if rows.len() < 2 {
            continue;
        }
        let lr = scheduled_lr(
            step_index,
            args.warmup_steps,
            args.cosine_steps,
            args.max_lr,
            args.min_lr,
        );
        set_optimizer_lr(&mut optimizer, lr);
        let batch = dataset.batch(rows, DType::F32, model.device())?;
        let loss = skyjepa_batch_loss_with_config(model, &batch.states, &batch.actions, loss_cfg)?;
        let scalars = SkyJepaLossScalars::from_loss(&loss)?;
        ensure_finite("latent", step_index, scalars.total)?;
        let grad_norm = optimizer.backward_step_clipped(&loss.total_loss, args.grad_clip)?;
        progress.global_step = step_index + 1;
        if progress.global_step == 1 || progress.global_step.is_multiple_of(args.log_every) {
            println!(
                "stage=latent epoch={} batch={} step={} loss={:.6} pred={:.6} sigreg={:.6} grad={:.4} lr={:.3e}",
                epoch + 1,
                batch_index + 1,
                progress.global_step,
                scalars.total,
                scalars.prediction,
                scalars.sigreg,
                grad_norm,
                lr
            );
            write_metric(
                metrics,
                json!({"stage":"latent","kind":"train","epoch":epoch+1,"batch":batch_index+1,"step":progress.global_step,"lr":lr,"grad_norm":grad_norm,"loss":scalars,"elapsed_sec":started.elapsed().as_secs_f64()}),
            )?;
        }
        let epoch_end = batch_index + 1 == batches_per_epoch;
        let requested_end = progress.global_step == requested_steps;
        if epoch_end || requested_end {
            let validation = validate_latent(
                dataset,
                validation_rows,
                model,
                args.batch_size,
                args.validation_batches,
            )?;
            println!(
                "stage=latent epoch={} validation_prediction={:.6} latent_std=[{:.4},{:.4}] active_dims={}",
                epoch + 1,
                validation.prediction_loss,
                validation.latent_std_min,
                validation.latent_std_max,
                validation.active_latent_dims
            );
            write_metric(
                metrics,
                json!({"stage":"latent","kind":"validation","epoch":epoch+1,"step":progress.global_step,"validation":validation,"elapsed_sec":started.elapsed().as_secs_f64()}),
            )?;
            if progress
                .best_validation
                .is_none_or(|best| validation.prediction_loss < best)
            {
                progress.best_validation = Some(validation.prediction_loss);
                progress.best_epoch = Some(epoch + 1);
                progress.best_step = Some(progress.global_step);
            }
        }
        if requested_end
            || epoch_end
            || (args.save_every > 0 && progress.global_step.is_multiple_of(args.save_every))
        {
            progress.completed_requested_steps = progress.global_step == requested_steps;
            progress.updated_at_unix = unix_seconds();
            save_stage_state(output_dir, "latent", vars, &optimizer, &progress)?;
        }
    }
    promote_best(output_dir, "latent")?;
    metrics.flush()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn train_prober(
    args: &Args,
    dataset: &SkyJepaDroneDataset,
    train_rows: &[usize],
    validation_rows: &[usize],
    model: &SkyJepaModel,
    prober: &SkyJepaProber,
    vars: &VarMap,
    output_dir: &Path,
    dataset_dir: &Path,
    metrics: &mut BufWriter<File>,
    started: Instant,
) -> anyhow::Result<()> {
    let batches_per_epoch = batch_count(train_rows.len(), args.batch_size);
    let requested_steps =
        requested_steps(args.prober_epochs, batches_per_epoch, args.prober_max_steps)?;
    let mut progress = load_or_create_progress(
        args,
        TrainingStage::Prober,
        dataset_dir,
        batches_per_epoch,
        args.prober_epochs,
        args.prober_max_steps,
        output_dir,
    )?;
    ensure!(
        progress.global_step <= requested_steps,
        "prober resume step {} is already at requested target {requested_steps}",
        progress.global_step
    );
    let mut optimizer = StatefulAdamW::new_from_varmap(
        vars,
        ParamsAdamW {
            lr: args.max_lr,
            weight_decay: args.weight_decay,
            ..ParamsAdamW::default()
        },
    )?;
    if args.resume {
        optimizer.load_state(
            TrainingSnapshot::load(output_dir, "prober")?.optimizer_path(),
            progress.global_step,
        )?;
    }
    let mut cached_epoch = None;
    let mut shuffled = Vec::new();
    for step_index in progress.global_step..requested_steps {
        let epoch = step_index / batches_per_epoch;
        let batch_index = step_index % batches_per_epoch;
        if cached_epoch != Some(epoch) {
            shuffled = dataset.shuffled_rows(
                train_rows,
                epoch_seed(args.seed.wrapping_add(0x0053_4b59_4a45_5041), epoch),
            );
            cached_epoch = Some(epoch);
        }
        let rows = batch_rows(&shuffled, batch_index, args.batch_size);
        let lr = scheduled_lr(
            step_index,
            args.warmup_steps,
            args.cosine_steps,
            args.max_lr,
            args.min_lr,
        );
        set_optimizer_lr(&mut optimizer, lr);
        let batch = dataset.batch(rows, DType::F32, model.device())?;
        let loss = skyjepa_prober_loss(
            model,
            prober,
            &batch.states,
            &batch.actions,
            &batch.metric_states,
            &batch.metric_actions,
            &batch.transition_dt,
        )?;
        let scalars = SkyJepaProberLossScalars::from_loss(&loss)?;
        ensure_finite("prober", step_index, scalars.total)?;
        let grad_norm = optimizer.backward_step_clipped(&loss.total_loss, args.grad_clip)?;
        progress.global_step = step_index + 1;
        if progress.global_step == 1 || progress.global_step.is_multiple_of(args.log_every) {
            println!(
                "stage=prober epoch={} batch={} step={} loss={:.6} pos={:.6} vel={:.6} attitude={:.6} omega={:.6} grad={:.4} lr={:.3e}",
                epoch + 1,
                batch_index + 1,
                progress.global_step,
                scalars.total,
                scalars.position,
                scalars.velocity,
                scalars.attitude,
                scalars.angular_velocity,
                grad_norm,
                lr
            );
            write_metric(
                metrics,
                json!({"stage":"prober","kind":"train","epoch":epoch+1,"batch":batch_index+1,"step":progress.global_step,"lr":lr,"grad_norm":grad_norm,"loss":scalars,"elapsed_sec":started.elapsed().as_secs_f64()}),
            )?;
        }
        let epoch_end = batch_index + 1 == batches_per_epoch;
        let requested_end = progress.global_step == requested_steps;
        if epoch_end || requested_end {
            let validation = validate_prober(
                dataset,
                validation_rows,
                model,
                prober,
                args.batch_size,
                args.validation_batches,
            )?;
            println!(
                "stage=prober epoch={} validation_metric_mse={validation:.6}",
                epoch + 1
            );
            write_metric(
                metrics,
                json!({"stage":"prober","kind":"validation","epoch":epoch+1,"step":progress.global_step,"metric_mse":validation,"elapsed_sec":started.elapsed().as_secs_f64()}),
            )?;
            if progress
                .best_validation
                .is_none_or(|best| validation < best)
            {
                progress.best_validation = Some(validation);
                progress.best_epoch = Some(epoch + 1);
                progress.best_step = Some(progress.global_step);
            }
        }
        if requested_end
            || epoch_end
            || (args.save_every > 0 && progress.global_step.is_multiple_of(args.save_every))
        {
            progress.completed_requested_steps = progress.global_step == requested_steps;
            progress.updated_at_unix = unix_seconds();
            save_stage_state(output_dir, "prober", vars, &optimizer, &progress)?;
        }
    }
    promote_best(output_dir, "prober")?;
    metrics.flush()?;
    Ok(())
}

fn validate_latent(
    dataset: &SkyJepaDroneDataset,
    rows: &[usize],
    model: &SkyJepaModel,
    batch_size: usize,
    max_batches: usize,
) -> anyhow::Result<LatentValidation> {
    let latent_dim = model.config().latent_dim;
    let mut weighted_loss = 0.0f64;
    let mut count = 0usize;
    let mut latent_count = 0usize;
    let mut mean = vec![0.0f64; latent_dim];
    let mut m2 = vec![0.0f64; latent_dim];
    for chunk in limited_chunks(rows, batch_size, max_batches) {
        let batch = dataset.batch(chunk, DType::F32, model.device())?;
        let rollout = skyjepa_latent_rollout(model, &batch.states, &batch.actions)?;
        let loss = (&rollout.predicted_latents - &rollout.target_latents)?
            .sqr()?
            .mean_all()?
            .to_scalar::<f32>()?;
        weighted_loss += f64::from(loss) * chunk.len() as f64;
        count += chunk.len();
        for trajectory in rollout.predicted_latents.to_vec3::<f32>()? {
            for latent in trajectory {
                latent_count += 1;
                let n = latent_count as f64;
                for (dim, value) in latent.into_iter().map(f64::from).enumerate() {
                    let delta = value - mean[dim];
                    mean[dim] += delta / n;
                    m2[dim] += delta * (value - mean[dim]);
                }
            }
        }
    }
    ensure!(count > 0, "latent validation evaluated zero rows");
    let denominator = latent_count.saturating_sub(1).max(1) as f64;
    let std = m2
        .into_iter()
        .map(|value| (value / denominator).sqrt())
        .collect::<Vec<_>>();
    Ok(LatentValidation {
        prediction_loss: weighted_loss / count as f64,
        latent_std_min: std.iter().copied().fold(f64::INFINITY, f64::min),
        latent_std_max: std.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        active_latent_dims: std.iter().filter(|value| **value >= 0.1).count(),
    })
}

fn validate_prober(
    dataset: &SkyJepaDroneDataset,
    rows: &[usize],
    model: &SkyJepaModel,
    prober: &SkyJepaProber,
    batch_size: usize,
    max_batches: usize,
) -> anyhow::Result<f64> {
    let mut weighted_loss = 0.0f64;
    let mut count = 0usize;
    for chunk in limited_chunks(rows, batch_size, max_batches) {
        let batch = dataset.batch(chunk, DType::F32, model.device())?;
        let loss = skyjepa_prober_loss(
            model,
            prober,
            &batch.states,
            &batch.actions,
            &batch.metric_states,
            &batch.metric_actions,
            &batch.transition_dt,
        )?;
        let value = loss.total_loss.to_scalar::<f32>()?;
        weighted_loss += f64::from(value) * chunk.len() as f64;
        count += chunk.len();
    }
    ensure!(count > 0, "prober validation evaluated zero rows");
    Ok(weighted_loss / count as f64)
}

fn load_audit(
    args: &Args,
    dataset_dir: &Path,
) -> anyhow::Result<(Option<PathBuf>, Option<AuditSummary>)> {
    if args.skip_audit {
        return Ok((None, None));
    }
    let path = absolute_path(
        &args
            .audit_report
            .clone()
            .unwrap_or_else(|| dataset_dir.join("audit.json")),
    )?;
    let audit: AuditSummary = read_json(&path)?;
    ensure!(
        audit.passed,
        "dataset audit {} did not pass",
        path.display()
    );
    ensure!(
        !audit.artifact_sha256.is_empty(),
        "dataset audit has no artifact fingerprint"
    );
    let current_fingerprint = skyjepa_artifact_fingerprint(dataset_dir)?;
    ensure!(
        current_fingerprint == audit.artifact_sha256,
        "dataset content no longer matches audit {}",
        path.display()
    );
    Ok((Some(path), Some(audit)))
}

fn prepare_output_dir(args: &Args, output_dir: &Path) -> anyhow::Result<()> {
    ensure!(
        !(args.resume && args.overwrite),
        "--resume and --overwrite are mutually exclusive"
    );
    ensure!(
        !(args.resume && args.stage == TrainingStage::Both),
        "resume one explicit stage at a time"
    );
    if args.resume {
        ensure!(output_dir.is_dir(), "resume output-dir does not exist");
        return Ok(());
    }
    for stage in match args.stage {
        TrainingStage::Both => vec!["latent", "prober"],
        TrainingStage::Latent => vec!["latent"],
        TrainingStage::Prober => vec!["prober"],
    } {
        let state = output_dir.join(format!("{stage}-current.json"));
        if state.exists() && !args.overwrite {
            anyhow::bail!(
                "{} already contains a {stage} run; pass --resume or --overwrite",
                output_dir.display()
            );
        }
    }
    Ok(())
}

fn clear_stage_artifacts(output_dir: &Path, stage: &str) -> anyhow::Result<()> {
    for name in [
        format!("{stage}.safetensors"),
        format!("{stage}-best.safetensors"),
        format!("{stage}-latest.safetensors"),
        format!("{stage}-optimizer.safetensors"),
        format!("{stage}-training-state.json"),
        format!("{stage}-current.json"),
        format!("{stage}-metrics.jsonl"),
    ] {
        let path = output_dir.join(name);
        if path.exists() {
            fs::remove_file(&path)
                .with_context(|| format!("failed to remove {}", path.display()))?;
        }
    }
    Ok(())
}

fn ensure_or_write_manifest(
    output_dir: &Path,
    stage: &str,
    manifest: &RunManifest,
    resume: bool,
    overwrite: bool,
) -> anyhow::Result<()> {
    let path = output_dir.join(format!("{stage}-run-manifest.json"));
    if resume {
        let saved: RunManifest = read_json(&path)?;
        let saved_key = manifest_compatibility(&saved)?;
        let requested_key = manifest_compatibility(manifest)?;
        let changed = saved_key
            .as_object()
            .unwrap()
            .iter()
            .filter(|(key, value)| requested_key.get(*key) != Some(*value))
            .map(|(key, _)| key.as_str())
            .collect::<Vec<_>>();
        ensure!(
            changed.is_empty(),
            "resume settings disagree with {stage}-run-manifest.json: {}",
            changed.join(", ")
        );
        Ok(())
    } else {
        ensure!(
            !path.exists() || overwrite,
            "stage manifest already exists; use --resume or --overwrite"
        );
        atomic_write_json(&path, manifest)
    }
}

fn manifest_compatibility(manifest: &RunManifest) -> anyhow::Result<serde_json::Value> {
    let mut value = serde_json::to_value(manifest)?;
    // These describe invocation provenance, not the experiment's numerical contract.
    for field in ["created_at_unix", "git_commit", "audit_report"] {
        value.as_object_mut().unwrap().remove(field);
    }
    Ok(value)
}

#[allow(clippy::too_many_arguments)]
fn load_or_create_progress(
    args: &Args,
    stage: TrainingStage,
    dataset_dir: &Path,
    batches_per_epoch: usize,
    epochs: usize,
    max_steps: Option<usize>,
    output_dir: &Path,
) -> anyhow::Result<StageProgress> {
    if args.resume {
        let name = if stage == TrainingStage::Latent {
            "latent"
        } else {
            "prober"
        };
        let snapshot = TrainingSnapshot::load(output_dir, name)?;
        let saved: StageProgress = serde_json::from_value(snapshot.manifest.progress)?;
        ensure!(
            saved.stage == stage
                && saved.dataset_dir == dataset_dir
                && saved.batch_size == args.batch_size
                && saved.epochs == epochs
                && saved.max_steps == max_steps
                && saved.seed == args.seed
                && saved.batches_per_epoch == batches_per_epoch,
            "resume state is incompatible with current arguments"
        );
        Ok(saved)
    } else {
        Ok(StageProgress {
            format_version: 2,
            stage,
            dataset_dir: dataset_dir.to_path_buf(),
            batch_size: args.batch_size,
            epochs,
            max_steps,
            seed: args.seed,
            batches_per_epoch,
            global_step: 0,
            best_validation: None,
            best_epoch: None,
            best_step: None,
            completed_requested_steps: false,
            updated_at_unix: unix_seconds(),
        })
    }
}

fn save_stage_state(
    output_dir: &Path,
    stage: &str,
    vars: &VarMap,
    optimizer: &StatefulAdamW,
    progress: &StageProgress,
) -> anyhow::Result<()> {
    ensure!(
        optimizer.step_t() == progress.global_step,
        "optimizer/progress step mismatch"
    );
    let contract: serde_json::Value =
        read_json(&output_dir.join(format!("{stage}-run-manifest.json")))?;
    TrainingSnapshot::publish(
        output_dir,
        stage,
        progress.global_step,
        serde_json::to_value(progress)?,
        contract,
        progress.best_step == Some(progress.global_step),
        |directory| {
            vars.save(directory.join("weights.safetensors"))?;
            optimizer.save_state(directory.join("optimizer.safetensors"))?;
            Ok(())
        },
    )?;
    Ok(())
}

fn promote_best(output_dir: &Path, stage: &str) -> anyhow::Result<()> {
    let source = TrainingSnapshot::load(output_dir, stage)?
        .best(output_dir)?
        .weights_path();
    atomic_copy(&source, &output_dir.join(format!("{stage}.safetensors")))
}

fn metrics_writer(path: &Path, append: bool) -> anyhow::Result<BufWriter<File>> {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append)
        .open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    Ok(BufWriter::new(file))
}

fn batch_count(rows: usize, batch_size: usize) -> usize {
    rows.div_ceil(batch_size)
}

fn batch_rows(rows: &[usize], batch_index: usize, batch_size: usize) -> &[usize] {
    let start = batch_index * batch_size;
    &rows[start..(start + batch_size).min(rows.len())]
}

fn requested_steps(
    epochs: usize,
    batches_per_epoch: usize,
    max_steps: Option<usize>,
) -> anyhow::Result<usize> {
    let epoch_steps = epochs
        .checked_mul(batches_per_epoch)
        .context("epochs * batches_per_epoch overflowed")?;
    Ok(max_steps.map_or(epoch_steps, |steps| steps.min(epoch_steps)))
}

fn limited_chunks(rows: &[usize], batch_size: usize, max_batches: usize) -> Vec<&[usize]> {
    let chunks = rows.chunks(batch_size);
    if max_batches == 0 {
        chunks.collect()
    } else {
        chunks.take(max_batches).collect()
    }
}

fn set_optimizer_lr(optimizer: &mut StatefulAdamW, lr: f64) {
    let mut params = optimizer.params().clone();
    params.lr = lr;
    optimizer.set_params(params);
}

fn scheduled_lr(step: usize, warmup: usize, cosine: usize, max_lr: f64, min_lr: f64) -> f64 {
    if warmup > 0 && step < warmup {
        return max_lr * (step + 1) as f64 / warmup as f64;
    }
    if cosine == 0 {
        return min_lr;
    }
    let progress = step.saturating_sub(warmup) as f64 / cosine as f64;
    let progress = progress.clamp(0.0, 1.0);
    min_lr + 0.5 * (max_lr - min_lr) * (1.0 + (std::f64::consts::PI * progress).cos())
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    ensure!(args.latent_epochs > 0, "latent_epochs must be positive");
    ensure!(args.prober_epochs > 0, "prober_epochs must be positive");
    ensure!(args.batch_size > 1, "batch_size must be at least two");
    ensure!(args.model_rate_hz > 0, "model_rate_hz must be positive");
    ensure!(args.log_every > 0, "log_every must be positive");
    ensure!(
        args.latent_max_steps.is_none_or(|steps| steps > 0),
        "latent_max_steps must be positive"
    );
    ensure!(
        args.prober_max_steps.is_none_or(|steps| steps > 0),
        "prober_max_steps must be positive"
    );
    ensure!(
        args.max_lr.is_finite() && args.max_lr > 0.0,
        "max_lr must be positive"
    );
    ensure!(
        args.min_lr.is_finite() && args.min_lr > 0.0 && args.min_lr <= args.max_lr,
        "min_lr must be positive and no greater than max_lr"
    );
    ensure!(
        args.weight_decay.is_finite() && args.weight_decay >= 0.0,
        "weight_decay must be non-negative"
    );
    ensure!(
        args.grad_clip.is_finite() && args.grad_clip > 0.0,
        "grad_clip must be positive"
    );
    ensure!(
        args.mass.is_finite() && args.mass > 0.0,
        "mass must be positive"
    );
    ensure!(
        args.hover_throttle.is_finite() && args.hover_throttle > 0.0,
        "hover_throttle must be positive"
    );
    Ok(())
}

fn ensure_finite(stage: &str, step: usize, value: f32) -> anyhow::Result<()> {
    ensure!(
        value.is_finite(),
        "{stage} loss became non-finite at step {step}"
    );
    Ok(())
}

fn parameter_count(vars: &VarMap) -> usize {
    vars.all_vars()
        .iter()
        .map(|var| var.shape().elem_count())
        .sum()
}

fn write_metric(writer: &mut BufWriter<File>, value: serde_json::Value) -> anyhow::Result<()> {
    serde_json::to_writer(&mut *writer, &value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn atomic_write_json<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    le_wm_nv::models::skyjepa::checkpoint::atomic_json(path, value)
}

fn atomic_copy(source: &Path, destination: &Path) -> anyhow::Result<()> {
    le_wm_nv::models::skyjepa::checkpoint::atomic_bytes(destination, &fs::read(source)?)
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> anyhow::Result<T> {
    serde_json::from_str(
        &fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))
}

fn absolute_path(path: &Path) -> anyhow::Result<PathBuf> {
    fs::canonicalize(path).with_context(|| format!("failed to resolve {}", path.display()))
}

fn absolute_output_path(path: &Path) -> anyhow::Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn git_commit() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn default_dataset_dir() -> PathBuf {
    user_home()
        .join(".stable_worldmodel")
        .join("le-wm-nv-data")
        .join("skyjepa-domain-randomized-20hz")
}

fn default_output_dir() -> PathBuf {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learning_rate_warms_up_and_decays() {
        assert_eq!(scheduled_lr(0, 4, 10, 1.0, 0.1), 0.25);
        assert_eq!(scheduled_lr(3, 4, 10, 1.0, 0.1), 1.0);
        assert_eq!(scheduled_lr(4, 4, 10, 1.0, 0.1), 1.0);
        assert!((scheduled_lr(14, 4, 10, 1.0, 0.1) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn batch_slicing_retains_partial_final_batch() {
        let rows = (0..10).collect::<Vec<_>>();
        assert_eq!(batch_count(rows.len(), 4), 3);
        assert_eq!(batch_rows(&rows, 2, 4), &[8, 9]);
    }
}
