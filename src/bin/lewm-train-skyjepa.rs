use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, ensure};
use candle::DType;
use candle_nn::{ParamsAdamW, VarBuilder, VarMap};
use clap::{Parser, ValueEnum};
use le_wm_nv::{
    data::{
        drone_racing::epoch_seed,
        skyjepa::{SkyJepaActionSpace, SkyJepaDatasetConfig, SkyJepaDroneDataset},
    },
    models::skyjepa::{
        SkyJepaConfig, SkyJepaLossConfig, SkyJepaLossScalars, SkyJepaModel, SkyJepaProber,
        SkyJepaProberConfig, SkyJepaProberLossScalars, skyjepa_batch_loss_with_config,
        skyjepa_latent_rollout, skyjepa_prober_loss,
    },
    optim::StatefulAdamW,
    runtime::DeviceSpec,
};
use serde::Serialize;
use serde_json::json;

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum TrainingStage {
    Both,
    Latent,
    Prober,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
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

    /// Required for --stage prober unless output-dir/latent.safetensors exists.
    #[arg(long)]
    latent_checkpoint: Option<PathBuf>,

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

    #[arg(long)]
    latent_max_steps: Option<usize>,

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

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let started = Instant::now();
    let dataset_dir = args.dataset_dir.clone().unwrap_or_else(default_dataset_dir);
    let output_dir = args.output_dir.clone().unwrap_or_else(default_output_dir);
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

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
    let dataset = SkyJepaDroneDataset::open(&dataset_dir, dataset_cfg)?;
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
    write_json(&output_dir.join("model-config.json"), &model_cfg)?;
    write_json(
        &output_dir.join("normalization.json"),
        dataset.normalization(),
    )?;
    write_json(&output_dir.join("episode-splits.json"), dataset.splits())?;
    write_json(&output_dir.join("dataset-config.json"), &dataset_cfg)?;

    let metrics_path = output_dir.join("metrics.jsonl");
    let mut metrics = BufWriter::new(
        File::create(&metrics_path)
            .with_context(|| format!("failed to create {}", metrics_path.display()))?,
    );
    let device = args.device.resolve()?;
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
        let checkpoint = args
            .latent_checkpoint
            .clone()
            .unwrap_or_else(|| output_dir.join("latent.safetensors"));
        latent_vars
            .load(&checkpoint)
            .with_context(|| format!("failed to load {}", checkpoint.display()))?;
    } else {
        train_latent(
            &args,
            &dataset,
            &train_rows,
            &validation_rows,
            &model,
            &latent_vars,
            &output_dir,
            &mut metrics,
            started,
        )?;
    }

    if args.stage != TrainingStage::Latent {
        let mut prober_cfg = SkyJepaProberConfig::paper_derived(model_cfg.latent_dim);
        prober_cfg.kinematics.mass = args.mass;
        prober_cfg.kinematics.hover_throttle = args.hover_throttle;
        prober_cfg.kinematics.action_space = dataset_cfg.action_space;
        write_json(&output_dir.join("prober-config.json"), &prober_cfg)?;
        let prober_vars = VarMap::new();
        let prober = SkyJepaProber::new(
            prober_cfg,
            VarBuilder::from_varmap(&prober_vars, dtype, &device),
        )?;
        println!("skyjepa prober_params={}", parameter_count(&prober_vars));
        train_prober(
            &args,
            &dataset,
            &train_rows,
            &validation_rows,
            &model,
            &prober,
            &prober_vars,
            &output_dir,
            &mut metrics,
            started,
        )?;
    }

    metrics.flush()?;
    println!(
        "skyjepa training complete output={} elapsed_sec={:.3}",
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
    metrics: &mut BufWriter<File>,
    started: Instant,
) -> anyhow::Result<()> {
    let mut optimizer = StatefulAdamW::new_from_varmap(
        vars,
        ParamsAdamW {
            lr: args.max_lr,
            weight_decay: args.weight_decay,
            ..ParamsAdamW::default()
        },
    )?;
    let loss_cfg = SkyJepaLossConfig::default();
    let mut step = 0usize;
    'epochs: for epoch in 0..args.latent_epochs {
        let shuffled = dataset.shuffled_rows(train_rows, epoch_seed(args.seed, epoch));
        for rows in shuffled.chunks(args.batch_size) {
            if rows.len() < 2 {
                continue;
            }
            if args.latent_max_steps.is_some_and(|limit| step >= limit) {
                break 'epochs;
            }
            let lr = scheduled_lr(
                step,
                args.warmup_steps,
                args.cosine_steps,
                args.max_lr,
                args.min_lr,
            );
            let mut params = optimizer.params().clone();
            params.lr = lr;
            optimizer.set_params(params);
            let batch = dataset.batch(rows, DType::F32, model_device(model))?;
            let loss =
                skyjepa_batch_loss_with_config(model, &batch.states, &batch.actions, loss_cfg)?;
            let scalars = SkyJepaLossScalars::from_loss(&loss)?;
            ensure_finite("latent", step, scalars.total)?;
            let grad_norm = optimizer.backward_step_clipped(&loss.total_loss, args.grad_clip)?;
            step += 1;
            if step == 1 || step.is_multiple_of(args.log_every) {
                println!(
                    "stage=latent epoch={} step={} loss={:.6} pred={:.6} sigreg={:.6} grad={:.4} lr={:.3e}",
                    epoch + 1,
                    step,
                    scalars.total,
                    scalars.prediction,
                    scalars.sigreg,
                    grad_norm,
                    lr
                );
                write_metric(
                    metrics,
                    json!({"stage":"latent","kind":"train","epoch":epoch+1,"step":step,"lr":lr,"grad_norm":grad_norm,"loss":scalars,"elapsed_sec":started.elapsed().as_secs_f64()}),
                )?;
            }
            if args.save_every > 0 && step.is_multiple_of(args.save_every) {
                vars.save(output_dir.join("latent-latest.safetensors"))?;
            }
        }
        let validation = validate_latent(
            dataset,
            validation_rows,
            model,
            args.batch_size,
            args.validation_batches,
        )?;
        println!(
            "stage=latent epoch={} validation_prediction={validation:.6}",
            epoch + 1
        );
        write_metric(
            metrics,
            json!({"stage":"latent","kind":"validation","epoch":epoch+1,"step":step,"prediction_loss":validation,"elapsed_sec":started.elapsed().as_secs_f64()}),
        )?;
    }
    ensure!(step > 0, "latent stage performed zero optimizer steps");
    vars.save(output_dir.join("latent.safetensors"))?;
    vars.save(output_dir.join("latent-latest.safetensors"))?;
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
    metrics: &mut BufWriter<File>,
    started: Instant,
) -> anyhow::Result<()> {
    let mut optimizer = StatefulAdamW::new_from_varmap(
        vars,
        ParamsAdamW {
            lr: args.max_lr,
            weight_decay: args.weight_decay,
            ..ParamsAdamW::default()
        },
    )?;
    let mut step = 0usize;
    'epochs: for epoch in 0..args.prober_epochs {
        let shuffled = dataset.shuffled_rows(
            train_rows,
            epoch_seed(args.seed.wrapping_add(0x0053_4b59_4a45_5041), epoch),
        );
        for rows in shuffled.chunks(args.batch_size) {
            if args.prober_max_steps.is_some_and(|limit| step >= limit) {
                break 'epochs;
            }
            let lr = scheduled_lr(
                step,
                args.warmup_steps,
                args.cosine_steps,
                args.max_lr,
                args.min_lr,
            );
            let mut params = optimizer.params().clone();
            params.lr = lr;
            optimizer.set_params(params);
            let batch = dataset.batch(rows, DType::F32, model_device(model))?;
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
            ensure_finite("prober", step, scalars.total)?;
            let grad_norm = optimizer.backward_step_clipped(&loss.total_loss, args.grad_clip)?;
            step += 1;
            if step == 1 || step.is_multiple_of(args.log_every) {
                println!(
                    "stage=prober epoch={} step={} loss={:.6} pos={:.6} vel={:.6} attitude={:.6} omega={:.6} grad={:.4} lr={:.3e}",
                    epoch + 1,
                    step,
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
                    json!({"stage":"prober","kind":"train","epoch":epoch+1,"step":step,"lr":lr,"grad_norm":grad_norm,"loss":scalars,"elapsed_sec":started.elapsed().as_secs_f64()}),
                )?;
            }
            if args.save_every > 0 && step.is_multiple_of(args.save_every) {
                vars.save(output_dir.join("prober-latest.safetensors"))?;
            }
        }
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
            json!({"stage":"prober","kind":"validation","epoch":epoch+1,"step":step,"metric_mse":validation,"elapsed_sec":started.elapsed().as_secs_f64()}),
        )?;
    }
    ensure!(step > 0, "prober stage performed zero optimizer steps");
    vars.save(output_dir.join("prober.safetensors"))?;
    vars.save(output_dir.join("prober-latest.safetensors"))?;
    Ok(())
}

fn validate_latent(
    dataset: &SkyJepaDroneDataset,
    rows: &[usize],
    model: &SkyJepaModel,
    batch_size: usize,
    max_batches: usize,
) -> anyhow::Result<f64> {
    let mut weighted_loss = 0.0f64;
    let mut count = 0usize;
    for chunk in limited_chunks(rows, batch_size, max_batches) {
        let batch = dataset.batch(chunk, DType::F32, model_device(model))?;
        let rollout = skyjepa_latent_rollout(model, &batch.states, &batch.actions)?;
        let loss = (&rollout.predicted_latents - &rollout.target_latents)?
            .sqr()?
            .mean_all()?
            .to_scalar::<f32>()?;
        weighted_loss += f64::from(loss) * chunk.len() as f64;
        count += chunk.len();
    }
    ensure!(count > 0, "latent validation evaluated zero rows");
    Ok(weighted_loss / count as f64)
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
        let batch = dataset.batch(chunk, DType::F32, model_device(model))?;
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

fn limited_chunks(rows: &[usize], batch_size: usize, max_batches: usize) -> Vec<&[usize]> {
    let chunks = rows.chunks(batch_size);
    if max_batches == 0 {
        chunks.collect()
    } else {
        chunks.take(max_batches).collect()
    }
}

fn model_device(model: &SkyJepaModel) -> &candle::Device {
    // The initial state encoder owns all model tensors on the same device. A
    // zero-size public accessor would add API surface; this constant probe is
    // replaced below by the model's direct device accessor.
    model.device()
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

fn write_json<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    fs::write(path, serde_json::to_string_pretty(value)?)
        .with_context(|| format!("failed to write {}", path.display()))
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
}
