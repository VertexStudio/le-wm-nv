use std::path::PathBuf;

use anyhow::Context;
use candle::{DType, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::Parser;
use le_wm_nv::{
    models::lewm::{LeWm, LeWmConfig, batch_loss},
    runtime::{DTypeSpec, DeviceSpec},
};

#[derive(Parser, Debug)]
struct Args {
    /// NPZ containing `pixels` [B,T,3,H,W] and `actions` [B,T,A].
    #[arg(long)]
    batch_npz: PathBuf,

    /// stable-worldmodel LeWM config JSON. If omitted, infer action_dim and use tiny Patch14/224.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Optional trainable initialization from safetensors.
    #[arg(long)]
    init_safetensors: Option<PathBuf>,

    /// Output safetensors path for updated weights.
    #[arg(long)]
    output: PathBuf,

    #[arg(long, default_value_t = DeviceSpec::Cuda(0))]
    device: DeviceSpec,

    #[arg(long, default_value_t = DTypeSpec::F32)]
    dtype: DTypeSpec,

    #[arg(long, default_value_t = 10)]
    steps: usize,

    #[arg(long, default_value_t = 1)]
    log_every: usize,

    #[arg(long, default_value_t = 1e-4)]
    lr: f64,

    #[arg(long, default_value_t = 0.01)]
    weight_decay: f64,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.steps == 0 {
        anyhow::bail!("--steps must be greater than zero");
    }
    if args.log_every == 0 {
        anyhow::bail!("--log-every must be greater than zero");
    }
    if !args.lr.is_finite() || args.lr <= 0.0 {
        anyhow::bail!("--lr must be finite and greater than zero");
    }
    if !args.weight_decay.is_finite() || args.weight_decay < 0.0 {
        anyhow::bail!("--weight-decay must be finite and non-negative");
    }

    let device = args.device.resolve()?;
    let dtype = args.dtype.dtype();
    if dtype != DType::F32 {
        anyhow::bail!("LeWM training currently requires --dtype f32");
    }

    let arrays = Tensor::read_npz_by_name(&args.batch_npz, &["pixels", "actions"])
        .with_context(|| format!("failed to read {}", args.batch_npz.display()))?;
    let pixels = arrays[0].to_device(&device)?.to_dtype(dtype)?;
    let actions = arrays[1].to_device(&device)?.to_dtype(dtype)?;
    let (_, time, channels, height, width) = pixels.dims5()?;
    let (_, action_time, action_dim) = actions.dims3()?;
    if channels != 3 {
        anyhow::bail!("pixels channel dimension must be 3, got {channels}");
    }
    if action_time != time {
        anyhow::bail!("pixels/actions time mismatch: {time} vs {action_time}");
    }

    let cfg = match args.config.as_ref() {
        Some(path) => load_config(path)?,
        None => LeWmConfig::tiny_patch14_224(action_dim),
    };
    if cfg.action_encoder.input_dim != action_dim {
        anyhow::bail!(
            "config action_dim {} does not match batch action_dim {action_dim}",
            cfg.action_encoder.input_dim
        );
    }
    if cfg.history_size != time {
        anyhow::bail!(
            "config history_size {} must match batch time {time}",
            cfg.history_size
        );
    }
    if cfg.encoder.image_size != height || cfg.encoder.image_size != width {
        anyhow::bail!(
            "config image_size {} must match batch image shape {height}x{width}",
            cfg.encoder.image_size
        );
    }

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
    let model = LeWm::new(cfg, vb)?;
    if let Some(path) = args.init_safetensors.as_ref() {
        varmap
            .load(path)
            .with_context(|| format!("failed to load {}", path.display()))?;
    }

    let params = ParamsAdamW {
        lr: args.lr,
        weight_decay: args.weight_decay,
        ..ParamsAdamW::default()
    };
    let mut optimizer = AdamW::new(varmap.all_vars(), params)?;
    let initial = batch_loss(&model, &pixels, &actions)?;
    let mut last_loss = initial.total_loss.to_scalar::<f32>()?;
    ensure_finite_loss(0, last_loss)?;
    print_loss(0, &initial)?;

    for step in 1..=args.steps {
        let loss = batch_loss(&model, &pixels, &actions)?;
        let total = loss.total_loss.to_scalar::<f32>()?;
        ensure_finite_loss(step, total)?;
        optimizer.backward_step(&loss.total_loss)?;
        last_loss = total;
        if step == 1 || step == args.steps || step % args.log_every == 0 {
            print_loss(step, &loss)?;
        }
    }

    let final_loss = batch_loss(&model, &pixels, &actions)?;
    let final_total = final_loss.total_loss.to_scalar::<f32>()?;
    ensure_finite_loss(args.steps, final_total)?;
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    varmap
        .save(&args.output)
        .with_context(|| format!("failed to save {}", args.output.display()))?;
    println!(
        "saved={} initial_total={:.8e} last_step_total={:.8e} final_total={:.8e}",
        args.output.display(),
        initial.total_loss.to_scalar::<f32>()?,
        last_loss,
        final_total
    );
    Ok(())
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

fn ensure_finite_loss(step: usize, value: f32) -> anyhow::Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        anyhow::bail!("loss at step {step} is not finite: {value}")
    }
}

fn print_loss(step: usize, loss: &le_wm_nv::models::lewm::LeWmBatchLoss) -> anyhow::Result<()> {
    println!(
        "step={} total={:.8e} prediction={:.8e} sigreg={:.8e}",
        step,
        loss.total_loss.to_scalar::<f32>()?,
        loss.prediction_loss.to_scalar::<f32>()?,
        loss.sigreg_loss.to_scalar::<f32>()?,
    );
    Ok(())
}
