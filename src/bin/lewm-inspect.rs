use std::path::PathBuf;

use candle::{IndexOp, Tensor};
use clap::Parser;
use le_wm_nv::{
    checkpoint,
    models::lewm::{LeWm, LeWmConfig},
    runtime::{DTypeSpec, DeviceSpec},
};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    weights: Option<PathBuf>,

    #[arg(long, default_value_t = 2)]
    action_dim: usize,

    #[arg(long, default_value_t = 1)]
    batch_size: usize,

    #[arg(long, default_value_t = 3)]
    history: usize,

    #[arg(long, default_value_t = 8)]
    horizon: usize,

    #[arg(long, default_value_t = DeviceSpec::Cuda(0))]
    device: DeviceSpec,

    #[arg(long, default_value_t = DTypeSpec::F32)]
    dtype: DTypeSpec,

    /// Deprecated; use --dtype bf16.
    #[arg(long, default_value_t = false)]
    bf16: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = args.device.resolve()?;
    let dtype = if args.bf16 {
        DTypeSpec::Bf16
    } else {
        args.dtype
    }
    .dtype();
    let cfg = LeWmConfig::tiny_patch14_224(args.action_dim);

    let vb = match args.weights.as_ref() {
        Some(path) => checkpoint::var_builder_from_path(path, dtype, &device)?,
        None => checkpoint::empty_var_builder(dtype, &device),
    };

    let model = LeWm::new(cfg, vb)?;
    println!("model built");

    let b = args.batch_size;
    let h = args.history;
    let image = Tensor::randn(0f32, 1f32, (b, h, 3, 224, 224), &device)?.to_dtype(dtype)?;
    let emb = model.encode_pixels(&image)?;
    println!("encoded pixels: {:?}", emb.shape());

    let emb_init = emb.unsqueeze(1)?;
    let actions = Tensor::randn(0f32, 1f32, (b, 1, args.horizon, args.action_dim), &device)?
        .to_dtype(dtype)?;
    let rollout = model.rollout_embeddings(&emb_init, &actions)?;
    println!("rollout embeddings: {:?}", rollout.shape());

    let cost = model.goal_cost(&rollout, &emb.i((.., emb.dim(1)? - 1, ..))?)?;
    println!("goal cost: {:?}", cost.shape());
    Ok(())
}
