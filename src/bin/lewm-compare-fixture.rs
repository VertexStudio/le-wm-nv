use std::path::PathBuf;

use anyhow::Context;
use candle::{DType, IndexOp, Tensor};
use clap::Parser;
use le_wm_nv::{
    checkpoint,
    models::lewm::{LeWm, LeWmConfig},
    runtime::DeviceSpec,
};

#[cfg(feature = "hub")]
use le_wm_nv::hub;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    fixture: PathBuf,

    #[arg(long)]
    weights: Option<PathBuf>,

    #[arg(long)]
    config: Option<PathBuf>,

    #[arg(long)]
    hf_repo: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long, default_value_t = 10)]
    action_dim: usize,

    #[arg(long, default_value_t = DeviceSpec::Cuda(0))]
    device: DeviceSpec,

    /// Override every per-output tolerance.
    #[arg(long)]
    tolerance: Option<f32>,

    #[arg(long, default_value_t = 1e-3)]
    emb_tolerance: f32,

    #[arg(long, default_value_t = 1e-5)]
    act_emb_tolerance: f32,

    #[arg(long, default_value_t = 1e-3)]
    pred_tolerance: f32,

    #[arg(long, default_value_t = 2e-3)]
    rollout_tolerance: f32,

    #[arg(long, default_value_t = 1e-2)]
    cost_tolerance: f32,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = args.device.resolve()?;
    let (weights, config) = resolve_files(&args)?;

    let cfg = match config.as_ref() {
        Some(path) => LeWmConfig::from_stable_worldmodel_json_file(path)?,
        None => LeWmConfig::tiny_patch14_224(args.action_dim),
    };
    let vb = checkpoint::var_builder_from_path(&weights, DType::F32, &device)?;
    let model = LeWm::new(cfg, vb)?;

    let arrays = Tensor::read_npz_by_name(
        &args.fixture,
        &[
            "pixels",
            "actions",
            "action_candidates",
            "goal_emb",
            "emb",
            "act_emb",
            "pred",
            "rollout",
            "cost",
        ],
    )?;
    let pixels = arrays[0].to_device(&device)?.to_dtype(DType::F32)?;
    let actions = arrays[1].to_device(&device)?.to_dtype(DType::F32)?;
    let action_candidates = arrays[2].to_device(&device)?.to_dtype(DType::F32)?;
    let goal_emb = arrays[3].to_device(&device)?.to_dtype(DType::F32)?;

    let emb = model.encode_pixels(&pixels)?;
    compare(
        "emb",
        &emb,
        &arrays[4],
        tolerance(args.tolerance, args.emb_tolerance),
    )?;

    let act_emb = model.encode_actions(&actions)?;
    compare(
        "act_emb",
        &act_emb,
        &arrays[5],
        tolerance(args.tolerance, args.act_emb_tolerance),
    )?;

    let pred = model.predict_from_action_embeddings(&emb, &act_emb)?;
    compare(
        "pred",
        &pred,
        &arrays[6],
        tolerance(args.tolerance, args.pred_tolerance),
    )?;

    let (b, s, _, _) = action_candidates.dims4()?;
    let (_, h, d) = emb.dims3()?;
    let emb_init = emb.unsqueeze(1)?.broadcast_as((b, s, h, d))?;
    let rollout = model.rollout_embeddings(&emb_init, &action_candidates)?;
    report_time_slices("rollout", &rollout, &arrays[7])?;
    compare(
        "rollout",
        &rollout,
        &arrays[7],
        tolerance(args.tolerance, args.rollout_tolerance),
    )?;

    let cost = model.goal_cost(&rollout, &goal_emb)?;
    compare(
        "cost",
        &cost,
        &arrays[8],
        tolerance(args.tolerance, args.cost_tolerance),
    )?;
    compare_cost_ordering(&cost, &arrays[8])?;

    Ok(())
}

fn tolerance(global: Option<f32>, output_default: f32) -> f32 {
    global.unwrap_or(output_default)
}

fn report_time_slices(name: &str, actual: &Tensor, expected: &Tensor) -> anyhow::Result<()> {
    if actual.shape() != expected.shape() || actual.rank() < 3 {
        return Ok(());
    }
    let time_dim = actual.rank() - 2;
    let time = actual.dim(time_dim)?;
    for idx in 0..time {
        let actual_slice = actual.i((.., .., idx, ..))?;
        let expected_slice = expected.i((.., .., idx, ..))?;
        let expected_slice = expected_slice
            .to_device(actual.device())?
            .to_dtype(DType::F32)?;
        let actual_slice = actual_slice.to_dtype(DType::F32)?;
        let diff = (actual_slice - expected_slice)?.abs()?;
        let max_abs = diff.max_all()?.to_scalar::<f32>()?;
        let mean_abs = diff.mean_all()?.to_scalar::<f32>()?;
        println!("{name}[t={idx}]: max_abs={max_abs:.6e} mean_abs={mean_abs:.6e}");
    }
    Ok(())
}

fn resolve_files(args: &Args) -> anyhow::Result<(PathBuf, Option<PathBuf>)> {
    if let Some(repo_id) = args.hf_repo.as_ref() {
        #[cfg(feature = "hub")]
        {
            let files =
                hub::download_stable_worldmodel_checkpoint(repo_id, args.revision.as_deref())?;
            return Ok((files.weights, Some(files.config)));
        }

        #[cfg(not(feature = "hub"))]
        {
            let _ = repo_id;
            anyhow::bail!("--hf-repo requires building with --features hub");
        }
    }

    let weights = args
        .weights
        .clone()
        .context("provide --weights or --hf-repo")?;
    Ok((weights, args.config.clone()))
}

fn compare(name: &str, actual: &Tensor, expected: &Tensor, tolerance: f32) -> anyhow::Result<()> {
    if actual.shape() != expected.shape() {
        anyhow::bail!(
            "{name} shape mismatch: Candle {:?}, Python {:?}",
            actual.shape(),
            expected.shape()
        );
    }
    ensure_finite(&format!("{name} Candle"), actual)?;
    ensure_finite(&format!("{name} Python"), expected)?;
    let expected = expected.to_device(actual.device())?.to_dtype(DType::F32)?;
    let actual = actual.to_dtype(DType::F32)?;
    let shape = actual.shape().clone();
    let diff = (actual - expected)?.abs()?;
    let max_abs = diff.max_all()?.to_scalar::<f32>()?;
    let mean_abs = diff.mean_all()?.to_scalar::<f32>()?;
    println!(
        "{name}: shape={:?} max_abs={max_abs:.6e} mean_abs={mean_abs:.6e}",
        shape
    );
    if max_abs > tolerance {
        anyhow::bail!("{name} max_abs {max_abs:.6e} exceeds tolerance {tolerance:.6e}");
    }
    Ok(())
}

fn ensure_finite(name: &str, tensor: &Tensor) -> anyhow::Result<()> {
    let values = tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if let Some((idx, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        anyhow::bail!("{name} contains non-finite value {value} at flat index {idx}");
    }
    Ok(())
}

fn compare_cost_ordering(actual: &Tensor, expected: &Tensor) -> anyhow::Result<()> {
    if actual.shape() != expected.shape() {
        return Ok(());
    }
    let actual = actual.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let expected = expected.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let mut mismatches = Vec::new();
    for (batch, (actual_row, expected_row)) in actual.iter().zip(expected.iter()).enumerate() {
        let actual_best = argmin(actual_row);
        let expected_best = argmin(expected_row);
        if actual_best != expected_best {
            mismatches.push((batch, actual_best, expected_best));
        }
    }
    if !mismatches.is_empty() {
        anyhow::bail!("cost argmin mismatch by batch: {mismatches:?}");
    }
    println!("cost argmin: stable for {} batch item(s)", actual.len());
    Ok(())
}

fn argmin(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .min_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}
