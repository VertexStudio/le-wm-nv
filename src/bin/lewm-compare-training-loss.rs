use std::path::PathBuf;

use candle::{DType, Tensor};
use clap::Parser;
use le_wm_nv::{
    models::lewm::{pldm_loss, temporal_straightening_loss},
    runtime::DeviceSpec,
};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    fixture: PathBuf,

    #[arg(long, default_value_t = DeviceSpec::Cuda(0))]
    device: DeviceSpec,

    #[arg(long, default_value_t = 1e-5)]
    tolerance: f32,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = args.device.resolve()?;
    let arrays = Tensor::read_npz_by_name(
        &args.fixture,
        &[
            "z",
            "a_pred",
            "a_target",
            "idm_loss",
            "temp_align_loss",
            "std_loss",
            "std_t_loss",
            "cov_loss",
            "cov_t_loss",
            "temporal_straightening_loss",
        ],
    )?;

    let z = arrays[0].to_device(&device)?.to_dtype(DType::F32)?;
    let a_pred = arrays[1].to_device(&device)?.to_dtype(DType::F32)?;
    let a_target = arrays[2].to_device(&device)?.to_dtype(DType::F32)?;
    let pldm = pldm_loss(&z, Some(&a_pred), Some(&a_target))?;
    let temporal = temporal_straightening_loss(&z)?;

    compare(
        "idm_loss",
        pldm.idm_loss.as_ref().expect("idm loss was requested"),
        &arrays[3],
        args.tolerance,
    )?;
    compare(
        "temp_align_loss",
        &pldm.temp_align_loss,
        &arrays[4],
        args.tolerance,
    )?;
    compare("std_loss", &pldm.std_loss, &arrays[5], args.tolerance)?;
    compare("std_t_loss", &pldm.std_t_loss, &arrays[6], args.tolerance)?;
    compare("cov_loss", &pldm.cov_loss, &arrays[7], args.tolerance)?;
    compare("cov_t_loss", &pldm.cov_t_loss, &arrays[8], args.tolerance)?;
    compare(
        "temporal_straightening_loss",
        &temporal,
        &arrays[9],
        args.tolerance,
    )?;

    Ok(())
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
    let diff = (actual.clone() - expected.clone())?.abs()?;
    let max_abs = diff.max_all()?.to_scalar::<f32>()?;
    let mean_abs = diff.mean_all()?.to_scalar::<f32>()?;
    let actual_scalar = actual.to_scalar::<f32>()?;
    let expected_scalar = expected.to_scalar::<f32>()?;
    println!(
        "{name}: actual={actual_scalar:.8e} expected={expected_scalar:.8e} max_abs={max_abs:.6e} mean_abs={mean_abs:.6e}"
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
