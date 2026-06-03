use candle::{D, IndexOp, Result, Tensor};

const VC_REG_EPS: f64 = 1e-4;
const COSINE_EPS: f64 = 1e-8;

#[derive(Debug)]
pub struct VcRegOutput {
    pub std_loss: Tensor,
    pub std_t_loss: Tensor,
    pub cov_loss: Tensor,
    pub cov_t_loss: Tensor,
}

#[derive(Debug)]
pub struct PldmLossOutput {
    pub idm_loss: Option<Tensor>,
    pub temp_align_loss: Tensor,
    pub std_loss: Tensor,
    pub std_t_loss: Tensor,
    pub cov_loss: Tensor,
    pub cov_t_loss: Tensor,
}

pub fn pldm_loss(
    z: &Tensor,
    a_pred: Option<&Tensor>,
    a_target: Option<&Tensor>,
) -> Result<PldmLossOutput> {
    let (_, time, _) = z.dims3()?;
    if time < 2 {
        candle::bail!("PLDM loss requires at least two latent frames");
    }

    let idm_loss = match (a_pred, a_target) {
        (Some(a_pred), Some(a_target)) => Some(mse_loss(a_pred, a_target)?),
        _ => None,
    };
    let prev = z.i((.., 0..(time - 1), ..))?;
    let next = z.i((.., 1..time, ..))?;
    let temp_align_loss = mse_loss(&prev, &next)?;
    let vc = vc_reg(z)?;

    Ok(PldmLossOutput {
        idm_loss,
        temp_align_loss,
        std_loss: vc.std_loss,
        std_t_loss: vc.std_t_loss,
        cov_loss: vc.cov_loss,
        cov_t_loss: vc.cov_t_loss,
    })
}

pub fn vc_reg(z: &Tensor) -> Result<VcRegOutput> {
    let (batch, time, dim) = z.dims3()?;
    if batch < 2 || time < 2 || dim < 2 {
        candle::bail!("VCReg requires batch, time, and latent dimensions of at least two");
    }

    let mean = z.mean_keepdim(0)?;
    let centered = z.broadcast_sub(&mean)?;
    let centered_t = centered.transpose(0, 1)?;

    Ok(VcRegOutput {
        std_loss: std_loss_by_time(&centered)?.mean_all()?,
        std_t_loss: std_loss_by_time(&centered_t)?.mean_all()?,
        cov_loss: cov_loss_by_time(&centered)?.mean_all()?,
        cov_t_loss: cov_loss_by_time(&centered_t)?.mean_all()?,
    })
}

pub fn temporal_straightening_loss(x: &Tensor) -> Result<Tensor> {
    let (_, time, _) = x.dims3()?;
    if time < 3 {
        candle::bail!("temporal straightening loss requires at least three frames");
    }

    let prev = x.i((.., 0..(time - 1), ..))?;
    let next = x.i((.., 1..time, ..))?;
    let velocities = (next - prev)?;
    let lhs = velocities.i((.., 0..(time - 2), ..))?;
    let rhs = velocities.i((.., 1..(time - 1), ..))?;
    cosine_similarity(&lhs, &rhs)?.mean_all()?.neg()
}

fn mse_loss(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    (lhs - rhs)?.sqr()?.mean_all()
}

fn std_loss_by_time(z: &Tensor) -> Result<Tensor> {
    let z = z.transpose(0, 1)?;
    let std = (z.var(1)? + VC_REG_EPS)?.sqrt()?;
    let one = Tensor::new(1f32, std.device())?.broadcast_as(std.shape())?;
    one.broadcast_sub(&std)?.relu()?.mean(D::Minus1)
}

fn cov_loss_by_time(z: &Tensor) -> Result<Tensor> {
    let (batch, _, dim) = z.dims3()?;
    let z = z.transpose(0, 1)?.contiguous()?;
    let cov = (z.t()?.contiguous()?.matmul(&z)? / (batch - 1) as f64)?;
    let cov_sq = cov.sqr()?;
    let total = cov_sq.sum(D::Minus1)?.sum(D::Minus1)?;
    let eye = Tensor::eye(dim, cov.dtype(), cov.device())?
        .unsqueeze(0)?
        .broadcast_as(cov.shape())?;
    let diag = cov_sq.broadcast_mul(&eye)?.sum(D::Minus1)?.sum(D::Minus1)?;
    (total - diag)? / (dim * dim - dim) as f64
}

fn cosine_similarity(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let dot = lhs.broadcast_mul(rhs)?.sum(D::Minus1)?;
    let lhs_norm = norm_last_dim(lhs)?;
    let rhs_norm = norm_last_dim(rhs)?;
    dot.broadcast_div(&lhs_norm.broadcast_mul(&rhs_norm)?)
}

fn norm_last_dim(x: &Tensor) -> Result<Tensor> {
    let norm = x.sqr()?.sum(D::Minus1)?.sqrt()?;
    let floor = Tensor::new(COSINE_EPS as f32, norm.device())?.broadcast_as(norm.shape())?;
    norm.broadcast_maximum(&floor)
}
