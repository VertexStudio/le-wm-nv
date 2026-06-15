use candle::{D, Result, Tensor};

const SIGREG_EPS: f64 = 1e-8;

pub const LEWM_SIGREG_WEIGHT: f64 = 0.09;
pub const LEWM_SIGREG_KNOTS: usize = 17;
pub const LEWM_SIGREG_NUM_PROJ: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SigRegConfig {
    pub weight: f64,
    pub knots: usize,
    pub num_proj: usize,
}

impl Default for SigRegConfig {
    fn default() -> Self {
        Self {
            weight: LEWM_SIGREG_WEIGHT,
            knots: LEWM_SIGREG_KNOTS,
            num_proj: LEWM_SIGREG_NUM_PROJ,
        }
    }
}

impl SigRegConfig {
    pub(crate) fn validate(self) -> Result<()> {
        if !self.weight.is_finite() || self.weight < 0.0 {
            candle::bail!("SIGReg weight must be finite and non-negative");
        }
        if self.knots < 2 {
            candle::bail!("SIGReg requires at least two knots");
        }
        if self.num_proj == 0 {
            candle::bail!("SIGReg requires at least one random projection");
        }
        Ok(())
    }
}

pub(crate) fn sigreg_loss(proj: &Tensor, cfg: SigRegConfig) -> Result<Tensor> {
    cfg.validate()?;
    let dims = proj.dims();
    if dims.len() != 3 {
        candle::bail!(
            "SIGReg expects [time, batch, dim] embeddings, got {:?}",
            proj.shape()
        );
    }
    let (time, batch, dim) = (dims[0], dims[1], dims[2]);
    if batch == 0 || dim == 0 {
        candle::bail!("SIGReg requires non-empty batch and embedding dimensions");
    }

    let dtype = proj.dtype();
    let device = proj.device();
    let projections = Tensor::randn(0f32, 1f32, (dim, cfg.num_proj), device)?.to_dtype(dtype)?;
    let projection_norm = (projections.sqr()?.sum_keepdim(0)? + SIGREG_EPS)?.sqrt()?;
    let projections = projections.broadcast_div(&projection_norm)?;
    let projected = proj
        .reshape((time * batch, dim))?
        .matmul(&projections)?
        .reshape((time, batch, cfg.num_proj))?;

    let t_values = (0..cfg.knots)
        .map(|idx| 3.0f32 * idx as f32 / (cfg.knots - 1) as f32)
        .collect::<Vec<_>>();
    let weights = trapezoid_weights(&t_values);
    let t = Tensor::from_vec(t_values, (cfg.knots,), device)?.to_dtype(dtype)?;
    let weights = Tensor::from_vec(weights, (cfg.knots,), device)?.to_dtype(dtype)?;

    let t4 = t.reshape((1, 1, 1, cfg.knots))?;
    let phi_values = (t.sqr()?.neg()? / 2.0)?.exp()?;
    let phi = phi_values.reshape((1, 1, cfg.knots))?;
    let windowed_weights = weights.broadcast_mul(&phi_values)?;
    let x_t = projected.unsqueeze(3)?.broadcast_mul(&t4)?;
    let cos_mean = x_t.cos()?.mean(1)?;
    let sin_mean = x_t.sin()?.mean(1)?;
    let err = (cos_mean.broadcast_sub(&phi)?.sqr()? + sin_mean.sqr()?)?;
    let weighted = err.broadcast_mul(&windowed_weights.reshape((1, 1, cfg.knots))?)?;
    (weighted.sum(D::Minus1)? * batch as f64)?.mean_all()
}

fn trapezoid_weights(values: &[f32]) -> Vec<f32> {
    let mut weights = vec![0f32; values.len()];
    for idx in 0..values.len() - 1 {
        let dt = values[idx + 1] - values[idx];
        weights[idx] += dt;
        weights[idx + 1] += dt;
    }
    weights
}

#[cfg(test)]
mod tests {
    use candle::{Device, Result, Tensor, Var};

    use super::*;

    #[test]
    fn sigreg_backward_is_finite() -> Result<()> {
        let device = Device::Cpu;
        let x = Var::from_tensor(&Tensor::randn(0f32, 1f32, (3, 4, 8), &device)?)?;
        let loss = sigreg_loss(
            x.as_tensor(),
            SigRegConfig {
                weight: 0.09,
                knots: 5,
                num_proj: 16,
            },
        )?;
        let value = loss.to_scalar::<f32>()?;
        assert!(value.is_finite());
        let grads = loss.backward()?;
        let grad = grads
            .get(x.as_tensor())
            .expect("SIGReg input should receive a gradient");
        let values = grad.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            values.iter().all(|value| value.is_finite()),
            "gradient contains non-finite values: {values:?}"
        );
        Ok(())
    }
}
