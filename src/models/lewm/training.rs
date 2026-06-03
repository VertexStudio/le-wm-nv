use candle::{IndexOp, Result, Tensor};

use super::{
    LeWm,
    loss::{pldm_loss, temporal_straightening_loss},
};

#[derive(Debug, Clone, Copy)]
pub struct LeWmLossWeights {
    pub prediction: f64,
    pub temporal_alignment: f64,
    pub std: f64,
    pub std_t: f64,
    pub covariance: f64,
    pub covariance_t: f64,
    pub temporal_straightening: f64,
}

impl Default for LeWmLossWeights {
    fn default() -> Self {
        Self {
            prediction: 1.0,
            temporal_alignment: 1.0,
            std: 1.0,
            std_t: 1.0,
            covariance: 1.0,
            covariance_t: 1.0,
            temporal_straightening: 1.0,
        }
    }
}

impl LeWmLossWeights {
    fn validate(self) -> Result<()> {
        for (name, value) in [
            ("prediction", self.prediction),
            ("temporal_alignment", self.temporal_alignment),
            ("std", self.std),
            ("std_t", self.std_t),
            ("covariance", self.covariance),
            ("covariance_t", self.covariance_t),
            ("temporal_straightening", self.temporal_straightening),
        ] {
            if !value.is_finite() || value < 0.0 {
                candle::bail!("LeWM loss weight {name} must be finite and non-negative");
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct LeWmBatchLoss {
    pub total_loss: Tensor,
    pub prediction_loss: Tensor,
    pub temporal_alignment_loss: Tensor,
    pub std_loss: Tensor,
    pub std_t_loss: Tensor,
    pub covariance_loss: Tensor,
    pub covariance_t_loss: Tensor,
    pub temporal_straightening_loss: Tensor,
}

pub fn batch_loss(
    model: &LeWm,
    pixels: &Tensor,
    actions: &Tensor,
    weights: LeWmLossWeights,
) -> Result<LeWmBatchLoss> {
    weights.validate()?;
    let pixel_dims = pixels.dims();
    let action_dims = actions.dims();
    if pixel_dims.len() != 5 {
        candle::bail!(
            "LeWM training pixels expect [batch, time, channels, height, width], got {:?}",
            pixels.shape()
        );
    }
    if action_dims.len() != 3 {
        candle::bail!(
            "LeWM training actions expect [batch, time, action_dim], got {:?}",
            actions.shape()
        );
    }
    if pixel_dims[0] != action_dims[0] || pixel_dims[1] != action_dims[1] {
        candle::bail!(
            "LeWM training pixels/actions batch-time mismatch: {:?} vs {:?}",
            pixels.shape(),
            actions.shape()
        );
    }
    if pixel_dims[1] < 3 {
        candle::bail!("LeWM training batch loss requires at least three frames");
    }

    let emb = model.encode_pixels(pixels)?;
    let pred = model.predict(&emb, actions)?;
    let time = emb.dim(1)?;
    let pred_next = pred.i((.., 0..(time - 1), ..))?;
    let target_next = emb.detach().i((.., 1..time, ..))?;
    let prediction_loss = mse_loss(&pred_next, &target_next)?;

    let pldm = pldm_loss(&emb, None, None)?;
    let temporal_straightening = temporal_straightening_loss(&emb)?;
    let total_loss = weighted_sum(
        &[
            (&prediction_loss, weights.prediction),
            (&pldm.temp_align_loss, weights.temporal_alignment),
            (&pldm.std_loss, weights.std),
            (&pldm.std_t_loss, weights.std_t),
            (&pldm.cov_loss, weights.covariance),
            (&pldm.cov_t_loss, weights.covariance_t),
            (&temporal_straightening, weights.temporal_straightening),
        ],
        prediction_loss.device(),
    )?;

    Ok(LeWmBatchLoss {
        total_loss,
        prediction_loss,
        temporal_alignment_loss: pldm.temp_align_loss,
        std_loss: pldm.std_loss,
        std_t_loss: pldm.std_t_loss,
        covariance_loss: pldm.cov_loss,
        covariance_t_loss: pldm.cov_t_loss,
        temporal_straightening_loss: temporal_straightening,
    })
}

fn mse_loss(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    (lhs - rhs)?.sqr()?.mean_all()
}

fn weighted_sum(terms: &[(&Tensor, f64)], device: &candle::Device) -> Result<Tensor> {
    let mut total = Tensor::new(0f32, device)?;
    for (term, weight) in terms {
        if *weight != 0.0 {
            total = (total + (*term * *weight)?)?;
        }
    }
    Ok(total)
}
