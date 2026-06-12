use candle::{IndexOp, Result, Tensor};
use serde::Serialize;

use super::WorldModel;
use crate::models::lewm::{pldm_loss, temporal_straightening_loss};

#[derive(Debug, Clone, Copy)]
pub struct VectorLossWeights {
    pub state_prediction: f64,
    pub temporal_alignment: f64,
    pub std: f64,
    pub std_t: f64,
    pub covariance: f64,
    pub covariance_t: f64,
    pub temporal_straightening: f64,
}

impl Default for VectorLossWeights {
    fn default() -> Self {
        Self {
            state_prediction: 1.0,
            temporal_alignment: 0.1,
            std: 0.1,
            std_t: 0.1,
            covariance: 0.1,
            covariance_t: 0.1,
            temporal_straightening: 0.1,
        }
    }
}

impl VectorLossWeights {
    fn validate(self) -> Result<()> {
        for (name, value) in [
            ("state_prediction", self.state_prediction),
            ("temporal_alignment", self.temporal_alignment),
            ("std", self.std),
            ("std_t", self.std_t),
            ("covariance", self.covariance),
            ("covariance_t", self.covariance_t),
            ("temporal_straightening", self.temporal_straightening),
        ] {
            if !value.is_finite() || value < 0.0 {
                candle::bail!(
                    "vector world-model loss weight {name} must be finite and non-negative"
                );
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct VectorBatchLoss {
    pub total_loss: Tensor,
    pub state_prediction_loss: Tensor,
    pub temporal_alignment_loss: Tensor,
    pub std_loss: Tensor,
    pub std_t_loss: Tensor,
    pub covariance_loss: Tensor,
    pub covariance_t_loss: Tensor,
    pub temporal_straightening_loss: Tensor,
}

#[derive(Debug, Clone, Serialize)]
pub struct VectorLossScalars {
    pub total: f32,
    pub state_prediction: f32,
    pub temporal_alignment: f32,
    pub std: f32,
    pub std_t: f32,
    pub covariance: f32,
    pub covariance_t: f32,
    pub temporal_straightening: f32,
}

impl VectorLossScalars {
    pub fn from_loss(loss: &VectorBatchLoss) -> candle::Result<Self> {
        Ok(Self {
            total: loss.total_loss.to_scalar::<f32>()?,
            state_prediction: loss.state_prediction_loss.to_scalar::<f32>()?,
            temporal_alignment: loss.temporal_alignment_loss.to_scalar::<f32>()?,
            std: loss.std_loss.to_scalar::<f32>()?,
            std_t: loss.std_t_loss.to_scalar::<f32>()?,
            covariance: loss.covariance_loss.to_scalar::<f32>()?,
            covariance_t: loss.covariance_t_loss.to_scalar::<f32>()?,
            temporal_straightening: loss.temporal_straightening_loss.to_scalar::<f32>()?,
        })
    }
}

pub fn vector_batch_loss(
    model: &WorldModel,
    observations: &Tensor,
    actions: &Tensor,
    target_deltas: &Tensor,
    weights: VectorLossWeights,
) -> Result<VectorBatchLoss> {
    weights.validate()?;
    let obs_dims = observations.dims();
    let action_dims = actions.dims();
    let target_dims = target_deltas.dims();
    if obs_dims.len() != 3 {
        candle::bail!(
            "vector observations expect [batch, time, obs_dim], got {:?}",
            observations.shape()
        );
    }
    if action_dims.len() != 3 {
        candle::bail!(
            "vector actions expect [batch, time, action_dim], got {:?}",
            actions.shape()
        );
    }
    if target_dims.len() != 3 {
        candle::bail!(
            "target deltas expect [batch, time, delta_dim], got {:?}",
            target_deltas.shape()
        );
    }
    if obs_dims[0] != action_dims[0]
        || obs_dims[1] != action_dims[1]
        || obs_dims[0] != target_dims[0]
        || obs_dims[1] != target_dims[1]
    {
        candle::bail!(
            "vector batch-time mismatch: obs={:?} actions={:?} targets={:?}",
            observations.shape(),
            actions.shape(),
            target_deltas.shape()
        );
    }
    if obs_dims[1] < 2 {
        candle::bail!("vector batch loss requires at least two timesteps");
    }

    let emb = model.encode_vector(observations)?;
    let pred_emb = model.predict(&emb, actions)?;
    let pred_state = model.predict_state_deltas_from_embeddings(&pred_emb)?;
    let time = emb.dim(1)?;
    let pred_next = pred_state.i((.., 0..(time - 1), ..))?;
    let target_next = target_deltas.i((.., 0..(time - 1), ..))?;
    let state_prediction_loss = mse_loss(&pred_next, &target_next)?;

    let pldm = pldm_loss(&emb, None, None)?;
    let temporal_straightening = temporal_straightening_loss(&emb)?;
    let total_loss = weighted_sum(
        &[
            (&state_prediction_loss, weights.state_prediction),
            (&pldm.temp_align_loss, weights.temporal_alignment),
            (&pldm.std_loss, weights.std),
            (&pldm.std_t_loss, weights.std_t),
            (&pldm.cov_loss, weights.covariance),
            (&pldm.cov_t_loss, weights.covariance_t),
            (&temporal_straightening, weights.temporal_straightening),
        ],
        state_prediction_loss.device(),
    )?;

    Ok(VectorBatchLoss {
        total_loss,
        state_prediction_loss,
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
