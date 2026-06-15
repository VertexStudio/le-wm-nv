use candle::{IndexOp, Result, Tensor};
use serde::Serialize;

use super::WorldModel;
use crate::models::lewm::{SigRegConfig, sigreg_loss};

#[derive(Debug)]
pub struct VectorBatchLoss {
    pub total_loss: Tensor,
    pub prediction_loss: Tensor,
    pub sigreg_loss: Tensor,
}

#[derive(Debug, Clone, Serialize)]
pub struct VectorLossScalars {
    pub total: f32,
    pub prediction: f32,
    pub sigreg: f32,
}

impl VectorLossScalars {
    pub fn from_loss(loss: &VectorBatchLoss) -> candle::Result<Self> {
        Ok(Self {
            total: loss.total_loss.to_scalar::<f32>()?,
            prediction: loss.prediction_loss.to_scalar::<f32>()?,
            sigreg: loss.sigreg_loss.to_scalar::<f32>()?,
        })
    }
}

pub fn vector_batch_loss(
    model: &WorldModel,
    observations: &Tensor,
    actions: &Tensor,
) -> Result<VectorBatchLoss> {
    vector_batch_loss_with_sigreg(model, observations, actions, SigRegConfig::default())
}

pub(crate) fn vector_batch_loss_with_sigreg(
    model: &WorldModel,
    observations: &Tensor,
    actions: &Tensor,
    sigreg: SigRegConfig,
) -> Result<VectorBatchLoss> {
    sigreg.validate()?;
    let obs_dims = observations.dims();
    let action_dims = actions.dims();
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
    if obs_dims[0] != action_dims[0] || obs_dims[1] != action_dims[1] {
        candle::bail!(
            "vector batch-time mismatch: obs={:?} actions={:?}",
            observations.shape(),
            actions.shape(),
        );
    }
    if obs_dims[1] < 2 {
        candle::bail!("vector batch loss requires at least two timesteps");
    }

    let time = obs_dims[1];
    let history_size = model.config().history_size;
    if time <= history_size {
        candle::bail!(
            "LeWM vector training expects time > history_size, got time={} history_size={}",
            time,
            history_size
        );
    }
    let num_preds = time - history_size;
    let emb = model.encode_vector(observations)?;
    let ctx_emb = emb.i((.., 0..history_size, ..))?;
    let ctx_actions = actions.i((.., 0..history_size, ..))?;
    let pred = model.predict(&ctx_emb, &ctx_actions)?;
    let target_next = emb.detach().i((.., num_preds..time, ..))?;
    let prediction_loss = mse_loss(&pred, &target_next)?;
    let sigreg_loss = sigreg_loss(&emb.transpose(0, 1)?, sigreg)?;
    let total_loss = (prediction_loss.clone() + (sigreg_loss.clone() * sigreg.weight)?)?;

    Ok(VectorBatchLoss {
        total_loss,
        prediction_loss,
        sigreg_loss,
    })
}

fn mse_loss(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    (lhs - rhs)?.sqr()?.mean_all()
}
