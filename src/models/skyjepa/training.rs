use candle::{Result, Tensor};
use serde::{Deserialize, Serialize};

use super::SkyJepaModel;
use crate::models::lewm::{SigRegConfig, sigreg_loss};

pub const SKYJEPA_SIGREG_WEIGHT: f64 = 0.02;
pub const SKYJEPA_SIGREG_KNOTS: usize = 17;
// SkyJEPA does not report M and states that it is not sensitive to this value.
// 64 keeps the paper's batch=2048, horizon=20 objective below the memory blow-up
// caused by materializing [T, B, M, knots] with LeWM's image default of 1024.
pub const SKYJEPA_SIGREG_NUM_PROJ: usize = 64;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct SkyJepaLossConfig {
    pub sigreg_weight: f64,
    pub sigreg_knots: usize,
    pub sigreg_num_proj: usize,
}

impl Default for SkyJepaLossConfig {
    fn default() -> Self {
        Self {
            sigreg_weight: SKYJEPA_SIGREG_WEIGHT,
            sigreg_knots: SKYJEPA_SIGREG_KNOTS,
            sigreg_num_proj: SKYJEPA_SIGREG_NUM_PROJ,
        }
    }
}

#[derive(Debug)]
pub struct SkyJepaBatchLoss {
    pub total_loss: Tensor,
    pub prediction_loss: Tensor,
    pub sigreg_loss: Tensor,
    pub predicted_latents: Tensor,
    pub target_latents: Tensor,
}

#[derive(Debug)]
pub struct SkyJepaLatentRollout {
    pub initial_latent: Tensor,
    pub predicted_latents: Tensor,
    pub target_latents: Tensor,
    pub action_embeddings: Tensor,
}

#[derive(Debug, Clone, Serialize)]
pub struct SkyJepaLossScalars {
    pub total: f32,
    pub prediction: f32,
    pub sigreg: f32,
}

impl SkyJepaLossScalars {
    pub fn from_loss(loss: &SkyJepaBatchLoss) -> Result<Self> {
        Ok(Self {
            total: loss.total_loss.to_scalar::<f32>()?,
            prediction: loss.prediction_loss.to_scalar::<f32>()?,
            sigreg: loss.sigreg_loss.to_scalar::<f32>()?,
        })
    }
}

/// Multi-step SkyJEPA objective.
///
/// `states` and `actions` are aligned sequences with shape `[B, H + T, D]`.
/// State and action histories are built as sliding windows in one batched
/// device operation, then the GRU is recursively unrolled for all `T` steps.
/// Unlike the legacy LeWM drone loss, target state embeddings are not detached.
pub fn skyjepa_batch_loss(
    model: &SkyJepaModel,
    states: &Tensor,
    actions: &Tensor,
) -> Result<SkyJepaBatchLoss> {
    skyjepa_batch_loss_with_config(model, states, actions, SkyJepaLossConfig::default())
}

pub fn skyjepa_batch_loss_with_config(
    model: &SkyJepaModel,
    states: &Tensor,
    actions: &Tensor,
    loss_cfg: SkyJepaLossConfig,
) -> Result<SkyJepaBatchLoss> {
    let rollout = skyjepa_latent_rollout(model, states, actions)?;
    let predicted_latents = rollout.predicted_latents;
    let target_latents = rollout.target_latents;
    let prediction_loss = (&predicted_latents - &target_latents)?.sqr()?.mean_all()?;
    let sigreg_cfg = SigRegConfig {
        weight: loss_cfg.sigreg_weight,
        knots: loss_cfg.sigreg_knots,
        num_proj: loss_cfg.sigreg_num_proj,
    };
    let sigreg_loss = sigreg_loss(&predicted_latents.transpose(0, 1)?, sigreg_cfg)?;
    let total_loss = (&prediction_loss + (&sigreg_loss * loss_cfg.sigreg_weight)?)?;

    Ok(SkyJepaBatchLoss {
        total_loss,
        prediction_loss,
        sigreg_loss,
        predicted_latents,
        target_latents,
    })
}

pub fn skyjepa_latent_rollout(
    model: &SkyJepaModel,
    states: &Tensor,
    actions: &Tensor,
) -> Result<SkyJepaLatentRollout> {
    let (batch, time, state_dim) = states.dims3()?;
    let (action_batch, action_time, action_dim) = actions.dims3()?;
    let cfg = model.config();
    if batch != action_batch || time != action_time {
        candle::bail!(
            "SkyJEPA state/action batch-time mismatch: states={:?} actions={:?}",
            states.shape(),
            actions.shape()
        );
    }
    if state_dim != cfg.state_dim || action_dim != cfg.action_dim {
        candle::bail!(
            "SkyJEPA feature mismatch: states={:?} actions={:?} expected state={} action={}",
            states.shape(),
            actions.shape(),
            cfg.state_dim,
            cfg.action_dim
        );
    }
    let expected_time = cfg
        .history_steps
        .checked_add(cfg.rollout_steps)
        .ok_or_else(|| candle::Error::Msg("SkyJEPA sequence length overflowed".to_string()))?;
    if time != expected_time {
        candle::bail!("SkyJEPA training expects exactly H+T={expected_time} steps, got {time}");
    }

    let state_windows = sliding_windows(states, cfg.history_steps, cfg.rollout_steps + 1)?;
    let action_windows = sliding_windows(actions, cfg.history_steps, cfg.rollout_steps)?;
    let state_latents = model.encode_state_windows(&state_windows)?;
    let initial_latent = state_latents.narrow(1, 0, 1)?.squeeze(1)?;
    let target_latents = state_latents.narrow(1, 1, cfg.rollout_steps)?;
    let action_embeddings = model.encode_action_windows(&action_windows)?;
    let predicted_latents =
        model.rollout_from_action_embeddings(&initial_latent, &action_embeddings)?;

    Ok(SkyJepaLatentRollout {
        initial_latent,
        predicted_latents,
        target_latents,
        action_embeddings,
    })
}

fn sliding_windows(values: &Tensor, width: usize, count: usize) -> Result<Tensor> {
    let (_, time, _) = values.dims3()?;
    if width == 0 || count == 0 || width + count - 1 > time {
        candle::bail!(
            "cannot build {count} sliding windows of width {width} from time dimension {time}"
        );
    }
    let windows = (0..count)
        .map(|start| values.narrow(1, start, width))
        .collect::<Result<Vec<_>>>()?;
    let refs = windows.iter().collect::<Vec<_>>();
    Tensor::stack(&refs, 1)
}
