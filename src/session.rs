use candle::{DType, Device, Result, Tensor};

use crate::models::lewm::LeWm;

#[derive(Debug)]
pub struct LeWmSession {
    model: LeWm,
    device: Device,
    dtype: DType,
    emb: Option<Tensor>,
}

impl LeWmSession {
    pub fn new(model: LeWm, device: Device, dtype: DType) -> Self {
        Self {
            model,
            device,
            dtype,
            emb: None,
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn reset_pixels(&mut self, pixels: &Tensor) -> Result<Tensor> {
        let emb = self.encode_pixels(pixels)?;
        self.emb = Some(emb.clone());
        Ok(emb)
    }

    pub fn encode_pixels(&self, pixels: &Tensor) -> Result<Tensor> {
        let pixels = pixels.to_device(&self.device)?.to_dtype(self.dtype)?;
        self.model.encode_pixels(&pixels)
    }

    pub fn cached_embedding(&self) -> Option<&Tensor> {
        self.emb.as_ref()
    }

    pub fn score_candidates(
        &self,
        action_candidates: &Tensor,
        goal_emb: &Tensor,
    ) -> Result<Tensor> {
        let emb = self.emb.as_ref().ok_or_else(|| {
            candle::Error::Msg("LeWmSession must be reset before scoring candidates".to_string())
        })?;
        let action_candidates = action_candidates
            .to_device(&self.device)?
            .to_dtype(self.dtype)?;
        let goal_emb = goal_emb.to_device(&self.device)?.to_dtype(self.dtype)?;
        let (b, s, _, _) = action_candidates.dims4()?;
        let (_, h, d) = emb.dims3()?;
        let emb_init = emb.unsqueeze(1)?.broadcast_as((b, s, h, d))?;
        let rollout =
            self.model
                .rollout_embeddings_with_history(&emb_init, &action_candidates, h)?;
        self.model.goal_cost(&rollout, &goal_emb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{checkpoint, models::lewm::LeWmConfig};

    #[test]
    fn lewm_session_scores_with_cached_input_history() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let dtype = DType::F32;
        let cfg = LeWmConfig::tiny_patch14_224(2);
        assert_eq!(cfg.history_size, 3);
        let model = LeWm::new(cfg, checkpoint::empty_var_builder(dtype, &device))?;
        let direct_model = model.clone();
        let mut session = LeWmSession::new(model, device.clone(), dtype);

        let pixels = Tensor::randn(0f32, 1f32, (1, 1, 3, 224, 224), &device)?;
        let goal_pixels = Tensor::randn(0f32, 1f32, (1, 1, 3, 224, 224), &device)?;
        let actions = Tensor::randn(0f32, 1f32, (1, 2, 5, 2), &device)?;

        let emb = session.reset_pixels(&pixels)?;
        let goal_emb = session.encode_pixels(&goal_pixels)?;
        let session_cost = session.score_candidates(&actions, &goal_emb)?;

        let emb_init = emb.unsqueeze(1)?.broadcast_as((1, 2, 1, emb.dim(2)?))?;
        let rollout = direct_model.rollout_embeddings_with_history(&emb_init, &actions, 1)?;
        let direct_cost = direct_model.goal_cost(&rollout, &goal_emb)?;
        let max_abs = (session_cost - direct_cost)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;

        assert!(max_abs <= 1e-6, "max abs diff {max_abs}");
        Ok(())
    }
}
