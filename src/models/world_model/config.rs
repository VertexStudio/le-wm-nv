use serde::{Deserialize, Serialize};

use crate::models::lewm::{
    ActionEmbedderConfig, MlpConfig, NormKind, PredictorConfig, VitEncoderConfig,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ObservationEncoderConfig {
    ImageVit {
        encoder: VitEncoderConfig,
        projector: MlpConfig,
    },
    VectorMlp(VectorMlpConfig),
}

impl ObservationEncoderConfig {
    pub fn output_dim(&self) -> usize {
        match self {
            Self::ImageVit { projector, .. } => projector.output_dim,
            Self::VectorMlp(cfg) => cfg.output_dim,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMlpConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub depth: usize,
    pub norm: NormKind,
}

impl VectorMlpConfig {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim,
            output_dim,
            depth: 3,
            norm: NormKind::LayerNorm,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StateHeadConfig {
    VectorDelta {
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        norm: NormKind,
    },
}

impl StateHeadConfig {
    pub fn output_dim(&self) -> usize {
        match self {
            Self::VectorDelta { output_dim, .. } => *output_dim,
        }
    }

    pub fn as_mlp_config(&self) -> MlpConfig {
        match self {
            Self::VectorDelta {
                input_dim,
                hidden_dim,
                output_dim,
                norm,
            } => MlpConfig {
                input_dim: *input_dim,
                hidden_dim: *hidden_dim,
                output_dim: *output_dim,
                norm: *norm,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldModelConfig {
    pub history_size: usize,
    pub observation_encoder: ObservationEncoderConfig,
    pub action_encoder: ActionEmbedderConfig,
    pub predictor: PredictorConfig,
    pub pred_proj: MlpConfig,
    pub state_head: Option<StateHeadConfig>,
}

impl WorldModelConfig {
    pub fn vector_drone_default(
        observation_dim: usize,
        action_dim: usize,
        state_delta_dim: usize,
    ) -> Self {
        let embed_dim = 192;
        let history_size = 8;
        Self {
            history_size,
            observation_encoder: ObservationEncoderConfig::VectorMlp(VectorMlpConfig {
                input_dim: observation_dim,
                hidden_dim: 512,
                output_dim: embed_dim,
                depth: 3,
                norm: NormKind::LayerNorm,
            }),
            action_encoder: ActionEmbedderConfig {
                input_dim: action_dim,
                smoothed_dim: action_dim,
                emb_dim: embed_dim,
                mlp_scale: 4,
            },
            predictor: PredictorConfig {
                num_frames: history_size,
                input_dim: embed_dim,
                hidden_dim: embed_dim,
                output_dim: embed_dim,
                depth: 6,
                heads: 8,
                dim_head: 32,
                mlp_dim: 1024,
            },
            pred_proj: MlpConfig {
                input_dim: embed_dim,
                hidden_dim: 512,
                output_dim: embed_dim,
                norm: NormKind::LayerNorm,
            },
            state_head: Some(StateHeadConfig::VectorDelta {
                input_dim: embed_dim,
                hidden_dim: 512,
                output_dim: state_delta_dim,
                norm: NormKind::LayerNorm,
            }),
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.history_size > 0,
            "history_size must be greater than zero"
        );
        let obs_dim = self.observation_encoder.output_dim();
        anyhow::ensure!(
            obs_dim == self.predictor.input_dim,
            "observation output_dim {obs_dim} must match predictor.input_dim {}",
            self.predictor.input_dim
        );
        anyhow::ensure!(
            self.predictor.output_dim == self.pred_proj.input_dim,
            "predictor.output_dim {} must match pred_proj.input_dim {}",
            self.predictor.output_dim,
            self.pred_proj.input_dim
        );
        anyhow::ensure!(
            self.pred_proj.output_dim == obs_dim,
            "pred_proj.output_dim {} must match observation output_dim {obs_dim}",
            self.pred_proj.output_dim
        );
        anyhow::ensure!(
            self.action_encoder.emb_dim == self.predictor.input_dim,
            "action_encoder.emb_dim {} must match predictor.input_dim {}",
            self.action_encoder.emb_dim,
            self.predictor.input_dim
        );
        anyhow::ensure!(
            self.predictor.num_frames >= self.history_size,
            "predictor.num_frames {} must be >= history_size {}",
            self.predictor.num_frames,
            self.history_size
        );
        if let Some(head) = &self.state_head {
            let head_cfg = head.as_mlp_config();
            anyhow::ensure!(
                head_cfg.input_dim == obs_dim,
                "state head input_dim {} must match embedding dim {obs_dim}",
                head_cfg.input_dim
            );
        }
        Ok(())
    }
}
