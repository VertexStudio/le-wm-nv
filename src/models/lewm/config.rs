use std::path::Path;

use anyhow::Context;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NormKind {
    BatchNorm1d,
    LayerNorm,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VitEncoderConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub layer_norm_eps: f64,
    pub num_channels: usize,
    pub qkv_bias: bool,
}

impl VitEncoderConfig {
    pub fn tiny_patch14_224() -> Self {
        Self {
            image_size: 224,
            patch_size: 14,
            hidden_size: 192,
            num_hidden_layers: 12,
            num_attention_heads: 3,
            intermediate_size: 768,
            layer_norm_eps: 1e-12,
            num_channels: 3,
            qkv_bias: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictorConfig {
    pub num_frames: usize,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub dim_head: usize,
    pub mlp_dim: usize,
}

impl PredictorConfig {
    pub fn lewm_tiny(embed_dim: usize, history_size: usize) -> Self {
        Self {
            num_frames: history_size,
            input_dim: embed_dim,
            hidden_dim: embed_dim,
            output_dim: embed_dim,
            depth: 6,
            heads: 16,
            dim_head: 64,
            mlp_dim: 2048,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionEmbedderConfig {
    pub input_dim: usize,
    pub smoothed_dim: usize,
    pub emb_dim: usize,
    pub mlp_scale: usize,
}

impl ActionEmbedderConfig {
    pub fn new(input_dim: usize, emb_dim: usize) -> Self {
        Self {
            input_dim,
            smoothed_dim: input_dim,
            emb_dim,
            mlp_scale: 4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlpConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub norm: NormKind,
}

impl MlpConfig {
    pub fn projector(embed_dim: usize) -> Self {
        Self {
            input_dim: embed_dim,
            hidden_dim: 2048,
            output_dim: embed_dim,
            norm: NormKind::BatchNorm1d,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeWmConfig {
    pub encoder: VitEncoderConfig,
    pub predictor: PredictorConfig,
    pub action_encoder: ActionEmbedderConfig,
    pub projector: MlpConfig,
    pub pred_proj: MlpConfig,
    pub history_size: usize,
}

impl LeWmConfig {
    pub fn tiny_patch14_224(action_dim: usize) -> Self {
        let embed_dim = 192;
        let history_size = 3;
        Self {
            encoder: VitEncoderConfig::tiny_patch14_224(),
            predictor: PredictorConfig::lewm_tiny(embed_dim, history_size),
            action_encoder: ActionEmbedderConfig::new(action_dim, embed_dim),
            projector: MlpConfig::projector(embed_dim),
            pred_proj: MlpConfig::projector(embed_dim),
            history_size,
        }
    }

    pub fn from_stable_worldmodel_json_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let json = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        Self::from_stable_worldmodel_json_str(&json)
            .with_context(|| format!("failed to parse {}", path.display()))
    }

    pub fn from_stable_worldmodel_json_str(json: &str) -> anyhow::Result<Self> {
        let stable: StableLeWmConfig = serde_json::from_str(json)?;
        stable.try_into()
    }
}

#[derive(Debug, Deserialize)]
struct StableLeWmConfig {
    encoder: StableVitEncoderConfig,
    predictor: StablePredictorConfig,
    action_encoder: StableActionEmbedderConfig,
    projector: StableMlpConfig,
    pred_proj: StableMlpConfig,
}

#[derive(Debug, Deserialize)]
struct StableVitEncoderConfig {
    size: String,
    patch_size: usize,
    image_size: usize,
}

#[derive(Debug, Deserialize)]
struct StablePredictorConfig {
    num_frames: usize,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: Option<usize>,
    depth: usize,
    heads: usize,
    mlp_dim: usize,
    dim_head: usize,
}

#[derive(Debug, Deserialize)]
struct StableActionEmbedderConfig {
    input_dim: usize,
    smoothed_dim: Option<usize>,
    emb_dim: usize,
    mlp_scale: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct StableMlpConfig {
    input_dim: usize,
    hidden_dim: usize,
    output_dim: Option<usize>,
    norm_fn: Option<StableTarget>,
}

#[derive(Debug, Deserialize)]
struct StableTarget {
    #[serde(rename = "_target_")]
    target: String,
}

impl TryFrom<StableLeWmConfig> for LeWmConfig {
    type Error = anyhow::Error;

    fn try_from(stable: StableLeWmConfig) -> anyhow::Result<Self> {
        let encoder = stable.encoder.try_into()?;
        let predictor: PredictorConfig = stable.predictor.into();
        let history_size = predictor.num_frames;
        Ok(Self {
            encoder,
            predictor,
            action_encoder: stable.action_encoder.into(),
            projector: stable.projector.try_into()?,
            pred_proj: stable.pred_proj.try_into()?,
            history_size,
        })
    }
}

impl TryFrom<StableVitEncoderConfig> for VitEncoderConfig {
    type Error = anyhow::Error;

    fn try_from(stable: StableVitEncoderConfig) -> anyhow::Result<Self> {
        if stable.size != "tiny" {
            anyhow::bail!("only LeWM ViT tiny is supported, got {}", stable.size);
        }
        let mut cfg = Self::tiny_patch14_224();
        cfg.patch_size = stable.patch_size;
        cfg.image_size = stable.image_size;
        Ok(cfg)
    }
}

impl From<StablePredictorConfig> for PredictorConfig {
    fn from(stable: StablePredictorConfig) -> Self {
        Self {
            num_frames: stable.num_frames,
            input_dim: stable.input_dim,
            hidden_dim: stable.hidden_dim,
            output_dim: stable.output_dim.unwrap_or(stable.input_dim),
            depth: stable.depth,
            heads: stable.heads,
            dim_head: stable.dim_head,
            mlp_dim: stable.mlp_dim,
        }
    }
}

impl From<StableActionEmbedderConfig> for ActionEmbedderConfig {
    fn from(stable: StableActionEmbedderConfig) -> Self {
        Self {
            input_dim: stable.input_dim,
            smoothed_dim: stable.smoothed_dim.unwrap_or(stable.input_dim),
            emb_dim: stable.emb_dim,
            mlp_scale: stable.mlp_scale.unwrap_or(4),
        }
    }
}

impl TryFrom<StableMlpConfig> for MlpConfig {
    type Error = anyhow::Error;

    fn try_from(stable: StableMlpConfig) -> anyhow::Result<Self> {
        Ok(Self {
            input_dim: stable.input_dim,
            hidden_dim: stable.hidden_dim,
            output_dim: stable.output_dim.unwrap_or(stable.input_dim),
            norm: parse_norm_kind(stable.norm_fn.as_ref())?,
        })
    }
}

fn parse_norm_kind(norm: Option<&StableTarget>) -> anyhow::Result<NormKind> {
    let Some(norm) = norm else {
        return Ok(NormKind::None);
    };
    match norm.target.as_str() {
        "torch.nn.BatchNorm1d" => Ok(NormKind::BatchNorm1d),
        "torch.nn.LayerNorm" => Ok(NormKind::LayerNorm),
        other => anyhow::bail!("unsupported LeWM MLP norm_fn target {other}"),
    }
}
