use candle::{IndexOp, Module, Result, Tensor};
use candle_nn::{LayerNorm, VarBuilder, layer_norm};
use candle_transformers::models::vit;

use super::config::VitEncoderConfig;

impl From<&VitEncoderConfig> for vit::Config {
    fn from(cfg: &VitEncoderConfig) -> Self {
        Self {
            hidden_size: cfg.hidden_size,
            num_hidden_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
            intermediate_size: cfg.intermediate_size,
            hidden_act: candle_nn::Activation::Gelu,
            layer_norm_eps: cfg.layer_norm_eps,
            image_size: cfg.image_size,
            patch_size: cfg.patch_size,
            num_channels: cfg.num_channels,
            qkv_bias: cfg.qkv_bias,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HfVitEncoder {
    embeddings: vit::Embeddings,
    encoder: vit::Encoder,
    layernorm: LayerNorm,
}

impl HfVitEncoder {
    pub fn new(cfg: &VitEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let vit_cfg = vit::Config::from(cfg);
        let embeddings = vit::Embeddings::new(&vit_cfg, false, vb.pp("embeddings"))?;
        let encoder = vit::Encoder::new(&vit_cfg, vb.pp("encoder"))?;
        let layernorm = layer_norm(
            vit_cfg.hidden_size,
            vit_cfg.layer_norm_eps,
            vb.pp("layernorm"),
        )?;
        Ok(Self {
            embeddings,
            encoder,
            layernorm,
        })
    }

    pub fn forward(&self, pixels: &Tensor) -> Result<Tensor> {
        let embeddings = self.embeddings.forward(pixels, None, false)?;
        let encoded = self.encoder.forward(&embeddings)?;
        self.layernorm.forward(&encoded)
    }

    pub fn cls(&self, pixels: &Tensor) -> Result<Tensor> {
        self.forward(pixels)?.i((.., 0, ..))?.contiguous()
    }
}
