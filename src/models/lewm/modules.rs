use candle::{D, DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{
    BatchNorm, BatchNormConfig, Conv1d, Conv1dConfig, LayerNorm, Linear, ModuleT, VarBuilder,
    batch_norm, conv1d, linear, linear_no_bias,
};

use super::config::{ActionEmbedderConfig, MlpConfig, NormKind, PredictorConfig};

fn layer_norm_no_affine(xs: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = xs.dtype();
    let internal_dtype = match dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    };
    let hidden = xs.dim(D::Minus1)? as f64;
    let xs = xs.to_dtype(internal_dtype)?;
    let mean = (xs.sum_keepdim(D::Minus1)? / hidden)?;
    let centered = xs.broadcast_sub(&mean)?;
    let var = (centered.sqr()?.sum_keepdim(D::Minus1)? / hidden)?;
    centered
        .broadcast_div(&(var + eps)?.sqrt()?)?
        .to_dtype(dtype)
}

fn modulate(xs: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    (xs * (scale + 1.0)?)? + shift
}

#[derive(Debug, Clone)]
enum MaybeLinear {
    Linear(Linear),
    Identity,
}

impl MaybeLinear {
    fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        if in_dim == out_dim {
            Ok(Self::Identity)
        } else {
            Ok(Self::Linear(linear(in_dim, out_dim, vb)?))
        }
    }
}

impl Module for MaybeLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Linear(linear) => linear.forward(xs),
            Self::Identity => Ok(xs.clone()),
        }
    }
}

#[derive(Debug, Clone)]
enum MlpNorm {
    BatchNorm(BatchNorm),
    LayerNorm(LayerNorm),
    None,
}

impl MlpNorm {
    fn new(kind: NormKind, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        match kind {
            NormKind::BatchNorm1d => {
                let cfg = BatchNormConfig {
                    eps: 1e-5,
                    ..Default::default()
                };
                Ok(Self::BatchNorm(batch_norm(hidden_dim, cfg, vb)?))
            }
            NormKind::LayerNorm => Ok(Self::LayerNorm(candle_nn::layer_norm(
                hidden_dim, 1e-5, vb,
            )?)),
            NormKind::None => Ok(Self::None),
        }
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::BatchNorm(norm) => norm.forward_t(xs, false),
            Self::LayerNorm(norm) => norm.forward(xs),
            Self::None => Ok(xs.clone()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Mlp {
    fc1: Linear,
    norm: MlpNorm,
    fc2: Linear,
}

impl Mlp {
    pub fn new(cfg: &MlpConfig, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(cfg.input_dim, cfg.hidden_dim, vb.pp("net").pp("0"))?;
        let norm = MlpNorm::new(cfg.norm, cfg.hidden_dim, vb.pp("net").pp("1"))?;
        let fc2 = linear(cfg.hidden_dim, cfg.output_dim, vb.pp("net").pp("3"))?;
        Ok(Self { fc1, norm, fc2 })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        let xs = self.norm.forward(&xs)?;
        let xs = xs.gelu()?;
        self.fc2.forward(&xs)
    }
}

#[derive(Debug, Clone)]
pub struct ActionEmbedder {
    patch_embed: Conv1d,
    fc1: Linear,
    fc2: Linear,
}

impl ActionEmbedder {
    pub fn new(cfg: &ActionEmbedderConfig, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv1dConfig {
            stride: 1,
            ..Default::default()
        };
        let patch_embed = conv1d(
            cfg.input_dim,
            cfg.smoothed_dim,
            1,
            conv_cfg,
            vb.pp("patch_embed"),
        )?;
        let fc1 = linear(
            cfg.smoothed_dim,
            cfg.mlp_scale * cfg.emb_dim,
            vb.pp("embed").pp("0"),
        )?;
        let fc2 = linear(
            cfg.mlp_scale * cfg.emb_dim,
            cfg.emb_dim,
            vb.pp("embed").pp("2"),
        )?;
        Ok(Self {
            patch_embed,
            fc1,
            fc2,
        })
    }
}

impl Module for ActionEmbedder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.to_dtype(DType::F32)?;
        let xs = xs.permute((0, 2, 1))?.contiguous()?;
        let xs = self.patch_embed.forward(&xs)?;
        let xs = xs.permute((0, 2, 1))?.contiguous()?;
        let xs = self.fc1.forward(&xs)?.silu()?;
        self.fc2.forward(&xs)
    }
}

#[derive(Debug, Clone)]
struct FeedForward {
    norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl FeedForward {
    fn new(dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        let norm = candle_nn::layer_norm(dim, 1e-5, vb.pp("net").pp("0"))?;
        let fc1 = linear(dim, hidden_dim, vb.pp("net").pp("1"))?;
        let fc2 = linear(hidden_dim, dim, vb.pp("net").pp("4"))?;
        Ok(Self { norm, fc1, fc2 })
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.norm.forward(xs)?;
        let xs = self.fc1.forward(&xs)?.gelu()?;
        self.fc2.forward(&xs)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    norm: LayerNorm,
    to_qkv: Linear,
    to_out: Linear,
    heads: usize,
    dim_head: usize,
    scale: f64,
}

impl Attention {
    fn new(dim: usize, heads: usize, dim_head: usize, vb: VarBuilder) -> Result<Self> {
        let inner_dim = heads * dim_head;
        let norm = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm"))?;
        let to_qkv = linear_no_bias(dim, inner_dim * 3, vb.pp("to_qkv"))?;
        let to_out = linear(inner_dim, dim, vb.pp("to_out").pp("0"))?;
        Ok(Self {
            norm,
            to_qkv,
            to_out,
            heads,
            dim_head,
            scale: (dim_head as f64).powf(-0.5),
        })
    }

    fn causal_mask(seq_len: usize, dtype: DType, device: &Device) -> Result<Tensor> {
        Tensor::tril2(seq_len, dtype, device)
    }
}

impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        let xs = self.norm.forward(xs)?;
        let qkv = self
            .to_qkv
            .forward(&xs)?
            .reshape((b, t, 3, self.heads, self.dim_head))?
            .permute((2, 0, 3, 1, 4))?;
        let q = (qkv.i(0)?.contiguous()? * self.scale)?;
        let k = qkv.i(1)?.contiguous()?;
        let v = qkv.i(2)?.contiguous()?;
        let mut scores = q.matmul(&k.t()?)?;
        let mask = Self::causal_mask(t, scores.dtype(), scores.device())?
            .reshape((1, 1, t, t))?
            .broadcast_as(scores.shape())?;
        let neg_inf = Tensor::full(f32::NEG_INFINITY, scores.shape(), scores.device())?
            .to_dtype(scores.dtype())?;
        scores = mask.eq(0f32)?.where_cond(&neg_inf, &scores)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = attn
            .matmul(&v)?
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((b, t, self.heads * self.dim_head))?;
        self.to_out.forward(&out)
    }
}

#[derive(Debug, Clone)]
struct ConditionalBlock {
    attn: Attention,
    mlp: FeedForward,
    ada_ln: Linear,
}

impl ConditionalBlock {
    fn new(
        dim: usize,
        heads: usize,
        dim_head: usize,
        mlp_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let attn = Attention::new(dim, heads, dim_head, vb.pp("attn"))?;
        let mlp = FeedForward::new(dim, mlp_dim, vb.pp("mlp"))?;
        let ada_ln = linear(dim, 6 * dim, vb.pp("adaLN_modulation").pp("1"))?;
        Ok(Self { attn, mlp, ada_ln })
    }

    fn forward(&self, xs: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let chunks = self.ada_ln.forward(&cond.silu()?)?.chunk(6, D::Minus1)?;
        let shift_msa = &chunks[0];
        let scale_msa = &chunks[1];
        let gate_msa = &chunks[2];
        let shift_mlp = &chunks[3];
        let scale_mlp = &chunks[4];
        let gate_mlp = &chunks[5];

        let attn_in = modulate(&layer_norm_no_affine(xs, 1e-6)?, shift_msa, scale_msa)?;
        let xs = (xs + (gate_msa * self.attn.forward(&attn_in)?)?)?;
        let mlp_in = modulate(&layer_norm_no_affine(&xs, 1e-6)?, shift_mlp, scale_mlp)?;
        &xs + (gate_mlp * self.mlp.forward(&mlp_in)?)?
    }
}

#[derive(Debug, Clone)]
struct Transformer {
    input_proj: MaybeLinear,
    cond_proj: MaybeLinear,
    output_proj: MaybeLinear,
    layers: Vec<ConditionalBlock>,
    norm: LayerNorm,
}

impl Transformer {
    fn new(cfg: &PredictorConfig, vb: VarBuilder) -> Result<Self> {
        let input_proj = MaybeLinear::new(cfg.input_dim, cfg.hidden_dim, vb.pp("input_proj"))?;
        let cond_proj = MaybeLinear::new(cfg.input_dim, cfg.hidden_dim, vb.pp("cond_proj"))?;
        let output_proj = MaybeLinear::new(cfg.hidden_dim, cfg.output_dim, vb.pp("output_proj"))?;
        let layers = (0..cfg.depth)
            .map(|idx| {
                ConditionalBlock::new(
                    cfg.hidden_dim,
                    cfg.heads,
                    cfg.dim_head,
                    cfg.mlp_dim,
                    vb.pp("layers").pp(idx),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let norm = candle_nn::layer_norm(cfg.hidden_dim, 1e-5, vb.pp("norm"))?;
        Ok(Self {
            input_proj,
            cond_proj,
            output_proj,
            layers,
            norm,
        })
    }

    fn forward(&self, xs: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let mut xs = self.input_proj.forward(xs)?;
        let cond = self.cond_proj.forward(cond)?;
        for block in &self.layers {
            xs = block.forward(&xs, &cond)?;
        }
        let xs = self.norm.forward(&xs)?;
        self.output_proj.forward(&xs)
    }
}

#[derive(Debug, Clone)]
pub struct Predictor {
    pos_embedding: Tensor,
    transformer: Transformer,
}

impl Predictor {
    pub fn new(cfg: &PredictorConfig, vb: VarBuilder) -> Result<Self> {
        let pos_embedding = vb.get((1, cfg.num_frames, cfg.input_dim), "pos_embedding")?;
        let transformer = Transformer::new(cfg, vb.pp("transformer"))?;
        Ok(Self {
            pos_embedding,
            transformer,
        })
    }

    pub fn forward(&self, xs: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let t = xs.dim(1)?;
        let pos = self.pos_embedding.i((.., ..t, ..))?;
        let xs = xs.broadcast_add(&pos)?;
        self.transformer.forward(&xs, cond)
    }
}
