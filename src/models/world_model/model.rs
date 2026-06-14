use candle::{Module, Result, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder, layer_norm, linear};

use super::config::{ObservationEncoderConfig, VectorMlpConfig, WorldModelConfig};
use crate::models::lewm::{
    MlpConfig, NormKind,
    modules::{ActionEmbedder, Mlp, Predictor},
    vit::HfVitEncoder,
};

#[derive(Debug, Clone)]
pub struct WorldModel {
    cfg: WorldModelConfig,
    observation_encoder: ObservationEncoder,
    predictor: Predictor,
    action_encoder: ActionEmbedder,
    pred_proj: Mlp,
}

impl WorldModel {
    pub fn new(cfg: WorldModelConfig, vb: VarBuilder) -> Result<Self> {
        cfg.validate()
            .map_err(|err| candle::Error::Msg(err.to_string()))?;
        let observation_encoder =
            ObservationEncoder::new(&cfg.observation_encoder, vb.pp("observation_encoder"))?;
        let predictor = Predictor::new(&cfg.predictor, vb.pp("predictor"))?;
        let action_encoder = ActionEmbedder::new(&cfg.action_encoder, vb.pp("action_encoder"))?;
        let pred_proj = Mlp::new(&cfg.pred_proj, vb.pp("pred_proj"))?;
        Ok(Self {
            cfg,
            observation_encoder,
            predictor,
            action_encoder,
            pred_proj,
        })
    }

    pub fn config(&self) -> &WorldModelConfig {
        &self.cfg
    }

    pub fn encode_pixels(&self, pixels: &Tensor) -> Result<Tensor> {
        self.observation_encoder.encode_pixels(pixels)
    }

    pub fn encode_vector(&self, observations: &Tensor) -> Result<Tensor> {
        self.observation_encoder.encode_vector(observations)
    }

    pub fn encode_actions(&self, actions: &Tensor) -> Result<Tensor> {
        self.action_encoder.forward(actions)
    }

    pub fn predict_from_action_embeddings(&self, emb: &Tensor, act_emb: &Tensor) -> Result<Tensor> {
        let dims = emb.dims();
        if dims.len() != 3 {
            candle::bail!("predict expects [batch, time, dim], got {:?}", emb.shape());
        }
        let (b, t, _) = (dims[0], dims[1], dims[2]);
        let preds = self.predictor.forward(emb, act_emb)?;
        let flat = preds.reshape((b * t, ()))?;
        let projected = self.pred_proj.forward(&flat)?;
        projected.reshape((b, t, ()))
    }

    fn predict_last_from_action_embeddings(
        &self,
        emb: &Tensor,
        act_emb: &Tensor,
    ) -> Result<Tensor> {
        let dims = emb.dims();
        if dims.len() != 3 {
            candle::bail!("predict expects [batch, time, dim], got {:?}", emb.shape());
        }
        let last_pred = self.predictor.forward_last(emb, act_emb)?;
        self.pred_proj.forward(&last_pred)
    }

    pub fn predict(&self, emb: &Tensor, actions: &Tensor) -> Result<Tensor> {
        let act_emb = self.encode_actions(actions)?;
        self.predict_from_action_embeddings(emb, &act_emb)
    }

    pub fn rollout_embeddings_with_history(
        &self,
        emb_init: &Tensor,
        actions: &Tensor,
        history_size: usize,
    ) -> Result<Tensor> {
        let emb_dims = emb_init.dims();
        let act_dims = actions.dims();
        if emb_dims.len() != 4 {
            candle::bail!(
                "emb_init expects [batch, samples, history, dim], got {:?}",
                emb_init.shape()
            );
        }
        if act_dims.len() != 4 {
            candle::bail!(
                "actions expects [batch, samples, horizon, action_dim], got {:?}",
                actions.shape()
            );
        }
        let (b, s, h, d) = (emb_dims[0], emb_dims[1], emb_dims[2], emb_dims[3]);
        let (ab, as_, t, a) = (act_dims[0], act_dims[1], act_dims[2], act_dims[3]);
        if (b, s) != (ab, as_) {
            candle::bail!(
                "emb/action batch sample mismatch: {:?} vs {:?}",
                emb_init.shape(),
                actions.shape()
            );
        }
        if t < h {
            candle::bail!("action horizon {t} is shorter than history {h}");
        }

        let bs = b * s;
        let emb_flat = emb_init.reshape((bs, h, d))?;
        let actions_flat = actions.reshape((bs, t, a))?;
        let all_act_emb = self.encode_actions(&actions_flat)?;

        let mut frames = (0..h)
            .map(|idx| emb_flat.narrow(1, idx, 1)?.squeeze(1))
            .collect::<Result<Vec<_>>>()?;
        let n_steps = t - h;
        for step in 0..=n_steps {
            let upper = h + step;
            let lo = upper.saturating_sub(history_size);
            let refs = frames[lo..].iter().collect::<Vec<_>>();
            let emb_trunc = Tensor::stack(&refs, 1)?;
            let act_trunc = all_act_emb.narrow(1, lo, upper - lo)?;
            frames.push(self.predict_last_from_action_embeddings(&emb_trunc, &act_trunc)?);
        }

        let refs = frames.iter().collect::<Vec<_>>();
        Tensor::stack(&refs, 1)?.reshape((b, s, (), d))
    }
}

#[derive(Debug, Clone)]
enum ObservationEncoder {
    ImageVit {
        encoder: HfVitEncoder,
        projector: Mlp,
    },
    VectorMlp(VectorMlpEncoder),
}

impl ObservationEncoder {
    fn new(cfg: &ObservationEncoderConfig, vb: VarBuilder) -> Result<Self> {
        match cfg {
            ObservationEncoderConfig::ImageVit { encoder, projector } => Ok(Self::ImageVit {
                encoder: HfVitEncoder::new(encoder, vb.pp("encoder"))?,
                projector: Mlp::new(projector, vb.pp("projector"))?,
            }),
            ObservationEncoderConfig::VectorMlp(cfg) => Ok(Self::VectorMlp(VectorMlpEncoder::new(
                cfg,
                vb.pp("vector_mlp"),
            )?)),
        }
    }

    fn encode_pixels(&self, pixels: &Tensor) -> Result<Tensor> {
        let Self::ImageVit { encoder, projector } = self else {
            candle::bail!("encode_pixels called on non-image observation encoder");
        };
        let dims = pixels.dims();
        if dims.len() != 5 {
            candle::bail!(
                "encode_pixels expects [batch, time, channels, height, width], got {:?}",
                pixels.shape()
            );
        }
        let (b, t, c, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        let pixels = pixels.reshape((b * t, c, h, w))?;
        let cls = encoder.cls(&pixels)?;
        let emb = projector.forward(&cls)?;
        emb.reshape((b, t, ()))
    }

    fn encode_vector(&self, observations: &Tensor) -> Result<Tensor> {
        let Self::VectorMlp(encoder) = self else {
            candle::bail!("encode_vector called on non-vector observation encoder");
        };
        encoder.forward(observations)
    }
}

#[derive(Debug, Clone)]
struct VectorMlpEncoder {
    input: Linear,
    hidden: Vec<Linear>,
    norms: Vec<MaybeNorm>,
    output: Linear,
}

impl VectorMlpEncoder {
    fn new(cfg: &VectorMlpConfig, vb: VarBuilder) -> Result<Self> {
        if cfg.depth < 2 {
            candle::bail!("VectorMlp depth must be at least 2");
        }
        let input = linear(cfg.input_dim, cfg.hidden_dim, vb.pp("layers").pp(0))?;
        let hidden_count = cfg.depth.saturating_sub(2);
        let mut hidden = Vec::with_capacity(hidden_count);
        let mut norms = Vec::with_capacity(hidden_count + 1);
        norms.push(MaybeNorm::new(
            cfg.norm,
            cfg.hidden_dim,
            vb.pp("norms").pp(0),
        )?);
        for idx in 0..hidden_count {
            hidden.push(linear(
                cfg.hidden_dim,
                cfg.hidden_dim,
                vb.pp("layers").pp(idx + 1),
            )?);
            norms.push(MaybeNorm::new(
                cfg.norm,
                cfg.hidden_dim,
                vb.pp("norms").pp(idx + 1),
            )?);
        }
        let output = linear(
            cfg.hidden_dim,
            cfg.output_dim,
            vb.pp("layers").pp(cfg.depth - 1),
        )?;
        Ok(Self {
            input,
            hidden,
            norms,
            output,
        })
    }
}

impl Module for VectorMlpEncoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dims = xs.dims();
        if dims.len() != 3 {
            candle::bail!(
                "VectorMlpEncoder expects [batch, time, dim], got {:?}",
                xs.shape()
            );
        }
        let (b, t, _) = (dims[0], dims[1], dims[2]);
        let mut xs = xs.reshape((b * t, ()))?;
        xs = self.input.forward(&xs)?;
        xs = self.norms[0].forward(&xs)?.gelu()?;
        for (idx, layer) in self.hidden.iter().enumerate() {
            xs = layer.forward(&xs)?;
            xs = self.norms[idx + 1].forward(&xs)?.gelu()?;
        }
        self.output.forward(&xs)?.reshape((b, t, ()))
    }
}

#[derive(Debug, Clone)]
enum MaybeNorm {
    LayerNorm(LayerNorm),
    None,
}

impl MaybeNorm {
    fn new(kind: NormKind, dim: usize, vb: VarBuilder) -> Result<Self> {
        match kind {
            NormKind::LayerNorm | NormKind::BatchNorm1d => {
                Ok(Self::LayerNorm(layer_norm(dim, 1e-5, vb)?))
            }
            NormKind::None => Ok(Self::None),
        }
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::LayerNorm(norm) => norm.forward(xs),
            Self::None => Ok(xs.clone()),
        }
    }
}

#[allow(dead_code)]
fn _mlp_config(
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    norm: NormKind,
) -> MlpConfig {
    MlpConfig {
        input_dim,
        hidden_dim,
        output_dim,
        norm,
    }
}
