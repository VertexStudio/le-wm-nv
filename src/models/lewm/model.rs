use candle::{D, IndexOp, Module, Result, Tensor};
use candle_nn::VarBuilder;

use super::{
    config::LeWmConfig,
    modules::{ActionEmbedder, Mlp, Predictor},
    vit::HfVitEncoder,
};

#[derive(Debug, Clone)]
pub struct LeWm {
    cfg: LeWmConfig,
    encoder: HfVitEncoder,
    predictor: Predictor,
    action_encoder: ActionEmbedder,
    projector: Mlp,
    pred_proj: Mlp,
}

impl LeWm {
    pub fn new(cfg: LeWmConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = HfVitEncoder::new(&cfg.encoder, vb.pp("encoder"))?;
        let predictor = Predictor::new(&cfg.predictor, vb.pp("predictor"))?;
        let action_encoder = ActionEmbedder::new(&cfg.action_encoder, vb.pp("action_encoder"))?;
        let projector = Mlp::new(&cfg.projector, vb.pp("projector"))?;
        let pred_proj = Mlp::new(&cfg.pred_proj, vb.pp("pred_proj"))?;
        Ok(Self {
            cfg,
            encoder,
            predictor,
            action_encoder,
            projector,
            pred_proj,
        })
    }

    pub fn config(&self) -> &LeWmConfig {
        &self.cfg
    }

    pub fn encode_pixels(&self, pixels: &Tensor) -> Result<Tensor> {
        let dims = pixels.dims();
        if dims.len() != 5 {
            candle::bail!(
                "encode_pixels expects [batch, time, channels, height, width], got {:?}",
                pixels.shape()
            );
        }
        let (b, t, c, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        let pixels = pixels.reshape((b * t, c, h, w))?;
        let cls = self.encoder.cls(&pixels)?;
        let emb = self.projector.forward(&cls)?;
        emb.reshape((b, t, ()))
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

    pub fn predict(&self, emb: &Tensor, actions: &Tensor) -> Result<Tensor> {
        let act_emb = self.encode_actions(actions)?;
        self.predict_from_action_embeddings(emb, &act_emb)
    }

    pub fn rollout_embeddings(&self, emb_init: &Tensor, actions: &Tensor) -> Result<Tensor> {
        self.rollout_embeddings_with_history(emb_init, actions, self.cfg.history_size)
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
            .map(|idx| emb_flat.i((.., idx, ..)))
            .collect::<Result<Vec<_>>>()?;
        let n_steps = t - h;
        for step in 0..=n_steps {
            let upper = h + step;
            let lo = upper.saturating_sub(history_size);
            let refs = frames[lo..].iter().collect::<Vec<_>>();
            let emb_trunc = Tensor::stack(&refs, 1)?;
            let act_trunc = all_act_emb.i((.., lo..upper, ..))?;
            let pred = self.predict_from_action_embeddings(&emb_trunc, &act_trunc)?;
            let last = pred.dim(1)? - 1;
            frames.push(pred.i((.., last, ..))?);
        }

        let refs = frames.iter().collect::<Vec<_>>();
        Tensor::stack(&refs, 1)?.reshape((b, s, (), d))
    }

    pub fn goal_cost(&self, predicted_emb: &Tensor, goal_emb: &Tensor) -> Result<Tensor> {
        let pred_dims = predicted_emb.dims();
        if pred_dims.len() != 4 {
            candle::bail!(
                "predicted_emb expects [batch, samples, time, dim], got {:?}",
                predicted_emb.shape()
            );
        }
        let goal = match goal_emb.dims() {
            [b, d] if *b == pred_dims[0] && *d == pred_dims[3] => goal_emb.clone(),
            [b, t, d] if *b == pred_dims[0] && *d == pred_dims[3] => goal_emb.i((.., t - 1, ..))?,
            other => candle::bail!("unsupported goal embedding shape {other:?}"),
        };
        let pred_last = predicted_emb.i((.., .., pred_dims[2] - 1, ..))?;
        let goal = goal.unsqueeze(1)?.broadcast_as(pred_last.shape())?;
        (pred_last - goal)?.sqr()?.sum(D::Minus1)
    }

    pub fn cost_to_goal(
        &self,
        emb_init: &Tensor,
        goal_emb: &Tensor,
        actions: &Tensor,
    ) -> Result<Tensor> {
        let rollout = self.rollout_embeddings(emb_init, actions)?;
        self.goal_cost(&rollout, goal_emb)
    }
}
