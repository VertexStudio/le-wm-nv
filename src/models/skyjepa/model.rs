use candle::{IndexOp, Module, Result, Tensor};
use candle_nn::{GRU, GRUConfig, Linear, RNN, VarBuilder, gru, linear, rnn::GRUState};

use super::{SkyJepaConfig, TemporalConvEncoder};

/// Paper-derived SkyJEPA latent dynamics model.
///
/// A full state history initializes the GRU hidden state. Each encoded rolling
/// action history advances that hidden state by one prediction step. Keeping
/// the recurrent state equal to the predicted latent avoids an extra decoder
/// or projection in the latency-sensitive rollout path.
#[derive(Debug, Clone)]
pub struct SkyJepaModel {
    cfg: SkyJepaConfig,
    state_encoder: TemporalConvEncoder,
    state_projection: Linear,
    action_encoder: TemporalConvEncoder,
    predictor: GRU,
}

impl SkyJepaModel {
    pub fn new(cfg: SkyJepaConfig, vb: VarBuilder) -> Result<Self> {
        cfg.validate()
            .map_err(|error| candle::Error::Msg(error.to_string()))?;
        let state_encoder = TemporalConvEncoder::new(&cfg.state_encoder, vb.pp("state_encoder"))?;
        let state_projection = linear(
            state_encoder.output_dim(),
            cfg.latent_dim,
            vb.pp("state_projection"),
        )?;
        let action_encoder =
            TemporalConvEncoder::new(&cfg.action_encoder, vb.pp("action_encoder"))?;
        let predictor = gru(
            action_encoder.output_dim(),
            cfg.latent_dim,
            GRUConfig::default(),
            vb.pp("predictor"),
        )?;
        Ok(Self {
            cfg,
            state_encoder,
            state_projection,
            action_encoder,
            predictor,
        })
    }

    pub fn config(&self) -> &SkyJepaConfig {
        &self.cfg
    }

    pub fn device(&self) -> &candle::Device {
        self.state_projection.weight().device()
    }

    pub fn encode_state_history(&self, states: &Tensor) -> Result<Tensor> {
        validate_history(states, self.cfg.history_steps, self.cfg.state_dim, "state")?;
        self.state_projection
            .forward(&self.state_encoder.forward_last(states)?)
    }

    pub fn encode_action_history(&self, actions: &Tensor) -> Result<Tensor> {
        validate_history(
            actions,
            self.cfg.history_steps,
            self.cfg.action_dim,
            "action",
        )?;
        self.action_encoder.forward_last(actions)
    }

    pub fn encode_state_windows(&self, windows: &Tensor) -> Result<Tensor> {
        let (batch, windows_count, history, state_dim) = windows.dims4()?;
        if history != self.cfg.history_steps || state_dim != self.cfg.state_dim {
            candle::bail!(
                "state windows expect [batch, windows, {}, {}], got {:?}",
                self.cfg.history_steps,
                self.cfg.state_dim,
                windows.shape()
            );
        }
        let flat = windows.reshape((batch * windows_count, history, state_dim))?;
        self.encode_state_history(&flat)?
            .reshape((batch, windows_count, self.cfg.latent_dim))
    }

    pub fn encode_action_windows(&self, windows: &Tensor) -> Result<Tensor> {
        let (batch, windows_count, history, action_dim) = windows.dims4()?;
        if history != self.cfg.history_steps || action_dim != self.cfg.action_dim {
            candle::bail!(
                "action windows expect [batch, windows, {}, {}], got {:?}",
                self.cfg.history_steps,
                self.cfg.action_dim,
                windows.shape()
            );
        }
        let flat = windows.reshape((batch * windows_count, history, action_dim))?;
        self.encode_action_history(&flat)?.reshape((
            batch,
            windows_count,
            self.action_encoder.output_dim(),
        ))
    }

    pub fn rollout_from_action_embeddings(
        &self,
        initial_latent: &Tensor,
        action_embeddings: &Tensor,
    ) -> Result<Tensor> {
        let (batch, latent_dim) = initial_latent.dims2()?;
        let (action_batch, rollout_steps, action_dim) = action_embeddings.dims3()?;
        if batch != action_batch || latent_dim != self.cfg.latent_dim {
            candle::bail!(
                "latent/action batch or latent dim mismatch: initial={:?} actions={:?}",
                initial_latent.shape(),
                action_embeddings.shape()
            );
        }
        if action_dim != self.action_encoder.output_dim() {
            candle::bail!(
                "action embedding dim {action_dim} does not match configured {}",
                self.action_encoder.output_dim()
            );
        }
        if rollout_steps == 0 {
            candle::bail!("SkyJEPA rollout requires at least one action embedding");
        }

        let mut state = GRUState {
            // CUDA GEMM requires a compact leading dimension. State encoder
            // projections can be views into a larger temporal allocation.
            h: initial_latent.contiguous()?,
        };
        let mut predictions = Vec::with_capacity(rollout_steps);
        for step in 0..rollout_steps {
            let action = action_embeddings.i((.., step, ..))?.contiguous()?;
            state = self.predictor.step(&action, &state)?;
            predictions.push(state.h().clone());
        }
        let refs = predictions.iter().collect::<Vec<_>>();
        Tensor::stack(&refs, 1)
    }

    pub fn rollout(
        &self,
        initial_state_history: &Tensor,
        action_windows: &Tensor,
    ) -> Result<Tensor> {
        let initial = self.encode_state_history(initial_state_history)?;
        let actions = self.encode_action_windows(action_windows)?;
        self.rollout_from_action_embeddings(&initial, &actions)
    }

    /// Rolls out all control candidates without a host-side loop over samples.
    ///
    /// `state_history` is `[B, H, Ds]`, `action_history` is `[B, H-1, Da]`,
    /// and `candidate_actions` is `[B, S, T, Da]`. The state history is
    /// encoded once per batch item, while every rolling action window is
    /// flattened into one batched TCN invocation. Only the short GRU horizon
    /// remains sequential.
    pub fn rollout_candidates(
        &self,
        state_history: &Tensor,
        action_history: &Tensor,
        candidate_actions: &Tensor,
    ) -> Result<Tensor> {
        validate_history(
            state_history,
            self.cfg.history_steps,
            self.cfg.state_dim,
            "state",
        )?;
        let (batch, action_history_steps, action_dim) = action_history.dims3()?;
        let (candidate_batch, samples, rollout_steps, candidate_action_dim) =
            candidate_actions.dims4()?;
        if action_history_steps + 1 != self.cfg.history_steps
            || action_dim != self.cfg.action_dim
            || candidate_batch != batch
            || candidate_action_dim != self.cfg.action_dim
        {
            candle::bail!(
                "invalid SkyJEPA candidate context shapes: states={:?} action_history={:?} candidates={:?}",
                state_history.shape(),
                action_history.shape(),
                candidate_actions.shape()
            );
        }
        if samples == 0 || rollout_steps == 0 {
            candle::bail!("SkyJEPA candidate rollout requires samples and steps");
        }

        let history = action_history.unsqueeze(1)?.broadcast_as((
            batch,
            samples,
            action_history_steps,
            action_dim,
        ))?;
        let actions = Tensor::cat(&[&history, candidate_actions], 2)?;
        let mut windows = Vec::with_capacity(rollout_steps);
        for step in 0..rollout_steps {
            windows.push(actions.narrow(2, step, self.cfg.history_steps)?);
        }
        let window_refs = windows.iter().collect::<Vec<_>>();
        let action_windows = Tensor::stack(&window_refs, 2)?;
        let embeddings = self
            .encode_action_history(&action_windows.reshape((
                batch * samples * rollout_steps,
                self.cfg.history_steps,
                action_dim,
            ))?)?
            .reshape((
                batch * samples,
                rollout_steps,
                self.action_encoder.output_dim(),
            ))?;
        let initial = self
            .encode_state_history(state_history)?
            .unsqueeze(1)?
            .broadcast_as((batch, samples, self.cfg.latent_dim))?
            .reshape((batch * samples, self.cfg.latent_dim))?;
        self.rollout_from_action_embeddings(&initial, &embeddings)?
            .reshape((batch, samples, rollout_steps, self.cfg.latent_dim))
    }
}

fn validate_history(
    values: &Tensor,
    expected_history: usize,
    expected_dim: usize,
    name: &str,
) -> Result<()> {
    let (_, history, dim) = values.dims3()?;
    if history != expected_history || dim != expected_dim {
        candle::bail!(
            "{name} history expects [batch, {expected_history}, {expected_dim}], got {:?}",
            values.shape()
        );
    }
    Ok(())
}
