use anyhow::ensure;
use candle::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};

use crate::{
    data::{
        drone_racing::RunningStats,
        skyjepa::{SKYJEPA_ACTION_DIM, SKYJEPA_STATE_DIM, SkyJepaNormalization},
    },
    planner::{ActionBounds, CandidateScorer, MppiConfig},
};

use super::{SkyJepaModel, SkyJepaProber};

/// The four grouped state weights and four per-axis control weights reported
/// by SkyJEPA. Rotation cost is the squared Frobenius error between the two
/// 3x3 matrices because the paper reports grouped weights but omits the exact
/// attitude-error parameterization used by the controller.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SkyJepaTrackingCost {
    pub position: f32,
    pub velocity: f32,
    pub attitude: f32,
    pub angular_velocity: f32,
    pub action: Vec<f32>,
}

impl SkyJepaTrackingCost {
    pub fn paper_derived() -> Self {
        Self {
            position: 400.0,
            velocity: 40.0,
            attitude: 20.0,
            angular_velocity: 20.0,
            action: vec![0.01, 0.05, 0.05, 0.10],
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        for (name, value) in [
            ("position", self.position),
            ("velocity", self.velocity),
            ("attitude", self.attitude),
            ("angular_velocity", self.angular_velocity),
        ] {
            ensure!(
                value.is_finite() && value >= 0.0,
                "SkyJEPA {name} cost must be finite and non-negative"
            );
        }
        ensure!(
            self.action.len() == SKYJEPA_ACTION_DIM,
            "SkyJEPA action cost must contain {SKYJEPA_ACTION_DIM} values"
        );
        ensure!(
            self.action
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0),
            "SkyJEPA action costs must be finite and non-negative"
        );
        Ok(())
    }
}

impl Default for SkyJepaTrackingCost {
    fn default() -> Self {
        Self::paper_derived()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SkyJepaControlConfig {
    pub horizon: usize,
    pub samples: usize,
    pub iterations: usize,
    pub dt: f32,
    pub temperature: f32,
    /// Per-action Gaussian sampling scales reported as diagonal action noise.
    pub action_noise: Vec<f32>,
    pub cost: SkyJepaTrackingCost,
}

impl SkyJepaControlConfig {
    pub fn paper_derived() -> Self {
        Self {
            horizon: 15,
            samples: 512,
            iterations: 1,
            dt: 0.05,
            temperature: 1e-4,
            action_noise: vec![0.60, 0.15, 0.15, 0.05],
            cost: SkyJepaTrackingCost::paper_derived(),
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(self.horizon > 0, "SkyJEPA control horizon must be positive");
        ensure!(self.samples > 0, "SkyJEPA MPPI samples must be positive");
        ensure!(
            self.iterations > 0,
            "SkyJEPA MPPI iterations must be positive"
        );
        ensure!(
            self.dt.is_finite() && self.dt > 0.0,
            "SkyJEPA control dt must be finite and positive"
        );
        ensure!(
            self.temperature.is_finite() && self.temperature > 0.0,
            "SkyJEPA MPPI temperature must be finite and positive"
        );
        ensure!(
            self.action_noise.len() == SKYJEPA_ACTION_DIM,
            "SkyJEPA action noise must contain {SKYJEPA_ACTION_DIM} values"
        );
        ensure!(
            self.action_noise
                .iter()
                .all(|value| value.is_finite() && *value > 0.0),
            "SkyJEPA action noise values must be finite and positive"
        );
        self.cost.validate()
    }

    pub fn mppi_config(&self, action_bounds: ActionBounds) -> anyhow::Result<MppiConfig> {
        self.validate()?;
        let mut config = MppiConfig::new(self.horizon, self.samples, SKYJEPA_ACTION_DIM);
        config.iterations = self.iterations;
        config.action_bounds = action_bounds;
        config.noise_std_per_action = Some(self.action_noise.clone());
        config.temperature = self.temperature;
        Ok(config)
    }
}

impl Default for SkyJepaControlConfig {
    fn default() -> Self {
        Self::paper_derived()
    }
}

/// Fully device-resident SkyJEPA rollout and tracking-cost adapter for MPPI.
///
/// The context and references are normalized once at construction. Every call
/// evaluates `[batch, samples, horizon, action]` candidates as one batched
/// TCN/GRU/prober/integrator pipeline.
pub struct SkyJepaMppiScorer<'a> {
    model: &'a SkyJepaModel,
    prober: &'a SkyJepaProber,
    normalized_state_history: Tensor,
    normalized_action_history: Tensor,
    metric_initial_state: Tensor,
    reference_states: Tensor,
    reference_actions: Tensor,
    action_mean: Tensor,
    action_std: Tensor,
    action_cost: Tensor,
    dt: f32,
    cost: SkyJepaTrackingCost,
}

impl<'a> SkyJepaMppiScorer<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: &'a SkyJepaModel,
        prober: &'a SkyJepaProber,
        metric_state_history: &Tensor,
        metric_action_history: &Tensor,
        reference_states: Tensor,
        reference_actions: Tensor,
        normalization: &SkyJepaNormalization,
        dt: f32,
        cost: SkyJepaTrackingCost,
    ) -> Result<Self> {
        cost.validate()
            .map_err(|error| candle::Error::Msg(error.to_string()))?;
        if !dt.is_finite() || dt <= 0.0 {
            candle::bail!("SkyJEPA control dt must be finite and positive");
        }
        let cfg = model.config();
        let (batch, history, state_dim) = metric_state_history.dims3()?;
        let (action_batch, action_history, action_dim) = metric_action_history.dims3()?;
        let (reference_batch, horizon, reference_dim) = reference_states.dims3()?;
        let action_reference_dims = reference_actions.dims3()?;
        if history != cfg.history_steps
            || state_dim != SKYJEPA_STATE_DIM
            || action_batch != batch
            || action_history + 1 != history
            || action_dim != SKYJEPA_ACTION_DIM
            || reference_batch != batch
            || reference_dim != SKYJEPA_STATE_DIM
            || action_reference_dims != (batch, horizon, SKYJEPA_ACTION_DIM)
        {
            candle::bail!(
                "invalid SkyJEPA control shapes: states={:?} action_history={:?} reference_states={:?} reference_actions={:?}",
                metric_state_history.shape(),
                metric_action_history.shape(),
                reference_states.shape(),
                reference_actions.shape()
            );
        }
        validate_stats(&normalization.state, SKYJEPA_STATE_DIM, "state")?;
        validate_stats(&normalization.action, SKYJEPA_ACTION_DIM, "action")?;
        let device = model.device();
        let dtype = metric_state_history.dtype();
        if metric_state_history.device().location() != device.location()
            || metric_action_history.device().location() != device.location()
            || reference_states.device().location() != device.location()
            || reference_actions.device().location() != device.location()
        {
            candle::bail!("SkyJEPA control tensors and model must share one device");
        }
        let (state_mean, state_std) =
            stats_tensors(&normalization.state, SKYJEPA_STATE_DIM, dtype, device)?;
        let (action_mean, action_std) =
            stats_tensors(&normalization.action, SKYJEPA_ACTION_DIM, dtype, device)?;
        let normalized_state_history = metric_state_history
            .broadcast_sub(&state_mean)?
            .broadcast_div(&state_std)?;
        let normalized_action_history = metric_action_history
            .broadcast_sub(&action_mean)?
            .broadcast_div(&action_std)?;
        let metric_initial_state = metric_state_history.narrow(1, history - 1, 1)?.squeeze(1)?;
        let action_cost =
            Tensor::from_vec(cost.action.clone(), (1, 1, 1, SKYJEPA_ACTION_DIM), device)?
                .to_dtype(dtype)?;
        Ok(Self {
            model,
            prober,
            normalized_state_history,
            normalized_action_history,
            metric_initial_state,
            reference_states,
            reference_actions,
            action_mean,
            action_std,
            action_cost,
            dt,
            cost,
        })
    }

    pub fn horizon(&self) -> Result<usize> {
        self.reference_states.dim(1)
    }

    pub fn predict_candidates(&self, metric_action_candidates: &Tensor) -> Result<Tensor> {
        let (batch, samples, horizon, action_dim) = metric_action_candidates.dims4()?;
        if batch != self.metric_initial_state.dim(0)?
            || horizon != self.horizon()?
            || action_dim != SKYJEPA_ACTION_DIM
        {
            candle::bail!(
                "SkyJEPA candidate actions {:?} do not match batch={}, horizon={}, action_dim={SKYJEPA_ACTION_DIM}",
                metric_action_candidates.shape(),
                self.metric_initial_state.dim(0)?,
                self.horizon()?
            );
        }
        let action_mean = self.action_mean.unsqueeze(1)?;
        let action_std = self.action_std.unsqueeze(1)?;
        let normalized_candidates = metric_action_candidates
            .broadcast_sub(&action_mean)?
            .broadcast_div(&action_std)?;
        let latents = self.model.rollout_candidates(
            &self.normalized_state_history,
            &self.normalized_action_history,
            &normalized_candidates,
        )?;
        let initial = self
            .metric_initial_state
            .unsqueeze(1)?
            .broadcast_as((batch, samples, SKYJEPA_STATE_DIM))?
            .reshape((batch * samples, SKYJEPA_STATE_DIM))?;
        let actions =
            metric_action_candidates.reshape((batch * samples, horizon, SKYJEPA_ACTION_DIM))?;
        let transition_dt = Tensor::full(
            self.dt,
            (batch * samples, horizon),
            metric_action_candidates.device(),
        )?
        .to_dtype(metric_action_candidates.dtype())?;
        self.prober
            .predict_metric_rollout_inference(
                &initial,
                &actions,
                &transition_dt,
                &latents.reshape((batch * samples, horizon, self.model.config().latent_dim))?,
            )?
            .reshape((batch, samples, horizon, SKYJEPA_STATE_DIM))
    }
}

impl CandidateScorer for SkyJepaMppiScorer<'_> {
    fn device(&self) -> &Device {
        self.model.device()
    }

    fn dtype(&self) -> DType {
        self.normalized_state_history.dtype()
    }

    fn batch_size(&self) -> Option<usize> {
        self.normalized_state_history.dims().first().copied()
    }

    fn score_candidates(&self, action_candidates: &Tensor) -> Result<Tensor> {
        let predicted = self.predict_candidates(action_candidates)?;
        tracking_scores(
            &predicted,
            action_candidates,
            &self.reference_states,
            &self.reference_actions,
            &self.action_cost,
            &self.cost,
        )
    }
}

pub(crate) trait MetricCandidateScorer: CandidateScorer {
    fn predict_metric_candidates(&self, actions: &Tensor) -> Result<Tensor>;
}

impl MetricCandidateScorer for SkyJepaMppiScorer<'_> {
    fn predict_metric_candidates(&self, actions: &Tensor) -> Result<Tensor> {
        self.predict_candidates(actions)
    }
}

/// A strong, trim-calibrated nominal rigid-body comparator. It receives only
/// observable state, command-derived motor estimates and the same hover trim
/// as the learned controller; never the randomized plant's hidden parameters.
pub struct SkyJepaNominalScorer {
    initial: Tensor,
    motors: Tensor,
    reference_states: Tensor,
    reference_actions: Tensor,
    action_cost: Tensor,
    dt: f32,
    trim_scale: f32,
    cost: SkyJepaTrackingCost,
}

impl SkyJepaNominalScorer {
    pub fn new(
        initial: Tensor,
        motors: Tensor,
        reference_states: Tensor,
        reference_actions: Tensor,
        dt: f32,
        trim_scale: f32,
        cost: SkyJepaTrackingCost,
    ) -> Result<Self> {
        cost.validate()
            .map_err(|error| candle::Error::Msg(error.to_string()))?;
        if !trim_scale.is_finite()
            || trim_scale <= 0.0
            || initial.dims2()?.1 != 18
            || motors.dims() != [initial.dim(0)?, 4]
            || reference_states.dims3()?.2 != 18
            || reference_actions.dims() != [initial.dim(0)?, reference_states.dim(1)?, 4]
        {
            candle::bail!("invalid nominal scorer context");
        }
        let action_cost = Tensor::from_vec(cost.action.clone(), (1, 1, 1, 4), initial.device())?;
        Ok(Self {
            initial,
            motors,
            reference_states,
            reference_actions,
            action_cost,
            dt,
            trim_scale,
            cost,
        })
    }
}

impl MetricCandidateScorer for SkyJepaNominalScorer {
    fn predict_metric_candidates(&self, actions: &Tensor) -> Result<Tensor> {
        let (batch, samples, horizon, dim) = actions.dims4()?;
        if batch != self.initial.dim(0)? || horizon != self.reference_states.dim(1)? || dim != 4 {
            candle::bail!("nominal candidate shape mismatch");
        }
        let initial = self
            .initial
            .unsqueeze(1)?
            .broadcast_as((batch, samples, 18))?
            .reshape((batch * samples, 18))?;
        let motors = self
            .motors
            .unsqueeze(1)?
            .broadcast_as((batch, samples, 4))?
            .reshape((batch * samples, 4))?;
        let calibrated_actions =
            (actions / self.trim_scale as f64)?.reshape((batch * samples, horizon, 4))?;
        super::nominal_physics_rollout(
            &initial,
            &calibrated_actions,
            &motors,
            self.dt,
            10,
            crate::skyjepa_sim::SkyJepaDomain::default(),
        )?
        .reshape((batch, samples, horizon, 18))
    }
}

impl CandidateScorer for SkyJepaNominalScorer {
    fn device(&self) -> &Device {
        self.initial.device()
    }
    fn dtype(&self) -> DType {
        self.initial.dtype()
    }
    fn batch_size(&self) -> Option<usize> {
        Some(self.initial.dims()[0])
    }
    fn score_candidates(&self, actions: &Tensor) -> Result<Tensor> {
        tracking_scores(
            &self.predict_metric_candidates(actions)?,
            actions,
            &self.reference_states,
            &self.reference_actions,
            &self.action_cost,
            &self.cost,
        )
    }
}

fn tracking_scores(
    predicted: &Tensor,
    action_candidates: &Tensor,
    reference_states: &Tensor,
    reference_actions: &Tensor,
    action_cost: &Tensor,
    cost: &SkyJepaTrackingCost,
) -> Result<Tensor> {
    let reference = reference_states.unsqueeze(1)?;
    let position = grouped_squared_error(predicted, &reference, 0, 3)?;
    let velocity = grouped_squared_error(predicted, &reference, 3, 3)?;
    let attitude = grouped_squared_error(predicted, &reference, 6, 9)?;
    let angular_velocity = grouped_squared_error(predicted, &reference, 15, 3)?;
    let action_error = action_candidates
        .broadcast_sub(&reference_actions.unsqueeze(1)?)?
        .sqr()?
        .broadcast_mul(action_cost)?
        .sum(3)?;
    let state_cost = (position * cost.position as f64)? + (velocity * cost.velocity as f64)?;
    let state_cost = state_cost? + (attitude * cost.attitude as f64)?;
    let state_cost = state_cost? + (angular_velocity * cost.angular_velocity as f64)?;
    (state_cost? + action_error)?.mean(2)
}

fn validate_stats(stats: &RunningStats, dim: usize, name: &str) -> Result<()> {
    if stats.mean.len() != dim || stats.std.len() != dim {
        candle::bail!(
            "SkyJEPA {name} normalization must contain {dim} means/stds, got {}/{}",
            stats.mean.len(),
            stats.std.len()
        );
    }
    if stats
        .mean
        .iter()
        .chain(stats.std.iter())
        .any(|value| !value.is_finite())
        || stats.std.iter().any(|value| *value <= 0.0)
    {
        candle::bail!("SkyJEPA {name} normalization is non-finite or has non-positive std");
    }
    Ok(())
}

fn stats_tensors(
    stats: &RunningStats,
    dim: usize,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let mean = Tensor::from_vec(stats.mean.clone(), (1, 1, dim), device)?.to_dtype(dtype)?;
    let std = Tensor::from_vec(stats.std.clone(), (1, 1, dim), device)?.to_dtype(dtype)?;
    Ok((mean, std))
}

fn grouped_squared_error(
    predicted: &Tensor,
    reference: &Tensor,
    start: usize,
    len: usize,
) -> Result<Tensor> {
    predicted
        .narrow(3, start, len)?
        .broadcast_sub(&reference.narrow(3, start, len)?)?
        .sqr()?
        .sum(3)
}
