use std::{collections::VecDeque, path::Path, time::Instant};

use anyhow::{Context, ensure};
use candle::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use serde::{Deserialize, Serialize};

use super::{
    SkyJepaConfig, SkyJepaControlConfig, SkyJepaModel, SkyJepaMppiScorer, SkyJepaNominalScorer,
    SkyJepaProber, checkpoint::SkyJepaCheckpoint, control::MetricCandidateScorer,
};
use crate::{
    checkpoint::var_builder_from_path,
    data::skyjepa::{
        SKYJEPA_ACTION_DIM, SKYJEPA_STATE_DIM, SkyJepaActionSpace, SkyJepaNormalization,
    },
    planner::{ActionBounds, MppiConfig, MppiPlanner, PlanDeviceResult},
    skyjepa_sim::{SkyJepaDomain, SkyJepaRotorState},
    skyjepa_task::skyjepa_geometric_action_prior,
};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum SkyJepaWarmStart {
    #[default]
    FreshPrior,
    ShiftedResidual,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum SkyJepaDynamics {
    #[default]
    Trained,
    Untrained,
    NominalPhysics,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SkyJepaSessionConfig {
    pub samples: usize,
    pub horizon: usize,
    pub planner_seed: u64,
    pub warm_start: SkyJepaWarmStart,
    pub residual_limit_n: f32,
    pub dynamics: SkyJepaDynamics,
    pub ablation_seed: u64,
    pub trim_multiplier: f32,
}

impl SkyJepaSessionConfig {
    pub fn paper_derived() -> Self {
        let control = SkyJepaControlConfig::paper_derived();
        Self {
            samples: control.samples,
            horizon: control.horizon,
            planner_seed: 7,
            warm_start: SkyJepaWarmStart::FreshPrior,
            residual_limit_n: 2.0,
            dynamics: SkyJepaDynamics::Trained,
            ablation_seed: 7,
            trim_multiplier: 1.0,
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(self.samples > 0, "SkyJEPA session samples must be positive");
        ensure!(self.horizon > 0, "SkyJEPA session horizon must be positive");
        ensure!(
            self.residual_limit_n.is_finite() && self.residual_limit_n > 0.0,
            "warm-start residual limit must be finite and positive"
        );
        ensure!(
            self.trim_multiplier.is_finite() && self.trim_multiplier > 0.0,
            "trim multiplier must be finite and positive"
        );
        Ok(())
    }
}

impl Default for SkyJepaSessionConfig {
    fn default() -> Self {
        Self::paper_derived()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SkyJepaControllerPlan {
    pub action: [f32; SKYJEPA_ACTION_DIM],
    pub prior_action: [f32; SKYJEPA_ACTION_DIM],
    pub action_correction: [f32; SKYJEPA_ACTION_DIM],
    pub action_sequence: Vec<[f32; SKYJEPA_ACTION_DIM]>,
    pub predicted_states: Vec<[f32; SKYJEPA_STATE_DIM]>,
    pub best_candidate_score: f32,
    pub plan_ms: f64,
    pub deadline_reached: bool,
}

/// Reusable checkpoint-backed SkyJEPA+MPPI control session.
///
/// Headless benchmarks and graphical simulators both use this type so history
/// alignment, normalization, candidate scoring, and warm starts cannot drift.
pub struct SkyJepaControllerSession {
    model: SkyJepaModel,
    prober: SkyJepaProber,
    device: Device,
    model_cfg: SkyJepaConfig,
    normalization: SkyJepaNormalization,
    control_cfg: SkyJepaControlConfig,
    planner_cfg: MppiConfig,
    planner: MppiPlanner,
    state_history: VecDeque<[f32; SKYJEPA_STATE_DIM]>,
    action_history: VecDeque<[f32; SKYJEPA_ACTION_DIM]>,
    hover_action: [f32; SKYJEPA_ACTION_DIM],
    trim_scale: f32,
    warm_start: SkyJepaWarmStart,
    residual_limit_n: f32,
    active_residual: Option<Tensor>,
    pending_residual: Option<Tensor>,
    dynamics: SkyJepaDynamics,
    trim_multiplier: f32,
    nominal_motor_forces: [f32; 4],
}

impl SkyJepaControllerSession {
    pub fn load(
        checkpoint_dir: impl AsRef<Path>,
        device: Device,
        cfg: SkyJepaSessionConfig,
        initial_state: SkyJepaRotorState,
    ) -> anyhow::Result<Self> {
        cfg.validate()?;
        let checkpoint_dir = checkpoint_dir.as_ref();
        let checkpoint = SkyJepaCheckpoint::load(checkpoint_dir)?;
        let model_cfg = checkpoint.contract.model.clone();
        let prober_cfg = checkpoint
            .contract
            .prober
            .clone()
            .context("control requires a trained prober")?;
        let dataset_cfg = checkpoint.contract.dataset;
        let normalization = checkpoint.contract.normalization.clone();
        ensure!(
            dataset_cfg.action_space == SkyJepaActionSpace::RotorForces,
            "SkyJEPA control requires a rotor-force checkpoint"
        );
        ensure!(
            dataset_cfg.history_steps == model_cfg.history_steps,
            "checkpoint model/dataset history dimensions disagree"
        );
        let (model, prober) = if cfg.dynamics == SkyJepaDynamics::Untrained {
            device.set_seed(cfg.ablation_seed)?;
            let latent_vars = VarMap::new();
            let _ = SkyJepaModel::new(
                model_cfg.clone(),
                VarBuilder::from_varmap(&latent_vars, DType::F32, &device),
            )?;
            device.set_seed(cfg.ablation_seed ^ 0x5052_4f42_4552_5f53)?;
            let prober_vars = VarMap::new();
            let _ = SkyJepaProber::new(
                prober_cfg.clone(),
                VarBuilder::from_varmap(&prober_vars, DType::F32, &device),
            )?;
            // Freeze random weights too: the ablation must not pay for a
            // training graph that the loaded, trained model does not build.
            let frozen = |vars: &VarMap| {
                VarBuilder::from_tensors(
                    vars.data()
                        .lock()
                        .unwrap()
                        .iter()
                        .map(|(name, var)| (name.clone(), var.as_tensor().detach()))
                        .collect(),
                    DType::F32,
                    &device,
                )
            };
            (
                SkyJepaModel::new(model_cfg.clone(), frozen(&latent_vars))?,
                SkyJepaProber::new(prober_cfg, frozen(&prober_vars))?,
            )
        } else {
            let model = SkyJepaModel::new(
                model_cfg.clone(),
                var_builder_from_path(
                    &checkpoint.latent_path(checkpoint_dir),
                    DType::F32,
                    &device,
                )?,
            )?;
            let prober = SkyJepaProber::new(
                prober_cfg,
                var_builder_from_path(
                    &checkpoint.prober_path(checkpoint_dir)?,
                    DType::F32,
                    &device,
                )?,
            )?;
            (model, prober)
        };
        let nominal = SkyJepaDomain::default();
        let hover_action = [nominal.mass * nominal.gravity / 4.0; SKYJEPA_ACTION_DIM];
        let mut control_cfg = SkyJepaControlConfig::paper_derived();
        control_cfg.dt = 1.0 / dataset_cfg.model_rate_hz as f32;
        control_cfg.samples = cfg.samples;
        control_cfg.horizon = cfg.horizon;
        let max_rotor_force = nominal.mass * nominal.gravity * nominal.max_thrust_weight / 4.0;
        let bounds = ActionBounds {
            low: vec![0.0; SKYJEPA_ACTION_DIM],
            high: vec![max_rotor_force; SKYJEPA_ACTION_DIM],
        };
        let mut planner_cfg = control_cfg.mppi_config(bounds)?;
        planner_cfg.seed = Some(cfg.planner_seed);
        planner_cfg.deadline_action = Some(hover_action.to_vec());
        let mut session = Self {
            model,
            prober,
            device,
            model_cfg,
            normalization,
            control_cfg,
            planner: MppiPlanner::new(planner_cfg.clone()),
            planner_cfg,
            state_history: VecDeque::new(),
            action_history: VecDeque::new(),
            hover_action,
            trim_scale: 1.0,
            warm_start: cfg.warm_start,
            residual_limit_n: cfg.residual_limit_n,
            active_residual: None,
            pending_residual: None,
            dynamics: cfg.dynamics,
            trim_multiplier: cfg.trim_multiplier,
            nominal_motor_forces: hover_action,
        };
        session.reset(initial_state)?;
        Ok(session)
    }

    pub fn reset(&mut self, initial_state: SkyJepaRotorState) -> anyhow::Result<()> {
        self.reset_with_action(initial_state, self.hover_action)
    }

    /// Resets control history with the actuator command that held the vehicle
    /// before model-based control took over. This trim is observable on a real
    /// flight stack and lets the prior start correctly across payload/thrust
    /// changes while SkyJEPA infers residual dynamics from history.
    pub fn reset_with_action(
        &mut self,
        initial_state: SkyJepaRotorState,
        initial_action: [f32; SKYJEPA_ACTION_DIM],
    ) -> anyhow::Result<()> {
        ensure!(
            initial_action
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0),
            "SkyJEPA initial action must be finite and non-negative"
        );
        let nominal_collective = self.hover_action.iter().sum::<f32>();
        ensure!(
            initial_action.iter().sum::<f32>() > 0.0,
            "initial trim must have positive collective force"
        );
        let trim_scale =
            initial_action.iter().sum::<f32>() / nominal_collective * self.trim_multiplier;
        ensure!(
            trim_scale.is_finite() && trim_scale > 0.0,
            "invalid calibrated trim"
        );
        self.trim_scale = trim_scale;
        self.nominal_motor_forces = initial_action.map(|force| force / trim_scale);
        self.active_residual = None;
        self.pending_residual = None;
        self.state_history = VecDeque::from(vec![
            initial_state.as_state18();
            self.model_cfg.history_steps
        ]);
        self.action_history =
            VecDeque::from(vec![initial_action; self.model_cfg.history_steps - 1]);
        self.planner = MppiPlanner::new(self.planner_cfg.clone());
        self.planner.set_warm_start_sequence(
            Tensor::from_vec(
                initial_action.repeat(self.control_cfg.horizon),
                (1, self.control_cfg.horizon, SKYJEPA_ACTION_DIM),
                &self.device,
            )?
            .to_dtype(DType::F32)?,
        );
        Ok(())
    }

    /// Adds the state reached after executing `action` for one controller step.
    pub fn commit_observation(
        &mut self,
        state: SkyJepaRotorState,
        action: [f32; SKYJEPA_ACTION_DIM],
    ) {
        // Shift only after an action was actually executed, not on every call
        // to plan (e.g. visualization or a repeated plan at the same state).
        self.active_residual = self.pending_residual.take();
        let response =
            1.0 - (-self.control_cfg.dt / SkyJepaDomain::default().motor_time_constant).exp();
        for (motor, command) in self.nominal_motor_forces.iter_mut().zip(action) {
            let target =
                (command / self.trim_scale).clamp(0.0, self.planner_cfg.action_bounds.high[0]);
            *motor += (target - *motor) * response;
        }
        self.state_history.pop_front();
        self.state_history.push_back(state.as_state18());
        self.action_history.pop_front();
        self.action_history.push_back(action);
    }

    pub fn plan(
        &mut self,
        reference_states: &[[f32; SKYJEPA_STATE_DIM]],
    ) -> anyhow::Result<SkyJepaControllerPlan> {
        self.plan_internal(reference_states, false)
    }

    pub fn plan_with_prediction(
        &mut self,
        reference_states: &[[f32; SKYJEPA_STATE_DIM]],
    ) -> anyhow::Result<SkyJepaControllerPlan> {
        self.plan_internal(reference_states, true)
    }

    fn plan_internal(
        &mut self,
        reference_states: &[[f32; SKYJEPA_STATE_DIM]],
        include_prediction: bool,
    ) -> anyhow::Result<SkyJepaControllerPlan> {
        let started = Instant::now();
        self.pending_residual = None;
        ensure!(
            reference_states.len() == self.control_cfg.horizon,
            "SkyJEPA session expected {} reference states, got {}",
            self.control_cfg.horizon,
            reference_states.len()
        );
        let state_tensor = Tensor::from_vec(
            self.state_history
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
            (1, self.model_cfg.history_steps, SKYJEPA_STATE_DIM),
            &self.device,
        )?;
        let action_tensor = Tensor::from_vec(
            self.action_history
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
            (1, self.model_cfg.history_steps - 1, SKYJEPA_ACTION_DIM),
            &self.device,
        )?;
        let prior_actions = self.prior_action_sequence(reference_states);
        let prior_tensor = Tensor::from_vec(
            prior_actions.iter().flatten().copied().collect::<Vec<_>>(),
            (1, self.control_cfg.horizon, SKYJEPA_ACTION_DIM),
            &self.device,
        )?;
        let warm = match (self.warm_start, &self.active_residual) {
            (SkyJepaWarmStart::ShiftedResidual, Some(residual)) => (&prior_tensor + residual)?,
            _ => prior_tensor.clone(),
        };
        self.planner
            .set_warm_start_sequence(warm.clamp(0.0, self.planner_cfg.action_bounds.high[0])?);
        let reference_states = Tensor::from_vec(
            reference_states
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
            (1, self.control_cfg.horizon, SKYJEPA_STATE_DIM),
            &self.device,
        )?;
        let reference_actions = prior_tensor.clone();
        let (result, predicted) = if self.dynamics == SkyJepaDynamics::NominalPhysics {
            let initial = state_tensor
                .narrow(1, self.model_cfg.history_steps - 1, 1)?
                .squeeze(1)?;
            let motors =
                Tensor::from_vec(self.nominal_motor_forces.to_vec(), (1, 4), &self.device)?;
            let scorer = SkyJepaNominalScorer::new(
                initial,
                motors,
                reference_states,
                reference_actions,
                self.control_cfg.dt,
                self.trim_scale,
                self.control_cfg.cost.clone(),
            )?;
            plan_metric_candidates(&mut self.planner, &scorer, include_prediction)?
        } else {
            let scorer = SkyJepaMppiScorer::new(
                &self.model,
                &self.prober,
                &state_tensor,
                &action_tensor,
                reference_states,
                reference_actions,
                &self.normalization,
                self.control_cfg.dt,
                self.control_cfg.cost.clone(),
            )?;
            plan_metric_candidates(&mut self.planner, &scorer, include_prediction)?
        };
        let actions = result.sequence.to_vec3::<f32>()?;
        let first_action = result.first_action.to_vec2::<f32>()?;
        let action: [f32; SKYJEPA_ACTION_DIM] = first_action[0]
            .as_slice()
            .try_into()
            .expect("planner action has four values");
        let prior_action = prior_actions[0];
        let best_candidate_score = result.scores.min_all()?.to_scalar::<f32>()?;
        ensure!(
            action.iter().all(|value| value.is_finite()) && best_candidate_score.is_finite(),
            "SkyJEPA planner produced a non-finite action or score"
        );
        self.pending_residual = if self.warm_start == SkyJepaWarmStart::ShiftedResidual {
            Some(shifted_residual(
                &result.sequence,
                &prior_tensor,
                self.residual_limit_n,
            )?)
        } else {
            None
        };
        Ok(SkyJepaControllerPlan {
            action,
            prior_action,
            action_correction: [0, 1, 2, 3].map(|index| action[index] - prior_action[index]),
            action_sequence: actions[0]
                .iter()
                .map(|action| {
                    action
                        .as_slice()
                        .try_into()
                        .expect("planner action has four values")
                })
                .collect(),
            predicted_states: predicted,
            best_candidate_score,
            plan_ms: started.elapsed().as_secs_f64() * 1e3,
            deadline_reached: result.deadline_reached,
        })
    }

    /// Compiles/initializes CUDA kernels, then restores pristine controller
    /// history and MPPI warm start so the first flight cycle has no cold hitch.
    pub fn warm_up(
        &mut self,
        initial_state: SkyJepaRotorState,
        reference_states: &[[f32; SKYJEPA_STATE_DIM]],
    ) -> anyhow::Result<f64> {
        let cold_ms = self.plan(reference_states)?.plan_ms;
        self.reset(initial_state)?;
        Ok(cold_ms)
    }

    pub fn dt(&self) -> f32 {
        self.control_cfg.dt
    }

    pub fn horizon(&self) -> usize {
        self.control_cfg.horizon
    }

    pub fn samples(&self) -> usize {
        self.control_cfg.samples
    }

    pub fn hover_action(&self) -> [f32; SKYJEPA_ACTION_DIM] {
        self.hover_action
    }

    pub fn trim_scale(&self) -> f32 {
        self.trim_scale
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn control_config(&self) -> &SkyJepaControlConfig {
        &self.control_cfg
    }

    pub fn action_bounds(&self) -> &ActionBounds {
        &self.planner_cfg.action_bounds
    }

    /// Identical trim calibration and actuator bounds for every comparator.
    pub fn prior_action_sequence(&self, reference: &[[f32; 18]]) -> Vec<[f32; 4]> {
        skyjepa_geometric_action_prior(
            *self.state_history.back().expect("initialized history"),
            reference,
            self.control_cfg.dt,
            SkyJepaDomain::default(),
        )
        .into_iter()
        .map(|action| {
            action.map(|force| {
                (force * self.trim_scale).clamp(0.0, self.planner_cfg.action_bounds.high[0])
            })
        })
        .collect()
    }
}

fn plan_metric_candidates<S: MetricCandidateScorer>(
    planner: &mut MppiPlanner,
    scorer: &S,
    include_prediction: bool,
) -> candle::Result<(PlanDeviceResult, Vec<[f32; 18]>)> {
    let result = planner.plan_device(scorer)?;
    let predicted = if include_prediction {
        scorer
            .predict_metric_candidates(&result.sequence.unsqueeze(1)?)?
            .squeeze(1)?
            .to_vec3::<f32>()?[0]
            .iter()
            .map(|state| state.as_slice().try_into().expect("state18 rollout"))
            .collect()
    } else {
        Vec::new()
    };
    Ok((result, predicted))
}

fn shifted_residual(sequence: &Tensor, prior: &Tensor, limit: f32) -> candle::Result<Tensor> {
    let (batch, horizon, actions) = sequence.dims3()?;
    let residual = (sequence - prior)?.clamp(-limit, limit)?;
    let tail = Tensor::zeros((batch, 1, actions), sequence.dtype(), sequence.device())?;
    if horizon == 1 {
        return Ok(tail);
    }
    Tensor::cat(&[&residual.narrow(1, 1, horizon - 1)?, &tail], 1)
}

#[cfg(test)]
mod warm_start_tests {
    use super::*;
    #[test]
    fn shifted_residual_is_bounded_relative_to_new_prior_and_has_zero_tail() -> candle::Result<()> {
        let prior = Tensor::new(&[[[3f32], [4.], [5.]]], &Device::Cpu)?;
        let selected = Tensor::new(&[[[3f32], [7.], [1.]]], &Device::Cpu)?;
        let residual = shifted_residual(&selected, &prior, 2.0)?;
        assert_eq!(
            residual.to_vec3::<f32>()?,
            vec![vec![vec![2.], vec![-2.], vec![0.]]]
        );
        let new_prior = Tensor::new(&[[[6f32], [6.], [6.]]], &Device::Cpu)?;
        assert_eq!(
            (&new_prior + &residual)?.to_vec3::<f32>()?,
            vec![vec![vec![8.], vec![4.], vec![6.]]]
        );
        Ok(())
    }
}
