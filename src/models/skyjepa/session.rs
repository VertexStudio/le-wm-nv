use std::{collections::VecDeque, fs, path::Path, time::Instant};

use anyhow::{Context, ensure};
use candle::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

use super::{
    SkyJepaConfig, SkyJepaControlConfig, SkyJepaModel, SkyJepaMppiScorer, SkyJepaProber,
    SkyJepaProberConfig,
};
use crate::{
    checkpoint::var_builder_from_path,
    data::skyjepa::{
        SKYJEPA_ACTION_DIM, SKYJEPA_STATE_DIM, SkyJepaActionSpace, SkyJepaDatasetConfig,
        SkyJepaNormalization,
    },
    planner::{ActionBounds, MppiConfig, MppiPlanner},
    skyjepa_sim::{SkyJepaDomain, SkyJepaRotorState},
    skyjepa_task::skyjepa_geometric_action_prior,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SkyJepaSessionConfig {
    pub samples: usize,
    pub horizon: usize,
    pub planner_seed: u64,
}

impl SkyJepaSessionConfig {
    pub fn paper_derived() -> Self {
        let control = SkyJepaControlConfig::paper_derived();
        Self {
            samples: control.samples,
            horizon: control.horizon,
            planner_seed: 7,
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(self.samples > 0, "SkyJEPA session samples must be positive");
        ensure!(self.horizon > 0, "SkyJEPA session horizon must be positive");
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
        let model_cfg: SkyJepaConfig = read_json(&checkpoint_dir.join("model-config.json"))?;
        let prober_cfg: SkyJepaProberConfig =
            read_json(&checkpoint_dir.join("prober-config.json"))?;
        let dataset_cfg: SkyJepaDatasetConfig =
            read_json(&checkpoint_dir.join("dataset-config.json"))?;
        let normalization: SkyJepaNormalization =
            read_json(&checkpoint_dir.join("normalization.json"))?;
        ensure!(
            dataset_cfg.action_space == SkyJepaActionSpace::RotorForces,
            "SkyJEPA control requires a rotor-force checkpoint"
        );
        ensure!(
            dataset_cfg.history_steps == model_cfg.history_steps,
            "checkpoint model/dataset history dimensions disagree"
        );
        let model = SkyJepaModel::new(
            model_cfg.clone(),
            var_builder_from_path(
                &checkpoint_dir.join("latent.safetensors"),
                DType::F32,
                &device,
            )?,
        )?;
        let prober = SkyJepaProber::new(
            prober_cfg,
            var_builder_from_path(
                &checkpoint_dir.join("prober.safetensors"),
                DType::F32,
                &device,
            )?,
        )?;
        let nominal = SkyJepaDomain::default();
        let hover_action = [nominal.mass * nominal.gravity / 4.0; SKYJEPA_ACTION_DIM];
        let mut control_cfg = SkyJepaControlConfig::paper_derived();
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
        self.trim_scale = initial_action.iter().sum::<f32>() / nominal_collective;
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
        let prior_actions = skyjepa_geometric_action_prior(
            *self
                .state_history
                .back()
                .expect("SkyJEPA state history is initialized"),
            reference_states,
            self.control_cfg.dt,
            SkyJepaDomain::default(),
        )
        .into_iter()
        .map(|action| action.map(|force| force * self.trim_scale))
        .collect::<Vec<_>>();
        let prior_tensor = Tensor::from_vec(
            prior_actions.iter().flatten().copied().collect::<Vec<_>>(),
            (1, self.control_cfg.horizon, SKYJEPA_ACTION_DIM),
            &self.device,
        )?;
        self.planner.set_warm_start_sequence(prior_tensor.clone());
        let reference_states = Tensor::from_vec(
            reference_states
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
            (1, self.control_cfg.horizon, SKYJEPA_STATE_DIM),
            &self.device,
        )?;
        let reference_actions = prior_tensor;
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
        let result = self.planner.plan_device(&scorer)?;
        let predicted = if include_prediction {
            scorer
                .predict_candidates(&result.sequence.unsqueeze(1)?)?
                .squeeze(1)?
                .to_vec3::<f32>()?[0]
                .iter()
                .map(|state| {
                    state
                        .as_slice()
                        .try_into()
                        .expect("predicted state has eighteen values")
                })
                .collect()
        } else {
            Vec::new()
        };
        let actions = result.sequence.to_vec3::<f32>()?;
        let first_action = result.first_action.to_vec2::<f32>()?;
        let action: [f32; SKYJEPA_ACTION_DIM] = first_action[0]
            .as_slice()
            .try_into()
            .expect("planner action has four values");
        let prior_action = prior_actions[0];
        let best_candidate_score = result.scores.min_all()?.to_scalar::<f32>()?;
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
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> anyhow::Result<T> {
    serde_json::from_str(
        &fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))
}
