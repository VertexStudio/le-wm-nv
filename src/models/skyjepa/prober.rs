use candle::{Module, Result, Tensor};
use candle_nn::{Init, Linear, VarBuilder, linear};
use serde::{Deserialize, Serialize};

use super::{
    KinematicConfig, SkyJepaLatentRollout, SkyJepaModel, integrate_metric_rollout,
    integrate_metric_rollout_inference,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SkyJepaProberConfig {
    pub latent_dim: usize,
    pub hidden_dim: usize,
    pub hidden_layers: usize,
    pub kinematics: KinematicConfig,
}

impl SkyJepaProberConfig {
    pub fn paper_derived(latent_dim: usize) -> Self {
        Self {
            latent_dim,
            hidden_dim: 32,
            hidden_layers: 2,
            kinematics: KinematicConfig::default(),
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.latent_dim > 0, "latent_dim must be greater than zero");
        anyhow::ensure!(self.hidden_dim > 0, "hidden_dim must be greater than zero");
        anyhow::ensure!(
            self.hidden_layers > 0,
            "hidden_layers must be greater than zero"
        );
        self.kinematics.validate()
    }
}

#[derive(Debug, Clone)]
pub struct SkyJepaProber {
    cfg: SkyJepaProberConfig,
    input: Linear,
    hidden: Vec<Linear>,
    output: Linear,
}

#[derive(Debug)]
pub struct SkyJepaProberOutput {
    pub residual_acceleration: Tensor,
    pub angular_action_map: Tensor,
}

impl SkyJepaProber {
    pub fn new(cfg: SkyJepaProberConfig, vb: VarBuilder) -> Result<Self> {
        cfg.validate()
            .map_err(|error| candle::Error::Msg(error.to_string()))?;
        let input = linear(cfg.latent_dim, cfg.hidden_dim, vb.pp("input"))?;
        let hidden = (1..cfg.hidden_layers)
            .map(|idx| linear(cfg.hidden_dim, cfg.hidden_dim, vb.pp("hidden").pp(idx - 1)))
            .collect::<Result<Vec<_>>>()?;
        let output_vb = vb.pp("output");
        let weight = output_vb.get_with_hints(
            (15, cfg.hidden_dim),
            "weight",
            Init::Randn {
                mean: 0.0,
                stdev: 1e-3,
            },
        )?;
        let bias = output_vb.get_with_hints(15, "bias", Init::Const(0.0))?;
        let output = Linear::new(weight, Some(bias));
        Ok(Self {
            cfg,
            input,
            hidden,
            output,
        })
    }

    pub fn config(&self) -> &SkyJepaProberConfig {
        &self.cfg
    }

    pub fn forward(&self, predicted_latents: &Tensor) -> Result<SkyJepaProberOutput> {
        let (batch, steps, latent_dim) = predicted_latents.dims3()?;
        if latent_dim != self.cfg.latent_dim {
            candle::bail!(
                "prober latent dim {latent_dim} does not match configured {}",
                self.cfg.latent_dim
            );
        }
        let mut hidden = predicted_latents.reshape((batch * steps, latent_dim))?;
        hidden = self.input.forward(&hidden)?.gelu()?;
        for layer in &self.hidden {
            hidden = layer.forward(&hidden)?.gelu()?;
        }
        let output = self.output.forward(&hidden)?.reshape((batch, steps, 15))?;
        Ok(SkyJepaProberOutput {
            residual_acceleration: output.narrow(2, 0, 3)?,
            angular_action_map: output.narrow(2, 3, 12)?.reshape((batch, steps, 3, 4))?,
        })
    }

    pub fn predict_metric_rollout(
        &self,
        initial_state: &Tensor,
        metric_actions: &Tensor,
        transition_dt: &Tensor,
        predicted_latents: &Tensor,
    ) -> Result<Tensor> {
        let output = self.forward(predicted_latents)?;
        integrate_metric_rollout(
            initial_state,
            metric_actions,
            transition_dt,
            &output.residual_acceleration,
            &output.angular_action_map,
            self.cfg.kinematics,
        )
    }

    pub fn predict_metric_rollout_inference(
        &self,
        initial_state: &Tensor,
        metric_actions: &Tensor,
        transition_dt: &Tensor,
        predicted_latents: &Tensor,
    ) -> Result<Tensor> {
        let output = self.forward(predicted_latents)?;
        integrate_metric_rollout_inference(
            initial_state,
            metric_actions,
            transition_dt,
            &output.residual_acceleration,
            &output.angular_action_map,
            self.cfg.kinematics,
        )
    }
}

#[derive(Debug)]
pub struct SkyJepaProberLoss {
    pub total_loss: Tensor,
    pub position_loss: Tensor,
    pub velocity_loss: Tensor,
    pub attitude_loss: Tensor,
    pub angular_velocity_loss: Tensor,
    pub predicted_states: Tensor,
    pub target_states: Tensor,
}

#[derive(Debug, Clone, Serialize)]
pub struct SkyJepaProberLossScalars {
    pub total: f32,
    pub position: f32,
    pub velocity: f32,
    pub attitude: f32,
    pub angular_velocity: f32,
}

impl SkyJepaProberLossScalars {
    pub fn from_loss(loss: &SkyJepaProberLoss) -> Result<Self> {
        Ok(Self {
            total: loss.total_loss.to_scalar::<f32>()?,
            position: loss.position_loss.to_scalar::<f32>()?,
            velocity: loss.velocity_loss.to_scalar::<f32>()?,
            attitude: loss.attitude_loss.to_scalar::<f32>()?,
            angular_velocity: loss.angular_velocity_loss.to_scalar::<f32>()?,
        })
    }
}

pub fn skyjepa_prober_loss(
    model: &SkyJepaModel,
    prober: &SkyJepaProber,
    normalized_states: &Tensor,
    normalized_actions: &Tensor,
    metric_states: &Tensor,
    metric_actions: &Tensor,
    transition_dt: &Tensor,
) -> Result<SkyJepaProberLoss> {
    let cfg = model.config();
    let SkyJepaLatentRollout {
        predicted_latents, ..
    } = super::skyjepa_latent_rollout(model, normalized_states, normalized_actions)?;
    let predicted_latents = predicted_latents.detach();
    let initial_state = metric_states
        .narrow(1, cfg.history_steps - 1, 1)?
        .squeeze(1)?;
    let future_actions = metric_actions.narrow(1, cfg.history_steps - 1, cfg.rollout_steps)?;
    let future_dt = transition_dt.narrow(1, cfg.history_steps - 1, cfg.rollout_steps)?;
    let target_states = metric_states.narrow(1, cfg.history_steps, cfg.rollout_steps)?;
    let predicted_states = prober.predict_metric_rollout(
        &initial_state,
        &future_actions,
        &future_dt,
        &predicted_latents,
    )?;

    let position_loss = component_mse(&predicted_states, &target_states, 0, 3)?;
    let velocity_loss = component_mse(&predicted_states, &target_states, 3, 3)?;
    let attitude_loss = component_mse(&predicted_states, &target_states, 6, 9)?;
    let angular_velocity_loss = component_mse(&predicted_states, &target_states, 15, 3)?;
    let total_loss = (&predicted_states - &target_states)?.sqr()?.mean_all()?;
    Ok(SkyJepaProberLoss {
        total_loss,
        position_loss,
        velocity_loss,
        attitude_loss,
        angular_velocity_loss,
        predicted_states,
        target_states,
    })
}

fn component_mse(lhs: &Tensor, rhs: &Tensor, start: usize, len: usize) -> Result<Tensor> {
    (lhs.narrow(2, start, len)? - rhs.narrow(2, start, len)?)?
        .sqr()?
        .mean_all()
}
