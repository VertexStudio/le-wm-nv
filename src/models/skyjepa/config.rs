use anyhow::ensure;
use serde::{Deserialize, Serialize};

/// Paper-derived temporal encoder configuration.
///
/// SkyJEPA does not currently publish its TCN kernel, dilation, activation, or
/// residual-block details. We use a standard causal residual TCN: two GELU
/// convolutions per level, exponentially increasing dilation, and no dropout.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TemporalConvConfig {
    pub input_dim: usize,
    pub channels: Vec<usize>,
    pub kernel_size: usize,
}

impl TemporalConvConfig {
    pub fn validate(&self, name: &str) -> anyhow::Result<()> {
        ensure!(
            self.input_dim > 0,
            "{name}.input_dim must be greater than zero"
        );
        ensure!(
            !self.channels.is_empty(),
            "{name}.channels must not be empty"
        );
        ensure!(
            self.channels.iter().all(|value| *value > 0),
            "{name}.channels must all be greater than zero"
        );
        ensure!(
            self.kernel_size > 0,
            "{name}.kernel_size must be greater than zero"
        );
        Ok(())
    }

    pub fn output_dim(&self) -> usize {
        *self
            .channels
            .last()
            .expect("validated TCN always has an output channel")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkyJepaConfig {
    pub state_dim: usize,
    pub action_dim: usize,
    pub history_steps: usize,
    pub rollout_steps: usize,
    pub latent_dim: usize,
    pub state_encoder: TemporalConvConfig,
    pub action_encoder: TemporalConvConfig,
}

impl SkyJepaConfig {
    /// Configuration reported in SkyJEPA, with explicitly documented choices
    /// for architecture details omitted by the paper.
    pub fn paper_derived() -> Self {
        Self {
            state_dim: 18,
            action_dim: 4,
            history_steps: 10,
            rollout_steps: 20,
            latent_dim: 24,
            state_encoder: TemporalConvConfig {
                input_dim: 18,
                channels: vec![8, 8, 16],
                kernel_size: 3,
            },
            action_encoder: TemporalConvConfig {
                input_dim: 4,
                channels: vec![4, 4, 8],
                kernel_size: 3,
            },
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(self.state_dim > 0, "state_dim must be greater than zero");
        ensure!(self.action_dim > 0, "action_dim must be greater than zero");
        ensure!(
            self.history_steps >= 2,
            "history_steps must be at least two"
        );
        ensure!(
            self.rollout_steps > 0,
            "rollout_steps must be greater than zero"
        );
        ensure!(self.latent_dim > 0, "latent_dim must be greater than zero");
        self.state_encoder.validate("state_encoder")?;
        self.action_encoder.validate("action_encoder")?;
        ensure!(
            self.state_encoder.input_dim == self.state_dim,
            "state_encoder.input_dim {} must match state_dim {}",
            self.state_encoder.input_dim,
            self.state_dim
        );
        ensure!(
            self.action_encoder.input_dim == self.action_dim,
            "action_encoder.input_dim {} must match action_dim {}",
            self.action_encoder.input_dim,
            self.action_dim
        );
        Ok(())
    }
}

impl Default for SkyJepaConfig {
    fn default() -> Self {
        Self::paper_derived()
    }
}
