use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::data::skyjepa::SKYJEPA_STATE_DIM;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum SkyJepaReferenceKind {
    Hover,
    Circle,
    FigureEight,
}

impl SkyJepaReferenceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hover => "hover",
            Self::Circle => "circle",
            Self::FigureEight => "figure-eight",
        }
    }
}

impl FromStr for SkyJepaReferenceKind {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "hover" => Ok(Self::Hover),
            "circle" => Ok(Self::Circle),
            "figure-eight" | "figure8" => Ok(Self::FigureEight),
            _ => anyhow::bail!("reference must be hover, circle, or figure-eight"),
        }
    }
}

pub fn skyjepa_reference_horizon(
    kind: SkyJepaReferenceKind,
    time: f32,
    dt: f32,
    horizon: usize,
    radius: f32,
    period: f32,
) -> Vec<[f32; SKYJEPA_STATE_DIM]> {
    (1..=horizon)
        .map(|offset| skyjepa_reference_state(kind, time + offset as f32 * dt, radius, period))
        .collect()
}

pub fn skyjepa_reference_state(
    kind: SkyJepaReferenceKind,
    time: f32,
    radius: f32,
    period: f32,
) -> [f32; SKYJEPA_STATE_DIM] {
    let mut state = [0.0; SKYJEPA_STATE_DIM];
    state[2] = 1.0;
    state[6] = 1.0;
    state[10] = 1.0;
    state[14] = 1.0;
    let omega = 2.0 * std::f32::consts::PI / period;
    match kind {
        SkyJepaReferenceKind::Hover => {}
        SkyJepaReferenceKind::Circle => {
            let angle = omega * time;
            state[0] = radius * (angle.cos() - 1.0);
            state[1] = radius * angle.sin();
            state[3] = -radius * omega * angle.sin();
            state[4] = radius * omega * angle.cos();
        }
        SkyJepaReferenceKind::FigureEight => {
            let angle = omega * time;
            state[0] = radius * angle.sin();
            state[1] = radius * 0.5 * (2.0 * angle).sin();
            state[2] = 1.2 + 0.25 * (0.5 * angle).sin();
            state[3] = radius * omega * angle.cos();
            state[4] = radius * omega * (2.0 * angle).cos();
            state[5] = 0.125 * omega * (0.5 * angle).cos();
        }
    }
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_starts_at_hover_position() {
        let state = skyjepa_reference_state(SkyJepaReferenceKind::Circle, 0.0, 2.0, 8.0);
        assert_eq!(&state[0..3], &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn reference_horizon_has_requested_length() {
        assert_eq!(
            skyjepa_reference_horizon(SkyJepaReferenceKind::FigureEight, 0.0, 0.05, 15, 2.0, 8.0)
                .len(),
            15
        );
    }
}
