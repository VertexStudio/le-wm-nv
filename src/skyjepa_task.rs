use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::{
    data::{
        drone_racing::{mat3_mul_vec3, mat3_t_mul_vec3},
        skyjepa::{SKYJEPA_ACTION_DIM, SKYJEPA_STATE_DIM},
    },
    skyjepa_sim::SkyJepaDomain,
};

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
    let omega = 2.0 * std::f32::consts::PI / period;
    let acceleration = match kind {
        SkyJepaReferenceKind::Hover => [0.0; 3],
        SkyJepaReferenceKind::Circle => {
            let angle = omega * time;
            [
                -radius * omega * omega * angle.cos(),
                -radius * omega * omega * angle.sin(),
                0.0,
            ]
        }
        SkyJepaReferenceKind::FigureEight => {
            let angle = omega * time;
            [
                -radius * omega * omega * angle.sin(),
                -2.0 * radius * omega * omega * (2.0 * angle).sin(),
                -0.0625 * omega * omega * (0.5 * angle).sin(),
            ]
        }
    };
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
    let rotation = desired_rotation(normalize3(
        [acceleration[0], acceleration[1], acceleration[2] + 9.81],
        [0.0, 0.0, 1.0],
    ));
    state[6..15].copy_from_slice(&rotation);
    state
}

/// Builds a dynamically feasible rotor-force prior for SkyJEPA MPPI.
///
/// The first action closes the loop around the measured state; later actions
/// provide differential-flatness feed-forward along the reference. SkyJEPA
/// still scores and updates sampled actions around this sequence, but the
/// optimizer always retains a safe, physically meaningful candidate.
pub fn skyjepa_geometric_action_prior(
    current_state: [f32; SKYJEPA_STATE_DIM],
    references: &[[f32; SKYJEPA_STATE_DIM]],
    dt: f32,
    domain: SkyJepaDomain,
) -> Vec<[f32; SKYJEPA_ACTION_DIM]> {
    let current_position: [f32; 3] = current_state[0..3].try_into().expect("position dimension");
    let current_velocity: [f32; 3] = current_state[3..6].try_into().expect("velocity dimension");
    let current_rotation: [f32; 9] = current_state[6..15].try_into().expect("rotation dimension");
    let current_omega: [f32; 3] = current_state[15..18].try_into().expect("omega dimension");
    references
        .iter()
        .enumerate()
        .map(|(index, reference)| {
            let acceleration = reference_acceleration(references, index, dt);
            let feedback_scale = if index == 0 { 1.0 } else { 0.0 };
            let desired_acceleration = [0, 1, 2].map(|axis| {
                acceleration[axis]
                    + feedback_scale
                        * (3.5 * (reference[axis] - current_position[axis])
                            + 2.6 * (reference[3 + axis] - current_velocity[axis]))
                    + if axis == 2 { domain.gravity } else { 0.0 }
            });
            let desired_up = normalize3(desired_acceleration, [0.0, 0.0, 1.0]);
            let heading = normalize3([reference[6], reference[9], 0.0], [1.0, 0.0, 0.0]);
            let desired = desired_rotation_with_heading(desired_up, heading);
            let (rotation, omega) = if index == 0 {
                (current_rotation, current_omega)
            } else {
                (
                    reference[6..15].try_into().expect("rotation dimension"),
                    reference[15..18].try_into().expect("omega dimension"),
                )
            };
            let error = attitude_error_body(desired, rotation);
            let torque = [0, 1, 2]
                .map(|axis| -domain.inertia[axis] * (25.0 * error[axis] + 8.0 * omega[axis]));
            let current_up = mat3_mul_vec3(rotation, [0.0, 0.0, 1.0]);
            let total_force = (domain.mass * dot3(desired_acceleration, current_up).max(0.0)
                / domain.thrust_scale)
                .clamp(0.0, domain.mass * domain.gravity * domain.max_thrust_weight);
            allocate_rotor_forces(total_force, torque, domain)
        })
        .collect()
}

fn reference_acceleration(
    references: &[[f32; SKYJEPA_STATE_DIM]],
    index: usize,
    dt: f32,
) -> [f32; 3] {
    let (before, after) = if index + 1 < references.len() {
        (&references[index], &references[index + 1])
    } else if index > 0 {
        (&references[index - 1], &references[index])
    } else {
        return [0.0; 3];
    };
    [0, 1, 2].map(|axis| (after[3 + axis] - before[3 + axis]) / dt)
}

fn allocate_rotor_forces(
    total_force: f32,
    torque: [f32; 3],
    domain: SkyJepaDomain,
) -> [f32; SKYJEPA_ACTION_DIM] {
    let arm = domain.arm_length * domain.thrust_scale;
    let yaw_lever = 0.025 * domain.torque_scale * domain.thrust_scale;
    let base = total_force / 4.0;
    let mut forces = [
        base - torque[1] / (2.0 * arm) + torque[2] / (4.0 * yaw_lever),
        base + torque[0] / (2.0 * arm) - torque[2] / (4.0 * yaw_lever),
        base + torque[1] / (2.0 * arm) + torque[2] / (4.0 * yaw_lever),
        base - torque[0] / (2.0 * arm) - torque[2] / (4.0 * yaw_lever),
    ];
    let maximum = domain.mass * domain.gravity * domain.max_thrust_weight / 4.0;
    forces
        .iter_mut()
        .for_each(|force| *force = force.clamp(0.0, maximum));
    forces
}

fn desired_rotation(body_up: [f32; 3]) -> [f32; 9] {
    let heading = if body_up[0].abs() < 0.95 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let body_y = normalize3(cross3(body_up, heading), [0.0, 1.0, 0.0]);
    let body_x = normalize3(cross3(body_y, body_up), heading);
    [
        body_x[0], body_y[0], body_up[0], body_x[1], body_y[1], body_up[1], body_x[2], body_y[2],
        body_up[2],
    ]
}

fn desired_rotation_with_heading(body_up: [f32; 3], mut heading: [f32; 3]) -> [f32; 9] {
    if dot3(body_up, heading).abs() > 0.95 {
        heading = [-heading[1], heading[0], 0.0];
    }
    let body_y = normalize3(cross3(body_up, heading), [0.0, 1.0, 0.0]);
    let body_x = normalize3(cross3(body_y, body_up), heading);
    [
        body_x[0], body_y[0], body_up[0], body_x[1], body_y[1], body_up[1], body_x[2], body_y[2],
        body_up[2],
    ]
}

fn attitude_error_body(desired: [f32; 9], current: [f32; 9]) -> [f32; 3] {
    let desired_x_current = [0, 1, 2].map(|column| {
        mat3_t_mul_vec3(
            desired,
            [current[column], current[3 + column], current[6 + column]],
        )
    });
    let product = [
        desired_x_current[0][0],
        desired_x_current[1][0],
        desired_x_current[2][0],
        desired_x_current[0][1],
        desired_x_current[1][1],
        desired_x_current[2][1],
        desired_x_current[0][2],
        desired_x_current[1][2],
        desired_x_current[2][2],
    ];
    [
        0.5 * (product[7] - product[5]),
        0.5 * (product[2] - product[6]),
        0.5 * (product[3] - product[1]),
    ]
}

fn dot3(lhs: [f32; 3], rhs: [f32; 3]) -> f32 {
    lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
}

fn cross3(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    ]
}

fn normalize3(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let norm = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
    if norm > 1e-6 {
        value.map(|component| component / norm)
    } else {
        fallback
    }
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

    #[test]
    fn circle_reference_tilts_into_centripetal_acceleration() {
        let state = skyjepa_reference_state(SkyJepaReferenceKind::Circle, 0.0, 2.0, 8.0);
        assert!(state[8] < 0.0);
        assert!(state[14] < 1.0);
    }

    #[test]
    fn geometric_hover_prior_balances_nominal_rotors() {
        let state = skyjepa_reference_state(SkyJepaReferenceKind::Hover, 0.0, 2.0, 8.0);
        let references = vec![state; 15];
        let actions =
            skyjepa_geometric_action_prior(state, &references, 0.05, SkyJepaDomain::default());
        let hover = 1.3 * 9.81 / 4.0;
        for action in actions {
            for force in action {
                assert!((force - hover).abs() < 1e-5);
            }
        }
    }
}
