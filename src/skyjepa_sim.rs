use serde::{Deserialize, Serialize};

use crate::data::drone_racing::{
    cross3, mat3_from_rotvec, mat3_mul, mat3_mul_vec3, mat3_t_mul_vec3,
};

pub const SKYJEPA_ROTORS: usize = 4;

/// Explicit experiment populations; the shifted population is not training data.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum SkyJepaDomainDistribution {
    #[default]
    TrainingRanges,
    ExtendedMassAndMotorLag,
}

impl SkyJepaDomainDistribution {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TrainingRanges => "training_ranges",
            Self::ExtendedMassAndMotorLag => "extended_mass_and_motor_lag",
        }
    }
}

/// One physically coherent member of the domain-randomized quadrotor family.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct SkyJepaDomain {
    pub mass: f32,
    pub inertia: [f32; 3],
    pub motor_time_constant: f32,
    pub drag: [f32; 3],
    pub thrust_scale: f32,
    pub torque_scale: f32,
    pub arm_length: f32,
    pub gravity: f32,
    pub max_thrust_weight: f32,
}

impl Default for SkyJepaDomain {
    fn default() -> Self {
        Self {
            mass: 1.3,
            inertia: [0.021, 0.023, 0.040],
            motor_time_constant: 0.04,
            drag: [0.25, 0.25, 0.30],
            thrust_scale: 1.0,
            torque_scale: 1.0,
            arm_length: 0.17,
            gravity: 9.81,
            max_thrust_weight: 4.0,
        }
    }
}

impl SkyJepaDomain {
    /// Paper ranges: mass ±50%, inertia ±30%, motor lag 0.01–0.1 s,
    /// drag 0.1–0.5, and thrust/torque coefficients ±50%. The paper does
    /// not report an arm-length range, so this clean-room simulator uses ±20%.
    pub fn sample(seed: u64) -> Self {
        Self::sample_with_distribution(seed, SkyJepaDomainDistribution::TrainingRanges)
    }

    pub fn sample_with_distribution(seed: u64, distribution: SkyJepaDomainDistribution) -> Self {
        let nominal = Self::default();
        let mut rng = SplitMix64::new(seed);
        let (mass_range, lag_range) = match distribution {
            SkyJepaDomainDistribution::TrainingRanges => ((0.5, 1.5), (0.01, 0.1)),
            SkyJepaDomainDistribution::ExtendedMassAndMotorLag => ((1.55, 1.75), (0.11, 0.14)),
        };
        Self {
            mass: nominal.mass * rng.range(mass_range.0, mass_range.1),
            inertia: nominal.inertia.map(|value| value * rng.range(0.7, 1.3)),
            motor_time_constant: rng.range(lag_range.0, lag_range.1),
            drag: [
                rng.range(0.1, 0.5),
                rng.range(0.1, 0.5),
                rng.range(0.1, 0.5),
            ],
            thrust_scale: rng.range(0.5, 1.5),
            torque_scale: rng.range(0.5, 1.5),
            arm_length: nominal.arm_length * rng.range(0.8, 1.2),
            ..nominal
        }
    }

    pub fn validate(self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.mass.is_finite() && self.mass > 0.0,
            "mass must be positive"
        );
        anyhow::ensure!(
            self.inertia
                .iter()
                .all(|value| value.is_finite() && *value > 0.0),
            "inertia must be positive"
        );
        anyhow::ensure!(
            self.motor_time_constant.is_finite() && self.motor_time_constant > 0.0,
            "motor time constant must be positive"
        );
        anyhow::ensure!(
            self.drag
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0),
            "drag must be non-negative"
        );
        anyhow::ensure!(
            self.thrust_scale.is_finite() && self.thrust_scale > 0.0,
            "thrust scale must be positive"
        );
        anyhow::ensure!(
            self.torque_scale.is_finite() && self.torque_scale > 0.0,
            "torque scale must be positive"
        );
        anyhow::ensure!(
            self.arm_length.is_finite() && self.arm_length > 0.0,
            "arm length must be positive"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct SkyJepaRotorState {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub rotation_world_from_body: [f32; 9],
    pub angular_velocity: [f32; 3],
}

impl SkyJepaRotorState {
    pub fn hover() -> Self {
        Self {
            position: [0.0, 0.0, 1.0],
            velocity: [0.0; 3],
            rotation_world_from_body: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            angular_velocity: [0.0; 3],
        }
    }

    pub fn as_state18(self) -> [f32; 18] {
        let mut state = [0.0; 18];
        state[0..3].copy_from_slice(&self.position);
        state[3..6].copy_from_slice(&self.velocity);
        state[6..15].copy_from_slice(&self.rotation_world_from_body);
        state[15..18].copy_from_slice(&self.angular_velocity);
        state
    }
}

impl Default for SkyJepaRotorState {
    fn default() -> Self {
        Self::hover()
    }
}

/// Rigid-body rotor-force plant with body-axis drag and first-order motors.
#[derive(Debug, Clone)]
pub struct SkyJepaRotorPlant {
    domain: SkyJepaDomain,
    state: SkyJepaRotorState,
    motor_forces: [f32; SKYJEPA_ROTORS],
}

impl SkyJepaRotorPlant {
    pub fn new(domain: SkyJepaDomain, state: SkyJepaRotorState) -> anyhow::Result<Self> {
        domain.validate()?;
        let hover_force = domain.mass * domain.gravity / (4.0 * domain.thrust_scale);
        Ok(Self {
            domain,
            state,
            motor_forces: [hover_force; SKYJEPA_ROTORS],
        })
    }

    pub fn state(&self) -> SkyJepaRotorState {
        self.state
    }

    pub fn domain(&self) -> SkyJepaDomain {
        self.domain
    }

    pub fn motor_forces(&self) -> [f32; SKYJEPA_ROTORS] {
        self.motor_forces
    }

    /// Restore an estimated motor state without advancing the rigid body.
    pub fn with_motor_forces(mut self, forces: [f32; SKYJEPA_ROTORS]) -> anyhow::Result<Self> {
        anyhow::ensure!(
            forces
                .iter()
                .all(|force| force.is_finite() && *force >= 0.0),
            "motor forces must be finite and nonnegative"
        );
        self.motor_forces = forces;
        Ok(self)
    }

    pub fn nominal_hover_action(&self) -> [f32; SKYJEPA_ROTORS] {
        [self.domain.mass * self.domain.gravity / (4.0 * self.domain.thrust_scale); 4]
    }

    pub fn step(&mut self, commanded_forces: [f32; SKYJEPA_ROTORS], dt: f32) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let max_force =
            self.domain.mass * self.domain.gravity * self.domain.max_thrust_weight / 4.0;
        let response = 1.0 - (-dt / self.domain.motor_time_constant).exp();
        for (rotor, commanded_force) in commanded_forces.iter().enumerate() {
            let target = commanded_force.clamp(0.0, max_force);
            self.motor_forces[rotor] += (target - self.motor_forces[rotor]) * response;
        }
        let forces = self
            .motor_forces
            .map(|force| force * self.domain.thrust_scale);
        let total_thrust = forces.iter().sum::<f32>();
        let arm = self.domain.arm_length;
        let yaw_lever = 0.025 * self.domain.torque_scale;
        let torque = [
            arm * (forces[1] - forces[3]),
            arm * (forces[2] - forces[0]),
            yaw_lever * (forces[0] - forces[1] + forces[2] - forces[3]),
        ];
        let inertia_omega = [
            self.domain.inertia[0] * self.state.angular_velocity[0],
            self.domain.inertia[1] * self.state.angular_velocity[1],
            self.domain.inertia[2] * self.state.angular_velocity[2],
        ];
        let gyro = cross3(self.state.angular_velocity, inertia_omega);
        for axis in 0..3 {
            let angular_acceleration =
                (torque[axis] - gyro[axis] - 0.015 * self.state.angular_velocity[axis])
                    / self.domain.inertia[axis];
            self.state.angular_velocity[axis] += angular_acceleration * dt;
        }
        let delta_rotation = mat3_from_rotvec(self.state.angular_velocity.map(|value| value * dt));
        self.state.rotation_world_from_body = orthonormalize(mat3_mul(
            self.state.rotation_world_from_body,
            delta_rotation,
        ));

        let body_velocity =
            mat3_t_mul_vec3(self.state.rotation_world_from_body, self.state.velocity);
        let body_drag = [
            self.domain.drag[0] * body_velocity[0],
            self.domain.drag[1] * body_velocity[1],
            self.domain.drag[2] * body_velocity[2],
        ];
        let drag_world = mat3_mul_vec3(self.state.rotation_world_from_body, body_drag);
        let body_up = mat3_mul_vec3(self.state.rotation_world_from_body, [0.0, 0.0, 1.0]);
        let acceleration = [
            body_up[0] * total_thrust / self.domain.mass - drag_world[0] / self.domain.mass,
            body_up[1] * total_thrust / self.domain.mass - drag_world[1] / self.domain.mass,
            body_up[2] * total_thrust / self.domain.mass
                - self.domain.gravity
                - drag_world[2] / self.domain.mass,
        ];
        for (axis, acceleration) in acceleration.iter().enumerate() {
            self.state.velocity[axis] += acceleration * dt;
            self.state.position[axis] += self.state.velocity[axis] * dt;
        }
        if self.state.position[2] < 0.05 {
            self.state.position[2] = 0.05;
            self.state.velocity[2] = self.state.velocity[2].max(0.0);
        }
    }
}

fn orthonormalize(matrix: [f32; 9]) -> [f32; 9] {
    let x = normalize([matrix[0], matrix[3], matrix[6]], [1.0, 0.0, 0.0]);
    let y0 = [matrix[1], matrix[4], matrix[7]];
    let dot = x[0] * y0[0] + x[1] * y0[1] + x[2] * y0[2];
    let y = normalize(
        [y0[0] - dot * x[0], y0[1] - dot * x[1], y0[2] - dot * x[2]],
        [0.0, 1.0, 0.0],
    );
    let z = cross3(x, y);
    [x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]]
}

fn normalize(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let norm = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
    if norm > 1e-6 {
        [value[0] / norm, value[1] / norm, value[2] / norm]
    } else {
        fallback
    }
}

#[cfg(test)]
mod distribution_tests {
    use super::*;

    #[test]
    fn deliberate_shift_is_outside_training_mass_and_lag_only() {
        for seed in 0..1000 {
            let standard = SkyJepaDomain::sample(seed);
            let shifted = SkyJepaDomain::sample_with_distribution(
                seed,
                SkyJepaDomainDistribution::ExtendedMassAndMotorLag,
            );
            assert!((0.65..=1.95).contains(&standard.mass));
            assert!((0.01..=0.1).contains(&standard.motor_time_constant));
            assert!((1.3 * 1.55..=1.3 * 1.75).contains(&shifted.mass));
            assert!((0.11..=0.14).contains(&shifted.motor_time_constant));
            assert_eq!(
                SkyJepaDomain {
                    mass: standard.mass,
                    motor_time_constant: standard.motor_time_constant,
                    ..shifted
                },
                standard
            );
            shifted.validate().unwrap();
        }
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(state: u64) -> Self {
        Self { state }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 + 0.5) / ((1u32 << 24) as f32)
    }

    fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.unit()
    }
}
