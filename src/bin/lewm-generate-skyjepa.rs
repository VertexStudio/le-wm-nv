use std::{fs, path::PathBuf, time::Instant};

use anyhow::{Context, ensure};
use clap::Parser;
use hdf5::File;
use le_wm_nv::{
    data::{
        drone_racing::{cross3, mat3_from_rotvec, mat3_mul, mat3_mul_vec3},
        skyjepa::{
            SKYJEPA_ACTION_DIM, SKYJEPA_SCHEMA_VERSION, SKYJEPA_STATE_DIM, SkyJepaActionSpace,
            SkyJepaDatasetMetadata,
        },
    },
    skyjepa_sim::{SkyJepaDomain, SkyJepaDomainDistribution, SkyJepaRotorPlant, SkyJepaRotorState},
};
use rayon::prelude::*;
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Generate canonical domain-randomized rotor-force data for SkyJEPA")]
struct Args {
    #[arg(long)]
    output_dir: PathBuf,

    #[arg(long, default_value_t = 500)]
    domains: usize,

    #[arg(long, value_enum, default_value_t = SkyJepaDomainDistribution::TrainingRanges)]
    domain_distribution: SkyJepaDomainDistribution,

    #[arg(long, default_value_t = 20_000)]
    trajectories: usize,

    #[arg(long, default_value_t = 10.0)]
    duration_seconds: f32,

    #[arg(long, default_value_t = 20)]
    sample_rate_hz: usize,

    #[arg(long, default_value_t = 200)]
    simulation_rate_hz: usize,

    /// Zero uses Rayon's detected worker count.
    #[arg(long, default_value_t = 0)]
    workers: usize,

    #[arg(long, default_value_t = 7)]
    seed: u64,

    #[arg(long)]
    overwrite: bool,
}

#[derive(Debug)]
struct Episode {
    states: Vec<f32>,
    actions: Vec<f32>,
    reference_states: Vec<f32>,
    motor_forces: Vec<f32>,
    episode_idx: Vec<i64>,
    step_idx: Vec<i64>,
    dt: Vec<f32>,
    domain_idx: Vec<i64>,
}

#[derive(Debug, Serialize)]
struct DomainRecord {
    index: usize,
    seed: u64,
    parameters: SkyJepaDomain,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let started = Instant::now();
    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("failed to create {}", args.output_dir.display()))?;
    let data_path = args.output_dir.join("data.h5");
    let metadata_path = args.output_dir.join("metadata.json");
    let domains_path = args.output_dir.join("domains.json");
    for path in [&data_path, &metadata_path, &domains_path] {
        if path.exists() && !args.overwrite {
            anyhow::bail!(
                "{} already exists; pass --overwrite to replace this generated artifact",
                path.display()
            );
        }
    }

    let domains = (0..args.domains)
        .map(|index| {
            let seed = mix_seed(args.seed, index as u64);
            DomainRecord {
                index,
                seed,
                parameters: SkyJepaDomain::sample_with_distribution(seed, args.domain_distribution),
            }
        })
        .collect::<Vec<_>>();
    let generate = || {
        (0..args.trajectories)
            .into_par_iter()
            .map(|episode| {
                generate_episode(episode, domains[episode % domains.len()].parameters, &args)
            })
            .collect::<anyhow::Result<Vec<_>>>()
    };
    let episodes = if args.workers == 0 {
        generate()?
    } else {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.workers)
            .build()
            .context("failed to build dataset worker pool")?
            .install(generate)?
    };
    let rows = episodes
        .iter()
        .map(|episode| episode.episode_idx.len())
        .sum();
    let mut states = Vec::with_capacity(rows * SKYJEPA_STATE_DIM);
    let mut actions = Vec::with_capacity(rows * SKYJEPA_ACTION_DIM);
    let mut reference_states = Vec::with_capacity(rows * SKYJEPA_STATE_DIM);
    let mut motor_forces = Vec::with_capacity(rows * SKYJEPA_ACTION_DIM);
    let mut episode_idx = Vec::with_capacity(rows);
    let mut step_idx = Vec::with_capacity(rows);
    let mut dt = Vec::with_capacity(rows);
    let mut domain_idx = Vec::with_capacity(rows);
    for episode in episodes {
        states.extend(episode.states);
        actions.extend(episode.actions);
        reference_states.extend(episode.reference_states);
        motor_forces.extend(episode.motor_forces);
        episode_idx.extend(episode.episode_idx);
        step_idx.extend(episode.step_idx);
        dt.extend(episode.dt);
        domain_idx.extend(episode.domain_idx);
    }

    if data_path.exists() {
        fs::remove_file(&data_path)
            .with_context(|| format!("failed to replace {}", data_path.display()))?;
    }
    let file = File::create(&data_path)
        .with_context(|| format!("failed to create {}", data_path.display()))?;
    write_f32(&file, "state", rows, SKYJEPA_STATE_DIM, &states)?;
    write_f32(&file, "action", rows, SKYJEPA_ACTION_DIM, &actions)?;
    write_f32(
        &file,
        "reference_state",
        rows,
        SKYJEPA_STATE_DIM,
        &reference_states,
    )?;
    write_f32(
        &file,
        "motor_force",
        rows,
        SKYJEPA_ACTION_DIM,
        &motor_forces,
    )?;
    write_i64(&file, "episode_idx", rows, &episode_idx)?;
    write_i64(&file, "step_idx", rows, &step_idx)?;
    write_f32(&file, "dt", rows, 1, &dt)?;
    write_i64(&file, "domain_idx", rows, &domain_idx)?;
    let metadata = SkyJepaDatasetMetadata {
        schema_version: SKYJEPA_SCHEMA_VERSION,
        data_h5: PathBuf::from("data.h5"),
        sample_rate_hz: args.sample_rate_hz,
        rows,
        episodes: args.trajectories,
        state_dim: SKYJEPA_STATE_DIM,
        action_dim: SKYJEPA_ACTION_DIM,
        action_space: SkyJepaActionSpace::RotorForces,
        generator: Some(
            "lewm-generate-skyjepa v2 balanced-domain RFF+geometric-PD simulator".into(),
        ),
        seed: Some(args.seed),
        domains: Some(args.domains),
        domain_distribution: Some(args.domain_distribution.as_str().into()),
        has_reference_state: true,
        has_motor_force: true,
    };
    fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?)?;
    fs::write(&domains_path, serde_json::to_string_pretty(&domains)?)?;
    println!(
        "generated SkyJEPA dataset={} trajectories={} domains={} rows={} elapsed_sec={:.3}",
        args.output_dir.display(),
        args.trajectories,
        args.domains,
        rows,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    ensure!(args.domains > 0, "domains must be positive");
    ensure!(
        args.trajectories / args.domains >= 2,
        "coverage requires at least two trajectories per domain"
    );
    ensure!(
        args.trajectories >= 3,
        "trajectories must be at least three"
    );
    ensure!(
        args.duration_seconds.is_finite() && args.duration_seconds > 0.0,
        "duration_seconds must be positive"
    );
    ensure!(args.sample_rate_hz > 0, "sample_rate_hz must be positive");
    ensure!(
        args.simulation_rate_hz >= args.sample_rate_hz
            && args.simulation_rate_hz.is_multiple_of(args.sample_rate_hz),
        "simulation_rate_hz must be an integer multiple of sample_rate_hz"
    );
    ensure!(
        (args.duration_seconds * args.sample_rate_hz as f32).round() as usize >= 30,
        "trajectory must contain at least 30 transitions for H=10, T=20"
    );
    Ok(())
}

fn generate_episode(episode: usize, domain: SkyJepaDomain, args: &Args) -> anyhow::Result<Episode> {
    let sample_dt = 1.0 / args.sample_rate_hz as f32;
    let sim_dt = 1.0 / args.simulation_rate_hz as f32;
    let substeps = args.simulation_rate_hz / args.sample_rate_hz;
    let transitions = (args.duration_seconds * args.sample_rate_hz as f32).round() as usize;
    let samples = transitions + 1;
    let trajectory = ReferenceTrajectory::sample(
        mix_seed(args.seed ^ 0x0053_4b59_4a45_5041, episode as u64),
        is_hover_episode(episode, args.domains, args.trajectories, args.seed),
    );
    let initial_reference = trajectory.at(0.0);
    let initial_acceleration = [
        initial_reference.acceleration[0],
        initial_reference.acceleration[1],
        initial_reference.acceleration[2] + domain.gravity,
    ];
    let mut initial_rng =
        SplitMix64::new(mix_seed(args.seed ^ 0x5245_434f_5645_5259, episode as u64));
    let desired_initial_rotation = desired_rotation(
        normalize3(initial_acceleration, [0.0, 0.0, 1.0]),
        initial_reference.yaw,
    );
    let initial_state = SkyJepaRotorState {
        position: [0, 1, 2]
            .map(|axis| initial_reference.position[axis] + initial_rng.range(-0.20, 0.20)),
        velocity: [0, 1, 2]
            .map(|axis| initial_reference.velocity[axis] + initial_rng.range(-0.25, 0.25)),
        rotation_world_from_body: mat3_mul(
            desired_initial_rotation,
            mat3_from_rotvec([
                initial_rng.range(-0.12, 0.12),
                initial_rng.range(-0.12, 0.12),
                initial_rng.range(-0.18, 0.18),
            ]),
        ),
        angular_velocity: [0, 1, 2].map(|_| initial_rng.range(-0.20, 0.20)),
    };
    let mut plant = SkyJepaRotorPlant::new(domain, initial_state)?;
    let mut result = Episode {
        states: Vec::with_capacity(samples * SKYJEPA_STATE_DIM),
        actions: Vec::with_capacity(samples * SKYJEPA_ACTION_DIM),
        reference_states: Vec::with_capacity(samples * SKYJEPA_STATE_DIM),
        motor_forces: Vec::with_capacity(samples * SKYJEPA_ACTION_DIM),
        episode_idx: vec![episode as i64; samples],
        step_idx: (0..samples).map(|step| step as i64).collect(),
        dt: vec![sample_dt; samples],
        domain_idx: vec![(episode % args.domains) as i64; samples],
    };
    let mut excitation = [0.0f32; 4];
    for step in 0..samples {
        let time = step as f32 * sample_dt;
        let reference = trajectory.at(time);
        let common_excitation = 0.04 * (time * 1.07 + episode as f32 * 0.11).sin();
        for (rotor, value) in excitation.iter_mut().enumerate() {
            let phase = episode as f32 * 0.17 + rotor as f32 * 1.31;
            let target = common_excitation
                + 0.035 * (time * (1.3 + 0.37 * rotor as f32) + phase).sin()
                + 0.020 * (time * (2.7 + 0.23 * rotor as f32) - phase).sin();
            *value = 0.88 * *value + 0.12 * target;
        }
        let action = tracking_action(plant.state(), reference, domain, excitation);
        result.states.extend_from_slice(&plant.state().as_state18());
        result.actions.extend_from_slice(&action);
        result
            .reference_states
            .extend_from_slice(&reference.as_state18(domain.gravity));
        result.motor_forces.extend_from_slice(&plant.motor_forces());
        if step < transitions {
            for _ in 0..substeps {
                plant.step(action, sim_dt);
            }
        }
    }
    ensure!(
        result
            .states
            .iter()
            .chain(result.actions.iter())
            .all(|value| value.is_finite()),
        "episode {episode} became non-finite"
    );
    Ok(result)
}

/// A domain's flight type changes on each visit. Every block of ten visits
/// includes one hover and nine moving references, independent of domain count.
/// Smaller artifacts reserve one hover per domain within their available visits.
fn is_hover_episode(episode: usize, domains: usize, trajectories: usize, seed: u64) -> bool {
    let domain = episode % domains;
    let visit = episode / domains;
    let period = (trajectories / domains).clamp(2, 10);
    let offset = (mix_seed(seed ^ 0x484f_5645_525f_434f, domain as u64) % period as u64) as usize;
    (visit + offset).is_multiple_of(period)
}

#[cfg(test)]
mod coverage_tests {
    use super::*;
    #[test]
    fn every_domain_gets_hover_and_motion_even_when_domain_count_is_a_multiple_of_ten() {
        for domains in [3, 10, 100, 500] {
            for visits in [2, 5, 20] {
                for domain in 0..domains {
                    let hover = (0..visits)
                        .filter(|visit| {
                            is_hover_episode(visit * domains + domain, domains, visits * domains, 7)
                        })
                        .count();
                    assert!(hover > 0 && hover < visits);
                    if visits == 20 {
                        assert_eq!(hover, 2);
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ReferencePoint {
    position: [f32; 3],
    velocity: [f32; 3],
    acceleration: [f32; 3],
    yaw: f32,
}

impl ReferencePoint {
    fn as_state18(self, gravity: f32) -> [f32; SKYJEPA_STATE_DIM] {
        let mut state = [0.0; SKYJEPA_STATE_DIM];
        state[0..3].copy_from_slice(&self.position);
        state[3..6].copy_from_slice(&self.velocity);
        let rotation = desired_rotation(
            normalize3(
                [
                    self.acceleration[0],
                    self.acceleration[1],
                    self.acceleration[2] + gravity,
                ],
                [0.0, 0.0, 1.0],
            ),
            self.yaw,
        );
        state[6..15].copy_from_slice(&rotation);
        state
    }
}

struct ReferenceTrajectory {
    center: [f32; 3],
    amplitude: [[f32; 3]; 3],
    omega: [[f32; 3]; 3],
    phase: [[f32; 3]; 3],
    yaw_center: f32,
    yaw_amplitude: f32,
    yaw_omega: f32,
    yaw_phase: f32,
}

impl ReferenceTrajectory {
    fn sample(seed: u64, hover: bool) -> Self {
        let mut rng = SplitMix64::new(seed);
        let periods_seconds = [[3.5, 5.5, 8.0], [4.0, 6.5, 9.0], [4.5, 7.0, 10.0]];
        let mut amplitude = [[0.0; 3]; 3];
        let mut omega = [[0.0; 3]; 3];
        let mut phase = [[0.0; 3]; 3];
        for axis in 0..3 {
            for component in 0..3 {
                let scale = if hover {
                    0.0
                } else if axis == 2 {
                    0.15
                } else {
                    0.45
                };
                amplitude[axis][component] = rng.range(0.35, 1.0) * scale;
                let seconds = periods_seconds[axis][component];
                omega[axis][component] = 2.0 * std::f32::consts::PI / seconds;
                phase[axis][component] = rng.range(0.0, 2.0 * std::f32::consts::PI);
            }
        }
        Self {
            center: [
                rng.range(-2.0, 2.0),
                rng.range(-2.0, 2.0),
                rng.range(0.9, 3.0),
            ],
            amplitude,
            omega,
            phase,
            yaw_center: rng.range(-std::f32::consts::PI, std::f32::consts::PI),
            yaw_amplitude: if hover { 0.0 } else { rng.range(0.25, 0.75) },
            yaw_omega: 2.0 * std::f32::consts::PI / rng.range(5.0, 9.0),
            yaw_phase: rng.range(0.0, 2.0 * std::f32::consts::PI),
        }
    }

    fn at(&self, time: f32) -> ReferencePoint {
        let mut position = self.center;
        let mut velocity = [0.0; 3];
        let mut acceleration = [0.0; 3];
        for axis in 0..3 {
            for component in 0..3 {
                let angle = self.omega[axis][component] * time + self.phase[axis][component];
                position[axis] += self.amplitude[axis][component] * angle.sin();
                velocity[axis] +=
                    self.amplitude[axis][component] * self.omega[axis][component] * angle.cos();
                acceleration[axis] -= self.amplitude[axis][component]
                    * self.omega[axis][component].powi(2)
                    * angle.sin();
            }
        }
        position[2] = position[2].max(0.6);
        ReferencePoint {
            position,
            velocity,
            acceleration,
            yaw: self.yaw_center
                + self.yaw_amplitude * (self.yaw_omega * time + self.yaw_phase).sin(),
        }
    }
}

fn tracking_action(
    state: SkyJepaRotorState,
    reference: ReferencePoint,
    domain: SkyJepaDomain,
    excitation: [f32; 4],
) -> [f32; 4] {
    let mut desired_acceleration = reference.acceleration;
    for (axis, value) in desired_acceleration.iter_mut().enumerate() {
        *value += 2.8 * (reference.position[axis] - state.position[axis])
            + 2.2 * (reference.velocity[axis] - state.velocity[axis]);
    }
    desired_acceleration[2] += domain.gravity;
    let acceleration_norm = norm(desired_acceleration).max(1e-4);
    let desired_up = desired_acceleration.map(|value| value / acceleration_norm);
    let current_up = mat3_mul_vec3(state.rotation_world_from_body, [0.0, 0.0, 1.0]);
    let attitude_error = attitude_error_body(
        desired_rotation(desired_up, reference.yaw),
        state.rotation_world_from_body,
    );
    let torque = [0, 1, 2].map(|axis| {
        -domain.inertia[axis] * (25.0 * attitude_error[axis] + 8.0 * state.angular_velocity[axis])
    });
    let thrust_acceleration = dot3(desired_acceleration, current_up).max(0.0);
    let total_force = (domain.mass * thrust_acceleration / domain.thrust_scale)
        .clamp(0.0, domain.mass * domain.gravity * domain.max_thrust_weight);
    let arm = domain.arm_length * domain.thrust_scale;
    let yaw_lever = 0.025 * domain.torque_scale * domain.thrust_scale;
    let base = total_force / 4.0;
    let mut forces = [
        base - torque[1] / (2.0 * arm) + torque[2] / (4.0 * yaw_lever),
        base + torque[0] / (2.0 * arm) - torque[2] / (4.0 * yaw_lever),
        base + torque[1] / (2.0 * arm) + torque[2] / (4.0 * yaw_lever),
        base - torque[0] / (2.0 * arm) - torque[2] / (4.0 * yaw_lever),
    ];
    let max_force = domain.mass * domain.gravity * domain.max_thrust_weight / 4.0;
    for rotor in 0..4 {
        forces[rotor] = (forces[rotor] * (1.0 + excitation[rotor])).clamp(0.0, max_force);
    }
    forces
}

fn norm(value: [f32; 3]) -> f32 {
    (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt()
}

fn normalize3(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let length = norm(value);
    if length > 1e-6 {
        value.map(|component| component / length)
    } else {
        fallback
    }
}

fn dot3(lhs: [f32; 3], rhs: [f32; 3]) -> f32 {
    lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
}

fn desired_rotation(body_up: [f32; 3], yaw: f32) -> [f32; 9] {
    let mut heading = [yaw.cos(), yaw.sin(), 0.0];
    if dot3(body_up, heading).abs() > 0.95 {
        heading = [-yaw.sin(), yaw.cos(), 0.0];
    }
    let body_y = normalize3(cross3(body_up, heading), [0.0, 1.0, 0.0]);
    let body_x = normalize3(cross3(body_y, body_up), heading);
    [
        body_x[0], body_y[0], body_up[0], body_x[1], body_y[1], body_up[1], body_x[2], body_y[2],
        body_up[2],
    ]
}

/// SO(3) error vee(0.5 * (R_d^T R - R^T R_d)) in current body axes.
fn attitude_error_body(desired: [f32; 9], current: [f32; 9]) -> [f32; 3] {
    let mut product = [0.0f32; 9];
    for row in 0..3 {
        for col in 0..3 {
            product[row * 3 + col] = (0..3)
                .map(|axis| desired[axis * 3 + row] * current[axis * 3 + col])
                .sum();
        }
    }
    [
        0.5 * (product[7] - product[5]),
        0.5 * (product[2] - product[6]),
        0.5 * (product[3] - product[1]),
    ]
}

fn write_f32(
    file: &File,
    name: &str,
    rows: usize,
    dim: usize,
    values: &[f32],
) -> anyhow::Result<()> {
    ensure!(values.len() == rows * dim, "invalid {name} value count");
    file.new_dataset::<f32>()
        .shape((rows, dim))
        .create(name)?
        .write_raw(values)?;
    Ok(())
}

fn write_i64(file: &File, name: &str, rows: usize, values: &[i64]) -> anyhow::Result<()> {
    ensure!(values.len() == rows, "invalid {name} value count");
    file.new_dataset::<i64>()
        .shape(rows)
        .create(name)?
        .write_raw(values)?;
    Ok(())
}

fn mix_seed(seed: u64, index: u64) -> u64 {
    seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(17)
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
