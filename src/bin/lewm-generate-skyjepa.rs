use std::{fs, path::PathBuf, time::Instant};

use anyhow::{Context, ensure};
use clap::Parser;
use hdf5::File;
use le_wm_nv::{
    data::{
        drone_racing::{cross3, mat3_mul_vec3},
        skyjepa::{
            SKYJEPA_ACTION_DIM, SKYJEPA_SCHEMA_VERSION, SKYJEPA_STATE_DIM, SkyJepaActionSpace,
            SkyJepaDatasetMetadata,
        },
    },
    skyjepa_sim::{SkyJepaDomain, SkyJepaRotorPlant, SkyJepaRotorState},
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
    episode_idx: Vec<i64>,
    step_idx: Vec<i64>,
    dt: Vec<f32>,
    domain_idx: Vec<i64>,
}

#[derive(Debug, Serialize)]
struct DomainRecord {
    index: usize,
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
        .map(|index| DomainRecord {
            index,
            parameters: SkyJepaDomain::sample(mix_seed(args.seed, index as u64)),
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
    let mut episode_idx = Vec::with_capacity(rows);
    let mut step_idx = Vec::with_capacity(rows);
    let mut dt = Vec::with_capacity(rows);
    let mut domain_idx = Vec::with_capacity(rows);
    for episode in episodes {
        states.extend(episode.states);
        actions.extend(episode.actions);
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
        generator: Some("lewm-generate-skyjepa clean-room RFF+geometric-PD simulator".into()),
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
        args.sample_rate_hz as f32,
    );
    let initial_reference = trajectory.at(0.0);
    let initial_state = SkyJepaRotorState {
        position: initial_reference.position,
        ..SkyJepaRotorState::hover()
    };
    let mut plant = SkyJepaRotorPlant::new(domain, initial_state)?;
    let mut result = Episode {
        states: Vec::with_capacity(samples * SKYJEPA_STATE_DIM),
        actions: Vec::with_capacity(samples * SKYJEPA_ACTION_DIM),
        episode_idx: vec![episode as i64; samples],
        step_idx: (0..samples).map(|step| step as i64).collect(),
        dt: vec![sample_dt; samples],
        domain_idx: vec![(episode % args.domains) as i64; samples],
    };
    let mut excitation = [0.0f32; 4];
    for step in 0..samples {
        let time = step as f32 * sample_dt;
        let reference = trajectory.at(time);
        for (rotor, value) in excitation.iter_mut().enumerate() {
            *value = 0.88 * *value
                + 0.12
                    * (0.09 * (time * (1.3 + 0.37 * rotor as f32) + episode as f32 * 0.17).sin());
        }
        let action = tracking_action(plant.state(), reference, excitation);
        result.states.extend_from_slice(&plant.state().as_state18());
        result.actions.extend_from_slice(&action);
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

#[derive(Debug, Clone, Copy)]
struct ReferencePoint {
    position: [f32; 3],
    velocity: [f32; 3],
}

struct ReferenceTrajectory {
    center: [f32; 3],
    amplitude: [[f32; 3]; 3],
    omega: [[f32; 3]; 3],
    phase: [[f32; 3]; 3],
}

impl ReferenceTrajectory {
    fn sample(seed: u64, sample_rate_hz: f32) -> Self {
        let mut rng = SplitMix64::new(seed);
        let periods_in_samples = [[37.0, 61.0, 13.0], [17.0, 23.0, 52.0], [19.0, 29.0, 53.0]];
        let mut amplitude = [[0.0; 3]; 3];
        let mut omega = [[0.0; 3]; 3];
        let mut phase = [[0.0; 3]; 3];
        for axis in 0..3 {
            for component in 0..3 {
                let scale = if axis == 2 { 0.18 } else { 0.55 };
                amplitude[axis][component] = rng.range(0.35, 1.0) * scale;
                let seconds = periods_in_samples[axis][component] / sample_rate_hz;
                omega[axis][component] = 2.0 * std::f32::consts::PI / seconds;
                phase[axis][component] = rng.range(0.0, 2.0 * std::f32::consts::PI);
            }
        }
        Self {
            center: [
                rng.range(-2.0, 2.0),
                rng.range(-2.0, 2.0),
                rng.range(1.5, 3.0),
            ],
            amplitude,
            omega,
            phase,
        }
    }

    fn at(&self, time: f32) -> ReferencePoint {
        let mut position = self.center;
        let mut velocity = [0.0; 3];
        for axis in 0..3 {
            for component in 0..3 {
                let angle = self.omega[axis][component] * time + self.phase[axis][component];
                position[axis] += self.amplitude[axis][component] * angle.sin();
                velocity[axis] +=
                    self.amplitude[axis][component] * self.omega[axis][component] * angle.cos();
            }
        }
        position[2] = position[2].max(0.6);
        ReferencePoint { position, velocity }
    }
}

fn tracking_action(
    state: SkyJepaRotorState,
    reference: ReferencePoint,
    excitation: [f32; 4],
) -> [f32; 4] {
    let nominal = SkyJepaDomain::default();
    let mut desired_acceleration = [0.0; 3];
    for (axis, value) in desired_acceleration.iter_mut().enumerate() {
        *value = 2.8 * (reference.position[axis] - state.position[axis])
            + 2.2 * (reference.velocity[axis] - state.velocity[axis]);
    }
    desired_acceleration[2] += nominal.gravity;
    let acceleration_norm = norm(desired_acceleration).max(1e-4);
    let desired_up = desired_acceleration.map(|value| value / acceleration_norm);
    let current_up = mat3_mul_vec3(state.rotation_world_from_body, [0.0, 0.0, 1.0]);
    let tilt_error = cross3(current_up, desired_up);
    let torque = [
        0.16 * tilt_error[0] - 0.035 * state.angular_velocity[0],
        0.16 * tilt_error[1] - 0.035 * state.angular_velocity[1],
        -0.025 * state.angular_velocity[2],
    ];
    let total_force = (nominal.mass * acceleration_norm).clamp(
        0.0,
        nominal.mass * nominal.gravity * nominal.max_thrust_weight,
    );
    let arm = nominal.arm_length;
    let yaw_lever = 0.025 * nominal.torque_scale;
    let base = total_force / 4.0;
    let mut forces = [
        base - torque[1] / (2.0 * arm) + torque[2] / (4.0 * yaw_lever),
        base + torque[0] / (2.0 * arm) - torque[2] / (4.0 * yaw_lever),
        base + torque[1] / (2.0 * arm) + torque[2] / (4.0 * yaw_lever),
        base - torque[0] / (2.0 * arm) - torque[2] / (4.0 * yaw_lever),
    ];
    let max_force = nominal.mass * nominal.gravity * nominal.max_thrust_weight / 4.0;
    for rotor in 0..4 {
        forces[rotor] = (forces[rotor] * (1.0 + excitation[rotor])).clamp(0.0, max_force);
    }
    forces
}

fn norm(value: [f32; 3]) -> f32 {
    (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt()
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
