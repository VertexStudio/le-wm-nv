use std::{collections::VecDeque, fs, path::PathBuf, time::Instant};

use anyhow::{Context, ensure};
use candle::{DType, Tensor};
use clap::Parser;
use le_wm_nv::{
    checkpoint::var_builder_from_path,
    data::skyjepa::{SkyJepaActionSpace, SkyJepaDatasetConfig, SkyJepaNormalization},
    models::skyjepa::{
        SkyJepaConfig, SkyJepaControlConfig, SkyJepaModel, SkyJepaMppiScorer, SkyJepaProber,
        SkyJepaProberConfig,
    },
    planner::{ActionBounds, MppiPlanner},
    runtime::DeviceSpec,
    skyjepa_sim::{SkyJepaDomain, SkyJepaRotorPlant, SkyJepaRotorState},
};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Run closed-loop SkyJEPA MPPI against the rotor-force UAV simulator")]
struct Args {
    #[arg(long)]
    checkpoint_dir: Option<PathBuf>,

    #[arg(long, default_value_t = DeviceSpec::Cuda(0))]
    device: DeviceSpec,

    #[arg(long, default_value_t = 400)]
    control_steps: usize,

    #[arg(long, default_value_t = 512)]
    samples: usize,

    #[arg(long, default_value_t = 15)]
    horizon: usize,

    #[arg(long, default_value_t = 200)]
    simulation_rate_hz: usize,

    #[arg(long, default_value_t = 2.0)]
    radius_m: f32,

    #[arg(long, default_value_t = 8.0)]
    period_seconds: f32,

    /// Enable one held-out randomized plant domain.
    #[arg(long)]
    randomize_domain: bool,

    #[arg(long, default_value_t = 9001)]
    domain_seed: u64,

    #[arg(long, default_value_t = 7)]
    planner_seed: u64,

    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct SimulationReport {
    control_steps: usize,
    samples: usize,
    horizon: usize,
    dt_seconds: f32,
    randomized_domain: bool,
    domain: SkyJepaDomain,
    position_rmse_m: f64,
    mean_plan_ms: f64,
    steady_mean_plan_ms: f64,
    p50_plan_ms: f64,
    p95_plan_ms: f64,
    max_plan_ms: f64,
    achieved_control_hz: f64,
    elapsed_seconds: f64,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let started = Instant::now();
    let checkpoint_dir = args.checkpoint_dir.unwrap_or_else(default_checkpoint_dir);
    let model_cfg: SkyJepaConfig = read_json(checkpoint_dir.join("model-config.json"))?;
    let prober_cfg: SkyJepaProberConfig = read_json(checkpoint_dir.join("prober-config.json"))?;
    let dataset_cfg: SkyJepaDatasetConfig = read_json(checkpoint_dir.join("dataset-config.json"))?;
    let normalization: SkyJepaNormalization = read_json(checkpoint_dir.join("normalization.json"))?;
    ensure!(
        dataset_cfg.action_space == SkyJepaActionSpace::RotorForces,
        "closed-loop SkyJEPA simulator requires a rotor-force checkpoint"
    );
    let device = args.device.resolve()?;
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

    let domain = if args.randomize_domain {
        SkyJepaDomain::sample(args.domain_seed)
    } else {
        SkyJepaDomain::default()
    };
    let mut plant = SkyJepaRotorPlant::new(domain, SkyJepaRotorState::hover())?;
    let nominal = SkyJepaDomain::default();
    let hover_action = [nominal.mass * nominal.gravity / 4.0; 4];
    let mut state_history =
        VecDeque::from(vec![plant.state().as_state18(); model_cfg.history_steps]);
    let mut action_history = VecDeque::from(vec![hover_action; model_cfg.history_steps - 1]);
    let mut control_cfg = SkyJepaControlConfig::paper_derived();
    control_cfg.samples = args.samples;
    control_cfg.horizon = args.horizon;
    let max_rotor_force = nominal.mass * nominal.gravity * nominal.max_thrust_weight / 4.0;
    let bounds = ActionBounds {
        low: vec![0.0; 4],
        high: vec![max_rotor_force; 4],
    };
    let mut mppi_cfg = control_cfg.mppi_config(bounds)?;
    mppi_cfg.seed = Some(args.planner_seed);
    mppi_cfg.deadline_action = Some(hover_action.to_vec());
    let mut planner = MppiPlanner::new(mppi_cfg);
    planner.set_warm_start_sequence(
        Tensor::from_vec(
            hover_action.repeat(args.horizon),
            (1, args.horizon, 4),
            &device,
        )?
        .to_dtype(DType::F32)?,
    );
    let dt = control_cfg.dt;
    let sim_substeps = (args.simulation_rate_hz as f32 * dt).round() as usize;
    ensure!(
        sim_substeps > 0,
        "simulation rate is too low for control dt"
    );
    let sim_dt = dt / sim_substeps as f32;
    let mut squared_position_error = 0.0f64;
    let mut total_plan_seconds = 0.0f64;
    let mut max_plan_seconds = 0.0f64;
    let mut plan_times = Vec::with_capacity(args.control_steps);

    for step in 0..args.control_steps {
        let time = step as f32 * dt;
        let state_tensor = Tensor::from_vec(
            state_history.iter().flatten().copied().collect::<Vec<_>>(),
            (1, model_cfg.history_steps, 18),
            &device,
        )?;
        let action_tensor = Tensor::from_vec(
            action_history.iter().flatten().copied().collect::<Vec<_>>(),
            (1, model_cfg.history_steps - 1, 4),
            &device,
        )?;
        let references = (1..=args.horizon)
            .flat_map(|offset| {
                reference_state(
                    time + offset as f32 * dt,
                    args.radius_m,
                    args.period_seconds,
                )
            })
            .collect::<Vec<_>>();
        let reference_states = Tensor::from_vec(references, (1, args.horizon, 18), &device)?;
        let reference_actions = Tensor::from_vec(
            hover_action.repeat(args.horizon),
            (1, args.horizon, 4),
            &device,
        )?;
        let scorer = SkyJepaMppiScorer::new(
            &model,
            &prober,
            &state_tensor,
            &action_tensor,
            reference_states,
            reference_actions,
            &normalization,
            dt,
            control_cfg.cost.clone(),
        )?;
        let plan_started = Instant::now();
        let result = planner.plan_device(&scorer)?;
        let action_values = result.first_action.to_vec2::<f32>()?;
        let plan_seconds = plan_started.elapsed().as_secs_f64();
        total_plan_seconds += plan_seconds;
        max_plan_seconds = max_plan_seconds.max(plan_seconds);
        plan_times.push(plan_seconds);
        let action: [f32; 4] = action_values[0]
            .as_slice()
            .try_into()
            .expect("planner action has four values");
        for _ in 0..sim_substeps {
            plant.step(action, sim_dt);
        }
        state_history.pop_front();
        state_history.push_back(plant.state().as_state18());
        action_history.pop_front();
        action_history.push_back(action);
        let reference = reference_state(time + dt, args.radius_m, args.period_seconds);
        let position = plant.state().position;
        squared_position_error += position
            .iter()
            .zip(reference[0..3].iter())
            .map(|(actual, target)| {
                let error = f64::from(*actual - *target);
                error * error
            })
            .sum::<f64>();
    }
    let mean_plan_seconds = total_plan_seconds / args.control_steps as f64;
    let steady_mean_plan_seconds = if plan_times.len() > 1 {
        plan_times[1..].iter().sum::<f64>() / (plan_times.len() - 1) as f64
    } else {
        mean_plan_seconds
    };
    let mut sorted_plan_times = plan_times.clone();
    sorted_plan_times.sort_by(f64::total_cmp);
    let percentile = |fraction: f64| {
        let index = ((sorted_plan_times.len() - 1) as f64 * fraction).round() as usize;
        sorted_plan_times[index]
    };
    let report = SimulationReport {
        control_steps: args.control_steps,
        samples: args.samples,
        horizon: args.horizon,
        dt_seconds: dt,
        randomized_domain: args.randomize_domain,
        domain,
        position_rmse_m: (squared_position_error / (args.control_steps * 3) as f64).sqrt(),
        mean_plan_ms: mean_plan_seconds * 1e3,
        steady_mean_plan_ms: steady_mean_plan_seconds * 1e3,
        p50_plan_ms: percentile(0.5) * 1e3,
        p95_plan_ms: percentile(0.95) * 1e3,
        max_plan_ms: max_plan_seconds * 1e3,
        achieved_control_hz: 1.0 / steady_mean_plan_seconds,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    };
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(output) = args.output {
        fs::write(&output, &json)
            .with_context(|| format!("failed to write {}", output.display()))?;
    }
    println!("{json}");
    Ok(())
}

fn reference_state(time: f32, radius: f32, period: f32) -> [f32; 18] {
    let omega = 2.0 * std::f32::consts::PI / period;
    let angle = omega * time;
    let mut state = [0.0; 18];
    state[0] = radius * (angle.cos() - 1.0);
    state[1] = radius * angle.sin();
    state[2] = 1.0;
    state[3] = -radius * omega * angle.sin();
    state[4] = radius * omega * angle.cos();
    state[6] = 1.0;
    state[10] = 1.0;
    state[14] = 1.0;
    state
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    ensure!(args.control_steps > 0, "control_steps must be positive");
    ensure!(args.samples > 0, "samples must be positive");
    ensure!(args.horizon > 0, "horizon must be positive");
    ensure!(
        args.simulation_rate_hz >= 20,
        "simulation_rate_hz must be at least 20"
    );
    ensure!(
        args.radius_m.is_finite() && args.radius_m > 0.0,
        "radius_m must be positive"
    );
    ensure!(
        args.period_seconds.is_finite() && args.period_seconds > 0.0,
        "period_seconds must be positive"
    );
    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: PathBuf) -> anyhow::Result<T> {
    serde_json::from_str(
        &fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))
}

fn default_checkpoint_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".stable_worldmodel")
        .join("le-wm-nv-runs")
        .join("skyjepa-drone-state18-20hz")
}
