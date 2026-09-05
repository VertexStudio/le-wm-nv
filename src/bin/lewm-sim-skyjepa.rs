use std::{fs, path::PathBuf, time::Instant};

use anyhow::{Context, ensure};
use clap::Parser;
use le_wm_nv::{
    models::skyjepa::{
        SkyJepaControllerSession, SkyJepaSessionConfig, SkyJepaWarmStart,
        checkpoint::{SkyJepaCheckpoint, file_sha256},
    },
    runtime::DeviceSpec,
    skyjepa_sim::{SkyJepaDomain, SkyJepaRotorPlant, SkyJepaRotorState},
    skyjepa_task::{SkyJepaReferenceKind, skyjepa_reference_horizon, skyjepa_reference_state},
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

    #[arg(long, value_enum, default_value_t = SkyJepaWarmStart::FreshPrior)]
    warm_start: SkyJepaWarmStart,

    #[arg(long, default_value_t = 200)]
    simulation_rate_hz: usize,

    #[arg(long, default_value_t = 2.0)]
    radius_m: f32,

    #[arg(long, default_value_t = 8.0)]
    period_seconds: f32,

    #[arg(long, default_value = "circle")]
    reference: SkyJepaReferenceKind,

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
    report_version: u32,
    checkpoint_sha256: String,
    executable_sha256: String,
    configuration: serde_json::Value,
    finite: bool,
    ground_contact: bool,
    trace: Vec<SimulationStep>,
    reference: SkyJepaReferenceKind,
    control_steps: usize,
    samples: usize,
    horizon: usize,
    dt_seconds: f32,
    randomized_domain: bool,
    domain: SkyJepaDomain,
    position_rmse_m: f64,
    position_vector_rmse_m: f64,
    maximum_position_error_m: f64,
    cold_warmup_ms: f64,
    mean_plan_ms: f64,
    steady_mean_plan_ms: f64,
    p50_plan_ms: f64,
    p95_plan_ms: f64,
    max_plan_ms: f64,
    achieved_control_hz: f64,
    elapsed_seconds: f64,
    trim_scale: f32,
    mean_model_correction_l2: f64,
    maximum_model_correction_l2: f64,
    first_action: [f32; 4],
    first_prior_action: [f32; 4],
    mean_action: [f64; 4],
    final_action: [f32; 4],
    final_state: SkyJepaRotorState,
    first_predicted_states: Vec<[f32; 18]>,
}

#[derive(Debug, Serialize)]
struct SimulationStep {
    time_seconds: f32,
    state: SkyJepaRotorState,
    reference: [f32; 18],
    action: [f32; 4],
    prior_action: [f32; 4],
    action_correction: [f32; 4],
    plan_ms: f64,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let started = Instant::now();
    let checkpoint_dir = args.checkpoint_dir.unwrap_or_else(default_checkpoint_dir);
    let checkpoint = SkyJepaCheckpoint::load(&checkpoint_dir)?;
    let device = args.device.resolve()?;
    let domain = if args.randomize_domain {
        SkyJepaDomain::sample(args.domain_seed)
    } else {
        SkyJepaDomain::default()
    };
    let mut plant = SkyJepaRotorPlant::new(domain, SkyJepaRotorState::hover())?;
    let session_cfg = SkyJepaSessionConfig {
        samples: args.samples,
        horizon: args.horizon,
        planner_seed: args.planner_seed,
        warm_start: args.warm_start,
        ..SkyJepaSessionConfig::default()
    };
    let mut controller = SkyJepaControllerSession::load(
        &checkpoint_dir,
        device,
        session_cfg.clone(),
        plant.state(),
    )?;
    let dt = controller.dt();
    let sim_substeps = (args.simulation_rate_hz as f32 * dt).round() as usize;
    ensure!(
        sim_substeps > 0,
        "simulation rate is too low for control dt"
    );
    let sim_dt = dt / sim_substeps as f32;
    let mut squared_position_error = 0.0f64;
    let mut maximum_position_error = 0.0f64;
    let mut total_plan_seconds = 0.0f64;
    let mut max_plan_seconds = 0.0f64;
    let mut plan_times = Vec::with_capacity(args.control_steps);
    let mut action_sum = [0.0f64; 4];
    let mut first_action = [0.0; 4];
    let mut first_prior_action = [0.0; 4];
    let mut final_action = [0.0; 4];
    let mut first_predicted_states = Vec::new();
    let mut correction_sum = 0.0f64;
    let mut maximum_correction = 0.0f64;
    let mut trace = Vec::with_capacity(args.control_steps);
    let mut ground_contact = false;
    let mut finite = true;
    let warmup_reference = skyjepa_reference_horizon(
        args.reference,
        0.0,
        dt,
        args.horizon,
        args.radius_m,
        args.period_seconds,
    );
    let cold_warmup_ms = controller.warm_up(plant.state(), &warmup_reference)?;
    controller.reset_with_action(plant.state(), plant.nominal_hover_action())?;

    for step in 0..args.control_steps {
        let time = step as f32 * dt;
        let references = skyjepa_reference_horizon(
            args.reference,
            time,
            dt,
            args.horizon,
            args.radius_m,
            args.period_seconds,
        );
        let plan = if step == 0 {
            controller.plan_with_prediction(&references)?
        } else {
            controller.plan(&references)?
        };
        let plan_seconds = plan.plan_ms / 1e3;
        total_plan_seconds += plan_seconds;
        max_plan_seconds = max_plan_seconds.max(plan_seconds);
        plan_times.push(plan_seconds);
        let action = plan.action;
        let correction = plan
            .action_correction
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt();
        correction_sum += correction;
        maximum_correction = maximum_correction.max(correction);
        if step == 0 {
            first_action = action;
            first_prior_action = plan.prior_action;
            first_predicted_states = plan.predicted_states;
        }
        final_action = action;
        for (sum, value) in action_sum.iter_mut().zip(action) {
            *sum += f64::from(value);
        }
        for _ in 0..sim_substeps {
            plant.step(action, sim_dt);
            ground_contact |= plant.state().position[2] <= 0.051;
        }
        controller.commit_observation(plant.state(), action);
        let reference = skyjepa_reference_state(
            args.reference,
            time + dt,
            args.radius_m,
            args.period_seconds,
        );
        let position = plant.state().position;
        let error_sq = position
            .iter()
            .zip(reference[0..3].iter())
            .map(|(actual, target)| {
                let error = f64::from(*actual - *target);
                error * error
            })
            .sum::<f64>();
        squared_position_error += error_sq;
        maximum_position_error = maximum_position_error.max(error_sq.sqrt());
        finite &= plant
            .state()
            .as_state18()
            .iter()
            .chain(action.iter())
            .all(|value| value.is_finite());
        trace.push(SimulationStep {
            time_seconds: time + dt,
            state: plant.state(),
            reference,
            action,
            prior_action: plan.prior_action,
            action_correction: plan.action_correction,
            plan_ms: plan.plan_ms,
        });
        ensure!(finite, "non-finite simulation state/action at step {step}");
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
        report_version: 2,
        checkpoint_sha256: checkpoint.fingerprint()?,
        executable_sha256: file_sha256(&std::env::current_exe()?)?,
        configuration: serde_json::json!({"argv":std::env::args().collect::<Vec<_>>(),
            "session":session_cfg,"domain_seed":args.domain_seed,"simulation_rate_hz":args.simulation_rate_hz,
            "radius_m":args.radius_m,"period_seconds":args.period_seconds,
            "timing_scope":"planning only; simulated time does not model scheduling overruns; first measured cycle includes prediction export"}),
        finite,
        ground_contact,
        trace,
        reference: args.reference,
        control_steps: args.control_steps,
        samples: args.samples,
        horizon: args.horizon,
        dt_seconds: dt,
        randomized_domain: args.randomize_domain,
        domain,
        position_rmse_m: (squared_position_error / (args.control_steps * 3) as f64).sqrt(),
        position_vector_rmse_m: (squared_position_error / args.control_steps as f64).sqrt(),
        maximum_position_error_m: maximum_position_error,
        cold_warmup_ms,
        mean_plan_ms: mean_plan_seconds * 1e3,
        steady_mean_plan_ms: steady_mean_plan_seconds * 1e3,
        p50_plan_ms: percentile(0.5) * 1e3,
        p95_plan_ms: percentile(0.95) * 1e3,
        max_plan_ms: max_plan_seconds * 1e3,
        achieved_control_hz: 1.0 / steady_mean_plan_seconds,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        trim_scale: controller.trim_scale(),
        mean_model_correction_l2: correction_sum / args.control_steps as f64,
        maximum_model_correction_l2: maximum_correction,
        first_action,
        first_prior_action,
        mean_action: action_sum.map(|value| value / args.control_steps as f64),
        final_action,
        final_state: plant.state(),
        first_predicted_states,
    };
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(output) = args.output {
        fs::write(&output, &json)
            .with_context(|| format!("failed to write {}", output.display()))?;
    }
    println!("{json}");
    Ok(())
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    ensure!(args.control_steps > 0, "control_steps must be positive");
    ensure!(args.samples > 0, "samples must be positive");
    ensure!(args.horizon > 0, "horizon must be positive");
    ensure!(
        args.simulation_rate_hz >= 20 && args.simulation_rate_hz.is_multiple_of(20),
        "simulation_rate_hz must be a positive multiple of the 20 Hz model rate"
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

fn default_checkpoint_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".stable_worldmodel")
        .join("le-wm-nv-runs")
        .join("skyjepa-drone-state18-20hz")
}
