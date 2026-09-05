use std::{fs, path::PathBuf, time::Instant};

use anyhow::{Context, ensure};
use clap::{Parser, ValueEnum};
use le_wm_nv::{
    models::skyjepa::{SkyJepaControllerSession, SkyJepaSessionConfig, SkyJepaWarmStart},
    runtime::DeviceSpec,
    skyjepa_sim::{SkyJepaDomain, SkyJepaRotorPlant, SkyJepaRotorState},
    skyjepa_task::{
        SkyJepaReferenceKind, skyjepa_geometric_action_prior, skyjepa_reference_horizon,
        skyjepa_reference_state,
    },
};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Benchmark SkyJEPA closed-loop control over nominal and held-out domains")]
struct Args {
    #[arg(long)]
    checkpoint_dir: PathBuf,

    #[arg(long, default_value_t = DeviceSpec::Cuda(0))]
    device: DeviceSpec,

    #[arg(long, default_value_t = 512)]
    samples: usize,

    #[arg(long, default_value_t = 15)]
    horizon: usize,

    #[arg(long, default_value_t = 200)]
    simulation_rate_hz: usize,

    #[arg(long, default_value_t = 8.0)]
    duration_seconds: f32,

    #[arg(long, default_value_t = 2.0)]
    radius_m: f32,

    #[arg(long, default_value_t = 8.0)]
    period_seconds: f32,

    #[arg(long, default_value_t = 20)]
    random_domains: usize,

    #[arg(long, default_value_t = 9001)]
    domain_seed: u64,

    #[arg(long, default_value_t = 7)]
    planner_seed: u64,

    #[arg(long, value_enum, default_value_t = SkyJepaWarmStart::FreshPrior)]
    warm_start: SkyJepaWarmStart,

    /// `prior` runs the same trim-aware geometric sequence without learned
    /// MPPI corrections, providing an apples-to-apples control baseline.
    #[arg(long, value_enum, default_value_t = ControllerMode::SkyJepa)]
    controller: ControllerMode,

    #[arg(long, default_value_t = 0.75)]
    max_position_rmse_m: f64,

    #[arg(long, default_value_t = 3.0)]
    max_position_error_m: f64,

    #[arg(long, default_value_t = 10.0)]
    max_p95_plan_ms: f64,

    #[arg(long, default_value_t = 0.95)]
    min_success_rate: f64,

    #[arg(long)]
    output: Option<PathBuf>,

    #[arg(long)]
    allow_fail: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum ControllerMode {
    SkyJepa,
    Prior,
}

#[derive(Debug, Serialize)]
struct ScenarioResult {
    reference: SkyJepaReferenceKind,
    randomized: bool,
    domain_seed: Option<u64>,
    domain: SkyJepaDomain,
    steps: usize,
    position_vector_rmse_m: f64,
    maximum_position_error_m: f64,
    p50_plan_ms: f64,
    p95_plan_ms: f64,
    maximum_plan_ms: f64,
    trim_scale: f32,
    mean_model_correction_l2: f64,
    maximum_model_correction_l2: f64,
    low_action_fraction: f64,
    high_action_fraction: f64,
    ground_contact: bool,
    finite: bool,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    warm_start: SkyJepaWarmStart,
    passed: bool,
    controller: ControllerMode,
    checkpoint_dir: PathBuf,
    samples: usize,
    horizon: usize,
    random_domains: usize,
    scenarios: usize,
    runs: usize,
    successful_runs: usize,
    success_rate: f64,
    cold_warmup_ms: f64,
    aggregate_p95_plan_ms: f64,
    worst_position_rmse_m: f64,
    worst_position_error_m: f64,
    elapsed_seconds: f64,
    failures: Vec<String>,
    results: Vec<ScenarioResult>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let started = Instant::now();
    let initial_state = SkyJepaRotorState::hover();
    let device = args.device.resolve()?;
    let mut controller = SkyJepaControllerSession::load(
        &args.checkpoint_dir,
        device,
        SkyJepaSessionConfig {
            samples: args.samples,
            horizon: args.horizon,
            planner_seed: args.planner_seed,
            warm_start: args.warm_start,
            ..SkyJepaSessionConfig::default()
        },
        initial_state,
    )?;
    let warmup_references = skyjepa_reference_horizon(
        SkyJepaReferenceKind::Circle,
        0.0,
        controller.dt(),
        controller.horizon(),
        args.radius_m,
        args.period_seconds,
    );
    let cold_warmup_ms = controller.warm_up(initial_state, &warmup_references)?;
    let references = [
        SkyJepaReferenceKind::Hover,
        SkyJepaReferenceKind::Circle,
        SkyJepaReferenceKind::FigureEight,
    ];
    let mut domains = vec![(false, None, SkyJepaDomain::default())];
    domains.extend((0..args.random_domains).map(|index| {
        let seed = mix_seed(args.domain_seed, index as u64);
        (true, Some(seed), SkyJepaDomain::sample(seed))
    }));
    let mut results = Vec::with_capacity(references.len() * domains.len());
    let mut all_plan_times = Vec::new();
    for reference in references {
        for (randomized, seed, domain) in domains.iter().copied() {
            let result = run_scenario(
                &args,
                &mut controller,
                reference,
                randomized,
                seed,
                domain,
                &mut all_plan_times,
            )?;
            println!(
                "reference={} randomized={} seed={:?} rmse={:.3}m max={:.3}m p95={:.2}ms passed={}",
                reference.as_str(),
                randomized,
                seed,
                result.position_vector_rmse_m,
                result.maximum_position_error_m,
                result.p95_plan_ms,
                result.passed
            );
            results.push(result);
        }
    }
    let successful_runs = results.iter().filter(|result| result.passed).count();
    let success_rate = successful_runs as f64 / results.len() as f64;
    let aggregate_p95 = percentile(&mut all_plan_times, 0.95);
    let worst_rmse = results
        .iter()
        .map(|result| result.position_vector_rmse_m)
        .fold(0.0, f64::max);
    let worst_error = results
        .iter()
        .map(|result| result.maximum_position_error_m)
        .fold(0.0, f64::max);
    let nominal_failures = results
        .iter()
        .filter(|result| !result.randomized && !result.passed)
        .count();
    let mut failures = Vec::new();
    if nominal_failures > 0 {
        failures.push(format!("{nominal_failures} nominal scenario(s) failed"));
    }
    if success_rate < args.min_success_rate {
        failures.push(format!(
            "success rate {success_rate:.3} is below {:.3}",
            args.min_success_rate
        ));
    }
    if aggregate_p95 > args.max_p95_plan_ms {
        failures.push(format!(
            "aggregate p95 {aggregate_p95:.3} ms exceeds {:.3} ms",
            args.max_p95_plan_ms
        ));
    }
    let report = BenchmarkReport {
        warm_start: args.warm_start,
        passed: failures.is_empty(),
        controller: args.controller,
        checkpoint_dir: fs::canonicalize(&args.checkpoint_dir)
            .unwrap_or(args.checkpoint_dir.clone()),
        samples: args.samples,
        horizon: args.horizon,
        random_domains: args.random_domains,
        scenarios: references.len(),
        runs: results.len(),
        successful_runs,
        success_rate,
        cold_warmup_ms,
        aggregate_p95_plan_ms: aggregate_p95,
        worst_position_rmse_m: worst_rmse,
        worst_position_error_m: worst_error,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        failures,
        results,
    };
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(output) = args.output.as_ref() {
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(output, &json)
            .with_context(|| format!("failed to write {}", output.display()))?;
    }
    println!("{json}");
    ensure!(
        report.passed || args.allow_fail,
        "SkyJEPA closed-loop benchmark failed with {} issue(s)",
        report.failures.len()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_scenario(
    args: &Args,
    controller: &mut SkyJepaControllerSession,
    reference_kind: SkyJepaReferenceKind,
    randomized: bool,
    domain_seed: Option<u64>,
    domain: SkyJepaDomain,
    aggregate_plan_times: &mut Vec<f64>,
) -> anyhow::Result<ScenarioResult> {
    let mut plant = SkyJepaRotorPlant::new(domain, SkyJepaRotorState::hover())?;
    controller.reset_with_action(plant.state(), plant.nominal_hover_action())?;
    let dt = controller.dt();
    let steps = (args.duration_seconds / dt).round() as usize;
    let substeps = (args.simulation_rate_hz as f32 * dt).round() as usize;
    let sim_dt = dt / substeps as f32;
    let mut position_error_sq = 0.0f64;
    let mut maximum_position_error = 0.0f64;
    let mut plan_times = Vec::with_capacity(steps);
    let mut low_actions = 0usize;
    let mut high_actions = 0usize;
    let mut correction_sum = 0.0f64;
    let mut maximum_correction = 0.0f64;
    let mut ground_contact = false;
    let mut finite = true;
    let maximum_force = domain.mass * domain.gravity * domain.max_thrust_weight / 4.0;
    for step in 0..steps {
        let time = step as f32 * dt;
        let references = skyjepa_reference_horizon(
            reference_kind,
            time,
            dt,
            controller.horizon(),
            args.radius_m,
            args.period_seconds,
        );
        let plan_started = Instant::now();
        let (action, correction, plan_ms) = match args.controller {
            ControllerMode::SkyJepa => {
                let plan = controller.plan(&references)?;
                let correction = plan
                    .action_correction
                    .iter()
                    .map(|value| f64::from(*value).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (plan.action, correction, plan.plan_ms)
            }
            ControllerMode::Prior => {
                let action = skyjepa_geometric_action_prior(
                    plant.state().as_state18(),
                    &references,
                    controller.dt(),
                    SkyJepaDomain::default(),
                )[0]
                .map(|force| force * controller.trim_scale());
                (action, 0.0, plan_started.elapsed().as_secs_f64() * 1e3)
            }
        };
        correction_sum += correction;
        maximum_correction = maximum_correction.max(correction);
        plan_times.push(plan_ms);
        aggregate_plan_times.push(plan_ms);
        for force in action {
            if force <= 1e-6 {
                low_actions += 1;
            }
            if force >= maximum_force * 0.999 {
                high_actions += 1;
            }
        }
        for _ in 0..substeps {
            plant.step(action, sim_dt);
        }
        controller.commit_observation(plant.state(), action);
        let reference = skyjepa_reference_state(
            reference_kind,
            time + dt,
            args.radius_m,
            args.period_seconds,
        );
        let error = distance(plant.state().position, reference[0..3].try_into().unwrap());
        position_error_sq += error * error;
        maximum_position_error = maximum_position_error.max(error);
        ground_contact |= plant.state().position[2] <= 0.051;
        finite &= plant
            .state()
            .as_state18()
            .iter()
            .chain(action.iter())
            .all(|value| value.is_finite());
        if !finite {
            break;
        }
    }
    let position_rmse = (position_error_sq / steps as f64).sqrt();
    let p50 = percentile(&mut plan_times.clone(), 0.50);
    let p95 = percentile(&mut plan_times.clone(), 0.95);
    let maximum_plan = plan_times.iter().copied().fold(0.0, f64::max);
    let action_count = (steps * 4) as f64;
    let passed = finite
        && !ground_contact
        && position_rmse <= args.max_position_rmse_m
        && maximum_position_error <= args.max_position_error_m
        && p95 <= args.max_p95_plan_ms;
    Ok(ScenarioResult {
        reference: reference_kind,
        randomized,
        domain_seed,
        domain,
        steps,
        position_vector_rmse_m: position_rmse,
        maximum_position_error_m: maximum_position_error,
        p50_plan_ms: p50,
        p95_plan_ms: p95,
        maximum_plan_ms: maximum_plan,
        trim_scale: controller.trim_scale(),
        mean_model_correction_l2: correction_sum / steps as f64,
        maximum_model_correction_l2: maximum_correction,
        low_action_fraction: low_actions as f64 / action_count,
        high_action_fraction: high_actions as f64 / action_count,
        ground_contact,
        finite,
        passed,
    })
}

fn percentile(values: &mut [f64], fraction: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * fraction).round() as usize]
}

fn distance(lhs: [f32; 3], rhs: [f32; 3]) -> f64 {
    lhs.into_iter()
        .zip(rhs)
        .map(|(lhs, rhs)| f64::from(lhs - rhs).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn mix_seed(seed: u64, index: u64) -> u64 {
    seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(17)
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    ensure!(
        args.checkpoint_dir.is_dir(),
        "checkpoint-dir does not exist"
    );
    ensure!(args.samples > 0, "samples must be positive");
    ensure!(args.horizon > 0, "horizon must be positive");
    ensure!(
        args.simulation_rate_hz >= 20,
        "simulation-rate-hz must be at least 20"
    );
    ensure!(
        args.duration_seconds.is_finite() && args.duration_seconds > 0.0,
        "duration-seconds must be positive"
    );
    ensure!(
        args.radius_m.is_finite() && args.radius_m > 0.0,
        "radius-m must be positive"
    );
    ensure!(
        args.period_seconds.is_finite() && args.period_seconds > 0.0,
        "period-seconds must be positive"
    );
    ensure!(
        args.max_position_rmse_m.is_finite() && args.max_position_rmse_m > 0.0,
        "max-position-rmse-m must be positive"
    );
    ensure!(
        args.max_position_error_m.is_finite() && args.max_position_error_m > 0.0,
        "max-position-error-m must be positive"
    );
    ensure!(
        args.max_p95_plan_ms.is_finite() && args.max_p95_plan_ms > 0.0,
        "max-p95-plan-ms must be positive"
    );
    ensure!(
        args.min_success_rate.is_finite() && (0.0..=1.0).contains(&args.min_success_rate),
        "min-success-rate must be in [0, 1]"
    );
    Ok(())
}
