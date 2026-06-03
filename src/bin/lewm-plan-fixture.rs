use std::{path::PathBuf, process::Command};

use anyhow::Context;
use candle::{DType, Tensor};
use clap::{Parser, ValueEnum};
use le_wm_nv::{
    checkpoint,
    models::lewm::{LeWm, LeWmConfig},
    planner::{
        CandidateScorer, CemConfig, CemPlanner, IcemConfig, IcemPlanner, LeWmGoalScorer,
        MppiConfig, MppiPlanner, PlanResult,
    },
    runtime::{DTypeSpec, DeviceSpec},
    session::LeWmSession,
};
use serde_json::json;

#[cfg(feature = "hub")]
use le_wm_nv::hub;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum PlannerArg {
    All,
    Cem,
    Mppi,
    Icem,
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    fixture: PathBuf,

    #[arg(long)]
    weights: Option<PathBuf>,

    #[arg(long)]
    config: Option<PathBuf>,

    #[arg(long)]
    hf_repo: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long, default_value_t = DeviceSpec::Cuda(0))]
    device: DeviceSpec,

    #[arg(long, default_value_t = DTypeSpec::F32)]
    dtype: DTypeSpec,

    #[arg(long, value_enum, default_value_t = PlannerArg::All)]
    planner: PlannerArg,

    #[arg(long)]
    horizon: Option<usize>,

    #[arg(long, default_value_t = 128)]
    samples: usize,

    #[arg(long)]
    elites: Option<usize>,

    #[arg(long, default_value_t = 3)]
    iterations: usize,

    #[arg(long, default_value_t = 1.0)]
    init_std: f32,

    #[arg(long, default_value_t = 1e-3)]
    min_std: f32,

    #[arg(long, default_value_t = 1.0)]
    noise_std: f32,

    #[arg(long, default_value_t = 1.0)]
    temperature: f32,

    #[arg(long)]
    seed: Option<u64>,

    #[arg(long, default_value_t = false)]
    json: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.samples < 2 {
        anyhow::bail!("--samples must be at least 2");
    }
    if args.iterations == 0 {
        anyhow::bail!("--iterations must be greater than zero");
    }

    let device = args.device.resolve()?;
    let dtype = args.dtype.dtype();
    let (weights, config) = resolve_files(&args)?;
    let cfg = match config.as_ref() {
        Some(path) => LeWmConfig::from_stable_worldmodel_json_file(path)?,
        None => LeWmConfig::tiny_patch14_224(2),
    };
    let action_dim = cfg.action_encoder.input_dim;
    let vb = checkpoint::var_builder_from_path(&weights, dtype, &device)?;
    let model = LeWm::new(cfg, vb)?;

    let arrays = Tensor::read_npz_by_name(
        &args.fixture,
        &["pixels", "action_candidates", "goal_emb", "cost"],
    )?;
    let pixels = arrays[0].to_device(&device)?.to_dtype(dtype)?;
    let fixture_candidates = arrays[1].to_device(&device)?.to_dtype(dtype)?;
    let goal_emb = arrays[2].to_device(&device)?.to_dtype(dtype)?;
    let python_fixture_cost = &arrays[3];
    let (batch, fixture_samples, fixture_horizon, fixture_action_dim) =
        fixture_candidates.dims4()?;
    if fixture_action_dim != action_dim {
        anyhow::bail!(
            "fixture action_dim {fixture_action_dim} does not match checkpoint action_dim {action_dim}"
        );
    }
    let horizon = args.horizon.unwrap_or(fixture_horizon);
    if horizon < model.config().history_size {
        anyhow::bail!(
            "planner horizon {horizon} must be >= LeWM history size {}",
            model.config().history_size
        );
    }

    let mut session = LeWmSession::new(model, device.clone(), dtype);
    let emb = session.reset_pixels(&pixels)?;
    let scorer = LeWmGoalScorer::new(&session, &goal_emb);
    let rust_baseline_cost = scorer.score_candidates(&fixture_candidates)?;
    let rust_baseline = cost_summary(&rust_baseline_cost)?;
    let python_baseline = cost_summary(python_fixture_cost)?;
    let baseline_argmin_stable = rust_baseline.argmin == python_baseline.argmin;

    let elites = args
        .elites
        .unwrap_or_else(|| (args.samples / 4).clamp(2, args.samples));
    if elites > args.samples {
        anyhow::bail!("--elites cannot exceed --samples");
    }

    let mut rows = Vec::new();
    if matches!(args.planner, PlannerArg::All | PlannerArg::Cem) {
        let mut cfg = CemConfig::new(horizon, args.samples, elites, action_dim);
        cfg.iterations = args.iterations;
        cfg.init_std = args.init_std;
        cfg.min_std = args.min_std;
        cfg.seed = args.seed;
        let planner = CemPlanner::new(cfg);
        rows.push(plan_summary(
            "cem",
            planner.plan(&scorer)?,
            &scorer,
            rust_baseline.best,
        )?);
    }
    if matches!(args.planner, PlannerArg::All | PlannerArg::Mppi) {
        let mut cfg = MppiConfig::new(horizon, args.samples, action_dim);
        cfg.iterations = args.iterations;
        cfg.noise_std = args.noise_std;
        cfg.temperature = args.temperature;
        cfg.seed = args.seed;
        let planner = MppiPlanner::new(cfg);
        rows.push(plan_summary(
            "mppi",
            planner.plan(&scorer)?,
            &scorer,
            rust_baseline.best,
        )?);
    }
    if matches!(args.planner, PlannerArg::All | PlannerArg::Icem) {
        let mut cfg = IcemConfig::new(horizon, args.samples, elites, action_dim);
        cfg.iterations = args.iterations;
        cfg.keep_elites = elites.min(args.samples);
        cfg.init_std = args.init_std;
        cfg.min_std = args.min_std;
        cfg.seed = args.seed;
        let mut planner = IcemPlanner::new(cfg);
        rows.push(plan_summary(
            "icem",
            planner.plan(&scorer)?,
            &scorer,
            rust_baseline.best,
        )?);
    }

    let payload = json!({
        "git_commit": git_commit(),
        "fixture": args.fixture.display().to_string(),
        "weights": weights.display().to_string(),
        "config": config.as_ref().map(|path| path.display().to_string()),
        "hf_repo": args.hf_repo,
        "device": args.device.to_string(),
        "dtype": args.dtype.to_string(),
        "batch_size": batch,
        "fixture_samples": fixture_samples,
        "fixture_horizon": fixture_horizon,
        "horizon": horizon,
        "samples": args.samples,
        "elites": elites,
        "iterations": args.iterations,
        "action_dim": action_dim,
        "embedding_shape": emb.dims(),
        "rust_baseline": rust_baseline,
        "python_fixture_baseline": python_baseline,
        "baseline_argmin_stable": baseline_argmin_stable,
        "planners": rows.clone(),
    });

    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!(
            "lewm-plan-fixture git={} fixture={} device={} dtype={} batch={} horizon={} samples={} elites={} iterations={} action_dim={}",
            git_commit(),
            args.fixture.display(),
            args.device,
            args.dtype,
            batch,
            horizon,
            args.samples,
            elites,
            args.iterations,
            action_dim,
        );
        println!(
            "fixture baseline: rust_best={:.6} python_best={:.6} argmin_stable={}",
            rust_baseline.best, python_baseline.best, baseline_argmin_stable
        );
        for row in rows {
            let best_cost = row["best_cost"].as_f64().unwrap_or_default();
            let improvement = row["baseline_improvement"].as_f64().unwrap_or_default();
            let elapsed = row["elapsed_ms"].as_f64().unwrap_or_default();
            println!(
                "{:<6} cost={:>10.6} improvement={:>10.6} elapsed={:>8.3}ms first_action={}",
                row["planner"].as_str().unwrap_or("planner"),
                best_cost,
                improvement,
                elapsed,
                row["first_action"],
            );
        }
    }

    Ok(())
}

#[derive(Debug, Clone, serde::Serialize)]
struct CostSummary {
    best: f32,
    mean: f32,
    argmin: Vec<usize>,
}

fn cost_summary(cost: &Tensor) -> anyhow::Result<CostSummary> {
    let rows = cost.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let mut argmin = Vec::with_capacity(rows.len());
    let mut best_values = Vec::with_capacity(rows.len());
    let mut total = 0.0f32;
    let mut count = 0usize;
    for row in rows {
        let mut best_idx = 0usize;
        let mut best = f32::INFINITY;
        for (idx, value) in row.iter().copied().enumerate() {
            if !value.is_finite() {
                anyhow::bail!("cost contains non-finite value {value}");
            }
            total += value;
            count += 1;
            if value < best {
                best = value;
                best_idx = idx;
            }
        }
        argmin.push(best_idx);
        best_values.push(best);
    }
    let best = best_values
        .iter()
        .copied()
        .fold(f32::INFINITY, |acc, value| acc.min(value));
    let mean = if count == 0 {
        f32::NAN
    } else {
        total / count as f32
    };
    Ok(CostSummary { best, mean, argmin })
}

fn plan_summary(
    name: &'static str,
    result: PlanResult,
    scorer: &LeWmGoalScorer<'_>,
    baseline_best: f32,
) -> anyhow::Result<serde_json::Value> {
    let sequence = result.sequence.unsqueeze(1)?;
    let best_cost_tensor = scorer.score_candidates(&sequence)?;
    let best_cost = cost_summary(&best_cost_tensor)?;
    let score_summary = cost_summary(&result.scores)?;
    let first_action = result.first_action.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    Ok(json!({
        "planner": name,
        "best_cost": best_cost.best,
        "score_best": score_summary.best,
        "score_mean": score_summary.mean,
        "baseline_improvement": baseline_best - best_cost.best,
        "elapsed_ms": result.elapsed.as_secs_f64() * 1000.0,
        "iterations_completed": result.iterations_completed,
        "deadline_reached": result.deadline_reached,
        "deadline_outcome": format!("{:?}", result.deadline_outcome),
        "used_host_elite_selection": result.used_host_elite_selection,
        "best_indices": result.best_indices,
        "first_action": first_action,
    }))
}

fn resolve_files(args: &Args) -> anyhow::Result<(PathBuf, Option<PathBuf>)> {
    if args.weights.is_some() || args.config.is_some() {
        let weights = args
            .weights
            .clone()
            .context("--weights is required when using local checkpoint files")?;
        return Ok((weights, args.config.clone()));
    }

    let repo = args
        .hf_repo
        .as_deref()
        .context("provide --hf-repo or --weights/--config")?;
    resolve_hf_files(repo, args.revision.as_deref())
}

fn resolve_hf_files(
    repo: &str,
    revision: Option<&str>,
) -> anyhow::Result<(PathBuf, Option<PathBuf>)> {
    #[cfg(feature = "hub")]
    {
        let files = hub::download_stable_worldmodel_checkpoint(repo, revision)?;
        return Ok((files.weights, Some(files.config)));
    }

    #[cfg(not(feature = "hub"))]
    {
        let _ = (repo, revision);
        anyhow::bail!("--hf-repo requires building with --features hub");
    }
}

fn git_commit() -> String {
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
            } else {
                None
            }
        })
        .filter(|commit| !commit.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}
