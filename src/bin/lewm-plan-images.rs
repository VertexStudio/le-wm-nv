use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

use anyhow::Context;
use candle::{DType, Device, Tensor};
use clap::{Parser, ValueEnum};
use le_wm_nv::media::nvjpeg::{NvJpegDecoder, NvJpegImageInfo};
use le_wm_nv::{
    checkpoint,
    media::{ImageHistoryPreprocessor, ImagePreprocess},
    models::lewm::{LeWm, LeWmConfig},
    planner::{
        CandidateScorer, CemConfig, CemPlanner, IcemConfig, IcemPlanner, LeWmGoalScorer,
        MppiConfig, MppiPlanner, PlanDeviceResult,
    },
    runtime::{DTypeSpec, DeviceSpec},
    session::LeWmSession,
};
use serde_json::json;

#[cfg(feature = "hub")]
use le_wm_nv::hub;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum PlannerArg {
    Cem,
    Mppi,
    Icem,
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    current: Vec<PathBuf>,

    #[arg(long)]
    goal: Vec<PathBuf>,

    #[arg(long)]
    output: PathBuf,

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

    #[arg(long, value_enum, default_value_t = PlannerArg::Icem)]
    planner: PlannerArg,

    #[arg(long)]
    horizon: Option<usize>,

    #[arg(long)]
    history_size: Option<usize>,

    #[arg(long, default_value_t = 1024)]
    samples: usize,

    #[arg(long)]
    elites: Option<usize>,

    #[arg(long, default_value_t = 5)]
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

    #[arg(long, default_value_t = 0)]
    warmup: usize,

    #[arg(long, default_value_t = 1)]
    iters: usize,
}

#[derive(Debug, Clone)]
struct DecodedHistory {
    pixels: Tensor,
    info: NvJpegImageInfo,
    image_names: Vec<String>,
    original_paths: Vec<String>,
}

#[derive(Debug, Clone)]
struct ScoreStats {
    best: f32,
    mean: f32,
    p50: f32,
    p95: f32,
    min: f32,
    max: f32,
    values: Vec<f32>,
}

#[derive(Debug, Clone)]
struct BenchStats {
    mean_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    validate_args(&args)?;

    let device = args.device.resolve()?;
    let dtype = args.dtype.dtype();
    let preprocess = ImagePreprocess::imagenet_224();
    let total_started = Instant::now();

    let load_started = Instant::now();
    let (weights, config) = resolve_files(&args)?;
    let cfg = match config.as_ref() {
        Some(path) => LeWmConfig::from_json_file(path)?,
        None => LeWmConfig::tiny_patch14_224(2),
    };
    let checkpoint_history = cfg.history_size;
    let history = args.history_size.unwrap_or(checkpoint_history);
    if history == 0 {
        anyhow::bail!("--history-size must be greater than zero");
    }
    let action_dim = cfg.action_encoder.input_dim;
    let horizon = args.horizon.unwrap_or_else(|| history.max(5));
    if horizon < history {
        anyhow::bail!("--horizon {horizon} must be >= input history size {history}");
    }
    let elites = args
        .elites
        .unwrap_or_else(|| (args.samples / 4).clamp(2, args.samples));
    if elites > args.samples {
        anyhow::bail!("--elites cannot exceed --samples");
    }

    let vb = checkpoint::var_builder_from_path(&weights, dtype, &device)?;
    let model = LeWm::new(cfg, vb)?;
    device.synchronize()?;
    let load_ms = duration_ms(load_started.elapsed());

    let output_dir = args
        .output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(output_dir)?;
    let current_names = copy_images("current", &args.current, output_dir)?;
    let goal_names = copy_images("goal", &args.goal, output_dir)?;

    let mut decoder = NvJpegDecoder::new(&device)?;
    let (current, current_decode_stats) =
        timed_cuda_bench(&device, args.warmup, args.iters, || {
            decode_history(
                "current",
                &args.current,
                &current_names,
                history,
                &mut decoder,
                &device,
                preprocess,
            )
        })?;
    let (goal, goal_decode_stats) = timed_cuda_bench(&device, args.warmup, args.iters, || {
        decode_history(
            "goal",
            &args.goal,
            &goal_names,
            history,
            &mut decoder,
            &device,
            preprocess,
        )
    })?;

    let mut session = LeWmSession::new(model, device.clone(), dtype);
    let (current_emb, current_encode_stats) =
        timed_cuda_bench(&device, args.warmup, args.iters, || {
            session
                .reset_pixels(&current.pixels)
                .map_err(anyhow::Error::from)
        })?;
    let (goal_emb, goal_encode_stats) = timed_cuda_bench(&device, args.warmup, args.iters, || {
        session
            .encode_pixels(&goal.pixels)
            .map_err(anyhow::Error::from)
    })?;
    let scorer = LeWmGoalScorer::new(&session, &goal_emb);

    let (plan_device_result, plan_stats) =
        timed_cuda_bench(&device, args.warmup, args.iters, || {
            run_planner(&args, &scorer, horizon, args.samples, elites, action_dim)
        })?;
    let selected_sequence = plan_device_result.sequence.clone();
    let selected_sequence_for_score = selected_sequence.unsqueeze(1)?;
    let (selected_cost_tensor, selected_score_stats) =
        timed_cuda_bench(&device, args.warmup, args.iters, || {
            scorer
                .score_candidates(&selected_sequence_for_score)
                .map_err(anyhow::Error::from)
        })?;
    let plan_result = plan_device_result.materialize()?;

    let first_action = plan_result
        .first_action
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?;
    let sequence = selected_sequence
        .to_dtype(DType::F32)?
        .to_vec3::<f32>()?
        .into_iter()
        .next()
        .context("planner produced empty batch")?;
    let selected_cost = selected_cost_tensor
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?
        .into_iter()
        .next()
        .and_then(|row| row.into_iter().next())
        .context("selected cost tensor was empty")?;
    let score_stats = score_stats(&plan_result.scores)?;
    let total_ms = duration_ms(total_started.elapsed());

    let benchmark_stats = benchmark_stats_json([
        ("current_decode_preprocess", &current_decode_stats),
        ("goal_decode_preprocess", &goal_decode_stats),
        ("current_encode", &current_encode_stats),
        ("goal_encode", &goal_encode_stats),
        ("planning", &plan_stats),
        ("selected_score", &selected_score_stats),
    ]);
    let timing = json!({
        "checkpoint_load": load_ms,
        "current_decode_preprocess": current_decode_stats.p50_ms,
        "goal_decode_preprocess": goal_decode_stats.p50_ms,
        "current_encode": current_encode_stats.p50_ms,
        "goal_encode": goal_encode_stats.p50_ms,
        "planning": plan_stats.p50_ms,
        "selected_score": selected_score_stats.p50_ms,
        "total": total_ms,
    });
    let payload = json!({
        "git_commit": git_commit(),
        "upstream_stable_worldmodel_commit": "40dff37fc983c5276ada65eb1c7873cefbcccd8a",
        "hf_repo": args.hf_repo,
        "weights": weights.display().to_string(),
        "config": config.as_ref().map(|path| path.display().to_string()),
        "device": args.device.to_string(),
        "dtype": args.dtype.to_string(),
        "planner": format!("{:?}", args.planner).to_ascii_lowercase(),
        "history_size": history,
        "checkpoint_history_size": checkpoint_history,
        "horizon": horizon,
        "samples": args.samples,
        "elites": elites,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "iters": args.iters,
        "action_dim": action_dim,
        "embedding_shape": current_emb.dims(),
        "goal_embedding_shape": goal_emb.dims(),
        "preprocess": {
            "output_height": preprocess.output_height,
            "output_width": preprocess.output_width,
            "mean": preprocess.mean,
            "std": preprocess.std,
        },
        "current_images": current.original_paths,
        "goal_images": goal.original_paths,
        "current_image_info": image_info_json(current.info),
        "goal_image_info": image_info_json(goal.info),
        "timing_ms": timing,
        "benchmark_stats": benchmark_stats,
        "score": {
            "selected_cost": selected_cost,
            "final_best": score_stats.best,
            "final_mean": score_stats.mean,
            "final_p50": score_stats.p50,
            "final_p95": score_stats.p95,
            "final_min": score_stats.min,
            "final_max": score_stats.max,
            "best_indices": plan_result.best_indices,
            "iterations_completed": plan_result.iterations_completed,
            "deadline_reached": plan_result.deadline_reached,
            "used_host_elite_selection": plan_result.used_host_elite_selection,
        },
        "first_action": first_action,
        "sequence": sequence,
    });

    let json_output = args.output.with_extension("json");
    fs::write(&json_output, serde_json::to_string_pretty(&payload)?)?;
    fs::write(
        &args.output,
        render_html(
            &args,
            &current,
            &goal,
            &sequence,
            &score_stats,
            selected_cost,
            &payload,
        )?,
    )?;

    println!("report={}", args.output.display());
    println!("json={}", json_output.display());
    println!(
        "planner={:?} selected_cost={:.6} final_best={:.6} plan_ms={:.3}",
        args.planner, selected_cost, score_stats.best, plan_stats.p50_ms
    );

    Ok(())
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    if args.current.is_empty() {
        anyhow::bail!("provide at least one --current JPEG");
    }
    if args.goal.is_empty() {
        anyhow::bail!("provide at least one --goal JPEG");
    }
    if args.samples < 2 {
        anyhow::bail!("--samples must be at least 2");
    }
    if args.iterations == 0 {
        anyhow::bail!("--iterations must be greater than zero");
    }
    if args.iters == 0 {
        anyhow::bail!("--iters must be greater than zero");
    }
    Ok(())
}

fn decode_history(
    label: &str,
    paths: &[PathBuf],
    image_names: &[String],
    history: usize,
    decoder: &mut NvJpegDecoder,
    device: &Device,
    preprocess: ImagePreprocess,
) -> anyhow::Result<DecodedHistory> {
    let first_bytes = fs::read(&paths[0])
        .with_context(|| format!("failed to read {label} image {}", paths[0].display()))?;
    let info = decoder.image_info(&first_bytes)?;
    let rgb_output = decoder.alloc_rgb_interleaved(info)?;
    let mut preprocessor =
        ImageHistoryPreprocessor::new(device, info.packed_rgb_shape(), history, preprocess)?;

    for slot in 0..history {
        let idx = history_path_index(paths, history, slot, label)?;
        let encoded = if idx == 0 {
            first_bytes.clone()
        } else {
            fs::read(&paths[idx])
                .with_context(|| format!("failed to read {label} image {}", paths[idx].display()))?
        };
        decoder.decode_preprocessed_history_slot_into(
            &encoded,
            &rgb_output,
            slot,
            &mut preprocessor,
        )?;
    }

    Ok(DecodedHistory {
        pixels: preprocessor.output().clone(),
        info,
        image_names: image_names.to_vec(),
        original_paths: paths
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
    })
}

fn history_path_index(
    paths: &[PathBuf],
    history: usize,
    slot: usize,
    label: &str,
) -> anyhow::Result<usize> {
    if paths.len() == 1 {
        return Ok(0);
    }
    if paths.len() != history {
        anyhow::bail!(
            "--{label} accepts either one JPEG or exactly history_size ({history}) JPEGs, got {}",
            paths.len()
        );
    }
    Ok(slot)
}

fn run_planner(
    args: &Args,
    scorer: &LeWmGoalScorer<'_>,
    horizon: usize,
    samples: usize,
    elites: usize,
    action_dim: usize,
) -> anyhow::Result<PlanDeviceResult> {
    match args.planner {
        PlannerArg::Cem => {
            let mut cfg = CemConfig::new(horizon, samples, elites, action_dim);
            cfg.iterations = args.iterations;
            cfg.init_std = args.init_std;
            cfg.min_std = args.min_std;
            cfg.seed = args.seed;
            CemPlanner::new(cfg)
                .plan_device(scorer)
                .map_err(anyhow::Error::from)
        }
        PlannerArg::Mppi => {
            let mut cfg = MppiConfig::new(horizon, samples, action_dim);
            cfg.iterations = args.iterations;
            cfg.noise_std = args.noise_std;
            cfg.temperature = args.temperature;
            cfg.seed = args.seed;
            MppiPlanner::new(cfg)
                .plan_device(scorer)
                .map_err(anyhow::Error::from)
        }
        PlannerArg::Icem => {
            let mut cfg = IcemConfig::new(horizon, samples, elites, action_dim);
            cfg.iterations = args.iterations;
            cfg.keep_elites = elites.min(samples);
            cfg.init_std = args.init_std;
            cfg.min_std = args.min_std;
            cfg.seed = args.seed;
            let mut planner = IcemPlanner::new(cfg);
            planner.plan_device(scorer).map_err(anyhow::Error::from)
        }
    }
}

fn score_stats(scores: &Tensor) -> anyhow::Result<ScoreStats> {
    let mut values = scores
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    if values.is_empty() {
        anyhow::bail!("planner produced empty score tensor");
    }
    for value in &values {
        if !value.is_finite() {
            anyhow::bail!("planner score contains non-finite value {value}");
        }
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let sum = values.iter().copied().sum::<f32>();
    let best = values[0];
    let max = *values.last().unwrap_or(&best);
    Ok(ScoreStats {
        best,
        mean: sum / values.len() as f32,
        p50: percentile(&values, 0.50),
        p95: percentile(&values, 0.95),
        min: best,
        max,
        values,
    })
}

fn percentile(sorted: &[f32], q: f32) -> f32 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = ((sorted.len() - 1) as f32 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn timed_cuda_bench<T>(
    device: &Device,
    warmup: usize,
    iters: usize,
    mut op: impl FnMut() -> anyhow::Result<T>,
) -> anyhow::Result<(T, BenchStats)> {
    for _ in 0..warmup {
        op()?;
    }
    device.synchronize()?;

    let mut samples = Vec::with_capacity(iters);
    let mut value = None;
    for _ in 0..iters {
        device.synchronize()?;
        let started = Instant::now();
        value = Some(op()?);
        device.synchronize()?;
        samples.push(started.elapsed());
    }
    let value = value.context("benchmark produced no timed value")?;
    Ok((value, bench_stats(samples)))
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn duration_ms_ref(duration: &Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn bench_stats(mut samples: Vec<Duration>) -> BenchStats {
    samples.sort_unstable();
    let total_ms = samples.iter().map(duration_ms_ref).sum::<f64>();
    BenchStats {
        mean_ms: total_ms / samples.len() as f64,
        p50_ms: duration_ms_ref(&samples[percentile_index(samples.len(), 0.50)]),
        p95_ms: duration_ms_ref(&samples[percentile_index(samples.len(), 0.95)]),
        p99_ms: duration_ms_ref(&samples[percentile_index(samples.len(), 0.99)]),
    }
}

fn percentile_index(len: usize, percentile: f64) -> usize {
    let idx = ((len.saturating_sub(1)) as f64 * percentile).ceil() as usize;
    idx.min(len.saturating_sub(1))
}

fn benchmark_stats_json<'a>(
    rows: impl IntoIterator<Item = (&'a str, &'a BenchStats)>,
) -> serde_json::Value {
    let stats = rows
        .into_iter()
        .map(|(name, stats)| {
            (
                name.to_string(),
                json!({
                    "mean_ms": stats.mean_ms,
                    "p50_ms": stats.p50_ms,
                    "p95_ms": stats.p95_ms,
                    "p99_ms": stats.p99_ms,
                }),
            )
        })
        .collect::<BTreeMap<_, _>>();
    json!(stats)
}

fn copy_images(prefix: &str, paths: &[PathBuf], output_dir: &Path) -> anyhow::Result<Vec<String>> {
    paths
        .iter()
        .enumerate()
        .map(|(idx, path)| {
            let name = format!("{prefix}-{idx:02}.jpg");
            fs::copy(path, output_dir.join(&name))
                .with_context(|| format!("failed to copy image {}", path.display()))?;
            Ok(name)
        })
        .collect()
}

fn image_info_json(info: NvJpegImageInfo) -> serde_json::Value {
    json!({
        "width": info.width,
        "height": info.height,
        "components": info.components,
    })
}

fn render_html(
    args: &Args,
    current: &DecodedHistory,
    goal: &DecodedHistory,
    sequence: &[Vec<f32>],
    scores: &ScoreStats,
    selected_cost: f32,
    payload: &serde_json::Value,
) -> anyhow::Result<String> {
    let timing = &payload["timing_ms"];
    let title = "LeWM PushT CUDA Plan";
    let json_name = args
        .output
        .with_extension("json")
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "plan.json".to_string());
    Ok(format!(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: dark; --bg: #0f1115; --panel: #171b22; --text: #f0f3f7; --muted: #9aa4b2; --line: #2c3340; --accent: #37d67a; --warn: #f5b84b; }}
body {{ margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
h1 {{ font-size: 28px; margin: 0 0 6px; }}
h2 {{ font-size: 16px; margin: 0 0 12px; }}
.sub {{ color: var(--muted); margin: 0 0 24px; }}
.grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
.panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }}
.images {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }}
img {{ width: 100%; border-radius: 6px; display: block; background: #080a0d; }}
.metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }}
.metric {{ border: 1px solid var(--line); border-radius: 6px; padding: 10px; }}
.label {{ color: var(--muted); font-size: 12px; }}
.value {{ font-size: 20px; font-weight: 650; margin-top: 3px; }}
svg {{ width: 100%; height: auto; display: block; }}
pre {{ overflow: auto; background: #0a0d12; border: 1px solid var(--line); border-radius: 6px; padding: 12px; color: #dbe4ef; }}
.wide {{ grid-column: 1 / -1; }}
@media (max-width: 800px) {{ main {{ padding: 16px; }} .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<main>
<h1>{title}</h1>
<p class="sub">stable-worldmodel LeWM checkpoint, JPEG decode through nvJPEG, Candle CUDA encode/rollout/scoring, Rust planner output. JSON: {json_name}</p>
<section class="grid">
<div class="panel">
<h2>Current Image History</h2>
{current_images}
</div>
<div class="panel">
<h2>Goal Image History</h2>
{goal_images}
</div>
<div class="panel wide">
<h2>Runtime</h2>
<div class="metric-grid">
{metrics}
</div>
</div>
<div class="panel wide">
<h2>Selected Action Sequence</h2>
{action_svg}
</div>
<div class="panel wide">
<h2>Final Candidate Cost Distribution</h2>
{cost_svg}
</div>
<div class="panel wide">
<h2>First Action</h2>
<pre>{first_action}</pre>
</div>
</section>
</main>
</body>
</html>"#,
        current_images = render_images(&current.image_names),
        goal_images = render_images(&goal.image_names),
        metrics = render_metrics(&[
            ("selected cost", selected_cost),
            ("final best", scores.best),
            ("final mean", scores.mean),
            (
                "load ms",
                timing["checkpoint_load"].as_f64().unwrap_or_default() as f32
            ),
            (
                "current media ms",
                timing["current_decode_preprocess"]
                    .as_f64()
                    .unwrap_or_default() as f32,
            ),
            (
                "goal media ms",
                timing["goal_decode_preprocess"]
                    .as_f64()
                    .unwrap_or_default() as f32,
            ),
            (
                "current encode ms",
                timing["current_encode"].as_f64().unwrap_or_default() as f32,
            ),
            (
                "goal encode ms",
                timing["goal_encode"].as_f64().unwrap_or_default() as f32,
            ),
            (
                "planning ms",
                timing["planning"].as_f64().unwrap_or_default() as f32,
            ),
        ]),
        action_svg = render_action_svg(sequence),
        cost_svg = render_cost_svg(&scores.values, selected_cost),
        first_action = html_escape(&serde_json::to_string_pretty(&payload["first_action"])?),
    ))
}

fn render_images(names: &[String]) -> String {
    names
        .iter()
        .map(|name| {
            format!(
                r#"<div><img src="{}" alt="{}"></div>"#,
                html_escape(name),
                html_escape(name)
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_metrics(metrics: &[(&str, f32)]) -> String {
    metrics
        .iter()
        .map(|(label, value)| {
            format!(
                r#"<div class="metric"><div class="label">{}</div><div class="value">{:.3}</div></div>"#,
                html_escape(label),
                value
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_action_svg(sequence: &[Vec<f32>]) -> String {
    let width = 980.0f32;
    let height = 300.0f32;
    let left = 42.0f32;
    let right = 16.0f32;
    let top = 16.0f32;
    let bottom = 32.0f32;
    let plot_w = width - left - right;
    let plot_h = height - top - bottom;
    let horizon = sequence.len().max(1);
    let action_dim = sequence.first().map(|row| row.len()).unwrap_or(0);
    let colors = [
        "#37d67a", "#5ec8ff", "#f5b84b", "#ff6b8a", "#b18cff", "#78e0d4", "#e3d55d", "#ff9f5a",
        "#9ae66e", "#c0c7d6",
    ];
    let mut lines = Vec::new();
    for dim in 0..action_dim {
        let mut points = Vec::new();
        for (t, row) in sequence.iter().enumerate() {
            let x = if horizon == 1 {
                left
            } else {
                left + (t as f32 / (horizon - 1) as f32) * plot_w
            };
            let value = row.get(dim).copied().unwrap_or_default().clamp(-1.0, 1.0);
            let y = top + ((1.0 - (value + 1.0) * 0.5) * plot_h);
            points.push(format!("{x:.1},{y:.1}"));
        }
        lines.push(format!(
            r#"<polyline fill="none" stroke="{}" stroke-width="2" points="{}"/>"#,
            colors[dim % colors.len()],
            points.join(" ")
        ));
    }
    format!(
        r##"<svg viewBox="0 0 {width} {height}" role="img" aria-label="Selected action sequence">
<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#0a0d12" stroke="#2c3340"/>
<line x1="{left}" y1="{mid}" x2="{right_x}" y2="{mid}" stroke="#3a4250"/>
<text x="8" y="{top}" fill="#9aa4b2" font-size="12">+1</text>
<text x="8" y="{mid}" fill="#9aa4b2" font-size="12">0</text>
<text x="8" y="{bottom_y}" fill="#9aa4b2" font-size="12">-1</text>
{lines}
</svg>"##,
        mid = top + plot_h / 2.0,
        right_x = left + plot_w,
        bottom_y = top + plot_h,
        lines = lines.join("\n")
    )
}

fn render_cost_svg(values: &[f32], selected_cost: f32) -> String {
    let width = 980.0f32;
    let height = 260.0f32;
    let left = 42.0f32;
    let right = 16.0f32;
    let top = 16.0f32;
    let bottom = 32.0f32;
    let plot_w = width - left - right;
    let plot_h = height - top - bottom;
    let bins = 32usize;
    let min = values.first().copied().unwrap_or(0.0);
    let max = values.last().copied().unwrap_or(min + 1.0);
    let span = (max - min).max(1e-6);
    let mut counts = vec![0usize; bins];
    for value in values {
        let idx = (((*value - min) / span) * (bins - 1) as f32).round() as usize;
        counts[idx.min(bins - 1)] += 1;
    }
    let max_count = counts.iter().copied().max().unwrap_or(1).max(1) as f32;
    let bar_w = plot_w / bins as f32;
    let bars = counts
        .iter()
        .enumerate()
        .map(|(idx, count)| {
            let h = (*count as f32 / max_count) * plot_h;
            let x = left + idx as f32 * bar_w;
            let y = top + plot_h - h;
            format!(
            r##"<rect x="{x:.1}" y="{y:.1}" width="{w:.1}" height="{h:.1}" fill="#37d67a" opacity="0.72"/>"##,
                w = (bar_w - 2.0).max(1.0)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let selected_x = left + (((selected_cost - min) / span).clamp(0.0, 1.0) * plot_w);
    format!(
        r##"<svg viewBox="0 0 {width} {height}" role="img" aria-label="Candidate cost distribution">
<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#0a0d12" stroke="#2c3340"/>
{bars}
<line x1="{selected_x:.1}" y1="{top}" x2="{selected_x:.1}" y2="{bottom_y}" stroke="#f5b84b" stroke-width="3"/>
<text x="{left}" y="{text_y}" fill="#9aa4b2" font-size="12">min {min:.3}</text>
<text x="{max_x}" y="{text_y}" fill="#9aa4b2" font-size="12" text-anchor="end">max {max:.3}</text>
</svg>"##,
        bottom_y = top + plot_h,
        text_y = height - 8.0,
        max_x = left + plot_w
    )
}

fn html_escape(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
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
