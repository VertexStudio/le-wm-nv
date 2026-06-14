use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, ensure};
use clap::{Parser, ValueEnum};
use le_wm_nv::data::drone_racing::{
    DRONE_ACTION_DIM, DroneBatchConfig, DroneRacingDataset, RunningStats, mat3_mul,
    mat3_t_mul_vec3, mat3_transpose, norm3, rotvec_from_mat3, sub3,
};
use serde::Serialize;

#[derive(Parser, Debug)]
struct Args {
    /// Imported drone dataset directory containing data.h5 and metadata.json.
    #[arg(long)]
    dataset_dir: Option<PathBuf>,

    /// Output JSON report.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Rows to scan.
    #[arg(long, value_enum, default_value_t = RowSource::All)]
    row_source: RowSource,

    /// History length used by downstream drone benchmarks.
    #[arg(long, default_value_t = 8)]
    history_steps: usize,

    /// Future rows that must remain in the same episode and are used for dataset-only scoring.
    #[arg(long, default_value_t = 40)]
    horizon_steps: usize,

    /// Number of rows to keep per category.
    #[arg(long, default_value_t = 12)]
    top_k: usize,

    /// Scan every Nth valid row.
    #[arg(long, default_value_t = 1)]
    stride: usize,
}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum RowSource {
    All,
    Train,
    Eval,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    validate_args(&args)?;

    let dataset_dir = args.dataset_dir.clone().unwrap_or_else(default_dataset_dir);
    let output = args.output.clone().unwrap_or_else(default_output);
    let batch_cfg = DroneBatchConfig {
        batch_size: 1,
        sequence_steps: args.history_steps.max(2),
        normalize_observations: false,
        normalize_actions: false,
        normalize_targets: false,
    };
    let dataset = DroneRacingDataset::open(&dataset_dir, batch_cfg)?;
    let rows = match args.row_source {
        RowSource::All => dataset.valid_rows().to_vec(),
        RowSource::Train => dataset.train_rows(),
        RowSource::Eval => dataset.eval_rows(),
    };
    ensure!(
        !rows.is_empty(),
        "no rows selected for row_source={:?}",
        args.row_source
    );
    let action_trim = baseline_action(&dataset.metadata().normalization.action)?;
    let mut metrics = Vec::with_capacity(rows.len() / args.stride.max(1));
    let mut skipped = 0usize;

    for (idx, row) in rows.iter().copied().enumerate() {
        if idx % args.stride != 0 {
            continue;
        }
        match row_metric(
            &dataset,
            row,
            args.history_steps,
            args.horizon_steps,
            action_trim,
        ) {
            Ok(Some(metric)) => metrics.push(metric),
            Ok(None) => skipped += 1,
            Err(err) => {
                skipped += 1;
                eprintln!("skipped row={row}: {err:#}");
            }
        }
    }
    ensure!(
        !metrics.is_empty(),
        "no candidate rows remain after horizon/episode filtering"
    );

    let categories = vec![
        CategoryReport::new(
            "hover_like",
            "Low-speed, low-angular-rate, low-future-path rows for hold/drift checks.",
            true,
            top_by(&metrics, args.top_k, Direction::Ascending, hover_like_score),
        ),
        CategoryReport::new(
            "clean_cruise",
            "Moderate-speed, low-angular-rate rows for in-distribution local-control starts.",
            true,
            top_by(
                &metrics,
                args.top_k,
                Direction::Ascending,
                clean_cruise_score,
            ),
        ),
        CategoryReport::new(
            "body_x_motion",
            "Rows whose recorded future moves mostly along the current body X axis.",
            false,
            top_by(
                &metrics,
                args.top_k,
                Direction::Descending,
                body_x_motion_score,
            ),
        ),
        CategoryReport::new(
            "body_y_motion",
            "Rows whose recorded future moves mostly along the current body Y axis.",
            false,
            top_by(
                &metrics,
                args.top_k,
                Direction::Descending,
                body_y_motion_score,
            ),
        ),
        CategoryReport::new(
            "body_z_motion",
            "Rows whose recorded future moves mostly along the current body Z axis.",
            false,
            top_by(
                &metrics,
                args.top_k,
                Direction::Descending,
                body_z_motion_score,
            ),
        ),
        CategoryReport::new(
            "yaw_motion",
            "Rows with meaningful future body yaw rotation.",
            false,
            top_by(
                &metrics,
                args.top_k,
                Direction::Descending,
                yaw_motion_score,
            ),
        ),
        CategoryReport::new(
            "action_excitation",
            "Rows with larger recorded action variation over the scoring horizon.",
            false,
            top_by(
                &metrics,
                args.top_k,
                Direction::Descending,
                action_excitation_score,
            ),
        ),
        CategoryReport::new(
            "stress_motion",
            "High-motion rows for rollout stress tests, not isolated local-control starts.",
            false,
            top_by(
                &metrics,
                args.top_k,
                Direction::Descending,
                stress_motion_score,
            ),
        ),
    ];

    let report = RowSearchReport {
        dataset_dir,
        output: output.clone(),
        row_source: args.row_source,
        history_steps: args.history_steps,
        horizon_steps: args.horizon_steps,
        top_k: args.top_k,
        stride: args.stride,
        selected_rows: rows.len(),
        scored_rows: metrics.len(),
        skipped_rows: skipped,
        sample_rate_hz: dataset.metadata().sample_rate_hz,
        action_trim,
        categories,
    };
    write_pretty_json(&output, &report)?;
    print_report(&report);
    println!("wrote {}", output.display());
    Ok(())
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    ensure!(
        args.history_steps >= 2,
        "--history-steps must be at least two"
    );
    ensure!(
        args.horizon_steps > 0,
        "--horizon-steps must be greater than zero"
    );
    ensure!(args.top_k > 0, "--top-k must be greater than zero");
    ensure!(args.stride > 0, "--stride must be greater than zero");
    Ok(())
}

fn row_metric(
    dataset: &DroneRacingDataset,
    row: usize,
    history_steps: usize,
    horizon_steps: usize,
    action_trim: [f32; DRONE_ACTION_DIM],
) -> anyhow::Result<Option<RowMetric>> {
    let current_row = row + history_steps - 1;
    let future_end_row = current_row + horizon_steps;
    if future_end_row >= dataset.metadata().rows {
        return Ok(None);
    }
    let history_start = dataset.frame(row)?;
    let current = dataset.frame(current_row)?;
    let future_end = dataset.frame(future_end_row)?;
    if history_start.episode_idx != current.episode_idx
        || current.episode_idx != future_end.episode_idx
    {
        return Ok(None);
    }

    let mut path_length_m = 0.0f32;
    let mut prev = current.clone();
    let mut action_sum_abs = [0.0f32; DRONE_ACTION_DIM];
    let mut action_min = [f32::INFINITY; DRONE_ACTION_DIM];
    let mut action_max = [f32::NEG_INFINITY; DRONE_ACTION_DIM];
    for step in 0..=horizon_steps {
        let frame = dataset.frame(current_row + step)?;
        if frame.episode_idx != current.episode_idx {
            return Ok(None);
        }
        if step > 0 {
            path_length_m += norm3(sub3(frame.pos_world, prev.pos_world));
        }
        for idx in 0..DRONE_ACTION_DIM {
            let value = frame.channels_norm[idx];
            action_sum_abs[idx] += value.abs();
            action_min[idx] = action_min[idx].min(value);
            action_max[idx] = action_max[idx].max(value);
        }
        prev = frame;
    }
    let denom = (horizon_steps + 1) as f32;
    let mut action_abs_mean = [0.0f32; DRONE_ACTION_DIM];
    let mut action_span = [0.0f32; DRONE_ACTION_DIM];
    for idx in 0..DRONE_ACTION_DIM {
        action_abs_mean[idx] = action_sum_abs[idx] / denom;
        action_span[idx] = action_max[idx] - action_min[idx];
    }

    let future_delta_world = sub3(future_end.pos_world, current.pos_world);
    let future_body_delta = mat3_t_mul_vec3(current.rotmat_world_from_body, future_delta_world);
    let future_rotvec_body = rotvec_from_mat3(mat3_mul(
        mat3_transpose(current.rotmat_world_from_body),
        future_end.rotmat_world_from_body,
    ));
    let speed_mps = norm3(current.lin_vel_body);
    let angular_speed_radps = norm3(current.ang_vel_body);
    let future_speed_delta_mps = norm3(future_end.lin_vel_body) - speed_mps;
    let action_dev_norm = action_distance(current.channels_norm, action_trim);
    let action_span_norm = action_distance(action_max, action_min);
    let future_net_m = norm3(future_delta_world);
    ensure!(
        finite_metric([
            path_length_m,
            future_net_m,
            speed_mps,
            angular_speed_radps,
            future_speed_delta_mps,
            action_dev_norm,
            action_span_norm,
        ]),
        "non-finite metric"
    );

    Ok(Some(RowMetric {
        row,
        current_row,
        future_end_row,
        episode_idx: current.episode_idx,
        step_idx: current.step_idx,
        dt: current.dt,
        pos_world: current.pos_world,
        rotmat_world_from_body: current.rotmat_world_from_body,
        lin_vel_body: current.lin_vel_body,
        ang_vel_body: current.ang_vel_body,
        vbat: current.vbat,
        channels_norm: current.channels_norm,
        speed_mps,
        angular_speed_radps,
        action_dev_norm,
        action_abs_mean,
        action_span,
        action_span_norm,
        future_path_m: path_length_m,
        future_net_m,
        future_delta_world,
        future_body_delta,
        future_rotvec_body,
        future_speed_delta_mps,
    }))
}

fn hover_like_score(metric: &RowMetric) -> f32 {
    metric.speed_mps
        + 0.65 * metric.angular_speed_radps
        + 0.25 * metric.action_dev_norm
        + 0.18 * metric.future_path_m
        + 0.05 * metric.future_body_delta[2].abs()
}

fn clean_cruise_score(metric: &RowMetric) -> f32 {
    (metric.speed_mps - 0.8).abs()
        + 0.45 * metric.angular_speed_radps
        + 0.15 * metric.action_dev_norm
        + 0.08 * metric.future_path_m
        + 0.05 * norm3(metric.future_rotvec_body)
}

fn body_x_motion_score(metric: &RowMetric) -> f32 {
    axis_motion_score(metric.future_body_delta, 0) - 0.08 * metric.angular_speed_radps
}

fn body_y_motion_score(metric: &RowMetric) -> f32 {
    axis_motion_score(metric.future_body_delta, 1) - 0.08 * metric.angular_speed_radps
}

fn body_z_motion_score(metric: &RowMetric) -> f32 {
    axis_motion_score(metric.future_body_delta, 2) - 0.06 * metric.angular_speed_radps
}

fn axis_motion_score(delta: [f32; 3], axis: usize) -> f32 {
    let primary = delta[axis].abs();
    let cross = (0..3)
        .filter(|idx| *idx != axis)
        .map(|idx| delta[idx].abs())
        .sum::<f32>();
    primary - 0.35 * cross
}

fn yaw_motion_score(metric: &RowMetric) -> f32 {
    metric.future_rotvec_body[2].abs() + 0.15 * metric.ang_vel_body[2].abs()
        - 0.08 * metric.speed_mps
        - 0.04 * (metric.future_rotvec_body[0].abs() + metric.future_rotvec_body[1].abs())
}

fn action_excitation_score(metric: &RowMetric) -> f32 {
    metric.action_span_norm + 0.5 * metric.action_dev_norm
}

fn stress_motion_score(metric: &RowMetric) -> f32 {
    metric.future_path_m + 0.25 * metric.speed_mps + 0.12 * metric.angular_speed_radps
}

fn top_by<F>(
    metrics: &[RowMetric],
    top_k: usize,
    direction: Direction,
    score_fn: F,
) -> Vec<RankedCandidate>
where
    F: Fn(&RowMetric) -> f32,
{
    let mut scored = metrics
        .iter()
        .filter_map(|metric| {
            let score = score_fn(metric);
            score.is_finite().then_some((score, metric.clone()))
        })
        .collect::<Vec<_>>();
    match direction {
        Direction::Ascending => {
            scored.sort_by(|lhs, rhs| lhs.0.total_cmp(&rhs.0));
        }
        Direction::Descending => {
            scored.sort_by(|lhs, rhs| rhs.0.total_cmp(&lhs.0));
        }
    }
    scored
        .into_iter()
        .take(top_k)
        .enumerate()
        .map(|(idx, (score, metric))| RankedCandidate {
            rank: idx + 1,
            score,
            metric,
        })
        .collect()
}

fn action_distance(lhs: [f32; DRONE_ACTION_DIM], rhs: [f32; DRONE_ACTION_DIM]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(l, r)| {
            let value = l - r;
            value * value
        })
        .sum::<f32>()
        .sqrt()
}

fn finite_metric(values: impl IntoIterator<Item = f32>) -> bool {
    values.into_iter().all(f32::is_finite)
}

fn baseline_action(stats: &RunningStats) -> anyhow::Result<[f32; DRONE_ACTION_DIM]> {
    ensure!(
        stats.mean.len() == DRONE_ACTION_DIM,
        "action mean length {} does not match action dim {DRONE_ACTION_DIM}",
        stats.mean.len()
    );
    Ok([
        stats.mean[0].clamp(-1.0, 1.0),
        stats.mean[1].clamp(-1.0, 1.0),
        stats.mean[2].clamp(0.0, 1.0),
        stats.mean[3].clamp(-1.0, 1.0),
    ])
}

fn print_report(report: &RowSearchReport) {
    println!(
        "row search source={:?} selected={} scored={} skipped={} history={} horizon={}",
        report.row_source,
        report.selected_rows,
        report.scored_rows,
        report.skipped_rows,
        report.history_steps,
        report.horizon_steps
    );
    for category in &report.categories {
        println!(
            "category={} prefer_low_score={}",
            category.name, category.prefer_low_score
        );
        for candidate in category.candidates.iter().take(5) {
            let metric = &candidate.metric;
            println!(
                "  #{:<2} row={:<6} current={:<6} ep={:<2} step={:<5} score={:>8.4} speed={:>6.3} ang={:>6.3} path={:>6.3} body=({:>6.3},{:>6.3},{:>6.3}) rot=({:>6.3},{:>6.3},{:>6.3})",
                candidate.rank,
                metric.row,
                metric.current_row,
                metric.episode_idx,
                metric.step_idx,
                candidate.score,
                metric.speed_mps,
                metric.angular_speed_radps,
                metric.future_path_m,
                metric.future_body_delta[0],
                metric.future_body_delta[1],
                metric.future_body_delta[2],
                metric.future_rotvec_body[0],
                metric.future_rotvec_body[1],
                metric.future_rotvec_body[2],
            );
        }
    }
}

fn write_pretty_json<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(value)?;
    fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))
}

fn default_dataset_dir() -> PathBuf {
    home_dir()
        .join(".stable_worldmodel")
        .join("le-wm-nv-data")
        .join("drone-racing-autonomous-100hz")
}

fn default_output() -> PathBuf {
    PathBuf::from("target/drone-eval/row-candidates-h40.json")
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

#[derive(Debug, Clone, Copy)]
enum Direction {
    Ascending,
    Descending,
}

#[derive(Debug, Serialize)]
struct RowSearchReport {
    dataset_dir: PathBuf,
    output: PathBuf,
    row_source: RowSource,
    history_steps: usize,
    horizon_steps: usize,
    top_k: usize,
    stride: usize,
    selected_rows: usize,
    scored_rows: usize,
    skipped_rows: usize,
    sample_rate_hz: usize,
    action_trim: [f32; DRONE_ACTION_DIM],
    categories: Vec<CategoryReport>,
}

#[derive(Debug, Serialize)]
struct CategoryReport {
    name: &'static str,
    description: &'static str,
    prefer_low_score: bool,
    candidates: Vec<RankedCandidate>,
}

impl CategoryReport {
    fn new(
        name: &'static str,
        description: &'static str,
        prefer_low_score: bool,
        candidates: Vec<RankedCandidate>,
    ) -> Self {
        Self {
            name,
            description,
            prefer_low_score,
            candidates,
        }
    }
}

#[derive(Debug, Serialize)]
struct RankedCandidate {
    rank: usize,
    score: f32,
    metric: RowMetric,
}

#[derive(Debug, Clone, Serialize)]
struct RowMetric {
    row: usize,
    current_row: usize,
    future_end_row: usize,
    episode_idx: i64,
    step_idx: i64,
    dt: f32,
    pos_world: [f32; 3],
    rotmat_world_from_body: [f32; 9],
    lin_vel_body: [f32; 3],
    ang_vel_body: [f32; 3],
    vbat: f32,
    channels_norm: [f32; DRONE_ACTION_DIM],
    speed_mps: f32,
    angular_speed_radps: f32,
    action_dev_norm: f32,
    action_abs_mean: [f32; DRONE_ACTION_DIM],
    action_span: [f32; DRONE_ACTION_DIM],
    action_span_norm: f32,
    future_path_m: f32,
    future_net_m: f32,
    future_delta_world: [f32; 3],
    future_body_delta: [f32; 3],
    future_rotvec_body: [f32; 3],
    future_speed_delta_mps: f32,
}
