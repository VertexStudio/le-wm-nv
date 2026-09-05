use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, ensure};
use clap::Parser;
use hdf5::File;
use le_wm_nv::{
    data::skyjepa::{
        SKYJEPA_ACTION_DIM, SKYJEPA_STATE_DIM, SkyJepaDatasetMetadata, skyjepa_artifact_fingerprint,
    },
    skyjepa_sim::SkyJepaDomain,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser, Serialize)]
#[command(about = "Audit canonical SkyJEPA data before training")]
struct Args {
    #[arg(long)]
    dataset_dir: PathBuf,

    #[arg(long)]
    output: Option<PathBuf>,

    #[arg(long, default_value_t = 1e-3)]
    max_rotation_error: f64,

    #[arg(long, default_value_t = 0.75)]
    max_tracking_rmse_m: f64,

    #[arg(long, default_value_t = 0.05)]
    max_ground_fraction: f64,

    #[arg(long, default_value_t = 0.25)]
    max_saturation_fraction: f64,

    #[arg(long, default_value_t = 1e-3)]
    min_action_std: f64,

    /// Minimum rotor-about-collective standard deviation. This prevents
    /// domain-dependent hover thrust from masquerading as control coverage.
    #[arg(long, default_value_t = 0.05)]
    min_differential_action_std: f64,

    /// Minimum consecutive within-episode command-delta standard deviation.
    #[arg(long, default_value_t = 0.005)]
    min_action_delta_std: f64,

    /// Emit a failed report without returning a non-zero exit status.
    #[arg(long)]
    allow_fail: bool,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    audit_version: u32,
    audit_config: serde_json::Value,
    passed: bool,
    dataset_dir: PathBuf,
    artifact_sha256: String,
    metadata: SkyJepaDatasetMetadata,
    rows: usize,
    episodes: usize,
    domains: usize,
    expected_dt_seconds: f64,
    max_dt_error_seconds: f64,
    max_rotation_orthogonality_error: f64,
    determinant_min: f64,
    determinant_max: f64,
    position_tracking_rmse_m: Option<f64>,
    velocity_tracking_rmse_mps: Option<f64>,
    ground_contact_fraction: f64,
    low_command_fraction: f64,
    high_command_fraction: f64,
    state: ChannelStats,
    action: ChannelStats,
    action_differential: ChannelStats,
    action_transition_delta: ChannelStats,
    reference_state: Option<ChannelStats>,
    motor_force: Option<ChannelStats>,
    domain_coverage: Vec<DomainCoverage>,
    failures: Vec<String>,
}

#[derive(Debug, Default, Clone, Serialize)]
struct DomainCoverage {
    domain: usize,
    episodes: usize,
    hover_episodes: usize,
    moving_episodes: usize,
    differential_action_std: Vec<f64>,
    action_delta_std: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct ChannelStats {
    mean: Vec<f64>,
    std: Vec<f64>,
    min: Vec<f64>,
    max: Vec<f64>,
}

#[derive(Debug)]
struct StatsAccumulator {
    count: usize,
    mean: Vec<f64>,
    m2: Vec<f64>,
    min: Vec<f64>,
    max: Vec<f64>,
}

impl StatsAccumulator {
    fn new(dim: usize) -> Self {
        Self {
            count: 0,
            mean: vec![0.0; dim],
            m2: vec![0.0; dim],
            min: vec![f64::INFINITY; dim],
            max: vec![f64::NEG_INFINITY; dim],
        }
    }

    fn push(&mut self, row: &[f32]) {
        self.count += 1;
        let count = self.count as f64;
        for (idx, value) in row.iter().copied().map(f64::from).enumerate() {
            let delta = value - self.mean[idx];
            self.mean[idx] += delta / count;
            self.m2[idx] += delta * (value - self.mean[idx]);
            self.min[idx] = self.min[idx].min(value);
            self.max[idx] = self.max[idx].max(value);
        }
    }

    fn finish(self) -> ChannelStats {
        let denominator = self.count.saturating_sub(1).max(1) as f64;
        ChannelStats {
            mean: self.mean,
            std: self
                .m2
                .into_iter()
                .map(|value| (value / denominator).sqrt())
                .collect(),
            min: self.min,
            max: self.max,
        }
    }
}

#[derive(Debug, Deserialize)]
struct DomainRecord {
    index: usize,
    parameters: SkyJepaDomain,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let metadata_path = args.dataset_dir.join("metadata.json");
    let data_path = args.dataset_dir.join("data.h5");
    let domains_path = args.dataset_dir.join("domains.json");
    let metadata: SkyJepaDatasetMetadata = read_json(&metadata_path)?;
    metadata.validate()?;
    ensure!(
        metadata.data_h5 == PathBuf::from("data.h5"),
        "auditor requires the canonical data.h5 artifact"
    );
    let domains: Vec<DomainRecord> = read_json(&domains_path)?;
    ensure!(!domains.is_empty(), "domains.json contains no domains");
    for (expected, domain) in domains.iter().enumerate() {
        ensure!(
            domain.index == expected,
            "domains.json index {} appears at position {expected}",
            domain.index
        );
        domain.parameters.validate()?;
    }
    if let Some(expected) = metadata.domains {
        ensure!(
            expected == domains.len(),
            "metadata declares {expected} domains but domains.json has {}",
            domains.len()
        );
    }

    let file = File::open(&data_path)
        .with_context(|| format!("failed to open {}", data_path.display()))?;
    let rows = metadata.rows;
    let states = read_f32(&file, "state", rows, SKYJEPA_STATE_DIM)?;
    let actions = read_f32(&file, "action", rows, SKYJEPA_ACTION_DIM)?;
    let episodes = read_i64(&file, "episode_idx", rows)?;
    let steps = read_i64(&file, "step_idx", rows)?;
    let dt = read_f32(&file, "dt", rows, 1)?;
    let domain_idx = read_i64(&file, "domain_idx", rows)?;
    let references = read_optional_f32(
        &file,
        "reference_state",
        rows,
        SKYJEPA_STATE_DIM,
        metadata.has_reference_state,
    )?;
    let motor_forces = read_optional_f32(
        &file,
        "motor_force",
        rows,
        SKYJEPA_ACTION_DIM,
        metadata.has_motor_force,
    )?;

    ensure_finite("state", &states)?;
    ensure_finite("action", &actions)?;
    ensure_finite("dt", &dt)?;
    if let Some(values) = references.as_ref() {
        ensure_finite("reference_state", values)?;
    }
    if let Some(values) = motor_forces.as_ref() {
        ensure_finite("motor_force", values)?;
    }

    let unique_episodes = validate_sequence_index(&episodes, &steps)?;
    ensure!(
        unique_episodes == metadata.episodes,
        "metadata declares {} episodes but data contains {unique_episodes}",
        metadata.episodes
    );
    let unique_domains = domain_idx.iter().copied().collect::<BTreeSet<_>>();
    ensure!(
        unique_domains.len() == domains.len(),
        "data contains {} unique domains but domains.json has {}",
        unique_domains.len(),
        domains.len()
    );
    ensure!(
        domain_idx
            .iter()
            .all(|index| *index >= 0 && (*index as usize) < domains.len()),
        "domain_idx contains an out-of-range value"
    );

    let mut state_stats = StatsAccumulator::new(SKYJEPA_STATE_DIM);
    let mut action_stats = StatsAccumulator::new(SKYJEPA_ACTION_DIM);
    let mut differential_action_stats = StatsAccumulator::new(SKYJEPA_ACTION_DIM);
    let mut action_delta_stats = StatsAccumulator::new(SKYJEPA_ACTION_DIM);
    let mut reference_stats = references
        .as_ref()
        .map(|_| StatsAccumulator::new(SKYJEPA_STATE_DIM));
    let mut motor_stats = motor_forces
        .as_ref()
        .map(|_| StatsAccumulator::new(SKYJEPA_ACTION_DIM));
    let mut domain_coverage =
        classify_domain_coverage(&episodes, &domain_idx, references.as_deref(), domains.len())?;
    let mut domain_differential = (0..domains.len())
        .map(|_| StatsAccumulator::new(4))
        .collect::<Vec<_>>();
    let mut domain_delta = (0..domains.len())
        .map(|_| StatsAccumulator::new(4))
        .collect::<Vec<_>>();
    let expected_dt = 1.0 / metadata.sample_rate_hz as f64;
    let mut max_dt_error = 0.0f64;
    let mut max_rotation_error = 0.0f64;
    let mut determinant_min = f64::INFINITY;
    let mut determinant_max = f64::NEG_INFINITY;
    let mut tracking_position_sq = 0.0f64;
    let mut tracking_velocity_sq = 0.0f64;
    let mut ground_contacts = 0usize;
    let mut low_commands = 0usize;
    let mut high_commands = 0usize;

    for row in 0..rows {
        let state = &states[row * SKYJEPA_STATE_DIM..(row + 1) * SKYJEPA_STATE_DIM];
        let action = &actions[row * SKYJEPA_ACTION_DIM..(row + 1) * SKYJEPA_ACTION_DIM];
        state_stats.push(state);
        action_stats.push(action);
        let collective = action.iter().sum::<f32>() / SKYJEPA_ACTION_DIM as f32;
        let differential = action
            .iter()
            .map(|value| *value - collective)
            .collect::<Vec<_>>();
        differential_action_stats.push(&differential);
        domain_differential[domain_idx[row] as usize].push(&differential);
        if row > 0 && episodes[row] == episodes[row - 1] {
            let previous = &actions[(row - 1) * SKYJEPA_ACTION_DIM..row * SKYJEPA_ACTION_DIM];
            let delta = action
                .iter()
                .zip(previous)
                .map(|(current, previous)| current - previous)
                .collect::<Vec<_>>();
            action_delta_stats.push(&delta);
            domain_delta[domain_idx[row] as usize].push(&delta);
        }
        max_dt_error = max_dt_error.max((f64::from(dt[row]) - expected_dt).abs());
        let rotation: [f32; 9] = state[6..15].try_into().expect("rotation has nine values");
        let (orthogonality, determinant) = rotation_quality(rotation);
        max_rotation_error = max_rotation_error.max(orthogonality);
        determinant_min = determinant_min.min(determinant);
        determinant_max = determinant_max.max(determinant);
        if state[2] <= 0.051 {
            ground_contacts += 1;
        }

        let domain = domains[domain_idx[row] as usize].parameters;
        let maximum = f64::from(domain.mass * domain.gravity * domain.max_thrust_weight / 4.0);
        for command in action.iter().copied().map(f64::from) {
            if command <= 1e-6 {
                low_commands += 1;
            }
            if command >= maximum * 0.999 {
                high_commands += 1;
            }
        }

        if let (Some(references), Some(stats)) = (references.as_ref(), reference_stats.as_mut()) {
            let reference = &references[row * SKYJEPA_STATE_DIM..(row + 1) * SKYJEPA_STATE_DIM];
            stats.push(reference);
            tracking_position_sq += squared_error(&state[0..3], &reference[0..3]);
            tracking_velocity_sq += squared_error(&state[3..6], &reference[3..6]);
        }
        if let (Some(motor_forces), Some(stats)) = (motor_forces.as_ref(), motor_stats.as_mut()) {
            stats.push(&motor_forces[row * SKYJEPA_ACTION_DIM..(row + 1) * SKYJEPA_ACTION_DIM]);
        }
    }

    let action = action_stats.finish();
    let action_differential = differential_action_stats.finish();
    let action_transition_delta = action_delta_stats.finish();
    let position_tracking_rmse = references
        .as_ref()
        .map(|_| (tracking_position_sq / rows as f64).sqrt());
    let velocity_tracking_rmse = references
        .as_ref()
        .map(|_| (tracking_velocity_sq / rows as f64).sqrt());
    let ground_fraction = ground_contacts as f64 / rows as f64;
    let action_values = rows * SKYJEPA_ACTION_DIM;
    let low_fraction = low_commands as f64 / action_values as f64;
    let high_fraction = high_commands as f64 / action_values as f64;
    let mut failures = Vec::new();
    if references.is_none() {
        failures.push("reference_state is required to audit per-domain flight coverage".into());
    }
    for ((coverage, differential), delta) in domain_coverage
        .iter_mut()
        .zip(domain_differential)
        .zip(domain_delta)
    {
        coverage.differential_action_std = differential.finish().std;
        coverage.action_delta_std = delta.finish().std;
        if coverage.hover_episodes == 0 || coverage.moving_episodes == 0 {
            failures.push(format!(
                "domain {} needs both hover and moving trajectories (hover={}, moving={})",
                coverage.domain, coverage.hover_episodes, coverage.moving_episodes
            ));
        }
        if coverage
            .differential_action_std
            .iter()
            .any(|std| *std < args.min_differential_action_std)
            || coverage
                .action_delta_std
                .iter()
                .any(|std| *std < args.min_action_delta_std)
        {
            failures.push(format!(
                "domain {} fails per-domain rotor excitation thresholds",
                coverage.domain
            ));
        }
    }
    if max_dt_error > 1e-6 {
        failures.push(format!(
            "maximum dt error {max_dt_error:.3e} exceeds 1e-6 s"
        ));
    }
    if max_rotation_error > args.max_rotation_error {
        failures.push(format!(
            "rotation orthogonality error {max_rotation_error:.3e} exceeds {:.3e}",
            args.max_rotation_error
        ));
    }
    if determinant_min <= 0.0 || (determinant_min - 1.0).abs() > args.max_rotation_error {
        failures.push(format!(
            "rotation determinant range [{determinant_min:.6}, {determinant_max:.6}] is invalid"
        ));
    }
    if let Some(rmse) = position_tracking_rmse {
        if rmse > args.max_tracking_rmse_m {
            failures.push(format!(
                "position tracking RMSE {rmse:.4} m exceeds {:.4} m",
                args.max_tracking_rmse_m
            ));
        }
    }
    if ground_fraction > args.max_ground_fraction {
        failures.push(format!(
            "ground-contact fraction {:.3}% exceeds {:.3}%",
            ground_fraction * 100.0,
            args.max_ground_fraction * 100.0
        ));
    }
    if low_fraction + high_fraction > args.max_saturation_fraction {
        failures.push(format!(
            "command saturation fraction {:.3}% exceeds {:.3}%",
            (low_fraction + high_fraction) * 100.0,
            args.max_saturation_fraction * 100.0
        ));
    }
    for (idx, std) in action.std.iter().enumerate() {
        if *std < args.min_action_std {
            failures.push(format!(
                "action channel {idx} std {std:.3e} is below {:.3e}",
                args.min_action_std
            ));
        }
    }
    for (idx, std) in action_differential.std.iter().enumerate() {
        if *std < args.min_differential_action_std {
            failures.push(format!(
                "differential action channel {idx} std {std:.3e} is below {:.3e}",
                args.min_differential_action_std
            ));
        }
    }
    for (idx, std) in action_transition_delta.std.iter().enumerate() {
        if *std < args.min_action_delta_std {
            failures.push(format!(
                "action delta channel {idx} std {std:.3e} is below {:.3e}",
                args.min_action_delta_std
            ));
        }
    }

    let report = AuditReport {
        audit_version: 2,
        audit_config: serde_json::to_value(&args)?,
        passed: failures.is_empty(),
        dataset_dir: fs::canonicalize(&args.dataset_dir).unwrap_or(args.dataset_dir.clone()),
        artifact_sha256: skyjepa_artifact_fingerprint(&args.dataset_dir)?,
        metadata,
        rows,
        episodes: unique_episodes,
        domains: unique_domains.len(),
        expected_dt_seconds: expected_dt,
        max_dt_error_seconds: max_dt_error,
        max_rotation_orthogonality_error: max_rotation_error,
        determinant_min,
        determinant_max,
        position_tracking_rmse_m: position_tracking_rmse,
        velocity_tracking_rmse_mps: velocity_tracking_rmse,
        ground_contact_fraction: ground_fraction,
        low_command_fraction: low_fraction,
        high_command_fraction: high_fraction,
        state: state_stats.finish(),
        action,
        action_differential,
        action_transition_delta,
        reference_state: reference_stats.map(StatsAccumulator::finish),
        motor_force: motor_stats.map(StatsAccumulator::finish),
        domain_coverage,
        failures,
    };
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = args.output.as_ref() {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, &json).with_context(|| format!("failed to write {}", path.display()))?;
    }
    println!("{json}");
    ensure!(
        report.passed || args.allow_fail,
        "SkyJEPA dataset audit failed with {} issue(s)",
        report.failures.len()
    );
    Ok(())
}

fn classify_domain_coverage(
    episodes: &[i64],
    domains: &[i64],
    references: Option<&[f32]>,
    count: usize,
) -> anyhow::Result<Vec<DomainCoverage>> {
    let mut coverage = (0..count)
        .map(|domain| DomainCoverage {
            domain,
            ..Default::default()
        })
        .collect::<Vec<_>>();
    let mut start = 0;
    while start < episodes.len() {
        let mut end = start + 1;
        while end < episodes.len() && episodes[end] == episodes[start] {
            end += 1;
        }
        ensure!(
            domains[start..end]
                .iter()
                .all(|domain| *domain == domains[start]),
            "episode {} changes domain mid-flight",
            episodes[start]
        );
        let domain = domains[start] as usize;
        coverage[domain].episodes += 1;
        if let Some(reference) = references {
            let moving = (start + 1..end).any(|row| {
                (0..3).any(|axis| {
                    (reference[row * 18 + axis] - reference[start * 18 + axis]).abs() > 1e-4
                })
            });
            if moving {
                coverage[domain].moving_episodes += 1;
            } else {
                coverage[domain].hover_episodes += 1;
            }
        }
        start = end;
    }
    Ok(coverage)
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    ensure!(args.dataset_dir.is_dir(), "dataset-dir is not a directory");
    ensure!(
        args.max_rotation_error.is_finite() && args.max_rotation_error > 0.0,
        "max-rotation-error must be positive"
    );
    ensure!(
        args.max_tracking_rmse_m.is_finite() && args.max_tracking_rmse_m > 0.0,
        "max-tracking-rmse-m must be positive"
    );
    for (name, value) in [
        ("max-ground-fraction", args.max_ground_fraction),
        ("max-saturation-fraction", args.max_saturation_fraction),
    ] {
        ensure!(
            value.is_finite() && (0.0..=1.0).contains(&value),
            "{name} must be in [0, 1]"
        );
    }
    ensure!(
        args.min_action_std.is_finite() && args.min_action_std >= 0.0,
        "min-action-std must be non-negative"
    );
    ensure!(
        args.min_differential_action_std.is_finite() && args.min_differential_action_std >= 0.0,
        "min-differential-action-std must be non-negative"
    );
    ensure!(
        args.min_action_delta_std.is_finite() && args.min_action_delta_std >= 0.0,
        "min-action-delta-std must be non-negative"
    );
    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> anyhow::Result<T> {
    serde_json::from_str(
        &fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))
}

fn read_f32(file: &File, name: &str, rows: usize, dim: usize) -> anyhow::Result<Vec<f32>> {
    let dataset = file
        .dataset(name)
        .with_context(|| format!("missing {name}"))?;
    ensure!(
        dataset.shape() == [rows, dim],
        "{name} shape {:?} does not match [{rows}, {dim}]",
        dataset.shape()
    );
    dataset.read_raw::<f32>().map_err(Into::into)
}

fn read_optional_f32(
    file: &File,
    name: &str,
    rows: usize,
    dim: usize,
    declared: bool,
) -> anyhow::Result<Option<Vec<f32>>> {
    let exists = file.link_exists(name);
    ensure!(
        exists == declared,
        "metadata {name} declaration ({declared}) disagrees with data.h5 ({exists})"
    );
    exists.then(|| read_f32(file, name, rows, dim)).transpose()
}

fn read_i64(file: &File, name: &str, rows: usize) -> anyhow::Result<Vec<i64>> {
    let dataset = file
        .dataset(name)
        .with_context(|| format!("missing {name}"))?;
    ensure!(
        dataset.shape() == [rows],
        "{name} shape {:?} does not match [{rows}]",
        dataset.shape()
    );
    dataset.read_raw::<i64>().map_err(Into::into)
}

fn ensure_finite(name: &str, values: &[f32]) -> anyhow::Result<()> {
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "{name} contains NaN or infinity"
    );
    Ok(())
}

fn validate_sequence_index(episodes: &[i64], steps: &[i64]) -> anyhow::Result<usize> {
    ensure!(episodes.len() == steps.len(), "episode/step lengths differ");
    let mut seen = BTreeSet::new();
    let mut previous_episode = None;
    let mut previous_step = 0i64;
    for (row, (&episode, &step)) in episodes.iter().zip(steps).enumerate() {
        ensure!(episode >= 0, "negative episode index at row {row}");
        if previous_episode == Some(episode) {
            ensure!(
                step == previous_step + 1,
                "episode {episode} step discontinuity at row {row}: {previous_step} -> {step}"
            );
        } else {
            ensure!(
                step == 0,
                "episode {episode} starts at step {step}, not zero"
            );
            ensure!(
                seen.insert(episode),
                "episode {episode} is split into non-contiguous blocks"
            );
        }
        previous_episode = Some(episode);
        previous_step = step;
    }
    Ok(seen.len())
}

fn rotation_quality(r: [f32; 9]) -> (f64, f64) {
    let mut error_sq = 0.0f64;
    for row in 0..3 {
        for col in 0..3 {
            let dot = (0..3)
                .map(|axis| f64::from(r[axis * 3 + row]) * f64::from(r[axis * 3 + col]))
                .sum::<f64>();
            let expected = f64::from((row == col) as u8);
            error_sq += (dot - expected).powi(2);
        }
    }
    let determinant = f64::from(r[0])
        * (f64::from(r[4]) * f64::from(r[8]) - f64::from(r[5]) * f64::from(r[7]))
        - f64::from(r[1]) * (f64::from(r[3]) * f64::from(r[8]) - f64::from(r[5]) * f64::from(r[6]))
        + f64::from(r[2]) * (f64::from(r[3]) * f64::from(r[7]) - f64::from(r[4]) * f64::from(r[6]));
    (error_sq.sqrt(), determinant)
}

fn squared_error(lhs: &[f32], rhs: &[f32]) -> f64 {
    lhs.iter()
        .zip(rhs)
        .map(|(lhs, rhs)| f64::from(*lhs - *rhs).powi(2))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_quality_accepts_identity() {
        let (error, determinant) = rotation_quality([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(error, 0.0);
        assert_eq!(determinant, 1.0);
    }

    #[test]
    fn sequence_validation_rejects_repeated_episode_blocks() {
        assert!(validate_sequence_index(&[0, 0, 1, 1, 0], &[0, 1, 0, 1, 0]).is_err());
    }

    #[test]
    fn coverage_detects_domain_task_aliasing_and_mid_episode_domain_changes() -> anyhow::Result<()>
    {
        let mut references = vec![0.0f32; 4 * 18];
        references[3 * 18] = 1.0;
        let coverage =
            classify_domain_coverage(&[0, 0, 1, 1], &[0, 0, 1, 1], Some(&references), 2)?;
        assert_eq!(coverage[0].hover_episodes, 1);
        assert_eq!(coverage[0].moving_episodes, 0);
        assert_eq!(coverage[1].hover_episodes, 0);
        assert_eq!(coverage[1].moving_episodes, 1);
        assert!(classify_domain_coverage(&[0, 0], &[0, 1], None, 2).is_err());
        Ok(())
    }
}
