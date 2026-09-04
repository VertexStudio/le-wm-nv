use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, ensure};
use candle::{DType, Device, Tensor};
use hdf5::File;
use serde::{Deserialize, Serialize};

use super::drone_racing::{DroneRacingMetadata, RunningStats, mat3_mul_vec3};

pub const SKYJEPA_STATE_DIM: usize = 18;
pub const SKYJEPA_ACTION_DIM: usize = 4;
pub const SKYJEPA_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SkyJepaActionSpace {
    /// Normalized roll-rate, pitch-rate, throttle, and yaw-rate commands used
    /// by the current drone-racing logs and low-level plant.
    BodyRatesThrottle,
    /// Four non-negative rotor force commands as written in the SkyJEPA paper.
    RotorForces,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkyJepaDatasetMetadata {
    pub schema_version: u32,
    pub data_h5: PathBuf,
    pub sample_rate_hz: usize,
    pub rows: usize,
    pub episodes: usize,
    pub state_dim: usize,
    pub action_dim: usize,
    pub action_space: SkyJepaActionSpace,
    #[serde(default)]
    pub generator: Option<String>,
}

impl SkyJepaDatasetMetadata {
    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            self.schema_version == SKYJEPA_SCHEMA_VERSION,
            "unsupported SkyJEPA schema version {}; expected {SKYJEPA_SCHEMA_VERSION}",
            self.schema_version
        );
        ensure!(self.sample_rate_hz > 0, "sample_rate_hz must be positive");
        ensure!(self.rows > 0, "SkyJEPA dataset must contain rows");
        ensure!(
            self.episodes >= 3,
            "SkyJEPA dataset requires at least 3 episodes"
        );
        ensure!(
            self.state_dim == SKYJEPA_STATE_DIM,
            "SkyJEPA state_dim {} must be {SKYJEPA_STATE_DIM}",
            self.state_dim
        );
        ensure!(
            self.action_dim == SKYJEPA_ACTION_DIM,
            "SkyJEPA action_dim {} must be {SKYJEPA_ACTION_DIM}",
            self.action_dim
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SkyJepaDatasetConfig {
    pub batch_size: usize,
    pub history_steps: usize,
    pub rollout_steps: usize,
    pub model_rate_hz: usize,
    pub normalize_states: bool,
    pub normalize_actions: bool,
    pub action_space: SkyJepaActionSpace,
}

impl SkyJepaDatasetConfig {
    pub fn paper_derived(batch_size: usize) -> Self {
        Self {
            batch_size,
            history_steps: 10,
            rollout_steps: 20,
            model_rate_hz: 20,
            normalize_states: true,
            normalize_actions: true,
            action_space: SkyJepaActionSpace::BodyRatesThrottle,
        }
    }

    pub fn validate(self) -> anyhow::Result<()> {
        ensure!(self.batch_size > 0, "batch_size must be greater than zero");
        ensure!(
            self.history_steps >= 2,
            "history_steps must be at least two"
        );
        ensure!(
            self.rollout_steps > 0,
            "rollout_steps must be greater than zero"
        );
        ensure!(
            self.model_rate_hz > 0,
            "model_rate_hz must be greater than zero"
        );
        Ok(())
    }

    pub fn sequence_steps(self) -> usize {
        self.history_steps + self.rollout_steps
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkyJepaNormalization {
    pub state: RunningStats,
    pub action: RunningStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkyJepaEpisodeSplit {
    pub train: Vec<i64>,
    pub validation: Vec<i64>,
    pub test: Vec<i64>,
}

#[derive(Debug)]
pub struct SkyJepaBatch {
    /// Standardized model inputs, `[batch, H + T, 18]`.
    pub states: Tensor,
    /// Unnormalized metric states for the physics prober.
    pub metric_states: Tensor,
    /// Standardized model inputs, `[batch, H + T, 4]`.
    pub actions: Tensor,
    /// Unnormalized controls in the configured action space.
    pub metric_actions: Tensor,
    /// Exact elapsed seconds for every transition, `[batch, H + T - 1]`.
    pub transition_dt: Tensor,
    pub rows: Vec<usize>,
    pub episode_idx: Vec<i64>,
    pub step_idx: Vec<i64>,
}

/// Canonical, pre-normalized full-state view of a drone trajectory dataset.
///
/// The loader consumes the existing imported HDF5 artifact but composes state18
/// once at open time. Training batches then perform only strided contiguous
/// copies and a single host-to-device transfer per tensor.
pub struct SkyJepaDroneDataset {
    root: PathBuf,
    config: SkyJepaDatasetConfig,
    source_rate_hz: usize,
    source_stride: usize,
    states_raw: Vec<f32>,
    states_normalized: Vec<f32>,
    actions_raw: Vec<f32>,
    actions_normalized: Vec<f32>,
    episode_idx: Vec<i64>,
    step_idx: Vec<i64>,
    dt: Vec<f32>,
    splits: SkyJepaEpisodeSplit,
    normalization: SkyJepaNormalization,
    valid_rows: Vec<usize>,
}

impl SkyJepaDroneDataset {
    pub fn open(root: impl AsRef<Path>, config: SkyJepaDatasetConfig) -> anyhow::Result<Self> {
        config.validate()?;
        let root = root.as_ref();
        let metadata_path = root.join("metadata.json");
        let metadata_json = fs::read_to_string(&metadata_path)
            .with_context(|| format!("failed to read {}", metadata_path.display()))?;
        let canonical = serde_json::from_str::<SkyJepaDatasetMetadata>(&metadata_json).ok();
        let legacy = if canonical.is_none() {
            Some(
                serde_json::from_str::<DroneRacingMetadata>(&metadata_json)
                    .with_context(|| format!("failed to parse {}", metadata_path.display()))?,
            )
        } else {
            None
        };
        if let Some(metadata) = canonical.as_ref() {
            metadata.validate()?;
            ensure!(
                metadata.action_space == config.action_space,
                "dataset action space {:?} does not match requested {:?}",
                metadata.action_space,
                config.action_space
            );
        } else {
            ensure!(
                config.action_space == SkyJepaActionSpace::BodyRatesThrottle,
                "legacy LeWM drone data contains body-rate/throttle actions; select body_rates_throttle"
            );
        }
        let sample_rate_hz = canonical
            .as_ref()
            .map(|metadata| metadata.sample_rate_hz)
            .or_else(|| legacy.as_ref().map(|metadata| metadata.sample_rate_hz))
            .expect("one metadata representation is present");
        let rows = canonical
            .as_ref()
            .map(|metadata| metadata.rows)
            .or_else(|| legacy.as_ref().map(|metadata| metadata.rows))
            .expect("one metadata representation is present");
        ensure!(
            sample_rate_hz >= config.model_rate_hz
                && sample_rate_hz.is_multiple_of(config.model_rate_hz),
            "dataset rate {} Hz must be an integer multiple of model rate {} Hz",
            sample_rate_hz,
            config.model_rate_hz
        );
        let source_stride = sample_rate_hz / config.model_rate_hz;
        let data_h5 = canonical
            .as_ref()
            .map(|metadata| &metadata.data_h5)
            .or_else(|| legacy.as_ref().map(|metadata| &metadata.data_h5))
            .expect("one metadata representation is present");
        let data_path = if data_h5.is_absolute() {
            data_h5.clone()
        } else {
            root.join(data_h5)
        };
        let file = File::open(&data_path)
            .with_context(|| format!("failed to open {}", data_path.display()))?;
        let states_raw = if file.link_exists("state") {
            read_f32_dataset(&file, "state", rows, SKYJEPA_STATE_DIM)?
        } else {
            compose_legacy_state(&file, rows)?
        };
        let actions_raw = if file.link_exists("action") {
            read_f32_dataset(&file, "action", rows, SKYJEPA_ACTION_DIM)?
        } else {
            read_f32_dataset(&file, "channels_norm", rows, SKYJEPA_ACTION_DIM)?
        };
        let episode_idx = read_i64_dataset(&file, "episode_idx", rows)?;
        let step_idx = read_i64_dataset(&file, "step_idx", rows)?;
        let dt = read_f32_dataset(&file, "dt", rows, 1)?;
        let episodes = episode_idx
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let splits = split_episodes(&episodes)?;
        let normalization =
            compute_normalization(&states_raw, &actions_raw, &episode_idx, &splits.train)?;
        let states_normalized = if config.normalize_states {
            normalize_rows(&states_raw, SKYJEPA_STATE_DIM, &normalization.state)
        } else {
            states_raw.clone()
        };
        let actions_normalized = if config.normalize_actions {
            normalize_rows(&actions_raw, SKYJEPA_ACTION_DIM, &normalization.action)
        } else {
            actions_raw.clone()
        };
        let valid_rows = valid_sequence_rows(
            &episode_idx,
            &step_idx,
            config.sequence_steps(),
            source_stride,
        )?;
        ensure!(
            !valid_rows.is_empty(),
            "SkyJEPA dataset has no valid sequences"
        );

        Ok(Self {
            root: root.to_path_buf(),
            config,
            source_rate_hz: sample_rate_hz,
            source_stride,
            states_raw,
            states_normalized,
            actions_raw,
            actions_normalized,
            episode_idx,
            step_idx,
            dt,
            splits,
            normalization,
            valid_rows,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn config(&self) -> SkyJepaDatasetConfig {
        self.config
    }

    pub fn source_rate_hz(&self) -> usize {
        self.source_rate_hz
    }

    pub fn source_stride(&self) -> usize {
        self.source_stride
    }

    pub fn splits(&self) -> &SkyJepaEpisodeSplit {
        &self.splits
    }

    pub fn normalization(&self) -> &SkyJepaNormalization {
        &self.normalization
    }

    pub fn train_rows(&self) -> Vec<usize> {
        self.rows_for_episodes(&self.splits.train)
    }

    pub fn validation_rows(&self) -> Vec<usize> {
        self.rows_for_episodes(&self.splits.validation)
    }

    pub fn test_rows(&self) -> Vec<usize> {
        self.rows_for_episodes(&self.splits.test)
    }

    pub fn shuffled_rows(&self, rows: &[usize], seed: u64) -> Vec<usize> {
        let mut rows = rows.to_vec();
        shuffle(&mut rows, seed);
        rows
    }

    pub fn batch(
        &self,
        rows: &[usize],
        dtype: DType,
        device: &Device,
    ) -> anyhow::Result<SkyJepaBatch> {
        ensure!(!rows.is_empty(), "cannot create an empty SkyJEPA batch");
        ensure!(
            rows.len() <= self.config.batch_size,
            "batch has {} rows, configured maximum is {}",
            rows.len(),
            self.config.batch_size
        );
        let batch = rows.len();
        let time = self.config.sequence_steps();
        let mut states = vec![0f32; batch * time * SKYJEPA_STATE_DIM];
        let mut metric_states = vec![0f32; batch * time * SKYJEPA_STATE_DIM];
        let mut actions = vec![0f32; batch * time * SKYJEPA_ACTION_DIM];
        let mut metric_actions = vec![0f32; batch * time * SKYJEPA_ACTION_DIM];
        let mut transition_dt = vec![0f32; batch * (time - 1)];
        let mut episodes = Vec::with_capacity(batch);
        let mut steps = Vec::with_capacity(batch);

        for (batch_idx, &start) in rows.iter().enumerate() {
            ensure!(
                self.valid_rows.binary_search(&start).is_ok(),
                "row {start} is not a valid SkyJEPA sequence start"
            );
            episodes.push(self.episode_idx[start]);
            steps.push(self.step_idx[start]);
            for time_idx in 0..time {
                let source = start + time_idx * self.source_stride;
                copy_row(
                    &self.states_normalized,
                    SKYJEPA_STATE_DIM,
                    source,
                    &mut states,
                    batch_idx * time + time_idx,
                );
                copy_row(
                    &self.states_raw,
                    SKYJEPA_STATE_DIM,
                    source,
                    &mut metric_states,
                    batch_idx * time + time_idx,
                );
                copy_row(
                    &self.actions_normalized,
                    SKYJEPA_ACTION_DIM,
                    source,
                    &mut actions,
                    batch_idx * time + time_idx,
                );
                copy_row(
                    &self.actions_raw,
                    SKYJEPA_ACTION_DIM,
                    source,
                    &mut metric_actions,
                    batch_idx * time + time_idx,
                );
                if time_idx + 1 < time {
                    transition_dt[batch_idx * (time - 1) + time_idx] = self.transition_dt(source);
                }
            }
        }

        Ok(SkyJepaBatch {
            states: Tensor::from_vec(states, (batch, time, SKYJEPA_STATE_DIM), device)?
                .to_dtype(dtype)?,
            metric_states: Tensor::from_vec(
                metric_states,
                (batch, time, SKYJEPA_STATE_DIM),
                device,
            )?
            .to_dtype(dtype)?,
            actions: Tensor::from_vec(actions, (batch, time, SKYJEPA_ACTION_DIM), device)?
                .to_dtype(dtype)?,
            metric_actions: Tensor::from_vec(
                metric_actions,
                (batch, time, SKYJEPA_ACTION_DIM),
                device,
            )?
            .to_dtype(dtype)?,
            transition_dt: Tensor::from_vec(transition_dt, (batch, time - 1), device)?
                .to_dtype(dtype)?,
            rows: rows.to_vec(),
            episode_idx: episodes,
            step_idx: steps,
        })
    }

    fn rows_for_episodes(&self, episodes: &[i64]) -> Vec<usize> {
        let episodes = episodes.iter().copied().collect::<BTreeSet<_>>();
        self.valid_rows
            .iter()
            .copied()
            .filter(|row| episodes.contains(&self.episode_idx[*row]))
            .collect()
    }

    fn transition_dt(&self, source: usize) -> f32 {
        (1..=self.source_stride)
            .map(|offset| self.dt[source + offset])
            .sum()
    }
}

fn compose_legacy_state(file: &File, rows: usize) -> anyhow::Result<Vec<f32>> {
    let positions = read_f32_dataset(file, "pos_world", rows, 3)?;
    let rotations = read_f32_dataset(file, "rotmat_world_from_body", rows, 9)?;
    let velocity_body = read_f32_dataset(file, "lin_vel_body", rows, 3)?;
    let angular_velocity = read_f32_dataset(file, "ang_vel_body", rows, 3)?;
    let mut states = vec![0f32; rows * SKYJEPA_STATE_DIM];
    for row in 0..rows {
        let output = &mut states[row * SKYJEPA_STATE_DIM..(row + 1) * SKYJEPA_STATE_DIM];
        output[0..3].copy_from_slice(&positions[row * 3..row * 3 + 3]);
        let mut rotation = [0f32; 9];
        rotation.copy_from_slice(&rotations[row * 9..row * 9 + 9]);
        let mut body_velocity = [0f32; 3];
        body_velocity.copy_from_slice(&velocity_body[row * 3..row * 3 + 3]);
        output[3..6].copy_from_slice(&mat3_mul_vec3(rotation, body_velocity));
        output[6..15].copy_from_slice(&rotation);
        output[15..18].copy_from_slice(&angular_velocity[row * 3..row * 3 + 3]);
    }
    Ok(states)
}

fn split_episodes(episodes: &[i64]) -> anyhow::Result<SkyJepaEpisodeSplit> {
    ensure!(
        episodes.len() >= 3,
        "SkyJEPA requires at least three episodes for train/validation/test splits"
    );
    let validation_count = ((episodes.len() as f64 * 0.1).round() as usize).max(1);
    let test_count = ((episodes.len() as f64 * 0.1).round() as usize).max(1);
    let train_end = episodes.len() - validation_count - test_count;
    ensure!(train_end > 0, "SkyJEPA split produced no training episodes");
    let validation_end = train_end + validation_count;
    Ok(SkyJepaEpisodeSplit {
        train: episodes[..train_end].to_vec(),
        validation: episodes[train_end..validation_end].to_vec(),
        test: episodes[validation_end..].to_vec(),
    })
}

fn valid_sequence_rows(
    episode_idx: &[i64],
    step_idx: &[i64],
    sequence_steps: usize,
    stride: usize,
) -> anyhow::Result<Vec<usize>> {
    ensure!(
        episode_idx.len() == step_idx.len(),
        "episode and step arrays differ in length"
    );
    let span = (sequence_steps - 1)
        .checked_mul(stride)
        .context("SkyJEPA sequence span overflowed")?;
    let mut rows = Vec::new();
    for start in 0..episode_idx.len() {
        let Some(end) = start.checked_add(span) else {
            continue;
        };
        if end >= episode_idx.len() || episode_idx[end] != episode_idx[start] {
            continue;
        }
        let expected_step = step_idx[start] + span as i64;
        if step_idx[end] == expected_step {
            rows.push(start);
        }
    }
    Ok(rows)
}

fn compute_normalization(
    states: &[f32],
    actions: &[f32],
    episode_idx: &[i64],
    train_episodes: &[i64],
) -> anyhow::Result<SkyJepaNormalization> {
    let train = train_episodes.iter().copied().collect::<BTreeSet<_>>();
    let mut state_stats = StatsAccumulator::new(SKYJEPA_STATE_DIM);
    let mut action_stats = StatsAccumulator::new(SKYJEPA_ACTION_DIM);
    for (row, episode) in episode_idx.iter().enumerate() {
        if train.contains(episode) {
            state_stats.push(&states[row * SKYJEPA_STATE_DIM..(row + 1) * SKYJEPA_STATE_DIM]);
            action_stats.push(&actions[row * SKYJEPA_ACTION_DIM..(row + 1) * SKYJEPA_ACTION_DIM]);
        }
    }
    Ok(SkyJepaNormalization {
        state: state_stats.finish()?,
        action: action_stats.finish()?,
    })
}

fn normalize_rows(values: &[f32], dim: usize, stats: &RunningStats) -> Vec<f32> {
    values
        .chunks_exact(dim)
        .flat_map(|row| {
            row.iter()
                .enumerate()
                .map(|(idx, value)| (*value - stats.mean[idx]) / stats.std[idx].max(1e-6))
        })
        .collect()
}

fn copy_row(source: &[f32], dim: usize, source_row: usize, output: &mut [f32], output_row: usize) {
    output[output_row * dim..(output_row + 1) * dim]
        .copy_from_slice(&source[source_row * dim..(source_row + 1) * dim]);
}

struct StatsAccumulator {
    count: usize,
    sum: Vec<f64>,
    sum_sq: Vec<f64>,
}

impl StatsAccumulator {
    fn new(dim: usize) -> Self {
        Self {
            count: 0,
            sum: vec![0.0; dim],
            sum_sq: vec![0.0; dim],
        }
    }

    fn push(&mut self, row: &[f32]) {
        self.count += 1;
        for (idx, value) in row.iter().enumerate() {
            let value = f64::from(*value);
            self.sum[idx] += value;
            self.sum_sq[idx] += value * value;
        }
    }

    fn finish(self) -> anyhow::Result<RunningStats> {
        ensure!(
            self.count > 0,
            "cannot compute normalization from zero rows"
        );
        let count = self.count as f64;
        let mut mean = Vec::with_capacity(self.sum.len());
        let mut std = Vec::with_capacity(self.sum.len());
        for idx in 0..self.sum.len() {
            let row_mean = self.sum[idx] / count;
            let variance = (self.sum_sq[idx] / count - row_mean * row_mean).max(1e-12);
            mean.push(row_mean as f32);
            std.push(variance.sqrt().max(1e-6) as f32);
        }
        Ok(RunningStats { mean, std })
    }
}

fn read_f32_dataset(file: &File, name: &str, rows: usize, dim: usize) -> anyhow::Result<Vec<f32>> {
    let dataset = file
        .dataset(name)
        .with_context(|| format!("missing SkyJEPA HDF5 dataset `{name}`"))?;
    ensure!(
        dataset.shape() == [rows, dim],
        "dataset `{name}` shape {:?} does not match [{rows}, {dim}]",
        dataset.shape()
    );
    dataset
        .read_raw::<f32>()
        .with_context(|| format!("failed to read `{name}`"))
}

fn read_i64_dataset(file: &File, name: &str, rows: usize) -> anyhow::Result<Vec<i64>> {
    let dataset = file
        .dataset(name)
        .with_context(|| format!("missing SkyJEPA HDF5 dataset `{name}`"))?;
    ensure!(
        dataset.shape() == [rows],
        "dataset `{name}` shape {:?} does not match [{rows}]",
        dataset.shape()
    );
    dataset
        .read_raw::<i64>()
        .with_context(|| format!("failed to read `{name}`"))
}

fn shuffle(values: &mut [usize], mut state: u64) {
    for idx in (1..values.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let swap = (state as usize) % (idx + 1);
        values.swap(idx, swap);
    }
}
