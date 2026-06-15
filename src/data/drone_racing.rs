use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, ensure};
use candle::{DType, Device, Tensor};
use hdf5::File;
use serde::{Deserialize, Serialize};

pub const DRONE_ACTION_DIM: usize = 4;
pub const DRONE_OBSERVATION_DIM: usize = 12;
pub const DRONE_STATE_DELTA_DIM: usize = 13;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DroneBatchConfig {
    pub batch_size: usize,
    pub sequence_steps: usize,
    pub normalize_observations: bool,
    pub normalize_actions: bool,
}

impl DroneBatchConfig {
    pub fn validate(self) -> anyhow::Result<()> {
        ensure!(self.batch_size > 0, "batch_size must be greater than zero");
        ensure!(
            self.sequence_steps >= 2,
            "sequence_steps must be at least two"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DroneRacingMetadata {
    pub source_root: PathBuf,
    pub data_h5: PathBuf,
    pub source_files: Vec<PathBuf>,
    pub sample_rate_hz: usize,
    pub source_rate_hz: usize,
    pub rows: usize,
    pub episodes: usize,
    pub action_dim: usize,
    pub observation_dim: usize,
    pub state_delta_dim: usize,
    pub train_episodes: Vec<i64>,
    pub eval_episodes: Vec<i64>,
    pub normalization: DroneNormalization,
    pub columns: DroneColumns,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DroneColumns {
    pub action: Vec<String>,
    pub observation: Vec<String>,
    pub state_delta: Vec<String>,
}

impl Default for DroneColumns {
    fn default() -> Self {
        Self {
            action: [
                "channels_roll_norm",
                "channels_pitch_norm",
                "channels_thrust_norm",
                "channels_yaw_norm",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            observation: ["pos_world[0..3]", "rotmat_world_from_body[0..9]"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            state_delta: [
                "delta_pos_body[0..3]",
                "delta_rot_body[0..3]",
                "next_lin_vel_body[0..3]",
                "next_ang_vel_body[0..3]",
                "delta_vbat",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DroneNormalization {
    pub observation: RunningStats,
    pub action: RunningStats,
    pub target_delta: RunningStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunningStats {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl RunningStats {
    pub fn identity(dim: usize) -> Self {
        Self {
            mean: vec![0.0; dim],
            std: vec![1.0; dim],
        }
    }
}

#[derive(Debug)]
pub struct DroneBatch {
    pub observations: Tensor,
    pub actions: Tensor,
    pub meta: DroneBatchMeta,
}

#[derive(Debug, Clone, Serialize)]
pub struct DroneBatchMeta {
    pub rows: Vec<usize>,
    pub episode_idx: Vec<i64>,
    pub step_idx: Vec<i64>,
}

pub struct DroneRacingDataset {
    root: PathBuf,
    data_h5: PathBuf,
    metadata: DroneRacingMetadata,
    config: DroneBatchConfig,
    pos_world: Vec<f32>,
    rotmat_world_from_body: Vec<f32>,
    lin_vel_body: Vec<f32>,
    ang_vel_body: Vec<f32>,
    vbat: Vec<f32>,
    channels_norm: Vec<f32>,
    episode_idx: Vec<i64>,
    step_idx: Vec<i64>,
    dt: Vec<f32>,
    valid_rows: Vec<usize>,
    valid_mask: Vec<bool>,
}

impl DroneRacingDataset {
    pub fn open(root: impl AsRef<Path>, config: DroneBatchConfig) -> anyhow::Result<Self> {
        config.validate()?;
        let root = root.as_ref();
        let metadata_path = root.join("metadata.json");
        let metadata: DroneRacingMetadata = serde_json::from_str(
            &fs::read_to_string(&metadata_path)
                .with_context(|| format!("failed to read {}", metadata_path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", metadata_path.display()))?;
        ensure!(
            metadata.action_dim == DRONE_ACTION_DIM,
            "metadata action_dim {} does not match expected {DRONE_ACTION_DIM}",
            metadata.action_dim
        );
        ensure!(
            metadata.observation_dim == DRONE_OBSERVATION_DIM,
            "metadata observation_dim {} does not match expected {DRONE_OBSERVATION_DIM}",
            metadata.observation_dim
        );
        ensure!(
            metadata.state_delta_dim == DRONE_STATE_DELTA_DIM,
            "metadata state_delta_dim {} does not match expected {DRONE_STATE_DELTA_DIM}",
            metadata.state_delta_dim
        );
        let data_h5 = if metadata.data_h5.is_absolute() {
            metadata.data_h5.clone()
        } else {
            root.join(&metadata.data_h5)
        };
        let file = File::open(&data_h5)
            .with_context(|| format!("failed to open {}", data_h5.display()))?;
        let pos_world = read_f32_dataset(&file, "pos_world", metadata.rows, 3)?;
        let rotmat_world_from_body =
            read_f32_dataset(&file, "rotmat_world_from_body", metadata.rows, 9)?;
        let lin_vel_body = read_f32_dataset(&file, "lin_vel_body", metadata.rows, 3)?;
        let ang_vel_body = read_f32_dataset(&file, "ang_vel_body", metadata.rows, 3)?;
        let vbat = read_f32_dataset(&file, "vbat", metadata.rows, 1)?;
        let channels_norm = read_f32_dataset(&file, "channels_norm", metadata.rows, 4)?;
        let episode_idx = read_i64_dataset(&file, "episode_idx", metadata.rows)?;
        let step_idx = read_i64_dataset(&file, "step_idx", metadata.rows)?;
        let dt = read_f32_dataset(&file, "dt", metadata.rows, 1)?;
        let ep_len = episode_lengths(&episode_idx, &step_idx)?;
        let valid_rows =
            compute_valid_rows(&episode_idx, &step_idx, &ep_len, config.sequence_steps)?;
        ensure!(
            !valid_rows.is_empty(),
            "no valid drone rows for sequence_steps={}",
            config.sequence_steps
        );
        let mut valid_mask = vec![false; metadata.rows];
        for &row in &valid_rows {
            valid_mask[row] = true;
        }
        Ok(Self {
            root: root.to_path_buf(),
            data_h5,
            metadata,
            config,
            pos_world,
            rotmat_world_from_body,
            lin_vel_body,
            ang_vel_body,
            vbat,
            channels_norm,
            episode_idx,
            step_idx,
            dt,
            valid_rows,
            valid_mask,
        })
    }

    pub fn metadata(&self) -> &DroneRacingMetadata {
        &self.metadata
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn data_h5(&self) -> &Path {
        &self.data_h5
    }

    pub fn valid_rows(&self) -> &[usize] {
        &self.valid_rows
    }

    pub fn config(&self) -> DroneBatchConfig {
        self.config
    }

    pub fn train_rows(&self) -> Vec<usize> {
        self.rows_for_episodes(&self.metadata.train_episodes)
    }

    pub fn eval_rows(&self) -> Vec<usize> {
        self.rows_for_episodes(&self.metadata.eval_episodes)
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
    ) -> anyhow::Result<DroneBatch> {
        ensure!(!rows.is_empty(), "cannot create an empty drone batch");
        ensure!(
            rows.len() <= self.config.batch_size,
            "batch has {} rows, configured batch_size is {}",
            rows.len(),
            self.config.batch_size
        );
        let batch = rows.len();
        let time = self.config.sequence_steps;
        let mut observations = vec![0f32; batch * time * DRONE_OBSERVATION_DIM];
        let mut actions = vec![0f32; batch * time * DRONE_ACTION_DIM];
        let mut episode_idx = Vec::with_capacity(batch);
        let mut step_idx = Vec::with_capacity(batch);

        for (batch_idx, &row) in rows.iter().enumerate() {
            self.ensure_valid_row(row)?;
            episode_idx.push(self.episode_idx[row]);
            step_idx.push(self.step_idx[row]);
            for t in 0..time {
                let src = row + t;
                self.write_observation(&mut observations, batch_idx, t, src)?;
                self.write_action(&mut actions, batch_idx, t, src);
            }
        }

        let observations =
            Tensor::from_vec(observations, (batch, time, DRONE_OBSERVATION_DIM), device)?
                .to_dtype(dtype)?;
        let actions =
            Tensor::from_vec(actions, (batch, time, DRONE_ACTION_DIM), device)?.to_dtype(dtype)?;
        Ok(DroneBatch {
            observations,
            actions,
            meta: DroneBatchMeta {
                rows: rows.to_vec(),
                episode_idx,
                step_idx,
            },
        })
    }

    pub fn replay_rows_for_episode(&self, episode: i64) -> Vec<usize> {
        self.episode_idx
            .iter()
            .enumerate()
            .filter_map(|(idx, value)| (*value == episode).then_some(idx))
            .collect()
    }

    pub fn frame(&self, row: usize) -> anyhow::Result<DroneFrame> {
        ensure!(row < self.metadata.rows, "row {row} outside dataset");
        Ok(DroneFrame {
            row,
            episode_idx: self.episode_idx[row],
            step_idx: self.step_idx[row],
            dt: self.dt[row],
            pos_world: vec3_at(&self.pos_world, row),
            rotmat_world_from_body: mat9_at(&self.rotmat_world_from_body, row),
            lin_vel_body: vec3_at(&self.lin_vel_body, row),
            ang_vel_body: vec3_at(&self.ang_vel_body, row),
            vbat: self.vbat[row],
            channels_norm: vec4_at(&self.channels_norm, row),
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

    fn ensure_valid_row(&self, row: usize) -> anyhow::Result<()> {
        ensure!(
            row < self.metadata.rows,
            "row {row} is outside dataset rows {}",
            self.metadata.rows
        );
        ensure!(
            self.valid_mask[row],
            "row {row} is not valid for sequence_steps={}",
            self.config.sequence_steps
        );
        Ok(())
    }

    fn write_observation(
        &self,
        output: &mut [f32],
        batch_idx: usize,
        time_idx: usize,
        row: usize,
    ) -> anyhow::Result<()> {
        let base = (batch_idx * self.config.sequence_steps + time_idx) * DRONE_OBSERVATION_DIM;
        output[base..base + 3].copy_from_slice(&self.pos_world[row * 3..row * 3 + 3]);
        output[base + 3..base + 12]
            .copy_from_slice(&self.rotmat_world_from_body[row * 9..row * 9 + 9]);
        if self.config.normalize_observations {
            normalize_in_place(
                &mut output[base..base + DRONE_OBSERVATION_DIM],
                &self.metadata.normalization.observation,
            )?;
        }
        Ok(())
    }

    fn write_action(&self, output: &mut [f32], batch_idx: usize, time_idx: usize, row: usize) {
        let base = (batch_idx * self.config.sequence_steps + time_idx) * DRONE_ACTION_DIM;
        output[base..base + 4].copy_from_slice(&self.channels_norm[row * 4..row * 4 + 4]);
        if self.config.normalize_actions {
            normalize_in_place_unchecked(
                &mut output[base..base + DRONE_ACTION_DIM],
                &self.metadata.normalization.action,
            );
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DroneFrame {
    pub row: usize,
    pub episode_idx: i64,
    pub step_idx: i64,
    pub dt: f32,
    pub pos_world: [f32; 3],
    pub rotmat_world_from_body: [f32; 9],
    pub lin_vel_body: [f32; 3],
    pub ang_vel_body: [f32; 3],
    pub vbat: f32,
    pub channels_norm: [f32; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateSequenceFile {
    pub flights: Vec<FlightGates>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlightGates {
    pub flight: String,
    pub episode_idx: i64,
    pub gates: Vec<GateSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateSpec {
    pub name: String,
    pub center: [f32; 3],
    pub normal: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub half_width: f32,
    pub half_height: f32,
}

#[derive(Debug, Default)]
pub struct ImportedDroneData {
    pub source_files: Vec<PathBuf>,
    pub pos_world: Vec<f32>,
    pub rotmat_world_from_body: Vec<f32>,
    pub lin_vel_body: Vec<f32>,
    pub ang_vel_body: Vec<f32>,
    pub vbat: Vec<f32>,
    pub channels_raw: Vec<f32>,
    pub channels_norm: Vec<f32>,
    pub accel_body: Vec<f32>,
    pub gyro_body: Vec<f32>,
    pub episode_idx: Vec<i64>,
    pub step_idx: Vec<i64>,
    pub elapsed_time: Vec<f32>,
    pub dt: Vec<f32>,
    pub gates: Vec<FlightGates>,
}

impl ImportedDroneData {
    pub fn rows(&self) -> usize {
        self.episode_idx.len()
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        let rows = self.rows();
        ensure!(rows > 0, "import produced zero rows");
        for (name, len, dim) in [
            ("pos_world", self.pos_world.len(), 3),
            (
                "rotmat_world_from_body",
                self.rotmat_world_from_body.len(),
                9,
            ),
            ("lin_vel_body", self.lin_vel_body.len(), 3),
            ("ang_vel_body", self.ang_vel_body.len(), 3),
            ("vbat", self.vbat.len(), 1),
            ("channels_raw", self.channels_raw.len(), 4),
            ("channels_norm", self.channels_norm.len(), 4),
            ("accel_body", self.accel_body.len(), 3),
            ("gyro_body", self.gyro_body.len(), 3),
            ("elapsed_time", self.elapsed_time.len(), 1),
            ("dt", self.dt.len(), 1),
        ] {
            ensure!(
                len == rows * dim,
                "{name} has {len} values, expected {}",
                rows * dim
            );
        }
        ensure!(
            self.step_idx.len() == rows,
            "step_idx length {} does not match rows {rows}",
            self.step_idx.len()
        );
        Ok(())
    }

    pub fn write_artifact(
        &self,
        output_dir: &Path,
        source_root: &Path,
        sample_rate_hz: usize,
        source_rate_hz: usize,
        eval_fraction: f32,
    ) -> anyhow::Result<DroneRacingMetadata> {
        self.validate()?;
        fs::create_dir_all(output_dir)
            .with_context(|| format!("failed to create {}", output_dir.display()))?;
        let data_h5 = output_dir.join("data.h5");
        let file = File::create(&data_h5)
            .with_context(|| format!("failed to create {}", data_h5.display()))?;
        write_f32_dataset(&file, "pos_world", self.rows(), 3, &self.pos_world)?;
        write_f32_dataset(
            &file,
            "rotmat_world_from_body",
            self.rows(),
            9,
            &self.rotmat_world_from_body,
        )?;
        write_f32_dataset(&file, "lin_vel_body", self.rows(), 3, &self.lin_vel_body)?;
        write_f32_dataset(&file, "ang_vel_body", self.rows(), 3, &self.ang_vel_body)?;
        write_f32_dataset(&file, "vbat", self.rows(), 1, &self.vbat)?;
        write_f32_dataset(&file, "channels_raw", self.rows(), 4, &self.channels_raw)?;
        write_f32_dataset(&file, "channels_norm", self.rows(), 4, &self.channels_norm)?;
        write_f32_dataset(&file, "accel_body", self.rows(), 3, &self.accel_body)?;
        write_f32_dataset(&file, "gyro_body", self.rows(), 3, &self.gyro_body)?;
        write_i64_dataset(&file, "episode_idx", self.rows(), &self.episode_idx)?;
        write_i64_dataset(&file, "step_idx", self.rows(), &self.step_idx)?;
        write_f32_dataset(&file, "elapsed_time", self.rows(), 1, &self.elapsed_time)?;
        write_f32_dataset(&file, "dt", self.rows(), 1, &self.dt)?;

        let episodes = self
            .episode_idx
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let eval_count = ((episodes.len() as f32) * eval_fraction)
            .round()
            .clamp(1.0, episodes.len().max(1) as f32) as usize;
        let split_at = episodes.len().saturating_sub(eval_count);
        let train_episodes = episodes[..split_at].to_vec();
        let eval_episodes = episodes[split_at..].to_vec();
        let normalization = compute_normalization(self, &train_episodes)?;
        let metadata = DroneRacingMetadata {
            source_root: source_root.to_path_buf(),
            data_h5: PathBuf::from("data.h5"),
            source_files: self.source_files.clone(),
            sample_rate_hz,
            source_rate_hz,
            rows: self.rows(),
            episodes: episodes.len(),
            action_dim: DRONE_ACTION_DIM,
            observation_dim: DRONE_OBSERVATION_DIM,
            state_delta_dim: DRONE_STATE_DELTA_DIM,
            train_episodes,
            eval_episodes,
            normalization,
            columns: DroneColumns::default(),
        };
        write_pretty_json(&output_dir.join("metadata.json"), &metadata)?;
        write_pretty_json(
            &output_dir.join("gates.json"),
            &GateSequenceFile {
                flights: self.gates.clone(),
            },
        )?;
        Ok(metadata)
    }
}

pub fn normalize_channels(raw: [f32; 4]) -> [f32; 4] {
    [
        ((raw[0] - 1500.0) / 500.0).clamp(-1.0, 1.0),
        ((raw[1] - 1500.0) / 500.0).clamp(-1.0, 1.0),
        ((raw[2] - 1000.0) / 1000.0).clamp(0.0, 1.0),
        ((raw[3] - 1500.0) / 500.0).clamp(-1.0, 1.0),
    ]
}

pub fn epoch_seed(seed: u64, epoch: usize) -> u64 {
    seed.wrapping_add((epoch as u64).wrapping_mul(0x9E3779B97F4A7C15))
}

fn compute_normalization(
    data: &ImportedDroneData,
    train_episodes: &[i64],
) -> anyhow::Result<DroneNormalization> {
    let train = train_episodes.iter().copied().collect::<BTreeSet<_>>();
    let rows = (0..data.rows())
        .filter(|row| train.contains(&data.episode_idx[*row]))
        .collect::<Vec<_>>();
    ensure!(!rows.is_empty(), "no rows selected for normalization");
    let mut obs_stats = StatsAccumulator::new(DRONE_OBSERVATION_DIM);
    let mut action_stats = StatsAccumulator::new(DRONE_ACTION_DIM);
    let mut target_stats = StatsAccumulator::new(DRONE_STATE_DELTA_DIM);
    for row in rows {
        let obs = raw_observation(data, row);
        let target = raw_target_delta(data, row);
        obs_stats.push(&obs);
        action_stats.push(&data.channels_norm[row * 4..row * 4 + 4]);
        target_stats.push(&target);
    }
    Ok(DroneNormalization {
        observation: obs_stats.finish(),
        action: action_stats.finish(),
        target_delta: target_stats.finish(),
    })
}

fn raw_observation(data: &ImportedDroneData, row: usize) -> [f32; DRONE_OBSERVATION_DIM] {
    let mut out = [0f32; DRONE_OBSERVATION_DIM];
    out[0..3].copy_from_slice(&data.pos_world[row * 3..row * 3 + 3]);
    out[3..12].copy_from_slice(&data.rotmat_world_from_body[row * 9..row * 9 + 9]);
    out
}

fn raw_target_delta(data: &ImportedDroneData, row: usize) -> [f32; DRONE_STATE_DELTA_DIM] {
    let next = if row + 1 < data.rows() && data.episode_idx[row + 1] == data.episode_idx[row] {
        row + 1
    } else {
        row
    };
    let pos = vec3_at(&data.pos_world, row);
    let next_pos = vec3_at(&data.pos_world, next);
    let rot = mat9_at(&data.rotmat_world_from_body, row);
    let next_rot = mat9_at(&data.rotmat_world_from_body, next);
    let delta_body = mat3_t_mul_vec3(rot, sub3(next_pos, pos));
    let rel_rot = mat3_mul(mat3_transpose(rot), next_rot);
    let delta_rot = rotvec_from_mat3(rel_rot);
    let mut out = [0f32; DRONE_STATE_DELTA_DIM];
    out[0..3].copy_from_slice(&delta_body);
    out[3..6].copy_from_slice(&delta_rot);
    out[6..9].copy_from_slice(&data.lin_vel_body[next * 3..next * 3 + 3]);
    out[9..12].copy_from_slice(&data.ang_vel_body[next * 3..next * 3 + 3]);
    out[12] = data.vbat[next] - data.vbat[row];
    out
}

fn read_f32_dataset(file: &File, name: &str, rows: usize, dim: usize) -> anyhow::Result<Vec<f32>> {
    let dataset = file
        .dataset(name)
        .with_context(|| format!("missing drone HDF5 dataset `{name}`"))?;
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
        .with_context(|| format!("missing drone HDF5 dataset `{name}`"))?;
    ensure!(
        dataset.shape() == [rows],
        "dataset `{name}` shape {:?} does not match [{rows}]",
        dataset.shape()
    );
    dataset
        .read_raw::<i64>()
        .with_context(|| format!("failed to read `{name}`"))
}

fn write_f32_dataset(
    file: &File,
    name: &str,
    rows: usize,
    dim: usize,
    values: &[f32],
) -> anyhow::Result<()> {
    ensure!(
        values.len() == rows * dim,
        "{name} value count {} does not match rows*dim {}",
        values.len(),
        rows * dim
    );
    let ds = file
        .new_dataset::<f32>()
        .shape((rows, dim))
        .create(name)
        .with_context(|| format!("failed to create `{name}`"))?;
    ds.write_raw(values)
        .with_context(|| format!("failed to write `{name}`"))
}

fn write_i64_dataset(file: &File, name: &str, rows: usize, values: &[i64]) -> anyhow::Result<()> {
    ensure!(
        values.len() == rows,
        "{name} value count {} does not match rows {rows}",
        values.len()
    );
    let ds = file
        .new_dataset::<i64>()
        .shape(rows)
        .create(name)
        .with_context(|| format!("failed to create `{name}`"))?;
    ds.write_raw(values)
        .with_context(|| format!("failed to write `{name}`"))
}

fn episode_lengths(episode_idx: &[i64], step_idx: &[i64]) -> anyhow::Result<Vec<usize>> {
    ensure!(
        episode_idx.len() == step_idx.len(),
        "episode_idx and step_idx lengths differ"
    );
    let mut max_step = BTreeMap::<i64, i64>::new();
    for (&episode, &step) in episode_idx.iter().zip(step_idx.iter()) {
        max_step
            .entry(episode)
            .and_modify(|value| *value = (*value).max(step))
            .or_insert(step);
    }
    let mut lengths = Vec::new();
    for (episode, step) in max_step {
        ensure!(episode >= 0, "negative episode index {episode}");
        let idx = episode as usize;
        if lengths.len() <= idx {
            lengths.resize(idx + 1, 0);
        }
        lengths[idx] = usize::try_from(step + 1).context("episode length overflow")?;
    }
    Ok(lengths)
}

fn compute_valid_rows(
    episode_idx: &[i64],
    step_idx: &[i64],
    ep_len: &[usize],
    sequence_steps: usize,
) -> anyhow::Result<Vec<usize>> {
    let mut rows = Vec::new();
    for row in 0..episode_idx.len() {
        let episode = usize::try_from(episode_idx[row])
            .with_context(|| format!("episode_idx[{row}] is negative"))?;
        ensure!(
            episode < ep_len.len(),
            "episode_idx[{row}]={episode} outside ep_len length {}",
            ep_len.len()
        );
        let step = usize::try_from(step_idx[row])
            .with_context(|| format!("step_idx[{row}] is negative"))?;
        if step + sequence_steps <= ep_len[episode] {
            rows.push(row);
        }
    }
    Ok(rows)
}

fn normalize_in_place(values: &mut [f32], stats: &RunningStats) -> anyhow::Result<()> {
    ensure!(
        values.len() == stats.mean.len() && values.len() == stats.std.len(),
        "normalization dim mismatch: values={} mean={} std={}",
        values.len(),
        stats.mean.len(),
        stats.std.len()
    );
    normalize_in_place_unchecked(values, stats);
    Ok(())
}

fn normalize_in_place_unchecked(values: &mut [f32], stats: &RunningStats) {
    for (idx, value) in values.iter_mut().enumerate() {
        *value = (*value - stats.mean[idx]) / stats.std[idx].max(1e-6);
    }
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

    fn push(&mut self, values: &[f32]) {
        self.count += 1;
        for (idx, value) in values.iter().enumerate() {
            let value = f64::from(*value);
            self.sum[idx] += value;
            self.sum_sq[idx] += value * value;
        }
    }

    fn finish(self) -> RunningStats {
        if self.count == 0 {
            return RunningStats::identity(self.sum.len());
        }
        let count = self.count as f64;
        let mut mean = Vec::with_capacity(self.sum.len());
        let mut std = Vec::with_capacity(self.sum.len());
        for idx in 0..self.sum.len() {
            let m = self.sum[idx] / count;
            let variance = (self.sum_sq[idx] / count - m * m).max(1e-12);
            mean.push(m as f32);
            std.push(variance.sqrt().max(1e-6) as f32);
        }
        RunningStats { mean, std }
    }
}

fn write_pretty_json<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    let json = serde_json::to_string_pretty(value)?;
    fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))
}

pub fn vec3_at(values: &[f32], row: usize) -> [f32; 3] {
    [values[row * 3], values[row * 3 + 1], values[row * 3 + 2]]
}

pub fn vec4_at(values: &[f32], row: usize) -> [f32; 4] {
    [
        values[row * 4],
        values[row * 4 + 1],
        values[row * 4 + 2],
        values[row * 4 + 3],
    ]
}

pub fn mat9_at(values: &[f32], row: usize) -> [f32; 9] {
    let base = row * 9;
    [
        values[base],
        values[base + 1],
        values[base + 2],
        values[base + 3],
        values[base + 4],
        values[base + 5],
        values[base + 6],
        values[base + 7],
        values[base + 8],
    ]
}

pub fn sub3(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]]
}

pub fn add3(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]]
}

pub fn scale3(value: [f32; 3], scale: f32) -> [f32; 3] {
    [value[0] * scale, value[1] * scale, value[2] * scale]
}

pub fn dot3(lhs: [f32; 3], rhs: [f32; 3]) -> f32 {
    lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
}

pub fn cross3(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    ]
}

pub fn norm3(value: [f32; 3]) -> f32 {
    dot3(value, value).sqrt()
}

pub fn normalize3(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let norm = norm3(value);
    if norm > 1e-6 {
        scale3(value, 1.0 / norm)
    } else {
        fallback
    }
}

pub fn mat3_transpose(m: [f32; 9]) -> [f32; 9] {
    [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]
}

pub fn mat3_mul(lhs: [f32; 9], rhs: [f32; 9]) -> [f32; 9] {
    let mut out = [0f32; 9];
    for r in 0..3 {
        for c in 0..3 {
            out[r * 3 + c] =
                lhs[r * 3] * rhs[c] + lhs[r * 3 + 1] * rhs[3 + c] + lhs[r * 3 + 2] * rhs[6 + c];
        }
    }
    out
}

pub fn mat3_mul_vec3(m: [f32; 9], v: [f32; 3]) -> [f32; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

pub fn mat3_t_mul_vec3(m: [f32; 9], v: [f32; 3]) -> [f32; 3] {
    [
        m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
        m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
        m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
    ]
}

pub fn rotvec_from_mat3(m: [f32; 9]) -> [f32; 3] {
    let trace = (m[0] + m[4] + m[8]).clamp(-1.0, 3.0);
    let cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    let skew = [m[7] - m[5], m[2] - m[6], m[3] - m[1]];
    if theta.abs() < 1e-5 {
        return scale3(skew, 0.5);
    }
    let denom = (2.0 * theta.sin()).abs().max(1e-6);
    scale3(skew, theta / denom)
}

pub fn mat3_from_rotvec(v: [f32; 3]) -> [f32; 9] {
    let theta = norm3(v);
    if theta < 1e-6 {
        return [1.0, -v[2], v[1], v[2], 1.0, -v[0], -v[1], v[0], 1.0];
    }
    let axis = scale3(v, 1.0 / theta);
    let (x, y, z) = (axis[0], axis[1], axis[2]);
    let c = theta.cos();
    let s = theta.sin();
    let one_c = 1.0 - c;
    [
        c + x * x * one_c,
        x * y * one_c - z * s,
        x * z * one_c + y * s,
        y * x * one_c + z * s,
        c + y * y * one_c,
        y * z * one_c - x * s,
        z * x * one_c - y * s,
        z * y * one_c + x * s,
        c + z * z * one_c,
    ]
}

pub fn shuffle<T>(values: &mut [T], seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for idx in (1..values.len()).rev() {
        let swap_idx = rng.next_usize(idx + 1);
        values.swap(idx, swap_idx);
    }
}

#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_usize(&mut self, upper: usize) -> usize {
        (self.next_u64() as usize) % upper
    }
}
