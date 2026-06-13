use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, ensure};
use candle::{DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use le_wm_nv::{
    data::drone_racing::{
        DRONE_ACTION_DIM, DRONE_OBSERVATION_DIM, DRONE_STATE_DELTA_DIM, DroneBatchConfig,
        DroneFrame, DroneRacingDataset, GateSequenceFile, RunningStats, add3, mat3_from_rotvec,
        mat3_mul, mat3_mul_vec3, norm3, scale3, sub3,
    },
    models::world_model::{
        VectorLossScalars, VectorLossWeights, WorldModel, WorldModelConfig, vector_batch_loss,
    },
    runtime::{DTypeSpec, DeviceSpec},
};
use serde::Serialize;

#[derive(Parser, Debug)]
struct Args {
    /// Imported drone dataset directory containing data.h5 and metadata.json.
    #[arg(long)]
    dataset_dir: Option<PathBuf>,

    /// Trained weights.
    #[arg(long)]
    weights: Option<PathBuf>,

    /// WorldModel config JSON.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Output report directory.
    #[arg(long)]
    output_dir: Option<PathBuf>,

    #[arg(long, default_value_t = DeviceSpec::Cuda(0))]
    device: DeviceSpec,

    #[arg(long, default_value_t = DTypeSpec::F32)]
    dtype: DTypeSpec,

    #[arg(long, default_value_t = 8)]
    history_steps: usize,

    #[arg(long, default_value_t = 100)]
    horizon_steps: usize,

    #[arg(long, default_value_t = 64)]
    batch_size: usize,

    #[arg(long, default_value_t = 8)]
    max_batches: usize,

    /// Dataset row to use for replay. Defaults to the highest-motion held-out window.
    #[arg(long)]
    replay_row: Option<usize>,

    /// Held-out episode to use for replay. Defaults to the episode containing the selected replay row.
    #[arg(long)]
    replay_episode: Option<i64>,

    /// Emit only one model horizon instead of the full selected episode.
    #[arg(long)]
    replay_window: bool,

    #[arg(long)]
    no_observation_normalize: bool,

    #[arg(long)]
    no_action_normalize: bool,

    #[arg(long)]
    no_target_normalize: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let dataset_dir = args.dataset_dir.clone().unwrap_or_else(default_dataset_dir);
    let weights = args.weights.clone().unwrap_or_else(default_weights);
    let config = args.config.clone().unwrap_or_else(default_config);
    let output_dir = args.output_dir.clone().unwrap_or_else(default_output_dir);
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let cfg: WorldModelConfig = serde_json::from_str(
        &fs::read_to_string(&config)
            .with_context(|| format!("failed to read {}", config.display()))?,
    )
    .with_context(|| format!("failed to parse {}", config.display()))?;
    let sequence_steps = cfg.predictor.num_frames;
    ensure!(
        sequence_steps > 1,
        "model config predictor.num_frames must be greater than one"
    );
    let replay_chunk_steps = args.horizon_steps.min(sequence_steps - 1);
    let batch_cfg = DroneBatchConfig {
        batch_size: args.batch_size,
        sequence_steps,
        normalize_observations: !args.no_observation_normalize,
        normalize_actions: !args.no_action_normalize,
        normalize_targets: !args.no_target_normalize,
    };
    let dataset = DroneRacingDataset::open(&dataset_dir, batch_cfg)?;
    let eval_rows = dataset.eval_rows();
    ensure!(!eval_rows.is_empty(), "dataset has no eval rows");
    let device = args.device.resolve()?;
    let dtype = args.dtype.dtype();
    if dtype != DType::F32 {
        anyhow::bail!("drone eval currently requires --dtype f32");
    }
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
    let model = WorldModel::new(cfg, vb)?;
    varmap
        .load(&weights)
        .with_context(|| format!("failed to load {}", weights.display()))?;

    let metrics = evaluate_batches(
        &model,
        &dataset,
        &eval_rows,
        args.max_batches,
        dtype,
        &device,
    )?;
    let candidate_replay_rows = replay_candidate_rows(&dataset, &eval_rows, args.replay_episode)?;
    let replay_row = select_replay_row(
        &dataset,
        &candidate_replay_rows,
        replay_chunk_steps,
        args.replay_row,
    )?;
    let replay_episode = args
        .replay_episode
        .unwrap_or(dataset.frame(replay_row)?.episode_idx);
    let replay = if args.replay_window {
        build_replay_window(
            &model,
            &dataset,
            replay_row,
            replay_chunk_steps,
            dtype,
            &device,
            !args.no_target_normalize,
        )?
    } else {
        build_episode_replay(
            &model,
            &dataset,
            replay_episode,
            replay_chunk_steps,
            dtype,
            &device,
            !args.no_target_normalize,
        )?
    };
    write_pretty_json(&output_dir.join("metrics.json"), &metrics)?;
    write_pretty_json(&output_dir.join("replay.json"), &replay)?;
    println!(
        "eval rows={} batches={} mean_total={:.8e} mean_state_prediction={:.8e}",
        eval_rows.len(),
        metrics.batches,
        metrics.mean_loss.total,
        metrics.mean_loss.state_prediction
    );
    println!(
        "replay kind={} prediction_mode={} row={} episode={} frames={} duration_s={:.2} actual_path_m={:.3} actual_net_m={:.3} model_chunk_steps={}",
        replay.replay_kind,
        replay.prediction_mode,
        replay.start_row,
        replay.episode_idx,
        replay.actual.len(),
        replay.duration_s,
        replay.actual_path_m,
        replay.actual_net_m,
        replay.model_chunk_steps,
    );
    println!("wrote {}", output_dir.join("replay.json").display());
    Ok(())
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    ensure!(
        args.history_steps > 0,
        "--history-steps must be greater than zero"
    );
    ensure!(
        args.horizon_steps > 0,
        "--horizon-steps must be greater than zero"
    );
    ensure!(
        args.batch_size > 0,
        "--batch-size must be greater than zero"
    );
    ensure!(
        args.max_batches > 0,
        "--max-batches must be greater than zero"
    );
    Ok(())
}

fn evaluate_batches(
    model: &WorldModel,
    dataset: &DroneRacingDataset,
    eval_rows: &[usize],
    max_batches: usize,
    dtype: DType,
    device: &candle::Device,
) -> anyhow::Result<EvalMetrics> {
    let batch_size = dataset.config().batch_size;
    let loss_weights = VectorLossWeights {
        state_prediction: 1.0,
        temporal_alignment: 0.0,
        std: 0.0,
        std_t: 0.0,
        covariance: 0.0,
        covariance_t: 0.0,
        temporal_straightening: 0.0,
    };
    let mut total = LossTotals::default();
    let mut batches = 0usize;
    for chunk in eval_rows.chunks(batch_size).take(max_batches) {
        let batch = dataset.batch(chunk, dtype, device)?;
        let loss = vector_batch_loss(
            model,
            &batch.observations,
            &batch.actions,
            &batch.target_deltas,
            loss_weights,
        )?;
        total.push(&VectorLossScalars::from_loss(&loss)?);
        batches += 1;
    }
    ensure!(batches > 0, "no eval batches were run");
    Ok(EvalMetrics {
        batches,
        mean_loss: total.mean(batches),
    })
}

fn replay_candidate_rows(
    dataset: &DroneRacingDataset,
    eval_rows: &[usize],
    replay_episode: Option<i64>,
) -> anyhow::Result<Vec<usize>> {
    let Some(episode) = replay_episode else {
        return Ok(eval_rows.to_vec());
    };
    let rows = eval_rows
        .iter()
        .copied()
        .filter(|row| {
            dataset
                .frame(*row)
                .is_ok_and(|frame| frame.episode_idx == episode)
        })
        .collect::<Vec<_>>();
    ensure!(
        !rows.is_empty(),
        "--replay-episode {episode} has no valid held-out replay rows"
    );
    Ok(rows)
}

fn select_replay_row(
    dataset: &DroneRacingDataset,
    eval_rows: &[usize],
    horizon_steps: usize,
    replay_row: Option<usize>,
) -> anyhow::Result<usize> {
    if let Some(row) = replay_row {
        ensure!(
            eval_rows.contains(&row),
            "--replay-row {row} is not a valid held-out replay row for this history/horizon"
        );
        return Ok(row);
    }
    let steps = horizon_steps.min(dataset.config().sequence_steps - 1);
    eval_rows
        .iter()
        .copied()
        .map(|row| Ok((row, replay_path_length(dataset, row, steps)?)))
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
        .map(|(row, _)| row)
        .context("failed to select replay row")
}

fn replay_path_length(
    dataset: &DroneRacingDataset,
    row: usize,
    steps: usize,
) -> anyhow::Result<f32> {
    let mut total = 0.0f32;
    let mut prev = dataset.frame(row)?.pos_world;
    for idx in 1..=steps {
        let current = dataset.frame(row + idx)?.pos_world;
        total += norm3(sub3(current, prev));
        prev = current;
    }
    Ok(total)
}

fn build_replay_window(
    model: &WorldModel,
    dataset: &DroneRacingDataset,
    row: usize,
    replay_chunk_steps: usize,
    dtype: DType,
    device: &candle::Device,
    target_normalized: bool,
) -> anyhow::Result<ReplayReport> {
    let batch = dataset.batch(&[row], dtype, device)?;
    let emb = model.encode_vector(&batch.observations)?;
    let pred_emb = model.predict(&emb, &batch.actions)?;
    let pred_delta = model.predict_state_deltas_from_embeddings(&pred_emb)?;
    let pred_values = pred_delta.flatten_all()?.to_vec1::<f32>()?;
    let steps = replay_chunk_steps.min(dataset.config().sequence_steps - 1);
    let mut actual = Vec::with_capacity(steps + 1);
    for idx in 0..=steps {
        actual.push(dataset.frame(row + idx)?);
    }
    let actual_path_m = path_length(&actual);
    let actual_net_m = net_displacement(&actual);
    let mut predicted = Vec::with_capacity(steps + 1);
    let mut baseline = Vec::with_capacity(steps + 1);
    predicted.push(actual[0].clone());
    baseline.push(actual[0].clone());
    let mut pred_state = actual[0].clone();
    let mut baseline_state = actual[0].clone();
    let start_vel_world = mat3_mul_vec3(actual[0].rotmat_world_from_body, actual[0].lin_vel_body);
    for step in 0..steps {
        let delta = denormalized_delta(
            &pred_values[step * DRONE_STATE_DELTA_DIM..(step + 1) * DRONE_STATE_DELTA_DIM],
            &dataset.metadata().normalization.target_delta,
            target_normalized,
        );
        pred_state = apply_delta(&pred_state, &delta);
        pred_state.row = actual[step + 1].row;
        pred_state.step_idx = actual[step + 1].step_idx;
        predicted.push(pred_state.clone());

        let dt = actual[step].dt;
        baseline_state.pos_world = add3(baseline_state.pos_world, scale3(start_vel_world, dt));
        baseline_state.row = actual[step + 1].row;
        baseline_state.step_idx = actual[step + 1].step_idx;
        baseline.push(baseline_state.clone());
    }

    let errors = frame_errors(&actual, &predicted, &baseline);

    Ok(ReplayReport {
        dataset_dir: dataset.root().to_path_buf(),
        source_data_h5: dataset.data_h5().to_path_buf(),
        replay_kind: "window".to_string(),
        prediction_mode: "single_open_loop_window".to_string(),
        start_row: row,
        episode_idx: actual[0].episode_idx,
        sample_rate_hz: dataset.metadata().sample_rate_hz,
        duration_s: replay_duration_s(&actual, dataset.metadata().sample_rate_hz),
        model_chunk_steps: steps,
        actual_path_m,
        actual_net_m,
        actual,
        predicted,
        baseline,
        errors,
        gates: read_gates(dataset.root()).unwrap_or(GateSequenceFile {
            flights: Vec::new(),
        }),
    })
}

fn build_episode_replay(
    model: &WorldModel,
    dataset: &DroneRacingDataset,
    episode: i64,
    replay_chunk_steps: usize,
    dtype: DType,
    device: &candle::Device,
    target_normalized: bool,
) -> anyhow::Result<ReplayReport> {
    let episode_rows = dataset.replay_rows_for_episode(episode);
    ensure!(!episode_rows.is_empty(), "episode {episode} has no rows");
    let actual = episode_rows
        .iter()
        .copied()
        .map(|row| dataset.frame(row))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let predicted =
        build_autoregressive_prediction(model, dataset, &actual, dtype, device, target_normalized)?;
    let baseline = build_chunked_baseline(&actual, replay_chunk_steps);
    let actual_path_m = path_length(&actual);
    let actual_net_m = net_displacement(&actual);
    let errors = frame_errors(&actual, &predicted, &baseline);
    Ok(ReplayReport {
        dataset_dir: dataset.root().to_path_buf(),
        source_data_h5: dataset.data_h5().to_path_buf(),
        replay_kind: "full_episode".to_string(),
        prediction_mode: "autoregressive_full_episode".to_string(),
        start_row: actual[0].row,
        episode_idx: episode,
        sample_rate_hz: dataset.metadata().sample_rate_hz,
        duration_s: replay_duration_s(&actual, dataset.metadata().sample_rate_hz),
        model_chunk_steps: actual.len().saturating_sub(1),
        actual_path_m,
        actual_net_m,
        actual,
        predicted,
        baseline,
        errors,
        gates: read_gates(dataset.root()).unwrap_or(GateSequenceFile {
            flights: Vec::new(),
        }),
    })
}

fn build_autoregressive_prediction(
    model: &WorldModel,
    dataset: &DroneRacingDataset,
    actual: &[DroneFrame],
    dtype: DType,
    device: &candle::Device,
    target_normalized: bool,
) -> anyhow::Result<Vec<DroneFrame>> {
    ensure!(
        !actual.is_empty(),
        "cannot build prediction for empty replay"
    );
    let history_size = model
        .config()
        .history_size
        .min(dataset.config().sequence_steps);
    ensure!(
        history_size > 0,
        "model history_size must be greater than zero"
    );
    if actual.len() <= history_size {
        return Ok(actual.to_vec());
    }

    let observations = initial_observations(dataset, actual, history_size, dtype, device)?;
    let emb_init = model.encode_vector(&observations)?.unsqueeze(1)?;
    let action_count = actual.len() - 1;
    let actions = replay_actions(dataset, actual, action_count, dtype, device)?
        .unsqueeze(0)?
        .unsqueeze(0)?;
    let rollout = model.rollout_embeddings_with_history(&emb_init, &actions, history_size)?;
    let (_, _, rollout_time, emb_dim) = rollout.dims4()?;
    let pred = model.predict_state_deltas_from_embeddings(&rollout.reshape((
        1,
        rollout_time,
        emb_dim,
    ))?)?;
    let pred_values = pred.flatten_all()?.to_vec1::<f32>()?;

    let mut predicted = actual[..history_size].to_vec();
    let mut pred_state = actual[history_size - 1].clone();
    for step in 0..actual.len() - history_size {
        let delta_idx = history_size + step;
        let delta = denormalized_delta(
            &pred_values
                [delta_idx * DRONE_STATE_DELTA_DIM..(delta_idx + 1) * DRONE_STATE_DELTA_DIM],
            &dataset.metadata().normalization.target_delta,
            target_normalized,
        );
        pred_state = apply_delta(&pred_state, &delta);
        let actual_frame = &actual[history_size + step];
        pred_state.row = actual_frame.row;
        pred_state.step_idx = actual_frame.step_idx;
        predicted.push(pred_state.clone());
    }
    Ok(predicted)
}

fn build_chunked_baseline(actual: &[DroneFrame], replay_chunk_steps: usize) -> Vec<DroneFrame> {
    if actual.is_empty() {
        return Vec::new();
    }
    let mut baseline = Vec::with_capacity(actual.len());
    baseline.push(actual[0].clone());
    let mut offset = 0usize;
    while offset + 1 < actual.len() {
        let steps = replay_chunk_steps.min(actual.len() - offset - 1);
        let mut state = actual[offset].clone();
        let start_vel_world = mat3_mul_vec3(state.rotmat_world_from_body, state.lin_vel_body);
        for local_step in 0..steps {
            let actual_frame = &actual[offset + local_step];
            let next_actual = &actual[offset + local_step + 1];
            state.pos_world = add3(state.pos_world, scale3(start_vel_world, actual_frame.dt));
            state.row = next_actual.row;
            state.step_idx = next_actual.step_idx;
            baseline.push(state.clone());
        }
        offset += steps;
    }
    baseline
}

fn initial_observations(
    dataset: &DroneRacingDataset,
    actual: &[DroneFrame],
    history_size: usize,
    dtype: DType,
    device: &candle::Device,
) -> anyhow::Result<Tensor> {
    let mut values = vec![0f32; history_size * DRONE_OBSERVATION_DIM];
    for idx in 0..history_size {
        let frame = &actual[idx];
        let base = idx * DRONE_OBSERVATION_DIM;
        values[base..base + 9].copy_from_slice(&frame.rotmat_world_from_body);
        values[base + 9..base + 12].copy_from_slice(&frame.lin_vel_body);
        values[base + 12..base + 15].copy_from_slice(&frame.ang_vel_body);
        values[base + 15] = frame.vbat;
        let prev_action = if idx > 0 {
            actual[idx - 1].channels_norm
        } else {
            frame.channels_norm
        };
        values[base + 16..base + 20].copy_from_slice(&prev_action);
        if dataset.config().normalize_observations {
            normalize_in_place(
                &mut values[base..base + DRONE_OBSERVATION_DIM],
                &dataset.metadata().normalization.observation,
            )?;
        }
    }
    Ok(
        Tensor::from_vec(values, (1, history_size, DRONE_OBSERVATION_DIM), device)?
            .to_dtype(dtype)?,
    )
}

fn replay_actions(
    dataset: &DroneRacingDataset,
    actual: &[DroneFrame],
    action_count: usize,
    dtype: DType,
    device: &candle::Device,
) -> anyhow::Result<Tensor> {
    let mut values = vec![0f32; action_count * DRONE_ACTION_DIM];
    for idx in 0..action_count {
        let base = idx * DRONE_ACTION_DIM;
        values[base..base + DRONE_ACTION_DIM].copy_from_slice(&actual[idx].channels_norm);
        if dataset.config().normalize_actions {
            normalize_in_place(
                &mut values[base..base + DRONE_ACTION_DIM],
                &dataset.metadata().normalization.action,
            )?;
        }
    }
    Ok(Tensor::from_vec(values, (action_count, DRONE_ACTION_DIM), device)?.to_dtype(dtype)?)
}

fn normalize_in_place(values: &mut [f32], stats: &RunningStats) -> anyhow::Result<()> {
    ensure!(
        values.len() == stats.mean.len() && values.len() == stats.std.len(),
        "normalization shape mismatch: values={} mean={} std={}",
        values.len(),
        stats.mean.len(),
        stats.std.len()
    );
    for (idx, value) in values.iter_mut().enumerate() {
        *value = (*value - stats.mean[idx]) / stats.std[idx].max(1e-6);
    }
    Ok(())
}

fn path_length(frames: &[DroneFrame]) -> f32 {
    frames
        .windows(2)
        .map(|pair| norm3(sub3(pair[1].pos_world, pair[0].pos_world)))
        .sum()
}

fn net_displacement(frames: &[DroneFrame]) -> f32 {
    let Some(first) = frames.first() else {
        return 0.0;
    };
    let Some(last) = frames.last() else {
        return 0.0;
    };
    norm3(sub3(last.pos_world, first.pos_world))
}

fn frame_errors(
    actual: &[DroneFrame],
    predicted: &[DroneFrame],
    baseline: &[DroneFrame],
) -> Vec<FrameError> {
    let len = actual.len().min(predicted.len()).min(baseline.len());
    (0..len)
        .map(|idx| FrameError {
            step: idx,
            position_error_m: norm3(sub3(predicted[idx].pos_world, actual[idx].pos_world)),
            baseline_position_error_m: norm3(sub3(baseline[idx].pos_world, actual[idx].pos_world)),
            attitude_error_rad: attitude_error(
                predicted[idx].rotmat_world_from_body,
                actual[idx].rotmat_world_from_body,
            ),
            velocity_error_mps: norm3(sub3(predicted[idx].lin_vel_body, actual[idx].lin_vel_body)),
            angular_velocity_error_radps: norm3(sub3(
                predicted[idx].ang_vel_body,
                actual[idx].ang_vel_body,
            )),
        })
        .collect()
}

fn replay_duration_s(frames: &[DroneFrame], sample_rate_hz: usize) -> f32 {
    if frames.len() <= 1 || sample_rate_hz == 0 {
        0.0
    } else {
        (frames.len() - 1) as f32 / sample_rate_hz as f32
    }
}

fn denormalized_delta(
    values: &[f32],
    stats: &le_wm_nv::data::drone_racing::RunningStats,
    normalized: bool,
) -> [f32; DRONE_STATE_DELTA_DIM] {
    let mut out = [0f32; DRONE_STATE_DELTA_DIM];
    for idx in 0..DRONE_STATE_DELTA_DIM {
        out[idx] = if normalized {
            values[idx] * stats.std[idx] + stats.mean[idx]
        } else {
            values[idx]
        };
    }
    out
}

fn apply_delta(frame: &DroneFrame, delta: &[f32; DRONE_STATE_DELTA_DIM]) -> DroneFrame {
    let delta_pos_body = [delta[0], delta[1], delta[2]];
    let delta_rot_body = [delta[3], delta[4], delta[5]];
    let delta_pos_world = mat3_mul_vec3(frame.rotmat_world_from_body, delta_pos_body);
    let delta_rot = mat3_from_rotvec(delta_rot_body);
    let next_rot = mat3_mul(frame.rotmat_world_from_body, delta_rot);
    DroneFrame {
        pos_world: add3(frame.pos_world, delta_pos_world),
        rotmat_world_from_body: next_rot,
        lin_vel_body: [delta[6], delta[7], delta[8]],
        ang_vel_body: [delta[9], delta[10], delta[11]],
        vbat: frame.vbat + delta[12],
        ..frame.clone()
    }
}

fn attitude_error(lhs: [f32; 9], rhs: [f32; 9]) -> f32 {
    let rel = mat3_mul(le_wm_nv::data::drone_racing::mat3_transpose(lhs), rhs);
    let trace = rel[0] + rel[4] + rel[8];
    (((trace - 1.0) * 0.5).clamp(-1.0, 1.0)).acos()
}

fn read_gates(dataset_dir: &Path) -> anyhow::Result<GateSequenceFile> {
    let path = dataset_dir.join("gates.json");
    serde_json::from_str(
        &fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))
}

fn default_dataset_dir() -> PathBuf {
    home_dir()
        .join(".stable_worldmodel")
        .join("le-wm-nv-data")
        .join("drone-racing-autonomous-100hz")
}

fn default_weights() -> PathBuf {
    home_dir()
        .join(".stable_worldmodel")
        .join("le-wm-nv-runs")
        .join("drone-state-lewm-autonomous-100hz")
        .join("latest.safetensors")
}

fn default_config() -> PathBuf {
    home_dir()
        .join(".stable_worldmodel")
        .join("le-wm-nv-runs")
        .join("drone-state-lewm-autonomous-100hz")
        .join("model-config.json")
}

fn default_output_dir() -> PathBuf {
    home_dir()
        .join(".stable_worldmodel")
        .join("le-wm-nv-reports")
        .join("drone-state-lewm-autonomous-100hz")
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn write_pretty_json<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(value)?;
    fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))
}

#[derive(Debug, Clone, Serialize)]
struct EvalMetrics {
    batches: usize,
    mean_loss: VectorLossScalars,
}

#[derive(Default)]
struct LossTotals {
    total: f64,
    state_prediction: f64,
    temporal_alignment: f64,
    std: f64,
    std_t: f64,
    covariance: f64,
    covariance_t: f64,
    temporal_straightening: f64,
}

impl LossTotals {
    fn push(&mut self, loss: &VectorLossScalars) {
        self.total += f64::from(loss.total);
        self.state_prediction += f64::from(loss.state_prediction);
        self.temporal_alignment += f64::from(loss.temporal_alignment);
        self.std += f64::from(loss.std);
        self.std_t += f64::from(loss.std_t);
        self.covariance += f64::from(loss.covariance);
        self.covariance_t += f64::from(loss.covariance_t);
        self.temporal_straightening += f64::from(loss.temporal_straightening);
    }

    fn mean(self, n: usize) -> VectorLossScalars {
        let n = n as f64;
        VectorLossScalars {
            total: (self.total / n) as f32,
            state_prediction: (self.state_prediction / n) as f32,
            temporal_alignment: (self.temporal_alignment / n) as f32,
            std: (self.std / n) as f32,
            std_t: (self.std_t / n) as f32,
            covariance: (self.covariance / n) as f32,
            covariance_t: (self.covariance_t / n) as f32,
            temporal_straightening: (self.temporal_straightening / n) as f32,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct ReplayReport {
    dataset_dir: PathBuf,
    source_data_h5: PathBuf,
    replay_kind: String,
    prediction_mode: String,
    start_row: usize,
    episode_idx: i64,
    sample_rate_hz: usize,
    duration_s: f32,
    model_chunk_steps: usize,
    actual_path_m: f32,
    actual_net_m: f32,
    actual: Vec<DroneFrame>,
    predicted: Vec<DroneFrame>,
    baseline: Vec<DroneFrame>,
    errors: Vec<FrameError>,
    gates: GateSequenceFile,
}

#[derive(Debug, Clone, Serialize)]
struct FrameError {
    step: usize,
    position_error_m: f32,
    baseline_position_error_m: f32,
    attitude_error_rad: f32,
    velocity_error_mps: f32,
    angular_velocity_error_radps: f32,
}
