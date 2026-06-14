use anyhow::ensure;
use candle::{DType, IndexOp, Tensor};
use serde::Serialize;

use crate::{
    data::drone_racing::{
        DRONE_ACTION_DIM, DRONE_STATE_DELTA_DIM, DroneFrame, RunningStats, add3, mat3_from_rotvec,
        mat3_mul, mat3_mul_vec3, norm3, sub3,
    },
    models::world_model::WorldModel,
    planner::ActionBounds,
};

pub const DRONE_ACTION_NAMES: [&str; DRONE_ACTION_DIM] = ["roll", "pitch", "throttle", "yaw"];

pub fn drone_action_bounds() -> ActionBounds {
    ActionBounds {
        low: vec![-1.0, -1.0, 0.0, -1.0],
        high: vec![1.0, 1.0, 1.0, 1.0],
    }
}

pub fn baseline_action(stats: &RunningStats) -> anyhow::Result<[f32; DRONE_ACTION_DIM]> {
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

pub fn history_action_prefix(
    history_actions: &Tensor,
    history_steps: usize,
) -> anyhow::Result<Tensor> {
    ensure!(
        history_steps >= 2,
        "history action prefix requires at least two history steps"
    );
    let (batch, time, action_dim) = history_actions.dims3()?;
    ensure!(batch == 1, "history action prefix expects batch=1");
    ensure!(
        time >= history_steps,
        "history action tensor has time={time}, expected at least {history_steps}"
    );
    ensure!(
        action_dim == DRONE_ACTION_DIM,
        "history action_dim {action_dim} does not match expected {DRONE_ACTION_DIM}"
    );
    Ok(history_actions
        .i((0, 0..history_steps - 1, ..))?
        .unsqueeze(0)?)
}

pub fn normalized_action_tensor(
    action: [f32; DRONE_ACTION_DIM],
    stats: &RunningStats,
    normalized: bool,
    dtype: DType,
    device: &candle::Device,
) -> anyhow::Result<Tensor> {
    let mut values = action;
    if normalized {
        for idx in 0..DRONE_ACTION_DIM {
            values[idx] = (values[idx] - stats.mean[idx]) / stats.std[idx].max(1e-6);
        }
    }
    Ok(Tensor::from_vec(values.to_vec(), (1, 1, DRONE_ACTION_DIM), device)?.to_dtype(dtype)?)
}

pub fn normalize_action_candidates(
    action_candidates: &Tensor,
    stats: &RunningStats,
    normalized: bool,
    dtype: DType,
    device: &candle::Device,
) -> candle::Result<Tensor> {
    if !normalized {
        return Ok(action_candidates.clone());
    }
    let mean = Tensor::from_vec(stats.mean.clone(), (1, 1, 1, DRONE_ACTION_DIM), device)?
        .to_dtype(dtype)?;
    let std = Tensor::from_vec(
        stats
            .std
            .iter()
            .map(|value| value.max(1e-6))
            .collect::<Vec<_>>(),
        (1, 1, 1, DRONE_ACTION_DIM),
        device,
    )?
    .to_dtype(dtype)?;
    action_candidates.broadcast_sub(&mean)?.broadcast_div(&std)
}

pub fn denormalize_delta_tensor(
    pred: Tensor,
    stats: &RunningStats,
    normalized: bool,
    dtype: DType,
    device: &candle::Device,
) -> candle::Result<Tensor> {
    if !normalized {
        return Ok(pred);
    }
    let mean = Tensor::from_vec(stats.mean.clone(), (1, 1, DRONE_STATE_DELTA_DIM), device)?
        .to_dtype(dtype)?;
    let std = Tensor::from_vec(
        stats
            .std
            .iter()
            .map(|value| value.max(1e-6))
            .collect::<Vec<_>>(),
        (1, 1, DRONE_STATE_DELTA_DIM),
        device,
    )?
    .to_dtype(dtype)?;
    pred.broadcast_mul(&std)?.broadcast_add(&mean)
}

pub fn denormalized_delta(
    values: &[f32],
    stats: &RunningStats,
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

pub fn rollout_deltas(
    model: &WorldModel,
    emb: &Tensor,
    action_prefix: &Tensor,
    raw_future_actions: &Tensor,
    action_stats: &RunningStats,
    target_stats: &RunningStats,
    action_normalized: bool,
    target_normalized: bool,
    dtype: DType,
    device: &candle::Device,
) -> anyhow::Result<Tensor> {
    let (_, samples, horizon, action_dim) = raw_future_actions.dims4()?;
    ensure!(
        action_dim == DRONE_ACTION_DIM,
        "future action dim {action_dim} does not match expected {DRONE_ACTION_DIM}"
    );
    let future_actions = normalize_action_candidates(
        raw_future_actions,
        action_stats,
        action_normalized,
        dtype,
        device,
    )?;
    let (_, history, emb_dim) = emb.dims3()?;
    let (_, prefix_len, prefix_dim) = action_prefix.dims3()?;
    ensure!(
        prefix_len + 1 == history && prefix_dim == DRONE_ACTION_DIM,
        "action prefix shape {:?} does not match history={history}",
        action_prefix.shape()
    );
    let emb_init = emb
        .unsqueeze(1)?
        .broadcast_as((1, samples, history, emb_dim))?;
    let prefix =
        action_prefix
            .unsqueeze(1)?
            .broadcast_as((1, samples, prefix_len, DRONE_ACTION_DIM))?;
    let model_actions = Tensor::cat(&[&prefix, &future_actions], 2)?;
    let rollout = model.rollout_embeddings_with_history(&emb_init, &model_actions, history)?;
    let future_emb = rollout
        .i((0, .., history..history + horizon, ..))?
        .contiguous()?;
    let pred = model
        .predict_state_deltas_from_embeddings(&future_emb.reshape((samples, horizon, emb_dim))?)?;
    Ok(denormalize_delta_tensor(
        pred,
        target_stats,
        target_normalized,
        dtype,
        device,
    )?)
}

pub fn apply_delta(frame: &DroneFrame, delta: &[f32; DRONE_STATE_DELTA_DIM]) -> DroneFrame {
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

pub fn rollout_one_step(
    model: &WorldModel,
    emb: &Tensor,
    action_prefix: &Tensor,
    current: &DroneFrame,
    action: [f32; DRONE_ACTION_DIM],
    action_stats: &RunningStats,
    target_stats: &RunningStats,
    action_normalized: bool,
    target_normalized: bool,
    dtype: DType,
    device: &candle::Device,
) -> anyhow::Result<(DroneFrame, Tensor, Tensor)> {
    let (_, history, emb_dim) = emb.dims3()?;
    let model_action =
        normalized_action_tensor(action, action_stats, action_normalized, dtype, device)?;
    let model_actions = Tensor::cat(&[action_prefix, &model_action], 1)?.unsqueeze(1)?;
    let rollout =
        model.rollout_embeddings_with_history(&emb.unsqueeze(1)?, &model_actions, history)?;
    let future = rollout
        .i((0, 0, history..history + 1, ..))?
        .reshape((1, 1, emb_dim))?;
    let pred = model.predict_state_deltas_from_embeddings(&future)?;
    let values = pred.flatten_all()?.to_vec1::<f32>()?;
    let delta = denormalized_delta(&values, target_stats, target_normalized);
    let mut next = apply_delta(current, &delta);
    next.row = next.row.saturating_add(1);
    next.step_idx += 1;
    next.channels_norm = action;
    let next_emb = rollout
        .i((0, 0, 1..history + 1, ..))?
        .reshape((1, history, emb_dim))?;
    let next_prefix = model_actions.i((0, 0, 1..history, ..))?.unsqueeze(0)?;
    Ok((next, next_emb, next_prefix))
}

pub fn integrate_future_deltas(
    current: &DroneFrame,
    deltas: &[f32],
    horizon: usize,
) -> anyhow::Result<Vec<DroneFrame>> {
    ensure!(
        deltas.len() >= horizon * DRONE_STATE_DELTA_DIM,
        "delta buffer has {}, expected at least {}",
        deltas.len(),
        horizon * DRONE_STATE_DELTA_DIM
    );
    let mut frames = Vec::with_capacity(horizon + 1);
    let mut frame = current.clone();
    frames.push(frame.clone());
    for step in 0..horizon {
        let offset = step * DRONE_STATE_DELTA_DIM;
        let mut delta = [0f32; DRONE_STATE_DELTA_DIM];
        delta.copy_from_slice(&deltas[offset..offset + DRONE_STATE_DELTA_DIM]);
        frame = apply_delta(&frame, &delta);
        frame.row = current.row + step + 1;
        frame.step_idx = current.step_idx + step as i64 + 1;
        frames.push(frame.clone());
    }
    Ok(frames)
}

pub fn frame_error(actual: &DroneFrame, predicted: &DroneFrame) -> FrameError {
    FrameError {
        position_error_m: norm3(sub3(predicted.pos_world, actual.pos_world)),
        attitude_error_rad: attitude_error(
            predicted.rotmat_world_from_body,
            actual.rotmat_world_from_body,
        ),
        velocity_error_mps: norm3(sub3(predicted.lin_vel_body, actual.lin_vel_body)),
        angular_velocity_error_radps: norm3(sub3(predicted.ang_vel_body, actual.ang_vel_body)),
    }
}

pub fn summarize_errors(errors: &[FrameError], sample_rate_hz: usize) -> HorizonErrorSummary {
    if errors.is_empty() {
        return HorizonErrorSummary {
            steps: 0,
            seconds: 0.0,
            position_mean_m: 0.0,
            position_rms_m: 0.0,
            position_max_m: 0.0,
            attitude_mean_rad: 0.0,
            attitude_max_rad: 0.0,
            velocity_mean_mps: 0.0,
            velocity_max_mps: 0.0,
        };
    }
    let mut pos_sum = 0.0f64;
    let mut pos_sq_sum = 0.0f64;
    let mut pos_max = 0.0f32;
    let mut att_sum = 0.0f64;
    let mut att_max = 0.0f32;
    let mut vel_sum = 0.0f64;
    let mut vel_max = 0.0f32;
    for err in errors {
        pos_sum += f64::from(err.position_error_m);
        pos_sq_sum += f64::from(err.position_error_m * err.position_error_m);
        pos_max = pos_max.max(err.position_error_m);
        att_sum += f64::from(err.attitude_error_rad);
        att_max = att_max.max(err.attitude_error_rad);
        vel_sum += f64::from(err.velocity_error_mps);
        vel_max = vel_max.max(err.velocity_error_mps);
    }
    let n = errors.len() as f64;
    HorizonErrorSummary {
        steps: errors.len(),
        seconds: errors.len() as f32 / sample_rate_hz.max(1) as f32,
        position_mean_m: (pos_sum / n) as f32,
        position_rms_m: (pos_sq_sum / n).sqrt() as f32,
        position_max_m: pos_max,
        attitude_mean_rad: (att_sum / n) as f32,
        attitude_max_rad: att_max,
        velocity_mean_mps: (vel_sum / n) as f32,
        velocity_max_mps: vel_max,
    }
}

fn attitude_error(lhs: [f32; 9], rhs: [f32; 9]) -> f32 {
    let rel = mat3_mul(crate::data::drone_racing::mat3_transpose(lhs), rhs);
    let trace = rel[0] + rel[4] + rel[8];
    (((trace - 1.0) * 0.5).clamp(-1.0, 1.0)).acos()
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct FrameError {
    pub position_error_m: f32,
    pub attitude_error_rad: f32,
    pub velocity_error_mps: f32,
    pub angular_velocity_error_radps: f32,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct HorizonErrorSummary {
    pub steps: usize,
    pub seconds: f32,
    pub position_mean_m: f32,
    pub position_rms_m: f32,
    pub position_max_m: f32,
    pub attitude_mean_rad: f32,
    pub attitude_max_rad: f32,
    pub velocity_mean_mps: f32,
    pub velocity_max_mps: f32,
}
