use std::sync::OnceLock;

use candle::{
    CudaStorage, DType, IndexOp, Result, Storage, Tensor,
    cuda_backend::{
        WrapErr,
        cudarc::{
            driver::{LaunchConfig, PushKernelArg},
            nvrtc,
        },
    },
    op::BackpropOp,
};
use serde::{Deserialize, Serialize};

use crate::data::skyjepa::{SKYJEPA_ACTION_DIM, SKYJEPA_STATE_DIM, SkyJepaActionSpace};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct KinematicConfig {
    pub mass: f64,
    pub gravity: f64,
    pub hover_throttle: f64,
    pub action_space: SkyJepaActionSpace,
}

impl KinematicConfig {
    pub fn validate(self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.mass.is_finite() && self.mass > 0.0,
            "mass must be positive"
        );
        anyhow::ensure!(
            self.gravity.is_finite() && self.gravity > 0.0,
            "gravity must be positive"
        );
        anyhow::ensure!(
            self.hover_throttle.is_finite() && self.hover_throttle > 0.0,
            "hover_throttle must be positive"
        );
        Ok(())
    }
}

impl Default for KinematicConfig {
    fn default() -> Self {
        Self {
            mass: 1.3,
            gravity: 9.81,
            hover_throttle: 0.2,
            action_space: SkyJepaActionSpace::BodyRatesThrottle,
        }
    }
}

/// Integrates the residual-corrected kinematics from the SkyJEPA paper.
/// Every rollout dimension stays batched on the selected Candle device.
pub fn integrate_metric_rollout(
    initial_state: &Tensor,
    actions: &Tensor,
    transition_dt: &Tensor,
    residual_acceleration: &Tensor,
    angular_action_map: &Tensor,
    config: KinematicConfig,
) -> Result<Tensor> {
    config
        .validate()
        .map_err(|error| candle::Error::Msg(error.to_string()))?;
    let (batch, state_dim) = initial_state.dims2()?;
    let (action_batch, steps, action_dim) = actions.dims3()?;
    let (dt_batch, dt_steps) = transition_dt.dims2()?;
    let (res_batch, res_steps, res_dim) = residual_acceleration.dims3()?;
    let map_dims = angular_action_map.dims();
    if state_dim != SKYJEPA_STATE_DIM
        || action_batch != batch
        || action_dim != SKYJEPA_ACTION_DIM
        || (dt_batch, dt_steps) != (batch, steps)
        || (res_batch, res_steps, res_dim) != (batch, steps, 3)
        || map_dims != [batch, steps, 3, SKYJEPA_ACTION_DIM]
    {
        candle::bail!(
            "invalid kinematic rollout shapes: state={:?} actions={:?} dt={:?} residual={:?} map={:?}",
            initial_state.shape(),
            actions.shape(),
            transition_dt.shape(),
            residual_acceleration.shape(),
            angular_action_map.shape()
        );
    }

    let mut position = initial_state.narrow(1, 0, 3)?;
    let mut velocity = initial_state.narrow(1, 3, 3)?;
    let mut rotation = initial_state
        .narrow(1, 6, 9)?
        .reshape((batch, 3, 3))?
        .contiguous()?;
    let mut angular_velocity = initial_state.narrow(1, 15, 3)?;
    let gravity = Tensor::from_vec(
        vec![0.0f32, 0.0, config.gravity as f32],
        (1, 3),
        initial_state.device(),
    )?
    .to_dtype(initial_state.dtype())?
    .broadcast_as((batch, 3))?;
    let mut predicted = Vec::with_capacity(steps);

    for step in 0..steps {
        let action = actions.i((.., step, ..))?.contiguous()?;
        let dt = transition_dt
            .i((.., step))?
            .reshape((batch, 1))?
            .contiguous()?;
        let residual = residual_acceleration.i((.., step, ..))?;
        let body_up = rotation.i((.., .., 2))?;
        let thrust_acceleration = match config.action_space {
            SkyJepaActionSpace::RotorForces => action.sum_keepdim(1)? / config.mass,
            SkyJepaActionSpace::BodyRatesThrottle => {
                let throttle = action.i((.., 2))?.clamp(0.0, 1.0)?.reshape((batch, 1))?;
                throttle * (config.gravity / config.hover_throttle)
            }
        }?;
        let acceleration =
            (body_up.broadcast_mul(&thrust_acceleration)? - &gravity)?.broadcast_add(&residual)?;
        let action_column = action.reshape((batch, SKYJEPA_ACTION_DIM, 1))?;
        let angular_acceleration = angular_action_map
            .i((.., step, .., ..))?
            .contiguous()?
            .matmul(&action_column)?
            .squeeze(2)?;

        position = (&position + velocity.broadcast_mul(&dt)?)?;
        velocity = (&velocity + acceleration.broadcast_mul(&dt)?)?;
        let rotation_delta = so3_exp(&angular_velocity.broadcast_mul(&dt)?)?;
        rotation = rotation
            .contiguous()?
            .matmul(&rotation_delta.contiguous()?)?;
        angular_velocity = (&angular_velocity + angular_acceleration.broadcast_mul(&dt)?)?;

        let flat_rotation = rotation.reshape((batch, 9))?;
        let state = Tensor::cat(
            &[&position, &velocity, &flat_rotation, &angular_velocity],
            1,
        )?;
        predicted.push(state);
    }
    let refs = predicted.iter().collect::<Vec<_>>();
    Tensor::stack(&refs, 1)
}

/// Inference-only integrator. CUDA f32 rollouts are fused into one kernel with
/// one thread per candidate trajectory; CPU and non-f32 calls use the
/// differentiable Candle implementation.
pub fn integrate_metric_rollout_inference(
    initial_state: &Tensor,
    actions: &Tensor,
    transition_dt: &Tensor,
    residual_acceleration: &Tensor,
    angular_action_map: &Tensor,
    config: KinematicConfig,
) -> Result<Tensor> {
    if initial_state.device().is_cuda()
        && initial_state.dtype() == DType::F32
        && actions.dtype() == DType::F32
        && transition_dt.dtype() == DType::F32
        && residual_acceleration.dtype() == DType::F32
        && angular_action_map.dtype() == DType::F32
    {
        return cuda_integrate_metric_rollout(
            initial_state,
            actions,
            transition_dt,
            residual_acceleration,
            angular_action_map,
            config,
        );
    }
    integrate_metric_rollout(
        initial_state,
        actions,
        transition_dt,
        residual_acceleration,
        angular_action_map,
        config,
    )
}

fn cuda_integrate_metric_rollout(
    initial_state: &Tensor,
    actions: &Tensor,
    transition_dt: &Tensor,
    residual_acceleration: &Tensor,
    angular_action_map: &Tensor,
    config: KinematicConfig,
) -> Result<Tensor> {
    config
        .validate()
        .map_err(|error| candle::Error::Msg(error.to_string()))?;
    let (batch, state_dim) = initial_state.dims2()?;
    let (action_batch, steps, action_dim) = actions.dims3()?;
    if state_dim != SKYJEPA_STATE_DIM
        || action_batch != batch
        || action_dim != SKYJEPA_ACTION_DIM
        || transition_dt.dims2()? != (batch, steps)
        || residual_acceleration.dims3()? != (batch, steps, 3)
        || angular_action_map.dims() != [batch, steps, 3, SKYJEPA_ACTION_DIM]
    {
        candle::bail!(
            "invalid fused kinematic rollout shapes: state={:?} actions={:?} dt={:?} residual={:?} map={:?}",
            initial_state.shape(),
            actions.shape(),
            transition_dt.shape(),
            residual_acceleration.shape(),
            angular_action_map.shape()
        );
    }
    for tensor in [
        actions,
        transition_dt,
        residual_acceleration,
        angular_action_map,
    ] {
        if tensor.device().location() != initial_state.device().location() {
            candle::bail!("fused SkyJEPA rollout tensors must share one CUDA device");
        }
    }

    let initial_state = initial_state.contiguous()?;
    let actions = actions.contiguous()?;
    let transition_dt = transition_dt.contiguous()?;
    let residual_acceleration = residual_acceleration.contiguous()?;
    let angular_action_map = angular_action_map.contiguous()?;
    let (initial_storage, initial_layout) = initial_state.storage_and_layout();
    let (action_storage, action_layout) = actions.storage_and_layout();
    let (dt_storage, dt_layout) = transition_dt.storage_and_layout();
    let (residual_storage, residual_layout) = residual_acceleration.storage_and_layout();
    let (map_storage, map_layout) = angular_action_map.storage_and_layout();
    let Storage::Cuda(initial_storage) = &*initial_storage else {
        candle::bail!("fused SkyJEPA rollout requires CUDA storage");
    };
    let Storage::Cuda(action_storage) = &*action_storage else {
        candle::bail!("fused SkyJEPA rollout requires CUDA storage");
    };
    let Storage::Cuda(dt_storage) = &*dt_storage else {
        candle::bail!("fused SkyJEPA rollout requires CUDA storage");
    };
    let Storage::Cuda(residual_storage) = &*residual_storage else {
        candle::bail!("fused SkyJEPA rollout requires CUDA storage");
    };
    let Storage::Cuda(map_storage) = &*map_storage else {
        candle::bail!("fused SkyJEPA rollout requires CUDA storage");
    };
    let initial_view = contiguous_cuda_view(initial_storage, initial_layout, "initial state")?;
    let action_view = contiguous_cuda_view(action_storage, action_layout, "actions")?;
    let dt_view = contiguous_cuda_view(dt_storage, dt_layout, "transition dt")?;
    let residual_view =
        contiguous_cuda_view(residual_storage, residual_layout, "residual acceleration")?;
    let map_view = contiguous_cuda_view(map_storage, map_layout, "angular action map")?;
    let cuda = initial_storage.device.clone();
    let elem_count = batch
        .checked_mul(steps)
        .and_then(|value| value.checked_mul(SKYJEPA_STATE_DIM))
        .ok_or_else(|| candle::Error::Msg("fused SkyJEPA output shape overflowed".into()))?;
    let mut output = unsafe { cuda.alloc::<f32>(elem_count)? };
    let ptx = cached_kinematics_ptx()?;
    let function =
        cuda.get_or_load_custom_func("lewm_skyjepa_integrate_f32", "lewm_skyjepa_kinematics", ptx)?;
    let block = 128u32;
    let grid = (batch as u32).div_ceil(block);
    let launch = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_u32 = batch as u32;
    let steps_u32 = steps as u32;
    let mass = config.mass as f32;
    let gravity = config.gravity as f32;
    let hover_throttle = config.hover_throttle as f32;
    let rotor_forces = u32::from(config.action_space == SkyJepaActionSpace::RotorForces);
    let stream = cuda.cuda_stream();
    let mut builder = stream.launch_builder(&function);
    builder.arg(&initial_view);
    builder.arg(&action_view);
    builder.arg(&dt_view);
    builder.arg(&residual_view);
    builder.arg(&map_view);
    builder.arg(&mut output);
    builder.arg(&batch_u32);
    builder.arg(&steps_u32);
    builder.arg(&mass);
    builder.arg(&gravity);
    builder.arg(&hover_throttle);
    builder.arg(&rotor_forces);
    unsafe { builder.launch(launch) }.w()?;
    let storage = CudaStorage::wrap_cuda_slice(output, cuda);
    Ok(Tensor::from_storage(
        Storage::Cuda(storage),
        (batch, steps, SKYJEPA_STATE_DIM),
        BackpropOp::none(),
        false,
    ))
}

fn contiguous_cuda_view<'a>(
    storage: &'a CudaStorage,
    layout: &candle::Layout,
    name: &str,
) -> Result<candle::cuda_backend::cudarc::driver::CudaView<'a, f32>> {
    let slice = storage.as_cuda_slice::<f32>()?;
    let Some((start, end)) = layout.contiguous_offsets() else {
        candle::bail!("fused SkyJEPA {name} must be contiguous");
    };
    Ok(slice.slice(start..end))
}

static KINEMATICS_PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();

fn cached_kinematics_ptx() -> Result<&'static str> {
    let cached = KINEMATICS_PTX.get_or_init(|| {
        nvrtc::safe::compile_ptx_with_opts(
            KINEMATICS_CUDA,
            nvrtc::CompileOptions {
                use_fast_math: Some(true),
                ..Default::default()
            },
        )
        .map(|ptx| ptx.to_src())
        .map_err(|error| error.to_string())
    });
    match cached {
        Ok(ptx) => Ok(ptx.as_str()),
        Err(error) => candle::bail!("SkyJEPA kinematics NVRTC compile failed: {error}"),
    }
}

/// Stable differentiable batched SO(3) exponential using Rodrigues' formula.
pub fn so3_exp(rotation_vector: &Tensor) -> Result<Tensor> {
    let (batch, dim) = rotation_vector.dims2()?;
    if dim != 3 {
        candle::bail!(
            "SO(3) exponential expects [batch, 3], got {:?}",
            rotation_vector.shape()
        );
    }
    let x = rotation_vector.i((.., 0))?;
    let y = rotation_vector.i((.., 1))?;
    let z = rotation_vector.i((.., 2))?;
    let zero = x.zeros_like()?;
    let neg_x = x.neg()?;
    let neg_y = y.neg()?;
    let neg_z = z.neg()?;
    let row0 = Tensor::stack(&[&zero, &neg_z, &y], 1)?;
    let row1 = Tensor::stack(&[&z, &zero, &neg_x], 1)?;
    let row2 = Tensor::stack(&[&neg_y, &x, &zero], 1)?;
    let skew = Tensor::stack(&[&row0, &row1, &row2], 1)?.contiguous()?;
    let skew_sq = skew.matmul(&skew)?;

    let theta_sq = rotation_vector
        .sqr()?
        .sum_keepdim(1)?
        .reshape((batch, 1, 1))?;
    let theta_four = theta_sq.sqr()?;
    let theta_safe = (&theta_sq + 1e-12)?.sqrt()?;
    let regular_a = theta_safe.sin()?.broadcast_div(&theta_safe)?;
    let regular_b = (theta_safe.cos()?.neg()? + 1.0)?.broadcast_div(&(&theta_sq + 1e-12)?)?;
    let series_a =
        ((&theta_sq * (-1.0 / 6.0))? + 1.0)?.broadcast_add(&(&theta_four * (1.0 / 120.0))?)?;
    let series_b =
        ((&theta_sq * (-1.0 / 24.0))? + 0.5)?.broadcast_add(&(&theta_four * (1.0 / 720.0))?)?;
    let small = theta_sq.lt(1e-6)?;
    let a = small.where_cond(&series_a, &regular_a)?;
    let b = small.where_cond(&series_b, &regular_b)?;
    let identity =
        Tensor::eye(3, rotation_vector.dtype(), rotation_vector.device())?.broadcast_left(batch)?;
    (identity + skew.broadcast_mul(&a)?)? + skew_sq.broadcast_mul(&b)?
}

const KINEMATICS_CUDA: &str = r#"
extern "C" __global__ void lewm_skyjepa_integrate_f32(
    const float* __restrict__ initial,
    const float* __restrict__ actions,
    const float* __restrict__ dt,
    const float* __restrict__ residual,
    const float* __restrict__ action_maps,
    float* __restrict__ output,
    unsigned int batch,
    unsigned int steps,
    float mass,
    float gravity,
    float hover_throttle,
    unsigned int rotor_forces
) {
    const unsigned int trajectory = blockIdx.x * blockDim.x + threadIdx.x;
    if (trajectory >= batch) return;

    const float* x0 = initial + (unsigned long long)trajectory * 18;
    float p[3] = {x0[0], x0[1], x0[2]};
    float v[3] = {x0[3], x0[4], x0[5]};
    float r[9];
    #pragma unroll
    for (int i = 0; i < 9; ++i) r[i] = x0[6 + i];
    float omega[3] = {x0[15], x0[16], x0[17]};

    for (unsigned int step = 0; step < steps; ++step) {
        const unsigned long long action_offset =
            ((unsigned long long)trajectory * steps + step) * 4;
        const unsigned long long vector_offset =
            ((unsigned long long)trajectory * steps + step) * 3;
        const float* action = actions + action_offset;
        const float step_dt = dt[(unsigned long long)trajectory * steps + step];
        float thrust_acceleration;
        if (rotor_forces) {
            thrust_acceleration =
                (action[0] + action[1] + action[2] + action[3]) / mass;
        } else {
            const float throttle = fminf(1.0f, fmaxf(0.0f, action[2]));
            thrust_acceleration = throttle * gravity / hover_throttle;
        }
        const float acceleration[3] = {
            r[2] * thrust_acceleration + residual[vector_offset],
            r[5] * thrust_acceleration + residual[vector_offset + 1],
            r[8] * thrust_acceleration - gravity + residual[vector_offset + 2]
        };
        const float* map = action_maps + action_offset * 3;
        float angular_acceleration[3];
        #pragma unroll
        for (int row = 0; row < 3; ++row) {
            angular_acceleration[row] =
                map[row * 4] * action[0] + map[row * 4 + 1] * action[1]
                + map[row * 4 + 2] * action[2] + map[row * 4 + 3] * action[3];
        }

        #pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
            p[axis] += v[axis] * step_dt;
            v[axis] += acceleration[axis] * step_dt;
        }

        const float wx = omega[0] * step_dt;
        const float wy = omega[1] * step_dt;
        const float wz = omega[2] * step_dt;
        const float theta_sq = wx * wx + wy * wy + wz * wz;
        float a;
        float b;
        if (theta_sq < 1.0e-6f) {
            const float theta_four = theta_sq * theta_sq;
            a = 1.0f - theta_sq / 6.0f + theta_four / 120.0f;
            b = 0.5f - theta_sq / 24.0f + theta_four / 720.0f;
        } else {
            const float theta = sqrtf(theta_sq);
            a = sinf(theta) / theta;
            b = (1.0f - cosf(theta)) / theta_sq;
        }
        const float delta[9] = {
            1.0f - b * (wy * wy + wz * wz), b * wx * wy - a * wz, b * wx * wz + a * wy,
            b * wy * wx + a * wz, 1.0f - b * (wx * wx + wz * wz), b * wy * wz - a * wx,
            b * wz * wx - a * wy, b * wz * wy + a * wx, 1.0f - b * (wx * wx + wy * wy)
        };
        float next_r[9];
        #pragma unroll
        for (int row = 0; row < 3; ++row) {
            #pragma unroll
            for (int col = 0; col < 3; ++col) {
                next_r[row * 3 + col] =
                    r[row * 3] * delta[col]
                    + r[row * 3 + 1] * delta[3 + col]
                    + r[row * 3 + 2] * delta[6 + col];
            }
        }
        #pragma unroll
        for (int i = 0; i < 9; ++i) r[i] = next_r[i];
        #pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
            omega[axis] += angular_acceleration[axis] * step_dt;
        }

        float* out = output + ((unsigned long long)trajectory * steps + step) * 18;
        out[0] = p[0]; out[1] = p[1]; out[2] = p[2];
        out[3] = v[0]; out[4] = v[1]; out[5] = v[2];
        #pragma unroll
        for (int i = 0; i < 9; ++i) out[6 + i] = r[i];
        out[15] = omega[0]; out[16] = omega[1]; out[17] = omega[2];
    }
}
"#;
