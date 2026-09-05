//! Batched rigid-body baseline for matched-budget MPPI ablations. The CUDA
//! path advances a complete candidate in one thread; no host candidate loop.
use crate::skyjepa_sim::{SkyJepaDomain, SkyJepaRotorPlant, SkyJepaRotorState};
use candle::{
    CudaStorage, DType, Result, Storage, Tensor,
    cuda_backend::{
        WrapErr,
        cudarc::{
            driver::{LaunchConfig, PushKernelArg},
            nvrtc,
        },
    },
    op::BackpropOp,
};
use std::sync::OnceLock;

pub fn nominal_physics_rollout(
    initial: &Tensor,
    actions: &Tensor,
    motors: &Tensor,
    dt: f32,
    substeps: usize,
    domain: SkyJepaDomain,
) -> Result<Tensor> {
    domain
        .validate()
        .map_err(|error| candle::Error::Msg(error.to_string()))?;
    let (batch, steps, dim) = actions.dims3()?;
    if batch == 0
        || steps == 0
        || dim != 4
        || initial.dims() != [batch, 18]
        || motors.dims() != [batch, 4]
        || !dt.is_finite()
        || dt <= 0.0
        || substeps == 0
        || substeps > u32::MAX as usize
    {
        candle::bail!("invalid nominal physics rollout shape or timestep");
    }
    for tensor in [initial, actions, motors] {
        if tensor.dtype() != DType::F32 || tensor.device().location() != actions.device().location()
        {
            candle::bail!("nominal physics requires f32 tensors on one device");
        }
    }
    if actions.device().is_cuda() {
        return cuda_rollout(initial, actions, motors, dt, substeps, domain);
    }
    let initial = initial.to_vec2::<f32>()?;
    let actions = actions.to_vec3::<f32>()?;
    let motors = motors.to_vec2::<f32>()?;
    let mut output = Vec::with_capacity(batch * steps * 18);
    for index in 0..batch {
        let state = SkyJepaRotorState {
            position: initial[index][0..3].try_into().unwrap(),
            velocity: initial[index][3..6].try_into().unwrap(),
            rotation_world_from_body: initial[index][6..15].try_into().unwrap(),
            angular_velocity: initial[index][15..18].try_into().unwrap(),
        };
        let mut plant = SkyJepaRotorPlant::new(domain, state)
            .and_then(|plant| plant.with_motor_forces(motors[index].as_slice().try_into().unwrap()))
            .map_err(|error| candle::Error::Msg(error.to_string()))?;
        for action in &actions[index] {
            for _ in 0..substeps {
                plant.step(action.as_slice().try_into().unwrap(), dt / substeps as f32);
            }
            output.extend_from_slice(&plant.state().as_state18());
        }
    }
    Tensor::from_vec(output, (batch, steps, 18), &candle::Device::Cpu)
}

fn cuda_rollout(
    initial: &Tensor,
    actions: &Tensor,
    motors: &Tensor,
    dt: f32,
    substeps: usize,
    domain: SkyJepaDomain,
) -> Result<Tensor> {
    let (batch, steps, _) = actions.dims3()?;
    if batch > u32::MAX as usize || steps > u32::MAX as usize {
        candle::bail!("nominal rollout index overflow");
    }
    let initial = initial.contiguous()?;
    let actions = actions.contiguous()?;
    let motors = motors.contiguous()?;
    let (initial_storage, initial_layout) = initial.storage_and_layout();
    let (action_storage, action_layout) = actions.storage_and_layout();
    let (motor_storage, motor_layout) = motors.storage_and_layout();
    let Storage::Cuda(initial_storage) = &*initial_storage else {
        candle::bail!("initial CUDA storage required");
    };
    let Storage::Cuda(action_storage) = &*action_storage else {
        candle::bail!("action CUDA storage required");
    };
    let Storage::Cuda(motor_storage) = &*motor_storage else {
        candle::bail!("motor CUDA storage required");
    };
    let initial_view = cuda_view(initial_storage, initial_layout)?;
    let action_view = cuda_view(action_storage, action_layout)?;
    let motor_view = cuda_view(motor_storage, motor_layout)?;
    let cuda = initial_storage.device.clone();
    let count = batch
        .checked_mul(steps)
        .and_then(|n| n.checked_mul(18))
        .ok_or_else(|| candle::Error::Msg("nominal rollout allocation overflow".into()))?;
    let mut output = unsafe { cuda.alloc::<f32>(count)? };
    let function = cuda.get_or_load_custom_func(
        "skyjepa_nominal_f32",
        "skyjepa_nominal_physics",
        cached_ptx()?,
    )?;
    let batch_u32 = batch as u32;
    let steps_u32 = steps as u32;
    let substeps_u32 = substeps as u32;
    let stream = cuda.cuda_stream();
    let mut launch = stream.launch_builder(&function);
    launch.arg(&initial_view);
    launch.arg(&action_view);
    launch.arg(&motor_view);
    launch.arg(&mut output);
    launch.arg(&batch_u32);
    launch.arg(&steps_u32);
    launch.arg(&dt);
    launch.arg(&substeps_u32);
    launch.arg(&domain.mass);
    launch.arg(&domain.inertia[0]);
    launch.arg(&domain.inertia[1]);
    launch.arg(&domain.inertia[2]);
    launch.arg(&domain.motor_time_constant);
    launch.arg(&domain.drag[0]);
    launch.arg(&domain.drag[1]);
    launch.arg(&domain.drag[2]);
    launch.arg(&domain.thrust_scale);
    launch.arg(&domain.torque_scale);
    launch.arg(&domain.arm_length);
    launch.arg(&domain.gravity);
    launch.arg(&domain.max_thrust_weight);
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (batch_u32.div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, cuda)),
        (batch, steps, 18),
        BackpropOp::none(),
        false,
    ))
}

fn cuda_view<'a>(
    storage: &'a CudaStorage,
    layout: &candle::Layout,
) -> Result<candle::cuda_backend::cudarc::driver::CudaView<'a, f32>> {
    let (start, end) = layout
        .contiguous_offsets()
        .ok_or_else(|| candle::Error::Msg("noncontiguous nominal input".into()))?;
    Ok(storage.as_cuda_slice::<f32>()?.slice(start..end))
}

static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
fn cached_ptx() -> Result<&'static str> {
    match PTX.get_or_init(|| {
        nvrtc::safe::compile_ptx_with_opts(
            include_str!("nominal.cu"),
            nvrtc::CompileOptions {
                use_fast_math: Some(false),
                ..Default::default()
            },
        )
        .map(|ptx| ptx.to_src())
        .map_err(|error| error.to_string())
    }) {
        Ok(ptx) => Ok(ptx),
        Err(error) => candle::bail!("nominal physics NVRTC failed: {error}"),
    }
}
