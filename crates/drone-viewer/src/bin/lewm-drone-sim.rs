use std::{env, fs, path::PathBuf, time::Instant};

use anyhow::{Context, ensure};
use bevy::prelude::*;
use candle::{DType, Device, IndexOp, Tensor};
use le_wm_nv::{
    checkpoint,
    data::drone_racing::{
        DRONE_ACTION_DIM, DRONE_STATE_DELTA_DIM, DroneBatchConfig, DroneFrame, DroneRacingDataset,
        RunningStats, add3, mat3_from_rotvec, mat3_mul, mat3_mul_vec3,
    },
    models::world_model::{WorldModel, WorldModelConfig},
    runtime::{DTypeSpec, DeviceSpec},
};

const TELEMETRY_FONT_SIZE: f32 = 0.11;

fn main() -> anyhow::Result<()> {
    let args = Args::parse()?;
    let sim = LewmDroneSim::load(&args)?;
    let render = SimRenderState::from_sim(&sim, args.max_trail, args.max_frame_steps);
    App::new()
        .insert_resource(args.camera)
        .insert_resource(InputControl::new(render.action))
        .insert_resource(render)
        .insert_non_send(sim)
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "le-wm-nv LeWM Drone Simulator".to_string(),
                resolution: (1280, 800).into(),
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                update_controls,
                step_lewm_sim,
                update_drone_mesh,
                update_follow_camera,
                draw_guides,
                draw_sim_overlays,
            )
                .chain(),
        )
        .run();
    Ok(())
}

#[derive(Debug, Clone)]
struct Args {
    dataset_dir: PathBuf,
    weights: PathBuf,
    config: PathBuf,
    row: Option<usize>,
    device: DeviceSpec,
    dtype: DTypeSpec,
    history_steps: usize,
    sim_hz: f32,
    max_frame_steps: usize,
    max_trail: usize,
    camera: FollowCameraConfig,
    normalize_observations: bool,
    normalize_actions: bool,
    normalize_targets: bool,
}

impl Args {
    fn parse() -> anyhow::Result<Self> {
        let mut args = Self {
            dataset_dir: default_dataset_dir(),
            weights: default_weights(),
            config: default_config(),
            row: None,
            device: DeviceSpec::Cuda(0),
            dtype: DTypeSpec::F32,
            history_steps: 8,
            sim_hz: 100.0,
            max_frame_steps: 4,
            max_trail: 2400,
            camera: FollowCameraConfig {
                distance: 7.0,
                height: 2.2,
                spring: 8.0,
            },
            normalize_observations: true,
            normalize_actions: true,
            normalize_targets: true,
        };

        let mut iter = env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--dataset-dir" => args.dataset_dir = next_path(&mut iter, &arg)?,
                "--weights" => args.weights = next_path(&mut iter, &arg)?,
                "--config" => args.config = next_path(&mut iter, &arg)?,
                "--row" => args.row = Some(next_parse(&mut iter, &arg)?),
                "--device" => args.device = next_parse(&mut iter, &arg)?,
                "--dtype" => args.dtype = next_parse(&mut iter, &arg)?,
                "--history-steps" => args.history_steps = next_parse(&mut iter, &arg)?,
                "--sim-hz" => args.sim_hz = next_parse(&mut iter, &arg)?,
                "--max-frame-steps" => args.max_frame_steps = next_parse(&mut iter, &arg)?,
                "--max-trail" => args.max_trail = next_parse(&mut iter, &arg)?,
                "--camera-distance" => args.camera.distance = next_parse(&mut iter, &arg)?,
                "--camera-height" => args.camera.height = next_parse(&mut iter, &arg)?,
                "--camera-spring" => args.camera.spring = next_parse(&mut iter, &arg)?,
                "--no-observation-normalize" => args.normalize_observations = false,
                "--no-action-normalize" => args.normalize_actions = false,
                "--no-target-normalize" => args.normalize_targets = false,
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                other => anyhow::bail!("unknown argument `{other}`; use --help"),
            }
        }

        ensure!(
            args.history_steps >= 2,
            "--history-steps must be at least two"
        );
        ensure!(args.sim_hz > 0.0, "--sim-hz must be greater than zero");
        ensure!(
            args.max_frame_steps > 0,
            "--max-frame-steps must be greater than zero"
        );
        ensure!(args.max_trail > 1, "--max-trail must be greater than one");
        ensure!(
            args.camera.distance > 0.25,
            "--camera-distance must be greater than 0.25"
        );
        ensure!(
            args.camera.height >= 0.0,
            "--camera-height must be non-negative"
        );
        ensure!(
            args.camera.spring > 0.0,
            "--camera-spring must be greater than zero"
        );
        Ok(args)
    }
}

fn next_path(iter: &mut impl Iterator<Item = String>, flag: &str) -> anyhow::Result<PathBuf> {
    iter.next()
        .map(PathBuf::from)
        .with_context(|| format!("missing value after {flag}"))
}

fn next_parse<T>(iter: &mut impl Iterator<Item = String>, flag: &str) -> anyhow::Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value = iter
        .next()
        .with_context(|| format!("missing value after {flag}"))?;
    value
        .parse::<T>()
        .map_err(|err| anyhow::anyhow!("invalid value `{value}` for {flag}: {err}"))
}

fn print_help() {
    println!(
        "Usage: lewm-drone-sim [options]\n\
         \n\
         Model/data options:\n\
           --dataset-dir <path>\n\
           --weights <path>\n\
           --config <path>\n\
           --row <usize>\n\
           --device cuda[:index]\n\
           --history-steps <usize>\n\
         \n\
         Sim options:\n\
           --sim-hz <hz>                 default 100\n\
           --max-frame-steps <n>         default 4\n\
           --max-trail <n>               default 2400\n\
         \n\
         Follow camera options:\n\
           --camera-distance <meters>    default 7\n\
           --camera-height <meters>      default 2.2\n\
           --camera-spring <rate>        default 8\n\
         \n\
         Controls:\n\
           W/S pitch, A/D roll, Q/E yaw, R/F throttle\n\
           Z zero roll/pitch/yaw, X set throttle to trim, P pause, Backspace reset\n\
           [/] camera distance, 3/4 camera height, 1/2 camera spring"
    );
}

#[derive(Resource, Debug, Clone)]
struct FollowCameraConfig {
    distance: f32,
    height: f32,
    spring: f32,
}

#[derive(Resource)]
struct InputControl {
    action: [f32; DRONE_ACTION_DIM],
    trim: [f32; DRONE_ACTION_DIM],
    paused: bool,
    accumulator: f32,
}

impl InputControl {
    fn new(trim: [f32; DRONE_ACTION_DIM]) -> Self {
        Self {
            action: trim,
            trim,
            paused: false,
            accumulator: 0.0,
        }
    }
}

#[derive(Resource, Clone)]
struct SimRenderState {
    frame: DroneFrame,
    action: [f32; DRONE_ACTION_DIM],
    step: usize,
    sim_hz: f32,
    last_step_ms: f32,
    avg_step_ms: f32,
    trail: Vec<[f32; 3]>,
    max_trail: usize,
    max_frame_steps: usize,
    error: Option<String>,
}

impl SimRenderState {
    fn from_sim(sim: &LewmDroneSim, max_trail: usize, max_frame_steps: usize) -> Self {
        Self {
            frame: sim.current.clone(),
            action: sim.action_trim,
            step: 0,
            sim_hz: sim.sim_hz,
            last_step_ms: 0.0,
            avg_step_ms: 0.0,
            trail: vec![sim.current.pos_world],
            max_trail,
            max_frame_steps,
            error: None,
        }
    }

    fn push_frame(&mut self, frame: DroneFrame, action: [f32; DRONE_ACTION_DIM], step_ms: f32) {
        self.frame = frame;
        self.action = action;
        self.step += 1;
        self.last_step_ms = step_ms;
        self.avg_step_ms = if self.avg_step_ms == 0.0 {
            step_ms
        } else {
            self.avg_step_ms * 0.95 + step_ms * 0.05
        };
        self.trail.push(self.frame.pos_world);
        if self.trail.len() > self.max_trail {
            let drop_count = self.trail.len() - self.max_trail;
            self.trail.drain(0..drop_count);
        }
    }

    fn reset_from_sim(&mut self, sim: &LewmDroneSim) {
        self.frame = sim.current.clone();
        self.action = sim.action_trim;
        self.step = 0;
        self.last_step_ms = 0.0;
        self.avg_step_ms = 0.0;
        self.trail.clear();
        self.trail.push(self.frame.pos_world);
        self.error = None;
    }
}

struct LewmDroneSim {
    dataset: DroneRacingDataset,
    model: WorldModel,
    device: Device,
    dtype: DType,
    row: usize,
    history_steps: usize,
    sim_hz: f32,
    normalize_actions: bool,
    normalize_targets: bool,
    emb: Tensor,
    action_prefix: Tensor,
    current: DroneFrame,
    action_trim: [f32; DRONE_ACTION_DIM],
}

impl LewmDroneSim {
    fn load(args: &Args) -> anyhow::Result<Self> {
        let dtype = args.dtype.dtype();
        ensure!(
            dtype == DType::F32,
            "drone simulator currently requires f32"
        );
        let device = args.device.resolve()?;
        ensure!(device.is_cuda(), "drone simulator requires CUDA");
        let batch_cfg = DroneBatchConfig {
            batch_size: 1,
            sequence_steps: args.history_steps.max(2),
            normalize_observations: args.normalize_observations,
            normalize_actions: args.normalize_actions,
            normalize_targets: args.normalize_targets,
        };
        let dataset = DroneRacingDataset::open(&args.dataset_dir, batch_cfg)?;
        let row = args
            .row
            .unwrap_or_else(|| dataset.eval_rows().first().copied().unwrap_or(0));
        let cfg: WorldModelConfig = serde_json::from_str(
            &fs::read_to_string(&args.config)
                .with_context(|| format!("failed to read {}", args.config.display()))?,
        )
        .with_context(|| format!("failed to parse {}", args.config.display()))?;
        let vb = checkpoint::var_builder_from_path(&args.weights, dtype, &device)
            .with_context(|| format!("failed to load {}", args.weights.display()))?;
        let model = WorldModel::new(cfg, vb)?;
        let action_trim = baseline_action(&dataset.metadata().normalization.action)?;
        let mut sim = Self {
            dataset,
            model,
            device,
            dtype,
            row,
            history_steps: args.history_steps,
            sim_hz: args.sim_hz,
            normalize_actions: args.normalize_actions,
            normalize_targets: args.normalize_targets,
            emb: Tensor::new(0f32, &Device::Cpu)?,
            action_prefix: Tensor::new(0f32, &Device::Cpu)?,
            current: DroneFrame {
                row,
                episode_idx: 0,
                step_idx: 0,
                dt: 1.0 / args.sim_hz,
                pos_world: [0.0, 0.0, 0.0],
                rotmat_world_from_body: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                lin_vel_body: [0.0, 0.0, 0.0],
                ang_vel_body: [0.0, 0.0, 0.0],
                vbat: 0.0,
                channels_norm: action_trim,
            },
            action_trim,
        };
        sim.reset()?;
        Ok(sim)
    }

    fn reset(&mut self) -> anyhow::Result<()> {
        let history = self.dataset.batch(&[self.row], self.dtype, &self.device)?;
        self.emb = self.model.encode_vector(&history.observations)?;
        self.action_prefix = history_action_prefix(&history.actions, self.history_steps)?;
        self.current = self.dataset.frame(self.row + self.history_steps - 1)?;
        self.current.channels_norm = self.action_trim;
        Ok(())
    }

    fn step(&mut self, action: [f32; DRONE_ACTION_DIM]) -> anyhow::Result<(DroneFrame, f32)> {
        let started = Instant::now();
        let model_action = normalized_action_tensor(
            action,
            &self.dataset.metadata().normalization.action,
            self.normalize_actions,
            self.dtype,
            &self.device,
        )?;
        let model_actions = Tensor::cat(&[&self.action_prefix, &model_action], 1)?.unsqueeze(1)?;
        let rollout = self.model.rollout_embeddings_with_history(
            &self.emb.unsqueeze(1)?,
            &model_actions,
            self.history_steps,
        )?;
        let (_, _, rollout_time, emb_dim) = rollout.dims4()?;
        ensure!(
            rollout_time >= self.history_steps + 1,
            "rollout_time {rollout_time} too short for one sim step"
        );
        let future = rollout
            .i((0, 0, self.history_steps..self.history_steps + 1, ..))?
            .reshape((1, 1, emb_dim))?;
        let pred = self.model.predict_state_deltas_from_embeddings(&future)?;
        let values = pred.flatten_all()?.to_vec1::<f32>()?;
        let delta = denormalized_delta(
            &values,
            &self.dataset.metadata().normalization.target_delta,
            self.normalize_targets,
        );
        self.current = apply_delta(&self.current, &delta);
        self.current.row = self.current.row.saturating_add(1);
        self.current.step_idx += 1;
        self.current.dt = 1.0 / self.sim_hz;
        self.current.channels_norm = action;
        self.emb = rollout.i((0, 0, 1..self.history_steps + 1, ..))?.reshape((
            1,
            self.history_steps,
            emb_dim,
        ))?;
        self.action_prefix = model_actions
            .i((0, 0, 1..self.history_steps, ..))?
            .unsqueeze(0)?;
        let step_ms = started.elapsed().as_secs_f32() * 1000.0;
        Ok((self.current.clone(), step_ms))
    }
}

#[derive(Component)]
enum DronePart {
    Body,
    ArmX,
    ArmY,
}

#[derive(Component)]
struct FollowCamera;

#[derive(Resource)]
struct SceneGuide {
    grid_extent: f32,
    grid_step: f32,
    axis_len: f32,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    state: Res<SimRenderState>,
    camera: Res<FollowCameraConfig>,
) {
    commands.insert_resource(SceneGuide {
        grid_extent: 40.0,
        grid_step: 1.0,
        axis_len: 8.0,
    });

    let frame_transform = transform_from_frame(&state.frame);
    let focus = frame_transform.translation + Vec3::Y * 0.4;
    let pos = focus + Vec3::new(0.0, camera.height, camera.distance);
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(pos).looking_at(focus, Vec3::Y),
        FollowCamera,
    ));

    commands.spawn((
        PointLight {
            intensity: 6000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(-8.0, 14.0, 10.0),
    ));

    let floor_mesh = meshes.add(Plane3d::default().mesh().size(80.0, 80.0));
    let floor_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.10, 0.105, 0.11),
        perceptual_roughness: 0.9,
        ..default()
    });
    commands.spawn((Mesh3d(floor_mesh), MeshMaterial3d(floor_mat)));

    let body_mesh = meshes.add(Cuboid::new(0.45, 0.12, 0.18));
    let arm_x_mesh = meshes.add(Cuboid::new(0.9, 0.04, 0.05));
    let arm_y_mesh = meshes.add(Cuboid::new(0.05, 0.04, 0.9));
    let drone_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.95, 0.88),
        ..default()
    });
    commands.spawn((
        Mesh3d(body_mesh),
        MeshMaterial3d(drone_mat.clone()),
        DronePart::Body,
    ));
    commands.spawn((
        Mesh3d(arm_x_mesh),
        MeshMaterial3d(drone_mat.clone()),
        DronePart::ArmX,
    ));
    commands.spawn((
        Mesh3d(arm_y_mesh),
        MeshMaterial3d(drone_mat),
        DronePart::ArmY,
    ));
}

fn update_controls(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut control: ResMut<InputControl>,
    mut camera: ResMut<FollowCameraConfig>,
    mut sim: NonSendMut<LewmDroneSim>,
    mut state: ResMut<SimRenderState>,
) {
    if keys.just_pressed(KeyCode::KeyP) {
        control.paused = !control.paused;
    }
    if keys.just_pressed(KeyCode::Backspace) {
        if let Err(err) = sim.reset() {
            state.error = Some(err.to_string());
            control.paused = true;
        } else {
            control.action = control.trim;
            control.accumulator = 0.0;
            state.reset_from_sim(&sim);
        }
    }
    if keys.just_pressed(KeyCode::KeyZ) {
        control.action[0] = 0.0;
        control.action[1] = 0.0;
        control.action[3] = 0.0;
    }
    if keys.just_pressed(KeyCode::KeyX) {
        control.action[2] = control.trim[2];
    }

    let dt = time.delta_secs();
    let target_roll = axis(keys.pressed(KeyCode::KeyA), keys.pressed(KeyCode::KeyD));
    let target_pitch = axis(keys.pressed(KeyCode::KeyS), keys.pressed(KeyCode::KeyW));
    let target_yaw = axis(keys.pressed(KeyCode::KeyQ), keys.pressed(KeyCode::KeyE));
    control.action[0] = approach(control.action[0], target_roll, 5.0 * dt);
    control.action[1] = approach(control.action[1], target_pitch, 5.0 * dt);
    control.action[3] = approach(control.action[3], target_yaw, 5.0 * dt);

    let throttle_delta = axis(keys.pressed(KeyCode::KeyF), keys.pressed(KeyCode::KeyR));
    control.action[2] = (control.action[2] + throttle_delta * 0.75 * dt).clamp(0.0, 1.0);
    state.action = control.action;

    let camera_dt = dt.max(1.0 / 240.0);
    if keys.pressed(KeyCode::BracketLeft) {
        camera.distance = (camera.distance - 3.0 * camera_dt).max(1.0);
    }
    if keys.pressed(KeyCode::BracketRight) {
        camera.distance = (camera.distance + 3.0 * camera_dt).min(30.0);
    }
    if keys.pressed(KeyCode::Digit3) {
        camera.height = (camera.height - 2.0 * camera_dt).max(0.2);
    }
    if keys.pressed(KeyCode::Digit4) {
        camera.height = (camera.height + 2.0 * camera_dt).min(20.0);
    }
    if keys.pressed(KeyCode::Digit1) {
        camera.spring = (camera.spring - 4.0 * camera_dt).max(0.5);
    }
    if keys.pressed(KeyCode::Digit2) {
        camera.spring = (camera.spring + 4.0 * camera_dt).min(40.0);
    }
}

fn step_lewm_sim(
    time: Res<Time>,
    mut control: ResMut<InputControl>,
    mut sim: NonSendMut<LewmDroneSim>,
    mut state: ResMut<SimRenderState>,
) {
    if control.paused || state.error.is_some() {
        return;
    }
    let frame_dt = 1.0 / state.sim_hz;
    control.accumulator += time.delta_secs();
    let mut steps = 0usize;
    while control.accumulator >= frame_dt && steps < state.max_frame_steps {
        control.accumulator -= frame_dt;
        steps += 1;
        match sim.step(control.action) {
            Ok((frame, step_ms)) => state.push_frame(frame, control.action, step_ms),
            Err(err) => {
                state.error = Some(err.to_string());
                control.paused = true;
                return;
            }
        }
    }
}

fn update_drone_mesh(state: Res<SimRenderState>, mut query: Query<(&DronePart, &mut Transform)>) {
    let base = transform_from_frame(&state.frame);
    for (_, mut transform) in &mut query {
        *transform = base;
    }
}

fn update_follow_camera(
    time: Res<Time>,
    state: Res<SimRenderState>,
    camera_cfg: Res<FollowCameraConfig>,
    mut query: Query<&mut Transform, With<FollowCamera>>,
) {
    let Ok(mut transform) = query.single_mut() else {
        return;
    };
    let drone_transform = transform_from_frame(&state.frame);
    let focus = drone_transform.translation + Vec3::Y * 0.35;
    let mut forward = body_forward_view(&state.frame);
    forward.y = 0.0;
    if forward.length_squared() < 1e-6 {
        forward = Vec3::NEG_Z;
    } else {
        forward = forward.normalize();
    }
    let desired = focus - forward * camera_cfg.distance + Vec3::Y * camera_cfg.height;
    let alpha = 1.0 - (-camera_cfg.spring * time.delta_secs()).exp();
    transform.translation = transform.translation.lerp(desired, alpha.clamp(0.0, 1.0));
    transform.look_at(focus, Vec3::Y);
}

fn draw_guides(mut gizmos: Gizmos, guide: Res<SceneGuide>) {
    let y = 0.0;
    let min = -guide.grid_extent;
    let max = guide.grid_extent;
    let grid_color = Color::srgb(0.24, 0.25, 0.26);
    let major_color = Color::srgb(0.34, 0.35, 0.36);
    let mut x = min;
    let mut idx = 0usize;
    while x <= max + 0.001 {
        let color = if idx % 5 == 0 {
            major_color
        } else {
            grid_color
        };
        gizmos.line(Vec3::new(x, y, min), Vec3::new(x, y, max), color);
        gizmos.line(Vec3::new(min, y, x), Vec3::new(max, y, x), color);
        x += guide.grid_step;
        idx += 1;
    }
    gizmos.line(
        Vec3::ZERO,
        Vec3::X * guide.axis_len,
        Color::srgb(1.0, 0.1, 0.1),
    );
    gizmos.line(
        Vec3::ZERO,
        Vec3::Y * guide.axis_len,
        Color::srgb(0.1, 0.9, 0.2),
    );
    gizmos.line(
        Vec3::ZERO,
        Vec3::Z * guide.axis_len,
        Color::srgb(0.2, 0.45, 1.0),
    );
}

fn draw_sim_overlays(
    mut gizmos: Gizmos,
    state: Res<SimRenderState>,
    control: Res<InputControl>,
    camera: Res<FollowCameraConfig>,
) {
    let frame = &state.frame;
    let pos = view_vec3(frame.pos_world);
    let transform = transform_from_frame(frame);
    gizmos.axes(transform, 0.55);
    gizmos.line(
        pos,
        Vec3::new(pos.x, 0.0, pos.z),
        Color::srgba(0.9, 0.9, 0.9, 0.35),
    );
    draw_trail(&mut gizmos, &state.trail);
    draw_action_bars(&mut gizmos, pos, &state.action);

    let lin_vel = body_vector_to_view_world(frame, frame.lin_vel_body);
    if lin_vel.length_squared() > 1e-6 {
        gizmos.arrow(pos, pos + lin_vel * 0.25, Color::srgb(0.35, 0.8, 1.0));
    }

    let status = if let Some(err) = state.error.as_ref() {
        format!("ERROR: {err}")
    } else if control.paused {
        "paused".to_string()
    } else {
        "running".to_string()
    };
    let text = format!(
        "{} step={} z={:.2} vbat={:.1} v=[{:.1} {:.1} {:.1}]\n\
         a roll={:.2} pitch={:.2} thr={:.2} yaw={:.2} step_ms={:.2} avg_ms={:.2}\n\
         cam dist={:.1} height={:.1} spring={:.1} | WASD pitch/roll Q/E yaw R/F throttle",
        status,
        state.step,
        frame.pos_world[2],
        frame.vbat,
        frame.lin_vel_body[0],
        frame.lin_vel_body[1],
        frame.lin_vel_body[2],
        state.action[0],
        state.action[1],
        state.action[2],
        state.action[3],
        state.last_step_ms,
        state.avg_step_ms,
        camera.distance,
        camera.height,
        camera.spring,
    );
    gizmos.text(
        Isometry3d::from_translation(pos + Vec3::Y * 1.25),
        &text,
        TELEMETRY_FONT_SIZE,
        Vec2::new(-0.5, 0.0),
        Color::srgb(0.95, 0.95, 0.85),
    );
}

fn draw_trail(gizmos: &mut Gizmos, trail: &[[f32; 3]]) {
    for pair in trail.windows(2) {
        gizmos.line(
            view_vec3(pair[0]),
            view_vec3(pair[1]),
            Color::srgb(0.95, 0.85, 0.15),
        );
    }
    if let Some(last) = trail.last() {
        gizmos.sphere(view_vec3(*last), 0.10, Color::srgb(0.15, 1.0, 0.35));
    }
}

fn draw_action_bars(gizmos: &mut Gizmos, current_pos: Vec3, action: &[f32; 4]) {
    let origin = current_pos + Vec3::new(-0.45, 0.25, -0.75);
    for (idx, value) in action.iter().enumerate() {
        let base = origin + Vec3::X * (idx as f32 * 0.25);
        let height = if idx == 2 {
            value.clamp(0.0, 1.0) * 0.70
        } else {
            value.clamp(-1.0, 1.0) * 0.55
        };
        let color = if *value >= 0.0 {
            Color::srgb(0.25, 1.0, 0.35)
        } else {
            Color::srgb(1.0, 0.25, 0.25)
        };
        gizmos.line(base, base + Vec3::Y * height, color);
        gizmos.sphere(base + Vec3::Y * height, 0.035, color);
    }
}

fn axis(negative: bool, positive: bool) -> f32 {
    match (negative, positive) {
        (true, false) => -1.0,
        (false, true) => 1.0,
        _ => 0.0,
    }
}

fn approach(value: f32, target: f32, max_delta: f32) -> f32 {
    let delta = (target - value).clamp(-max_delta, max_delta);
    value + delta
}

fn normalized_action_tensor(
    action: [f32; DRONE_ACTION_DIM],
    stats: &RunningStats,
    normalized: bool,
    dtype: DType,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let mut values = action;
    if normalized {
        for idx in 0..DRONE_ACTION_DIM {
            values[idx] = (values[idx] - stats.mean[idx]) / stats.std[idx].max(1e-6);
        }
    }
    Ok(Tensor::from_vec(values.to_vec(), (1, 1, DRONE_ACTION_DIM), device)?.to_dtype(dtype)?)
}

fn denormalized_delta(
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

fn history_action_prefix(history_actions: &Tensor, history_steps: usize) -> anyhow::Result<Tensor> {
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

fn transform_from_frame(frame: &DroneFrame) -> Transform {
    let m = frame.rotmat_world_from_body;
    let drone_col_x = Vec3::new(m[0], m[3], m[6]);
    let drone_col_y = Vec3::new(m[1], m[4], m[7]);
    let drone_col_z = Vec3::new(m[2], m[5], m[8]);
    let rotation = Quat::from_mat3(&Mat3::from_cols(
        drone_to_view_vec(drone_col_x),
        drone_to_view_vec(drone_col_z),
        drone_to_view_vec(drone_col_y),
    ));
    Transform {
        translation: view_vec3(frame.pos_world),
        rotation,
        scale: Vec3::ONE,
    }
}

fn body_forward_view(frame: &DroneFrame) -> Vec3 {
    let m = frame.rotmat_world_from_body;
    drone_to_view_vec(Vec3::new(m[0], m[3], m[6])).normalize_or_zero()
}

fn body_vector_to_view_world(frame: &DroneFrame, body_vector: [f32; 3]) -> Vec3 {
    let m = frame.rotmat_world_from_body;
    let x = Vec3::new(m[0], m[3], m[6]);
    let y = Vec3::new(m[1], m[4], m[7]);
    let z = Vec3::new(m[2], m[5], m[8]);
    drone_to_view_vec(x * body_vector[0] + y * body_vector[1] + z * body_vector[2])
}

fn view_vec3(value: [f32; 3]) -> Vec3 {
    drone_to_view_vec(Vec3::new(value[0], value[1], value[2]))
}

fn drone_to_view_vec(value: Vec3) -> Vec3 {
    Vec3::new(value.x, value.z, value.y)
}

fn default_dataset_dir() -> PathBuf {
    home_dir()
        .join(".stable_worldmodel")
        .join("le-wm-nv-data")
        .join("drone-racing-autonomous-100hz")
}

fn default_run_dir() -> PathBuf {
    home_dir()
        .join(".stable_worldmodel")
        .join("le-wm-nv-runs")
        .join("drone-state-lewm-all-data-20260612-235255")
}

fn default_weights() -> PathBuf {
    default_run_dir().join("final.safetensors")
}

fn default_config() -> PathBuf {
    default_run_dir().join("model-config.json")
}

fn home_dir() -> PathBuf {
    env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}
