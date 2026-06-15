use std::env;

use anyhow::ensure;
use bevy::{input::mouse::MouseWheel, prelude::*};

const ACTION_DIM: usize = 4;
const OBS_DIM: usize = 16;
const TELEMETRY_FONT_SIZE: f32 = 0.11;

fn main() -> anyhow::Result<()> {
    let args = Args::parse()?;
    App::new()
        .insert_resource(args.camera)
        .insert_resource(SimControl::new(args.dynamics.hover_throttle))
        .insert_resource(SimState::new(args.dynamics, args.max_trail))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "le-wm-nv Drone Dynamics Sim".to_string(),
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
                step_simulation,
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
    dynamics: DynamicsConfig,
    camera: FollowCameraConfig,
    max_trail: usize,
}

impl Args {
    fn parse() -> anyhow::Result<Self> {
        let mut args = Self {
            dynamics: DynamicsConfig::default(),
            camera: FollowCameraConfig {
                distance: 7.0,
                height: 2.2,
                spring: 8.0,
            },
            max_trail: 2400,
        };
        let mut iter = env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--sim-hz" => args.dynamics.sim_hz = next_parse(&mut iter, &arg)?,
                "--time-scale" => args.dynamics.time_scale = next_parse(&mut iter, &arg)?,
                "--max-frame-steps" => args.dynamics.max_frame_steps = next_parse(&mut iter, &arg)?,
                "--mass" => args.dynamics.mass = next_parse(&mut iter, &arg)?,
                "--gravity" => args.dynamics.gravity = next_parse(&mut iter, &arg)?,
                "--hover-throttle" => args.dynamics.hover_throttle = next_parse(&mut iter, &arg)?,
                "--max-thrust-weight" => {
                    args.dynamics.max_thrust_weight = next_parse(&mut iter, &arg)?
                }
                "--max-roll-rate" => args.dynamics.max_roll_rate = next_parse(&mut iter, &arg)?,
                "--max-pitch-rate" => args.dynamics.max_pitch_rate = next_parse(&mut iter, &arg)?,
                "--max-yaw-rate" => args.dynamics.max_yaw_rate = next_parse(&mut iter, &arg)?,
                "--rate-kp" => args.dynamics.rate_kp = next_parse(&mut iter, &arg)?,
                "--rate-damping" => args.dynamics.rate_damping = next_parse(&mut iter, &arg)?,
                "--linear-drag" => args.dynamics.linear_drag = next_parse(&mut iter, &arg)?,
                "--quadratic-drag" => args.dynamics.quadratic_drag = next_parse(&mut iter, &arg)?,
                "--max-trail" => args.max_trail = next_parse(&mut iter, &arg)?,
                "--camera-distance" => args.camera.distance = next_parse(&mut iter, &arg)?,
                "--camera-height" => args.camera.height = next_parse(&mut iter, &arg)?,
                "--camera-spring" => args.camera.spring = next_parse(&mut iter, &arg)?,
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                other => anyhow::bail!("unknown argument `{other}`; use --help"),
            }
        }
        args.validate()?;
        Ok(args)
    }

    fn validate(&self) -> anyhow::Result<()> {
        ensure!(self.dynamics.sim_hz > 0.0, "--sim-hz must be positive");
        ensure!(
            self.dynamics.max_frame_steps > 0,
            "--max-frame-steps must be positive"
        );
        ensure!(
            self.dynamics.time_scale > 0.0,
            "--time-scale must be positive"
        );
        ensure!(self.dynamics.mass > 0.0, "--mass must be positive");
        ensure!(self.dynamics.gravity > 0.0, "--gravity must be positive");
        ensure!(
            self.dynamics.hover_throttle > 0.01 && self.dynamics.hover_throttle <= 1.0,
            "--hover-throttle must be in (0.01, 1.0]"
        );
        ensure!(
            self.dynamics.max_thrust_weight >= 1.0,
            "--max-thrust-weight must be at least 1.0"
        );
        ensure!(self.max_trail > 1, "--max-trail must be greater than 1");
        ensure!(
            self.camera.distance > 0.25,
            "--camera-distance must be greater than 0.25"
        );
        ensure!(
            self.camera.height >= 0.0,
            "--camera-height must be non-negative"
        );
        ensure!(self.camera.spring > 0.0, "--camera-spring must be positive");
        Ok(())
    }
}

fn next_parse<T>(iter: &mut impl Iterator<Item = String>, flag: &str) -> anyhow::Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value = iter
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing value after {flag}"))?;
    value
        .parse::<T>()
        .map_err(|err| anyhow::anyhow!("invalid value `{value}` for {flag}: {err}"))
}

fn print_help() {
    println!(
        "Usage: lewm-drone-sim [options]\n\
         \n\
         Dynamics options:\n\
           --sim-hz <hz>                 default 1000\n\
           --time-scale <scale>          default 1\n\
           --max-frame-steps <n>         default 20\n\
           --mass <kg>                   default 1.3\n\
           --gravity <m/s^2>             default 9.81\n\
           --hover-throttle <0..1>       default 0.5\n\
           --max-thrust-weight <ratio>   default 2.2\n\
           --max-roll-rate <rad/s>       default 8\n\
           --max-pitch-rate <rad/s>      default 8\n\
           --max-yaw-rate <rad/s>        default 5\n\
           --rate-kp <rad/s^2>           default 22\n\
           --rate-damping <gain>         default 2.5\n\
           --linear-drag <gain>          default 0.18\n\
           --quadratic-drag <gain>       default 0.03\n\
         \n\
         Camera options:\n\
           --camera-distance <meters>    default 7\n\
           --camera-height <meters>      default 2.2\n\
           --camera-spring <rate>        default 8\n\
           --max-trail <steps>           default 2400\n\
         \n\
         Controls:\n\
           W/S pitch forward/back, A/D roll left/right, Q/E yaw left/right, R/F throttle\n\
           Z zero roll/pitch/yaw, X hover throttle, P pause, Backspace reset\n\
           mouse wheel or [/] camera distance, 3/4 camera height, 1/2 camera spring"
    );
}

#[derive(Resource, Debug, Clone)]
struct DynamicsConfig {
    sim_hz: f32,
    time_scale: f32,
    max_frame_steps: usize,
    mass: f32,
    gravity: f32,
    hover_throttle: f32,
    max_thrust_weight: f32,
    max_roll_rate: f32,
    max_pitch_rate: f32,
    max_yaw_rate: f32,
    rate_kp: f32,
    rate_damping: f32,
    linear_drag: f32,
    quadratic_drag: f32,
}

impl Default for DynamicsConfig {
    fn default() -> Self {
        Self {
            sim_hz: 1000.0,
            time_scale: 1.0,
            max_frame_steps: 20,
            mass: 1.3,
            gravity: 9.81,
            hover_throttle: 0.5,
            max_thrust_weight: 2.2,
            max_roll_rate: 8.0,
            max_pitch_rate: 8.0,
            max_yaw_rate: 5.0,
            rate_kp: 22.0,
            rate_damping: 2.5,
            linear_drag: 0.18,
            quadratic_drag: 0.03,
        }
    }
}

#[derive(Resource, Debug, Clone)]
struct FollowCameraConfig {
    distance: f32,
    height: f32,
    spring: f32,
}

#[derive(Resource)]
struct SimControl {
    action: [f32; ACTION_DIM],
    hover_action: [f32; ACTION_DIM],
    paused: bool,
    accumulator: f32,
}

impl SimControl {
    fn new(hover_throttle: f32) -> Self {
        let hover_action = [0.0, 0.0, hover_throttle, 0.0];
        Self {
            action: hover_action,
            hover_action,
            paused: false,
            accumulator: 0.0,
        }
    }
}

#[derive(Resource)]
struct SimState {
    dynamics: DynamicsConfig,
    pose: DronePose,
    previous_action: [f32; ACTION_DIM],
    time: f32,
    step: usize,
    last_step_ms: f32,
    avg_step_ms: f32,
    trail: Vec<Vec3>,
    max_trail: usize,
    target: TargetPose,
}

impl SimState {
    fn new(dynamics: DynamicsConfig, max_trail: usize) -> Self {
        let pose = DronePose::initial();
        Self {
            dynamics,
            pose,
            previous_action: [0.0, 0.0, 0.0, 0.0],
            time: 0.0,
            step: 0,
            last_step_ms: 0.0,
            avg_step_ms: 0.0,
            trail: vec![pose.pos_world],
            max_trail,
            target: TargetPose {
                pos_world: Vec3::new(4.0, 0.0, 1.6),
                rot_world_from_body: Quat::IDENTITY,
            },
        }
    }

    fn reset(&mut self, hover_action: [f32; ACTION_DIM]) {
        self.pose = DronePose::initial();
        self.previous_action = hover_action;
        self.time = 0.0;
        self.step = 0;
        self.last_step_ms = 0.0;
        self.avg_step_ms = 0.0;
        self.trail.clear();
        self.trail.push(self.pose.pos_world);
    }

    fn step(&mut self, action: [f32; ACTION_DIM], dt: f32) {
        let started = std::time::Instant::now();
        self.pose.integrate(action, &self.dynamics, dt);
        self.previous_action = action;
        self.time += dt;
        self.step += 1;
        self.trail.push(self.pose.pos_world);
        if self.trail.len() > self.max_trail {
            let drop_count = self.trail.len() - self.max_trail;
            self.trail.drain(0..drop_count);
        }
        self.last_step_ms = started.elapsed().as_secs_f32() * 1000.0;
        self.avg_step_ms = if self.avg_step_ms == 0.0 {
            self.last_step_ms
        } else {
            self.avg_step_ms * 0.95 + self.last_step_ms * 0.05
        };
    }

    fn obs16(&self) -> [f32; OBS_DIM] {
        let mut obs = [0.0; OBS_DIM];
        obs[0..3].copy_from_slice(&vec3_array(self.pose.pos_world));
        obs[3..12].copy_from_slice(&rotmat_world_from_body_array(self.pose.rot_world_from_body));
        obs[12..16].copy_from_slice(&self.previous_action);
        obs
    }
}

#[derive(Clone, Copy)]
struct DronePose {
    pos_world: Vec3,
    vel_world: Vec3,
    rot_world_from_body: Quat,
    ang_vel_body: Vec3,
}

impl DronePose {
    fn initial() -> Self {
        Self {
            pos_world: Vec3::new(0.0, 0.0, 1.0),
            vel_world: Vec3::ZERO,
            rot_world_from_body: Quat::IDENTITY,
            ang_vel_body: Vec3::ZERO,
        }
    }

    fn integrate(&mut self, action: [f32; ACTION_DIM], cfg: &DynamicsConfig, dt: f32) {
        let roll = action[0].clamp(-1.0, 1.0);
        let pitch = action[1].clamp(-1.0, 1.0);
        let throttle = action[2].clamp(0.0, 1.0);
        let yaw = action[3].clamp(-1.0, 1.0);

        let desired_rates = Vec3::new(
            -roll * cfg.max_roll_rate,
            pitch * cfg.max_pitch_rate,
            yaw * cfg.max_yaw_rate,
        );
        let rate_error = desired_rates - self.ang_vel_body;
        let angular_accel = rate_error * cfg.rate_kp - self.ang_vel_body * cfg.rate_damping;
        self.ang_vel_body += angular_accel * dt;
        self.ang_vel_body = self.ang_vel_body.clamp_length_max(
            cfg.max_roll_rate
                .max(cfg.max_pitch_rate)
                .max(cfg.max_yaw_rate)
                * 1.5,
        );

        let delta_rot = Quat::from_scaled_axis(self.ang_vel_body * dt);
        self.rot_world_from_body = (self.rot_world_from_body * delta_rot).normalize();

        let thrust_weight = (throttle / cfg.hover_throttle).clamp(0.0, cfg.max_thrust_weight);
        let thrust = cfg.mass * cfg.gravity * thrust_weight;
        let body_up_world = self.rot_world_from_body * Vec3::Z;
        let speed = self.vel_world.length();
        let drag = self.vel_world * cfg.linear_drag + self.vel_world * speed * cfg.quadratic_drag;
        let accel_world =
            body_up_world * (thrust / cfg.mass) + Vec3::new(0.0, 0.0, -cfg.gravity) - drag;

        self.vel_world += accel_world * dt;
        self.pos_world += self.vel_world * dt;

        if self.pos_world.z < 0.04 {
            self.pos_world.z = 0.04;
            if self.vel_world.z < 0.0 {
                self.vel_world.z = 0.0;
            }
            self.vel_world.x *= 0.985;
            self.vel_world.y *= 0.985;
        }
    }
}

#[derive(Clone, Copy)]
struct TargetPose {
    pos_world: Vec3,
    rot_world_from_body: Quat,
}

#[derive(Component)]
enum DronePart {
    Body,
    ArmX,
    ArmY,
    Nose,
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
    state: Res<SimState>,
    camera: Res<FollowCameraConfig>,
) {
    commands.insert_resource(SceneGuide {
        grid_extent: 40.0,
        grid_step: 1.0,
        axis_len: 8.0,
    });

    let drone_transform = transform_from_pose(&state.pose);
    let focus = drone_transform.translation + Vec3::Y * 0.35;
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
    let arm_x_mesh = meshes.add(Cuboid::new(0.95, 0.04, 0.05));
    let arm_y_mesh = meshes.add(Cuboid::new(0.05, 0.04, 0.95));
    let nose_mesh = meshes.add(Cuboid::new(0.18, 0.07, 0.07));
    let drone_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.92, 0.92, 0.86),
        perceptual_roughness: 0.75,
        ..default()
    });
    let nose_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.8, 0.1),
        emissive: LinearRgba::rgb(0.5, 0.35, 0.0),
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
    commands.spawn((Mesh3d(nose_mesh), MeshMaterial3d(nose_mat), DronePart::Nose));
}

fn update_controls(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut scroll: MessageReader<MouseWheel>,
    mut control: ResMut<SimControl>,
    mut camera: ResMut<FollowCameraConfig>,
    mut state: ResMut<SimState>,
) {
    if keys.just_pressed(KeyCode::KeyP) {
        control.paused = !control.paused;
    }
    if keys.just_pressed(KeyCode::Backspace) {
        control.action = control.hover_action;
        control.accumulator = 0.0;
        state.reset(control.hover_action);
    }
    if keys.just_pressed(KeyCode::KeyZ) {
        control.action[0] = 0.0;
        control.action[1] = 0.0;
        control.action[3] = 0.0;
    }
    if keys.just_pressed(KeyCode::KeyX) {
        control.action[2] = control.hover_action[2];
    }

    let dt = time.delta_secs().max(1.0 / 240.0);
    let target_roll = axis(keys.pressed(KeyCode::KeyA), keys.pressed(KeyCode::KeyD));
    let target_pitch = axis(keys.pressed(KeyCode::KeyS), keys.pressed(KeyCode::KeyW));
    let target_yaw = axis(keys.pressed(KeyCode::KeyQ), keys.pressed(KeyCode::KeyE));
    control.action[0] = approach(control.action[0], target_roll, 5.0 * dt);
    control.action[1] = approach(control.action[1], target_pitch, 5.0 * dt);
    control.action[3] = approach(control.action[3], target_yaw, 5.0 * dt);

    let throttle_delta = axis(keys.pressed(KeyCode::KeyF), keys.pressed(KeyCode::KeyR));
    control.action[2] = (control.action[2] + throttle_delta * 0.75 * dt).clamp(0.0, 1.0);

    for event in scroll.read() {
        camera.distance = (camera.distance - event.y * 0.6).clamp(1.0, 40.0);
    }
    if keys.pressed(KeyCode::BracketLeft) {
        camera.distance = (camera.distance - 4.0 * dt).max(1.0);
    }
    if keys.pressed(KeyCode::BracketRight) {
        camera.distance = (camera.distance + 4.0 * dt).min(40.0);
    }
    if keys.pressed(KeyCode::Digit3) {
        camera.height = (camera.height - 3.0 * dt).max(0.2);
    }
    if keys.pressed(KeyCode::Digit4) {
        camera.height = (camera.height + 3.0 * dt).min(30.0);
    }
    if keys.pressed(KeyCode::Digit1) {
        camera.spring = (camera.spring - 5.0 * dt).max(0.5);
    }
    if keys.pressed(KeyCode::Digit2) {
        camera.spring = (camera.spring + 5.0 * dt).min(50.0);
    }
}

fn step_simulation(time: Res<Time>, mut control: ResMut<SimControl>, mut state: ResMut<SimState>) {
    if control.paused {
        return;
    }
    let dt = 1.0 / state.dynamics.sim_hz;
    control.accumulator += time.delta_secs() * state.dynamics.time_scale;
    let mut steps = 0usize;
    while control.accumulator >= dt && steps < state.dynamics.max_frame_steps {
        control.accumulator -= dt;
        steps += 1;
        state.step(control.action, dt);
    }
}

fn update_drone_mesh(state: Res<SimState>, mut query: Query<(&DronePart, &mut Transform)>) {
    let base = transform_from_pose(&state.pose);
    for (part, mut transform) in &mut query {
        *transform = match part {
            DronePart::Nose => base * Transform::from_translation(Vec3::X * 0.28),
            _ => base,
        };
    }
}

fn update_follow_camera(
    time: Res<Time>,
    state: Res<SimState>,
    camera_cfg: Res<FollowCameraConfig>,
    mut query: Query<&mut Transform, With<FollowCamera>>,
) {
    let Ok(mut transform) = query.single_mut() else {
        return;
    };
    let drone_transform = transform_from_pose(&state.pose);
    let focus = drone_transform.translation + Vec3::Y * 0.35;
    let mut forward = drone_to_view_vec(state.pose.rot_world_from_body * Vec3::X);
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
    state: Res<SimState>,
    control: Res<SimControl>,
    camera: Res<FollowCameraConfig>,
) {
    let pose = state.pose;
    let pos = view_vec3(pose.pos_world);
    let transform = transform_from_pose(&pose);
    gizmos.axes(transform, 0.65);
    gizmos.line(
        pos,
        Vec3::new(pos.x, 0.0, pos.z),
        Color::srgba(0.9, 0.9, 0.9, 0.35),
    );
    draw_trail(&mut gizmos, &state.trail);
    draw_action_bars(&mut gizmos, pos, &control.action);
    draw_target_pose(&mut gizmos, state.target);

    let vel_view = drone_to_view_vec(pose.vel_world);
    if vel_view.length_squared() > 1e-6 {
        gizmos.arrow(pos, pos + vel_view * 0.25, Color::srgb(0.35, 0.8, 1.0));
    }

    let obs = state.obs16();
    let target_dist = (state.target.pos_world - pose.pos_world).length();
    let status = if control.paused { "paused" } else { "running" };
    let text = format!(
        "{} t={:.2}s step={} pos=[{:.2} {:.2} {:.2}] dist={:.2}\n\
         a roll={:.2} pitch={:.2} thr={:.2} yaw={:.2} vel=[{:.2} {:.2} {:.2}] rates=[{:.2} {:.2} {:.2}]\n\
         obs16 pos=[{:.2} {:.2} {:.2}] cam dist={:.1} height={:.1} spring={:.1} step_ms={:.4}",
        status,
        state.time,
        state.step,
        pose.pos_world.x,
        pose.pos_world.y,
        pose.pos_world.z,
        target_dist,
        control.action[0],
        control.action[1],
        control.action[2],
        control.action[3],
        pose.vel_world.x,
        pose.vel_world.y,
        pose.vel_world.z,
        pose.ang_vel_body.x,
        pose.ang_vel_body.y,
        pose.ang_vel_body.z,
        obs[0],
        obs[1],
        obs[2],
        camera.distance,
        camera.height,
        camera.spring,
        state.avg_step_ms,
    );
    gizmos.text(
        Isometry3d::from_translation(pos + Vec3::Y * 1.35),
        &text,
        TELEMETRY_FONT_SIZE,
        Vec2::new(-0.5, 0.0),
        Color::srgb(0.95, 0.95, 0.85),
    );
}

fn draw_target_pose(gizmos: &mut Gizmos, target: TargetPose) {
    let pos = view_vec3(target.pos_world);
    let transform = transform_from_target(target);
    gizmos.sphere(pos, 0.18, Color::srgb(1.0, 0.25, 0.95));
    gizmos.axes(transform, 0.75);
    draw_cross(gizmos, pos, 0.45, Color::srgb(1.0, 0.25, 0.95));
    gizmos.text(
        Isometry3d::from_translation(pos + Vec3::Y * 0.55),
        "target pose",
        TELEMETRY_FONT_SIZE,
        Vec2::new(-0.5, 0.0),
        Color::srgb(1.0, 0.25, 0.95),
    );
}

fn draw_trail(gizmos: &mut Gizmos, trail: &[Vec3]) {
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

fn draw_action_bars(gizmos: &mut Gizmos, current_pos: Vec3, action: &[f32; ACTION_DIM]) {
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

fn draw_cross(gizmos: &mut Gizmos, center: Vec3, size: f32, color: Color) {
    gizmos.line(center - Vec3::X * size, center + Vec3::X * size, color);
    gizmos.line(center - Vec3::Y * size, center + Vec3::Y * size, color);
    gizmos.line(center - Vec3::Z * size, center + Vec3::Z * size, color);
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

fn transform_from_pose(pose: &DronePose) -> Transform {
    Transform {
        translation: view_vec3(pose.pos_world),
        rotation: view_rotation_from_drone_rotation(pose.rot_world_from_body),
        scale: Vec3::ONE,
    }
}

fn transform_from_target(target: TargetPose) -> Transform {
    Transform {
        translation: view_vec3(target.pos_world),
        rotation: view_rotation_from_drone_rotation(target.rot_world_from_body),
        scale: Vec3::ONE,
    }
}

fn view_rotation_from_drone_rotation(rot_world_from_body: Quat) -> Quat {
    let body_x = rot_world_from_body * Vec3::X;
    let body_y = rot_world_from_body * Vec3::Y;
    let body_z = rot_world_from_body * Vec3::Z;
    Quat::from_mat3(&Mat3::from_cols(
        drone_to_view_vec(body_x),
        drone_to_view_vec(body_z),
        drone_to_view_vec(body_y),
    ))
}

fn rotmat_world_from_body_array(rot: Quat) -> [f32; 9] {
    let x = rot * Vec3::X;
    let y = rot * Vec3::Y;
    let z = rot * Vec3::Z;
    [x.x, x.y, x.z, y.x, y.y, y.z, z.x, z.y, z.z]
}

fn vec3_array(value: Vec3) -> [f32; 3] {
    [value.x, value.y, value.z]
}

fn view_vec3(value: Vec3) -> Vec3 {
    drone_to_view_vec(value)
}

fn drone_to_view_vec(value: Vec3) -> Vec3 {
    Vec3::new(value.x, value.z, value.y)
}
