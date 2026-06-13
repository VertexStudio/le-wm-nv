use std::{env, fs, path::PathBuf};

use anyhow::Context;
use bevy::input::mouse::MouseMotion;
use bevy::prelude::*;
use serde::Deserialize;

fn main() -> anyhow::Result<()> {
    let args = Args::parse()?;
    let replay: ReplayReport = serde_json::from_str(
        &fs::read_to_string(&args.replay)
            .with_context(|| format!("failed to read {}", args.replay.display()))?,
    )
    .with_context(|| format!("failed to parse {}", args.replay.display()))?;
    App::new()
        .insert_resource(args)
        .insert_resource(ReplayState {
            replay,
            frame: 0,
            playing: true,
            speed: 1.0,
            follow_predicted: false,
            accumulator: 0.0,
        })
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "le-wm-nv Drone Replay".to_string(),
                resolution: (1280, 800).into(),
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                playback_controls,
                free_camera_controls,
                update_drone,
                draw_paths,
            ),
        )
        .run();
    Ok(())
}

#[derive(Resource)]
struct Args {
    replay: PathBuf,
}

impl Args {
    fn parse() -> anyhow::Result<Self> {
        let mut replay = None;
        let mut iter = env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--replay" => {
                    replay = iter.next().map(PathBuf::from);
                }
                other => anyhow::bail!("unknown argument `{other}`, expected --replay <path>"),
            }
        }
        let replay = replay.unwrap_or_else(default_replay);
        Ok(Self { replay })
    }
}

fn default_replay() -> PathBuf {
    env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".stable_worldmodel")
        .join("le-wm-nv-reports")
        .join("drone-state-lewm-autonomous-100hz")
        .join("replay.json")
}

#[derive(Resource)]
struct ReplayState {
    replay: ReplayReport,
    frame: usize,
    playing: bool,
    speed: f32,
    follow_predicted: bool,
    accumulator: f32,
}

#[derive(Component)]
struct FreeCamera {
    yaw: f32,
    pitch: f32,
    speed: f32,
    fast_multiplier: f32,
    sensitivity: f32,
}

#[derive(Component)]
enum DronePart {
    Body,
    ArmX,
    ArmY,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    replay: Res<ReplayState>,
) {
    let bounds = replay_bounds(&replay.replay);
    let center = bounds.center();
    let radius = bounds.radius().max(4.0);
    let camera_pos = center + Vec3::new(-0.9 * radius, -1.4 * radius, 0.8 * radius);
    let camera_transform = Transform::from_translation(camera_pos).looking_at(center, Vec3::Z);
    let (yaw, pitch, _) = camera_transform.rotation.to_euler(EulerRot::YXZ);
    commands.spawn((
        Camera3d::default(),
        camera_transform,
        FreeCamera {
            yaw,
            pitch,
            speed: (radius * 0.5).max(4.0),
            fast_multiplier: 4.0,
            sensitivity: 0.002,
        },
    ));
    commands.spawn((
        PointLight {
            intensity: 5000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_translation(center + Vec3::new(-0.5 * radius, 0.7 * radius, radius)),
    ));

    let floor_mesh = meshes.add(Plane3d::default().mesh().size(radius * 4.0, radius * 4.0));
    let floor_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.08, 0.09, 0.09),
        perceptual_roughness: 0.9,
        ..default()
    });
    commands.spawn((
        Mesh3d(floor_mesh),
        MeshMaterial3d(floor_mat),
        Transform::from_xyz(center.x, center.y, 0.0),
    ));

    let body_mesh = meshes.add(Cuboid::new(0.45, 0.18, 0.12));
    let arm_x_mesh = meshes.add(Cuboid::new(0.9, 0.05, 0.04));
    let arm_y_mesh = meshes.add(Cuboid::new(0.05, 0.9, 0.04));
    let drone_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.95, 0.9),
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

    let gate_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.0, 0.8, 0.85),
        emissive: Color::srgb(0.0, 0.08, 0.08).into(),
        ..default()
    });
    let bar = meshes.add(Cuboid::new(1.0, 1.0, 1.0));
    let has_matching_gates = replay
        .replay
        .gates
        .flights
        .iter()
        .any(|flight| flight.episode_idx == replay.replay.episode_idx);
    for flight in &replay.replay.gates.flights {
        if has_matching_gates && flight.episode_idx != replay.replay.episode_idx {
            continue;
        }
        for gate in &flight.gates {
            spawn_gate(&mut commands, &bar, &gate_mat, gate);
        }
    }
}

fn playback_controls(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<ReplayState>,
) {
    if keys.just_pressed(KeyCode::Space) {
        state.playing = !state.playing;
    }
    if keys.just_pressed(KeyCode::Tab) {
        state.follow_predicted = !state.follow_predicted;
    }
    if keys.just_pressed(KeyCode::ArrowRight) {
        state.frame = (state.frame + 1).min(state.replay.actual.len().saturating_sub(1));
    }
    if keys.just_pressed(KeyCode::ArrowLeft) {
        state.frame = state.frame.saturating_sub(1);
    }
    if keys.just_pressed(KeyCode::Equal) {
        state.speed = (state.speed * 1.25).min(8.0);
    }
    if keys.just_pressed(KeyCode::Minus) {
        state.speed = (state.speed / 1.25).max(0.125);
    }
    if state.playing && state.replay.sample_rate_hz > 0 {
        state.accumulator += time.delta_secs() * state.speed;
        let frame_dt = 1.0 / state.replay.sample_rate_hz as f32;
        while state.accumulator >= frame_dt {
            state.accumulator -= frame_dt;
            state.frame = (state.frame + 1) % state.replay.actual.len().max(1);
        }
    }
}

fn free_camera_controls(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: MessageReader<MouseMotion>,
    mut query: Query<(&mut Transform, &mut FreeCamera)>,
) {
    let Ok((mut transform, mut camera)) = query.single_mut() else {
        return;
    };

    let mut mouse_delta = Vec2::ZERO;
    for event in mouse_motion.read() {
        if mouse_buttons.pressed(MouseButton::Right) {
            mouse_delta += event.delta;
        }
    }
    if mouse_delta != Vec2::ZERO {
        camera.yaw -= mouse_delta.x * camera.sensitivity;
        camera.pitch = (camera.pitch - mouse_delta.y * camera.sensitivity).clamp(-1.5, 1.5);
        transform.rotation = Quat::from_euler(EulerRot::YXZ, camera.yaw, camera.pitch, 0.0);
    }

    let forward = transform.rotation * -Vec3::Z;
    let right = transform.rotation * Vec3::X;
    let mut movement = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        movement += forward;
    }
    if keys.pressed(KeyCode::KeyS) {
        movement -= forward;
    }
    if keys.pressed(KeyCode::KeyD) {
        movement += right;
    }
    if keys.pressed(KeyCode::KeyA) {
        movement -= right;
    }
    if keys.pressed(KeyCode::KeyE) {
        movement += Vec3::Z;
    }
    if keys.pressed(KeyCode::KeyQ) {
        movement -= Vec3::Z;
    }
    if movement.length_squared() > 0.0 {
        let multiplier = if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
            camera.fast_multiplier
        } else {
            1.0
        };
        transform.translation +=
            movement.normalize() * camera.speed * multiplier * time.delta_secs();
    }
}

fn update_drone(state: Res<ReplayState>, mut query: Query<(&DronePart, &mut Transform)>) {
    let frames = if state.follow_predicted {
        &state.replay.predicted
    } else {
        &state.replay.actual
    };
    let Some(frame) = frames.get(state.frame) else {
        return;
    };
    let base = transform_from_frame(frame);
    for (part, mut transform) in &mut query {
        *transform = match part {
            DronePart::Body => base,
            DronePart::ArmX => base,
            DronePart::ArmY => base,
        };
    }
}

fn draw_paths(mut gizmos: Gizmos, state: Res<ReplayState>) {
    draw_line_strip(
        &mut gizmos,
        &state.replay.actual,
        Color::srgb(0.15, 0.8, 0.25),
    );
    draw_line_strip(
        &mut gizmos,
        &state.replay.predicted,
        Color::srgb(0.95, 0.85, 0.15),
    );
    draw_line_strip(
        &mut gizmos,
        &state.replay.baseline,
        Color::srgb(0.55, 0.55, 0.65),
    );
    if let Some(frame) = state.replay.actual.get(state.frame) {
        let pos = vec3(frame.pos_world);
        gizmos.sphere(pos, 0.12, Color::srgb(0.15, 1.0, 0.35));
    }
    if let Some(frame) = state.replay.predicted.get(state.frame) {
        let pos = vec3(frame.pos_world);
        gizmos.sphere(pos, 0.1, Color::srgb(1.0, 0.9, 0.1));
    }
}

fn draw_line_strip(gizmos: &mut Gizmos, frames: &[DroneFrame], color: Color) {
    for pair in frames.windows(2) {
        gizmos.line(vec3(pair[0].pos_world), vec3(pair[1].pos_world), color);
    }
}

fn spawn_gate(
    commands: &mut Commands,
    bar_mesh: &Handle<Mesh>,
    material: &Handle<StandardMaterial>,
    gate: &GateSpec,
) {
    let center = vec3(gate.center);
    let right = vec3(gate.right).normalize_or_zero();
    let up = vec3(gate.up).normalize_or_zero();
    let normal = vec3(gate.normal).normalize_or_zero();
    let rotation = Quat::from_mat3(&Mat3::from_cols(right, up, normal));
    let width = gate.half_width * 2.0;
    let height = gate.half_height * 2.0;
    let thickness = 0.08;
    for offset in [up * gate.half_height, -up * gate.half_height] {
        commands.spawn((
            Mesh3d(bar_mesh.clone()),
            MeshMaterial3d(material.clone()),
            Transform {
                translation: center + offset,
                rotation,
                scale: Vec3::new(width, thickness, thickness),
            },
        ));
    }
    for offset in [right * gate.half_width, -right * gate.half_width] {
        commands.spawn((
            Mesh3d(bar_mesh.clone()),
            MeshMaterial3d(material.clone()),
            Transform {
                translation: center + offset,
                rotation,
                scale: Vec3::new(thickness, height, thickness),
            },
        ));
    }
}

fn transform_from_frame(frame: &DroneFrame) -> Transform {
    let m = frame.rotmat_world_from_body;
    let rotation = Quat::from_mat3(&Mat3::from_cols(
        Vec3::new(m[0], m[3], m[6]),
        Vec3::new(m[1], m[4], m[7]),
        Vec3::new(m[2], m[5], m[8]),
    ));
    Transform {
        translation: vec3(frame.pos_world),
        rotation,
        scale: Vec3::ONE,
    }
}

fn vec3(value: [f32; 3]) -> Vec3 {
    Vec3::new(value[0], value[1], value[2])
}

#[derive(Clone, Copy)]
struct SceneBounds {
    min: Vec3,
    max: Vec3,
}

impl SceneBounds {
    fn empty() -> Self {
        Self {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        }
    }

    fn include(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    fn center(&self) -> Vec3 {
        if self.min.is_finite() && self.max.is_finite() {
            (self.min + self.max) * 0.5
        } else {
            Vec3::ZERO
        }
    }

    fn radius(&self) -> f32 {
        if self.min.is_finite() && self.max.is_finite() {
            (self.max - self.min).length() * 0.5
        } else {
            10.0
        }
    }
}

fn replay_bounds(replay: &ReplayReport) -> SceneBounds {
    let mut bounds = SceneBounds::empty();
    for frame in &replay.actual {
        bounds.include(vec3(frame.pos_world));
    }
    for flight in &replay.gates.flights {
        if flight.episode_idx != replay.episode_idx {
            continue;
        }
        for gate in &flight.gates {
            let center = vec3(gate.center);
            let right = vec3(gate.right).normalize_or_zero() * gate.half_width;
            let up = vec3(gate.up).normalize_or_zero() * gate.half_height;
            for sx in [-1.0, 1.0] {
                for sy in [-1.0, 1.0] {
                    bounds.include(center + right * sx + up * sy);
                }
            }
        }
    }
    bounds
}

#[derive(Debug, Deserialize)]
struct ReplayReport {
    episode_idx: i64,
    sample_rate_hz: usize,
    actual: Vec<DroneFrame>,
    predicted: Vec<DroneFrame>,
    baseline: Vec<DroneFrame>,
    gates: GateSequenceFile,
}

#[derive(Debug, Clone, Deserialize)]
struct DroneFrame {
    pos_world: [f32; 3],
    rotmat_world_from_body: [f32; 9],
}

#[derive(Debug, Deserialize)]
struct GateSequenceFile {
    flights: Vec<FlightGates>,
}

#[derive(Debug, Deserialize)]
struct FlightGates {
    episode_idx: i64,
    gates: Vec<GateSpec>,
}

#[derive(Debug, Deserialize)]
struct GateSpec {
    center: [f32; 3],
    normal: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    half_width: f32,
    half_height: f32,
}
