use std::{env, fs, path::PathBuf};

use anyhow::Context;
use bevy::prelude::*;
use bevy_camera_controller::free_camera::{FreeCamera, FreeCameraPlugin};
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
        .add_plugins(FreeCameraPlugin)
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                playback_controls,
                draw_scene_guides,
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

#[derive(Resource)]
struct SceneGuide {
    grid_center: Vec2,
    grid_extent: f32,
    grid_step: f32,
    axis_len: f32,
}

impl SceneGuide {
    fn new(center: Vec3, radius: f32) -> Self {
        let grid_step = nice_grid_step(radius / 10.0);
        let grid_extent = (radius * 1.4).max(grid_step * 8.0);
        Self {
            grid_center: Vec2::new(center.x, center.z),
            grid_extent,
            grid_step,
            axis_len: (radius * 1.2).max(grid_step * 6.0),
        }
    }
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
    commands.insert_resource(SceneGuide::new(center, radius));
    let camera_pos = center + Vec3::new(-0.9 * radius, 0.8 * radius, -1.4 * radius);
    let camera_transform = Transform::from_translation(camera_pos).looking_at(center, Vec3::Y);
    commands.spawn((
        Camera3d::default(),
        camera_transform,
        FreeCamera {
            sensitivity: 0.2,
            friction: 25.0,
            walk_speed: (radius * 0.45).max(3.0),
            run_speed: (radius * 1.8).max(12.0),
            mouse_key_cursor_grab: MouseButton::Right,
            ..default()
        },
    ));
    commands.spawn((
        PointLight {
            intensity: 5000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_translation(center + Vec3::new(-0.5 * radius, radius, 0.7 * radius)),
    ));

    let floor_mesh = meshes.add(Plane3d::default().mesh().size(radius * 4.0, radius * 4.0));
    let floor_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.11, 0.115, 0.12),
        perceptual_roughness: 0.9,
        ..default()
    });
    commands.spawn((
        Mesh3d(floor_mesh),
        MeshMaterial3d(floor_mat),
        Transform::from_xyz(center.x, 0.0, center.z),
    ));

    let body_mesh = meshes.add(Cuboid::new(0.45, 0.12, 0.18));
    let arm_x_mesh = meshes.add(Cuboid::new(0.9, 0.04, 0.05));
    let arm_y_mesh = meshes.add(Cuboid::new(0.05, 0.04, 0.9));
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

fn draw_scene_guides(mut gizmos: Gizmos, guide: Res<SceneGuide>) {
    let y = 0.0;
    let min_x = snap_down(guide.grid_center.x - guide.grid_extent, guide.grid_step);
    let max_x = snap_up(guide.grid_center.x + guide.grid_extent, guide.grid_step);
    let min_z = snap_down(guide.grid_center.y - guide.grid_extent, guide.grid_step);
    let max_z = snap_up(guide.grid_center.y + guide.grid_extent, guide.grid_step);
    let grid_color = Color::srgb(0.24, 0.25, 0.26);
    let major_color = Color::srgb(0.34, 0.35, 0.36);

    let mut x = min_x;
    let mut i = 0usize;
    while x <= max_x + guide.grid_step * 0.5 {
        let color = if i % 5 == 0 { major_color } else { grid_color };
        gizmos.line(Vec3::new(x, y, min_z), Vec3::new(x, y, max_z), color);
        x += guide.grid_step;
        i += 1;
    }

    let mut z = min_z;
    i = 0;
    while z <= max_z + guide.grid_step * 0.5 {
        let color = if i % 5 == 0 { major_color } else { grid_color };
        gizmos.line(Vec3::new(min_x, y, z), Vec3::new(max_x, y, z), color);
        z += guide.grid_step;
        i += 1;
    }

    let axis_len = guide.axis_len;
    gizmos.line(
        Vec3::ZERO,
        Vec3::new(axis_len, 0.0, 0.0),
        Color::srgb(1.0, 0.1, 0.1),
    );
    gizmos.line(
        Vec3::ZERO,
        Vec3::new(0.0, axis_len, 0.0),
        Color::srgb(0.1, 0.9, 0.2),
    );
    gizmos.line(
        Vec3::ZERO,
        Vec3::new(0.0, 0.0, axis_len),
        Color::srgb(0.2, 0.45, 1.0),
    );
}

fn nice_grid_step(value: f32) -> f32 {
    let value = value.max(0.05);
    let exponent = value.log10().floor();
    let base = 10.0f32.powf(exponent);
    let fraction = value / base;
    let multiplier = if fraction <= 1.0 {
        1.0
    } else if fraction <= 2.0 {
        2.0
    } else if fraction <= 5.0 {
        5.0
    } else {
        10.0
    };
    base * multiplier
}

fn snap_down(value: f32, step: f32) -> f32 {
    (value / step).floor() * step
}

fn snap_up(value: f32, step: f32) -> f32 {
    (value / step).ceil() * step
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
    if state.replay.mode.as_deref() == Some("gate_loop") {
        draw_line_strip(
            &mut gizmos,
            &state.replay.predicted,
            Color::srgb(0.95, 0.85, 0.15),
        );
    } else {
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
    }
    if let Some(frame) = state.replay.actual.get(state.frame) {
        let pos = view_vec3(frame.pos_world);
        gizmos.sphere(pos, 0.12, Color::srgb(0.15, 1.0, 0.35));
    }
    if let Some(frame) = state.replay.predicted.get(state.frame) {
        let pos = view_vec3(frame.pos_world);
        gizmos.sphere(pos, 0.1, Color::srgb(1.0, 0.9, 0.1));
    }
}

fn draw_line_strip(gizmos: &mut Gizmos, frames: &[DroneFrame], color: Color) {
    for pair in frames.windows(2) {
        gizmos.line(
            view_vec3(pair[0].pos_world),
            view_vec3(pair[1].pos_world),
            color,
        );
    }
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

fn view_vec3(value: [f32; 3]) -> Vec3 {
    drone_to_view_vec(Vec3::new(value[0], value[1], value[2]))
}

fn drone_to_view_vec(value: Vec3) -> Vec3 {
    Vec3::new(value.x, value.z, value.y)
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
        bounds.include(view_vec3(frame.pos_world));
    }
    for frame in &replay.predicted {
        bounds.include(view_vec3(frame.pos_world));
    }
    for frame in &replay.baseline {
        bounds.include(view_vec3(frame.pos_world));
    }
    bounds
}

#[derive(Debug, Deserialize)]
struct ReplayReport {
    mode: Option<String>,
    sample_rate_hz: usize,
    actual: Vec<DroneFrame>,
    predicted: Vec<DroneFrame>,
    baseline: Vec<DroneFrame>,
}

#[derive(Debug, Clone, Deserialize)]
struct DroneFrame {
    pos_world: [f32; 3],
    rotmat_world_from_body: [f32; 9],
}
