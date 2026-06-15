use std::{env, fs, path::PathBuf};

use anyhow::Context;
use bevy::prelude::*;
use bevy_camera_controller::free_camera::{FreeCamera, FreeCameraPlugin};
use serde::Deserialize;

const TARGET_LABEL_FONT_SIZE: f32 = 0.18;
const TELEMETRY_FONT_SIZE: f32 = 0.11;

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
                draw_analysis_overlays,
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
                    replay =
                        Some(PathBuf::from(iter.next().ok_or_else(|| {
                            anyhow::anyhow!("missing value after --replay")
                        })?));
                }
                other => anyhow::bail!("unknown argument `{other}`, expected --replay <path>"),
            }
        }
        let replay = replay
            .ok_or_else(|| anyhow::anyhow!("missing required --replay <path>; replay files must be generated explicitly from the current code"))?;
        Ok(Self { replay })
    }
}

#[derive(Resource)]
struct ReplayState {
    replay: ReplayReport,
    frame: usize,
    playing: bool,
    speed: f32,
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
    if keys.just_pressed(KeyCode::ArrowRight) {
        state.frame = (state.frame + 1).min(state.replay.frames.len().saturating_sub(1));
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
            state.frame = (state.frame + 1) % state.replay.frames.len().max(1);
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
    let Some(frame) = state.replay.frames.get(state.frame) else {
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
        &state.replay.frames,
        Color::srgb(0.95, 0.85, 0.15),
    );
    if let Some(frame) = state.replay.frames.get(state.frame) {
        let pos = view_vec3(frame.pos_world);
        gizmos.sphere(pos, 0.12, Color::srgb(0.15, 1.0, 0.35));
    }
}

fn draw_analysis_overlays(mut gizmos: Gizmos, state: Res<ReplayState>) {
    let replay = &state.replay;
    let frames = &state.replay.frames;
    let Some(frame) = frames.get(state.frame) else {
        return;
    };
    let current_pos = view_vec3(frame.pos_world);
    let active_replan = active_replan(replay, state.frame);
    let active_gate = active_replan
        .and_then(|replan| replay.gate_loop.get(replan.gate_index))
        .or_else(|| replay.gate_loop.first());

    draw_reference_path(&mut gizmos, replay, frames);
    draw_target_points(&mut gizmos, replay, active_gate);
    draw_replan_markers(&mut gizmos, replay, frames);
    draw_current_state_overlay(
        &mut gizmos,
        replay,
        state.frame,
        frame,
        current_pos,
        active_replan,
        active_gate,
    );
}

fn draw_reference_path(gizmos: &mut Gizmos, replay: &ReplayReport, frames: &[DroneFrame]) {
    if replay.gate_loop.is_empty() {
        return;
    }
    let mut points = Vec::with_capacity(replay.gate_loop.len() + 1);
    if let Some(start) = frames.first() {
        points.push(view_vec3(start.pos_world));
    }
    points.extend(replay.gate_loop.iter().map(|gate| view_vec3(gate.center)));
    if points.len() > 1 {
        gizmos.linestrip(points.iter().copied(), Color::srgb(0.1, 0.85, 1.0));
    }
    if replay.gate_loop.len() > 2 {
        gizmos.lineloop(
            replay.gate_loop.iter().map(|gate| view_vec3(gate.center)),
            Color::srgba(0.0, 0.55, 0.95, 0.55),
        );
    }
}

fn draw_target_points(gizmos: &mut Gizmos, replay: &ReplayReport, active_gate: Option<&GateSpec>) {
    for (idx, gate) in replay.gate_loop.iter().enumerate() {
        let center = view_vec3(gate.center);
        let is_active = active_gate.is_some_and(|active| std::ptr::eq(active, gate));
        let color = if is_active {
            Color::srgb(1.0, 0.25, 0.95)
        } else {
            Color::srgb(0.1, 0.85, 1.0)
        };
        let size = if is_active { 0.34 } else { 0.2 };
        gizmos.sphere(center, size, color);
        draw_cross(gizmos, center, size * 1.5, color);

        let normal = drone_to_view_vec(Vec3::from_array(gate.normal)).normalize_or_zero();
        let right = drone_to_view_vec(Vec3::from_array(gate.right)).normalize_or_zero();
        let up = drone_to_view_vec(Vec3::from_array(gate.up)).normalize_or_zero();
        gizmos.arrow(center, center + normal * 0.75, Color::srgb(1.0, 0.25, 0.95));
        gizmos.line(
            center - right * 0.35,
            center + right * 0.35,
            Color::srgb(1.0, 0.7, 0.2),
        );
        gizmos.line(
            center - up * 0.35,
            center + up * 0.35,
            Color::srgb(0.25, 1.0, 0.35),
        );

        let label = if gate.name.is_empty() {
            format!("T{idx}")
        } else {
            gate.name.clone()
        };
        gizmos.text(
            Isometry3d::from_translation(center + Vec3::Y * (size + 0.35)),
            &label,
            TARGET_LABEL_FONT_SIZE,
            Vec2::ZERO,
            color,
        );
    }
}

fn draw_replan_markers(gizmos: &mut Gizmos, replay: &ReplayReport, frames: &[DroneFrame]) {
    if replay.replans.is_empty() || frames.is_empty() {
        return;
    }
    for replan in &replay.replans {
        let frame_idx = replan.executed_steps.min(frames.len().saturating_sub(1));
        let pos = view_vec3(frames[frame_idx].pos_world);
        let color = if replan.passed_gate {
            Color::srgb(0.2, 1.0, 0.25)
        } else {
            Color::srgb(1.0, 0.45, 0.1)
        };
        gizmos.sphere(pos + Vec3::Y * 0.08, 0.055, color);
        gizmos.line(pos, pos + Vec3::Y * 0.45, color);
    }
}

fn draw_current_state_overlay(
    gizmos: &mut Gizmos,
    replay: &ReplayReport,
    frame_index: usize,
    frame: &DroneFrame,
    current_pos: Vec3,
    active_replan: Option<&ReplanStep>,
    active_gate: Option<&GateSpec>,
) {
    let transform = transform_from_frame(frame);
    gizmos.axes(transform, 0.55);
    gizmos.line(
        current_pos,
        Vec3::new(current_pos.x, 0.0, current_pos.z),
        Color::srgba(0.9, 0.9, 0.9, 0.35),
    );

    let lin_vel = body_vector_to_view_world(frame, frame.lin_vel_body);
    if lin_vel.length_squared() > 1e-6 {
        gizmos.arrow(
            current_pos,
            current_pos + lin_vel * 0.25,
            Color::srgb(0.35, 0.8, 1.0),
        );
    }

    if let Some(gate) = active_gate {
        let target = view_vec3(gate.center);
        gizmos.arrow(current_pos, target, Color::srgb(1.0, 0.15, 0.85));
        draw_cross(gizmos, target, 0.65, Color::srgb(1.0, 0.15, 0.85));
    }

    if let Some(replan) = active_replan {
        let anchor = view_vec3(replan.path_anchor);
        let carrot = view_vec3(replan.carrot);
        gizmos.line(anchor, carrot, Color::srgb(1.0, 0.55, 0.05));
        gizmos.sphere(anchor, 0.08, Color::srgb(1.0, 0.55, 0.05));
        gizmos.sphere(carrot, 0.22, Color::srgb(1.0, 0.55, 0.05));
        draw_cross(gizmos, carrot, 0.38, Color::srgb(1.0, 0.55, 0.05));
        gizmos.arrow(current_pos, carrot, Color::srgb(1.0, 0.55, 0.05));
    }

    if let Some(action) = action_at_frame(replay, frame_index.saturating_sub(1)) {
        draw_action_bars(gizmos, current_pos, action);
    }

    let replan_text = active_replan
        .map(|replan| {
            format!(
                "gate={} score={:.2} mean={:.2} eval/s={:.0}",
                replan.gate_name,
                replan.score_summary.best,
                replan.score_summary.mean,
                replan.planner_evals_per_sec
            )
        })
        .unwrap_or_else(|| "no replan metadata".to_string());
    let action_text = action_at_frame(replay, frame_index.saturating_sub(1))
        .map(|action| {
            format!(
                "a=[{:.2} {:.2} {:.2} {:.2}]",
                action[0], action[1], action[2], action[3]
            )
        })
        .unwrap_or_else(|| "a=[n/a]".to_string());
    let carrot_text = active_replan
        .map(|replan| {
            let carrot = replan.carrot;
            format!(
                " carrot=[{:.1} {:.1} {:.1}]",
                carrot[0], carrot[1], carrot[2]
            )
        })
        .unwrap_or_default();
    let text = format!(
        "f={} step={} row={} z={:.2} vbat={:.1} {}{}\n{}",
        frame_index,
        frame.step_idx,
        frame.row,
        frame.pos_world[2],
        frame.vbat,
        action_text,
        carrot_text,
        replan_text
    );
    gizmos.text(
        Isometry3d::from_translation(current_pos + Vec3::Y * 1.15),
        &text,
        TELEMETRY_FONT_SIZE,
        Vec2::new(-0.5, 0.0),
        Color::srgb(0.95, 0.95, 0.85),
    );
}

fn draw_cross(gizmos: &mut Gizmos, center: Vec3, size: f32, color: Color) {
    gizmos.line(center - Vec3::X * size, center + Vec3::X * size, color);
    gizmos.line(center - Vec3::Y * size, center + Vec3::Y * size, color);
    gizmos.line(center - Vec3::Z * size, center + Vec3::Z * size, color);
}

fn draw_action_bars(gizmos: &mut Gizmos, current_pos: Vec3, action: &[f32; 4]) {
    let origin = current_pos + Vec3::new(-0.45, 0.25, -0.75);
    for (idx, value) in action.iter().enumerate() {
        let base = origin + Vec3::X * (idx as f32 * 0.25);
        let height = value.clamp(-1.0, 1.0) * 0.55;
        let color = if *value >= 0.0 {
            Color::srgb(0.25, 1.0, 0.35)
        } else {
            Color::srgb(1.0, 0.25, 0.25)
        };
        gizmos.line(base, base + Vec3::Y * height, color);
        gizmos.sphere(base + Vec3::Y * height, 0.035, color);
    }
}

fn active_replan(replay: &ReplayReport, frame: usize) -> Option<&ReplanStep> {
    replay
        .replans
        .iter()
        .find(|replan| frame <= replan.executed_steps)
        .or_else(|| replay.replans.last())
}

fn action_at_frame(replay: &ReplayReport, frame: usize) -> Option<&[f32; 4]> {
    if replay.actions.is_empty() {
        None
    } else {
        replay.actions.get(frame.min(replay.actions.len() - 1))
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
    for frame in &replay.frames {
        bounds.include(view_vec3(frame.pos_world));
    }
    for gate in &replay.gate_loop {
        bounds.include(view_vec3(gate.center));
    }
    for replan in &replay.replans {
        bounds.include(view_vec3(replan.path_anchor));
        bounds.include(view_vec3(replan.carrot));
    }
    bounds
}

#[derive(Debug, Deserialize)]
struct ReplayReport {
    sample_rate_hz: usize,
    gate_loop: Vec<GateSpec>,
    frames: Vec<DroneFrame>,
    actions: Vec<[f32; 4]>,
    replans: Vec<ReplanStep>,
}

#[derive(Debug, Clone, Deserialize)]
struct DroneFrame {
    row: usize,
    step_idx: i64,
    pos_world: [f32; 3],
    rotmat_world_from_body: [f32; 9],
    lin_vel_body: [f32; 3],
    vbat: f32,
}

#[derive(Debug, Clone, Deserialize)]
struct GateSpec {
    name: String,
    center: [f32; 3],
    normal: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
}

#[derive(Debug, Clone, Deserialize)]
struct ReplanStep {
    executed_steps: usize,
    gate_index: usize,
    gate_name: String,
    passed_gate: bool,
    path_anchor: [f32; 3],
    carrot: [f32; 3],
    score_summary: ScoreSummary,
    planner_evals_per_sec: f32,
}

#[derive(Debug, Clone, Deserialize)]
struct ScoreSummary {
    best: f32,
    mean: f32,
}
