use std::{collections::VecDeque, env, path::PathBuf};

use anyhow::ensure;
use bevy::{
    gizmos::config::{DefaultGizmoConfigGroup, GizmoConfigStore},
    prelude::*,
};
use bevy_camera_controller::free_camera::{FreeCamera, FreeCameraPlugin, FreeCameraState};
use le_wm_nv::{
    models::skyjepa::{SkyJepaControllerSession, SkyJepaSessionConfig, SkyJepaWarmStart},
    runtime::DeviceSpec,
    skyjepa_sim::{SkyJepaDomain, SkyJepaRotorPlant, SkyJepaRotorState},
    skyjepa_task::{SkyJepaReferenceKind, skyjepa_reference_horizon, skyjepa_reference_state},
};

const STATE_DIM: usize = 18;
const ACTION_DIM: usize = 4;

fn main() -> anyhow::Result<()> {
    let args = Args::parse()?;
    let domain = if args.randomize_domain {
        SkyJepaDomain::sample(args.domain_seed)
    } else {
        SkyJepaDomain::default()
    };
    let initial_state = SkyJepaRotorState::hover();
    let device = args.device.resolve()?;
    let mut controller = SkyJepaControllerSession::load(
        &args.checkpoint_dir,
        device,
        SkyJepaSessionConfig {
            samples: args.samples,
            horizon: args.horizon,
            planner_seed: args.planner_seed,
            warm_start: args.warm_start,
            ..SkyJepaSessionConfig::default()
        },
        initial_state,
    )?;
    let reference = skyjepa_reference_horizon(
        args.scenario,
        0.0,
        controller.dt(),
        controller.horizon(),
        args.radius_m,
        args.period_seconds,
    );
    let cold_warmup_ms = controller.warm_up(initial_state, &reference)?;
    let state = SimState::new(
        domain,
        initial_state,
        cold_warmup_ms,
        args.scenario,
        args.domain_seed,
    )?;
    controller.reset_with_action(initial_state, state.plant.nominal_hover_action())?;
    let mut app = App::new();
    app.insert_resource(args)
        .insert_resource(state)
        .insert_non_send(controller)
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "SkyJEPA rotor-force UAV simulator".to_string(),
                    resolution: (1280, 800).into(),
                    ..default()
                }),
                ..default()
            }),
            FreeCameraPlugin,
        ))
        .add_systems(Startup, (configure_gizmos, setup).chain())
        .add_systems(
            Update,
            (
                keyboard_controls,
                step_control,
                update_drone_mesh,
                update_follow_camera,
                update_telemetry,
                draw_scene,
            )
                .chain(),
        );
    app.run();
    Ok(())
}

#[derive(Resource, Debug, Clone)]
struct Args {
    checkpoint_dir: PathBuf,
    device: DeviceSpec,
    samples: usize,
    horizon: usize,
    planner_seed: u64,
    warm_start: SkyJepaWarmStart,
    randomize_domain: bool,
    domain_seed: u64,
    simulation_rate_hz: usize,
    time_scale: f32,
    radius_m: f32,
    period_seconds: f32,
    scenario: SkyJepaReferenceKind,
    max_trail: usize,
}

impl Args {
    fn parse() -> anyhow::Result<Self> {
        let mut args = Self {
            checkpoint_dir: default_checkpoint_dir(),
            device: DeviceSpec::Cuda(0),
            samples: 512,
            horizon: 15,
            planner_seed: 7,
            warm_start: SkyJepaWarmStart::FreshPrior,
            randomize_domain: false,
            domain_seed: 9001,
            simulation_rate_hz: 200,
            time_scale: 1.0,
            radius_m: 2.0,
            period_seconds: 8.0,
            scenario: SkyJepaReferenceKind::Circle,
            max_trail: 2400,
        };
        let mut iter = env::args().skip(1);
        while let Some(flag) = iter.next() {
            match flag.as_str() {
                "--checkpoint-dir" => args.checkpoint_dir = next_path(&mut iter, &flag)?,
                "--device" => args.device = next_parse(&mut iter, &flag)?,
                "--samples" => args.samples = next_parse(&mut iter, &flag)?,
                "--horizon" => args.horizon = next_parse(&mut iter, &flag)?,
                "--planner-seed" => args.planner_seed = next_parse(&mut iter, &flag)?,
                "--warm-start" => {
                    args.warm_start = match iter.next().as_deref() {
                        Some("fresh-prior") => SkyJepaWarmStart::FreshPrior,
                        Some("shifted-residual") => SkyJepaWarmStart::ShiftedResidual,
                        _ => anyhow::bail!("--warm-start requires fresh-prior or shifted-residual"),
                    };
                }
                "--randomize-domain" => args.randomize_domain = true,
                "--domain-seed" => args.domain_seed = next_parse(&mut iter, &flag)?,
                "--simulation-rate-hz" => args.simulation_rate_hz = next_parse(&mut iter, &flag)?,
                "--time-scale" => args.time_scale = next_parse(&mut iter, &flag)?,
                "--radius-m" => args.radius_m = next_parse(&mut iter, &flag)?,
                "--period-seconds" => args.period_seconds = next_parse(&mut iter, &flag)?,
                "--scenario" => args.scenario = next_string(&mut iter, &flag)?.parse()?,
                "--max-trail" => args.max_trail = next_parse(&mut iter, &flag)?,
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                other => anyhow::bail!("unknown argument `{other}`; use --help"),
            }
        }
        ensure!(
            args.checkpoint_dir.is_dir(),
            "checkpoint-dir does not exist"
        );
        ensure!(args.samples > 0, "samples must be positive");
        ensure!(args.horizon > 0, "horizon must be positive");
        ensure!(
            args.simulation_rate_hz >= 20 && args.simulation_rate_hz.is_multiple_of(20),
            "simulation-rate-hz must be an integer multiple of 20"
        );
        ensure!(
            args.time_scale.is_finite() && args.time_scale > 0.0,
            "time-scale must be positive"
        );
        ensure!(
            args.radius_m.is_finite() && args.radius_m > 0.0,
            "radius-m must be positive"
        );
        ensure!(
            args.period_seconds.is_finite() && args.period_seconds > 0.0,
            "period-seconds must be positive"
        );
        ensure!(args.max_trail > 1, "max-trail must exceed one");
        Ok(args)
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
        .parse()
        .map_err(|error| anyhow::anyhow!("invalid value `{value}` for {flag}: {error}"))
}

fn next_string(iter: &mut impl Iterator<Item = String>, flag: &str) -> anyhow::Result<String> {
    iter.next()
        .ok_or_else(|| anyhow::anyhow!("missing value after {flag}"))
}

fn next_path(iter: &mut impl Iterator<Item = String>, flag: &str) -> anyhow::Result<PathBuf> {
    Ok(PathBuf::from(next_string(iter, flag)?))
}

fn print_help() {
    println!(
        "Usage: skyjepa-drone-sim [options]\n\
         --checkpoint-dir <dir>       trained SkyJEPA run directory\n\
         --device <cuda[:idx]>        default cuda:0\n\
         --samples <n>                MPPI samples, default 512\n\
         --horizon <n>                MPPI horizon, default 15\n\
         --scenario <name>            hover, circle, or figure-eight\n\
         --radius-m <meters>           default 2\n\
         --period-seconds <seconds>    default 8\n\
         --randomize-domain           use a held-out randomized plant\n\
         --domain-seed <u64>          default 9001\n\
         --planner-seed <u64>         default 7\n\
         --simulation-rate-hz <hz>    default 200\n\
         --time-scale <scale>         default 1\n\
         Controls: Space pause, L controller, Backspace reset, R new domain,\n\
         1 hover, 2 circle, 3 figure-eight; right mouse enables free camera"
    );
}

#[derive(Resource)]
struct SimState {
    plant: SkyJepaRotorPlant,
    initial_state: SkyJepaRotorState,
    action: [f32; ACTION_DIM],
    prior_action: [f32; ACTION_DIM],
    model_correction_l2: f32,
    time: f32,
    wall_accumulator: f32,
    paused: bool,
    controller_enabled: bool,
    scenario: SkyJepaReferenceKind,
    domain_seed: u64,
    trail: Vec<[f32; 3]>,
    reference: Vec<[f32; STATE_DIM]>,
    prediction: Vec<[f32; STATE_DIM]>,
    plan_times_ms: VecDeque<f64>,
    best_score: f32,
    cold_warmup_ms: f64,
    position_error_m: f32,
    max_position_error_m: f32,
    control_steps: usize,
    last_error: Option<String>,
}

impl SimState {
    fn new(
        domain: SkyJepaDomain,
        initial_state: SkyJepaRotorState,
        cold_warmup_ms: f64,
        scenario: SkyJepaReferenceKind,
        domain_seed: u64,
    ) -> anyhow::Result<Self> {
        let plant = SkyJepaRotorPlant::new(domain, initial_state)?;
        let hover_action = plant.nominal_hover_action();
        Ok(Self {
            action: hover_action,
            prior_action: hover_action,
            model_correction_l2: 0.0,
            plant,
            initial_state,
            time: 0.0,
            wall_accumulator: 0.0,
            paused: false,
            controller_enabled: true,
            scenario,
            domain_seed,
            trail: vec![initial_state.position],
            reference: Vec::new(),
            prediction: Vec::new(),
            plan_times_ms: VecDeque::with_capacity(240),
            best_score: f32::NAN,
            cold_warmup_ms,
            position_error_m: 0.0,
            max_position_error_m: 0.0,
            control_steps: 0,
            last_error: None,
        })
    }

    fn reset(&mut self, controller: &mut SkyJepaControllerSession) -> anyhow::Result<()> {
        let domain = self.plant.domain();
        self.plant = SkyJepaRotorPlant::new(domain, self.initial_state)?;
        self.action = self.plant.nominal_hover_action();
        self.prior_action = self.action;
        self.model_correction_l2 = 0.0;
        controller.reset_with_action(self.initial_state, self.action)?;
        self.time = 0.0;
        self.wall_accumulator = 0.0;
        self.trail.clear();
        self.trail.push(self.initial_state.position);
        self.reference.clear();
        self.prediction.clear();
        self.plan_times_ms.clear();
        self.best_score = f32::NAN;
        self.position_error_m = 0.0;
        self.max_position_error_m = 0.0;
        self.control_steps = 0;
        self.last_error = None;
        Ok(())
    }
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

#[derive(Component)]
struct Telemetry;

fn configure_gizmos(mut store: ResMut<GizmoConfigStore>) {
    store.config_mut::<DefaultGizmoConfigGroup>().0.line.width = 1.5;
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 4.0, 8.0).looking_at(Vec3::Y, Vec3::Y),
        FreeCamera {
            walk_speed: 6.0,
            run_speed: 24.0,
            ..default()
        },
        disabled_free_camera_state(),
        FollowCamera,
    ));
    commands.spawn((
        Text::new(""),
        TextFont::from_font_size(14.0),
        TextColor(Color::srgb(0.95, 0.95, 0.86)),
        Node {
            position_type: PositionType::Absolute,
            top: px(10),
            left: px(10),
            max_width: px(980),
            ..default()
        },
        Telemetry,
    ));
    commands.spawn((
        PointLight {
            intensity: 6000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(-8.0, 14.0, 10.0),
    ));
    let floor = meshes.add(Plane3d::default().mesh().size(80.0, 80.0));
    let floor_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.10, 0.105, 0.11),
        perceptual_roughness: 0.9,
        ..default()
    });
    commands.spawn((Mesh3d(floor), MeshMaterial3d(floor_material)));
    let drone_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.92, 0.92, 0.86),
        perceptual_roughness: 0.7,
        ..default()
    });
    let nose_material = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.75, 0.05),
        emissive: LinearRgba::rgb(0.5, 0.3, 0.0),
        ..default()
    });
    for (mesh, material, part) in [
        (
            meshes.add(Cuboid::new(0.45, 0.12, 0.18)),
            drone_material.clone(),
            DronePart::Body,
        ),
        (
            meshes.add(Cuboid::new(0.95, 0.04, 0.05)),
            drone_material.clone(),
            DronePart::ArmX,
        ),
        (
            meshes.add(Cuboid::new(0.05, 0.04, 0.95)),
            drone_material,
            DronePart::ArmY,
        ),
        (
            meshes.add(Cuboid::new(0.18, 0.07, 0.07)),
            nose_material,
            DronePart::Nose,
        ),
    ] {
        commands.spawn((Mesh3d(mesh), MeshMaterial3d(material), part));
    }
}

fn disabled_free_camera_state() -> FreeCameraState {
    let mut state = FreeCameraState::default();
    state.enabled = false;
    state
}

fn keyboard_controls(
    keys: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<SimState>,
    mut controller: NonSendMut<SkyJepaControllerSession>,
) {
    if keys.just_pressed(KeyCode::Space) {
        state.paused = !state.paused;
    }
    if keys.just_pressed(KeyCode::KeyL) {
        state.controller_enabled = !state.controller_enabled;
    }
    if keys.just_pressed(KeyCode::Digit1) {
        state.scenario = SkyJepaReferenceKind::Hover;
    }
    if keys.just_pressed(KeyCode::Digit2) {
        state.scenario = SkyJepaReferenceKind::Circle;
    }
    if keys.just_pressed(KeyCode::Digit3) {
        state.scenario = SkyJepaReferenceKind::FigureEight;
    }
    if keys.just_pressed(KeyCode::Backspace)
        && let Err(error) = state.reset(&mut controller)
    {
        state.last_error = Some(error.to_string());
        state.paused = true;
    }
    if keys.just_pressed(KeyCode::KeyR) {
        state.domain_seed = state.domain_seed.wrapping_add(1);
        let domain = SkyJepaDomain::sample(state.domain_seed);
        match SkyJepaRotorPlant::new(domain, state.initial_state) {
            Ok(plant) => {
                state.plant = plant;
                if let Err(error) = state.reset(&mut controller) {
                    state.last_error = Some(error.to_string());
                    state.paused = true;
                }
            }
            Err(error) => {
                state.last_error = Some(error.to_string());
                state.paused = true;
            }
        }
    }
}

fn step_control(
    time: Res<Time>,
    args: Res<Args>,
    mut state: ResMut<SimState>,
    mut controller: NonSendMut<SkyJepaControllerSession>,
) {
    if state.paused {
        return;
    }
    state.wall_accumulator += time.delta_secs() * args.time_scale;
    let control_dt = controller.dt();
    while state.wall_accumulator >= control_dt {
        state.wall_accumulator -= control_dt;
        let references = skyjepa_reference_horizon(
            state.scenario,
            state.time,
            control_dt,
            controller.horizon(),
            args.radius_m,
            args.period_seconds,
        );
        if state.controller_enabled {
            match controller.plan_with_prediction(&references) {
                Ok(plan) => {
                    state.action = plan.action;
                    state.prior_action = plan.prior_action;
                    state.model_correction_l2 = plan
                        .action_correction
                        .iter()
                        .map(|value| value * value)
                        .sum::<f32>()
                        .sqrt();
                    state.prediction = plan.predicted_states;
                    state.best_score = plan.best_candidate_score;
                    state.plan_times_ms.push_back(plan.plan_ms);
                    if state.plan_times_ms.len() > 240 {
                        state.plan_times_ms.pop_front();
                    }
                }
                Err(error) => {
                    state.last_error = Some(error.to_string());
                    state.paused = true;
                    return;
                }
            }
        } else {
            state.action = state.plant.nominal_hover_action();
            state.prior_action = state.action;
            state.model_correction_l2 = 0.0;
            state.prediction.clear();
        }
        state.reference = references;
        let substeps = args.simulation_rate_hz / 20;
        let sim_dt = control_dt / substeps as f32;
        let action = state.action;
        for _ in 0..substeps {
            state.plant.step(action, sim_dt);
        }
        state.time += control_dt;
        state.control_steps += 1;
        controller.commit_observation(state.plant.state(), action);
        let target = skyjepa_reference_state(
            state.scenario,
            state.time,
            args.radius_m,
            args.period_seconds,
        );
        state.position_error_m = distance(
            state.plant.state().position,
            target[0..3].try_into().unwrap(),
        );
        state.max_position_error_m = state.max_position_error_m.max(state.position_error_m);
        let position = state.plant.state().position;
        state.trail.push(position);
        if state.trail.len() > args.max_trail {
            state.trail.remove(0);
        }
    }
}

fn update_drone_mesh(state: Res<SimState>, mut query: Query<(&DronePart, &mut Transform)>) {
    let base = transform_from_state(state.plant.state());
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
    mut query: Query<(&mut Transform, &FreeCameraState), With<FollowCamera>>,
) {
    let Ok((mut transform, free)) = query.single_mut() else {
        return;
    };
    if free.enabled {
        return;
    }
    let focus = view_position(state.plant.state().position) + Vec3::Y * 0.3;
    let desired = focus + Vec3::new(0.0, 2.8, 7.5);
    let alpha = 1.0 - (-8.0 * time.delta_secs()).exp();
    transform.translation = transform.translation.lerp(desired, alpha);
    transform.look_at(focus, Vec3::Y);
}

fn update_telemetry(
    state: Res<SimState>,
    controller: NonSend<SkyJepaControllerSession>,
    mut query: Query<&mut Text, With<Telemetry>>,
) {
    let Ok(mut text) = query.single_mut() else {
        return;
    };
    let mut times = state.plan_times_ms.iter().copied().collect::<Vec<_>>();
    times.sort_by(f64::total_cmp);
    let percentile = |fraction: f64| {
        if times.is_empty() {
            0.0
        } else {
            times[((times.len() - 1) as f64 * fraction).round() as usize]
        }
    };
    let domain = state.plant.domain();
    let status = if state.paused { "paused" } else { "running" };
    let control = if state.controller_enabled {
        "on"
    } else {
        "off"
    };
    let error = state
        .last_error
        .as_deref()
        .map(|error| format!("\nERROR: {error}"))
        .unwrap_or_default();
    **text = format!(
        "SkyJEPA {} controller={} scenario={} t={:.2}s step={}\n\
         position=[{:.2} {:.2} {:.2}] error={:.3}m max={:.3}m\n\
         rotor-force=[{:.2} {:.2} {:.2} {:.2}] prior=[{:.2} {:.2} {:.2} {:.2}] model-delta={:.3} score={:.3}\n\
         MPPI samples={} horizon={} p50={:.2}ms p95={:.2}ms cold={:.1}ms\n\
         domain seed={} mass={:.3}kg inertia=[{:.4} {:.4} {:.4}] lag={:.3}s drag=[{:.2} {:.2} {:.2}]\n\
         Space pause | L controller | Backspace reset | R new domain | 1/2/3 reference{}",
        status,
        control,
        state.scenario.as_str(),
        state.time,
        state.control_steps,
        state.plant.state().position[0],
        state.plant.state().position[1],
        state.plant.state().position[2],
        state.position_error_m,
        state.max_position_error_m,
        state.action[0],
        state.action[1],
        state.action[2],
        state.action[3],
        state.prior_action[0],
        state.prior_action[1],
        state.prior_action[2],
        state.prior_action[3],
        state.model_correction_l2,
        state.best_score,
        controller.samples(),
        controller.horizon(),
        percentile(0.5),
        percentile(0.95),
        state.cold_warmup_ms,
        state.domain_seed,
        domain.mass,
        domain.inertia[0],
        domain.inertia[1],
        domain.inertia[2],
        domain.motor_time_constant,
        domain.drag[0],
        domain.drag[1],
        domain.drag[2],
        error,
    );
}

fn draw_scene(mut gizmos: Gizmos, state: Res<SimState>) {
    let grid = Color::srgb(0.23, 0.24, 0.25);
    for index in -20..=20 {
        let value = index as f32;
        gizmos.line(
            Vec3::new(value, 0.0, -20.0),
            Vec3::new(value, 0.0, 20.0),
            grid,
        );
        gizmos.line(
            Vec3::new(-20.0, 0.0, value),
            Vec3::new(20.0, 0.0, value),
            grid,
        );
    }
    draw_polyline(
        &mut gizmos,
        state.trail.iter().copied(),
        Color::srgb(0.95, 0.82, 0.12),
    );
    draw_polyline(
        &mut gizmos,
        state
            .reference
            .iter()
            .map(|state| [state[0], state[1], state[2]]),
        Color::srgb(0.25, 0.9, 1.0),
    );
    draw_polyline(
        &mut gizmos,
        state
            .prediction
            .iter()
            .map(|state| [state[0], state[1], state[2]]),
        Color::srgb(1.0, 0.25, 0.85),
    );
    if let Some(target) = state.reference.first() {
        let target = view_position([target[0], target[1], target[2]]);
        gizmos.sphere(target, 0.13, Color::srgb(0.25, 0.9, 1.0));
    }
    let position = view_position(state.plant.state().position);
    for (index, force) in state.action.iter().enumerate() {
        let base = position + Vec3::new(-0.45 + index as f32 * 0.3, 0.25, -0.65);
        let height = force * 0.10;
        gizmos.line(base, base + Vec3::Y * height, Color::srgb(0.3, 1.0, 0.35));
    }
}

fn draw_polyline(gizmos: &mut Gizmos, points: impl Iterator<Item = [f32; 3]>, color: Color) {
    let mut previous = None;
    for point in points {
        let point = view_position(point);
        if let Some(previous) = previous {
            gizmos.line(previous, point, color);
        }
        previous = Some(point);
    }
}

fn transform_from_state(state: SkyJepaRotorState) -> Transform {
    let rotation = Quat::from_mat3(&Mat3::from_cols(
        Vec3::new(
            state.rotation_world_from_body[0],
            state.rotation_world_from_body[3],
            state.rotation_world_from_body[6],
        ),
        Vec3::new(
            state.rotation_world_from_body[1],
            state.rotation_world_from_body[4],
            state.rotation_world_from_body[7],
        ),
        Vec3::new(
            state.rotation_world_from_body[2],
            state.rotation_world_from_body[5],
            state.rotation_world_from_body[8],
        ),
    ));
    let body_x = rotation * Vec3::X;
    let body_y = rotation * Vec3::Y;
    let body_z = rotation * Vec3::Z;
    Transform {
        translation: view_position(state.position),
        rotation: Quat::from_mat3(&Mat3::from_cols(
            view_vector(body_x),
            view_vector(body_z),
            view_vector(body_y),
        )),
        scale: Vec3::ONE,
    }
}

fn view_position(value: [f32; 3]) -> Vec3 {
    Vec3::new(value[0], value[2], value[1])
}

fn view_vector(value: Vec3) -> Vec3 {
    Vec3::new(value.x, value.z, value.y)
}

fn distance(lhs: [f32; 3], rhs: [f32; 3]) -> f32 {
    ((lhs[0] - rhs[0]).powi(2) + (lhs[1] - rhs[1]).powi(2) + (lhs[2] - rhs[2]).powi(2)).sqrt()
}

fn default_checkpoint_dir() -> PathBuf {
    env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".stable_worldmodel")
        .join("le-wm-nv-runs")
        .join("skyjepa-drone-state18-20hz")
}
