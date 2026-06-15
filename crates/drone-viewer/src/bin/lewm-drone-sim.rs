use std::{env, fs, path::PathBuf, time::Instant};

use anyhow::{Context, ensure};
use bevy::{
    gizmos::config::{DefaultGizmoConfigGroup, GizmoConfigStore},
    input::mouse::MouseWheel,
    prelude::*,
};
use bevy_camera_controller::free_camera::{FreeCamera, FreeCameraPlugin, FreeCameraState};
use candle::{D, DType, Device as CandleDevice, Tensor};
use le_wm_nv::{
    checkpoint,
    data::drone_racing::{
        DroneBatchConfig, DroneNormalization, DroneRacingDataset, GateSequenceFile, GateSpec,
        RunningStats,
    },
    drone_plant::{DronePlantConfig, DronePlantState},
    models::world_model::{ObservationEncoderConfig, WorldModel, WorldModelConfig},
    planner::{ActionBounds, CandidateScorer, IcemConfig, IcemPlanner},
    runtime::{DTypeSpec, DeviceSpec},
};

const ACTION_DIM: usize = 4;
const OBS_DIM: usize = 12;
const TELEMETRY_FONT_SIZE: f32 = 0.11;
const KEYBOARD_MAX_TILT_RAD: f32 = 0.42;
const KEYBOARD_ATTITUDE_KP: f32 = 6.0;
const KEYBOARD_YAW_LIMIT: f32 = 0.28;
const KEYBOARD_RATE_SLEW: f32 = 4.0;
const KEYBOARD_CLIMB_RATE_MPS: f32 = 2.0;
const KEYBOARD_VERTICAL_VEL_KP: f32 = 0.05;
const KEYBOARD_THROTTLE_SLEW: f32 = 1.5;
const GATE_ENTRY_LEAD_STEPS: usize = 12;
const GATE_REFERENCE_SEARCH_LIMIT_STEPS: usize = 450;

fn main() -> anyhow::Result<()> {
    let args = Args::parse()?;
    if args.inspect_gate_targets {
        return inspect_gate_targets(args);
    }
    if args.oracle_replay_rows > 0 {
        return run_oracle_replay(args);
    }
    if args.headless_steps > 0 {
        return run_headless(args);
    }

    let mut sim_state = initial_sim_state(&args)?;
    let gate_loop = args
        .gate_loop
        .as_ref()
        .map(|cfg| GateLoop::load(cfg, args.start.as_ref(), &mut sim_state))
        .transpose()?;
    let controller = args
        .lewm
        .as_ref()
        .map(|cfg| DroneLeWmController::load(cfg, &args.dynamics, &sim_state))
        .transpose()?;
    let mut app = App::new();
    app.insert_resource(args.camera)
        .insert_resource(SimControl::new(args.dynamics.hover_throttle))
        .insert_resource(sim_state)
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "le-wm-nv Drone Dynamics Sim".to_string(),
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
                update_controls,
                update_free_camera_state,
                step_simulation,
                update_drone_mesh,
                update_follow_camera,
                update_telemetry_ui,
                draw_guides,
                draw_sim_overlays,
            )
                .chain(),
        );
    if let Some(controller) = controller {
        app.insert_non_send(controller);
    }
    if let Some(gate_loop) = gate_loop {
        app.insert_resource(gate_loop);
    }
    app.run();
    Ok(())
}

fn run_headless(args: Args) -> anyhow::Result<()> {
    let mut state = initial_sim_state(&args)?;
    let mut gate_loop = args
        .gate_loop
        .as_ref()
        .map(|cfg| GateLoop::load(cfg, args.start.as_ref(), &mut state))
        .transpose()?;
    let mut control = SimControl::new(args.dynamics.hover_throttle);
    let lewm = args
        .lewm
        .as_ref()
        .context("--headless-steps requires LeWM control config")?;
    let mut controller = DroneLeWmController::load(lewm, &args.dynamics, &state)?;
    control.action = controller.plan(&state)?;
    let start_dist = (state.target.pos_world - state.pose.pos_world).length();
    let started = Instant::now();
    let dt = 1.0 / state.dynamics.sim_hz;

    for _ in 0..args.headless_steps {
        state.step(control.action, dt);
        if let Some(gate_loop) = gate_loop.as_mut() {
            gate_loop.update_state_target(&mut state);
            if gate_loop.finished {
                break;
            }
        }
        if controller.observe_if_due(&state) && controller.should_plan_now() {
            control.action = controller.plan(&state)?;
        }
    }
    let end_dist = (state.target.pos_world - state.pose.pos_world).length();
    let gate_text = gate_loop
        .as_ref()
        .map(|gate_loop| {
            format!(
                " gates={} lap={}/{} gate={}/{} finished={} best_gate_dists=[{}]",
                gate_loop.pass_count,
                gate_loop.laps_completed,
                gate_loop.desired_laps,
                gate_loop.current_index + 1,
                gate_loop.gates.len(),
                gate_loop.finished,
                gate_loop.best_distance_text(),
            )
        })
        .unwrap_or_default();
    println!(
        "headless steps={} sim_time={:.3}s wall={:.3}s start_dist={:.3} end_dist={:.3} pos=[{:.3} {:.3} {:.3}] action=[{:.3} {:.3} {:.3} {:.3}] plans={} last_plan_ms={:.3} best={:.6}{}",
        args.headless_steps,
        state.time,
        started.elapsed().as_secs_f32(),
        start_dist,
        end_dist,
        state.pose.pos_world.x,
        state.pose.pos_world.y,
        state.pose.pos_world.z,
        control.action[0],
        control.action[1],
        control.action[2],
        control.action[3],
        controller.plan_count,
        controller.last_plan_ms,
        controller.last_best_score,
        gate_text,
    );
    Ok(())
}

fn inspect_gate_targets(args: Args) -> anyhow::Result<()> {
    let mut state = initial_sim_state(&args)?;
    let gate_cfg = args
        .gate_loop
        .as_ref()
        .context("--inspect-gate-targets requires --gates")?;
    let gate_loop = GateLoop::load(gate_cfg, args.start.as_ref(), &mut state)?;
    println!("start {}", pose_summary(None, state.pose));
    println!(
        "flight={} gates={} order_target_lead_steps={}",
        gate_loop.flight,
        gate_loop.gates.len(),
        GATE_ENTRY_LEAD_STEPS
    );
    for (idx, gate) in gate_loop.gates.iter().enumerate() {
        println!(
            "{} {} center=[{:.3} {:.3} {:.3}] display_normal=[{:.3} {:.3} {:.3}]",
            idx + 1,
            gate.name,
            gate.center.x,
            gate.center.y,
            gate.center.z,
            gate.display_normal.x,
            gate.display_normal.y,
            gate.display_normal.z,
        );
        println!(
            "  entry {}",
            target_pose_summary(Some(gate.entry_row), gate.entry_pose())
        );
        println!(
            "  pass  {}",
            target_pose_summary(Some(gate.pass_row), gate.pass_pose())
        );
    }
    Ok(())
}

fn run_oracle_replay(args: Args) -> anyhow::Result<()> {
    let start = args
        .start
        .as_ref()
        .context("--oracle-replay-rows requires --start-row")?;
    let start_row = start
        .row
        .context("--oracle-replay-rows requires --start-row")?;
    let dataset = DroneRacingDataset::open(
        &start.dataset_dir,
        DroneBatchConfig {
            batch_size: 1,
            sequence_steps: 2,
            normalize_observations: false,
            normalize_actions: false,
        },
    )
    .with_context(|| {
        format!(
            "failed to open oracle dataset {}",
            start.dataset_dir.display()
        )
    })?;
    let mut state = initial_sim_state(&args)?;
    let mut gate_loop = args
        .gate_loop
        .as_ref()
        .map(|cfg| GateLoop::load(cfg, args.start.as_ref(), &mut state))
        .transpose()?;

    let mut compared = 0usize;
    let mut pos_sq_sum = 0.0f64;
    let mut rot_sq_sum = 0.0f64;
    let mut max_pos_err = 0.0f32;
    let mut final_pos_err = f32::NAN;
    let mut final_rot_err = f32::NAN;
    let mut final_row = start_row;
    let started = Instant::now();

    for offset in 0..args.oracle_replay_rows {
        let row = start_row + offset;
        let frame = dataset.frame(row)?;
        let next_row = row + 1;
        if next_row >= dataset.metadata().rows {
            break;
        }
        let next = dataset.frame(next_row)?;
        if frame.episode_idx != next.episode_idx {
            break;
        }
        let duration = frame.dt.max(1.0 / state.dynamics.sim_hz);
        let substeps = (duration * state.dynamics.sim_hz).round().max(1.0) as usize;
        let dt = duration / substeps as f32;
        for _ in 0..substeps {
            state.step(frame.channels_norm, dt);
            if let Some(gate_loop) = gate_loop.as_mut() {
                gate_loop.update_state_target(&mut state);
            }
        }
        let expected = DronePose::from_plant(DronePlantState::from_frame(&next));
        final_pos_err = state.pose.pos_world.distance(expected.pos_world);
        final_rot_err = state
            .pose
            .rot_world_from_body
            .angle_between(expected.rot_world_from_body);
        max_pos_err = max_pos_err.max(final_pos_err);
        pos_sq_sum += f64::from(final_pos_err).powi(2);
        rot_sq_sum += f64::from(final_rot_err).powi(2);
        compared += 1;
        final_row = next_row;
        if gate_loop
            .as_ref()
            .is_some_and(|gate_loop| gate_loop.finished)
        {
            break;
        }
    }

    let gate_text = gate_loop
        .as_ref()
        .map(|gate_loop| {
            format!(
                " gates={} lap={}/{} gate={}/{} finished={} best_gate_dists=[{}]",
                gate_loop.pass_count,
                gate_loop.laps_completed,
                gate_loop.desired_laps,
                gate_loop.current_index + 1,
                gate_loop.gates.len(),
                gate_loop.finished,
                gate_loop.best_distance_text(),
            )
        })
        .unwrap_or_default();
    let pos_rmse = if compared > 0 {
        (pos_sq_sum / compared as f64).sqrt()
    } else {
        f64::NAN
    };
    let rot_rmse = if compared > 0 {
        (rot_sq_sum / compared as f64).sqrt()
    } else {
        f64::NAN
    };
    println!(
        "oracle_replay start_row={} final_row={} compared_rows={} sim_time={:.3}s wall={:.3}s pos_rmse={:.3} max_pos_err={:.3} final_pos_err={:.3} rot_rmse={:.3} final_rot_err={:.3} final_pos=[{:.3} {:.3} {:.3}]{}",
        start_row,
        final_row,
        compared,
        state.time,
        started.elapsed().as_secs_f32(),
        pos_rmse,
        max_pos_err,
        final_pos_err,
        rot_rmse,
        final_rot_err,
        state.pose.pos_world.x,
        state.pose.pos_world.y,
        state.pose.pos_world.z,
        gate_text,
    );
    Ok(())
}

#[derive(Debug, Clone)]
struct Args {
    dynamics: DynamicsConfig,
    camera: FollowCameraConfig,
    target: TargetConfig,
    gate_loop: Option<GateLoopConfig>,
    start: Option<StartConfig>,
    lewm: Option<LeWmControlConfig>,
    max_trail: usize,
    headless_steps: usize,
    inspect_gate_targets: bool,
    oracle_replay_rows: usize,
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
            target: TargetConfig {
                pos_world: Vec3::new(4.0, 0.0, 1.6),
                yaw_rad: 0.0,
            },
            gate_loop: None,
            start: None,
            lewm: None,
            max_trail: 2400,
            headless_steps: 0,
            inspect_gate_targets: false,
            oracle_replay_rows: 0,
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
                "--target-pos" => args.target.pos_world = next_vec3(&mut iter, &arg)?,
                "--target-yaw" => args.target.yaw_rad = next_parse(&mut iter, &arg)?,
                "--start-dataset" => args.start_mut().dataset_dir = next_path(&mut iter, &arg)?,
                "--start-row" => args.start_mut().row = Some(next_parse(&mut iter, &arg)?),
                "--gates" => args.gate_loop_mut().path = next_path(&mut iter, &arg)?,
                "--gate-flight" => {
                    args.gate_loop_mut().flight = Some(next_string(&mut iter, &arg)?)
                }
                "--gate-episode" => {
                    args.gate_loop_mut().episode_idx = Some(next_parse(&mut iter, &arg)?)
                }
                "--gate-laps" => args.gate_loop_mut().desired_laps = next_parse(&mut iter, &arg)?,
                "--gate-radius" => {
                    args.gate_loop_mut().pass_radius_m = next_parse(&mut iter, &arg)?
                }
                "--gate-order" => {
                    args.gate_loop_mut().order = Some(next_gate_order(&mut iter, &arg)?)
                }
                "--model-dir" => args.lewm_mut().model_dir = Some(next_path(&mut iter, &arg)?),
                "--weights" => args.lewm_mut().weights = Some(next_path(&mut iter, &arg)?),
                "--config" => args.lewm_mut().config = Some(next_path(&mut iter, &arg)?),
                "--normalization" => {
                    args.lewm_mut().normalization = Some(next_path(&mut iter, &arg)?)
                }
                "--device" => args.lewm_mut().device = next_parse(&mut iter, &arg)?,
                "--dtype" => args.lewm_mut().dtype = next_parse(&mut iter, &arg)?,
                "--model-hz" => args.lewm_mut().model_hz = next_parse(&mut iter, &arg)?,
                "--planner-horizon" => args.lewm_mut().horizon = next_parse(&mut iter, &arg)?,
                "--planner-samples" => args.lewm_mut().samples = next_parse(&mut iter, &arg)?,
                "--planner-elites" => args.lewm_mut().elites = next_parse(&mut iter, &arg)?,
                "--planner-iterations" => args.lewm_mut().iterations = next_parse(&mut iter, &arg)?,
                "--planner-every" => {
                    args.lewm_mut().plan_every_model_steps = next_parse(&mut iter, &arg)?
                }
                "--planner-init-std" => args.lewm_mut().init_std = next_parse(&mut iter, &arg)?,
                "--planner-min-std" => args.lewm_mut().min_std = next_parse(&mut iter, &arg)?,
                "--planner-objective" => args.lewm_mut().objective = next_parse(&mut iter, &arg)?,
                "--seed" => args.lewm_mut().seed = Some(next_parse(&mut iter, &arg)?),
                "--headless-steps" => args.headless_steps = next_parse(&mut iter, &arg)?,
                "--inspect-gate-targets" => args.inspect_gate_targets = true,
                "--oracle-replay-rows" => args.oracle_replay_rows = next_parse(&mut iter, &arg)?,
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

    fn lewm_mut(&mut self) -> &mut LeWmControlConfig {
        self.lewm.get_or_insert_with(LeWmControlConfig::default)
    }

    fn gate_loop_mut(&mut self) -> &mut GateLoopConfig {
        self.gate_loop.get_or_insert_with(GateLoopConfig::default)
    }

    fn start_mut(&mut self) -> &mut StartConfig {
        self.start.get_or_insert_with(StartConfig::default)
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
        ensure!(
            self.dynamics.max_roll_rate > 0.0,
            "--max-roll-rate must be positive"
        );
        ensure!(
            self.dynamics.max_pitch_rate > 0.0,
            "--max-pitch-rate must be positive"
        );
        ensure!(
            self.dynamics.max_yaw_rate > 0.0,
            "--max-yaw-rate must be positive"
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
        ensure!(
            self.target.pos_world.is_finite(),
            "--target-pos must contain finite values"
        );
        ensure!(
            self.target.yaw_rad.is_finite(),
            "--target-yaw must be finite"
        );
        if let Some(lewm) = self.lewm.as_ref() {
            lewm.validate()?;
        }
        if let Some(gate_loop) = self.gate_loop.as_ref() {
            gate_loop.validate()?;
        }
        if let Some(start) = self.start.as_ref() {
            start.validate()?;
        }
        if self.headless_steps > 0 {
            ensure!(
                self.lewm.is_some(),
                "--headless-steps requires --model-dir or --weights"
            );
        }
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

fn next_path(iter: &mut impl Iterator<Item = String>, flag: &str) -> anyhow::Result<PathBuf> {
    Ok(PathBuf::from(iter.next().ok_or_else(|| {
        anyhow::anyhow!("missing value after {flag}")
    })?))
}

fn next_string(iter: &mut impl Iterator<Item = String>, flag: &str) -> anyhow::Result<String> {
    iter.next()
        .ok_or_else(|| anyhow::anyhow!("missing value after {flag}"))
}

fn next_vec3(iter: &mut impl Iterator<Item = String>, flag: &str) -> anyhow::Result<Vec3> {
    let value = iter
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing value after {flag}"))?;
    parse_vec3(&value).with_context(|| format!("invalid value `{value}` for {flag}"))
}

fn next_gate_order(
    iter: &mut impl Iterator<Item = String>,
    flag: &str,
) -> anyhow::Result<Vec<usize>> {
    let value = iter
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing value after {flag}"))?;
    let order = value
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .with_context(|| format!("invalid gate index in {flag}: {part}"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    ensure!(
        !order.is_empty(),
        "{flag} must contain at least one gate index"
    );
    ensure!(
        order.iter().all(|idx| *idx > 0),
        "{flag} uses 1-based gate indexes"
    );
    Ok(order)
}

fn parse_vec3(value: &str) -> anyhow::Result<Vec3> {
    let parts = value.split(',').collect::<Vec<_>>();
    ensure!(
        parts.len() == 3,
        "expected comma-separated x,y,z, got {value}"
    );
    let x = parts[0].trim().parse::<f32>()?;
    let y = parts[1].trim().parse::<f32>()?;
    let z = parts[2].trim().parse::<f32>()?;
    Ok(Vec3::new(x, y, z))
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
           --hover-throttle <0..1>       default 0.2\n\
           --max-thrust-weight <ratio>   default 5.73\n\
           --max-roll-rate <rad/s>       default 14\n\
           --max-pitch-rate <rad/s>      default 12\n\
           --max-yaw-rate <rad/s>        default 10\n\
           --rate-kp <rad/s^2>           default 32\n\
           --rate-damping <gain>         default 8\n\
           --linear-drag <gain>          default 0.05\n\
           --quadratic-drag <gain>       default 0.03\n\
         \n\
         Camera options:\n\
           --camera-distance <meters>    default 7\n\
           --camera-height <meters>      default 2.2\n\
           --camera-spring <rate>        default 8\n\
           --max-trail <steps>           default 2400\n\
         \n\
         Target options:\n\
           --target-pos <x,y,z>           default 4,0,1.6\n\
           --target-yaw <rad>             default 0\n\
           --start-dataset <dir>          imported drone dataset dir for --start-row\n\
           --start-row <row>              initialize pose/velocity/action from dataset row\n\
           --gates <path>                 load gate loop JSON from lewm-drone-import\n\
           --gate-flight <name>           select flight name from gates JSON\n\
           --gate-episode <idx>           select episode_idx from gates JSON\n\
           --gate-laps <n>                default 1\n\
           --gate-radius <meters>         default 0.85\n\
           --gate-order <1,2,...>         explicit 1-based traversal order\n\
         \n\
         LeWM control options:\n\
           --model-dir <dir>              run dir containing final.safetensors, model-config.json, normalization.json\n\
           --weights <path>               overrides --model-dir/final.safetensors\n\
           --config <path>                overrides --model-dir/model-config.json\n\
           --normalization <path>         overrides --model-dir/normalization.json\n\
           --device <cuda[:idx]>          default cuda:0\n\
           --dtype <f32|bf16|f16>         default f32\n\
           --model-hz <hz>                default 100\n\
           --planner-horizon <steps>      future LeWM rollout steps, default 12\n\
           --planner-samples <n>          default 64\n\
           --planner-elites <n>           default 8\n\
           --planner-iterations <n>       default 1\n\
          --planner-every <model steps>  default 5\n\
          --planner-init-std <norm units> default 0.25\n\
          --planner-min-std <norm units>  default 0.005\n\
          --planner-objective <terminal|future-mean|future-min> default future-mean\n\
          --seed <u64>                   deterministic CUDA candidate noise\n\
           --headless-steps <n>           run planner/sim smoke test without Bevy window\n\
           --inspect-gate-targets         print ordered gate entry/pass poses and exit\n\
           --oracle-replay-rows <n>       replay recorded dataset actions through the analytic plant\n\
         \n\
         Controls:\n\
           W/S forward/back tilt, A/D left/right tilt, Q/E yaw left/right, R/F climb/descent\n\
           L toggles LeWM control when loaded\n\
           Z level roll/pitch/yaw, X hover throttle now, P pause/free camera, Backspace reset\n\
           running: mouse wheel or [/] camera distance, 3/4 camera height, 1/2 camera spring\n\
           paused: Bevy FreeCamera controls, WASD/QE move, Shift fast, right mouse or M look"
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
            hover_throttle: 0.2,
            max_thrust_weight: 5.73,
            max_roll_rate: 14.0,
            max_pitch_rate: 12.0,
            max_yaw_rate: 10.0,
            rate_kp: 32.0,
            rate_damping: 8.0,
            linear_drag: 0.05,
            quadratic_drag: 0.03,
        }
    }
}

impl DynamicsConfig {
    fn plant_config(&self) -> DronePlantConfig {
        DronePlantConfig {
            sim_hz: self.sim_hz,
            mass: self.mass,
            gravity: self.gravity,
            hover_throttle: self.hover_throttle,
            max_thrust_weight: self.max_thrust_weight,
            max_roll_rate: self.max_roll_rate,
            max_pitch_rate: self.max_pitch_rate,
            max_yaw_rate: self.max_yaw_rate,
            rate_kp: self.rate_kp,
            rate_damping: self.rate_damping,
            linear_drag: self.linear_drag,
            quadratic_drag: self.quadratic_drag,
            ..DronePlantConfig::default()
        }
    }
}

#[derive(Resource, Debug, Clone)]
struct FollowCameraConfig {
    distance: f32,
    height: f32,
    spring: f32,
}

#[derive(Debug, Clone, Copy)]
struct TargetConfig {
    pos_world: Vec3,
    yaw_rad: f32,
}

impl TargetConfig {
    fn to_pose(self) -> TargetPose {
        TargetPose {
            pos_world: self.pos_world,
            rot_world_from_body: Quat::from_rotation_z(self.yaw_rad),
        }
    }
}

#[derive(Debug, Clone)]
struct StartConfig {
    dataset_dir: PathBuf,
    row: Option<usize>,
}

impl Default for StartConfig {
    fn default() -> Self {
        Self {
            dataset_dir: default_dataset_dir(),
            row: None,
        }
    }
}

impl StartConfig {
    fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            self.row.is_some(),
            "--start-row is required when using --start-dataset"
        );
        ensure!(
            self.dataset_dir.is_dir(),
            "--start-dataset does not exist or is not a directory: {}",
            self.dataset_dir.display()
        );
        Ok(())
    }
}

fn initial_sim_state(args: &Args) -> anyhow::Result<SimState> {
    let Some(start) = args.start.as_ref() else {
        let target = args.target.to_pose();
        return Ok(SimState::new(args.dynamics.clone(), target, args.max_trail));
    };
    let dataset = DroneRacingDataset::open(
        &start.dataset_dir,
        DroneBatchConfig {
            batch_size: 1,
            sequence_steps: 2,
            normalize_observations: false,
            normalize_actions: false,
        },
    )
    .with_context(|| {
        format!(
            "failed to open start dataset {}",
            start.dataset_dir.display()
        )
    })?;
    let row = start.row.context("--start-row is required")?;
    let frame = dataset.frame(row)?;
    let prev_action = if frame.step_idx > 0
        && row > 0
        && dataset.frame(row - 1)?.episode_idx == frame.episode_idx
    {
        dataset.frame(row - 1)?.channels_norm
    } else {
        frame.channels_norm
    };
    let target = args.target.to_pose();
    let pose = DronePose::from_plant(DronePlantState::from_frame(&frame));
    Ok(SimState::from_start_frame(
        args.dynamics.clone(),
        target,
        args.max_trail,
        pose,
        prev_action,
    ))
}

#[derive(Debug, Clone)]
struct GateLoopConfig {
    path: PathBuf,
    flight: Option<String>,
    episode_idx: Option<i64>,
    order: Option<Vec<usize>>,
    desired_laps: usize,
    pass_radius_m: f32,
}

impl Default for GateLoopConfig {
    fn default() -> Self {
        Self {
            path: default_gates_path(),
            flight: None,
            episode_idx: None,
            order: None,
            desired_laps: 1,
            pass_radius_m: 0.85,
        }
    }
}

impl GateLoopConfig {
    fn validate(&self) -> anyhow::Result<()> {
        ensure!(self.desired_laps > 0, "--gate-laps must be positive");
        ensure!(
            self.pass_radius_m.is_finite() && self.pass_radius_m > 0.0,
            "--gate-radius must be positive and finite"
        );
        ensure!(
            self.path.is_file(),
            "--gates path does not exist or is not a file: {}",
            self.path.display()
        );
        if let Some(order) = self.order.as_ref() {
            ensure!(!order.is_empty(), "--gate-order cannot be empty");
            ensure!(
                order.iter().all(|idx| *idx > 0),
                "--gate-order uses 1-based gate indexes"
            );
        }
        Ok(())
    }
}

fn default_dataset_dir() -> PathBuf {
    env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".stable_worldmodel")
        .join("le-wm-nv-data")
        .join("drone-racing-autonomous-100hz-pose12")
}

fn default_gates_path() -> PathBuf {
    default_dataset_dir().join("gates.json")
}

#[derive(Debug, Clone)]
struct LeWmControlConfig {
    model_dir: Option<PathBuf>,
    weights: Option<PathBuf>,
    config: Option<PathBuf>,
    normalization: Option<PathBuf>,
    device: DeviceSpec,
    dtype: DTypeSpec,
    model_hz: f32,
    horizon: usize,
    samples: usize,
    elites: usize,
    iterations: usize,
    plan_every_model_steps: usize,
    init_std: f32,
    min_std: f32,
    objective: PlannerObjective,
    seed: Option<u64>,
}

impl Default for LeWmControlConfig {
    fn default() -> Self {
        Self {
            model_dir: None,
            weights: None,
            config: None,
            normalization: None,
            device: DeviceSpec::default(),
            dtype: DTypeSpec::default(),
            model_hz: 100.0,
            horizon: 12,
            samples: 64,
            elites: 8,
            iterations: 1,
            plan_every_model_steps: 5,
            init_std: 0.25,
            min_std: 0.005,
            objective: PlannerObjective::FutureMean,
            seed: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlannerObjective {
    Terminal,
    FutureMean,
    FutureMin,
}

impl std::str::FromStr for PlannerObjective {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> anyhow::Result<Self> {
        match value {
            "terminal" => Ok(Self::Terminal),
            "future-mean" => Ok(Self::FutureMean),
            "future-min" => Ok(Self::FutureMin),
            other => anyhow::bail!(
                "unsupported --planner-objective `{other}`; expected terminal, future-mean, or future-min"
            ),
        }
    }
}

impl LeWmControlConfig {
    fn validate(&self) -> anyhow::Result<()> {
        let has_model_dir = self.model_dir.is_some();
        let has_explicit_files =
            self.weights.is_some() && self.config.is_some() && self.normalization.is_some();
        ensure!(
            has_model_dir || has_explicit_files,
            "LeWM control requires --model-dir or all of --weights, --config, --normalization"
        );
        ensure!(self.model_hz > 0.0, "--model-hz must be positive");
        ensure!(
            self.horizon > 0,
            "--planner-horizon must be greater than zero"
        );
        ensure!(self.samples >= 2, "--planner-samples must be at least two");
        ensure!(self.elites >= 2, "--planner-elites must be at least two");
        ensure!(
            self.elites <= self.samples,
            "--planner-elites cannot exceed --planner-samples"
        );
        ensure!(
            self.iterations > 0,
            "--planner-iterations must be greater than zero"
        );
        ensure!(
            self.plan_every_model_steps > 0,
            "--planner-every must be greater than zero"
        );
        ensure!(
            self.init_std.is_finite() && self.init_std > 0.0,
            "--planner-init-std must be finite and positive"
        );
        ensure!(
            self.min_std.is_finite() && self.min_std >= 0.0,
            "--planner-min-std must be finite and non-negative"
        );
        Ok(())
    }

    fn weights_path(&self) -> anyhow::Result<PathBuf> {
        self.file_path(self.weights.as_ref(), "final.safetensors", "--weights")
    }

    fn config_path(&self) -> anyhow::Result<PathBuf> {
        self.file_path(self.config.as_ref(), "model-config.json", "--config")
    }

    fn normalization_path(&self) -> anyhow::Result<PathBuf> {
        self.file_path(
            self.normalization.as_ref(),
            "normalization.json",
            "--normalization",
        )
    }

    fn file_path(
        &self,
        explicit: Option<&PathBuf>,
        model_dir_name: &str,
        flag: &str,
    ) -> anyhow::Result<PathBuf> {
        if let Some(path) = explicit {
            return Ok(path.clone());
        }
        let dir = self
            .model_dir
            .as_ref()
            .with_context(|| format!("missing {flag} and --model-dir"))?;
        Ok(dir.join(model_dir_name))
    }
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
    initial_pose: DronePose,
    previous_action: [f32; ACTION_DIM],
    initial_previous_action: [f32; ACTION_DIM],
    time: f32,
    step: usize,
    last_step_ms: f32,
    avg_step_ms: f32,
    trail: Vec<Vec3>,
    max_trail: usize,
    target: TargetPose,
}

impl SimState {
    fn new(dynamics: DynamicsConfig, target: TargetPose, max_trail: usize) -> Self {
        let pose = DronePose::initial();
        let hover_action = [0.0, 0.0, dynamics.hover_throttle, 0.0];
        Self {
            dynamics,
            pose,
            initial_pose: pose,
            previous_action: hover_action,
            initial_previous_action: hover_action,
            time: 0.0,
            step: 0,
            last_step_ms: 0.0,
            avg_step_ms: 0.0,
            trail: vec![pose.pos_world],
            max_trail,
            target,
        }
    }

    fn from_start_frame(
        dynamics: DynamicsConfig,
        target: TargetPose,
        max_trail: usize,
        pose: DronePose,
        previous_action: [f32; ACTION_DIM],
    ) -> Self {
        Self {
            dynamics,
            pose,
            initial_pose: pose,
            previous_action,
            initial_previous_action: previous_action,
            time: 0.0,
            step: 0,
            last_step_ms: 0.0,
            avg_step_ms: 0.0,
            trail: vec![pose.pos_world],
            max_trail,
            target,
        }
    }

    fn reset(&mut self) {
        self.pose = self.initial_pose;
        self.previous_action = self.initial_previous_action;
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

    fn obs12(&self) -> [f32; OBS_DIM] {
        let mut obs = [0.0; OBS_DIM];
        obs[0..3].copy_from_slice(&vec3_array(self.pose.pos_world));
        obs[3..12].copy_from_slice(&rotmat_row_major_array(self.pose.rot_world_from_body));
        obs
    }
}

#[derive(Resource)]
struct GateLoop {
    flight: String,
    gates: Vec<GateTarget>,
    current_index: usize,
    desired_laps: usize,
    laps_completed: usize,
    pass_count: usize,
    pass_radius_m: f32,
    best_dist_m: Vec<f32>,
    finished: bool,
}

impl GateLoop {
    fn load(
        cfg: &GateLoopConfig,
        start: Option<&StartConfig>,
        state: &mut SimState,
    ) -> anyhow::Result<Self> {
        let file: GateSequenceFile = serde_json::from_str(
            &fs::read_to_string(&cfg.path)
                .with_context(|| format!("failed to read {}", cfg.path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", cfg.path.display()))?;
        let flight = file
            .flights
            .iter()
            .find(|flight| {
                cfg.flight
                    .as_ref()
                    .is_none_or(|wanted| flight.flight == *wanted)
                    && cfg
                        .episode_idx
                        .is_none_or(|wanted| flight.episode_idx == wanted)
                    && !flight.gates.is_empty()
            })
            .with_context(|| {
                format!(
                    "no matching non-empty gate flight in {}; use --gate-flight or --gate-episode",
                    cfg.path.display()
                )
            })?;
        let reference_dataset_dir = start
            .map(|start| start.dataset_dir.clone())
            .unwrap_or_else(default_dataset_dir);
        let reference_dataset = DroneRacingDataset::open(
            &reference_dataset_dir,
            DroneBatchConfig {
                batch_size: 1,
                sequence_steps: 2,
                normalize_observations: false,
                normalize_actions: false,
            },
        )
        .with_context(|| {
            format!(
                "failed to open gate reference dataset {}",
                reference_dataset_dir.display()
            )
        })?;
        let reference_rows = reference_dataset.replay_rows_for_episode(flight.episode_idx);
        ensure!(
            !reference_rows.is_empty(),
            "reference dataset has no rows for episode_idx={}",
            flight.episode_idx
        );
        let gate_specs = if let Some(order) = cfg.order.as_ref() {
            order
                .iter()
                .map(|idx| {
                    flight.gates.get(idx - 1).cloned().with_context(|| {
                        format!(
                            "--gate-order index {idx} is out of range for selected flight with {} gates",
                            flight.gates.len()
                        )
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?
        } else {
            flight.gates.clone()
        };
        let start_row = start
            .and_then(|start| start.row)
            .unwrap_or_else(|| reference_rows[0]);
        let gates = GateTarget::from_ordered_specs(
            &gate_specs,
            &reference_dataset,
            &reference_rows,
            start_row,
        )?;
        ensure!(
            !gates.is_empty(),
            "selected flight {} has no gates",
            flight.flight
        );
        let gate_count = gates.len();
        let mut loop_state = Self {
            flight: flight.flight.clone(),
            gates,
            current_index: 0,
            desired_laps: cfg.desired_laps,
            laps_completed: 0,
            pass_count: 0,
            pass_radius_m: cfg.pass_radius_m,
            best_dist_m: vec![f32::INFINITY; gate_count],
            finished: false,
        };
        loop_state.update_state_target(state);
        Ok(loop_state)
    }

    fn reset(&mut self, state: &mut SimState) {
        self.current_index = 0;
        self.laps_completed = 0;
        self.pass_count = 0;
        self.best_dist_m.fill(f32::INFINITY);
        self.finished = false;
        self.update_state_target(state);
    }

    fn update_state_target(&mut self, state: &mut SimState) {
        if self.finished {
            return;
        }
        let active = &self.gates[self.current_index];
        let dist = (active.center - state.pose.pos_world).length();
        self.best_dist_m[self.current_index] = self.best_dist_m[self.current_index].min(dist);
        if dist <= self.pass_radius_m {
            self.advance();
        }
        if !self.finished {
            state.target = self.target_pose_for(state.pose);
        }
    }

    fn advance(&mut self) {
        self.pass_count += 1;
        self.current_index += 1;
        if self.current_index >= self.gates.len() {
            self.current_index = 0;
            self.laps_completed += 1;
            if self.laps_completed >= self.desired_laps {
                self.finished = true;
            }
        }
    }

    fn target_pose_for(&self, pose: DronePose) -> TargetPose {
        let gate = &self.gates[self.current_index];
        if (gate.entry_pos_world - pose.pos_world).length() > (self.pass_radius_m * 0.65).max(0.35)
        {
            gate.entry_pose()
        } else {
            gate.pass_pose()
        }
    }

    fn active_gate(&self) -> Option<&GateTarget> {
        (!self.finished).then(|| &self.gates[self.current_index])
    }

    fn best_distance_text(&self) -> String {
        self.best_dist_m
            .iter()
            .enumerate()
            .map(|(idx, dist)| {
                if dist.is_finite() {
                    format!("{}:{:.2}", idx + 1, dist)
                } else {
                    format!("{}:inf", idx + 1)
                }
            })
            .collect::<Vec<_>>()
            .join(",")
    }
}

#[derive(Clone)]
struct GateTarget {
    name: String,
    center: Vec3,
    display_normal: Vec3,
    visual_radius_m: f32,
    entry_row: usize,
    pass_row: usize,
    entry_pos_world: Vec3,
    entry_rot_world_from_body: Quat,
    pass_pos_world: Vec3,
    pass_rot_world_from_body: Quat,
}

impl GateTarget {
    fn from_ordered_specs(
        specs: &[GateSpec],
        dataset: &DroneRacingDataset,
        reference_rows: &[usize],
        start_row: usize,
    ) -> anyhow::Result<Vec<Self>> {
        let mut search_start = reference_rows
            .iter()
            .position(|row| *row >= start_row)
            .unwrap_or(0);
        let mut gates = Vec::with_capacity(specs.len());
        for spec in specs {
            let gate = Self::from_spec_after(spec, dataset, reference_rows, search_start)?;
            search_start = reference_rows
                .iter()
                .position(|row| *row > gate.pass_row)
                .unwrap_or(reference_rows.len());
            gates.push(gate);
        }
        Ok(gates)
    }

    fn from_spec_after(
        spec: &GateSpec,
        dataset: &DroneRacingDataset,
        reference_rows: &[usize],
        search_start: usize,
    ) -> anyhow::Result<Self> {
        let center = Vec3::from_array(spec.center);
        let pass_idx =
            closest_reference_index_after(dataset, reference_rows, center, search_start)?;
        let pass_row = reference_rows[pass_idx];
        let pass_pose = pose_at_row(dataset, pass_row)?;
        let entry_idx = entry_reference_index(pass_idx, search_start);
        let entry_row = reference_rows[entry_idx];
        let entry_pose = pose_at_row(dataset, entry_row)?;
        let display_normal = horizontal_unit(pass_pose.pos_world - entry_pose.pos_world)
            .or_else(|| {
                trajectory_direction(dataset, reference_rows, entry_idx, pass_idx)
                    .ok()
                    .flatten()
            })
            .unwrap_or(Vec3::X);
        Ok(Self {
            name: spec.name.clone(),
            center,
            display_normal,
            visual_radius_m: spec.half_height.max(0.35),
            entry_row,
            pass_row,
            entry_pos_world: entry_pose.pos_world,
            entry_rot_world_from_body: entry_pose.rot_world_from_body,
            pass_pos_world: pass_pose.pos_world,
            pass_rot_world_from_body: pass_pose.rot_world_from_body,
        })
    }

    fn entry_pose(&self) -> TargetPose {
        TargetPose {
            pos_world: self.entry_pos_world,
            rot_world_from_body: self.entry_rot_world_from_body,
        }
    }

    fn pass_pose(&self) -> TargetPose {
        TargetPose {
            pos_world: self.pass_pos_world,
            rot_world_from_body: self.pass_rot_world_from_body,
        }
    }
}

fn closest_reference_index_after(
    dataset: &DroneRacingDataset,
    rows: &[usize],
    center: Vec3,
    search_start: usize,
) -> anyhow::Result<usize> {
    let start = search_start.min(rows.len().saturating_sub(1));
    let end = (start + GATE_REFERENCE_SEARCH_LIMIT_STEPS).min(rows.len());
    ensure!(start < end, "no reference rows left for gate pose");
    let mut best: Option<(usize, f32)> = None;
    for (idx, &row) in rows[start..end].iter().enumerate() {
        let frame = dataset.frame(row)?;
        let pos = Vec3::from_array(frame.pos_world);
        let dist_sq = pos.distance_squared(center);
        if best.is_none_or(|(_, best_dist)| dist_sq < best_dist) {
            best = Some((start + idx, dist_sq));
        }
    }
    let (idx, _) = best.context("no reference rows for gate pose")?;
    Ok(idx)
}

fn entry_reference_index(pass_idx: usize, min_idx: usize) -> usize {
    let min_idx = min_idx.min(pass_idx);
    pass_idx.saturating_sub(GATE_ENTRY_LEAD_STEPS).max(min_idx)
}

fn pose_at_row(dataset: &DroneRacingDataset, row: usize) -> anyhow::Result<DronePose> {
    let frame = dataset.frame(row)?;
    Ok(DronePose::from_plant(DronePlantState::from_frame(&frame)))
}

fn trajectory_direction(
    dataset: &DroneRacingDataset,
    rows: &[usize],
    start_idx: usize,
    end_idx: usize,
) -> anyhow::Result<Option<Vec3>> {
    if rows.is_empty() {
        return Ok(None);
    }
    let start_idx = start_idx.min(rows.len() - 1);
    let end_idx = end_idx.min(rows.len() - 1);
    if start_idx == end_idx {
        return Ok(None);
    }
    let start = Vec3::from_array(dataset.frame(rows[start_idx])?.pos_world);
    let end = Vec3::from_array(dataset.frame(rows[end_idx])?.pos_world);
    Ok(horizontal_unit(end - start))
}

fn horizontal_unit(value: Vec3) -> Option<Vec3> {
    let horizontal = Vec3::new(value.x, value.y, 0.0);
    let len = horizontal.length();
    (len > 1e-5).then_some(horizontal / len)
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
        Self::from_plant(DronePlantState::initial())
    }

    fn integrate(&mut self, action: [f32; ACTION_DIM], cfg: &DynamicsConfig, dt: f32) {
        let mut state = self.to_plant();
        state.integrate(action, &cfg.plant_config(), dt);
        *self = Self::from_plant(state);
    }

    fn to_plant(self) -> DronePlantState {
        DronePlantState {
            pos_world: vec3_array(self.pos_world),
            vel_world: vec3_array(self.vel_world),
            rotmat_world_from_body: rotmat_row_major_array(self.rot_world_from_body),
            ang_vel_body: vec3_array(self.ang_vel_body),
        }
    }

    fn from_plant(state: DronePlantState) -> Self {
        Self {
            pos_world: vec3_from_array(state.pos_world),
            vel_world: vec3_from_array(state.vel_world),
            rot_world_from_body: quat_from_rotmat_world_from_body(state.rotmat_world_from_body),
            ang_vel_body: vec3_from_array(state.ang_vel_body),
        }
    }
}

#[derive(Clone, Copy)]
struct TargetPose {
    pos_world: Vec3,
    rot_world_from_body: Quat,
}

fn pose_summary(row: Option<usize>, pose: DronePose) -> String {
    attitude_summary(row, pose.pos_world, pose.rot_world_from_body)
}

fn target_pose_summary(row: Option<usize>, pose: TargetPose) -> String {
    attitude_summary(row, pose.pos_world, pose.rot_world_from_body)
}

fn attitude_summary(row: Option<usize>, pos_world: Vec3, rot_world_from_body: Quat) -> String {
    let body_forward = rot_world_from_body * Vec3::X;
    let body_up = rot_world_from_body * Vec3::Z;
    let body_forward_horizontal = Vec3::new(body_forward.x, body_forward.y, 0.0);
    let pitch_deg = body_forward
        .z
        .atan2(body_forward_horizontal.length().max(1e-6))
        .to_degrees();
    let up_z = body_up.z;
    let row_text = row
        .map(|row| format!("row={row} "))
        .unwrap_or_else(|| "".to_string());
    format!(
        "{}pos=[{:.3} {:.3} {:.3}] pitch={:+.1}deg up_z={:+.3} forward=[{:+.3} {:+.3} {:+.3}] up=[{:+.3} {:+.3} {:+.3}]",
        row_text,
        pos_world.x,
        pos_world.y,
        pos_world.z,
        pitch_deg,
        up_z,
        body_forward.x,
        body_forward.y,
        body_forward.z,
        body_up.x,
        body_up.y,
        body_up.z,
    )
}

struct DroneLeWmController {
    model: WorldModel,
    device: CandleDevice,
    dtype: DType,
    obs_mean: Tensor,
    obs_std: Tensor,
    action_mean: [f32; ACTION_DIM],
    action_std: [f32; ACTION_DIM],
    target_emb: Tensor,
    planner: IcemPlanner,
    objective: PlannerObjective,
    history_raw: Vec<[f32; OBS_DIM]>,
    last_action: [f32; ACTION_DIM],
    enabled: bool,
    sim_steps_per_model_step: usize,
    next_observe_step: usize,
    model_step: usize,
    plan_every_model_steps: usize,
    history_actions_raw: Vec<[f32; ACTION_DIM]>,
    last_target_pos: Vec3,
    plan_count: usize,
    last_best_score: f32,
    last_plan_ms: f32,
    last_iterations: usize,
    last_error: Option<String>,
}

impl DroneLeWmController {
    fn load(
        cfg: &LeWmControlConfig,
        dynamics: &DynamicsConfig,
        state: &SimState,
    ) -> anyhow::Result<Self> {
        let weights = cfg.weights_path()?;
        let config = cfg.config_path()?;
        let normalization = cfg.normalization_path()?;
        let model_cfg: WorldModelConfig = serde_json::from_str(
            &fs::read_to_string(&config)
                .with_context(|| format!("failed to read {}", config.display()))?,
        )
        .with_context(|| format!("failed to parse {}", config.display()))?;
        let normalization: DroneNormalization = serde_json::from_str(
            &fs::read_to_string(&normalization)
                .with_context(|| format!("failed to read {}", normalization.display()))?,
        )
        .with_context(|| "failed to parse LeWM normalization")?;
        validate_stats("observation", &normalization.observation, OBS_DIM)?;
        validate_stats("action", &normalization.action, ACTION_DIM)?;
        let action_mean = fixed_action_stats(&normalization.action.mean)?;
        let action_std = fixed_action_stats(&normalization.action.std)?;
        let action_bounds = normalized_action_bounds(&normalization.action)?;

        let device = cfg.device.resolve()?;
        let dtype = cfg.dtype.dtype();
        let vb = checkpoint::var_builder_from_path(&weights, dtype, &device)
            .with_context(|| format!("failed to load weights {}", weights.display()))?;
        let model = WorldModel::new(model_cfg.clone(), vb)?;
        ensure!(
            model_cfg.history_size > 0,
            "model history_size must be greater than zero"
        );
        ensure!(
            model_cfg.action_encoder.input_dim == ACTION_DIM,
            "model action dim {} does not match simulator action dim {ACTION_DIM}",
            model_cfg.action_encoder.input_dim
        );
        match &model_cfg.observation_encoder {
            ObservationEncoderConfig::VectorMlp(vector_cfg) => ensure!(
                vector_cfg.input_dim == OBS_DIM,
                "model observation dim {} does not match simulator obs dim {OBS_DIM}",
                vector_cfg.input_dim
            ),
            ObservationEncoderConfig::ImageVit { .. } => {
                anyhow::bail!("drone simulator requires a vector-observation LeWM model")
            }
        }
        let history_size = model_cfg.history_size;
        let horizon = cfg.horizon;
        let elites = cfg.elites.min(cfg.samples);
        let mut planner_cfg = IcemConfig::new(horizon, cfg.samples, elites, ACTION_DIM);
        planner_cfg.iterations = cfg.iterations;
        planner_cfg.keep_elites = elites;
        planner_cfg.init_std = cfg.init_std;
        planner_cfg.min_std = cfg.min_std;
        planner_cfg.return_mean = false;
        planner_cfg.seed = cfg.seed;
        planner_cfg.action_bounds = action_bounds;

        let obs_mean = Tensor::from_vec(normalization.observation.mean, (OBS_DIM,), &device)?
            .to_dtype(dtype)?
            .reshape((1, 1, OBS_DIM))?;
        let obs_std = Tensor::from_vec(normalization.observation.std, (OBS_DIM,), &device)?
            .to_dtype(dtype)?
            .reshape((1, 1, OBS_DIM))?;

        let sim_steps_per_model_step = ((dynamics.sim_hz / cfg.model_hz).round() as usize).max(1);
        let target_pose = state.target;
        let target_emb =
            encode_target_embedding(&model, &obs_mean, &obs_std, dtype, &device, target_pose)?;
        let mut controller = Self {
            model,
            device,
            dtype,
            obs_mean,
            obs_std,
            action_mean,
            action_std,
            target_emb,
            planner: IcemPlanner::new(planner_cfg),
            objective: cfg.objective,
            history_raw: vec![state.obs12(); history_size],
            last_action: state.previous_action,
            enabled: true,
            sim_steps_per_model_step,
            next_observe_step: sim_steps_per_model_step,
            model_step: 0,
            plan_every_model_steps: cfg.plan_every_model_steps,
            history_actions_raw: vec![state.previous_action; history_size.saturating_sub(1)],
            last_target_pos: state.target.pos_world,
            plan_count: 0,
            last_best_score: f32::NAN,
            last_plan_ms: 0.0,
            last_iterations: 0,
            last_error: None,
        };
        controller.reset_warm_start(state.previous_action)?;
        controller.device.synchronize()?;
        Ok(controller)
    }

    fn reset(&mut self, state: &SimState) -> anyhow::Result<()> {
        let obs = state.obs12();
        for slot in &mut self.history_raw {
            *slot = obs;
        }
        self.next_observe_step = state.step + self.sim_steps_per_model_step;
        self.model_step = 0;
        self.plan_count = 0;
        self.last_best_score = f32::NAN;
        self.last_plan_ms = 0.0;
        self.last_iterations = 0;
        self.last_error = None;
        self.last_action = state.previous_action;
        self.last_target_pos = state.target.pos_world;
        self.history_actions_raw =
            vec![state.previous_action; self.history_raw.len().saturating_sub(1)];
        self.reset_warm_start(state.previous_action)
    }

    fn reset_warm_start(&mut self, action: [f32; ACTION_DIM]) -> anyhow::Result<()> {
        let horizon = self.planner.config().horizon;
        let action = self.normalize_action(action);
        let mut values = Vec::with_capacity(horizon * ACTION_DIM);
        for _ in 0..horizon {
            values.extend_from_slice(&action);
        }
        let sequence = Tensor::from_vec(values, (1, horizon, ACTION_DIM), &self.device)?
            .to_dtype(self.dtype)?;
        self.planner.set_warm_start_sequence(sequence);
        Ok(())
    }

    fn observe_if_due(&mut self, state: &SimState) -> bool {
        if state.step < self.next_observe_step {
            return false;
        }
        while self.next_observe_step <= state.step {
            self.next_observe_step += self.sim_steps_per_model_step;
        }
        self.observe(state.obs12(), state.previous_action);
        self.model_step += 1;
        true
    }

    fn observe(&mut self, obs: [f32; OBS_DIM], action: [f32; ACTION_DIM]) {
        self.history_raw.rotate_left(1);
        if let Some(last) = self.history_raw.last_mut() {
            *last = obs;
        }
        if !self.history_actions_raw.is_empty() {
            self.history_actions_raw.rotate_left(1);
            if let Some(last) = self.history_actions_raw.last_mut() {
                *last = action;
            }
        }
    }

    fn should_plan_now(&self) -> bool {
        self.model_step % self.plan_every_model_steps == 0
    }

    fn plan(&mut self, state: &SimState) -> anyhow::Result<[f32; ACTION_DIM]> {
        self.device.synchronize()?;
        let started = Instant::now();
        let target_pose = state.target;
        let target_emb = encode_target_embedding(
            &self.model,
            &self.obs_mean,
            &self.obs_std,
            self.dtype,
            &self.device,
            target_pose,
        )?;
        let history = self.history_tensor()?;
        let history = history
            .broadcast_sub(&self.obs_mean)?
            .broadcast_div(&self.obs_std)?;
        let history_emb = self.model.encode_vector(&history)?;
        let action_prefix = self.action_history_prefix_tensor()?;
        let scorer = DroneLeWmScorer {
            model: &self.model,
            device: &self.device,
            dtype: self.dtype,
            history_emb: &history_emb,
            target_emb: &target_emb,
            action_prefix: action_prefix.as_ref(),
            objective: self.objective,
        };
        let result = self.planner.plan_device(&scorer)?;
        self.device.synchronize()?;

        let first = result.first_action.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let row = first
            .first()
            .context("LeWM planner produced an empty first action")?;
        ensure!(
            row.len() == ACTION_DIM,
            "LeWM first action dim {} does not match {ACTION_DIM}",
            row.len()
        );
        let mut normalized_action = [0.0f32; ACTION_DIM];
        normalized_action.copy_from_slice(row);
        let action = clamp_action(self.denormalize_action(normalized_action));

        self.last_best_score = best_score(&result.scores).unwrap_or(f32::NAN);
        self.last_plan_ms = started.elapsed().as_secs_f32() * 1000.0;
        self.last_iterations = result.iterations_completed;
        self.plan_count += 1;
        self.last_action = action;
        self.last_target_pos = target_pose.pos_world;
        self.target_emb = target_emb;
        Ok(action)
    }

    fn history_tensor(&self) -> anyhow::Result<Tensor> {
        let mut values = Vec::with_capacity(self.history_raw.len() * OBS_DIM);
        for obs in &self.history_raw {
            values.extend_from_slice(obs);
        }
        Ok(
            Tensor::from_vec(values, (1, self.history_raw.len(), OBS_DIM), &self.device)?
                .to_dtype(self.dtype)?,
        )
    }

    fn action_history_prefix_tensor(&self) -> anyhow::Result<Option<Tensor>> {
        if self.history_raw.len() <= 1 {
            return Ok(None);
        }
        let prefix_len = self.history_raw.len() - 1;
        let mut values = Vec::with_capacity(prefix_len * ACTION_DIM);
        ensure!(
            self.history_actions_raw.len() == prefix_len,
            "LeWM action history length {} does not match expected prefix length {prefix_len}",
            self.history_actions_raw.len()
        );
        for action in &self.history_actions_raw {
            let normalized = self.normalize_action(*action);
            values.extend_from_slice(&normalized);
        }
        Ok(Some(
            Tensor::from_vec(values, (1, 1, prefix_len, ACTION_DIM), &self.device)?
                .to_dtype(self.dtype)?,
        ))
    }

    fn normalize_action(&self, action: [f32; ACTION_DIM]) -> [f32; ACTION_DIM] {
        let mut out = [0.0f32; ACTION_DIM];
        for idx in 0..ACTION_DIM {
            out[idx] = (action[idx] - self.action_mean[idx]) / self.action_std[idx].max(1e-6);
        }
        out
    }

    fn denormalize_action(&self, action: [f32; ACTION_DIM]) -> [f32; ACTION_DIM] {
        let mut out = [0.0f32; ACTION_DIM];
        for idx in 0..ACTION_DIM {
            out[idx] = action[idx] * self.action_std[idx] + self.action_mean[idx];
        }
        out
    }
}

struct DroneLeWmScorer<'a> {
    model: &'a WorldModel,
    device: &'a CandleDevice,
    dtype: DType,
    history_emb: &'a Tensor,
    target_emb: &'a Tensor,
    action_prefix: Option<&'a Tensor>,
    objective: PlannerObjective,
}

impl CandidateScorer for DroneLeWmScorer<'_> {
    fn device(&self) -> &CandleDevice {
        self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn batch_size(&self) -> Option<usize> {
        Some(1)
    }

    fn score_candidates(&self, action_candidates: &Tensor) -> candle::Result<Tensor> {
        let action_candidates = action_candidates
            .to_device(self.device)?
            .to_dtype(self.dtype)?;
        let (_, samples, _, _) = action_candidates.dims4()?;
        let actions = match self.action_prefix {
            Some(prefix) => {
                let prefix = prefix.broadcast_as((1, samples, prefix.dim(2)?, ACTION_DIM))?;
                Tensor::cat(&[&prefix, &action_candidates], 2)?
            }
            None => action_candidates,
        };
        let (_, history, dim) = self.history_emb.dims3()?;
        let emb_init = self
            .history_emb
            .unsqueeze(1)?
            .broadcast_as((1, samples, history, dim))?;
        let rollout = self
            .model
            .rollout_embeddings_with_history(&emb_init, &actions, history)?;
        rollout_cost(
            self.model,
            &rollout,
            self.target_emb,
            history,
            self.objective,
        )
    }
}

fn rollout_cost(
    model: &WorldModel,
    rollout: &Tensor,
    target_emb: &Tensor,
    history_size: usize,
    objective: PlannerObjective,
) -> candle::Result<Tensor> {
    if objective == PlannerObjective::Terminal {
        return model.goal_cost(rollout, target_emb);
    }
    let (batch, samples, time, dim) = rollout.dims4()?;
    if history_size >= time {
        candle::bail!("rollout history_size {history_size} is outside time {time}");
    }
    let future_len = time - history_size;
    let future = rollout.narrow(2, history_size, future_len)?;
    let target = match target_emb.dims() {
        [b, d] if *b == batch && *d == dim => target_emb.clone(),
        [b, t, d] if *b == batch && *d == dim => target_emb.narrow(1, t - 1, 1)?.squeeze(1)?,
        other => candle::bail!("unsupported target embedding shape {other:?}"),
    };
    let target = target
        .unsqueeze(1)?
        .unsqueeze(2)?
        .broadcast_as((batch, samples, future_len, dim))?;
    let step_cost = (future - target)?.sqr()?.sum(D::Minus1)?;
    match objective {
        PlannerObjective::Terminal => unreachable!(),
        PlannerObjective::FutureMean => step_cost.mean(2),
        PlannerObjective::FutureMin => step_cost.min_keepdim(2)?.squeeze(2),
    }
}

fn encode_target_embedding(
    model: &WorldModel,
    obs_mean: &Tensor,
    obs_std: &Tensor,
    dtype: DType,
    device: &CandleDevice,
    target: TargetPose,
) -> anyhow::Result<Tensor> {
    let target_obs = target_obs12(target);
    let target = Tensor::from_vec(target_obs.to_vec(), (1, 1, OBS_DIM), device)?.to_dtype(dtype)?;
    let target = target.broadcast_sub(obs_mean)?.broadcast_div(obs_std)?;
    Ok(model.encode_vector(&target)?)
}

fn target_obs12(target: TargetPose) -> [f32; OBS_DIM] {
    let mut obs = [0.0; OBS_DIM];
    obs[0..3].copy_from_slice(&vec3_array(target.pos_world));
    obs[3..12].copy_from_slice(&rotmat_row_major_array(target.rot_world_from_body));
    obs
}

fn validate_stats(name: &str, stats: &RunningStats, dim: usize) -> anyhow::Result<()> {
    ensure!(
        stats.mean.len() == dim && stats.std.len() == dim,
        "{name} normalization dims mean={} std={} expected {dim}",
        stats.mean.len(),
        stats.std.len()
    );
    for (idx, (&mean, &std)) in stats.mean.iter().zip(stats.std.iter()).enumerate() {
        ensure!(mean.is_finite(), "{name} mean[{idx}] is not finite");
        ensure!(
            std.is_finite() && std > 0.0,
            "{name} std[{idx}] must be positive and finite"
        );
    }
    Ok(())
}

fn fixed_action_stats(values: &[f32]) -> anyhow::Result<[f32; ACTION_DIM]> {
    ensure!(
        values.len() == ACTION_DIM,
        "action stats length {} does not match {ACTION_DIM}",
        values.len()
    );
    let mut out = [0.0f32; ACTION_DIM];
    out.copy_from_slice(values);
    Ok(out)
}

fn normalized_action_bounds(stats: &RunningStats) -> anyhow::Result<ActionBounds> {
    ensure!(
        stats.mean.len() == ACTION_DIM && stats.std.len() == ACTION_DIM,
        "action stats dims mean={} std={} expected {ACTION_DIM}",
        stats.mean.len(),
        stats.std.len()
    );
    let raw_low = [-1.0, -1.0, 0.0, -1.0];
    let raw_high = [1.0, 1.0, 1.0, 1.0];
    let mut low = Vec::with_capacity(ACTION_DIM);
    let mut high = Vec::with_capacity(ACTION_DIM);
    for idx in 0..ACTION_DIM {
        let std = stats.std[idx].max(1e-6);
        low.push((raw_low[idx] - stats.mean[idx]) / std);
        high.push((raw_high[idx] - stats.mean[idx]) / std);
    }
    Ok(ActionBounds { low, high })
}

fn clamp_action(action: [f32; ACTION_DIM]) -> [f32; ACTION_DIM] {
    [
        action[0].clamp(-1.0, 1.0),
        action[1].clamp(-1.0, 1.0),
        action[2].clamp(0.0, 1.0),
        action[3].clamp(-1.0, 1.0),
    ]
}

fn best_score(scores: &Tensor) -> anyhow::Result<f32> {
    let values = scores
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    values
        .into_iter()
        .reduce(f32::min)
        .context("planner produced empty score tensor")
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
struct TelemetryText;

#[derive(Resource)]
struct SceneGuide {
    grid_extent: f32,
    grid_step: f32,
    axis_len: f32,
}

fn configure_gizmos(mut store: ResMut<GizmoConfigStore>) {
    store.config_mut::<DefaultGizmoConfigGroup>().0.line.width = 1.0;
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
        TextFont::from_font_size(13.0),
        TextColor(Color::srgb(0.95, 0.95, 0.85)),
        Node {
            position_type: PositionType::Absolute,
            top: px(10),
            left: px(10),
            max_width: px(820),
            ..default()
        },
        TelemetryText,
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

fn disabled_free_camera_state() -> FreeCameraState {
    let mut state = FreeCameraState::default();
    state.enabled = false;
    state
}

fn update_controls(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut scroll: MessageReader<MouseWheel>,
    mut control: ResMut<SimControl>,
    mut camera: ResMut<FollowCameraConfig>,
    mut state: ResMut<SimState>,
    mut gate_loop: Option<ResMut<GateLoop>>,
    mut controller: Option<NonSendMut<DroneLeWmController>>,
) {
    if keys.just_pressed(KeyCode::KeyP) {
        control.paused = !control.paused;
    }
    if keys.just_pressed(KeyCode::KeyL)
        && let Some(controller) = controller.as_mut()
    {
        controller.enabled = !controller.enabled;
    }
    if keys.just_pressed(KeyCode::Backspace) {
        control.action = control.hover_action;
        control.accumulator = 0.0;
        state.reset();
        if let Some(gate_loop) = gate_loop.as_mut() {
            gate_loop.reset(&mut state);
        }
        if let Some(controller) = controller.as_mut() {
            controller
                .reset(&state)
                .expect("failed to reset LeWM controller");
            control.action = controller.last_action;
        }
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
    if control.paused {
        return;
    }

    let lewm_enabled = controller
        .as_ref()
        .is_some_and(|controller| controller.enabled);
    if !lewm_enabled {
        let target_roll_angle =
            axis(keys.pressed(KeyCode::KeyD), keys.pressed(KeyCode::KeyA)) * KEYBOARD_MAX_TILT_RAD;
        let target_pitch_angle =
            axis(keys.pressed(KeyCode::KeyS), keys.pressed(KeyCode::KeyW)) * KEYBOARD_MAX_TILT_RAD;
        let target_yaw =
            axis(keys.pressed(KeyCode::KeyE), keys.pressed(KeyCode::KeyQ)) * KEYBOARD_YAW_LIMIT;
        let (roll_angle, pitch_angle, body_up_z) = heading_local_roll_pitch_from_pose(state.pose);
        let desired_roll_rate = (target_roll_angle - roll_angle) * KEYBOARD_ATTITUDE_KP;
        let desired_pitch_rate = (target_pitch_angle - pitch_angle) * KEYBOARD_ATTITUDE_KP;
        let target_roll = desired_roll_rate / state.dynamics.max_roll_rate;
        let target_pitch = desired_pitch_rate / state.dynamics.max_pitch_rate;
        control.action[0] = approach(
            control.action[0],
            target_roll.clamp(-1.0, 1.0),
            KEYBOARD_RATE_SLEW * dt,
        );
        control.action[1] = approach(
            control.action[1],
            target_pitch.clamp(-1.0, 1.0),
            KEYBOARD_RATE_SLEW * dt,
        );
        control.action[3] = approach(control.action[3], target_yaw, KEYBOARD_RATE_SLEW * dt);

        let target_climb_rate = axis(keys.pressed(KeyCode::KeyF), keys.pressed(KeyCode::KeyR))
            * KEYBOARD_CLIMB_RATE_MPS;
        let vertical_error = target_climb_rate - state.pose.vel_world.z;
        let target_throttle = control.hover_action[2] + vertical_error * KEYBOARD_VERTICAL_VEL_KP;
        let compensated_throttle = target_throttle / body_up_z.clamp(0.35, 1.0);
        control.action[2] = approach(
            control.action[2],
            compensated_throttle.clamp(0.0, 1.0),
            KEYBOARD_THROTTLE_SLEW * 2.0 * dt,
        );
    }

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

fn update_free_camera_state(
    control: Res<SimControl>,
    mut query: Query<&mut FreeCameraState, With<FollowCamera>>,
) {
    let Ok(mut state) = query.single_mut() else {
        return;
    };
    state.enabled = control.paused;
}

fn step_simulation(
    time: Res<Time>,
    mut control: ResMut<SimControl>,
    mut state: ResMut<SimState>,
    mut gate_loop: Option<ResMut<GateLoop>>,
    mut controller: Option<NonSendMut<DroneLeWmController>>,
) {
    if control.paused {
        return;
    }
    let dt = 1.0 / state.dynamics.sim_hz;
    control.accumulator += time.delta_secs() * state.dynamics.time_scale;
    let mut steps = 0usize;
    while control.accumulator >= dt && steps < state.dynamics.max_frame_steps {
        control.accumulator -= dt;
        steps += 1;
        let mut action = control.action;
        if let Some(controller) = controller.as_ref()
            && controller.enabled
        {
            action = controller.last_action;
        }
        state.step(action, dt);
        if let Some(gate_loop) = gate_loop.as_mut() {
            gate_loop.update_state_target(&mut state);
            if gate_loop.finished {
                control.paused = true;
                break;
            }
        }
        control.action = action;
        if let Some(controller) = controller.as_mut()
            && controller.enabled
            && controller.observe_if_due(&state)
            && controller.should_plan_now()
        {
            control.action = controller
                .plan(&state)
                .expect("LeWM controller planning failed");
        }
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
    mut query: Query<(&mut Transform, &FreeCameraState), With<FollowCamera>>,
) {
    let Ok((mut transform, free_camera)) = query.single_mut() else {
        return;
    };
    if free_camera.enabled {
        return;
    }
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

fn update_telemetry_ui(
    state: Res<SimState>,
    control: Res<SimControl>,
    camera: Res<FollowCameraConfig>,
    gate_loop: Option<Res<GateLoop>>,
    controller: Option<NonSend<DroneLeWmController>>,
    mut query: Query<&mut Text, With<TelemetryText>>,
) {
    let Ok(mut text) = query.single_mut() else {
        return;
    };
    text.0 = telemetry_text(
        &state,
        &control,
        &camera,
        gate_loop.as_deref(),
        controller.as_deref(),
    );
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
    gate_loop: Option<Res<GateLoop>>,
    controller: Option<NonSend<DroneLeWmController>>,
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
    if let Some(gate_loop) = gate_loop.as_ref() {
        draw_gate_loop(&mut gizmos, gate_loop);
    }
    draw_target_pose(&mut gizmos, state.target);
    if let Some(controller) = controller.as_ref() {
        let target = view_vec3(controller.last_target_pos);
        gizmos.sphere(target, 0.13, Color::srgb(1.0, 0.55, 0.05));
        draw_cross(&mut gizmos, target, 0.25, Color::srgb(1.0, 0.55, 0.05));
        gizmos.line(pos, target, Color::srgb(1.0, 0.55, 0.05));
    }

    let vel_view = drone_to_view_vec(pose.vel_world);
    if vel_view.length_squared() > 1e-6 {
        gizmos.arrow(pos, pos + vel_view * 0.25, Color::srgb(0.35, 0.8, 1.0));
    }
}

fn telemetry_text(
    state: &SimState,
    control: &SimControl,
    camera: &FollowCameraConfig,
    gate_loop: Option<&GateLoop>,
    controller: Option<&DroneLeWmController>,
) -> String {
    let pose = state.pose;
    let obs = state.obs12();
    let target_dist = (state.target.pos_world - pose.pos_world).length();
    let status = if control.paused { "paused" } else { "running" };
    let lewm_text = controller
        .map(|controller| {
            let mode = if controller.enabled {
                "lewm:on"
            } else {
                "lewm:off"
            };
            format!(
                "{} plans={} plan_ms={:.2} best={:.3} model_step={} every={} target=[{:.2} {:.2} {:.2}]",
                mode,
                controller.plan_count,
                controller.last_plan_ms,
                controller.last_best_score,
                controller.model_step,
                controller.plan_every_model_steps,
                controller.last_target_pos.x,
                controller.last_target_pos.y,
                controller.last_target_pos.z,
            )
        })
        .unwrap_or_else(|| "lewm:not-loaded".to_string());
    let gate_text = gate_loop
        .map(|gate_loop| {
            let active = gate_loop
                .active_gate()
                .map(|gate| {
                    format!(
                        "{} entry={} pass={}",
                        gate.name, gate.entry_row, gate.pass_row
                    )
                })
                .unwrap_or_else(|| "done".to_string());
            format!(
                " gate={} lap={}/{} passed={} flight={}",
                active,
                gate_loop.laps_completed,
                gate_loop.desired_laps,
                gate_loop.pass_count,
                gate_loop.flight,
            )
        })
        .unwrap_or_default();
    format!(
        "{} t={:.2}s step={} pos=[{:.2} {:.2} {:.2}] dist={:.2}\n\
         a roll={:.2} pitch={:.2} thr={:.2} yaw={:.2} vel=[{:.2} {:.2} {:.2}] rates=[{:.2} {:.2} {:.2}]\n\
         obs12 pos=[{:.2} {:.2} {:.2}] cam dist={:.1} height={:.1} spring={:.1} step_ms={:.4}\n\
         {}{}",
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
        lewm_text,
        gate_text,
    )
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

fn draw_gate_loop(gizmos: &mut Gizmos, gate_loop: &GateLoop) {
    for (idx, gate) in gate_loop.gates.iter().enumerate() {
        let active = !gate_loop.finished && idx == gate_loop.current_index;
        let color = if active {
            Color::srgb(1.0, 0.35, 0.95)
        } else {
            Color::srgba(0.10, 0.85, 1.0, 0.65)
        };
        draw_gate_cylinder(gizmos, gate, gate_loop.pass_radius_m, active, color);
    }
}

fn draw_gate_cylinder(
    gizmos: &mut Gizmos,
    gate: &GateTarget,
    pass_radius_m: f32,
    active: bool,
    color: Color,
) {
    let center = view_vec3(gate.center);
    let normal = drone_to_view_vec(gate.display_normal).normalize_or(Vec3::Z);
    let radius = gate.visual_radius_m.max(pass_radius_m);
    let half_depth = if active { 0.18 } else { 0.10 };
    let front = center + normal * half_depth;
    let back = center - normal * half_depth;
    let rotation = Quat::from_rotation_arc(Vec3::Z, normal);

    gizmos
        .circle(Isometry3d::new(front, rotation), radius, color)
        .resolution(48);
    gizmos
        .circle(Isometry3d::new(back, rotation), radius, color)
        .resolution(48);

    let (right, up) = gate_ring_basis(normal);
    for dir in [right, -right, up, -up] {
        gizmos.line(front + dir * radius, back + dir * radius, color);
    }
    if active {
        gizmos.arrow(center, center + normal * 0.75, color);
        draw_cross(gizmos, center, pass_radius_m * 0.35, color);
        let entry_pos = view_vec3(gate.entry_pos_world);
        let pass_pos = view_vec3(gate.pass_pos_world);
        gizmos.sphere(entry_pos, 0.08, Color::srgb(0.95, 0.95, 0.85));
        draw_cross(gizmos, entry_pos, 0.20, Color::srgb(0.95, 0.95, 0.85));
        gizmos.sphere(pass_pos, 0.08, Color::srgb(0.35, 1.0, 0.40));
        draw_cross(gizmos, pass_pos, 0.20, Color::srgb(0.35, 1.0, 0.40));
        gizmos.line(entry_pos, pass_pos, Color::srgb(0.35, 1.0, 0.40));
    }
}

fn gate_ring_basis(normal: Vec3) -> (Vec3, Vec3) {
    let helper = if normal.z.abs() < 0.9 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let right = helper.cross(normal).normalize_or(Vec3::X);
    let up = normal.cross(right).normalize_or(Vec3::Y);
    (right, up)
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

fn heading_local_roll_pitch_from_pose(pose: DronePose) -> (f32, f32, f32) {
    let body_forward = pose.rot_world_from_body * Vec3::X;
    let yaw = body_forward.y.atan2(body_forward.x);
    let yaw_only = Quat::from_rotation_z(yaw);
    let heading_local_rot = yaw_only.inverse() * pose.rot_world_from_body;
    let body_up = heading_local_rot * Vec3::Z;
    let roll = (-body_up.y).atan2(body_up.z);
    let pitch = body_up.x.atan2(body_up.z);
    let world_body_up_z = (pose.rot_world_from_body * Vec3::Z).z;
    (roll, pitch, world_body_up_z)
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

fn rotmat_row_major_array(rot: Quat) -> [f32; 9] {
    let x = rot * Vec3::X;
    let y = rot * Vec3::Y;
    let z = rot * Vec3::Z;
    [x.x, y.x, z.x, x.y, y.y, z.y, x.z, y.z, z.z]
}

fn vec3_array(value: Vec3) -> [f32; 3] {
    [value.x, value.y, value.z]
}

fn vec3_from_array(value: [f32; 3]) -> Vec3 {
    Vec3::new(value[0], value[1], value[2])
}

fn quat_from_rotmat_world_from_body(m: [f32; 9]) -> Quat {
    Quat::from_mat3(&Mat3::from_cols(
        Vec3::new(m[0], m[3], m[6]),
        Vec3::new(m[1], m[4], m[7]),
        Vec3::new(m[2], m[5], m[8]),
    ))
}

fn view_vec3(value: Vec3) -> Vec3 {
    drone_to_view_vec(value)
}

fn drone_to_view_vec(value: Vec3) -> Vec3 {
    Vec3::new(value.x, value.z, value.y)
}
