use std::{env, path::PathBuf};

use anyhow::{Context, ensure};
use le_wm_nv::{
    data::drone_racing::{DroneBatchConfig, DroneFrame, DroneRacingDataset, mat3_mul_vec3, sub3},
    drone_plant::{DronePlantConfig, DronePlantState, config_summary, rotmat_distance_rad},
};

fn main() -> anyhow::Result<()> {
    let args = Args::parse()?;
    let dataset = DroneRacingDataset::open(
        &args.dataset_dir,
        DroneBatchConfig {
            batch_size: 1,
            sequence_steps: args.window_steps + 1,
            normalize_observations: false,
            normalize_actions: false,
        },
    )?;
    let frames = load_frames(&dataset)?;
    let windows = collect_windows(&frames, args.window_steps, args.stride, args.max_windows)?;
    ensure!(
        !windows.is_empty(),
        "no valid windows found for window_steps={} stride={}",
        args.window_steps,
        args.stride
    );

    println!(
        "dataset={} rows={} windows={} window_steps={} sim_hz={:.1}",
        args.dataset_dir.display(),
        frames.len(),
        windows.len(),
        args.window_steps,
        args.sim_hz
    );
    println!();

    let mut best = DronePlantConfig {
        sim_hz: args.sim_hz,
        ..DronePlantConfig::default()
    };
    let default_score = score_config(&best, &frames, &windows, args.window_steps);
    print_named_score("default", &best, default_score);

    let regression = DronePlantConfig {
        sim_hz: args.sim_hz,
        hover_throttle: 0.2544,
        max_thrust_weight: 5.73,
        max_roll_rate: 12.2572,
        max_pitch_rate: 9.9288,
        max_yaw_rate: 7.8290,
        roll_rate_sign: 1.0,
        pitch_rate_sign: 1.0,
        yaw_rate_sign: -1.0,
        ..DronePlantConfig::default()
    };
    let regression_score = score_config(&regression, &frames, &windows, args.window_steps);
    print_named_score("single-step-regression", &regression, regression_score);

    let mut best_score = default_score;
    if regression_score.total < best_score.total {
        best = regression;
        best_score = regression_score;
    }

    let sign_sets = [
        (-1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, 1.0),
        (1.0, -1.0, 1.0),
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (1.0, -1.0, -1.0),
    ];
    for (roll, pitch, yaw) in sign_sets {
        let mut cfg = best;
        cfg.roll_rate_sign = roll;
        cfg.pitch_rate_sign = pitch;
        cfg.yaw_rate_sign = yaw;
        let score = score_config(&cfg, &frames, &windows, args.window_steps);
        if score.total < best_score.total {
            println!(
                "sign search improved total {:.6} -> {:.6}",
                best_score.total, score.total
            );
            best = cfg;
            best_score = score;
        }
    }

    for pass in 0..args.passes {
        let before = best_score.total;
        best_score = tune_scalar(
            "hover_throttle",
            best_score,
            &mut best,
            &[0.20, 0.22, 0.24, 0.2544, 0.28, 0.305, 0.33, 0.36],
            |cfg, value| cfg.hover_throttle = value,
            &frames,
            &windows,
            args.window_steps,
        );
        best_score = tune_scalar(
            "max_thrust_weight",
            best_score,
            &mut best,
            &[1.4, 1.8, 2.2, 2.8, 3.6, 4.6, 5.73],
            |cfg, value| cfg.max_thrust_weight = value,
            &frames,
            &windows,
            args.window_steps,
        );
        best_score = tune_scalar(
            "max_roll_rate",
            best_score,
            &mut best,
            &[4.0, 6.0, 8.0, 10.0, 12.2572, 14.0, 16.0],
            |cfg, value| cfg.max_roll_rate = value,
            &frames,
            &windows,
            args.window_steps,
        );
        best_score = tune_scalar(
            "max_pitch_rate",
            best_score,
            &mut best,
            &[4.0, 6.0, 8.0, 9.9288, 12.0, 14.0, 16.0],
            |cfg, value| cfg.max_pitch_rate = value,
            &frames,
            &windows,
            args.window_steps,
        );
        best_score = tune_scalar(
            "max_yaw_rate",
            best_score,
            &mut best,
            &[3.0, 5.0, 7.8290, 10.0, 12.0, 14.0, 16.0],
            |cfg, value| cfg.max_yaw_rate = value,
            &frames,
            &windows,
            args.window_steps,
        );
        best_score = tune_scalar(
            "rate_kp",
            best_score,
            &mut best,
            &[6.0, 10.0, 16.0, 22.0, 32.0, 48.0, 64.0],
            |cfg, value| cfg.rate_kp = value,
            &frames,
            &windows,
            args.window_steps,
        );
        best_score = tune_scalar(
            "rate_damping",
            best_score,
            &mut best,
            &[0.25, 0.75, 1.5, 2.5, 4.0, 6.0, 8.0],
            |cfg, value| cfg.rate_damping = value,
            &frames,
            &windows,
            args.window_steps,
        );
        best_score = tune_scalar(
            "linear_drag",
            best_score,
            &mut best,
            &[0.0, 0.05, 0.10, 0.18, 0.30, 0.50, 0.80],
            |cfg, value| cfg.linear_drag = value,
            &frames,
            &windows,
            args.window_steps,
        );
        best_score = tune_scalar(
            "quadratic_drag",
            best_score,
            &mut best,
            &[0.0, 0.01, 0.03, 0.06, 0.10, 0.20],
            |cfg, value| cfg.quadratic_drag = value,
            &frames,
            &windows,
            args.window_steps,
        );
        println!(
            "pass={} total={:.6} delta={:+.6}",
            pass + 1,
            best_score.total,
            best_score.total - before
        );
        if (before - best_score.total).abs() < 1e-5 {
            break;
        }
    }

    println!();
    print_named_score("best", &best, best_score);
    println!();
    println!("simulator flags:");
    println!(
        "  --hover-throttle {:.5} --max-thrust-weight {:.3} --max-roll-rate {:.4} --max-pitch-rate {:.4} --max-yaw-rate {:.4} --rate-kp {:.3} --rate-damping {:.3} --linear-drag {:.4} --quadratic-drag {:.4}",
        best.hover_throttle,
        best.max_thrust_weight,
        best.max_roll_rate,
        best.max_pitch_rate,
        best.max_yaw_rate,
        best.rate_kp,
        best.rate_damping,
        best.linear_drag,
        best.quadratic_drag,
    );
    if best.roll_rate_sign != DronePlantConfig::default().roll_rate_sign
        || best.pitch_rate_sign != DronePlantConfig::default().pitch_rate_sign
        || best.yaw_rate_sign != DronePlantConfig::default().yaw_rate_sign
    {
        println!(
            "  note: best signs are roll={:+.0} pitch={:+.0} yaw={:+.0}; the Bevy CLI keeps sign convention fixed, so apply this only by changing defaults after validating LeWM control.",
            best.roll_rate_sign, best.pitch_rate_sign, best.yaw_rate_sign
        );
    }
    Ok(())
}

#[derive(Debug)]
struct Args {
    dataset_dir: PathBuf,
    window_steps: usize,
    max_windows: usize,
    stride: usize,
    sim_hz: f32,
    passes: usize,
}

impl Args {
    fn parse() -> anyhow::Result<Self> {
        let mut args = Self {
            dataset_dir: default_dataset_dir(),
            window_steps: 30,
            max_windows: 512,
            stride: 25,
            sim_hz: 1000.0,
            passes: 3,
        };
        let mut iter = env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--dataset-dir" => {
                    args.dataset_dir =
                        PathBuf::from(iter.next().context("missing value after --dataset-dir")?);
                }
                "--window-steps" => args.window_steps = next_parse(&mut iter, &arg)?,
                "--max-windows" => args.max_windows = next_parse(&mut iter, &arg)?,
                "--stride" => args.stride = next_parse(&mut iter, &arg)?,
                "--sim-hz" => args.sim_hz = next_parse(&mut iter, &arg)?,
                "--passes" => args.passes = next_parse(&mut iter, &arg)?,
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                other => anyhow::bail!("unknown argument `{other}`; use --help"),
            }
        }
        ensure!(args.window_steps > 0, "--window-steps must be positive");
        ensure!(args.max_windows > 0, "--max-windows must be positive");
        ensure!(args.stride > 0, "--stride must be positive");
        ensure!(args.sim_hz > 0.0, "--sim-hz must be positive");
        Ok(args)
    }
}

fn default_dataset_dir() -> PathBuf {
    env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".stable_worldmodel")
        .join("le-wm-nv-data")
        .join("drone-racing-autonomous-100hz-pose16")
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
        "Usage: lewm-drone-fit-plant [--dataset-dir <dir>] [--window-steps <n>] [--max-windows <n>] [--stride <n>] [--sim-hz <hz>] [--passes <n>]\n\
         \n\
         Fits the simple Bevy drone plant by replaying real dataset control windows\n\
         from real initial poses and minimizing short-horizon pose trajectory error.\n\
         This does not train or modify LeWM."
    );
}

#[derive(Debug, Clone, Copy)]
struct Window {
    start: usize,
}

fn load_frames(dataset: &DroneRacingDataset) -> anyhow::Result<Vec<DroneFrame>> {
    let mut frames = Vec::with_capacity(dataset.metadata().rows);
    for row in 0..dataset.metadata().rows {
        frames.push(dataset.frame(row)?);
    }
    Ok(frames)
}

fn collect_windows(
    frames: &[DroneFrame],
    window_steps: usize,
    stride: usize,
    max_windows: usize,
) -> anyhow::Result<Vec<Window>> {
    let mut candidates = Vec::new();
    let mut row = 0usize;
    while row + window_steps < frames.len() {
        if valid_window(frames, row, window_steps) {
            candidates.push(Window { start: row });
            row += stride;
        } else {
            row += 1;
        }
    }
    ensure!(
        !candidates.is_empty(),
        "no valid windows found before downsampling"
    );
    if candidates.len() <= max_windows {
        return Ok(candidates);
    }
    let mut windows = Vec::with_capacity(max_windows);
    for idx in 0..max_windows {
        let src = idx * candidates.len() / max_windows;
        windows.push(candidates[src]);
    }
    Ok(windows)
}

fn valid_window(frames: &[DroneFrame], start: usize, window_steps: usize) -> bool {
    for offset in 0..window_steps {
        let frame = &frames[start + offset];
        let next = &frames[start + offset + 1];
        if frame.episode_idx != next.episode_idx {
            return false;
        }
        if next.step_idx != frame.step_idx + 1 {
            return false;
        }
        let dt = frame.dt.max(next.dt);
        if !dt.is_finite() || dt <= 1e-5 || dt > 0.05 {
            return false;
        }
    }
    true
}

#[derive(Debug, Clone, Copy)]
struct PlantScore {
    total: f32,
    pos_rmse: f32,
    rot_rmse_rad: f32,
    vel_rmse: f32,
    rate_rmse: f32,
    final_pos_rmse: f32,
    samples: usize,
}

fn score_config(
    cfg: &DronePlantConfig,
    frames: &[DroneFrame],
    windows: &[Window],
    window_steps: usize,
) -> PlantScore {
    let mut pos_sq = 0.0f64;
    let mut rot_sq = 0.0f64;
    let mut vel_sq = 0.0f64;
    let mut rate_sq = 0.0f64;
    let mut final_pos_sq = 0.0f64;
    let mut samples = 0usize;

    for window in windows {
        let mut state = DronePlantState::from_frame(&frames[window.start]);
        for offset in 0..window_steps {
            let frame = &frames[window.start + offset];
            let next = &frames[window.start + offset + 1];
            let dt = frame.dt.max(next.dt);
            let substeps = ((dt * cfg.sim_hz).round() as usize).max(1);
            let sub_dt = dt / substeps as f32;
            for _ in 0..substeps {
                state.integrate(frame.channels_norm, cfg, sub_dt);
            }
            let pos_err = sub3(state.pos_world, next.pos_world);
            pos_sq += dot3_local(pos_err, pos_err) as f64;
            let rot_err =
                rotmat_distance_rad(state.rotmat_world_from_body, next.rotmat_world_from_body);
            rot_sq += (rot_err * rot_err) as f64;
            let target_vel = mat3_mul_vec3(next.rotmat_world_from_body, next.lin_vel_body);
            let vel_err = sub3(state.vel_world, target_vel);
            vel_sq += dot3_local(vel_err, vel_err) as f64;
            let rate_err = sub3(state.ang_vel_body, next.ang_vel_body);
            rate_sq += dot3_local(rate_err, rate_err) as f64;
            samples += 1;
        }
        let final_frame = &frames[window.start + window_steps];
        let final_pos_err = sub3(state.pos_world, final_frame.pos_world);
        final_pos_sq += dot3_local(final_pos_err, final_pos_err) as f64;
    }

    let denom = samples.max(1) as f64;
    let windows_denom = windows.len().max(1) as f64;
    let pos_rmse = (pos_sq / denom).sqrt() as f32;
    let rot_rmse_rad = (rot_sq / denom).sqrt() as f32;
    let vel_rmse = (vel_sq / denom).sqrt() as f32;
    let rate_rmse = (rate_sq / denom).sqrt() as f32;
    let final_pos_rmse = (final_pos_sq / windows_denom).sqrt() as f32;
    let total = pos_rmse
        + 0.75 * final_pos_rmse
        + 0.35 * rot_rmse_rad
        + 0.025 * vel_rmse
        + 0.01 * rate_rmse;
    PlantScore {
        total,
        pos_rmse,
        rot_rmse_rad,
        vel_rmse,
        rate_rmse,
        final_pos_rmse,
        samples,
    }
}

fn tune_scalar(
    name: &str,
    current_score: PlantScore,
    best: &mut DronePlantConfig,
    values: &[f32],
    mut set: impl FnMut(&mut DronePlantConfig, f32),
    frames: &[DroneFrame],
    windows: &[Window],
    window_steps: usize,
) -> PlantScore {
    let mut best_cfg = *best;
    let mut best_score = current_score;
    for value in values {
        let mut cfg = *best;
        set(&mut cfg, *value);
        let score = score_config(&cfg, frames, windows, window_steps);
        if score.total < best_score.total {
            best_score = score;
            best_cfg = cfg;
        }
    }
    if best_score.total < current_score.total {
        println!(
            "  {name} improved total {:.6} -> {:.6}",
            current_score.total, best_score.total
        );
        *best = best_cfg;
    }
    best_score
}

fn print_named_score(name: &str, cfg: &DronePlantConfig, score: PlantScore) {
    println!("{name}: {}", config_summary(cfg));
    println!(
        "  total={:.6} pos_rmse={:.4}m final_pos_rmse={:.4}m rot_rmse={:.4}rad vel_rmse={:.4}m/s rate_rmse={:.4}rad/s samples={}",
        score.total,
        score.pos_rmse,
        score.final_pos_rmse,
        score.rot_rmse_rad,
        score.vel_rmse,
        score.rate_rmse,
        score.samples,
    );
}

fn dot3_local(lhs: [f32; 3], rhs: [f32; 3]) -> f32 {
    lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
}
