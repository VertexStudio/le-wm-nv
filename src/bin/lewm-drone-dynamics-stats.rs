use std::{env, path::PathBuf};

use anyhow::{Context, ensure};
use le_wm_nv::data::drone_racing::{
    DroneBatchConfig, DroneFrame, DroneRacingDataset, add3, dot3, mat3_mul_vec3, scale3, sub3,
};

const CHANNELS: [&str; 4] = ["roll", "pitch", "thrust", "yaw"];
const AXES: [&str; 3] = ["x", "y", "z"];

fn main() -> anyhow::Result<()> {
    let args = Args::parse()?;
    let dataset = DroneRacingDataset::open(
        &args.dataset_dir,
        DroneBatchConfig {
            batch_size: 1,
            sequence_steps: 2,
            normalize_observations: false,
            normalize_actions: false,
        },
    )?;
    let report = analyze(&dataset, args.gravity)?;
    report.print(args.gravity);
    Ok(())
}

#[derive(Debug)]
struct Args {
    dataset_dir: PathBuf,
    gravity: f32,
}

impl Args {
    fn parse() -> anyhow::Result<Self> {
        let mut dataset_dir = default_dataset_dir();
        let mut gravity = 9.81;
        let mut iter = env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--dataset-dir" => {
                    dataset_dir =
                        PathBuf::from(iter.next().context("missing value after --dataset-dir")?);
                }
                "--gravity" => {
                    let value = iter.next().context("missing value after --gravity")?;
                    gravity = value
                        .parse::<f32>()
                        .with_context(|| format!("invalid --gravity value `{value}`"))?;
                }
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                other => anyhow::bail!("unknown argument `{other}`; use --help"),
            }
        }
        ensure!(gravity > 0.0, "--gravity must be positive");
        Ok(Self {
            dataset_dir,
            gravity,
        })
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

fn print_help() {
    println!(
        "Usage: lewm-drone-dynamics-stats [--dataset-dir <dir>] [--gravity <m/s^2>]\n\
         \n\
         Reads the imported drone HDF5 artifact and reports action/rate/thrust\n\
         statistics useful for calibrating the Bevy simulator plant."
    );
}

#[derive(Debug)]
struct DynamicsReport {
    rows: usize,
    samples: usize,
    action_stats: [Moments; 4],
    angular_rate_stats: [Moments; 3],
    action_to_angular_rate: [[Bivariate; 3]; 4],
    action_to_angular_accel: [[Bivariate; 3]; 4],
    action_to_body_delta: [[Bivariate; 3]; 4],
    throttle_to_specific_thrust: Bivariate,
    hover_throttle_candidates: Moments,
    dt_stats: Moments,
}

impl DynamicsReport {
    fn print(&self, gravity: f32) {
        println!("rows={} consecutive_samples={}", self.rows, self.samples);
        println!(
            "dt mean={:.6}s std={:.6}s min={:.6}s max={:.6}s",
            self.dt_stats.mean(),
            self.dt_stats.std(),
            self.dt_stats.min,
            self.dt_stats.max,
        );
        println!();
        println!("action stats:");
        for (idx, name) in CHANNELS.iter().enumerate() {
            let stats = self.action_stats[idx];
            println!(
                "  {:>6}: mean={:+.5} std={:.5} min={:+.5} max={:+.5}",
                name,
                stats.mean(),
                stats.std(),
                stats.min,
                stats.max,
            );
        }
        println!();
        println!("body angular-rate stats [rad/s]:");
        for (idx, axis) in AXES.iter().enumerate() {
            let stats = self.angular_rate_stats[idx];
            println!(
                "  {axis}: mean={:+.5} std={:.5} min={:+.5} max={:+.5}",
                stats.mean(),
                stats.std(),
                stats.min,
                stats.max,
            );
        }
        println!();
        println!("action -> body angular-rate regression: rate_axis = intercept + slope * channel");
        print_matrix(&self.action_to_angular_rate);
        println!();
        println!(
            "action -> body angular-accel regression: accel_axis = intercept + slope * channel"
        );
        print_matrix(&self.action_to_angular_accel);
        println!();
        println!(
            "action -> next body-frame velocity regression from position delta/dt: vel_axis = intercept + slope * channel"
        );
        print_matrix(&self.action_to_body_delta);
        println!();

        let thrust = self.throttle_to_specific_thrust;
        let hover = if thrust.slope().abs() > 1e-6 {
            (gravity - thrust.intercept()) / thrust.slope()
        } else {
            f32::NAN
        };
        let max_thrust_weight = (thrust.intercept() + thrust.slope()).max(0.0) / gravity;
        println!(
            "throttle -> specific thrust: intercept={:+.5} slope={:+.5} corr={:+.4}",
            thrust.intercept(),
            thrust.slope(),
            thrust.corr(),
        );
        println!(
            "  linear hover_throttle_for_{gravity:.2}m/s^2={:.5} implied_max_thrust_weight={:.3}",
            hover, max_thrust_weight,
        );
        println!(
            "  near-level/low-vertical-accel throttle: n={} mean={:.5} std={:.5} min={:.5} max={:.5}",
            self.hover_throttle_candidates.count,
            self.hover_throttle_candidates.mean(),
            self.hover_throttle_candidates.std(),
            self.hover_throttle_candidates.min,
            self.hover_throttle_candidates.max,
        );
    }
}

fn print_matrix(matrix: &[[Bivariate; 3]; 4]) {
    print!("          ");
    for axis in AXES {
        print!(" {:>28}", axis);
    }
    println!();
    for (channel_idx, channel) in CHANNELS.iter().enumerate() {
        print!("  {:>6}: ", channel);
        for axis_idx in 0..3 {
            let v = matrix[channel_idx][axis_idx];
            print!(" slope={:+8.4} corr={:+6.3}", v.slope(), v.corr(),);
        }
        println!();
    }
}

fn analyze(dataset: &DroneRacingDataset, gravity: f32) -> anyhow::Result<DynamicsReport> {
    let rows = dataset.metadata().rows;
    let mut report = DynamicsReport {
        rows,
        samples: 0,
        action_stats: [Moments::new(); 4],
        angular_rate_stats: [Moments::new(); 3],
        action_to_angular_rate: [[Bivariate::new(); 3]; 4],
        action_to_angular_accel: [[Bivariate::new(); 3]; 4],
        action_to_body_delta: [[Bivariate::new(); 3]; 4],
        throttle_to_specific_thrust: Bivariate::new(),
        hover_throttle_candidates: Moments::new(),
        dt_stats: Moments::new(),
    };

    for row in 0..rows.saturating_sub(1) {
        let frame = dataset.frame(row)?;
        let next = dataset.frame(row + 1)?;
        if frame.episode_idx != next.episode_idx {
            continue;
        }
        let dt = frame.dt.max(next.dt);
        if !dt.is_finite() || dt <= 1e-5 {
            continue;
        }
        report.samples += 1;
        report.dt_stats.push(dt);

        let body_delta = scale3(
            mat3_t_mul_vec3_local(
                frame.rotmat_world_from_body,
                sub3(next.pos_world, frame.pos_world),
            ),
            1.0 / dt,
        );
        let angular_accel = [
            (next.ang_vel_body[0] - frame.ang_vel_body[0]) / dt,
            (next.ang_vel_body[1] - frame.ang_vel_body[1]) / dt,
            (next.ang_vel_body[2] - frame.ang_vel_body[2]) / dt,
        ];
        let world_accel = world_accel_from_body_velocity(&frame, &next, dt);
        let body_up = mat3_mul_vec3(frame.rotmat_world_from_body, [0.0, 0.0, 1.0]);
        let specific_thrust = dot3(add3(world_accel, [0.0, 0.0, gravity]), body_up);

        for channel_idx in 0..4 {
            let action = frame.channels_norm[channel_idx];
            report.action_stats[channel_idx].push(action);
            for axis_idx in 0..3 {
                report.action_to_angular_rate[channel_idx][axis_idx]
                    .push(action, frame.ang_vel_body[axis_idx]);
                report.action_to_angular_accel[channel_idx][axis_idx]
                    .push(action, angular_accel[axis_idx]);
                report.action_to_body_delta[channel_idx][axis_idx]
                    .push(action, body_delta[axis_idx]);
            }
        }
        for axis_idx in 0..3 {
            report.angular_rate_stats[axis_idx].push(frame.ang_vel_body[axis_idx]);
        }
        report
            .throttle_to_specific_thrust
            .push(frame.channels_norm[2], specific_thrust);

        if body_up[2] > 0.85 && frame.lin_vel_body[2].abs() < 0.75 && world_accel[2].abs() < 3.0 {
            report
                .hover_throttle_candidates
                .push(frame.channels_norm[2]);
        }
    }

    ensure!(
        report.samples > 0,
        "dataset produced zero consecutive samples"
    );
    Ok(report)
}

fn mat3_t_mul_vec3_local(m: [f32; 9], v: [f32; 3]) -> [f32; 3] {
    [
        m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
        m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
        m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
    ]
}

fn world_accel_from_body_velocity(frame: &DroneFrame, next: &DroneFrame, dt: f32) -> [f32; 3] {
    let vel = mat3_mul_vec3(frame.rotmat_world_from_body, frame.lin_vel_body);
    let next_vel = mat3_mul_vec3(next.rotmat_world_from_body, next.lin_vel_body);
    scale3(sub3(next_vel, vel), 1.0 / dt)
}

#[derive(Debug, Clone, Copy)]
struct Moments {
    count: usize,
    sum: f64,
    sum_sq: f64,
    min: f32,
    max: f32,
}

impl Moments {
    const fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
        }
    }

    fn push(&mut self, value: f32) {
        if !value.is_finite() {
            return;
        }
        self.count += 1;
        self.sum += value as f64;
        self.sum_sq += (value as f64) * (value as f64);
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    fn mean(self) -> f32 {
        if self.count == 0 {
            return f32::NAN;
        }
        (self.sum / self.count as f64) as f32
    }

    fn std(self) -> f32 {
        if self.count == 0 {
            return f32::NAN;
        }
        let mean = self.sum / self.count as f64;
        let var = (self.sum_sq / self.count as f64 - mean * mean).max(0.0);
        var.sqrt() as f32
    }
}

#[derive(Debug, Clone, Copy)]
struct Bivariate {
    count: usize,
    x: Moments,
    y: Moments,
    sum_xy: f64,
}

impl Bivariate {
    const fn new() -> Self {
        Self {
            count: 0,
            x: Moments::new(),
            y: Moments::new(),
            sum_xy: 0.0,
        }
    }

    fn push(&mut self, x: f32, y: f32) {
        if !x.is_finite() || !y.is_finite() {
            return;
        }
        self.count += 1;
        self.x.push(x);
        self.y.push(y);
        self.sum_xy += (x as f64) * (y as f64);
    }

    fn slope(self) -> f32 {
        let var_x = self.var_x();
        if var_x <= 1e-12 {
            return f32::NAN;
        }
        (self.cov_xy() / var_x) as f32
    }

    fn intercept(self) -> f32 {
        self.y.mean() - self.slope() * self.x.mean()
    }

    fn corr(self) -> f32 {
        let denom = (self.var_x() * self.var_y()).sqrt();
        if denom <= 1e-12 {
            return f32::NAN;
        }
        (self.cov_xy() / denom) as f32
    }

    fn cov_xy(self) -> f64 {
        if self.count == 0 {
            return f64::NAN;
        }
        self.sum_xy / self.count as f64
            - (self.x.sum / self.count as f64) * (self.y.sum / self.count as f64)
    }

    fn var_x(self) -> f64 {
        variance(self.x)
    }

    fn var_y(self) -> f64 {
        variance(self.y)
    }
}

fn variance(values: Moments) -> f64 {
    if values.count == 0 {
        return f64::NAN;
    }
    let mean = values.sum / values.count as f64;
    (values.sum_sq / values.count as f64 - mean * mean).max(0.0)
}
