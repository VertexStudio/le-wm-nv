use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use anyhow::{Context, ensure};
use clap::Parser;
use csv::StringRecord;
use le_wm_nv::data::drone_racing::{
    FlightGates, GateSpec, ImportedDroneData, add3, cross3, mat3_t_mul_vec3, mat3_transpose,
    normalize_channels, normalize3, scale3, sub3,
};
use walkdir::WalkDir;

#[derive(Parser, Debug)]
struct Args {
    /// Extracted drone-racing-dataset flight root, e.g. ~/.stable_worldmodel/drone-racing-dataset/data/autonomous.
    #[arg(long)]
    input_dir: Option<PathBuf>,

    /// Output artifact directory.
    #[arg(long)]
    output_dir: Option<PathBuf>,

    /// Source CSV rate in Hz.
    #[arg(long, default_value_t = 500)]
    source_rate: usize,

    /// Imported artifact sample rate in Hz.
    #[arg(long, default_value_t = 100)]
    sample_rate: usize,

    /// Fraction of flights reserved for held-out evaluation.
    #[arg(long, default_value_t = 0.2)]
    eval_fraction: f32,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let input_dir = args.input_dir.clone().unwrap_or_else(default_input_dir);
    let output_dir = args.output_dir.clone().unwrap_or_else(default_output_dir);
    let stride = args.source_rate / args.sample_rate;
    let files = find_sync_csvs(&input_dir)?;
    ensure!(
        !files.is_empty(),
        "no *_500hz_freq_sync.csv files found under {}",
        input_dir.display()
    );

    println!(
        "importing files={} source_rate={}Hz sample_rate={}Hz stride={} output_dir={}",
        files.len(),
        args.source_rate,
        args.sample_rate,
        stride,
        output_dir.display()
    );
    let mut imported = ImportedDroneData::default();
    imported.source_files = files.clone();
    for (episode, path) in files.iter().enumerate() {
        let rows_before = imported.rows();
        import_flight(
            path,
            episode as i64,
            stride,
            args.sample_rate,
            &mut imported,
        )
        .with_context(|| format!("failed to import {}", path.display()))?;
        println!(
            "flight={} episode={} imported_rows={}",
            path.display(),
            episode,
            imported.rows() - rows_before
        );
    }

    let metadata = imported.write_artifact(
        &output_dir,
        &input_dir,
        args.sample_rate,
        args.source_rate,
        args.eval_fraction,
    )?;
    println!(
        "wrote rows={} episodes={} train_episodes={} eval_episodes={} data={}",
        metadata.rows,
        metadata.episodes,
        metadata.train_episodes.len(),
        metadata.eval_episodes.len(),
        output_dir.join("data.h5").display()
    );
    Ok(())
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    ensure!(
        args.source_rate > 0,
        "--source-rate must be greater than zero"
    );
    ensure!(
        args.sample_rate > 0,
        "--sample-rate must be greater than zero"
    );
    ensure!(
        args.source_rate >= args.sample_rate,
        "--source-rate must be >= --sample-rate"
    );
    ensure!(
        args.source_rate % args.sample_rate == 0,
        "--source-rate must be divisible by --sample-rate"
    );
    ensure!(
        args.eval_fraction.is_finite() && args.eval_fraction > 0.0 && args.eval_fraction < 1.0,
        "--eval-fraction must be finite and in (0, 1)"
    );
    Ok(())
}

fn default_input_dir() -> PathBuf {
    home_dir()
        .join(".stable_worldmodel")
        .join("drone-racing-dataset")
        .join("data")
        .join("autonomous")
}

fn default_output_dir() -> PathBuf {
    home_dir()
        .join(".stable_worldmodel")
        .join("le-wm-nv-data")
        .join("drone-racing-autonomous-100hz-pose16")
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn find_sync_csvs(root: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in WalkDir::new(root).follow_links(true) {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy();
        if name.ends_with("_500hz_freq_sync.csv") {
            files.push(entry.path().to_path_buf());
        }
    }
    files.sort();
    Ok(files)
}

fn import_flight(
    path: &Path,
    episode: i64,
    stride: usize,
    sample_rate: usize,
    output: &mut ImportedDroneData,
) -> anyhow::Result<()> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open CSV {}", path.display()))?;
    let headers = reader.headers()?.clone();
    let columns = HeaderIndex::new(&headers);
    let mut kept_step = 0i64;
    let mut last_elapsed: Option<f32> = None;
    let mut first_record = None;
    for (raw_idx, row) in reader.records().enumerate() {
        let row = row?;
        if raw_idx % stride != 0 {
            continue;
        }
        if first_record.is_none() {
            first_record = Some(row.clone());
        }
        let Some(sample) = FlightSample::from_record(&columns, &row)? else {
            continue;
        };
        let dt = last_elapsed
            .map(|prev| sample.elapsed_time - prev)
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(1.0 / sample_rate as f32);
        last_elapsed = Some(sample.elapsed_time);
        output.pos_world.extend_from_slice(&sample.pos_world);
        output
            .rotmat_world_from_body
            .extend_from_slice(&sample.rotmat_world_from_body);
        output.lin_vel_body.extend_from_slice(&sample.lin_vel_body);
        output.ang_vel_body.extend_from_slice(&sample.ang_vel_body);
        output.vbat.push(sample.vbat);
        output.channels_raw.extend_from_slice(&sample.channels_raw);
        output
            .channels_norm
            .extend_from_slice(&sample.channels_norm);
        output.accel_body.extend_from_slice(&sample.accel_body);
        output.gyro_body.extend_from_slice(&sample.gyro_body);
        output.episode_idx.push(episode);
        output.step_idx.push(kept_step);
        output.elapsed_time.push(sample.elapsed_time);
        output.dt.push(dt);
        kept_step += 1;
    }
    ensure!(
        kept_step > 1,
        "flight {} produced fewer than two rows",
        path.display()
    );
    if let Some(record) = first_record.as_ref() {
        let gates = extract_gates(path, episode, &columns, record)?;
        output.gates.push(gates);
    }
    Ok(())
}

struct FlightSample {
    elapsed_time: f32,
    pos_world: [f32; 3],
    rotmat_world_from_body: [f32; 9],
    lin_vel_body: [f32; 3],
    ang_vel_body: [f32; 3],
    vbat: f32,
    channels_raw: [f32; 4],
    channels_norm: [f32; 4],
    accel_body: [f32; 3],
    gyro_body: [f32; 3],
}

impl FlightSample {
    fn from_record(columns: &HeaderIndex, row: &StringRecord) -> anyhow::Result<Option<Self>> {
        let elapsed_time = columns.f32(row, "elapsed_time")?;
        let pos_world = [
            columns.f32(row, "drone_x")?,
            columns.f32(row, "drone_y")?,
            columns.f32(row, "drone_z")?,
        ];
        let csv_rot = [
            columns.f32(row, "drone_rot[0]")?,
            columns.f32(row, "drone_rot[1]")?,
            columns.f32(row, "drone_rot[2]")?,
            columns.f32(row, "drone_rot[3]")?,
            columns.f32(row, "drone_rot[4]")?,
            columns.f32(row, "drone_rot[5]")?,
            columns.f32(row, "drone_rot[6]")?,
            columns.f32(row, "drone_rot[7]")?,
            columns.f32(row, "drone_rot[8]")?,
        ];
        let rotmat_world_from_body = mat3_transpose(csv_rot);
        let lin_vel_world = [
            columns.f32(row, "drone_velocity_linear_x")?,
            columns.f32(row, "drone_velocity_linear_y")?,
            columns.f32(row, "drone_velocity_linear_z")?,
        ];
        let ang_vel_world = [
            columns.f32(row, "drone_velocity_angular_x")?,
            columns.f32(row, "drone_velocity_angular_y")?,
            columns.f32(row, "drone_velocity_angular_z")?,
        ];
        let lin_vel_body = mat3_t_mul_vec3(rotmat_world_from_body, lin_vel_world);
        let ang_vel_body = mat3_t_mul_vec3(rotmat_world_from_body, ang_vel_world);
        let channels_raw = [
            columns.f32(row, "channels_roll")?,
            columns.f32(row, "channels_pitch")?,
            columns.f32(row, "channels_thrust")?,
            columns.f32(row, "channels_yaw")?,
        ];
        let channels_norm = normalize_channels(channels_raw);
        let sample = Self {
            elapsed_time,
            pos_world,
            rotmat_world_from_body,
            lin_vel_body,
            ang_vel_body,
            vbat: columns.f32(row, "vbat")?,
            channels_raw,
            channels_norm,
            accel_body: [
                columns.f32(row, "accel_x")?,
                columns.f32(row, "accel_y")?,
                columns.f32(row, "accel_z")?,
            ],
            gyro_body: [
                columns.f32(row, "gyro_x")?,
                columns.f32(row, "gyro_y")?,
                columns.f32(row, "gyro_z")?,
            ],
        };
        if sample.is_finite() {
            Ok(Some(sample))
        } else {
            Ok(None)
        }
    }

    fn is_finite(&self) -> bool {
        self.elapsed_time.is_finite()
            && self.pos_world.iter().all(|v| v.is_finite())
            && self.rotmat_world_from_body.iter().all(|v| v.is_finite())
            && self.lin_vel_body.iter().all(|v| v.is_finite())
            && self.ang_vel_body.iter().all(|v| v.is_finite())
            && self.vbat.is_finite()
            && self.channels_raw.iter().all(|v| v.is_finite())
            && self.channels_norm.iter().all(|v| v.is_finite())
            && self.accel_body.iter().all(|v| v.is_finite())
            && self.gyro_body.iter().all(|v| v.is_finite())
    }
}

fn extract_gates(
    path: &Path,
    episode: i64,
    columns: &HeaderIndex,
    row: &StringRecord,
) -> anyhow::Result<FlightGates> {
    let mut gates = Vec::new();
    for gate_idx in 1..=8 {
        let names = (1..=4)
            .map(|marker| {
                (
                    format!("gate{gate_idx}_marker{marker}_x"),
                    format!("gate{gate_idx}_marker{marker}_y"),
                    format!("gate{gate_idx}_marker{marker}_z"),
                )
            })
            .collect::<Vec<_>>();
        if !names
            .iter()
            .all(|(x, y, z)| columns.contains(x) && columns.contains(y) && columns.contains(z))
        {
            continue;
        }
        let mut marker = [[0f32; 3]; 4];
        for (idx, (x, y, z)) in names.iter().enumerate() {
            marker[idx] = [
                columns.f32(row, x)?,
                columns.f32(row, y)?,
                columns.f32(row, z)?,
            ];
        }
        if !marker.iter().flatten().all(|v| v.is_finite()) {
            continue;
        }
        let center = scale3(
            add3(add3(marker[0], marker[1]), add3(marker[2], marker[3])),
            0.25,
        );
        let right = normalize3(
            scale3(
                add3(sub3(marker[1], marker[0]), sub3(marker[2], marker[3])),
                0.5,
            ),
            [1.0, 0.0, 0.0],
        );
        let up = normalize3(
            scale3(
                add3(sub3(marker[3], marker[0]), sub3(marker[2], marker[1])),
                0.5,
            ),
            [0.0, 0.0, 1.0],
        );
        let normal = normalize3(cross3(right, up), [0.0, 1.0, 0.0]);
        let half_width = 0.25
            * (distance(marker[1], marker[0])
                + distance(marker[2], marker[3])
                + distance(marker[2], marker[0])
                + distance(marker[1], marker[3]));
        let half_height = 0.25 * (distance(marker[3], marker[0]) + distance(marker[2], marker[1]));
        gates.push(GateSpec {
            name: format!("gate{gate_idx}"),
            center,
            normal,
            right,
            up,
            half_width,
            half_height,
        });
    }
    Ok(FlightGates {
        flight: path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("flight")
            .to_string(),
        episode_idx: episode,
        gates,
    })
}

fn distance(lhs: [f32; 3], rhs: [f32; 3]) -> f32 {
    let d = sub3(lhs, rhs);
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

struct HeaderIndex {
    map: HashMap<String, usize>,
}

impl HeaderIndex {
    fn new(headers: &StringRecord) -> Self {
        Self {
            map: headers
                .iter()
                .enumerate()
                .map(|(idx, name)| (name.to_string(), idx))
                .collect(),
        }
    }

    fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    fn f32(&self, row: &StringRecord, name: &str) -> anyhow::Result<f32> {
        let idx = self
            .map
            .get(name)
            .copied()
            .with_context(|| format!("missing CSV column `{name}`"))?;
        row.get(idx)
            .with_context(|| format!("missing value for CSV column `{name}`"))?
            .parse::<f32>()
            .with_context(|| format!("failed to parse `{name}` as f32"))
    }
}
