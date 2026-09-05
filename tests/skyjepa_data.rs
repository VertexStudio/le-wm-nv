use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use candle::{DType, Device, Result};
use hdf5::File;
use le_wm_nv::data::{
    drone_racing::{ImportedDroneData, RunningStats},
    skyjepa::{
        SKYJEPA_SCHEMA_VERSION, SKYJEPA_STATE_DIM, SkyJepaActionSpace, SkyJepaDatasetConfig,
        SkyJepaDatasetMetadata, SkyJepaDroneDataset, SkyJepaNormalization, SkyJepaSplitBy,
    },
};

#[test]
fn builds_strided_full_state_batches_from_imported_data() -> Result<()> {
    let root = temp_dir();
    let mut data = ImportedDroneData::default();
    let rows_per_episode = 40;
    for episode in 0..4i64 {
        for step in 0..rows_per_episode {
            let value = episode as f32 + step as f32 * 0.01;
            data.pos_world.extend_from_slice(&[value, 2.0, 3.0]);
            data.rotmat_world_from_body
                .extend_from_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
            data.lin_vel_body.extend_from_slice(&[value, 0.5, -0.25]);
            data.ang_vel_body.extend_from_slice(&[0.1, 0.2, 0.3]);
            data.vbat.push(16.0);
            data.channels_raw
                .extend_from_slice(&[1500.0, 1500.0, 1200.0, 1500.0]);
            data.channels_norm.extend_from_slice(&[0.0, 0.0, 0.2, 0.0]);
            data.accel_body.extend_from_slice(&[0.0; 3]);
            data.gyro_body.extend_from_slice(&[0.0; 3]);
            data.episode_idx.push(episode);
            data.step_idx.push(step as i64);
            data.elapsed_time.push(step as f32 * 0.01);
            data.dt.push(0.01);
        }
    }
    data.write_artifact(&root, &root, 100, 500, 0.25).unwrap();

    let dataset = SkyJepaDroneDataset::open(
        &root,
        SkyJepaDatasetConfig {
            split_by: SkyJepaSplitBy::Episodes,
            batch_size: 2,
            history_steps: 2,
            rollout_steps: 2,
            model_rate_hz: 20,
            normalize_states: true,
            normalize_actions: true,
            ..SkyJepaDatasetConfig::paper_derived(2)
        },
    )
    .unwrap();
    assert_eq!(dataset.source_stride(), 5);
    assert_eq!(dataset.splits().train.len(), 2);
    assert_eq!(dataset.splits().validation.len(), 1);
    assert_eq!(dataset.splits().test.len(), 1);
    let rows = dataset.train_rows();
    let batch = dataset.batch(&rows[..2], DType::F32, &Device::Cpu).unwrap();
    assert_eq!(batch.states.dims(), [2, 4, SKYJEPA_STATE_DIM]);
    assert_eq!(batch.metric_states.dims(), [2, 4, SKYJEPA_STATE_DIM]);
    assert_eq!(batch.actions.dims(), [2, 4, 4]);
    assert_eq!(batch.transition_dt.dims(), [2, 3]);
    let raw = batch.metric_states.to_vec3::<f32>().unwrap();
    assert_eq!(raw[0][0][3..6], [0.0, 0.5, -0.25]);
    assert!((raw[0][1][0] - 0.05).abs() < 1e-6);
    for value in batch.transition_dt.flatten_all()?.to_vec1::<f32>()? {
        assert!((value - 0.05).abs() < 1e-6);
    }
    fs::remove_dir_all(root).ok();
    Ok(())
}

#[test]
fn opens_canonical_rotor_force_schema() -> Result<()> {
    let root = temp_dir();
    fs::create_dir_all(&root).unwrap();
    let episodes = 3usize;
    let rows_per_episode = 35usize;
    let rows = episodes * rows_per_episode;
    let mut states = vec![0f32; rows * 18];
    for (row, state) in states.chunks_exact_mut(18).enumerate() {
        state[0] = row as f32 * 0.01;
        state[2] = 1.0;
        state[6] = 1.0;
        state[10] = 1.0;
        state[14] = 1.0;
    }
    let actions = vec![3.2f32; rows * 4];
    let episode_idx = (0..episodes)
        .flat_map(|episode| std::iter::repeat_n(episode as i64, rows_per_episode))
        .collect::<Vec<_>>();
    let step_idx = (0..episodes)
        .flat_map(|_| (0..rows_per_episode).map(|step| step as i64))
        .collect::<Vec<_>>();
    let dt = vec![0.05f32; rows];
    let file = File::create(root.join("data.h5")).unwrap();
    file.new_dataset::<f32>()
        .shape((rows, 18))
        .create("state")
        .unwrap()
        .write_raw(&states)
        .unwrap();
    file.new_dataset::<f32>()
        .shape((rows, 4))
        .create("action")
        .unwrap()
        .write_raw(&actions)
        .unwrap();
    file.new_dataset::<i64>()
        .shape(rows)
        .create("episode_idx")
        .unwrap()
        .write_raw(&episode_idx)
        .unwrap();
    file.new_dataset::<i64>()
        .shape(rows)
        .create("step_idx")
        .unwrap()
        .write_raw(&step_idx)
        .unwrap();
    file.new_dataset::<f32>()
        .shape((rows, 1))
        .create("dt")
        .unwrap()
        .write_raw(&dt)
        .unwrap();
    drop(file);
    let metadata = SkyJepaDatasetMetadata {
        schema_version: SKYJEPA_SCHEMA_VERSION,
        data_h5: PathBuf::from("data.h5"),
        sample_rate_hz: 20,
        rows,
        episodes,
        state_dim: 18,
        action_dim: 4,
        action_space: SkyJepaActionSpace::RotorForces,
        generator: Some("test".into()),
        seed: Some(7),
        domains: Some(3),
        domain_distribution: None,
        has_reference_state: false,
        has_motor_force: false,
    };
    fs::write(
        root.join("metadata.json"),
        serde_json::to_string(&metadata).unwrap(),
    )
    .unwrap();
    let dataset = SkyJepaDroneDataset::open(
        &root,
        SkyJepaDatasetConfig {
            split_by: SkyJepaSplitBy::Episodes,
            batch_size: 2,
            history_steps: 2,
            rollout_steps: 2,
            model_rate_hz: 20,
            normalize_states: true,
            normalize_actions: true,
            action_space: SkyJepaActionSpace::RotorForces,
        },
    )
    .unwrap();
    assert_eq!(dataset.source_stride(), 1);
    assert!(!dataset.train_rows().is_empty());
    let batch = dataset
        .batch(&dataset.train_rows()[..2], DType::F32, &Device::Cpu)
        .unwrap();
    assert_eq!(batch.metric_actions.to_vec3::<f32>()?[0][0], [3.2; 4]);

    let fixed = SkyJepaNormalization {
        state: RunningStats {
            mean: vec![1.0; 18],
            std: vec![2.0; 18],
        },
        action: RunningStats {
            mean: vec![3.0; 4],
            std: vec![0.5; 4],
        },
    };
    let fixed_dataset = SkyJepaDroneDataset::open_with_normalization(
        &root,
        SkyJepaDatasetConfig {
            split_by: SkyJepaSplitBy::Episodes,
            batch_size: 2,
            history_steps: 2,
            rollout_steps: 2,
            model_rate_hz: 20,
            normalize_states: true,
            normalize_actions: true,
            action_space: SkyJepaActionSpace::RotorForces,
        },
        Some(fixed),
    )
    .unwrap();
    let fixed_batch = fixed_dataset
        .batch(&fixed_dataset.train_rows()[..1], DType::F32, &Device::Cpu)
        .unwrap();
    for value in &fixed_batch.actions.to_vec3::<f32>()?[0][0] {
        assert!((*value - 0.4).abs() < 1e-6);
    }
    assert!((fixed_batch.states.to_vec3::<f32>()?[0][0][2] - 0.0).abs() < 1e-6);
    fs::remove_dir_all(root).ok();
    Ok(())
}

fn temp_dir() -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "le-wm-nv-skyjepa-data-{}-{stamp}",
        std::process::id()
    ))
}
