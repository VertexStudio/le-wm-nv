use le_wm_nv::{
    data::skyjepa::SkyJepaDroneDataset,
    models::skyjepa::checkpoint::{SkyJepaCheckpoint, TrainingSnapshot, file_sha256, json_sha256},
};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "skyjepa-training-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn successful(output: Output) {
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
fn generate(path: &Path, seed: &str) {
    successful(
        Command::new(env!("CARGO_BIN_EXE_lewm-generate-skyjepa"))
            .arg("--output-dir")
            .arg(path)
            .args([
                "--domains",
                "3",
                "--trajectories",
                "12",
                "--duration-seconds",
                "2",
                "--seed",
                seed,
            ])
            .output()
            .unwrap(),
    );
}
fn trainer(data: &Path, output: &Path, stage: &str) -> Command {
    trainer_with_steps(data, output, stage, "2")
}
fn trainer_with_steps(data: &Path, output: &Path, stage: &str, steps: &str) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_lewm-train-skyjepa"));
    cmd.arg("--dataset-dir")
        .arg(data)
        .arg("--output-dir")
        .arg(output)
        .args([
            "--stage",
            stage,
            "--batch-size",
            "16",
            "--split-by",
            "episodes",
            "--latent-max-steps",
            steps,
            "--prober-max-steps",
            steps,
            "--warmup-steps",
            "1",
            "--cosine-steps",
            "3",
            "--save-every",
            "1",
            "--log-every",
            "1",
            "--validation-batches",
            "1",
            "--skip-audit",
        ]);
    cmd
}
fn tree_fingerprint(root: &Path) -> anyhow::Result<String> {
    let mut files = walkdir::WalkDir::new(root)
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    files.sort_by_key(|entry| entry.path().to_owned());
    let hashes = files
        .iter()
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| {
            Ok((
                entry.path().strip_prefix(root)?.to_owned(),
                file_sha256(entry.path())?,
            ))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    json_sha256(&hashes)
}

#[test]
fn prober_reuses_parent_normalization_and_rejects_resume_changes_without_writes()
-> anyhow::Result<()> {
    let scratch = Scratch::new();
    let data_a = scratch.0.join("data-a");
    let data_b = scratch.0.join("data-b");
    let latent = scratch.0.join("latent");
    let prober = scratch.0.join("prober");
    generate(&data_a, "7");
    generate(&data_b, "17");
    successful(trainer(&data_a, &latent, "latent").output()?);
    let parent = SkyJepaCheckpoint::load(&latent)?;
    let dataset_b = SkyJepaDroneDataset::open(&data_b, parent.contract.dataset)?;
    assert_ne!(
        json_sha256(dataset_b.normalization())?,
        json_sha256(&parent.contract.normalization)?
    );
    successful(
        trainer(&data_b, &prober, "prober")
            .arg("--latent-checkpoint")
            .arg(&latent)
            .output()?,
    );
    let trained = SkyJepaCheckpoint::load(&prober)?;
    assert_eq!(
        json_sha256(&parent.contract.normalization)?,
        json_sha256(&trained.contract.normalization)?
    );
    let before = tree_fingerprint(&prober)?;
    let rejected = trainer(&data_b, &prober, "prober")
        .args(["--resume", "--mass", "2.0"])
        .output()?;
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("resume settings disagree"));
    assert_eq!(before, tree_fingerprint(&prober)?);
    let other_parent = scratch.0.join("other-parent");
    successful(trainer(&data_b, &other_parent, "latent").output()?);
    let rejected = trainer(&data_b, &prober, "prober")
        .arg("--resume")
        .arg("--latent-checkpoint")
        .arg(&other_parent)
        .output()?;
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("resume settings disagree"));
    assert_eq!(before, tree_fingerprint(&prober)?);
    // Re-publication after the last training snapshot is idempotent, including
    // recovery from interruption between the last update and package export.
    successful(
        trainer(&data_b, &prober, "prober")
            .arg("--resume")
            .output()?,
    );
    assert_eq!(
        trained.fingerprint()?,
        SkyJepaCheckpoint::load(&prober)?.fingerprint()?
    );
    let snapshot = TrainingSnapshot::load(&prober, "prober")?;
    fs::write(
        snapshot.optimizer_path(),
        b"interrupted or corrupted optimizer",
    )?;
    let corrupted = tree_fingerprint(&prober)?;
    let rejected = trainer(&data_b, &prober, "prober")
        .arg("--resume")
        .output()?;
    assert!(!rejected.status.success());
    assert!(
        String::from_utf8_lossy(&rejected.stderr).contains("optimizer.safetensors fingerprint")
    );
    assert_eq!(corrupted, tree_fingerprint(&prober)?);
    Ok(())
}

fn assert_tensor_files_equal(lhs: &Path, rhs: &Path) -> anyhow::Result<()> {
    let a = candle::safetensors::load(lhs, &candle::Device::Cpu)?;
    let b = candle::safetensors::load(rhs, &candle::Device::Cpu)?;
    assert_eq!(a.len(), b.len());
    for (key, tensor) in a {
        assert_eq!(tensor.dims(), b[&key].dims());
        assert_eq!(
            tensor.flatten_all()?.to_vec1::<f32>()?,
            b[&key].flatten_all()?.to_vec1::<f32>()?,
            "tensor {key} differs between uninterrupted and resumed training"
        );
    }
    Ok(())
}

fn training_metrics(path: &Path) -> anyhow::Result<Vec<serde_json::Value>> {
    fs::read_to_string(path)?
        .lines()
        .map(|line| {
            let mut value: serde_json::Value = serde_json::from_str(line)?;
            value.as_object_mut().unwrap().remove("elapsed_sec");
            Ok(value)
        })
        .collect()
}

#[test]
fn cuda_resume_matches_uninterrupted_updates_across_validation_and_epoch_boundaries()
-> anyhow::Result<()> {
    let scratch = Scratch::new();
    let data = scratch.0.join("data");
    generate(&data, "31");
    let continuous = scratch.0.join("continuous");
    successful(trainer_with_steps(&data, &continuous, "latent", "16").output()?);
    let original = TrainingSnapshot::load(&continuous, "latent")?;
    // This fixture has eight batches per epoch. Exercise both sides of the
    // epoch validation and the first batch of the next epoch.
    assert_eq!(original.manifest.progress["batches_per_epoch"], 8);
    for interruption in [7, 8, 9] {
        let resumed = scratch.0.join(format!("resumed-{interruption}"));
        successful(
            trainer_with_steps(&data, &resumed, "latent", "16")
                .arg("--stop-after-step")
                .arg(interruption.to_string())
                .output()?,
        );
        assert_eq!(
            TrainingSnapshot::load(&resumed, "latent")?.manifest.step,
            interruption
        );
        // Model a crash after logging an update but before publishing its
        // snapshot. Recovery must not leave duplicate/uncommitted metrics.
        let mut log = fs::OpenOptions::new()
            .append(true)
            .open(resumed.join("latent-metrics.jsonl"))?;
        log.write_all(b"{\"step\":999,\"kind\":\"train\"}\n{\"partial\":")?;
        drop(log);
        successful(
            trainer_with_steps(&data, &resumed, "latent", "16")
                .arg("--resume")
                .output()?,
        );
        let actual = TrainingSnapshot::load(&resumed, "latent")?;
        assert_tensor_files_equal(&original.weights_path(), &actual.weights_path())?;
        assert_tensor_files_equal(&original.optimizer_path(), &actual.optimizer_path())?;
        assert_eq!(
            training_metrics(&continuous.join("latent-metrics.jsonl"))?,
            training_metrics(&resumed.join("latent-metrics.jsonl"))?
        );
        assert_eq!(
            original.manifest.progress["best_step"],
            actual.manifest.progress["best_step"]
        );
    }
    let prober_a = scratch.0.join("prober-continuous");
    let prober_b = scratch.0.join("prober-resumed");
    successful(
        trainer_with_steps(&data, &prober_a, "prober", "16")
            .arg("--latent-checkpoint")
            .arg(&continuous)
            .output()?,
    );
    successful(
        trainer_with_steps(&data, &prober_b, "prober", "16")
            .arg("--latent-checkpoint")
            .arg(&continuous)
            .args(["--stop-after-step", "8"])
            .output()?,
    );
    successful(
        trainer_with_steps(&data, &prober_b, "prober", "16")
            .arg("--latent-checkpoint")
            .arg(&continuous)
            .arg("--resume")
            .output()?,
    );
    let a = TrainingSnapshot::load(&prober_a, "prober")?;
    let b = TrainingSnapshot::load(&prober_b, "prober")?;
    assert_tensor_files_equal(&a.weights_path(), &b.weights_path())?;
    assert_tensor_files_equal(&a.optimizer_path(), &b.optimizer_path())?;
    assert_eq!(
        training_metrics(&prober_a.join("prober-metrics.jsonl"))?,
        training_metrics(&prober_b.join("prober-metrics.jsonl"))?
    );
    Ok(())
}

#[test]
fn trainer_rejects_noncanonical_rate_before_creating_a_run() -> anyhow::Result<()> {
    let scratch = Scratch::new();
    let output_dir = scratch.0.join("must-not-exist");
    let output = Command::new(env!("CARGO_BIN_EXE_lewm-train-skyjepa"))
        .arg("--output-dir")
        .arg(&output_dir)
        .args(["--model-rate-hz", "10"])
        .output()?;
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("20 Hz"));
    assert!(!output_dir.exists());
    Ok(())
}

#[test]
fn domain_split_and_external_evaluation_report_the_actual_population() -> anyhow::Result<()> {
    let scratch = Scratch::new();
    let data = scratch.0.join("data");
    let external = scratch.0.join("external");
    let run = scratch.0.join("run");
    generate(&data, "71");
    generate(&external, "72");
    successful(
        Command::new(env!("CARGO_BIN_EXE_lewm-train-skyjepa"))
            .arg("--dataset-dir")
            .arg(&data)
            .arg("--output-dir")
            .arg(&run)
            .args([
                "--stage",
                "both",
                "--batch-size",
                "16",
                "--latent-max-steps",
                "2",
                "--prober-max-steps",
                "2",
                "--warmup-steps",
                "1",
                "--cosine-steps",
                "3",
                "--validation-batches",
                "1",
                "--skip-audit",
            ])
            .output()?,
    );
    let package = SkyJepaCheckpoint::load(&run)?;
    let dataset = SkyJepaDroneDataset::open(&data, package.contract.dataset)?;
    let train = dataset.domain_ids_for_rows(&dataset.train_rows());
    let validation = dataset.domain_ids_for_rows(&dataset.validation_rows());
    let test = dataset.domain_ids_for_rows(&dataset.test_rows());
    assert_eq!((train.len(), validation.len(), test.len()), (1, 1, 1));
    assert!(
        train
            .iter()
            .all(|id| !test.contains(id) && !validation.contains(id))
    );
    let report = scratch.0.join("eval.json");
    let evaluate = |dataset: &Path, split: &str, limit: &str| {
        Command::new(env!("CARGO_BIN_EXE_lewm-eval-skyjepa"))
            .arg("--dataset-dir")
            .arg(dataset)
            .arg("--checkpoint-dir")
            .arg(&run)
            .arg("--output")
            .arg(&report)
            .args([
                "--split",
                split,
                "--rollout-steps",
                "5",
                "--batch-size",
                "16",
                "--max-batches",
                limit,
            ])
            .output()
    };
    assert!(!evaluate(&data, "all", "0")?.status.success());
    successful(evaluate(&data, "test", "0")?);
    let own: serde_json::Value = serde_json::from_slice(&fs::read(&report)?)?;
    assert_eq!(own["scope"], "held_out_domains");
    assert_eq!(own["evaluated_episode_ids"].as_array().unwrap().len(), 4);
    assert_eq!(own["training_domain_overlap"], 0);
    successful(evaluate(&external, "all", "0")?);
    let full: serde_json::Value = serde_json::from_slice(&fs::read(&report)?)?;
    assert_eq!(full["evaluated_episode_ids"].as_array().unwrap().len(), 12);
    assert_eq!(full["complete_population"], true);
    assert_eq!(full["training_domain_overlap"], 0);
    successful(evaluate(&external, "all", "1")?);
    let limited: serde_json::Value = serde_json::from_slice(&fs::read(&report)?)?;
    assert_eq!(limited["windows"], 16);
    assert_eq!(limited["complete_population"], false);
    for mode in [
        "prior",
        "nominal-physics-mppi",
        "untrained-mppi",
        "trained-mppi",
    ] {
        successful(
            Command::new(env!("CARGO_BIN_EXE_lewm-bench-skyjepa"))
                .arg("--checkpoint-dir")
                .arg(&run)
                .arg("--output")
                .arg(&report)
                .args([
                    "--controller",
                    mode,
                    "--random-domains",
                    "0",
                    "--duration-seconds",
                    "0.1",
                    "--samples",
                    "8",
                    "--horizon",
                    "3",
                    "--trim-multiplier",
                    "0.9",
                    "--allow-fail",
                ])
                .output()?,
        );
        let bench: serde_json::Value = serde_json::from_slice(&fs::read(&report)?)?;
        assert_eq!(bench["controller"], mode.replace('-', "_"));
        assert_eq!(bench["runs"], 3);
        assert!(bench["checkpoint_sha256"].as_str().unwrap().len() == 64);
        assert!(bench["executable_sha256"].as_str().unwrap().len() == 64);
        for scenario in bench["results"].as_array().unwrap() {
            assert!(scenario["tracking_passed"].is_boolean());
            assert!(scenario["timing_passed"].is_boolean());
            assert!((scenario["trim_scale"].as_f64().unwrap() - 0.9).abs() < 1e-6);
            assert_eq!(scenario["plan_times_ms"].as_array().unwrap().len(), 2);
        }
    }
    Ok(())
}
