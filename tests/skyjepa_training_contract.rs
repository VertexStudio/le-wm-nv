use le_wm_nv::{
    data::skyjepa::SkyJepaDroneDataset,
    models::skyjepa::checkpoint::{SkyJepaCheckpoint, file_sha256, json_sha256},
};
use std::{
    fs,
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
            "--latent-max-steps",
            "2",
            "--prober-max-steps",
            "2",
            "--warmup-steps",
            "1",
            "--cosine-steps",
            "3",
            "--save-every",
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
    Ok(())
}
