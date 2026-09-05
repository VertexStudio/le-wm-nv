use le_wm_nv::{
    data::{
        drone_racing::RunningStats,
        skyjepa::{SkyJepaDatasetConfig, SkyJepaNormalization},
    },
    models::skyjepa::{
        SkyJepaConfig, SkyJepaProberConfig,
        checkpoint::{ModelContract, SkyJepaCheckpoint, atomic_json},
    },
};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "skyjepa-checkpoint-test-{}-{}",
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

fn contract() -> ModelContract {
    ModelContract {
        model: SkyJepaConfig::paper_derived(),
        dataset: SkyJepaDatasetConfig::paper_derived(4),
        normalization: SkyJepaNormalization {
            state: RunningStats::identity(18),
            action: RunningStats::identity(4),
        },
        prober: None,
    }
}

#[test]
fn package_detects_modified_weights_and_preprocessing() -> anyhow::Result<()> {
    let scratch = Scratch::new();
    let weights = scratch.0.join("input.safetensors");
    fs::write(&weights, b"test-weight-bytes")?;
    let package = SkyJepaCheckpoint::publish(
        &scratch.0,
        contract(),
        &weights,
        None,
        serde_json::json!({}),
    )?;
    assert_eq!(
        package.latent_identity()?,
        SkyJepaCheckpoint::load(&scratch.0)?.latent_identity()?
    );
    let mut edited = package.clone();
    edited.contract.normalization.state.mean[0] = 1.0;
    atomic_json(&scratch.0.join("checkpoint.json"), &edited)?;
    assert!(
        SkyJepaCheckpoint::load(&scratch.0)
            .unwrap_err()
            .to_string()
            .contains("contract fingerprint")
    );
    atomic_json(&scratch.0.join("checkpoint.json"), &package)?;
    fs::write(package.latent_path(&scratch.0), b"different weights")?;
    assert!(
        SkyJepaCheckpoint::load(&scratch.0)
            .unwrap_err()
            .to_string()
            .contains("weights fingerprint")
    );
    Ok(())
}

#[test]
fn latent_identity_ignores_prober_but_binds_preprocessing() -> anyhow::Result<()> {
    let scratch = Scratch::new();
    let weights = scratch.0.join("weights");
    fs::write(&weights, b"weight fixture")?;
    let package = SkyJepaCheckpoint::publish(
        &scratch.0,
        contract(),
        &weights,
        None,
        serde_json::json!({}),
    )?;
    let mut full = contract();
    full.prober = Some(SkyJepaProberConfig::paper_derived(24));
    full.dataset.batch_size = 64;
    let full = SkyJepaCheckpoint::publish(
        &scratch.0,
        full,
        &weights,
        Some(&weights),
        serde_json::json!({"new_run":true}),
    )?;
    assert_eq!(package.latent_identity()?, full.latent_identity()?);
    let mut altered = package.clone();
    altered.contract.normalization.action.std[0] = 2.0;
    assert_ne!(package.latent_identity()?, altered.latent_identity()?);
    assert!(
        SkyJepaCheckpoint::load(&scratch.0)?
            .prober_path(&scratch.0)?
            .is_file()
    );
    Ok(())
}

#[test]
fn invalid_publication_keeps_previous_package() -> anyhow::Result<()> {
    let scratch = Scratch::new();
    let weights = scratch.0.join("weights");
    fs::write(&weights, b"weight fixture")?;
    let original = SkyJepaCheckpoint::publish(
        &scratch.0,
        contract(),
        &weights,
        None,
        serde_json::json!({}),
    )?;
    let mut invalid = contract();
    invalid.prober = Some(SkyJepaProberConfig::paper_derived(12));
    assert!(
        SkyJepaCheckpoint::publish(
            &scratch.0,
            invalid,
            &weights,
            Some(&weights),
            serde_json::json!({})
        )
        .is_err()
    );
    assert_eq!(
        original.fingerprint()?,
        SkyJepaCheckpoint::load(&scratch.0)?.fingerprint()?
    );
    Ok(())
}
