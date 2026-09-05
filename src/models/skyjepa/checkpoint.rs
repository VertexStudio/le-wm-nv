//! SkyJEPA packages are immutable content-addressed weights plus one atomic
//! manifest. Loading never guesses preprocessing from neighbouring loose files.
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{SkyJepaConfig, SkyJepaProberConfig};
use crate::data::skyjepa::{
    SKYJEPA_ACTION_DIM, SKYJEPA_STATE_DIM, SkyJepaDatasetConfig, SkyJepaNormalization,
    validate_normalization,
};

pub const CHECKPOINT_VERSION: u32 = 2;
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelContract {
    pub model: SkyJepaConfig,
    pub dataset: SkyJepaDatasetConfig,
    pub normalization: SkyJepaNormalization,
    pub prober: Option<SkyJepaProberConfig>,
}

impl ModelContract {
    pub fn validate(&self) -> anyhow::Result<()> {
        self.model.validate()?;
        self.dataset.validate()?;
        validate_normalization(&self.normalization)?;
        ensure!(
            self.model.state_dim == SKYJEPA_STATE_DIM
                && self.model.action_dim == SKYJEPA_ACTION_DIM,
            "checkpoint requires state18/action4"
        );
        ensure!(
            self.model.history_steps == self.dataset.history_steps
                && self.model.rollout_steps == self.dataset.rollout_steps,
            "checkpoint model/dataset sequence dimensions disagree"
        );
        ensure!(
            self.dataset.normalize_states && self.dataset.normalize_actions,
            "checkpoint requires normalized state/action model inputs"
        );
        if let Some(prober) = &self.prober {
            prober.validate()?;
            ensure!(
                prober.latent_dim == self.model.latent_dim
                    && prober.kinematics.action_space == self.dataset.action_space,
                "checkpoint prober/model/action-space mismatch"
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkyJepaCheckpoint {
    pub format_version: u32,
    pub contract: ModelContract,
    pub contract_sha256: String,
    pub latent_sha256: String,
    pub prober_sha256: Option<String>,
    pub provenance: serde_json::Value,
}

impl SkyJepaCheckpoint {
    pub fn load(root: &Path) -> anyhow::Result<Self> {
        let path = root.join("checkpoint.json");
        let checkpoint: Self = read_json(&path).with_context(
            || "SkyJEPA requires a versioned checkpoint package, not loose safetensors",
        )?;
        ensure!(
            checkpoint.format_version == CHECKPOINT_VERSION,
            "unsupported SkyJEPA checkpoint version"
        );
        checkpoint.contract.validate()?;
        ensure!(
            checkpoint.contract_sha256 == json_sha256(&checkpoint.contract)?,
            "checkpoint contract fingerprint mismatch"
        );
        ensure!(
            checkpoint.contract.prober.is_some() == checkpoint.prober_sha256.is_some(),
            "checkpoint prober weights/configuration mismatch"
        );
        verify_object(root, &checkpoint.latent_sha256)?;
        if let Some(hash) = &checkpoint.prober_sha256 {
            verify_object(root, hash)?;
        }
        Ok(checkpoint)
    }

    pub fn publish(
        root: &Path,
        contract: ModelContract,
        latent: &Path,
        prober: Option<&Path>,
        provenance: serde_json::Value,
    ) -> anyhow::Result<Self> {
        contract.validate()?;
        ensure!(
            contract.prober.is_some() == prober.is_some(),
            "prober configuration/weights must be paired"
        );
        let checkpoint = Self {
            format_version: CHECKPOINT_VERSION,
            contract_sha256: json_sha256(&contract)?,
            contract,
            latent_sha256: store_object(root, latent)?,
            prober_sha256: prober.map(|path| store_object(root, path)).transpose()?,
            provenance,
        };
        atomic_json(&root.join("checkpoint.json"), &checkpoint)?;
        Ok(checkpoint)
    }

    pub fn latent_path(&self, root: &Path) -> PathBuf {
        object_path(root, &self.latent_sha256)
    }

    pub fn prober_path(&self, root: &Path) -> anyhow::Result<PathBuf> {
        let hash = self
            .prober_sha256
            .as_ref()
            .context("checkpoint has no trained prober")?;
        Ok(object_path(root, hash))
    }

    /// Identity of the frozen latent function, independent of packaging path,
    /// prober training settings, or the prober later attached to this package.
    pub fn latent_identity(&self) -> anyhow::Result<String> {
        let mut dataset = self.contract.dataset;
        dataset.batch_size = 1;
        json_sha256(
            &serde_json::json!({"model": self.contract.model, "dataset": dataset,
            "normalization": self.contract.normalization, "weights": self.latent_sha256}),
        )
    }

    pub fn fingerprint(&self) -> anyhow::Result<String> {
        json_sha256(self)
    }
}

fn object_path(root: &Path, hash: &str) -> PathBuf {
    root.join("objects").join(format!("{hash}.safetensors"))
}

fn verify_object(root: &Path, hash: &str) -> anyhow::Result<()> {
    ensure!(
        hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid checkpoint object hash"
    );
    ensure!(
        file_sha256(&object_path(root, hash))? == hash,
        "checkpoint weights fingerprint mismatch: {hash}"
    );
    Ok(())
}

fn store_object(root: &Path, source: &Path) -> anyhow::Result<String> {
    let bytes = fs::read(source).with_context(|| format!("read weights {}", source.display()))?;
    let hash = bytes_sha256(&bytes);
    let destination = object_path(root, &hash);
    if destination.exists() {
        verify_object(root, &hash)?;
    } else {
        atomic_bytes(&destination, &bytes)?;
    }
    Ok(hash)
}

pub fn bytes_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn json_sha256<T: Serialize>(value: &T) -> anyhow::Result<String> {
    Ok(bytes_sha256(&serde_json::to_vec(value)?))
}

pub fn file_sha256(path: &Path) -> anyhow::Result<String> {
    let mut file = File::open(path).with_context(|| format!("read {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> anyhow::Result<T> {
    Ok(serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read {}", path.display()))?,
    )?)
}

pub fn atomic_json<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    atomic_bytes(path, &serde_json::to_vec_pretty(value)?)
}

pub fn atomic_bytes(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let parent = path.parent().context("atomic destination has no parent")?;
    fs::create_dir_all(parent)?;
    let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
    let temp = parent.join(format!(".publish-{}-{id}.tmp", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(&temp, path)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}
