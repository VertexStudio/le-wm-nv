use std::path::PathBuf;

use anyhow::Context;
use hf_hub::{Repo, RepoType, api::sync::Api};

#[derive(Debug, Clone)]
pub struct StableWorldModelFiles {
    pub config: PathBuf,
    pub weights: PathBuf,
}

pub fn download_stable_worldmodel_checkpoint(
    repo_id: &str,
    revision: Option<&str>,
) -> anyhow::Result<StableWorldModelFiles> {
    let _ = repo_id.split_once('/').with_context(|| {
        format!("expected Hugging Face model repo id like owner/name, got {repo_id}")
    })?;
    let api = Api::new()?;
    let repo = match revision {
        Some(revision) => api.repo(Repo::with_revision(
            repo_id.to_string(),
            RepoType::Model,
            revision.to_string(),
        )),
        None => api.model(repo_id.to_string()),
    };

    let config = repo.get("config.json")?;
    let weights = repo.get("weights.pt")?;

    Ok(StableWorldModelFiles { config, weights })
}
