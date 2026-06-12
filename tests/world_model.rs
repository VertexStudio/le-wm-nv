use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use candle::{DType, Device, Tensor};
use candle_nn::{ParamsAdamW, VarBuilder, VarMap};
use le_wm_nv::{
    checkpoint,
    models::{
        lewm::{ActionEmbedderConfig, MlpConfig, NormKind, PredictorConfig},
        world_model::{
            ObservationEncoderConfig, StateHeadConfig, VectorLossWeights, VectorMlpConfig,
            WorldModel, WorldModelConfig, vector_batch_loss,
        },
    },
    optim::StatefulAdamW,
};

#[test]
fn vector_world_model_training_step_updates_and_reloads_cuda_weights() -> candle::Result<()> {
    let device = Device::new_cuda(0)?;
    let cfg = tiny_vector_config();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = WorldModel::new(cfg.clone(), vb)?;

    let observations = Tensor::randn(0f32, 1f32, (2, cfg.predictor.num_frames, 6), &device)?;
    let actions = Tensor::randn(0f32, 1f32, (2, cfg.predictor.num_frames, 4), &device)?;
    let targets = Tensor::randn(0f32, 1f32, (2, cfg.predictor.num_frames, 5), &device)?;
    let vars = varmap.all_vars();
    assert!(!vars.is_empty());
    let before = vars
        .iter()
        .map(|var| var.as_tensor().sum_all()?.to_scalar::<f32>())
        .collect::<candle::Result<Vec<_>>>()?;
    let mut opt = StatefulAdamW::new_from_varmap(
        &varmap,
        ParamsAdamW {
            lr: 1e-4,
            weight_decay: 0.0,
            ..ParamsAdamW::default()
        },
    )?;

    let loss = vector_batch_loss(
        &model,
        &observations,
        &actions,
        &targets,
        VectorLossWeights::default(),
    )?;
    let loss_before = loss.total_loss.to_scalar::<f32>()?;
    assert!(loss_before.is_finite());
    opt.backward_step(&loss.total_loss)?;

    let changed = vars
        .iter()
        .zip(before.iter())
        .map(|(var, before)| Ok((var.as_tensor().sum_all()?.to_scalar::<f32>()? - before).abs()))
        .collect::<candle::Result<Vec<_>>>()?
        .into_iter()
        .filter(|diff| *diff > 0.0)
        .count();
    assert!(
        changed > 0,
        "AdamW step did not update any world-model variable"
    );

    let loss_after = vector_batch_loss(
        &model,
        &observations,
        &actions,
        &targets,
        VectorLossWeights::default(),
    )?
    .total_loss
    .to_scalar::<f32>()?;
    assert!(loss_after.is_finite());

    let weights_path = temp_safetensors_path();
    varmap.save(&weights_path)?;
    let reloaded_vb = checkpoint::var_builder_from_path(&weights_path, DType::F32, &device)?;
    let reloaded = WorldModel::new(cfg, reloaded_vb)?;
    let reload_loss = vector_batch_loss(
        &reloaded,
        &observations,
        &actions,
        &targets,
        VectorLossWeights::default(),
    )?
    .total_loss
    .to_scalar::<f32>()?;
    assert!(reload_loss.is_finite());
    assert!((reload_loss - loss_after).abs() < 1e-4);
    fs::remove_file(weights_path)?;
    Ok(())
}

fn tiny_vector_config() -> WorldModelConfig {
    let embed_dim = 24;
    let time = 5;
    WorldModelConfig {
        history_size: 3,
        observation_encoder: ObservationEncoderConfig::VectorMlp(VectorMlpConfig {
            input_dim: 6,
            hidden_dim: 32,
            output_dim: embed_dim,
            depth: 3,
            norm: NormKind::LayerNorm,
        }),
        action_encoder: ActionEmbedderConfig {
            input_dim: 4,
            smoothed_dim: 4,
            emb_dim: embed_dim,
            mlp_scale: 2,
        },
        predictor: PredictorConfig {
            num_frames: time,
            input_dim: embed_dim,
            hidden_dim: embed_dim,
            output_dim: embed_dim,
            depth: 1,
            heads: 2,
            dim_head: 12,
            mlp_dim: 64,
        },
        pred_proj: MlpConfig {
            input_dim: embed_dim,
            hidden_dim: 32,
            output_dim: embed_dim,
            norm: NormKind::LayerNorm,
        },
        state_head: Some(StateHeadConfig::VectorDelta {
            input_dim: embed_dim,
            hidden_dim: 32,
            output_dim: 5,
            norm: NormKind::LayerNorm,
        }),
    }
}

fn temp_safetensors_path() -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "le-wm-nv-world-model-training-{}-{stamp}.safetensors",
        std::process::id()
    ))
}
