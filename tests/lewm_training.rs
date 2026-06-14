use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use candle::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use le_wm_nv::{
    checkpoint,
    models::lewm::{
        ActionEmbedderConfig, LeWm, LeWmConfig, MlpConfig, NormKind, PredictorConfig,
        VitEncoderConfig, batch_loss,
    },
};

#[test]
fn lewm_training_step_updates_and_reloads_cuda_weights() -> candle::Result<()> {
    let device = Device::new_cuda(0)?;
    let cfg = tiny_training_config();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = LeWm::new(cfg.clone(), vb)?;

    let pixels = Tensor::randn(
        0f32,
        1f32,
        (
            2,
            cfg.history_size,
            cfg.encoder.num_channels,
            cfg.encoder.image_size,
            cfg.encoder.image_size,
        ),
        &device,
    )?;
    let actions = Tensor::randn(
        0f32,
        1f32,
        (2, cfg.history_size, cfg.action_encoder.input_dim),
        &device,
    )?;

    let vars = varmap.all_vars();
    assert!(!vars.is_empty());
    let before = vars
        .iter()
        .map(|var| var.as_tensor().sum_all()?.to_scalar::<f32>())
        .collect::<candle::Result<Vec<_>>>()?;
    let mut opt = AdamW::new(
        vars.clone(),
        ParamsAdamW {
            lr: 1e-4,
            weight_decay: 0.0,
            ..ParamsAdamW::default()
        },
    )?;

    let loss = batch_loss(&model, &pixels, &actions)?;
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
    assert!(changed > 0, "AdamW step did not update any LeWM variable");

    let loss_after = batch_loss(&model, &pixels, &actions)?;
    let total_after = loss_after.total_loss.to_scalar::<f32>()?;
    let prediction_after = loss_after.prediction_loss.to_scalar::<f32>()?;
    assert!(total_after.is_finite());
    assert!(prediction_after.is_finite());

    let weights_path = temp_safetensors_path();
    varmap.save(&weights_path)?;
    let reloaded_vb = checkpoint::var_builder_from_path(&weights_path, DType::F32, &device)?;
    let reloaded = LeWm::new(cfg, reloaded_vb)?;
    let reload_loss = batch_loss(&reloaded, &pixels, &actions)?;
    let reload_total = reload_loss.total_loss.to_scalar::<f32>()?;
    let reload_prediction = reload_loss.prediction_loss.to_scalar::<f32>()?;
    assert!(reload_total.is_finite());
    assert!(reload_prediction.is_finite());
    assert!((reload_prediction - prediction_after).abs() < 1e-4);
    fs::remove_file(weights_path)?;

    Ok(())
}

fn tiny_training_config() -> LeWmConfig {
    let embed_dim = 32;
    let history_size = 3;
    LeWmConfig {
        encoder: VitEncoderConfig {
            image_size: 28,
            patch_size: 14,
            hidden_size: embed_dim,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            intermediate_size: 64,
            layer_norm_eps: 1e-5,
            num_channels: 3,
            qkv_bias: true,
        },
        predictor: PredictorConfig {
            num_frames: history_size,
            input_dim: embed_dim,
            hidden_dim: embed_dim,
            output_dim: embed_dim,
            depth: 1,
            heads: 2,
            dim_head: 16,
            mlp_dim: 64,
        },
        action_encoder: ActionEmbedderConfig {
            input_dim: 2,
            smoothed_dim: 2,
            emb_dim: embed_dim,
            mlp_scale: 2,
        },
        projector: MlpConfig {
            input_dim: embed_dim,
            hidden_dim: 64,
            output_dim: embed_dim,
            norm: NormKind::LayerNorm,
        },
        pred_proj: MlpConfig {
            input_dim: embed_dim,
            hidden_dim: 64,
            output_dim: embed_dim,
            norm: NormKind::LayerNorm,
        },
        history_size,
    }
}

fn temp_safetensors_path() -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "le-wm-nv-lewm-training-{}-{stamp}.safetensors",
        std::process::id()
    ))
}
