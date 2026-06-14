use le_wm_nv::models::{
    lewm::{LeWmConfig, NormKind},
    world_model::WorldModelConfig,
};

#[test]
fn tiny_patch14_defaults_match_python_config() {
    let cfg = LeWmConfig::tiny_patch14_224(2);

    assert_eq!(cfg.encoder.image_size, 224);
    assert_eq!(cfg.encoder.patch_size, 14);
    assert_eq!(cfg.encoder.hidden_size, 192);
    assert_eq!(cfg.encoder.num_hidden_layers, 12);
    assert_eq!(cfg.encoder.num_attention_heads, 3);

    assert_eq!(cfg.predictor.num_frames, 3);
    assert_eq!(cfg.predictor.depth, 6);
    assert_eq!(cfg.predictor.heads, 16);
    assert_eq!(cfg.predictor.dim_head, 64);
    assert_eq!(cfg.predictor.mlp_dim, 2048);

    assert_eq!(cfg.action_encoder.input_dim, 2);
    assert_eq!(cfg.projector.norm, NormKind::BatchNorm1d);
    assert_eq!(cfg.pred_proj.norm, NormKind::BatchNorm1d);
}

#[test]
fn config_round_trips_through_json() {
    let cfg = LeWmConfig::tiny_patch14_224(4);
    let json = serde_json::to_string(&cfg).unwrap();
    let decoded: LeWmConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(decoded.action_encoder.input_dim, 4);
    assert_eq!(decoded.encoder.patch_size, cfg.encoder.patch_size);
    assert_eq!(decoded.predictor.hidden_dim, cfg.predictor.hidden_dim);
}

#[test]
fn vector_world_model_config_round_trips_through_json() {
    let cfg = WorldModelConfig::vector_drone_default(20, 4);
    let json = serde_json::to_string(&cfg).unwrap();
    let decoded: WorldModelConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(decoded.history_size, cfg.history_size);
    assert_eq!(decoded.action_encoder.input_dim, 4);
    assert_eq!(decoded.observation_encoder.output_dim(), 192);
}

#[test]
fn parses_stable_worldmodel_lewm_config_json() {
    let json = r#"{
      "_target_": "stable_worldmodel.wm.lewm.LeWM",
      "encoder": {
        "_target_": "stable_pretraining.backbone.utils.vit_hf",
        "size": "tiny",
        "patch_size": 14,
        "image_size": 224,
        "pretrained": false,
        "use_mask_token": false
      },
      "predictor": {
        "_target_": "stable_worldmodel.wm.lewm.module.Predictor",
        "num_frames": 3,
        "input_dim": 192,
        "hidden_dim": 192,
        "output_dim": 192,
        "depth": 6,
        "heads": 16,
        "mlp_dim": 2048,
        "dim_head": 64,
        "dropout": 0.1,
        "emb_dropout": 0.0
      },
      "action_encoder": {
        "_target_": "stable_worldmodel.wm.lewm.module.Embedder",
        "input_dim": 10,
        "emb_dim": 192
      },
      "projector": {
        "_target_": "stable_worldmodel.wm.lewm.module.MLP",
        "input_dim": 192,
        "output_dim": 192,
        "hidden_dim": 2048,
        "norm_fn": {
          "_target_": "torch.nn.BatchNorm1d",
          "_partial_": true
        }
      },
      "pred_proj": {
        "_target_": "stable_worldmodel.wm.lewm.module.MLP",
        "input_dim": 192,
        "output_dim": 192,
        "hidden_dim": 2048,
        "norm_fn": {
          "_target_": "torch.nn.BatchNorm1d",
          "_partial_": true
        }
      }
    }"#;

    let cfg = LeWmConfig::from_stable_worldmodel_json_str(json).unwrap();

    assert_eq!(cfg.action_encoder.input_dim, 10);
    assert_eq!(cfg.action_encoder.smoothed_dim, 10);
    assert_eq!(cfg.history_size, 3);
    assert_eq!(cfg.predictor.output_dim, 192);
    assert_eq!(cfg.projector.norm, NormKind::BatchNorm1d);
    assert_eq!(cfg.pred_proj.norm, NormKind::BatchNorm1d);
}
