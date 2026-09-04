use candle::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use le_wm_nv::{
    data::{drone_racing::RunningStats, skyjepa::SkyJepaNormalization},
    models::skyjepa::{
        KinematicConfig, SkyJepaConfig, SkyJepaLossConfig, SkyJepaModel, SkyJepaMppiScorer,
        SkyJepaProber, SkyJepaProberConfig, SkyJepaTrackingCost, TemporalConvConfig,
        integrate_metric_rollout, integrate_metric_rollout_inference,
        skyjepa_batch_loss_with_config, skyjepa_prober_loss, so3_exp,
    },
    planner::CandidateScorer,
};

fn tiny_config() -> SkyJepaConfig {
    SkyJepaConfig {
        state_dim: 6,
        action_dim: 2,
        history_steps: 4,
        rollout_steps: 3,
        latent_dim: 6,
        state_encoder: TemporalConvConfig {
            input_dim: 6,
            channels: vec![4, 6],
            kernel_size: 3,
        },
        action_encoder: TemporalConvConfig {
            input_dim: 2,
            channels: vec![3, 4],
            kernel_size: 3,
        },
    }
}

#[test]
fn paper_configuration_matches_reported_dimensions() {
    let cfg = SkyJepaConfig::paper_derived();
    cfg.validate().unwrap();
    assert_eq!(cfg.state_dim, 18);
    assert_eq!(cfg.action_dim, 4);
    assert_eq!(cfg.history_steps, 10);
    assert_eq!(cfg.rollout_steps, 20);
    assert_eq!(cfg.latent_dim, 24);
    assert_eq!(cfg.state_encoder.channels, [8, 8, 16]);
    assert_eq!(cfg.action_encoder.channels, [4, 4, 8]);
}

#[test]
fn recursive_loss_has_finite_gradients() -> Result<()> {
    let device = Device::Cpu;
    let cfg = tiny_config();
    let varmap = VarMap::new();
    let model = SkyJepaModel::new(
        cfg.clone(),
        VarBuilder::from_varmap(&varmap, DType::F32, &device),
    )?;
    let time = cfg.history_steps + cfg.rollout_steps;
    let states = Tensor::randn(0f32, 1f32, (3, time, cfg.state_dim), &device)?;
    let actions = Tensor::randn(0f32, 1f32, (3, time, cfg.action_dim), &device)?;
    let loss = skyjepa_batch_loss_with_config(
        &model,
        &states,
        &actions,
        SkyJepaLossConfig {
            sigreg_weight: 0.02,
            sigreg_knots: 5,
            sigreg_num_proj: 8,
        },
    )?;
    assert_eq!(loss.predicted_latents.dims(), [3, 3, 6]);
    assert_eq!(loss.target_latents.dims(), [3, 3, 6]);
    assert!(loss.total_loss.to_scalar::<f32>()?.is_finite());
    let grads = loss.total_loss.backward()?;
    let trainable = varmap.all_vars();
    assert!(!trainable.is_empty());
    assert!(trainable.iter().any(|var| grads.get(var).is_some()));
    Ok(())
}

#[test]
fn candidate_rollout_batches_samples_and_windows() -> Result<()> {
    let device = Device::Cpu;
    let cfg = tiny_config();
    let vars = VarMap::new();
    let model = SkyJepaModel::new(
        cfg.clone(),
        VarBuilder::from_varmap(&vars, DType::F32, &device),
    )?;
    let states = Tensor::randn(0f32, 1f32, (2, cfg.history_steps, cfg.state_dim), &device)?;
    let action_history = Tensor::randn(
        0f32,
        1f32,
        (2, cfg.history_steps - 1, cfg.action_dim),
        &device,
    )?;
    let candidates = Tensor::randn(0f32, 1f32, (2, 5, 3, cfg.action_dim), &device)?;

    let rollout = model.rollout_candidates(&states, &action_history, &candidates)?;

    assert_eq!(rollout.dims(), &[2, 5, 3, cfg.latent_dim]);
    assert!(
        rollout
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|value| value.is_finite())
    );
    Ok(())
}

#[test]
fn so3_exponential_preserves_rotation_geometry() -> Result<()> {
    let vectors = Tensor::new(&[[0.0f32, 0.0, 0.0], [0.2, -0.1, 0.3]], &Device::Cpu)?;
    let rotations = so3_exp(&vectors)?.to_vec3::<f32>()?;
    for rotation in rotations {
        for row in rotation.iter().take(3) {
            let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4);
        }
        let dot01 = (0..3)
            .map(|col| rotation[0][col] * rotation[1][col])
            .sum::<f32>();
        assert!(dot01.abs() < 1e-4);
    }
    Ok(())
}

#[test]
fn nominal_rotor_hover_remains_stationary() -> Result<()> {
    let device = Device::Cpu;
    let mass = 1.3f32;
    let gravity = 9.81f32;
    let initial = Tensor::new(
        &[
            0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0,
        ],
        &device,
    )?
    .reshape((1, 18))?;
    let rotor_force = mass * gravity / 4.0;
    let actions = Tensor::new(&[[[rotor_force; 4]; 3]], &device)?;
    let dt = Tensor::new(&[[0.05f32; 3]], &device)?;
    let residual = Tensor::zeros((1, 3, 3), DType::F32, &device)?;
    let angular_map = Tensor::zeros((1, 3, 3, 4), DType::F32, &device)?;
    let predicted = integrate_metric_rollout(
        &initial,
        &actions,
        &dt,
        &residual,
        &angular_map,
        KinematicConfig {
            mass: mass as f64,
            gravity: gravity as f64,
            action_space: le_wm_nv::data::skyjepa::SkyJepaActionSpace::RotorForces,
            ..KinematicConfig::default()
        },
    )?
    .to_vec3::<f32>()?;
    for state in &predicted[0] {
        assert!((state[2] - 1.0).abs() < 1e-5);
        assert!(state[3..6].iter().all(|value| value.abs() < 1e-5));
    }
    Ok(())
}

#[test]
fn fused_cuda_integrator_matches_training_integrator() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let batch = 5;
    let steps = 4;
    let mut initial_values = vec![0f32; batch * 18];
    for state in initial_values.chunks_exact_mut(18) {
        state[0] = 0.3;
        state[2] = 1.2;
        state[3] = 0.1;
        state[5] = -0.05;
        state[6] = 1.0;
        state[10] = 1.0;
        state[14] = 1.0;
        state[15] = 0.02;
        state[16] = -0.03;
        state[17] = 0.01;
    }
    let initial = Tensor::from_vec(initial_values, (batch, 18), &device)?;
    let actions = Tensor::rand(2.5f32, 3.8f32, (batch, steps, 4), &device)?;
    let dt = Tensor::full(0.05f32, (batch, steps), &device)?;
    let residual = Tensor::randn(0f32, 0.1f32, (batch, steps, 3), &device)?;
    let angular_map = Tensor::randn(0f32, 0.02f32, (batch, steps, 3, 4), &device)?;
    let config = KinematicConfig {
        action_space: le_wm_nv::data::skyjepa::SkyJepaActionSpace::RotorForces,
        ..KinematicConfig::default()
    };

    let expected =
        integrate_metric_rollout(&initial, &actions, &dt, &residual, &angular_map, config)?;
    let actual = integrate_metric_rollout_inference(
        &initial,
        &actions,
        &dt,
        &residual,
        &angular_map,
        config,
    )?;
    let max_error = (actual - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(max_error < 2e-5, "fused integrator max error {max_error}");
    Ok(())
}

#[test]
fn prober_stage_freezes_latent_model_and_backpropagates_through_integration() -> Result<()> {
    let device = Device::Cpu;
    let cfg = SkyJepaConfig {
        state_dim: 18,
        action_dim: 4,
        history_steps: 3,
        rollout_steps: 2,
        latent_dim: 6,
        state_encoder: TemporalConvConfig {
            input_dim: 18,
            channels: vec![5, 6],
            kernel_size: 3,
        },
        action_encoder: TemporalConvConfig {
            input_dim: 4,
            channels: vec![4],
            kernel_size: 3,
        },
    };
    let latent_vars = VarMap::new();
    let model = SkyJepaModel::new(
        cfg.clone(),
        VarBuilder::from_varmap(&latent_vars, DType::F32, &device),
    )?;
    let prober_vars = VarMap::new();
    let prober = SkyJepaProber::new(
        SkyJepaProberConfig::paper_derived(cfg.latent_dim),
        VarBuilder::from_varmap(&prober_vars, DType::F32, &device),
    )?;
    let time = cfg.history_steps + cfg.rollout_steps;
    let normalized_states = Tensor::randn(0f32, 0.1f32, (2, time, 18), &device)?;
    let normalized_actions = Tensor::randn(0f32, 0.1f32, (2, time, 4), &device)?;
    let mut raw_states = vec![0f32; 2 * time * 18];
    for state in raw_states.chunks_exact_mut(18) {
        state[2] = 1.0;
        state[6] = 1.0;
        state[10] = 1.0;
        state[14] = 1.0;
    }
    let metric_states = Tensor::from_vec(raw_states, (2, time, 18), &device)?;
    let mut raw_actions = vec![0f32; 2 * time * 4];
    for action in raw_actions.chunks_exact_mut(4) {
        action[2] = 0.2;
    }
    let metric_actions = Tensor::from_vec(raw_actions, (2, time, 4), &device)?;
    let transition_dt = Tensor::full(0.05f32, (2, time - 1), &device)?;
    let loss = skyjepa_prober_loss(
        &model,
        &prober,
        &normalized_states,
        &normalized_actions,
        &metric_states,
        &metric_actions,
        &transition_dt,
    )?;
    assert!(loss.total_loss.to_scalar::<f32>()?.is_finite());
    let grads = loss.total_loss.backward()?;
    assert!(
        prober_vars
            .all_vars()
            .iter()
            .any(|var| grads.get(var).is_some())
    );
    assert!(
        latent_vars
            .all_vars()
            .iter()
            .all(|var| grads.get(var).is_none())
    );
    Ok(())
}

#[test]
fn control_scorer_keeps_candidate_rollout_on_device() -> Result<()> {
    let device = Device::Cpu;
    let mut cfg = SkyJepaConfig::paper_derived();
    cfg.history_steps = 3;
    cfg.rollout_steps = 2;
    let latent_vars = VarMap::new();
    let model = SkyJepaModel::new(
        cfg.clone(),
        VarBuilder::from_varmap(&latent_vars, DType::F32, &device),
    )?;
    let prober_vars = VarMap::new();
    let prober = SkyJepaProber::new(
        SkyJepaProberConfig::paper_derived(cfg.latent_dim),
        VarBuilder::from_varmap(&prober_vars, DType::F32, &device),
    )?;
    let batch = 2;
    let horizon = 2;
    let mut history_values = vec![0f32; batch * cfg.history_steps * 18];
    for state in history_values.chunks_exact_mut(18) {
        state[2] = 1.0;
        state[6] = 1.0;
        state[10] = 1.0;
        state[14] = 1.0;
    }
    let state_history = Tensor::from_vec(history_values, (batch, cfg.history_steps, 18), &device)?;
    let action_history = Tensor::zeros((batch, cfg.history_steps - 1, 4), DType::F32, &device)?;
    let reference_states = state_history
        .narrow(1, cfg.history_steps - 1, 1)?
        .broadcast_as((batch, horizon, 18))?;
    let reference_actions = Tensor::zeros((batch, horizon, 4), DType::F32, &device)?;
    let normalization = SkyJepaNormalization {
        state: RunningStats::identity(18),
        action: RunningStats::identity(4),
    };
    let scorer = SkyJepaMppiScorer::new(
        &model,
        &prober,
        &state_history,
        &action_history,
        reference_states,
        reference_actions,
        &normalization,
        0.05,
        SkyJepaTrackingCost::paper_derived(),
    )?;
    let mut candidate_values = vec![0f32; batch * 4 * horizon * 4];
    for action in candidate_values.chunks_exact_mut(4) {
        action[2] = 0.2;
    }
    let candidates = Tensor::from_vec(candidate_values, (batch, 4, horizon, 4), &device)?;

    let predictions = scorer.predict_candidates(&candidates)?;
    let scores = scorer.score_candidates(&candidates)?;

    assert_eq!(predictions.dims(), &[batch, 4, horizon, 18]);
    assert_eq!(scores.dims(), &[batch, 4]);
    assert!(
        scores
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|value| value.is_finite())
    );
    Ok(())
}
