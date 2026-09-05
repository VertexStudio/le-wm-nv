use candle::{DType, Device, Tensor};
use le_wm_nv::{
    models::skyjepa::{SkyJepaNominalScorer, SkyJepaTrackingCost, nominal_physics_rollout},
    planner::CandidateScorer,
    skyjepa_sim::{SkyJepaDomain, SkyJepaRotorState},
};

#[test]
fn fused_nominal_candidates_match_the_cpu_rigid_body_plant() -> anyhow::Result<()> {
    let cuda = Device::new_cuda(0)?;
    for domain in [SkyJepaDomain::default(), SkyJepaDomain::sample(918)] {
        let initial = SkyJepaRotorState {
            velocity: [0.3, -0.2, 0.1],
            angular_velocity: [0.05, -0.02, 0.03],
            ..SkyJepaRotorState::hover()
        };
        let batch = 5;
        let horizon = 30;
        let hover = domain.mass * domain.gravity / (4.0 * domain.thrust_scale);
        let initial = Tensor::from_vec(
            initial.as_state18().repeat(batch),
            (batch, 18),
            &Device::Cpu,
        )?;
        let motors = Tensor::full(hover, (batch, 4), &Device::Cpu)?;
        let action_values = (0..batch * horizon * 4)
            .map(|i| hover + 0.02 * (i as f32 * 0.71).sin())
            .collect::<Vec<_>>();
        let actions = Tensor::from_vec(action_values, (batch, horizon, 4), &Device::Cpu)?;
        let expected = nominal_physics_rollout(&initial, &actions, &motors, 0.05, 10, domain)?;
        let actual = nominal_physics_rollout(
            &initial.to_device(&cuda)?,
            &actions.to_device(&cuda)?,
            &motors.to_device(&cuda)?,
            0.05,
            10,
            domain,
        )?;
        assert!(actual.device().is_cuda());
        assert_eq!(actual.dtype(), DType::F32);
        let error = (&expected - &actual.to_device(&Device::Cpu)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            error < 2e-3,
            "CPU/CUDA nominal rollout maximum state error {error}"
        );
    }
    Ok(())
}

#[test]
fn nominal_rollout_rejects_incompatible_shapes_and_steps() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let initial = Tensor::zeros((1, 18), DType::F32, &device)?;
    let actions = Tensor::zeros((1, 3, 4), DType::F32, &device)?;
    let motors = Tensor::zeros((1, 4), DType::F32, &device)?;
    assert!(
        nominal_physics_rollout(
            &initial,
            &actions,
            &motors,
            0.05,
            0,
            SkyJepaDomain::default()
        )
        .is_err()
    );
    assert!(
        nominal_physics_rollout(
            &initial,
            &actions,
            &motors,
            f32::NAN,
            10,
            SkyJepaDomain::default()
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn nominal_scorer_uses_observed_trim_and_the_shared_tracking_cost() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let state = SkyJepaRotorState::hover().as_state18();
    let horizon = 15;
    let nominal = SkyJepaDomain::default();
    let hover = nominal.mass * nominal.gravity / 4.0;
    for trim in [1.0f32, 2.0] {
        let scorer = SkyJepaNominalScorer::new(
            Tensor::from_vec(state.to_vec(), (1, 18), &device)?,
            Tensor::full(hover, (1, 4), &device)?,
            Tensor::from_vec(state.repeat(horizon), (1, horizon, 18), &device)?,
            Tensor::full(hover * trim, (1, horizon, 4), &device)?,
            0.05,
            trim,
            SkyJepaTrackingCost::paper_derived(),
        )?;
        let values = [0.5f32, 0.0, -0.5]
            .into_iter()
            .flat_map(|delta| vec![hover * trim + delta; horizon * 4])
            .collect::<Vec<_>>();
        let actions = Tensor::from_vec(values, (1, 3, horizon, 4), &device)?;
        let scores = scorer.score_candidates(&actions)?.to_vec2::<f32>()?;
        assert!(scores[0].iter().all(|value| value.is_finite()));
        assert!(scores[0][1] < scores[0][0] && scores[0][1] < scores[0][2]);
        assert!(
            scores[0][1] < 1e-6,
            "calibrated hover should have effectively zero cost"
        );
    }
    Ok(())
}
