use le_wm_nv::skyjepa_sim::{SkyJepaDomain, SkyJepaRotorPlant, SkyJepaRotorState};

#[test]
fn sampled_domains_stay_in_reported_ranges() {
    let nominal = SkyJepaDomain::default();
    for seed in 0..100 {
        let domain = SkyJepaDomain::sample(seed);
        domain.validate().unwrap();
        assert!((nominal.mass * 0.5..=nominal.mass * 1.5).contains(&domain.mass));
        assert!((0.01..=0.1).contains(&domain.motor_time_constant));
        assert!(domain.drag.iter().all(|value| (0.1..=0.5).contains(value)));
        assert!((0.5..=1.5).contains(&domain.thrust_scale));
        assert!((0.5..=1.5).contains(&domain.torque_scale));
    }
}

#[test]
fn nominal_rotor_plant_hovers() {
    let mut plant =
        SkyJepaRotorPlant::new(SkyJepaDomain::default(), SkyJepaRotorState::hover()).unwrap();
    let action = plant.nominal_hover_action();
    for _ in 0..1000 {
        plant.step(action, 0.001);
    }
    let state = plant.state();
    assert!((state.position[2] - 1.0).abs() < 1e-3);
    assert!(state.velocity.iter().all(|value| value.abs() < 1e-3));
    assert!(
        state
            .angular_velocity
            .iter()
            .all(|value| value.abs() < 1e-3)
    );
}
