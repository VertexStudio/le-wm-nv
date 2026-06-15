use le_wm_nv::data::drone_racing::{
    DRONE_OBSERVATION_DIM, DroneColumns, mat3_from_rotvec, mat3_mul, normalize_channels,
    rotvec_from_mat3,
};

#[test]
fn drone_lewm_observation_is_pose_only() {
    let columns = DroneColumns::default();
    assert_eq!(DRONE_OBSERVATION_DIM, 12);
    assert_eq!(
        columns.observation,
        ["pos_world[0..3]", "rotmat_world_from_body[0..9]",]
    );
}

#[test]
fn normalizes_betaflight_channels_to_rc_action_space() {
    assert_eq!(
        normalize_channels([1500.0, 1000.0, 2000.0, 2000.0]),
        [0.0, -1.0, 1.0, 1.0]
    );
    assert_eq!(
        normalize_channels([2500.0, 500.0, 500.0, 1500.0]),
        [1.0, -1.0, 0.0, 0.0]
    );
}

#[test]
fn rotation_vector_round_trip_is_stable_for_small_gate_rollouts() {
    let rotvec = [0.05, -0.02, 0.03];
    let mat = mat3_from_rotvec(rotvec);
    let recovered = rotvec_from_mat3(mat3_mul([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], mat));
    for idx in 0..3 {
        assert!((rotvec[idx] - recovered[idx]).abs() < 1e-5);
    }
}
