use crate::data::drone_racing::{
    DroneFrame, dot3, mat3_from_rotvec, mat3_mul, mat3_mul_vec3, mat3_t_mul_vec3, norm3, scale3,
};

pub const DRONE_ACTION_DIM: usize = 4;

#[derive(Debug, Clone, Copy)]
pub struct DronePlantConfig {
    pub sim_hz: f32,
    pub mass: f32,
    pub gravity: f32,
    pub hover_throttle: f32,
    pub max_thrust_weight: f32,
    pub thrust_curve: f32,
    pub max_roll_rate: f32,
    pub max_pitch_rate: f32,
    pub max_yaw_rate: f32,
    pub rate_kp: f32,
    pub rate_damping: f32,
    pub linear_drag: f32,
    pub quadratic_drag: f32,
    pub body_linear_drag: [f32; 3],
    pub body_quadratic_drag: [f32; 3],
    pub roll_rate_sign: f32,
    pub pitch_rate_sign: f32,
    pub yaw_rate_sign: f32,
}

impl Default for DronePlantConfig {
    fn default() -> Self {
        Self {
            sim_hz: 1000.0,
            mass: 1.3,
            gravity: 9.81,
            hover_throttle: 0.2,
            max_thrust_weight: 5.73,
            thrust_curve: 0.0,
            max_roll_rate: 14.0,
            max_pitch_rate: 12.0,
            max_yaw_rate: 10.0,
            rate_kp: 32.0,
            rate_damping: 8.0,
            linear_drag: 0.05,
            quadratic_drag: 0.03,
            body_linear_drag: [0.0; 3],
            body_quadratic_drag: [0.0; 3],
            roll_rate_sign: 1.0,
            pitch_rate_sign: 1.0,
            yaw_rate_sign: -1.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DronePlantState {
    pub pos_world: [f32; 3],
    pub vel_world: [f32; 3],
    pub rotmat_world_from_body: [f32; 9],
    pub ang_vel_body: [f32; 3],
}

impl DronePlantState {
    pub fn initial() -> Self {
        Self {
            pos_world: [0.0, 0.0, 1.0],
            vel_world: [0.0, 0.0, 0.0],
            rotmat_world_from_body: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ang_vel_body: [0.0, 0.0, 0.0],
        }
    }

    pub fn from_frame(frame: &DroneFrame) -> Self {
        Self {
            pos_world: frame.pos_world,
            vel_world: mat3_mul_vec3(frame.rotmat_world_from_body, frame.lin_vel_body),
            rotmat_world_from_body: orthonormalize(frame.rotmat_world_from_body),
            ang_vel_body: frame.ang_vel_body,
        }
    }

    pub fn lin_vel_body(&self) -> [f32; 3] {
        mat3_t_mul_vec3(self.rotmat_world_from_body, self.vel_world)
    }

    pub fn integrate(&mut self, action: [f32; DRONE_ACTION_DIM], cfg: &DronePlantConfig, dt: f32) {
        let roll = action[0].clamp(-1.0, 1.0);
        let pitch = action[1].clamp(-1.0, 1.0);
        let throttle = action[2].clamp(0.0, 1.0);
        let yaw = action[3].clamp(-1.0, 1.0);

        let desired_rates = [
            cfg.roll_rate_sign * roll * cfg.max_roll_rate,
            cfg.pitch_rate_sign * pitch * cfg.max_pitch_rate,
            cfg.yaw_rate_sign * yaw * cfg.max_yaw_rate,
        ];
        let angular_accel = [
            (desired_rates[0] - self.ang_vel_body[0]) * cfg.rate_kp
                - self.ang_vel_body[0] * cfg.rate_damping,
            (desired_rates[1] - self.ang_vel_body[1]) * cfg.rate_kp
                - self.ang_vel_body[1] * cfg.rate_damping,
            (desired_rates[2] - self.ang_vel_body[2]) * cfg.rate_kp
                - self.ang_vel_body[2] * cfg.rate_damping,
        ];
        self.ang_vel_body = [
            self.ang_vel_body[0] + angular_accel[0] * dt,
            self.ang_vel_body[1] + angular_accel[1] * dt,
            self.ang_vel_body[2] + angular_accel[2] * dt,
        ];
        let max_rate = cfg
            .max_roll_rate
            .max(cfg.max_pitch_rate)
            .max(cfg.max_yaw_rate)
            * 1.5;
        self.ang_vel_body = clamp_length3(self.ang_vel_body, max_rate);

        let delta_rot = mat3_from_rotvec(scale3(self.ang_vel_body, dt));
        self.rotmat_world_from_body =
            orthonormalize(mat3_mul(self.rotmat_world_from_body, delta_rot));

        let thrust_input = shaped_throttle(throttle, cfg.thrust_curve);
        let hover_input = shaped_throttle(cfg.hover_throttle, cfg.thrust_curve).max(1e-4);
        let thrust_weight = (thrust_input / hover_input).clamp(0.0, cfg.max_thrust_weight);
        let thrust = cfg.mass * cfg.gravity * thrust_weight;
        let body_up_world = mat3_mul_vec3(self.rotmat_world_from_body, [0.0, 0.0, 1.0]);
        let speed = norm3(self.vel_world);
        let mut drag = [
            self.vel_world[0] * cfg.linear_drag + self.vel_world[0] * speed * cfg.quadratic_drag,
            self.vel_world[1] * cfg.linear_drag + self.vel_world[1] * speed * cfg.quadratic_drag,
            self.vel_world[2] * cfg.linear_drag + self.vel_world[2] * speed * cfg.quadratic_drag,
        ];
        if cfg.body_linear_drag.iter().any(|v| *v != 0.0)
            || cfg.body_quadratic_drag.iter().any(|v| *v != 0.0)
        {
            let vel_body = mat3_t_mul_vec3(self.rotmat_world_from_body, self.vel_world);
            let body_drag = [
                vel_body[0] * cfg.body_linear_drag[0]
                    + vel_body[0] * vel_body[0].abs() * cfg.body_quadratic_drag[0],
                vel_body[1] * cfg.body_linear_drag[1]
                    + vel_body[1] * vel_body[1].abs() * cfg.body_quadratic_drag[1],
                vel_body[2] * cfg.body_linear_drag[2]
                    + vel_body[2] * vel_body[2].abs() * cfg.body_quadratic_drag[2],
            ];
            let body_drag_world = mat3_mul_vec3(self.rotmat_world_from_body, body_drag);
            drag = [
                drag[0] + body_drag_world[0],
                drag[1] + body_drag_world[1],
                drag[2] + body_drag_world[2],
            ];
        }
        let accel_world = [
            body_up_world[0] * (thrust / cfg.mass) - drag[0],
            body_up_world[1] * (thrust / cfg.mass) - drag[1],
            body_up_world[2] * (thrust / cfg.mass) - cfg.gravity - drag[2],
        ];
        self.vel_world = [
            self.vel_world[0] + accel_world[0] * dt,
            self.vel_world[1] + accel_world[1] * dt,
            self.vel_world[2] + accel_world[2] * dt,
        ];
        self.pos_world = [
            self.pos_world[0] + self.vel_world[0] * dt,
            self.pos_world[1] + self.vel_world[1] * dt,
            self.pos_world[2] + self.vel_world[2] * dt,
        ];

        if self.pos_world[2] < 0.04 {
            self.pos_world[2] = 0.04;
            if self.vel_world[2] < 0.0 {
                self.vel_world[2] = 0.0;
            }
            self.vel_world[0] *= 0.985;
            self.vel_world[1] *= 0.985;
        }
    }
}

pub fn config_summary(cfg: &DronePlantConfig) -> String {
    format!(
        "hover={:.5} thrust_w={:.3} thrust_curve={:.3} rates=[{:.3},{:.3},{:.3}] signs=[{:+.0},{:+.0},{:+.0}] kp={:.3} damp={:.3} drag=[{:.4},{:.4}] body_drag=[{:.3},{:.3},{:.3}] body_qdrag=[{:.3},{:.3},{:.3}]",
        cfg.hover_throttle,
        cfg.max_thrust_weight,
        cfg.thrust_curve,
        cfg.max_roll_rate,
        cfg.max_pitch_rate,
        cfg.max_yaw_rate,
        cfg.roll_rate_sign,
        cfg.pitch_rate_sign,
        cfg.yaw_rate_sign,
        cfg.rate_kp,
        cfg.rate_damping,
        cfg.linear_drag,
        cfg.quadratic_drag,
        cfg.body_linear_drag[0],
        cfg.body_linear_drag[1],
        cfg.body_linear_drag[2],
        cfg.body_quadratic_drag[0],
        cfg.body_quadratic_drag[1],
        cfg.body_quadratic_drag[2],
    )
}

fn shaped_throttle(throttle: f32, curve: f32) -> f32 {
    let throttle = throttle.clamp(0.0, 1.0);
    let curve = curve.clamp(-0.95, 0.95);
    ((1.0 - curve) * throttle + curve * throttle * throttle).max(0.0)
}

pub fn rotmat_distance_rad(lhs: [f32; 9], rhs: [f32; 9]) -> f32 {
    let rel = mat3_mul(crate::data::drone_racing::mat3_transpose(lhs), rhs);
    let trace = (rel[0] + rel[4] + rel[8]).clamp(-1.0, 3.0);
    let cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
    cos_theta.acos()
}

fn clamp_length3(value: [f32; 3], max_len: f32) -> [f32; 3] {
    let len = norm3(value);
    if len > max_len && len > 1e-6 {
        scale3(value, max_len / len)
    } else {
        value
    }
}

fn orthonormalize(m: [f32; 9]) -> [f32; 9] {
    let x0 = [m[0], m[3], m[6]];
    let y0 = [m[1], m[4], m[7]];
    let x = normalize_or(x0, [1.0, 0.0, 0.0]);
    let y_reject = [
        y0[0] - x[0] * dot3(y0, x),
        y0[1] - x[1] * dot3(y0, x),
        y0[2] - x[2] * dot3(y0, x),
    ];
    let y = normalize_or(y_reject, [0.0, 1.0, 0.0]);
    let z = crate::data::drone_racing::cross3(x, y);
    [x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]]
}

fn normalize_or(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len = norm3(value);
    if len > 1e-6 {
        scale3(value, 1.0 / len)
    } else {
        fallback
    }
}
