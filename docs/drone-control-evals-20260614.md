# Drone World-Model Control Evaluations

Date: 2026-06-14

Checkpoint:

```text
~/.stable_worldmodel/le-wm-nv-runs/drone-state-lewm-all-data-20260612-235255/final.safetensors
```

These evaluations strengthen the fast learned-dynamics claim without relying on
gate-specific behavior. They measure:

- checkpoint quality versus training time
- action controllability under direct action sweeps
- closed-loop short-horizon local control with replanning

The checkpoint was trained with `--train-all-data`, so these are
in-distribution dynamics/control diagnostics, not held-out generalization.

## 1. Checkpoint Quality Curve

Tool:

```bash
cargo run --release --locked --bin lewm-eval-drone-checkpoints -- \
  --device cuda \
  --run-dir "$HOME/.stable_worldmodel/le-wm-nv-runs/drone-state-lewm-all-data-20260612-235255" \
  --dataset-dir "$HOME/.stable_worldmodel/le-wm-nv-data/drone-racing-autonomous-100hz" \
  --output target/drone-eval/checkpoint-curve-all-b16.json \
  --history-steps 8 \
  --horizon-steps 100 \
  --report-horizons 20,40,100 \
  --batch-size 256 \
  --max-batches 16
```

This evaluates every saved checkpoint. It computes normalized state-prediction
loss on `16` metadata-eval batches and runs an autoregressive replay on a fixed
short horizon. The default full-curve run selected replay row `55981`, a
high-motion stress row.

For continuity with the earlier dynamics report, this representative run used
row `53401`:

```bash
cargo run --release --locked --bin lewm-eval-drone-checkpoints -- \
  --device cuda \
  --run-dir "$HOME/.stable_worldmodel/le-wm-nv-runs/drone-state-lewm-all-data-20260612-235255" \
  --dataset-dir "$HOME/.stable_worldmodel/le-wm-nv-data/drone-racing-autonomous-100hz" \
  --output target/drone-eval/checkpoint-curve-row53401-b16.json \
  --steps 500,1000,2000,5000,10000,15000,20000,22300 \
  --replay-row 53401 \
  --history-steps 8 \
  --horizon-steps 100 \
  --report-horizons 20,40,100 \
  --batch-size 256 \
  --max-batches 16
```

Representative row `53401` results:

| Step | Train Time | State Loss | 0.2s Pos RMS | 0.4s Pos RMS | 1.0s Pos RMS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `500` | `14.1s` | `0.889676` | `0.659m` | `1.256m` | `2.927m` |
| `1,000` | `28.1s` | `0.670009` | `0.537m` | `1.121m` | `2.824m` |
| `2,000` | `56.4s` | `0.486630` | `0.403m` | `0.946m` | `2.727m` |
| `5,000` | `141.2s` | `0.330363` | `0.222m` | `0.526m` | `1.357m` |
| `10,000` | `282.0s` | `0.257670` | `0.291m` | `0.651m` | `1.572m` |
| `15,000` | `421.8s` | `0.244679` | `0.247m` | `0.507m` | `1.222m` |
| `20,000` | `563.4s` | `0.239547` | `0.238m` | `0.453m` | `1.115m` |
| `22,300` | `628.7s` | `0.238248` | `0.235m` | `0.423m` | `1.074m` |

Interpretation:

- The model becomes useful quickly. By step `5,000` / `141s`, the row `53401`
  0.4s RMS position error drops from `1.256m` to `0.526m`.
- Most scalar loss improvement happens early. From the first saved checkpoint
  to final, about `86%` of the state-loss drop is achieved by `141s`.
- The final checkpoint still improves the 0.4s and 1.0s rollouts, but with
  diminishing returns after the first few minutes.
- The high-motion stress row `55981` is harder: final 0.4s RMS was `1.447m`
  and final 1.0s RMS was `4.714m`.

## 2. Action Controllability

Tool:

```bash
cargo run --release --locked --bin lewm-probe-drone-actions -- \
  --device cuda \
  --dataset-dir "$HOME/.stable_worldmodel/le-wm-nv-data/drone-racing-autonomous-100hz" \
  --weights "$HOME/.stable_worldmodel/le-wm-nv-runs/drone-state-lewm-all-data-20260612-235255/final.safetensors" \
  --config "$HOME/.stable_worldmodel/le-wm-nv-runs/drone-state-lewm-all-data-20260612-235255/model-config.json" \
  --output target/drone-eval/action-controllability-h40.json \
  --row 40847 \
  --history-steps 8 \
  --horizon 40 \
  --sweep-steps 9
```

Result:

| Action | Final Pos Span | Final Z Span | Final Speed Span |
| --- | ---: | ---: | ---: |
| roll | `1.453m` | `0.505m` | `1.018m/s` |
| pitch | `4.853m` | `4.601m` | `1.366m/s` |
| throttle | `2.049m` | `1.861m` | `2.059m/s` |
| yaw | `1.693m` | `0.510m` | `3.928m/s` |

Interpretation:

- The learned model is action-conditioned. Sweeping each RC channel produces a
  different predicted state response over the same `0.4s` horizon.
- Pitch and throttle have large positional effects. Yaw strongly changes
  predicted speed. Roll produces a smaller but still visible positional span.
- This test is a controllability probe, not a controller. It answers whether
  the model reacts to actions at all.

## 3. Closed-Loop Local Control

Tool:

```bash
cargo run --release --locked --bin lewm-bench-drone-closed-loop -- \
  --device cuda \
  --dataset-dir "$HOME/.stable_worldmodel/le-wm-nv-data/drone-racing-autonomous-100hz" \
  --weights "$HOME/.stable_worldmodel/le-wm-nv-runs/drone-state-lewm-all-data-20260612-235255/final.safetensors" \
  --config "$HOME/.stable_worldmodel/le-wm-nv-runs/drone-state-lewm-all-data-20260612-235255/model-config.json" \
  --output target/drone-eval/closed-loop-local-h40-s128-i2.json \
  --row 40847 \
  --history-steps 8 \
  --horizon 40 \
  --samples 128 \
  --elites 32 \
  --keep-elites 8 \
  --iterations 2 \
  --loop-steps 30 \
  --target-distance-m 0.75 \
  --target-yaw-rad 0.35
```

This benchmark runs a short-horizon iCEM planner over simple body-frame targets.
It replans every simulated model step. The expensive path stays on CUDA through
Candle: candidate sampling, LeWM rollout, state-head decoding, and candidate
scoring. The CPU side only receives the selected action and tiny predicted
state for the next replan/report.

Result:

| Task | Expected Body Delta | Actual Body Delta | Progress | Cross Track | Planner |
| --- | ---: | ---: | ---: | ---: | ---: |
| hold | `(0.000, 0.000, 0.000)` | `(0.443, -0.417, -0.227)` | `0.000m` | `0.650m` | `189.5ms` |
| body_x | `(0.563, 0.000, 0.000)` | `(0.624, 0.001, 0.172)` | `0.624m` | `0.172m` | `184.0ms` |
| body_y | `(0.000, 0.563, 0.000)` | `(0.931, -0.121, -0.347)` | `-0.121m` | `0.994m` | `181.3ms` |
| body_z | `(0.000, 0.000, 0.563)` | `(0.876, -0.396, 0.364)` | `0.364m` | `0.961m` | `184.8ms` |
| yaw_z | `(0.000, 0.000, 0.000)` | `(0.647, -0.040, -0.005)` | `0.000m` | `0.648m` | `184.3ms` |

Interpretation:

- The body-X task is the cleanest local-control success: expected `0.563m`,
  achieved `0.624m`, with `0.172m` cross-track error.
- Body-Z shows partial progress but strong coupling into body X/Y.
- Body-Y, yaw, and hold reveal that the current model/planner pair is not yet
  a clean decoupled local controller.
- This is still useful: it separates "the model reacts to controls" from "the
  current short-horizon controller can shape all desired local behaviors."

## Current Claim After These Tests

The evidence now supports this narrower statement:

> LeWM can learn a compact action-conditioned drone dynamics model from logged
> state/action data in minutes on one GPU. The learned model is responsive to
> RC-channel actions and supports short-horizon MPC-style local control probes,
> with strongest current behavior on forward/body-X motion and weaker behavior
> on decoupled lateral, vertical, yaw, and hold tasks.

The next improvement should focus on local-control quality, not long open-loop
episode matching. Useful next probes:

- run the closed-loop benchmark after additional training
- tune the body-frame local scorer while keeping planner budget fixed
- add scripted viewer playback for these local tasks
- compare LeWM against simple linear/MLP baselines on the same three reports
