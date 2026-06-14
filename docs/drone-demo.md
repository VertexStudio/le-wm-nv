# Drone LeWM Demo

Date: 2026-06-14

This document is the current entry point for the drone world-model demo. It
summarizes what has been built, which artifacts are local, how to run the
viewer/simulator, and which measurements should be treated as runtime
optimization results versus behavior-changing planner experiments.

## Goal

The drone work tests whether this repo can learn a usable dynamics world model
from real drone racing logs and use that learned model for closed-loop behavior.
This is not a hand-written physics simulator. The hot path uses the repo-native
LeWM `WorldModel` with vector observations, RC-channel actions, and a learned
state-delta head.

The gate loop is one evaluation task for the model. It is not the only goal of
the project.

## Data

Imported dataset:

```text
~/.stable_worldmodel/le-wm-nv-data/drone-racing-autonomous-100hz
```

Current imported data summary:

- Rows: `58,239`
- Episodes/flights: `18`
- Sample rate: `100 Hz`
- Observation dim: `20`
- Action dim: `4`
- State-delta dim: `13`
- Valid all-data training windows: `57,195`

Observation schema:

- `rotmat_world_from_body[0..9]`
- `lin_vel_body[0..3]`
- `ang_vel_body[0..3]`
- `vbat`
- `previous_channels_norm[0..4]`

Action schema:

- roll `[-1, 1]`
- pitch `[-1, 1]`
- throttle `[0, 1]`
- yaw `[-1, 1]`

Target delta schema:

- `delta_pos_body[0..3]`
- `delta_rot_body[0..3]`
- `next_lin_vel_body[0..3]`
- `next_ang_vel_body[0..3]`
- `delta_vbat`

`vbat` is intentionally kept in the model input/target schema.

## Model

The drone model uses repo-native modular observation support:

- Observation encoder: `VectorMlp`, not vision.
- Action encoder: same LeWM action encoder pattern.
- Predictor: AdaLN-conditioned autoregressive LeWM predictor.
- Output: latent prediction projection plus `VectorDelta` state head.

Current checkpoint:

```text
~/.stable_worldmodel/le-wm-nv-runs/drone-state-lewm-all-data-20260612-235255/final.safetensors
```

Checkpoint size:

- Weights: about `22 MB`
- Optimizer state: about `44 MB`
- Rough parameter count: about `5.5M` F32 parameters

Config:

```text
~/.stable_worldmodel/le-wm-nv-runs/drone-state-lewm-all-data-20260612-235255/model-config.json
```

## Training Run

The main run is documented in detail in:

```text
docs/drone-all-data-training-20260612-235255.md
```

Summary:

- Run ID: `drone-state-lewm-all-data-20260612-235255`
- Training data: all valid sliding windows
- Epochs: `100`
- Batch size: `256`
- Batches per epoch: `223`
- Optimizer steps: `22,300`
- History steps: `8`
- Horizon steps: `50`
- Sequence steps: `59`
- Learning rate: `1e-4`
- Weight decay: `1e-2`
- Elapsed training time: `628.8 s`
- Mean step rate: `35.49 steps/s`
- Mean step time: `28.18 ms/step`
- GPU observed during run: RTX 4090 at about `99-100%` utilization

Final logged loss at step `22,300`:

- total: `0.2661571`
- state_prediction: `0.2043576`
- temporal_alignment: `0.010423715`
- temporal_straightening: `-0.54001254`

Resume note: `lewm-train-drone --resume-dir ... --epochs N` treats `N` as the
total target epoch count, not additional epochs. To train 200 more epochs after
the 100-epoch run, use `--epochs 300`.

## Behavior-Preserving Runtime Optimization

Runtime optimization metrics must hold planner settings fixed. Do not mix these
numbers with lower planner horizon, fewer samples, fewer iterations, or changed
control stride.

Current behavior-preserving LeWM/Candle optimizations:

- Cache non-learned causal attention bias tensors.
- Use a fixed-shape rollout path that projects only the last autoregressive
  token needed for the drone state head.

Fixed rollout benchmark:

```bash
cargo run --release --locked --bin lewm-bench-drone-rollout -- \
  --device cuda \
  --dataset-dir "$HOME/.stable_worldmodel/le-wm-nv-data/drone-racing-autonomous-100hz" \
  --weights "$HOME/.stable_worldmodel/le-wm-nv-runs/drone-state-lewm-all-data-20260612-235255/final.safetensors" \
  --config "$HOME/.stable_worldmodel/le-wm-nv-runs/drone-state-lewm-all-data-20260612-235255/model-config.json" \
  --row 40847 \
  --samples 528 \
  --horizon 40 \
  --warmup 4 \
  --iterations 30 \
  --output target/bench/drone-rollout-h40-s528-candle-future-head.json
```

Measured fixed-setting result:

- Baseline total mean: `0.207182 s`
- Optimized total mean: `0.194838 s`
- Latency reduction: about `5.96%`
- Throughput increase: about `6.33%`
- Candidate rollouts/s: `2548.5 -> 2709.9`
- Checksum remained `40992.58`

Memory tradeoff:

- Cached causal bias for this model is about `82 KB`.
- The model weights are about `22 MB`.
- Planner/runtime activation workspace is the main memory pressure, not the
  cached masks.

## Gate-Loop Planner Demo

The gate-loop planner uses the learned LeWM model for rollout and scores the
predicted state deltas on CUDA. The current default is iCEM with settings that
have been verified to complete one lap on the current checkpoint.

All-gates command:

```bash
cargo run --release --locked --bin lewm-plan-drone-gates -- \
  --device cuda \
  --dataset-dir "$HOME/.stable_worldmodel/le-wm-nv-data/drone-racing-autonomous-100hz" \
  --weights "$HOME/.stable_worldmodel/le-wm-nv-runs/drone-state-lewm-all-data-20260612-235255/final.safetensors" \
  --config "$HOME/.stable_worldmodel/le-wm-nv-runs/drone-state-lewm-all-data-20260612-235255/model-config.json" \
  --row 40847 \
  --loop-steps 800 \
  --laps 1 \
  --output target/drone-plans/defaults-full-lap-passcheck.json
```

Verified result for `target/drone-plans/defaults-full-lap-passcheck.json`:

- Planner: iCEM
- Horizon: `40`
- Samples: `512`
- Elites: `64`
- Keep elites: `16`
- Iterations: `4`
- Control stride: `5`
- Completed laps: `1`
- Executed model steps: `645`
- Total replans: `129`
- Total planner time: `96.947329 s`
- Total planner evals: `270,384`
- Planner throughput: `2788.98 evals/s`

Gate pass points:

- `gate1`: step `40`
- `gate2`: step `150`
- `gate3`: step `165`
- `gate4`: step `525`
- `gate5`: step `550`
- `gate6`: step `600`
- `gate7`: step `645`

Planner-budget experiments are controller changes, not runtime optimizations.
The aggressive `horizon=25`, `samples=256`, `iterations=2`, `control_stride=8`
profile was much faster but only passed `gate1` and remained on `gate2`. A
half-sample `horizon=40`, `samples=256`, `iterations=4`, `control_stride=5`
profile passed through `gate5` but did not complete the lap within 800 steps.

## Replay Viewer

Use the replay viewer for saved planner JSON:

```bash
cargo run --release --locked -p lewm-drone-viewer -- \
  --replay target/drone-plans/defaults-full-lap-passcheck.json
```

The viewer renders:

- drone pose
- predicted path/trail
- floor grid and world axes
- gate targets
- active carrot/target metadata
- action bars
- planner telemetry

## Interactive LeWM Simulator

`lewm-drone-sim` is an interactive Bevy simulator driven by the learned LeWM
model. Each sim step:

1. Reads live RC-channel input.
2. Normalizes action using dataset stats.
3. Runs one LeWM autoregressive model step on CUDA.
4. Decodes one state delta through the learned state head.
5. Copies the tiny predicted delta/state back for Bevy rendering.

Run:

```bash
cargo run --release --locked -p lewm-drone-viewer --bin lewm-drone-sim -- \
  --dataset-dir "$HOME/.stable_worldmodel/le-wm-nv-data/drone-racing-autonomous-100hz" \
  --weights "$HOME/.stable_worldmodel/le-wm-nv-runs/drone-state-lewm-all-data-20260612-235255/final.safetensors" \
  --config "$HOME/.stable_worldmodel/le-wm-nv-runs/drone-state-lewm-all-data-20260612-235255/model-config.json" \
  --row 40847
```

Controls:

- `W/S`: pitch
- `A/D`: roll
- `Q/E`: yaw
- `R/F`: throttle
- `Z`: zero roll/pitch/yaw
- `X`: return throttle to trim
- `P`: pause
- Backspace: reset
- `[` / `]`: follow camera distance
- `3` / `4`: follow camera height
- `1` / `2`: camera spring strength

The simulator uses a spring follow camera so WASD stays reserved for piloting.

## Current Caveats

- The model is a learned dynamics model, not a physics engine.
- Good behavior should be expected mainly near the action/state distribution
  present in the dataset.
- Gate-loop success is one task-specific probe, not proof of full drone
  controllability.
- Planner budget changes can change behavior and must be reported separately
  from runtime optimizations.
- The interactive simulator is a qualitative closed-loop probe. It is meant to
  expose model dynamics failures quickly before deciding whether more training
  is useful.

## Useful Next Comparisons

- Run `lewm-drone-sim` before and after any continued training checkpoint.
- Compare the all-gates planner artifact at fixed planner settings.
- Add per-dimension one-step and autoregressive state-delta error summaries.
- Track simulator step time and closed-loop drift for fixed scripted controls.
- If training continues for 200 more epochs, document it as a new run instead
  of overwriting this baseline.
