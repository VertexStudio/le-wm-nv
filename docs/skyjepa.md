# Native SkyJEPA

This repository supports SkyJEPA as a second model family alongside LeWM.
LeWM's models, checkpoint-compatible image runtime, vector extensions,
trainers, planners, and simulator remain available. SkyJEPA has an independent
state/action schema and independent checkpoints; there is no forced conversion
between the two model families.

The implementation is a clean-room Rust/Candle interpretation of the
[SkyJEPA paper](https://arxiv.org/html/2606.23444) and is CUDA-oriented from
training through candidate rollout and MPPI scoring. The
[authors' repository](https://github.com/arplaboratory/SkyJEPA) did not contain
model, training, data, or controller source when this implementation was
written, so paper-omitted details are explicit below rather than represented as
upstream parity.

## Implemented pipeline

Stage one learns a latent dynamics model:

```text
state history [B,10,18] -> causal residual TCN [8,8,16] -> linear -> latent24
action windows [B,20,10,4] -> causal residual TCN [4,4,8]
latent24 + action8 -> one-layer GRU unrolled for 20 steps -> predicted latent24
loss = multi-step latent MSE + 0.02 * SIGReg
```

Stage two freezes the complete stage-one path. A small prober maps each
predicted latent to residual linear acceleration `[3]` and a latent-dependent
angular action map `[3,4]`. A differentiable kinematic integrator propagates
position, inertial velocity, rotation on SO(3), and body angular velocity. Only
the prober parameters receive gradients in this stage.

The control path encodes a state history once, flattens all sampled rolling
action windows into one TCN batch, unrolls the GRU over the short horizon, and
passes all candidates through the prober/integrator without a host-side sample
loop. The native MPPI implementation supports a per-action noise vector and
persists a shifted warm start between control cycles. Prober training uses the
fully differentiable Candle integrator; control uses an equivalent fused CUDA
forward kernel that advances one complete candidate trajectory per thread.

The paper-derived defaults are:

| Setting | Value |
| --- | ---: |
| State / action dimensions | 18 / 4 |
| State / action TCN channels | `[8,8,16]` / `[4,4,8]` |
| Latent / GRU hidden size | 24 |
| History / training rollout | 10 / 20 at 20 Hz |
| Training epochs / batch | 50 / 2048 |
| SIGReg knots / weight | 17 / 0.02 |
| AdamW decay / gradient clip | `1e-5` / `0.5` |
| LR schedule | 4k warmup to `5e-3`, 20k cosine to `1e-4` |
| MPPI horizon / samples / dt | 15 / 512 / 0.05 s |
| MPPI temperature | `1e-4` |
| MPPI action noise scales | `[0.60,0.15,0.15,0.05]` |
| State group weights | `[400,40,20,20]` |
| Per-action weights | `[0.01,0.05,0.05,0.10]` |

## Canonical dataset

`lewm-generate-skyjepa` writes schema version 1:

```text
metadata.json
data.h5
  state       float32 [N,18]
  action      float32 [N,4]
  episode_idx int64   [N]
  step_idx    int64   [N]
  dt          float32 [N,1]
  domain_idx  int64   [N]
domains.json
```

State order is position in world `[3]`, velocity in world `[3]`, row-major
world-from-body rotation matrix `[9]`, and body angular velocity `[3]`. The
canonical action is four commanded rotor forces in newtons. `metadata.json`
declares the action space, dimensions, sample rate, row/episode counts, and
schema version. Normalization is fit only on training episodes, and deterministic
episode splits are 80/10/10.

The loader can also compose state18 from the existing LeWM drone import. That
legacy route must explicitly select `--action-space body-rates-throttle`; it is
an adaptation route and not the canonical SkyJEPA rotor-force model.

Generate a small dataset first, then scale to the paper's 500 domains and
20,000 ten-second trajectories:

```bash
cargo run --release --locked --bin lewm-generate-skyjepa -- \
  --output-dir "$HOME/.stable_worldmodel/le-wm-nv-data/skyjepa-domain-randomized-20hz" \
  --domains 500 \
  --trajectories 20000 \
  --duration-seconds 10
```

Generation is parallel across trajectories. Each domain samples the reported
mass, inertia, motor-time-constant, drag, thrust, and torque ranges. The plant
includes rotor allocation, first-order motor response, rigid-body angular
dynamics, body-axis drag, and SO(3) attitude integration.

## Train, evaluate, and simulate

Train both stages on CUDA:

```bash
cargo run --release --locked --bin lewm-train-skyjepa -- \
  --dataset-dir "$HOME/.stable_worldmodel/le-wm-nv-data/skyjepa-domain-randomized-20hz" \
  --output-dir "$HOME/.stable_worldmodel/le-wm-nv-runs/skyjepa-drone-state18-20hz" \
  --stage both \
  --action-space rotor-forces
```

The run directory contains model, prober, dataset, normalization, and split
configuration JSON; latent and prober safetensors; and JSONL training metrics.

Report latent RMSE and metric position, velocity, attitude-geodesic, and angular
velocity RMSE at each of the 20 open-loop horizons:

```bash
cargo run --release --locked --bin lewm-eval-skyjepa -- \
  --dataset-dir "$HOME/.stable_worldmodel/le-wm-nv-data/skyjepa-domain-randomized-20hz" \
  --checkpoint-dir "$HOME/.stable_worldmodel/le-wm-nv-runs/skyjepa-drone-state18-20hz"
```

Exercise the learned model in closed-loop MPPI against a held-out randomized
rotor-force plant:

```bash
cargo run --release --locked --bin lewm-sim-skyjepa -- \
  --checkpoint-dir "$HOME/.stable_worldmodel/le-wm-nv-runs/skyjepa-drone-state18-20hz" \
  --samples 512 \
  --horizon 15 \
  --randomize-domain
```

Inference loads immutable safetensors directly rather than creating trainable
variables. Candidate tensors, latent rollouts, metric rollouts, costs, MPPI
weights, and selected actions remain on the Candle CUDA path; only the executed
action and final reports are materialized on the host.

The simulator reports both cold-start and steady planning latency. NVRTC and
library initialization make the first cycle intentionally visible in
`mean_plan_ms`/`max_plan_ms`; `steady_mean_plan_ms`, p50, and p95 are the useful
control-loop measures after that first compile cycle.

Local implementation smoke benchmark on 2026-09-03, RTX 4090, release build,
batch 1, 512 MPPI samples, horizon 15, and 30 closed-loop cycles: steady mean
`5.92 ms`, p50 `5.79 ms`, p95 `7.18 ms` (`168.8 Hz`). The first NVRTC/library
initialization cycle was `299.7 ms` and is excluded only from the steady mean.
The benchmark used a one-step smoke-test checkpoint, so it validates execution
latency and tensor shape, not learned control quality.

## Explicit clean-room choices

The following details are not specified by the paper or released source:

- TCN blocks use two causal GELU Conv1d layers per level, residual projections,
  kernel size 3, exponentially increasing dilation, and no dropout.
- The prober is a two-hidden-layer GELU MLP with width 32 and a small-output
  initialization. The paper specifies its outputs but not its internal layers.
- SIGReg uses 64 projection directions. The paper gives 17 knots and weight
  0.02, does not report projection count, and says the result is insensitive to
  that count. This bounded choice avoids a very large `[T,B,M,17]` training
  allocation at batch 2048.
- Prober training defaults to 50 epochs because its epoch count is not reported.
- Grouped attitude control cost uses squared rotation-matrix Frobenius error;
  the paper reports the group weight but not the attitude error formula.
- The paper defines actions as rotor forces but later describes PX4 deployment
  using collective-thrust/body-rate commands. The canonical implementation
  follows the mathematical rotor-force definition; the legacy loader retains a
  separately tagged body-rate/throttle option.
- The data generator uses a random-Fourier approximation of the reported
  multi-periodic GP references and a geometric PD tracker with smooth action
  excitation. It is dynamically valid and uses the paper's domain ranges, but
  it is not claimed to reproduce the authors' unreleased NMPC/MPPI data
  synthesis implementation.
- Arm length is randomized ±20% because the methodology lists it as a domain
  parameter but the parameter table omits its range.

Python is not part of the SkyJEPA runtime or training path. The repository's
existing `uv`-locked Python environment remains available for LeWM parity and
analysis tooling; this implementation adds no Python sidecar.
