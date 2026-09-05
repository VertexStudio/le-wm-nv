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
an optional shifted-residual warm start (`--warm-start shifted-residual`).
The default is `fresh-prior`; carried residuals are bounded to +/-2 N per rotor,
shifted only after executing an action, and cleared on reset. MPPI optimizes learned
corrections around a trim-aware geometric flight prior instead of rediscovering
basic stabilization from noise on every cycle. Prober training uses the fully
differentiable Candle integrator; control uses an equivalent fused CUDA forward
kernel that advances one complete candidate trajectory per thread.

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
  state           float32 [N,18]
  action          float32 [N,4]
  reference_state float32 [N,18]
  motor_force     float32 [N,4]
  episode_idx     int64   [N]
  step_idx        int64   [N]
  dt              float32 [N,1]
  domain_idx      int64   [N]
domains.json
```

State order is position in world `[3]`, velocity in world `[3]`, row-major
world-from-body rotation matrix `[9]`, and body angular velocity `[3]`. The
canonical action is four commanded rotor forces in newtons. `metadata.json`
declares the action space, dimensions, sample rate, row/episode counts, and
schema version. Normalization is fit only on the training partition. The trainer
defaults to deterministic, domain-disjoint 80/10/10 splits (`--split-by domains`);
`--split-by episodes` is an explicit diagnostic alternative that can share
physical domains across partitions. The model/controller rate is fixed at
20 Hz. Higher-rate raw data must be explicitly strided to this rate.

The loader can also compose state18 from the existing LeWM drone import. That
legacy route must explicitly select `--action-space body-rates-throttle`; it is
an adaptation route and not the canonical SkyJEPA rotor-force model.

Generate a small dataset first, then scale to the paper's 500 domains and
20,000 ten-second trajectories. The commands below deliberately use a shell
variable other than `HOME` so they are safe to paste into automation:

```bash
SKYJEPA_DATA="$HOME/.stable_worldmodel/le-wm-nv-data/skyjepa-domain-randomized-20hz"
cargo run --release --locked --bin lewm-generate-skyjepa -- \
  --output-dir "$SKYJEPA_DATA" \
  --domains 500 \
  --trajectories 20000 \
  --duration-seconds 10

cargo run --release --locked --bin lewm-audit-skyjepa -- \
  --dataset-dir "$SKYJEPA_DATA" \
  --output "$SKYJEPA_DATA/audit.json"
```

Generation is parallel across trajectories. Each domain samples the reported
mass, inertia, motor-time-constant, drag, thrust, and torque ranges. The plant
includes rotor allocation, first-order motor response, rigid-body angular
dynamics, body-axis drag, and SO(3) attitude integration.

The audit is a required training gate. It validates schema and shapes, finite
values, episode/domain indices, sample interval, SO(3) orthogonality and
determinant, reference tracking, ground contact, command saturation, per-rotor
variance, rotor-about-collective variance, and within-episode command deltas.
Audit version 2 also requires both hover and moving trajectories and adequate
command excitation in every domain. Generator scheduling changes flight type
on successive visits to a domain, avoiding domain/trajectory-type aliasing.
It also records a SHA-256 fingerprint over the dataset artifact. Training
refuses a failed, missing, or stale audit unless `--skip-audit` is explicitly
used for a developer smoke test.

## Train, evaluate, and simulate

Train both stages on CUDA:

```bash
SKYJEPA_DATA="$HOME/.stable_worldmodel/le-wm-nv-data/skyjepa-domain-randomized-20hz"
SKYJEPA_RUN="$HOME/.stable_worldmodel/le-wm-nv-runs/skyjepa-drone-state18-20hz"
cargo run --release --locked --bin lewm-train-skyjepa -- \
  --dataset-dir "$SKYJEPA_DATA" \
  --output-dir "$SKYJEPA_RUN" \
  --stage both \
  --action-space rotor-forces
```

For operational runs, the two stages may instead use independent step budgets
and output directories. A standalone prober run copies the frozen latent model
into its directory, producing a self-contained deployable checkpoint:

```bash
SKYJEPA_DATA="$HOME/.stable_worldmodel/le-wm-nv-data/skyjepa-domain-randomized-20hz"
SKYJEPA_LATENT="$HOME/.stable_worldmodel/le-wm-nv-runs/skyjepa-latent"
SKYJEPA_RUN="$HOME/.stable_worldmodel/le-wm-nv-runs/skyjepa-prober"
cargo run --release --locked --bin lewm-train-skyjepa -- \
  --dataset-dir "$SKYJEPA_DATA" --output-dir "$SKYJEPA_LATENT" \
  --stage latent --batch-size 2048 --latent-max-steps 20000
cargo run --release --locked --bin lewm-train-skyjepa -- \
  --dataset-dir "$SKYJEPA_DATA" --output-dir "$SKYJEPA_RUN" \
  --stage prober --batch-size 2048 --prober-max-steps 20000 \
  --latent-checkpoint "$SKYJEPA_LATENT"
```

Deployable packages use version-2 `checkpoint.json` and content-addressed,
immutable safetensors in `objects/`. The manifest binds architecture,
normalization, timestep, action semantics, prober physics, latent ancestry and
weight hashes. Controller/evaluator loading rejects incomplete, changed or
loose legacy checkpoints. Historical artifacts remain untouched; these new
tools do not automatically migrate them.

Each training snapshot is a complete immutable generation under `snapshots/`:
weights, optimizer, progress, numerical contract and metrics byte boundary.
Atomic `{stage}-current.json` publication selects a complete generation; the
previous one remains recoverable. Best-validation publication uses the same
package contract. Restart an interrupted explicit stage with identical training
arguments plus `--resume`. Changes to data, preprocessing, parent latent model,
physics, loss, seed, batch size or optimizer schedule are rejected before any
existing run is modified. A completed-stage resume can recover package export.

`--stop-after-step N` pauses an explicit stage without changing its training
target or adding artificial validation. Step-addressed SIGReg randomness and
metrics-tail recovery make resumed updates bit-exact on the tested CUDA stack,
including interruption across validation/epoch boundaries. This is not a
cross-hardware or cross-library-version determinism guarantee.

Report latent RMSE and metric position, velocity, attitude-geodesic, and angular
velocity RMSE at each requested open-loop horizon:

```bash
cargo run --release --locked --bin lewm-eval-skyjepa -- \
  --dataset-dir "$SKYJEPA_DATA" \
  --checkpoint-dir "$SKYJEPA_RUN" \
  --rollout-steps 60 \
  --output "$SKYJEPA_RUN/eval-test-60.json"
```

Evaluation always uses the checkpoint's fixed training normalization, including
when `--dataset-dir` points at independent data. For an external dataset, pass
`--split all` to evaluate every valid window; the default `test` is only its
held-out partition. Reports state actual episode/domain IDs, physical-domain
overlap, artifact hashes, window counts and whether evaluation was truncated.
Training-episode overlap is rejected. Independently sampled domains within
training ranges are not automatically a distribution shift. Generate a named
stress population with `--domain-distribution extended-mass-and-motor-lag` to
place mass and motor lag outside training support. Reports
include constant-velocity and zero-residual kinematic baselines so learned
rollout error is not interpreted without context.

Exercise the learned model in closed-loop MPPI against a held-out randomized
rotor-force plant:

```bash
cargo run --release --locked --bin lewm-sim-skyjepa -- \
  --checkpoint-dir "$SKYJEPA_RUN" \
  --samples 512 \
  --horizon 15 \
  --randomize-domain
```

Gate nominal plus held-out hover, circle, and figure-eight scenarios and compare
the learned corrections against the exact same flight prior:

```bash
cargo run --release --locked --bin lewm-bench-skyjepa -- \
  --checkpoint-dir "$SKYJEPA_RUN" --controller trained-mppi \
  --random-domains 20 --output "$SKYJEPA_RUN/bench-skyjepa.json"
cargo run --release --locked --bin lewm-bench-skyjepa -- \
  --checkpoint-dir "$SKYJEPA_RUN" --controller prior \
  --random-domains 20 --output "$SKYJEPA_RUN/bench-prior.json"
```

Also compare `--controller nominal-physics-mppi` and `--controller untrained-mppi`.
All MPPI modes share the candidate budget, cost, prior and actuator bounds.
Nominal physics uses a fused CUDA rigid-body model with motor lag/drag and only
nominal parameters, not hidden test-domain parameters. Untrained MPPI has the
same architecture/normalization but fixed random weights (`--ablation-seed 7`).
All modes use observed hover trim; `--trim-multiplier 0.9` or `1.1` perturbs
calibration without falsifying observed command history. This remains a
trim-initialized state-feedback test, not unknown-trim or visual flight.

Benchmark v2 reports tracking and timing separately, incomplete/failed runs,
raw per-cycle planning times, p50/p95/p99/max, 10 ms budget exceedances and
50 ms control-deadline misses. Checkpoint/executable hashes, full configuration
and GPU/process snapshots accompany each result. Simulated time does not model
wall-clock overruns; passing these gates is not hard real-time certification.

Run the dedicated Bevy rotor-force simulator (the existing LeWM simulator is a
different binary and remains available):

```bash
cargo run --release --locked -p lewm-drone-viewer \
  --bin skyjepa-drone-sim -- \
  --checkpoint-dir "$SKYJEPA_RUN" \
  --scenario figure-eight --randomize-domain
```

The window shows actual, reference, and learned predicted paths, commanded rotor
forces, prior action, learned action delta, CUDA plan latency, and tracking
telemetry. Use `Space` to pause, `L` to toggle the controller (off applies the
hover command; it is not a prior-only tracking comparison),
`Backspace` to reset, `R` to randomize the plant, `1`/`2`/`3` to select
hover/circle/figure-eight, and the free-camera mouse/keyboard controls to inspect
the flight.

Inference loads immutable safetensors directly rather than creating trainable
variables. Candidate tensors, latent rollouts, metric rollouts, costs, MPPI
weights, and selected actions remain on the Candle CUDA path until selection.
The selected action sequence and optional single predicted path are copied to
the host for control/visualization; the full candidate population is not.

The simulator reports both cold-start and steady planning latency. NVRTC and
library initialization make the first cycle intentionally visible in
`mean_plan_ms`/`max_plan_ms`; `steady_mean_plan_ms`, p50, and p95 are the useful
control-loop measures after that first compile cycle.

## Historical pilot evidence (2026-09-03)

These results and the README GIF refer to the preserved historical checkpoint,
not the corrected three-seed pilot. The historical episode-disjoint split shared
physical domains, and a later audit found domain/flight-type coverage aliasing.
The corrected generator, domain-disjoint splits, checkpoint/resume safeguards,
and preregistered follow-up are documented in
[skyjepa-remediation.md](skyjepa-remediation.md).

The 2026-09-03 RTX 4090 pilot is an end-to-end implementation validation, not a
claim of reproducing the authors' unreleased 20,000-trajectory checkpoint. It
used 2,000 ten-second trajectories over 100 randomized domains (402,000 rows),
a 1,000-step latent stage, and a 5,000-step prober stage. The accepted dataset
fingerprint is
`67928c082cb7fd627aa6132df8738a80cd5fed6aa778bce25d2be6c4a379a37d`.

- Audit passed with `0.263 m` reference-position RMSE, zero ground contact,
  rotor-about-collective standard deviations `0.273`-`0.275 N`, and command
  delta standard deviations `0.0724`-`0.0727 N`.
- The latent best validation loss was `0.2383`; all 24 latent dimensions were
  active. The prober best validation loss was `0.07390`.
- A separate 500-trajectory/100-domain artifact sampled within the training
  parameter ranges passed the then-current audit. The original report evaluated
  its test partition, not all 500 trajectories. At
  5 steps (`0.25 s`), SkyJEPA position-vector RMSE was `0.0519 m` versus
  `0.0535 m` for constant velocity. At 20 steps (`1.0 s`), it was `0.818 m`
  versus `0.769 m`; at 60 steps it degraded to `10.30 m` versus `3.43 m`.
  The pilot therefore validates the complete learned path but does not yet
  establish superior long-horizon open-loop prediction.
- Closed-loop gating at the paper's 512 samples/horizon 15 passed all 63
  nominal and held-out-domain scenario runs. Aggregate p95 planning latency was
  `8.69 ms`, worst trajectory RMSE `0.407 m`, and worst position error
  `0.984 m`. The matched prior-only baseline passed 62/63, with worst RMSE
  `0.888 m` and worst error `1.394 m`; learned corrections materially improved
  the worst randomized case while mean RMSE moved from `0.2245` to `0.2210 m`.

This historical control result did not isolate the benefit of learning against
nominal-physics MPPI or untrained MPPI. The corrected experiment adds those
comparators and imperfect-trim tests before making a stronger learning claim.

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
