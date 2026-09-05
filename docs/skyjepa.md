# SkyJEPA

SkyJEPA is this repository's learned UAV dynamics model and model-predictive
control pipeline. It trains from state/action logs, predicts vehicle motion,
and helps select rotor commands in a 200 Hz quadrotor simulator. Training and
planning run natively in Rust/Candle on NVIDIA CUDA; LeWM remains a separate,
supported model family.

Three trained models complete every tracking case in the current simulator
suite. Training reduces tracking error by about two-thirds versus random
weights, and improves the hand-written controller's worst trajectory RMSE by
26–34%. Each model took about 34 minutes to train on a shared RTX 4090.

## How the system works

| Component | Responsibility |
| --- | --- |
| Hand-written geometric controller | Uses current state and the desired path to propose thrust, attitude corrections and four rotor commands. |
| SkyJEPA model and physics prober | Learn to predict motion under candidate rotor commands. |
| MPPI planner | Samples alternatives around the proposed commands and uses predicted tracking cost to choose an action. |
| Rotor simulator | Independently executes that action using rigid-body physics, motor lag and drag. |

The geometric controller is code we wrote, not another trained network.
It supplies basic stabilization and trajectory tracking. It receives full
position, velocity, attitude and angular-rate state, nominal physics parameters,
and a hover-thrust calibration. In the simulator, that calibration comes from
the command that holds the randomized plant in hover.

SkyJEPA supplies predictions to MPPI, which improves commands around that
starting sequence. Learned dynamics residuals modify the predictor; MPPI's
action corrections modify rotor commands. The simulator itself is not modified
by learned residuals. The flight in the README shows this hybrid system.

### Model and acceleration

Stage one learns action-conditioned latent dynamics:

```text
state history [B,10,18] -> causal TCN [8,8,16] -> latent24
action windows [B,20,10,4] -> causal TCN [4,4,8] -> action8
latent24 + action8 -> GRU unrolled 20 steps -> predicted latent24
loss = multi-step latent MSE + 0.02 * SIGReg
```

Stage two freezes the latent model. A small prober maps each predicted latent
to a linear-acceleration residual [3] and an angular-action map [3,4].
Known thrust/gravity and those learned outputs drive a differentiable integrator
for position, velocity, SO(3) orientation and angular velocity. State-prediction
loss backpropagates through integration into the prober.

Planning uses an equivalent fused CUDA forward integrator: one thread advances
a candidate trajectory. It has no backward implementation, because MPPI uses
sampling and scoring rather than gradients. The independent rotor simulator
also runs outside autodiff.

The runtime encodes state history once, batches candidate action windows into
one TCN call, and keeps latent rollouts, metric rollouts, costs and MPPI weights
on CUDA. Only the selected action sequence and optional single visualization
path return to the CPU.

| Setting | Value |
| --- | --- |
| State / action | 18 values / 4 rotor forces |
| History / training rollout | 10 / 20 steps at 20 Hz |
| Latent / GRU width | 24 |
| SIGReg | 17 knots, weight 0.02, 64 projections |
| MPPI horizon / candidates | 15 steps (0.75 s) / 512 |
| MPPI temperature / noise scales | `1e-4` / `[0.60,0.15,0.15,0.05]` |
| State group / action cost weights | `[400,40,20,20]` / `[0.01,0.05,0.05,0.10]` |

The selected learned-controller warm start is `fresh-prior`: build a new
geometric command sequence each cycle. Optional `shifted-residual` carries
the previous action correction, bounded to ±2 N, with a zero tail. It is
activated only after execution and cleared on reset. Validation selected
fresh-prior for trained/untrained MPPI and shifted-residual for nominal-physics
MPPI.

## Results

Measurements below use training seeds 7, 17 and 29 on a shared RTX 4090
(2026-09-05). Each model uses 1,000 latent updates and 5,000 prober updates,
batch size 2,048. Training takes about 34 minutes per model, 102.47 minutes
total. Checkpoint selection uses eight validation batches per epoch.

The dataset has 2,000 ten-second trajectories across 100 physical domains,
with two hover and eighteen moving trajectories per domain. Domain-disjoint
80/10/10 splits yield 1,600 training, 200 validation and 200 test episodes.
Training uses 4.44 simulated hours; the full dataset contains 5.56 hours.
Overlapping training windows reuse those observations. These times and data
volumes describe this experiment's budget; the learned objective is dynamics
prediction rather than an RL policy.

### Flight control

Every exact-trim row measures the same 63 cases: hover, circle and figure-eight
in 20 randomized domains plus a nominal anchor. MPPI uses 512 candidates,
horizon 15 and 20 Hz control. Test-domain seed 271828 is separate from the
warm-start validation seed 31415.

| Controller | Tracking passes | Mean RMSE | Worst RMSE | Planning p95 |
| --- | ---: | ---: | ---: | ---: |
| Hand-written controller | 63/63 | 0.2153 m | 0.5706 m | 0.0015 ms |
| Random-weight model + MPPI | 51/63 | 0.6035 m | 1.8637 m | 8.931 ms |
| SkyJEPA seed 7 + MPPI | 63/63 | 0.2009 m | 0.4024 m | 8.818 ms |
| SkyJEPA seed 17 + MPPI | 63/63 | 0.2059 m | 0.4231 m | 8.936 ms |
| SkyJEPA seed 29 + MPPI | 63/63 | 0.2029 m | 0.3781 m | 8.849 ms |
| Nominal-physics MPPI | 63/63 | 0.1609 m | 0.3038 m | 3.215 ms |

Training has a consistent positive effect across all three seeds. Relative to
the hand-written controller alone, mean tracking RMSE improves by 4.4–6.7% and
worst trajectory RMSE by 25.9–33.7%. Most of that benefit is in figure-eight
tracking; the hand-written controller already hovers very accurately.

Nominal-physics MPPI is the strongest mean-error/latency baseline here. It uses
explicit rigid-body equations with nominal parameters, command-derived motor
estimates and the same hover calibration. It does not receive the randomized
plant's hidden parameters. All MPPI modes share the cost, candidate budget,
geometric starting sequence and actuator bounds; only the prediction model and
validation-selected warm-start strategy differ. The nonlearned/random baselines
are measured once per condition, rather than treated as independent training seeds.

The wider suite tests exact and ±10% hover calibration, plus mass 1.55–1.75×
nominal and motor lag 0.11–0.14 s, outside training support. Each count below
sums three models on the same 63 cases; the shift includes 60 shifted cases
and three nominal anchors.

| Condition | Trained tracking passes | Trained timing passes |
| --- | ---: | ---: |
| Exact calibration | 189/189 | 189/189 |
| Calibration ×0.9 | 189/189 | 183/189 |
| Calibration ×1.1 | 189/189 | 181/189 |
| Mass/motor-lag shift | 189/189 | 188/189 |
| Total | **756/756** | **741/756** |

All trained flights complete without ground contact. Each trained model also
fixes the hand-written controller's one tracking failure at calibration ×0.9.
Under the mass/lag shift, mean error is 0.204–0.208 m versus the hand-written
controller's 0.196 m; nominal-physics MPPI has the lowest mean error in every
condition.

Tracking requires finite state/actions, no ground contact, trajectory RMSE
≤0.75 m and maximum position error ≤3 m. Timing is a separate per-case p95
≤10 ms check. A report passes the combined gate when all nominal anchors pass,
at least 95% of cases pass both checks and aggregate p95 is ≤10 ms.
Two seed-29 reports, at perturbed trim, miss that combined gate on timing.

None of the 1,512 final comparator runs records a 50 ms control-deadline miss.
Other GPU workloads are recorded in the reports. Planning latency is measured
after warm-up; simulator time does not model wall-clock scheduling overruns.

### Prediction quality

Each model is evaluated on all 26,400 valid windows from 200 held-out-domain
test episodes and all 132,000 windows in each independent 1,000-episode
in-range/shifted population. Those domains do not overlap training. The external
in-range position-vector RMSE is:

| Prediction horizon | Seed 7 | Seed 17 | Seed 29 | Constant velocity |
| --- | ---: | ---: | ---: | ---: |
| 0.25 s | 0.0398 m | 0.0562 m | 0.0372 m | 0.0515 m |
| 0.75 s | 0.3697 m | 0.5639 m | 0.3633 m | 0.4353 m |
| 1.00 s | 0.6360 m | 0.9994 m | 0.6346 m | 0.7400 m |
| 3.00 s | 10.399 m | 12.277 m | 10.501 m | 3.300 m |

Two seeds improve short-horizon prediction over constant velocity. Long
uncorrected rollouts are the main model-quality weakness: all three lose the
three-second comparison. Under shifted dynamics, one-second error is
2.145–2.924 m versus 0.881 m for constant velocity.

The controller succeeds with much shorter predictions: it looks 0.75 seconds
ahead, applies one action and replans every 0.05 seconds with fresh state
feedback. The next model-quality target is reducing accumulated prediction
error; the next runtime target is tighter planning-latency tails.

The [machine-readable results](../benchmarks/skyjepa/results.json) retain all
24 control reports, nine prediction reports, individual seeds, paired
comparisons, timing failures and artifact hashes. The
[experiment protocol](../benchmarks/skyjepa/protocol.json) records the fixed
training budgets, validation/test seeds and selection rules.

## Dataset

The canonical schema uses commanded rotor forces in newtons and state18 ordered
as world position [3], world velocity [3], row-major world-from-body rotation [9]
and body angular velocity [3]. Generated datasets contain:

```text
metadata.json       schema, rates, action semantics, counts and distribution
domains.json        physical parameters for each domain
data.h5
  state            float32 [N,18]
  action           float32 [N,4]
  reference_state  float32 [N,18]
  motor_force      float32 [N,4]
  episode_idx      int64   [N]
  step_idx         int64   [N]
  dt               float32 [N,1]
  domain_idx       int64   [N]
```

The plant models rotor allocation, first-order motor response, gyroscopic
torque, angular damping, body-axis drag and SO(3) attitude integration.
Generation runs trajectories in parallel and includes hover and moving flight
in every domain. Physical parameters are randomized across domains.

The required version-2 audit checks schema, finite values, trajectory indices,
20 Hz sampling, valid rotations, tracking, ground contact, saturation and
per-domain flight/rotor-excitation coverage. Its SHA-256 binds the data files;
training rejects failed, missing or stale audits. Normalization is fitted on
training domains and reused by the prober, evaluator and controller.
Higher-rate raw data can be explicitly strided to 20 Hz. A separately tagged
`body-rates-throttle` import path exists for LeWM drone logs; the experiments
in this guide use the canonical rotor-force schema.

## Training and evaluation

Use Linux/NVIDIA with the repo's CUDA dependencies. The following commands
create a new seed-7 experiment with the measured 1,000/5,000-update budget;
choose an unused work directory. For a three-seed comparison, reuse the same
dataset and repeat only the two training commands with seeds 17 and 29 and
separate model directories.

```bash
cargo build --release --locked --workspace
SKYJEPA_WORK="$HOME/.stable_worldmodel/le-wm-nv-runs/skyjepa-example"
SKYJEPA_DATA="$SKYJEPA_WORK/data"
SKYJEPA_LATENT="$SKYJEPA_WORK/latent"
SKYJEPA_RUN="$SKYJEPA_WORK/prober"

target/release/lewm-generate-skyjepa \
  --output-dir "$SKYJEPA_DATA" --seed 7 --domains 100 \
  --trajectories 2000 --duration-seconds 10
target/release/lewm-audit-skyjepa \
  --dataset-dir "$SKYJEPA_DATA" --output "$SKYJEPA_DATA/audit.json"

target/release/lewm-train-skyjepa \
  --dataset-dir "$SKYJEPA_DATA" --output-dir "$SKYJEPA_LATENT" \
  --stage latent --seed 7 --split-by domains --batch-size 2048 \
  --latent-max-steps 1000 --warmup-steps 200 --cosine-steps 800 \
  --max-lr 0.005 --min-lr 0.0001
target/release/lewm-train-skyjepa \
  --dataset-dir "$SKYJEPA_DATA" --output-dir "$SKYJEPA_RUN" \
  --stage prober --latent-checkpoint "$SKYJEPA_LATENT" \
  --seed 7 --split-by domains --batch-size 2048 \
  --prober-max-steps 5000 --warmup-steps 500 --cosine-steps 4500 \
  --max-lr 0.005 --min-lr 0.0001

target/release/lewm-eval-skyjepa \
  --dataset-dir "$SKYJEPA_DATA" --checkpoint-dir "$SKYJEPA_RUN" \
  --split test --rollout-steps 60 --output "$SKYJEPA_RUN/eval.json"
target/release/lewm-bench-skyjepa \
  --checkpoint-dir "$SKYJEPA_RUN" --controller trained-mppi \
  --warm-start fresh-prior --samples 512 --horizon 15 \
  --domain-seed 271828 --random-domains 20 --output "$SKYJEPA_RUN/control.json"
```

The evaluator reports every prediction horizon against constant-velocity and
zero-residual kinematic baselines. Use `--split all` for an independent dataset;
reports include actual episode/domain IDs, training overlap and population
completeness. Generate out-of-support data with
`--domain-distribution extended-mass-and-motor-lag`.

The benchmark also accepts `--controller prior`, `untrained-mppi`, and
`nominal-physics-mppi`. For the settings in the results above, use fresh-prior
for trained/untrained modes, shifted-residual for nominal physics, and
`--ablation-seed 7`. `--trim-multiplier 0.9` or `1.1` changes hover calibration
without changing the model's observed action history. The shifted population
uses `--domain-distribution extended-mass-and-motor-lag --domain-seed 90002`.
Use a new output file per comparison. `--allow-fail` retains negative benchmark
results without making the process exit unsuccessfully.

### Checkpoints and resume

A deployable prober directory is self-contained: version-2 `checkpoint.json`
binds architecture, normalization, 20 Hz timing, action semantics, physics,
latent ancestry and immutable hashed safetensors under `objects/`.
Loading verifies that contract and the weight files.

Snapshots under `snapshots/` contain weights, optimizer, progress and the
committed metrics boundary. Atomic `{stage}-current.json` publication selects
a complete generation; the previous generation remains recoverable. Export
selects the best validation checkpoint. To resume an interrupted explicit
stage, repeat its original command with `--resume`; incompatible settings are
rejected before writes. `--stop-after-step N` deliberately pauses a stage while
preserving its total budget. Resume equivalence is tested across validation
and epoch boundaries on the same hardware/software environment.

## Run the simulator

Using the prober directory produced above:

```bash
target/release/skyjepa-drone-sim \
  --checkpoint-dir "$SKYJEPA_RUN" --scenario figure-eight \
  --randomize-domain --domain-seed 31415 \
  --samples 512 --horizon 15 --warm-start fresh-prior --time-scale 1
```

Controls: `Space` pauses; `Backspace` resets; `R` randomizes the plant;
`1`/`2`/`3` select hover/circle/figure-eight. `L` toggles the learned controller;
off applies the hover command, so use the benchmark's `prior` mode for an
actual hand-written-controller tracking comparison. Free-camera mouse/keyboard
controls inspect the scene.

For a headless state/action trace:

```bash
target/release/lewm-sim-skyjepa \
  --checkpoint-dir "$SKYJEPA_RUN" --reference figure-eight \
  --control-steps 400 --randomize-domain --domain-seed 31415 \
  --output "$SKYJEPA_RUN/flight.json"
```

The README's [20-second MP4](skyjepa-v3-figure-eight.mp4) runs seed 7 with the
settings above. Additional 20-second headless runs achieve 0.190 m circle RMSE
and 0.293 m figure-eight RMSE, with no ground contact. The HUD shows commanded
forces, prior action, learned correction, prediction and render-contended
latency. Yellow is executed flight, cyan reference, magenta prediction, and
green bars rotor commands.

The [GIF](skyjepa-v3-figure-eight.gif) is an eight-second excerpt (video seconds
8–16), at normal speed, 800×500 and 10 fps. To regenerate it from the repo root:

```bash
ffmpeg -hide_banner -loglevel warning -n \
  -ss 8 -t 8 -i docs/skyjepa-v3-figure-eight.mp4 \
  -filter_complex '[0:v]fps=10,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle' \
  -loop 0 /tmp/skyjepa-v3-preview.gif
```

## Verification and implementation basis

The current implementation passes 85 Rust tests and five local-uv protocol
tests, covering checkpoints, resume, data contracts, CPU/CUDA integration
agreement, controller comparisons and existing LeWM regressions:

```bash
cargo test --locked --workspace -- --test-threads=1
uv run --locked scripts/test_skyjepa_evaluation.py
```

This implementation is built from the
[SkyJEPA paper](https://arxiv.org/html/2606.23444), rather than translated from
upstream source code. The [authors' project](https://github.com/arplaboratory/SkyJEPA)
is the research reference. The paper supplies the state/action contract,
encoder dimensions, latent/GRU size, two-stage objective, prober outputs,
integration structure and MPPI settings. Repo-specific choices include:

- Two causal GELU convolutions per TCN residual block, kernel 3, exponential
  dilation, residual projections and no dropout.
- A two-hidden-layer, width-32 GELU prober and 64 SIGReg projections.
- Rotation-matrix Frobenius attitude cost and a hand-written geometric
  action prior.
- Random-Fourier reference generation, a geometric data-collection tracker
  with rotor excitation, and arm-length randomization of ±20%.

These choices describe the model we run. Its measured scope is state-feedback
control in the rotor simulator; real-vehicle deployment requires its own
validation. Python is used through local `uv` for experiment orchestration and
analysis, while model training and the control runtime are Rust/CUDA.

[Development history](skyjepa-history.md) preserves experiment decisions,
lessons, artifact provenance and the workflow for extending this evaluation.
