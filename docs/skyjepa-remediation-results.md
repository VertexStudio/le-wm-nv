# Corrected SkyJEPA pilot — 2026-09-05

Status: training, full-population open-loop evaluation, warm-start selection,
final control measurements and simulator capture are complete.
Nothing has been pushed; LeWM and historical pilot artifacts remain intact.

The corrected stack works end to end and all trained controllers pass tracking.
It does **not** beat nominal-physics MPPI in mean error or latency, and two
trained seed/trim combinations fail the strict combined timing/tracking gate.
Long-horizon and shifted open-loop prediction remain weak. These are retained
results, not reasons to change the frozen test or select a different checkpoint.

## What this experiment tests

This is a clean-room, native Rust/Candle implementation validation, not an
upstream-source reproduction, paper-scale result or real-flight demonstration.
The changes and regression evidence are described in
[the remediation log](skyjepa-remediation.md). Budgets, populations and test
seeds were fixed in [the experiment protocol](skyjepa-remediation-experiment.json)
before training and final testing.

The corrected dataset has 2,000 ten-second trajectories in 100 domains, with
two hover and eighteen moving trajectories in every domain. Domain-disjoint
80/10/10 splits yield 1,600 training, 200 validation and 200 test episodes.
Normalization is fitted on the training partition and bound to both model stages.

That is 5.56 hours of generated simulation, of which 4.44 hours belong to the
training partition. Overlapping windows are reused across updates; they are not
independent new experiences. The three seeds reuse the same dataset. This is a
fixed pilot budget, not a measurement of the minimum data needed, and it trains
dynamics through supervised/self-supervised losses, not an RL policy.

## Engineering verification

The final workspace regression run passes **85 Rust tests**, and the local-uv
evaluation/summary suite passes **5 tests**. Coverage includes contract mismatch
rejection before writes, interrupted snapshot publication, corrupted artifacts,
same-environment uninterrupted/resumed CUDA equivalence for both stages,
canonical 20 Hz rejection, domain/flight coverage, external evaluation
populations, all four control modes, and CPU/CUDA nominal-rollout agreement.
Existing LeWM regression coverage remains passing.

Commands: `cargo test --locked --workspace -- --test-threads=1` and
`uv run --locked scripts/test_skyjepa_evaluation.py`. The final Rust log is
`final-workspace-tests.log` in the artifact root. The compact result was
regenerated in memory and matched the checked-in JSON exactly; video,
screenshot and checkpoint hashes were also rechecked.

## Training

Every seed uses 1,000 latent updates and 5,000 prober updates, batch size 2,048.
Checkpoints are selected using eight validation batches per epoch; the final
evaluations below cover their entire declared populations. All latent stages
finish with 24 variance-active dimensions. That count is not an effective-rank
test or a guarantee that all dimensions contain independent information.

| Training seed | Latent time | Prober time | Best prober step | Best validation metric MSE |
| --- | ---: | ---: | ---: | ---: |
| 7 | 28.74 min | 5.72 min | 4,860 | 0.019535 |
| 17 | 28.52 min | 5.26 min | 4,320 | 0.025486 |
| 29 | 29.01 min | 5.23 min | 3,915 | 0.014413 |

Total training time is 102.47 minutes, approximately 34 minutes per model.
These are observed wall times on a shared RTX 4090, including concurrent
implementation diagnostics during parts of training—not isolated throughput
benchmarks. The frozen trainer's hash is recorded in the remediation log.

## Complete-population open-loop evaluation

Each seed is evaluated on 26,400 valid windows from 200 pilot test episodes
(10 unseen physical domains), 132,000 windows from 1,000 independent in-range
episodes (100 new domains), and 132,000 windows from 1,000 deliberately shifted
episodes (100 new domains). No evaluated physical domain overlaps training.

The shifted population uses mass 1.55–1.75 times nominal and motor lag
0.11–0.14 seconds, both outside training support. All other parameter ranges
are unchanged. Merely drawing a fresh seed is not called distribution shift.

Position-vector RMSE in metres on the independent **in-range** population:

| Horizon | Seed 7 | Seed 17 | Seed 29 | Constant velocity |
| --- | ---: | ---: | ---: | ---: |
| 0.25 s | 0.0398 | 0.0562 | 0.0372 | 0.0515 |
| 0.75 s | 0.3697 | 0.5639 | 0.3633 | 0.4353 |
| 1.00 s | 0.6360 | 0.9994 | 0.6346 | 0.7400 |
| 3.00 s | 10.399 | 12.277 | 10.501 | 3.300 |

On the **shifted** population:

| Horizon | Seed 7 | Seed 17 | Seed 29 | Constant velocity |
| --- | ---: | ---: | ---: | ---: |
| 0.25 s | 0.1164 | 0.1567 | 0.1159 | 0.0626 |
| 1.00 s | 2.1450 | 2.9236 | 2.1510 | 0.8805 |
| 3.00 s | 21.260 | 26.377 | 21.549 | 3.655 |

Two seeds improve short-horizon in-range prediction, but the gain is not
consistent across seeds. All three lose the three-second and shifted-dynamics
comparisons. These results do not establish reliable long-horizon extrapolation
or sim-to-real transfer. Raw reports also include every intermediate horizon
and the zero-residual kinematic baseline.

Closed-loop tracking and open-loop prediction are different tests: MPPI plans
only 0.75 seconds ahead, applies one action, and replans at 20 Hz with fresh
state feedback and a stabilizing geometric prior. Successful tracking therefore
does not validate a three-second uncorrected prediction.

## Warm-start decision, using validation only

Validation uses domain seed 31415 and eight randomized domains plus a nominal
anchor, each on hover, circle and figure-eight. Trained metrics below aggregate
the three seeds (81 runs); the fixed baselines each have 27 runs.

| Controller | Fresh mean RMSE | Shifted mean RMSE | Selected |
| --- | ---: | ---: | --- |
| Nominal-physics MPPI | 0.1691 m | 0.1575 m | Shifted residual |
| Untrained MPPI | 0.5047 m | 1.8131 m | Fresh prior |
| Trained SkyJEPA MPPI | 0.2037 m | 0.2196 m | Fresh prior |

Trained MPPI passes 81/81 tracking cases with either strategy; fresh has 79/81
timing passes versus 81/81 shifted. Shifted residuals nevertheless fail the
preregistered mean-error improvement rule. Fresh-prior therefore remains the
trained controller's default and is used for its final tests and capture.
Untrained shifted MPPI passes only 4/27 tracking cases versus 24/27 fresh.
Nominal physics passes 27/27 tracking and timing cases under either strategy.

## Final control comparison

The protocol measures all three trained seeds against the same
geometric prior, fused nominal-physics MPPI and fixed-initialization MPPI.
The latter three are executed once per condition, not counted as three
independent training seeds. Reuse requires identical model/preprocessing/physics
contracts across all three trained packages.

Every final report contains 63 cases: three references times twenty random
domains plus one nominal anchor. Test-domain seed 271828 is used only after
warm-start selection. In-range tests include trim multipliers 1.0, 0.9 and 1.1;
the deliberate shift uses seed 90002 at unit trim.

Tracking gates require finite state/actions, no ground contact, RMSE <=0.75 m
and maximum position error <=3 m. Timing is reported separately, with per-case
p95 <=10 ms, raw per-cycle latencies, tail percentiles and 50 ms deadline misses.
The overall gate additionally requires all nominal anchors to pass, >=95% joint
case success and aggregate p95 <=10 ms. Other GPU applications remain running
and are recorded. Simulated time does not model wall-clock scheduling overruns.

All modes share prior generation, candidate budget, costs and planner bounds.
The nominal comparator receives nominal physics, command-derived motor estimates
and hover trim—not hidden randomized plant parameters. Trim perturbations affect
all geometric priors and nominal-model calibration; learned models retain raw
observed commands. This measures calibration sensitivity, not proof that learning
is necessary to estimate trim; a nonlearned controller could also adapt its trim.

### Exact trim, in-range test domains

These reports are complete. Each row measures the same 63 cases; trained seeds
are listed individually rather than averaging away failures.

| Controller | Tracking | Timing | Mean RMSE | Worst RMSE | Aggregate p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Geometric prior | 63/63 | 63/63 | 0.2153 m | 0.5706 m | 0.0015 ms |
| Nominal-physics MPPI | 63/63 | 63/63 | 0.1609 m | 0.3038 m | 3.215 ms |
| Untrained MPPI | 51/63 | 63/63 | 0.6035 m | 1.8637 m | 8.931 ms |
| Trained, seed 7 | 63/63 | 63/63 | 0.2009 m | 0.4024 m | 8.818 ms |
| Trained, seed 17 | 63/63 | 63/63 | 0.2059 m | 0.4231 m | 8.936 ms |
| Trained, seed 29 | 63/63 | 63/63 | 0.2029 m | 0.3781 m | 8.849 ms |

Trained mean error improves by 4.4–6.7% over the prior and worst-case trajectory
RMSE improves by 25.9–33.7%. Unlike the historical result, the corrected prior
also passes all 63 cases, so there is no exact-trim tracking-success gain.
Nominal-physics MPPI has lower mean and worst error than every trained seed;
trained p95 planning latency is approximately 2.7–2.8 times its latency.
Training clearly improves on these fixed random weights, but this experiment does not demonstrate
an advantage over the matched nonlearned predictive controller.

The prior and nominal controller hover essentially exactly; learned MPPI adds
0.023–0.031 m mean hover RMSE. Its mean gains over the prior mainly come from
figure-eight tracking (0.332–0.338 m versus 0.397 m), with circle tracking close
to the prior (0.245–0.249 m versus 0.249 m). Trained seeds beat the prior's RMSE
in only 30–31 of 63 paired cases; none beats nominal-physics MPPI in any
exact-trim case. One untrained run touches the ground.

No exact-trim comparator misses a 50 ms control deadline. Passing a p95 gate
does not mean every cycle stays below 10 ms: the three trained runs have
80/97/76 such exceedances across 10,080 cycles per seed, with maxima of
16.509/12.510/13.559 ms respectively.

### Calibration and distribution stress

All reports are complete. Each row has 63 cases. The shifted condition includes
60 deliberately shifted randomized cases and three nominal anchors; it is not
63 out-of-support plants. "Gate" includes the nominal-anchor requirement and
>=95% joint tracking/timing success, not just aggregate latency.

| Condition | Controller | Tracking | Timing | Mean RMSE | p95 | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Trim 0.9 | Prior | 62/63 | 63/63 | 0.4270 m | 0.0015 ms | Pass |
| Trim 0.9 | Nominal MPPI | 63/63 | 63/63 | 0.2470 m | 3.029 ms | Pass |
| Trim 0.9 | Untrained MPPI | 27/63 | 63/63 | 0.9223 m | 8.929 ms | Fail |
| Trim 0.9 | Trained seed 7 | 63/63 | 63/63 | 0.3869 m | 8.764 ms | Pass |
| Trim 0.9 | Trained seed 17 | 63/63 | 63/63 | 0.3652 m | 8.852 ms | Pass |
| Trim 0.9 | Trained seed 29 | 63/63 | 57/63 | 0.3803 m | 9.530 ms | **Fail** |
| Trim 1.1 | Prior | 63/63 | 63/63 | 0.3218 m | 0.0016 ms | Pass |
| Trim 1.1 | Nominal MPPI | 63/63 | 63/63 | 0.2235 m | 2.018 ms | Pass |
| Trim 1.1 | Untrained MPPI | 52/63 | 62/63 | 0.5802 m | 9.365 ms | Fail |
| Trim 1.1 | Trained seed 7 | 63/63 | 61/63 | 0.3149 m | 9.201 ms | Pass |
| Trim 1.1 | Trained seed 17 | 63/63 | 61/63 | 0.3049 m | 9.390 ms | Pass |
| Trim 1.1 | Trained seed 29 | 63/63 | 59/63 | 0.3072 m | 9.244 ms | **Fail** |
| Mass/lag shift | Prior | 63/63 | 63/63 | 0.1957 m | 0.0013 ms | Pass |
| Mass/lag shift | Nominal MPPI | 63/63 | 63/63 | 0.1568 m | 2.128 ms | Pass |
| Mass/lag shift | Untrained MPPI | 62/63 | 61/63 | 0.4229 m | 9.085 ms | Fail |
| Mass/lag shift | Trained seed 7 | 63/63 | 63/63 | 0.2044 m | 9.062 ms | Pass |
| Mass/lag shift | Trained seed 17 | 63/63 | 63/63 | 0.2078 m | 8.854 ms | Pass |
| Mass/lag shift | Trained seed 29 | 63/63 | 62/63 | 0.2063 m | 9.092 ms | Pass |

Across four conditions, trained controllers pass **756/756 tracking cases** and
741/756 timing cases. Two of twelve trained reports fail the combined gate:
seed 29 at either perturbed trim. There are no nonfinite runs and no 50 ms
control-deadline misses across any of the 1,512 final comparator runs. Eight
untrained runs touch the ground; none of the prior, nominal or trained runs do.
Shared-GPU timing does not isolate contention from implementation latency, and
these failures are not discarded or repaired by tuning on the final test set.

Each trained seed fixes the prior's single tracking failure at trim 0.9, but
nominal MPPI also fixes it and has lower mean error in every condition. Under
mass/lag shift, trained mean error is 4.4–6.2% **worse** than the prior despite
passing the tracking gate. Nonlearned model-based control remains the stronger
mean-error/latency baseline in this experiment.

Planner and plant saturation are distinct: the largest per-case high-command
fraction for trained MPPI is 3.44% at planner bounds, while no commanded action
hits the randomized plant's upper force bound. Raw reports retain both measures;
planner clipping is not described as physical actuator saturation.

## Simulator evidence and artifacts

The selected seed-7 checkpoint runs two additional 400-step, 20-second headless
flights at raw domain seed 31415 with fresh-prior MPPI, 512 samples and horizon
15. Both have finite state, no ground contact and nonzero learned corrections:

| Reference | Position-vector RMSE | Maximum error | Planning p95 |
| --- | ---: | ---: | ---: |
| Circle | 0.1900 m | 0.6925 m | 9.510 ms |
| Figure-eight | 0.2932 m | 0.9320 m | 8.588 ms |

Full traces are `simulation/final-seed-7-circle.json` and
`simulation/final-seed-7-figure-eight.json`. These are additional demonstration
runs, not added to the frozen 63-case gates. The older preliminary 160-step
figure-eight trace remains preserved; its training-contended timing is excluded.

The actual Bevy simulator was then recorded separately at normal simulation
speed: 20 seconds, H.264, 1280×800, 30 fps, 600 frames. A frame at 16 seconds
was visually inspected: controller on, figure-eight running, executed/reference/
predicted paths and nonzero rotor corrections visible. The GUI log contains no
warnings or errors. HUD p95 is about 17.69 ms in that frame, with rendering
and prediction export active; it is not the headless timing claim.

The recording is local only:

```text
/home/rozgo/.stable_worldmodel/le-wm-nv-data/skyjepa-remediation-v3/simulation/seed-7-figure-eight.mp4
```

Its `.capture.json` sidecar records the exact commands, checkpoint, executable,
video and screenshot hashes. Video SHA-256:
`12a96c30fc64b494566ac701628f47af1fe96964c3af88a127f07caad7606314`.
GUI executable SHA-256:
`0cec102fa31f875b418aa22c485db8c1c82b50f8c2a7eab81c63629f13d3245b`.
Only the simulator process launched for capture was terminated afterward;
other desktop/GPU applications were left running.

All model, dataset and raw report artifacts are local under:

```text
/home/rozgo/.stable_worldmodel/le-wm-nv-data/skyjepa-remediation-v3
```

The README GIF remains the preserved historical pilot capture, not evidence
that its older checkpoint has the corrected contracts or data split.

## Rechecking the evidence

The [checked-in compact results](skyjepa-remediation-results.json) retain raw-report hashes, all individual trained
seeds, paired domain-matched comparisons and the validation-only warm-start
decision. Dataset/checkpoint files and raw per-cycle traces remain outside Git.

Using the existing frozen artifacts (completed reports are verified and reused):

```bash
SKYJEPA_REMEDIATION_ROOT=/home/rozgo/.stable_worldmodel/le-wm-nv-data/skyjepa-remediation-v3
uv run --locked scripts/evaluate-skyjepa-remediation.py "$SKYJEPA_REMEDIATION_ROOT" validation
uv run --locked scripts/evaluate-skyjepa-remediation.py "$SKYJEPA_REMEDIATION_ROOT" test
uv run --locked scripts/evaluate-skyjepa-remediation.py "$SKYJEPA_REMEDIATION_ROOT" open-loop
uv run --locked scripts/summarize-skyjepa-remediation.py "$SKYJEPA_REMEDIATION_ROOT" /tmp/skyjepa-review-summary.json
```

Choose a new summary output path if that file already exists. Training, builds
and visual capture must not overlap timed control measurements. Reusing reports
does not rerun them; a genuinely new experiment needs a fresh artifact root and
its own recorded protocol. The original snapshots and historical pilot are not
overwritten by this workflow.
