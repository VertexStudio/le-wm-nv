# SkyJEPA remediation and validation

Status: implementation in progress. All commits stay local until the user
reviews the results and explicitly authorizes a push. LeWM and the accepted
pilot artifacts are preserved.

## Implementation sequence

- [x] Versioned checkpoint contract: bind architecture, preprocessing, physics,
  latent parent, data fingerprint, and training settings; reject mismatches
  before modifying an existing run.
- [x] Crash-safe checkpoint generations: publish weights, optimizer, progress,
  and contract together; retain the previous committed generation.
- [x] Deterministic resume: step-addressed SIGReg randomness and independent
  validation; uninterrupted/interrupted equivalence tests.
- [x] Canonical 20 Hz contract across trainer, evaluator, and controller.
- [x] Domain/trajectory coverage audit and explicit evaluation populations.
- [ ] Compare fresh-prior and shifted-residual warm starts under matched settings.
- [ ] Matched-budget nominal-physics/untrained/trained MPPI ablations, imperfect
  trim tests, and separate tracking/timing reporting with artifact provenance.
- [ ] Corrected pilot, multiple training seeds, held-out-domain and deliberate
  distribution-shift evaluations, simulator verification, and evidence report.

## Acceptance rules

Contract and resume regressions must pass before pilot training. Benchmark
seeds/settings are recorded before evaluating a candidate; the final test set
is not used to tune it. Timing benchmarks run without concurrent review builds
or training, and record other GPU workloads rather than terminating them.

The control gate retains the baseline limits: finite state/actions, no ground
contact, position-vector RMSE <= 0.75 m, maximum position error <= 3 m, and
per-run p95 planning <= 10 ms. Tracking and timing outcomes are reported
separately. No stronger long-horizon or sim-to-real claim follows merely from
passing the included simulated control gate.

Historical baseline: commit `613083a`, accepted dataset
`skyjepa-pilot-v2-20hz`, latent source `skyjepa-pilot-v2-run`, final model
`skyjepa-pilot-v2-prober-long` under
`/home/rozgo/.stable_worldmodel/le-wm-nv-data`. Its saved results and the review
artifacts in `/tmp/skyjepa-review.EmiA5v` remain unchanged.

## Results

Checkpoint contract: `cargo check --locked --workspace --all-targets` passed.
Three package tests and the CUDA trainer integration test passed. The latter
trains the prober on a differently normalized dataset, verifies parent
normalization is retained, and verifies changed mass/latent-parent resume
requests leave the entire output directory byte-for-byte unchanged.

New checkpoints use `checkpoint.json` with immutable, hashed weights under
`objects/`. Evaluator and controller reject loose legacy artifacts. Parent
`--latent-checkpoint` now takes a package directory. Stage-specific run
manifests include preprocessing, latent identity, physics and loss settings.
The historical baseline is not modified or automatically converted.

Training snapshots now live in immutable `snapshots/` generations. A flushed,
atomically replaced `{stage}-current.json` selects the complete generation.
Snapshots bind the optimizer step, weights, progress and training contract;
best selection retains a complete earlier generation. Failed writes leave only
unreferenced directories, never a partially updated current generation.
Fault-injection tests cover interruption before weights, between weights and
optimizer, and before publication; corruption is rejected on load. Completed
stages can be resumed to recover package export without repeating updates.

Further experiment results pending.

Resume regression: on the current CUDA environment, interrupted latent runs
at steps 7, 8, and 9 (before/after epoch validation and into the next epoch)
match a continuous 16-step run exactly: every f32 weight, every optimizer
tensor, learning rates, losses, validation metrics, and best-step selection.
The prober passes the same check across its epoch boundary. SIGReg randomness
is keyed by run seed and optimizer step, with separate validation seeds.
Singleton tail batches are excluded from the step schedule. `--stop-after-step`
pauses an explicit stage without changing its target or adding validation.

Snapshots also commit the durable metrics byte boundary. Resume preserves
uncommitted/partial log tails in separate files and restores the committed log
before appending, so replay does not duplicate updates in reported metrics.
This is a same-hardware/software determinism guarantee, not a claim of bitwise
equivalence across GPU models or CUDA/library versions.

The dataset/model contract now requires 20 Hz; raw higher-rate data can still
be explicitly strided into that model rate. Trainer rejection occurs before
creating a run. Package validation rejects noncanonical rates even when the
package hash is internally consistent, and the controller derives its timestep
from validated metadata. Existing strided-data tests remain passing.

Coverage correction: the new 2,000-trajectory/100-domain pilot contains exactly
two hover and eighteen moving trajectories in every domain. Audit v2 passes
with position tracking RMSE 0.257921 m, zero ground contact, and per-domain
excitation checks passing. Dataset SHA-256:
`77acffd68edbe3d5b352bb290693bfa86ef80045554488f2c94c2aaed72e5fe0`.
Re-auditing the untouched historical dataset correctly reports missing flight
type coverage in all 100 domains. Both reports are under the new artifact root;
the historical audit remains unchanged. Training binds the audit file hash and
requires audit version 2. Generator/auditor coverage regression tests pass.

The corrected-pilot budgets, training seeds, validation/test control seeds,
comparators, trim perturbations, and deliberate distribution shift are frozen
in `skyjepa-remediation-experiment.json` before pilot training or final testing.

Domain-disjoint training is now the default (`--split-by domains`): 80/10/10
physical domains for this pilot. Episode-disjoint splitting remains an explicit
diagnostic option, not a claim of unseen dynamics. Normalization is fit on the
selected training partition only. Checkpoints preserve both stages' training
episode IDs and physical-domain fingerprints, including latent ancestry.
JSON float round-trip parsing is enabled so nested numerical provenance is
stable through publication and resume.

Evaluation supports `--split all` for external data, reports the actual episode
and domain IDs/fingerprints and population completeness, and rejects training
episode overlap with either stage. A batch-limited report is explicitly marked
incomplete. The integration test verifies domain-disjoint splits, full external
evaluation, overlap rejection, and accurate reporting of truncated evaluation.
All four trainer/evaluator integration tests and existing data/package tests pass.

Warm-start comparison support is implemented but no default change is promoted:
`--warm-start fresh-prior` remains the default; `shifted-residual` carries the
previous correction around the new geometric prior, bounded to +/-2 N per rotor
with a zero tail and actuator clamping. A correction becomes active only after
the action is committed, and reset clears it. CPU tensor regression and workspace
checks pass. The validation comparison will run after training, without GPU
training contention, before any final-test evaluation.

The pilot runner uses a frozen trainer binary with SHA-256
`057571a4afa52cde3ae5c4b258c18536716f91c4f60ab601286074244804096d`
(training source at `1711787`; runner committed at `2d6295c`). Runtime
`git_commit` fields describe the invocation checkout; the frozen executable hash
is the authority for this experiment's training implementation. Training started
with approximately 11 GiB GPU memory occupied by other workloads, including
Isaac Sim; those workloads were left untouched.

Nominal-physics baseline: a batched CUDA rigid-body rollout now mirrors the CPU
plant's motor lag, gyroscopic torque, angular damping, SO(3) update, body-axis
drag, semi-implicit translation, actuator limits, and ground plane. It uses ten
substeps per 20 Hz action. CPU/CUDA agreement is tested over thirty-step,
five-candidate rollouts in nominal and randomized parameter fixtures (maximum
state-component tolerance 0.002). The controller comparator itself only uses
nominal parameters, command-derived motor estimates and the same observable
hover trim; randomized test-plant parameters are never passed to it.

Learned and nominal scorers share exactly the same tracking/action cost code.
A calibrated-hover test verifies both unit trim and non-unit trim have effectively
zero cost at their corresponding hover command. All three new nominal tests and
the eight existing SkyJEPA model/integrator tests pass.

All four controller modes now run through the benchmark: geometric prior,
nominal-physics MPPI, deterministic untrained MPPI, and trained MPPI. They share
the prior, action bounds and trim calibration. Untrained weights are detached
from autodiff and use a recorded initialization seed. Trim perturbations change
calibration only; observed command histories remain the actual hover commands.
The integration smoke test exercises every mode with perturbed trim.

Benchmark report v2 separates tracking and timing success, records incomplete
or non-finite runs explicitly, and retains every planning latency. Reports bind
checkpoint/executable hashes, the full controller configuration, domain seeds,
GPU/process snapshots, p50/p95/p99/max latency, 10 ms budget exceedances and
50 ms control-deadline misses. Simulation time still does not model scheduling
overruns; this is not a hard real-time certification.

External evaluation data is generated and audit-v2 clean: 1,000 trajectories
across 100 domains per population, each domain with one hover and nine moving
trajectories. The ordinary external population uses training parameter ranges
(seed 90001; artifact SHA-256
`afbc11b19ce91cf60a4d600e5dd47f551f4974f03a9f99c22aff4199494c81b3`).
The deliberate shift (seed 90002) uses mass 1.55–1.75 times nominal and motor
lag 0.11–0.14 s, both outside training support. Other parameter draws are
unchanged; a 1,000-seed regression verifies this. Metadata, audits and control
reports explicitly label the population; unseen seeds alone are not called OOD.

The serial evaluation runner is `scripts/evaluate-skyjepa-remediation.py`, run
with local `uv run --locked`. Before looking at validation results, its warm-start
selection rule is fixed: shifted residuals must improve mean RMSE by at least
1%, lose no tracking or timing successes, keep aggregate p95 <=10 ms, and worsen
worst-case RMSE by no more than 5%. Otherwise fresh-prior remains selected.
Nominal and untrained MPPI get their own matched validation comparison; trained
MPPI selection aggregates all three training seeds. The final seed is never
used for this choice. Final tests cover all four comparators at trim multipliers
1.0/0.9/1.1, plus the deliberately shifted population at unit trim.

Nonlearned and fixed-initialization baselines are executed once, not presented
as three independent training-seed measurements. Reuse requires exact equality
of all three packages' model/preprocessing/physics contracts. Trained results
are measured separately for every seed. The runner refuses concurrent training
or compilation for timing measurements and preserves each invocation/report.

Full `cargo test --locked --workspace -- --test-threads=1` passes, including
LeWM CUDA training/runtime regressions. The three local-uv evaluation-protocol
tests also pass. Simulator reports now bind the checkpoint/executable and
configuration, retain a per-control-step state/reference/action/correction trace,
and report finite-state and substep ground-contact checks. Its new integration
smoke test passes. Trim stress affects the geometric prior in every comparator
and the nominal model's calibration; learned/untrained model inputs retain their
true metric actions, matching the training data contract.

Seed 7 completed its fixed budgets: latent 1,000 steps in 1,724.665 seconds,
prober 5,000 steps in 343.128 seconds. Best latent validation prediction loss
0.259284 (24 active dimensions), best prober validation MSE 0.019535 at step
4,860. Training checkpoint selection uses the recorded eight validation batches
per epoch, not the complete validation population. Follow-up open-loop evaluation
will cover the complete selected test populations. These validation numbers are
not directly comparable to the historical pilot's different preprocessing/split.

The final implementation verification log is `workspace-tests.log` in the
artifact root: 85 Rust tests pass, plus five local-uv protocol/summary tests.
Clippy completes with existing warnings outside the newly introduced code.
Evaluation uses frozen binaries with SHA-256:

- Benchmark: `56462cf55bdaf2c96e42b830632d9e61f61608509996787764308c8839969c3c`
- Open-loop evaluator: `e93a75b0aaaff2725bef0026d18db0dc4079b49f918c185cc28190c89afcae3a`
- Headless simulator: `2a1c09f5c84cb76dd3c654f33ce6fb3778c2805832ad9bcfc5cdbbb97fcdf87d`

Seed 7's complete open-loop populations are evaluated: 26,400 windows from
200 held-out-domain pilot episodes, plus 132,000 windows from each 1,000-episode
external population. On fresh in-range domains, learned versus constant-velocity
position-vector RMSE is 0.0398/0.0515 m at 0.25 s, 0.6360/0.7400 m at 1 s,
and 10.399/3.300 m at 3 s. Under deliberate mass/lag shift, it is
0.1164/0.0626 m, 2.1450/0.8805 m and 21.260/3.655 m respectively. Thus the first
seed improves short-horizon in-range prediction but fails long-horizon and
shifted-dynamics comparisons. Other seeds and all control comparisons remain
pending; these are not promoted as final multi-seed results.

A preliminary seed-7, fresh-prior figure-eight simulation (raw domain seed
31415, 160 control steps) completes with finite state, no ground contact,
0.3682 m position-vector RMSE and 0.9320 m maximum error. Its full trace is in
`simulation/preliminary-seed-7-figure-eight.json`. It ran alongside training:
its 17.49 ms p95 is explicitly contention-affected and excluded from timing gates.
