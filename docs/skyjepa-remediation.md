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
