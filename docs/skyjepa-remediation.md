# SkyJEPA remediation and validation

Status: implementation in progress. All commits stay local until the user
reviews the results and explicitly authorizes a push. LeWM and the accepted
pilot artifacts are preserved.

## Implementation sequence

- [ ] Versioned checkpoint contract: bind architecture, preprocessing, physics,
  latent parent, data fingerprint, and training settings; reject mismatches
  before modifying an existing run.
- [ ] Crash-safe checkpoint generations: publish weights, optimizer, progress,
  and contract together; retain the previous committed generation.
- [ ] Deterministic resume: step-addressed SIGReg randomness and independent
  validation; uninterrupted/interrupted equivalence tests.
- [ ] Canonical 20 Hz contract across trainer, evaluator, and controller.
- [ ] Domain/trajectory coverage audit and explicit evaluation populations.
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

Pending. Record commands, artifact fingerprints, environment, tests, timings,
and both successful and unsuccessful experiments here as work progresses.
