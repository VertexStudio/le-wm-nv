# SkyJEPA development history

The [SkyJEPA guide](skyjepa.md) describes the current system and how to use it.
This document keeps the decisions, experiment provenance and lessons needed to
improve it. The numerical records are in
[benchmarks/skyjepa](../benchmarks/skyjepa/results.json).

## Lessons to retain

- Separate basic flight control from learned dynamics. The hand-written
  geometric controller can fly by itself; the learned model's contribution
  must be measured against that same controller and a predictive physics baseline.
- Split physical domains when testing unseen dynamics. Different trajectories
  from the same vehicle configuration answer a different question.
- Audit coverage per domain, not only aggregate statistics. A dataset can have
  the right global hover/motion ratio while individual domains lack one type.
- Record the actual evaluation population. New random seeds within the same
  parameter ranges are in-range tests; a distribution-shift test changes support.
- Bind weights to preprocessing and training provenance. Correct tensors with
  the wrong normalization, latent parent or timestep are the wrong model.
- Keep tracking and timing results separate, and retain failed cases. A good
  aggregate p95 can hide individual domains that miss their per-case target.
- Distinguish measurements from broader conclusions. The 24 variance-active
  latent dimensions are not a representation-rank test; a fixed training budget
  is not a minimum-data learning curve. Measure those properties directly when
  they become experiment goals.

## September 3: first working implementation

The initial implementation was reconstructed from the paper while the authors'
model, training and data source were unavailable. Paper-specified interfaces
and dimensions were kept separate from local choices such as residual TCN
blocks, the prober MLP and random-Fourier reference generation.

An initial dataset was rejected for weak differential rotor excitation. A
stronger geometric data-collection tracker with smooth excitation produced the
2,000-trajectory, 100-domain `skyjepa-pilot-v2-20hz` dataset. A later audit found
a separate coverage issue in this dataset, described below.

Raw rotor-force sampling was not a reliable flight initializer. Commit
`6bc8a36` added the trim-aware geometric action prior: feedback on the first
command and reference-based feed-forward over the remainder of the horizon.
This established the hybrid design: hand-written stabilization plus MPPI
corrections scored using learned dynamics. The prior also became a separately
measured baseline.

At baseline commit `613083a`, the trained controller passed 63/63 flight cases
versus 62/63 for the prior. Mean trajectory RMSE was 0.2210 m versus 0.2245 m;
worst RMSE was 0.407 m versus 0.888 m. Reported planning p95 was 8.69 ms.
The [original MP4](skyjepa-trained-figure-eight.mp4) and
[GIF](skyjepa-trained-figure-eight.gif) remain unchanged. These are genuine
measurements of that setup, separate from the current three-seed experiment.

The original model/data artifacts remain under
`/home/rozgo/.stable_worldmodel/le-wm-nv-data`:

- Dataset: `skyjepa-pilot-v2-20hz`; SHA-256
  `67928c082cb7fd627aa6132df8738a80cd5fed6aa778bce25d2be6c4a379a37d`.
- Latent run: `skyjepa-pilot-v2-run`.
- Prober run: `skyjepa-pilot-v2-prober-long`.
- Review scratch evidence: `/tmp/skyjepa-review.EmiA5v`.

## September 5: data, evaluation and training safeguards

| Finding | Change and verification |
| --- | --- |
| Flight type was correlated with domain ID: ten domains had only hover and ninety only motion. | Generator scheduling now supplies both types in every domain; audit v2 rejects the original dataset's 100 coverage gaps. The new 2,000-trajectory dataset has two hover and eighteen moving episodes per domain. |
| Offline episode splits reused physical domains. | Domain-disjoint 80/10/10 splits, training-only normalization and reported domain fingerprints. Tests check disjointness and external population completeness. |
| A 500-trajectory external artifact was called OOD although it used training ranges; only its test partition was evaluated. | Explicit `--split all` and actual episode/window counts. Named mass/motor-lag shifts distinguish out-of-support tests from fresh in-range samples. |
| Checkpoint loading/resume lacked complete preprocessing, parent and numerical contracts. | Versioned packages bind normalization, physics, rate, latent ancestry and hashes. Tests reject incompatible resumes before any writes, including a changed latent parent or mass. |
| Multi-file saves and stochastic resume could diverge after interruption. | Atomic snapshot generations, optimizer/progress/metrics boundaries, step-addressed randomness and separate validation seeds. Interrupted CUDA fixtures match continuous runs across epoch/validation boundaries. |
| Rate configuration could disagree with the intended model timestep. | Canonical 20 Hz enforcement and explicit raw-data striding, checked in trainer/package/controller tests. |

These changes repaired specific experimental assumptions and reliability risks;
they did not invalidate the fact that the original trained stack executed and
flew. The old offline metrics describe their actual partitions, not the stronger
unseen-domain or out-of-support populations that some descriptions implied.

The physics comparator uses a fused CUDA rollout matching the CPU rigid-body
plant, with nominal parameters only. CPU/CUDA agreement is tested across five
candidates and thirty steps, with maximum state-component tolerance 0.002.
The prior, nominal, untrained and trained comparators share costs, calibration
and planner bounds. Baseline reuse across training seeds requires identical
model/preprocessing/physics contracts.

## Validation decisions

The [protocol](../benchmarks/skyjepa/protocol.json) fixed the data budget, three
training seeds, test populations and controller settings before final testing.
Validation used seed 31415; the ordinary final control test used seed 271828.
Each validation report had 27 cases. Trained metrics below aggregate three
seeds (81 cases); nominal and untrained baselines each use 27.

| Controller | Fresh-prior mean RMSE | Shifted-residual mean RMSE | Selected |
| --- | ---: | ---: | --- |
| Trained SkyJEPA MPPI | 0.2037 m | 0.2196 m | Fresh prior |
| Untrained MPPI | 0.5047 m | 1.8131 m | Fresh prior |
| Nominal-physics MPPI | 0.1691 m | 0.1575 m | Shifted residual |

The selection rule required shifted residuals to improve mean error by ≥1%,
lose no tracking/timing passes, keep aggregate p95 ≤10 ms and worsen the worst
trajectory RMSE by no more than 5%. Trained tracking passed 81/81 under both
strategies, but shifted residuals increased mean error. Untrained shifted MPPI
passed only 4/27 tracking cases versus 24/27 fresh. No final-test tuning followed.

The completed experiment contains 24 final control reports (1,512 runs) and
nine full-population prediction reports (871,200 overlapping windows). Current
results are summarized in the guide and preserved individually in the
[results JSON](../benchmarks/skyjepa/results.json). The engineering regressions
passed 85 Rust and five Python tests; `final-workspace-tests.log` records the
post-experiment Rust run.

Useful directions for subsequent experiments are better long-horizon prediction,
lower tail latency, and a stronger learning advantage over explicit physics.
Keep the hand-written and physics baselines: trained-versus-random improvement
and improvement beyond a capable conventional controller are separate measurements.

## Frozen experiment artifacts

Root: `/home/rozgo/.stable_worldmodel/le-wm-nv-data/skyjepa-remediation-v3`.
Dataset, checkpoint, raw latency and trace files remain outside Git. The
protocol/results JSON files were moved into `benchmarks/skyjepa` without changing
their bytes. The original executable hashes, not later invocation checkout IDs,
identify the implementation used for these measurements.

| Artifact | SHA-256 |
| --- | --- |
| Trainer (training source `1711787`, runner `2d6295c`) | `057571a4afa52cde3ae5c4b258c18536716f91c4f60ab601286074244804096d` |
| Benchmark | `9090e0031d9a6d2c918c79d2691e2d7a92d44a976a66653b401dc233004c2e89` |
| Open-loop evaluator | `e93a75b0aaaff2725bef0026d18db0dc4079b49f918c185cc28190c89afcae3a` |
| Headless simulator | `2a1c09f5c84cb76dd3c654f33ce6fb3778c2805832ad9bcfc5cdbbb97fcdf87d` |
| GUI simulator | `0cec102fa31f875b418aa22c485db8c1c82b50f8c2a7eab81c63629f13d3245b` |
| Training dataset | `77acffd68edbe3d5b352bb290693bfa86ef80045554488f2c94c2aaed72e5fe0` |
| Independent in-range dataset | `afbc11b19ce91cf60a4d600e5dd47f551f4974f03a9f99c22aff4199494c81b3` |
| Shifted dataset | `709ffc3eec7ee96280eba0382c5015a7d822dafaae7e990220a85e9e116fb3bb` |

Before control validation, a reporting-only benchmark revision separated
planner-bound clipping from physical plant saturation. The original executable
(`56462cf55bdaf2c96e42b830632d9e61f61608509996787764308c8839969c3c`)
remains as `bin/lewm-bench-skyjepa-initial`. Paired four-controller smoke tests
in `reporting-regression/` verify unchanged numerical tracking/correction results.

The earlier `simulation/preliminary-seed-7-figure-eight.json` ran concurrently
with training; its 17.49 ms p95 is excluded from timing gates. Final control
measurements ran without our training/builds; other GPU applications, including
Isaac Sim, remained running and are captured in report snapshots. GUI timing is
also separate from headless benchmarking.

### Simulator media provenance

The current MP4 is a byte-identical copy of
`simulation/seed-7-figure-eight.mp4` in the artifact root. Its `.capture.json`
sidecar records exact simulator/capture commands and hashes. The run uses seed-7
weights, raw domain seed 31415, planner seed 7, 512 candidates, horizon 15 and
fresh-prior warm start. This is an in-range demonstration.

- MP4: 20 seconds, 1280×800, H.264, 30 fps, 600 frames, 5,328,617 bytes;
  SHA-256 `12a96c30fc64b494566ac701628f47af1fe96964c3af88a127f07caad7606314`.
- GIF: video seconds 8–16, 800×500, 10 fps, 80 frames, 5,799,904 bytes;
  SHA-256 `a43535defd33d94b2d5a03c649935244f2ee9b1ab6f36bbd8f63a92aa8def86d`.
- Checkpoint manifest file SHA-256:
  `cf97165c25a71906ce0cb8ae36ecbdcccc380137a7405dd83c38476550d660c6`.
- GIF encoder: FFmpeg `n8.1.2-22-g94138f6973-20260717`, palette 64, no dithering.
  Reproduction command is in the current guide. Media decoding, distinct frames,
  source/hash agreement and visual inspection were checked before publication.

### Rechecking or extending the experiment

The local-uv runner verifies and reuses completed reports only when command,
checkpoint, executable and dataset identities match:

```bash
SKYJEPA_ARTIFACT_ROOT=/home/rozgo/.stable_worldmodel/le-wm-nv-data/skyjepa-remediation-v3
uv run --locked scripts/evaluate-skyjepa-remediation.py "$SKYJEPA_ARTIFACT_ROOT" validation
uv run --locked scripts/evaluate-skyjepa-remediation.py "$SKYJEPA_ARTIFACT_ROOT" test
uv run --locked scripts/evaluate-skyjepa-remediation.py "$SKYJEPA_ARTIFACT_ROOT" open-loop
uv run --locked scripts/summarize-skyjepa-remediation.py "$SKYJEPA_ARTIFACT_ROOT" /tmp/skyjepa-review-summary.json
```

Choose a new summary filename if it already exists. A new experiment needs a
fresh artifact root and a recorded protocol before inspecting its test results.
The staged training runner is `scripts/train-skyjepa-remediation.sh`; it requires
an audited `data-pilot` directory and an explicitly supplied frozen trainer.
The capture runner is `scripts/capture-skyjepa-remediation.py`; it refuses to
overwrite existing captures. Keep training, builds and GUI capture separate
from timed control measurements, and retain unsuccessful cases with their settings.
