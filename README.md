# le-wm-nv

NVIDIA/CUDA-first LeWM and SkyJEPA training, inference, and control runtime.

![le-wm-nv CUDA runtime architecture](docs/le-wm-nv.png)

This repo supports two model families. LeWM remains the upstream-compatible
image/vector baseline from `stable-worldmodel`. SkyJEPA is a separate,
repo-native state/action world model for long-horizon quadrotor dynamics and
metric MPPI control. SkyJEPA does not replace or remove LeWM.
The runtime target is Linux with NVIDIA hardware, CUDA, cuDNN, nvJPEG,
NVDECODE, and Candle CUDA tensors. The hot paths are:

```text
image/video observation -> CUDA preprocess -> LeWM encode -> candidate rollout -> cost -> action
vector/state observation -> normalize -> LeWM encode -> candidate rollout -> cost -> action
UAV state18/action4 -> SkyJEPA TCN/GRU -> physics prober -> metric MPPI -> rotor forces
```

## SkyJEPA UAV control

![SkyJEPA pipeline from simulation data synthesis through latent dynamics and physics-inspired probing to sampling-based control](docs/skyjepa-pipeline.jpg)

SkyJEPA learns UAV dynamics from state/action logs and uses those predictions
to plan rotor commands. Our Rust/Candle implementation trains and runs on CUDA,
with a dedicated 200 Hz rotor-physics simulator and a 20 Hz controller.

The flight controller is a hybrid: a **hand-written geometric controller**
provides the initial commands, and **MPPI uses SkyJEPA's learned predictions**
to improve them. Basic stabilization and learned dynamics have distinct roles.

[![SkyJEPA seed 7 flying a randomized UAV on a figure-eight; click for the full video](docs/skyjepa-v3-figure-eight.gif)](docs/skyjepa-v3-figure-eight.mp4)

[Full 20-second simulator video](docs/skyjepa-v3-figure-eight.mp4), at normal
speed. Yellow: executed. Cyan: reference. Magenta: predicted.
Green bars: commanded rotor forces. HUD timing includes the visual workload;
the headless measurements below are the performance reference.

### Training and control results

Three models trained in approximately **34 minutes each on a shared RTX 4090**,
using 1,600 ten-second training episodes from a 2,000-episode dataset.
Training, validation and test partitions use different physical vehicle
configurations. MPPI uses 512 candidates and a 15-step horizon. On the same
63 hover/circle/figure-eight test cases:

| Controller | Tracking passes | Mean trajectory RMSE | Worst trajectory RMSE | Planning p95 |
| --- | ---: | ---: | ---: | ---: |
| Random-weight model + MPPI | 51/63 | 0.6035 m | 1.8637 m | 8.931 ms |
| Hand-written controller alone | 63/63 | 0.2153 m | 0.5706 m | 0.0015 ms |
| Trained SkyJEPA + controller, three seeds | **63/63 each** | **0.2009–0.2059 m** | **0.3781–0.4231 m** | 8.818–8.936 ms |
| Nominal-physics MPPI + controller | 63/63 | 0.1609 m | 0.3038 m | 3.215 ms |

**Training produces useful behavior:** tracking error is roughly two-thirds
lower than with random weights. Adding SkyJEPA to the hand-written controller
reduces mean error by 4.4–6.7% and worst trajectory RMSE by 25.9–33.7%.

Across all three training seeds and four test conditions—including ±10% hover
calibration error and heavier, slower-motor plants—the learned controllers pass
**756/756 tracking runs with no ground contact**. They pass 741/756 of the
stricter 10 ms p95 timing checks, with zero 50 ms control-deadline misses.

This is a working learned-control system. The next targets are better
long-horizon predictions and tighter latency tails; nominal-physics MPPI is
currently the most accurate and fastest predictive controller in these tests.

The [SkyJEPA guide](docs/skyjepa.md) explains the architecture, what each
controller contributes, current results, dataset format, and how to train,
evaluate and run the simulator. The implementation follows the
[SkyJEPA paper](https://arxiv.org/html/2606.23444), with its paper-derived
contracts and repo-specific design choices documented in that guide.

## Mandate

Performance is the primary acceptance criterion. The repo is not a portability
layer, and non-Linux/non-NVIDIA targets are intentionally out of scope.

Runtime work should keep media buffers, preprocessed tensors, embeddings,
candidate action batches, rollouts, costs, and selected actions in the
Rust/Candle CUDA path. Python is included for bootstrap, checkpoint conversion,
data export, and parity checks against the official implementation. Python is
not the deployment runtime.

When Candle lacks a needed NVIDIA primitive, the preferred direction is a
focused Candle CUDA op, a direct NVIDIA library binding, or a CUDA-compatible
crate that preserves device residency.

The strategic use case is fast learned dynamics for control. Given recent
observation/action logs from an unknown platform, the repo should be able to
train a compact action-conditioned LeWM world model in minutes on one GPU, then
use that model as the predictive core for MPC-style rollout, scoring, and
control selection. That makes pre-deployment or in-transit model refresh
plausible: collect logs, train, run fixed validation probes, and load the model
before the vehicle starts the real task. This is a world-model claim, not a
claim about vision, navigation, or full autonomy.

Validation claims must stay inside the logged data distribution. Trainers
produce model-family-specific checkpoints; task-specific evaluators must back
control or dynamics-quality claims. SkyJEPA includes per-horizon open-loop
metrics and a closed-loop simulator, while LeWM keeps its existing parity and
drone evaluation paths.

## Capabilities

- LeWM model runtime: ViT-Tiny image encoder, vector MLP observation encoder,
  projector, action encoder, predictor, latent rollout, goal embedding, goal
  cost, and session caching.
- LeWM planning: CEM, MPPI, and iCEM over Candle CUDA tensors.
- NVIDIA image/video ingest: nvJPEG decode into CUDA tensors, packed RGB/BGR
  CUDA preprocessing, NV12 CUDA preprocessing, and NVDECODE capability/parser
  plumbing.
- LeWM training surface: upstream-style predicted embedding loss plus SIGReg,
  batch-loss API, AdamW training CLIs, PushT HDF5 dataset streaming, drone
  vector-observation dataset training, and safetensors save/reload.
- Native SkyJEPA surface: full UAV state18/rotor-force action4 schema,
  causal-TCN encoders, recursive GRU latent dynamics, SIGReg, frozen-latent
  physics-inspired prober training, differentiable SO(3) integration, batched MPPI,
  audited domain-randomized data generation, resumable best-checkpoint staged
  training, fixed-normalization long-horizon evaluation with baselines,
  trim-aware low-latency control, closed-loop scenario gating, and a dedicated
  Bevy rotor-force simulator.
- Python bootstrap tooling: official `stable-worldmodel[train]` package via
  `uv`, checkpoint conversion, PushT batch export, Python parity fixture export,
  and Python-vs-Rust image-planning benchmark scripts.
- Hugging Face checkpoint download is available with `--features hub`.

The audited upstream `stable-worldmodel` commit is tracked in
[docs/upstream-stable-worldmodel.md](docs/upstream-stable-worldmodel.md).
The SkyJEPA implementation and paper/code assumptions are tracked in
[docs/skyjepa.md](docs/skyjepa.md).

## LeWM Runtime Extensions

The image LeWM runtime is a Rust/Candle port of the audited upstream
`stable-worldmodel` architecture: ViT image encoder, projector, action encoder,
AdaLN-conditioned predictor, prediction projection, autoregressive latent
rollout, and goal-embedding cost. Checkpoint tensor layout and model math are
kept compatible with upstream image LeWM parity fixtures.

Repo-native extensions live around that core instead of replacing it:

- Modular observation encoders: image observations use the upstream-compatible
  ViT path; vector/state observations use a `VectorMlp` encoder with the same
  LeWM action encoder, predictor, and autoregressive rollout pattern.
- Drone vector LeWM: `lewm-drone-import` imports vector/state logs and
  `lewm-train-drone` trains the modular vector-observation model with the same
  LeWM objective used by the image trainers. No supervised decoder head is part
  of the model. The drone trainer follows upstream LeWM history semantics:
  each training sample contains `history_steps + num_preds` observations, the
  predictor has `history_steps` positional frames, and longer horizons are
  produced only by autoregressive rollout during planning/evaluation.

Architecture-preserving performance work is allowed and should be documented
here when landed. It may cache non-learned tensors, reduce tensor assembly,
reuse fixed-shape workspaces, add focused CUDA kernels, or use CUDA graph
capture. It must not change learned layer shapes, checkpoint tensor layout,
positional-embedding semantics, history semantics, predictor depth/heads, action
encoder math, rollout horizon, planner sample budget, controller cadence, or
silently introduce CPU planning/scoring paths. Runtime optimization benchmarks
must hold model and planner settings fixed.

## Prerequisites

- Linux host with NVIDIA GPU
- CUDA toolkit and driver libraries
- cuDNN available to Candle
- `libnvjpeg.so`
- `libnvcuvid.so`
- Rust toolchain from `rust-toolchain.toml`
- `uv`

The build script requires `libnvjpeg.so` and `libnvcuvid.so`. Set `CUDA_HOME`,
`CUDA_PATH`, or `NVIDIA_VIDEO_CODEC_SDK_PATH` if they are not under standard
system library paths.

## Build

```bash
cargo check --locked --all-targets
cargo test --locked
```

With Hugging Face Hub checkpoint download:

```bash
cargo check --locked --features hub --all-targets
```

## Python Bootstrap

The repo includes `.python-version`, `pyproject.toml`, and `uv.lock`.
`pyproject.toml` defines the supported Python range and dependencies, including
`stable-worldmodel[train]`.

```bash
uv sync --locked --no-dev
```

Convert a PyTorch state dict to safetensors:

```bash
uv run --locked --no-dev \
  python tools/convert_state_dict_safetensors.py \
  --input /path/to/weights.pt \
  --output target/model.safetensors
```

## LeWM Parity

Export a deterministic CUDA fixture from the official Python implementation:

```bash
uv run --locked --no-dev \
  python tools/export_lewm_fixture.py \
  --model quentinll/lewm-pusht \
  --device cuda \
  --output target/lewm-pusht-python-cuda.npz
```

Compare Rust/Candle CUDA against that fixture:

```bash
cargo run --release --locked --features hub --bin lewm-compare-fixture -- \
  --device cuda \
  --fixture target/lewm-pusht-python-cuda.npz \
  --hf-repo quentinll/lewm-pusht
```

Run checkpoint-backed planning from fixture tensors:

```bash
cargo run --release --locked --features hub --bin lewm-plan-fixture -- \
  --device cuda \
  --fixture target/lewm-pusht-python-cuda.npz \
  --hf-repo quentinll/lewm-pusht \
  --planner icem \
  --samples 128 \
  --iterations 3 \
  --seed 7
```

Validation snapshot on 2026-06-03, RTX 4090, `quentinll/lewm-pusht`,
PyTorch `2.12.0+cu130`, CUDA `13.0`:

| Output | Max Abs |
| --- | ---: |
| `emb` | `5.731881e-4` |
| `act_emb` | `4.768372e-7` |
| `pred` | `7.328391e-4` |
| `rollout` | `6.533712e-4` |
| `cost` | `5.619049e-3` |

Cost argmin was stable for the fixture batch.

## Performance Snapshot

![LeWM PushT image planning latency: Python/PyTorch vs Rust/Candle](docs/lewm-image-plan-python-rust-benchmark.svg)

Snapshot on 2026-06-03, RTX 4090, `quentinll/lewm-pusht`, CUDA 13.0,
`planner=icem`, `samples=1024`, `iterations=5`, `horizon=5`, `history_size=3`.
Metric is synchronized CUDA p50 wall time after 2 warmup runs and 5 measured
runs. Python is vanilla `stable-worldmodel` LeWM through PyTorch; Rust is
`lewm-plan-images` with nvJPEG decode, Candle CUDA encode/rollout/scoring, and
Rust-native planning. In this image-input PushT benchmark, Rust/Candle is faster
across the hot path: 3-4x for media decode/preprocess, 1.37-1.51x for image
encoding, 1.13x for iCEM planning, and 1.66x for selected-score evaluation.

## Image Planning

Plan from JPEG current/goal images through nvJPEG, CUDA preprocessing, LeWM,
and Rust-native planning:

```bash
cargo run --release --locked --features hub --bin lewm-plan-images -- \
  --device cuda \
  --hf-repo quentinll/lewm-pusht \
  --current current.jpg \
  --goal goal.jpg \
  --planner icem \
  --samples 1024 \
  --iterations 5 \
  --output target/reports/lewm-pusht-plan.html
```

## Training

The drone vector trainer follows the upstream LeWM shifted-context objective:

```text
ctx = embedding[:, :history_size]
target = stopgrad(embedding[:, num_preds:])
pred = predictor(ctx, action[:, :history_size])

loss = mse(pred, target)
     + 0.09 * SIGReg(online_embeddings)
```

Shared SIGReg uses the upstream default `17` knots, `1024` random projections,
and the upstream Gaussian-windowed integration weights. The trainer CLIs do not
expose alternate auxiliary loss weights.

Export a PushT image/action batch and run a Rust/Candle CUDA training step:

```bash
uv run --locked --no-dev \
  python tools/export_pusht_lewm_training_batch.py \
  --output target/pusht-lewm-training-batch.npz \
  --batch-size 2 \
  --history-size 3 \
  --action-block 5 \
  --seed 7

cargo run --release --locked --bin lewm-train-batch -- \
  --device cuda \
  --batch-npz target/pusht-lewm-training-batch.npz \
  --steps 10 \
  --lr 1e-5 \
  --output target/pusht-lewm-trained.safetensors
```

Train LeWM from the PushT HDF5 dataset without Python in the data/training
path. Long-running training outputs should live outside `target/`, because
`target/` is disposable build output:

```bash
tools/launch_pusht_from_scratch.sh
```

By default the launcher writes checkpoints, optimizer state, metrics, logs, and
`train.pid` to
`~/.stable_worldmodel/le-wm-nv-runs/pusht-from-scratch-b96`. If
`training-state.json` already exists there, it resumes from that run directory.
Override settings with environment variables, for example
`RUN_DIR=/mnt/runs/pusht-b96 BATCH_SIZE=64 tools/launch_pusht_from_scratch.sh`.

Equivalent direct command:

```bash
cargo run --release --locked --bin lewm-train-pusht -- \
  --device cuda \
  --dataset-h5 ~/.stable_worldmodel/pusht_expert_train.h5 \
  --epochs 100 \
  --batch-size 96 \
  --history-size 3 \
  --action-block 5 \
  --output-dir ~/.stable_worldmodel/le-wm-nv-runs/pusht-from-scratch-b96
```

`lewm-train-pusht` reads `pusht_expert_train.h5` natively through Rust HDF5
with in-process Blosc filter support. It reproduces the Python exporter dataset
semantics: valid row selection from `episode_idx`, `step_idx`, and `ep_len`;
image history rows at `row + idx * action_block`; flattened action blocks; and
dataset-wide action mean/std normalization. Because PushT H5 pixels are already
decoded RGB arrays, the optimized path is HDF5 host reads, raw RGB
host-to-CUDA transfer, CUDA resize/normalize/history assembly, and LeWM
training on Candle CUDA tensors. It does not use nvJPEG or NVDECODE.

The trainer writes `metrics.jsonl`, `dataset-summary.json`, `model-config.json`,
`training-state.json`, `latest.safetensors`, periodic
`checkpoint-step-*.safetensors` files, `optimizer.safetensors`, periodic
`optimizer-step-*.safetensors` files, `final.safetensors`, and
`final-optimizer.safetensors`. Use `--init-safetensors` for a weights-only warm
start. Use `--resume-dir` for exact continuation from `latest.safetensors`,
`optimizer.safetensors`, and `training-state.json`; the trainer resumes from the
saved `global_step`, which maps deterministically back to the same epoch shuffle
and next batch.

```bash
cargo run --release --locked --bin lewm-train-pusht -- \
  --device cuda \
  --dataset-h5 ~/.stable_worldmodel/pusht_expert_train.h5 \
  --resume-dir ~/.stable_worldmodel/le-wm-nv-runs/pusht-from-scratch-b96 \
  --epochs 100 \
  --batch-size 96 \
  --history-size 3 \
  --action-block 5 \
  --output-dir ~/.stable_worldmodel/le-wm-nv-runs/pusht-from-scratch-b96
```

## Reports

Run the PushT environment demo through Rust planning:

```bash
uv run --locked --no-dev \
  python tools/run_pusht_lewm_rust_demo.py \
  --hf-repo quentinll/lewm-pusht \
  --planner icem \
  --replans 2 \
  --output-dir target/reports/pusht-lewm-demo
```

Run the same demo with a locally trained Rust checkpoint:

```bash
uv run --locked --no-dev \
  python tools/run_pusht_lewm_rust_demo.py \
  --weights ~/.stable_worldmodel/le-wm-nv-runs/pusht-from-scratch-b96/latest.safetensors \
  --config ~/.stable_worldmodel/le-wm-nv-runs/pusht-from-scratch-b96/model-config.json \
  --planner icem \
  --history-size 1 \
  --replans 2 \
  --output-dir ~/.stable_worldmodel/le-wm-nv-reports/pusht-from-scratch-demo
```

Run Python-vs-Rust image-planning benchmark tooling:

```bash
uv run --locked --no-dev \
  python tools/benchmark_lewm_plan_images_python.py \
  --model quentinll/lewm-pusht \
  --current current.jpg \
  --goal goal.jpg \
  --output target/bench/lewm-plan-images-python.json
```
