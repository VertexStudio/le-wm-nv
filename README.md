# le-wm-nv

NVIDIA/CUDA-first LeWM training and inference runtime.

This repo is focused on one model family: LeWM image world models from
`stable-worldmodel`. The runtime target is Linux with NVIDIA hardware, CUDA,
cuDNN, nvJPEG, NVDECODE, and Candle CUDA tensors. The hot path is:

```text
image/video observation -> CUDA preprocess -> LeWM encode -> candidate rollout -> cost -> action
```

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

## Capabilities

- LeWM model runtime: ViT-Tiny encoder, projector, action encoder, predictor,
  latent rollout, goal embedding, goal cost, and session caching.
- LeWM planning: CEM, MPPI, and iCEM over Candle CUDA tensors.
- NVIDIA image/video ingest: nvJPEG decode into CUDA tensors, packed RGB/BGR
  CUDA preprocessing, NV12 CUDA preprocessing, and NVDECODE capability/parser
  plumbing.
- LeWM training surface: PLDM, VCReg, temporal-straightening losses,
  batch-loss API, AdamW training CLI, and safetensors save/reload.
- Python bootstrap tooling: official `stable-worldmodel[train]` package via
  `uv`, checkpoint conversion, PushT batch export, Python parity fixture export,
  and Python-vs-Rust image-planning benchmark scripts.
- Hugging Face checkpoint download is available with `--features hub`.

The audited upstream `stable-worldmodel` commit is tracked in
[docs/upstream-stable-worldmodel.md](docs/upstream-stable-worldmodel.md).

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

Compare Rust/Candle LeWM losses against official Python CUDA losses:

```bash
uv run --locked --no-dev \
  python tools/export_lewm_training_loss_fixture.py \
  --device cuda \
  --output target/lewm-training-loss-python-cuda.npz

cargo run --release --locked --bin lewm-compare-training-loss -- \
  --device cuda \
  --fixture target/lewm-training-loss-python-cuda.npz \
  --tolerance 1e-5
```

Validation snapshot on 2026-06-03, RTX 4090:

| Loss | Max Abs |
| --- | ---: |
| `idm_loss` | `0` |
| `temp_align_loss` | `1.192093e-7` |
| `std_loss` | `0` |
| `std_t_loss` | `0` |
| `cov_loss` | `2.980232e-8` |
| `cov_t_loss` | `0` |
| `temporal_straightening_loss` | `0` |

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

Run Python-vs-Rust image-planning benchmark tooling:

```bash
uv run --locked --no-dev \
  python tools/benchmark_lewm_plan_images_python.py \
  --model quentinll/lewm-pusht \
  --current current.jpg \
  --goal goal.jpg \
  --output target/bench/lewm-plan-images-python.json
```
