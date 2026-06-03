# TODO: LeWM NVIDIA Runtime

This roadmap is scoped to LeWM training and inference on NVIDIA hardware.

## Working Protocol

- Work on `main` unless asked otherwise.
- Commit each completed capability chunk.
- Push after each commit or tight group of commits.
- Run the relevant Rust and Python checks before committing.
- Keep Python in bootstrap, conversion, export, parity, and reporting scripts.

## Baseline

- LeWM Rust/Candle model, loss, training, session, and planner code is
  scaffolded from the validated source repo.
- Python bootstrap tooling uses the official `stable-worldmodel[train]`
  package through `uv`.
- nvJPEG and NVDECODE are required NVIDIA runtime libraries.
- TD-MPC2, generic model config, C ABI, and state/vector runtime surfaces are
  intentionally absent from this repo.
- Initial validation passed on 2026-06-03:
  - `cargo check --locked --all-targets`
  - `cargo check --locked --features hub --all-targets`
  - `cargo test --locked`
  - Python tool py_compile through `uv`
  - LeWM training-loss CUDA parity
  - LeWM PushT checkpoint CUDA inference parity

## Next Work

1. Make the PushT data path training-grade.
   - Stream H5 batches instead of fixed NPZ-only mini-batches.
   - Add shuffling, batch repeat, epoch counters, and checkpoint cadence.
   - Track throughput, GPU memory, and loss curves.

2. Tighten LeWM image/video ingest.
   - Reuse nvJPEG output buffers across decode calls.
   - Reuse history/preprocess tensors across replans.
   - Add direct video packet/demo coverage through NVDECODE.
   - Benchmark decode, preprocess, encode, rollout, and planner sections with
     CUDA synchronization.

3. Improve planner performance.
   - Reduce per-plan allocations.
   - Keep elite selection and score summaries device-resident.
   - Add planner tests and benchmarks that use LeWM goal scoring directly.
   - Add selected-action agreement checks for F16/BF16.

4. Expand Rust training.
   - Add a full PushT training CLI around the batch-loss API.
   - Save optimizer/training metadata with safetensors checkpoints.
   - Add resume support from repo-native training artifacts.
   - Compare trained checkpoint behavior against official Python outputs.

5. Add deployment interfaces.
   - Rust package loader for `config.json`, `model.safetensors`, and
     preprocess metadata.
   - LeWM-only C ABI after the Rust API is stable.
   - Explicit media input entrypoints for JPEG bytes and video packets.
