# Drone All-Data World Model Training Run

Run ID: `drone-state-lewm-all-data-20260612-235255`

Date: 2026-06-13

## Purpose

Train the drone vector LeWM on every valid sliding window from the imported
drone racing dataset. This run intentionally ignores the metadata train/eval
episode split so the model can use all available real flight dynamics data.

This report records the run as a comparison point for later model, planner, and
dataset changes. Checkpoints are local artifacts and are not tracked in git.

## Command

```bash
target/release/lewm-train-drone \
  --device cuda \
  --dataset-dir /home/rozgo/.stable_worldmodel/le-wm-nv-data/drone-racing-autonomous-100hz \
  --output-dir /home/rozgo/.stable_worldmodel/le-wm-nv-runs/drone-state-lewm-all-data-20260612-235255 \
  --train-all-data \
  --epochs 100 \
  --batch-size 256 \
  --history-steps 8 \
  --horizon-steps 50 \
  --log-every 10 \
  --save-every 500
```

## Dataset Coverage

- Imported rows: `58,239`
- Imported episodes/flights: `18`
- Sample rate: `100 Hz`
- Training row source: `all_valid_rows`
- Valid sliding windows: `57,195`
- Batches per epoch: `223`
- Total epochs: `100`
- Total optimizer steps: `22,300`

Window shape:

- `history_steps = 8`
- `horizon_steps = 50`
- `sequence_steps = 59`
- observations: `[batch, 59, 20]`
- actions: `[batch, 59, 4]`
- target deltas: `[batch, 59, 13]`

The observation and target schema keeps `vbat` and `delta_vbat`.

## Performance

Hardware observed during the run:

- GPU: NVIDIA GeForce RTX 4090
- Trainer VRAM: about `4.18 GiB`
- Total VRAM used during training: about `5.45 GiB`
- GPU utilization: `99-100%`
- Memory-controller utilization: about `56-58%`
- Power draw: about `371-378 W`
- GPU temperature: about `70-75 C`

Throughput:

- Elapsed training time: `628.8 s`
- Mean step rate: `35.49 steps/s`
- Mean step time: `28.18 ms/step`

## Loss Summary

Initial logged loss at step 1:

- total: `2.1279259`
- state_prediction: `1.9654360`

Final logged loss at step 22,300:

- total: `0.2661571`
- state_prediction: `0.2043576`
- temporal_alignment: `0.010423715`
- std: `0.24455294`
- std_t: `0.6662478`
- covariance: `0.019967476`
- covariance_t: `0.21681556`
- temporal_straightening: `-0.54001254`

Last 100 optimizer steps average:

- total: `0.2879059`
- state_prediction: `0.2274850`

Last 500 optimizer steps average:

- total: `0.2860373`
- state_prediction: `0.2259356`

Interpretation: the scalar losses show clear learning from random/init state,
but they are only a first filter. The next comparison should use per-dimension
error, action sensitivity, and autoregressive rollout behavior.

## Local Artifacts

Run directory:

```text
/home/rozgo/.stable_worldmodel/le-wm-nv-runs/drone-state-lewm-all-data-20260612-235255
```

Saved weights:

```text
/home/rozgo/.stable_worldmodel/le-wm-nv-runs/drone-state-lewm-all-data-20260612-235255/final.safetensors
/home/rozgo/.stable_worldmodel/le-wm-nv-runs/drone-state-lewm-all-data-20260612-235255/latest.safetensors
```

Optimizer states:

```text
/home/rozgo/.stable_worldmodel/le-wm-nv-runs/drone-state-lewm-all-data-20260612-235255/final-optimizer.safetensors
/home/rozgo/.stable_worldmodel/le-wm-nv-runs/drone-state-lewm-all-data-20260612-235255/optimizer.safetensors
```

Metrics:

```text
/home/rozgo/.stable_worldmodel/le-wm-nv-runs/drone-state-lewm-all-data-20260612-235255/metrics.jsonl
/home/rozgo/.stable_worldmodel/le-wm-nv-runs/drone-state-lewm-all-data-20260612-235255/training-state.json
```

## Comparison Notes

Earlier smoke training with `--max-steps 20` only performed 20 optimizer steps.
With batch size 64, that exposed the model to only 1,280 training windows. This
all-data run performs 22,300 optimizer steps with batch size 256 over all 57,195
valid windows per epoch.

Future comparisons should keep these fields fixed or explicitly report changes:

- dataset artifact path
- `train_all_data`
- `history_steps`
- `horizon_steps`
- `batch_size`
- `epochs`
- learning rate and weight decay
- model config
- normalized target schema
- per-dimension prediction error
- action sensitivity spans
- rollout/planning behavior
