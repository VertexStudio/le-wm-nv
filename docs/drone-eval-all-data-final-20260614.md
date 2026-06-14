# Drone All-Data World Model Evaluation

Date: 2026-06-14

Checkpoint:

```text
~/.stable_worldmodel/le-wm-nv-runs/drone-state-lewm-all-data-20260612-235255/final.safetensors
```

This report records the first quantitative dynamics evaluation for the
100-epoch all-data drone LeWM checkpoint. The checkpoint was trained on all
valid windows, so these numbers measure in-distribution model fit and rollout
behavior, not held-out generalization.

## Command

```bash
cargo run --release --locked --bin lewm-eval-drone -- \
  --device cuda \
  --dataset-dir "$HOME/.stable_worldmodel/le-wm-nv-data/drone-racing-autonomous-100hz" \
  --weights "$HOME/.stable_worldmodel/le-wm-nv-runs/drone-state-lewm-all-data-20260612-235255/final.safetensors" \
  --config "$HOME/.stable_worldmodel/le-wm-nv-runs/drone-state-lewm-all-data-20260612-235255/model-config.json" \
  --output-dir target/drone-eval/all-data-final \
  --history-steps 8 \
  --horizon-steps 100 \
  --batch-size 256 \
  --max-batches 64
```

Local artifacts:

```text
target/drone-eval/all-data-final/metrics.json
target/drone-eval/all-data-final/replay.json
```

## Batch Loss

The eval tool used the dataset metadata eval partition:

- Eval rows: `18,160`
- Eval batches run: `64`
- Batch size: `256`
- Mean normalized state-prediction loss: `0.25623435`

Because this checkpoint was trained with `--train-all-data`, the metadata eval
partition was not held out during training.

## Replay Rollout

The default replay selection chose the highest-motion metadata eval window and
then rolled the whole episode autoregressively:

- Replay kind: `full_episode`
- Prediction mode: `autoregressive_full_episode`
- Start row: `53,401`
- Episode: `17`
- Duration: `48.37 s`
- Frames: `4,838`
- Actual path length: `193.568 m`
- Actual net displacement: `7.296 m`
- Model chunk steps: `4,837`

The first `8` frames are copied history. Physical replay error starts after the
history context.

## Horizon Error Summary

The current all-gates planner uses a `40` step horizon at `100 Hz`, so the most
relevant open-loop MPC horizon is `0.4 s`.

| Horizon | Mean Pos Error | RMS Pos Error | Max Pos Error | Mean Att Error | Max Att Error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0.4 s` / 40 steps | `0.277 m` | `0.360 m` | `0.601 m` | `0.113 rad` | `0.217 rad` |
| `1.0 s` / 100 steps | `0.738 m` | `0.917 m` | `1.968 m` | `0.294 rad` | `0.708 rad` |
| `2.0 s` / 200 steps | `2.066 m` | `2.549 m` | `4.647 m` | `0.993 rad` | `2.737 rad` |
| `5.0 s` / 500 steps | `5.798 m` | `6.808 m` | `12.003 m` | `1.412 rad` | `3.136 rad` |
| `48.37 s` full episode | `38.569 m` | `41.732 m` | `56.301 m` | `2.080 rad` | `3.142 rad` |

Endpoint errors:

- `0.4 s`: position `0.580 m`, attitude `0.213 rad`
- `1.0 s`: position `1.968 m`, attitude `0.708 rad`
- `5.0 s`: position `12.003 m`, attitude `2.935 rad`
- `48.37 s`: position `50.597 m`, attitude `2.384 rad`

## Interpretation

This strengthens the current claim in a narrow way: the learned LeWM dynamics
model produces usable short-horizon predictions at the same horizon used by the
current MPC-style gate planner. The full-episode open-loop rollout drifts
substantially, so the current checkpoint should not be described as a long-run
drone simulator without replanning or correction.

For control, the relevant result is the `0.4 s` planner-horizon behavior plus
the existing all-gates closed-loop replay. For stronger evidence, the next
measurement should evaluate the same command across checkpoints saved during
training to show how quickly the model reaches usable short-horizon quality.
