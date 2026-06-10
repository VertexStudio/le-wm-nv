#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-$HOME/.stable_worldmodel/le-wm-nv-runs/pusht-from-scratch-b96}"
DATASET_H5="${DATASET_H5:-$HOME/.stable_worldmodel/pusht_expert_train.h5}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-96}"
HISTORY_SIZE="${HISTORY_SIZE:-3}"
ACTION_BLOCK="${ACTION_BLOCK:-5}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LOG_EVERY="${LOG_EVERY:-100}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
BLOSC_THREADS="${BLOSC_THREADS:-4}"

mkdir -p "$RUN_DIR/logs"

if [ -f "$RUN_DIR/train.pid" ]; then
  old_pid="$(cat "$RUN_DIR/train.pid")"
  if [ -n "$old_pid" ] && ps -p "$old_pid" >/dev/null 2>&1; then
    echo "trainer already running: pid=$old_pid run_dir=$RUN_DIR" >&2
    exit 1
  fi
fi

if [ ! -f "$DATASET_H5" ]; then
  echo "missing PushT dataset: $DATASET_H5" >&2
  exit 1
fi

cargo build --release --locked --bin lewm-train-pusht

cmd=(
  target/release/lewm-train-pusht
  --device "$DEVICE"
  --dataset-h5 "$DATASET_H5"
  --output-dir "$RUN_DIR"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --history-size "$HISTORY_SIZE"
  --action-block "$ACTION_BLOCK"
  --image-size "$IMAGE_SIZE"
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --log-every "$LOG_EVERY"
  --save-every "$SAVE_EVERY"
  --blosc-threads "$BLOSC_THREADS"
)

if [ -f "$RUN_DIR/training-state.json" ]; then
  cmd+=(--resume-dir "$RUN_DIR")
fi

log="$RUN_DIR/logs/train-$(date +%Y%m%d-%H%M%S).log"
setsid env RUST_BACKTRACE=1 "${cmd[@]}" > "$log" 2>&1 < /dev/null &
pid="$!"
echo "$pid" > "$RUN_DIR/train.pid"

echo "pid=$pid"
echo "run_dir=$RUN_DIR"
echo "log=$log"
