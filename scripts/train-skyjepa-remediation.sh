#!/usr/bin/env bash
# Reproduce the preregistered pilot. Requires a version-2 audit in data-pilot.
# The caller supplies a frozen trainer executable so later builds cannot change
# the experiment between stages. No existing artifacts are overwritten.
set -euo pipefail
artifact_root="${1:?usage: train-skyjepa-remediation.sh ARTIFACT_ROOT FROZEN_TRAINER}"
trainer_bin="${2:?provide the frozen trainer executable}"
test -x "$trainer_bin"
test -f "$artifact_root/data-pilot/audit.json"

for training_seed in 7 17 29; do
    latent_dir="$artifact_root/seed-$training_seed/latent"
    prober_dir="$artifact_root/seed-$training_seed/prober"
    mkdir -p "$artifact_root/seed-$training_seed"
    resume_args=()
    if test -f "$latent_dir/latent-current.json"; then resume_args=(--resume); fi
    "$trainer_bin" --dataset-dir "$artifact_root/data-pilot" --output-dir "$latent_dir" \
        --stage latent --split-by domains --seed "$training_seed" --batch-size 2048 \
        --latent-max-steps 1000 --warmup-steps 200 --cosine-steps 800 \
        --max-lr 0.005 --min-lr 0.0001 --log-every 100 --save-every 1000 \
        "${resume_args[@]}" | tee -a "$artifact_root/seed-$training_seed/latent-console.log"
    resume_args=()
    if test -f "$prober_dir/prober-current.json"; then resume_args=(--resume); fi
    "$trainer_bin" --dataset-dir "$artifact_root/data-pilot" --output-dir "$prober_dir" \
        --stage prober --latent-checkpoint "$latent_dir" --split-by domains \
        --seed "$training_seed" --batch-size 2048 --prober-max-steps 5000 \
        --warmup-steps 500 --cosine-steps 4500 --max-lr 0.005 --min-lr 0.0001 \
        --log-every 100 --save-every 1000 "${resume_args[@]}" \
        | tee -a "$artifact_root/seed-$training_seed/prober-console.log"
done
