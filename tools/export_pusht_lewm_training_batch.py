#!/usr/bin/env python3
"""Export a PushT LeWM training batch as model-ready pixels/actions NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 - registers Blosc and other HDF5 dataset filters
import numpy as np
from PIL import Image

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    default_pusht_h5 = Path.home() / ".stable_worldmodel" / "pusht_expert_train.h5"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-h5", type=Path, default=default_pusht_h5)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dataset-row", type=int, action="append", default=None)
    parser.add_argument("--no-action-normalize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not args.dataset_h5.exists():
        raise FileNotFoundError(f"PushT H5 not found: {args.dataset_h5}")

    with h5py.File(args.dataset_h5, "r") as h5:
        rows = select_rows(h5, args)
        mean, std = action_stats(h5)
        pixels = np.stack([pixel_history(h5, row, args) for row in rows], axis=0)
        actions = np.stack(
            [
                action_history(
                    h5,
                    row,
                    args,
                    None if args.no_action_normalize else mean,
                    None if args.no_action_normalize else std,
                )
                for row in rows
            ],
            axis=0,
        )
        episode_idx = h5["episode_idx"][rows].astype(np.int64)
        step_idx = h5["step_idx"][rows].astype(np.int64)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        pixels=np.ascontiguousarray(pixels.astype("float32")),
        actions=np.ascontiguousarray(actions.astype("float32")),
        rows=np.asarray(rows, dtype=np.int64),
        episode_idx=episode_idx,
        step_idx=step_idx,
        action_mean=mean.astype("float32"),
        action_std=std.astype("float32"),
        action_normalized=np.asarray([not args.no_action_normalize], dtype=np.bool_),
    )
    print(f"batch={args.output}")
    print(f"dataset={args.dataset_h5}")
    print(f"rows={rows.tolist()}")
    print(f"pixels_shape={pixels.shape}")
    print(f"actions_shape={actions.shape}")
    print(f"action_normalized={not args.no_action_normalize}")


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    if args.history_size <= 0:
        raise ValueError("--history-size must be greater than zero")
    if args.action_block <= 0:
        raise ValueError("--action-block must be greater than zero")
    if args.image_size <= 0:
        raise ValueError("--image-size must be greater than zero")
    if args.dataset_row is not None and len(args.dataset_row) != args.batch_size:
        raise ValueError("--dataset-row count must match --batch-size")


def select_rows(h5: h5py.File, args: argparse.Namespace) -> np.ndarray:
    episode_idx = h5["episode_idx"][:]
    step_idx = h5["step_idx"][:]
    ep_len = h5["ep_len"][:]
    valid_until = np.asarray(
        [ep_len[int(ep)] - args.history_size * args.action_block for ep in episode_idx],
        dtype=np.int64,
    )
    valid = np.nonzero(step_idx <= valid_until)[0]
    if args.dataset_row is not None:
        rows = np.asarray(args.dataset_row, dtype=np.int64)
        valid_set = set(int(row) for row in valid)
        bad = [int(row) for row in rows if int(row) not in valid_set]
        if bad:
            raise ValueError(f"dataset rows are not valid for requested history/action block: {bad}")
        return rows
    if len(valid) < args.batch_size:
        raise ValueError(
            f"only {len(valid)} valid rows available, cannot export batch {args.batch_size}"
        )
    rng = np.random.default_rng(args.seed)
    return np.sort(rng.choice(valid, size=args.batch_size, replace=False)).astype(np.int64)


def action_stats(h5: h5py.File, chunk_size: int = 262_144) -> tuple[np.ndarray, np.ndarray]:
    actions = h5["action"]
    count = 0
    total = np.zeros(actions.shape[1], dtype=np.float64)
    total_sq = np.zeros(actions.shape[1], dtype=np.float64)
    for start in range(0, actions.shape[0], chunk_size):
        batch = actions[start : start + chunk_size].astype(np.float64)
        total += batch.sum(axis=0)
        total_sq += (batch * batch).sum(axis=0)
        count += batch.shape[0]
    mean = total / count
    variance = np.maximum(total_sq / count - mean * mean, 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def pixel_history(h5: h5py.File, row: int, args: argparse.Namespace) -> np.ndarray:
    frames = []
    for idx in range(args.history_size):
        frame_row = int(row + idx * args.action_block)
        frames.append(preprocess_image(h5["pixels"][frame_row], args.image_size))
    return np.stack(frames, axis=0)


def preprocess_image(image: np.ndarray, image_size: int) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    if image.shape[:2] != (image_size, image_size):
        image = np.asarray(
            Image.fromarray(image).resize((image_size, image_size), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN.reshape(1, 1, 3)) / IMAGENET_STD.reshape(1, 1, 3)
    return np.transpose(image, (2, 0, 1))


def action_history(
    h5: h5py.File,
    row: int,
    args: argparse.Namespace,
    mean: np.ndarray | None,
    std: np.ndarray | None,
) -> np.ndarray:
    blocks = []
    for idx in range(args.history_size):
        start = int(row + idx * args.action_block)
        block = h5["action"][start : start + args.action_block].astype(np.float32)
        if mean is not None and std is not None:
            block = (block - mean.reshape(1, -1)) / std.reshape(1, -1)
        blocks.append(block.reshape(-1))
    return np.stack(blocks, axis=0)


if __name__ == "__main__":
    main()
