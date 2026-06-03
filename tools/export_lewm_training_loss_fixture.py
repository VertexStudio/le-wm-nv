#!/usr/bin/env python3
"""Export deterministic LeWM training-loss tensors from the Python implementation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--stable-worldmodel-root",
        default=os.environ.get("STABLE_WORLDMODEL_ROOT")
        or os.environ.get("STABLE_WORLDMODEL_PY"),
        help="local stable-worldmodel source tree to prepend to PYTHONPATH",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--time", type=int, default=5)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--device",
        choices=("cuda",),
        default="cuda",
        help="CUDA backend used for loss export",
    )
    return parser.parse_args()


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(tensor.detach().cpu().numpy().astype("float32"))


def scalar_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32)


def main() -> None:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2")
    if args.time < 3:
        raise ValueError("--time must be at least 3")
    if args.dim < 2:
        raise ValueError("--dim must be at least 2")
    if args.action_dim < 1:
        raise ValueError("--action-dim must be at least 1")
    if args.stable_worldmodel_root:
        sys.path.insert(0, str(Path(args.stable_worldmodel_root).resolve()))

    from stable_worldmodel.wm.loss import PLDMLoss, TemporalStraighteningLoss

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA loss export requires torch.cuda.is_available()")
    device = torch.device(args.device)

    pldm = PLDMLoss().to(device).eval()
    temporal = TemporalStraighteningLoss().to(device).eval()

    with torch.no_grad():
        z = torch.randn(
            args.batch_size,
            args.time,
            args.dim,
            dtype=torch.float32,
            device=device,
        )
        a_pred = torch.randn(
            args.batch_size,
            args.time - 1,
            args.action_dim,
            dtype=torch.float32,
            device=device,
        )
        a_target = torch.randn_like(a_pred)

        pldm_out = pldm(z, a_pred, a_target)
        temporal_loss = temporal(z)
        torch.cuda.synchronize(device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        z=tensor_to_numpy(z),
        a_pred=tensor_to_numpy(a_pred),
        a_target=tensor_to_numpy(a_target),
        idm_loss=scalar_to_numpy(pldm_out["idm_loss"]),
        temp_align_loss=scalar_to_numpy(pldm_out["temp_align_loss"]),
        std_loss=scalar_to_numpy(pldm_out["std_loss"]),
        std_t_loss=scalar_to_numpy(pldm_out["std_t_loss"]),
        cov_loss=scalar_to_numpy(pldm_out["cov_loss"]),
        cov_t_loss=scalar_to_numpy(pldm_out["cov_t_loss"]),
        temporal_straightening_loss=scalar_to_numpy(temporal_loss),
    )

    print(f"fixture={args.output}")
    print(f"device={device}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_device={torch.cuda.get_device_name(device)}")
    print(f"batch_size={args.batch_size}")
    print(f"time={args.time}")
    print(f"dim={args.dim}")
    print(f"action_dim={args.action_dim}")


if __name__ == "__main__":
    main()
