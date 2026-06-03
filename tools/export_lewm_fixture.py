#!/usr/bin/env python3
"""Export deterministic LeWM inference tensors from the Python implementation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="quentinll/lewm-pusht",
        help="stable_worldmodel load_pretrained name, folder, .pt path, or HF repo id",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--stable-worldmodel-root",
        default=os.environ.get("STABLE_WORLDMODEL_ROOT")
        or os.environ.get("STABLE_WORLDMODEL_PY"),
        help="local stable-worldmodel source tree to prepend to PYTHONPATH",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        choices=("cuda",),
        default="cuda",
        help="CUDA backend used for model inference; inputs are generated on host and copied to this device",
    )
    return parser.parse_args()


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(tensor.detach().cpu().numpy().astype("float32"))


def main() -> None:
    args = parse_args()
    if args.stable_worldmodel_root:
        sys.path.insert(0, str(Path(args.stable_worldmodel_root).resolve()))

    from stable_worldmodel.wm.utils import load_pretrained

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA fixture export requires torch.cuda.is_available()")
    device = torch.device(args.device)

    model = load_pretrained(args.model, cache_dir=args.cache_dir).to(device).eval()
    action_dim = model.action_encoder.input_dim
    history = model.predictor.num_frames
    if args.horizon < history:
        raise ValueError(f"--horizon must be >= model history ({history})")

    with torch.no_grad():
        pixels = torch.randn(
            args.batch_size,
            history,
            3,
            args.image_size,
            args.image_size,
            dtype=torch.float32,
        )
        actions = torch.randn(
            args.batch_size, history, action_dim, dtype=torch.float32
        )
        action_candidates = torch.randn(
            args.batch_size,
            args.samples,
            args.horizon,
            action_dim,
            dtype=torch.float32,
        ).clamp(-1.0, 1.0)

        pixels_device = pixels.to(device)
        actions_device = actions.to(device)
        action_candidates_device = action_candidates.to(device)

        encoded = model.encode(
            {"pixels": pixels_device.clone(), "action": actions_device.clone()}
        )
        emb = encoded["emb"].contiguous()
        act_emb = encoded["act_emb"].contiguous()
        pred = model.predict(emb, act_emb).contiguous()

        emb_init = (
            emb.unsqueeze(1)
            .expand(args.batch_size, args.samples, history, emb.shape[-1])
            .contiguous()
        )
        rollout_info = {
            "pixels": pixels_device.unsqueeze(1)
            .expand(
                args.batch_size,
                args.samples,
                history,
                3,
                args.image_size,
                args.image_size,
            )
            .contiguous(),
            "emb": emb_init,
        }
        rollout = model.rollout(
            rollout_info, action_candidates_device, history_size=history
        )["predicted_emb"].contiguous()

        goal_emb = emb[:, -1].contiguous()
        cost = ((rollout[:, :, -1] - goal_emb[:, None]) ** 2).sum(
            dim=-1
        ).contiguous()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        pixels=tensor_to_numpy(pixels),
        actions=tensor_to_numpy(actions),
        action_candidates=tensor_to_numpy(action_candidates),
        goal_emb=tensor_to_numpy(goal_emb),
        emb=tensor_to_numpy(emb),
        act_emb=tensor_to_numpy(act_emb),
        pred=tensor_to_numpy(pred),
        rollout=tensor_to_numpy(rollout),
        cost=tensor_to_numpy(cost),
    )

    print(f"fixture={args.output}")
    print(f"model={args.model}")
    print(f"device={device}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    if device.type == "cuda":
        print(f"cuda_device={torch.cuda.get_device_name(device)}")
    print(f"batch_size={args.batch_size}")
    print(f"samples={args.samples}")
    print(f"history={history}")
    print(f"horizon={args.horizon}")
    print(f"action_dim={action_dim}")


if __name__ == "__main__":
    main()
