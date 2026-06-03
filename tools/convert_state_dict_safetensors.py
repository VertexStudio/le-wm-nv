#!/usr/bin/env python3
"""Convert a PyTorch state_dict checkpoint into safetensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--strip-prefix",
        action="append",
        default=[],
        help="prefix to remove from checkpoint keys; may be passed more than once",
    )
    return parser.parse_args()


def state_dict_from_checkpoint(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
        state_dict = checkpoint.get("state_dict")
        if isinstance(state_dict, dict) and state_dict:
            if all(torch.is_tensor(value) for value in state_dict.values()):
                return state_dict
    raise TypeError(
        "checkpoint must be a raw state_dict or contain a tensor-only 'state_dict'"
    )


def strip_prefixes(key: str, prefixes: list[str]) -> str:
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    state_dict = state_dict_from_checkpoint(checkpoint)

    tensors: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        out_key = strip_prefixes(key, args.strip_prefix)
        if out_key in tensors:
            raise ValueError(f"duplicate output tensor key after prefix stripping: {out_key}")
        tensors[out_key] = tensor.detach().cpu().contiguous()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output, metadata={"format": "pt"})
    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"tensors={len(tensors)}")


if __name__ == "__main__":
    main()
