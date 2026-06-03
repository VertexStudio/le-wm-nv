#!/usr/bin/env python3
"""Benchmark image-input LeWM planning through the Python implementation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
UPSTREAM_COMMIT = "40dff37fc983c5276ada65eb1c7873cefbcccd8a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="quentinll/lewm-pusht")
    parser.add_argument("--current", required=True, action="append", type=Path)
    parser.add_argument("--goal", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--stable-worldmodel-root",
        default=os.environ.get("STABLE_WORLDMODEL_ROOT")
        or os.environ.get("STABLE_WORLDMODEL_PY"),
    )
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--planner", choices=("cem", "mppi", "icem"), default="icem")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--history-size", type=int, default=None)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--elites", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--init-std", type=float, default=1.0)
    parser.add_argument("--min-std", type=float, default=1e-3)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.stable_worldmodel_root:
        sys.path.insert(0, str(Path(args.stable_worldmodel_root).resolve()))

    from stable_worldmodel.wm.utils import load_pretrained

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)

    total_started = time.perf_counter()
    model, load_ms = timed_cuda(
        device, lambda: load_pretrained(args.model, cache_dir=args.cache_dir).to(device).eval()
    )
    action_dim = model.action_encoder.input_dim
    checkpoint_history = model.predictor.num_frames
    history = args.history_size or checkpoint_history
    if history <= 0:
        raise ValueError("--history-size must be greater than zero")
    horizon = args.horizon or max(history, 5)
    if horizon < history:
        raise ValueError(f"--horizon {horizon} must be >= model history size {history}")
    elites = args.elites or min(max(args.samples // 4, 2), args.samples)
    if elites > args.samples:
        raise ValueError("--elites cannot exceed --samples")

    current_loaded, current_media_stats = bench_cuda(
        device, args, lambda: load_history(args.current, history, args.image_size, device)
    )
    current_pixels, current_info = current_loaded
    goal_loaded, goal_media_stats = bench_cuda(
        device, args, lambda: load_history(args.goal, history, args.image_size, device)
    )
    goal_pixels, goal_info = goal_loaded
    current_emb, current_encode_stats = bench_cuda(
        device, args, lambda: encode_pixels(model, current_pixels)
    )
    goal_emb, goal_encode_stats = bench_cuda(
        device, args, lambda: encode_pixels(model, goal_pixels)
    )

    def score(candidates: torch.Tensor) -> torch.Tensor:
        return score_candidates(
            model,
            current_pixels,
            current_emb,
            goal_emb,
            candidates,
            history,
        )

    planner_fn: Callable[[], PlanResult]
    if args.planner == "cem":
        planner_fn = lambda: plan_cem(args, score, horizon, action_dim, elites, device)
    elif args.planner == "mppi":
        planner_fn = lambda: plan_mppi(args, score, horizon, action_dim, device)
    else:
        planner_fn = lambda: plan_icem(args, score, horizon, action_dim, elites, device)
    plan_result, planning_stats = bench_cuda(device, args, planner_fn)
    selected_cost_tensor, selected_score_stats = bench_cuda(
        device, args, lambda: score(plan_result.sequence.unsqueeze(1))
    )

    selected_cost = float(selected_cost_tensor.detach().cpu()[0, 0])
    score_stats = summarize_scores(plan_result.scores)
    total_ms = (time.perf_counter() - total_started) * 1000.0
    payload = {
        "git_commit": git_commit(),
        "upstream_stable_worldmodel_commit": UPSTREAM_COMMIT,
        "model": args.model,
        "device": str(device),
        "dtype": "f32",
        "planner": args.planner,
        "history_size": history,
        "checkpoint_history_size": checkpoint_history,
        "horizon": horizon,
        "samples": args.samples,
        "elites": elites,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "iters": args.iters,
        "action_dim": action_dim,
        "embedding_shape": list(current_emb.shape),
        "goal_embedding_shape": list(goal_emb.shape),
        "preprocess": {
            "output_height": args.image_size,
            "output_width": args.image_size,
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
        "current_images": [str(path) for path in args.current],
        "goal_images": [str(path) for path in args.goal],
        "current_image_info": current_info,
        "goal_image_info": goal_info,
        "timing_ms": {
            "checkpoint_load": load_ms,
            "current_decode_preprocess": current_media_stats["p50_ms"],
            "goal_decode_preprocess": goal_media_stats["p50_ms"],
            "current_encode": current_encode_stats["p50_ms"],
            "goal_encode": goal_encode_stats["p50_ms"],
            "planning": planning_stats["p50_ms"],
            "selected_score": selected_score_stats["p50_ms"],
            "total": total_ms,
        },
        "benchmark_stats": {
            "current_decode_preprocess": current_media_stats,
            "goal_decode_preprocess": goal_media_stats,
            "current_encode": current_encode_stats,
            "goal_encode": goal_encode_stats,
            "planning": planning_stats,
            "selected_score": selected_score_stats,
        },
        "score": {
            "selected_cost": selected_cost,
            "final_best": score_stats["best"],
            "final_mean": score_stats["mean"],
            "final_p50": score_stats["p50"],
            "final_p95": score_stats["p95"],
            "final_min": score_stats["min"],
            "final_max": score_stats["max"],
            "best_indices": plan_result.best_indices,
            "iterations_completed": plan_result.iterations_completed,
        },
        "first_action": plan_result.sequence[:, 0, :].detach().cpu().numpy().tolist(),
        "sequence": plan_result.sequence.detach().cpu()[0].numpy().tolist(),
        "backend": "python-pytorch",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"json={args.output}")
    print(
        "planner={} selected_cost={:.6f} final_best={:.6f} plan_ms={:.3f}".format(
            args.planner, selected_cost, score_stats["best"], planning_stats["p50_ms"]
        )
    )


class PlanResult:
    def __init__(
        self,
        sequence: torch.Tensor,
        scores: torch.Tensor,
        best_indices: list[int],
        iterations_completed: int,
    ) -> None:
        self.sequence = sequence
        self.scores = scores
        self.best_indices = best_indices
        self.iterations_completed = iterations_completed


def validate_args(args: argparse.Namespace) -> None:
    if not args.current:
        raise ValueError("provide at least one --current JPEG")
    if not args.goal:
        raise ValueError("provide at least one --goal JPEG")
    if args.samples < 2:
        raise ValueError("--samples must be at least 2")
    if args.iterations <= 0:
        raise ValueError("--iterations must be greater than zero")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.iters <= 0:
        raise ValueError("--iters must be greater than zero")


def load_history(
    paths: list[Path],
    history: int,
    image_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    tensors: list[torch.Tensor] = []
    info = None
    for slot in range(history):
        path = paths[0] if len(paths) == 1 else history_path(paths, history, slot)
        image = Image.open(path).convert("RGB")
        if info is None:
            info = {
                "width": image.width,
                "height": image.height,
                "components": 3,
            }
        tensors.append(preprocess_image(image, image_size, device))
    return torch.stack(tensors, dim=0).unsqueeze(0).contiguous(), info or {}


def history_path(paths: list[Path], history: int, slot: int) -> Path:
    if len(paths) != history:
        raise ValueError(
            f"image history accepts one JPEG or exactly history_size ({history}) JPEGs, got {len(paths)}"
        )
    return paths[slot]


def preprocess_image(
    image: Image.Image,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    array = np.asarray(image, dtype=np.uint8)
    tensor = torch.from_numpy(array).to(device=device)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(torch.float32).div_(255.0)
    if tensor.shape[-2:] != (image_size, image_size):
        tensor = F.interpolate(
            tensor,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return ((tensor - mean) / std).squeeze(0).contiguous()


def encode_pixels(model: torch.nn.Module, pixels: torch.Tensor) -> torch.Tensor:
    return model.encode({"pixels": pixels.clone()})["emb"].contiguous()


def score_candidates(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    emb: torch.Tensor,
    goal_emb: torch.Tensor,
    candidates: torch.Tensor,
    history: int,
) -> torch.Tensor:
    batch, samples, _, _ = candidates.shape
    rollout_info = {
        "pixels": pixels.unsqueeze(1)
        .expand(batch, samples, history, 3, pixels.shape[-2], pixels.shape[-1])
        .contiguous(),
        "emb": emb.unsqueeze(1)
        .expand(batch, samples, history, emb.shape[-1])
        .contiguous(),
    }
    rollout = model.rollout(rollout_info, candidates, history_size=history)[
        "predicted_emb"
    ].contiguous()
    goal_last = goal_emb[:, -1].contiguous()
    return ((rollout[:, :, -1] - goal_last[:, None]) ** 2).sum(dim=-1).contiguous()


def plan_cem(
    args: argparse.Namespace,
    score: Callable[[torch.Tensor], torch.Tensor],
    horizon: int,
    action_dim: int,
    elites: int,
    device: torch.device,
) -> PlanResult:
    mean = torch.zeros((1, horizon, action_dim), device=device)
    std = torch.full_like(mean, args.init_std)
    last_candidates = None
    last_scores = None
    for _ in range(args.iterations):
        candidates = sample_candidates(mean, std, args.samples)
        scores = score(candidates)
        elite = gather_elites(candidates, scores, elites)
        mean = elite.mean(dim=1)
        std = torch.var(elite, dim=1, unbiased=True).sqrt().clamp_min(args.min_std)
        last_candidates = candidates
        last_scores = scores
    return best_plan(last_candidates, last_scores, args.iterations)


def plan_icem(
    args: argparse.Namespace,
    score: Callable[[torch.Tensor], torch.Tensor],
    horizon: int,
    action_dim: int,
    elites: int,
    device: torch.device,
) -> PlanResult:
    mean = torch.zeros((1, horizon, action_dim), device=device)
    std = torch.full_like(mean, args.init_std)
    carried_elites = None
    last_candidates = None
    last_scores = None
    keep_elites = elites
    for _ in range(args.iterations):
        sampled = sample_candidates(mean, std, args.samples)
        candidates = (
            torch.cat([sampled, carried_elites], dim=1)
            if carried_elites is not None
            else sampled
        )
        scores = score(candidates)
        elite = gather_elites(candidates, scores, elites)
        mean = elite.mean(dim=1)
        std = torch.var(elite, dim=1, unbiased=True).sqrt().clamp_min(args.min_std)
        carried_elites = elite[:, :keep_elites].contiguous()
        last_candidates = candidates
        last_scores = scores
    return best_plan(last_candidates, last_scores, args.iterations)


def plan_mppi(
    args: argparse.Namespace,
    score: Callable[[torch.Tensor], torch.Tensor],
    horizon: int,
    action_dim: int,
    device: torch.device,
) -> PlanResult:
    mean = torch.zeros((1, horizon, action_dim), device=device)
    std = torch.full_like(mean, args.noise_std)
    last_scores = None
    for _ in range(args.iterations):
        candidates = sample_candidates(mean, std, args.samples)
        scores = score(candidates)
        weights = torch.softmax(
            -((scores - scores.min(dim=1, keepdim=True).values) / args.temperature),
            dim=1,
        )
        mean = (candidates * weights[:, :, None, None]).sum(dim=1)
        last_scores = scores
    best_indices = torch.argsort(last_scores, dim=1)[:, 0]
    return PlanResult(
        sequence=mean.contiguous(),
        scores=last_scores.contiguous(),
        best_indices=[int(idx) for idx in best_indices.detach().cpu()],
        iterations_completed=args.iterations,
    )


def sample_candidates(
    mean: torch.Tensor,
    std: torch.Tensor,
    samples: int,
) -> torch.Tensor:
    noise = torch.randn(
        (mean.shape[0], samples, mean.shape[1], mean.shape[2]),
        device=mean.device,
        dtype=mean.dtype,
    )
    return (mean[:, None] + noise * std[:, None]).clamp_(-1.0, 1.0).contiguous()


def gather_elites(
    candidates: torch.Tensor,
    scores: torch.Tensor,
    elite_count: int,
) -> torch.Tensor:
    indices = torch.argsort(scores, dim=1)[:, :elite_count]
    gather_indices = indices[:, :, None, None].expand(
        -1, -1, candidates.shape[2], candidates.shape[3]
    )
    return torch.gather(candidates, dim=1, index=gather_indices).contiguous()


def best_plan(
    candidates: torch.Tensor | None,
    scores: torch.Tensor | None,
    iterations: int,
) -> PlanResult:
    if candidates is None or scores is None:
        raise RuntimeError("planner produced no candidates")
    best_indices = torch.argsort(scores, dim=1)[:, 0]
    gather_indices = best_indices[:, None, None, None].expand(
        -1, 1, candidates.shape[2], candidates.shape[3]
    )
    sequence = torch.gather(candidates, dim=1, index=gather_indices).squeeze(1)
    return PlanResult(
        sequence=sequence.contiguous(),
        scores=scores.contiguous(),
        best_indices=[int(idx) for idx in best_indices.detach().cpu()],
        iterations_completed=iterations,
    )


def summarize_scores(scores: torch.Tensor) -> dict[str, float]:
    values = scores.detach().to(torch.float32).cpu().flatten().numpy()
    if values.size == 0:
        raise RuntimeError("empty score tensor")
    if not np.isfinite(values).all():
        raise RuntimeError("score tensor contains non-finite values")
    return {
        "best": float(values.min()),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def timed_cuda(
    device: torch.device,
    fn: Callable[[], object],
) -> tuple[object, float]:
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    value = fn()
    torch.cuda.synchronize(device)
    return value, (time.perf_counter() - started) * 1000.0


def bench_cuda(
    device: torch.device,
    args: argparse.Namespace,
    fn: Callable[[], object],
) -> tuple[object, dict[str, float]]:
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize(device)

    samples: list[float] = []
    value = None
    for _ in range(args.iters):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        value = fn()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - started) * 1000.0)
    if value is None:
        raise RuntimeError("benchmark produced no timed value")

    samples.sort()
    return value, {
        "mean_ms": sum(samples) / len(samples),
        "p50_ms": percentile_ms(samples, 0.50),
        "p95_ms": percentile_ms(samples, 0.95),
        "p99_ms": percentile_ms(samples, 0.99),
    }


def percentile_ms(samples: list[float], pct: float) -> float:
    idx = min(len(samples) - 1, int((len(samples) - 1) * pct + 0.999999))
    return samples[idx]


def git_commit() -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return "unknown"
    return output.strip() or "unknown"


if __name__ == "__main__":
    main()
