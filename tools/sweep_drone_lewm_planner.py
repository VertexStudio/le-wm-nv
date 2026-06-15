#!/usr/bin/env python3
"""Run headless LeWM drone planner sweeps and write JSONL metrics."""

from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


HEADLESS_RE = re.compile(
    r"headless steps=(?P<steps>\d+) sim_time=(?P<sim_time>[-+0-9.eE]+)s "
    r"wall=(?P<wall>[-+0-9.eE]+)s start_dist=(?P<start_dist>[-+0-9.eE]+) "
    r"end_dist=(?P<end_dist>[-+0-9.eE]+) pos=\[(?P<pos>[^\]]+)\] "
    r"action=\[(?P<action>[^\]]+)\] plans=(?P<plans>\d+) "
    r"last_plan_ms=(?P<last_plan_ms>[-+0-9.eE]+) best=(?P<best>[-+0-9.eE]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark headless lewm-drone-sim planner settings."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--sim-bin",
        type=Path,
        default=Path("target/release/lewm-drone-sim"),
        help="Built lewm-drone-sim binary.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--horizons", default="12,25,40")
    parser.add_argument("--samples", default="32,64,128")
    parser.add_argument("--lookaheads", default="0.5,1.0,2.0")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--planner-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-pos", nargs=3, type=float, default=[4.0, 0.0, 1.6])
    parser.add_argument("--target-yaw", type=float, default=0.0)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    if args.planner_every <= 0:
        raise SystemExit("--planner-every must be positive")
    sim_bin = args.sim_bin.expanduser()
    if not sim_bin.exists():
        raise SystemExit(f"sim binary not found: {sim_bin}; build lewm-drone-sim first")
    model_dir = args.model_dir.expanduser()
    if not model_dir.exists():
        raise SystemExit(f"model dir not found: {model_dir}")

    horizons = parse_int_list(args.horizons, "--horizons")
    samples = parse_int_list(args.samples, "--samples")
    lookaheads = parse_float_list(args.lookaheads, "--lookaheads")
    output = args.output or default_output_path()
    output.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    with output.open("w", encoding="utf-8") as f:
        for horizon, sample_count, lookahead in itertools.product(
            horizons, samples, lookaheads
        ):
            elites = max(2, sample_count // 4)
            result = run_case(
                sim_bin=sim_bin,
                model_dir=model_dir,
                steps=args.steps,
                horizon=horizon,
                samples=sample_count,
                elites=elites,
                iterations=args.iterations,
                planner_every=args.planner_every,
                lookahead=lookahead,
                seed=args.seed,
                target_pos=args.target_pos,
                target_yaw=args.target_yaw,
                timeout_sec=args.timeout_sec,
            )
            f.write(json.dumps(result, sort_keys=True) + "\n")
            f.flush()
            results.append(result)
            print(format_case(result))
            if args.fail_fast and result["returncode"] != 0:
                break

    print_summary(results, output, time.perf_counter() - started)
    if any(result["returncode"] != 0 for result in results):
        raise SystemExit(1)


def parse_int_list(value: str, flag: str) -> list[int]:
    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not out or any(item <= 0 for item in out):
        raise SystemExit(f"{flag} must contain positive integers")
    return out


def parse_float_list(value: str, flag: str) -> list[float]:
    out = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not out or any(item <= 0.0 for item in out):
        raise SystemExit(f"{flag} must contain positive floats")
    return out


def default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("target") / "drone-sim-sweeps" / f"sweep-{stamp}.jsonl"


def run_case(
    *,
    sim_bin: Path,
    model_dir: Path,
    steps: int,
    horizon: int,
    samples: int,
    elites: int,
    iterations: int,
    planner_every: int,
    lookahead: float,
    seed: int,
    target_pos: list[float],
    target_yaw: float,
    timeout_sec: float,
) -> dict[str, Any]:
    cmd = [
        str(sim_bin),
        "--headless-steps",
        str(steps),
        "--model-dir",
        str(model_dir),
        "--planner-horizon",
        str(horizon),
        "--planner-samples",
        str(samples),
        "--planner-elites",
        str(elites),
        "--planner-iterations",
        str(iterations),
        "--planner-every",
        str(planner_every),
        "--target-lookahead",
        f"{lookahead}",
        "--target-pos",
        ",".join(f"{value}" for value in target_pos),
        "--target-yaw",
        f"{target_yaw}",
        "--seed",
        str(seed),
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
        )
        elapsed = time.perf_counter() - started
        parsed = parse_headless_output(completed.stdout)
        result = {
            "returncode": completed.returncode,
            "elapsed_sec": elapsed,
            "horizon": horizon,
            "samples": samples,
            "elites": elites,
            "iterations": iterations,
            "planner_every": planner_every,
            "lookahead": lookahead,
            "seed": seed,
            "target_pos": target_pos,
            "target_yaw": target_yaw,
            "command": cmd,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "metrics": parsed,
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        result = {
            "returncode": 124,
            "elapsed_sec": elapsed,
            "horizon": horizon,
            "samples": samples,
            "elites": elites,
            "iterations": iterations,
            "planner_every": planner_every,
            "lookahead": lookahead,
            "seed": seed,
            "target_pos": target_pos,
            "target_yaw": target_yaw,
            "command": cmd,
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "").strip() if isinstance(exc.stderr, str) else "",
            "metrics": {},
            "error": "timeout",
        }
    metrics = result.get("metrics") or {}
    if metrics:
        metrics["distance_delta"] = metrics["end_dist"] - metrics["start_dist"]
        metrics["final_action_abs_max"] = max(abs(value) for value in metrics["action"])
        metrics["final_action_saturated"] = any(abs(value) >= 0.98 for value in metrics["action"])
    return result


def parse_headless_output(stdout: str) -> dict[str, Any]:
    match = HEADLESS_RE.search(stdout)
    if match is None:
        return {}
    groups = match.groupdict()
    return {
        "steps": int(groups["steps"]),
        "sim_time": float(groups["sim_time"]),
        "wall": float(groups["wall"]),
        "start_dist": float(groups["start_dist"]),
        "end_dist": float(groups["end_dist"]),
        "pos": parse_float_vector(groups["pos"]),
        "action": parse_float_vector(groups["action"]),
        "plans": int(groups["plans"]),
        "last_plan_ms": float(groups["last_plan_ms"]),
        "best": float(groups["best"]),
    }


def parse_float_vector(value: str) -> list[float]:
    return [float(part) for part in value.replace(",", " ").split()]


def format_case(result: dict[str, Any]) -> str:
    metrics = result.get("metrics") or {}
    if not metrics:
        return (
            "h={horizon:>2} samples={samples:>3} lookahead={lookahead:.2f} "
            "rc={returncode} no metrics"
        ).format(**result)
    return (
        "h={horizon:>2} samples={samples:>3} lookahead={lookahead:.2f} "
        "end={end:.3f} delta={delta:+.3f} wall={wall:.3f}s "
        "plan_ms={plan:.2f} saturated={sat}"
    ).format(
        horizon=result["horizon"],
        samples=result["samples"],
        lookahead=result["lookahead"],
        end=metrics["end_dist"],
        delta=metrics["distance_delta"],
        wall=metrics["wall"],
        plan=metrics["last_plan_ms"],
        sat=metrics["final_action_saturated"],
    )


def print_summary(results: list[dict[str, Any]], output: Path, elapsed_sec: float) -> None:
    valid = [result for result in results if result.get("metrics")]
    print(f"jsonl={output}")
    print(f"cases={len(results)} valid={len(valid)} elapsed_sec={elapsed_sec:.3f}")
    if not valid:
        return
    ranked = sorted(valid, key=lambda result: result["metrics"]["end_dist"])
    print("best_cases:")
    for result in ranked[: min(5, len(ranked))]:
        metrics = result["metrics"]
        print(
            "  h={} samples={} lookahead={:.2f} end={:.3f} delta={:+.3f} plan_ms={:.2f}".format(
                result["horizon"],
                result["samples"],
                result["lookahead"],
                metrics["end_dist"],
                metrics["distance_delta"],
                metrics["last_plan_ms"],
            )
        )


if __name__ == "__main__":
    main()
