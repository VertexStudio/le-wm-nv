#!/usr/bin/env python3
"""Compare two LeWM NPZ fixtures with backend-specific parity tolerances."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_ORDER = (
    "pixels",
    "actions",
    "action_candidates",
    "goal_emb",
    "emb",
    "act_emb",
    "pred",
    "rollout",
    "cost",
)

DEFAULT_TOLERANCES = {
    "pixels": 0.0,
    "actions": 0.0,
    "action_candidates": 0.0,
    "goal_emb": 1e-3,
    "emb": 1e-3,
    "act_emb": 1e-5,
    "pred": 1e-3,
    "rollout": 2e-3,
    "cost": 1e-2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument(
        "--key",
        action="append",
        dest="keys",
        help="array key to compare; may be repeated; defaults to common LeWM keys",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="override every per-key max-absolute tolerance",
    )
    parser.add_argument("--emb-tolerance", type=float, default=None)
    parser.add_argument("--act-emb-tolerance", type=float, default=None)
    parser.add_argument("--pred-tolerance", type=float, default=None)
    parser.add_argument("--rollout-tolerance", type=float, default=None)
    parser.add_argument("--cost-tolerance", type=float, default=None)
    return parser.parse_args()


def tolerance_for(args: argparse.Namespace, key: str) -> float:
    if args.tolerance is not None:
        return args.tolerance
    overrides = {
        "emb": args.emb_tolerance,
        "act_emb": args.act_emb_tolerance,
        "pred": args.pred_tolerance,
        "rollout": args.rollout_tolerance,
        "cost": args.cost_tolerance,
    }
    if overrides.get(key) is not None:
        return overrides[key]
    return DEFAULT_TOLERANCES.get(key, 1e-4)


def is_finite(name: str, array: np.ndarray) -> tuple[bool, str | None]:
    finite = np.isfinite(array)
    if bool(finite.all()):
        return True, None
    flat_idx = int(np.flatnonzero(~finite)[0])
    return False, f"{name} has non-finite value at flat index {flat_idx}"


def argmin_stable(left: np.ndarray, right: np.ndarray) -> bool:
    if left.ndim < 2 or right.ndim < 2:
        return True
    return bool(np.array_equal(np.argmin(left, axis=-1), np.argmin(right, axis=-1)))


def compare_key(
    key: str,
    left: np.ndarray,
    right: np.ndarray,
    tolerance: float,
    left_label: str,
    right_label: str,
) -> list[str]:
    failures = []
    if left.shape != right.shape:
        return [
            f"{key}: shape mismatch {left_label}={left.shape} "
            f"{right_label}={right.shape}"
        ]

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)

    for label, array in ((left_label, left), (right_label, right)):
        ok, message = is_finite(f"{label}.{key}", array)
        if not ok and message is not None:
            failures.append(message)

    diff = np.abs(left - right)
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0
    status = "ok" if max_abs <= tolerance and not failures else "FAIL"
    extra = ""

    if key == "cost":
        stable = argmin_stable(left, right)
        extra = f" argmin={'ok' if stable else 'FAIL'}"
        if not stable:
            failures.append(f"{key}: top candidate argmin changed")

    print(
        f"{status:4} {key:18} shape={left.shape} "
        f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} "
        f"tol={tolerance:.6e}{extra}"
    )

    if max_abs > tolerance:
        failures.append(
            f"{key}: max_abs {max_abs:.6e} exceeds tolerance {tolerance:.6e}"
        )
    return failures


def main() -> None:
    args = parse_args()
    failures = []
    with np.load(args.left) as left_npz, np.load(args.right) as right_npz:
        keys = args.keys or [key for key in DEFAULT_ORDER if key in left_npz]
        missing = [
            key
            for key in keys
            if key not in left_npz or key not in right_npz
        ]
        if missing:
            missing_keys = ", ".join(missing)
            raise SystemExit(f"missing key(s): {missing_keys}")

        print(f"left={args.left} ({args.left_label})")
        print(f"right={args.right} ({args.right_label})")
        for key in keys:
            failures.extend(
                compare_key(
                    key,
                    left_npz[key],
                    right_npz[key],
                    tolerance_for(args, key),
                    args.left_label,
                    args.right_label,
                )
            )

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
