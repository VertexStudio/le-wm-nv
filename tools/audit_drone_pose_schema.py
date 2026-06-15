#!/usr/bin/env python3
"""Audit the drone pose12 observation contract used by the LeWM simulator."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np


EXPECTED_ACTION_DIM = 4
EXPECTED_OBS_DIM = 12
EXPECTED_COLUMNS = [
    "pos_world[0..3]",
    "rotmat_world_from_body[0..9]",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate drone pose12 dataset/model schema and CSV rotation conventions."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path.home()
        / ".stable_worldmodel"
        / "le-wm-nv-data"
        / "drone-racing-autonomous-100hz-pose12",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Optional trained model directory containing model-config.json and normalization.json.",
    )
    parser.add_argument(
        "--csv",
        action="append",
        type=Path,
        default=None,
        help="Optional source CSV path. Defaults to metadata source_files.",
    )
    parser.add_argument("--sample-stride", type=int, default=100)
    parser.add_argument("--max-csv-files", type=int, default=18)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_stride <= 0:
        raise SystemExit("--sample-stride must be positive")
    if args.max_csv_files <= 0:
        raise SystemExit("--max-csv-files must be positive")

    dataset_dir = args.dataset_dir.expanduser()
    metadata_path = dataset_dir / "metadata.json"
    metadata = read_json(metadata_path)
    report: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "metadata": audit_metadata(metadata),
        "h5": audit_h5(dataset_dir, metadata),
    }

    csv_paths = args.csv or [Path(path) for path in metadata.get("source_files", [])]
    csv_paths = [path.expanduser() for path in csv_paths[: args.max_csv_files]]
    report["csv_rotation"] = audit_csv_rotation(csv_paths, args.sample_stride)

    if args.model_dir is not None:
        model_dir = args.model_dir.expanduser()
        report["model_dir"] = str(model_dir)
        report["model"] = audit_model_dir(model_dir)

    failures = collect_failures(report)
    report["ok"] = not failures
    report["failures"] = failures
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_report(report)
    if failures:
        raise SystemExit(1)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def audit_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    columns = metadata.get("columns", {}).get("observation", [])
    return {
        "rows": metadata.get("rows"),
        "episodes": metadata.get("episodes"),
        "sample_rate_hz": metadata.get("sample_rate_hz"),
        "action_dim": metadata.get("action_dim"),
        "observation_dim": metadata.get("observation_dim"),
        "columns": columns,
        "action_dim_ok": metadata.get("action_dim") == EXPECTED_ACTION_DIM,
        "observation_dim_ok": metadata.get("observation_dim") == EXPECTED_OBS_DIM,
        "columns_ok": columns == EXPECTED_COLUMNS,
        "normalization": stats_dims(metadata.get("normalization", {})),
    }


def audit_h5(dataset_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    data_h5 = Path(metadata["data_h5"])
    if not data_h5.is_absolute():
        data_h5 = dataset_dir / data_h5
    rows = int(metadata["rows"])
    required_shapes = {
        "pos_world": [rows, 3],
        "rotmat_world_from_body": [rows, 9],
        "channels_norm": [rows, 4],
        "episode_idx": [rows],
        "step_idx": [rows],
    }
    report: dict[str, Any] = {"path": str(data_h5), "datasets": {}}
    with h5py.File(data_h5, "r") as h5:
        for name, expected_shape in required_shapes.items():
            if name not in h5:
                report["datasets"][name] = {"exists": False, "shape_ok": False}
                continue
            shape = list(h5[name].shape)
            report["datasets"][name] = {
                "exists": True,
                "shape": shape,
                "expected_shape": expected_shape,
                "shape_ok": shape == expected_shape,
            }
        if all(name in h5 for name in ("pos_world", "rotmat_world_from_body", "channels_norm")):
            pos = h5["pos_world"][: min(rows, 2048)]
            rot = h5["rotmat_world_from_body"][: min(rows, 2048)].reshape(-1, 3, 3)
            action = h5["channels_norm"][: min(rows, 2048)]
            report["pos_min"] = np.min(pos, axis=0).astype(float).tolist()
            report["pos_max"] = np.max(pos, axis=0).astype(float).tolist()
            report["action_min"] = np.min(action, axis=0).astype(float).tolist()
            report["action_max"] = np.max(action, axis=0).astype(float).tolist()
            report["rot_orth_max"] = float(
                np.max(np.abs(rot @ np.swapaxes(rot, 1, 2) - np.eye(3)))
            )
            report["rot_det_min"] = float(np.min(np.linalg.det(rot)))
            report["rot_det_max"] = float(np.max(np.linalg.det(rot)))
    report["shape_ok"] = all(item.get("shape_ok") for item in report["datasets"].values())
    report["rotation_ok"] = (
        report.get("rot_orth_max", 1.0) < 1e-4
        and abs(report.get("rot_det_min", 0.0) - 1.0) < 1e-4
        and abs(report.get("rot_det_max", 0.0) - 1.0) < 1e-4
    )
    return report


def audit_model_dir(model_dir: Path) -> dict[str, Any]:
    model_cfg = read_json(model_dir / "model-config.json")
    normalization = read_json(model_dir / "normalization.json")
    obs = model_cfg.get("observation_encoder", {})
    action = model_cfg.get("action_encoder", {})
    return {
        "observation_encoder_kind": obs.get("kind"),
        "observation_input_dim": obs.get("input_dim"),
        "action_input_dim": action.get("input_dim"),
        "history_size": model_cfg.get("history_size"),
        "predictor_num_frames": model_cfg.get("predictor", {}).get("num_frames"),
        "vector_pose12_ok": obs.get("kind") == "vector_mlp"
        and obs.get("input_dim") == EXPECTED_OBS_DIM,
        "action_ok": action.get("input_dim") == EXPECTED_ACTION_DIM,
        "normalization": stats_dims(normalization),
    }


def stats_dims(normalization: dict[str, Any]) -> dict[str, Any]:
    def dims(name: str) -> dict[str, Any]:
        stats = normalization.get(name, {})
        mean = stats.get("mean", [])
        std = stats.get("std", [])
        return {"mean": len(mean), "std": len(std)}

    return {
        "observation": dims("observation"),
        "action": dims("action"),
        "target_delta": dims("target_delta"),
    }


def audit_csv_rotation(csv_paths: list[Path], stride: int) -> dict[str, Any]:
    if not csv_paths:
        return {"sampled_rows": 0, "ok": False, "error": "no CSV files provided"}
    orders = {
        "RzRyRx": lambda r, p, y: rot_z(y) @ rot_y(p) @ rot_x(r),
        "RxRyRz": lambda r, p, y: rot_x(r) @ rot_y(p) @ rot_z(y),
        "RzRxRy": lambda r, p, y: rot_z(y) @ rot_x(r) @ rot_y(p),
        "RyRxRz": lambda r, p, y: rot_y(p) @ rot_x(r) @ rot_z(y),
    }
    errors: dict[str, dict[str, list[float]]] = {
        name: {"csv": [], "csv_transposed": []} for name in orders
    }
    orth_err: list[float] = []
    dets: list[float] = []
    pos: list[list[float]] = []
    rpy: list[list[float]] = []
    ang_vel: list[list[float]] = []
    sampled = 0
    used_files: list[str] = []
    for path in csv_paths:
        if not path.exists():
            continue
        used_files.append(str(path))
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                if row_idx % stride != 0:
                    continue
                sample = parse_csv_sample(row)
                if sample is None:
                    continue
                roll, pitch, yaw, matrix, xyz, omega = sample
                sampled += 1
                orth_err.append(float(np.max(np.abs(matrix @ matrix.T - np.eye(3)))))
                dets.append(float(np.linalg.det(matrix)))
                pos.append(xyz)
                rpy.append([roll, pitch, yaw])
                ang_vel.append(omega)
                for name, fn in orders.items():
                    expected = fn(roll, pitch, yaw)
                    errors[name]["csv"].append(float(np.max(np.abs(matrix - expected))))
                    errors[name]["csv_transposed"].append(
                        float(np.max(np.abs(matrix.T - expected)))
                    )

    if sampled == 0:
        return {"sampled_rows": 0, "ok": False, "error": "no finite CSV samples"}

    candidates: dict[str, dict[str, float]] = {}
    for name, variants in errors.items():
        for variant, values in variants.items():
            arr = np.asarray(values, dtype=np.float64)
            candidates[f"{name}.{variant}"] = {
                "mean": float(np.mean(arr)),
                "p95": float(np.percentile(arr, 95)),
                "max": float(np.max(arr)),
            }
    best = min(candidates, key=lambda key: candidates[key]["mean"])
    pos_arr = np.asarray(pos, dtype=np.float64)
    rpy_arr = np.asarray(rpy, dtype=np.float64)
    ang_arr = np.asarray(ang_vel, dtype=np.float64)
    expected_best = "RxRyRz.csv_transposed"
    return {
        "files": used_files,
        "sample_stride": stride,
        "sampled_rows": sampled,
        "rotation_orth_max": float(np.max(orth_err)),
        "rotation_det_min": float(np.min(dets)),
        "rotation_det_max": float(np.max(dets)),
        "candidate_errors": candidates,
        "best_convention": best,
        "expected_convention": expected_best,
        "expected_convention_ok": best == expected_best
        and candidates[expected_best]["p95"] < 1e-2,
        "position_min": pos_arr.min(axis=0).astype(float).tolist(),
        "position_max": pos_arr.max(axis=0).astype(float).tolist(),
        "rpy_min": rpy_arr.min(axis=0).astype(float).tolist(),
        "rpy_max": rpy_arr.max(axis=0).astype(float).tolist(),
        "rpy_units_look_like_radians": bool(np.max(np.abs(rpy_arr)) <= math.pi + 0.05),
        "angular_velocity_min": ang_arr.min(axis=0).astype(float).tolist(),
        "angular_velocity_max": ang_arr.max(axis=0).astype(float).tolist(),
    }


def parse_csv_sample(
    row: dict[str, str],
) -> tuple[float, float, float, np.ndarray, list[float], list[float]] | None:
    try:
        roll = float(row["drone_roll"])
        pitch = float(row["drone_pitch"])
        yaw = float(row["drone_yaw"])
        matrix = np.asarray(
            [float(row[f"drone_rot[{idx}]"]) for idx in range(9)], dtype=np.float64
        ).reshape(3, 3)
        pos = [
            float(row["drone_x"]),
            float(row["drone_y"]),
            float(row["drone_z"]),
        ]
        ang_vel = [
            float(row["drone_velocity_angular_x"]),
            float(row["drone_velocity_angular_y"]),
            float(row["drone_velocity_angular_z"]),
        ]
    except (KeyError, ValueError):
        return None
    values = [roll, pitch, yaw, *matrix.reshape(-1).tolist(), *pos, *ang_vel]
    if not np.all(np.isfinite(values)):
        return None
    return roll, pitch, yaw, matrix, pos, ang_vel


def rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def collect_failures(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    metadata = report["metadata"]
    if not metadata["action_dim_ok"]:
        failures.append("metadata action_dim is not 4")
    if not metadata["observation_dim_ok"]:
        failures.append("metadata observation_dim is not 16")
    if not metadata["columns_ok"]:
        failures.append("metadata observation columns are not pose12")
    obs_norm = metadata["normalization"]["observation"]
    action_norm = metadata["normalization"]["action"]
    if obs_norm != {"mean": EXPECTED_OBS_DIM, "std": EXPECTED_OBS_DIM}:
        failures.append("metadata observation normalization is not 16D")
    if action_norm != {"mean": EXPECTED_ACTION_DIM, "std": EXPECTED_ACTION_DIM}:
        failures.append("metadata action normalization is not 4D")
    if not report["h5"]["shape_ok"]:
        failures.append("HDF5 dataset shapes do not match metadata")
    if not report["h5"]["rotation_ok"]:
        failures.append("HDF5 rotation matrices are not orthonormal det=1")
    csv_report = report["csv_rotation"]
    if not csv_report.get("expected_convention_ok", False):
        failures.append("CSV rotation convention does not match expected transposed RxRyRz")
    if not csv_report.get("rpy_units_look_like_radians", False):
        failures.append("CSV roll/pitch/yaw do not look like radians")
    model = report.get("model")
    if model is not None:
        if not model["vector_pose12_ok"]:
            failures.append("model is not vector pose12")
        if not model["action_ok"]:
            failures.append("model action dimension is not 4")
        model_obs_norm = model["normalization"]["observation"]
        model_action_norm = model["normalization"]["action"]
        if model_obs_norm != {"mean": EXPECTED_OBS_DIM, "std": EXPECTED_OBS_DIM}:
            failures.append("model observation normalization is not 16D")
        if model_action_norm != {"mean": EXPECTED_ACTION_DIM, "std": EXPECTED_ACTION_DIM}:
            failures.append("model action normalization is not 4D")
    return failures


def print_report(report: dict[str, Any]) -> None:
    metadata = report["metadata"]
    h5 = report["h5"]
    csv_report = report["csv_rotation"]
    print(
        "dataset rows={} episodes={} sample_rate={}Hz obs_dim={} action_dim={}".format(
            metadata["rows"],
            metadata["episodes"],
            metadata["sample_rate_hz"],
            metadata["observation_dim"],
            metadata["action_dim"],
        )
    )
    print("observation columns={}".format(metadata["columns"]))
    print(
        "h5 shape_ok={} rotation_ok={} rot_orth_max={:.3e} det=[{:.6f},{:.6f}]".format(
            h5["shape_ok"],
            h5["rotation_ok"],
            h5.get("rot_orth_max", float("nan")),
            h5.get("rot_det_min", float("nan")),
            h5.get("rot_det_max", float("nan")),
        )
    )
    print(
        "csv sampled_rows={} best_convention={} expected_ok={} rpy_radians={}".format(
            csv_report.get("sampled_rows"),
            csv_report.get("best_convention"),
            csv_report.get("expected_convention_ok"),
            csv_report.get("rpy_units_look_like_radians"),
        )
    )
    if "model" in report:
        model = report["model"]
        print(
            "model kind={} obs_input_dim={} action_input_dim={} history={} predictor_frames={}".format(
                model["observation_encoder_kind"],
                model["observation_input_dim"],
                model["action_input_dim"],
                model["history_size"],
                model["predictor_num_frames"],
            )
        )
    if report["ok"]:
        print("audit=ok")
    else:
        for failure in report["failures"]:
            print(f"failure={failure}")


if __name__ == "__main__":
    main()
