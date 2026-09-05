"""Summarize every preregistered result without pooling repeated baselines.

Usage: uv run --locked scripts/summarize-skyjepa-remediation.py ROOT OUTPUT.json
Raw reports stay in ROOT; the compact output includes their SHA-256 fingerprints.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def complete_mean(values):
    return statistics.mean(values) if values and all(value is not None for value in values) else None


def control_summary(path, report):
    cases = report["results"]
    config = report["configuration"]
    return {"file": str(path), "report_sha256": sha(path),
            "training_seed": int(path.name.split("_")[0].removeprefix("seed-")) if report["controller"] == "trained_mppi" else None,
            "controller": report["controller"], "warm_start": report["warm_start"],
            "domain_distribution": config["domain_distribution"], "trim_multiplier": round(config["session"]["trim_multiplier"], 2),
            "runs": report["runs"], "tracking_successes": report["tracking_successful_runs"],
            "timing_successes": report["timing_successful_runs"], "joint_successes": report["successful_runs"],
            "complete_runs": report["completed_runs"], "gate_passed": report["passed"],
            "mean_rmse_m": complete_mean([case["position_vector_rmse_m"] for case in cases]),
            "mean_randomized_rmse_m": complete_mean([case["position_vector_rmse_m"] for case in cases if case["randomized"]]),
            "worst_rmse_m": report["worst_position_rmse_m"], "worst_error_m": report["worst_position_error_m"],
            "aggregate_p95_plan_ms": report["aggregate_p95_plan_ms"],
            "aggregate_p99_plan_ms": report["aggregate_p99_plan_ms"], "maximum_plan_ms": report["maximum_plan_ms"],
            "control_deadline_misses": report["control_deadline_misses"],
            "planning_budget_exceedances": report["planning_budget_exceedances"],
            "maximum_planner_high_command_fraction": max(case["planner_high_command_fraction"] for case in cases),
            "maximum_plant_high_command_fraction": max(case["plant_high_command_fraction"] for case in cases),
            "ground_contact_runs": sum(case["ground_contact"] for case in cases),
            "nonfinite_runs": sum(not case["finite"] for case in cases),
            "errors": [case["failure"] for case in cases if case["failure"]],
            "checkpoint_sha256": report["checkpoint_sha256"], "executable_sha256": report["executable_sha256"]}


def case_key(case):
    return case["reference"], case["randomized"], case["domain_seed"]


def compare(trained, baseline):
    reference = {case_key(case): case for case in baseline["results"]}
    deltas, wins, tracking_wins, tracking_losses = [], 0, 0, 0
    for case in trained["results"]:
        other = reference[case_key(case)]
        assert case["domain"] == other["domain"], "comparison requires identical physical domains"
        lhs, rhs = case["position_vector_rmse_m"], other["position_vector_rmse_m"]
        deltas.append(lhs - rhs if lhs is not None and rhs is not None else None)
        wins += lhs is not None and rhs is not None and lhs < rhs
        tracking_wins += case["tracking_passed"] and not other["tracking_passed"]
        tracking_losses += other["tracking_passed"] and not case["tracking_passed"]
    assert len(deltas) == len(reference)
    return {"paired_runs": len(deltas), "mean_rmse_delta_m": complete_mean(deltas),
            "lower_rmse_runs": wins, "tracking_gains": tracking_wins, "tracking_losses": tracking_losses,
            "delta_direction": "trained minus baseline; negative is better"}


def training_summary(root, seed):
    result = {"seed": seed}
    for stage in ("latent", "prober"):
        directory = root / f"seed-{seed}" / stage
        pointer = read(directory / f"{stage}-current.json")
        snapshot = read(directory / "snapshots" / pointer["generation"] / "manifest.json")
        progress = snapshot["progress"]
        assert progress["completed_requested_steps"], directory
        elapsed = re.findall(r"requested stages complete .*?elapsed_sec=([0-9.]+)",
                             (directory.parent / f"{stage}-console.log").read_text())
        assert len(elapsed) == 1, "resumed elapsed accounting needs explicit handling, not silent undercounting"
        manifest = read(directory / f"{stage}-run-manifest.json")
        result[stage] = {"steps": progress["global_step"], "best_step": progress["best_step"],
                         "best_validation": progress["best_validation"], "elapsed_seconds": float(elapsed[0]),
                         "validation_batches_per_epoch": manifest["validation_batches"],
                         "training_domains": len(manifest["training_domains"]),
                         "training_episodes": len(manifest["training_episodes"]),
                         "checkpoint_file_sha256": sha(directory / "checkpoint.json")}
    return result


def summarize(root):
    selection = read(root / "warm-start-selection.json")
    reports = [(path, read(path)) for path in sorted((root / "test").glob("*.json"))
               if not path.name.endswith(".invocation.json")]
    assert len(reports) == 24, "expected six comparator/seed runs in each of four test conditions"
    assert all(report["runs"] == 63 for _, report in reports)
    controls = [control_summary(path, report) for path, report in reports]
    paired = []
    for path, trained in reports:
        if trained["controller"] != "trained_mppi":
            continue
        for _, baseline in reports:
            if baseline["controller"] == "trained_mppi":
                continue
            if (trained["configuration"]["domain_distribution"], trained["configuration"]["session"]["trim_multiplier"]) != (
                baseline["configuration"]["domain_distribution"], baseline["configuration"]["session"]["trim_multiplier"]):
                continue
            paired.append({"trained_report": path.name, "baseline": baseline["controller"], **compare(trained, baseline)})
    open_loop = []
    for path in sorted((root / "open-loop").glob("*.json")):
        if path.name.endswith(".invocation.json"):
            continue
        report = read(path)
        assert report["complete_population"] and report["training_domain_overlap"] == 0, path
        horizon = lambda key: [{"seconds": item["time_seconds"], "position_vector_rmse_m": item["position_vector_rmse_m"]}
                               for item in report[key] if item["step"] in (5, 15, 20, 60)]
        open_loop.append({"file": str(path), "report_sha256": sha(path), "windows": report["windows"],
                          "episodes": len(report["evaluated_episode_ids"]), "domains": len(report["evaluated_domain_ids"]),
                          "scope": report["scope"], "domain_distribution": report["domain_distribution"],
                          "dataset_artifact_sha256": report["dataset_artifact_sha256"],
                          "skyjepa": horizon("skyjepa"), "constant_velocity": horizon("constant_velocity_baseline"),
                          "kinematic": horizon("kinematic_baseline")})
    assert len(open_loop) == 9, "expected three populations for every training seed"
    return {"artifact_root": str(root), "training": [training_summary(root, seed) for seed in (7, 17, 29)],
            "warm_start_selection": selection, "control": controls, "paired_control_comparisons": paired,
            "open_loop": open_loop,
            "limitations": ["three training seeds are not a large-scale statistical study",
                            "prior/nominal/fixed-random baselines are measured once per condition, not three independent seeds",
                            "simulator supplies a hover-initialized state and observed trim; trim stress is +/-10%",
                            "timing is measured on a shared RTX 4090; other GPU workloads are retained",
                            "simulated time does not model real-time scheduling overruns",
                            "not upstream source parity, paper-scale reproduction, real flight or sim-to-real validation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = summarize(args.artifact_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"wrote {args.output}: {len(result['control'])} control reports, {len(result['open_loop'])} open-loop reports")


if __name__ == "__main__":
    main()
