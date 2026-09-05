"""Run the frozen pilot protocol serially: uv run --locked scripts/...py ROOT PHASE.

No training or build may overlap measured control runs. Other GPU applications
are not stopped; the Rust reports record them. Existing reports are immutable
and reused only when the invocation, executable and checkpoint hashes match.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess


SEEDS = (7, 17, 29)
MODES = ("prior", "nominal-physics-mppi", "untrained-mppi", "trained-mppi")
WARM_STARTS = ("fresh-prior", "shifted-residual")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def require_no_training_or_build():
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            name = (process / "exe").resolve(strict=True).name
        except (OSError, RuntimeError):
            continue
        if name in ("lewm-train-skyjepa", "cargo", "rustc", "nvcc"):
            raise RuntimeError(f"timing would overlap {name} (PID {process.name})")


def execute(root, binary, arguments, output):
    executable = root / "bin" / binary
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(executable), *map(str, arguments), "--output", str(output)]
    if output.exists():
        report = read(output)
        assert report["configuration"]["argv"] == command, output
        assert report["executable_sha256"] == sha(executable), output
        # Package fingerprint is a SHA of the serialized manifest, not file bytes;
        # the sidecar records the byte hash independently for safe report reuse.
        sidecar = read(output.with_suffix(".invocation.json"))
        checkpoint = Path(command[command.index("--checkpoint-dir") + 1]) / "checkpoint.json"
        assert sidecar["checkpoint_file_sha256"] == sha(checkpoint), output
        print(f"reuse {output.name}", flush=True)
        return report
    checkpoint = Path(command[command.index("--checkpoint-dir") + 1]) / "checkpoint.json"
    sidecar_path = output.with_suffix(".invocation.json")
    invocation = {"command": command, "executable_sha256": sha(executable),
                  "checkpoint_file_sha256": sha(checkpoint)}
    if sidecar_path.exists():
        assert read(sidecar_path) == invocation, sidecar_path
    else:
        save_new(sidecar_path, invocation)
    print(f"run {output.name}", flush=True)
    with output.with_suffix(".console.log").open("a") as console:
        subprocess.run(command, check=True, stdout=console, stderr=subprocess.STDOUT)
    report = read(output)
    print(f"done {output.name}: tracking={report.get('tracking_successful_runs')} "
          f"p95={report.get('aggregate_p95_plan_ms')} windows={report.get('windows')}", flush=True)
    return report


def control(root, seed, mode, warm, trim, population, phase, domain_seed, domains):
    require_no_training_or_build()
    return execute(root, "lewm-bench-skyjepa", [
        "--checkpoint-dir", root / f"seed-{seed}" / "prober",
        "--controller", mode, "--warm-start", warm, "--trim-multiplier", trim,
        "--domain-distribution", population, "--domain-seed", domain_seed,
        "--random-domains", domains, "--samples", 512, "--horizon", 15,
        "--planner-seed", 7, "--ablation-seed", 7, "--duration-seconds", 8,
        "--radius-m", 2, "--period-seconds", 8, "--allow-fail",
    ], root / phase / f"seed-{seed}_{mode}_{warm}_trim-{trim}_{population}.json")


def aggregate(reports):
    cases = [case for report in reports for case in report["results"]]
    errors = [case["position_vector_rmse_m"] for case in cases]
    complete = all(error is not None and math.isfinite(error) for error in errors)
    return {"runs": len(cases), "tracking": sum(case["tracking_passed"] for case in cases),
            "timing": sum(case["timing_passed"] for case in cases), "complete": complete,
            "mean_rmse_m": sum(errors) / len(errors) if complete else None,
            "worst_rmse_m": max(errors) if complete else None,
            "maximum_aggregate_p95_ms": max(report["aggregate_p95_plan_ms"] for report in reports)}


def choose_warm_start(fresh, shifted):
    """Conservative rule frozen before validation; no final-test tuning."""
    improved = (
        fresh["complete"] and shifted["complete"]
        and shifted["tracking"] >= fresh["tracking"]
        and shifted["timing"] >= fresh["timing"]
        and shifted["maximum_aggregate_p95_ms"] <= 10.0
        and shifted["mean_rmse_m"] <= fresh["mean_rmse_m"] * 0.99
        and shifted["worst_rmse_m"] <= fresh["worst_rmse_m"] * 1.05
    )
    return "shifted-residual" if improved else "fresh-prior"


def verify_shared_baselines(root):
    contracts = [read(root / f"seed-{seed}" / "prober" / "checkpoint.json")["contract"] for seed in SEEDS]
    assert contracts[0] == contracts[1] == contracts[2], "baseline reuse requires identical model/preprocessing/physics contracts"
    return {"reused_for_training_seeds": list(SEEDS), "executed_with_package_seed": 7,
            "reason": "identical architecture, normalization and physics; untrained initialization seed fixed at 7; trained weights not used"}


def validate(root):
    reuse = verify_shared_baselines(root)
    metrics, selection = {}, {"prior": "fresh-prior"}
    for mode in MODES[1:]:
        metrics[mode] = {}
        for warm in WARM_STARTS:
            reports = [control(root, seed, mode, warm, 1.0, "training-ranges", "validation", 31415, 8)
                       for seed in (SEEDS if mode == "trained-mppi" else (7,))]
            metrics[mode][warm] = aggregate(reports)
        selection[mode] = choose_warm_start(*[metrics[mode][warm] for warm in WARM_STARTS])
    result = {"selection": selection, "validation": metrics, "baseline_reuse": reuse,
              "rule": "shift only if complete, tracking/timing counts do not regress, p95<=10ms, mean RMSE improves >=1%, worst RMSE regresses <=5%; otherwise fresh",
              "validation_domain_seed": 31415, "final_test_domain_seed": 271828}
    path = root / "warm-start-selection.json"
    if path.exists():
        assert read(path) == result, "selection is frozen; do not overwrite"
    else:
        save_new(path, result)
    print(json.dumps(result, indent=2), flush=True)


def test(root):
    verify_shared_baselines(root)
    selected = read(root / "warm-start-selection.json")["selection"]
    for population, domain_seed, trims in (
        ("training-ranges", 271828, (1.0, 0.9, 1.1)),
        ("extended-mass-and-motor-lag", 90002, (1.0,)),
    ):
        for trim in trims:
            for mode in MODES:
                for seed in (SEEDS if mode == "trained-mppi" else (7,)):
                    control(root, seed, mode, selected[mode], trim, population, "test", domain_seed, 20)


def open_loop(root):
    for seed in SEEDS:
        for data, split in (("data-pilot", "test"), ("data-external", "all"), ("data-shift", "all")):
            execute(root, "lewm-eval-skyjepa", [
                "--checkpoint-dir", root / f"seed-{seed}" / "prober", "--dataset-dir", root / data,
                "--split", split, "--rollout-steps", 60, "--batch-size", 512,
            ], root / "open-loop" / f"seed-{seed}_{data}.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("phase", choices=("validation", "test", "open-loop"))
    args = parser.parse_args()
    {"validation": validate, "test": test, "open-loop": open_loop}[args.phase](args.artifact_root.resolve())


if __name__ == "__main__":
    main()
