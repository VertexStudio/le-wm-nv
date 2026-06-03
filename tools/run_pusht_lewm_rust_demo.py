#!/usr/bin/env python3
"""Run a PushT scene through the Rust LeWM planner and execute it."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import gymnasium as gym
import h5py
import hdf5plugin  # noqa: F401 - registers Blosc and other HDF5 dataset filters
import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default_pusht_h5 = Path.home() / ".stable_worldmodel" / "pusht_expert_train.h5"
    parser.add_argument("--output-dir", type=Path, default=Path("target/reports/pusht-demo"))
    parser.add_argument("--hf-repo", default="quentinll/lewm-pusht")
    parser.add_argument("--planner", choices=("cem", "mppi", "icem"), default="icem")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--history-size", type=int, default=1)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--replans", type=int, default=2)
    parser.add_argument("--execute-blocks", type=int, default=None)
    parser.add_argument("--dataset-h5", type=Path, default=default_pusht_h5)
    parser.add_argument("--dataset-row", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--eval-index", type=int, default=0)
    parser.add_argument("--goal-offset-steps", type=int, default=25)
    parser.add_argument(
        "--action-stats-h5",
        type=Path,
        default=default_pusht_h5,
    )
    parser.add_argument("--no-action-inverse-transform", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--open", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)

    import stable_worldmodel.envs  # noqa: F401 - registers swm/PushT-v1

    input_dir = args.output_dir / "input"
    rollout_dir = args.output_dir / "rollout"
    input_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224)
    eval_case = load_eval_case(
        args.dataset_h5,
        args.goal_offset_steps,
        args.eval_seed,
        args.eval_index,
        args.dataset_row,
    )
    start_state_np = np.asarray(eval_case["start_state"], dtype=np.float64)
    goal_state_np = np.asarray(eval_case["goal_state"], dtype=np.float64)
    obs, info = env.reset(
        seed=args.eval_seed,
        options={"state": start_state_np, "goal_state": goal_state_np},
    )
    reset_state = obs["state"].tolist()
    goal_state = goal_state_np.astype(np.float64).tolist()
    dataset_current_path = input_dir / "dataset-current.jpg"
    dataset_goal_path = input_dir / "dataset-goal.jpg"
    rendered_current_path = input_dir / "rendered-current.jpg"
    rendered_goal_path = input_dir / "rendered-goal.jpg"
    Image.fromarray(eval_case["start_pixels"]).save(dataset_current_path)
    Image.fromarray(eval_case["goal_pixels"]).save(dataset_goal_path)
    Image.fromarray(np.asarray(env.render(), dtype=np.uint8)).save(rendered_current_path)
    Image.fromarray(np.asarray(info["goal"], dtype=np.uint8)).save(rendered_goal_path)

    action_stats = None
    if not args.no_action_inverse_transform:
        action_stats = load_action_stats(args.action_stats_h5)

    execute_blocks = args.execute_blocks or args.horizon
    history_frames = [np.asarray(env.render(), dtype=np.uint8)]
    initial_history_paths = [dataset_current_path]
    start_state = obs["state"].tolist()
    rollout_frames = [history_frames[-1]]
    planner_runs = []
    executed_actions: list[list[float]] = []
    rewards: list[float] = []
    final_obs = obs
    final_info = info
    stopped = False

    for replan_idx in range(args.replans):
        if replan_idx == 0:
            history_paths = [dataset_current_path]
        else:
            history_paths = save_history(input_dir, history_frames, replan_idx)
        report_html = args.output_dir / f"lewm-pusht-rust-plan-r{replan_idx:02}.html"
        run_rust_planner(args, history_paths, dataset_goal_path, report_html)
        planner_json = report_html.with_suffix(".json")
        plan_payload = json.loads(planner_json.read_text())
        sequence = np.asarray(plan_payload["sequence"], dtype=np.float32)
        planner_runs.append(
            {
                "report": str(report_html),
                "json": str(planner_json),
                "selected_cost": plan_payload["score"]["selected_cost"],
                "planning_ms": plan_payload["timing_ms"]["planning"],
                "best_cost": plan_payload["score"]["final_best"],
            }
        )
        step_frames, step_actions, step_rewards, final_obs, final_info, stopped = execute_plan(
            env,
            sequence[:execute_blocks],
            args.action_block,
            action_stats,
        )
        rollout_frames.extend(step_frames[1:])
        executed_actions.extend(step_actions)
        rewards.extend(step_rewards)
        history_frames = (history_frames + step_frames[1:])[-args.history_size :]
        if stopped:
            break
    success, final_distance = env.unwrapped.eval_state(goal_state_np, final_obs["state"])
    env.close()

    gif_path = rollout_dir / "rollout.gif"
    montage_path = rollout_dir / "rollout-montage.jpg"
    final_path = rollout_dir / "final.jpg"
    save_gif(rollout_frames, gif_path)
    save_montage(rollout_frames, montage_path)
    Image.fromarray(rollout_frames[-1]).save(final_path)

    demo_payload = {
        "seed": args.seed,
        "env": "swm/PushT-v1",
        "hf_repo": args.hf_repo,
        "dataset": {
            "path": str(args.dataset_h5),
            "row": eval_case["row"],
            "episode": eval_case["episode"],
            "step": eval_case["step"],
            "goal_row": eval_case["goal_row"],
            "goal_offset_steps": args.goal_offset_steps,
            "eval_seed": args.eval_seed,
            "eval_index": args.eval_index,
        },
        "planner_report": str(report_html),
        "planner_json": str(planner_json),
        "goal_image": str(dataset_goal_path),
        "dataset_current_image": str(dataset_current_path),
        "rendered_current_image": str(rendered_current_path),
        "rendered_goal_image": str(rendered_goal_path),
        "current_history": [str(path) for path in initial_history_paths],
        "final_history": [str(path) for path in save_history(input_dir, history_frames, "final")],
        "rollout_gif": str(gif_path),
        "rollout_montage": str(montage_path),
        "final_image": str(final_path),
        "action_inverse_transform": action_stats,
        "replans_requested": args.replans,
        "replans_completed": len(planner_runs),
        "history_size": args.history_size,
        "execute_blocks": execute_blocks,
        "planner_runs": planner_runs,
        "executed_actions": executed_actions,
        "rewards": rewards,
        "reset_state": reset_state,
        "start_state": start_state,
        "final_state": final_obs["state"].tolist(),
        "goal_state": goal_state,
        "success": bool(success),
        "final_distance": float(final_distance),
        "selected_cost": planner_runs[-1]["selected_cost"],
        "planning_ms": sum(run["planning_ms"] for run in planner_runs),
        "stopped": stopped,
    }
    demo_json = args.output_dir / "pusht-demo.json"
    demo_json.write_text(json.dumps(demo_payload, indent=2) + "\n")
    demo_html = args.output_dir / "pusht-demo.html"
    demo_html.write_text(render_demo_html(demo_payload), encoding="utf-8")

    print(f"demo={demo_html}")
    print(f"json={demo_json}")
    print(f"rollout_gif={gif_path}")
    print(f"planner_report={report_html}")
    print(
        "selected_cost={:.6f} planning_ms={:.3f} executed_actions={}".format(
            float(planner_runs[-1]["selected_cost"]),
            float(sum(run["planning_ms"] for run in planner_runs)),
            len(executed_actions),
        )
    )
    if args.open:
        subprocess.run(["/usr/bin/open", str(demo_html.resolve())], check=False)


def validate_args(args: argparse.Namespace) -> None:
    if args.history_size <= 0:
        raise ValueError("--history-size must be greater than zero")
    if args.action_block <= 0:
        raise ValueError("--action-block must be greater than zero")
    if args.horizon <= 0:
        raise ValueError("--horizon must be greater than zero")
    if args.samples < 2:
        raise ValueError("--samples must be at least 2")
    if args.iterations <= 0:
        raise ValueError("--iterations must be greater than zero")
    if args.replans <= 0:
        raise ValueError("--replans must be greater than zero")
    if args.execute_blocks is not None and args.execute_blocks <= 0:
        raise ValueError("--execute-blocks must be greater than zero")
    if args.goal_offset_steps <= 0:
        raise ValueError("--goal-offset-steps must be greater than zero")
    if args.eval_index < 0:
        raise ValueError("--eval-index must be non-negative")


def save_history(input_dir: Path, frames: list[np.ndarray], label: int | str) -> list[Path]:
    paths = []
    for idx, frame in enumerate(frames):
        path = input_dir / f"current-{label}-{idx:02}.jpg"
        Image.fromarray(frame).save(path)
        paths.append(path)
    return paths


def load_eval_case(
    path: Path,
    goal_offset_steps: int,
    eval_seed: int,
    eval_index: int,
    dataset_row: int | None,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"PushT H5 not found: {path}")
    with h5py.File(path, "r") as h5:
        episode_idx = h5["episode_idx"][:]
        step_idx = h5["step_idx"][:]
        ep_len = h5["ep_len"][:]
        ep_offset = h5["ep_offset"][:]

        if dataset_row is None:
            max_start_by_episode = {
                ep: int(length) - goal_offset_steps - 1 for ep, length in enumerate(ep_len)
            }
            max_start_per_row = np.asarray(
                [max_start_by_episode[int(ep)] for ep in episode_idx],
                dtype=np.int64,
            )
            valid_indices = np.nonzero(step_idx <= max_start_per_row)[0]
            if len(valid_indices) == 0:
                raise ValueError("no valid PushT start rows found for requested goal offset")
            if eval_index >= len(valid_indices):
                raise ValueError(
                    f"--eval-index {eval_index} exceeds {len(valid_indices)} valid rows"
                )
            rng = np.random.default_rng(eval_seed)
            selected = np.sort(
                valid_indices[
                    rng.choice(len(valid_indices) - 1, size=eval_index + 1, replace=False)
                ]
            )
            row = int(selected[eval_index])
        else:
            row = int(dataset_row)
            if row < 0 or row >= len(step_idx):
                raise ValueError(f"--dataset-row {row} is outside dataset row range")

        episode = int(episode_idx[row])
        step = int(step_idx[row])
        goal_row = int(ep_offset[episode] + step + goal_offset_steps)
        episode_end = int(ep_offset[episode] + ep_len[episode])
        if goal_row >= episode_end:
            raise ValueError(
                f"goal row {goal_row} leaves episode {episode}; choose an earlier row"
            )

        return {
            "row": row,
            "episode": episode,
            "step": step,
            "goal_row": goal_row,
            "start_state": h5["state"][row].astype(np.float64),
            "goal_state": h5["state"][goal_row].astype(np.float64),
            "start_pixels": h5["pixels"][row].astype(np.uint8),
            "goal_pixels": h5["pixels"][goal_row].astype(np.uint8),
        }


def run_rust_planner(
    args: argparse.Namespace,
    current_paths: list[Path],
    goal_path: Path,
    output_html: Path,
) -> None:
    cmd = [
        args.cargo,
        "run",
        "--release",
        "--locked",
        "--features",
        "hub",
        "--bin",
        "lewm-plan-images",
        "--",
        "--hf-repo",
        args.hf_repo,
        "--goal",
        str(goal_path),
        "--planner",
        args.planner,
        "--samples",
        str(args.samples),
        "--iterations",
        str(args.iterations),
        "--horizon",
        str(args.horizon),
        "--history-size",
        str(args.history_size),
        "--seed",
        str(args.seed),
        "--output",
        str(output_html),
    ]
    for path in current_paths:
        cmd.extend(["--current", str(path)])
    subprocess.run(cmd, check=True)


def execute_plan(
    env: gym.Env,
    sequence: np.ndarray,
    action_block: int,
    action_stats: dict[str, list[float]] | None,
) -> tuple[list[np.ndarray], list[list[float]], list[float], dict, dict, bool]:
    frames = [np.asarray(env.render(), dtype=np.uint8)]
    executed_actions: list[list[float]] = []
    rewards: list[float] = []
    final_obs = None
    final_info = None
    for block in sequence:
        actions = block.reshape(action_block, -1)
        if action_stats is not None:
            mean = np.asarray(action_stats["mean"], dtype=np.float32)
            std = np.asarray(action_stats["std"], dtype=np.float32)
            actions = actions[:, :2] * std + mean
        for action in actions:
            action = np.clip(action[:2], -1.0, 1.0).astype(np.float32)
            final_obs, reward, terminated, truncated, final_info = env.step(action)
            executed_actions.append(action.tolist())
            rewards.append(float(reward))
            frames.append(np.asarray(env.render(), dtype=np.uint8))
            if terminated or truncated:
                return frames, executed_actions, rewards, final_obs, final_info, True
    assert final_obs is not None and final_info is not None
    return frames, executed_actions, rewards, final_obs, final_info, False


def load_action_stats(path: Path, chunk_size: int = 262_144) -> dict[str, list[float]]:
    if not path.exists():
        raise FileNotFoundError(f"PushT action stats H5 not found: {path}")
    count = 0
    total = np.zeros(2, dtype=np.float64)
    total_sq = np.zeros(2, dtype=np.float64)
    with h5py.File(path, "r") as h5:
        actions = h5["action"]
        for start in range(0, actions.shape[0], chunk_size):
            batch = actions[start : start + chunk_size].astype(np.float64)
            total += batch.sum(axis=0)
            total_sq += (batch * batch).sum(axis=0)
            count += batch.shape[0]
    mean = total / count
    variance = total_sq / count - mean * mean
    std = np.sqrt(np.maximum(variance, 1e-12))
    return {
        "path": str(path),
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
        "count": count,
    }


def save_gif(frames: list[np.ndarray], path: Path) -> None:
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0,
    )


def save_montage(frames: list[np.ndarray], path: Path, cols: int = 6) -> None:
    picks = np.linspace(0, len(frames) - 1, num=min(12, len(frames)), dtype=int)
    images = [Image.fromarray(frames[idx]) for idx in picks]
    w, h = images[0].size
    rows = int(np.ceil(len(images) / cols))
    canvas = Image.new("RGB", (cols * w, rows * (h + 22)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (idx, image) in enumerate(zip(picks, images, strict=False)):
        x = (i % cols) * w
        y = (i // cols) * (h + 22)
        canvas.paste(image, (x, y))
        draw.text((x + 6, y + h + 4), f"t={idx}", fill=(0, 0, 0))
    canvas.save(path)


def render_demo_html(payload: dict) -> str:
    def rel(path: str) -> str:
        return Path(path).relative_to(Path(payload["rollout_gif"]).parents[1]).as_posix()

    current_images = "\n".join(
        f'<img src="{rel(path)}" alt="current history frame">'
        for path in payload["current_history"]
    )
    final_history = "\n".join(
        f'<img src="{rel(path)}" alt="final history frame">'
        for path in payload["final_history"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PushT LeWM Rust Demo</title>
<style>
body {{ margin: 0; background: #101217; color: #f2f5f8; font-family: system-ui, sans-serif; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
h1 {{ margin: 0 0 8px; }}
.grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
.panel {{ border: 1px solid #2d3541; background: #171b22; border-radius: 8px; padding: 16px; }}
.wide {{ grid-column: 1 / -1; }}
.history {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px; }}
img {{ max-width: 100%; border-radius: 6px; background: #080a0d; }}
a {{ color: #78e0d4; }}
code {{ color: #dbe4ef; }}
</style>
</head>
<body>
<main>
<h1>PushT LeWM Rust Demo</h1>
<p><code>swm/PushT-v1</code> frames, <code>{payload["hf_repo"]}</code> checkpoint, Rust/Candle CUDA planner, then selected actions executed in PushT.</p>
<section class="grid">
<div class="panel"><h2>Initial Current History</h2><div class="history">{current_images}</div></div>
<div class="panel"><h2>Goal</h2><img src="{rel(payload["goal_image"])}" alt="goal"></div>
<div class="panel"><h2>Executed Rollout</h2><img src="{rel(payload["rollout_gif"])}" alt="rollout gif"></div>
<div class="panel"><h2>Final Frame</h2><img src="{rel(payload["final_image"])}" alt="final frame"></div>
<div class="panel wide"><h2>Montage</h2><img src="{rel(payload["rollout_montage"])}" alt="rollout montage"></div>
<div class="panel wide"><h2>Final History</h2><div class="history">{final_history}</div></div>
<div class="panel wide">
<h2>Planner</h2>
<p>Dataset row: <code>{payload["dataset"]["row"]}</code>. Episode: <code>{payload["dataset"]["episode"]}</code>. Start step: <code>{payload["dataset"]["step"]}</code>. Goal offset: <code>{payload["dataset"]["goal_offset_steps"]}</code>.</p>
<p>Success: <code>{payload["success"]}</code>. Final distance: <code>{payload["final_distance"]:.6f}</code>. Last selected cost: <code>{payload["selected_cost"]:.6f}</code>. Total planning time: <code>{payload["planning_ms"]:.3f} ms</code>. Replans: <code>{payload["replans_completed"]}</code>. Executed actions: <code>{len(payload["executed_actions"])}</code>.</p>
<p><a href="{Path(payload["planner_runs"][-1]["report"]).name}">Open final planner report</a></p>
</div>
</section>
</main>
</body>
</html>
"""


if __name__ == "__main__":
    main()
