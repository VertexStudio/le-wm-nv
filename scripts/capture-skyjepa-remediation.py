"""Capture the selected seed-7 simulator locally on X11, without other windows.

Run after training/control measurements:
  uv run --locked scripts/capture-skyjepa-remediation.py ARTIFACT_ROOT
Requires ffmpeg (x11grab/libx264), xwininfo and xprop. Never uploads anything.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def owned_window(pid):
    tree = subprocess.check_output(["xwininfo", "-root", "-tree"], text=True)
    ids = re.findall(r'(0x[0-9a-fA-F]+) "SkyJEPA rotor-force UAV simulator"', tree)
    for window in ids:
        owner = subprocess.check_output(["xprop", "-id", window, "_NET_WM_PID"], text=True)
        if re.search(rf"=\s*{pid}\s*$", owner):
            return window
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    display = os.environ.get("DISPLAY")
    if not display:
        parser.error("an X11 DISPLAY is required")
    warm = json.loads((root / "warm-start-selection.json").read_text())["selection"]["trained-mppi"]
    binary = root / "bin/skyjepa-drone-sim"
    checkpoint = root / "seed-7/prober"
    directory = root / "simulation"
    directory.mkdir(exist_ok=True)
    stem = directory / "seed-7-figure-eight"
    for suffix in (".mp4", ".png", ".capture.json", ".gui.log"):
        if stem.with_suffix(suffix).exists():
            parser.error(f"refusing to overwrite {stem.with_suffix(suffix)}")
    command = [str(binary), "--checkpoint-dir", str(checkpoint), "--scenario", "figure-eight",
               "--randomize-domain", "--domain-seed", "31415", "--warm-start", warm,
               "--samples", "512", "--horizon", "15", "--planner-seed", "7", "--time-scale", "1"]
    with stem.with_suffix(".gui.log").open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, "WINIT_UNIX_BACKEND": "x11"})
        try:
            window = None
            for _ in range(60):
                if process.poll() is not None:
                    raise RuntimeError(f"simulator exited: inspect {stem.with_suffix('.gui.log')}")
                window = owned_window(process.pid)
                if window:
                    break
                time.sleep(0.5)
            if not window:
                raise RuntimeError("could not resolve an X11 window owned by this simulator process")
            time.sleep(3)
            capture = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-n", "-f", "x11grab",
                       "-window_id", str(int(window, 16)), "-draw_mouse", "0", "-framerate", "30",
                       "-i", display, "-t", "20", "-vf", "scale=1280:-2", "-c:v", "libx264",
                       "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                       str(stem.with_suffix(".mp4"))]
            subprocess.run(capture, check=True)
            if process.poll() is not None:
                raise RuntimeError("simulator exited during capture")
        finally:
            if process.poll() is None:
                process.terminate()  # Only the process launched above, never another user's simulator.
                process.wait(timeout=10)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "warning", "-n", "-ss", "16",
                    "-i", str(stem.with_suffix(".mp4")), "-frames:v", "1", "-update", "1",
                    str(stem.with_suffix(".png"))], check=True)
    report = {"simulator_command": command, "capture_command": capture, "window_id": window,
              "checkpoint_file_sha256": sha(checkpoint / "checkpoint.json"), "executable_sha256": sha(binary),
              "video_sha256": sha(stem.with_suffix(".mp4")), "screenshot_sha256": sha(stem.with_suffix(".png")),
              "notes": "20 seconds, normal simulation speed, seed 7 selected warm start; render-contended HUD latency is not the headless benchmark"}
    with stem.with_suffix(".capture.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(stem.with_suffix(".mp4"))


if __name__ == "__main__":
    main()
