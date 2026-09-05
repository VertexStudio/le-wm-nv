# Corrected SkyJEPA pilot media

The README uses the corrected seed-7 checkpoint from the
[2026-09-05 three-seed experiment](skyjepa-remediation-results.md), not the
historical pilot. This is an actual Bevy simulator capture, not a generated
animation. Publication to `origin/main` was authorized on 2026-09-05.

- [Full MP4](skyjepa-v3-figure-eight.mp4): unchanged source recording,
  20 seconds, 1280×800, 30 fps, H.264, 5,328,617 bytes.
- [GIF preview](skyjepa-v3-figure-eight.gif): seconds 8–16 of that MP4,
  normal speed, 800×500, 10 fps, 80 frames, looping, 5,799,904 bytes.
  Palette reduction is for file size; no telemetry or flight paths are altered.
- Historical [MP4](skyjepa-trained-figure-eight.mp4) and
  [GIF](skyjepa-trained-figure-eight.gif) remain unchanged.

The flight uses a randomized in-range plant (raw domain seed 31415),
figure-eight reference, training seed 7, planner seed 7, 512 candidates,
horizon 15 and the validation-selected fresh-prior warm start. The capture is
a demonstration, not a new benchmark or evidence of out-of-support flight.
Render-contended HUD latency includes prediction export and must not be
substituted for headless benchmark latency.

Yellow: executed trajectory. Cyan: reference. Magenta: learned prediction.
Green bars: commanded rotor forces. The HUD separates prior action and model
correction.

## Provenance

The original local MP4 and its `.capture.json` sidecar remain under:

```text
/home/rozgo/.stable_worldmodel/le-wm-nv-data/skyjepa-remediation-v3/simulation/seed-7-figure-eight.mp4
```

SHA-256 fingerprints:

- MP4: `12a96c30fc64b494566ac701628f47af1fe96964c3af88a127f07caad7606314`
- GIF: `a43535defd33d94b2d5a03c649935244f2ee9b1ab6f36bbd8f63a92aa8def86d`
- Checkpoint manifest file: `cf97165c25a71906ce0cb8ae36ecbdcccc380137a7405dd83c38476550d660c6`
- GUI executable: `0cec102fa31f875b418aa22c485db8c1c82b50f8c2a7eab81c63629f13d3245b`

## Reproduce the GIF

Generated with FFmpeg `n8.1.2-22-g94138f6973-20260717`. From the repository root:

```bash
ffmpeg -hide_banner -loglevel warning -n \
  -ss 8 -t 8 -i docs/skyjepa-v3-figure-eight.mp4 \
  -filter_complex '[0:v]fps=10,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle' \
  -loop 0 /tmp/skyjepa-v3-preview.gif
```

Choose a new output path if it exists; `-n` prevents overwriting. Encoding
details may vary between FFmpeg versions. The committed preview was decoded
and visually inspected, and the MP4 hash was checked against the capture
sidecar before publication.
