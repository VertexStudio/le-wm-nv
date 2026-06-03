# Upstream Stable-Worldmodel Support

This repo tracks the upstream `stable-worldmodel` commit it has audited and
supports for checkpoint/runtime architecture.

## Supported Upstream

- Upstream repo: `https://github.com/galilai-group/stable-worldmodel`
- Supported upstream branch: `main`
- Supported upstream commit: `40dff37fc983c5276ada65eb1c7873cefbcccd8a`
- Commit title: `Fix SIGReg hardcoded device='cuda' (#231)`
- Audit date: `2026-06-02`
- Python package in `uv.lock`: `stable-worldmodel==0.1.0`
- PyPI tag commit: `56e64a6`

Diff from PyPI tag `56e64a6` to supported upstream commit `40dff37` touches:

- `README.md`
- `docs/index.md`
- `pyproject.toml`
- `stable_worldmodel/wm/loss.py`

The only Python source change is a training/loss device fix in `SIGReg`: random
projection tensors use `proj.device` instead of a hardcoded CUDA device.
This does not change LeWM inference architecture, checkpoint tensor layout,
pixel encoding, action embedding, latent rollout, or goal-cost computation.

## Audit Command

```bash
git ls-remote https://github.com/galilai-group/stable-worldmodel.git \
  HEAD refs/heads/main

git -C ../stable-worldmodel fetch origin main --prune
git -C ../stable-worldmodel diff --stat 40dff37fc983c5276ada65eb1c7873cefbcccd8a..origin/main
git -C ../stable-worldmodel diff --name-only 40dff37fc983c5276ada65eb1c7873cefbcccd8a..origin/main \
  -- stable_worldmodel/wm pyproject.toml
```

If upstream changes files under `stable_worldmodel/wm`, audit the affected model
graph before moving this supported commit forward.
