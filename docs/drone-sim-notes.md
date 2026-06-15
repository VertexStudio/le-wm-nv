# Drone Sim Notes

## Plant fitting experiments

The gate-segment oracle replay is the current plant-fidelity check:

```text
start_row=1020
gate_order=1,4,3,2
oracle_replay_rows=300
gate_radius=0.85
```

Baseline analytic plant:

```text
gates=2/4
pos_rmse=8.434m
final_pos_err=9.371m
best_gate_dists=[1:0.84,2:0.85,3:2.69,4:inf]
```

Replay-aligned scalar fit found:

```text
--hover-throttle 0.20000
--max-thrust-weight 3.600
--max-roll-rate 8.0000
--max-pitch-rate 12.0000
--max-yaw-rate 10.0000
--rate-kp 32.000
--rate-damping 8.000
--linear-drag 0.3000
--quadratic-drag 0.0300
```

Oracle replay with those flags:

```text
gates=3/4
pos_rmse=2.964m
final_pos_err=3.728m
best_gate_dists=[1:0.84,2:0.85,3:0.85,4:1.07]
```

That is a real expert-action replay improvement, but it is not yet a better
closed-loop LeWM planner setting. With the same planner budget, the default
plant reached 2/4 gates while this fitted scalar plant reached 1/4 gates.

Tested and rejected: cubic stick curves and quadratic throttle curves. A sweep
around the fitted scalar plant did not improve the oracle replay; the best
setting stayed equivalent to the original linear channel mapping. Those curve
knobs were intentionally not kept in the runtime plant.

## LeWM expert-action latent rollout

`lewm-drone-rollout-eval` evaluates the trained drone WorldModel directly,
without the analytic plant. It uses real dataset history rows, rolls LeWM with
recorded expert actions, and compares predicted embeddings to embeddings of the
actual future observations.

The current drone LeWM observation contract is pose-only:

```text
pos_world[0..3]
rotmat_world_from_body[0..9]
```

Regenerate rollout numbers after training a pose12 model.
