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

Reference run:

```text
model=/home/rozgo/.stable_worldmodel/le-wm-nv-runs/drone-pose16-lewm-sigreg-all-data-20260614-175103
dataset=/home/rozgo/.stable_worldmodel/le-wm-nv-data/drone-racing-autonomous-100hz-pose16
start_row=1020
history=8
```

Results:

```text
horizon=25
one_step:      mean_l2=3.2474 mean_cos=0.9762 final_l2=3.3272 final_cos=0.9768
autoregressive mean_l2=8.2077 mean_cos=0.7924 final_l2=14.8561 final_cos=0.4218

horizon=50
one_step:      mean_l2=3.9211 mean_cos=0.9661 final_l2=8.6083 final_cos=0.8715
autoregressive mean_l2=15.1003 mean_cos=0.4552 final_l2=23.0681 final_cos=0.3509

horizon=300
one_step:      mean_l2=6.6681 mean_cos=0.9234 final_l2=6.5364 final_cos=0.9199
autoregressive mean_l2=13.3634 mean_cos=0.6336 final_l2=22.4809 final_cos=0.2389
```

Interpretation: local teacher-forced prediction is reasonably aligned on this
gate segment, but autoregressive latent rollout degrades quickly. That points
at rollout stability and planner/model coupling before more plant tuning.
