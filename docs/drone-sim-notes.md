# Drone Sim Notes

## Plant fitting experiments

The gate-segment oracle replay is the current plant-fidelity check. It replays
recorded expert actions through the Bevy plant and compares against the dataset
trajectory. This isolates plant fidelity from LeWM planning quality.

```text
start_row=1020
gate_order=1,4,3,2
oracle_replay_rows=260
gate_radius=0.85
```

Baseline analytic plant:

```text
gates=2/4
pos_rmse=7.587m
final_pos_err=15.112m
best_gate_dists=[1:0.84,2:0.85,3:2.69,4:inf]
```

Replay-aligned aggressive plant fit found:

```text
--hover-throttle 0.20000
--max-thrust-weight 3.600
--thrust-curve 0.000
--max-roll-rate 8.0000
--max-pitch-rate 12.0000
--max-yaw-rate 10.0000
--rate-kp 32.000
--rate-damping 8.000
--linear-drag 0.3000
--quadratic-drag 0.0300
--body-linear-drag 0.0500,0.8000,0.0000
--body-quadratic-drag 0.0300,0.1000,0.0000
```

Oracle replay with those flags:

```text
gates=4/4
finished=true
pos_rmse=2.712m
final_pos_err=1.247m
best_gate_dists=[1:0.84,2:0.84,3:0.85,4:0.85]
```

This is a real plant-fidelity improvement: expert actions now complete the lap
in the plant. It is not yet a closed-loop LeWM planner success. With the current
pose12 model and `h6/s64/i1` planner, the fitted plant still reaches 2/4 gates.
With a larger `h8/s256/i3` planner budget, default plant reached 3/4 while the
fitted plant reached 2/4. That means plant fidelity improved, but current
closed-loop behavior is still dominated by LeWM planner/model action selection.

The runtime plant keeps the added terms with neutral defaults, so existing
simulator behavior does not change unless these flags are passed. Sign flips
remain opt-in in the fitter and are not part of the accepted plant.

## Plan trace diagnostics

`lewm-drone-sim --headless-steps ... --plan-trace <path>` writes per-plan JSON
diagnostics from the live simulator. Each event records current pose, target
pose, active gate, selected action sequence, matching dataset expert action
sequence when available, LeWM scores for both sequences, per-step embedding
costs, and source-row pose drift.

Current traced failure:

```text
trace=target/drone-plan-traces/full-lap-h6-fitted-plant.json
planner=h6/s64/e8/i1 future-min
plant=fitted aggressive plant
result=gates=2/4, best_gate_dists=[1:0.85,2:0.84,3:1.63,4:inf]
```

Gate-3 summary from that trace:

```text
events=142
selected_better_than_expert=103
expert_better_than_selected=39
mean_selected_score=425.7
mean_expert_score=484.2
mean_first_action_l2=1.00
source_pose_drift=0.98m..49.90m
target_distance_min=0.60m
gate3_best_distance=1.63m
```

Interpretation: this is not just a plant fidelity problem. During gate 3, the
LeWM objective often scores the selected action sequence better than the
dataset expert sequence, while the plant state is already drifting away from
the recorded source-row pose. The next useful work is to improve closed-loop
target/objective/model behavior, not to add more plant knobs.

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
