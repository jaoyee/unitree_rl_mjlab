# Go2 V3 TRACE T=2 protocol

This version is additive and does not overwrite V2 artifacts.

## Fixed protocol

- TRACE action temperature: `2.0`
- imperfect-simulator rollout: `200` policy steps (`4.0 s`)
- reset states: `1024`
- trajectories per reset state: `4`
- candidate collectors: `80%` T=2 expert sampling and `20%` fixed random
  boundary sampling
- candidate transitions: `1024 * 4 * 200 = 819200`
- command range: `vx [-0.5, 0.5]`, `vy [-0.2, 0.2]`, `yaw [-0.4, 0.4]`
- selected replay: top `25%`, with at least `20%` terminal trajectories when available
- terminal transition reward: `-10.0`
- TRACE policy replay ratios: `0.10` and `0.25`
- policy seed: `300`
- final policy budget: `50M` environment steps

## Failure handling

`build_trace_replay_v3.py` combines the dataset unsafe-orientation termination with
environment `done && !timeout`. It retains terminal-aligned variable-length prefixes
of at least 20 steps, assigns the terminal transition a negative reward, and reserves
a configured fraction of selected trajectories for failures. Training is blocked unless
the selected replay contains at least 32 terminal trajectories, a terminal-trajectory
ratio of at least 5%, terminal replay rows, and a negative terminal reward.

## Checkpoint selection

Every 5000-step checkpoint plus the final checkpoint is evaluated in MJLab with the
`calibrated_default` randomization preset. Selection is survival-first, then uses
termination count, return, and command error as tie breakers. The selected checkpoint
replaces the stage artifact pointer; training still retains every checkpoint.

## Alignment report

`compare_go2_sim_real_dynamics.py` compares:

- all six state blocks;
- one-step state deltas;
- action magnitude and saturation;
- contact fraction;
- command scalar distributions;
- effective continuous 40-step history starts.

This report diagnoses remaining sim/real support mismatch. It does not rewrite physical
states or silently scale transitions.
