# TRACE adapter contract

`trace-core` never imports the target RWM implementation. A target baseline
provides the tools named in `pipeline.example.json` and must satisfy these
behavioral contracts.

## Candidate collector

- Restore the complete dataset snapshot, including root state, joint state,
  command and action histories.
- Use the baseline actor checkpoint only; do not load an expert actor or
  critic.
- Map `simulator.rollout_actor_arguments` to the target collector's neutral
  policy-loading interface. The template value
  `{rollout_actor_checkpoint}` is always resolved from the baseline adapter.
- Roll out in the normal simulator with automatic reset disabled.
- Create all branches for one source from the same realized state, command and
  simulator parameters.
- Record finite/reset/terminal masks and a stable `start_state_id`.

## Summary adapter

The scorer/rule boundary is behavioral JSONL. It includes command mode,
per-axis command values, survival, true-simulator tracking errors, direction,
signed transport, posture, action safety and gait stability.

Combined commands additionally provide:

- `minimum_active_axis_response_ratio`;
- `minimum_active_axis_transport_ratio`.

These weakest-component fields prevent a branch that only realizes forward
motion from passing an `x+yaw` or `x+y` command.

## Replay materializer

- Consume `portable_trace_selection_v1`; accept either
  `score_source.kind=rule` or `score_source.kind=learned_scorer`.
- Recompute transition rewards only after selection, using the frozen baseline
  reward configuration.
- Match baseline `n_step`, `gamma`, observation order and action history.
- Treat a window boundary as a bootstrap boundary, not an environment
  terminal.
- Preserve real terminal transitions and never bootstrap across them.
- Consume `TRACE_REPLAY_ZERO_OBSERVATION_INDICES`,
  `TRACE_REPLAY_REWARD_VERSION` and
  `TRACE_REPLAY_RECOMPUTE_REWARD_AFTER_SELECTION` from the replay command
  environment. Refuse to materialize replay if these constraints cannot be
  honored.

The current V12 route is asymmetric rather than fully No-BLV: the actor drops
base linear velocity `[0, 1, 2]`, while the critic consumes all 48 observation
coordinates. The world model does not consume or supervise those state
coordinates but still emits nonzero predictions, and the frozen
`v1_1_dense_progress` reward reads them.

Therefore V12 TRACE must not zero `[0, 1, 2]` only in the secondary replay.
The initial adapter uses `source_native_full_state`: physical simulator
velocity in TRACE replay and native RWM output in primary replay. Because this
can reveal replay source to the critic, a source-leakage audit is mandatory
before training. Any later projection policy must be compared as an explicit
ablation, not silently substituted.

## Trainer adapter

The trainer mixes the selected replay at the configured batch fraction. For
the current V12 protocol, baseline, random-TRACE and selected-TRACE arms all
initialize actor, critic, target critic, temperature and optimizers from zero
with matched seeds. The frozen baseline actor checkpoint is used only to
generate candidate rollouts. A same-size random replay arm uses the same
candidate pool, command distribution, reward materializer and transition
count.
