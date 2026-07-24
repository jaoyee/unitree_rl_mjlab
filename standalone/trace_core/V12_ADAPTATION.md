# Current V12 adaptation

This contract targets the completed
`go2_rwm_v12_10x5m_no_bc_20260724` route, not the abandoned
`v1_1_dense_progress_noblv` experiment.

## Frozen baseline facts

- policy, critic, target critic and temperature initialize from zero;
- no policy checkpoint resume and no behavior cloning;
- world model input is 42-D and output is 45-D;
- base linear velocity `[0, 1, 2]` is excluded from world-model input and
  state loss, but remains a nonzero output;
- actor observation is 45-D after removing those three coordinates;
- critic observation is the full 48-D observation; the critic receives the
  action separately;
- reward is `v1_1_dense_progress`;
- replay uses `n_step=3`, `gamma=0.99`, batch size 2048;
- candidate source datasets contain exact MJLab snapshots.

## Required target-project bridge

The V12 project needs four small integration points. Keep them outside
`trace-core`; they are baseline adapters.

1. Candidate policy API
   - load a completed V12 baseline actor checkpoint;
   - accept a TRACE-only action temperature in the proprioceptive agent;
   - never substitute the E0 collector actor.
2. Reward materializer
   - load the complete selected checkpoint `rwm_flashsac_config.yaml`;
   - instantiate reward state with `reward_version=v1_1_dense_progress`;
   - copy every reward weight/threshold used by the RWM environment;
   - use physical simulator state and zero epistemic uncertainty;
   - do not add terminal, failure-backprop, saturation or action-delta
     penalties absent from the baseline.
3. Replay mixer
   - implement `configure_trace_replay` on the actual proprioceptive agent;
   - mix before actor-observation slicing;
   - apply the 45-D actor view to both primary and TRACE rows;
   - leave the critic input 48-D;
   - report the realized TRACE row count for every update.
   - if `TRACE_SOURCE_LEAKAGE_AUDIT_REQUIRED=1`, refuse to start training until
     a passing audit report has been produced from the actual primary and
     TRACE critic-observation rows.
4. Evaluation adapter
   - run fixed-horizon rollouts without resetting a failed environment;
   - export `[steps, envs]` termination, `[steps, envs, 2 or 3]` base-linear
     velocity, `[steps, envs]` base-yaw velocity and `[steps, envs, 3]`
     command arrays;
   - compute success percent, mean survival steps and velocity error with
     `trace-evaluate-three-metrics`.

## Base-linear-velocity boundary

Do not zero `[0, 1, 2]` only in TRACE replay. The initial faithful policy is:

- primary replay retains native RWM outputs;
- TRACE replay retains physical simulator values;
- actor never sees either;
- critic sees both;
- training is blocked until `trace-audit-source-leakage` measures how easily a
  linear discriminator separates the sources.

If leakage is excessive, evaluate `frozen_rwm_projection` as a named ablation.
Changing the critic input for both arms would define a different baseline and
must not be presented as the current V12 result.

## Experiment controls

For each selected size/condition checkpoint, recollect five independent
candidate pools by launching otherwise identical pipeline configurations with
`candidate_group_index` 0 through 4 and distinct output roots. The group index
is folded into both source-selection and rollout seeds. Compare matched
from-zero arms:

1. no TRACE;
2. same-size command-distribution-matched random TRACE;
3. rule-selected TRACE;
4. learned-scorer TRACE only after rule selection improves all three metrics.
