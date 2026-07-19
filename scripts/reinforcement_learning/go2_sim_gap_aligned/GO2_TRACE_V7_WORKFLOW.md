# Go2 Corrected TRACE V7 Workflow

This is the canonical collaborator entry point for the corrected V7
experiments. V5/V6 experiment directories and their results are audit-only and
must not be used as final results.

## Protocol

- Sides: `sim`, `real`
- Conditions: `g0`, `rr05`, `p5`, `rr03`, `p75`
- Methods per condition: RWM baseline, TRACE `r10`, TRACE `r25`
- Dataset: one immutable, condition-specific 25K dataset
- Frozen RWM: one model trained from that condition's dataset
- Common policy warmup: 10M environment steps
- Final policy budget: 50M environment steps for every method
- TRACE: `T=2`, `H=100`, 8 refreshes, 5M steps per refresh
- Candidate selection: scorer top 25%
- TRACE replay ratios: 10% and 25%
- Failure quota: disabled; naturally terminated trajectories remain eligible
- Replay continuity: every refresh resumes the previous RWM replay buffer

The real `g0` dataset must be the go2sun recollection. The legacy go2 `g0`
dataset is retained only for cross-robot audit and must not enter the main
five-gap table.

## State and Policy Shapes

The RWM state is 45-dimensional:

1. base linear velocity: 3
2. base angular velocity: 3
3. projected gravity: 3
4. relative joint position: 12
5. joint velocity: 12
6. actuator force: 12

The RWM predicts the next 45-dimensional state plus contact and termination.
The final actor receives 45-dimensional hardware-observable proprioception;
the three base-linear-velocity dimensions are not actor inputs.

## Inputs

Create a dataset manifest from
`go2_trace_v7_datasets.example.env`. Every path must point to a validated 25K
dataset. The launcher refuses missing files.

```bash
export V7_DATASET_MANIFEST=/absolute/path/to/go2_trace_v7_datasets.env
export FORMAL_ROOT=$PWD/logs/experiments/go2_trace_v7_corrected/$(date +%Y%m%d_%H%M%S)
```

By default each condition trains its RWM, common warmup, initial scorer and
baseline from scratch. To reuse immutable prepared inputs, set `REUSE_ROOT` to
a V7-compatible experiment root containing `ready.env`, `scorer_ready.env`,
and `prepare_completed.txt` for each side/condition. Refresh artifacts are
never reused.

## Formal Training

```bash
bash scripts/reinforcement_learning/go2_sim_gap_aligned/launch_go2_trace_v7_formal.sh
```

GPU assignments can be overridden with variables such as `GPU_SIM_G0=1` and
`GPU_REAL_P75=7`. Each job runs in a `v7f_<side>_<condition>` tmux session.

## Paired Evaluation

After training, use the same fixed command sequences for baseline, r10 and
r25:

```bash
export FORMAL_ROOT=/absolute/path/to/completed/v7/root
export EVAL_ROOT=$PWD/logs/evaluations/go2_trace_v7/$(date +%Y%m%d_%H%M%S)
bash scripts/reinforcement_learning/go2_sim_gap_aligned/launch_go2_trace_v7_gap_only_eval.sh
```

The standard evaluation uses 256 environments, 1000 steps, command switches
every 50 steps, and seeds 400/401/402. Primary metrics are fixed-horizon
return, termination rate, restricted survival time, XY/yaw tracking error,
post-switch error, action saturation and epistemic uncertainty.

## Required Completion Evidence

- `baseline/stage/artifact_path.txt` exists for every condition.
- `trace_r10/completed.txt` and `trace_r25/completed.txt` exist.
- Each refresh loads the preceding replay buffer.
- Intermediate checkpoints do not contain replay buffers.
- The latest committed refresh contains exactly one replay buffer.
- Replay metadata records `failure_quota_mode=disabled_natural`.
- `paired_summary.csv` contains three seeds for each method and condition.

Low-level files that retain `v5` in their names implement the stable artifact
schema inherited by V7. They are internal compatibility components, not older
experiment entry points.
