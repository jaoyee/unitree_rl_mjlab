# Go2 TRACE V10

This directory contains the current Go2 TRACE implementation. V8/V9 run directories,
label caches, scorer checkpoints, and replay shards are deliberately not part of V10.

## Canonical Paths

- Frozen protocol: `scripts/reinforcement_learning/rwm_trace/trace_v10_protocol.json`
- Formal launcher: `launch_go2_trace_v10_formal.sh`
- Pilot launcher: `launch_go2_trace_v10_pilots.sh`
- Condition runner: `run_go2_trace_v10_condition.sh`
- Refresh runner: `run_go2_trace_v10_branch.sh`
- Replay builder: `scripts/reinforcement_learning/rwm_trace/build_trace_replay_v10.py`
- Server artifact template: `trace_v10_artifacts.example.env`

The formal algorithm remains:

```text
condition 25K dataset -> frozen condition RWM
dataset state -> nominal simulator rollout
LLM labels -> condition-specific scorer
distribution-constrained global Top-25% -> simbuffer
90% continuous RWMbuffer + 10% simbuffer -> FlashSAC update
```

`trace_v10_protocol.json` is the only source of frozen TRACE parameters. Launchers may
override paths, GPU pools, and run roots, but not the protocol values.

## Diagnostic Branches

`launch_go2_trace_v10_metric_pilot.sh` is the completed scorer-free diagnostic. It
compares global metric Top-25%, random Top-25%, and no-TRACE control from one shared
candidate set. It is retained as evidence for the reward-collapse diagnosis; it is not
the formal TRACE entry point.

The current next experiment is the RWM-only RR05 reward sanity pilot:

```bash
V10_ARTIFACT_ENV=/absolute/server/artifacts.env \
V10_RUN_BASE=/absolute/new/run/root \
V10_GPU_POOL=1,2 \
bash scripts/reinforcement_learning/go2_sim_gap_aligned/launch_go2_reward_v2_rr05_pilot.sh
```

It runs paired V1 and hierarchical-V2 arms with the same actor, seed, 5M budget, and
evaluation commands. Both arms use `actor_only`, an empty replay buffer, and a fresh
reward normalizer. Loading the old V1 replay or normalizer into V2 is prohibited because
stored rewards are not compatible across reward versions.

Reward V2 is intentionally not the V10 formal default until this pilot shows correct
forward, lateral, and yaw response without a survival regression. RWMbuffer and
simbuffer must use the same reward version before any TRACE experiment uses V2.

Two RR05 follow-up launchers isolate critic adaptation without changing the reward or
TRACE protocol:

- `launch_go2_reward_v2_rr05_critic_warmup_pilot.sh` delays actor and temperature
  updates for the first 3,000 critic updates in one 5M run.
- `launch_go2_reward_v2_rr05_staged_pilot.sh` first runs a critic-only 5M stage, saves
  the critic, replay buffer, and reward normalizer, then restores the complete state
  for a separate 5M actor-finetuning stage.

These are reward-diagnostic pilots, not formal V10 TRACE launchers. Their outputs must
use new run roots and must not be treated as interchangeable with the paired V1/V2
comparison above.

## Validation

Run before syncing or committing:

```bash
python -m py_compile \
  src/tasks/rwm_velocity/mdp/rewards.py \
  scripts/reinforcement_learning/rwm_trace/build_trace_replay_v10.py
python -m unittest discover -s tests -p 'test_go2_reward_v2.py' -v
python -m unittest discover -s tests -p 'test_trace_v10_protocol.py' -v
python -m unittest discover -s tests -p 'test_trace_ood_audit.py' -v
```

Server datasets, models, policy checkpoints, logs, tmux state, and generated label
caches are runtime artifacts and must not be committed.
