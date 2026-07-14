# Go2 0.5-strength friend-flat DR 2x2 experiment

## Fixed assumptions

- Reuse the best existing 0.5-strength expert policy. Do not retrain the expert.
- Keep the 0.5-strength `RR_calf_joint=0.5` task in dataset collection.
- Train an unmasked proprioceptive RWM from each collected dataset.
- Apply physical DR only in the real MJLab collection environment.
- Apply action/observation interface DR only while training the final policy against the frozen RWM.

## Strict 2x2 design

| Collection branch | Final policy branch | Shared upstream artifact |
| --- | --- | --- |
| friend-flat physical DR | clean frozen-RWM interface | base dataset + base RWM |
| friend-flat physical DR | randomized frozen-RWM interface | base dataset + base RWM |
| friend-flat physical DR + action noise 0.03 | clean frozen-RWM interface | action-noise dataset + action-noise RWM |
| friend-flat physical DR + action noise 0.03 | randomized frozen-RWM interface | action-noise dataset + action-noise RWM |

The two final policies under one collection branch always reuse the exact same dataset and RWM checkpoint. This isolates the final-interface factor from upstream training randomness.

## Collection DR mapping

The `friend_flat` preset is based on `wty-yy/go2_rl_gym@vanilla_train`:

- foot friction: `[0.0, 2.0]`
- base mass addition: `[-1.0, 1.0] kg`
- non-base link mass scale: `[0.9, 1.1]`
- base COM offset: `[-0.03, 0.03] m` per axis
- PD stiffness/damping scale: `[0.9, 1.1]`
- motor zero offset: `[-0.035, 0.035] rad`
- motor strength: `[0.8, 1.2]`
- pushes every `4 s`, with friend linear/angular velocity ranges
- actuator delay: `0..4` physics steps at `0.005 s` per step
- feature-specific uniform actor-observation noise

MuJoCo has no direct coefficient-of-restitution field. The friend restitution setting is intentionally not approximated with an unvalidated `solref` mapping; this omission is recorded in dataset metadata.

## Frozen-RWM interface DR

- action Gaussian noise: `0.02`
- persistent action bias std: `0.005`
- persistent action scale: `[0.95, 1.05]`
- action delay: `0..1` policy steps
- persistent joint-position observation bias: `[-0.035, 0.035]`
- feature-specific uniform observation noise:
  - base angular velocity: `+-0.05`
  - projected gravity: `+-0.05`
  - joint position: `+-0.01`
  - joint velocity: `+-0.075`
  - command and previous action: no noise
  - privileged base linear velocity: no noise

## Dry-run

```bash
cd /root/unitree_rl_mjlab

EXPERT_POLICY_PATH=/path/to/best_0p5_expert \
DRY_RUN=true CONFIRM_RUN=false \
bash scripts/reinforcement_learning/go2_0p5_proprioceptive/launch_friend_flat_dr_rwm_ablation_2x2.sh
```

## Full run

The launcher waits for idle GPUs. When called outside tmux, a confirmed full run automatically starts its watcher in a detached tmux session.

```bash
cd /root/unitree_rl_mjlab

EXPERT_POLICY_PATH=/path/to/best_0p5_expert \
COLLECT_RANDOMIZATION_COMPONENTS=friction,mass_com,motor,push,observation \
GPU_POOL="0 1 2 3 4 5 6 7" \
DRY_RUN=false CONFIRM_RUN=true \
bash scripts/reinforcement_learning/go2_0p5_proprioceptive/launch_friend_flat_dr_rwm_ablation_2x2.sh
```

`COLLECT_RANDOMIZATION_COMPONENTS` must come from the completed short
combination validation. The default remains `all` for backward compatibility;
do not use that default for the formal run when the full preset fails the
preregistered quality gates.

Useful overrides:

```bash
GPU_MAX_MEMORY_USED_MIB=2500
GPU_MAX_UTILIZATION=15
GPU_POLL_SECONDS=30
```

## Monitoring

```bash
tmux ls
find logs/queues/go2_0p5_friend_flat_dr -name state.txt -print -exec tail -n 6 {} \;
find logs/queues/go2_0p5_friend_flat_dr -name launcher.log -print -exec tail -n 30 {} \;
```
