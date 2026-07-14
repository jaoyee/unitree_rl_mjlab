# Normal Go2 FixStand-v2 RWM DR 3x3x4 comparison

## Material Passport

- Origin: experiment planning discussion
- Date: 2026-07-12
- Version: v2.0
- Status: implementation complete; full 36-policy run not started
- Verification status: VERIFIED for v2 configuration, clean stance, complete
  E0-D0-P0 plumbing, and four-domain evaluation entry
- Canonical code root: `/root/unitree_rl_mjlab_model_based_broken`
- Canonical experiment root: `logs/experiments/go2_normal_dr_3x3x4/<RUN_ID>`
- Friend source repository: `wty-yy/go2_rl_gym`, branch `vanilla_train`
- Friend source commit: `34c237ba7d34cc8ff4e23e2952e1f420aecd901b`
- Conversion analysis: `scripts/reinforcement_learning/go2_normal_proprioceptive/references/wty_fullDR_analysis.md`

This document is the source of truth for the normal-strength Go2 domain
randomization comparison. Future scripts, launch commands, run directories, and
result summaries must follow this document. A deviation is allowed only when it
is recorded in the run manifest before the affected stage starts.

The v2 baseline supersedes the v1 friction, posture, gain, and command values.
The friend's original `[0, 2]` friction remains available as `friend_flat` for
source reproduction only. Formal normal-v2 runs use the calibrated presets
defined below and must use a new RUN_ID; v1 smoke artifacts are plumbing
history and must not be mixed into v2 performance comparisons.

## 1. Objective

The experiment first verifies the complete normal Go2 pipeline:

```text
MJLab expert training
  -> expert-driven dataset collection
  -> offline proprioceptive RWM training
  -> final policy training against the frozen RWM
  -> common robustness evaluation
  -> MuJoCo and real-robot deployment checks
```

The scientific objective is to measure the main effects and interactions of DR
at three stages:

1. DR used while training the data-collection expert.
2. DR used while that expert collects the offline dataset.
3. Small action/observation interface noise used while the final policy
   interacts with a frozen RWM.

The target full factorial contains `3 x 3 x 4 = 36` final policies.

The command support is enlarged conservatively rather than jumping to the
friend project's high-speed range:

```text
expert and collection: x=+-0.8 m/s, y=+-0.3 m/s, yaw=+-0.6 rad/s
frozen-RWM imagination: x=+-0.5 m/s, y=+-0.25 m/s, yaw=+-0.5 rad/s
```

The expert starts with the collection support rather than a narrow command
curriculum. Final-policy imagination stays inside the denser center of the
dataset support. This removes the earlier mismatch where the
dataset stopped near `0.5/0.2/0.4` and final-policy imagination used only
`0.3/0.15/0.3`.

## 2. Non-negotiable normal-Go2 baseline

This is not a continuation of the broken-joint or 0.5-strength experiment.
Every branch must satisfy all of the following:

- Every joint has nominal actuator strength `1.0`.
- `RR_calf_joint` is not fixed at `0.5`.
- No action, observation, world-model, or policy joint mask is active.
- No broken-joint task or reward override is active.
- The robot model, calibrated PD gains, action scale, control period, reward,
  command distribution, observation order, and termination definitions are
  identical across all 36 branches.
- The deploy template must be a new normal-strength template. It must not reuse
  `rwm_proprioceptive_0p5_rr_calf`.
- Training, evaluation, ONNX deployment, and FixStand use the same per-leg
  target pose `[hip=0.0, thigh=0.8, calf=-1.5]`.
- Training and deployment use moderate locomotion gains
  `kp=[30,40,50]`, `kd=[1.5,2,2.5]`; normal effort limits remain enabled.
  These gains sit between the old soft controller and FixStand, while the
  deployment transition ramps from FixStand over `0.5 s`.
- The normal task includes a base-height penalty targeted at `0.32 m`, the
  measured settled height of this pose in MJLab, rather than copying the
  friend's simulator-specific `0.38 m` target.
- Friend DR ranges are applied around the moderate locomotion actuator model
  (`hip=30/1.5`, `thigh=40/2`, `calf=50/2.5`, action scale `0.25`, 50 Hz policy
  control). The friend's nominal all-joint `20/0.5` controller is not copied,
  because changing nominal gains would confound DR with a different robot
  controller.
- The deploy actor remains proprioceptive. `base_lin_vel` must not become a
  hardware policy input. Privileged observations may be used by a training
  critic only when the existing asymmetric-training path supports them.
- A dedicated normal proprioceptive task must be registered before launch. Its
  actor is the 45-dimensional hardware-observable prefix, its critic may append
  privileged `base_lin_vel` to form 48 dimensions, and it is built from the
  normal Go2 task without broken-joint robot construction or broken-expert
  reward overrides. The existing 48-dimensional normal RWM task and the
  45-dimensional broken-RR-calf expert task are not valid substitutes.
- Exported deployment ONNX models use IR version 9, opset 18, and the verified
  policy observation/action signature.
- The companion `unitree_mujoco` normal Go2 foot/collision friction is `0.8`,
  matching the calibrated clean task instead of the previous `0.4`.
- The companion `unitree_cpp_deploy` controller blends measured entry pose and
  existing kp/kd into policy targets and gains over `0.5 s`; it must not switch
  FixStand pose, gains, and a nonzero first policy action in one control frame.

Before any full run, the launcher must assert that all joint strength scales are
`1.0`, all mask lists are empty, and no path contains an inherited 0.5 task
override.

## 3. Experimental factors

### 3.1 Expert-training factor E

Three experts are trained independently. They share the same normal Go2 task,
training budget, seed policy, network, replay settings, commands, rewards, and
checkpoint schedule. Only the DR preset differs.

| ID | Expert-training DR | Code mapping |
| --- | --- | --- |
| `E0` | MJLab default components with calibrated friction | `randomization_preset=calibrated_default` |
| `E1` | calibrated friend-flat DR at half range | `randomization_preset=calibrated_friend_flat`, `randomization_scale=0.5` |
| `E2` | calibrated friend-flat DR at full range | `randomization_preset=calibrated_friend_flat`, `randomization_scale=1.0` |

For all three expert branches:

```text
use_domain_randomization = true
use_push_randomization = true
use_observation_noise = true
joint_strength_scales = {}
broken_joint_names = []
```

For E1/E2, `calibrated_friend_flat` means the complete friend flat-task
conversion components with only the friction interval replaced by the
empirically calibrated range:

```text
friction,mass_com,motor,delay,observation,push,initial_state
```

Complex terrain and the friend's wide command curriculum are intentionally not
included. They remain fixed controls because this experiment targets flat-ground
locomotion and needs to attribute differences to DR rather than a changed task.

`E0`, `E1`, and `E2` are different DR definitions, not a strictly ordered
low/medium/high scalar. In particular, the legacy default has some observation,
COM, and push ranges that are larger than the friend preset. Analysis must not
label `E0` as "weak DR".

### 3.2 Dataset-collection factor D

Each expert is used to collect three datasets, producing nine datasets total.
The expert checkpoint is frozen during collection. Within one expert branch,
all three collection runs use the same command coverage, collector mixture,
transition count, environment count, policy checkpoints, and collector noise
settings. Only the environment DR preset differs.

| ID | Collection DR | Code mapping |
| --- | --- | --- |
| `D0` | MJLab default components with calibrated friction | `randomization_preset=calibrated_default` |
| `D1` | calibrated friend-flat DR at half range | `randomization_preset=calibrated_friend_flat`, `randomization_scale=0.5` |
| `D2` | calibrated friend-flat DR at full range | `randomization_preset=calibrated_friend_flat`, `randomization_scale=1.0` |

For all three collection branches:

```text
use_domain_randomization = true
use_push_randomization = true
use_observation_noise = true
joint_strength_scales = {}
broken_joint_names = []
```

For D1/D2, the same complete flat-task component list is mandatory:

```text
friction,mass_com,motor,delay,observation,push,initial_state
```

The collector's behavioral mixture noise is not the D factor. Existing
`random`, `noisy_expert`, `expert`, and `medium` proportions and their action
noise levels must be identical in D0, D1, and D2. Any additional
`env_action_noise_std`, action scale, bias, or delay outside the selected DR
preset must remain neutral in the first 3x3x4 experiment. This prevents
double-counting action noise and preserves a single interpretable D factor.

Each dataset must store both its exact command and the complete DR manifest.
Where the collection path exposes noisy actor observations and clean critic
state separately, the RWM transition target must use the established clean
state target rather than silently treating sensor corruption as physical
dynamics.

Physical properties in the friend source are sampled at environment creation.
The source relies on 8192 parallel environments, while the established RWM
collector uses 1024. To improve coverage without changing total dataset size,
each E/D dataset must be collected as four independently created shards (1024
environments per shard) and then merged with globally unique episode IDs. The
fixed vector-row schema cannot represent exactly 1,000,000 transitions because
that value is not divisible by 1024. The frozen counts are therefore
`249856, 249856, 249856, 250880`, totalling `1,000,448` transitions. This is
0.045% above the nominal target and produces 4096 independently sampled startup
domains while preserving trajectories long enough for the 32+8 sequence
window. The same protocol applies to D0, D1, and D2 for fairness.

### 3.3 Frozen-RWM interface factor P

Each of the nine datasets trains exactly one RWM. That RWM checkpoint is frozen
and reused by four final-policy branches. No branch may retrain or modify the
RWM after P is selected.

| ID | Action noise | Observation noise | Purpose |
| --- | ---: | ---: | --- |
| `P0` | off | off | clean final-policy control |
| `P1` | on | off | action-channel robustness |
| `P2` | off | on | sensor-channel robustness |
| `P3` | on | on | combined interface robustness |

The first full factorial intentionally uses small, simple noise. It does not mix
delay, persistent motor bias, action scaling, or low-pass filtering into P.
Those structured interface perturbations are reserved for a follow-up ablation
after the 36-policy screen.

Canonical P settings:

| Parameter | `P0` | `P1` | `P2` | `P3` |
| --- | ---: | ---: | ---: | ---: |
| `interface_action_noise_std` | `0.0` | `0.01` | `0.0` | `0.01` |
| `interface_action_bias_std` | `0.0` | `0.0` | `0.0` | `0.0` |
| action scale range | `[1.0, 1.0]` | `[1.0, 1.0]` | `[1.0, 1.0]` | `[1.0, 1.0]` |
| action delay steps | `[0, 0]` | `[0, 0]` | `[0, 0]` | `[0, 0]` |
| `interface_obs_noise_profile` | `none` | `none` | `deployment_small` | `deployment_small` |
| `interface_obs_noise_scale` | `0.0` | `0.0` | `0.25` | `0.25` |
| persistent joint-position bias | `[0.0, 0.0]` | `[0.0, 0.0]` | `[0.0, 0.0]` | `[0.0, 0.0]` |

At observation scale `0.25`, the intended per-step uniform ranges are:

| Actor feature | Noise range |
| --- | --- |
| base angular velocity | `[-0.0125, 0.0125]` |
| projected gravity | `[-0.0125, 0.0125]` |
| command | no noise |
| joint position | `[-0.0025, 0.0025] rad` |
| joint velocity | `[-0.01875, 0.01875] rad/s` |
| previous action | no independent noise |

`deployment_small` is deliberately separate from the environment-side friend
profile. The source friend observation noise contains joint-velocity noise up
to `+-1.5 rad/s`, which is appropriate to test in a real physics environment
but is too large for this first frozen-RWM interface ablation. `WorldModelEnv`
now implements this profile independently and its exact feature mapping has a
focused tensor test. The launcher records the resolved P settings before
starting training.

## 4. Frozen DR definitions

### 4.1 Calibrated MJLab default (`E0` and `D0`)

This branch keeps the current normal Go2 default COM, push, observation, and
legacy encoder-bias behavior, but replaces its low/wide friction interval with
the real-floor-calibrated interval:

| Component | Current default range |
| --- | --- |
| collision-geom friction | `[0.6, 1.0]`, center `0.8`, shared per environment |
| base COM offset | each axis `[-0.05, 0.05] m` |
| configured encoder-bias event | `[-0.015, 0.015] rad` |
| push interval | `[5, 6] s` |
| push linear velocity x/y | `[-0.5, 0.5] m/s` |
| push linear velocity z | `[-0.4, 0.4] m/s` |
| push angular roll/pitch | `[-0.52, 0.52] rad/s` |
| push angular yaw | `[-0.78, 0.78] rad/s` |
| actor base angular velocity noise | `[-0.2, 0.2]` |
| actor projected-gravity noise | `[-0.05, 0.05]` |
| actor joint-position noise | `[-0.01, 0.01] rad` |
| actor joint-velocity noise | `[-1.5, 1.5] rad/s` |

The current actor `joint_pos_rel` term uses `biased=False`, and the built-in PD
actuator also uses true joint position. Consequently, the legacy
`encoder_bias` event is configured but does not currently perturb either the
actor input or PD target. This ineffective legacy behavior is retained for E0/D0
so they remain a regression baseline; the manifest must label it as inactive.
The legacy preset also does not add friend-style base/link mass, Kp/Kd
multiplier, motor-strength, or actuator-delay randomization. A launcher must
save the resolved task config because these defaults can change with code.

### 4.2 Calibrated friend-flat full (`E2` and `D2`)

The friend-flat preset is based on
`wty-yy/go2_rl_gym@vanilla_train/legged_gym/envs/go2/go2_config.py` and mapped
to MJLab by `flash_rl/envs/mjlab.py`.

| Component | Full range |
| --- | --- |
| collision-geom friction | `[0.6, 1.0]`, one shared value per environment |
| base mass addition | `[-1.0, 1.0] kg`, with inertia recomputed/scaled consistently |
| non-base link mass/inertia scale | `[0.9, 1.1]` |
| base COM offset | each axis `[-0.03, 0.03] m` |
| Kp multiplier | `[0.9, 1.1]` |
| Kd multiplier | `[0.9, 1.1]` |
| motor target zero offset | `[-0.035, 0.035] rad` |
| motor strength | `[0.8, 1.2]` |
| actuator delay | `0..4` physics steps (`0..20 ms` at `0.005 s`) |
| push interval | `4 s` |
| push linear velocity x/y | `[-0.4, 0.4] m/s` |
| push linear velocity z | fixed at `0` |
| push angular velocity | each axis `[-0.6, 0.6] rad/s` |
| actor base angular velocity noise | `[-0.2, 0.2] rad/s` |
| actor projected-gravity noise | `[-0.05, 0.05]` |
| actor joint-position noise | `[-0.01, 0.01] rad` |
| actor joint-velocity noise | `[-1.5, 1.5] rad/s` |
| reset joint position | `q_default * Uniform(0.5, 1.5)` |
| reset base linear/angular velocity | each component `[-0.5, 0.5]` |

MuJoCo/MJLab has no direct coefficient-of-restitution field equivalent to the
source implementation. Restitution is intentionally not approximated through
an unvalidated contact parameter. The omission must appear in the manifest.

The friend source applies friction to all robot rigid collision shapes, not
only the four feet. The implemented MJLab mapping therefore samples one shared
coefficient per environment and applies it to every robot collision geom. Base
and non-base mass changes update inertia consistently rather than changing only
`body_mass`.

Motor zero offset is a persistent target-angle calibration error in the source:

```text
tau = Kp * (q_target + zero_offset - q) - Kd * dq
```

It is not ordinary observation noise. The friend branches implement it in the
actuator target/affine-bias path; the ineffective legacy `dr.encoder_bias`
behavior remains isolated to E0/D0 for regression compatibility.

### 4.3 Calibrated friend-flat half (`E1` and `D1`)

Half DR scales each range around its physical nominal center. It does not divide
both numeric endpoints by two.

| Component | Half range |
| --- | --- |
| collision-geom friction | `[0.7, 0.9]` |
| base mass addition | `[-0.5, 0.5] kg`, with consistent inertia update |
| non-base link mass/inertia scale | `[0.95, 1.05]` |
| base COM offset | each axis `[-0.015, 0.015] m` |
| Kp multiplier | `[0.95, 1.05]` |
| Kd multiplier | `[0.95, 1.05]` |
| motor target zero offset | `[-0.0175, 0.0175] rad` |
| motor strength | `[0.9, 1.1]` |
| actuator delay | `0..2` physics steps (`0..10 ms`) |
| push interval | `4 s` |
| push linear velocity x/y | `[-0.2, 0.2] m/s` |
| push angular velocity | each axis `[-0.3, 0.3] rad/s` |
| actor base angular velocity noise | `[-0.1, 0.1] rad/s` |
| actor projected-gravity noise | `[-0.025, 0.025]` |
| actor joint-position noise | `[-0.005, 0.005] rad` |
| actor joint-velocity noise | `[-0.75, 0.75] rad/s` |
| reset joint position | `q_default * Uniform(0.75, 1.25)` |
| reset base linear/angular velocity | each component `[-0.25, 0.25]` |

Friend parameters that represent probabilities or discrete schedules must be
resolved explicitly in the run manifest. They must not be changed merely to
make a failing branch look better.

### 4.4 Source-to-MJLab conversion audit

The following audit was performed against friend commit
`34c237ba7d34cc8ff4e23e2952e1f420aecd901b`, the current MJLab package, and the
current repository implementation.

| Source behavior | MJLab conversion decision | Readiness |
| --- | --- | --- |
| deployable 45-dim proprioceptive actor | registered normal-strength 45-dim actor/48-dim critic task without broken rewards | implemented and probed |
| friction shared across all robot rigid shapes, startup sampled | randomize all robot collision geoms with one shared per-environment value | implemented and probed |
| restitution `[0, 0.5]` | no direct MuJoCo coefficient; do not invent a `solref` proxy | intentionally omitted |
| base added mass with inertia recomputation | update base mass and inertia consistently while preserving COM semantics | implemented and probed |
| independent non-base link mass with inertia recomputation | physics-consistent per-link pseudo-inertia density scaling | implemented |
| base COM x/y/z offset | `body_com_offset` on `base_link`, startup sampled | implemented |
| per-joint Kp/Kd multiplier per episode | reset-time per-actuator gain/bias scaling | implemented |
| motor strength after torque clipping | scale gain/bias and force range by the same positive sample; mathematically equivalent to `strength * clip(raw_tau)` | implemented; parity test passed |
| motor target zero offset per episode | affine PD target offset, not encoder observation bias | implemented and probed |
| one per-environment delay sampled in `{0,5,10,15,20} ms` and shared by all joints | synchronized delayed position targets with lag `0..4` physics steps, refreshed at policy boundaries | implemented and probed at `dt=0.005`, decimation `4` |
| feature-specific actor observation noise | exact source ranges for friend full and centered half scaling | implemented and probed |
| velocity-overwrite push every 4 s | MJLab velocity-reset push with source x/y and angular ranges | implemented |
| joint reset `q_default * U(0.5,1.5)` | custom multiplicative reset, half uses `U(0.75,1.25)` | implemented and probed |
| base velocity reset `U(-0.5,0.5)` | set reset-event velocity ranges; half uses `[-0.25,0.25]` | implemented and probed |
| 8192 startup domains | four independent 1024-env collection shards, yielding 4096 startup domains | implemented and smoke-tested |
| rough-terrain curriculum | keep plane terrain | intentionally excluded |
| wide command curriculum | keep the common experiment command distribution | intentionally excluded |

The motor-strength conversion deserves an explicit test. For a positive strength
`m`, the current intended MuJoCo mapping is:

```text
clip(m * raw_tau, -m * limit, m * limit) = m * clip(raw_tau, -limit, limit)
```

This preserves the friend's post-clipping multiplication, including the fact
that `m > 1` can produce force above the nominal limit. It must not be silently
replaced by clipping again at the unscaled nominal limit.

Every conversion row above now has a focused check, and the resolved-config
probe confirms the expected sampling period, shape selection, and range. Future
behavioral changes require a new run ID and a repeated probe; an incomplete
mapping must be reported as a blocked experiment, not renamed "friend full".

## 5. Complete 36-policy matrix

Each row shares one dataset and one RWM. The four P branches under a row must
start from the exact same RWM checkpoint and use the exact same final-policy
seed during the single-seed screen.

| Expert | Dataset | RWM artifact | Final policies |
| --- | --- | --- | --- |
| `E0` | `D0` | `RWM_E0_D0` | `E0_D0_P0`, `E0_D0_P1`, `E0_D0_P2`, `E0_D0_P3` |
| `E0` | `D1` | `RWM_E0_D1` | `E0_D1_P0`, `E0_D1_P1`, `E0_D1_P2`, `E0_D1_P3` |
| `E0` | `D2` | `RWM_E0_D2` | `E0_D2_P0`, `E0_D2_P1`, `E0_D2_P2`, `E0_D2_P3` |
| `E1` | `D0` | `RWM_E1_D0` | `E1_D0_P0`, `E1_D0_P1`, `E1_D0_P2`, `E1_D0_P3` |
| `E1` | `D1` | `RWM_E1_D1` | `E1_D1_P0`, `E1_D1_P1`, `E1_D1_P2`, `E1_D1_P3` |
| `E1` | `D2` | `RWM_E1_D2` | `E1_D2_P0`, `E1_D2_P1`, `E1_D2_P2`, `E1_D2_P3` |
| `E2` | `D0` | `RWM_E2_D0` | `E2_D0_P0`, `E2_D0_P1`, `E2_D0_P2`, `E2_D0_P3` |
| `E2` | `D1` | `RWM_E2_D1` | `E2_D1_P0`, `E2_D1_P1`, `E2_D1_P2`, `E2_D1_P3` |
| `E2` | `D2` | `RWM_E2_D2` | `E2_D2_P0`, `E2_D2_P1`, `E2_D2_P2`, `E2_D2_P3` |

The complete workload is:

```text
3 expert training runs
9 dataset collection runs
9 offline RWM training runs
36 final-policy training runs
36 x evaluation-domain runs
```

It is incorrect to rerun the expert, dataset, or RWM separately for each P
branch. Artifact reuse is part of the experimental control.

## 6. Fixed controls

The following must remain identical unless a factor table explicitly changes
them:

- task and normal robot XML/model
- observation and action definitions
- expert architecture and optimizer
- expert training budget and checkpoint selection rule
- command distribution and standing-command probability
- collector mixture and dataset size
- dataset transition schema and feature normalization
- RWM architecture, ensemble size, horizon, optimizer, batch size, training
  budget, and checkpoint selection rule
- final-policy architecture, optimizer, replay settings, update ratio, reward,
  uncertainty setting already used by the established pipeline, and training
  budget
- logging cadence and checkpoint cadence
- evaluation seeds, commands, episode count, termination rules, and episode
  horizon

This experiment does not change the RWM algorithm. It does not add a latent DR
variable, context conditioning, a new uncertainty objective, or a different RL
algorithm.

## 7. Seed protocol

### 7.1 Screening pass

Run all 36 final policies with one controlled seed set:

```text
expert_seed = 0
collection_seed = 100
rwm_seed = 200
policy_seed = 300
evaluation_seeds = [400, 401, 402]
```

The launcher may derive branch-specific environment streams from the branch ID,
but the derivation must be deterministic and recorded. It must not select a
different favorable seed manually for a branch.

### 7.2 Confirmation pass

The single-seed grid is a screen, not final statistical evidence. Select the top
3 to 5 configurations using the preregistered evaluation rule and repeat their
complete affected stochastic stages with at least three seeds. Report mean,
standard deviation, median, and worst-seed performance.

## 8. Artifact layout and provenance

Required layout:

```text
logs/experiments/go2_normal_dr_3x3x4/<RUN_ID>/
  experiment_manifest.json
  source_snapshot/
  experts/
    E0_default/
    E1_friend_half/
    E2_friend_full/
  datasets/
    E0/D0/dataset.pt
    E0/D1/dataset.pt
    ...
    E2/D2/dataset.pt
  rwms/
    E0/D0/<run>/
    ...
    E2/D2/<run>/
  policies/
    E0/D0/P0/<run>/
    ...
    E2/D2/P3/<run>/
  evaluations/
    clean/
    default_dr/
    friend_half/
    friend_full/
    holdout_dr/
  queues/
  summaries/
```

Every stage directory must contain:

- `command.sh`: exact executable command
- `resolved_config.yaml` or JSON equivalent
- `state.txt`: `pending`, `running`, `completed`, `failed`, or `blocked`
- `run.log`
- `summary.json`
- upstream artifact paths and SHA256 hashes
- current git commit and `git diff --stat`
- CUDA device, Python, Torch, MJLab, MuJoCo, and driver versions
- start/end timestamps and elapsed time

Never choose an artifact by an ambiguous "latest directory" rule. The launcher
must pass explicit paths and verify hashes before reuse.

The merged dataset must additionally preserve DR provenance without repeating
episode-constant tensors on every transition unnecessarily:

```text
startup_domain_table (one row per shard/environment)
  friction
  base mass delta and realized mass/inertia scale
  per-link mass/inertia scales
  base COM offset

episode_domain_table (one row per episode)
  episode_id and startup_domain_id
  Kp/Kd multipliers
  motor strengths
  motor target zero offsets
  reset joint multiplier
  reset base velocity

transition interface fields
  raw policy action
  effective delayed action
  sampled action-delay substep
  clean RWM observation/state and clean next state
  noisy actor observation seen by the expert
  command, reward, termination reason, collector type
```

These fields are analysis metadata and are not automatically appended to
the RWM input. Their purpose is to measure per-domain model error, verify DR
coverage, and diagnose whether robustness failures come from data quality or
model averaging.

## 9. Stage gates

### 9.1 Expert gate

A completed expert must:

- contain a loadable policy checkpoint and resolved config
- contain no NaN/Inf metric or parameter
- run a clean smoke evaluation without runtime errors
- report fall rate, episode length, command tracking, orientation, action
  saturation, and checkpoint step

Poor task performance is an experimental result and should be marked with a
warning, not silently hidden. A branch is hard-blocked only for invalid artifacts,
NaN/Inf, load failure, or failure to produce usable trajectories.

### 9.2 Dataset gate

Each dataset must verify:

- exact resolved vector-row count (`1,000,448` for the nominal 1024-env run)
- finite observations, actions, rewards, next observations, and terminals
- expected feature dimensions and observation order
- command-mode coverage and collector-source proportions
- action min/max/mean/std and clipping fraction
- terminal/fall fraction and episode-length distribution
- DR manifest and effective ranges
- startup-domain and episode-domain table integrity
- empirical min/max/quantiles and unique-domain coverage for every DR field
- raw-policy-action versus effective-action statistics and delay histogram
- clean-state versus noisy-policy-observation statistics
- no accidental 0.5 joint strength or mask

Dataset quality summaries must be produced before RWM training starts.

### 9.3 RWM gate

Each RWM must verify:

- loadable checkpoint and complete training log
- finite train/held-out losses
- one-step and multi-step rollout errors at horizons `1`, `5`, and `20`
- per-feature errors for base velocity, angular velocity, projected gravity,
  joint position, and joint velocity
- episode-level train/held-out splitting, with zero episode overlap

The same held-out split rule must be used for all nine datasets.

### 9.4 Final-policy gate

Each policy must verify:

- loadable actor checkpoint
- finite losses, rewards, and action outputs
- the exact expected P manifest
- no RWM weight update after the frozen checkpoint is loaded
- the same RWM SHA256 for P0 through P3 under one E/D row

## 10. Common evaluation suite

Every final policy is evaluated under exactly the same five domains:

| Evaluation ID | Environment |
| --- | --- |
| `V0` | clean normal MJLab |
| `V1` | calibrated default DR |
| `V2` | calibrated friend-flat half DR |
| `V3` | calibrated friend-flat full DR |
| `V4` | preregistered holdout DR outside or recombined beyond training support |

`V4` must be specified before inspecting the 36-policy ranking. It must remain
physically plausible and must not introduce broken joints or RR calf 0.5.

Use a fixed command schedule containing:

- zero-command standing
- forward and backward velocity
- lateral velocity in both directions
- yaw in both directions
- mixed linear/yaw commands

Primary metrics:

- fall-free success rate
- episode length
- linear/yaw command tracking MAE
- zero-command drift
- roll/pitch RMS and 95th percentile
- action RMS, action rate, and clipping/saturation fraction
- joint-position tracking error
- joint velocity and estimated/available torque statistics

Selection is lexicographic:

1. Pass the clean-domain viability gate.
2. Maximize the minimum fall-free success rate across V0-V4.
3. Minimize worst-domain tracking error.
4. Prefer lower orientation excursion and smoother actions.
5. Use mean reward only as a supporting metric, not the sole rank criterion.

After MJLab ranking, only shortlisted policies proceed to the common MuJoCo
closed-loop test. Real-robot testing starts with zero command and low speed,
with an immediate Passive transition available.

## 11. Analysis plan

The screening report must show:

- all 36 rows, including failed or blocked rows
- E, D, and P main-effect summaries
- E x D, E x P, D x P, and E x D x P interaction plots
- per-domain and worst-domain rankings
- artifact provenance and any deviations

Do not claim statistical significance from the one-seed screen. After the
confirmation pass, use a factorial regression or mixed-effects analysis with
seed and evaluation domain represented appropriately. Report effect sizes and
confidence intervals in addition to p-values.

## 12. Execution order

1. Implement a normal-strength config and deploy template.
2. Fix and test all `fix required` rows in the conversion audit.
3. Implement `deployment_small` independently from source friend observation noise.
4. Implement four-shard collection, globally unique episode IDs, and DR metadata tables.
5. Implement the E/D/P presets and manifest assertions.
6. Implement a dry-run launcher that generates all commands but starts no job.
7. Run syntax/import/config-resolution and DR-distribution tests.
8. Run short environment-only DR probes for default, friend half, and friend full.
9. Run a shortened end-to-end `E0_D0_P0` smoke test.
10. Run shortened `E1_D1_P3` and `E2_D2_P3` mapping smoke tests.
11. Run and evaluate E0, E1, and E2 experts.
12. Collect and validate the nine datasets.
13. Train and validate the nine RWMs.
14. Fan out four final policies from each frozen RWM.
15. Run V0-V4 evaluation for all completed policies.
16. Produce the single-seed screening report.
17. Repeat the selected configurations with at least three seeds.
18. Export shortlisted policies as ONNX IR 9/opset 18 and run MuJoCo checks.
19. Proceed to staged real-robot validation only after deployment checks pass.

The full launcher must run inside detached tmux sessions, assign GPUs explicitly,
write queue events, and resume only from completed validated artifacts. A failed
stage must mark downstream dependent stages `blocked`; it must not silently
retry with altered parameters.

## 13. Implemented launcher contract

The launcher implements the following interface and defaults to dry-run:

```bash
cd /root/unitree_rl_mjlab_model_based_broken

RUN_ID=normal_go2_dr_3x3x4_seed0_<timestamp> \
GPU_POOL="0 1 2 3 4 5 6 7" \
DRY_RUN=true \
CONFIRM_RUN=false \
bash scripts/reinforcement_learning/go2_normal_proprioceptive/launch_normal_go2_dr_3x3x4.sh
```

This generates 57 training jobs. Set `INCLUDE_EVALUATIONS=true` to append one
four-domain evaluation job per policy, for 93 jobs total. The V4 holdout must be
preregistered before final ranking and is intentionally not invented by the
launcher.

An actual run must require both:

```text
DRY_RUN=false
CONFIRM_RUN=true
```

The launcher must refuse to start when the normal-strength assertions, source
snapshot, output-root uniqueness, or GPU checks fail.

## 14. Monitoring contract

The launcher supports monitoring without attaching to tmux:

```bash
tmux ls

find logs/experiments/go2_normal_dr_3x3x4/<RUN_ID> \
  -name state.txt -print -exec tail -n 8 {} \;

find logs/experiments/go2_normal_dr_3x3x4/<RUN_ID>/queues \
  -name '*.log' -print -exec tail -n 30 {} \;

nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
  --format=csv
```

## 15. Change-control rule

Once the first full expert starts, this document and all resolved factor values
are frozen for that `RUN_ID`. Bug fixes that alter behavior require a new
`RUN_ID`; the old run remains available for audit. Formatting, monitoring, or
logging-only fixes may continue only when they provably do not change training
or evaluation behavior.

## 16. Verification record

Completed on 2026-07-12 in the 4090 container:

- v2 FixStand invariant probe: actor/critic `45/48`, default pose exactly
  `[0,0.8,-1.5]` per leg, calibrated full friction
  sampled inside `[0.6,1.0]`, shared delay `0..4`, and all motor/mass checks passed
- v2 clean zero-command stance: 16 environments for 100 steps, zero
  terminations, mean base height `0.31765 m`, and mean absolute joint tracking
  error `0.03842 rad`; the height reward target was consequently fixed at `0.32 m`
- v2 full 57-job dry-run:
  `normal_go2_fixstand_v2_dryrun_20260712_v1`; manifest contains 3 experts,
  9 datasets, 9 RWMs, and 36 policies with no training process started
- v2 shortened end-to-end run:
  `normal_go2_fixstand_v2_smoke_E0D0P0_20260712_v1`; expert, four-shard
  dataset, RWM, and final policy completed with `completed=4, failed=0, blocked=0`
- v2 four-domain evaluation entry completed on the smoke policy, including
  negative backward/lateral/yaw commands after fixing repeatable CLI parsing

Historical v1 plumbing checks retained for audit:

- full 57-job dry-run: 3 expert, 9 dataset, 9 RWM, and 36 policy jobs; no
  training process was started
- friend-full 8-env physics probe: actor/critic `45/48`, all collision geom
  friction shared per environment, normal mass/inertia and motor fields finite,
  synchronized delay sampled in `0..4`, and three consecutive steps finite
- collector v2 probe: raw/effective action, noisy actor observation, delay, and
  startup/episode DR tables written with no episode/timestep discontinuity
- four-shard merge probe: globally unique startup/episode IDs and no sampled
  sequence crossing a shard or episode boundary
- episode-level split probe: 28 training episodes, 7 validation episodes, zero
  overlap in the shortened dataset
- shortened `E2_D2_P3` run:
  `normal_go2_dr_smoke_E2D2P3_20260712_v3`, all four stages completed after a
  tested failed-policy-only resume; this is a plumbing test, not a performance result
- shortened E0/E1 parallel run:
  `normal_go2_dr_smoke_E0E1_20260712_v1`; `E0_D0_P0` and `E1_D1_P3` both
  completed. The initial E1 shard merge exposed a table-only terminal episode
  ID collision; the merge offset was fixed and the same RUN_ID resumed by
  reusing all four validated shards, then completed dataset, RWM, and policy
  stages with final state `completed=8, failed=0, blocked=0`
- historical normal deployment export probe: ONNX `1x45 -> 1x12`, IR 9,
  opset 18, and maximum PyTorch/ONNX parity error about `7.34e-7`; v2 export
  must additionally use FixStand pose and `60/80/80` gains
- friend-full real-MJLab evaluation entry loaded and stepped the exported policy;
  the play-only `randomize_terrain` event was then removed to keep evaluation flat

The full 36-policy run has not been launched. Its results must not be inferred
from the shortened smoke artifacts.
