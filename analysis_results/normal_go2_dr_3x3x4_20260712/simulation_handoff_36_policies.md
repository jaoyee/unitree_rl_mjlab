# Normal Go2 3x3x4 Policy Simulation Handoff

## 1. Experiment Identity

- Run ID: `normal_go2_fixstand_v2_moderate_seed0_20260712`
- Task: `Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert`
- Robot: normal Go2; no 0.5-strength joint, broken joint, action mask, or observation mask
- Matrix: 3 expert DR settings x 3 dataset DR settings x 4 RWM-policy interface settings
- Final policies: 36
- Expert training: 100,000,000 environment steps, 1,024 environments
- Dataset: at least 1,000,000 transitions per E/D pair, collected as four shards
- RWM: 5,000 offline iterations per E/D pair
- Final policy: 50,000,000 imagination environment steps, 1,024 imagination environments
- Seeds: dataset `100`, RWM `200`, final policy `300`, evaluation `400`
- Current screen has one seed per condition. Rankings are screening results, not statistical confidence claims.

## 2. Common Robot and Task Configuration

All 36 policies share the following configuration:

| Item | Value |
|---|---|
| Default joint pose per leg | hip `0.0`, thigh `0.8`, calf `-1.5` rad |
| Stiffness per leg | hip `30`, thigh `40`, calf `50` |
| Damping per leg | hip `1.5`, thigh `2.0`, calf `2.5` |
| Target base height | `0.32 m` |
| Calibrated friction center | `0.8` |
| Full calibrated friction range | `[0.6, 1.0]` |
| Expert/collection command x | `[-0.8, 0.8] m/s` |
| Expert/collection command y | `[-0.3, 0.3] m/s` |
| Expert/collection yaw | `[-0.6, 0.6] rad/s` |
| Final imagination command x | `[-0.5, 0.5] m/s` |
| Final imagination command y | `[-0.25, 0.25] m/s` |
| Final imagination yaw | `[-0.5, 0.5] rad/s` |

The evaluated actor uses a 48-dimensional observation in this order:

1. `base_lin_vel` (3)
2. `base_ang_vel` (3)
3. `projected_gravity` (3)
4. command (3)
5. joint position (12)
6. joint velocity (12)
7. previous action (12)

The action is a 12-dimensional joint-position target. A simulator adapter must preserve this exact order, scaling, default pose, action scale, and control period. These policies must not be loaded with the old broken-joint or 0.5-strength adapter.

## 3. Factor Definitions

### Expert DR

| ID | Setting |
|---|---|
| `E0` | `calibrated_default`, scale 1.0 |
| `E1` | `calibrated_friend_flat`, scale 0.5 |
| `E2` | `calibrated_friend_flat`, scale 1.0 |

### Dataset Collection DR

| ID | Setting |
|---|---|
| `D0` | `calibrated_default`, scale 1.0 |
| `D1` | `calibrated_friend_flat`, scale 0.5 |
| `D2` | `calibrated_friend_flat`, scale 1.0 |

`calibrated_default` keeps the task's lightweight DR and calibrates shared robot collision friction around 0.8. `calibrated_friend_flat` adds the selected flat-ground friend components: friction, mass/COM, motor parameters, delay, observation corruption, initial-state randomization, and pushes. At full scale it includes base mass +/-1 kg, non-base link mass/inertia about 0.9-1.1x, stiffness/damping 0.9-1.1x, motor strength 0.8-1.2x, motor zero offset +/-0.035 rad, synchronized position delay up to one policy step, and periodic pushes. Half scale contracts these ranges toward nominal values.

### Frozen RWM to Final Policy Interface

| ID | Action noise | Observation noise |
|---|---:|---|
| `P0` | 0 | none |
| `P1` | Gaussian std `0.01` | none |
| `P2` | 0 | `deployment_small` at scale `0.25` |
| `P3` | Gaussian std `0.01` | `deployment_small` at scale `0.25` |

Before the 0.25 multiplier, `deployment_small` uses uniform half-ranges of 0.05 for angular velocity, 0.05 for projected gravity, 0.01 for joint position, and 0.075 for joint velocity. It does not add mass, COM, friction, or motor randomization inside the frozen RWM.

## 4. Evaluation Protocol

Each policy was evaluated in real MJLab physics with 256 environments for 2,400 steps. Commands changed every 300 steps:

1. stand: `(0, 0, 0)`
2. forward: `(0.5, 0, 0)`
3. backward: `(-0.4, 0, 0)`
4. left: `(0, 0.25, 0)`
5. right: `(0, -0.25, 0)`
6. positive yaw: `(0, 0, 0.5)`
7. negative yaw: `(0, 0, -0.5)`
8. combined: `(0.4, 0.2, 0.3)`

Evaluation domains:

- `V0`: clean
- `V1`: calibrated default DR
- `V2`: calibrated friend-flat at 0.5 scale
- `V3`: calibrated friend-flat at 1.0 scale

The robust score is a 0-100 screening score: 35% mean V1-V3 survival, 25% worst-domain survival, 20% domain-normalized return, and 20% tracking quality. It is not an environment reward.

## 5. Complete Policy Ranking and Rewards

|Rank|Policy|E|D|P|Score|V0 reward|V1 reward|V2 reward|V3 reward|Mean DR reward|
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
|1|E0D0P2|default|default|obs|70.35|142.53|124.03|141.30|55.94|107.09|
|2|E0D0P3|default|default|action+obs|68.79|142.24|141.31|141.13|47.13|109.86|
|3|E1D1P2|half|half|obs|57.51|144.79|46.60|143.52|48.84|79.65|
|4|E2D1P0|full|half|none|54.61|145.82|129.74|144.86|9.88|94.83|
|5|E0D0P0|default|default|none|53.65|142.50|141.78|141.72|6.32|96.61|
|6|E2D0P2|full|default|obs|51.99|144.99|144.20|143.86|0.98|96.35|
|7|E1D0P2|half|default|obs|51.20|141.00|76.60|139.82|21.07|79.16|
|8|E1D1P3|half|half|action+obs|51.00|145.16|31.14|144.28|42.14|72.52|
|9|E1D0P0|half|default|none|50.93|143.02|142.02|141.50|-0.96|94.18|
|10|E0D0P1|default|default|action|50.29|143.22|142.35|142.29|-3.37|93.75|
|11|E1D0P1|half|default|action|49.37|141.95|141.22|140.83|-4.20|92.62|
|12|E0D1P2|default|half|obs|49.16|144.80|41.22|143.41|26.73|70.45|
|13|E1D1P0|half|half|none|46.14|144.87|54.71|143.61|13.46|70.59|
|14|E0D2P2|default|full|obs|45.23|140.92|42.62|140.18|18.92|67.24|
|15|E2D1P1|full|half|action|43.81|145.43|31.49|144.73|14.06|63.43|
|16|E1D2P1|half|full|action|43.14|138.68|21.57|138.05|26.13|61.92|
|17|E2D2P1|full|full|action|40.04|143.69|13.75|143.31|14.05|57.04|
|18|E0D2P1|default|full|action|39.42|31.45|25.56|74.91|35.98|45.48|
|19|E2D2P0|full|full|none|39.04|142.87|11.89|142.21|11.99|55.36|
|20|E1D0P3|half|default|action+obs|38.74|141.91|15.04|140.73|8.94|54.90|
|21|E0D1P0|default|half|none|38.12|144.60|16.11|143.39|4.88|54.79|
|22|E1D2P0|half|full|none|37.33|139.83|27.06|138.82|5.09|56.99|
|23|E0D1P1|default|half|action|29.28|144.86|37.17|19.77|16.54|24.49|
|24|E0D1P3|default|half|action+obs|29.08|144.81|31.81|15.68|23.27|23.59|
|25|E1D1P1|half|half|action|28.30|145.01|42.33|9.77|14.49|22.20|
|26|E0D2P3|default|full|action+obs|27.14|13.98|27.84|21.63|17.87|22.45|
|27|E0D2P0|default|full|none|25.35|31.30|18.19|25.68|11.76|18.55|
|28|E2D1P2|full|half|obs|24.59|145.50|17.45|19.28|11.25|15.99|
|29|E2D0P1|full|default|action|24.26|16.07|43.09|13.03|5.57|20.56|
|30|E2D2P2|full|full|obs|22.69|143.18|11.18|20.35|9.12|13.55|
|31|E2D1P3|full|half|action+obs|21.04|145.27|9.44|14.02|6.24|9.90|
|32|E2D2P3|full|full|action+obs|20.42|15.43|9.45|11.82|8.02|9.76|
|33|E2D0P0|full|default|none|20.36|145.13|15.09|10.45|5.48|10.34|
|34|E2D0P3|full|default|action+obs|20.27|145.16|6.33|16.63|5.55|9.50|
|35|E1D2P3|half|full|action+obs|19.61|34.31|14.15|20.14|13.89|16.06|
|36|E1D2P2|half|full|obs|12.96|7.98|7.64|7.96|8.06|7.88|

## 6. Analysis

The clean-domain rewards for most policies are around 140-146, showing that the policy architecture and normal-Go2 task can learn stable behavior. The major separation appears under randomized evaluation, especially V3.

The best screen is `E0D0P2`: default expert DR, default collection DR, and small observation noise during frozen-RWM policy training. `E0D0P3` has a slightly higher mean DR reward, but lower worst-domain survival than `E0D0P2`. Observation noise is therefore useful, while the added action noise is not consistently beneficial.

At the factor level, default expert DR scores higher than friend-half and friend-full. Default dataset DR is also clearly strongest. Friend-full dataset collection broadens the transition distribution beyond what the current unconditioned RWM models reliably. D2 models show higher autoregressive error and weaker epistemic-error correlation, and their final policies are generally poor.

The result does not mean physical DR is inherently harmful. It means the current combination of broad randomized data, one unconditioned RWM, fixed model capacity/training budget, and final imagination training does not preserve that diversity well. The V3 results also show that none of the 36 policies is yet a mature real-robot candidate: even `E0D0P2` reaches only about 44.5% worst-domain survival in the automated screen.

## 7. Recommended Simulation Order

Do not begin by manually viewing all 36 policies. Use this order:

1. `E0D0P2`: primary candidate
2. `E0D0P3`: tests whether action noise improves motion despite lower worst-domain survival
3. `E1D1P2`: best matched half-DR pipeline
4. `E0D0P0`: clean baseline without interface noise
5. `E0D0P1`: isolates action noise
6. `E0D2P2`: checks the effect of full dataset DR with the same E0/P2 endpoints
7. `E2D0P2`: checks the effect of full expert DR with D0/P2

For each policy, first stand for at least 10 seconds, then test forward, backward, lateral, yaw, and combined commands. Record falls, illegal contacts, base height, body pitch/roll, foot slip, contact duty, action saturation, joint tracking error, and recovery after command switching. Use identical simulator friction, gains, action scale, control frequency, initial pose, and command script for every policy.

## 8. Checkpoint Layout

On the 3090 server, each final checkpoint is located at:

```text
/data1/xjy/unitree_rl_mjlab_normal_fixstand_v2/logs/experiments/
  go2_normal_dr_3x3x4/normal_go2_fixstand_v2_moderate_seed0_20260712/
  policies/EX/DY/PZ/run/step48828
```

Replace `EX`, `DY`, and `PZ` with the policy ID components. For example:

```text
policies/E0/D0/P2/run/step48828
```

Transfer the checkpoint together with its `rwm_flashsac_config.yaml`, actor state, normalization state, and the matching simulator/deployment configuration. Do not transfer only an anonymous `actor.pt` without its E/D/P identity and configuration.

## 9. Files to Keep with the Simulation Results

- `experiment_manifest.json`
- `policy_ranking.csv`
- `factor_summary.csv`
- `evaluation_report.md`
- this handoff document
- each tested checkpoint directory
- simulator configuration and commit hash
- raw trajectory logs from each visualization/simulation
