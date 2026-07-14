# Deployable Go2 Gap Expert 2x3 Screens

Both screens train 45D deployable actors with 48D privileged critics.

## Payload screen

- G0: fixed 0 kg payload.
- G1: per-environment back-mounted payload sampled uniformly from 5 to 10 kg.
- D0/D1/D2: calibrated default, friend-half, and friend-full DR.

## RR calf screen

- G0: RR calf strength fixed at 1.0.
- G1: per-environment RR calf motor strength sampled uniformly from 0.5 to 1.0.
- The multiplier affects PD torque and force limits; it does not mask or scale the action.
- D0/D1/D2 use the same three DR settings as the payload screen.

Generate commands without training:

```bash
GAP_KIND=rr_calf DRY_RUN=true bash scripts/reinforcement_learning/go2_normal_proprioceptive/prepare_gap_expert_2x3.sh
GAP_KIND=payload DRY_RUN=true bash scripts/reinforcement_learning/go2_normal_proprioceptive/prepare_gap_expert_2x3.sh
```

The payload is currently represented as a rigid box centered at `(0, 0, 0.10) m`
in the `base_link` frame, with size `0.20 x 0.12 x 0.05 m`. Its mass, combined
center of mass, and inertia are applied using the parallel-axis theorem. Before
the final robot experiment, replace the nominal center and dimensions with the
measured mounting geometry.

After training, generate the preregistered cross-evaluation commands:

```bash
GAP_KIND=rr_calf EXPERIMENT_ROOT=/path/to/completed/run \
  bash scripts/reinforcement_learning/go2_normal_proprioceptive/prepare_gap_expert_cross_eval.sh

GAP_KIND=payload EXPERIMENT_ROOT=/path/to/completed/run \
  bash scripts/reinforcement_learning/go2_normal_proprioceptive/prepare_gap_expert_cross_eval.sh
```

The RR calf screen evaluates strengths 1.0, 0.75, and 0.5. The payload screen evaluates 0, 5, 7.5, and 10 kg.

## Launch all 12 experts on the 3090 server

The launcher uses GPUs 1, 2, 3, and 7. Each GPU runs three jobs serially and
stops before starting a new job if `/data1` has less than 40 GB available.

```bash
bash scripts/reinforcement_learning/go2_normal_proprioceptive/launch_gap_experts_12.sh
```
