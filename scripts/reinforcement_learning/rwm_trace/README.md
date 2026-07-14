# Go2 RWM + TRACE integration

This package ports TRACE's imperfect-simulator trajectory scoring into the
existing Go2 RWM-U/FlashSAC pipeline. It does not replace the ensemble world
model or its epistemic uncertainty penalty.

## Data flow

1. Generate short trajectories in a deliberately mismatched MJLab simulator.
   Candidate datasets use the normal Go2 dataset tensor schema and may include
   `trace_start_state_ids` to identify multiple rollouts reconstructed from the
   same offline state.
2. Summarize candidates and build Go2-specific LLM pair prompts.
3. Train a Bradley-Terry scorer from confidence-filtered pair labels.
4. Rank all candidates, retain a fixed top ratio, and convert selected windows
   to the same n-step targets used by FlashSAC.
5. Enable `trace` in the RWM policy config. A fixed fraction of every critic
   batch is then replaced by selected simulator replay. Actor, critic, reward,
   temperature, and RWM uncertainty losses remain unchanged.

## Commands

Collect 20-step candidates reconstructed from offline states. The normal
collector's DR/gap flags define the imperfect simulator. Keep `collector_mix`
expert-only for the primary comparison:

```bash
uv run python scripts/reinforcement_learning/rwm_dataset/collect_go2_expert_command_coverage_dataset.py \
  --task Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert \
  --expert_policy_path <EXPERT_CHECKPOINT> \
  --collector_mix expert:1.0 \
  --trace_reset_dataset <OFFLINE_DATASET.pt> \
  --trace_rollout_length 20 --trace_trajectories_per_state 4 \
  --trace_action_temperature 1.0 \
  --num_envs 4096 --num_transitions 1000000 \
  --randomization_preset <MISMATCH_PRESET> \
  --save_path logs/trace/imperfect_sim_candidates.pt
```

Here `4096 = 1024 starting states x 4 trajectories per state`. Lower
`num_envs` proportionally reduces the number of distinct starting states.
`trace_action_temperature` scales the expert policy's sampling noise only in
these imperfect-simulator candidate rollouts. Ordinary dataset collection and
evaluation remain deterministic. Record and tune this value independently of
the SAC entropy coefficient and environment action-interface noise.

Create summaries before the first scorer exists:

```bash
uv run python scripts/reinforcement_learning/rwm_trace/build_trace_replay.py \
  --candidate_dataset logs/trace/imperfect_sim_candidates.pt \
  --summaries_only --trajectory_length 20 --trajectory_stride 20 \
  --output logs/trace/candidates.pt
```

Build an initial pair set from candidate summaries:

```bash
uv run python scripts/reinforcement_learning/rwm_trace/build_go2_feedback_pairs.py \
  --summaries logs/trace/candidates.summaries.jsonl \
  --output logs/trace/pairs.jsonl \
  --budget 200 --cross_start_fraction 0.2
```

After labeling the prompts, train the scorer:

```bash
uv run python scripts/reinforcement_learning/rwm_trace/train_go2_trace_scorer.py \
  --labels logs/trace/labels.jsonl \
  --pairs logs/trace/pairs.jsonl \
  --output logs/trace/go2_trace_scorer.pt
```

Build selected n-step replay. `n_step` and `gamma` must match the policy config:

```bash
uv run python scripts/reinforcement_learning/rwm_trace/build_trace_replay.py \
  --candidate_dataset logs/trace/imperfect_sim_candidates.pt \
  --scorer_checkpoint logs/trace/go2_trace_scorer.pt \
  --selection scorer --select_ratio 0.10 \
  --trajectory_length 20 --trajectory_stride 20 \
  --n_step 3 --gamma 0.99 \
  --reward_source rwm_aligned \
  --output logs/trace/selected_replay.pt
```

`rwm_aligned` is the default and is required for the primary comparison. It
recomputes candidate rewards from physical simulator states/actions/contacts
with the same tensor reward used by the frozen-RWM environment. Raw MJLab
collector reward remains only as `dataset_return` diagnostics.

Start the unchanged RWM-U trainer with TRACE replay enabled:

```bash
uv run python scripts/reinforcement_learning/rwm_flashsac/train_flashsac_world_model_go2_proprioceptive.py \
  --config_path scripts/reinforcement_learning/rwm_flashsac/configs/go2_flashsac_rwm_proprioceptive.yaml \
  --overrides trace.enabled=true \
  --overrides trace.replay_path=logs/trace/selected_replay.pt \
  --overrides trace.replay_ratio=0.10
```

For a feedback refresh, collect candidates with the latest saved policy,
rebuild the scorer/replay artifact, then resume rather than restarting:

```bash
uv run python scripts/reinforcement_learning/rwm_flashsac/train_flashsac_world_model_go2_proprioceptive.py \
  --policy_resume_path <LATEST_STEP_DIR> \
  --overrides trace.enabled=true \
  --overrides trace.replay_path=logs/trace/selected_replay_refresh.pt
```

## Reset limitation

`reset_go2_from_rwm_state` reconstructs roll/pitch, body velocities, joint
position, and joint velocity from the 45D state. Root x/y/z/yaw and actuator
force cannot be recovered from that state. Candidate generation must record
the reset reconstruction error and must not describe this reset as exact.
