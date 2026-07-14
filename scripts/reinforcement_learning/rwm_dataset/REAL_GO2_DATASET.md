# Real Go2 CSV to RWM Dataset

The formal real-robot dataset path uses the same
`go2_mixed_rwm_dataset_v2` container as simulation. It keeps the 45D state
interface but sets `base_lin_vel_b[0:3]` to zero placeholders. Training must
use `system_dynamics.state_loss_ignored_indices=[0,1,2]`.

## 1. Build the updated deployment logger

The controller must log `policy_obs_0..44` and either
`effective_policy_action_0..11` or `raw_policy_action_0..11`.

```bash
cd /root/unitree_cpp_deploy/deploy/robots/go2
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

Use `logging_dt: 0.02` for a 50 Hz policy. Re-run MuJoCo before collecting on
hardware. After collection, verify the header:

```bash
head -1 /path/to/run_data.csv | tr ',' '\n' | grep -E \
  '^(policy_obs_0|policy_obs_44|effective_policy_action_0|effective_policy_action_11|raw_policy_action_0|raw_policy_action_11|episode_step|unsafe_orientation)$'
```

## 2. Convert one or more trajectories

Repeat `--csv` for every trajectory. Each CSV is treated as a separate
episode boundary.

```bash
cd /root/unitree_rl_mjlab
uv run python scripts/reinforcement_learning/rwm_dataset/convert_go2_real_csv_to_dataset.py \
  --csv /path/to/run_001.csv \
  --csv /path/to/run_002.csv \
  --deploy-yaml /path/to/the/exported/policy/params/deploy.yaml \
  --output logs/rwm_datasets/go2_real_proprioceptive/dataset.pt \
  --minimum-transitions 100000
```

The command also writes `dataset.json`. Formal data must report:

- `conversion_kind: exact`
- `state_dim: 45`, `action_dim: 12`, `contact_dim: 4`
- `base_lin_vel_source: zero_placeholder_unsupervised`
- a small `max_next_obs_action_alignment_error` (default limit `1e-3`)
- no unexpected rejected rows

Do not use `--allow-legacy-reconstruction` for a formal experiment. That mode
exists only to check old CSV files and marks the result `approximate_legacy`.

## 3. Train the RWM

Point the offline world-model config at the generated `dataset.pt` and keep:

```yaml
system_dynamics:
  state_loss_ignored_indices: [0, 1, 2]
```

The converter executes the existing 32-step-history plus 8-step-forecast
sampler before saving, so a successful strict conversion is directly readable
by the current offline RWM trainer.
