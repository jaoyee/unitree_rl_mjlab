#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/root/unitree_rl_mjlab_normal_fixstand_v2
cd "${REPO_ROOT}"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
SMOKE_GAP_ID="${SMOKE_GAP_ID:-rr05}"
SMOKE_RR_STRENGTH="${SMOKE_RR_STRENGTH:-0.5}"
SMOKE_FAILURE_NOISE="${SMOKE_FAILURE_NOISE:-0.0}"
SMOKE_ROLLOUT_LENGTH="${SMOKE_ROLLOUT_LENGTH:-200}"
SMOKE_COLLECTOR_MIX="${SMOKE_COLLECTOR_MIX:-expert:0.80,random:0.20}"
SMOKE_LABEL="${SMOKE_LABEL:-${SMOKE_GAP_ID}_n${SMOKE_FAILURE_NOISE}_h${SMOKE_ROLLOUT_LENGTH}}"
ROOT="${REPO_ROOT}/logs/experiments/go2_v3_t2_long_failure/smoke_${SMOKE_LABEL}"
DATASET="${REPO_ROOT}/logs/transfer_real_trace_inputs/${SMOKE_GAP_ID}/dataset.pt"
EXPERT="${REPO_ROOT}/logs/experiments/go2_gap_experts/selected_g0d0/step97656"
CANDIDATES="${ROOT}/candidates_t2_16x4x${SMOKE_ROLLOUT_LENGTH}.pt"
REPLAY="${ROOT}/random_replay.pt"
mkdir -p "${ROOT}"

if [[ ! -f "${CANDIDATES}" ]]; then
  "${PYTHON_BIN}" scripts/reinforcement_learning/rwm_dataset/collect_go2_expert_command_coverage_dataset.py \
    --task Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert \
    --device cuda:0 --seed 142 --num_envs 64 --num_transitions "$((64 * SMOKE_ROLLOUT_LENGTH))" \
    --save_path "${CANDIDATES}" --expert_policy_path "${EXPERT}" \
    --collector_mix "${SMOKE_COLLECTOR_MIX}" \
    --fixed_collector_assignment --collector_assignment_seed 142 \
    --dataset_obs_kind proprioceptive --action_noise_std 0.0 \
    --failure_action_noise_std "${SMOKE_FAILURE_NOISE}" \
    --randomization_preset calibrated_default \
    --randomization_components friction,mass_com,motor,delay,observation,push,initial_state \
    --randomization_scale 1.0 \
    --payload_mass_range_kg 0.0 0.0 \
    --rr_calf_strength_range "${SMOKE_RR_STRENGTH}" "${SMOKE_RR_STRENGTH}" \
    --env_action_noise_std 0.0 --env_action_bias_std 0.0 \
    --env_action_scale_range 1.0 1.0 --env_action_delay_steps 0 0 \
    --command_modes stand,pure_x,pure_y,pure_yaw,xy,x_yaw,y_yaw,xy_yaw \
    --command_mode_weights stand:0.08,pure_x:0.25,pure_y:0.10,pure_yaw:0.08,xy:0.14,x_yaw:0.17,y_yaw:0.05,xy_yaw:0.13 \
    --x_range -0.5 0.5 --signed_x --x_abs_range 0.08 0.50 \
    --y_abs_range 0.05 0.20 --yaw_abs_range 0.08 0.40 \
    --command_resample_interval_min 120 --command_resample_interval_max 300 \
    --use_domain_randomization --use_push_randomization --use_observation_noise \
    --trace_reset_dataset "${DATASET}" \
    --trace_rollout_length "${SMOKE_ROLLOUT_LENGTH}" --trace_trajectories_per_state 4 \
    --trace_action_temperature 2.0 --headless
fi

"${PYTHON_BIN}" scripts/reinforcement_learning/rwm_trace/build_trace_replay_v3.py \
  --candidate_dataset "${CANDIDATES}" \
  --selection random --select_ratio 0.50 \
  --trajectory_length "${SMOKE_ROLLOUT_LENGTH}" --trajectory_stride "${SMOKE_ROLLOUT_LENGTH}" \
  --include_terminal_prefixes --minimum_terminal_length 20 \
  --failure_trajectory_ratio 0.20 --terminal_penalty -10.0 \
  --n_step 3 --gamma 0.99 --reward_source rwm_aligned --seed 142 \
  --output "${REPLAY}"

"${PYTHON_BIN}" - "${CANDIDATES}" "${REPLAY}" "${ROOT}/smoke_summary.json" <<'PY'
import json
import sys
from pathlib import Path
import torch

candidate_path, replay_path, output = sys.argv[1:]
candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
replay = torch.load(replay_path, map_location="cpu", weights_only=False)
metadata = replay["metadata"]
payload = {
    "candidate_metadata": candidate.get("metadata", {}),
    "replay_metadata": metadata,
    "replay_terminated_count": int(replay["terminated"].bool().sum()),
    "replay_reward_min": float(replay["reward"].min()),
    "replay_reward_max": float(replay["reward"].max()),
    "replay_action_abs_max": float(replay["action"].abs().max()),
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
assert metadata["trajectory_length"] == int(candidate["metadata"]["trace_candidates"]["rollout_length"])
assert metadata["failure_trajectory_ratio"] == 0.20
assert metadata["terminal_penalty"] == -10.0
assert replay["observation"].shape[-1] == 48
assert replay["action"].shape[-1] == 12
print(json.dumps(payload, indent=2, sort_keys=True))
PY
