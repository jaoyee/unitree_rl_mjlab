#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
SIDE="${SIDE:?Set SIDE to sim or real}"
case "${SIDE}" in sim|real) ;; *) echo "SIDE must be sim or real" >&2; exit 2 ;; esac

PYTHON="${REPO}/.venv/bin/python"
EXPERT="${EXPERT_OVERRIDE:-${REPO}/logs/experiments/go2_gap_experts/selected_g0d0/step97656}"
PARTS="${PARTS_OVERRIDE:-${REPO}/logs/rwm_datasets_v4/${SIDE}_v4/pooled_40_20_20_10_10/parts}"
ROOT="${RUN_ROOT_OVERRIDE:-${REPO}/logs/experiments/go2_pooled_v4/20260717_40_20_20_10_10/${SIDE}/trace}"
OUT="${ROOT}/replay_source"
mkdir -p "${OUT}"

export WANDB_MODE=offline MUJOCO_GL=egl
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

collect_one() {
  local gap="$1" start_states="$2" payload="$3" rr_strength="$4"
  local envs=$((start_states * 4))
  local transitions=$((envs * 100))
  local reset_dataset="${PARTS}/${gap}.pt"
  local output="${OUT}/candidates_${gap}_t1_h100.pt"
  [[ -s "${output}" ]] && return
  [[ -s "${reset_dataset}" ]] || { echo "Missing ${reset_dataset}" >&2; exit 2; }
  "${PYTHON}" scripts/reinforcement_learning/rwm_dataset/collect_go2_expert_command_coverage_dataset.py \
    --task Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert \
    --device cuda:0 --seed 42 --num_envs "${envs}" --num_transitions "${transitions}" \
    --save_path "${output}" --expert_policy_path "${EXPERT}" \
    --collector_mix expert:1.0 --fixed_collector_assignment --collector_assignment_seed 42 \
    --dataset_obs_kind proprioceptive --action_noise_std 0.0 \
    --failure_action_noise_std 0.0 --randomization_preset calibrated_default \
    --randomization_components friction,mass_com,motor,delay,initial_state \
    --randomization_scale 0.2 --payload_mass_range_kg "${payload}" "${payload}" \
    --payload_position_body_m 0.0 0.0 0.10 --payload_box_size_m 0.20 0.12 0.05 \
    --rr_calf_strength_range "${rr_strength}" "${rr_strength}" --env_action_noise_std 0.0 \
    --env_action_bias_std 0.0 --env_action_scale_range 1.0 1.0 \
    --env_action_delay_steps 0 0 --chunk_size 0 \
    --command_modes stand,pure_x,pure_y,pure_yaw,xy,x_yaw,y_yaw,xy_yaw \
    --command_mode_weights stand:0.08,pure_x:0.25,pure_y:0.10,pure_yaw:0.08,xy:0.14,x_yaw:0.17,y_yaw:0.05,xy_yaw:0.13 \
    --x_range -0.5 0.5 --signed_x --x_abs_range 0.08 0.50 \
    --y_abs_range 0.05 0.20 --yaw_abs_range 0.08 0.40 \
    --command_resample_interval_min 100 --command_resample_interval_max 250 \
    --use_domain_randomization --trace_reset_dataset "${reset_dataset}" \
    --trace_rollout_length 100 --trace_trajectories_per_state 4 \
    --trace_action_temperature 1.0 --headless
}

# 1024 reset states distributed as 410/205/205/102/102.
collect_one g0   410 0.0 1.0
collect_one rr05 205 0.0 0.5
collect_one p5   205 5.0 1.0
collect_one rr03 102 0.0 0.3
collect_one p75  102 7.5 1.0

sha256sum "${OUT}"/candidates_*_t1_h100.pt > "${ROOT}/candidate_sha256.txt"
printf 'side=%s\ntemperature=1.0\nhorizon=100\nstart_states=1024\ntrajectories_per_state=4\nstatus=completed\n' \
  "${SIDE}" > "${ROOT}/candidate_summary.txt"
