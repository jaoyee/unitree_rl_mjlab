#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/unitree_rl_mjlab_model_based_broken}"
VIEWER_GPUS="${VIEWER_GPUS:-3 5 6 7}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/viewers/best_four_manual}"

read -r -a GPUS <<<"${VIEWER_GPUS}"
if [[ "${#GPUS[@]}" -ne 4 ]]; then
  echo "VIEWER_GPUS must contain exactly four GPU indices, got: ${VIEWER_GPUS}" >&2
  exit 2
fi

POLICY_NAMES=(
  nodr_baseline
  half_dr_base_clean
  half_dr_action_noise_clean
  exp1_olddr
)
CHECKPOINTS=(
  "${REPO_ROOT}/logs/imported/nodr_regression_20260711/step48828"
  "${REPO_ROOT}/logs/experiments/go2_0p5_friend_flat_dr/formal_2x2_core_half_20260711/policies/base_final_clean/2026-07-11_05-11-35/step48828"
  "${REPO_ROOT}/logs/experiments/go2_0p5_friend_flat_dr/formal_2x2_core_half_20260711/policies/action_noise_final_clean/2026-07-11_05-11-41/step48828"
  "/root/unitree_rl_mjlab_zkq_exp1_olddr_unmasked/logs/repro/exp1_olddr_unmasked_20260709_1005/model_based/go2_rr_calf_strength_0p5_proprioceptive_olddr_unmasked/2026-07-09_11-37-31/step48828"
)
PORTS=(8080 8081 8082 8083)
SESSIONS=(
  go2_manual_nodr_8080
  go2_manual_half_base_8081
  go2_manual_half_action_8082
  go2_manual_olddr_8083
)

mkdir -p "${LOG_ROOT}"

# Retire viewers that occupied the first two ports with automatic schedules.
for old_session in go2_dr_best_visual go2_exp1_olddr_visual_8081; do
  tmux kill-session -t "${old_session}" 2>/dev/null || true
done

for index in 0 1 2 3; do
  checkpoint="${CHECKPOINTS[$index]}"
  session="${SESSIONS[$index]}"
  policy_name="${POLICY_NAMES[$index]}"
  gpu="${GPUS[$index]}"
  port="${PORTS[$index]}"
  log_file="${LOG_ROOT}/${policy_name}.log"

  if [[ ! -f "${checkpoint}/actor.pt" ]]; then
    echo "Missing actor checkpoint: ${checkpoint}/actor.pt" >&2
    exit 3
  fi
  tmux kill-session -t "${session}" 2>/dev/null || true

  command="cd '${REPO_ROOT}' && exec env CUDA_VISIBLE_DEVICES='${gpu}' MUJOCO_GL=egl uv run python scripts/reinforcement_learning/rwm_flashsac/play_flashsac_go2_mjlab_proprioceptive.py --checkpoint_path '${checkpoint}' --task Unitree-Go2-Flat-RWM-Pretrain-Ens --device cuda:0 --num_envs 1 --seed 123 --viewer viser --viser_port '${port}' --frame_rate 60 --manual_command --random_lin_vel_x -0.5 0.5 --random_lin_vel_y -0.2 0.2 --random_ang_vel_z -0.4 0.4 --clean --joint_strength_scales RR_calf_joint=0.5 2>&1 | tee '${log_file}'"
  tmux new-session -d -s "${session}" "bash -lc \"${command}\""
  echo "started policy=${policy_name} session=${session} gpu=${gpu} port=${port} log=${log_file}"
done

echo
echo "All viewers start at command (0, 0, 0). Open the Viser Commands folder to adjust vx, vy, and yaw."
