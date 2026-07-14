#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
RR_ROOT="${RR_ROOT:-${REPO_ROOT}/logs/experiments/go2_gap_experts/rr_calf_20260713_rrcalf_1_vs_0p5to1}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/viewers/gap_rr_0p5_20260713}"
VIEWER_GPUS="${VIEWER_GPUS:-2 5 6 1 3 4}"
read -r -a GPUS <<< "${VIEWER_GPUS}"

if [[ "${#GPUS[@]}" -ne 6 ]]; then
  echo "VIEWER_GPUS must contain six GPU indices" >&2
  exit 2
fi

POLICIES=(G0_D0 G0_D1 G0_D2 G1_D0 G1_D1 G1_D2)
PORTS=(8090 8094 8095 8091 8092 8093)
mkdir -p "${LOG_ROOT}"

for index in 0 1 2 3 4 5; do
  policy="${POLICIES[$index]}"
  gpu="${GPUS[$index]}"
  port="${PORTS[$index]}"
  policy_slug="${policy//_/}"
  session="go2_gap_rr05_${policy_slug,,}_${port}"
  checkpoint="$(cat "${RR_ROOT}/${policy}/artifact_path.txt")"
  log_file="${LOG_ROOT}/${policy}.log"
  if [[ ! -f "${checkpoint}/actor.pt" ]]; then
    echo "Missing checkpoint: ${checkpoint}/actor.pt" >&2
    exit 3
  fi
  tmux kill-session -t "${session}" 2>/dev/null || true
  tmux new-session -d -s "${session}" \
    "cd '${REPO_ROOT}' && exec env CUDA_VISIBLE_DEVICES='${gpu}' MUJOCO_GL=egl '${REPO_ROOT}/.venv/bin/python' scripts/play_flashsac_mjlab.py --checkpoint_path '${checkpoint}' --device cuda:0 --num_envs 1 --viewer viser --viser_port '${port}' --frame_rate 60 --manual_command --manual_lin_vel_x -0.5 0.5 --manual_lin_vel_y -0.2 0.2 --manual_ang_vel_z -0.4 0.4 --joint_strength_scales RR_calf_joint=0.5 2>&1 | tee '${log_file}'"
  echo "started policy=${policy} gpu=${gpu} port=${port} session=${session}"
done

echo "All viewers use RR_calf_joint strength 0.5 and start with command (0, 0, 0)."
