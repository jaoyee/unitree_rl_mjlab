#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SOURCE_ROOT="${SOURCE_ROOT:-${REPO_ROOT}/logs/experiments/go2_normal_dr_3x3x4/normal_go2_fixstand_v2_moderate_seed0_20260712}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/experiments/go2_base_lin_vel_loss_ablation/e1d1p2_no_base_lin_vel_loss_${RUN_ID}}"
DATASET_PATH="${DATASET_PATH:-${SOURCE_ROOT}/datasets/E1/D1/dataset.pt}"
CONTROL_POLICY_PATH="${CONTROL_POLICY_PATH:-${SOURCE_ROOT}/policies/E1/D1/P2/run/step48828}"
DRY_RUN="${DRY_RUN:-true}"
CONFIRM_RUN="${CONFIRM_RUN:-false}"
DEVICE="${DEVICE:-cuda:0}"

if [[ "${DRY_RUN}" != "true" && "${CONFIRM_RUN}" != "true" ]]; then
  echo "Execution requires DRY_RUN=false CONFIRM_RUN=true" >&2
  exit 2
fi
if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "E1D1 dataset not found: ${DATASET_PATH}" >&2
  exit 1
fi
if [[ ! -f "${CONTROL_POLICY_PATH}/actor.pt" ]]; then
  echo "E1D1P2 control policy not found: ${CONTROL_POLICY_PATH}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/rwm" "${OUTPUT_ROOT}/policy" "${OUTPUT_ROOT}/evaluation"
cat > "${OUTPUT_ROOT}/manifest.json" <<EOF
{
  "comparison": "E1D1P2 control versus no base-linear-velocity supervision",
  "control_policy_path": "${CONTROL_POLICY_PATH}",
  "dataset_path": "${DATASET_PATH}",
  "expert_dr": "friend_half",
  "dataset_dr": "friend_half",
  "policy_interface": "obs_noise",
  "rwm_input_state_dim": 42,
  "rwm_output_state_dim": 45,
  "state_loss_ignored_indices": [0, 1, 2],
  "remaining_state_loss_scale": 1.0714285714285714,
  "rwm_seed": 200,
  "policy_seed": 300,
  "evaluation_seed": 400
}
EOF

run_command() {
  local name="$1"
  shift
  printf '%q ' "$@" > "${OUTPUT_ROOT}/${name}.command.sh"
  printf '\n' >> "${OUTPUT_ROOT}/${name}.command.sh"
  chmod +x "${OUTPUT_ROOT}/${name}.command.sh"
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[dry-run] ${name}: $(cat "${OUTPUT_ROOT}/${name}.command.sh")"
    return 0
  fi
  "$@" 2>&1 | tee "${OUTPUT_ROOT}/${name}.log"
}

run_command train_rwm env \
  STAGE_DIR="${OUTPUT_ROOT}/rwm" \
  OUTPUT_DIR="${OUTPUT_ROOT}/rwm/run" \
  DATASET_PATH="${DATASET_PATH}" \
  STATE_LOSS_IGNORED_INDICES="0,1,2" \
  SEED=200 DEVICE="${DEVICE}" \
  bash "${SCRIPT_DIR}/03_train_normal_rwm.sh"

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[dry-run] policy and evaluation depend on the generated RWM artifact."
  echo "output_root=${OUTPUT_ROOT}"
  exit 0
fi

MODEL_PATH="$(cat "${OUTPUT_ROOT}/rwm/artifact_path.txt")"
run_command train_policy env \
  STAGE_DIR="${OUTPUT_ROOT}/policy" \
  OUTPUT_DIR="${OUTPUT_ROOT}/policy/run" \
  DATASET_PATH="${DATASET_PATH}" MODEL_PATH="${MODEL_PATH}" \
  P_ID=P2 SEED=300 DEVICE="${DEVICE}" \
  bash "${SCRIPT_DIR}/04_train_normal_policy.sh"

POLICY_PATH="$(cat "${OUTPUT_ROOT}/policy/artifact_path.txt")"
run_command evaluate env \
  STAGE_DIR="${OUTPUT_ROOT}/evaluation" \
  CHECKPOINT_PATH="${POLICY_PATH}" \
  SEED=400 DEVICE="${DEVICE}" \
  bash "${SCRIPT_DIR}/05_evaluate_normal_policy.sh"

echo "ablation_complete=${OUTPUT_ROOT}"
