#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

: "${STAGE_DIR:?Set STAGE_DIR}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
: "${DATASET_PATH:?Set DATASET_PATH}"
: "${MODEL_PATH:?Set MODEL_PATH}"
: "${P_ID:?Set P_ID to P0, P1, P2, or P3}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
SEED="${SEED:-300}"
DEVICE="${DEVICE:-cuda:0}"
NUM_IMAGINATION_ENVS="${NUM_IMAGINATION_ENVS:-1024}"
NUM_ENV_STEPS="${NUM_ENV_STEPS:-50000000}"
LIN_VEL_X_RANGE="${LIN_VEL_X_RANGE:--0.5 0.5}"
LIN_VEL_Y_RANGE="${LIN_VEL_Y_RANGE:--0.25 0.25}"
ANG_VEL_Z_RANGE="${ANG_VEL_Z_RANGE:--0.5 0.5}"
read -r LIN_VEL_X_MIN LIN_VEL_X_MAX <<< "${LIN_VEL_X_RANGE}"
read -r LIN_VEL_Y_MIN LIN_VEL_Y_MAX <<< "${LIN_VEL_Y_RANGE}"
read -r ANG_VEL_Z_MIN ANG_VEL_Z_MAX <<< "${ANG_VEL_Z_RANGE}"

case "${P_ID}" in
  P0)
    ACTION_NOISE_STD=0.0
    OBS_PROFILE=none
    OBS_SCALE=0.0
    ;;
  P1)
    ACTION_NOISE_STD=0.01
    OBS_PROFILE=none
    OBS_SCALE=0.0
    ;;
  P2)
    ACTION_NOISE_STD=0.0
    OBS_PROFILE=deployment_small
    OBS_SCALE=0.25
    ;;
  P3)
    ACTION_NOISE_STD=0.01
    OBS_PROFILE=deployment_small
    OBS_SCALE=0.25
    ;;
  *)
    echo "Unknown P_ID: ${P_ID}" >&2
    exit 1
    ;;
esac

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${DATASET_PATH}" || ! -f "${MODEL_PATH}" ]]; then
  echo "Missing dataset or RWM artifact" >&2
  exit 1
fi
OUTPUT_RELATIVE="${OUTPUT_DIR#${REPO_ROOT}/}"
if [[ "${OUTPUT_RELATIVE}" == *0p5* || "${OUTPUT_RELATIVE}" == *broken* ]]; then
  echo "Normal policy output path contains a forbidden 0.5/broken marker: ${OUTPUT_DIR}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Policy output directory is not empty: ${OUTPUT_DIR}" >&2
  exit 1
fi
mkdir -p "${STAGE_DIR}" "${OUTPUT_DIR}"

"${PYTHON_BIN}" scripts/reinforcement_learning/rwm_flashsac/train_flashsac_world_model_go2_proprioceptive.py \
  --config_path scripts/reinforcement_learning/rwm_flashsac/configs/go2_flashsac_rwm_proprioceptive.yaml \
  --model_resume_path "${MODEL_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --num_imagination_envs "${NUM_IMAGINATION_ENVS}" \
  --num_env_steps "${NUM_ENV_STEPS}" \
  --device "${DEVICE}" \
  --save_path "${OUTPUT_DIR}" \
  --overrides "seed=${SEED}" \
  --overrides "world_model.policy_action_mask_indices=[]" \
  --overrides "world_model.world_model_action_mask_indices=[]" \
  --overrides "world_model.policy_observation_mask_indices=[]" \
  --overrides "world_model.broken_joint_names=[]" \
  --overrides "world_model.lin_vel_x_min=${LIN_VEL_X_MIN}" \
  --overrides "world_model.lin_vel_x_max=${LIN_VEL_X_MAX}" \
  --overrides "world_model.lin_vel_y_min=${LIN_VEL_Y_MIN}" \
  --overrides "world_model.lin_vel_y_max=${LIN_VEL_Y_MAX}" \
  --overrides "world_model.ang_vel_z_min=${ANG_VEL_Z_MIN}" \
  --overrides "world_model.ang_vel_z_max=${ANG_VEL_Z_MAX}" \
  --overrides "world_model.interface_action_noise_std=${ACTION_NOISE_STD}" \
  --overrides "world_model.interface_action_bias_std=0.0" \
  --overrides "world_model.interface_action_scale_min=1.0" \
  --overrides "world_model.interface_action_scale_max=1.0" \
  --overrides "world_model.interface_action_delay_steps_min=0" \
  --overrides "world_model.interface_action_delay_steps_max=0" \
  --overrides "world_model.interface_obs_noise_profile=${OBS_PROFILE}" \
  --overrides "world_model.interface_obs_noise_scale=${OBS_SCALE}" \
  --overrides "world_model.interface_obs_joint_pos_bias_min=0.0" \
  --overrides "world_model.interface_obs_joint_pos_bias_max=0.0" \
  --overrides "world_model.interface_obs_noise_std=0.0" \
  --overrides "world_model.interface_obs_bias_std=0.0"

mapfile -t CHECKPOINTS < <(
  find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'step*' -print \
    | while read -r path; do
        step="${path##*/step}"
        [[ "${step}" =~ ^[0-9]+$ ]] && printf '%012d %s\n' "${step}" "${path}"
      done \
    | sort -n
)
if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
  echo "Final-policy training produced no step checkpoint in ${OUTPUT_DIR}" >&2
  exit 1
fi
ARTIFACT_PATH="${CHECKPOINTS[-1]#* }"
if [[ ! -f "${ARTIFACT_PATH}/actor.pt" || ! -f "${ARTIFACT_PATH}/rwm_flashsac_config.yaml" ]]; then
  echo "Incomplete final-policy checkpoint: ${ARTIFACT_PATH}" >&2
  exit 1
fi
ARTIFACT_PATH="$(realpath "${ARTIFACT_PATH}")"
ARTIFACT_SHA256="$(sha256sum "${ARTIFACT_PATH}/actor.pt" | awk '{print $1}')"
MODEL_SHA256="$(sha256sum "${MODEL_PATH}" | awk '{print $1}')"
DATASET_SHA256="$(sha256sum "${DATASET_PATH}" | awk '{print $1}')"
printf '%s\n' "${ARTIFACT_PATH}" > "${STAGE_DIR}/artifact_path.txt"
printf '%s\n' "${ARTIFACT_SHA256}" > "${STAGE_DIR}/artifact_sha256.txt"

"${PYTHON_BIN}" - "${STAGE_DIR}/summary.json" "${ARTIFACT_PATH}" "${ARTIFACT_SHA256}" \
  "$(realpath "${MODEL_PATH}")" "${MODEL_SHA256}" "$(realpath "${DATASET_PATH}")" "${DATASET_SHA256}" \
  "${P_ID}" "${ACTION_NOISE_STD}" "${OBS_PROFILE}" "${OBS_SCALE}" "${SEED}" "${NUM_ENV_STEPS}" <<'PY'
import json
import sys
from pathlib import Path

(
    output, artifact, digest, model, model_digest, dataset, dataset_digest,
    p_id, action_noise, obs_profile, obs_scale, seed, env_steps,
) = sys.argv[1:]
Path(output).write_text(json.dumps({
    "status": "completed",
    "artifact_path": artifact,
    "artifact_sha256": digest,
    "model_path": model,
    "model_sha256": model_digest,
    "dataset_path": dataset,
    "dataset_sha256": dataset_digest,
    "p_id": p_id,
    "seed": int(seed),
    "num_env_steps": int(env_steps),
    "interface": {
        "action_noise_std": float(action_noise),
        "action_bias_std": 0.0,
        "action_scale": [1.0, 1.0],
        "action_delay_steps": [0, 0],
        "obs_noise_profile": obs_profile,
        "obs_noise_scale": float(obs_scale),
        "obs_joint_pos_bias": [0.0, 0.0],
    },
    "policy_action_mask_indices": [],
    "world_model_action_mask_indices": [],
    "broken_joint_names": [],
}, indent=2, sort_keys=True), encoding="utf-8")
PY

echo "policy_artifact=${ARTIFACT_PATH}"
echo "policy_actor_sha256=${ARTIFACT_SHA256}"
