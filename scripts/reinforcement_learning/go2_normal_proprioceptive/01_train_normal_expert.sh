#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

: "${STAGE_DIR:?Set STAGE_DIR}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
: "${DR_PRESET:?Set DR_PRESET}"
: "${DR_SCALE:?Set DR_SCALE}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
SEED="${SEED:-0}"
NUM_TRAIN_ENVS="${NUM_TRAIN_ENVS:-1024}"
NUM_ENV_STEPS="${NUM_ENV_STEPS:-100000000}"
DEVICE="${DEVICE:-cuda:0}"
DR_COMPONENTS="${DR_COMPONENTS:-friction,mass_com,motor,delay,observation,push,initial_state}"
PAYLOAD_MASS_MIN_KG="${PAYLOAD_MASS_MIN_KG:-0.0}"
PAYLOAD_MASS_MAX_KG="${PAYLOAD_MASS_MAX_KG:-0.0}"
RR_CALF_STRENGTH_MIN="${RR_CALF_STRENGTH_MIN:-1.0}"
RR_CALF_STRENGTH_MAX="${RR_CALF_STRENGTH_MAX:-1.0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 1
fi
OUTPUT_RELATIVE="${OUTPUT_DIR#${REPO_ROOT}/}"
if [[ "${OUTPUT_RELATIVE}" == *broken* ]]; then
  echo "Normal expert output path contains a forbidden broken marker: ${OUTPUT_DIR}" >&2
  exit 1
fi
mkdir -p "${STAGE_DIR}"
exec 9>"${STAGE_DIR}/run.lock"
flock 9
if [[ -f "${STAGE_DIR}/summary.json" ]] \
  && grep -q '"status": "completed"' "${STAGE_DIR}/summary.json"; then
  echo "Expert stage already completed: ${STAGE_DIR}"
  exit 0
fi
if [[ -e "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Expert output directory is not empty: ${OUTPUT_DIR}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" scripts/train_flashsac.py \
  --config_name flashsac_go2_normal_proprioceptive_expert \
  --no-export_deploy_policy \
  --overrides "seed=${SEED}" \
  --overrides "num_train_envs=${NUM_TRAIN_ENVS}" \
  --overrides "num_env_steps=${NUM_ENV_STEPS}" \
  --overrides "save_path=${OUTPUT_DIR}" \
  --overrides "env.device=${DEVICE}" \
  --overrides "env.use_domain_randomization=true" \
  --overrides "env.use_push_randomization=true" \
  --overrides "env.use_observation_noise=true" \
  --overrides "env.randomization_preset=${DR_PRESET}" \
  --overrides "env.randomization_components=[${DR_COMPONENTS}]" \
  --overrides "env.randomization_scale=${DR_SCALE}" \
  --overrides "env.payload_mass_range_kg=[${PAYLOAD_MASS_MIN_KG},${PAYLOAD_MASS_MAX_KG}]" \
  --overrides "env.rr_calf_strength_range=[${RR_CALF_STRENGTH_MIN},${RR_CALF_STRENGTH_MAX}]" \
  --overrides "env.broken_joint_names=[]" \
  --overrides "env.joint_strength_scales={}" \
  --overrides "env.action_mask_indices=[]" \
  --overrides "env.actuator_curriculum.enabled=false"

mapfile -t CHECKPOINTS < <(
  find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'step*' -print \
    | while read -r path; do
        step="${path##*/step}"
        [[ "${step}" =~ ^[0-9]+$ ]] && printf '%012d %s\n' "${step}" "${path}"
      done \
    | sort -n
)
if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
  echo "Expert training produced no step checkpoint in ${OUTPUT_DIR}" >&2
  exit 1
fi
ARTIFACT_PATH="${CHECKPOINTS[-1]#* }"
if [[ ! -f "${ARTIFACT_PATH}/actor.pt" || ! -f "${ARTIFACT_PATH}/flashsac_config.yaml" ]]; then
  echo "Incomplete expert checkpoint: ${ARTIFACT_PATH}" >&2
  exit 1
fi
ARTIFACT_PATH="$(realpath "${ARTIFACT_PATH}")"
ARTIFACT_SHA256="$(sha256sum "${ARTIFACT_PATH}/actor.pt" | awk '{print $1}')"
printf '%s\n' "${ARTIFACT_PATH}" > "${STAGE_DIR}/artifact_path.txt"
printf '%s\n' "${ARTIFACT_SHA256}" > "${STAGE_DIR}/artifact_sha256.txt"

"${PYTHON_BIN}" - "${STAGE_DIR}/summary.json" "${ARTIFACT_PATH}" "${ARTIFACT_SHA256}" \
  "${DR_PRESET}" "${DR_SCALE}" "${SEED}" "${NUM_ENV_STEPS}" \
  "${PAYLOAD_MASS_MIN_KG}" "${PAYLOAD_MASS_MAX_KG}" \
  "${RR_CALF_STRENGTH_MIN}" "${RR_CALF_STRENGTH_MAX}" <<'PY'
import json
import sys
from pathlib import Path

(
    output, artifact, digest, preset, scale, seed, env_steps,
    payload_min, payload_max, strength_min, strength_max,
) = sys.argv[1:]
Path(output).write_text(json.dumps({
    "status": "completed",
    "artifact_path": artifact,
    "artifact_sha256": digest,
    "randomization_preset": preset,
    "randomization_scale": float(scale),
    "seed": int(seed),
    "num_env_steps": int(env_steps),
    "task": "Unitree-Go2-Flat-Proprioceptive-Expert",
    "joint_strength_scales": {},
    "payload_mass_range_kg": [float(payload_min), float(payload_max)],
    "rr_calf_strength_range": [float(strength_min), float(strength_max)],
    "broken_joint_names": [],
}, indent=2, sort_keys=True), encoding="utf-8")
PY

echo "expert_artifact=${ARTIFACT_PATH}"
echo "expert_actor_sha256=${ARTIFACT_SHA256}"
