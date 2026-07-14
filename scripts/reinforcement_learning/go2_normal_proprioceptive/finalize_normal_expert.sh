#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

: "${STAGE_DIR:?Set STAGE_DIR}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
: "${DR_PRESET:?Set DR_PRESET}"
: "${DR_SCALE:?Set DR_SCALE}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
SEED="${SEED:-0}"
NUM_ENV_STEPS="${NUM_ENV_STEPS:-100000000}"
EXPECTED_FINAL_STEP="${EXPECTED_FINAL_STEP:-97656}"
PAYLOAD_MASS_MIN_KG="${PAYLOAD_MASS_MIN_KG:-0.0}"
PAYLOAD_MASS_MAX_KG="${PAYLOAD_MASS_MAX_KG:-0.0}"
RR_CALF_STRENGTH_MIN="${RR_CALF_STRENGTH_MIN:-1.0}"
RR_CALF_STRENGTH_MAX="${RR_CALF_STRENGTH_MAX:-1.0}"

if [[ -f "${STAGE_DIR}/summary.json" ]] \
  && grep -q '"status": "completed"' "${STAGE_DIR}/summary.json"; then
  exit 0
fi

mapfile -t CHECKPOINTS < <(
  find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'step*' -print 2>/dev/null \
    | while read -r path; do
        step="${path##*/step}"
        [[ "${step}" =~ ^[0-9]+$ ]] && printf '%012d %s\n' "${step}" "${path}"
      done \
    | sort -n
)
if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
  exit 3
fi

LATEST_STEP_PADDED="${CHECKPOINTS[-1]%% *}"
LATEST_STEP="$((10#${LATEST_STEP_PADDED}))"
if (( LATEST_STEP < EXPECTED_FINAL_STEP )); then
  exit 3
fi

mkdir -p "${STAGE_DIR}"
exec 9>"${STAGE_DIR}/run.lock"
flock 9
if [[ -f "${STAGE_DIR}/summary.json" ]] \
  && grep -q '"status": "completed"' "${STAGE_DIR}/summary.json"; then
  exit 0
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

echo "finalized=${STAGE_DIR}"
