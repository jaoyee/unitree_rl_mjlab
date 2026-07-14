#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

GAP_KIND="${GAP_KIND:-rr_calf}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/experiments/go2_gap_experts/${GAP_KIND}_${RUN_ID}}"
DRY_RUN="${DRY_RUN:-true}"
CONFIRM_RUN="${CONFIRM_RUN:-false}"
SEED="${SEED:-0}"
NUM_TRAIN_ENVS="${NUM_TRAIN_ENVS:-1024}"
NUM_ENV_STEPS="${NUM_ENV_STEPS:-100000000}"

if [[ "${GAP_KIND}" != "payload" && "${GAP_KIND}" != "rr_calf" ]]; then
  echo "GAP_KIND must be payload or rr_calf" >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "true" && "${CONFIRM_RUN}" != "true" ]]; then
  echo "Execution requires DRY_RUN=false CONFIRM_RUN=true" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/commands"
cat > "${OUTPUT_ROOT}/manifest.json" <<EOF
{
  "gap_kind": "${GAP_KIND}",
  "run_id": "${RUN_ID}",
  "actor_observation_dim": 45,
  "critic_observation_dim": 48,
  "num_policies": 6,
  "seed": ${SEED},
  "num_train_envs": ${NUM_TRAIN_ENVS},
  "num_env_steps": ${NUM_ENV_STEPS},
  "dr": {"D0": ["calibrated_default", 1.0], "D1": ["calibrated_friend_flat", 0.5], "D2": ["calibrated_friend_flat", 1.0]},
  "payload_conditions_kg": {"G0": [0.0, 0.0], "G1": [5.0, 10.0]},
  "payload_position_body_m": [0.0, 0.0, 0.10],
  "payload_box_size_m": [0.20, 0.12, 0.05],
  "rr_calf_strength_conditions": {"G0": [1.0, 1.0], "G1": [0.5, 1.0]}
}
EOF

write_command() {
  local gap_id="$1" dr_id="$2" preset="$3" scale="$4"
  local payload_min=0.0 payload_max=0.0 strength_min=1.0 strength_max=1.0
  if [[ "${GAP_KIND}" == "payload" && "${gap_id}" == "G1" ]]; then
    payload_min=5.0; payload_max=10.0
  elif [[ "${GAP_KIND}" == "rr_calf" && "${gap_id}" == "G1" ]]; then
    strength_min=0.5; strength_max=1.0
  fi
  local job="${gap_id}_${dr_id}"
  local stage_dir="${OUTPUT_ROOT}/${job}"
  local command_file="${OUTPUT_ROOT}/commands/${job}.sh"
  mkdir -p "${stage_dir}"
  cat > "${command_file}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
STAGE_DIR="${stage_dir}" \\
OUTPUT_DIR="${stage_dir}/run" \\
DR_PRESET="${preset}" DR_SCALE="${scale}" \\
PAYLOAD_MASS_MIN_KG="${payload_min}" PAYLOAD_MASS_MAX_KG="${payload_max}" \\
RR_CALF_STRENGTH_MIN="${strength_min}" RR_CALF_STRENGTH_MAX="${strength_max}" \\
SEED="${SEED}" NUM_TRAIN_ENVS="${NUM_TRAIN_ENVS}" NUM_ENV_STEPS="${NUM_ENV_STEPS}" \\
DEVICE="cuda:0" \\
bash "${SCRIPT_DIR}/01_train_normal_expert.sh" 2>&1 | tee -a "${stage_dir}/run.log"
EOF
  chmod +x "${command_file}"
  echo "${job}|${command_file}" >> "${OUTPUT_ROOT}/jobs.list"
}

: > "${OUTPUT_ROOT}/jobs.list"
for gap_id in G0 G1; do
  write_command "${gap_id}" D0 calibrated_default 1.0
  write_command "${gap_id}" D1 calibrated_friend_flat 0.5
  write_command "${gap_id}" D2 calibrated_friend_flat 1.0
done

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[dry-run] prepared six ${GAP_KIND} expert commands in ${OUTPUT_ROOT}"
  cat "${OUTPUT_ROOT}/jobs.list"
  exit 0
fi

echo "Commands are prepared. Run them through the GPU scheduler after smoke validation."
