#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${V10_ARTIFACT_ENV:?Set the server-local V10 artifact manifest}"
: "${V10_RUN_BASE:?Set a new reward-pilot run base}"
: "${V10_GPU_POOL:?Set physical GPU pool}"
PY="${PYTHON_BIN:-${REPO}/.venv/bin/python}"
CONFIG="${REPO}/scripts/reinforcement_learning/rwm_trace/trace_v10_reward_v2_pilot.json"
PROTOCOL="${REPO}/scripts/reinforcement_learning/rwm_trace/trace_v10_protocol.json"
source "${V10_ARTIFACT_ENV}"

readarray -t SETTINGS < <("${PY}" - "${CONFIG}" "${PROTOCOL}" <<'PY'
import hashlib
import json
import sys

config_path, protocol_path = sys.argv[1:]
config = json.load(open(config_path, encoding="utf-8"))
with open(protocol_path, "rb") as stream:
    protocol_sha = hashlib.sha256(stream.read()).hexdigest()
if config.get("schema") != "go2_trace_v10_reward_v2_pilot_v1":
    raise SystemExit("Unsupported reward-pilot config schema")
if config.get("base_protocol_sha256") != protocol_sha:
    raise SystemExit("Reward-pilot config is bound to a different V10 protocol")
training = config["training"]
evaluation = config["evaluation"]
reward = config["reward_v2"]
if training["policy_resume_mode"] != "actor_only":
    raise SystemExit("Reward comparison must use actor-only initialization")
if training["load_replay_buffer"] or training["load_reward_normalizer"]:
    raise SystemExit("Reward comparison cannot load V1 replay or reward normalization state")
values = [
    config["side"], config["condition"], training["environment_steps"], training["seed"],
    " ".join(map(str, evaluation["seeds"])), evaluation["num_envs"], evaluation["steps"],
    reward["version"],
    reward["command_response_weight"], reward["yaw_command_response_weight"],
    reward["wrong_direction_weight"], reward["uncertainty_penalty_weight"],
    reward["action_rate_l2"], reward["action_saturation"],
    reward["action_saturation_threshold"], reward["dof_acc_l2"],
    reward["dof_torques_l2"], reward["command_active_threshold"],
    reward["motion_gate_low"], reward["motion_gate_high"],
]
for value in values:
    print(value)
PY
)
SIDE="${SETTINGS[0]}"; CONDITION="${SETTINGS[1]}"; ENV_STEPS="${SETTINGS[2]}"
SEED="${SETTINGS[3]}"; EVAL_SEEDS="${SETTINGS[4]}"; EVAL_NUM_ENVS="${SETTINGS[5]}"
EVAL_STEPS="${SETTINGS[6]}"; V2_VERSION="${SETTINGS[7]}"
DATASET_PATH="${REAL_RR05_DATASET:-}"; MODEL_PATH="${REAL_RR05_MODEL:-}"
INITIAL_POLICY_PATH="${REAL_RR05_WARMUP:-}"
for path in "${DATASET_PATH}" "${MODEL_PATH}" "${INITIAL_POLICY_PATH}/actor.pt"; do
  [[ -s "${path}" ]] || { echo "Missing reward-pilot artifact: ${path}" >&2; exit 2; }
done
mkdir -p "${V10_RUN_BASE}"

controller="v10_reward_rr05_controller"
for session in "${controller}" v10_reward_rr05_v1 v10_reward_rr05_v2; do
  tmux has-session -t "${session}" 2>/dev/null && {
    echo "Session already exists: ${session}" >&2; exit 2;
  }
done

tmux new-session -d -s "${controller}" env \
  REPO="${REPO}" RUN_ROOT="${V10_RUN_BASE}" V10_GPU_POOL="${V10_GPU_POOL}" \
  SIDE="${SIDE}" CONDITION="${CONDITION}" DATASET_PATH="${DATASET_PATH}" \
  MODEL_PATH="${MODEL_PATH}" INITIAL_POLICY_PATH="${INITIAL_POLICY_PATH}" \
  ENV_STEPS="${ENV_STEPS}" SEED="${SEED}" EVAL_SEEDS="${EVAL_SEEDS}" \
  EVAL_NUM_ENVS="${EVAL_NUM_ENVS}" EVAL_STEPS="${EVAL_STEPS}" \
  V2_VERSION="${V2_VERSION}" CONFIG="${CONFIG}" \
  V2_COMMAND_RESPONSE="${SETTINGS[8]}" V2_YAW_RESPONSE="${SETTINGS[9]}" \
  V2_WRONG_DIRECTION="${SETTINGS[10]}" V2_UNCERTAINTY="${SETTINGS[11]}" \
  V2_ACTION_RATE="${SETTINGS[12]}" V2_ACTION_SATURATION="${SETTINGS[13]}" \
  V2_ACTION_SATURATION_THRESHOLD="${SETTINGS[14]}" V2_DOF_ACC="${SETTINGS[15]}" \
  V2_DOF_TORQUES="${SETTINGS[16]}" V2_ACTIVE_THRESHOLD="${SETTINGS[17]}" \
  V2_GATE_LOW="${SETTINGS[18]}" V2_GATE_HIGH="${SETTINGS[19]}" \
  bash -lc '
set -euo pipefail
for arm in reward_v1 reward_v2; do
  session="v10_reward_rr05_${arm##reward_}"
  reward_version=v1
  extra_env=()
  if [[ "${arm}" == reward_v2 ]]; then
    reward_version="${V2_VERSION}"
    extra_env=(
      REWARD_COMMAND_RESPONSE_WEIGHT="${V2_COMMAND_RESPONSE}"
      REWARD_YAW_COMMAND_RESPONSE_WEIGHT="${V2_YAW_RESPONSE}"
      REWARD_WRONG_DIRECTION_WEIGHT="${V2_WRONG_DIRECTION}"
      REWARD_UNCERTAINTY_PENALTY_WEIGHT="${V2_UNCERTAINTY}"
      REWARD_ACTION_RATE_L2="${V2_ACTION_RATE}"
      REWARD_ACTION_SATURATION="${V2_ACTION_SATURATION}"
      REWARD_ACTION_SATURATION_THRESHOLD="${V2_ACTION_SATURATION_THRESHOLD}"
      REWARD_DOF_ACC_L2="${V2_DOF_ACC}"
      REWARD_DOF_TORQUES_L2="${V2_DOF_TORQUES}"
      REWARD_COMMAND_ACTIVE_THRESHOLD="${V2_ACTIVE_THRESHOLD}"
      REWARD_MOTION_GATE_LOW="${V2_GATE_LOW}"
      REWARD_MOTION_GATE_HIGH="${V2_GATE_HIGH}"
    )
  fi
  tmux new-session -d -s "${session}" env \
    REPO="${REPO}" SIDE="${SIDE}" CONDITION="${CONDITION}" \
    RUN_ROOT="${RUN_ROOT}/${arm}" DATASET_PATH="${DATASET_PATH}" MODEL_PATH="${MODEL_PATH}" \
    INITIAL_POLICY_PATH="${INITIAL_POLICY_PATH}" V10_GPU_POOL="${V10_GPU_POOL}" \
    RUN_KIND=pilot CONTROL_NUM_ENV_STEPS="${ENV_STEPS}" SEED="${SEED}" \
    POLICY_RESUME_MODE=actor_only LOAD_REPLAY_BUFFER=false LOAD_REWARD_NORMALIZER=false \
    REWARD_VERSION="${reward_version}" V10_EVAL_SEEDS="${EVAL_SEEDS}" \
    V10_EVAL_NUM_ENVS="${EVAL_NUM_ENVS}" V10_EVAL_STEPS="${EVAL_STEPS}" \
    "${extra_env[@]}" \
    bash "${REPO}/scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_trace_v10_no_trace_control.sh"
done

for arm in reward_v1 reward_v2; do
  session="v10_reward_rr05_${arm##reward_}"
  marker="${RUN_ROOT}/${arm}/completed.txt"
  while [[ ! -s "${marker}" ]]; do
    tmux has-session -t "${session}" 2>/dev/null || {
      echo "Reward-pilot arm exited without completion: ${arm}" >&2; exit 2;
    }
    sleep 30
  done
done
"${REPO}/.venv/bin/python" \
  "${REPO}/scripts/reinforcement_learning/rwm_trace/summarize_go2_reward_v2_pilot.py" \
  --run-root "${RUN_ROOT}" --config "${CONFIG}" --output "${RUN_ROOT}/comparison.json"
'
echo "Started ${controller}: paired fresh-buffer V1/V2 RR05 RWM-only pilot."
