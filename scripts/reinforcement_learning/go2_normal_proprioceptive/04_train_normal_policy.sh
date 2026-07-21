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
TRACE_REPLAY_PATH="${TRACE_REPLAY_PATH:-}"
TRACE_REPLAY_RATIO="${TRACE_REPLAY_RATIO:-0.10}"
POLICY_RESUME_PATH="${POLICY_RESUME_PATH:-}"
POLICY_RESUME_MODE="${POLICY_RESUME_MODE:-full}"
SAVE_REPLAY_BUFFER="${SAVE_REPLAY_BUFFER:-false}"
LOAD_REPLAY_BUFFER="${LOAD_REPLAY_BUFFER:-true}"
LOAD_REWARD_NORMALIZER="${LOAD_REWARD_NORMALIZER:-auto}"
ACTOR_LEARNING_STARTS_UPDATES="${ACTOR_LEARNING_STARTS_UPDATES:-0}"
LIN_VEL_X_RANGE="${LIN_VEL_X_RANGE:--0.5 0.5}"
LIN_VEL_Y_RANGE="${LIN_VEL_Y_RANGE:--0.25 0.25}"
ANG_VEL_Z_RANGE="${ANG_VEL_Z_RANGE:--0.5 0.5}"
REWARD_VERSION="${REWARD_VERSION:-v1}"
REWARD_COMMAND_RESPONSE_WEIGHT="${REWARD_COMMAND_RESPONSE_WEIGHT:-2.0}"
REWARD_YAW_COMMAND_RESPONSE_WEIGHT="${REWARD_YAW_COMMAND_RESPONSE_WEIGHT:-1.0}"
REWARD_WRONG_DIRECTION_WEIGHT="${REWARD_WRONG_DIRECTION_WEIGHT:--2.0}"
REWARD_RESPONSE_SHORTFALL_WEIGHT="${REWARD_RESPONSE_SHORTFALL_WEIGHT:--6.0}"
REWARD_RESPONSE_FLOOR="${REWARD_RESPONSE_FLOOR:-0.30}"
REWARD_ACTIVE_COMMAND_BIAS="${REWARD_ACTIVE_COMMAND_BIAS:--0.20}"
REWARD_COMMAND_ACTIVE_THRESHOLD="${REWARD_COMMAND_ACTIVE_THRESHOLD:-0.02}"
REWARD_MOTION_GATE_LOW="${REWARD_MOTION_GATE_LOW:-0.05}"
REWARD_MOTION_GATE_HIGH="${REWARD_MOTION_GATE_HIGH:-0.30}"
REWARD_UNCERTAINTY_PENALTY_WEIGHT="${REWARD_UNCERTAINTY_PENALTY_WEIGHT:--2.0}"
REWARD_ACTION_RATE_L2="${REWARD_ACTION_RATE_L2:-}"
REWARD_ACTION_SATURATION="${REWARD_ACTION_SATURATION:-}"
REWARD_ACTION_SATURATION_THRESHOLD="${REWARD_ACTION_SATURATION_THRESHOLD:-0.9}"
REWARD_DOF_ACC_L2="${REWARD_DOF_ACC_L2:-}"
REWARD_DOF_TORQUES_L2="${REWARD_DOF_TORQUES_L2:-}"
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

TRACE_OVERRIDES=()
if [[ -n "${TRACE_REPLAY_PATH}" ]]; then
  if [[ ! -f "${TRACE_REPLAY_PATH}" ]]; then
    echo "Missing TRACE replay artifact: ${TRACE_REPLAY_PATH}" >&2
    exit 1
  fi
  TRACE_REPLAY_PATH="$(realpath "${TRACE_REPLAY_PATH}")"
  TRACE_OVERRIDES+=(
    --overrides "trace.enabled=true"
    --overrides "trace.protocol=go2_trace_v10"
    --overrides "trace.allow_legacy=false"
    --overrides "trace.replay_path=${TRACE_REPLAY_PATH}"
    --overrides "trace.replay_ratio=${TRACE_REPLAY_RATIO}"
  )
fi

POLICY_RESUME_ARGS=()
if [[ -n "${POLICY_RESUME_PATH}" ]]; then
  if [[ ! -f "${POLICY_RESUME_PATH}/actor.pt" || ! -f "${POLICY_RESUME_PATH}/agent_state.pt" ]]; then
    echo "Incomplete policy resume checkpoint: ${POLICY_RESUME_PATH}" >&2
    exit 1
  fi
  POLICY_RESUME_PATH="$(realpath "${POLICY_RESUME_PATH}")"
  if [[ "${LOAD_REWARD_NORMALIZER}" == auto ]]; then
    LOAD_REWARD_NORMALIZER=true
  fi
  if [[ "${LOAD_REWARD_NORMALIZER}" == true && ! -f "${POLICY_RESUME_PATH}/reward_normalizer.pt" ]]; then
    echo "Strict continuation requires reward_normalizer.pt: ${POLICY_RESUME_PATH}" >&2
    exit 1
  fi
  case "${POLICY_RESUME_MODE}" in full|actor_only) ;; *) echo "POLICY_RESUME_MODE must be full or actor_only" >&2; exit 1 ;; esac
  POLICY_RESUME_ARGS+=(--policy_resume_path "${POLICY_RESUME_PATH}" --policy_resume_mode "${POLICY_RESUME_MODE}")
elif [[ "${LOAD_REWARD_NORMALIZER}" == auto ]]; then
  LOAD_REWARD_NORMALIZER=false
fi

case "${SAVE_REPLAY_BUFFER}" in true|false) ;; *) echo "SAVE_REPLAY_BUFFER must be true or false" >&2; exit 1 ;; esac
case "${LOAD_REPLAY_BUFFER}" in true|false) ;; *) echo "LOAD_REPLAY_BUFFER must be true or false" >&2; exit 1 ;; esac
case "${LOAD_REWARD_NORMALIZER}" in true|false) ;; *) echo "LOAD_REWARD_NORMALIZER must be true, false, or auto" >&2; exit 1 ;; esac
case "${ACTOR_LEARNING_STARTS_UPDATES}" in
  ''|*[!0-9]*) echo "ACTOR_LEARNING_STARTS_UPDATES must be a non-negative integer" >&2; exit 1 ;;
esac
SAVE_REPLAY_ARGS=(--no-save_replay_buffer)
[[ "${SAVE_REPLAY_BUFFER}" == true ]] && SAVE_REPLAY_ARGS=(--save_replay_buffer)
LOAD_REPLAY_ARGS=(--no-load_replay_buffer)
[[ "${LOAD_REPLAY_BUFFER}" == true ]] && LOAD_REPLAY_ARGS=(--load_replay_buffer)
REWARD_OVERRIDES=(
  --overrides "world_model.reward_version=${REWARD_VERSION}"
  --overrides "world_model.reward_command_response_weight=${REWARD_COMMAND_RESPONSE_WEIGHT}"
  --overrides "world_model.reward_yaw_command_response_weight=${REWARD_YAW_COMMAND_RESPONSE_WEIGHT}"
  --overrides "world_model.reward_wrong_direction_weight=${REWARD_WRONG_DIRECTION_WEIGHT}"
  --overrides "world_model.reward_response_shortfall_weight=${REWARD_RESPONSE_SHORTFALL_WEIGHT}"
  --overrides "world_model.reward_response_floor=${REWARD_RESPONSE_FLOOR}"
  --overrides "world_model.reward_active_command_bias=${REWARD_ACTIVE_COMMAND_BIAS}"
  --overrides "world_model.reward_command_active_threshold=${REWARD_COMMAND_ACTIVE_THRESHOLD}"
  --overrides "world_model.reward_motion_gate_low=${REWARD_MOTION_GATE_LOW}"
  --overrides "world_model.reward_motion_gate_high=${REWARD_MOTION_GATE_HIGH}"
  --overrides "world_model.uncertainty_penalty_weight=${REWARD_UNCERTAINTY_PENALTY_WEIGHT}"
)
[[ -n "${REWARD_ACTION_RATE_L2}" ]] && REWARD_OVERRIDES+=(--overrides "world_model.reward_action_rate_l2=${REWARD_ACTION_RATE_L2}")
[[ -n "${REWARD_ACTION_SATURATION}" ]] && REWARD_OVERRIDES+=(--overrides "world_model.reward_action_saturation=${REWARD_ACTION_SATURATION}")
REWARD_OVERRIDES+=(--overrides "world_model.reward_action_saturation_threshold=${REWARD_ACTION_SATURATION_THRESHOLD}")
[[ -n "${REWARD_DOF_ACC_L2}" ]] && REWARD_OVERRIDES+=(--overrides "world_model.reward_dof_acc_l2=${REWARD_DOF_ACC_L2}")
[[ -n "${REWARD_DOF_TORQUES_L2}" ]] && REWARD_OVERRIDES+=(--overrides "world_model.reward_dof_torques_l2=${REWARD_DOF_TORQUES_L2}")

"${PYTHON_BIN}" scripts/reinforcement_learning/rwm_flashsac/train_flashsac_world_model_go2_proprioceptive.py \
  --config_path scripts/reinforcement_learning/rwm_flashsac/configs/go2_flashsac_rwm_proprioceptive.yaml \
  --model_resume_path "${MODEL_PATH}" \
  "${POLICY_RESUME_ARGS[@]}" \
  --dataset_path "${DATASET_PATH}" \
  --num_imagination_envs "${NUM_IMAGINATION_ENVS}" \
  --num_env_steps "${NUM_ENV_STEPS}" \
  --device "${DEVICE}" \
  --save_path "${OUTPUT_DIR}" \
  "${LOAD_REPLAY_ARGS[@]}" \
  "${SAVE_REPLAY_ARGS[@]}" \
  --overrides "seed=${SEED}" \
  --overrides "agent.load_reward_normalizer=${LOAD_REWARD_NORMALIZER}" \
  --overrides "agent.actor_learning_starts_updates=${ACTOR_LEARNING_STARTS_UPDATES}" \
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
  --overrides "world_model.interface_obs_bias_std=0.0" \
  "${REWARD_OVERRIDES[@]}" \
  "${TRACE_OVERRIDES[@]}"

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
  "${P_ID}" "${ACTION_NOISE_STD}" "${OBS_PROFILE}" "${OBS_SCALE}" "${SEED}" "${NUM_ENV_STEPS}" \
  "${TRACE_REPLAY_PATH}" "${TRACE_REPLAY_RATIO}" "${POLICY_RESUME_PATH}" \
  "${POLICY_RESUME_MODE}" "${LOAD_REPLAY_BUFFER}" "${ACTOR_LEARNING_STARTS_UPDATES}" \
  "${REWARD_VERSION}" "${REWARD_COMMAND_RESPONSE_WEIGHT}" "${REWARD_YAW_COMMAND_RESPONSE_WEIGHT}" \
  "${REWARD_WRONG_DIRECTION_WEIGHT}" "${REWARD_RESPONSE_SHORTFALL_WEIGHT}" \
  "${REWARD_RESPONSE_FLOOR}" "${REWARD_ACTIVE_COMMAND_BIAS}" "${REWARD_UNCERTAINTY_PENALTY_WEIGHT}" \
  "${REWARD_MOTION_GATE_LOW}" "${REWARD_MOTION_GATE_HIGH}" \
  "${REWARD_ACTION_RATE_L2}" "${REWARD_ACTION_SATURATION}" "${REWARD_ACTION_SATURATION_THRESHOLD}" \
  "${REWARD_DOF_ACC_L2}" "${REWARD_DOF_TORQUES_L2}" <<'PY'
import json
import sys
from pathlib import Path

(
    output, artifact, digest, model, model_digest, dataset, dataset_digest,
    p_id, action_noise, obs_profile, obs_scale, seed, env_steps,
    trace_replay_path, trace_replay_ratio, policy_resume_path,
    policy_resume_mode, load_replay_buffer, actor_learning_starts_updates,
    reward_version, reward_command_response, reward_yaw_command_response,
    reward_wrong_direction, reward_response_shortfall, reward_response_floor,
    reward_active_command_bias, reward_uncertainty, reward_gate_low, reward_gate_high,
    reward_action_rate_l2, reward_action_saturation, reward_action_saturation_threshold,
    reward_dof_acc_l2, reward_dof_torques_l2,
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
    "policy_resume_path": policy_resume_path or None,
    "policy_resume_mode": policy_resume_mode,
    "load_replay_buffer": load_replay_buffer == "true",
    "actor_learning_starts_updates": int(actor_learning_starts_updates),
    "trace": {
        "enabled": bool(trace_replay_path),
        "replay_path": trace_replay_path or None,
        "replay_ratio": float(trace_replay_ratio) if trace_replay_path else 0.0,
    },
    "reward": {
        "version": reward_version,
        "command_response_weight": float(reward_command_response),
        "yaw_command_response_weight": float(reward_yaw_command_response),
        "wrong_direction_weight": float(reward_wrong_direction),
        "response_shortfall_weight": float(reward_response_shortfall),
        "response_floor": float(reward_response_floor),
        "active_command_bias": float(reward_active_command_bias),
        "uncertainty_penalty_weight": float(reward_uncertainty),
        "motion_gate_low": float(reward_gate_low),
        "motion_gate_high": float(reward_gate_high),
        "action_rate_l2": None if reward_action_rate_l2 == "" else float(reward_action_rate_l2),
        "action_saturation": None if reward_action_saturation == "" else float(reward_action_saturation),
        "action_saturation_threshold": float(reward_action_saturation_threshold),
        "dof_acc_l2": None if reward_dof_acc_l2 == "" else float(reward_dof_acc_l2),
        "dof_torques_l2": None if reward_dof_torques_l2 == "" else float(reward_dof_torques_l2),
    },
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
