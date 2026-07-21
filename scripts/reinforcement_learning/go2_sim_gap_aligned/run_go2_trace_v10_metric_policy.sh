#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${SELECTION:?Set metric or random}"
: "${SIDE:?Set sim or real}"
: "${CONDITION:?Set condition}"
: "${RUN_ROOT:?Set branch root}"
: "${DATASET_PATH:?Set dataset}"
: "${MODEL_PATH:?Set frozen RWM}"
: "${INITIAL_POLICY_PATH:?Set common warmup}"
: "${TRACE_MANIFEST:?Set staged V10 replay manifest}"
: "${V10_GPU_POOL:?Set physical GPU pool}"
case "${SELECTION}" in metric|random) ;; *) echo "Bad selection ${SELECTION}" >&2; exit 2 ;; esac
PY="${PYTHON_BIN:-${REPO}/.venv/bin/python}"
PROTOCOL="${REPO}/scripts/reinforcement_learning/rwm_trace/trace_v10_protocol.json"
GPU_EXEC="${REPO}/scripts/reinforcement_learning/go2_sim_gap_aligned/run_v10_gpu_stage.sh"
read -r STEPS IMAG_ENVS RATIO <<< "$(${PY} - "${PROTOCOL}" <<'PY'
import json, sys
p=json.load(open(sys.argv[1])); print(p["training"]["environment_steps_per_refresh"], p["training"]["imagination_environments"], p["replay"]["ratio"])
PY
)"
POLICY_RESUME_MODE="${POLICY_RESUME_MODE:-full}"
LOAD_REPLAY_BUFFER="${LOAD_REPLAY_BUFFER:-true}"
LOAD_REWARD_NORMALIZER="${LOAD_REWARD_NORMALIZER:-true}"
REWARD_VERSION="${REWARD_VERSION:-v1}"
REWARD_COMMAND_RESPONSE_WEIGHT="${REWARD_COMMAND_RESPONSE_WEIGHT:-2.0}"
REWARD_YAW_COMMAND_RESPONSE_WEIGHT="${REWARD_YAW_COMMAND_RESPONSE_WEIGHT:-1.0}"
REWARD_WRONG_DIRECTION_WEIGHT="${REWARD_WRONG_DIRECTION_WEIGHT:--2.0}"
REWARD_UNCERTAINTY_PENALTY_WEIGHT="${REWARD_UNCERTAINTY_PENALTY_WEIGHT:--2.0}"
REWARD_ACTION_RATE_L2="${REWARD_ACTION_RATE_L2:-}"
REWARD_DOF_ACC_L2="${REWARD_DOF_ACC_L2:-}"
REWARD_DOF_TORQUES_L2="${REWARD_DOF_TORQUES_L2:-}"
REWARD_COMMAND_ACTIVE_THRESHOLD="${REWARD_COMMAND_ACTIVE_THRESHOLD:-0.02}"
REWARD_MOTION_GATE_LOW="${REWARD_MOTION_GATE_LOW:-0.05}"
REWARD_MOTION_GATE_HIGH="${REWARD_MOTION_GATE_HIGH:-0.30}"
for path in "${DATASET_PATH}" "${MODEL_PATH}" "${INITIAL_POLICY_PATH}/actor.pt" \
  "${INITIAL_POLICY_PATH}/replay_buffer.pt" "${INITIAL_POLICY_PATH}/reward_normalizer.pt" \
  "${TRACE_MANIFEST}"; do
  [[ -s "${path}" ]] || { echo "Missing ${SELECTION} policy input ${path}" >&2; exit 2; }
done
mkdir -p "${RUN_ROOT}/policy" "${RUN_ROOT}/behavior"
cd "${REPO}"
if [[ ! -s "${RUN_ROOT}/policy/stage/summary.json" ]]; then
  V10_GPU_POOL="${V10_GPU_POOL}" "${GPU_EXEC}" env \
    STAGE_DIR="${RUN_ROOT}/policy/stage" OUTPUT_DIR="${RUN_ROOT}/policy/run" \
    DATASET_PATH="${DATASET_PATH}" MODEL_PATH="${MODEL_PATH}" P_ID=P0 DEVICE=cuda:0 \
    SEED=300 NUM_ENV_STEPS="${STEPS}" NUM_IMAGINATION_ENVS="${IMAG_ENVS}" \
    POLICY_RESUME_PATH="${INITIAL_POLICY_PATH}" POLICY_RESUME_MODE="${POLICY_RESUME_MODE}" \
    SAVE_REPLAY_BUFFER=true \
    LOAD_REPLAY_BUFFER="${LOAD_REPLAY_BUFFER}" \
    LOAD_REWARD_NORMALIZER="${LOAD_REWARD_NORMALIZER}" TRACE_REPLAY_PATH="${TRACE_MANIFEST}" \
    TRACE_REPLAY_RATIO="${RATIO}" LIN_VEL_X_RANGE='-0.5 0.5' \
    REWARD_VERSION="${REWARD_VERSION}" \
    REWARD_COMMAND_RESPONSE_WEIGHT="${REWARD_COMMAND_RESPONSE_WEIGHT}" \
    REWARD_YAW_COMMAND_RESPONSE_WEIGHT="${REWARD_YAW_COMMAND_RESPONSE_WEIGHT}" \
    REWARD_WRONG_DIRECTION_WEIGHT="${REWARD_WRONG_DIRECTION_WEIGHT}" \
    REWARD_UNCERTAINTY_PENALTY_WEIGHT="${REWARD_UNCERTAINTY_PENALTY_WEIGHT}" \
    REWARD_ACTION_RATE_L2="${REWARD_ACTION_RATE_L2}" \
    REWARD_DOF_ACC_L2="${REWARD_DOF_ACC_L2}" \
    REWARD_DOF_TORQUES_L2="${REWARD_DOF_TORQUES_L2}" \
    REWARD_COMMAND_ACTIVE_THRESHOLD="${REWARD_COMMAND_ACTIVE_THRESHOLD}" \
    REWARD_MOTION_GATE_LOW="${REWARD_MOTION_GATE_LOW}" \
    REWARD_MOTION_GATE_HIGH="${REWARD_MOTION_GATE_HIGH}" \
    LIN_VEL_Y_RANGE='-0.2 0.2' ANG_VEL_Z_RANGE='-0.4 0.4' \
    bash scripts/reinforcement_learning/go2_normal_proprioceptive/04_train_normal_policy.sh \
    2>&1 | tee -a "${RUN_ROOT}/run.log"
fi
POLICY="$(<"${RUN_ROOT}/policy/stage/artifact_path.txt")"
[[ -s "${POLICY}/replay_buffer.pt" ]] || { echo "${SELECTION} final RWM buffer missing" >&2; exit 2; }
"${PY}" scripts/reinforcement_learning/rwm_trace/v10_replay_manifest.py commit \
  --staged "${TRACE_MANIFEST}" --output "${RUN_ROOT}/replay/committed_manifest.json" \
  --policy-checkpoint "${POLICY}"
V10_GPU_POOL="${V10_GPU_POOL}" "${GPU_EXEC}" env REPO="${REPO}" GAP_ID="${CONDITION}" \
  CHECKPOINT="${POLICY}" OUTPUT_ROOT="${RUN_ROOT}/behavior" DEVICE=cuda:0 \
  NUM_ENVS=64 STEPS=2400 SEEDS='901 902 903' MODES=clean \
  bash scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_command_eval_v4.sh \
  2>&1 | tee -a "${RUN_ROOT}/run.log"
cat > "${RUN_ROOT}/completed.txt.new" <<EOF
protocol=go2_trace_v10_metric_pilot_v1
selection=${SELECTION}
side=${SIDE}
condition=${CONDITION}
policy=${POLICY}
training_steps=${STEPS}
EOF
mv "${RUN_ROOT}/completed.txt.new" "${RUN_ROOT}/completed.txt"
