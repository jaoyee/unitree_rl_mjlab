#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${CONDITION:?Set condition}"
: "${SIDE:?Set sim or real}"
: "${RUN_ROOT:?Set a new control root}"
: "${DATASET_PATH:?Set dataset}"
: "${MODEL_PATH:?Set frozen RWM}"
: "${INITIAL_POLICY_PATH:?Set common warmup}"
: "${V10_GPU_POOL:?Set physical GPU pool}"
PY="${PYTHON_BIN:-${REPO}/.venv/bin/python}"
PROTOCOL="${REPO}/scripts/reinforcement_learning/rwm_trace/trace_v10_protocol.json"
GPU_EXEC="${REPO}/scripts/reinforcement_learning/go2_sim_gap_aligned/run_v10_gpu_stage.sh"
STEPS="$(${PY} -c 'import json,sys; print(json.load(open(sys.argv[1]))["training"]["environment_steps_per_refresh"])' "${PROTOCOL}")"
IMAG_ENVS="$(${PY} -c 'import json,sys; print(json.load(open(sys.argv[1]))["training"]["imagination_environments"])' "${PROTOCOL}")"
RUN_KIND="${RUN_KIND:-pilot}"
CONTROL_NUM_ENV_STEPS="${CONTROL_NUM_ENV_STEPS:-}"
SEED="${SEED:-300}"
V10_EVAL_NUM_ENVS="${V10_EVAL_NUM_ENVS:-64}"
V10_EVAL_STEPS="${V10_EVAL_STEPS:-2400}"
POLICY_RESUME_MODE="${POLICY_RESUME_MODE:-full}"
LOAD_REPLAY_BUFFER="${LOAD_REPLAY_BUFFER:-true}"
LOAD_REWARD_NORMALIZER="${LOAD_REWARD_NORMALIZER:-true}"
ACTOR_LEARNING_STARTS_UPDATES="${ACTOR_LEARNING_STARTS_UPDATES:-0}"
REWARD_VERSION="${REWARD_VERSION:-v1}"
REWARD_COMMAND_RESPONSE_WEIGHT="${REWARD_COMMAND_RESPONSE_WEIGHT:-2.0}"
REWARD_YAW_COMMAND_RESPONSE_WEIGHT="${REWARD_YAW_COMMAND_RESPONSE_WEIGHT:-1.0}"
REWARD_WRONG_DIRECTION_WEIGHT="${REWARD_WRONG_DIRECTION_WEIGHT:--2.0}"
REWARD_RESPONSE_SHORTFALL_WEIGHT="${REWARD_RESPONSE_SHORTFALL_WEIGHT:--6.0}"
REWARD_RESPONSE_FLOOR="${REWARD_RESPONSE_FLOOR:-0.30}"
REWARD_ACTIVE_COMMAND_BIAS="${REWARD_ACTIVE_COMMAND_BIAS:--0.20}"
REWARD_UNCERTAINTY_PENALTY_WEIGHT="${REWARD_UNCERTAINTY_PENALTY_WEIGHT:--2.0}"
REWARD_ACTION_RATE_L2="${REWARD_ACTION_RATE_L2:-}"
REWARD_ACTION_SATURATION="${REWARD_ACTION_SATURATION:-}"
REWARD_ACTION_SATURATION_THRESHOLD="${REWARD_ACTION_SATURATION_THRESHOLD:-0.9}"
REWARD_DOF_ACC_L2="${REWARD_DOF_ACC_L2:-}"
REWARD_DOF_TORQUES_L2="${REWARD_DOF_TORQUES_L2:-}"
REWARD_COMMAND_ACTIVE_THRESHOLD="${REWARD_COMMAND_ACTIVE_THRESHOLD:-0.02}"
REWARD_MOTION_GATE_LOW="${REWARD_MOTION_GATE_LOW:-0.05}"
REWARD_MOTION_GATE_HIGH="${REWARD_MOTION_GATE_HIGH:-0.30}"
case "${RUN_KIND}" in
  pilot) ;;
  formal) STEPS=$((STEPS * 8)) ;;
  *) echo "Bad RUN_KIND=${RUN_KIND}" >&2; exit 2 ;;
esac
if [[ -n "${CONTROL_NUM_ENV_STEPS}" ]]; then
  case "${CONTROL_NUM_ENV_STEPS}" in
    ''|*[!0-9]*) echo "CONTROL_NUM_ENV_STEPS must be a positive integer" >&2; exit 2 ;;
  esac
  [[ "${CONTROL_NUM_ENV_STEPS}" -gt 0 ]] || { echo "CONTROL_NUM_ENV_STEPS must be positive" >&2; exit 2; }
  STEPS="${CONTROL_NUM_ENV_STEPS}"
fi
for path in "${DATASET_PATH}" "${MODEL_PATH}" "${INITIAL_POLICY_PATH}/actor.pt" \
  "${INITIAL_POLICY_PATH}/replay_buffer.pt" "${INITIAL_POLICY_PATH}/reward_normalizer.pt"; do
  [[ -s "${path}" ]] || { echo "Missing no-TRACE control input ${path}" >&2; exit 2; }
done
mkdir -p "${RUN_ROOT}/policy"
cd "${REPO}"
if [[ ! -s "${RUN_ROOT}/policy/stage/summary.json" ]]; then
  V10_GPU_POOL="${V10_GPU_POOL}" "${GPU_EXEC}" env \
    STAGE_DIR="${RUN_ROOT}/policy/stage" OUTPUT_DIR="${RUN_ROOT}/policy/run" \
    DATASET_PATH="${DATASET_PATH}" MODEL_PATH="${MODEL_PATH}" P_ID=P0 DEVICE=cuda:0 \
    SEED="${SEED}" NUM_ENV_STEPS="${STEPS}" NUM_IMAGINATION_ENVS="${IMAG_ENVS}" \
    POLICY_RESUME_PATH="${INITIAL_POLICY_PATH}" \
    POLICY_RESUME_MODE="${POLICY_RESUME_MODE}" \
    SAVE_REPLAY_BUFFER=true LOAD_REPLAY_BUFFER="${LOAD_REPLAY_BUFFER}" \
    LOAD_REWARD_NORMALIZER="${LOAD_REWARD_NORMALIZER}" TRACE_REPLAY_PATH='' \
    ACTOR_LEARNING_STARTS_UPDATES="${ACTOR_LEARNING_STARTS_UPDATES}" \
    REWARD_VERSION="${REWARD_VERSION}" \
    REWARD_COMMAND_RESPONSE_WEIGHT="${REWARD_COMMAND_RESPONSE_WEIGHT}" \
    REWARD_YAW_COMMAND_RESPONSE_WEIGHT="${REWARD_YAW_COMMAND_RESPONSE_WEIGHT}" \
    REWARD_WRONG_DIRECTION_WEIGHT="${REWARD_WRONG_DIRECTION_WEIGHT}" \
    REWARD_RESPONSE_SHORTFALL_WEIGHT="${REWARD_RESPONSE_SHORTFALL_WEIGHT}" \
    REWARD_RESPONSE_FLOOR="${REWARD_RESPONSE_FLOOR}" \
    REWARD_ACTIVE_COMMAND_BIAS="${REWARD_ACTIVE_COMMAND_BIAS}" \
    REWARD_UNCERTAINTY_PENALTY_WEIGHT="${REWARD_UNCERTAINTY_PENALTY_WEIGHT}" \
    REWARD_ACTION_RATE_L2="${REWARD_ACTION_RATE_L2}" \
    REWARD_ACTION_SATURATION="${REWARD_ACTION_SATURATION}" \
    REWARD_ACTION_SATURATION_THRESHOLD="${REWARD_ACTION_SATURATION_THRESHOLD}" \
    REWARD_DOF_ACC_L2="${REWARD_DOF_ACC_L2}" \
    REWARD_DOF_TORQUES_L2="${REWARD_DOF_TORQUES_L2}" \
    REWARD_COMMAND_ACTIVE_THRESHOLD="${REWARD_COMMAND_ACTIVE_THRESHOLD}" \
    REWARD_MOTION_GATE_LOW="${REWARD_MOTION_GATE_LOW}" \
    REWARD_MOTION_GATE_HIGH="${REWARD_MOTION_GATE_HIGH}" \
    LIN_VEL_X_RANGE='-0.5 0.5' LIN_VEL_Y_RANGE='-0.2 0.2' ANG_VEL_Z_RANGE='-0.4 0.4' \
    bash scripts/reinforcement_learning/go2_normal_proprioceptive/04_train_normal_policy.sh
fi
POLICY="$(<"${RUN_ROOT}/policy/stage/artifact_path.txt")"
[[ -s "${POLICY}/replay_buffer.pt" ]] || { echo "Control final buffer missing" >&2; exit 2; }
V10_GPU_POOL="${V10_GPU_POOL}" "${GPU_EXEC}" env REPO="${REPO}" GAP_ID="${CONDITION}" \
  CHECKPOINT="${POLICY}" OUTPUT_ROOT="${RUN_ROOT}/behavior" DEVICE=cuda:0 \
  NUM_ENVS="${V10_EVAL_NUM_ENVS}" STEPS="${V10_EVAL_STEPS}" \
  SEEDS="${V10_EVAL_SEEDS:-901}" MODES=clean \
  bash scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_command_eval_v4.sh
cat > "${RUN_ROOT}/completed.txt.new" <<EOF
protocol=go2_trace_v10_no_trace_control
side=${SIDE}
condition=${CONDITION}
policy=${POLICY}
training_steps=${STEPS}
EOF
mv "${RUN_ROOT}/completed.txt.new" "${RUN_ROOT}/completed.txt"
"${PY}" - "${RUN_ROOT}/v10_baseline_manifest.json" "${PROTOCOL}" "${SIDE}" "${CONDITION}" \
  "${DATASET_PATH}" "${MODEL_PATH}" "${INITIAL_POLICY_PATH}" "${POLICY}" "${STEPS}" \
  "${SEED}" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
out, protocol, side, condition, dataset, model, warmup, policy, steps, seed = sys.argv[1:]
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()
repo=Path(protocol).resolve().parents[3]
payload={
 "schema":"go2_trace_v10_baseline_manifest_v1", "protocol_sha256":sha(protocol),
 "side":side, "condition":condition, "dataset":str(Path(dataset).resolve()),
 "dataset_sha256":sha(dataset), "model":str(Path(model).resolve()), "model_sha256":sha(model),
 "warmup":str(Path(warmup).resolve()), "warmup_actor_sha256":sha(Path(warmup)/"actor.pt"),
 "warmup_replay_buffer_sha256":sha(Path(warmup)/"replay_buffer.pt"), "seed":int(seed),
 "training_steps":int(steps), "reward_sha256":sha(repo/"src/tasks/rwm_velocity/mdp/rewards.py"),
 "policy":str(Path(policy).resolve()), "policy_actor_sha256":sha(Path(policy)/"actor.pt")}
tmp=out+f".tmp.{os.getpid()}"; Path(tmp).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
os.replace(tmp,out)
PY
