#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
ARTIFACTS_ENV="${TRACE_V10_ARTIFACTS_ENV:-${REPO}/trace_v10_metric_artifacts.env}"
RUN_ROOT="${RUN_ROOT:-${REPO}/logs/experiments/go2_reward_v2_rr05/20260721_v2k_staged}"
V10_GPU_POOL="${V10_GPU_POOL:-5}"

[[ -s "${ARTIFACTS_ENV}" ]] || { echo "Missing artifacts env: ${ARTIFACTS_ENV}" >&2; exit 2; }
# shellcheck disable=SC1090
source "${ARTIFACTS_ENV}"

for name in REAL_RR05_DATASET REAL_RR05_MODEL REAL_RR05_WARMUP; do
  [[ -n "${!name:-}" ]] || { echo "Missing ${name} in ${ARTIFACTS_ENV}" >&2; exit 2; }
done

run_phase() {
  local phase_root="$1"
  local initial_policy="$2"
  local resume_mode="$3"
  local load_replay="$4"
  local load_normalizer="$5"
  local actor_learning_starts="$6"

  env \
    REPO="${REPO}" SIDE=real CONDITION=rr05 RUN_KIND=pilot \
    RUN_ROOT="${phase_root}" V10_GPU_POOL="${V10_GPU_POOL}" \
    DATASET_PATH="${REAL_RR05_DATASET}" MODEL_PATH="${REAL_RR05_MODEL}" \
    INITIAL_POLICY_PATH="${initial_policy}" CONTROL_NUM_ENV_STEPS=5000000 \
    POLICY_RESUME_MODE="${resume_mode}" LOAD_REPLAY_BUFFER="${load_replay}" \
    LOAD_REWARD_NORMALIZER="${load_normalizer}" \
    ACTOR_LEARNING_STARTS_UPDATES="${actor_learning_starts}" \
    REWARD_VERSION=v2_hierarchical \
    REWARD_COMMAND_RESPONSE_WEIGHT=4.0 REWARD_YAW_COMMAND_RESPONSE_WEIGHT=1.5 \
    REWARD_WRONG_DIRECTION_WEIGHT=-8.0 REWARD_RESPONSE_SHORTFALL_WEIGHT=-10.0 \
    REWARD_RESPONSE_FLOOR=0.45 REWARD_ACTIVE_COMMAND_BIAS=-0.30 \
    REWARD_ACTION_SATURATION=-0.2 REWARD_UNCERTAINTY_PENALTY_WEIGHT=-0.15 \
    REWARD_ACTION_RATE_L2=-0.005 REWARD_DOF_ACC_L2=-2e-8 REWARD_DOF_TORQUES_L2=-5e-6 \
    V10_EVAL_NUM_ENVS=64 V10_EVAL_STEPS=2400 V10_EVAL_SEEDS=901 \
    bash "${REPO}/scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_trace_v10_no_trace_control.sh"
}

CRITIC_ROOT="${RUN_ROOT}/critic_pretrain"
ACTOR_ROOT="${RUN_ROOT}/actor_finetune"

# A value beyond this phase's update budget keeps actor and temperature frozen.
run_phase "${CRITIC_ROOT}" "${REAL_RR05_WARMUP}" actor_only false false 1000000000

CRITIC_POLICY="$(<"${CRITIC_ROOT}/policy/stage/artifact_path.txt")"
[[ -s "${CRITIC_POLICY}/critic.pt" && -s "${CRITIC_POLICY}/replay_buffer.pt" \
   && -s "${CRITIC_POLICY}/reward_normalizer.pt" ]] || {
  echo "Incomplete critic-pretrain checkpoint: ${CRITIC_POLICY}" >&2
  exit 2
}

run_phase "${ACTOR_ROOT}" "${CRITIC_POLICY}" full true true 0

cat > "${RUN_ROOT}/completed.txt.new" <<EOF
protocol=go2_reward_v2_rr05_staged_pilot
critic_pretrain=${CRITIC_POLICY}
actor_finetune=$(<"${ACTOR_ROOT}/policy/stage/artifact_path.txt")
EOF
mv "${RUN_ROOT}/completed.txt.new" "${RUN_ROOT}/completed.txt"
