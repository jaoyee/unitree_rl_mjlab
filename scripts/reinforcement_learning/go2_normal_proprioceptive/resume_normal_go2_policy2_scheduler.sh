#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data1/xjy/unitree_rl_mjlab_normal_fixstand_v2}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${REPO_ROOT}/logs/experiments/go2_normal_dr_3x3x4/normal_go2_fixstand_v2_moderate_seed0_20260712}"

export GPU_POOL="${GPU_POOL:-1 2 3 4 5 7}"
export GPU_MAX_MEMORY_USED_MIB="${GPU_MAX_MEMORY_USED_MIB:-6000}"
export GPU_MAX_UTILIZATION="${GPU_MAX_UTILIZATION:-20}"
export POLICY_JOBS_PER_GPU="${POLICY_JOBS_PER_GPU:-2}"
export POLICY_SECOND_SLOT_MAX_MEMORY_USED_MIB="${POLICY_SECOND_SLOT_MAX_MEMORY_USED_MIB:-16000}"
export POLL_SECONDS="${POLL_SECONDS:-30}"

exec bash "${REPO_ROOT}/scripts/reinforcement_learning/go2_normal_proprioceptive/normal_go2_dr_scheduler.sh" \
  "${EXPERIMENT_ROOT}"
