#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:?gpu id}"
JOB_ID="${2:?job id}"
RELATIVE_JOB_DIR="${3:?relative job directory}"

REPO_ROOT="${REPO_ROOT:-/data1/xjy/unitree_rl_mjlab_normal_fixstand_v2}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${REPO_ROOT}/logs/experiments/go2_normal_dr_3x3x4/normal_go2_fixstand_v2_moderate_seed0_20260712}"
JOB_DIR="${EXPERIMENT_ROOT}/${RELATIVE_JOB_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
exec bash "${REPO_ROOT}/scripts/reinforcement_learning/go2_normal_proprioceptive/run_stage_job.sh" \
  "${JOB_ID}" \
  "${JOB_DIR}" \
  "${JOB_DIR}/command.sh" \
  "${EXPERIMENT_ROOT}/queues/queue_events.log"
