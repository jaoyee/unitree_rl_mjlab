#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data1/xjy/unitree_rl_mjlab_normal_fixstand_v2}"
PRUNER="${REPO_ROOT}/scripts/reinforcement_learning/go2_normal_proprioceptive/prune_experiment_checkpoints.sh"
LOG_FILE="${REPO_ROOT}/logs/checkpoint_pruner.log"

while true; do
  "${PRUNER}" >> "${LOG_FILE}" 2>&1
  sleep 300
done
