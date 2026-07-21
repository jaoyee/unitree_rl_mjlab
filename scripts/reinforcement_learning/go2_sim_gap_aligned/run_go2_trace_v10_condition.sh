#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${SIDE:?Set side}"
: "${CONDITION:?Set condition}"
: "${RUN_ROOT:?Set branch root}"
: "${DATASET_PATH:?Set dataset}"
: "${MODEL_PATH:?Set frozen RWM}"
: "${INITIAL_POLICY_PATH:?Set warmup policy}"
: "${INITIAL_SCORER_ROOT:?Set initial scorer root}"
: "${V10_GPU_POOL:?Set GPU pool}"
mkdir -p "${RUN_ROOT}" "${INITIAL_SCORER_ROOT}"
if [[ ! -s "${INITIAL_SCORER_ROOT}/ready.env" ]]; then
  REPO="${REPO}" DATASET_PATH="${DATASET_PATH}" CONDITION="${CONDITION}" \
    OUTPUT_ROOT="${INITIAL_SCORER_ROOT}" CODEX_LOCK="${CODEX_LOCK:-$(dirname "${RUN_ROOT}")/codex_feedback.lock}" \
    bash "${REPO}/scripts/reinforcement_learning/go2_sim_gap_aligned/prepare_go2_trace_v10_initial_scorer.sh"
fi
exec env REPO="${REPO}" SIDE="${SIDE}" CONDITION="${CONDITION}" RUN_ROOT="${RUN_ROOT}" \
  DATASET_PATH="${DATASET_PATH}" MODEL_PATH="${MODEL_PATH}" \
  INITIAL_POLICY_PATH="${INITIAL_POLICY_PATH}" INITIAL_SCORER_ROOT="${INITIAL_SCORER_ROOT}" \
  V10_GPU_POOL="${V10_GPU_POOL}" RUN_KIND="${RUN_KIND:-formal}" \
  CODEX_LOCK="${CODEX_LOCK:-$(dirname "${RUN_ROOT}")/codex_feedback.lock}" \
  bash "${REPO}/scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_trace_v10_branch.sh"
