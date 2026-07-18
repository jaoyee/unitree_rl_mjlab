#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${SIDE:?Set SIDE=sim or SIDE=real}"
: "${CONDITION:?Set CONDITION=g0/rr05/p5/rr03/p75}"
: "${DATASET_PATH:?Set the condition-specific 25K dataset path}"
: "${RUN_ROOT:?Set the condition-specific experiment root}"
: "${CUDA_VISIBLE_DEVICES:?Set exactly one physical GPU}"

case "${SIDE}" in sim|real) ;; *) echo "Bad SIDE=${SIDE}" >&2; exit 2 ;; esac
case "${CONDITION}" in g0|rr05|p5|rr03|p75) ;; *) echo "Bad CONDITION=${CONDITION}" >&2; exit 2 ;; esac

PY="${REPO}/.venv/bin/python"
LOG="${RUN_ROOT}/condition_full.log"
mkdir -p "${RUN_ROOT}"
cd "${REPO}"

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${LOG}"
}

if [[ ! -s "${RUN_ROOT}/scorer_ready.env" ]]; then
  log "starting initial scorer for ${SIDE}/${CONDITION}"
  SIDE="${SIDE}" DATASET_PATH="${DATASET_PATH}" SIDE_ROOT="${RUN_ROOT}" \
    bash scripts/reinforcement_learning/go2_sim_gap_aligned/prepare_go2_trace_v5_initial_scorer.sh \
    2>&1 | tee -a "${LOG}" &
  scorer_pid=$!
else
  scorer_pid=""
fi

if [[ ! -s "${RUN_ROOT}/prepare_completed.txt" ]]; then
  log "starting RWM/common-warmup/baseline for ${SIDE}/${CONDITION}"
  SIDE="${SIDE}" DATASET_PATH="${DATASET_PATH}" RUN_ROOT="${RUN_ROOT}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    bash scripts/reinforcement_learning/go2_sim_gap_aligned/prepare_go2_trace_v5_side.sh \
    2>&1 | tee -a "${LOG}"
else
  log "prepare already completed for ${SIDE}/${CONDITION}"
fi

if [[ -n "${scorer_pid:-}" ]]; then
  log "waiting initial scorer for ${SIDE}/${CONDITION}"
  wait "${scorer_pid}"
fi

[[ -s "${RUN_ROOT}/ready.env" ]] || { echo "Missing ready.env" >&2; exit 2; }
[[ -s "${RUN_ROOT}/scorer_ready.env" ]] || { echo "Missing scorer_ready.env" >&2; exit 2; }
# shellcheck disable=SC1090
source "${RUN_ROOT}/ready.env"
# shellcheck disable=SC1090
source "${RUN_ROOT}/scorer_ready.env"

for branch in r10 r25; do
  branch_root="${RUN_ROOT}/trace_${branch}"
  if [[ -s "${branch_root}/completed.txt" ]]; then
    log "TRACE ${branch} already completed for ${SIDE}/${CONDITION}"
    continue
  fi
  log "starting TRACE ${branch} for ${SIDE}/${CONDITION}"
  SIDE="${SIDE}" CONDITION="${CONDITION}" BRANCH_ID="${branch}" RUN_ROOT="${branch_root}" \
    DATASET_PATH="${DATASET_PATH}" MODEL_PATH="${MODEL_PATH}" RESET_DATASET_PATH="${DATASET_PATH}" \
    INITIAL_POLICY_PATH="${INITIAL_POLICY_PATH}" INITIAL_SCORER_PATH="${INITIAL_SCORER_PATH}" \
    INITIAL_PAIRS_PATH="${INITIAL_PAIRS_PATH}" INITIAL_LABELS_PATH="${INITIAL_LABELS_PATH}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    bash scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_trace_v5_dynamic_branch.sh \
    2>&1 | tee -a "${LOG}"
done

printf 'side=%s\ncondition=%s\nstatus=completed\n' "${SIDE}" "${CONDITION}" \
  > "${RUN_ROOT}/condition_full_completed.txt"
log "completed ${SIDE}/${CONDITION}"
