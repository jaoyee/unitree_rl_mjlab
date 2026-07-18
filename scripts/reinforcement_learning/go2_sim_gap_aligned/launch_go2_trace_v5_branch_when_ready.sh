#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${SIDE:?Set SIDE=sim or SIDE=real}"
: "${BRANCH_ID:?Set BRANCH_ID=r10 or BRANCH_ID=r25}"
: "${SIDE_ROOT:?Set the V5 side root containing ready.env}"
: "${CUDA_VISIBLE_DEVICES:?Set exactly one physical GPU}"
CONDITION="${CONDITION:-}"
if [[ -n "${CONDITION}" ]]; then
  case "${CONDITION}" in g0|rr05|p5|rr03|p75) ;; *) echo "Bad CONDITION=${CONDITION}" >&2; exit 2 ;; esac
else
  case "${SIDE}" in
    sim) RESET_PARTS_ROOT="${REPO}/logs/rwm_datasets_v5/sim_exact_40_20_20_10_10/parts" ;;
    real) RESET_PARTS_ROOT="${REPO}/logs/rwm_datasets_v4/real_v4/pooled_40_20_20_10_10/parts" ;;
    *) exit 2 ;;
  esac
fi

until [[ -s "${SIDE_ROOT}/ready.env" && -s "${SIDE_ROOT}/scorer_ready.env" ]]; do sleep 20; done
# shellcheck disable=SC1090
source "${SIDE_ROOT}/ready.env"
# shellcheck disable=SC1090
source "${SIDE_ROOT}/scorer_ready.env"
RUN_ROOT="${SIDE_ROOT}/trace_${BRANCH_ID}"
env_args=(
  SIDE="${SIDE}" BRANCH_ID="${BRANCH_ID}" RUN_ROOT="${RUN_ROOT}"
  DATASET_PATH="${DATASET_PATH}" MODEL_PATH="${MODEL_PATH}"
  INITIAL_POLICY_PATH="${INITIAL_POLICY_PATH}"
  INITIAL_SCORER_PATH="${INITIAL_SCORER_PATH}" INITIAL_PAIRS_PATH="${INITIAL_PAIRS_PATH}"
  INITIAL_LABELS_PATH="${INITIAL_LABELS_PATH}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
)
if [[ -n "${CONDITION}" ]]; then
  env_args+=(CONDITION="${CONDITION}" RESET_DATASET_PATH="${DATASET_PATH}")
else
  env_args+=(RESET_PARTS_ROOT="${RESET_PARTS_ROOT}")
fi
exec env "${env_args[@]}" \
  bash "${REPO}/scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_trace_v5_dynamic_branch.sh"
