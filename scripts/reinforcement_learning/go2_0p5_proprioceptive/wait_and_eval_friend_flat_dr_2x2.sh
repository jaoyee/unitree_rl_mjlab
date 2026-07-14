#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

: "${FORMAL_RUN_GROUP:?Set FORMAL_RUN_GROUP to the active strict-2x2 run group.}"

FORMAL_QUEUE_ROOT="${FORMAL_QUEUE_ROOT:-logs/queues/go2_0p5_friend_flat_dr/${FORMAL_RUN_GROUP}}"
FORMAL_ARTIFACT_ROOT="${FORMAL_ARTIFACT_ROOT:-logs/experiments/go2_0p5_friend_flat_dr/${FORMAL_RUN_GROUP}}"
EVAL_ID="${EVAL_ID:-selected_dr_eval_$(date +%Y%m%d_%H%M%S)}"
EVAL_ROOT="${EVAL_ROOT:-logs/evals/go2_0p5_friend_flat_dr/${FORMAL_RUN_GROUP}/${EVAL_ID}}"
GPU_POOL="${GPU_POOL:-1 3 4 5}"
GPU_POOL="${GPU_POOL//,/ }"
RANDOMIZATION_COMPONENTS="${RANDOMIZATION_COMPONENTS:-all}"
RANDOMIZATION_SCALE="${RANDOMIZATION_SCALE:-1.0}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-43200}"
STATE_FILE="${EVAL_ROOT}/watcher_state.txt"

mkdir -p "${EVAL_ROOT}"

write_state() {
  local stage="$1" status="$2"
  {
    echo "time=$(date --iso-8601=seconds)"
    echo "formal_run_group=${FORMAL_RUN_GROUP}"
    echo "stage=${stage}"
    echo "status=${status}"
    echo "randomization_components=${RANDOMIZATION_COMPONENTS}"
    echo "randomization_scale=${RANDOMIZATION_SCALE}"
  } > "${STATE_FILE}"
}

deadline=$(( $(date +%s) + MAX_WAIT_SECONDS ))
write_state wait_for_formal running
while true; do
  if find "${FORMAL_QUEUE_ROOT}/jobs" -name state.txt -type f \
      -exec grep -l '^status=failed$' {} + 2>/dev/null | grep -q .; then
    write_state wait_for_formal formal_failed
    exit 1
  fi

  checkpoint_count=0
  if [[ -d "${FORMAL_ARTIFACT_ROOT}/policies" ]]; then
    checkpoint_count="$(find "${FORMAL_ARTIFACT_ROOT}/policies" -mindepth 3 -maxdepth 3 \
      -type d -name step48828 | wc -l)"
  fi
  if [[ "${checkpoint_count}" -eq 4 ]] \
      && grep -Eq '^(completed_at=|All strict 2x2 jobs completed)' \
        "${FORMAL_QUEUE_ROOT}/launch_summary.txt" 2>/dev/null; then
    break
  fi

  if (( $(date +%s) >= deadline )); then
    write_state wait_for_formal timed_out
    exit 124
  fi
  sleep "${POLL_SECONDS}"
done

write_state evaluate running
RUN_GROUP="${FORMAL_RUN_GROUP}" \
EVAL_ID="${EVAL_ID}" \
EVAL_ROOT="${EVAL_ROOT}" \
GPU_POOL="${GPU_POOL}" \
RANDOMIZATION_COMPONENTS="${RANDOMIZATION_COMPONENTS}" \
RANDOMIZATION_SCALE="${RANDOMIZATION_SCALE}" \
bash "${SCRIPT_DIR}/eval_friend_flat_dr_2x2.sh"
write_state completed completed
