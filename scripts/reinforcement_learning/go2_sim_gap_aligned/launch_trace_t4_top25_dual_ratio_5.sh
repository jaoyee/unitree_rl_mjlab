#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
BASE_ROOT="${BASE_ROOT:-${REPO_ROOT}/logs/experiments/go2_sim_gap_aligned/20260714_aligned_g0d0_v1}"
TRACE_BATCH_ROOT="${TRACE_BATCH_ROOT:-${BASE_ROOT}/trace_t4_top25_dual_${RUN_ID}}"

mkdir -p "${TRACE_BATCH_ROOT}"

launch() {
  local gap_id="$1" gpu="$2"
  local session="trace_t4_${gap_id}_${RUN_ID}"
  local run_root="${TRACE_BATCH_ROOT}/${gap_id}"
  mkdir -p "${run_root}"
  cat > "${run_root}/command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export GAP_ID="${gap_id}"
export RUN_ROOT="${run_root}"
export BASE_ROOT="${BASE_ROOT}"
export TRACE_BATCH_ROOT="${TRACE_BATCH_ROOT}"
export TRACE_ACTION_TEMPERATURE=4.0
export TRACE_SELECT_RATIO=0.25
bash scripts/reinforcement_learning/go2_sim_gap_aligned/run_sim_gap_trace_t4_dual_ratio_pipeline.sh
EOF
  chmod +x "${run_root}/command.sh"
  tmux new-session -d -s "${session}" \
    "bash '${run_root}/command.sh' 2>&1 | tee -a '${run_root}/tmux.log'"
  printf '%s|gpu=%s|session=%s|root=%s\n' "${gap_id}" "${gpu}" "${session}" "${run_root}" \
    | tee -a "${TRACE_BATCH_ROOT}/sessions.txt"
}

launch g0 2
launch rr05 4
launch rr03 5
launch p5 6
launch p75 7

printf '%s\n' "${TRACE_BATCH_ROOT}" > "${BASE_ROOT}/latest_trace_t4_top25_dual_root.txt"
echo "trace_batch_root=${TRACE_BATCH_ROOT}"
