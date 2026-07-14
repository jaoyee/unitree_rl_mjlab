#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${REPO_ROOT}/logs/experiments/go2_sim_gap_aligned/${RUN_ID}}"

mkdir -p "${EXPERIMENT_ROOT}"

launch() {
  local gap_id="$1" gpu="$2" payload="$3" strength="$4"
  local session="sim_gap_${gap_id}_${RUN_ID}"
  local run_root="${EXPERIMENT_ROOT}/${gap_id}"
  mkdir -p "${run_root}"
  cat > "${run_root}/command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export GAP_ID="${gap_id}"
export RUN_ROOT="${run_root}"
export PAYLOAD_KG="${payload}"
export RR_STRENGTH="${strength}"
bash scripts/reinforcement_learning/go2_sim_gap_aligned/run_sim_gap_rwm_pipeline.sh
EOF
  chmod +x "${run_root}/command.sh"
  tmux new-session -d -s "${session}" "bash '${run_root}/command.sh'"
  printf '%s|gpu=%s|session=%s|root=%s\n' "${gap_id}" "${gpu}" "${session}" "${run_root}" \
    | tee -a "${EXPERIMENT_ROOT}/sessions.txt"
}

launch g0 2 0.0 1.0
launch rr05 3 0.0 0.5
launch rr03 4 0.0 0.3
launch p5 5 5.0 1.0
launch p75 6 7.5 1.0

printf '%s\n' "${EXPERIMENT_ROOT}" > "${REPO_ROOT}/logs/experiments/go2_sim_gap_aligned/latest_root.txt"
echo "experiment_root=${EXPERIMENT_ROOT}"
