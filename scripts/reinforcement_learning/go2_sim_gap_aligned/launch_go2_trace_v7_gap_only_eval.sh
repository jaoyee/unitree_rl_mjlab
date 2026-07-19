#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${FORMAL_ROOT:?Set FORMAL_ROOT to a corrected V7 formal experiment root}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EVAL_ROOT="${EVAL_ROOT:-${REPO}/logs/evaluations/go2_trace_v7/${RUN_ID}}"
SESSION="v7e_default_gap_only"
mkdir -p "${EVAL_ROOT}"
tmux has-session -t "${SESSION}" 2>/dev/null && exit 0

cat > "${EVAL_ROOT}/command.sh.new" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO}"
export REPO="${REPO}" EVAL_ROOT="${EVAL_ROOT}"
export FORMAL_ROOT="${FORMAL_ROOT}"
bash scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_trace_v7_gap_only_eval.sh
EOF
mv "${EVAL_ROOT}/command.sh.new" "${EVAL_ROOT}/command.sh"
chmod +x "${EVAL_ROOT}/command.sh"

cat > "${EVAL_ROOT}/queue.sh.new" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export WORKER_COMMAND="${EVAL_ROOT}/command.sh"
export INITIAL_DELAY=20
export FORMAL_ROOT="${FORMAL_ROOT}"
bash "${REPO}/scripts/reinforcement_learning/go2_sim_gap_aligned/run_with_available_gpu_v7.sh"
EOF
mv "${EVAL_ROOT}/queue.sh.new" "${EVAL_ROOT}/queue.sh"
chmod +x "${EVAL_ROOT}/queue.sh"
tmux new-session -d -s "${SESSION}" "bash '${EVAL_ROOT}/queue.sh' 2>&1 | tee '${EVAL_ROOT}/launcher.log'"
