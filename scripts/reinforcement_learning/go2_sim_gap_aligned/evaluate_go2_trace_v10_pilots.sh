#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${V10_RUN_BASE:?Set pilot run base}"
PY="${PYTHON_BIN:-${REPO}/.venv/bin/python}"
PROTOCOL="${REPO}/scripts/reinforcement_learning/rwm_trace/trace_v10_protocol.json"
all_passed=true
for condition in rr05 p5; do
  trace="${V10_RUN_BASE}/trace/real/${condition}/refresh_01/behavior/summary.json"
  control="${V10_RUN_BASE}/control/real/${condition}/behavior/summary.json"
  output="${V10_RUN_BASE}/pilot_gate_real_${condition}.json"
  if ! "${PY}" "${REPO}/scripts/reinforcement_learning/rwm_trace/evaluate_v10_pilot_gate.py" \
    --trace-summary "${trace}" --control-summary "${control}" --protocol "${PROTOCOL}" --output "${output}"; then
    all_passed=false
  fi
done
[[ "${all_passed}" == true ]] || exit 3
printf 'passed=true\ntime=%s\n' "$(date --iso-8601=seconds)" > "${V10_RUN_BASE}/pilots_passed.env.new"
mv "${V10_RUN_BASE}/pilots_passed.env.new" "${V10_RUN_BASE}/pilots_passed.env"
