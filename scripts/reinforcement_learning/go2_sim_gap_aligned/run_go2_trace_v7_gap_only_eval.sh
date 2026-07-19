#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${FORMAL_ROOT:?Set FORMAL_ROOT to a corrected V7 formal experiment root}"
EVAL_ROOT="${EVAL_ROOT:-${REPO}/logs/evaluations/go2_trace_v7/$(date +%Y%m%d_%H%M%S)}"
PY="${REPO}/.venv/bin/python"
mkdir -p "${EVAL_ROOT}/commands"
cd "${REPO}"

write_state() {
  printf 'time=%s\nstage=%s\nstatus=%s\ngpu=%s\n' \
    "$(date --iso-8601=seconds)" "$1" "$2" "${CUDA_VISIBLE_DEVICES}" > "${EVAL_ROOT}/state.txt"
}

baseline_for() { cat "${FORMAL_ROOT}/$1/$2/baseline/stage/artifact_path.txt"; }
model_for() {
  local ready="${FORMAL_ROOT}/$1/$2/ready.env"
  bash -c "source '${ready}'; printf '%s' \"\${MODEL_PATH}\""
}
trace_for() {
  local state="${FORMAL_ROOT}/$1/$2/trace_$3/state.txt"
  grep -qx 'status=completed' "${state}" 2>/dev/null || return 1
  cat "${FORMAL_ROOT}/$1/$2/trace_$3/refresh_08/policy/stage/artifact_path.txt"
}

while true; do
  pending=0
  for side in sim real; do
    for condition in g0 rr05 p5 rr03 p75; do
      model="$(model_for "${side}" "${condition}")"
      for variant in baseline r10 r25; do
        if [[ "${variant}" == baseline ]]; then
          checkpoint="$(baseline_for "${side}" "${condition}")"
        else
          checkpoint="$(trace_for "${side}" "${condition}" "${variant}" || true)"
          if [[ -z "${checkpoint}" ]]; then pending=$((pending + 1)); continue; fi
        fi
        output="${EVAL_ROOT}/gap_only/${side}/${condition}/${variant}"
        if [[ $(find "${output}" -maxdepth 1 -name 'seed_*.json' -type f 2>/dev/null | wc -l) -lt 3 ]]; then
          write_state "${side}_${condition}_${variant}" running
          CHECKPOINT_PATH="${checkpoint}" MODEL_PATH="${model}" CONDITION="${condition}" \
            OUTPUT_DIR="${output}" COMMAND_ROOT="${EVAL_ROOT}/commands" \
            bash scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_v7_random50_one.sh
          "${PY}" scripts/reinforcement_learning/go2_sim_gap_aligned/summarize_go2_trace_v7_random50_eval.py \
            --eval-root "${EVAL_ROOT}"
        fi
      done
    done
  done
  if [[ "${pending}" -eq 0 ]]; then break; fi
  write_state "waiting_for_${pending}_formal_branches" waiting
  sleep 300
done
"${PY}" scripts/reinforcement_learning/go2_sim_gap_aligned/summarize_go2_trace_v7_random50_eval.py \
  --eval-root "${EVAL_ROOT}"
bash scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_trace_v7_refresh_eval.sh
write_state done completed
