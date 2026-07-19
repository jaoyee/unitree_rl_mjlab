#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${FORMAL_ROOT:?Set FORMAL_ROOT to a corrected V7 formal experiment root}"
: "${EVAL_ROOT:?Set EVAL_ROOT to the paired V7 evaluation root}"
PY="${REPO}/.venv/bin/python"
cd "${REPO}"

for condition in g0 rr03; do
  source "${FORMAL_ROOT}/sim/${condition}/ready.env"
  model="${MODEL_PATH}"
  baseline="$(cat "${FORMAL_ROOT}/sim/${condition}/baseline/stage/artifact_path.txt")"
  output="${EVAL_ROOT}/refresh_curves/sim/${condition}/baseline/refresh_0"
  if [[ $(find "${output}" -maxdepth 1 -name 'seed_*.json' -type f 2>/dev/null | wc -l) -lt 3 ]]; then
    CHECKPOINT_PATH="${baseline}" MODEL_PATH="${model}" CONDITION="${condition}" \
      OUTPUT_DIR="${output}" COMMAND_ROOT="${EVAL_ROOT}/commands" \
      bash scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_v7_random50_one.sh
  fi
  for variant in r10 r25; do
    for refresh in $(seq 1 8); do
      checkpoint="$(cat "${FORMAL_ROOT}/sim/${condition}/trace_${variant}/refresh_$(printf '%02d' "${refresh}")/policy/stage/artifact_path.txt")"
      output="${EVAL_ROOT}/refresh_curves/sim/${condition}/${variant}/refresh_${refresh}"
      if [[ $(find "${output}" -maxdepth 1 -name 'seed_*.json' -type f 2>/dev/null | wc -l) -lt 3 ]]; then
        CHECKPOINT_PATH="${checkpoint}" MODEL_PATH="${model}" CONDITION="${condition}" \
          OUTPUT_DIR="${output}" COMMAND_ROOT="${EVAL_ROOT}/commands" \
          bash scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_v7_random50_one.sh
      fi
      "${PY}" scripts/reinforcement_learning/go2_sim_gap_aligned/summarize_go2_trace_v7_refresh_eval.py \
        --eval-root "${EVAL_ROOT}"
    done
  done
done
