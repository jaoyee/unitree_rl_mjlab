#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
SIDE="${SIDE:?Set SIDE to sim or real}"
case "${SIDE}" in sim|real) ;; *) echo "SIDE must be sim or real" >&2; exit 2 ;; esac

ROOT="${REPO}/logs/experiments/go2_pooled_v4/20260717_40_20_20_10_10/${SIDE}/trace"
mkdir -p "${ROOT}/feedback"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

for gap in g0 rr05 p5 rr03 p75; do
  candidate="${ROOT}/replay_source/candidates_${gap}_t1_h100.pt"
  summary="${ROOT}/feedback/${gap}.summaries.jsonl"
  [[ -s "${summary}" ]] && continue
  [[ -s "${candidate}" ]] || { echo "Missing ${candidate}" >&2; exit 2; }
  "${REPO}/.venv/bin/python" \
    "${REPO}/scripts/reinforcement_learning/rwm_trace/build_trace_replay_v4.py" \
    --candidate_dataset "${candidate}" \
    --output "${ROOT}/feedback/bootstrap_${gap}.pt" \
    --summary_jsonl "${summary}" --summaries_only \
    --trajectory_length 100 --trajectory_stride 100 \
    --include_terminal_prefixes --minimum_terminal_length 20 \
    --terminal_penalty -10.0 --failure_backprop_steps 40 \
    --failure_backprop_penalty 2.0 --action_saturation_threshold 0.95 \
    --action_saturation_penalty_scale 10.0 --action_delta_penalty_scale 0.10 \
    --reward_source rwm_aligned
done

printf 'side=%s\nstatus=completed\n' "${SIDE}" > "${ROOT}/feedback/summary_stage.txt"
