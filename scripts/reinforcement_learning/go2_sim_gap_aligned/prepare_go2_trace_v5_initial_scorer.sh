#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${SIDE:?Set SIDE=sim or SIDE=real}"
: "${DATASET_PATH:?Set the immutable expert dataset path}"
: "${SIDE_ROOT:?Set the V5 side root}"
case "${SIDE}" in sim|real) ;; *) exit 2 ;; esac
PY="${REPO}/.venv/bin/python"
ROOT="${SIDE_ROOT}/initial_scorer"
SUMMARIES="${ROOT}/dataset_windows.jsonl"
PAIRS="${ROOT}/pairs.jsonl"
LABEL_DIR="${ROOT}/labels"
LABELS="${LABEL_DIR}/codex_labels_filtered.jsonl"
SCORER="${ROOT}/go2_trace_scorer.pt"
CODEX_LOCK="$(dirname "${SIDE_ROOT}")/codex_feedback.lock"
mkdir -p "${ROOT}"
cd "${REPO}"

for codex_dir in \
  /root/.vscode-server/extensions/openai.chatgpt-26.707.71524-linux-x64/bin/linux-x86_64 \
  /tmp/.X11-unix/codex_projects/runtime/codex-preflight
do
  [[ -x "${codex_dir}/codex" ]] && export PATH="${codex_dir}:${PATH}" && break
done

[[ -s "${SUMMARIES}" ]] || "${PY}" \
  scripts/reinforcement_learning/rwm_trace/build_go2_dataset_window_summaries.py \
  --dataset "${DATASET_PATH}" --output "${SUMMARIES}" --window_length 40 --stride 20
[[ -s "${PAIRS}" ]] || "${PY}" \
  scripts/reinforcement_learning/rwm_trace/build_go2_feedback_pairs.py \
  --summaries "${SUMMARIES}" --output "${PAIRS}" --budget 500 \
  --pair_prefix "${SIDE}_dataset_init" --cross_start_fraction 0 --seed 42

pair_count=$(grep -cve '^[[:space:]]*$' "${PAIRS}" || true)
minimum_labels=$(( pair_count * 2 / 5 ))
(( minimum_labels > 200 )) && minimum_labels=200
(( minimum_labels >= 80 )) || { echo "Too few controlled dataset-window pairs: ${pair_count}" >&2; exit 3; }
while [[ ! -s "${LABELS}" ]] || (( $(grep -cve '^[[:space:]]*$' "${LABELS}" || true) < minimum_labels )); do
  "${PY}" scripts/reinforcement_learning/rwm_trace/label_feedback_with_codex.py \
    --prompts "${PAIRS}" --output-dir "${LABEL_DIR}" \
    --schema scripts/reinforcement_learning/rwm_trace/feedback_label_batch.schema.json \
    --repo-root "${REPO}" --batch-size 20 --confidence-threshold 0.70 \
    --max-retries 2 --timeout-sec 300 --lock-path "${CODEX_LOCK}" \
    --codex-service-tier default 2>&1 | tee -a "${ROOT}/label.log" || true
  [[ -s "${LABELS}" ]] && count=$(grep -cve '^[[:space:]]*$' "${LABELS}" || true) || count=0
  (( count >= minimum_labels )) || sleep 120
done

"${PY}" scripts/reinforcement_learning/rwm_trace/train_go2_trace_scorer.py \
  --labels "${LABELS}" --pairs "${PAIRS}" --output "${SCORER}" \
  --epochs 50 --learning_rate 0.001 --hidden_dim 256 --confidence_threshold 0.70 \
  --val_ratio 0.20 --seed 42
"${PY}" - "${SCORER}" <<'PY'
import math, sys, torch
m = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["metadata"]
n, acc = int(m["num_val_pairs"]), float(m["val_balanced_accuracy"])
if n < 16 or not math.isfinite(acc) or acc < 0.60:
    raise SystemExit(f"initial scorer gate failed: num_val={n}, balanced_accuracy={acc}")
PY

{
  printf 'INITIAL_SCORER_PATH=%q\n' "$(realpath "${SCORER}")"
  printf 'INITIAL_PAIRS_PATH=%q\n' "$(realpath "${PAIRS}")"
  printf 'INITIAL_LABELS_PATH=%q\n' "$(realpath "${LABELS}")"
} > "${SIDE_ROOT}/scorer_ready.env.tmp"
mv "${SIDE_ROOT}/scorer_ready.env.tmp" "${SIDE_ROOT}/scorer_ready.env"
