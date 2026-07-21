#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${DATASET_PATH:?Set the immutable condition 25K dataset path}"
: "${CONDITION:?Set g0, rr05, rr03, p5, or p75}"
: "${OUTPUT_ROOT:?Set a new V10 initial-scorer directory}"
PY="${PYTHON_BIN:-${REPO}/.venv/bin/python}"
PROTOCOL="${REPO}/scripts/reinforcement_learning/rwm_trace/trace_v10_protocol.json"
CODEX_LOCK="${CODEX_LOCK:-$(dirname "${OUTPUT_ROOT}")/codex_feedback.lock}"
MODES=(stand pure_x pure_y pure_yaw xy x_yaw y_yaw xy_yaw)

case "${CONDITION}" in g0|rr05|rr03|p5|p75) ;; *) echo "Bad CONDITION=${CONDITION}" >&2; exit 2 ;; esac
[[ -x "${PY}" && -s "${DATASET_PATH}" && -s "${PROTOCOL}" ]] || exit 2
[[ ! -e "${OUTPUT_ROOT}/ready.env" ]] || { echo "Initial scorer already committed: ${OUTPUT_ROOT}" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}/labels" "${OUTPUT_ROOT}/seeds"
cd "${REPO}"

read -r WINDOW STRIDE PAIR_BUDGET PAIR_MIN LABEL_MIN LABEL_BUCKET LABEL_SIDE HIDDEN GLOBAL_ACC MODE_ACC MODE_VAL CONFIDENCE <<< "$(${PY} - "${PROTOCOL}" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))["scorer"]
print(p["initial_window_length"], p["initial_window_stride"], p["initial_pair_pool_size"],
      p["minimum_pairs_per_partition_mode"], p["minimum_initial_filtered_labels"],
      p["minimum_initial_labels_per_partition_mode"], p["minimum_initial_labels_per_side"],
      p["hidden_dimension"], p["minimum_global_balanced_accuracy"],
      p["minimum_mode_balanced_accuracy"], p["minimum_mode_validation_count"],
      p["confidence_threshold"])
PY
)"

SUMMARIES="${OUTPUT_ROOT}/dataset_windows.jsonl"
PAIRS="${OUTPUT_ROOT}/pairs.jsonl"
LABELS="${OUTPUT_ROOT}/labels/codex_labels_filtered.jsonl"
SCORER="${OUTPUT_ROOT}/go2_trace_scorer.pt"

"${PY}" scripts/reinforcement_learning/rwm_trace/build_go2_dataset_window_summaries.py \
  --dataset "${DATASET_PATH}" --output "${SUMMARIES}" --window_length "${WINDOW}" \
  --stride "${STRIDE}" --comparison_group start
"${PY}" scripts/reinforcement_learning/rwm_trace/build_go2_partitioned_feedback_pairs.py \
  --summaries "${SUMMARIES}" --output "${PAIRS}" \
  --partition_manifest "${OUTPUT_ROOT}/pair_partition.json" --condition "${CONDITION}" \
  --budget "${PAIR_BUDGET}" --validation_fraction 0.20 \
  --required_pair_modes "${MODES[@]}" \
  --min_pairs_per_mode_per_partition "${PAIR_MIN}" --allow_cross_group_pairs \
  --pair_prefix "v10_initial_${CONDITION}" --seed 1042
"${PY}" scripts/reinforcement_learning/rwm_trace/audit_go2_pair_partition.py \
  --pairs "${PAIRS}" --output "${OUTPUT_ROOT}/pair_audit.json" \
  --required_modes "${MODES[@]}" --min_pairs_per_partition_mode "${PAIR_MIN}"

"${PY}" scripts/reinforcement_learning/rwm_trace/label_feedback_with_codex.py \
  --prompts "${PAIRS}" --output-dir "${OUTPUT_ROOT}/labels" \
  --schema scripts/reinforcement_learning/rwm_trace/feedback_label_batch.schema.json \
  --repo-root "${REPO}" --batch-size 20 --confidence-threshold "${CONFIDENCE}" \
  --minimum-filtered-labels "${LABEL_MIN}" --max-low-confidence-relabels 16 \
  --max-retries 2 --timeout-sec 300 --lock-path "${CODEX_LOCK}" --codex-service-tier default
"${PY}" scripts/reinforcement_learning/rwm_trace/audit_go2_pair_partition.py \
  --pairs "${PAIRS}" --labels "${LABELS}" --output "${OUTPUT_ROOT}/label_audit.json" \
  --required_modes "${MODES[@]}" --min_pairs_per_partition_mode "${PAIR_MIN}" \
  --min_labels_per_partition_mode "${LABEL_BUCKET}" --min_labels_per_side "${LABEL_SIDE}" \
  --confidence_threshold "${CONFIDENCE}"
"${PY}" - "${LABELS}" "${LABEL_MIN}" <<'PY'
import sys
count=sum(bool(line.strip()) for line in open(sys.argv[1], encoding="utf-8"))
minimum=int(sys.argv[2])
if count < minimum:
    raise SystemExit(f"Initial V10 high-confidence label shortfall: {count}/{minimum}")
PY

CANDIDATES=()
for seed in 42 43 44 45 46; do
  checkpoint="${OUTPUT_ROOT}/seeds/scorer_seed${seed}.pt"
  "${PY}" scripts/reinforcement_learning/rwm_trace/train_go2_trace_scorer.py \
    --labels "${LABELS}" --pairs "${PAIRS}" --output "${checkpoint}" \
    --epochs 80 --learning_rate 0.001 --hidden_dim "${HIDDEN}" \
    --confidence_threshold "${CONFIDENCE}" --val_ratio 0.20 \
    --required_val_modes "${MODES[@]}" --min_val_mode_count "${MODE_VAL}" \
    --zero_all_missing_feature_weights --seed "${seed}"
  CANDIDATES+=(--candidate "${checkpoint}")
done
"${PY}" scripts/reinforcement_learning/rwm_trace/select_go2_scorer_checkpoint.py \
  "${CANDIDATES[@]}" --output "${SCORER}" --report "${OUTPUT_ROOT}/seed_selection.json" \
  --required_modes "${MODES[@]}" --min_global_balanced_accuracy "${GLOBAL_ACC}" \
  --min_mode_balanced_accuracy "${MODE_ACC}" --min_mode_count "${MODE_VAL}"
"${PY}" scripts/reinforcement_learning/rwm_trace/validate_v10_scorer.py \
  --checkpoint "${SCORER}" --output "${OUTPUT_ROOT}/scorer_validation.json" \
  --minimum-global-balanced-accuracy "${GLOBAL_ACC}" \
  --minimum-mode-balanced-accuracy "${MODE_ACC}" \
  --minimum-mode-validation-count "${MODE_VAL}"

DATASET_SHA="$(sha256sum "${DATASET_PATH}" | awk '{print $1}')"
PROTOCOL_SHA="$(sha256sum "${PROTOCOL}" | awk '{print $1}')"
SCORER_SHA="$(sha256sum "${SCORER}" | awk '{print $1}')"
cat > "${OUTPUT_ROOT}/ready.env.new" <<EOF
TRACE_PROTOCOL=go2_trace_v10
TRACE_PROTOCOL_SHA256=${PROTOCOL_SHA}
INITIAL_SCORER_CONDITION=${CONDITION}
INITIAL_SCORER_DATASET=$(realpath "${DATASET_PATH}")
INITIAL_SCORER_DATASET_SHA256=${DATASET_SHA}
INITIAL_SCORER_PATH=$(realpath "${SCORER}")
INITIAL_SCORER_SHA256=${SCORER_SHA}
INITIAL_PAIRS_PATH=$(realpath "${PAIRS}")
INITIAL_LABELS_PATH=$(realpath "${LABELS}")
EOF
mv "${OUTPUT_ROOT}/ready.env.new" "${OUTPUT_ROOT}/ready.env"
echo "V10 initial scorer ready: ${OUTPUT_ROOT}"
