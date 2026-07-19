#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${REPO}"

: "${SOURCE:?Set SOURCE to the converted go2sun-g0 dataset_all.pt}"
: "${TARGET:?Set TARGET to common_signed_magnitude_target_v2.json}"
OUT_DIR="${OUT_DIR:-${REPO}/logs/rwm_datasets_v2/real_v2/g0_go2sun_recollect}"
PYTHON_BIN="${PYTHON_BIN:-${REPO}/.venv/bin/python}"

[[ -s "${SOURCE}" ]] || { echo "Missing source dataset: ${SOURCE}" >&2; exit 2; }
[[ -s "${TARGET}" ]] || { echo "Missing target specification: ${TARGET}" >&2; exit 2; }
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python interpreter: ${PYTHON_BIN}" >&2; exit 2; }

mkdir -p "$OUT_DIR"

"${PYTHON_BIN}" \
  scripts/reinforcement_learning/rwm_dataset/select_go2_matched_sequence_milp_v2.py \
  --input "$SOURCE" \
  --output "$OUT_DIR/official_25k.pt" \
  --report "$OUT_DIR/selection_manifest.json" \
  --condition-id g0-go2sun \
  --minimum-run 40 \
  --enforce-minimum-run \
  --marginal-target-json "$TARGET" \
  --time-limit 1800 \
  --mip-rel-gap 0.0 \
  >"$OUT_DIR/selection.log" 2>&1

date --iso-8601=seconds > "$OUT_DIR/SELECTION_COMPLETE"
