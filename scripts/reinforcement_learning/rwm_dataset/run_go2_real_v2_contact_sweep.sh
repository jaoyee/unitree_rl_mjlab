#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data1/xjy/unitree_rl_mjlab_normal_fixstand_v2}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
CONVERTER="$ROOT/scripts/reinforcement_learning/rwm_dataset/convert_go2_real_csv_to_dataset.py"
OUT_ROOT="${OUT_ROOT:-$ROOT/logs/rwm_datasets_v2/contact_sweep}"
read -r -a THRESHOLDS <<< "${THRESHOLDS_TEXT:-10 12 15 18}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"

conditions=(g0 rr05 rr03 p5 p75)
declare -A dirs=(
  [g0]="logs/rwm_datasets/go2_real_g0_baseline_25k_20260714"
  [rr05]="logs/rwm_datasets/go2_real_g0_d0_rrcalf_0p5_25k_20260714"
  [rr03]="logs/rwm_datasets/go2_real_g0_d0_rrcalf_0p3_25k_20260715"
  [p5]="logs/rwm_datasets/go2_real_g0_d0_p5_25k_20260715"
  [p75]="logs/rwm_datasets/go2_real_g0_d0_p75_25k_20260715"
)
declare -A deploys=(
  [g0]="config/deploy.yaml"
  [rr05]="config/deploy.yaml"
  [rr03]="config/deploy.yaml"
  [p5]="source/params/deploy.yaml"
  [p75]="source/params/deploy.yaml"
)
declare -A csv_dirs=(
  [g0]="raw"
  [rr05]="raw"
  [rr03]="raw"
  [p5]="source/logs"
  [p75]="source/logs"
)

run_one() {
  local condition="$1" threshold="$2" source_dir="$3" deploy="$4" robot_id="$5"
  shift 5
  local -a csv_args=("$@")
  local output_dir="$OUT_ROOT/$condition/threshold_$threshold"
  mkdir -p "$output_dir"
  echo "[start] condition=$condition threshold=$threshold $(date -Is)"
  "$PYTHON" "$CONVERTER" \
    "${csv_args[@]}" \
    --deploy-yaml "$deploy" \
    --output "$output_dir/dataset_all.pt" \
    --report-json "$output_dir/conversion_report.json" \
    --contact-source recompute \
    --contact-force-threshold "$threshold" \
    --condition-id "$condition" \
    --robot-id "$robot_id" \
    --collection-id "v1_existing_csv_reconverted_v2" \
    > "$output_dir/conversion_stdout.log"
  echo "[done] condition=$condition threshold=$threshold $(date -Is)"
}

mkdir -p "$OUT_ROOT"
for condition in "${conditions[@]}"; do
  source_dir="$ROOT/${dirs[$condition]}"
  deploy="$source_dir/${deploys[$condition]}"
  robot_id="go2sun"
  [[ "$condition" == "g0" ]] && robot_id="go2_original"
  mapfile -t csvs < <(find "$source_dir/${csv_dirs[$condition]}" -maxdepth 1 -type f -name 'run_data_*.csv' | sort)
  if [[ ${#csvs[@]} -eq 0 ]]; then
    echo "no CSV files found for $condition" >&2
    exit 1
  fi
  csv_args=()
  for csv in "${csvs[@]}"; do
    csv_args+=(--csv "$csv")
  done
  for threshold in "${THRESHOLDS[@]}"; do
    while (( $(jobs -pr | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
    run_one "$condition" "$threshold" "$source_dir" "$deploy" "$robot_id" "${csv_args[@]}" &
  done
done
wait
