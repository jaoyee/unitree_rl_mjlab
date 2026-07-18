#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

domain="${1:?usage: $0 real|sim}"
case "$domain" in
  real) base="logs/rwm_datasets_v2/real_v2" ;;
  sim) base="logs/rwm_datasets_v2/sim_v2" ;;
  *) echo "domain must be real or sim" >&2; exit 2 ;;
esac

paths=(
  "logs/rwm_datasets_v2/specs"
  "scripts/reinforcement_learning/rwm/dynamics.py"
  "scripts/reinforcement_learning/rwm_dataset/dataset.py"
  "scripts/reinforcement_learning/rwm_dataset/proprioceptive_dynamics.py"
  "scripts/reinforcement_learning/rwm_dataset/train_world_model_offline_go2_proprioceptive.py"
  "scripts/reinforcement_learning/rwm_dataset/validate_go2_matched_v2.py"
  "scripts/reinforcement_learning/rwm_dataset/smoke_go2_v2_dataset.py"
  "scripts/reinforcement_learning/rwm_dataset/export_go2_v2_qa_features.py"
  "scripts/reinforcement_learning/rwm_dataset/freeze_go2_v2_manifest.py"
)

if [[ "$domain" == real ]]; then
  paths+=(
    "scripts/reinforcement_learning/rwm_dataset/convert_go2_real_csv_to_dataset.py"
    "scripts/reinforcement_learning/rwm_dataset/select_go2_real_command_coverage_dataset.py"
    "scripts/reinforcement_learning/rwm_dataset/build_go2_termination_aux_v2.py"
    "logs/rwm_datasets_v2/pair_qa"
  )
else
  paths+=(
    "scripts/reinforcement_learning/rwm_dataset/select_go2_balanced_transitions_v2.py"
    "scripts/reinforcement_learning/rwm_dataset/select_go2_matched_blocks_v2.py"
    "scripts/reinforcement_learning/rwm_dataset/select_go2_matched_intervals_v2.py"
    "scripts/reinforcement_learning/rwm_dataset/select_go2_matched_sequence_milp_v2.py"
  )
  transfer_manifest="logs/rwm_datasets_v2/freeze/transfer_3090_to_4090_v2.sha256"
  [[ -f "$transfer_manifest" ]] && paths+=("$transfer_manifest")
fi

for condition in g0 rr05 rr03 p5 p75; do
  paths+=(
    "$base/$condition/dataset.pt"
    "$base/$condition/selection_report.json"
    "$base/$condition/validation_report.json"
    "$base/$condition/smoke_report.json"
  )
  if [[ "$domain" == real ]]; then
    paths+=(
      "$base/$condition/threshold_18/dataset_all.pt"
      "$base/$condition/threshold_18/conversion_report.json"
      "$base/$condition/termination_aux_report.json"
    )
    [[ -f "$base/$condition/termination_aux.pt" ]] \
      && paths+=("$base/$condition/termination_aux.pt")
  else
    paths+=("$base/$condition/expert_pool_150k.pt")
  fi
done

args=()
for path in "${paths[@]}"; do
  args+=(--path "$path")
done

.venv/bin/python \
  scripts/reinforcement_learning/rwm_dataset/freeze_go2_v2_manifest.py \
  --root "$repo_root" \
  --domain "$domain" \
  "${args[@]}" \
  --output "logs/rwm_datasets_v2/freeze/${domain}_v2_manifest.json"
