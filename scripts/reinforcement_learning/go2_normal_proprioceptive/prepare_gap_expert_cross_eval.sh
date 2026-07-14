#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

: "${EXPERIMENT_ROOT:?Set EXPERIMENT_ROOT to a completed 2x3 expert directory}"
GAP_KIND="${GAP_KIND:-rr_calf}"
NUM_ENVS="${NUM_ENVS:-256}"
STEPS="${STEPS:-2400}"
SEED="${SEED:-400}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXPERIMENT_ROOT}/cross_eval}"
mkdir -p "${OUTPUT_ROOT}/commands"
: > "${OUTPUT_ROOT}/jobs.list"

if [[ "${GAP_KIND}" == "payload" ]]; then
  eval_ids=(M0 M5 M7p5 M10)
  eval_values=(0.0 5.0 7.5 10.0)
elif [[ "${GAP_KIND}" == "rr_calf" ]]; then
  eval_ids=(S1p0 S0p75 S0p5)
  eval_values=(1.0 0.75 0.5)
else
  echo "GAP_KIND must be payload or rr_calf" >&2
  exit 2
fi

while IFS='|' read -r train_id command_file; do
  stage_dir="${EXPERIMENT_ROOT}/${train_id}"
  artifact_file="${stage_dir}/artifact_path.txt"
  for idx in "${!eval_ids[@]}"; do
    eval_id="${eval_ids[$idx]}" value="${eval_values[$idx]}"
    output_dir="${OUTPUT_ROOT}/${train_id}/${eval_id}"
    output_json="${output_dir}/metrics.json"
    eval_command="${OUTPUT_ROOT}/commands/${train_id}_${eval_id}.sh"
    mkdir -p "${output_dir}"
    payload=0.0; strength=1.0
    [[ "${GAP_KIND}" == "payload" ]] && payload="${value}"
    [[ "${GAP_KIND}" == "rr_calf" ]] && strength="${value}"
    cat > "${eval_command}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
checkpoint=\$(cat "${artifact_file}")
"${REPO_ROOT}/.venv/bin/python" scripts/eval_flashsac_mjlab.py \\
  --checkpoint_path "\${checkpoint}" --num_envs "${NUM_ENVS}" --steps "${STEPS}" \\
  --seed "${SEED}" --device cuda:0 --clean \\
  --payload_mass_kg "${payload}" --rr_calf_strength "${strength}" \\
  --output_json "${output_json}" 2>&1 | tee "${output_dir}/run.log"
EOF
    chmod +x "${eval_command}"
    echo "${train_id}_${eval_id}|${eval_command}" >> "${OUTPUT_ROOT}/jobs.list"
  done
done < "${EXPERIMENT_ROOT}/jobs.list"

echo "Prepared $(wc -l < "${OUTPUT_ROOT}/jobs.list") cross-evaluation commands in ${OUTPUT_ROOT}"
