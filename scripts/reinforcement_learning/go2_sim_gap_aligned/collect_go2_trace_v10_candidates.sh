#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${SIDE:?Set sim or real}"
: "${DATASET_PATH:?Set the condition 25K dataset}"
: "${POLICY_PATH:?Set the current policy checkpoint}"
: "${CYCLE_ROOT:?Set the refresh output directory}"
: "${V10_GPU_POOL:?Set physical GPU pool}"
PY="${PYTHON_BIN:-${REPO}/.venv/bin/python}"
PROTOCOL="${REPO}/scripts/reinforcement_learning/rwm_trace/trace_v10_protocol.json"
GPU_EXEC="${REPO}/scripts/reinforcement_learning/go2_sim_gap_aligned/run_v10_gpu_stage.sh"
case "${SIDE}" in sim) RESET_MODE=exact_snapshot ;; real) RESET_MODE=canonical_real_projection ;; *) exit 2 ;; esac
mkdir -p "${CYCLE_ROOT}/source_batches" "${CYCLE_ROOT}/candidate_parts"
cd "${REPO}"

read -r H STARTS BRANCHES MIN_T <<< "$(${PY} - "${PROTOCOL}" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))["candidate"]
print(p["horizon"], p["source_start_count"], p["trajectories_per_start"], p["minimum_source_timestep"])
PY
)"
SOURCE_IDS="${CYCLE_ROOT}/source_ids.pt"
FINAL="${CYCLE_ROOT}/candidates.pt"
if [[ -s "${FINAL}" && -s "${SOURCE_IDS}" ]]; then
  "${PY}" scripts/reinforcement_learning/rwm_trace/audit_v10_candidate_set.py \
    --candidate "${FINAL}" --source-ids "${SOURCE_IDS}" --protocol "${PROTOCOL}" \
    --output "${CYCLE_ROOT}/candidate_audit.json"
  echo "V10 candidates already valid: ${FINAL}"
  exit 0
fi
"${PY}" scripts/reinforcement_learning/rwm_trace/select_v10_source_starts.py \
  --dataset "${DATASET_PATH}" --output "${SOURCE_IDS}" \
  --report "${CYCLE_ROOT}/source_ids_report.json" --protocol "${PROTOCOL}" \
  --seed "$((10420 + ${REFRESH_CYCLE:?Set REFRESH_CYCLE}))" \
  --batch-output-dir "${CYCLE_ROOT}/source_batches" --batch-size 256

for batch in 0 1 2 3; do
  batch_id="$(printf '%02d' "${batch}")"
  source_batch="${CYCLE_ROOT}/source_batches/source_ids_batch_${batch_id}.pt"
  part="${CYCLE_ROOT}/candidate_parts/batch_${batch_id}.pt"
  if [[ ! -s "${part}" ]]; then
    V10_GPU_POOL="${V10_GPU_POOL}" "${GPU_EXEC}" "${PY}" \
      scripts/reinforcement_learning/rwm_dataset/collect_go2_expert_command_coverage_dataset.py \
      --task Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert \
      --device cuda:0 --seed "$((20420 + REFRESH_CYCLE * 10 + batch))" --num_envs 4096 \
      --num_transitions "$((4096 * H))" --save_path "${part}" \
      --expert_policy_path "${POLICY_PATH}" --collector_mix expert:1.0 \
      --fixed_collector_assignment --collector_assignment_seed 42 \
      --dataset_obs_kind proprioceptive --action_noise_std 0.0 \
      --medium_action_noise_std 0.0 --failure_action_noise_std 0.0 \
      --no-use_domain_randomization --no-use_push_randomization --no-use_observation_noise \
      --payload_mass_range_kg 0.0 0.0 --rr_calf_strength_range 1.0 1.0 \
      --env_action_delay_steps 0 0 --env_action_noise_std 0.0 \
      --command_modes stand,pure_x,pure_y,pure_yaw,xy,x_yaw,y_yaw,xy_yaw \
      --trace_reset_dataset "${DATASET_PATH}" --trace_reset_mode "${RESET_MODE}" \
      --trace_source_ids_path "${source_batch}" --trace_min_source_timestep "${MIN_T}" \
      --trace_source_history_horizon "${MIN_T}" --trace_rollout_length "${H}" \
      --trace_trajectories_per_state "${BRANCHES}" --trace_action_temperature 1.0 \
      --trace_identity_tolerance 5e-3 --no-trace_require_source_transition_identity --headless
  fi
done

merged="${CYCLE_ROOT}/candidate_parts/batch_00.pt"
for batch in 1 2 3; do
  batch_id="$(printf '%02d' "${batch}")"
  next="${CYCLE_ROOT}/candidate_parts/merged_through_${batch_id}.pt"
  "${PY}" scripts/reinforcement_learning/rwm_trace/merge_go2_trace_candidate_branches.py \
    --base "${merged}" --append "${CYCLE_ROOT}/candidate_parts/batch_${batch_id}.pt" \
    --output "${next}"
  merged="${next}"
done
mv "${merged}" "${FINAL}.new"
mv "${FINAL}.new" "${FINAL}"
"${PY}" scripts/reinforcement_learning/rwm_trace/audit_v10_candidate_set.py \
  --candidate "${FINAL}" --source-ids "${SOURCE_IDS}" --protocol "${PROTOCOL}" \
  --output "${CYCLE_ROOT}/candidate_audit.json"
rm -f -- "${CYCLE_ROOT}"/candidate_parts/*.pt
echo "V10 candidates ready: ${FINAL}"
