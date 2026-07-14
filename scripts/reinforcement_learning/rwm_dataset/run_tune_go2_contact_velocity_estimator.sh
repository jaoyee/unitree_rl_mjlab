#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_ROOT="${OUTPUT_ROOT:-logs/evaluations/contact_velocity_tuning_20260713}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-logs/experiments/go2_normal_dr_3x3x4/normal_go2_fixstand_v2_moderate_seed0_20260712/policies/E1/D1/P2/run/step48828}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-64}"
mkdir -p "${OUTPUT_ROOT}/clean" "${OUTPUT_ROOT}/half"

COMMON_ARGS=(
  --checkpoint_path "${CHECKPOINT_PATH}"
  --task Unitree-Go2-Flat-Normal-FixStand-RWM-Pretrain-Ens
  --device "${DEVICE}"
  --num_envs "${NUM_ENVS}"
  --steps 2400
  --seed 400
  --command_sequence=0.0,0.0,0.0
  --command_sequence=0.5,0.0,0.0
  --command_sequence=-0.4,0.0,0.0
  --command_sequence=0.0,0.25,0.0
  --command_sequence=0.0,-0.25,0.0
  --command_sequence=0.0,0.0,0.5
  --command_sequence=0.0,0.0,-0.5
  --command_sequence=0.4,0.2,0.3
  --command_switch_steps 300
  --validate_contact_velocity_estimator
)

.venv/bin/python scripts/reinforcement_learning/rwm_flashsac/eval_flashsac_go2_mjlab_proprioceptive.py \
  "${COMMON_ARGS[@]}" \
  --clean \
  --velocity_estimator_samples_npz "${OUTPUT_ROOT}/clean/samples.npz" \
  --output_json "${OUTPUT_ROOT}/clean/summary.json" \
  > "${OUTPUT_ROOT}/clean/run.log" 2>&1

.venv/bin/python scripts/reinforcement_learning/rwm_flashsac/eval_flashsac_go2_mjlab_proprioceptive.py \
  "${COMMON_ARGS[@]}" \
  --no-clean \
  --randomization_preset calibrated_friend_flat \
  --randomization_components all \
  --randomization_scale 0.5 \
  --velocity_estimator_samples_npz "${OUTPUT_ROOT}/half/samples.npz" \
  --output_json "${OUTPUT_ROOT}/half/summary.json" \
  > "${OUTPUT_ROOT}/half/run.log" 2>&1

.venv/bin/python scripts/reinforcement_learning/rwm_dataset/tune_go2_contact_velocity_estimator.py \
  --train-npz "${OUTPUT_ROOT}/clean/samples.npz" \
  --validation-npz "${OUTPUT_ROOT}/half/samples.npz" \
  --output-json "${OUTPUT_ROOT}/tuning_report.json" \
  > "${OUTPUT_ROOT}/tuning.log" 2>&1

echo "tuning_report=${OUTPUT_ROOT}/tuning_report.json"
