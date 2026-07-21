#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/unitree_rl_mjlab_normal_fixstand_v2}"
: "${GAP_ID:?Set GAP_ID to g0, rr03, rr05, p5, or p75}"
: "${CHECKPOINT:?Set CHECKPOINT to a FlashSAC checkpoint directory}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new evaluation output directory}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-128}"
STEPS="${STEPS:-2400}"
SEEDS="${SEEDS:-901 902 903}"
MODES="${MODES:-clean calibrated}"

case "${GAP_ID}" in
  g0) PAYLOAD=0.0; RR_STRENGTH=1.0 ;;
  rr03) PAYLOAD=0.0; RR_STRENGTH=0.3 ;;
  rr05) PAYLOAD=0.0; RR_STRENGTH=0.5 ;;
  p5) PAYLOAD=5.0; RR_STRENGTH=1.0 ;;
  p75) PAYLOAD=7.5; RR_STRENGTH=1.0 ;;
  *) echo "Unsupported GAP_ID=${GAP_ID}" >&2; exit 2 ;;
esac

cd "${REPO}"
mkdir -p "${OUTPUT_ROOT}"
COMMANDS=(
  "0,0,0"
  "0.35,0,0"
  "0,0.15,0"
  "0,0,0.30"
  "0.30,0.15,0"
  "0.30,0,0.30"
  "0,0.15,0.30"
  "0.30,0.15,0.30"
)
COMMAND_ARGS=()
for command in "${COMMANDS[@]}"; do
  COMMAND_ARGS+=("--command_sequence=${command}")
done

for mode in ${MODES}; do
  case "${mode}" in clean|calibrated) ;; *) echo "Bad evaluation mode: ${mode}" >&2; exit 2 ;; esac
  for seed in ${SEEDS}; do
    output="${OUTPUT_ROOT}/${mode}_seed${seed}.json"
    [[ -s "${output}" ]] && continue
    mode_args=(--clean)
    if [[ "${mode}" == calibrated ]]; then
      mode_args=(
        --no-clean
        --randomization_preset calibrated_default
        --randomization_components all
        --randomization_scale 1.0
      )
    fi
    "${REPO}/.venv/bin/python" \
      scripts/reinforcement_learning/rwm_flashsac/eval_flashsac_go2_mjlab_proprioceptive_gap_v4.py \
      --checkpoint_path "${CHECKPOINT}" --device "${DEVICE}" \
      --num_envs "${NUM_ENVS}" --steps "${STEPS}" --seed "${seed}" \
      --payload_mass_kg "${PAYLOAD}" --rr_calf_strength "${RR_STRENGTH}" \
      --command_switch_steps 300 --command_settle_steps 50 \
      --reset_between_commands \
      --sustained_response_steps 10 --xy_error_threshold 0.15 \
      --yaw_error_threshold 0.15 --minimum_response_gain 0.50 \
      --maximum_response_gain 1.50 --no_response_gain 0.20 \
      --action_saturation_threshold 0.95 \
      "${COMMAND_ARGS[@]}" "${mode_args[@]}" --output_json "${output}"
  done
done

"${REPO}/.venv/bin/python" \
  scripts/reinforcement_learning/go2_sim_gap_aligned/summarize_command_eval_v4.py \
  --input-dir "${OUTPUT_ROOT}" --output-json "${OUTPUT_ROOT}/summary.json"
