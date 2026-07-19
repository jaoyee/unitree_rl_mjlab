#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${SIDE:?Set SIDE=sim or SIDE=real}"
: "${BRANCH_ID:?Set a unique branch identifier}"
: "${RUN_ROOT:?Set a new, empty V5 branch directory}"
: "${DATASET_PATH:?Set the exact 25K RWM dataset path}"
: "${MODEL_PATH:?Set the frozen RWM model_5000.pt path}"
: "${INITIAL_POLICY_PATH:?Set the common from-scratch warm-up checkpoint}"
: "${INITIAL_SCORER_PATH:?Set the dataset-window pretrained scorer checkpoint}"
: "${INITIAL_PAIRS_PATH:?Set the dataset-window initial feedback pairs}"
: "${INITIAL_LABELS_PATH:?Set the dataset-window initial filtered labels}"
: "${CUDA_VISIBLE_DEVICES:?Set exactly one physical GPU}"

case "${SIDE}" in sim) RESET_MODE=exact_snapshot ;; real) RESET_MODE=canonical_real_projection ;; *) exit 2 ;; esac
if [[ -n "${REPLAY_RATIO_OVERRIDE:-}" ]]; then
  REPLAY_RATIO="${REPLAY_RATIO_OVERRIDE}"
else
  case "${BRANCH_ID}" in r10) REPLAY_RATIO=0.10 ;; r25) REPLAY_RATIO=0.25 ;; *)
    echo "Set REPLAY_RATIO_OVERRIDE for BRANCH_ID=${BRANCH_ID}" >&2
    exit 2
  esac
fi
"${REPO}/.venv/bin/python" - "${REPLAY_RATIO}" <<'PY'
import sys
value = float(sys.argv[1])
if not 0.0 < value <= 1.0:
    raise SystemExit(f"REPLAY_RATIO must be in (0, 1], got {value}")
PY
CONDITION="${CONDITION:-}"
if [[ -n "${CONDITION}" ]]; then
  case "${CONDITION}" in g0|rr05|p5|rr03|p75) ;; *) echo "Bad CONDITION=${CONDITION}" >&2; exit 2 ;; esac
  RESET_DATASET_PATH="${RESET_DATASET_PATH:-${DATASET_PATH}}"
else
  : "${RESET_PARTS_ROOT:?Set the directory containing g0/rr05/p5/rr03/p75 reset datasets}"
fi

PY="${REPO}/.venv/bin/python"
# Ten million common warm-up steps plus eight 5M feedback/update cycles gives
# every method the same 50M-step policy budget while keeping TRACE feedback
# genuinely periodic.
REFRESH_CYCLES="${REFRESH_CYCLES:-8}"
SEGMENT_ENV_STEPS="${SEGMENT_ENV_STEPS:-5000000}"
TRACE_T="${TRACE_T:-2.0}"
TRACE_H="${TRACE_H:-100}"
TRACE_IDENTITY_TOLERANCE="${TRACE_IDENTITY_TOLERANCE:-5e-3}"
TRAJECTORIES_PER_STATE=4
TRACE_NUM_DATASET_STATES="${TRACE_NUM_DATASET_STATES:-1024}"
SELECT_RATIO="${SELECT_RATIO:-0.25}"
TRACE_QUERY_BUDGET_INITIAL="${TRACE_QUERY_BUDGET_INITIAL:-200}"
TRACE_QUERY_BUDGET_MIN="${TRACE_QUERY_BUDGET_MIN:-20}"
TRACE_QUERY_BUDGET_DECAY="${TRACE_QUERY_BUDGET_DECAY:-0.8}"
TRACE_QUERY_STOP_CYCLE="${TRACE_QUERY_STOP_CYCLE:-${REFRESH_CYCLES}}"
TRACE_MIN_FILTERED_LABEL_FRACTION="${TRACE_MIN_FILTERED_LABEL_FRACTION:-0.15}"
TRACE_MIN_FILTERED_LABELS="${TRACE_MIN_FILTERED_LABELS:-10}"
CODEX_LOCK="${CODEX_LOCK:-$(dirname "${RUN_ROOT}")/codex_feedback.lock}"
STATE_FILE="${RUN_ROOT}/state.txt"
RUN_LOG="${RUN_ROOT}/run.log"
POST_REFRESH_HOOK="${POST_REFRESH_HOOK:-}"
LAST_COMPLETED_CYCLE=0
STOP_REASON=""

export MUJOCO_GL=egl WANDB_MODE=offline
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
for codex_dir in \
  /root/.vscode-server/extensions/openai.chatgpt-26.707.71524-linux-x64/bin/linux-x86_64 \
  /tmp/.X11-unix/codex_projects/runtime/codex-preflight
do
  [[ -x "${codex_dir}/codex" ]] && export PATH="${codex_dir}:${PATH}" && break
done
cd "${REPO}"

write_state() {
  printf 'time=%s\nside=%s\ncondition=%s\nbranch=%s\ncycle=%s\nstage=%s\nstatus=%s\ngpu=%s\n' \
    "$(date --iso-8601=seconds)" "${SIDE}" "${CONDITION:-pooled}" "${BRANCH_ID}" "${1}" "${2}" "${3}" \
    "${CUDA_VISIBLE_DEVICES}" > "${STATE_FILE}"
}

for path in "${DATASET_PATH}" "${MODEL_PATH}" "${INITIAL_POLICY_PATH}/actor.pt" \
  "${INITIAL_POLICY_PATH}/agent_state.pt" "${INITIAL_POLICY_PATH}/reward_normalizer.pt" \
  "${INITIAL_POLICY_PATH}/replay_buffer.pt" "${INITIAL_SCORER_PATH}" \
  "${INITIAL_PAIRS_PATH}" "${INITIAL_LABELS_PATH}"
do
  [[ -s "${path}" ]] || { echo "Missing required artifact ${path}" >&2; exit 2; }
done
if [[ -n "${CONDITION}" ]]; then
  [[ -s "${RESET_DATASET_PATH}" ]] || { echo "Missing reset dataset ${RESET_DATASET_PATH}" >&2; exit 2; }
else
  for gap in g0 rr05 p5 rr03 p75; do
    [[ -s "${RESET_PARTS_ROOT}/${gap}.pt" ]] || { echo "Missing reset part ${gap}" >&2; exit 2; }
  done
fi
[[ ! -e "${RUN_ROOT}/completed.txt" ]] || { echo "Branch already completed" >&2; exit 2; }
mkdir -p "${RUN_ROOT}"

manifest_inputs=(
  --input "dataset=${DATASET_PATH}" --input "frozen_rwm=${MODEL_PATH}"
  --input "initial_actor=${INITIAL_POLICY_PATH}/actor.pt"
  --input "initial_agent_state=${INITIAL_POLICY_PATH}/agent_state.pt"
  --input "initial_scorer=${INITIAL_SCORER_PATH}"
  --input "initial_pairs=${INITIAL_PAIRS_PATH}" --input "initial_labels=${INITIAL_LABELS_PATH}"
)
if [[ -n "${CONDITION}" ]]; then
  manifest_inputs+=(--input "reset_${CONDITION}=${RESET_DATASET_PATH}")
else
  manifest_inputs+=(--input "reset_g0=${RESET_PARTS_ROOT}/g0.pt" --input "reset_rr05=${RESET_PARTS_ROOT}/rr05.pt")
  manifest_inputs+=(--input "reset_p5=${RESET_PARTS_ROOT}/p5.pt" --input "reset_rr03=${RESET_PARTS_ROOT}/rr03.pt")
  manifest_inputs+=(--input "reset_p75=${RESET_PARTS_ROOT}/p75.pt")
fi
"${PY}" scripts/reinforcement_learning/rwm_trace/artifact_manifest.py create \
  --output "${RUN_ROOT}/input_manifest.json" \
  "${manifest_inputs[@]}"

CURRENT_POLICY="$(realpath "${INITIAL_POLICY_PATH}")"
CURRENT_SCORER="$(realpath "${INITIAL_SCORER_PATH}")"
printf 'query_budget_initial=%s\nquery_budget_min=%s\nquery_budget_decay=%s\nquery_stop_cycle=%s\n' \
  "${TRACE_QUERY_BUDGET_INITIAL}" "${TRACE_QUERY_BUDGET_MIN}" "${TRACE_QUERY_BUDGET_DECAY}" \
  "${TRACE_QUERY_STOP_CYCLE}" \
  > "${RUN_ROOT}/feedback_budget_schedule.txt"

gap_args() {
  local starts
  if [[ -n "${CONDITION}" ]]; then
    starts="${TRACE_NUM_DATASET_STATES}"
  fi
  case "$1" in
    g0)   echo "${starts:-410} 0.0 1.0" ;;
    rr05) echo "${starts:-205} 0.0 0.5" ;;
    p5)   echo "${starts:-205} 5.0 1.0" ;;
    rr03) echo "${starts:-102} 0.0 0.3" ;;
    p75)  echo "${starts:-102} 7.5 1.0" ;;
  esac
}

if [[ -n "${CONDITION}" ]]; then
  GAPS=("${CONDITION}")
else
  GAPS=(g0 rr05 p5 rr03 p75)
fi

for cycle in $(seq 1 "${REFRESH_CYCLES}"); do
  CYCLE_ROOT="${RUN_ROOT}/refresh_$(printf '%02d' "${cycle}")"
  mkdir -p "${CYCLE_ROOT}/candidates" "${CYCLE_ROOT}/summaries" \
    "${CYCLE_ROOT}/feedback" "${CYCLE_ROOT}/replay" "${CYCLE_ROOT}/policy/stage"

  # A completed policy segment is the refresh commit point.  On restart, use
  # its policy/scorer as the inputs to the next refresh instead of rebuilding
  # already committed candidates, labels, and replay artifacts.
  completed_policy_file="${CYCLE_ROOT}/policy/stage/artifact_path.txt"
  completed_scorer="${CYCLE_ROOT}/feedback/go2_trace_scorer.pt"
  if [[ -s "${CYCLE_ROOT}/policy/stage/summary.json" \
        && -s "${completed_policy_file}" \
        && -s "${completed_scorer}" ]]; then
    completed_policy="$(<"${completed_policy_file}")"
    [[ -d "${completed_policy}" ]] || {
      echo "Completed refresh ${cycle} points to missing policy: ${completed_policy}" >&2
      exit 2
    }
    CURRENT_POLICY="$(realpath "${completed_policy}")"
    CURRENT_SCORER="$(realpath "${completed_scorer}")"
    write_state "${cycle}" resume_skip completed
    "${PY}" scripts/reinforcement_learning/rwm_trace/artifact_manifest.py verify \
      --manifest "${RUN_ROOT}/input_manifest.json"
    continue
  fi
  [[ -s "${CURRENT_POLICY}/replay_buffer.pt" ]] || {
    echo "TRACE requires a continuous RWMbuffer; missing ${CURRENT_POLICY}/replay_buffer.pt" >&2
    exit 2
  }
  write_state "${cycle}" collect_candidates running

  for gap in "${GAPS[@]}"; do
    read -r starts payload rr <<< "$(gap_args "${gap}")"
    envs=$((starts * TRAJECTORIES_PER_STATE))
    transitions=$((envs * TRACE_H))
    candidate="${CYCLE_ROOT}/candidates/${gap}.pt"
    if [[ -n "${CONDITION}" ]]; then
      reset_dataset="${RESET_DATASET_PATH}"
    else
      reset_dataset="${RESET_PARTS_ROOT}/${gap}.pt"
    fi
    if [[ ! -s "${candidate}" ]]; then
      "${PY}" scripts/reinforcement_learning/rwm_dataset/collect_go2_expert_command_coverage_dataset.py \
        --task Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert \
        --device cuda:0 --seed "$((4200 + cycle))" --num_envs "${envs}" \
        --num_transitions "${transitions}" --save_path "${candidate}" \
        --expert_policy_path "${CURRENT_POLICY}" --collector_mix expert:1.0 \
        --fixed_collector_assignment --collector_assignment_seed 42 \
        --dataset_obs_kind proprioceptive --action_noise_std 0.0 --failure_action_noise_std 0.0 \
        --no-use_domain_randomization --no-use_push_randomization --no-use_observation_noise \
        --payload_mass_range_kg "${payload}" "${payload}" \
        --rr_calf_strength_range "${rr}" "${rr}" --env_action_delay_steps 0 0 \
        --command_modes stand,pure_x,pure_y,pure_yaw,xy,x_yaw,y_yaw,xy_yaw \
        --command_mode_weights stand:0.08,pure_x:0.25,pure_y:0.10,pure_yaw:0.08,xy:0.14,x_yaw:0.17,y_yaw:0.05,xy_yaw:0.13 \
        --x_range -0.5 0.5 --signed_x --x_abs_range 0.08 0.50 \
        --y_abs_range 0.05 0.20 --yaw_abs_range 0.08 0.40 \
        --trace_reset_dataset "${reset_dataset}" \
        --trace_reset_mode "${RESET_MODE}" --trace_rollout_length "${TRACE_H}" \
        --trace_trajectories_per_state "${TRAJECTORIES_PER_STATE}" \
        --trace_action_temperature "${TRACE_T}" --trace_identity_tolerance "${TRACE_IDENTITY_TOLERANCE}" --headless \
        2>&1 | tee -a "${RUN_LOG}"
    fi
    summary="${CYCLE_ROOT}/summaries/${gap}.jsonl"
    if [[ ! -s "${summary}" ]]; then
      "${PY}" scripts/reinforcement_learning/rwm_trace/build_trace_replay_v4.py \
        --candidate_dataset "${candidate}" --output "${CYCLE_ROOT}/summaries/${gap}.bootstrap.pt" \
        --summary_jsonl "${summary}" --summaries_only --trajectory_length "${TRACE_H}" \
        --trajectory_stride "${TRACE_H}" --include_terminal_prefixes --minimum_terminal_length 20 \
        --terminal_penalty=-10 --failure_backprop_steps 40 --failure_backprop_penalty 2 \
        --action_saturation_penalty_scale 10 --action_delta_penalty_scale 0.10 \
        --reward_source rwm_aligned 2>&1 | tee -a "${RUN_LOG}"
    fi
  done

  ALL_SUMMARIES="${CYCLE_ROOT}/summaries/all.jsonl"
  : > "${ALL_SUMMARIES}"
  for gap in "${GAPS[@]}"; do
    cat "${CYCLE_ROOT}/summaries/${gap}.jsonl" >> "${ALL_SUMMARIES}"
  done
  PAIRS="${CYCLE_ROOT}/feedback/pairs.jsonl"
  query_budget="$("${PY}" -c \
    'import sys; initial, minimum, decay, cycle, stop = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]); print(0 if cycle > stop else max(minimum, int(initial * decay ** (cycle - 1))))' \
    "${TRACE_QUERY_BUDGET_INITIAL}" "${TRACE_QUERY_BUDGET_MIN}" "${TRACE_QUERY_BUDGET_DECAY}" "${cycle}" "${TRACE_QUERY_STOP_CYCLE}")"
  (( query_budget > 0 )) || { echo "Query budget exhausted before refresh ${cycle}" >&2; exit 2; }
  printf 'cycle_%02d=%s\n' "${cycle}" "${query_budget}" >> "${RUN_ROOT}/feedback_budget_schedule.txt"
  "${PY}" scripts/reinforcement_learning/rwm_trace/build_go2_feedback_pairs.py \
    --summaries "${ALL_SUMMARIES}" --output "${PAIRS}" --budget "${query_budget}" \
    --scorer_checkpoint "${CURRENT_SCORER}" \
    --pair_prefix "${SIDE}_${BRANCH_ID}_c$(printf '%02d' "${cycle}")" \
    --cross_start_fraction 0 --seed "$((4200 + cycle))" 2>&1 | tee -a "${RUN_LOG}"

  LABEL_DIR="${CYCLE_ROOT}/feedback/labels"
  FILTERED="${LABEL_DIR}/codex_labels_filtered.jsonl"
  minimum_labels="$("${PY}" -c \
    'import math, sys; print(max(int(sys.argv[3]), int(math.ceil(int(sys.argv[1]) * float(sys.argv[2])))))' \
    "${query_budget}" "${TRACE_MIN_FILTERED_LABEL_FRACTION}" "${TRACE_MIN_FILTERED_LABELS}")"
  while [[ ! -s "${FILTERED}" ]] || (( $(grep -cve '^[[:space:]]*$' "${FILTERED}" || true) < minimum_labels )); do
    write_state "${cycle}" label_feedback waiting
    "${PY}" scripts/reinforcement_learning/rwm_trace/label_feedback_with_codex.py \
      --prompts "${PAIRS}" --output-dir "${LABEL_DIR}" \
      --schema scripts/reinforcement_learning/rwm_trace/feedback_label_batch.schema.json \
      --repo-root "${REPO}" --batch-size 20 --confidence-threshold 0.70 \
      --minimum-filtered-labels "${minimum_labels}" --max-low-confidence-relabels 7 \
      --max-retries 2 --timeout-sec 300 --lock-path "${CODEX_LOCK}" \
      --codex-service-tier default 2>&1 | tee -a "${RUN_LOG}" || true
    [[ -s "${FILTERED}" ]] && count=$(grep -cve '^[[:space:]]*$' "${FILTERED}" || true) || count=0
    (( count >= minimum_labels )) || sleep 120
  done

  CUM_PAIRS="${CYCLE_ROOT}/feedback/cumulative_pairs.jsonl"
  CUM_LABELS="${CYCLE_ROOT}/feedback/cumulative_labels.jsonl"
  cat "${INITIAL_PAIRS_PATH}" > "${CUM_PAIRS}"
  find "${RUN_ROOT}" -path '*/feedback/pairs.jsonl' -print0 | sort -z | xargs -0 cat >> "${CUM_PAIRS}"
  cat "${INITIAL_LABELS_PATH}" > "${CUM_LABELS}"
  find "${RUN_ROOT}" -path '*/feedback/labels/codex_labels_filtered.jsonl' -print0 | sort -z | xargs -0 cat >> "${CUM_LABELS}"
  SCORER="${CYCLE_ROOT}/feedback/go2_trace_scorer.pt"
  write_state "${cycle}" train_scorer running
  "${PY}" scripts/reinforcement_learning/rwm_trace/train_go2_trace_scorer.py \
    --labels "${CUM_LABELS}" --pairs "${CUM_PAIRS}" --output "${SCORER}" \
    --epochs 50 --learning_rate 0.001 --hidden_dim 256 --confidence_threshold 0.70 \
    --val_ratio 0.20 --seed 42 2>&1 | tee -a "${RUN_LOG}"
  "${PY}" - "${SCORER}" <<'PY'
import math, sys, torch
m = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["metadata"]
n, acc = int(m["num_val_pairs"]), float(m["val_balanced_accuracy"])
if n < 40 or not math.isfinite(acc) or acc < 0.60:
    raise SystemExit(f"scorer quality gate failed: num_val={n}, val_balanced_accuracy={acc}")
PY
  CURRENT_SCORER="$(realpath "${SCORER}")"

  merge_args=()
  for gap in "${GAPS[@]}"; do
    replay="${CYCLE_ROOT}/replay/${gap}.pt"
    "${PY}" scripts/reinforcement_learning/rwm_trace/build_trace_replay_v4.py \
      --candidate_dataset "${CYCLE_ROOT}/candidates/${gap}.pt" --output "${replay}" \
      --scorer_checkpoint "${CURRENT_SCORER}" --selection scorer --select_ratio "${SELECT_RATIO}" \
      --trajectory_length "${TRACE_H}" --trajectory_stride "${TRACE_H}" \
      --include_terminal_prefixes --minimum_terminal_length 20 --failure_transition_ratio 0.0 \
      --terminal_penalty=-10 --failure_backprop_steps 40 --failure_backprop_penalty 2 \
      --action_saturation_penalty_scale 10 --action_delta_penalty_scale 0.10 \
      --n_step 3 --gamma 0.99 --reward_source rwm_aligned --seed 42 \
      2>&1 | tee -a "${RUN_LOG}"
    merge_args+=(--input "${gap}=${replay}")
  done
  MERGED_REPLAY="${CYCLE_ROOT}/replay/pooled.pt"
  "${PY}" scripts/reinforcement_learning/rwm_trace/merge_trace_replays_v5.py \
    "${merge_args[@]}" --output "${MERGED_REPLAY}" 2>&1 | tee -a "${RUN_LOG}"

  write_state "${cycle}" train_policy_segment running
  PREVIOUS_POLICY="$(realpath "${CURRENT_POLICY}")"
  if [[ ! -s "${CYCLE_ROOT}/policy/stage/summary.json" ]]; then
    STAGE_DIR="${CYCLE_ROOT}/policy/stage" OUTPUT_DIR="${CYCLE_ROOT}/policy/run" \
      DATASET_PATH="${DATASET_PATH}" MODEL_PATH="${MODEL_PATH}" P_ID=P0 DEVICE=cuda:0 \
      SEED=300 NUM_ENV_STEPS="${SEGMENT_ENV_STEPS}" POLICY_RESUME_PATH="${CURRENT_POLICY}" \
      SAVE_REPLAY_BUFFER=true LOAD_REWARD_NORMALIZER=true TRACE_REPLAY_PATH="${MERGED_REPLAY}" \
      TRACE_REPLAY_RATIO="${REPLAY_RATIO}" LIN_VEL_X_RANGE='-0.5 0.5' \
      LIN_VEL_Y_RANGE='-0.2 0.2' ANG_VEL_Z_RANGE='-0.4 0.4' \
      bash scripts/reinforcement_learning/go2_normal_proprioceptive/04_train_normal_policy.sh \
      2>&1 | tee -a "${RUN_LOG}"
  fi
  CURRENT_POLICY="$(<"${CYCLE_ROOT}/policy/stage/artifact_path.txt")"
  [[ -s "${CURRENT_POLICY}/replay_buffer.pt" ]] || {
    echo "Policy segment did not save its final RWMbuffer: ${CURRENT_POLICY}" >&2
    exit 2
  }
  "${PY}" scripts/reinforcement_learning/rwm_trace/artifact_manifest.py verify \
    --manifest "${RUN_ROOT}/input_manifest.json"
  LAST_COMPLETED_CYCLE="${cycle}"
  if [[ -n "${POST_REFRESH_HOOK}" ]]; then
    hook_status=0
    "${POST_REFRESH_HOOK}" "${cycle}" "${CURRENT_POLICY}" "${RUN_ROOT}" || hook_status=$?
    if [[ "${hook_status}" -eq 20 ]]; then
      STOP_REASON="pilot_early_stop_hook"
      printf 'time=%s\ncycle=%s\nreason=%s\npolicy=%s\n' \
        "$(date --iso-8601=seconds)" "${cycle}" "${STOP_REASON}" "${CURRENT_POLICY}" \
        > "${RUN_ROOT}/stopped_early.txt"
      write_state "${cycle}" pilot_early_stop stopped
      break
    elif [[ "${hook_status}" -ne 0 ]]; then
      echo "Post-refresh hook failed at cycle ${cycle} with status ${hook_status}" >&2
      exit "${hook_status}"
    fi
  fi
  # Once the new policy+buffer is committed, retain only the latest branch
  # buffer.  Never remove the shared common-warmup buffer used to start r10
  # and r25 independently.
  case "${PREVIOUS_POLICY}" in
    "${RUN_ROOT}"/refresh_*/policy/run/step4882)
      [[ "${PREVIOUS_POLICY}" == "${CURRENT_POLICY}" ]] || \
        rm -f -- "${PREVIOUS_POLICY}/replay_buffer.pt"
      ;;
  esac
done

if [[ -z "${STOP_REASON}" ]]; then
  printf 'side=%s\ncondition=%s\nbranch=%s\nfinal_policy=%s\ntemperature=%s\nhorizon=%s\ncycles=%s\nreplay_ratio=%s\nselect_ratio=%s\n' \
    "${SIDE}" "${CONDITION:-pooled}" "${BRANCH_ID}" "${CURRENT_POLICY}" "${TRACE_T}" "${TRACE_H}" \
    "${REFRESH_CYCLES}" "${REPLAY_RATIO}" "${SELECT_RATIO}" > "${RUN_ROOT}/completed.txt"
  write_state "${REFRESH_CYCLES}" done completed
fi
