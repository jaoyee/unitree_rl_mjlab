#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${SIDE:?Set sim or real}"
: "${CONDITION:?Set g0, rr05, rr03, p5, or p75}"
: "${RUN_ROOT:?Set a new V10 branch root}"
: "${DATASET_PATH:?Set the immutable condition 25K dataset}"
: "${MODEL_PATH:?Set the frozen condition RWM model_5000.pt}"
: "${INITIAL_POLICY_PATH:?Set the common warmup checkpoint directory}"
: "${INITIAL_SCORER_ROOT:?Set the matching V10 initial scorer directory}"
: "${V10_GPU_POOL:?Set comma-separated physical GPU IDs}"
RUN_KIND="${RUN_KIND:-formal}"
PY="${PYTHON_BIN:-${REPO}/.venv/bin/python}"
PROTOCOL="${REPO}/scripts/reinforcement_learning/rwm_trace/trace_v10_protocol.json"
GPU_EXEC="${REPO}/scripts/reinforcement_learning/go2_sim_gap_aligned/run_v10_gpu_stage.sh"
MODES=(stand pure_x pure_y pure_yaw xy x_yaw y_yaw xy_yaw)
case "${SIDE}" in sim|real) ;; *) exit 2 ;; esac
case "${CONDITION}" in g0|rr05|rr03|p5|p75) ;; *) exit 2 ;; esac
case "${RUN_KIND}" in formal|pilot|preflight) ;; *) echo "Bad RUN_KIND=${RUN_KIND}" >&2; exit 2 ;; esac
for path in "${PY}" "${DATASET_PATH}" "${MODEL_PATH}" "${PROTOCOL}" \
  "${INITIAL_POLICY_PATH}/actor.pt" "${INITIAL_POLICY_PATH}/agent_state.pt" \
  "${INITIAL_POLICY_PATH}/reward_normalizer.pt" "${INITIAL_POLICY_PATH}/replay_buffer.pt" \
  "${INITIAL_SCORER_ROOT}/ready.env"; do
  [[ -e "${path}" ]] || { echo "Missing V10 input: ${path}" >&2; exit 2; }
done
mkdir -p "${RUN_ROOT}"
cd "${REPO}"
source "${INITIAL_SCORER_ROOT}/ready.env"
[[ "${TRACE_PROTOCOL}" == go2_trace_v10 && "${INITIAL_SCORER_CONDITION}" == "${CONDITION}" ]] || {
  echo "Initial scorer condition/protocol mismatch" >&2; exit 2;
}
[[ "${TRACE_PROTOCOL_SHA256}" == "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" ]] || {
  echo "Initial scorer protocol SHA mismatch" >&2; exit 2;
}
[[ "${INITIAL_SCORER_DATASET_SHA256}" == "$(sha256sum "${DATASET_PATH}" | awk '{print $1}')" ]] || {
  echo "Initial scorer dataset SHA mismatch" >&2; exit 2;
}

read -r CYCLES STEPS IMAG_ENVS H T RATIO HIDDEN GLOBAL_ACC MODE_ACC MODE_VAL CONFIDENCE <<< "$(${PY} - "${PROTOCOL}" <<'PY'
import json, sys
p=json.load(open(sys.argv[1])); c=p["candidate"]; s=p["scorer"]; r=p["replay"]; t=p["training"]
print(t["refresh_cycles"], t["environment_steps_per_refresh"], t["imagination_environments"],
      c["horizon"], c["action_temperature"],
      r["ratio"], s["hidden_dimension"], s["minimum_global_balanced_accuracy"],
      s["minimum_mode_balanced_accuracy"], s["minimum_mode_validation_count"], s["confidence_threshold"])
PY
)"
[[ "${RUN_KIND}" == pilot || "${RUN_KIND}" == preflight ]] && CYCLES=1
CODEX_LOCK="${CODEX_LOCK:-$(dirname "${RUN_ROOT}")/codex_feedback.lock}"
STATE="${RUN_ROOT}/state.txt"
RUN_LOG="${RUN_ROOT}/run.log"
MANIFEST="${RUN_ROOT}/replay_manifest.json"

write_state() {
  local cycle="$1" stage="$2" status="$3"
  local percent=$(( (100 * (cycle - 1)) / CYCLES ))
  [[ "${status}" == completed ]] && percent=$((100 * cycle / CYCLES))
  printf 'time=%s\nprotocol=go2_trace_v10\nside=%s\ncondition=%s\ncycle=%s/%s\nstage=%s\nstatus=%s\nprogress_percent=%s\ngpu_pool=%s\n' \
    "$(date --iso-8601=seconds)" "${SIDE}" "${CONDITION}" "${cycle}" "${CYCLES}" \
    "${stage}" "${status}" "${percent}" "${V10_GPU_POOL}" > "${STATE}.new"
  mv "${STATE}.new" "${STATE}"
}

CURRENT_POLICY="$(realpath "${INITIAL_POLICY_PATH}")"
CURRENT_SCORER="$(realpath "${INITIAL_SCORER_PATH}")"
LAST_COMMITTED=0
for cycle in $(seq 1 "${CYCLES}"); do
  ROOT="${RUN_ROOT}/refresh_$(printf '%02d' "${cycle}")"
  [[ -s "${ROOT}/cycle_commit.json" ]] || break
  LAST_COMMITTED="${cycle}"
  CURRENT_POLICY="$(realpath "$(<"${ROOT}/policy/stage/artifact_path.txt")")"
  CURRENT_SCORER="$(realpath "${ROOT}/feedback/go2_trace_scorer.pt")"
done
if (( LAST_COMMITTED > 0 )); then
  ROOT="${RUN_ROOT}/refresh_$(printf '%02d' "${LAST_COMMITTED}")"
  "${PY}" scripts/reinforcement_learning/rwm_trace/v10_cycle_commit.py verify \
    --policy "${CURRENT_POLICY}" --scorer "${CURRENT_SCORER}" \
    --manifest "${ROOT}/replay/committed_manifest.json" --protocol "${PROTOCOL}" \
    --cycle "${LAST_COMMITTED}" --output "${ROOT}/cycle_commit.json"
fi
START_CYCLE=$((LAST_COMMITTED + 1))
if (( START_CYCLE <= CYCLES )); then
for cycle in $(seq "${START_CYCLE}" "${CYCLES}"); do
  CYCLE="$(printf '%02d' "${cycle}")"
  ROOT="${RUN_ROOT}/refresh_${CYCLE}"
  COMMIT="${ROOT}/cycle_commit.json"
  mkdir -p "${ROOT}/feedback/labels" "${ROOT}/feedback/seeds" "${ROOT}/replay" "${ROOT}/policy"
  [[ -s "${CURRENT_POLICY}/replay_buffer.pt" ]] || { echo "RWM buffer continuity broken" >&2; exit 2; }

  write_state "${cycle}" collect_candidates running
  SIDE="${SIDE}" DATASET_PATH="${DATASET_PATH}" POLICY_PATH="${CURRENT_POLICY}" \
    CYCLE_ROOT="${ROOT}" REFRESH_CYCLE="${cycle}" V10_GPU_POOL="${V10_GPU_POOL}" REPO="${REPO}" \
    bash scripts/reinforcement_learning/go2_sim_gap_aligned/collect_go2_trace_v10_candidates.sh \
    2>&1 | tee -a "${RUN_LOG}"

  CANDIDATE="${ROOT}/candidates.pt"
  REFERENCE="${ROOT}/rwm_reference.jsonl"
  write_state "${cycle}" rwm_reference running
  V10_GPU_POOL="${V10_GPU_POOL}" "${GPU_EXEC}" "${PY}" \
    scripts/reinforcement_learning/rwm_trace/build_go2_rwm_reference_summaries.py \
    --candidate_dataset "${CANDIDATE}" --source_dataset "${DATASET_PATH}" \
    --protocol "${PROTOCOL}" --model_path "${MODEL_PATH}" --policy_path "${CURRENT_POLICY}" \
    --output "${REFERENCE}" --device cuda:0 --horizon "${H}" --batch_size 256 \
    --action_temperature "${T}" --seed "$((30420 + cycle))"

  SUMMARIES="${ROOT}/summaries.jsonl"
  "${PY}" scripts/reinforcement_learning/rwm_trace/build_trace_replay_v10.py \
    --candidate_dataset "${CANDIDATE}" --target_dataset "${DATASET_PATH}" \
    --protocol "${PROTOCOL}" --refresh_cycle "${cycle}" --policy_checkpoint "${CURRENT_POLICY}" \
    --output "${ROOT}/summaries_only.pt" --summary_jsonl "${SUMMARIES}" --summaries_only \
    --reference_summaries "${REFERENCE}"

  QUERY_BUDGET="$(${PY} - "${PROTOCOL}" "${cycle}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["scorer"]["refresh_query_budgets"][int(sys.argv[2])-1])
PY
)"
  PAIR_POOL="${ROOT}/feedback/pair_pool.jsonl"
  PAIRS="${ROOT}/feedback/pairs.jsonl"
  write_state "${cycle}" prepare_feedback running
  "${PY}" scripts/reinforcement_learning/rwm_trace/build_go2_partitioned_feedback_pairs.py \
    --summaries "${SUMMARIES}" --output "${PAIR_POOL}" \
    --partition_manifest "${ROOT}/feedback/pair_partition.json" \
    --scorer_checkpoint "${CURRENT_SCORER}" --condition "${CONDITION}" \
    --budget "$((QUERY_BUDGET * 2))" --validation_fraction 0.20 \
    --required_pair_modes "${MODES[@]}" --min_pairs_per_mode_per_partition 4 \
    --pair_prefix "v10_${SIDE}_${CONDITION}_c${CYCLE}" --seed 1042
  "${PY}" scripts/reinforcement_learning/rwm_trace/schedule_v10_feedback_queries.py \
    --pool "${PAIR_POOL}" --output "${PAIRS}" --budget "${QUERY_BUDGET}" \
    --minimum-per-partition-mode 4 --seed "$((40420 + cycle))"
  "${PY}" scripts/reinforcement_learning/rwm_trace/audit_go2_pair_partition.py \
    --pairs "${PAIRS}" --output "${ROOT}/feedback/pair_audit.json" \
    --required_modes "${MODES[@]}" --min_pairs_per_partition_mode 4

  write_state "${cycle}" label_feedback running
  "${PY}" scripts/reinforcement_learning/rwm_trace/label_feedback_with_codex.py \
    --prompts "${PAIRS}" --output-dir "${ROOT}/feedback/labels" \
    --schema scripts/reinforcement_learning/rwm_trace/feedback_label_batch.schema.json \
    --repo-root "${REPO}" --batch-size 16 --confidence-threshold "${CONFIDENCE}" \
    --minimum-filtered-labels 64 --max-low-confidence-relabels 8 --max-retries 2 \
    --timeout-sec 300 --lock-path "${CODEX_LOCK}" --codex-service-tier default
  LABELS="${ROOT}/feedback/labels/codex_labels_filtered.jsonl"
  "${PY}" scripts/reinforcement_learning/rwm_trace/audit_go2_pair_partition.py \
    --pairs "${PAIRS}" --labels "${LABELS}" --output "${ROOT}/feedback/label_audit.json" \
    --required_modes "${MODES[@]}" --min_pairs_per_partition_mode 4 \
    --min_labels_per_partition_mode 4 --min_labels_per_side 1 \
    --confidence_threshold "${CONFIDENCE}"

  pair_inputs=(--input "${INITIAL_PAIRS_PATH}")
  label_inputs=(--input "${INITIAL_LABELS_PATH}")
  for previous in $(seq 1 "${cycle}"); do
    previous_root="${RUN_ROOT}/refresh_$(printf '%02d' "${previous}")/feedback"
    pair_inputs+=(--input "${previous_root}/pairs.jsonl")
    label_inputs+=(--input "${previous_root}/labels/codex_labels_filtered.jsonl")
  done
  CUM_PAIRS="${ROOT}/feedback/cumulative_pairs.jsonl"
  CUM_LABELS="${ROOT}/feedback/cumulative_labels.jsonl"
  "${PY}" scripts/reinforcement_learning/rwm_trace/merge_jsonl_atomic.py "${pair_inputs[@]}" --output "${CUM_PAIRS}"
  "${PY}" scripts/reinforcement_learning/rwm_trace/merge_jsonl_atomic.py "${label_inputs[@]}" --output "${CUM_LABELS}"

  write_state "${cycle}" train_scorer running
  scorer_candidates=()
  for seed in 42 43 44 45 46; do
    checkpoint="${ROOT}/feedback/seeds/scorer_seed${seed}.pt"
    "${PY}" scripts/reinforcement_learning/rwm_trace/train_go2_trace_scorer.py \
      --labels "${CUM_LABELS}" --pairs "${CUM_PAIRS}" --output "${checkpoint}" \
      --epochs 80 --learning_rate 0.001 --hidden_dim "${HIDDEN}" \
      --confidence_threshold "${CONFIDENCE}" --required_val_modes "${MODES[@]}" \
      --min_val_mode_count "${MODE_VAL}" --seed "${seed}"
    scorer_candidates+=(--candidate "${checkpoint}")
  done
  SCORER="${ROOT}/feedback/go2_trace_scorer.pt"
  "${PY}" scripts/reinforcement_learning/rwm_trace/select_go2_scorer_checkpoint.py \
    "${scorer_candidates[@]}" --output "${SCORER}" --report "${ROOT}/feedback/seed_selection.json" \
    --required_modes "${MODES[@]}" --min_global_balanced_accuracy "${GLOBAL_ACC}" \
    --min_mode_balanced_accuracy "${MODE_ACC}" --min_mode_count "${MODE_VAL}"
  "${PY}" scripts/reinforcement_learning/rwm_trace/validate_v10_scorer.py \
    --checkpoint "${SCORER}" --output "${ROOT}/feedback/scorer_validation.json" \
    --minimum-global-balanced-accuracy "${GLOBAL_ACC}" \
    --minimum-mode-balanced-accuracy "${MODE_ACC}" --minimum-mode-validation-count "${MODE_VAL}"

  SHARD="${ROOT}/replay/trace_shard.pt"
  write_state "${cycle}" build_replay running
  "${PY}" scripts/reinforcement_learning/rwm_trace/build_trace_replay_v10.py \
    --candidate_dataset "${CANDIDATE}" --target_dataset "${DATASET_PATH}" \
    --protocol "${PROTOCOL}" --refresh_cycle "${cycle}" --policy_checkpoint "${CURRENT_POLICY}" \
    --output "${SHARD}" --summary_jsonl "${ROOT}/replay/scored_summaries.jsonl" \
    --scorer_checkpoint "${SCORER}" --selection scorer --reference_summaries "${REFERENCE}" --seed 42
  "${PY}" scripts/reinforcement_learning/rwm_trace/audit_trace_replay_ood.py \
    --dataset "${DATASET_PATH}" --replay "${SHARD}" --output "${ROOT}/replay/ood_audit.json" --seed 42
  if [[ "${RUN_KIND}" == pilot || "${RUN_KIND}" == preflight ]]; then
    "${PY}" scripts/reinforcement_learning/rwm_trace/audit_v10_candidate_preflight.py \
      --dataset "${DATASET_PATH}" --replay "${SHARD}" \
      --summaries "${ROOT}/replay/scored_summaries.jsonl" --protocol "${PROTOCOL}" \
      --output "${ROOT}/replay/candidate_preflight.json"
  fi
  if [[ "${RUN_KIND}" == preflight ]]; then
    cat > "${RUN_ROOT}/preflight_passed.env.new" <<EOF
passed=true
protocol=go2_trace_v10
protocol_sha256=$(sha256sum "${PROTOCOL}" | awk '{print $1}')
side=${SIDE}
condition=${CONDITION}
candidate_preflight=$(realpath "${ROOT}/replay/candidate_preflight.json")
EOF
    mv "${RUN_ROOT}/preflight_passed.env.new" "${RUN_ROOT}/preflight_passed.env"
    write_state "${cycle}" candidate_preflight completed
    exit 0
  fi
  STAGED="${ROOT}/replay/staged_manifest.json"
  stage_args=()
  [[ -s "${MANIFEST}" ]] && stage_args+=(--committed-manifest "${MANIFEST}")
  "${PY}" scripts/reinforcement_learning/rwm_trace/v10_replay_manifest.py stage \
    --output "${STAGED}" "${stage_args[@]}" --shard "${SHARD}" --cycle "${cycle}" \
    --protocol "${PROTOCOL}" --side "${SIDE}" --condition "${CONDITION}"

  PREVIOUS_POLICY="${CURRENT_POLICY}"
  write_state "${cycle}" train_policy running
  if [[ ! -s "${ROOT}/policy/stage/summary.json" ]]; then
    if [[ -d "${ROOT}/policy/run" && -n "$(find "${ROOT}/policy/run" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
      echo "Uncommitted partial policy output requires manual quarantine: ${ROOT}/policy/run" >&2
      exit 2
    fi
    V10_GPU_POOL="${V10_GPU_POOL}" "${GPU_EXEC}" env \
      STAGE_DIR="${ROOT}/policy/stage" OUTPUT_DIR="${ROOT}/policy/run" \
      DATASET_PATH="${DATASET_PATH}" MODEL_PATH="${MODEL_PATH}" P_ID=P0 DEVICE=cuda:0 \
      SEED=300 NUM_ENV_STEPS="${STEPS}" NUM_IMAGINATION_ENVS="${IMAG_ENVS}" \
      POLICY_RESUME_PATH="${CURRENT_POLICY}" \
      SAVE_REPLAY_BUFFER=true LOAD_REWARD_NORMALIZER=true TRACE_REPLAY_PATH="${STAGED}" \
      TRACE_REPLAY_RATIO="${RATIO}" LIN_VEL_X_RANGE='-0.5 0.5' \
      LIN_VEL_Y_RANGE='-0.2 0.2' ANG_VEL_Z_RANGE='-0.4 0.4' \
      bash scripts/reinforcement_learning/go2_normal_proprioceptive/04_train_normal_policy.sh \
      2>&1 | tee -a "${RUN_LOG}"
  fi
  POLICY="$(<"${ROOT}/policy/stage/artifact_path.txt")"
  [[ -s "${POLICY}/replay_buffer.pt" ]] || { echo "Final RWM buffer missing" >&2; exit 2; }
  "${PY}" scripts/reinforcement_learning/rwm_trace/v10_replay_manifest.py commit \
    --staged "${STAGED}" --output "${MANIFEST}" --policy-checkpoint "${POLICY}" --delete-expired
  cp --reflink=auto "${MANIFEST}" "${ROOT}/replay/committed_manifest.json.new"
  mv "${ROOT}/replay/committed_manifest.json.new" "${ROOT}/replay/committed_manifest.json"

  write_state "${cycle}" behavior_diagnostic running
  V10_GPU_POOL="${V10_GPU_POOL}" "${GPU_EXEC}" env REPO="${REPO}" GAP_ID="${CONDITION}" \
    CHECKPOINT="${POLICY}" OUTPUT_ROOT="${ROOT}/behavior" DEVICE=cuda:0 \
    NUM_ENVS=64 STEPS=2400 SEEDS=901 MODES=clean \
    bash scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_command_eval_v4.sh \
    2>&1 | tee -a "${RUN_LOG}"

  "${PY}" scripts/reinforcement_learning/rwm_trace/v10_cycle_commit.py create \
    --policy "${POLICY}" --scorer "${SCORER}" --manifest "${ROOT}/replay/committed_manifest.json" \
    --protocol "${PROTOCOL}" --cycle "${cycle}" --output "${COMMIT}"
  CURRENT_POLICY="$(realpath "${POLICY}")"; CURRENT_SCORER="$(realpath "${SCORER}")"
  case "${PREVIOUS_POLICY}" in
    "${RUN_ROOT}"/refresh_*/policy/run/step*) rm -f -- "${PREVIOUS_POLICY}/replay_buffer.pt" ;;
  esac
  rm -f -- "${CANDIDATE}"
  write_state "${cycle}" refresh completed
done
fi

cat > "${RUN_ROOT}/completed.txt.new" <<EOF
protocol=go2_trace_v10
side=${SIDE}
condition=${CONDITION}
final_policy=${CURRENT_POLICY}
cycles=${CYCLES}
temperature=${T}
horizon=${H}
replay_ratio=${RATIO}
EOF
mv "${RUN_ROOT}/completed.txt.new" "${RUN_ROOT}/completed.txt"
write_state "${CYCLES}" done completed
