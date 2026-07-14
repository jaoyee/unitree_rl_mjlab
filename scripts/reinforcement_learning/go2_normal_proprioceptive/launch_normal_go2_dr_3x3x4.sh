#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-normal_go2_dr_3x3x4_seed0_$(date +%Y-%m-%d_%H-%M-%S)}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-logs/experiments/go2_normal_dr_3x3x4/${RUN_ID}}"
DRY_RUN="${DRY_RUN:-true}"
CONFIRM_RUN="${CONFIRM_RUN:-false}"
RESUME="${RESUME:-false}"
MATRIX_SELECTION="${MATRIX_SELECTION:-all}"
INCLUDE_EVALUATIONS="${INCLUDE_EVALUATIONS:-false}"
GPU_POOL="${GPU_POOL:-0 1 2 3 4 5 6 7}"
GPU_MAX_MEMORY_USED_MIB="${GPU_MAX_MEMORY_USED_MIB:-6000}"
GPU_MAX_UTILIZATION="${GPU_MAX_UTILIZATION:-20}"
POLL_SECONDS="${POLL_SECONDS:-30}"

EXPERT_NUM_ENVS="${EXPERT_NUM_ENVS:-1024}"
EXPERT_NUM_ENV_STEPS="${EXPERT_NUM_ENV_STEPS:-100000000}"
DATASET_NUM_ENVS="${DATASET_NUM_ENVS:-1024}"
DATASET_SHARD_TRANSITIONS="${DATASET_SHARD_TRANSITIONS:-249856 249856 249856 250880}"
DATASET_SHARD_TRANSITIONS="${DATASET_SHARD_TRANSITIONS//,/ }"
DATASET_MINIMUM_TRANSITIONS="${DATASET_MINIMUM_TRANSITIONS:-1000000}"
RWM_MAX_ITERATIONS="${RWM_MAX_ITERATIONS:-5000}"
RWM_BATCH_SIZE="${RWM_BATCH_SIZE:-1024}"
RWM_MICRO_BATCH_SIZE="${RWM_MICRO_BATCH_SIZE:-256}"
RWM_NUM_EVAL_SEQUENCES="${RWM_NUM_EVAL_SEQUENCES:-4096}"
POLICY_NUM_IMAGINATION_ENVS="${POLICY_NUM_IMAGINATION_ENVS:-1024}"
POLICY_NUM_ENV_STEPS="${POLICY_NUM_ENV_STEPS:-50000000}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-256}"
EVAL_STEPS="${EVAL_STEPS:-2400}"
EVAL_COMMAND_SWITCH_STEPS="${EVAL_COMMAND_SWITCH_STEPS:-300}"

if [[ "${EXPERIMENT_ROOT}" == *0p5* || "${EXPERIMENT_ROOT}" == *broken* ]]; then
  echo "Normal experiment root contains a forbidden 0.5/broken marker: ${EXPERIMENT_ROOT}" >&2
  exit 1
fi
if [[ "${DRY_RUN}" != "true" && "${DRY_RUN}" != "false" ]]; then
  echo "DRY_RUN must be true or false" >&2
  exit 1
fi
if [[ "${DRY_RUN}" == "false" && "${CONFIRM_RUN}" != "true" ]]; then
  echo "Actual execution requires DRY_RUN=false CONFIRM_RUN=true" >&2
  exit 1
fi
if [[ -e "${EXPERIMENT_ROOT}" && "${RESUME}" != "true" ]]; then
  echo "Experiment root already exists; use RESUME=true only for the same frozen RUN_ID" >&2
  exit 1
fi

for required in \
  configs/flashsac_go2_normal_proprioceptive_expert.yaml \
  scripts/reinforcement_learning/go2_normal_proprioceptive/01_train_normal_expert.sh \
  scripts/reinforcement_learning/go2_normal_proprioceptive/02_collect_normal_dataset_sharded.sh \
  scripts/reinforcement_learning/go2_normal_proprioceptive/03_train_normal_rwm.sh \
  scripts/reinforcement_learning/go2_normal_proprioceptive/04_train_normal_policy.sh \
  scripts/reinforcement_learning/go2_normal_proprioceptive/05_evaluate_normal_policy.sh \
  scripts/reinforcement_learning/go2_normal_proprioceptive/verify_normal_go2_dr.py \
  scripts/reinforcement_learning/go2_normal_proprioceptive/normal_go2_dr_scheduler.sh; do
  [[ -f "${required}" ]] || { echo "Missing required file: ${required}" >&2; exit 1; }
done

grep -q 'Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert' configs/flashsac_go2_normal_proprioceptive_expert.yaml
"${REPO_ROOT}/.venv/bin/python" - <<'PY'
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/flashsac_go2_normal_proprioceptive_expert.yaml")
assert cfg.env.env_name == "Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert"
assert list(cfg.env.broken_joint_names) == []
assert dict(cfg.env.joint_strength_scales) == {}
assert list(cfg.env.action_mask_indices) == []
PY
bash -n "${SCRIPT_DIR}"/*.sh
"${REPO_ROOT}/.venv/bin/python" -m py_compile \
  flash_rl/envs/mjlab.py \
  flash_rl/envs/mjlab_dr.py \
  src/assets/robots/unitree_go2/go2_constants.py \
  src/tasks/velocity/mdp/rewards.py \
  src/tasks/rwm_velocity/config/go2/__init__.py \
  src/tasks/rwm_velocity/config/go2/env_cfgs.py \
  scripts/reinforcement_learning/rwm_dataset/collect_go2_expert_command_coverage_dataset.py \
  scripts/reinforcement_learning/rwm_dataset/merge_go2_dataset_shards.py \
  scripts/reinforcement_learning/rwm_dataset/validate_go2_dataset.py \
  scripts/reinforcement_learning/rwm_flashsac/world_model_env.py \
  scripts/reinforcement_learning/rwm_flashsac/eval_flashsac_go2_mjlab_proprioceptive.py \
  scripts/reinforcement_learning/go2_normal_proprioceptive/verify_normal_go2_dr.py

mkdir -p "${EXPERIMENT_ROOT}/queues" "${EXPERIMENT_ROOT}/source_snapshot" "${EXPERIMENT_ROOT}/summaries"
JOBS_FILE="${EXPERIMENT_ROOT}/queues/jobs.list"
QUEUE_LOG="${EXPERIMENT_ROOT}/queues/queue_events.log"
: > "${JOBS_FILE}"
touch "${QUEUE_LOG}"

SOURCE_FILES=(
  flash_rl/envs/mjlab.py
  flash_rl/envs/mjlab_dr.py
  scripts/train_flashsac.py
  src/assets/robots/unitree_go2/go2_constants.py
  src/tasks/velocity/mdp/rewards.py
  src/tasks/rwm_velocity/config/go2/__init__.py
  src/tasks/rwm_velocity/config/go2/env_cfgs.py
  configs/flashsac_go2_normal_proprioceptive_expert.yaml
  scripts/reinforcement_learning/rwm_dataset/collect_go2_expert_command_coverage_dataset.py
  scripts/reinforcement_learning/rwm_dataset/dataset.py
  scripts/reinforcement_learning/rwm_dataset/merge_go2_dataset_shards.py
  scripts/reinforcement_learning/rwm_dataset/validate_go2_dataset.py
  scripts/reinforcement_learning/rwm_flashsac/world_model_env.py
  scripts/reinforcement_learning/rwm_flashsac/eval_flashsac_go2_mjlab_proprioceptive.py
  scripts/reinforcement_learning/go2_normal_proprioceptive/01_train_normal_expert.sh
  scripts/reinforcement_learning/go2_normal_proprioceptive/02_collect_normal_dataset_sharded.sh
  scripts/reinforcement_learning/go2_normal_proprioceptive/03_train_normal_rwm.sh
  scripts/reinforcement_learning/go2_normal_proprioceptive/04_train_normal_policy.sh
  scripts/reinforcement_learning/go2_normal_proprioceptive/05_evaluate_normal_policy.sh
  scripts/reinforcement_learning/go2_normal_proprioceptive/verify_normal_go2_dr.py
  scripts/reinforcement_learning/go2_normal_proprioceptive/run_stage_job.sh
  scripts/reinforcement_learning/go2_normal_proprioceptive/normal_go2_dr_scheduler.sh
  scripts/reinforcement_learning/go2_normal_proprioceptive/launch_normal_go2_dr_3x3x4.sh
  scripts/export_flashsac_go2_normal_proprioceptive.py
  deploy/robots/go2/config/policy/velocity/rwm_proprioceptive_normal/params/deploy.yaml
  scripts/reinforcement_learning/go2_normal_proprioceptive/NORMAL_GO2_DR_3X3X4.md
)
sha256sum "${SOURCE_FILES[@]}" > "${EXPERIMENT_ROOT}/source_snapshot/files.sha256"

declare -A E_NAME E_PRESET E_SCALE D_NAME D_PRESET D_SCALE
E_NAME[E0]=calibrated_default
E_NAME[E1]=calibrated_friend_half
E_NAME[E2]=calibrated_friend_full
E_PRESET[E0]=calibrated_default
E_PRESET[E1]=calibrated_friend_flat
E_PRESET[E2]=calibrated_friend_flat
E_SCALE[E0]=1.0
E_SCALE[E1]=0.5
E_SCALE[E2]=1.0
D_NAME[D0]=calibrated_default
D_NAME[D1]=calibrated_friend_half
D_NAME[D2]=calibrated_friend_full
D_PRESET[D0]=calibrated_default
D_PRESET[D1]=calibrated_friend_flat
D_PRESET[D2]=calibrated_friend_flat
D_SCALE[D0]=1.0
D_SCALE[D1]=0.5
D_SCALE[D2]=1.0

policy_selected() {
  local e="$1" d="$2" p="$3"
  [[ "${MATRIX_SELECTION}" == "all" ]] && return 0
  [[ ",${MATRIX_SELECTION}," == *",${e}_${d}_${p},"* ]]
}

pair_selected() {
  local e="$1" d="$2" p
  for p in P0 P1 P2 P3; do
    policy_selected "${e}" "${d}" "${p}" && return 0
  done
  return 1
}

expert_selected() {
  local e="$1" d
  for d in D0 D1 D2; do
    pair_selected "${e}" "${d}" && return 0
  done
  return 1
}

initial_status() {
  [[ "${DRY_RUN}" == "true" ]] && printf 'dry_run\n' || printf 'pending\n'
}

register_job() {
  local job_id="$1" job_dir="$2" dependencies="$3"
  mkdir -p "${job_dir}"
  printf '%s|%s|%s\n' "${job_id}" "$(realpath "${job_dir}")" "${dependencies}" >> "${JOBS_FILE}"
  if [[ -f "${job_dir}/summary.json" ]] && \
     "${REPO_ROOT}/.venv/bin/python" - "${job_dir}/summary.json" <<'PY' >/dev/null 2>&1
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("status") == "completed" else 1)
PY
  then
    status=completed
  else
    status="$(initial_status)"
  fi
  {
    printf 'time=%s\n' "$(date --iso-8601=seconds)"
    printf 'job=%s\n' "${job_id}"
    printf 'status=%s\n' "${status}"
    printf 'pid=\n'
    printf 'gpu=\n'
    printf 'detail=generated_by_launcher\n'
  } > "${job_dir}/state.txt"
}

for e in E0 E1 E2; do
  expert_selected "${e}" || continue
  job_id="expert_${e}"
  job_dir="${EXPERIMENT_ROOT}/experts/${e}_${E_NAME[${e}]}"
  register_job "${job_id}" "${job_dir}" ""
  cat > "${job_dir}/command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
STAGE_DIR="$(realpath "${job_dir}")" \\
OUTPUT_DIR="$(realpath "${job_dir}")/run" \\
DR_PRESET="${E_PRESET[${e}]}" DR_SCALE="${E_SCALE[${e}]}" \\
SEED=0 NUM_TRAIN_ENVS="${EXPERT_NUM_ENVS}" NUM_ENV_STEPS="${EXPERT_NUM_ENV_STEPS}" DEVICE=cuda:0 \\
bash "${SCRIPT_DIR}/01_train_normal_expert.sh"
EOF
done

for e in E0 E1 E2; do
  for d in D0 D1 D2; do
    pair_selected "${e}" "${d}" || continue
    expert_dir="${EXPERIMENT_ROOT}/experts/${e}_${E_NAME[${e}]}"
    job_id="dataset_${e}_${d}"
    job_dir="${EXPERIMENT_ROOT}/datasets/${e}/${d}"
    register_job "${job_id}" "${job_dir}" "expert_${e}"
    cat > "${job_dir}/command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
EXPERT_POLICY_PATH="\$(cat "$(realpath "${expert_dir}")/artifact_path.txt")" \\
STAGE_DIR="$(realpath "${job_dir}")" OUTPUT_DIR="$(realpath "${job_dir}")" \\
DR_PRESET="${D_PRESET[${d}]}" DR_SCALE="${D_SCALE[${d}]}" \\
SEED=100 NUM_ENVS="${DATASET_NUM_ENVS}" SHARD_TRANSITIONS_SPEC="${DATASET_SHARD_TRANSITIONS}" \\
MINIMUM_TRANSITIONS="${DATASET_MINIMUM_TRANSITIONS}" DEVICE=cuda:0 \\
bash "${SCRIPT_DIR}/02_collect_normal_dataset_sharded.sh"
EOF

    rwm_job_id="rwm_${e}_${d}"
    rwm_dir="${EXPERIMENT_ROOT}/rwms/${e}/${d}"
    register_job "${rwm_job_id}" "${rwm_dir}" "${job_id}"
    cat > "${rwm_dir}/command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
DATASET_PATH="\$(cat "$(realpath "${job_dir}")/artifact_path.txt")" \\
STAGE_DIR="$(realpath "${rwm_dir}")" OUTPUT_DIR="$(realpath "${rwm_dir}")/runs" \\
SEED=200 MAX_ITERATIONS="${RWM_MAX_ITERATIONS}" BATCH_SIZE="${RWM_BATCH_SIZE}" \\
MICRO_BATCH_SIZE="${RWM_MICRO_BATCH_SIZE}" NUM_EVAL_SEQUENCES="${RWM_NUM_EVAL_SEQUENCES}" DEVICE=cuda:0 \\
bash "${SCRIPT_DIR}/03_train_normal_rwm.sh"
EOF

    for p in P0 P1 P2 P3; do
      policy_selected "${e}" "${d}" "${p}" || continue
      policy_job_id="policy_${e}_${d}_${p}"
      policy_dir="${EXPERIMENT_ROOT}/policies/${e}/${d}/${p}"
      register_job "${policy_job_id}" "${policy_dir}" "${rwm_job_id}"
      cat > "${policy_dir}/command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
DATASET_PATH="\$(cat "$(realpath "${job_dir}")/artifact_path.txt")" \\
MODEL_PATH="\$(cat "$(realpath "${rwm_dir}")/artifact_path.txt")" \\
STAGE_DIR="$(realpath "${policy_dir}")" OUTPUT_DIR="$(realpath "${policy_dir}")/run" \\
P_ID="${p}" SEED=300 NUM_IMAGINATION_ENVS="${POLICY_NUM_IMAGINATION_ENVS}" \\
NUM_ENV_STEPS="${POLICY_NUM_ENV_STEPS}" DEVICE=cuda:0 \\
bash "${SCRIPT_DIR}/04_train_normal_policy.sh"
EOF
      if [[ "${INCLUDE_EVALUATIONS}" == "true" ]]; then
        eval_job_id="eval_${e}_${d}_${p}"
        eval_dir="${EXPERIMENT_ROOT}/evaluations/${e}/${d}/${p}"
        register_job "${eval_job_id}" "${eval_dir}" "${policy_job_id}"
        cat > "${eval_dir}/command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
CHECKPOINT_PATH="\$(cat "$(realpath "${policy_dir}")/artifact_path.txt")" \\
STAGE_DIR="$(realpath "${eval_dir}")" SEED=400 NUM_ENVS="${EVAL_NUM_ENVS}" \\
STEPS="${EVAL_STEPS}" COMMAND_SWITCH_STEPS="${EVAL_COMMAND_SWITCH_STEPS}" DEVICE=cuda:0 \\
bash "${SCRIPT_DIR}/05_evaluate_normal_policy.sh"
EOF
      fi
    done
  done
done

chmod +x "${SCRIPT_DIR}"/*.sh
find "${EXPERIMENT_ROOT}" -name command.sh -exec chmod +x {} +

JOB_COUNT="$(grep -c '^[^#]' "${JOBS_FILE}")"
if [[ "${JOB_COUNT}" -eq 0 ]]; then
  echo "MATRIX_SELECTION selected no valid policy IDs" >&2
  exit 1
fi

export RUN_ID EXPERIMENT_ROOT MATRIX_SELECTION DRY_RUN JOB_COUNT GPU_POOL INCLUDE_EVALUATIONS
export EXPERT_NUM_ENVS EXPERT_NUM_ENV_STEPS DATASET_NUM_ENVS DATASET_SHARD_TRANSITIONS
export RWM_MAX_ITERATIONS POLICY_NUM_IMAGINATION_ENVS POLICY_NUM_ENV_STEPS
"${REPO_ROOT}/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["EXPERIMENT_ROOT"])
manifest = {
    "run_id": os.environ["RUN_ID"],
    "status": "dry_run" if os.environ["DRY_RUN"] == "true" else "queued",
    "matrix_selection": os.environ["MATRIX_SELECTION"],
    "job_count": int(os.environ["JOB_COUNT"]),
    "include_evaluations": os.environ["INCLUDE_EVALUATIONS"] == "true",
    "gpu_pool": os.environ["GPU_POOL"].split(),
    "normal_go2": True,
    "task": "Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert",
    "common_baseline": {
        "default_joint_pos_per_leg": [0.0, 0.8, -1.5],
        "stiffness_per_leg": [30.0, 40.0, 50.0],
        "damping_per_leg": [1.5, 2.0, 2.5],
        "friction_center": 0.8,
        "friction_full_range": [0.6, 1.0],
        "base_height_target": 0.32,
        "command_ranges": {
            "expert_and_collection_lin_vel_x": [-0.8, 0.8],
            "expert_and_collection_lin_vel_y": [-0.3, 0.3],
            "expert_and_collection_ang_vel_z": [-0.6, 0.6],
            "imagination_lin_vel_x": [-0.5, 0.5],
            "imagination_lin_vel_y": [-0.25, 0.25],
            "imagination_ang_vel_z": [-0.5, 0.5],
        },
    },
    "joint_strength_scales": {},
    "broken_joint_names": [],
    "mask_indices": [],
    "factors": {
        "E": {
            "E0": ["calibrated_default", 1.0],
            "E1": ["calibrated_friend_flat", 0.5],
            "E2": ["calibrated_friend_flat", 1.0],
        },
        "D": {
            "D0": ["calibrated_default", 1.0],
            "D1": ["calibrated_friend_flat", 0.5],
            "D2": ["calibrated_friend_flat", 1.0],
        },
        "P": {
            "P0": [0.0, "none", 0.0],
            "P1": [0.01, "none", 0.0],
            "P2": [0.0, "deployment_small", 0.25],
            "P3": [0.01, "deployment_small", 0.25],
        },
    },
    "budgets": {
        "expert_num_envs": int(os.environ["EXPERT_NUM_ENVS"]),
        "expert_num_env_steps": int(os.environ["EXPERT_NUM_ENV_STEPS"]),
        "dataset_num_envs": int(os.environ["DATASET_NUM_ENVS"]),
        "dataset_shard_transitions": [int(x) for x in os.environ["DATASET_SHARD_TRANSITIONS"].split()],
        "rwm_max_iterations": int(os.environ["RWM_MAX_ITERATIONS"]),
        "policy_num_imagination_envs": int(os.environ["POLICY_NUM_IMAGINATION_ENVS"]),
        "policy_num_env_steps": int(os.environ["POLICY_NUM_ENV_STEPS"]),
    },
}
(root / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
PY

echo "experiment_root=$(realpath "${EXPERIMENT_ROOT}")"
echo "jobs=${JOB_COUNT}"
echo "jobs_file=$(realpath "${JOBS_FILE}")"
if [[ "${DRY_RUN}" == "true" ]]; then
  echo "dry_run=true; no tmux session or training process was started"
  exit 0
fi

SESSION_NAME="${SESSION_NAME:-go2_normal_${RUN_ID}}"
SESSION_NAME="${SESSION_NAME//./_}"
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  exit 1
fi
SCHEDULER_LAUNCH="${EXPERIMENT_ROOT}/queues/launch_scheduler.sh"
cat > "${SCHEDULER_LAUNCH}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export GPU_POOL="${GPU_POOL}"
export GPU_MAX_MEMORY_USED_MIB="${GPU_MAX_MEMORY_USED_MIB}"
export GPU_MAX_UTILIZATION="${GPU_MAX_UTILIZATION}"
export POLL_SECONDS="${POLL_SECONDS}"
bash "${SCRIPT_DIR}/normal_go2_dr_scheduler.sh" "$(realpath "${EXPERIMENT_ROOT}")"
EOF
chmod +x "${SCHEDULER_LAUNCH}"
tmux new-session -d -s "${SESSION_NAME}" "bash '$(realpath "${SCHEDULER_LAUNCH}")'"
printf '%s\n' "${SESSION_NAME}" > "${EXPERIMENT_ROOT}/queues/tmux_session.txt"
echo "tmux_session=${SESSION_NAME}"
tmux has-session -t "${SESSION_NAME}"
