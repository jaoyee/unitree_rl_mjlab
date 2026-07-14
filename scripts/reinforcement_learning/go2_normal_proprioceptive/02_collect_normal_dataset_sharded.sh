#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

: "${STAGE_DIR:?Set STAGE_DIR}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
: "${EXPERT_POLICY_PATH:?Set EXPERT_POLICY_PATH}"
: "${DR_PRESET:?Set DR_PRESET}"
: "${DR_SCALE:?Set DR_SCALE}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
SEED="${SEED:-100}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-1024}"
DR_COMPONENTS="${DR_COMPONENTS:-friction,mass_com,motor,delay,observation,push,initial_state}"
COLLECTOR_MIX="${COLLECTOR_MIX:-expert:0.45,noisy_expert:0.25,medium:0.10,failure_border:0.15,random:0.05}"
MEDIUM_POLICY_PATH="${MEDIUM_POLICY_PATH:-${EXPERT_POLICY_PATH}}"
ACTION_NOISE_STD="${ACTION_NOISE_STD:-0.12}"
MEDIUM_ACTION_NOISE_STD="${MEDIUM_ACTION_NOISE_STD:-0.25}"
FAILURE_ACTION_NOISE_STD="${FAILURE_ACTION_NOISE_STD:-0.45}"
SHARD_TRANSITIONS_SPEC="${SHARD_TRANSITIONS_SPEC:-249856 249856 249856 250880}"
MINIMUM_TRANSITIONS="${MINIMUM_TRANSITIONS:-1000000}"
X_ABS_RANGE="${X_ABS_RANGE:-0.08 0.80}"
Y_ABS_RANGE="${Y_ABS_RANGE:-0.05 0.30}"
YAW_ABS_RANGE="${YAW_ABS_RANGE:-0.08 0.60}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${EXPERT_POLICY_PATH}/actor.pt" ]]; then
  echo "Expert actor not found: ${EXPERT_POLICY_PATH}/actor.pt" >&2
  exit 1
fi
OUTPUT_RELATIVE="${OUTPUT_DIR#${REPO_ROOT}/}"
if [[ "${OUTPUT_RELATIVE}" == *0p5* || "${OUTPUT_RELATIVE}" == *broken* ]]; then
  echo "Normal dataset output path contains a forbidden 0.5/broken marker: ${OUTPUT_DIR}" >&2
  exit 1
fi
mkdir -p "${STAGE_DIR}" "${OUTPUT_DIR}/shards"

# The legacy vectorized format stores complete [time, env] rows, so exactly
# 1,000,000 transitions is not divisible by 1024. These four independent
# shards total 1,000,448 transitions, only 0.045% above the target.
read -r -a SHARD_TRANSITIONS <<< "${SHARD_TRANSITIONS_SPEC}"
if [[ "${#SHARD_TRANSITIONS[@]}" -ne 4 ]]; then
  echo "SHARD_TRANSITIONS_SPEC must contain exactly four counts" >&2
  exit 1
fi
SHARD_PATHS=()
for shard_id in 0 1 2 3; do
  shard_dir="${OUTPUT_DIR}/shards/shard_${shard_id}"
  shard_path="${shard_dir}/dataset.pt"
  shard_seed=$((SEED + shard_id))
  if [[ -e "${shard_path}" ]]; then
    "${PYTHON_BIN}" - "${shard_path}" "${shard_id}" "${shard_seed}" \
      "${SHARD_TRANSITIONS[shard_id]}" "${NUM_ENVS}" "${DR_PRESET}" "${DR_SCALE}" <<'PY'
import sys
import torch

path, shard_id, seed, transitions, num_envs, preset, scale = sys.argv[1:]
data = torch.load(path, map_location="cpu", weights_only=False)
metadata = data.get("metadata") or {}
assert int(metadata["shard_id"]) == int(shard_id)
assert int(metadata["seed"]) == int(seed)
assert int(data["num_envs"]) == int(num_envs)
assert len(data["states"]) * int(data["num_envs"]) == int(transitions)
randomization = metadata["env_randomization"]
assert str(randomization["randomization_preset"]) == preset
assert abs(float(randomization["randomization_scale"]) - float(scale)) < 1e-9
PY
    echo "[reuse] validated shard ${shard_path}"
    SHARD_PATHS+=("${shard_path}")
    continue
  fi
  mkdir -p "${shard_dir}"
  "${PYTHON_BIN}" scripts/reinforcement_learning/rwm_dataset/collect_go2_expert_command_coverage_dataset.py \
    --task Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert \
    --device "${DEVICE}" \
    --seed "${shard_seed}" \
    --shard_id "${shard_id}" \
    --num_envs "${NUM_ENVS}" \
    --num_transitions "${SHARD_TRANSITIONS[shard_id]}" \
    --save_path "${shard_path}" \
    --expert_policy_path "${EXPERT_POLICY_PATH}" \
    --medium_policy_path "${MEDIUM_POLICY_PATH}" \
    --collector_mix "${COLLECTOR_MIX}" \
    --dataset_obs_kind proprioceptive \
    --action_noise_std "${ACTION_NOISE_STD}" \
    --medium_action_noise_std "${MEDIUM_ACTION_NOISE_STD}" \
    --failure_action_noise_std "${FAILURE_ACTION_NOISE_STD}" \
    --randomization_preset "${DR_PRESET}" \
    --randomization_components "${DR_COMPONENTS}" \
    --randomization_scale "${DR_SCALE}" \
    --env_action_noise_std 0.0 \
    --env_action_bias_std 0.0 \
    --env_action_scale_range 1.0 1.0 \
    --env_action_delay_steps 0 0 \
    --chunk_size 0 \
    --command_modes stand,pure_x,pure_y,pure_yaw,xy,x_yaw,y_yaw,xy_yaw \
    --command_mode_weights stand:0.08,pure_x:0.25,pure_y:0.10,pure_yaw:0.08,xy:0.14,x_yaw:0.17,y_yaw:0.05,xy_yaw:0.13 \
    --x_range -0.8 0.8 \
    --signed_x \
    --x_abs_range ${X_ABS_RANGE} \
    --y_abs_range ${Y_ABS_RANGE} \
    --yaw_abs_range ${YAW_ABS_RANGE} \
    --command_resample_interval_min 120 \
    --command_resample_interval_max 300 \
    --use_domain_randomization \
    --use_push_randomization \
    --use_observation_noise \
    --headless
  SHARD_PATHS+=("${shard_path}")
done

DATASET_PATH="${OUTPUT_DIR}/dataset.pt"
if [[ ! -e "${DATASET_PATH}" ]]; then
  "${PYTHON_BIN}" scripts/reinforcement_learning/rwm_dataset/merge_go2_dataset_shards.py \
    --inputs "${SHARD_PATHS[@]}" \
    --output "${DATASET_PATH}" \
    --expected_shards 4 \
    --minimum_transitions "${MINIMUM_TRANSITIONS}"
else
  echo "[reuse] validating existing merged dataset ${DATASET_PATH}"
fi

"${PYTHON_BIN}" scripts/reinforcement_learning/rwm_dataset/validate_go2_dataset.py \
  --dataset_path "${DATASET_PATH}" \
  --output_json "${STAGE_DIR}/dataset_quality.json" \
  --minimum_transitions "${MINIMUM_TRANSITIONS}" \
  --expected_num_envs "${NUM_ENVS}" \
  --expected_preset "${DR_PRESET}" \
  --expected_scale "${DR_SCALE}" \
  --expected_shards 4

DATASET_PATH="$(realpath "${DATASET_PATH}")"
DATASET_SHA256="$(sha256sum "${DATASET_PATH}" | awk '{print $1}')"
printf '%s\n' "${DATASET_PATH}" > "${STAGE_DIR}/artifact_path.txt"
printf '%s\n' "${DATASET_SHA256}" > "${STAGE_DIR}/artifact_sha256.txt"

"${PYTHON_BIN}" - "${STAGE_DIR}/summary.json" "${DATASET_PATH}" "${DATASET_SHA256}" \
  "${EXPERT_POLICY_PATH}" "${DR_PRESET}" "${DR_SCALE}" "${SEED}" <<'PY'
import json
import sys
from pathlib import Path

output, artifact, digest, expert, preset, scale, seed = sys.argv[1:]
quality = json.loads((Path(output).parent / "dataset_quality.json").read_text(encoding="utf-8"))
Path(output).write_text(json.dumps({
    "status": "completed",
    "artifact_path": artifact,
    "artifact_sha256": digest,
    "expert_policy_path": expert,
    "randomization_preset": preset,
    "randomization_scale": float(scale),
    "seed_base": int(seed),
    "num_transitions": quality["num_transitions"],
    "startup_domains": quality["startup_domains"],
    "episodes": quality["episodes"],
}, indent=2, sort_keys=True), encoding="utf-8")
PY

echo "dataset_artifact=${DATASET_PATH}"
echo "dataset_sha256=${DATASET_SHA256}"
