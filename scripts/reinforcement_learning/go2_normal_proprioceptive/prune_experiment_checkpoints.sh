#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data1/xjy/unitree_rl_mjlab_normal_fixstand_v2}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${REPO_ROOT}/logs/experiments/go2_normal_dr_3x3x4/normal_go2_fixstand_v2_moderate_seed0_20260712}"
KEEP_RUNNING="${KEEP_RUNNING:-2}"

safe_remove() {
  local target resolved
  target="$1"
  resolved="$(realpath -m "${target}")"
  case "${resolved}" in
    "${EXPERIMENT_ROOT}"/*) rm -rf -- "${resolved}" ;;
    *) echo "Refusing to remove unsafe path: ${resolved}" >&2; return 1 ;;
  esac
}

# Completed experts are consumed through artifact_path.txt, so only that final
# checkpoint is required by downstream collection jobs.
for expert_dir in "${EXPERIMENT_ROOT}"/experts/*; do
  [[ -s "${expert_dir}/artifact_path.txt" ]] || continue
  keep="$(realpath "$(cat "${expert_dir}/artifact_path.txt")")"
  for checkpoint in "${expert_dir}"/run/step*; do
    [[ -d "${checkpoint}" ]] || continue
    [[ "$(realpath "${checkpoint}")" == "${keep}" ]] || safe_remove "${checkpoint}"
  done
done

# Keep the newest checkpoints for active/future policies. Two checkpoints
# preserve a fallback if the newest directory is observed while being written.
for run_dir in "${EXPERIMENT_ROOT}"/policies/*/*/*/run; do
  [[ -d "${run_dir}" ]] || continue
  mapfile -t checkpoints < <(find "${run_dir}" -mindepth 1 -maxdepth 1 -type d -name 'step*' -printf '%f\n' | sort -V)
  remove_count=$(( ${#checkpoints[@]} - KEEP_RUNNING ))
  (( remove_count > 0 )) || continue
  for ((i = 0; i < remove_count; i++)); do
    safe_remove "${run_dir}/${checkpoints[i]}"
  done
done

# Completed merged datasets are self-contained and hash-verified. Their source
# shard provenance remains in dataset_quality.json, so duplicate shard payloads
# can be removed after successful validation.
for dataset_dir in "${EXPERIMENT_ROOT}"/datasets/*/*; do
  [[ -s "${dataset_dir}/summary.json" ]] || continue
  [[ -d "${dataset_dir}/shards" ]] && safe_remove "${dataset_dir}/shards"
done

df -h "${EXPERIMENT_ROOT}"
