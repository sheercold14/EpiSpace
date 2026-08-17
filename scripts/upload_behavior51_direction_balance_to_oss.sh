#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 oss://bucket/prefix" >&2
  exit 2
fi

oss_prefix="${1%/}"
repo_root="${EPISPACE_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
direction_root="${EPISPACE_DIRECTION_OUTPUT:-${repo_root}/outputs/behavior51_direction_balance_v1}"
shard_root="${EPISPACE_DIRECTION_SHARDS:-${repo_root}/outputs/behavior51_direction_balance_shards_v1}"
backend_root="${EPISPACE_OG_ROOT:-/home/wmq/project/bench/OminiGibson}"
conda_env="${EPISPACE_CONDA_ENV:-behavior}"
conda_bin="${EPISPACE_CONDA_BIN:-$(command -v conda)}"
ossutil_bin="${OSSUTIL_BIN:-ossutil}"
overwrite="${EPISPACE_SHARD_OVERWRITE:-0}"
tasks=(pure_rotation_left pure_rotation_right pure_translation_left pure_translation_right)
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${repo_root}"

if tmux has-session -t epispace-direction-balance 2>/dev/null; then
  echo "local direction renderer is still queued/running; stop it before sharding" >&2
  exit 1
fi
if ! command -v "${ossutil_bin}" >/dev/null 2>&1; then
  echo "ossutil is unavailable: ${ossutil_bin}" >&2
  exit 1
fi
if ! git -C "${repo_root}" diff --quiet HEAD -- || \
   ! git -C "${repo_root}" diff --cached --quiet; then
  echo "EpiSpace tracked code must be committed before building Git-bound shards" >&2
  exit 1
fi
required_tracked=(
  scripts/coverage_render_shards.py
  scripts/direction_balance_shards.py
  scripts/launch_behavior51_direction_balance_remote.sh
  scripts/merge_behavior51_direction_balance_from_oss.sh
  scripts/run_behavior51_shard_from_oss.sh
  scripts/run_coverage_render_shard.sh
  src/spatial_episode/scriptgen/binding_coverage.py
  src/spatial_episode/scriptgen/occlusion.py
)
for required in "${required_tracked[@]}"; do
  if ! git -C "${repo_root}" ls-files --error-unmatch "${required}" >/dev/null 2>&1; then
    echo "required runtime file is not committed: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${shard_root}"
for task in "${tasks[@]}"; do
  manifest="${direction_root}/${task}/coverage.plan.json"
  "${conda_bin}" run --no-capture-output -n "${conda_env}" \
    python -m spatial_episode.scriptgen.dataset_cli coverage-init-status \
    --manifest "${manifest}"

  task_shards="${shard_root}/${task}"
  if [[ ! -f "${task_shards}/distribution.json" || "${overwrite}" == 1 ]]; then
    args=(
      python "${repo_root}/scripts/coverage_render_shards.py" build
      --manifest "${manifest}"
      --output "${task_shards}"
      --num-shards 1
      --backend-root "${backend_root}"
      --hardlink
    )
    if [[ "${overwrite}" == 1 ]]; then
      args+=(--overwrite)
    fi
    "${args[@]}"
  else
    echo "reusing existing immutable shard task=${task}"
  fi
done

index_path="${shard_root}/direction_balance.distribution.json"
python "${repo_root}/scripts/direction_balance_shards.py" build-index \
  --direction-root "${direction_root}" \
  --shard-root "${shard_root}" \
  --oss-prefix "${oss_prefix}" \
  --output "${index_path}"

"${ossutil_bin}" cp "${index_path}" \
  "${oss_prefix}/direction_balance.distribution.json" --update --force
for task in "${tasks[@]}"; do
  bash "${repo_root}/scripts/upload_coverage_render_shards_to_oss.sh" \
    "${shard_root}/${task}" "${oss_prefix}/${task}/input"
done

echo "direction-balance inputs uploaded: ${oss_prefix}"
echo "distribution: ${oss_prefix}/direction_balance.distribution.json"
