#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 oss://bucket/prefix [DOWNLOAD_ROOT]" >&2
  exit 2
fi

oss_prefix="${1%/}"
repo_root="${EPISPACE_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
direction_root="${EPISPACE_DIRECTION_OUTPUT:-${repo_root}/outputs/behavior51_direction_balance_v1}"
download_root="$(realpath -m "${2:-${repo_root}/outputs/behavior51_direction_balance_remote_results_v1}")"
conda_env="${EPISPACE_CONDA_ENV:-behavior}"
conda_bin="${EPISPACE_CONDA_BIN:-$(command -v conda)}"
ossutil_bin="${OSSUTIL_BIN:-ossutil}"
index_path="${download_root}/direction_balance.distribution.json"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${repo_root}"

if tmux has-session -t epispace-direction-balance 2>/dev/null; then
  echo "local direction renderer is active; stop it before merging remote state" >&2
  exit 1
fi
mkdir -p "${download_root}"
"${ossutil_bin}" cp "${oss_prefix}/direction_balance.distribution.json" \
  "${index_path}" --update --force
python "${repo_root}/scripts/direction_balance_shards.py" verify-local \
  --index "${index_path}" --direction-root "${direction_root}"

while IFS= read -r task; do
  [[ -n "${task}" ]] || continue
  shard_name="$(
    python "${repo_root}/scripts/direction_balance_shards.py" task-field \
      --index "${index_path}" --task "${task}" --field shard_name
  )"
  result_root="$(
    bash "${repo_root}/scripts/sync_coverage_shard_results_from_oss.sh" \
      "${oss_prefix}/${task}/results" "${shard_name}" \
      "${download_root}/${task}"
  )"
  python "${repo_root}/scripts/coverage_render_shards.py" merge \
    --manifest "${direction_root}/${task}/coverage.plan.json" \
    --status "${direction_root}/${task}/coverage.status.json" \
    --results "${result_root}"
  "${conda_bin}" run --no-capture-output -n "${conda_env}" \
    python -m spatial_episode.scriptgen.dataset_cli coverage-package \
    --manifest "${direction_root}/${task}/coverage.plan.json"
done < <(
  python "${repo_root}/scripts/direction_balance_shards.py" list-tasks \
    --index "${index_path}"
)

python "${repo_root}/scripts/direction_balance_shards.py" merge-summary \
  --index "${index_path}" \
  --direction-root "${direction_root}" \
  --output "${direction_root}/direction_balance.merge.json"
echo "direction-balance remote results merged: ${direction_root}"
