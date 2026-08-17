#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 oss://bucket/prefix [LOCAL_ROOT]" >&2
  exit 2
}

script_path="$(realpath "${BASH_SOURCE[0]}")"
repo_root="${EPISPACE_REPO_ROOT:-$(cd "$(dirname "${script_path}")/.." && pwd)}"

if [[ "${1:-}" == "--worker" ]]; then
  [[ $# -eq 3 ]] || usage
  oss_prefix="${2%/}"
  local_root="$(realpath -m "$3")"
  ossutil_bin="${OSSUTIL_BIN:-ossutil}"
  index_path="${local_root}/direction_balance.distribution.json"
  log_path="${local_root}/logs/direction-balance-remote.log"
  mkdir -p "${local_root}/logs"
  exec > >(tee -a "${log_path}") 2>&1
  "${ossutil_bin}" cp "${oss_prefix}/direction_balance.distribution.json" \
    "${index_path}" --update --force
  while IFS= read -r task; do
    [[ -n "${task}" ]] || continue
    echo "direction remote task=${task} started=$(date --iso-8601=seconds)"
    bash "${repo_root}/scripts/run_behavior51_shard_from_oss.sh" \
      "${oss_prefix}/${task}" 0 "${local_root}/${task}"
  done < <(
    python "${repo_root}/scripts/direction_balance_shards.py" list-tasks \
      --index "${index_path}"
  )
  echo "direction remote complete=$(date --iso-8601=seconds)"
  exit 0
fi

[[ $# -ge 1 && $# -le 2 ]] || usage
oss_prefix="${1%/}"
local_root="$(realpath -m "${2:-${EPISPACE_LOCAL_ROOT:-/home/wmq/epispace-direction-balance}}")"
session_name="${EPISPACE_TMUX_SESSION:-epispace-direction-balance-remote}"
log_path="${local_root}/logs/direction-balance-remote.log"

export OSS_REGION="${OSS_REGION:-cn-shanghai}"
export OSSUTIL_BIN="${OSSUTIL_BIN:-/home/wmq/.local/bin/ossutil}"
export EPISPACE_REPO_ROOT="${repo_root}"
export EPISPACE_OG_ROOT="${EPISPACE_OG_ROOT:-/home/wmq/project/bench/OminiGibson}"
export EPISPACE_DATA_ROOT="${EPISPACE_DATA_ROOT:-/home/wmq/project/bench/BEHAVIOR-1K/datasets}"
export EPISPACE_CONDA_ENV="${EPISPACE_CONDA_ENV:-behavior}"
export EPISPACE_GPU_IDS="${EPISPACE_GPU_IDS:-0 1 2 3}"
export EPISPACE_WORKERS="${EPISPACE_WORKERS:-4}"
export EPISPACE_REMOTE_BACKFILL=1

if tmux has-session -t "=${session_name}" 2>/dev/null; then
  echo "tmux session already exists: ${session_name}" >&2
  exit 1
fi
mkdir -p "$(dirname "${log_path}")"
printf -v worker_command '%q --worker %q %q' \
  "${script_path}" "${oss_prefix}" "${local_root}"
tmux_environment=()
for variable in \
  PATH OSS_REGION OSSUTIL_BIN \
  OSS_ACCESS_KEY_ID OSS_ACCESS_KEY_SECRET OSS_STS_TOKEN \
  EPISPACE_REPO_ROOT EPISPACE_OG_ROOT EPISPACE_DATA_ROOT \
  EPISPACE_CONDA_ENV EPISPACE_CONDA_BIN EPISPACE_GPU_IDS \
  EPISPACE_WORKERS EPISPACE_REMOTE_BACKFILL EPISPACE_TIMEOUT_MINUTES; do
  if [[ -v "${variable}" ]]; then
    tmux_environment+=(-e "${variable}=${!variable}")
  fi
done
tmux new-session -d -s "${session_name}" "${tmux_environment[@]}" \
  "${worker_command}"

echo "started tmux session: ${session_name}"
echo "attach: tmux attach -t ${session_name}"
echo "log: tail -f ${log_path}"
