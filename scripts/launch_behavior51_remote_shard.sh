#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: launch_behavior51_remote_shard.sh SHARD_INDEX

Launch one Behavior-51 coverage shard in a detached tmux session.
Use shard index 0 on one four-GPU machine and index 1 on the other.

Optional environment overrides:
  OSS_PREFIX                 OSS task root
  OSS_REGION                 OSS region (default: cn-shanghai)
  OSSUTIL_BIN                ossutil executable
  EPISPACE_REPO_ROOT         EpiSpace checkout
  EPISPACE_OG_ROOT           custom omnigibson_episode checkout
  EPISPACE_DATA_ROOT         BEHAVIOR dataset root
  EPISPACE_CONDA_ENV         conda environment (default: behavior)
  EPISPACE_GPU_IDS           space-separated GPU ids (default: 0 1 2 3)
  EPISPACE_WORKERS           renderer worker count (default: 4)
  EPISPACE_REMOTE_BACKFILL   enable geometry backfill (default: 1)
  EPISPACE_LOCAL_ROOT        persistent worker root
  EPISPACE_TMUX_SESSION      tmux session name
EOF
  exit 2
}

script_path="$(realpath "${BASH_SOURCE[0]}")"
script_root="$(cd "$(dirname "${script_path}")" && pwd)"
default_repo_root="$(cd "${script_root}/.." && pwd)"

if [[ "${1:-}" == "--worker" ]]; then
  [[ $# -eq 2 ]] || usage
  shard_index="$2"
  repo_root="${EPISPACE_REPO_ROOT:-${default_repo_root}}"
  local_root="${EPISPACE_LOCAL_ROOT:-/home/wmq/epispace-distributed}"
  log_root="${local_root}/logs"
  log_path="${log_root}/shard-${shard_index}.log"
  oss_prefix="${OSS_PREFIX:-oss://brain-imagegen-sh/wmq/epispace/behavior51_coverage_v1_git_runner_v1}"

  mkdir -p "${log_root}"
  cd "${repo_root}"
  echo "worker shard=${shard_index} started=$(date --iso-8601=seconds)"
  echo "oss_prefix=${oss_prefix}"
  echo "log=${log_path}"
  bash scripts/run_behavior51_shard_from_oss.sh \
    "${oss_prefix}" "${shard_index}" "${local_root}" \
    2>&1 | tee -a "${log_path}"
  exit "${PIPESTATUS[0]}"
fi

[[ $# -eq 1 ]] || usage
shard_index="$1"
if [[ ! "${shard_index}" =~ ^[0-9]+$ ]]; then
  echo "SHARD_INDEX must be a non-negative integer: ${shard_index}" >&2
  exit 2
fi

export OSS_PREFIX="${OSS_PREFIX:-oss://brain-imagegen-sh/wmq/epispace/behavior51_coverage_v1_git_runner_v1}"
export OSS_REGION="${OSS_REGION:-cn-shanghai}"
export OSSUTIL_BIN="${OSSUTIL_BIN:-/home/wmq/.local/bin/ossutil}"
export EPISPACE_REPO_ROOT="${EPISPACE_REPO_ROOT:-${default_repo_root}}"
export EPISPACE_OG_ROOT="${EPISPACE_OG_ROOT:-/home/wmq/project/bench/OminiGibson}"
export EPISPACE_DATA_ROOT="${EPISPACE_DATA_ROOT:-/home/wmq/project/bench/BEHAVIOR-1K/datasets}"
export EPISPACE_CONDA_ENV="${EPISPACE_CONDA_ENV:-behavior}"
export EPISPACE_GPU_IDS="${EPISPACE_GPU_IDS:-0 1 2 3}"
export EPISPACE_WORKERS="${EPISPACE_WORKERS:-4}"
export EPISPACE_REMOTE_BACKFILL="${EPISPACE_REMOTE_BACKFILL:-1}"
export EPISPACE_LOCAL_ROOT="${EPISPACE_LOCAL_ROOT:-/home/wmq/epispace-distributed}"
session_name="${EPISPACE_TMUX_SESSION:-epispace-s${shard_index}}"
log_path="${EPISPACE_LOCAL_ROOT}/logs/shard-${shard_index}.log"

command -v tmux >/dev/null 2>&1 || {
  echo "tmux is unavailable" >&2
  exit 1
}
if [[ ! -x "${OSSUTIL_BIN}" ]] && ! command -v "${OSSUTIL_BIN}" >/dev/null 2>&1; then
  echo "ossutil is unavailable: ${OSSUTIL_BIN}" >&2
  exit 1
fi
if tmux has-session -t "=${session_name}" 2>/dev/null; then
  echo "tmux session already exists: ${session_name}" >&2
  echo "attach with: tmux attach -t ${session_name}" >&2
  exit 1
fi

mkdir -p "$(dirname "${log_path}")"
printf -v worker_command '%q --worker %q' "${script_path}" "${shard_index}"
tmux_environment=()
for variable in \
  PATH \
  OSS_PREFIX OSS_REGION OSSUTIL_BIN \
  OSS_ACCESS_KEY_ID OSS_ACCESS_KEY_SECRET OSS_STS_TOKEN \
  EPISPACE_REPO_ROOT EPISPACE_OG_ROOT EPISPACE_DATA_ROOT \
  EPISPACE_CONDA_ENV EPISPACE_CONDA_BIN EPISPACE_GPU_IDS \
  EPISPACE_WORKERS EPISPACE_REMOTE_BACKFILL EPISPACE_LOCAL_ROOT \
  EPISPACE_TIMEOUT_MINUTES; do
  if [[ -v "${variable}" ]]; then
    tmux_environment+=(-e "${variable}=${!variable}")
  fi
done
tmux new-session -d -s "${session_name}" \
  "${tmux_environment[@]}" "${worker_command}"

echo "started tmux session: ${session_name}"
echo "attach: tmux attach -t ${session_name}"
echo "log: tail -f ${log_path}"
