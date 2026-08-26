#!/usr/bin/env bash
# Compatibility entrypoint; the old unsharded runner was intentionally retired.
set -euo pipefail

script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -ne 4 ]]; then
  echo "The unsafe all-in-one renderer has been retired." >&2
  echo "usage: $0 PHASE SHARD_INDEX OSS_TASK_ROOT WORK_ROOT" >&2
  exit 2
fi
exec bash "${script_root}/run_p23_shard_from_oss.sh" "$@"
