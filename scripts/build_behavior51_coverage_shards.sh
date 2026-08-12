#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${EPISPACE_COVERAGE_MANIFEST:-${repo_root}/outputs/behavior51_coverage_v1/coverage.plan.json}"
output_root="${EPISPACE_SHARD_OUTPUT:-${repo_root}/outputs/behavior51_coverage_shards_v1}"
shard_count="${EPISPACE_SHARD_COUNT:-2}"
backend_root="${EPISPACE_BACKEND_ROOT:-/home/wmq/project/bench/OminiGibson}"

args=(
  python "${repo_root}/scripts/coverage_render_shards.py" build
  --manifest "${manifest}"
  --output "${output_root}"
  --num-shards "${shard_count}"
  --backend-root "${backend_root}"
)

if [[ "${EPISPACE_SHARD_PREVIEW:-0}" == 1 ]]; then
  args+=(--preview)
fi
if [[ "${EPISPACE_SHARD_OVERWRITE:-0}" == 1 ]]; then
  args+=(--overwrite)
fi

cd "${repo_root}"
"${args[@]}"
