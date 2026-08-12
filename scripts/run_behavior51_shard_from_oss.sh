#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 oss://bucket/epispace/behavior51_coverage_v1 SHARD_INDEX LOCAL_ROOT" >&2
  echo "example: $0 oss://my-bucket/epispace/behavior51_coverage_v1 0 /data/epispace-worker" >&2
  exit 2
fi

oss_base_prefix="${1%/}"
shard_index="$2"
local_root="$(realpath -m "$3")"
input_prefix="${oss_base_prefix}/input"
result_prefix="${oss_base_prefix}/results"
ossutil_bin="${OSSUTIL_BIN:-ossutil}"
script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
epispace_repo_root="${EPISPACE_REPO_ROOT:-$(cd "${script_root}/.." && pwd)}"

if [[ ! "${shard_index}" =~ ^[0-9]+$ ]]; then
  echo "SHARD_INDEX must be a non-negative integer" >&2
  exit 2
fi
if ! command -v "${ossutil_bin}" >/dev/null 2>&1; then
  echo "ossutil is unavailable: ${ossutil_bin}" >&2
  exit 1
fi
if ! command -v zstd >/dev/null 2>&1; then
  echo "zstd is required to extract the shard archive" >&2
  exit 1
fi

archive_root="${local_root}/archives"
package_parent="${local_root}/packages"
mkdir -p "${archive_root}" "${package_parent}"

distribution_path="${archive_root}/distribution.json"
"${ossutil_bin}" cp "${input_prefix}/distribution.json" "${distribution_path}" \
  --update --force

archive_name="$(python - "${distribution_path}" "${shard_index}" <<'PY'
import json
import sys

path, raw_index = sys.argv[1:]
payload = json.load(open(path, encoding="utf-8"))
index = int(raw_index)
archives = payload.get("archives", [])
if index >= len(archives):
    raise SystemExit(f"shard index {index} is out of range; archive count={len(archives)}")
row = archives[index]
print(row["name"] if isinstance(row, dict) else row.rsplit("/", 1)[-1])
PY
)"
package_name="${archive_name%.tar.zst}"
archive_path="${archive_root}/${archive_name}"
checksum_path="${archive_path}.sha256"

echo "worker shard_index=${shard_index} archive=${archive_name}"
"${ossutil_bin}" cp "${input_prefix}/${archive_name}" "${archive_path}" \
  --update --force
"${ossutil_bin}" cp "${input_prefix}/${archive_name}.sha256" "${checksum_path}" \
  --update --force
(cd "${archive_root}" && sha256sum --check "$(basename "${checksum_path}")")

tar --zstd -xf "${archive_path}" -C "${package_parent}"
package_root="${package_parent}/${package_name}"
work_root="${local_root}/work/${package_name}"
if [[ ! -f "${package_root}/shard.json" ]]; then
  echo "invalid extracted shard package: ${package_root}" >&2
  exit 1
fi

# Defaults target the current EpiSpace / BEHAVIOR installation. Override them
# in the environment when a worker uses different absolute paths.
export EPISPACE_REPO_ROOT="${epispace_repo_root}"
export EPISPACE_OG_ROOT="${EPISPACE_OG_ROOT:-/home/wmq/project/bench/OminiGibson}"
export EPISPACE_DATA_ROOT="${EPISPACE_DATA_ROOT:-/home/wmq/project/bench/BEHAVIOR-1K/datasets}"
export EPISPACE_CONDA_ENV="${EPISPACE_CONDA_ENV:-behavior}"
export EPISPACE_GPU_IDS="${EPISPACE_GPU_IDS:-0 1 2 3}"
export EPISPACE_WORKERS="${EPISPACE_WORKERS:-4}"
export EPISPACE_REMOTE_BACKFILL="${EPISPACE_REMOTE_BACKFILL:-1}"

bash "${epispace_repo_root}/scripts/run_coverage_render_shard.sh" \
  "${package_root}" "${work_root}"
bash "${epispace_repo_root}/scripts/upload_coverage_shard_results_to_oss.sh" \
  "${work_root}" "${result_prefix}"

echo "worker complete shard_index=${shard_index} result_prefix=${result_prefix}/${package_name}"
