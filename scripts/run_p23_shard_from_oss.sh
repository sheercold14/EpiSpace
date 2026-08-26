#!/usr/bin/env bash
# Download, verify, render, validate and upload exactly one P2/P3 shard.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_p23_shard_from_oss.sh PHASE SHARD_INDEX OSS_TASK_ROOT WORK_ROOT

PHASE is p2 or p3. OSS_TASK_ROOT contains:
  p2/input, p2/results, p3/input, p3/results

Required environment overrides on each worker normally include:
  EPISPACE_OG_ROOT EPISPACE_DATA_ROOT EPISPACE_CONDA_ENV
Optional: EPISPACE_GPU_IDS, EPISPACE_WORKERS, EPISPACE_MIN_FREE_GIB,
          EPISPACE_BYTES_PER_FRAME, EPISPACE_SCRATCH_RESERVE_GIB,
          EPISPACE_SCRATCH_ROOT, OSSUTIL_BIN.
Remote geometry backfill is forbidden for this frozen 7k render.
EOF
  exit 2
}

[[ $# -eq 4 ]] || usage
phase=$1
shard_index=$2
oss_task_root="${3%/}"
local_root="$(realpath -m "$4")"
[[ "${phase}" == p2 || "${phase}" == p3 ]] || usage
[[ "${shard_index}" =~ ^[0-9]+$ ]] || usage
if [[ -n "${EPISPACE_REMOTE_BACKFILL:-}" && "${EPISPACE_REMOTE_BACKFILL}" != 0 ]]; then
  echo "P2/P3 frozen shards forbid EPISPACE_REMOTE_BACKFILL != 0" >&2
  exit 1
fi
export EPISPACE_REMOTE_BACKFILL=0

script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${EPISPACE_REPO_ROOT:-$(cd "${script_root}/.." && pwd)}"
backend_root="${EPISPACE_OG_ROOT:?set EPISPACE_OG_ROOT to the OminiGibson checkout}"
data_root="${EPISPACE_DATA_ROOT:?set EPISPACE_DATA_ROOT to the OmniGibson data root}"
conda_env="${EPISPACE_CONDA_ENV:-behavior-spatialep}"
conda_bin="${EPISPACE_CONDA_BIN:-conda}"
ossutil_bin="${OSSUTIL_BIN:-ossutil}"
read -r -a gpu_ids <<< "${EPISPACE_GPU_IDS:-0 1 2 3}"
export EPISPACE_WORKERS="${EPISPACE_WORKERS:-${#gpu_ids[@]}}"

for command in "${ossutil_bin}" "${conda_bin}" zstd tar sha256sum git nvidia-smi; do
  command -v "${command}" >/dev/null 2>&1 || {
    echo "required worker command is unavailable: ${command}" >&2
    exit 1
  }
done

input_prefix="${oss_task_root}/${phase}/input"
result_prefix="${oss_task_root}/${phase}/results"
archive_root="${local_root}/${phase}/archives"
package_parent="${local_root}/${phase}/packages"
mkdir -p "${archive_root}" "${package_parent}"
distribution="${archive_root}/distribution.json"
quality_audit="${archive_root}/p23_7k_trajectory_quality_audit.json"
quality_checksum="${quality_audit}.sha256"

"${ossutil_bin}" cp "${input_prefix}/distribution.json" "${distribution}" \
  --update --force
"${ossutil_bin}" cp \
  "${input_prefix}/p23_7k_trajectory_quality_audit.json" "${quality_audit}" \
  --update --force
"${ossutil_bin}" cp \
  "${input_prefix}/p23_7k_trajectory_quality_audit.json.sha256" "${quality_checksum}" \
  --update --force
(
  cd "${archive_root}"
  sha256sum --check "$(basename "${quality_checksum}")"
)

archive_name="$("${conda_bin}" run -n "${conda_env}" python - \
  "${distribution}" "${shard_index}" <<'PY'
import json
import sys

distribution_path, raw_index = sys.argv[1:]
payload = json.load(open(distribution_path, encoding="utf-8"))
index = int(raw_index)
rows = payload.get("archives", [])
if index >= len(rows):
    raise SystemExit(f"shard index {index} outside archive count {len(rows)}")
row = rows[index]
if not isinstance(row, dict) or not row.get("name", "").endswith(".tar.zst"):
    raise SystemExit("distribution archive row is malformed")
print(row["name"])
PY
)"
package_name="${archive_name%.tar.zst}"
archive="${archive_root}/${archive_name}"
archive_checksum="${archive}.sha256"
"${ossutil_bin}" cp "${input_prefix}/${archive_name}" "${archive}" --update --force
"${ossutil_bin}" cp "${input_prefix}/${archive_name}.sha256" \
  "${archive_checksum}" --update --force
(
  cd "${archive_root}"
  sha256sum --check "$(basename "${archive_checksum}")"
)

mapfile -t archive_entries < <(tar --zstd -tf "${archive}")
if (( ${#archive_entries[@]} == 0 )); then
  echo "shard archive is empty: ${archive}" >&2
  exit 1
fi
for entry in "${archive_entries[@]}"; do
  if [[ "${entry}" == /* || "${entry}" == .. || "${entry}" == ../* \
    || "${entry}" == */../* || "${entry}" != "${package_name}"/* ]]; then
    echo "unsafe or unexpected shard archive entry: ${entry}" >&2
    exit 1
  fi
done

package_root="${package_parent}/${package_name}"
if [[ ! -f "${package_root}/shard.json" ]]; then
  tar --zstd -xf "${archive}" -C "${package_parent}"
fi
[[ -f "${package_root}/shard.json" ]] || {
  echo "archive did not contain expected package: ${package_root}" >&2
  exit 1
}

work_root="${local_root}/${phase}/work/${package_name}"
preflight_output="${work_root}/output/p23.worker_preflight.json"
"${conda_bin}" run --no-capture-output -n "${conda_env}" \
  python "${repo_root}/scripts/p23_worker_preflight.py" \
  --phase "${phase}" \
  --shard-index "${shard_index}" \
  --package "${package_root}" \
  --distribution "${distribution}" \
  --quality-audit "${quality_audit}" \
  --archive "${archive}" \
  --repo-root "${repo_root}" \
  --backend-root "${backend_root}" \
  --data-root "${data_root}" \
  --work-root "${work_root}" \
  --gpu-ids "${gpu_ids[@]}" \
  --min-free-gib "${EPISPACE_MIN_FREE_GIB:-100}" \
  --bytes-per-frame "${EPISPACE_BYTES_PER_FRAME:-4194304}" \
  --scratch-reserve-gib "${EPISPACE_SCRATCH_RESERVE_GIB:-20}" \
  --output "${preflight_output}"

export EPISPACE_REPO_ROOT="${repo_root}"
export EPISPACE_OG_ROOT="${backend_root}"
export EPISPACE_DATA_ROOT="${data_root}"
export EPISPACE_CONDA_ENV="${conda_env}"
export EPISPACE_GPU_IDS="${gpu_ids[*]}"
bash "${repo_root}/scripts/run_coverage_render_shard.sh" \
  "${package_root}" "${work_root}"

"${conda_bin}" run --no-capture-output -n "${conda_env}" \
  python "${repo_root}/scripts/verify_p23_shard_output.py" \
  --phase "${phase}" --work-root "${work_root}"

bash "${repo_root}/scripts/upload_coverage_shard_results_to_oss.sh" \
  "${work_root}" "${result_prefix}"
echo "P2/P3 shard uploaded: phase=${phase} shard=${shard_index}"
