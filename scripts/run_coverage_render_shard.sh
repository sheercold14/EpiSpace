#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 PACKAGE_ROOT WORK_ROOT" >&2
  exit 2
fi

package_root="$(realpath "$1")"
work_root="$(realpath -m "$2")"
script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
epispace_repo_root="${EPISPACE_REPO_ROOT:-$(cd "${script_root}/.." && pwd)}"
tool="${epispace_repo_root}/scripts/coverage_render_shards.py"

epispace_og_root="${EPISPACE_OG_ROOT:-/home/wmq/project/bench/OminiGibson}"
epispace_data_root="${EPISPACE_DATA_ROOT:-/home/wmq/project/bench/BEHAVIOR-1K/datasets}"
epispace_conda_env="${EPISPACE_CONDA_ENV:-behavior}"
epispace_conda_bin="${EPISPACE_CONDA_BIN:-conda}"
epispace_timeout_minutes="${EPISPACE_TIMEOUT_MINUTES:-20}"
epispace_remote_backfill="${EPISPACE_REMOTE_BACKFILL:-1}"
read -r -a epispace_gpu_ids <<< "${EPISPACE_GPU_IDS:-0 1 2 3}"
epispace_workers="${EPISPACE_WORKERS:-${#epispace_gpu_ids[@]}}"

if [[ ! -f "${tool}" ]]; then
  echo "missing EpiSpace shard tool: ${tool}" >&2
  exit 1
fi

expected_commit() {
  python - "$package_root/shard.json" "$1" <<'PY'
import json
import sys
metadata_path, repository = sys.argv[1:]
metadata = json.load(open(metadata_path, encoding="utf-8"))
print(metadata["code_revisions"][repository]["commit"])
PY
}

epispace_expected="$(expected_commit epispace)"
epispace_actual="$(git -C "${epispace_repo_root}" rev-parse HEAD)"
backend_expected="$(expected_commit omnigibson_episode)"
backend_actual="$(git -C "${epispace_og_root}" rev-parse HEAD)"
if [[ "${epispace_actual}" != "${epispace_expected}" ]]; then
  echo "EpiSpace commit mismatch: ${epispace_actual} != ${epispace_expected}" >&2
  exit 1
fi
if [[ "${backend_actual}" != "${backend_expected}" ]]; then
  echo "omnigibson_episode commit mismatch: ${backend_actual} != ${backend_expected}" >&2
  exit 1
fi
if [[ ! -f "${epispace_og_root}/src/omnigibson_episode/cli.py" ]]; then
  echo "custom omnigibson_episode backend is missing: ${epispace_og_root}" >&2
  exit 1
fi
if [[ "${epispace_remote_backfill}" != 0 && "${epispace_remote_backfill}" != 1 ]]; then
  echo "EPISPACE_REMOTE_BACKFILL must be 0 or 1" >&2
  exit 1
fi

mkdir -p "${work_root}"
export PYTHONPATH="${epispace_repo_root}/src:${epispace_og_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMNIGIBSON_DATA_PATH="${epispace_data_root}"
export OMNIGIBSON_HEADLESS=True
export OMNI_KIT_ACCEPT_EULA=YES
unset DISPLAY || true

"${epispace_conda_bin}" run --no-capture-output -n "${epispace_conda_env}" \
  python "${tool}" prepare --package "${package_root}" --work-root "${work_root}"

coverage_run_args=(
  python -m spatial_episode.scriptgen.dataset_cli coverage-run
  --manifest "${work_root}/output/coverage.plan.json"
  --og-root "${epispace_og_root}"
  --conda-env "${epispace_conda_env}"
  --data-root "${epispace_data_root}"
  --gpu-ids "${epispace_gpu_ids[@]}"
  --workers "${epispace_workers}"
  --timeout-minutes "${epispace_timeout_minutes}"
  --cell-ids-file "${work_root}/output/cell_ids.json"
)
if [[ -f "${package_root}/credit_cell_ids.json" ]]; then
  cp "${package_root}/credit_cell_ids.json" \
    "${work_root}/output/credit_cell_ids.json"
  coverage_run_args+=(
    --credit-cell-ids-file "${work_root}/output/credit_cell_ids.json"
  )
fi
if [[ "${epispace_remote_backfill}" == 0 ]]; then
  coverage_run_args+=(--no-backfill)
fi
echo "remote geometry backfill=${epispace_remote_backfill} (maximum attempts inherited from manifest)"
"${epispace_conda_bin}" run --no-capture-output -n "${epispace_conda_env}" \
  "${coverage_run_args[@]}"

"${epispace_conda_bin}" run --no-capture-output -n "${epispace_conda_env}" \
  python "${tool}" finalize --work-root "${work_root}"

echo "render shard complete: ${work_root}/output"
