#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 V2_OVERLAY_ROOT" >&2
  exit 2
fi

overlay_root="$(realpath "$1")"
repo_root="${EPISPACE_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
og_root="${EPISPACE_OG_ROOT:-/home/wmq/project/bench/OminiGibson}"
data_root="${EPISPACE_DATA_ROOT:-/home/wmq/project/bench/BEHAVIOR-1K/datasets}"
conda_env="${EPISPACE_CONDA_ENV:-behavior}"
conda_bin="${EPISPACE_CONDA_BIN:-$(command -v conda)}"
timeout_minutes="${EPISPACE_TIMEOUT_MINUTES:-20}"
read -r -a gpu_ids <<< "${EPISPACE_GPU_IDS:-0}"
workers="${EPISPACE_WORKERS:-${#gpu_ids[@]}}"

for required in \
  "${overlay_root}/coverage.plan.json" \
  "${overlay_root}/coverage.status.json" \
  "${overlay_root}/producer_cell_ids.json" \
  "${overlay_root}/credit_cell_ids.json" \
  "${overlay_root}/overlay.provenance.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing local repair input: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${overlay_root}/logs"
export PYTHONPATH="${repo_root}/src:${og_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMNIGIBSON_DATA_PATH="${data_root}"
export OMNIGIBSON_HEADLESS=True
export OMNI_KIT_ACCEPT_EULA=YES
unset DISPLAY || true

echo "local v2 repair: gpu_ids=${gpu_ids[*]} workers=${workers} overlay=${overlay_root}"
"${conda_bin}" run --no-capture-output -n "${conda_env}" \
  python -m spatial_episode.scriptgen.dataset_cli coverage-run \
  --manifest "${overlay_root}/coverage.plan.json" \
  --og-root "${og_root}" \
  --conda-env "${conda_env}" \
  --data-root "${data_root}" \
  --gpu-ids "${gpu_ids[@]}" \
  --workers "${workers}" \
  --timeout-minutes "${timeout_minutes}" \
  --cell-ids-file "${overlay_root}/producer_cell_ids.json" \
  --credit-cell-ids-file "${overlay_root}/credit_cell_ids.json"

"${conda_bin}" run --no-capture-output -n "${conda_env}" \
  python -m spatial_episode.scriptgen.dataset_cli coverage-package \
  --manifest "${overlay_root}/coverage.plan.json"

echo "local v2 repair pass complete: ${overlay_root}"
