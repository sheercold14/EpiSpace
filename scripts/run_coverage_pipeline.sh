#!/usr/bin/env bash
set -euo pipefail

EPISPACE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPISPACE_PYTHON="${EPISPACE_PYTHON:-/home/wmq/.conda/envs/behavior/bin/python}"
EPISPACE_MANIFEST="${EPISPACE_MANIFEST:-${EPISPACE_PROJECT_ROOT}/outputs/behavior51_coverage_v1/coverage.plan.json}"
EPISPACE_OG_ROOT="${EPISPACE_OG_ROOT:-/home/wmq/project/bench/OminiGibson}"
EPISPACE_DATA_ROOT="${EPISPACE_DATA_ROOT:-/home/wmq/project/bench/BEHAVIOR-1K/datasets}"
EPISPACE_CONDA_ENV="${EPISPACE_CONDA_ENV:-behavior}"
EPISPACE_GPU_IDS="${EPISPACE_GPU_IDS:-0}"
EPISPACE_TIMEOUT_MINUTES="${EPISPACE_TIMEOUT_MINUTES:-20}"

if [[ ! -x "${EPISPACE_PYTHON}" ]]; then
  echo "Python executable is unavailable: ${EPISPACE_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${EPISPACE_MANIFEST}" ]]; then
  echo "Coverage manifest is unavailable: ${EPISPACE_MANIFEST}" >&2
  exit 1
fi

read -r -a EPISPACE_GPU_ARRAY <<<"${EPISPACE_GPU_IDS}"
EPISPACE_WORKERS="${EPISPACE_WORKERS:-${#EPISPACE_GPU_ARRAY[@]}}"
EPISPACE_IMPORT_PATH="${EPISPACE_PROJECT_ROOT}/src"
if [[ -n "${PYTHONPATH:-}" ]]; then
  EPISPACE_IMPORT_PATH="${EPISPACE_IMPORT_PATH}:${PYTHONPATH}"
fi

cd "${EPISPACE_PROJECT_ROOT}"
exec env PYTHONPATH="${EPISPACE_IMPORT_PATH}" "${EPISPACE_PYTHON}" \
  -m spatial_episode.scriptgen.dataset_cli coverage-pipeline \
  --manifest "${EPISPACE_MANIFEST}" \
  --og-root "${EPISPACE_OG_ROOT}" \
  --conda-env "${EPISPACE_CONDA_ENV}" \
  --data-root "${EPISPACE_DATA_ROOT}" \
  --gpu-ids "${EPISPACE_GPU_ARRAY[@]}" \
  --workers "${EPISPACE_WORKERS}" \
  --timeout-minutes "${EPISPACE_TIMEOUT_MINUTES}"
