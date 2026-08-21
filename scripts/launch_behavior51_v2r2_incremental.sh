#!/usr/bin/env bash
set -euo pipefail

repo_root="${EPISPACE_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
conda_env="${EPISPACE_CONDA_ENV:-behavior}"
conda_bin="${EPISPACE_CONDA_BIN:-$(command -v conda)}"
output_root="${1:-${repo_root}/outputs/behavior51_coverage_v2r2_incremental}"

source_root="${EPISPACE_V1_ROOT:-${repo_root}/outputs/behavior51_coverage_v1}"
audit_root="${EPISPACE_FINAL_AUDIT_ROOT:-${repo_root}/outputs/behavior51_coverage_v2_full_s01/audit}"
prior_repair_root="${EPISPACE_V2R1_ROOT:-${repo_root}/outputs/behavior51_coverage_v2_local_repair}"
direction_root="${EPISPACE_DIRECTION_ROOT:-${repo_root}/outputs/behavior51_direction_balance_v1}"
occlusion_root="${EPISPACE_OCCLUSION_ROOT:-${repo_root}/outputs/behavior51_occlusion_repair_v1}"

if [[ ! -f "${output_root}/coverage.plan.json" ]]; then
  "${conda_bin}" run --no-capture-output -n "${conda_env}" \
    python "${repo_root}/scripts/prepare_behavior51_v2r2_incremental_overlay.py" \
    --source "${source_root}" \
    --audit "${audit_root}" \
    --prior-repair "${prior_repair_root}" \
    --direction-root "${direction_root}/pure_rotation_left" \
    --direction-root "${direction_root}/pure_rotation_right" \
    --direction-root "${direction_root}/pure_translation_left" \
    --direction-root "${direction_root}/pure_translation_right" \
    --occlusion-root "${occlusion_root}" \
    --output "${output_root}" \
    --collection-id behavior51_coverage_v2r2_incremental
else
  echo "v2r2 overlay already exists; resuming ${output_root}"
fi

mkdir -p "${output_root}/logs"
exec bash "${repo_root}/scripts/run_behavior51_v2_local_repair.sh" "${output_root}"
