#!/usr/bin/env bash
set -euo pipefail

repo_root="${EPISPACE_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source_index="${EPISPACE_SOURCE_INDEX:-${repo_root}/outputs/behavior51_sources_v1/source.index.json}"
output_root="${EPISPACE_OCCLUSION_OUTPUT:-${repo_root}/outputs/behavior51_occlusion_repair_v1}"
og_root="${EPISPACE_OG_ROOT:-/home/wmq/project/bench/OminiGibson}"
data_root="${EPISPACE_DATA_ROOT:-/home/wmq/project/bench/BEHAVIOR-1K/datasets}"
conda_env="${EPISPACE_CONDA_ENV:-behavior}"
conda_bin="${EPISPACE_CONDA_BIN:-$(command -v conda)}"
timeout_minutes="${EPISPACE_TIMEOUT_MINUTES:-20}"
wait_session="${EPISPACE_WAIT_FOR_TMUX:-epispace-v2-local-repair}"
binding_allowlist="${EPISPACE_OCCLUSION_BINDINGS:-${repo_root}/outputs/behavior51_occlusion_binding_allowlist_v1.json}"
read -r -a gpu_ids <<< "${EPISPACE_GPU_IDS:-0}"
workers="${EPISPACE_WORKERS:-${#gpu_ids[@]}}"

export PYTHONPATH="${repo_root}/src:${og_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMNIGIBSON_DATA_PATH="${data_root}"
export OMNIGIBSON_HEADLESS=True
export OMNI_KIT_ACCEPT_EULA=YES
unset DISPLAY || true

mkdir -p "${output_root}/logs"
if [[ ! -f "${binding_allowlist}" ]]; then
  "${conda_bin}" run --no-capture-output -n "${conda_env}" \
    python "${repo_root}/scripts/build_occlusion_binding_allowlist.py" \
    --dataset "${repo_root}/outputs/behavior51_coverage_v1/dataset.json" \
    --out "${binding_allowlist}"
fi
if [[ ! -f "${output_root}/coverage.plan.json" ]]; then
  echo "planning self_motion_update_after_occlusion into ${output_root}"
  "${conda_bin}" run --no-capture-output -n "${conda_env}" \
    python -m spatial_episode.scriptgen.dataset_cli coverage-plan \
    --source-index "${source_index}" \
    --output-root "${output_root}" \
    --collection-id behavior51_occlusion_repair_v1 \
    --accepted-per-binding 10 \
    --attempts-per-binding 150 \
    --initial-attempts-per-binding 30 \
    --binding-allowlist "${binding_allowlist}" \
    --capabilities \
      self_motion_update_after_occlusion \
      occluder_identification \
      disappearance_cause
fi

if [[ -n "${wait_session}" ]]; then
  while tmux has-session -t "${wait_session}" 2>/dev/null; do
    echo "waiting for active GPU session ${wait_session} to finish"
    sleep 30
  done
fi

echo "rendering occlusion repair: gpu_ids=${gpu_ids[*]} workers=${workers}"
"${conda_bin}" run --no-capture-output -n "${conda_env}" \
  python -m spatial_episode.scriptgen.dataset_cli coverage-run \
  --manifest "${output_root}/coverage.plan.json" \
  --og-root "${og_root}" \
  --conda-env "${conda_env}" \
  --data-root "${data_root}" \
  --gpu-ids "${gpu_ids[@]}" \
  --workers "${workers}" \
  --timeout-minutes "${timeout_minutes}"

"${conda_bin}" run --no-capture-output -n "${conda_env}" \
  python -m spatial_episode.scriptgen.dataset_cli coverage-package \
  --manifest "${output_root}/coverage.plan.json"

echo "occlusion repair complete: ${output_root}"
