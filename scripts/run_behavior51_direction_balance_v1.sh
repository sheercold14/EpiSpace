#!/usr/bin/env bash
set -euo pipefail

repo_root="${EPISPACE_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source_root="${EPISPACE_DIRECTION_SOURCE:-${repo_root}/outputs/behavior51_coverage_v1}"
output_root="${EPISPACE_DIRECTION_OUTPUT:-${repo_root}/outputs/behavior51_direction_balance_v1}"
og_root="${EPISPACE_OG_ROOT:-/home/wmq/project/bench/OminiGibson}"
data_root="${EPISPACE_DATA_ROOT:-/home/wmq/project/bench/BEHAVIOR-1K/datasets}"
conda_env="${EPISPACE_CONDA_ENV:-behavior}"
conda_bin="${EPISPACE_CONDA_BIN:-$(command -v conda)}"
timeout_minutes="${EPISPACE_TIMEOUT_MINUTES:-20}"
read -r -a gpu_ids <<< "${EPISPACE_GPU_IDS:-0}"
workers="${EPISPACE_WORKERS:-${#gpu_ids[@]}}"
read -r -a wait_sessions <<< "${EPISPACE_WAIT_FOR_TMUX:-epispace-v2-local-repair epispace-occlusion-repair}"

export PYTHONPATH="${repo_root}/src:${og_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMNIGIBSON_DATA_PATH="${data_root}"
export OMNIGIBSON_HEADLESS=True
export OMNI_KIT_ACCEPT_EULA=YES
unset DISPLAY || true

mkdir -p "${output_root}/logs"
"${conda_bin}" run --no-capture-output -n "${conda_env}" \
  python "${repo_root}/scripts/build_behavior51_direction_balance_v1.py" \
  --source "${source_root}" \
  --output "${output_root}" \
  --accepted-per-label 3 \
  --attempts-per-binding 150 \
  --initial-attempts-per-binding 30

for session in "${wait_sessions[@]}"; do
  while tmux has-session -t "${session}" 2>/dev/null; do
    echo "waiting for active GPU session ${session} to finish"
    sleep 30
  done
done

tasks=(
  pure_rotation_left
  pure_rotation_right
  pure_translation_left
  pure_translation_right
)
for task in "${tasks[@]}"; do
  manifest="${output_root}/${task}/coverage.plan.json"
  echo "rendering direction balance task=${task} gpu_ids=${gpu_ids[*]} workers=${workers}"
  "${conda_bin}" run --no-capture-output -n "${conda_env}" \
    python -m spatial_episode.scriptgen.dataset_cli coverage-run \
    --manifest "${manifest}" \
    --og-root "${og_root}" \
    --conda-env "${conda_env}" \
    --data-root "${data_root}" \
    --gpu-ids "${gpu_ids[@]}" \
    --workers "${workers}" \
    --timeout-minutes "${timeout_minutes}"

  "${conda_bin}" run --no-capture-output -n "${conda_env}" \
    python -m spatial_episode.scriptgen.dataset_cli coverage-package \
    --manifest "${manifest}"
done

echo "direction balance complete: ${output_root}"
