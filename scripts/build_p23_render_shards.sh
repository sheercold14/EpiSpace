#!/usr/bin/env bash
# Audit and freeze the final P2/P3 plans into two scene-disjoint shards each.
set -euo pipefail

script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${EPISPACE_REPO_ROOT:-$(cd "${script_root}/.." && pwd)}"
python_bin="${EPISPACE_PYTHON:-python}"
backend_root="${EPISPACE_OG_ROOT:-${repo_root}/../OminiGibson}"
shard_count="${EPISPACE_SHARD_COUNT:-2}"

p2_manifest="${EPISPACE_P2_MANIFEST:-${repo_root}/outputs/p23_p2_7k_v1/coverage.plan.json}"
p3_manifest="${EPISPACE_P3_MANIFEST:-${repo_root}/outputs/p23_p3stream_7k_v1/coverage.plan.json}"
contract="${EPISPACE_P23_CONTRACT:-${repo_root}/outputs/p23_7k_search_contract.json}"
allowlist="${EPISPACE_P23_ALLOWLIST:-${repo_root}/outputs/p23_7k_binding_allowlist.json}"
audit="${EPISPACE_P23_AUDIT:-${repo_root}/outputs/p23_7k_trajectory_quality_audit.json}"
p2_output="${EPISPACE_P2_SHARD_OUTPUT:-${repo_root}/outputs/p23_p2_7k_render_shards_v1}"
p3_output="${EPISPACE_P3_SHARD_OUTPUT:-${repo_root}/outputs/p23_p3stream_7k_render_shards_v1}"

for required in "${p2_manifest}" "${p3_manifest}" "${contract}" "${allowlist}"; do
  [[ -f "${required}" ]] || {
    echo "missing required P2/P3 input: ${required}" >&2
    exit 1
  }
done

cd "${repo_root}"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

# A shard snapshot always starts from an explicit idle status.  This performs
# no rendering and rejects a status whose cells/candidates differ from plan.
"${python_bin}" -m spatial_episode.scriptgen.dataset_cli \
  coverage-init-status --manifest "${p2_manifest}"
"${python_bin}" -m spatial_episode.scriptgen.dataset_cli \
  coverage-init-status --manifest "${p3_manifest}"

"${python_bin}" scripts/audit_p23_7k_trajectory_quality.py \
  --p2 "${p2_manifest}" \
  --p3 "${p3_manifest}" \
  --contract "${contract}" \
  --allowlist "${allowlist}" \
  --output "${audit}"

build_phase() {
  local manifest=$1
  local output=$2
  "${python_bin}" scripts/coverage_render_shards.py build \
    --manifest "${manifest}" \
    --output "${output}" \
    --num-shards "${shard_count}" \
    --backend-root "${backend_root}"
  cp "${audit}" "${output}/p23_7k_trajectory_quality_audit.json"
  (
    cd "${output}"
    sha256sum p23_7k_trajectory_quality_audit.json \
      > p23_7k_trajectory_quality_audit.json.sha256
  )
}

build_phase "${p2_manifest}" "${p2_output}"
build_phase "${p3_manifest}" "${p3_output}"

echo "P2 shards: ${p2_output}"
echo "P3 shards: ${p3_output}"
echo "quality audit: ${audit}"
