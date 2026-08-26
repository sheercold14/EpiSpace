#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SHARD_DISTRIBUTION_DIR oss://bucket/prefix" >&2
  exit 2
fi

distribution_dir="$(realpath "$1")"
oss_prefix="${2%/}"
ossutil_bin="${OSSUTIL_BIN:-ossutil}"

if [[ ! -f "${distribution_dir}/distribution.json" ]]; then
  echo "missing distribution.json under ${distribution_dir}" >&2
  exit 1
fi

shopt -s nullglob
archives=("${distribution_dir}"/*.tar.zst)
if (( ${#archives[@]} == 0 )); then
  echo "no .tar.zst shard archives under ${distribution_dir}" >&2
  exit 1
fi

"${ossutil_bin}" cp "${distribution_dir}/distribution.json" \
  "${oss_prefix}/distribution.json" --update --force
for archive in "${archives[@]}"; do
  checksum="${archive}.sha256"
  if [[ ! -f "${checksum}" ]]; then
    echo "missing archive checksum: ${checksum}" >&2
    exit 1
  fi
  "${ossutil_bin}" cp "${archive}" "${oss_prefix}/$(basename "${archive}")" \
    --update --force
  "${ossutil_bin}" cp "${checksum}" "${oss_prefix}/$(basename "${checksum}")" \
    --update --force
done

# P2/P3 distributions carry the hash-bound global 7k quality audit.  Generic
# coverage distributions remain valid without it.
quality_audit="${distribution_dir}/p23_7k_trajectory_quality_audit.json"
if [[ -f "${quality_audit}" ]]; then
  quality_checksum="${quality_audit}.sha256"
  if [[ ! -f "${quality_checksum}" ]]; then
    echo "missing P2/P3 quality audit checksum: ${quality_checksum}" >&2
    exit 1
  fi
  (
    cd "${distribution_dir}"
    sha256sum --check "$(basename "${quality_checksum}")"
  )
  "${ossutil_bin}" cp "${quality_audit}" \
    "${oss_prefix}/$(basename "${quality_audit}")" --update --force
  "${ossutil_bin}" cp "${quality_checksum}" \
    "${oss_prefix}/$(basename "${quality_checksum}")" --update --force
fi

echo "uploaded ${#archives[@]} shard archives to ${oss_prefix}"
