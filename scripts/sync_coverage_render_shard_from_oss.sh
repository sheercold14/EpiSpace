#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 oss://bucket/prefix SHARD_ARCHIVE_NAME DOWNLOAD_ROOT" >&2
  echo "SHARD_ARCHIVE_NAME may be given with or without .tar.zst" >&2
  exit 2
fi

oss_prefix="${1%/}"
archive_name="${2%.tar.zst}.tar.zst"
download_root="$(realpath -m "$3")"
ossutil_bin="${OSSUTIL_BIN:-ossutil}"

mkdir -p "${download_root}/archives" "${download_root}/packages"
local_archive="${download_root}/archives/${archive_name}"
local_checksum="${local_archive}.sha256"
"${ossutil_bin}" cp "${oss_prefix}/${archive_name}" "${local_archive}" \
  --update --force
"${ossutil_bin}" cp "${oss_prefix}/${archive_name}.sha256" "${local_checksum}" \
  --update --force
(cd "${download_root}/archives" && sha256sum --check "$(basename "${local_checksum}")")
tar --zstd -xf "${local_archive}" -C "${download_root}/packages"

package_root="${download_root}/packages/${archive_name%.tar.zst}"
if [[ ! -f "${package_root}/shard.json" ]]; then
  echo "archive did not contain the expected package: ${package_root}" >&2
  exit 1
fi
echo "${package_root}"
