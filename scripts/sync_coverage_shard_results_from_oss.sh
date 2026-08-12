#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 oss://bucket/prefix/results SHARD_NAME LOCAL_RESULTS_ROOT" >&2
  exit 2
fi

oss_prefix="${1%/}"
shard_name="$2"
local_root="$(realpath -m "$3")"
ossutil_bin="${OSSUTIL_BIN:-ossutil}"
destination="${local_root}/${shard_name}"

mkdir -p "${destination}"
"${ossutil_bin}" sync "${oss_prefix}/${shard_name}" "${destination}" \
  --update --force
if [[ ! -f "${destination}/shard.complete.json" ]]; then
  echo "download is incomplete: ${destination}/shard.complete.json is missing" >&2
  exit 1
fi
echo "${destination}"
