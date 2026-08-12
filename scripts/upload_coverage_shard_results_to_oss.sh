#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 WORK_ROOT oss://bucket/prefix/results" >&2
  exit 2
fi

work_root="$(realpath "$1")"
oss_prefix="${2%/}"
output_root="${work_root}/output"
ossutil_bin="${OSSUTIL_BIN:-ossutil}"

if [[ ! -f "${output_root}/shard.complete.json" ]]; then
  echo "shard is not finalized: ${output_root}/shard.complete.json is missing" >&2
  exit 1
fi

shard_name="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["shard_name"])' \
  "${output_root}/shard.json")"
destination="${oss_prefix}/${shard_name}"
"${ossutil_bin}" sync "${output_root}" "${destination}" --update --force
echo "uploaded shard result to ${destination}"
