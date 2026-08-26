#!/usr/bin/env bash
# Remove only sentinel-marked EpiSpace scratch directories under one run root.
set -euo pipefail

usage() {
  echo "usage: $0 [--once] SCRATCH_ROOT" >&2
  exit 2
}

once=0
if [[ "${1:-}" == "--once" ]]; then
  once=1
  shift
fi
[[ $# -eq 1 ]] || usage

grace_minutes="${GRACE_MINUTES:-30}"
interval_seconds="${INTERVAL_SECONDS:-600}"
[[ "${grace_minutes}" =~ ^[0-9]+$ ]] || {
  echo "GRACE_MINUTES must be a non-negative integer" >&2
  exit 2
}
[[ "${interval_seconds}" =~ ^[1-9][0-9]*$ ]] || {
  echo "INTERVAL_SECONDS must be a positive integer" >&2
  exit 2
}

scratch_root="$(realpath -e "$1")"
case "${scratch_root}" in
  /|/tmp|/var|/var/tmp|/data|/home)
    echo "refusing broad scratch root: ${scratch_root}" >&2
    exit 1
    ;;
esac
[[ -d "${scratch_root}" ]] || {
  echo "scratch root is not a directory: ${scratch_root}" >&2
  exit 1
}

current_uid="$(id -u)"

directory_is_live() {
  local candidate=$1
  local proc path link
  for proc in /proc/[0-9]*; do
    [[ -d "${proc}" ]] || continue
    for link in "${proc}/cwd" "${proc}"/fd/*; do
      path="$(readlink -f "${link}" 2>/dev/null || true)"
      if [[ "${path}" == "${candidate}" || "${path}" == "${candidate}/"* ]]; then
        return 0
      fi
    done
    if grep -Fq -- "${candidate}/" "${proc}/maps" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

clean_once() {
  local now removed spared candidate resolved owner modified age
  now="$(date +%s)"
  removed=0
  spared=0
  shopt -s nullglob
  for candidate in "${scratch_root}"/*; do
    [[ -d "${candidate}" && -f "${candidate}/.epispace_scratch" ]] || continue
    [[ "$(head -n 1 "${candidate}/.epispace_scratch")" == "epispace_scratch.v1" ]] || {
      echo "skipping invalid scratch sentinel: ${candidate}" >&2
      spared=$((spared + 1))
      continue
    }
    resolved="$(realpath -e "${candidate}")"
    [[ "${resolved}" == "${scratch_root}/"* ]] || {
      echo "skipping scratch path outside root: ${resolved}" >&2
      spared=$((spared + 1))
      continue
    }
    owner="$(stat -c %u "${resolved}")"
    [[ "${owner}" == "${current_uid}" ]] || {
      echo "skipping scratch owned by uid ${owner}: ${resolved}" >&2
      spared=$((spared + 1))
      continue
    }
    modified="$(stat -c %Y "${resolved}")"
    age=$((now - modified))
    if (( age < grace_minutes * 60 )) || directory_is_live "${resolved}"; then
      spared=$((spared + 1))
      continue
    fi
    rm -rf --one-file-system -- "${resolved}"
    removed=$((removed + 1))
  done
  echo "[$(date +%F' '%T)] EpiSpace scratch removed=${removed} spared=${spared} root=${scratch_root}"
}

while true; do
  clean_once
  (( once == 1 )) && exit 0
  sleep "${interval_seconds}"
done
