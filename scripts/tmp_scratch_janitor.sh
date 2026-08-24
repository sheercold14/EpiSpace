#!/usr/bin/env bash
# Reclaims the USD scratch directories OmniGibson leaves in /tmp.
#
# Each scene render stages roughly 2.4 GB under /tmp/tmpXXXXXXXX and does not
# remove it, so an unattended overnight run fills the root filesystem and dies
# with "No space left on device" - which is how Wainscott_0_garden was lost on
# 2026-08-24.  This deletes a scratch directory only when it is both older than
# GRACE_MINUTES and referenced by no live process, so a directory belonging to
# a render still in progress is never touched.
set -uo pipefail

GRACE_MINUTES=${GRACE_MINUTES:-30}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-600}

# Scratch paths any live process still has open, as its cwd, or mapped. Reading
# /proc directly costs a few hundred milliseconds, where "lsof +D /tmp" would
# walk every file in every scratch directory.
live_paths() {
  {
    ls -l /proc/[0-9]*/fd/* 2>/dev/null | grep -o '/tmp/tmp[a-z0-9_]\{8\}'
    ls -l /proc/[0-9]*/cwd 2>/dev/null | grep -o '/tmp/tmp[a-z0-9_]\{8\}'
    grep -ho '/tmp/tmp[a-z0-9_]\{8\}' /proc/[0-9]*/maps 2>/dev/null
  } | sort -u
}

while true; do
  free_gib=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  live=$(mktemp /tmp/janitor-live.XXXXXX)
  stale=$(mktemp /tmp/janitor-stale.XXXXXX)
  live_paths > "$live"
  find /tmp -maxdepth 1 -type d -name 'tmp????????' -mmin "+$GRACE_MINUTES" 2>/dev/null |
    sort > "$stale"
  mapfile -t doomed < <(comm -23 "$stale" "$live")
  if [ "${#doomed[@]}" -gt 0 ]; then
    printf '%s\n' "${doomed[@]}" | xargs -d '\n' -P 4 -n 25 rm -rf 2>/dev/null
    echo "[$(date +%F' '%T)] removed ${#doomed[@]}, spared $(wc -l < "$live") live, free ${free_gib}G -> $(df -BG --output=avail / | tail -1 | tr -d ' ')"
  else
    echo "[$(date +%F' '%T)] nothing stale, free ${free_gib}G"
  fi
  rm -f "$live" "$stale"
  sleep "$INTERVAL_SECONDS"
done
