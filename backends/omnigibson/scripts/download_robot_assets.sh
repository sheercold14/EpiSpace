#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${OMNIGIBSON_DATA_PATH:-$ROOT/.data/omnigibson}"
CACHE_ROOT="${BEHAVIOR_DOWNLOAD_CACHE:-$ROOT/.cache/downloads}"
FAST_DOWNLOADER="${HF_FAST_DOWNLOADER:-/home/shichao/.codex/skills/hf-fast-download/scripts/hf_dynamic_fast_download.sh}"
ARIA2C="${ARIA2C:-$ROOT/.tools/aria2/bin/aria2c}"
FILENAME="omnigibson-robot-assets-3.8.2.zip"
EXPECTED_SIZE=641004353
URL="${OMNIGIBSON_ROBOT_ASSET_URL:-https://hf-mirror.com/datasets/behavior-1k/zipped-datasets/resolve/main/$FILENAME}"
TRANSPORT="fast"
ARCHIVE_OVERRIDE=""

usage() {
  cat <<'EOF'
Usage: download_robot_assets.sh [--transport fast|proxy] [--archive PATH]

Installs the OmniGibson 3.8.2 runtime assets. OmniGibson checks this bundle at
every simulator launch, including camera-only environments with robots=[].
Extraction is verified, staged and moved into place atomically.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --transport) TRANSPORT="$2"; shift 2 ;;
    --archive) ARCHIVE_OVERRIDE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$TRANSPORT" != fast && "$TRANSPORT" != proxy ]]; then
  echo "--transport must be fast or proxy" >&2
  exit 2
fi

mkdir -p "$CACHE_ROOT" "$DATA_ROOT"
ARCHIVE="${ARCHIVE_OVERRIDE:-$CACHE_ROOT/$FILENAME}"
if [[ -n "$ARCHIVE_OVERRIDE" ]]; then
  ARCHIVE="$(readlink -f "$ARCHIVE")"
  [[ -f "$ARCHIVE" ]] || { echo "Asset archive not found: $ARCHIVE" >&2; exit 2; }
elif [[ "$TRANSPORT" == fast ]]; then
  [[ -x "$FAST_DOWNLOADER" ]] || { echo "Fast downloader not found: $FAST_DOWNLOADER" >&2; exit 2; }
  "$FAST_DOWNLOADER" "$URL" "$ARCHIVE" --connections 8
else
  if [[ ! -x "$ARIA2C" ]]; then
    ARIA2C="$(command -v aria2c || true)"
  fi
  [[ -n "$ARIA2C" && -x "$ARIA2C" ]] || { echo "aria2c is required for proxy transport" >&2; exit 2; }
  : "${http_proxy:=http://127.0.0.1:7890}"
  : "${https_proxy:=http://127.0.0.1:7890}"
  export http_proxy https_proxy
  "$ARIA2C" --continue=true --file-allocation=none --max-connection-per-server=16 \
    --split=16 --min-split-size=8M --all-proxy="$https_proxy" \
    --dir="$CACHE_ROOT" --out="$FILENAME" "$URL"
fi

python - "$ARCHIVE" "$EXPECTED_SIZE" <<'PY'
import os
import sys
import zipfile

path = sys.argv[1]
expected_size = int(sys.argv[2])
actual_size = os.path.getsize(path)
if actual_size != expected_size:
    raise SystemExit(f"asset archive size mismatch: expected {expected_size}, got {actual_size}")
with zipfile.ZipFile(path) as archive:
    bad = archive.testzip()
if bad is not None:
    raise SystemExit(f"corrupt member in downloaded archive: {bad}")
print("zip integrity: pass")
PY

TARGET="$DATA_ROOT/omnigibson-robot-assets"
STAGING="$DATA_ROOT/.omnigibson-robot-assets.staging.$$"
rm -rf "$STAGING"
mkdir -p "$STAGING"
python - "$ARCHIVE" "$STAGING" <<'PY'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as archive:
    archive.extractall(sys.argv[2])
PY
[[ "$(<"$STAGING/VERSION")" == "3.8.2" ]] || {
  echo "Unexpected or missing robot asset VERSION" >&2
  rm -rf "$STAGING"
  exit 2
}
rm -rf "$TARGET"
mv "$STAGING" "$TARGET"

echo "OmniGibson robot assets installed at: $TARGET"
