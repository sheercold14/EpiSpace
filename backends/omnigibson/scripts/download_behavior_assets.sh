#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${BEHAVIOR_ENV_NAME:-behavior-spatialep}"
DATA_ROOT="${OMNIGIBSON_DATA_PATH:-$ROOT/.data/omnigibson}"
CACHE_ROOT="${BEHAVIOR_DOWNLOAD_CACHE:-$ROOT/.cache/downloads}"
FAST_DOWNLOADER="${HF_FAST_DOWNLOADER:-/home/shichao/.codex/skills/hf-fast-download/scripts/hf_dynamic_fast_download.sh}"
ARIA2C="${ARIA2C:-$ROOT/.tools/aria2/bin/aria2c}"
FILENAME="behavior-1k-assets-3.9.0.zip"
EXPECTED_SIZE=31457673073
URL="${BEHAVIOR_ASSET_URL:-https://hf-mirror.com/datasets/behavior-1k/zipped-datasets/resolve/main/$FILENAME}"
TRANSPORT="fast"
ACCEPT_LICENSE=false
ARCHIVE_OVERRIDE=""

usage() {
  cat <<'EOF'
Usage: download_behavior_assets.sh --accept-license [--transport fast|proxy]
                                   [--archive PATH]

Downloads only the BEHAVIOR scene/object bundle required by the static episode
worker. Robot assets and challenge task instances are intentionally omitted.

--accept-license is an explicit acknowledgement of the BEHAVIOR Data Bundle
agreement printed by the upstream installer. See https://behavior.stanford.edu/.
--archive uses an already transferred archive without downloading another copy.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --accept-license) ACCEPT_LICENSE=true; shift ;;
    --transport) TRANSPORT="$2"; shift 2 ;;
    --archive) ARCHIVE_OVERRIDE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$ACCEPT_LICENSE" != true ]]; then
  echo "Refusing to download licensed BEHAVIOR assets without --accept-license." >&2
  echo "Read the agreement in the upstream setup output or at the BEHAVIOR site." >&2
  exit 2
fi
if [[ "$TRANSPORT" != fast && "$TRANSPORT" != proxy ]]; then
  echo "--transport must be fast or proxy" >&2
  exit 2
fi
if [[ ! -d "$ROOT/.deps/BEHAVIOR-1K/OmniGibson" ]]; then
  echo "Pinned BEHAVIOR source is missing; run scripts/bootstrap_behavior.sh first." >&2
  exit 2
fi
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' is missing; run bootstrap first." >&2
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
  [[ -n "$ARIA2C" && -x "$ARIA2C" ]] || {
    echo "aria2c is required for proxy transport" >&2
    exit 2
  }
  : "${http_proxy:=http://127.0.0.1:7890}"
  : "${https_proxy:=http://127.0.0.1:7890}"
  export http_proxy https_proxy
  "$ARIA2C" --continue=true --max-connection-per-server=16 --split=16 \
    --min-split-size=16M --dir="$CACHE_ROOT" --out="$FILENAME" "$URL"
fi

python - "$ARCHIVE" "$EXPECTED_SIZE" <<'PY'
import os
import sys, zipfile
path = sys.argv[1]
expected_size = int(sys.argv[2])
actual_size = os.path.getsize(path)
if actual_size != expected_size:
    raise SystemExit(
        f"asset archive size mismatch: expected {expected_size}, got {actual_size}"
    )
with zipfile.ZipFile(path) as archive:
    bad = archive.testzip()
if bad is not None:
    raise SystemExit(f"corrupt member in downloaded archive: {bad}")
print("zip integrity: pass")
PY

TARGET="$DATA_ROOT/behavior-1k-assets"
STAGING="$DATA_ROOT/.behavior-1k-assets.staging.$$"
rm -rf "$STAGING"
mkdir -p "$STAGING"
python - "$ARCHIVE" "$STAGING" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as archive:
    archive.extractall(sys.argv[2])
PY
rm -rf "$TARGET"
mv "$STAGING" "$TARGET"

# Install the encryption key after explicit acceptance. Since the asset target
# now exists, the upstream function will not redownload the 31 GB archive.
OMNIGIBSON_NO_OMNIVERSE=1 OMNIGIBSON_DATA_PATH="$DATA_ROOT" \
  conda run -n "$ENV_NAME" python -c \
  "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)"
chmod 600 "$DATA_ROOT/omnigibson.key"

echo "BEHAVIOR assets installed at: $TARGET"
