#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${BEHAVIOR_ENV_NAME:-behavior-spatialep}"
LIB_DIR="$ROOT/.data/sysroot/usr/lib/x86_64-linux-gnu"

if [[ "${1:-}" == "--accept-eula" ]]; then
  export OMNI_KIT_ACCEPT_EULA=YES
  shift
fi
if [[ "${OMNI_KIT_ACCEPT_EULA:-}" != YES ]]; then
  echo "NVIDIA Omniverse EULA acceptance is required." >&2
  echo "Read the EULA, then pass --accept-eula or set OMNI_KIT_ACCEPT_EULA=YES." >&2
  exit 2
fi
if [[ $# -eq 0 ]]; then
  echo "Usage: $0 --accept-eula COMMAND [ARG ...]" >&2
  exit 2
fi
if [[ ! -e "$LIB_DIR/libGLU.so.1" || ! -e "$LIB_DIR/libOpenGL.so.0" ]]; then
  echo "Headless libraries are missing; run scripts/install_headless_runtime.sh." >&2
  exit 2
fi

export LD_LIBRARY_PATH="$LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMNIGIBSON_DATA_PATH="${OMNIGIBSON_DATA_PATH:-$ROOT/.data/omnigibson}"
exec conda run --no-capture-output -n "$ENV_NAME" "$@"
