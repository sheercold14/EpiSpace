#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="$ROOT/.deps/BEHAVIOR-1K"
TAG="${BEHAVIOR_TAG:-v3.9.0}"
ENV_NAME="${BEHAVIOR_ENV_NAME:-behavior-spatialep}"

if [[ ! -d "$UPSTREAM/.git" ]]; then
  mkdir -p "$ROOT/.deps"
  git clone --depth 1 --filter=blob:none --single-branch --branch "$TAG" \
    https://github.com/StanfordVL/BEHAVIOR-1K.git "$UPSTREAM"
fi

cd "$UPSTREAM"
if ! git describe --tags --exact-match 2>/dev/null | grep -qx "$TAG"; then
  git fetch --depth 1 origin "refs/tags/$TAG:refs/tags/$TAG"
  git checkout --detach "$TAG"
fi

echo "The official installer will request Conda and NVIDIA license acceptance."
echo "No license acceptance is automated by this project."
echo "BEHAVIOR assets are intentionally handled in a separate mirror-aware phase."
./setup.sh --new-env "$ENV_NAME" --omnigibson --bddl
