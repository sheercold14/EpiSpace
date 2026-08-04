#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${BEHAVIOR_ENV_NAME:-behavior-spatialep}"
CACHE_ROOT="${ISAACSIM_WHEEL_CACHE:-$ROOT/.cache/isaacsim-5.1.0}"
ARIA2C="${ARIA2C:-$ROOT/.tools/aria2/bin/aria2c}"
ACCEPT_EULA=false

if [[ "${1:-}" == "--accept-eula" ]]; then
  ACCEPT_EULA=true
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: $0 --accept-eula" >&2
  exit 2
fi
if [[ "$ACCEPT_EULA" != true ]]; then
  echo "Refusing to install Isaac Sim without explicit NVIDIA EULA acceptance." >&2
  echo "Read the NVIDIA Isaac Sim EULA, then rerun with --accept-eula." >&2
  exit 2
fi
if [[ ! -x "$ARIA2C" ]]; then
  ARIA2C="$(command -v aria2c || true)"
fi
[[ -n "$ARIA2C" && -x "$ARIA2C" ]] || {
  echo "aria2c is required for resumable Isaac Sim wheel downloads." >&2
  exit 2
}

packages=(
  omniverse_kit-107.3.1.206797
  isaacsim_kernel-5.1.0.0
  isaacsim_app-5.1.0.0
  isaacsim_core-5.1.0.0
  isaacsim_gui-5.1.0.0
  isaacsim_utils-5.1.0.0
  isaacsim_storage-5.1.0.0
  isaacsim_asset-5.1.0.0
  isaacsim_sensor-5.1.0.0
  isaacsim_robot_motion-5.1.0.0
  isaacsim_robot-5.1.0.0
  isaacsim_benchmark-5.1.0.0
  isaacsim_code_editor-5.1.0.0
  isaacsim_ros1-5.1.0.0
  isaacsim_cortex-5.1.0.0
  isaacsim_example-5.1.0.0
  isaacsim_replicator-5.1.0.0
  isaacsim_rl-5.1.0.0
  isaacsim_robot_setup-5.1.0.0
  isaacsim_ros2-5.1.0.0
  isaacsim_template-5.1.0.0
  isaacsim_test-5.1.0.0
  isaacsim-5.1.0.0
  isaacsim_extscache_physics-5.1.0.0
  isaacsim_extscache_kit-5.1.0.0
  isaacsim_extscache_kit_sdk-5.1.0.0
)

mkdir -p "$CACHE_ROOT"
urls=()
wheels=()
for package in "${packages[@]}"; do
  package_name="${package%-*}"
  filename="${package}-cp311-none-manylinux_2_35_x86_64.whl"
  urls+=("https://pypi.nvidia.com/${package_name//_/-}/$filename")
  wheels+=("$CACHE_ROOT/$filename")
done

"$ARIA2C" \
  --force-sequential=true \
  --continue=true \
  --max-connection-per-server=4 \
  --split=4 \
  --min-split-size=16M \
  --file-allocation=none \
  --auto-file-renaming=false \
  --allow-overwrite=true \
  --max-tries=0 \
  --retry-wait=5 \
  --timeout=60 \
  --summary-interval=30 \
  --dir="$CACHE_ROOT" \
  "${urls[@]}"

python - "${wheels[@]}" <<'PY'
import pathlib
import sys
import zipfile

for raw_path in sys.argv[1:]:
    path = pathlib.Path(raw_path)
    if not path.is_file():
        raise SystemExit(f"missing Isaac Sim wheel: {path}")
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
    if bad is not None:
        raise SystemExit(f"corrupt member {bad!r} in {path}")
print(f"wheel integrity: pass ({len(sys.argv) - 1} files)")
PY

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export OMNI_KIT_ACCEPT_EULA=YES
python -m pip install "${wheels[@]}"
python -m pip install --force-reinstall cffi==1.17.1 "websockets>=15.0.1"
python -c "import isaacsim; print('Isaac Sim import: pass')"

echo "Isaac Sim 5.1.0 installed from persistent cache: $CACHE_ROOT"
