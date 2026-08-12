#!/usr/bin/env bash
set -euo pipefail

epispace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
backend_root="${EPISPACE_OG_BACKEND_ROOT:-/home/wmq/project/bench/OminiGibson}"
behavior_root="${EPISPACE_BEHAVIOR_ROOT:-/home/wmq/project/bench/BEHAVIOR-1K}"
python_bin="${EPISPACE_BEHAVIOR_PYTHON:-/home/wmq/.conda/envs/behavior/bin/python}"

scene_model="gates_bedroom"
output_dir=""
gpu_id="0"
accept_eula=false
overwrite=false
compile_only=false
dry_run=false

usage() {
    cat <<'EOF'
Usage:
  scripts/generate_scene_ir.sh --accept-eula [options]

Options:
  --scene NAME       BEHAVIOR scene model (default: gates_bedroom)
  --output DIR       Output bundle directory
  --gpu-id ID        GPU identifier (default: 0)
  --overwrite        Replace an existing acquisition bundle
  --compile-only     Compile an already acquired bundle without launching OmniGibson
  --dry-run          Validate paths and generated recipe without acquiring
  --accept-eula      Confirm acceptance of the NVIDIA Omniverse EULA
  -h, --help         Show this help

Environment overrides:
  EPISPACE_OG_BACKEND_ROOT
  EPISPACE_BEHAVIOR_ROOT
  EPISPACE_BEHAVIOR_PYTHON
EOF
}

die() {
    echo "error: $*" >&2
    exit 2
}

while (($#)); do
    case "$1" in
        --scene)
            [[ $# -ge 2 ]] || die "--scene requires a value"
            scene_model="$2"
            shift 2
            ;;
        --output)
            [[ $# -ge 2 ]] || die "--output requires a value"
            output_dir="$2"
            shift 2
            ;;
        --gpu-id)
            [[ $# -ge 2 ]] || die "--gpu-id requires a value"
            gpu_id="$2"
            shift 2
            ;;
        --overwrite)
            overwrite=true
            shift
            ;;
        --compile-only)
            compile_only=true
            shift
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        --accept-eula)
            accept_eula=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

if [[ -z "$output_dir" ]]; then
    output_dir="$epispace_root/outputs/${scene_model}_baseline"
elif [[ "$output_dir" != /* ]]; then
    output_dir="$epispace_root/$output_dir"
fi

case "$output_dir" in
    /|"$epispace_root"|"$backend_root"|"$behavior_root")
        die "refusing unsafe output directory: $output_dir"
        ;;
esac

assets_root="$behavior_root/datasets/behavior-1k-assets"
scene_file="$assets_root/scenes/$scene_model/json/${scene_model}_best.json"
template="$backend_root/configs/omnigibson_static_m1.yaml"

[[ -x "$python_bin" ]] || die "behavior Python is missing: $python_bin"
[[ -d "$backend_root/src/omnigibson_episode" ]] || die "backend is missing: $backend_root"
[[ -d "$epispace_root/src/spatial_episode" ]] || die "EpiSpace package is missing"
[[ -s "$assets_root/VERSION" ]] || die "BEHAVIOR assets are not installed: $assets_root"
[[ -s "$behavior_root/datasets/omnigibson.key" ]] || die "omnigibson.key is missing"
[[ -s "$scene_file" ]] || die "scene instance is missing: $scene_file"

export OMNIGIBSON_DATA_PATH="$behavior_root/datasets"
export PYTHONPATH="$epispace_root/src:$backend_root/src${PYTHONPATH:+:$PYTHONPATH}"

if $compile_only; then
    for member in scene_snapshot.json trajectory_plan.json render_report.json; do
        [[ -s "$output_dir/$member" ]] || die "compile input is missing: $output_dir/$member"
    done
else
    [[ -s "$template" ]] || die "bootstrap recipe is missing: $template"
    if ! $dry_run && ! $accept_eula && [[ "${OMNI_KIT_ACCEPT_EULA:-}" != "YES" ]]; then
        die "read the NVIDIA Omniverse EULA, then pass --accept-eula"
    fi
fi

recipe_path=""
cleanup() {
    if [[ -n "$recipe_path" ]]; then
        rm -f -- "$recipe_path"
    fi
}
trap cleanup EXIT

if ! $compile_only; then
    recipe_path="$(mktemp --tmpdir --suffix=.yaml epispace-scene-ir-recipe.XXXXXX)"
    sed \
        -e "s/recipe_id: omnigibson_static_m1/recipe_id: ${scene_model}_baseline/" \
        -e "s/scene_model: Rs_int/scene_model: ${scene_model}/" \
        "$template" > "$recipe_path"

    "$python_bin" - "$recipe_path" <<'PY'
import sys
from pathlib import Path
from omnigibson_episode.config import load_recipe

recipe = load_recipe(Path(sys.argv[1]))
print(
    f"Validated recipe: scene={recipe.source.scene_model}, "
    f"strategy={recipe.trajectory.sampling_strategy}, seed={recipe.seed}"
)
PY
fi

echo "Scene:  $scene_model"
echo "Bundle: $output_dir"
echo "GPU:    $gpu_id"

if $dry_run; then
    echo "Dry run passed; OmniGibson was not launched."
    exit 0
fi

mkdir -p "$(dirname "$output_dir")"

if ! $compile_only; then
    export OMNI_KIT_ACCEPT_EULA=YES
    acquire_args=(
        -m omnigibson_episode.cli acquire
        --recipe "$recipe_path"
        --output "$output_dir"
        --gpu-id "$gpu_id"
        --headless
    )
    if $overwrite; then
        acquire_args+=(--overwrite)
    fi
    "$python_bin" "${acquire_args[@]}"
fi

"$python_bin" -m omnigibson_episode.cli compile \
    --bundle "$output_dir" \
    --output "$output_dir/spatial_episode.json"

scene_ir="$output_dir/scene_ir.json"
[[ -s "$scene_ir" ]] || die "compile completed without scene_ir.json"
echo "Generated: $scene_ir"
