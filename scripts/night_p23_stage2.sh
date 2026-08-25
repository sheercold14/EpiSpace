#!/usr/bin/env bash
# Finish the marker-enabled P2 + streaming-only P3 CPU plan.  This script does
# not launch a renderer; its outputs are tomorrow's reviewed render contract.
set -euo pipefail

ROOT=/data/shichao/data/dataV100/code/EpiSpace
SOURCES=$ROOT/outputs/p23_sources_all_v1
P2_NAME=p23_p2_badged_v2
P3_NAME=p23_p3stream_badged_v1
SELECTION=$ROOT/outputs/$P3_NAME/binding.selection.json

cd "$ROOT"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate behavior-spatialep
export PYTHONPATH=src

P2=(reference_frame_transform reference_frame_transform_yaw45 reference_frame_transform_yaw90
    reference_frame_transform_yaw135 reference_frame_transform_yaw180
    reference_frame_transform_yawm45 reference_frame_transform_yawm90
    reference_frame_transform_yawm135
    reference_frame_visibility reference_frame_visibility_yaw45 reference_frame_visibility_yaw90
    reference_frame_visibility_yaw135 reference_frame_visibility_yaw180
    reference_frame_visibility_yawm45 reference_frame_visibility_yawm90
    reference_frame_visibility_yawm135)
P3=(cross_view_ego_k1 cross_view_ego_k2 cross_view_ego_k3
    cross_view_anchor_k1 cross_view_anchor_k2 cross_view_anchor_k3
    cross_view_closer_k1 cross_view_closer_k2 cross_view_closer_k3)

echo "[$(date +%T)] selecting exact streaming k mix"
python scripts/select_p3_streaming_bindings.py \
  --source-index "$SOURCES/source.index.json" \
  --output "$SELECTION" \
  --per-scene-cap 12 \
  --search-buffer-factor 2.0 \
  --workers 12 \
  --maximum-multislot-bindings 128

echo "[$(date +%T)] planning marker-enabled P2"
python -m spatial_episode.scriptgen.dataset_cli coverage-plan \
  --source-index "$SOURCES/source.index.json" \
  --output-root "$ROOT/outputs/$P2_NAME" \
  --collection-id "$P2_NAME" \
  --capabilities "${P2[@]}" \
  --maximum-multislot-bindings 128 \
  --limit-bindings-per-capability 8 \
  --accepted-per-binding 2 \
  --initial-attempts-per-binding 24

echo "[$(date +%T)] planning streaming-only P3"
python -m spatial_episode.scriptgen.dataset_cli coverage-plan \
  --source-index "$SOURCES/source.index.json" \
  --output-root "$ROOT/outputs/$P3_NAME" \
  --collection-id "$P3_NAME" \
  --capabilities "${P3[@]}" \
  --binding-allowlist "$SELECTION" \
  --maximum-multislot-bindings 128 \
  --limit-bindings-per-capability 12 \
  --accepted-per-binding 2 \
  --initial-attempts-per-binding 24

python scripts/finalize_p23_search_plan.py \
  --p2 "$ROOT/outputs/$P2_NAME/coverage.plan.json" \
  --p3 "$ROOT/outputs/$P3_NAME/coverage.plan.json" \
  --selection "$SELECTION" \
  --output "$ROOT/outputs/p23_search_plan_finalization.json" \
  --p2-per-scene-cap 8

python scripts/repair_p23_search_plan.py \
  --manifest "$ROOT/outputs/$P3_NAME/coverage.plan.json" \
  --attempt-limit 150 \
  --output "$ROOT/outputs/p23_search_plan_repair.json"

python scripts/make_first_batch.py \
  --plan "$ROOT/outputs/$P2_NAME/coverage.plan.json" \
  --output "$ROOT/outputs/$P2_NAME/batch1.cells.json" --fraction 0.2
python scripts/make_first_batch.py \
  --plan "$ROOT/outputs/$P3_NAME/coverage.plan.json" \
  --output "$ROOT/outputs/$P3_NAME/batch1.cells.json" --fraction 0.2

python scripts/audit_p23_search_plan.py \
  --p2 "$ROOT/outputs/$P2_NAME/coverage.plan.json" \
  --p3 "$ROOT/outputs/$P3_NAME/coverage.plan.json" \
  --selection "$SELECTION" \
  --output "$ROOT/outputs/p23_search_plan_audit.json"

echo "[$(date +%T)] CPU search plan complete; no render was started"
