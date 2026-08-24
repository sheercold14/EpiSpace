#!/usr/bin/env bash
# Waits for the all-scene source render, then runs the two CPU stages that
# decide what tonight's GPU batch will contain: the per-scene binding screen
# and the two coverage plans.  Nothing here touches a GPU, and every step is
# re-runnable, so it is safe to leave unattended.  The render itself is
# launched by hand after the plan is read.
set -euo pipefail

ROOT=/data/shichao/data/dataV100/code/EpiSpace
SOURCES=$ROOT/outputs/p23_sources_all_v1
PREPARE_PID=${1:?usage: night_p23_stage2.sh <prepare-scenes-pid>}

cd "$ROOT"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate behavior-spatialep
export PYTHONPATH=src

echo "[$(date +%T)] waiting on prepare-scenes pid=$PREPARE_PID"
while kill -0 "$PREPARE_PID" 2>/dev/null; do sleep 60; done
echo "[$(date +%T)] sources done: $(grep -c 'source ready' /tmp/prepare_all51.log) scenes"

# A scene that died on a full disk is worth one retry before it is written off:
# the garden scenes are the only ones that have ever produced deep chains, and
# losing one to an environment fault would quietly shrink the P3 scene pool.
mapfile -t FAILED < <(grep -o 'source failed scene=\S*' /tmp/prepare_all51.log |
  cut -d= -f2 | sort -u)
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "[$(date +%T)] retrying failed sources: ${FAILED[*]}"
  python -m spatial_episode.scriptgen.dataset_cli prepare-scenes \
    --scenes "${FAILED[@]}" \
    --source-root "$SOURCES" \
    --source-id p23_sources_all_v1 \
    --og-root /data/shichao/data/dataV100/code/OminiGibson \
    --conda-env behavior-spatialep \
    --data-root /data/shichao/data/dataV100/code/OminiGibson/.data/omnigibson \
    --gpu-ids 0 1 2 3 --workers 4 --accept-eula \
    >> /tmp/prepare_all51.log 2>&1 || echo "[$(date +%T)] retry did not fully succeed"
fi

echo "[$(date +%T)] screening per-scene binding yield"
# 128 is coverage-plan's own binding pool size, so the screened counts are the
# counts the plan will see rather than a raw capacity figure.
python scripts/screen_scene_yield.py \
  --bundles-root "$SOURCES/scenes" \
  --output "$ROOT/outputs/p23_yield_all_v1.json" \
  --maximum-multislot-bindings 128 \
  --workers 12

# The snapshot line is the main training line; the walking line is held out
# for evaluation, so the two are planned as separate collections and only the
# snapshot collection is rendered tonight.
P2=(reference_frame_transform reference_frame_transform_yaw45 reference_frame_transform_yaw90
    reference_frame_transform_yaw135 reference_frame_transform_yaw180
    reference_frame_transform_yawm45 reference_frame_transform_yawm90
    reference_frame_transform_yawm135
    reference_frame_visibility reference_frame_visibility_yaw45 reference_frame_visibility_yaw90
    reference_frame_visibility_yaw135 reference_frame_visibility_yaw180
    reference_frame_visibility_yawm45 reference_frame_visibility_yawm90
    reference_frame_visibility_yawm135)
SNAPSHOT=(cross_view_snapshot_ego_k1 cross_view_snapshot_ego_k2 cross_view_snapshot_ego_k3
          cross_view_snapshot_anchor_k1 cross_view_snapshot_anchor_k2 cross_view_snapshot_anchor_k3
          cross_view_snapshot_closer_k1 cross_view_snapshot_closer_k2 cross_view_snapshot_closer_k3)
WALKING=(cross_view_ego_k1 cross_view_ego_k2 cross_view_ego_k3
         cross_view_anchor_k1 cross_view_anchor_k2 cross_view_anchor_k3
         cross_view_closer_k1 cross_view_closer_k2 cross_view_closer_k3)

echo "[$(date +%T)] planning main collection (P2 + P3 snapshot)"
python -m spatial_episode.scriptgen.dataset_cli coverage-plan \
  --source-index "$SOURCES/source.index.json" \
  --output-root "$ROOT/outputs/p23_main_v1" \
  --collection-id p23_main_v1 \
  --capabilities "${P2[@]}" "${SNAPSHOT[@]}" \
  --limit-bindings-per-capability 32 \
  --accepted-per-binding 2

echo "[$(date +%T)] planning holdout collection (P3 walking)"
python -m spatial_episode.scriptgen.dataset_cli coverage-plan \
  --source-index "$SOURCES/source.index.json" \
  --output-root "$ROOT/outputs/p23_holdout_v1" \
  --collection-id p23_holdout_v1 \
  --capabilities "${WALKING[@]}" \
  --limit-bindings-per-capability 32 \
  --accepted-per-binding 2

echo "[$(date +%T)] stage 2 complete"
