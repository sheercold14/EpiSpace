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
#
# The retry has to repeat the original all-scene invocation rather than name
# the failed scenes.  prepare-scenes compares the requested scene count against
# the count recorded in source.index.json and refuses a run that differs, so
# "--scenes Wainscott_0_garden" is rejected before it renders anything.  The
# full command is resumable - a scene whose bundle is still on disk is adopted
# untouched - so this costs one scene's render, not fifty-one.
FAILED=$(python - "$SOURCES/source.index.json" <<'PY'
import json, pathlib, sys
index = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(" ".join(sorted(s["scene_key"] for s in index["scenes"] if s["status"] != "ready")))
PY
)
if [ -n "$FAILED" ]; then
  echo "[$(date +%T)] retrying failed sources: $FAILED"
  python -m spatial_episode.scriptgen.dataset_cli prepare-scenes \
    --scene-set all \
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

python scripts/report_scene_concentration.py \
  --yield-report "$ROOT/outputs/p23_yield_all_v1.json" | tee "$ROOT/outputs/p23_concentration_v1.txt"

# Three collections, because the per-scene binding cap has to differ by line
# and coverage-plan takes one value per collection.  Shared credit runs inside
# the reference curve and inside each cross-view trio, never between them, so
# splitting on that boundary costs no coverage - including splitting the chain
# lines by k, as long as each ego/anchor/closer trio stays whole.
#
# The 50-scene screen decides the split.  P2 draws on 37 scenes with the top
# two holding 15% at a cap of 32, so it takes the wide cap.  The chains come
# from a handful of rooms and no cap can add rooms, so the cap is only there to
# stop one room dominating what few there are: at 8 the snapshot k1 and k2
# top-two shares sit at 33% and 43%, and above that k2 crosses half.
#
# k3 does not go into training at either cap.  Snapshot k3 exists in four
# scenes and walking k3 in three, and two of them hold two thirds of the supply
# no matter how it is capped, so a model trained on it learns those rooms.  It
# renders into the holdout collection with the walking line and is evaluated,
# not trained on.
P2=(reference_frame_transform reference_frame_transform_yaw45 reference_frame_transform_yaw90
    reference_frame_transform_yaw135 reference_frame_transform_yaw180
    reference_frame_transform_yawm45 reference_frame_transform_yawm90
    reference_frame_transform_yawm135
    reference_frame_visibility reference_frame_visibility_yaw45 reference_frame_visibility_yaw90
    reference_frame_visibility_yaw135 reference_frame_visibility_yaw180
    reference_frame_visibility_yawm45 reference_frame_visibility_yawm90
    reference_frame_visibility_yawm135)
SNAPSHOT=(cross_view_snapshot_ego_k1 cross_view_snapshot_ego_k2
          cross_view_snapshot_anchor_k1 cross_view_snapshot_anchor_k2
          cross_view_snapshot_closer_k1 cross_view_snapshot_closer_k2)
HOLDOUT=(cross_view_snapshot_ego_k3 cross_view_snapshot_anchor_k3 cross_view_snapshot_closer_k3
         cross_view_ego_k1 cross_view_ego_k2 cross_view_ego_k3
         cross_view_anchor_k1 cross_view_anchor_k2 cross_view_anchor_k3
         cross_view_closer_k1 cross_view_closer_k2 cross_view_closer_k3)

plan_collection() {
  local name=$1 limit=$2; shift 2
  echo "[$(date +%T)] planning $name (limit $limit)"
  python -m spatial_episode.scriptgen.dataset_cli coverage-plan \
    --source-index "$SOURCES/source.index.json" \
    --output-root "$ROOT/outputs/$name" \
    --collection-id "$name" \
    --capabilities "$@" \
    --limit-bindings-per-capability "$limit" \
    --accepted-per-binding 2
}

plan_collection p23_p2_v1      32 "${P2[@]}"
plan_collection p23_p3snap_v1   8 "${SNAPSHOT[@]}"
plan_collection p23_holdout_v1  8 "${HOLDOUT[@]}"

for name in p23_p2_v1 p23_p3snap_v1 p23_holdout_v1; do
  echo "[$(date +%T)] --- $name"
  python - "$ROOT/outputs/$name/coverage.plan.json" <<'PY'
import collections, json, pathlib, sys

plan = json.loads(pathlib.Path(sys.argv[1]).read_text())
cells = plan["cells"]
scenes = {cell["scene_key"] for cell in cells}
print(f"  {len(cells)} cells over {len(scenes)} scenes, "
      f"{plan['accepted_per_binding']} accepted per cell "
      f"=> {len(cells) * plan['accepted_per_binding']} trajectories at full quota")
by_capability = collections.Counter(cell["capability"] for cell in cells)
per_scene = collections.defaultdict(collections.Counter)
for cell in cells:
    per_scene[cell["capability"]][cell["scene_key"]] += 1
for capability, count in sorted(by_capability.items()):
    spread = per_scene[capability]
    top = sum(n for _, n in spread.most_common(2))
    print(f"    {capability:<40} {count:>4} cells  {len(spread):>3} scenes  "
          f"top2 {top / count:>5.1%}")
PY
done

echo "[$(date +%T)] stage 2 complete"
