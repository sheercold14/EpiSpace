#!/usr/bin/env bash
# Renders tonight's episodes for the three planned collections.
#
# Batch one comes first and is drawn across every supplying scene rather than
# taken from the head of the plan, so the morning's review sample represents
# the whole collection instead of the handful of rooms the plan happens to
# enumerate first.  The review sets are cut as soon as batch one lands, so the
# material the gate needs exists no matter how far the rest of the night gets.
#
# Only then does the remainder run.  Rendering ahead of the review costs
# nothing that was not already idle GPU time - nothing reaches a training set
# until the review passes and the balance gate has dropped the skewed strata -
# but if the review fails, the batch-one boundary is where the diagnosis
# starts.
#
# Every stage is resumable: coverage-run adopts episodes already on disk, so
# re-running this script after an interruption continues rather than restarts.
set -uo pipefail

ROOT=/data/shichao/data/dataV100/code/EpiSpace
OG_ROOT=/data/shichao/data/dataV100/code/OminiGibson
DATA_ROOT=$OG_ROOT/.data/omnigibson
COLLECTIONS=(p23_p2_v1 p23_p3snap_v1 p23_holdout_v1)

cd "$ROOT"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate behavior-spatialep
export PYTHONPATH=src

run_coverage() {
  local name=$1; shift
  python -m spatial_episode.scriptgen.dataset_cli coverage-run \
    --manifest "$ROOT/outputs/$name/coverage.plan.json" \
    --og-root "$OG_ROOT" \
    --conda-env behavior-spatialep \
    --data-root "$DATA_ROOT" \
    --gpu-ids 0 1 2 3 --workers 4 \
    --timeout-minutes 30 \
    "$@"
}

for name in "${COLLECTIONS[@]}"; do
  echo "[$(date +%T)] batch 1: $name"
  run_coverage "$name" --cell-ids-file "$ROOT/outputs/$name/batch1.cells.json" ||
    echo "[$(date +%T)] batch 1 $name returned nonzero, continuing"
done

# Cut the review sets before the remainder starts competing for the GPUs, so a
# long tail of repair searches cannot delay the one artefact the gate needs.
for name in "${COLLECTIONS[@]}"; do
  echo "[$(date +%T)] review set: $name"
  # These collections put groups/ at the collection root; only the pilot had
  # the extra output/ level that make_pilot_review's usage example shows.
  python scripts/make_pilot_review.py \
    --output-root "$ROOT/outputs/$name" --count 20 ||
    echo "[$(date +%T)] review set $name unavailable"
done

for name in "${COLLECTIONS[@]}"; do
  echo "[$(date +%T)] remainder: $name"
  run_coverage "$name" || echo "[$(date +%T)] remainder $name returned nonzero, continuing"
done

echo "[$(date +%T)] render night complete"
