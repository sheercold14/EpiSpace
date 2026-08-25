#!/usr/bin/env bash
# Tomorrow's renderer: only the frozen marker-enabled P2 and walking P3 plans.
set -euo pipefail

ROOT=/data/shichao/data/dataV100/code/EpiSpace
OG_ROOT=/data/shichao/data/dataV100/code/OminiGibson
DATA_ROOT=$OG_ROOT/.data/omnigibson
COLLECTIONS=(p23_p2_badged_v2 p23_p3stream_badged_v1)

cd "$ROOT"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate behavior-spatialep
export PYTHONPATH=src

# Refuse GPU work if either manifest changed after tonight's freeze.
python scripts/audit_p23_search_plan.py \
  --p2 "$ROOT/outputs/p23_p2_badged_v2/coverage.plan.json" \
  --p3 "$ROOT/outputs/p23_p3stream_badged_v1/coverage.plan.json" \
  --selection "$ROOT/outputs/p23_p3stream_badged_v1/binding.selection.json" \
  --output "$ROOT/outputs/p23_search_plan_audit.json"

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

for name in "${COLLECTIONS[@]}"; do
  echo "[$(date +%T)] review set: $name"
  python scripts/make_pilot_review.py \
    --output-root "$ROOT/outputs/$name" --count 20 ||
    echo "[$(date +%T)] review set $name unavailable"
done

for name in "${COLLECTIONS[@]}"; do
  echo "[$(date +%T)] remainder: $name"
  run_coverage "$name" || echo "[$(date +%T)] remainder $name returned nonzero, continuing"
done

echo "[$(date +%T)] P2 + streaming P3 render complete"
