#!/usr/bin/env bash
# CPU-only P2/P3 search. It never invokes coverage-run or a GPU renderer.
set -euo pipefail

ROOT=/data/shichao/data/dataV100/code/EpiSpace
SOURCES=$ROOT/outputs/p23_sources_all_v1/source.index.json
P2_NAME=p23_p2_7k_v1
P3_NAME=p23_p3stream_7k_v1
CONTRACT=$ROOT/outputs/p23_7k_search_contract.json
ALLOWLIST=$ROOT/outputs/p23_7k_binding_allowlist.json

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

mkdir -p "$ROOT/outputs/$P2_NAME" "$ROOT/outputs/$P3_NAME"

echo "[$(date +%T)] starting P2 search: target pool cap 10 x up to 128 bindings/scene"
python -m spatial_episode.scriptgen.dataset_cli coverage-plan \
  --source-index "$SOURCES" \
  --output-root "$ROOT/outputs/$P2_NAME" \
  --collection-id "$P2_NAME" \
  --seed-namespace p23_p2_badged_v2 \
  --capabilities "${P2[@]}" \
  --maximum-multislot-bindings 128 \
  --limit-bindings-per-capability 128 \
  --accepted-per-binding 10 \
  --attempts-per-binding 80 \
  --initial-attempts-per-binding 80 \
  --raw-plan-oversample 8 \
  >>"$ROOT/outputs/$P2_NAME/search.log" 2>&1 &
p2_pid=$!

echo "[$(date +%T)] starting walking-only P3 search: frozen 182/109/73 bindings"
python -m spatial_episode.scriptgen.dataset_cli coverage-plan \
  --source-index "$SOURCES" \
  --output-root "$ROOT/outputs/$P3_NAME" \
  --collection-id "$P3_NAME" \
  --seed-namespace p23_p3stream_badged_v1 \
  --capabilities "${P3[@]}" \
  --binding-allowlist "$ALLOWLIST" \
  --maximum-multislot-bindings 128 \
  --limit-bindings-per-capability 12 \
  --accepted-per-binding 12 \
  --attempts-per-binding 80 \
  --initial-attempts-per-binding 80 \
  --raw-plan-oversample 5 \
  >>"$ROOT/outputs/$P3_NAME/search.log" 2>&1 &
p3_pid=$!

p2_status=0
p3_status=0
wait "$p2_pid" || p2_status=$?
wait "$p3_pid" || p3_status=$?
if ((p2_status != 0 || p3_status != 0)); then
  echo "CPU search failed: P2=$p2_status P3=$p3_status" >&2
  echo "See outputs/$P2_NAME/search.log and outputs/$P3_NAME/search.log" >&2
  exit 1
fi

echo "[$(date +%T)] selecting the exact balanced 7,000-trajectory plan"
python scripts/finalize_p23_7k_search.py \
  --p2 "$ROOT/outputs/$P2_NAME/coverage.plan.json" \
  --p3 "$ROOT/outputs/$P3_NAME/coverage.plan.json" \
  --contract "$CONTRACT" \
  --output "$ROOT/outputs/p23_7k_search_finalization.json"

echo "[$(date +%T)] running the hard pre-render quality audit"
python scripts/audit_p23_7k_trajectory_quality.py \
  --p2 "$ROOT/outputs/$P2_NAME/coverage.plan.json" \
  --p3 "$ROOT/outputs/$P3_NAME/coverage.plan.json" \
  --contract "$CONTRACT" \
  --allowlist "$ALLOWLIST" \
  --output "$ROOT/outputs/p23_7k_trajectory_quality_audit.json"

python scripts/export_p23_7k_trajectory_index.py \
  --p2 "$ROOT/outputs/$P2_NAME/coverage.plan.json" \
  --p3 "$ROOT/outputs/$P3_NAME/coverage.plan.json" \
  --contract "$CONTRACT" \
  --audit "$ROOT/outputs/p23_7k_trajectory_quality_audit.json" \
  --output "$ROOT/outputs/p23_7k_trajectory_index.json"

python scripts/make_first_batch.py \
  --plan "$ROOT/outputs/$P2_NAME/coverage.plan.json" \
  --output "$ROOT/outputs/$P2_NAME/batch1.cells.json" --fraction 0.2
python scripts/make_first_batch.py \
  --plan "$ROOT/outputs/$P3_NAME/coverage.plan.json" \
  --output "$ROOT/outputs/$P3_NAME/batch1.cells.json" --fraction 0.2

echo "[$(date +%T)] PASS: CPU search plan complete; no render was started"
