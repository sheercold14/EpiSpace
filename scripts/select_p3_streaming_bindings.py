#!/usr/bin/env python
"""Select an exact 50/30/20 streaming-P3 binding mix before trajectory search.

The expensive planner should only search bindings that can enter tomorrow's
render.  This CPU prepass uses the same marker pool, source-render co-visibility
graph and ranked binding code as ``coverage-plan``, caps each scene, then draws
round-robin across scenes.  The resulting allowlist is immutable input to the
coverage manifest and contains no snapshot capability.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

from spatial_episode.scriptgen.behavior import layout_from_scene_ir
from spatial_episode.scriptgen.binding_coverage import _source_scene
from spatial_episode.scriptgen.collection import (
    _chain_binding_filter,
    _scene_can_bind,
    _source_render_evidence,
    ranked_bindings,
)
from spatial_episode.scriptgen.library import SCRIPT_LIBRARY
from spatial_episode.scriptgen.source_inventory import load_source_index
from spatial_episode.scriptgen.standards import STD_V1

TARGETS = {1: 182, 2: 109, 3: 73}  # two renders each -> 364/218/146 = 50/30/20


def _screen_scene(item):
    scene, maximum, per_scene_cap = item
    layout = layout_from_scene_ir(scene.scene_ir, std=STD_V1)
    evidence = _source_render_evidence(scene, STD_V1)
    result: dict[int, tuple[dict[str, str], ...]] = {}
    for k in TARGETS:
        script = SCRIPT_LIBRARY[f"cross_view_ego_k{k}"]
        if not _scene_can_bind(layout, script):
            result[k] = ()
            continue
        bindings = ranked_bindings(
            layout,
            script,
            maximum=maximum,
            allowed_entity_ids=evidence.visible_entities,
            binding_filter=_chain_binding_filter(layout, script, evidence),
            chain_adjacency=evidence.covisible_pairs,
            std=STD_V1,
        )
        result[k] = bindings[:per_scene_cap]
    return scene.scene_key, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-scene-cap", type=int, default=8)
    parser.add_argument("--maximum-multislot-bindings", type=int, default=128)
    parser.add_argument(
        "--search-buffer-factor",
        type=float,
        default=1.35,
        help="plan extra bindings, then retain exact quotas with two trajectories",
    )
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.per_scene_cap <= 0:
        raise ValueError("--per-scene-cap must be positive")

    index = load_source_index(args.source_index.resolve())
    if index.ready_scene_count != index.requested_scene_count:
        raise ValueError("source inventory is incomplete")
    scenes = [_source_scene(record) for record in index.scenes if record.status == "ready"]
    pools: dict[int, dict[str, tuple[dict[str, str], ...]]] = {k: {} for k in TARGETS}
    work = [(scene, args.maximum_multislot_bindings, args.per_scene_cap) for scene in scenes]
    with mp.Pool(args.workers) as workers:
        for scene_key, result in workers.imap_unordered(_screen_scene, work):
            for k in TARGETS:
                pools[k][scene_key] = result[k]
            counts = " ".join(f"k{k}={len(result[k])}" for k in TARGETS)
            print(f"streaming binding screen scene={scene_key} {counts}", flush=True)

    bindings_by_scene: dict[str, list[dict[str, str]]] = defaultdict(list)
    selected_rows: dict[str, list[dict[str, object]]] = {}
    availability: dict[str, int] = {}
    search_targets = {
        k: math.ceil(target * args.search_buffer_factor) for k, target in TARGETS.items()
    }
    for k, target in search_targets.items():
        pool = pools[k]
        available = sum(len(bindings) for bindings in pool.values())
        availability[f"k{k}"] = available
        if available < target:
            raise RuntimeError(
                f"streaming k{k} has {available} bindings after scene cap, needs {target}"
            )
        ordered: list[tuple[str, dict[str, str]]] = []
        max_depth = max((len(bindings) for bindings in pool.values()), default=0)
        for rank in range(max_depth):
            for scene_key in sorted(pool):
                if rank < len(pool[scene_key]):
                    ordered.append((scene_key, pool[scene_key][rank]))
        chosen = ordered[:target]
        selected_rows[f"k{k}"] = [
            {"scene_key": scene_key, "binding": binding} for scene_key, binding in chosen
        ]
        for scene_key, binding in chosen:
            bindings_by_scene[scene_key].append(binding)

    payload = {
        "schema_version": "p23_streaming_binding_selection.v1",
        "standard_version": STD_V1.standard_version,
        "source_index": str(args.source_index.resolve()),
        "per_scene_cap": args.per_scene_cap,
        "maximum_multislot_bindings": args.maximum_multislot_bindings,
        "target_bindings": {f"k{k}": target for k, target in TARGETS.items()},
        "target_trajectories": {f"k{k}": target * 2 for k, target in TARGETS.items()},
        "search_bindings": {f"k{k}": target for k, target in search_targets.items()},
        "available_bindings": availability,
        "bindings_by_scene": dict(sorted(bindings_by_scene.items())),
        "selected": selected_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "selected "
        + " ".join(f"k{k}={search_targets[k]} (retain {TARGETS[k]})" for k in TARGETS)
        + f" -> {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
