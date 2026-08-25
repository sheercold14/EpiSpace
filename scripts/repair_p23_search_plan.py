#!/usr/bin/env python
"""Expand the CPU attempt frontier until every frozen producer has two plans."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from spatial_episode.scriptgen.binding_coverage import CoverageManifest, _new_candidates
from spatial_episode.scriptgen.standards import STD_V1


def _write(path: Path, manifest: CoverageManifest) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--attempt-limit", type=int, default=150)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = CoverageManifest.model_validate_json(args.manifest.read_text(encoding="utf-8"))
    scenes = {scene.scene_key: scene for scene in manifest.scenes}

    def producer(capability: str) -> bool:
        return capability == "reference_frame_transform" or capability.startswith(
            "cross_view_ego_k"
        )

    repaired = 0
    failures: collections.Counter[str] = collections.Counter()
    cells = list(manifest.cells)
    for index, cell in enumerate(cells):
        if not producer(cell.capability) or len(cell.candidates) >= cell.target_accepted:
            continue
        needed = cell.target_accepted - len(cell.candidates)
        search = _new_candidates(
            cell,
            scenes[cell.scene_key],
            manifest,
            requested=needed,
            attempt_limit=args.attempt_limit,
            std=STD_V1,
        )
        merged = (*cell.candidates, *search.candidates)
        counts = collections.Counter(cell.rejection_counts)
        counts.update(search.rejection_counts)
        cell = cell.model_copy(
            update={
                "candidates": merged,
                "geometry_pool_size": max(cell.geometry_pool_size, search.geometry_pool_size),
                "diverse_pool_size": max(cell.diverse_pool_size, search.diverse_pool_size),
                "search_attempt_limit": max(cell.search_attempt_limit, search.search_attempt_limit),
                "raw_plan_limit": max(cell.raw_plan_limit, search.raw_plan_limit),
                "search_pool_exhausted": search.search_pool_exhausted,
                "rejection_counts": dict(counts),
            }
        )
        cells[index] = cell
        manifest = manifest.model_copy(update={"cells": tuple(cells)})
        _write(args.manifest, manifest)
        if len(merged) >= cell.target_accepted:
            repaired += 1
        else:
            failures[cell.capability] += 1
        print(
            f"repair cell={cell.cell_id} candidates={len(merged)}/{cell.target_accepted}",
            flush=True,
        )

    report = {
        "schema_version": "p23_search_plan_repair.v1",
        "attempt_limit": args.attempt_limit,
        "repaired_cells": repaired,
        "unfilled_by_capability": dict(sorted(failures.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
