#!/usr/bin/env python
"""Freeze quota-valid producer bindings with concrete search-stage trajectories.

Bindings with two candidates are preferred.  A later bounded repair pass is
responsible for bringing any retained one-candidate P3 producer up to two.
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
from pathlib import Path


def _canonical(binding: dict[str, str]) -> str:
    return json.dumps(binding, sort_keys=True, separators=(",", ":"))


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _round_robin(cells: list[dict], count: int) -> list[dict]:
    by_scene: dict[str, list[dict]] = collections.defaultdict(list)
    for cell in cells:
        by_scene[cell["scene_key"]].append(cell)
    for rows in by_scene.values():
        rows.sort(key=lambda cell: (-len(cell["candidates"]), cell["cell_id"]))
    selected: list[dict] = []
    depth = 0
    while len(selected) < count:
        added = False
        for scene_key in sorted(by_scene):
            rows = by_scene[scene_key]
            if depth < len(rows):
                selected.append(rows[depth])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        depth += 1
    return selected


def _round_robin_successes(cells: list[dict], count: int) -> list[dict]:
    """Prefer complete two-trajectory producers, preserving scene diversity."""
    complete = [cell for cell in cells if len(cell["candidates"]) >= 2]
    selected = _round_robin(complete, count)
    if len(selected) == count:
        return selected
    chosen_ids = {cell["cell_id"] for cell in selected}
    fallback = [cell for cell in cells if cell["cell_id"] not in chosen_ids]
    selected.extend(_round_robin(fallback, count - len(selected)))
    return selected


def _freeze_p2(manifest: dict, per_scene_cap: int) -> dict[str, object]:
    successful_producers = [
        cell
        for cell in manifest["cells"]
        if cell["capability"] == "reference_frame_transform" and len(cell["candidates"]) >= 2
    ]
    capabilities_by_key: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for cell in manifest["cells"]:
        capabilities_by_key[(cell["scene_key"], _canonical(cell["binding"]))].add(
            cell["capability"]
        )
    required_capabilities = set(manifest["capabilities"])
    producers = [
        cell
        for cell in successful_producers
        if capabilities_by_key[(cell["scene_key"], _canonical(cell["binding"]))]
        == required_capabilities
    ]
    selected: list[dict] = []
    by_scene: dict[str, list[dict]] = collections.defaultdict(list)
    for cell in producers:
        by_scene[cell["scene_key"]].append(cell)
    for scene_key in sorted(by_scene):
        selected.extend(
            sorted(by_scene[scene_key], key=lambda cell: cell["cell_id"])[:per_scene_cap]
        )
    keys = {(cell["scene_key"], _canonical(cell["binding"])) for cell in selected}
    before = len(manifest["cells"])
    manifest["cells"] = [
        cell
        for cell in manifest["cells"]
        if (cell["scene_key"], _canonical(cell["binding"])) in keys
    ]
    return {
        "producer_bindings": len(selected),
        "candidate_trajectories": sum(len(cell["candidates"]) for cell in selected),
        "incomplete_successes_excluded": len(successful_producers) - len(producers),
        "cells_before": before,
        "cells_after": len(manifest["cells"]),
        "scenes": len({cell["scene_key"] for cell in selected}),
    }


def _freeze_p3(manifest: dict, selection: dict) -> dict[str, object]:
    selected_keys: set[tuple[str, str]] = set()
    summary: dict[str, object] = {}
    capabilities_by_key: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for cell in manifest["cells"]:
        capabilities_by_key[(cell["scene_key"], _canonical(cell["binding"]))].add(
            cell["capability"]
        )
    for k in (1, 2, 3):
        capability = f"cross_view_ego_k{k}"
        required_capabilities = {
            f"cross_view_ego_k{k}",
            f"cross_view_anchor_k{k}",
            f"cross_view_closer_k{k}",
        }
        allowlisted = {
            (row["scene_key"], _canonical(row["binding"])) for row in selection["selected"][f"k{k}"]
        }
        allowlisted_successes = [
            cell
            for cell in manifest["cells"]
            if cell["capability"] == capability
            and (cell["scene_key"], _canonical(cell["binding"])) in allowlisted
            and len(cell["candidates"]) >= 1
        ]
        successes = [
            cell
            for cell in allowlisted_successes
            if capabilities_by_key[(cell["scene_key"], _canonical(cell["binding"]))]
            >= required_capabilities
        ]
        target = int(selection["target_bindings"][f"k{k}"])
        selected = _round_robin_successes(successes, target)
        if len(selected) != target:
            raise RuntimeError(
                f"k{k} has {len(successes)} bindings with a trajectory; needs {target}"
            )
        selected_keys.update((cell["scene_key"], _canonical(cell["binding"])) for cell in selected)
        summary[f"k{k}"] = {
            "searched_bindings": sum(
                cell["capability"] == capability
                and (cell["scene_key"], _canonical(cell["binding"])) in allowlisted
                for cell in manifest["cells"]
            ),
            "successful_bindings": len(successes),
            "incomplete_successes_excluded": len(allowlisted_successes) - len(successes),
            "two_trajectory_bindings": sum(len(cell["candidates"]) >= 2 for cell in successes),
            "retained_bindings": len(selected),
            "retained_trajectories": sum(len(cell["candidates"]) for cell in selected),
            "scenes": len({cell["scene_key"] for cell in selected}),
        }
    before = len(manifest["cells"])
    manifest["cells"] = [
        cell
        for cell in manifest["cells"]
        if (cell["scene_key"], _canonical(cell["binding"])) in selected_keys
    ]
    summary["cells_before"] = before
    summary["cells_after"] = len(manifest["cells"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2", type=Path, required=True)
    parser.add_argument("--p3", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--p2-per-scene-cap", type=int, default=8)
    args = parser.parse_args()

    p2, p3, selection = _read(args.p2), _read(args.p3), _read(args.selection)
    p2_summary = _freeze_p2(p2, args.p2_per_scene_cap)
    p3_summary = _freeze_p3(p3, selection)
    if p2_summary["producer_bindings"] == 0:
        raise RuntimeError("P2 has no binding with two search-stage trajectories")

    for path, manifest in ((args.p2, p2), (args.p3, p3)):
        backup = path.with_name("coverage.search_pool.json")
        if not backup.exists():
            shutil.copy2(path, backup)
        _write(path, manifest)

    report = {
        "schema_version": "p23_search_plan_finalization.v1",
        "p2": p2_summary,
        "p3": p3_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
