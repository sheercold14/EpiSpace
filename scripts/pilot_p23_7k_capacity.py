#!/usr/bin/env python
"""Measure quality-gated trajectory yield before scaling P2/P3 to 7k.

The pilot never writes render candidates.  It replays a scene-balanced sample
of known-feasible bindings through the production geometry generator, applies
the same pairwise structural-diversity gate as coverage planning, and reports
how many genuinely different pose sequences survive a bounded attempt budget.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
from typing import Any

from spatial_episode.scriptgen.behavior import layout_from_scene_ir
from spatial_episode.scriptgen.binding_coverage import (
    CoverageManifest,
    _diverse_sequence,
    _source_scene,
)
from spatial_episode.scriptgen.generate import generate_plans
from spatial_episode.scriptgen.library import SCRIPT_LIBRARY
from spatial_episode.scriptgen.source_inventory import load_source_index
from spatial_episode.scriptgen.standards import STD_V1


def _pose_fingerprint(plan: Any) -> str:
    poses = [(round(pose.x, 4), round(pose.y, 4), round(pose.yaw_deg, 3)) for pose in plan.poses]
    payload = json.dumps(poses, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _path_distance(plan: Any) -> float:
    return sum(
        math.hypot(right.x - left.x, right.y - left.y)
        for left, right in zip(plan.poses, plan.poses[1:], strict=False)
    )


def _worker(job: tuple[Any, str, dict[str, str], int, int]) -> dict[str, Any]:
    scene, capability, binding, seed, attempts = job
    layout = layout_from_scene_ir(scene.scene_ir, std=STD_V1)
    report = generate_plans(
        layout,
        SCRIPT_LIBRARY[capability],
        STD_V1,
        seed=seed,
        attempts_per_binding=attempts,
        plans_per_binding=attempts,
        candidate_bindings=(binding,),
    )
    diverse = _diverse_sequence(report.plans)
    return {
        "scene_key": scene.scene_key,
        "capability": capability,
        "binding": binding,
        "seed": seed,
        "attempts": attempts,
        "raw_plans": len(report.plans),
        "diverse_plans": len(diverse),
        "pose_fingerprints": [_pose_fingerprint(plan) for plan in diverse],
        "frame_counts": [len(plan.poses) for plan in diverse],
        "path_distances_m": [round(_path_distance(plan), 3) for plan in diverse],
        "rejection_counts": dict(sorted(report.rejection_counts.items())),
    }


def _producer(cell: Any) -> bool:
    return cell.capability == "reference_frame_transform" or cell.capability.startswith(
        "cross_view_ego_k"
    )


def _balanced_sample(cells: list[Any], count: int) -> list[Any]:
    by_scene: dict[str, list[Any]] = collections.defaultdict(list)
    for cell in cells:
        by_scene[cell.scene_key].append(cell)
    for rows in by_scene.values():
        rows.sort(key=lambda cell: cell.cell_id)
    selected: list[Any] = []
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


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    yields = [int(row["diverse_plans"]) for row in rows]
    raw = [int(row["raw_plans"]) for row in rows]
    fingerprints = [item for row in rows for item in row["pose_fingerprints"]]
    return {
        "bindings": len(rows),
        "scenes": len({row["scene_key"] for row in rows}),
        "attempts_per_binding": rows[0]["attempts"] if rows else 0,
        "raw_plans": sum(raw),
        "diverse_plans": sum(yields),
        "globally_unique_pose_sequences": len(set(fingerprints)),
        "bindings_with_zero_diverse_plans": sum(value == 0 for value in yields),
        "diverse_yield": {
            "minimum": min(yields, default=0),
            "p25": round(_percentile(yields, 0.25), 2),
            "median": round(_percentile(yields, 0.5), 2),
            "p75": round(_percentile(yields, 0.75), 2),
            "maximum": max(yields, default=0),
            "mean": round(sum(yields) / len(yields), 2) if yields else 0.0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--p2", type=Path, required=True)
    parser.add_argument("--p3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--p2-bindings", type=int, default=24)
    parser.add_argument("--p3-bindings-per-k", type=int, default=12)
    parser.add_argument("--attempts-per-binding", type=int, default=80)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    source_index = load_source_index(args.source_index.resolve())
    scenes = {
        record.scene_key: _source_scene(record)
        for record in source_index.scenes
        if record.status == "ready"
    }
    p2 = CoverageManifest.model_validate_json(args.p2.read_text(encoding="utf-8"))
    p3 = CoverageManifest.model_validate_json(args.p3.read_text(encoding="utf-8"))
    groups: dict[str, list[Any]] = {
        "p2": _balanced_sample(
            [cell for cell in p2.cells if _producer(cell) and cell.candidates],
            args.p2_bindings,
        )
    }
    for k in (1, 2, 3):
        capability = f"cross_view_ego_k{k}"
        groups[f"p3_k{k}"] = _balanced_sample(
            [cell for cell in p3.cells if cell.capability == capability and cell.candidates],
            args.p3_bindings_per_k,
        )

    jobs: list[tuple[Any, str, dict[str, str], int, int]] = []
    group_by_job: list[str] = []
    for group, cells in groups.items():
        for cell in cells:
            jobs.append(
                (
                    scenes[cell.scene_key],
                    cell.capability,
                    dict(cell.binding),
                    cell.seed,
                    args.attempts_per_binding,
                )
            )
            group_by_job.append(group)

    results_by_group: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    with mp.Pool(args.workers) as workers:
        for group, row in zip(
            group_by_job,
            workers.imap(_worker, jobs),
            strict=True,
        ):
            results_by_group[group].append(row)
            print(
                f"capacity group={group} scene={row['scene_key']} "
                f"raw={row['raw_plans']} diverse={row['diverse_plans']}",
                flush=True,
            )

    all_rows = [row for rows in results_by_group.values() for row in rows]
    all_fingerprints = [fingerprint for row in all_rows for fingerprint in row["pose_fingerprints"]]
    payload = {
        "schema_version": "p23_7k_capacity_pilot.v1",
        "standard_version": STD_V1.standard_version,
        "quality_contract": {
            "pairwise_structural_diversity": (
                "start/end>=0.5m OR mean_path>=0.30m OR mean_yaw>=15deg OR "
                "net_turn>=30deg OR key_frame_delta>=2 OR blocker differs"
            ),
            "global_exact_pose_duplicates_allowed": 0,
            "p3_motion": "walking_only",
        },
        "groups": {
            group: {**_summary(rows), "rows": rows}
            for group, rows in sorted(results_by_group.items())
        },
        "overall": {
            "sampled_bindings": len(all_rows),
            "diverse_plans": len(all_fingerprints),
            "globally_unique_pose_sequences": len(set(all_fingerprints)),
            "exact_pose_duplicates": len(all_fingerprints) - len(set(all_fingerprints)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    concise_groups = {
        group: {key: value for key, value in summary.items() if key != "rows"}
        for group, summary in payload["groups"].items()
    }
    print(json.dumps({"groups": concise_groups, "overall": payload["overall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
