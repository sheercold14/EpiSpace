#!/usr/bin/env python
"""Audit the final P2/P3 pre-render manifests and summarize actual trajectories."""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

from spatial_episode.scriptgen.plan import TrajectoryPlan
from spatial_episode.scriptgen.standards import STD_V1

P2_CAPABILITIES = (
    "reference_frame_transform",
    "reference_frame_transform_yaw45",
    "reference_frame_transform_yaw90",
    "reference_frame_transform_yaw135",
    "reference_frame_transform_yaw180",
    "reference_frame_transform_yawm45",
    "reference_frame_transform_yawm90",
    "reference_frame_transform_yawm135",
    "reference_frame_visibility",
    "reference_frame_visibility_yaw45",
    "reference_frame_visibility_yaw90",
    "reference_frame_visibility_yaw135",
    "reference_frame_visibility_yaw180",
    "reference_frame_visibility_yawm45",
    "reference_frame_visibility_yawm90",
    "reference_frame_visibility_yawm135",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _k(capability: str) -> int | None:
    if "_k" not in capability:
        return None
    return int(capability.rsplit("_k", 1)[1])


def _canonical(binding: dict[str, str]) -> str:
    return json.dumps(binding, sort_keys=True, separators=(",", ":"))


def _binding_keys(manifest: dict, capability: str) -> set[tuple[str, str]]:
    return {
        (cell["scene_key"], _canonical(cell["binding"]))
        for cell in manifest["cells"]
        if cell["capability"] == capability
    }


def _plan_stats(manifest: dict) -> dict:
    producer_cells = [
        cell
        for cell in manifest["cells"]
        if cell["capability"] == "reference_frame_transform"
        or cell["capability"].startswith("cross_view_ego_k")
    ]
    labels: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    movement_violations: list[str] = []
    candidate_count = 0
    trajectories_by_capability: collections.Counter[str] = collections.Counter()
    frame_counts: collections.Counter[int] = collections.Counter()
    path_distances: list[float] = []
    path_turns: list[float] = []
    empty_cells: list[str] = []
    underfilled_cells: list[str] = []
    for cell in producer_cells:
        if not cell["candidates"]:
            empty_cells.append(cell["cell_id"])
        if len(cell["candidates"]) < cell["target_accepted"]:
            underfilled_cells.append(cell["cell_id"])
        for candidate in cell["candidates"]:
            candidate_count += 1
            trajectories_by_capability[cell["capability"]] += 1
            plan = TrajectoryPlan.model_validate_json(
                Path(candidate["plan_record"]).read_text(encoding="utf-8")
            )
            frame_counts[len(plan.poses)] += 1
            labels[cell["capability"]][plan.provisional_answer.label] += 1
            distance = 0.0
            turn = 0.0
            for left, right in zip(plan.poses, plan.poses[1:], strict=False):
                translation = math.hypot(right.x - left.x, right.y - left.y)
                yaw = abs(((right.yaw_deg - left.yaw_deg + 180.0) % 360.0) - 180.0)
                distance += translation
                turn += yaw
                if cell["capability"].startswith("cross_view_ego_k") and (
                    translation > STD_V1.max_step_translation_m + 1e-6
                    or yaw > STD_V1.max_step_turn_deg + 1e-6
                ):
                    movement_violations.append(plan.plan_id)
            path_distances.append(distance)
            path_turns.append(turn)

    def numeric_summary(values: list[float]) -> dict[str, float]:
        if not values:
            return {"minimum": 0.0, "mean": 0.0, "maximum": 0.0}
        return {
            "minimum": round(min(values), 3),
            "mean": round(sum(values) / len(values), 3),
            "maximum": round(max(values), 3),
        }

    return {
        "cells": len(manifest["cells"]),
        "producer_cells": len(producer_cells),
        "producer_scenes": sorted({cell["scene_key"] for cell in producer_cells}),
        "candidate_trajectories": candidate_count,
        "trajectories_by_capability": dict(sorted(trajectories_by_capability.items())),
        "frame_count_distribution": {
            str(frame_count): count for frame_count, count in sorted(frame_counts.items())
        },
        "path_distance_m": numeric_summary(path_distances),
        "path_turn_degrees": numeric_summary(path_turns),
        "empty_producer_cells": empty_cells,
        "underfilled_producer_cells": underfilled_cells,
        "provisional_labels": {
            capability: dict(sorted(counts.items()))
            for capability, counts in sorted(labels.items())
        },
        "movement_violations": sorted(set(movement_violations)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2", type=Path, required=True)
    parser.add_argument("--p3", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    p2, p3, selection = _load(args.p2), _load(args.p3), _load(args.selection)
    p2_capabilities = tuple(p2["capabilities"])
    p2_unexpected = sorted(set(p2_capabilities) - set(P2_CAPABILITIES))
    p2_missing = sorted(set(P2_CAPABILITIES) - set(p2_capabilities))
    p2_binding_sets = {capability: _binding_keys(p2, capability) for capability in P2_CAPABILITIES}
    p2_producer_bindings = p2_binding_sets["reference_frame_transform"]
    p2_shared_bindings_ok = all(
        bindings == p2_producer_bindings for bindings in p2_binding_sets.values()
    )
    capabilities = tuple(p3["capabilities"])
    snapshot = sorted(capability for capability in capabilities if "snapshot" in capability)
    nonwalking = sorted(
        capability
        for capability in capabilities
        if not capability.startswith(
            ("cross_view_ego_k", "cross_view_anchor_k", "cross_view_closer_k")
        )
    )
    by_k: dict[str, dict[str, int]] = {}
    for k in (1, 2, 3):
        cap_counts = {
            mode: sum(cell["capability"] == f"cross_view_{mode}_k{k}" for cell in p3["cells"])
            for mode in ("ego", "anchor", "closer")
        }
        by_k[f"k{k}"] = cap_counts
    p3_shared_bindings_ok = all(
        _binding_keys(p3, f"cross_view_{mode}_k{k}") == _binding_keys(p3, f"cross_view_ego_k{k}")
        for k in (1, 2, 3)
        for mode in ("anchor", "closer")
    )
    expected = selection["target_bindings"]
    allowlisted = {
        k: {
            (row["scene_key"], _canonical(row["binding"])) for row in selection["selected"][f"k{k}"]
        }
        for k in (1, 2, 3)
    }
    outside_allowlist = sorted(
        cell["cell_id"]
        for cell in p3["cells"]
        if cell["capability"].startswith("cross_view_ego_k")
        and (
            cell["scene_key"],
            _canonical(cell["binding"]),
        )
        not in allowlisted[_k(cell["capability"])]
    )
    quota_ok = all(set(by_k[f"k{k}"].values()) == {expected[f"k{k}"]} for k in (1, 2, 3))
    p2_stats, p3_stats = _plan_stats(p2), _plan_stats(p3)
    status = "pass"
    errors: list[str] = []
    if snapshot:
        errors.append("snapshot capability present")
    if nonwalking:
        errors.append("non-walking capability present")
    if p2_unexpected or p2_missing:
        errors.append("P2 capability contract mismatch")
    if not p2_shared_bindings_ok:
        errors.append("P2 question cells do not share the producer binding set")
    if not quota_ok:
        errors.append("P3 50/30/20 binding quota mismatch")
    if not p3_shared_bindings_ok:
        errors.append("P3 question cells do not share the producer binding set")
    if outside_allowlist:
        errors.append("P3 binding is outside the frozen search allowlist")
    if p2_stats["movement_violations"] or p3_stats["movement_violations"]:
        errors.append("step-motion bound violation")
    if p2_stats["underfilled_producer_cells"] or p3_stats["underfilled_producer_cells"]:
        errors.append("producer cell is below its search-stage trajectory quota")
    if errors:
        status = "fail"
    report = {
        "schema_version": "p23_search_plan_audit.v1",
        "status": status,
        "standard_version": STD_V1.standard_version,
        "errors": errors,
        "search_policy": {
            "marker_object_candidates_per_slot": 18,
            "marker_object_maximum_size_m": 4.0,
            "p2_viewpoint_facing_pair_frontier": 16,
            "p3_path_beam": 8192,
            "p3_admitted_binding_early_stop": 256,
            "p3_binding_constraints": [
                "source_render_covisibility_graph",
                "distinct_landmark_footprints",
                "motif_edge_station_exists",
            ],
        },
        "p2_contract": {
            "unexpected_capabilities": p2_unexpected,
            "missing_capabilities": p2_missing,
            "bindings_by_question": {
                capability: len(bindings) for capability, bindings in p2_binding_sets.items()
            },
            "shared_bindings_ok": p2_shared_bindings_ok,
        },
        "p3_contract": {
            "snapshot_capabilities": snapshot,
            "nonwalking_capabilities": nonwalking,
            "bindings_by_k_and_question": by_k,
            "quota_ok": quota_ok,
            "shared_bindings_ok": p3_shared_bindings_ok,
            "outside_allowlist_cells": outside_allowlist,
            "target_trajectory_ratio": {"k1": 50, "k2": 30, "k3": 20},
        },
        "p2": p2_stats,
        "p3": p3_stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
