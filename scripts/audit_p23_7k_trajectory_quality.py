#!/usr/bin/env python
"""Hard pre-render audit for the finalized 7,000-trajectory P2/P3 plan."""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

from spatial_episode.scriptgen.behavior import layout_from_scene_ir
from spatial_episode.scriptgen.binding_coverage import trajectories_are_diverse
from spatial_episode.scriptgen.collection import _reference_binding_eligible
from spatial_episode.scriptgen.library import SCRIPT_LIBRARY
from spatial_episode.scriptgen.markers import script_uses_markers
from spatial_episode.scriptgen.motifs import imagined_station_placement
from spatial_episode.scriptgen.plan import TrajectoryPlan
from spatial_episode.scriptgen.standards import STD_V1


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _canonical(binding: dict[str, str]) -> str:
    return json.dumps(binding, sort_keys=True, separators=(",", ":"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _key(cell: dict[str, Any]) -> tuple[str, str]:
    return cell["scene_key"], _canonical(cell["binding"])


def _pose_fingerprint(plan: TrajectoryPlan) -> str:
    poses = [
        (pose.frame, round(pose.x, 6), round(pose.y, 6), round(pose.yaw_deg, 6))
        for pose in plan.poses
    ]
    return hashlib.sha256(json.dumps(poses, separators=(",", ":")).encode("utf-8")).hexdigest()


def _ratio_targets(total: int, weights: dict[str, int]) -> dict[str, int]:
    denominator = sum(weights.values())
    floors = {key: total * weight // denominator for key, weight in weights.items()}
    remainder = total - sum(floors.values())
    priority = sorted(
        weights,
        key=lambda key: (-(total * weights[key] % denominator), key),
    )
    for key in priority[:remainder]:
        floors[key] += 1
    return floors


def _binding_set(manifest: dict[str, Any], capability: str) -> set[tuple[str, str]]:
    return {_key(cell) for cell in manifest["cells"] if cell["capability"] == capability}


def _producer_cells(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        cell
        for cell in manifest["cells"]
        if cell["capability"] == "reference_frame_transform"
        or cell["capability"].startswith("cross_view_ego_k")
    ]


def _question_sharing_errors(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    producer_by_key = {_key(cell): cell for cell in _producer_cells(manifest)}
    for cell in manifest["cells"]:
        producer = producer_by_key.get(_key(cell))
        if producer is None:
            errors.append(f"question_without_producer:{cell['cell_id']}")
            continue
        if cell["target_accepted"] != len(producer["candidates"]):
            errors.append(f"shared_target_mismatch:{cell['cell_id']}")
        if cell["cell_id"] != producer["cell_id"] and cell["candidates"]:
            errors.append(f"deferred_question_has_private_candidate:{cell['cell_id']}")
    return errors


def _motion_violations(plan: TrajectoryPlan) -> list[str]:
    errors: list[str] = []
    for index, (left, right) in enumerate(zip(plan.poses, plan.poses[1:], strict=False)):
        translation = math.hypot(right.x - left.x, right.y - left.y)
        yaw = abs(((right.yaw_deg - left.yaw_deg + 180.0) % 360.0) - 180.0)
        if translation > STD_V1.max_step_translation_m + 1e-6:
            errors.append(f"{plan.plan_id}:step{index}:translation={translation:.6f}")
        if yaw > STD_V1.max_step_turn_deg + 1e-6:
            errors.append(f"{plan.plan_id}:step{index}:yaw={yaw:.6f}")
    return errors


def _p2_camera_station_violations(manifest: dict[str, Any]) -> list[str]:
    """Recompute every selected P2 imagined station under current geometry."""
    scene_ir = {row["scene_key"]: row["scene_ir"] for row in manifest["scenes"]}
    layouts: dict[str, Any] = {}
    errors: list[str] = []
    for cell in _producer_cells(manifest):
        if cell["capability"] != "reference_frame_transform":
            continue
        scene_key = cell["scene_key"]
        if scene_key not in layouts:
            layouts[scene_key] = layout_from_scene_ir(
                Path(scene_ir[scene_key]), std=STD_V1
            )
        layout = layouts[scene_key]
        binding = cell["binding"]
        placement = imagined_station_placement(
            layout,
            binding["viewpoint"],
            binding["facing"],
            STD_V1.camera_height_m,
        )
        if placement is None or not _reference_binding_eligible(layout, binding, STD_V1):
            errors.append(f"binding_ineligible:{cell['cell_id']}")
            continue
        for candidate in cell["candidates"]:
            payload = _read(Path(candidate["render_plan"]))
            auxiliary = payload.get("auxiliary_views") or []
            if len(auxiliary) != len(STD_V1.imagined_viewpoint_offsets_deg):
                errors.append(f"auxiliary_count:{candidate['candidate_id']}")
                continue
            mismatch = False
            for view in auxiliary:
                position = view.get("world_from_agent", {}).get("translation_m", ())
                mismatch = mismatch or bool(
                    len(position) < 2
                    or view.get("station_method") != placement.method
                    or abs(float(position[0]) - placement.xy[0]) > 1e-8
                    or abs(float(position[1]) - placement.xy[1]) > 1e-8
                    or abs(float(view.get("station_offset_m", math.inf)) - placement.offset_m)
                    > 1e-8
                    or abs(
                        float(view.get("station_heading_shift_deg", math.inf))
                        - placement.heading_shift_deg
                    )
                    > 1e-8
                    or abs(
                        float(view.get("station_surface_standoff_m", math.inf))
                        - placement.surface_standoff_m
                    )
                    > 1e-8
                )
            if mismatch:
                errors.append(f"station_mismatch:{candidate['candidate_id']}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2", type=Path, required=True)
    parser.add_argument("--p3", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    p2, p3 = _read(args.p2), _read(args.p3)
    contract, allowlist = _read(args.contract), _read(args.allowlist)
    allocation_policy = contract["allocation_policy"]
    quality = contract["quality"]
    errors: list[str] = []

    p2_producer = _producer_cells(p2)
    p3_producer = _producer_cells(p3)
    p2_count = sum(len(cell["candidates"]) for cell in p2_producer)
    p3_counts = {
        f"k{k}": sum(
            len(cell["candidates"])
            for cell in p3_producer
            if cell["capability"] == f"cross_view_ego_k{k}"
        )
        for k in (1, 2, 3)
    }
    if (
        not int(allocation_policy["p2_minimum"])
        <= p2_count
        <= int(allocation_policy["p2_preferred"])
    ):
        errors.append(f"p2_trajectory_count_outside_policy:{p2_count}")
    expected_p3_counts = _ratio_targets(
        int(contract["physical_trajectory_target"]) - p2_count,
        {key: int(value) for key, value in allocation_policy["p3_k_ratio"].items()},
    )
    for group, count in p3_counts.items():
        if count != expected_p3_counts[group]:
            errors.append(f"p3_{group}_trajectory_count:{count}!={expected_p3_counts[group]}")

    p2_bindings = len(p2_producer)
    if p2_bindings < math.ceil(p2_count / int(quality["p2_max_trajectories_per_binding"])):
        errors.append(f"p2_binding_count:{p2_bindings}")
    if any(
        len(cell["candidates"]) > int(quality["p2_max_trajectories_per_binding"])
        for cell in p2_producer
    ):
        errors.append("p2_per_binding_cap_exceeded")
    p3_bindings: dict[str, int] = {}
    for k in (1, 2, 3):
        group = f"k{k}"
        cells = [cell for cell in p3_producer if cell["capability"] == f"cross_view_ego_k{k}"]
        p3_bindings[group] = len(cells)
        if len(cells) < math.ceil(
            p3_counts[group] / int(quality["p3_max_trajectories_per_binding"])
        ):
            errors.append(f"p3_{group}_binding_count:{len(cells)}")
        if any(
            len(cell["candidates"]) > int(quality["p3_max_trajectories_per_binding"])
            for cell in cells
        ):
            errors.append(f"p3_{group}_per_binding_cap_exceeded")

    expected_p2 = _binding_set(p2, "reference_frame_transform")
    for capability in p2["capabilities"]:
        if _binding_set(p2, capability) != expected_p2:
            errors.append(f"p2_question_binding_mismatch:{capability}")
    for k in (1, 2, 3):
        expected_p3 = _binding_set(p3, f"cross_view_ego_k{k}")
        for mode in ("anchor", "closer"):
            capability = f"cross_view_{mode}_k{k}"
            if _binding_set(p3, capability) != expected_p3:
                errors.append(f"p3_question_binding_mismatch:{capability}")

    allowed = {
        (row["scene_key"], _canonical(row["binding"]))
        for k in (1, 2, 3)
        for row in allowlist["selected"][f"k{k}"]
    }
    outside_allowlist = sorted(cell["cell_id"] for cell in p3_producer if _key(cell) not in allowed)
    if outside_allowlist:
        errors.append(f"p3_outside_allowlist:{len(outside_allowlist)}")

    snapshot_or_nonwalking = sorted(
        capability
        for capability in p3["capabilities"]
        if "snapshot" in capability
        or not capability.startswith(
            ("cross_view_ego_k", "cross_view_anchor_k", "cross_view_closer_k")
        )
    )
    if snapshot_or_nonwalking:
        errors.append("p3_nonwalking_capability_present")

    errors.extend(_question_sharing_errors(p2))
    errors.extend(_question_sharing_errors(p3))
    p2_camera_station_violations = _p2_camera_station_violations(p2)
    if p2_camera_station_violations:
        errors.append(
            f"p2_camera_station_violations:{len(p2_camera_station_violations)}"
        )

    fingerprints: dict[tuple[str, str], str] = {}
    duplicate_pairs: list[tuple[str, str]] = []
    near_duplicate_pairs: list[tuple[str, str]] = []
    movement_violations: list[str] = []
    integrity_violations: list[str] = []
    plan_ids: set[str] = set()
    duplicate_plan_ids: list[str] = []
    candidate_ids: set[str] = set()
    duplicate_candidate_ids: list[str] = []
    frame_counts: collections.Counter[int] = collections.Counter()
    scene_counts: collections.Counter[str] = collections.Counter()
    for tier, manifest, producer_cells in (
        ("p2", p2, p2_producer),
        ("p3", p3, p3_producer),
    ):
        plans_by_scene: dict[str, list[TrajectoryPlan]] = collections.defaultdict(list)
        for cell in producer_cells:
            for candidate in cell["candidates"]:
                if candidate["candidate_id"] in candidate_ids:
                    duplicate_candidate_ids.append(candidate["candidate_id"])
                candidate_ids.add(candidate["candidate_id"])
                plan = TrajectoryPlan.model_validate_json(
                    Path(candidate["plan_record"]).read_text(encoding="utf-8")
                )
                if candidate["plan_id"] != plan.plan_id:
                    integrity_violations.append(f"candidate_plan_id:{candidate['candidate_id']}")
                if plan.scene_id != cell["scene_id"]:
                    integrity_violations.append(f"scene_id:{candidate['candidate_id']}")
                if plan.capability != cell["capability"]:
                    integrity_violations.append(f"capability:{candidate['candidate_id']}")
                if plan.binding != cell["binding"]:
                    integrity_violations.append(f"binding:{candidate['candidate_id']}")
                if plan.standard_version != manifest["standard_version"]:
                    integrity_violations.append(f"standard_version:{candidate['candidate_id']}")
                if [pose.frame for pose in plan.poses] != list(range(len(plan.poses))):
                    integrity_violations.append(f"pose_frames:{candidate['candidate_id']}")
                plans_by_scene[cell["scene_key"]].append(plan)
                if plan.plan_id in plan_ids:
                    duplicate_plan_ids.append(plan.plan_id)
                plan_ids.add(plan.plan_id)
                fingerprint_key = (cell["scene_key"], _pose_fingerprint(plan))
                if fingerprint_key in fingerprints:
                    duplicate_pairs.append((fingerprints[fingerprint_key], plan.plan_id))
                else:
                    fingerprints[fingerprint_key] = plan.plan_id
                frame_counts[len(plan.poses)] += 1
                scene_counts[cell["scene_key"]] += 1
                if tier == "p3":
                    movement_violations.extend(_motion_violations(plan))
        for plans in plans_by_scene.values():
            for left, right in itertools.combinations(plans, 2):
                if not trajectories_are_diverse(left, right):
                    near_duplicate_pairs.append((left.plan_id, right.plan_id))

    if duplicate_pairs:
        errors.append(f"exact_pose_duplicates_per_scene:{len(duplicate_pairs)}")
    if near_duplicate_pairs:
        errors.append(f"within_scene_tier_near_duplicates:{len(near_duplicate_pairs)}")
    if duplicate_plan_ids:
        errors.append(f"duplicate_plan_ids:{len(duplicate_plan_ids)}")
    if duplicate_candidate_ids:
        errors.append(f"duplicate_candidate_ids:{len(duplicate_candidate_ids)}")
    if integrity_violations:
        errors.append(f"plan_manifest_integrity_violations:{len(integrity_violations)}")
    if movement_violations:
        errors.append(f"p3_motion_violations:{len(movement_violations)}")

    marker_capabilities = tuple(p2["capabilities"]) + tuple(p3["capabilities"])
    marker_contract_ok = all(
        script_uses_markers(SCRIPT_LIBRARY[capability]) for capability in marker_capabilities
    ) and STD_V1.landmark_min_visible_frames == int(quality["marker_min_badged_frames"])
    if not marker_contract_ok:
        errors.append("marker_runtime_contract_mismatch")

    total = p2_count + sum(p3_counts.values())
    if total != int(contract["physical_trajectory_target"]):
        errors.append(f"physical_trajectory_total:{total}")
    report = {
        "schema_version": "p23_7k_trajectory_quality_audit.v1",
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "inputs": {
            "p2": {
                "path": str(args.p2.resolve()),
                "sha256": _file_sha256(args.p2.resolve()),
            },
            "p3": {
                "path": str(args.p3.resolve()),
                "sha256": _file_sha256(args.p3.resolve()),
            },
            "contract": {
                "path": str(args.contract.resolve()),
                "sha256": _file_sha256(args.contract.resolve()),
            },
            "allowlist": {
                "path": str(args.allowlist.resolve()),
                "sha256": _file_sha256(args.allowlist.resolve()),
            },
        },
        "physical_trajectories": total,
        "actual_allocation": {"p2": p2_count, "p3": p3_counts},
        "expected_p3_ratio_allocation": expected_p3_counts,
        "exact_pose_duplicates_per_scene": len(duplicate_pairs),
        "within_scene_tier_near_duplicates": len(near_duplicate_pairs),
        "duplicate_plan_ids": len(duplicate_plan_ids),
        "duplicate_candidate_ids": len(duplicate_candidate_ids),
        "plan_manifest_integrity_violations": len(integrity_violations),
        "p3_motion_violations": len(movement_violations),
        "p2_camera_station_violations": len(p2_camera_station_violations),
        "p2": {"trajectories": p2_count, "bindings": p2_bindings},
        "p3": {"trajectories": p3_counts, "bindings": p3_bindings},
        "scenes": len(scene_counts),
        "trajectories_by_scene": dict(sorted(scene_counts.items())),
        "frame_count_distribution": {
            str(count): frequency for count, frequency in sorted(frame_counts.items())
        },
        "marker_contract": {
            "all_capabilities_use_persistent_markers": marker_contract_ok,
            "minimum_badged_frames_enforced_after_render": int(quality["marker_min_badged_frames"]),
            "pre_render_marker_visibility": "pending_authoritative_instance_masks",
        },
        "details": {
            "outside_allowlist_cells": outside_allowlist,
            "exact_duplicate_pairs": duplicate_pairs[:20],
            "near_duplicate_pairs": near_duplicate_pairs[:20],
            "movement_violations": movement_violations[:20],
            "integrity_violations": integrity_violations[:20],
            "camera_station_violations": p2_camera_station_violations[:20],
        },
    }
    _write(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
