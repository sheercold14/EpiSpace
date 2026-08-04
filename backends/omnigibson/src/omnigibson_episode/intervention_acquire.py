"""Execute a certified single-object intervention on a fixed base trajectory.

The intervention worker deliberately mirrors the acquisition boundary: Isaac Sim
is only responsible for changing and observing the world.  Pair construction and
language generation remain lightweight downstream steps.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

from omnigibson_episode.acquire import (
    _as_float_list,
    _capture_scene_snapshot,
    _environment_config,
    _render_views,
    _scene_source_id,
)
from omnigibson_episode.config import EpisodeRecipe
from omnigibson_episode.geometry import dominant_planar_relation
from omnigibson_episode.io import sha256_file, write_json_atomic
from omnigibson_episode.scene_inventory import STRUCTURE_CATEGORIES

_CONTACT_TOLERANT_CATEGORIES = STRUCTURE_CATEGORIES | {"carpet", "rug"}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _select_proposal(plan: dict[str, Any], proposal_id: str) -> dict[str, Any]:
    matches = [
        item
        for item in plan.get("proposals", [])
        if isinstance(item, dict) and item.get("proposal_id") == proposal_id
    ]
    if len(matches) != 1:
        raise ValueError(f"proposal ID must resolve exactly once: {proposal_id}")
    return matches[0]


def _model_swap_scene_file(
    base_snapshot: dict[str, Any], proposal: dict[str, Any]
) -> dict[str, Any]:
    source_path = Path(str(base_snapshot.get("source_path", "")))
    if not source_path.is_file():
        raise FileNotFoundError("base source scene JSON is unavailable for model swap")
    scene_file = deepcopy(_read_json(source_path))
    target_name = str(proposal["target_source_entity_id"])
    try:
        args = scene_file["objects_info"]["init_info"][target_name]["args"]
    except KeyError as error:
        raise ValueError("model-swap target is absent from source scene JSON") from error
    if str(args.get("category")) != str(proposal["target_category"]):
        raise ValueError("model-swap target category does not match source scene")
    if str(args.get("model")) != str(proposal["source_model"]):
        raise ValueError("model-swap source model does not match source scene")
    args["model"] = str(proposal["replacement_model"])
    args.pop("scale", None)
    args.pop("expected_file_hash", None)
    args["bounding_box"] = [float(value) for value in proposal["target_bbox_extent_m"]]
    return scene_file


def _pose(obj: Any) -> tuple[list[float], list[float]]:
    position, orientation = obj.get_position_orientation(frame="world")
    return _as_float_list(position), _as_float_list(orientation)


def _quaternion_angle_deg(left: list[float], right: list[float]) -> float:
    dot = abs(sum(a * b for a, b in zip(left, right, strict=True)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def _aabb_overlap(
    center: list[float],
    extent: list[float],
    other_center: list[float],
    other_extent: list[float],
    *,
    penetration_tolerance_m: float = 0.01,
) -> bool:
    return all(
        abs(center[axis] - other_center[axis])
        < (extent[axis] + other_extent[axis]) / 2.0 - penetration_tolerance_m
        for axis in range(3)
    )


def _snapshot_overlap_names(
    snapshot: dict[str, Any], *, target_name: str, anchor_name: str
) -> set[str]:
    """Return pre-existing target overlaps from the immutable base snapshot."""

    entities = {
        str(item["source_entity_id"]): item for item in snapshot.get("entities", [])
    }
    target = entities.get(target_name)
    if target is None:
        raise ValueError("intervention target is absent from the base snapshot")
    target_center = [float(value) for value in target["aabb_center_m"]]
    target_extent = [float(value) for value in target["aabb_extent_m"]]
    overlaps = set()
    for name, item in entities.items():
        if name in {target_name, anchor_name}:
            continue
        category = str(item.get("category", "object"))
        if category in _CONTACT_TOLERANT_CATEGORIES:
            continue
        if _aabb_overlap(
            target_center,
            target_extent,
            [float(value) for value in item["aabb_center_m"]],
            [float(value) for value in item["aabb_extent_m"]],
        ):
            overlaps.add(name)
    return overlaps


def relation_from_centers(
    *, proposal: dict[str, Any], target_center: list[float], anchor_center: list[float]
) -> tuple[str, float]:
    if proposal.get("relation_frame") == "first_view_yaw":
        relation, axis, margin, _, _ = dominant_planar_relation(
            target_center,
            anchor_center,
            proposal["world_from_relation_frame_rotation_xyzw"],
        )
        expected_axis = proposal["relation_axis"]
        if axis != expected_axis:
            return relation, margin
        return relation, margin
    axis = 0 if proposal["relation_axis"] == "x" else 1
    orthogonal = 1 - axis
    delta = target_center[axis] - anchor_center[axis]
    relation = (
        ("right_of" if delta > 0 else "left_of")
        if axis == 0
        else ("in_front_of" if delta > 0 else "behind")
    )
    return relation, abs(delta) - abs(
        target_center[orthogonal] - anchor_center[orthogonal]
    )


def _room_at(scene: Any, center: list[float]) -> str | None:
    segmentation = getattr(scene, "seg_map", None)
    if segmentation is None:
        return None
    room = segmentation.get_room_instance_by_point(center[:2])
    return None if room is None else str(room)


def _execution_checks(
    *,
    proposal: dict[str, Any],
    target_center: list[float],
    anchor_center: list[float],
    proposed_room: str | None,
    expected_room: str,
    target_settle_drift_m: float,
    maximum_non_target_position_drift_m: float,
    maximum_non_target_orientation_drift_deg: float,
    overlap_names: list[str],
    target_visible_after: bool | None,
) -> list[dict[str, Any]]:
    relation, margin = relation_from_centers(
        proposal=proposal,
        target_center=target_center,
        anchor_center=anchor_center,
    )
    checks = [
        {
            "name": "same_room_containment",
            "passed": proposed_room == expected_room,
            "measured_value": proposed_room or "outside_annotated_rooms",
            "threshold": expected_room,
        },
        {
            "name": "target_settle_drift_m",
            "passed": target_settle_drift_m <= 0.08,
            "measured_value": round(target_settle_drift_m, 6),
            "threshold": 0.08,
        },
        {
            "name": "non_target_position_invariance_m",
            "passed": maximum_non_target_position_drift_m <= 0.02,
            "measured_value": round(maximum_non_target_position_drift_m, 6),
            "threshold": 0.02,
        },
        {
            "name": "non_target_orientation_invariance_deg",
            "passed": maximum_non_target_orientation_drift_deg <= 2.0,
            "measured_value": round(maximum_non_target_orientation_drift_deg, 6),
            "threshold": 2.0,
        },
        {
            "name": "no_new_aabb_penetration",
            "passed": not overlap_names,
            "measured_value": ",".join(overlap_names) if overlap_names else "none",
            "threshold": "none",
        },
        {
            "name": "declared_relation_after",
            "passed": relation == proposal["relation_after"],
            "measured_value": relation,
            "threshold": proposal["relation_after"],
        },
        {
            "name": "relation_axis_margin_m",
            "passed": margin >= 0.4,
            "measured_value": round(margin, 6),
            "threshold": 0.4,
        },
    ]
    if target_visible_after is not None:
        checks.append(
            {
                "name": "target_visible_after",
                "passed": target_visible_after,
                "measured_value": target_visible_after,
                "threshold": True,
            }
        )
    return checks


def _assert_checks(checks: list[dict[str, Any]]) -> None:
    failed = [item["name"] for item in checks if not item["passed"]]
    if failed:
        raise RuntimeError("intervention validation failed: " + ", ".join(failed))


def acquire_intervention(
    *,
    recipe: EpisodeRecipe,
    base_bundle: Path,
    intervention_plan_path: Path,
    proposal_id: str,
    output_directory: Path,
    gpu_id: int,
    headless: bool,
    overwrite: bool = False,
    settle_steps: int = 12,
) -> Path:
    """Render one relation-flip variant while holding every other variable fixed."""

    if gpu_id < 0:
        raise ValueError("gpu_id cannot be negative")
    if settle_steps < 0:
        raise ValueError("settle_steps must be non-negative")
    base_bundle = base_bundle.resolve()
    output_directory = output_directory.resolve()
    plan_path = intervention_plan_path.resolve()
    for member in (
        base_bundle / "render_report.json",
        base_bundle / "scene_snapshot.json",
        base_bundle / "trajectory_plan.json",
        base_bundle / "spatial_episode.json",
        plan_path,
    ):
        if not member.is_file():
            raise FileNotFoundError(member)
    if output_directory == base_bundle:
        raise ValueError("intervention output must not overwrite the base bundle")
    if output_directory.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output_directory}")
        shutil.rmtree(output_directory)
    if overwrite:
        for stale in output_directory.parent.glob(f".{output_directory.name}.staging.*"):
            shutil.rmtree(stale)
    staging = output_directory.with_name(f".{output_directory.name}.staging.{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    base_report = _read_json(base_bundle / "render_report.json")
    base_snapshot = _read_json(base_bundle / "scene_snapshot.json")
    base_episode = _read_json(base_bundle / "spatial_episode.json")
    trajectory = _read_json(base_bundle / "trajectory_plan.json")
    plan = _read_json(plan_path)
    proposal = _select_proposal(plan, proposal_id)
    intervention_type = str(plan.get("intervention_type", ""))
    if intervention_type not in {
        "single_object_relation_flip",
        "single_object_model_swap",
    }:
        raise ValueError(f"unsupported intervention type: {intervention_type}")
    if intervention_type == "single_object_relation_flip" and settle_steps < 1:
        raise ValueError("relation flips require at least one settle step")
    execution_mode = str(proposal.get("execution_mode", "physics_settle"))
    if intervention_type == "single_object_model_swap":
        expected_steps = int(proposal.get("settle_steps", settle_steps))
        if settle_steps != expected_steps:
            raise ValueError("runtime settle_steps do not match the model-swap proposal")
        if execution_mode == "kinematic_counterfactual" and settle_steps != 0:
            raise ValueError("kinematic model swaps require zero settle steps")
    if base_snapshot.get("source_scene_id") != _scene_source_id(recipe):
        raise ValueError("recipe scene does not match the base bundle")
    if plan.get("episode_id") != base_episode.get("episode_id"):
        raise ValueError("intervention plan was generated from a different base episode")
    base_trajectory_digest = sha256_file(base_bundle / "trajectory_plan.json")
    if base_report.get("trajectory_sha256") != base_trajectory_digest:
        raise ValueError("base render report and trajectory digest disagree")

    os.environ["OMNIGIBSON_GPU_ID"] = str(gpu_id)
    env = None
    result_path: Path | None = None
    pending_error: Exception | None = None
    diagnostic: dict[str, Any] = {}
    try:
        import numpy as np
        import omnigibson as og
        import torch as th
        from omnigibson.macros import gm
        from omnigibson.utils.constants import STRUCTURE_CATEGORIES as OG_STRUCTURE_CATEGORIES

        random.seed(recipe.seed)
        np.random.seed(recipe.seed)
        th.manual_seed(recipe.seed)
        gm.HEADLESS = headless
        gm.ENABLE_HQ_RENDERING = recipe.sensor.high_quality_rendering
        config = _environment_config(recipe)
        if intervention_type == "single_object_model_swap":
            config["scene"]["scene_file"] = _model_swap_scene_file(
                base_snapshot, proposal
            )
        if config["scene"].get("load_object_categories") == "__STRUCTURE_CATEGORIES__":
            config["scene"]["load_object_categories"] = sorted(OG_STRUCTURE_CATEGORIES)
        env = og.Environment(configs=config)
        objects = {str(obj.name): obj for obj in env.scene.objects}
        target_name = str(proposal["target_source_entity_id"])
        anchor_name = str(proposal["anchor_source_entity_id"])
        if target_name not in objects or anchor_name not in objects:
            raise RuntimeError("target or anchor object is unavailable in the loaded scene")
        target = objects[target_name]
        anchor = objects[anchor_name]
        target_before_center = _as_float_list(target.aabb_center)
        expected_target_center = [
            float(value)
            for value in proposal[
                "before_center_m"
                if intervention_type == "single_object_relation_flip"
                else "expected_target_center_m"
            ]
        ]
        if (
            intervention_type == "single_object_relation_flip"
            and math.dist(target_before_center, expected_target_center) > 0.05
        ):
            raise RuntimeError("loaded target pose does not reproduce the planned base state")
        loaded_model = str(
            getattr(target, "model", getattr(target, "model_name", "unknown"))
        )
        if (
            intervention_type == "single_object_model_swap"
            and loaded_model != str(proposal["replacement_model"])
        ):
            raise RuntimeError(
                f"replacement model did not load: {loaded_model} != "
                f"{proposal['replacement_model']}"
            )
        target_position, target_orientation = _pose(target)
        before_poses = {name: _pose(obj) for name, obj in objects.items()}
        expected_rooms = getattr(target, "in_rooms", None)
        if isinstance(expected_rooms, str):
            expected_room_set = {expected_rooms}
        else:
            expected_room_set = {str(room) for room in expected_rooms or []}
        if not expected_room_set:
            expected_room_set = {
                str(item.get("region", "unassigned"))
                for item in base_snapshot["entities"]
                if item["source_entity_id"] == target_name
            }
        proposed_center = (
            [float(value) for value in proposal["proposed_center_m"]]
            if intervention_type == "single_object_relation_flip"
            else expected_target_center
        )
        proposed_room = _room_at(env.scene, proposed_center)
        if proposed_room not in expected_room_set:
            raise RuntimeError(
                f"proposed target center leaves its room: {proposed_room!r} not in "
                f"{sorted(expected_room_set)}"
            )

        delta = [
            proposed_center[index] - target_before_center[index] for index in range(3)
        ]
        target.set_position_orientation(
            position=th.tensor(
                [target_position[index] + delta[index] for index in range(3)],
                dtype=th.float32,
            ),
            orientation=th.tensor(target_orientation, dtype=th.float32),
            frame="world",
        )
        target.keep_still()
        for _ in range(settle_steps):
            og.sim.step_physics()

        target_after_center = _as_float_list(target.aabb_center)
        target_after_extent = _as_float_list(target.aabb_extent)
        anchor_after_center = _as_float_list(anchor.aabb_center)
        after_poses = {name: _pose(obj) for name, obj in objects.items()}
        position_drifts = {
            name: math.dist(before_poses[name][0], after_poses[name][0])
            for name in objects
            if name != target_name
        }
        orientation_drifts = {
            name: _quaternion_angle_deg(before_poses[name][1], after_poses[name][1])
            for name in objects
            if name != target_name
        }
        pose_drift_audit = sorted(
            (
                {
                    "source_entity_id": name,
                    "position_drift_m": round(position_drifts[name], 6),
                    "orientation_drift_deg": round(orientation_drifts[name], 6),
                }
                for name in position_drifts
            ),
            key=lambda item: (
                -float(item["position_drift_m"]),
                -float(item["orientation_drift_deg"]),
                str(item["source_entity_id"]),
            ),
        )[:10]
        baseline_overlap_names = _snapshot_overlap_names(
            base_snapshot, target_name=target_name, anchor_name=anchor_name
        )
        after_overlap_names = set()
        for name, obj in objects.items():
            if name in {target_name, anchor_name}:
                continue
            category = str(getattr(obj, "category", "object"))
            if category in _CONTACT_TOLERANT_CATEGORIES:
                continue
            if _aabb_overlap(
                target_after_center,
                target_after_extent,
                _as_float_list(obj.aabb_center),
                _as_float_list(obj.aabb_extent),
            ):
                after_overlap_names.add(name)
        new_overlap_names = after_overlap_names - baseline_overlap_names
        overlap_audit = {
            "baseline_overlap_names": sorted(baseline_overlap_names),
            "after_overlap_names": sorted(after_overlap_names),
            "new_overlap_names": sorted(new_overlap_names),
        }
        expected_room = proposed_room or sorted(expected_room_set)[0]
        checks = _execution_checks(
            proposal=proposal,
            target_center=target_after_center,
            anchor_center=anchor_after_center,
            proposed_room=_room_at(env.scene, target_after_center),
            expected_room=expected_room,
            target_settle_drift_m=math.dist(target_after_center, proposed_center),
            maximum_non_target_position_drift_m=max(position_drifts.values(), default=0.0),
            maximum_non_target_orientation_drift_deg=max(
                orientation_drifts.values(), default=0.0
            ),
            overlap_names=sorted(new_overlap_names),
            target_visible_after=None,
        )
        if intervention_type == "single_object_model_swap":
            checks.extend(
                [
                    {
                        "name": "replacement_model_loaded",
                        "passed": loaded_model == proposal["replacement_model"],
                        "measured_value": loaded_model,
                        "threshold": proposal["replacement_model"],
                    },
                    {
                        "name": "semantic_category_invariance",
                        "passed": str(getattr(target, "category", ""))
                        == proposal["target_category"],
                        "measured_value": str(getattr(target, "category", "")),
                        "threshold": proposal["target_category"],
                    },
                ]
            )
        diagnostic = {
            "checks": checks,
            "overlap_audit": overlap_audit,
            "non_target_pose_drift_audit": pose_drift_audit,
        }
        _assert_checks(checks)

        write_json_atomic(staging / "trajectory_plan.json", trajectory)
        selection_path = base_bundle / "trajectory_selection.json"
        if selection_path.is_file():
            write_json_atomic(
                staging / "trajectory_selection.json", _read_json(selection_path)
            )
        sensor = env.external_sensors["episode_camera"]
        category_by_name = {
            str(obj.name): str(getattr(obj, "category", "object"))
            for obj in env.scene.objects
        }
        object_name_by_prim_path = {
            str(obj.prim_path).rstrip("/"): str(obj.name)
            for obj in env.scene.objects
            if str(getattr(obj, "prim_path", "")).startswith("/")
        }
        rendered, instance_registry, semantic_registry = _render_views(
            og=og,
            sensor=sensor,
            trajectory=trajectory,
            recipe=recipe,
            category_by_name=category_by_name,
            object_name_by_prim_path=object_name_by_prim_path,
            staging_root=staging,
            public_root=output_directory,
        )
        target_runtime_ids = {
            identifier
            for identifier, name in instance_registry.items()
            if name == target_name
        }
        target_visible_after = any(
            target_runtime_ids & set(view["visible_runtime_instance_ids"])
            for view in rendered
        )
        checks = _execution_checks(
            proposal=proposal,
            target_center=target_after_center,
            anchor_center=anchor_after_center,
            proposed_room=_room_at(env.scene, target_after_center),
            expected_room=expected_room,
            target_settle_drift_m=math.dist(target_after_center, proposed_center),
            maximum_non_target_position_drift_m=max(position_drifts.values(), default=0.0),
            maximum_non_target_orientation_drift_deg=max(
                orientation_drifts.values(), default=0.0
            ),
            overlap_names=sorted(new_overlap_names),
            target_visible_after=target_visible_after,
        )
        if intervention_type == "single_object_model_swap":
            checks.extend(
                [
                    {
                        "name": "replacement_model_loaded",
                        "passed": loaded_model == proposal["replacement_model"],
                        "measured_value": loaded_model,
                        "threshold": proposal["replacement_model"],
                    },
                    {
                        "name": "semantic_category_invariance",
                        "passed": str(getattr(target, "category", ""))
                        == proposal["target_category"],
                        "measured_value": str(getattr(target, "category", "")),
                        "threshold": proposal["target_category"],
                    },
                ]
            )
        diagnostic = {"checks": checks, "overlap_audit": overlap_audit}
        _assert_checks(checks)
        snapshot = _capture_scene_snapshot(
            env, recipe, instance_registry, semantic_registry
        )
        write_json_atomic(staging / "scene_snapshot.json", snapshot)
        report = {
            "protocol_version": "omnigibson_render_bundle.v1",
            "status": "success",
            "scene_id": base_report["scene_id"],
            "trajectory_sha256": sha256_file(staging / "trajectory_plan.json"),
            "gpu_id": gpu_id,
            "headless": headless,
            "high_quality_rendering": recipe.sensor.high_quality_rendering,
            "sensor_contract": dict(base_report["sensor_contract"]),
            "intervention": {
                "proposal_id": proposal_id,
                "type": intervention_type,
                "base_episode_id": base_episode["episode_id"],
            },
            "views": rendered,
        }
        write_json_atomic(staging / "render_report.json", report)
        execution = {
            "schema_version": "omnigibson_intervention_execution.v1",
            "status": "success",
            "family_variant": (
                f"relation_flip__{proposal_id}"
                if intervention_type == "single_object_relation_flip"
                else f"model_swap__{proposal_id}"
            ),
            "intervention_type": intervention_type,
            "execution_mode": execution_mode,
            "proposal": proposal,
            "base": {
                "bundle": str(base_bundle),
                "episode_id": base_episode["episode_id"],
                "family_id": base_episode["family_id"],
                "scene_id": base_episode["scene_id"],
                "render_report_sha256": sha256_file(base_bundle / "render_report.json"),
                "trajectory_sha256": base_trajectory_digest,
            },
            "actual": {
                "target_center_before_m": target_before_center,
                "target_center_after_m": target_after_center,
                "loaded_target_model": loaded_model,
                "target_extent_after_m": target_after_extent,
                "anchor_center_after_m": anchor_after_center,
                "target_room_after": _room_at(env.scene, target_after_center),
                "settle_steps": settle_steps,
                "target_visible_after": target_visible_after,
            },
            "checks": checks,
            "overlap_audit": overlap_audit,
            "non_target_pose_drift_audit": pose_drift_audit,
        }
        write_json_atomic(staging / "intervention_execution.json", execution)
        os.replace(staging, output_directory)
        result_path = output_directory / "render_report.json"
    except Exception as error:
        failure = {
            "protocol_version": "omnigibson_intervention_failure.v1",
            "status": "failure",
            "proposal_id": proposal_id,
            "error": {"type": type(error).__name__, "message": str(error)},
            **diagnostic,
        }
        write_json_atomic(staging / "failure_report.json", failure)
        if output_directory.exists():
            shutil.rmtree(output_directory)
        os.replace(staging, output_directory)
        pending_error = error
    finally:
        if env is not None:
            try:
                import omnigibson as og

                og.shutdown()
            except BaseException:
                if pending_error is None:
                    raise
    if pending_error is not None:
        raise pending_error
    if result_path is None:
        raise RuntimeError("intervention ended without a success or failure result")
    return result_path
