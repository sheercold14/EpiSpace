"""Plan single-object relation-flip interventions from verified bundle geometry."""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from omnigibson_episode.geometry import (
    dominant_planar_relation,
    planar_aabb_half_extents_in_yaw_frame,
    planar_delta_to_world,
)
from omnigibson_episode.io import write_json_atomic
from omnigibson_episode.scene_inventory import STRUCTURE_CATEGORIES

IGNORED_COLLISION_CATEGORIES = STRUCTURE_CATEGORIES | {"carpet", "rug"}
_LAYOUT_RESOLUTION_M = 0.01


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _overlaps(
    center: list[float], half: list[float], other: dict[str, Any], margin: float = 0.03
) -> bool:
    other_center = other["obb"]["center_m"]
    other_half = other["obb"]["half_extents_m"]
    return all(
        abs(center[axis] - float(other_center[axis]))
        < half[axis] + float(other_half[axis]) + margin
        for axis in range(3)
    )


@lru_cache(maxsize=64)
def _load_room_layout(source_path: Path) -> tuple[Any, dict[int, str]] | None:
    """Load and index one immutable room layout once per planning process."""

    import numpy as np
    from PIL import Image

    scene_directory = source_path.parent.parent
    instance_path = scene_directory / "layout" / "floor_insseg_0.png"
    semantic_path = scene_directory / "layout" / "floor_semseg_0.png"
    categories_path = scene_directory.parent.parent / "metadata" / "room_categories.txt"
    if not all(path.is_file() for path in (instance_path, semantic_path, categories_path)):
        return None
    instance_map = np.asarray(Image.open(instance_path).convert("L"))
    semantic_map = np.asarray(Image.open(semantic_path).convert("L"))
    if instance_map.ndim != 2 or instance_map.shape != semantic_map.shape:
        raise ValueError("room instance and semantic maps must be aligned grayscale images")
    height, width = instance_map.shape
    if height != width:
        raise ValueError("room maps must be square")
    categories = categories_path.read_text(encoding="utf-8").splitlines()
    semantic_to_instances: dict[int, list[int]] = {}
    for candidate_id in sorted(int(value) for value in np.unique(instance_map) if value):
        first_flat_index = int(np.flatnonzero(instance_map == candidate_id)[0])
        semantic_id = int(semantic_map.flat[first_flat_index])
        semantic_to_instances.setdefault(semantic_id, []).append(candidate_id)
    instance_names = {}
    for semantic_id, instance_ids in semantic_to_instances.items():
        if not 1 <= semantic_id <= len(categories):
            raise ValueError(f"room semantic ID is out of range: {semantic_id}")
        for index, instance_id in enumerate(instance_ids):
            instance_names[instance_id] = f"{categories[semantic_id - 1]}_{index}"
    return instance_map, instance_names


def _layout_room_instance(source_path: Path, xy: list[float]) -> str | None:
    """Resolve a world XY point through the licensed scene's room-instance map."""

    layout = _load_room_layout(source_path.resolve())
    if layout is None:
        return None
    instance_map, instance_names = layout
    height, width = instance_map.shape
    row = int(float(xy[1]) / _LAYOUT_RESOLUTION_M + height / 2.0)
    column = int(float(xy[0]) / _LAYOUT_RESOLUTION_M + width / 2.0)
    if row < 0 or row >= height or column < 0 or column >= width:
        return None
    return instance_names.get(int(instance_map[row, column]))


def plan_relation_flip_interventions(
    *,
    bundle_directory: Path,
    object_inventory_path: Path,
    output_path: Path | None = None,
    maximum_proposals: int = 20,
) -> dict[str, Any]:
    bundle = bundle_directory.resolve()
    snapshot = _read(bundle / "scene_snapshot.json")
    scene = _read(bundle / "scene_ir.json")
    episode = _read(bundle / "spatial_episode.json")
    trajectory = _read(bundle / "trajectory_plan.json")
    object_inventory = _read(object_inventory_path)
    source_path = Path(str(snapshot.get("source_path", "")))
    if not source_path.is_file():
        raise FileNotFoundError("licensed source scene JSON is unavailable")
    source = _read(source_path)
    init_info = source["objects_info"]["init_info"]
    catalog = {
        (record["category"], record["model"]): record
        for record in object_inventory["models"]
    }
    entities = {item["source_entity_id"]: item for item in scene["entities"]}
    entity_id_to_source = {
        item["entity_id"]: item["source_entity_id"] for item in scene["entities"]
    }
    visible_steps: dict[str, list[int]] = {name: [] for name in entities}
    for index, observation in enumerate(episode["observations"]):
        for entity_id in observation["visible_entity_ids"]:
            source_name = entity_id_to_source.get(entity_id)
            if source_name is not None:
                visible_steps[source_name].append(index)
    floor_height = float(trajectory["views"][0]["world_from_agent"]["translation_m"][2])
    canonical_observation = episode["observations"][0]
    canonical_rotation = canonical_observation["world_from_camera"]["rotation_xyzw"]

    proposals: list[dict[str, Any]] = []
    for target_name, target in entities.items():
        init = init_info.get(target_name, {}).get("args", {})
        category = str(init.get("category", target.get("raw_label", "object")))
        model = str(init.get("model", ""))
        raw_target_rooms = init.get("in_rooms")
        if isinstance(raw_target_rooms, str):
            room_values = [raw_target_rooms]
        elif isinstance(raw_target_rooms, (list, tuple, set)):
            room_values = raw_target_rooms
        else:
            room_values = []
        target_rooms = {str(value) for value in room_values if str(value).strip()}
        asset = catalog.get((category, model))
        if (
            asset is None
            or not asset["minimal_pair_eligible"]
            or asset["placement_tier"] not in {"floor", "large_floor"}
            or bool(init.get("fixed_base", False))
            or bool(init.get("visual_only", False))
            or len(set(visible_steps[target_name])) < 2
        ):
            continue
        target_center = [float(value) for value in target["obb"]["center_m"]]
        target_half = [float(value) for value in target["obb"]["half_extents_m"]]
        if target_center[2] - target_half[2] > floor_height + 0.2:
            continue
        for anchor_name, anchor in entities.items():
            if anchor_name == target_name or anchor.get("region_id") != target.get("region_id"):
                continue
            if len(set(visible_steps[anchor_name])) < 1:
                continue
            anchor_center = [float(value) for value in anchor["obb"]["center_m"]]
            anchor_half = [float(value) for value in anchor["obb"]["half_extents_m"]]
            relation_before, axis_name, margin_before, right_delta, front_delta = (
                dominant_planar_relation(
                    target_center, anchor_center, canonical_rotation
                )
            )
            deltas = [right_delta, front_delta]
            axis = 0 if axis_name == "right" else 1
            if abs(deltas[axis]) < 0.4 or margin_before < 0.4:
                continue
            orthogonal = 1 - axis
            target_frame_half = planar_aabb_half_extents_in_yaw_frame(
                target_half, canonical_rotation
            )
            anchor_frame_half = planar_aabb_half_extents_in_yaw_frame(
                anchor_half, canonical_rotation
            )
            separation = (
                target_frame_half[axis]
                + anchor_frame_half[axis]
                + abs(deltas[orthogonal])
                + 0.6
            )
            candidate = target_center.copy()
            candidate_frame = deltas.copy()
            candidate_frame[axis] = -math.copysign(separation, deltas[axis])
            candidate_dx, candidate_dy = planar_delta_to_world(
                candidate_frame[0], candidate_frame[1], canonical_rotation
            )
            candidate[0] = anchor_center[0] + candidate_dx
            candidate[1] = anchor_center[1] + candidate_dy
            planned_room = _layout_room_instance(source_path, candidate[:2])
            if target_rooms and planned_room not in target_rooms:
                continue
            movement = math.dist(target_center, candidate)
            if not 0.5 <= movement <= 4.0:
                continue
            collides = any(
                other_name not in {target_name, anchor_name}
                and str(other.get("raw_label", "")) not in IGNORED_COLLISION_CATEGORIES
                and _overlaps(candidate, target_half, other)
                for other_name, other in entities.items()
            )
            if collides:
                continue
            relation_after, after_axis, margin_after, _, _ = dominant_planar_relation(
                candidate, anchor_center, canonical_rotation
            )
            if after_axis != axis_name:
                continue
            payload = {
                "target_source_entity_id": target_name,
                "anchor_source_entity_id": anchor_name,
                "target_category": category,
                "target_model": model,
                "before_center_m": target_center,
                "proposed_center_m": [round(value, 6) for value in candidate],
                "movement_m": round(movement, 6),
                "relation_before": relation_before,
                "relation_after": relation_after,
                "relation_axis": axis_name,
                "relation_frame": "first_view_yaw",
                "relation_frame_view_id": canonical_observation["view_id"],
                "world_from_relation_frame_rotation_xyzw": canonical_rotation,
                "before_axis_margin_m": round(margin_before, 6),
                "expected_after_axis_margin_m": round(margin_after, 6),
                "target_visible_view_ids_before": sorted(
                    {
                        episode["observations"][step]["view_id"]
                        for step in visible_steps[target_name]
                    }
                ),
                "anchor_visible_view_ids_before": sorted(
                    {
                        episode["observations"][step]["view_id"]
                        for step in visible_steps[anchor_name]
                    }
                ),
                "placement_tier": asset["placement_tier"],
                "planned_room_instance": planned_room,
            }
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            payload["proposal_id"] = f"ogflip-{digest[:16]}"
            payload["required_simulator_checks"] = [
                "proposed center remains in the target room instance",
                "target settles without collision or excessive pose drift",
                "all non-target object poses remain invariant",
                "target is visible after intervention",
                "declared relation flip re-executes with >=0.4 m margin",
            ]
            proposals.append(payload)

    proposals.sort(
        key=lambda item: (
            -len(item["target_visible_view_ids_before"]),
            item["movement_m"],
            item["proposal_id"],
        )
    )
    proposals = proposals[:maximum_proposals]
    manifest = {
        "schema_version": "omnigibson_intervention_plan.v1",
        "scene_id": scene["scene_id"],
        "episode_id": episode["episode_id"],
        "intervention_type": "single_object_relation_flip",
        "causal_contract": {
            "mutable_state": ["target_object.world_pose"],
            "invariants": [
                "scene identity",
                "camera trajectory",
                "renderer and sensor settings",
                "all non-target object states",
            ],
        },
        "proposal_count": len(proposals),
        "proposals": proposals,
    }
    output = output_path or bundle / "intervention_proposals.json"
    write_json_atomic(output, manifest)
    return manifest
