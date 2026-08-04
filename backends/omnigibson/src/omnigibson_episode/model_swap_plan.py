"""Plan category-preserving model swaps for visual-invariance minimal pairs."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from omnigibson_episode.geometry import dominant_planar_relation
from omnigibson_episode.io import sha256_file, write_json_atomic
from omnigibson_episode.scene_inventory import STRUCTURE_CATEGORIES


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _center(entity: dict[str, Any]) -> list[float]:
    return [float(value) for value in entity["obb"]["center_m"]]


def _extent(entity: dict[str, Any]) -> list[float]:
    return [2.0 * float(value) for value in entity["obb"]["half_extents_m"]]


def _relation(
    subject: dict[str, Any],
    reference: dict[str, Any],
    world_from_frame_xyzw: list[float],
) -> tuple[str, str, float]:
    relation, axis, margin, _, _ = dominant_planar_relation(
        _center(subject), _center(reference), world_from_frame_xyzw
    )
    return relation, axis, margin


def _aspect_distance(left: list[float], right: list[float]) -> float:
    left_log = [math.log(max(value, 1e-6)) for value in left]
    right_log = [math.log(max(value, 1e-6)) for value in right]
    left_mean = sum(left_log) / 3.0
    right_mean = sum(right_log) / 3.0
    return max(
        abs((left_log[index] - left_mean) - (right_log[index] - right_mean))
        for index in range(3)
    )


def _aabb_clearance(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_center = _center(left)
    right_center = _center(right)
    left_half = [float(value) for value in left["obb"]["half_extents_m"]]
    right_half = [float(value) for value in right["obb"]["half_extents_m"]]
    gaps = [
        max(
            0.0,
            abs(left_center[axis] - right_center[axis])
            - left_half[axis]
            - right_half[axis],
        )
        for axis in range(3)
    ]
    return math.sqrt(sum(value * value for value in gaps))


def _clearance_summary(
    target_name: str,
    target: dict[str, Any],
    anchor_name: str,
    entities: dict[str, dict[str, Any]],
) -> tuple[int, float]:
    clearances = []
    for other_name, other in entities.items():
        if other_name in {target_name, anchor_name}:
            continue
        if str(other.get("raw_label", "")).casefold() in STRUCTURE_CATEGORIES:
            continue
        clearances.append(_aabb_clearance(target, other))
    overlap_count = sum(value <= 0.01 for value in clearances)
    nonoverlap = [value for value in clearances if value > 0.01]
    return overlap_count, min(nonoverlap, default=99.0)


def plan_model_swap_interventions(
    *,
    bundle_directory: Path,
    object_inventory_path: Path,
    output_path: Path | None = None,
    maximum_proposals: int = 20,
    maximum_baseline_overlap_count: int = 1,
    minimum_nonoverlap_clearance_m: float = 0.05,
    maximum_aspect_ratio_factor: float = 1.5,
    execution_mode: str = "physics_settle",
    settle_steps: int = 12,
) -> dict[str, Any]:
    """Select one logical scene instance and replace only its visual asset model."""

    if maximum_baseline_overlap_count < 0:
        raise ValueError("maximum_baseline_overlap_count must be non-negative")
    if minimum_nonoverlap_clearance_m < 0:
        raise ValueError("minimum_nonoverlap_clearance_m must be non-negative")
    if maximum_aspect_ratio_factor <= 1.0:
        raise ValueError("maximum_aspect_ratio_factor must be greater than one")
    if execution_mode not in {"physics_settle", "kinematic_counterfactual"}:
        raise ValueError(f"unsupported model-swap execution_mode: {execution_mode}")
    if execution_mode == "physics_settle" and settle_steps < 1:
        raise ValueError("physics_settle requires at least one settle step")
    if execution_mode == "kinematic_counterfactual" and settle_steps != 0:
        raise ValueError("kinematic_counterfactual requires zero settle steps")

    bundle = bundle_directory.resolve()
    snapshot = _read(bundle / "scene_snapshot.json")
    scene = _read(bundle / "scene_ir.json")
    episode = _read(bundle / "spatial_episode.json")
    inventory_path = object_inventory_path.resolve()
    inventory = _read(inventory_path)
    source_path = Path(str(snapshot.get("source_path", "")))
    if not source_path.is_file():
        raise FileNotFoundError("licensed source scene JSON is unavailable")
    source = _read(source_path)
    init_info = source["objects_info"]["init_info"]

    assets_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    assets_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in inventory["models"]:
        category = str(record["category"])
        model = str(record["model"])
        assets_by_key[(category, model)] = record
        if (
            record.get("minimal_pair_eligible")
            and record.get("collision_ready")
            and int(record.get("link_count", 0)) == 1
        ):
            assets_by_category[category].append(record)

    entities = {str(item["source_entity_id"]): item for item in scene["entities"]}
    entity_id_to_source = {
        str(item["entity_id"]): str(item["source_entity_id"])
        for item in scene["entities"]
    }
    visible_steps: dict[str, list[int]] = {name: [] for name in entities}
    for index, observation in enumerate(episode["observations"]):
        for entity_id in observation["visible_entity_ids"]:
            source_name = entity_id_to_source.get(str(entity_id))
            if source_name is not None:
                visible_steps[source_name].append(index)
    canonical_observation = episode["observations"][0]
    canonical_rotation = canonical_observation["world_from_camera"]["rotation_xyzw"]

    proposals: list[dict[str, Any]] = []
    maximum_aspect_log_delta = math.log(maximum_aspect_ratio_factor)
    for target_name, target in entities.items():
        args = init_info.get(target_name, {}).get("args", {})
        category = str(args.get("category", target.get("raw_label", "object")))
        source_model = str(args.get("model", ""))
        source_asset = assets_by_key.get((category, source_model))
        if (
            category.casefold() in STRUCTURE_CATEGORIES
            or source_asset is None
            or not source_asset.get("minimal_pair_eligible")
            or int(source_asset.get("link_count", 0)) != 1
            or bool(args.get("fixed_base", False))
            or bool(args.get("visual_only", False))
            or len(set(visible_steps[target_name])) < 2
        ):
            continue

        anchor_options = []
        for anchor_name, anchor in entities.items():
            if (
                anchor_name == target_name
                or anchor.get("region_id") != target.get("region_id")
                or not visible_steps.get(anchor_name)
                or str(anchor.get("raw_label", "")).casefold() in STRUCTURE_CATEGORIES
            ):
                continue
            relation, axis, margin = _relation(target, anchor, canonical_rotation)
            distance = math.dist(_center(target), _center(anchor))
            if margin >= 0.4 and 0.4 <= distance <= 8.0:
                anchor_options.append((-margin, distance, anchor_name, relation, axis))
        if not anchor_options:
            continue
        neg_margin, distance, anchor_name, relation, axis = min(anchor_options)
        baseline_overlap_count, minimum_clearance_m = _clearance_summary(
            target_name, target, anchor_name, entities
        )
        if (
            baseline_overlap_count > maximum_baseline_overlap_count
            or minimum_clearance_m < minimum_nonoverlap_clearance_m
        ):
            continue

        source_bbox = [float(value) for value in source_asset["bbox_size_m"]]
        alternatives = []
        for replacement in assets_by_category.get(category, []):
            if (
                replacement["model"] == source_model
                or replacement.get("placement_tier") != source_asset.get("placement_tier")
            ):
                continue
            replacement_bbox = [float(value) for value in replacement["bbox_size_m"]]
            aspect_distance = _aspect_distance(source_bbox, replacement_bbox)
            if aspect_distance <= maximum_aspect_log_delta:
                alternatives.append((aspect_distance, str(replacement["model"]), replacement))
        if not alternatives:
            continue
        aspect_distance, replacement_model, replacement = min(alternatives)
        payload = {
            "target_source_entity_id": target_name,
            "anchor_source_entity_id": anchor_name,
            "target_category": category,
            "source_model": source_model,
            "replacement_model": replacement_model,
            "source_asset_bbox_m": source_bbox,
            "replacement_asset_bbox_m": replacement["bbox_size_m"],
            "target_bbox_extent_m": [round(value, 6) for value in _extent(target)],
            "expected_target_center_m": [round(value, 6) for value in _center(target)],
            "relation_before": relation,
            "relation_after": relation,
            "relation_axis": axis,
            "relation_frame": "first_view_yaw",
            "relation_frame_view_id": canonical_observation["view_id"],
            "world_from_relation_frame_rotation_xyzw": canonical_rotation,
            "relation_margin_before_m": round(-neg_margin, 6),
            "anchor_distance_m": round(distance, 6),
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
            "placement_tier": source_asset["placement_tier"],
            "source_link_count": source_asset["link_count"],
            "replacement_link_count": replacement["link_count"],
            "aspect_log_distance": round(aspect_distance, 6),
            "difficulty_tier": "appearance_invariance_conservative",
            "baseline_overlap_count": baseline_overlap_count,
            "minimum_nonoverlap_clearance_m": round(minimum_clearance_m, 6),
            "execution_mode": execution_mode,
            "settle_steps": settle_steps,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        payload["proposal_id"] = f"ogswap-{digest[:16]}"
        payload["required_simulator_checks"] = [
            "replacement model loads under the same category and logical instance name",
            "target center, room, and all non-target poses remain invariant",
            (
                "replacement settles without new collision penetration"
                if execution_mode == "physics_settle"
                else (
                    "replacement occupies the fixed target bounds without new "
                    "collision penetration"
                )
            ),
            "target remains visible on the exact same camera trajectory",
            "declared spatial answer remains unchanged with >=0.4 m margin",
        ]
        proposals.append(payload)

    proposals.sort(
        key=lambda item: (
            -len(item["target_visible_view_ids_before"]),
            item["baseline_overlap_count"],
            -item["minimum_nonoverlap_clearance_m"],
            item["aspect_log_distance"],
            item["proposal_id"],
        )
    )
    proposals = proposals[:maximum_proposals]
    manifest = {
        "schema_version": "omnigibson_intervention_plan.v1",
        "scene_id": scene["scene_id"],
        "episode_id": episode["episode_id"],
        "intervention_type": "single_object_model_swap",
        "object_inventory_sha256": sha256_file(inventory_path),
        "causal_contract": {
            "mutable_state": ["target_object.asset_model", "target_object.visual_geometry"],
            "invariants": [
                "logical target identity and semantic category",
                "scene identity and camera trajectory",
                "target world center and declared spatial relations",
                "all non-target object states",
            ],
        },
        "candidate_policy": {
            "difficulty_tier": "appearance_invariance_conservative",
            "maximum_baseline_overlap_count": maximum_baseline_overlap_count,
            "minimum_nonoverlap_clearance_m": minimum_nonoverlap_clearance_m,
            "maximum_aspect_ratio_factor": maximum_aspect_ratio_factor,
            "execution_mode": execution_mode,
            "settle_steps": settle_steps,
        },
        "proposal_count": len(proposals),
        "proposals": proposals,
    }
    output = output_path or bundle / "model_swap_proposals.json"
    write_json_atomic(output, manifest)
    return manifest
