"""Build a simulator-free inventory from installed BEHAVIOR scene metadata.

The inventory is intentionally aggregate-only: it records enough information to
plan and audit acquisition without redistributing licensed scene/object assets.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from omnigibson_episode.io import sha256_file, write_json_atomic

STRUCTURE_CATEGORIES = frozenset(
    {"floors", "walls", "ceilings", "lawn", "driveway", "fence", "roof", "background"}
)


def _scene_domain(scene_model: str) -> str:
    if scene_model.endswith("_int"):
        return "residential_scan"
    prefix = scene_model.split("_", 1)[0]
    return {
        "gates": "residential_synthetic",
        "house": "residential_synthetic",
        "hotel": "hospitality",
        "restaurant": "restaurant",
        "grocery": "retail",
        "office": "office",
        "school": "school",
        "hall": "public_hall",
    }.get(prefix, "other_indoor")


def _scene_classification(
    *, room_count: int, usable_object_count: int, usable_category_count: int
) -> str:
    rich = usable_object_count >= 20 and usable_category_count >= 10
    if rich and room_count >= 2:
        return "multi_room_rich"
    if rich:
        return "single_room_rich"
    return "sparse"


def _scene_record(scene_root: Path, assets_root: Path) -> dict[str, Any]:
    scene_model = scene_root.name
    source_files = sorted((scene_root / "json").glob("*.json"))
    if len(source_files) != 1:
        raise ValueError(f"scene {scene_model} must expose exactly one JSON instance")
    source_file = source_files[0]
    payload = json.loads(source_file.read_text(encoding="utf-8"))
    init_info = payload.get("objects_info", {}).get("init_info")
    registry = payload.get("state", {}).get("registry", {}).get("object_registry")
    if not isinstance(init_info, dict) or not isinstance(registry, dict):
        raise ValueError(f"scene {scene_model} has an unsupported metadata schema")

    categories: Counter[str] = Counter()
    usable_categories: Counter[str] = Counter()
    rooms: set[str] = set()
    rearrangeable_count = 0
    for object_name, item in init_info.items():
        args = item.get("args", {}) if isinstance(item, dict) else {}
        category = str(args.get("category", "object"))
        categories[category] += 1
        in_rooms = args.get("in_rooms") or []
        if isinstance(in_rooms, str):
            in_rooms = [in_rooms]
        rooms.update(str(room).strip() for room in in_rooms if str(room).strip())
        if category not in STRUCTURE_CATEGORIES:
            usable_categories[category] += 1
            state = registry.get(object_name, {})
            has_pose = isinstance(state, dict) and isinstance(state.get("root_link"), dict)
            if not bool(args.get("fixed_base", False)) and not bool(
                args.get("visual_only", False)
            ) and has_pose:
                rearrangeable_count += 1

    floor_files = sorted((scene_root / "layout").glob("floor_trav_[0-9]*.png"))
    scene_kind = "garden" if scene_model.endswith("_garden") else "indoor"
    classification = _scene_classification(
        room_count=len(rooms),
        usable_object_count=sum(usable_categories.values()),
        usable_category_count=len(usable_categories),
    )
    versions = payload.get("versions", {})
    asset_version = versions.get("behavior-1k-assets", {}).get("version")
    return {
        "scene_model": scene_model,
        "scene_kind": scene_kind,
        "domain": _scene_domain(scene_model) if scene_kind == "indoor" else "garden",
        "classification": classification,
        "source_instance": source_file.stem.removeprefix(f"{scene_model}_"),
        "source_json": str(source_file.relative_to(assets_root)),
        "source_json_sha256": sha256_file(source_file),
        "source_asset_version": asset_version,
        "floor_count": len(floor_files),
        "room_count": len(rooms),
        "room_types": sorted({room.rsplit("_", 1)[0] for room in rooms}),
        "object_count": len(init_info),
        "category_count": len(categories),
        "usable_object_count": sum(usable_categories.values()),
        "usable_category_count": len(usable_categories),
        "rearrangeable_object_count": rearrangeable_count,
        "eligibility": {
            "static_multiview": scene_kind == "indoor" and bool(floor_files),
            "episode_rich": classification == "multi_room_rich",
            "rearrangement": scene_kind == "indoor" and rearrangeable_count >= 5,
        },
    }


def build_scene_inventory(assets_root: Path, output_path: Path | None = None) -> dict[str, Any]:
    """Return a deterministic inventory and optionally write it atomically."""

    assets_root = assets_root.resolve()
    scenes_root = assets_root / "scenes"
    if not scenes_root.is_dir():
        raise FileNotFoundError(f"BEHAVIOR scenes directory is missing: {scenes_root}")
    scenes = [
        _scene_record(path, assets_root)
        for path in sorted(scenes_root.iterdir())
        if path.is_dir()
    ]
    indoor = [scene for scene in scenes if scene["scene_kind"] == "indoor"]
    classifications = Counter(scene["classification"] for scene in indoor)
    summary = {
        "scene_count": len(scenes),
        "indoor_scene_count": len(indoor),
        "garden_scene_count": len(scenes) - len(indoor),
        "static_multiview_eligible_count": sum(
            bool(scene["eligibility"]["static_multiview"]) for scene in scenes
        ),
        "rearrangement_eligible_count": sum(
            bool(scene["eligibility"]["rearrangement"]) for scene in scenes
        ),
        "indoor_classification_counts": dict(sorted(classifications.items())),
        "indoor_aggregate": {
            "object_count": sum(scene["object_count"] for scene in indoor),
            "usable_object_count": sum(scene["usable_object_count"] for scene in indoor),
            "rearrangeable_object_count": sum(
                scene["rearrangeable_object_count"] for scene in indoor
            ),
            "mean_room_count": round(
                sum(scene["room_count"] for scene in indoor) / len(indoor), 3
            ),
            "mean_usable_category_count": round(
                sum(scene["usable_category_count"] for scene in indoor) / len(indoor), 3
            ),
        },
    }
    inventory = {
        "schema_version": "omnigibson_scene_inventory.v1",
        "inventory_id": "behavior-1k-v3.9.0-installed-scenes-v1",
        "license_boundary": {
            "contains_external_assets": False,
            "redistribution": "aggregate_metadata_only",
            "source_assets_required_to_reproduce": True,
        },
        "summary": summary,
        "scenes": scenes,
    }
    if output_path is not None:
        write_json_atomic(output_path, inventory)
    return inventory
