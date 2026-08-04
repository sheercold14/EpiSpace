"""Inventory installed BEHAVIOR object models for controlled interventions."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from omnigibson_episode.io import write_json_atomic
from omnigibson_episode.scene_inventory import STRUCTURE_CATEGORIES


def _placement_tier(dimensions: list[float], eligible: bool) -> str:
    if not eligible:
        return "exclude"
    maximum = max(dimensions)
    volume = math.prod(dimensions)
    if maximum <= 0.6 and volume <= 0.125:
        return "tabletop"
    if maximum <= 2.5:
        return "floor"
    return "large_floor"


def build_object_inventory(
    assets_root: Path,
    output_path: Path | None = None,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    assets_root = assets_root.resolve()
    objects_root = assets_root / "objects"
    if not objects_root.is_dir():
        raise FileNotFoundError(f"BEHAVIOR objects directory is missing: {objects_root}")
    records: list[dict[str, Any]] = []
    for category_root in sorted(path for path in objects_root.iterdir() if path.is_dir()):
        category = category_root.name
        for model_root in sorted(path for path in category_root.iterdir() if path.is_dir()):
            metadata_path = model_root / "misc" / "metadata.json"
            usd_files = sorted((model_root / "usd").glob("*.usd"))
            parse_error = None
            metadata: dict[str, Any] = {}
            if metadata_path.is_file():
                try:
                    loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        metadata = loaded
                    else:
                        parse_error = "metadata is not an object"
                except (OSError, json.JSONDecodeError) as error:
                    parse_error = str(error)
            else:
                parse_error = "metadata missing"
            dimensions_raw = metadata.get("bbox_size")
            dimensions = (
                [float(value) for value in dimensions_raw]
                if isinstance(dimensions_raw, list)
                and len(dimensions_raw) == 3
                and all(float(value) > 0 for value in dimensions_raw)
                else None
            )
            link_boxes = metadata.get("link_bounding_boxes", {})
            collision_links = sum(
                isinstance(value, dict) and isinstance(value.get("collision"), dict)
                for value in link_boxes.values()
            ) if isinstance(link_boxes, dict) else 0
            renderable = bool(usd_files)
            collision_ready = collision_links > 0
            scale_valid = bool(
                dimensions
                and min(dimensions) >= 0.01
                and max(dimensions) <= 3.0
                and math.prod(dimensions) <= 8.0
            )
            eligible = bool(
                renderable
                and collision_ready
                and scale_valid
                and category not in STRUCTURE_CATEGORIES
                and parse_error is None
            )
            orientations = metadata.get("orientations", [])
            record = {
                "category": category,
                "model": model_root.name,
                "asset_id": f"{category}/{model_root.name}",
                "model_root": str(model_root.relative_to(assets_root)),
                "usd": (
                    str(usd_files[0].relative_to(assets_root)) if usd_files else None
                ),
                "metadata": (
                    str(metadata_path.relative_to(assets_root))
                    if metadata_path.is_file()
                    else None
                ),
                "bbox_size_m": dimensions,
                "renderable": renderable,
                "collision_ready": collision_ready,
                "link_count": len(metadata.get("meta_links", {})),
                "collision_link_count": collision_links,
                "orientation_evidence_count": (
                    len(orientations) if isinstance(orientations, list) else 0
                ),
                "minimal_pair_eligible": eligible,
                "placement_tier": _placement_tier(dimensions or [math.inf] * 3, eligible),
                "exclusion_reason": (
                    None
                    if eligible
                    else parse_error
                    or ("USD missing" if not renderable else None)
                    or ("collision metadata missing" if not collision_ready else None)
                    or ("scale outside conservative gate" if not scale_valid else None)
                    or "structural category"
                ),
            }
            records.append(record)

    tiers = Counter(record["placement_tier"] for record in records)
    exclusion_reasons = Counter(
        record["exclusion_reason"] for record in records if record["exclusion_reason"]
    )
    eligible_categories = {
        record["category"] for record in records if record["minimal_pair_eligible"]
    }
    summary = {
        "category_count": len({record["category"] for record in records}),
        "model_count": len(records),
        "renderable_model_count": sum(record["renderable"] for record in records),
        "collision_ready_model_count": sum(record["collision_ready"] for record in records),
        "minimal_pair_eligible_model_count": sum(
            record["minimal_pair_eligible"] for record in records
        ),
        "minimal_pair_eligible_category_count": len(eligible_categories),
        "placement_tier_counts": dict(sorted(tiers.items())),
        "exclusion_reason_counts": dict(sorted(exclusion_reasons.items())),
    }
    inventory = {
        "schema_version": "omnigibson_object_inventory.v1",
        "inventory_id": "behavior-1k-v3.9.0-object-models-v1",
        "source_asset_version": (
            (assets_root / "VERSION").read_text(encoding="utf-8").strip()
            if (assets_root / "VERSION").is_file()
            else None
        ),
        "license_boundary": {
            "contains_external_assets": False,
            "redistribution": "local_metadata_index",
            "source_assets_required_to_reproduce": True,
        },
        "eligibility_policy": {
            "purpose": (
                "candidate generation only; simulator placement must still pass "
                "physics and visibility gates"
            ),
            "requires": [
                "encrypted USD present",
                "collision bounding-box metadata",
                "all bbox dimensions in [0.01, 3.0] m",
                "bbox volume <= 8 m^3",
                "non-structural category",
            ],
        },
        "summary": summary,
        "models": records,
    }
    if output_path is not None:
        write_json_atomic(output_path, inventory)
    if summary_path is not None:
        write_json_atomic(
            summary_path,
            {
                "schema_version": "omnigibson_object_inventory_summary.v1",
                "inventory_id": inventory["inventory_id"],
                "source_asset_version": inventory["source_asset_version"],
                "eligibility_policy": inventory["eligibility_policy"],
                "summary": summary,
            },
        )
    return inventory
