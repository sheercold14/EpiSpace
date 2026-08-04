from __future__ import annotations

import json
from pathlib import Path

from omnigibson_episode.model_swap_plan import plan_model_swap_interventions


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_model_swap_planner_preserves_category_and_relation(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    _write(
        source,
        {
            "objects_info": {
                "init_info": {
                    "chair_0": {
                        "args": {
                            "category": "chair",
                            "model": "aaaaaa",
                            "fixed_base": False,
                            "visual_only": False,
                        }
                    },
                    "table_0": {"args": {"category": "table", "model": "cccccc"}},
                }
            }
        },
    )
    _write(tmp_path / "scene_snapshot.json", {"source_path": str(source)})
    _write(
        tmp_path / "scene_ir.json",
        {
            "scene_id": "scene",
            "entities": [
                {
                    "entity_id": "chair-id",
                    "source_entity_id": "chair_0",
                    "raw_label": "chair",
                    "region_id": "room",
                    "obb": {
                        "center_m": [-1.5, 0.0, 0.5],
                        "half_extents_m": [0.4, 0.4, 0.5],
                    },
                },
                {
                    "entity_id": "table-id",
                    "source_entity_id": "table_0",
                    "raw_label": "table",
                    "region_id": "room",
                    "obb": {
                        "center_m": [0.0, 0.0, 0.5],
                        "half_extents_m": [0.5, 0.5, 0.5],
                    },
                },
            ],
        },
    )
    _write(
        tmp_path / "spatial_episode.json",
        {
            "episode_id": "episode",
            "observations": [
                {
                    "view_id": "view-000",
                    "visible_entity_ids": ["chair-id", "table-id"],
                    "world_from_camera": {"rotation_xyzw": [0.0, 0.0, 0.0, 1.0]},
                },
                {
                    "view_id": "view-001",
                    "visible_entity_ids": ["chair-id"],
                    "world_from_camera": {"rotation_xyzw": [0.0, 0.0, 0.0, 1.0]},
                },
            ],
        },
    )
    inventory = tmp_path / "inventory.json"
    _write(
        inventory,
        {
            "models": [
                {
                    "category": "chair",
                    "model": "aaaaaa",
                    "bbox_size_m": [0.8, 0.8, 1.0],
                    "minimal_pair_eligible": True,
                    "collision_ready": True,
                    "placement_tier": "floor",
                    "link_count": 1,
                },
                {
                    "category": "chair",
                    "model": "bbbbbb",
                    "bbox_size_m": [0.9, 0.7, 1.1],
                    "minimal_pair_eligible": True,
                    "collision_ready": True,
                    "placement_tier": "floor",
                    "link_count": 1,
                },
                {
                    "category": "chair",
                    "model": "dddddd",
                    "bbox_size_m": [1.1, 0.6, 0.9],
                    "minimal_pair_eligible": True,
                    "collision_ready": True,
                    "placement_tier": "floor",
                    "link_count": 1,
                },
            ]
        },
    )

    manifest = plan_model_swap_interventions(
        bundle_directory=tmp_path,
        object_inventory_path=inventory,
    )

    assert manifest["proposal_count"] == 1
    proposal = manifest["proposals"][0]
    assert proposal["source_model"] == "aaaaaa"
    assert proposal["replacement_model"] == "bbbbbb"
    assert proposal["difficulty_tier"] == "appearance_invariance_conservative"
    assert proposal["baseline_overlap_count"] == 0
    assert proposal["minimum_nonoverlap_clearance_m"] == 99.0
    assert manifest["candidate_policy"] == {
        "difficulty_tier": "appearance_invariance_conservative",
        "maximum_baseline_overlap_count": 1,
        "minimum_nonoverlap_clearance_m": 0.05,
        "maximum_aspect_ratio_factor": 1.5,
        "execution_mode": "physics_settle",
        "settle_steps": 12,
    }
    assert proposal["execution_mode"] == "physics_settle"
    assert proposal["settle_steps"] == 12
    assert proposal["relation_before"] == proposal["relation_after"] == "left_of"
    assert proposal["target_bbox_extent_m"] == [0.8, 0.8, 1.0]
