import json
from pathlib import Path

import numpy as np
from PIL import Image

from omnigibson_episode.intervention_plan import (
    _layout_room_instance,
    plan_relation_flip_interventions,
)


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


def test_planner_proposes_one_object_relation_flip(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    _write(
        source,
        {
            "objects_info": {
                "init_info": {
                    "chair_0": {
                        "args": {
                            "category": "chair",
                            "model": "abc",
                            "fixed_base": False,
                            "visual_only": False,
                            "in_rooms": None,
                        }
                    },
                    "table_0": {"args": {"category": "table", "model": "def"}},
                }
            }
        },
    )
    _write(tmp_path / "scene_snapshot.json", {"source_path": str(source)})
    entities = [
        {
            "entity_id": "target-id",
            "source_entity_id": "chair_0",
            "raw_label": "chair",
            "region_id": "room",
            "obb": {"center_m": [-1.5, 0.0, 0.5], "half_extents_m": [0.4, 0.4, 0.5]},
        },
        {
            "entity_id": "anchor-id",
            "source_entity_id": "table_0",
            "raw_label": "table",
            "region_id": "room",
            "obb": {"center_m": [0.0, 0.0, 0.5], "half_extents_m": [0.5, 0.5, 0.5]},
        },
    ]
    _write(tmp_path / "scene_ir.json", {"scene_id": "scene", "entities": entities})
    _write(
        tmp_path / "spatial_episode.json",
        {
            "episode_id": "episode",
            "observations": [
                {
                    "view_id": "view-000",
                    "visible_entity_ids": ["target-id", "anchor-id"],
                    "world_from_camera": {"rotation_xyzw": [0.0, 0.0, 0.0, 1.0]},
                },
                {
                    "view_id": "view-001",
                    "visible_entity_ids": ["target-id"],
                    "world_from_camera": {"rotation_xyzw": [0.0, 0.0, 0.0, 1.0]},
                },
            ],
        },
    )
    _write(
        tmp_path / "trajectory_plan.json",
        {"views": [{"world_from_agent": {"translation_m": [0.0, 0.0, 0.0]}}]},
    )
    catalog = tmp_path / "objects.json"
    _write(
        catalog,
        {
            "models": [
                {
                    "category": "chair",
                    "model": "abc",
                    "minimal_pair_eligible": True,
                    "placement_tier": "floor",
                }
            ]
        },
    )
    manifest = plan_relation_flip_interventions(
        bundle_directory=tmp_path,
        object_inventory_path=catalog,
    )
    assert manifest["proposal_count"] == 1
    proposal = manifest["proposals"][0]
    assert proposal["relation_before"] == "left_of"
    assert proposal["relation_after"] == "right_of"
    assert proposal["target_source_entity_id"] == "chair_0"
    assert proposal["relation_frame"] == "first_view_yaw"


def test_layout_room_lookup_matches_omnigibson_world_frame(tmp_path: Path) -> None:
    scene = tmp_path / "assets" / "scenes" / "example"
    layout = scene / "layout"
    source = scene / "json" / "example.json"
    metadata = tmp_path / "assets" / "metadata"
    layout.mkdir(parents=True)
    source.parent.mkdir()
    metadata.mkdir()
    source.write_text("{}", encoding="utf-8")
    metadata.joinpath("room_categories.txt").write_text(
        "dining_room\nliving_room\n", encoding="utf-8"
    )
    instance_map = np.zeros((20, 20), dtype=np.uint8)
    instance_map[:, :10] = 1
    instance_map[:, 10:] = 2
    semantic_map = np.zeros((20, 20), dtype=np.uint8)
    semantic_map[:, :10] = 1
    semantic_map[:, 10:] = 2
    Image.fromarray(instance_map).save(layout / "floor_insseg_0.png")
    Image.fromarray(semantic_map).save(layout / "floor_semseg_0.png")

    assert _layout_room_instance(source, [-0.05, 0.0]) == "dining_room_0"
    assert _layout_room_instance(source, [0.05, 0.0]) == "living_room_0"
    assert _layout_room_instance(source, [1.0, 1.0]) is None
