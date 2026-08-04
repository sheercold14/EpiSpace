from __future__ import annotations

from omnigibson_episode.intervention_acquire import (
    _execution_checks,
    _model_swap_scene_file,
    _snapshot_overlap_names,
    relation_from_centers,
)


def _proposal() -> dict[str, object]:
    return {
        "relation_axis": "x",
        "relation_after": "right_of",
    }


def test_relation_from_centers_uses_declared_axis_and_dominance_margin() -> None:
    relation, margin = relation_from_centers(
        proposal=_proposal(),
        target_center=[2.0, 0.2, 0.5],
        anchor_center=[0.0, 0.0, 0.5],
    )

    assert relation == "right_of"
    assert margin == 1.8


def test_execution_checks_reject_causal_leakage() -> None:
    checks = _execution_checks(
        proposal=_proposal(),
        target_center=[2.0, 0.2, 0.5],
        anchor_center=[0.0, 0.0, 0.5],
        proposed_room="living_room_0",
        expected_room="living_room_0",
        target_settle_drift_m=0.01,
        maximum_non_target_position_drift_m=0.03,
        maximum_non_target_orientation_drift_deg=0.1,
        overlap_names=[],
        target_visible_after=True,
    )

    by_name = {item["name"]: item for item in checks}
    assert by_name["declared_relation_after"]["passed"]
    assert by_name["target_visible_after"]["passed"]
    assert not by_name["non_target_position_invariance_m"]["passed"]


def test_model_swap_scene_file_changes_only_target_asset_arguments(
    tmp_path,
) -> None:
    import json

    source = tmp_path / "scene.json"
    source.write_text(
        json.dumps(
            {
                "objects_info": {
                    "init_info": {
                        "chair_0": {
                            "args": {
                                "category": "chair",
                                "model": "aaaaaa",
                                "scale": [1.0, 1.0, 1.0],
                                "expected_file_hash": "old",
                            }
                        },
                        "table_0": {"args": {"category": "table", "model": "cccccc"}},
                    }
                },
                "state": {"unchanged": True},
            }
        ),
        encoding="utf-8",
    )
    result = _model_swap_scene_file(
        {"source_path": str(source)},
        {
            "target_source_entity_id": "chair_0",
            "target_category": "chair",
            "source_model": "aaaaaa",
            "replacement_model": "bbbbbb",
            "target_bbox_extent_m": [0.8, 0.8, 1.0],
        },
    )

    args = result["objects_info"]["init_info"]["chair_0"]["args"]
    assert args["model"] == "bbbbbb"
    assert args["bounding_box"] == [0.8, 0.8, 1.0]
    assert "scale" not in args
    assert "expected_file_hash" not in args
    assert result["objects_info"]["init_info"]["table_0"]["args"]["model"] == "cccccc"
    assert result["state"] == {"unchanged": True}


def test_overlap_audit_preserves_baseline_contacts() -> None:
    snapshot = {
        "entities": [
            {
                "source_entity_id": "chair_0",
                "category": "chair",
                "aabb_center_m": [0.0, 0.0, 0.5],
                "aabb_extent_m": [1.0, 1.0, 1.0],
            },
            {
                "source_entity_id": "table_0",
                "category": "table",
                "aabb_center_m": [0.0, 0.0, 0.5],
                "aabb_extent_m": [1.0, 1.0, 1.0],
            },
            {
                "source_entity_id": "cabinet_0",
                "category": "cabinet",
                "aabb_center_m": [0.7, 0.0, 0.5],
                "aabb_extent_m": [0.6, 0.6, 1.0],
            },
            {
                "source_entity_id": "wall_0",
                "category": "walls",
                "aabb_center_m": [0.0, 0.0, 0.5],
                "aabb_extent_m": [1.0, 1.0, 1.0],
            },
        ]
    }

    assert _snapshot_overlap_names(
        snapshot, target_name="chair_0", anchor_name="table_0"
    ) == {"cabinet_0"}
