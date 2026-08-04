from __future__ import annotations

import pytest

from omnigibson_episode.reasoning_tasks import (
    _metric_graph,
    _model_visible_entity_references,
    _perspective_graph,
    _perspective_quadrant,
    _relation_graph,
    _view_relation,
)


def _entity(label: str) -> dict[str, object]:
    return {"raw_label": label}


def _observation(view_id: str, entity_ids: list[str]) -> dict[str, object]:
    return {"view_id": view_id, "visible_entity_ids": entity_ids}


def test_repeated_entity_names_use_model_visible_view_anchors() -> None:
    referenced, appearances, metadata = _model_visible_entity_references(
        {
            "chair-a": _entity("straight_chair"),
            "chair-b": _entity("straight_chair"),
            "lamp": _entity("floor_lamp"),
        },
        [
            _observation("view-000", ["chair-a", "lamp"]),
            _observation("view-001", ["chair-a", "chair-b"]),
            _observation("view-002", ["chair-b"]),
        ],
    )

    assert referenced["chair-a"]["_display_name"] == "第1个视角里看到的straight_chair"
    assert referenced["chair-b"]["_display_name"] == "第3个视角里看到的straight_chair"
    assert referenced["lamp"]["_display_name"] == "floor_lamp"
    assert appearances["chair-a"] == [0, 1]
    assert metadata["chair-b"] == {
        "surface_name": "第3个视角里看到的straight chair",
        "basis": "single_instance_in_anchor_view",
        "anchor_view_id": "view-002",
    }


def test_repeated_entities_without_a_unique_visible_anchor_are_excluded() -> None:
    referenced, _, _ = _model_visible_entity_references(
        {"chair-a": _entity("chair"), "chair-b": _entity("chair")},
        [_observation("view-000", ["chair-a", "chair-b"])],
    )

    assert referenced == {}


def test_view_relation_respects_camera_frame() -> None:
    relation, margin, right, front = _view_relation(
        [2.0, 0.2, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    )
    assert relation == "right_of"
    assert margin == pytest.approx(1.8)
    assert right == pytest.approx(2.0)
    assert front == pytest.approx(0.2)


def test_perspective_quadrant_uses_object_defined_heading() -> None:
    result = _perspective_quadrant(
        viewpoint=[0.0, 0.0, 0.0],
        facing=[0.0, 2.0, 0.0],
        target=[-1.0, 1.0, 0.0],
    )
    assert result is not None
    assert result[0] == "front_left"


def test_generated_operation_graphs_are_typed() -> None:
    metric = _metric_graph("entity-a", "entity-b")
    relation = _relation_graph("entity-a", "entity-b", "left_of")
    perspective = _perspective_graph(
        ("entity-a", "entity-b", "entity-c"), "front_right"
    )
    assert metric["answer_node"] == "verify_metric"
    assert relation["answer_node"] == "verify_relation"
    assert perspective["answer_node"] == "verify_perspective"
