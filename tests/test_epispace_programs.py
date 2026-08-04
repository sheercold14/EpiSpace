from __future__ import annotations

import json

import pytest

from episode3d.programs import program_for


@pytest.mark.parametrize(
    "task_type",
    [
        "grounding_presence",
        "last_seen_memory",
        "metric_distance",
        "egocentric_relation",
        "cross_view_relation",
        "cross_view_unknown",
        "evidence_presence_unknown",
        "evidence_presence_reveal",
        "counterfactual_verification",
        "object_centric_perspective",
        "unknown_abstention",
        "rotation_change_detection",
        "orbit_identity",
        "elevation_relation_transfer",
        "occlusion_unknown",
        "occlusion_reveal",
        "target_view_prediction",
    ],
)
def test_semantic_programs_are_typed_and_answer_free(task_type: str) -> None:
    payload = program_for(task_type).as_dict()
    serialized = json.dumps(payload)

    assert payload["answer_node"] == "answer"
    assert set(payload["atoms"]) <= {"G", "F", "B", "M", "R", "P", "V"}
    assert "expected_relation" not in serialized
    assert "expected_quadrant" not in serialized


def test_cross_view_and_ego_programs_have_distinct_semantics() -> None:
    ego = program_for("egocentric_relation")
    cross_view = program_for("cross_view_relation")

    assert ego.semantic_signature != cross_view.semantic_signature
    assert "F_ego" in ego.semantic_signature
    assert "F*_cross_view" in cross_view.semantic_signature
