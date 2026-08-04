"""Frozen task signatures for the first Transform Pilot census."""

from __future__ import annotations

TASK_SCHEMAS = {
    "self_rotation_query.v1": {
        "semantic_signature": (
            "G(ordered_rgb,entity_hint)->B_view_memory->"
            "F_self(turn_sequence)->R_bearing->V_direction"
        ),
        "model_visible": ["ordered_rgb", "natural_language_motion", "question"],
        "assistant_roles": ["cue", "transform", "conclusion"],
        "oracle_only": ["camera_pose", "entity_obb", "target_orientation_view"],
    },
    "among5_layout.v1": {
        "semantic_signature": (
            "G(view_set,anchor+satellites)->F_register(anchor)->"
            "B_layout5->R/P(query_frame)->V_answer"
        ),
        "model_visible": ["ordered_rgb", "question"],
        "assistant_roles": ["cue", "transform", "conclusion"],
        "oracle_only": ["camera_pose", "entity_obb", "canonical_layout"],
    },
}


def task_schema(task_id: str) -> dict[str, object]:
    try:
        return dict(TASK_SCHEMAS[task_id])
    except KeyError as error:
        raise ValueError(f"unknown Transform Pilot task: {task_id}") from error
