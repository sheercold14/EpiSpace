from __future__ import annotations

import math

import pytest

from episode3d.transform_pilot.geometry import (
    direction_measurement,
    horizontal_obb_radius,
    wrap_degrees,
    yaw_facing,
)
from episode3d.transform_pilot.schemas import task_schema


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(0.0, 0.0), (180.0, -180.0), (270.0, -90.0), (-181.0, 179.0)],
)
def test_wrap_degrees_uses_one_half_open_interval(angle: float, expected: float) -> None:
    assert wrap_degrees(angle) == expected


def test_right_turn_sign_matches_acquisition_contract() -> None:
    # Before turning, a point on world +Y is straight ahead.
    before = direction_measurement(
        point_xy=(0.0, 2.0), origin_xy=(0.0, 0.0), camera_yaw_rad=0.0
    )
    assert before.label == "front"

    # A right turn is a negative mathematical camera-yaw update.  The same
    # world point is therefore on the observer's left after a 90° right turn.
    after = direction_measurement(
        point_xy=(0.0, 2.0),
        origin_xy=(0.0, 0.0),
        camera_yaw_rad=-math.pi / 2,
    )
    assert after.label == "left"
    assert after.right_m == pytest.approx(-2.0)
    assert after.front_m == pytest.approx(0.0, abs=1e-8)


def test_obb_radius_reduces_direction_confidence() -> None:
    point = (1.0, 2.0)
    center_only = direction_measurement(
        point_xy=point, origin_xy=(0.0, 0.0), camera_yaw_rad=0.0
    )
    with_extent = direction_measurement(
        point_xy=point,
        origin_xy=(0.0, 0.0),
        camera_yaw_rad=0.0,
        horizontal_radius_m=horizontal_obb_radius((1.0, 1.0, 1.0)),
    )
    assert with_extent.label == center_only.label
    assert with_extent.effective_margin_deg < center_only.effective_margin_deg


def test_yaw_facing_places_target_straight_ahead() -> None:
    origin = (1.5, -2.0)
    for target in ((1.5, 3.0), (4.0, -2.0), (-2.0, -2.0), (1.5, -5.0)):
        measurement = direction_measurement(
            point_xy=target,
            origin_xy=origin,
            camera_yaw_rad=yaw_facing(origin, target),
        )
        assert measurement.right_m == pytest.approx(0.0, abs=1e-8)
        assert measurement.front_m > 0.0


def test_object_anchored_direction_is_equivariant_to_global_rotation() -> None:
    anchor = (0.0, 0.0)
    reference = (0.0, -2.0)
    subject = (2.0, 0.0)
    baseline = direction_measurement(
        point_xy=subject,
        origin_xy=reference,
        camera_yaw_rad=yaw_facing(reference, anchor),
    )

    def rotate(point: tuple[float, float]) -> tuple[float, float]:
        return -point[1], point[0]

    rotated_anchor = rotate(anchor)
    rotated_reference = rotate(reference)
    rotated_subject = rotate(subject)
    rotated = direction_measurement(
        point_xy=rotated_subject,
        origin_xy=rotated_reference,
        camera_yaw_rad=yaw_facing(rotated_reference, rotated_anchor),
    )
    assert rotated.label == baseline.label
    assert rotated.bearing_deg == pytest.approx(baseline.bearing_deg)


def test_transform_task_schemas_expose_grounded_cot_roles_but_hide_oracle() -> None:
    for task_id in ("self_rotation_query.v1", "among5_layout.v1"):
        schema = task_schema(task_id)
        assert schema["assistant_roles"] == ["cue", "transform", "conclusion"]
        assert "camera_pose" in schema["oracle_only"]
        assert "camera_pose" not in schema["model_visible"]
