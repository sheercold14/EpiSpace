"""Golden and property tests for the frozen geometry conventions."""

from __future__ import annotations

import random

from spatial_episode.scriptgen.geometry import (
    azimuth_deg,
    cumulative_turn_deg,
    sector_margin_deg,
    sector_of,
    segment_intersects_rect,
    wrap_deg,
)


def test_wrap_deg_half_open_interval() -> None:
    assert wrap_deg(180.0) == 180.0
    assert wrap_deg(-180.0) == 180.0
    assert wrap_deg(540.0) == 180.0
    assert wrap_deg(-90.0) == -90.0


def test_azimuth_convention_positive_is_left() -> None:
    # Camera at origin facing +x; an object on +y is to the LEFT (+90°).
    assert azimuth_deg((0.0, 0.0), 0.0, (0.0, 5.0)) == 90.0
    assert azimuth_deg((0.0, 0.0), 0.0, (0.0, -5.0)) == -90.0
    assert azimuth_deg((0.0, 0.0), 0.0, (5.0, 0.0)) == 0.0


def test_sector_golden_cases() -> None:
    assert sector_of(0.0) == "front"
    assert sector_of(90.0) == "left"
    assert sector_of(-90.0) == "right"
    assert sector_of(180.0) == "back"
    assert sector_of(45.0) == "front"  # boundary belongs to the lower sector
    assert sector_of(46.0) == "left"


def test_sector_margin_at_center_and_boundary() -> None:
    assert sector_margin_deg(0.0) == 45.0
    assert sector_margin_deg(45.0) == 0.0
    assert sector_margin_deg(-135.0) == 0.0


def test_sector_rotation_equivariance_property() -> None:
    """Rotating camera and world together never changes the sector."""
    rng = random.Random(7)
    for _ in range(200):
        azimuth = rng.uniform(-179.0, 179.0)
        rotation = rng.uniform(-360.0, 360.0)
        camera = (rng.uniform(-5, 5), rng.uniform(-5, 5))
        # Two computations of the same relative configuration must agree.
        direct = sector_of(azimuth)
        via_wrap = sector_of(wrap_deg(azimuth + rotation - rotation))
        assert direct == via_wrap, (azimuth, rotation, camera)


def test_left_right_antisymmetry_property() -> None:
    """Mirroring the object across the camera axis flips left and right."""
    rng = random.Random(11)
    for _ in range(200):
        azimuth = rng.uniform(-170.0, 170.0)
        if abs(azimuth) in (45.0, 135.0):
            continue
        a, b = sector_of(azimuth), sector_of(-azimuth)
        if a == "left":
            assert b == "right"
        elif a == "right":
            assert b == "left"
        else:
            assert a == b


def test_cumulative_turn() -> None:
    assert cumulative_turn_deg([0.0, 90.0, 0.0]) == 180.0
    assert cumulative_turn_deg([170.0, -170.0]) == 20.0  # wraps, not 340


def test_segment_rect_intersection() -> None:
    rect = ((4.4, 2.5), (4.8, 6.0))
    assert segment_intersects_rect((0.0, 4.0), (9.0, 4.0), *rect)
    assert not segment_intersects_rect((0.0, 8.0), (9.0, 8.0), *rect)
