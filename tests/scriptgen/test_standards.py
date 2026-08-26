"""Version-lineage regression tests for compile standards."""

from __future__ import annotations

from dataclasses import fields

import pytest

from spatial_episode.scriptgen.standards import (
    COMPATIBLE_PLAN_STANDARDS,
    STD_V3,
    STD_V4_ADDED_FIELDS,
    STD_V5_ADDED_FIELDS,
    STD_V6_ADDED_FIELDS,
    STD_V7_ADDED_FIELDS,
    STD_V8_ADDED_FIELDS,
    STD_V10,
    STD_V10_ADDED_FIELDS,
    STD_V11,
    STD_V11_ADDED_FIELDS,
    CompileStandard,
    standard_for_version,
)

STD_V3_FIELDS = {
    "standard_version",
    "geom_min_visible_ratio",
    "geom_max_invisible_ratio",
    "fov_half_angle_deg",
    "max_view_distance_m",
    "min_unoccluded_ratio",
    "render_min_visible_pixels",
    "render_max_invisible_pixels",
    "render_min_unoccluded_ratio",
    "absence_min_frames",
    "sector_count",
    "sector_margin_deg",
    "max_step_turn_deg",
    "max_step_translation_m",
    "body_radius_m",
    "clearance_z_low_m",
    "clearance_z_high_m",
    "search_tighten_factor",
}


def test_std_v11_declares_its_compatible_lineage() -> None:
    actual = {field.name for field in fields(CompileStandard)}
    assert actual == (
        STD_V3_FIELDS
        | set(STD_V4_ADDED_FIELDS)
        | set(STD_V5_ADDED_FIELDS)
        | set(STD_V6_ADDED_FIELDS)
        | set(STD_V7_ADDED_FIELDS)
        | set(STD_V8_ADDED_FIELDS)
        | set(STD_V10_ADDED_FIELDS)
        | set(STD_V11_ADDED_FIELDS)
    )
    assert COMPATIBLE_PLAN_STANDARDS == {
        "std.v4": ("std.v3",),
        "std.v5": ("std.v4", "std.v3"),
        "std.v6": ("std.v5", "std.v4", "std.v3"),
        "std.v7": ("std.v6", "std.v5", "std.v4", "std.v3"),
        "std.v8": ("std.v7", "std.v6", "std.v5", "std.v4", "std.v3"),
        "std.v9": ("std.v8", "std.v7", "std.v6", "std.v5", "std.v4", "std.v3"),
        "std.v10": (),
        # std.v11 re-judges the imagined curve over eight offsets: no ancestors.
        "std.v11": (),
    }


def test_version_dispatch_requires_explicit_legacy_choice() -> None:
    assert standard_for_version("std.v11") is STD_V11
    assert standard_for_version("std.v3", allow_legacy=True) is STD_V3
    assert standard_for_version("std.v10", allow_legacy=True) is STD_V10
    assert STD_V3.imagined_viewpoint_offsets_deg == (0, 45, 90, 135, 180)
    assert STD_V10.imagined_viewpoint_offsets_deg == (0, 45, 90, 135, 180)
    with pytest.raises(ValueError, match="requires explicit"):
        standard_for_version("std.v3")
    with pytest.raises(ValueError, match="unsupported"):
        standard_for_version("std.v9", allow_legacy=True)
