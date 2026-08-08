"""Version-lineage regression tests for compile standards."""

from __future__ import annotations

from dataclasses import fields

from spatial_episode.scriptgen.standards import (
    COMPATIBLE_PLAN_STANDARDS,
    STD_V4_ADDED_FIELDS,
    CompileStandard,
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


def test_std_v4_is_declared_pure_extension_of_v3() -> None:
    actual = {field.name for field in fields(CompileStandard)}
    assert actual == STD_V3_FIELDS | set(STD_V4_ADDED_FIELDS)
    assert COMPATIBLE_PLAN_STANDARDS == {"std.v4": ("std.v3",)}
