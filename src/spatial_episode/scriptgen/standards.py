"""Versioned compile standards: every threshold used by predicates lives here.

A standard is frozen. Changing any value requires bumping ``standard_version``
so that every certificate and trajectory plan records exactly which numbers
produced it. Predicates must never hard-code thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompileStandard:
    """All numeric thresholds shared by search-phase and compile-phase checks.

    Visibility uses a double threshold: values at or above the ``*_min_visible``
    bound count as visible, values at or below ``*_max_invisible`` count as
    invisible, and anything in between is *ambiguous* — a frame that must not
    be used as evidence and rejects the surrounding question candidate.
    """

    standard_version: str = "std.v1"

    # --- geometry-backend visibility (search phase, render-free estimate) ---
    # Value is projected angular size ratio: object_size_m / distance_m.
    geom_min_visible_ratio: float = 0.10
    geom_max_invisible_ratio: float = 0.06
    # Camera model used by the geometry backend.
    fov_half_angle_deg: float = 45.0
    max_view_distance_m: float = 8.0
    # Fraction of line-of-sight sample rays that must be unobstructed.
    min_unoccluded_ratio: float = 0.5

    # --- render-backend visibility (compile phase, authoritative) ---
    render_min_visible_pixels: int = 900
    render_max_invisible_pixels: int = 300
    render_min_unoccluded_ratio: float = 0.30

    # --- disappearance ---
    absence_min_frames: int = 3

    # --- direction discretisation ---
    sector_count: int = 4  # front / left / back / right
    sector_margin_deg: float = 15.0

    # --- search phase tightening ---
    # Search-phase margins are tightened by this factor so that most candidates
    # survive the authoritative post-render re-check.
    search_tighten_factor: float = 1.2


STD_V1 = CompileStandard()
