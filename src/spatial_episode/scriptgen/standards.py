"""Versioned compile standards: every threshold used by predicates lives here.

A standard is frozen. Changing any value requires bumping ``standard_version``
so that every certificate and trajectory plan records exactly which numbers
produced it. Predicates must never hard-code thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class CompileStandard:
    """All numeric thresholds shared by search-phase and compile-phase checks.

    Visibility uses a double threshold: values at or above the ``*_min_visible``
    bound count as visible, values at or below ``*_max_invisible`` count as
    invisible, and anything in between is *ambiguous* — a frame that must not
    be used as evidence and rejects the surrounding question candidate.
    """

    standard_version: str = "std.v11"

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

    # --- rendered occluder attribution (std.v10) ---
    occlusion_center_patch_radius_px: int = 4
    occlusion_min_support_pixels: int = 9
    occlusion_min_depth_margin_m: float = 0.10
    occlusion_min_dominance_ratio: float = 2.0
    # Geometry search treats an OBB as a conservative proxy for a mesh.  A
    # sightline that only clips the very top of that box is not reliable
    # evidence of a real rendered occlusion (pillows and chair backs make the
    # fitted box substantially taller than the solid mesh in some scenes).
    geom_occlusion_vertical_margin_m: float = 0.15
    geom_occlusion_footprint_margin_ratio: float = 0.35

    # --- disappearance ---
    absence_min_frames: int = 3

    # --- direction discretisation ---
    sector_count: int = 4  # front / left / back / right
    sector_margin_deg: float = 15.0

    # --- ego-motion trackability (std.v2) ---
    # Self-motion updating is only a fair question when the camera's own
    # motion can be tracked from the image stream: consecutive frames need
    # visual overlap. With a 90-degree horizontal FOV, capping per-frame yaw
    # change at 40 degrees preserves >50% overlap; translation is capped so
    # the scene does not jump discontinuously between frames.
    max_step_turn_deg: float = 40.0
    max_step_translation_m: float = 1.2

    # --- traversability clearance (std.v3) ---
    body_radius_m: float = 0.30
    clearance_z_low_m: float = 0.10
    clearance_z_high_m: float = 1.70

    # --- answer boundary margins (std.v4) ---
    view_side_margin_deg: float = 15.0
    net_turn_margin_deg: float = 15.0
    homing_min_distance_m: float = 1.0

    # --- self-motion subtype envelopes (std.v5) ---
    pure_rotation_max_displacement_m: float = 0.10
    pure_translation_max_turn_deg: float = 2.0
    net_turn_magnitude_deg: float = 90.0
    turn_segment_min_step_deg: float = 5.0
    multi_turn_min_segments: int = 2
    multi_turn_max_segments: int = 3
    camera_height_m: float = 1.5

    # --- imagined-viewpoint transformation (std.v6) ---
    landmark_min_visible_frames: int = 2
    imagined_min_anchor_distance_m: float = 1.0
    # Extended in std.v11 with the mirrored negative offsets. The curve
    # predicates iterate this tuple, so the extension changes how old plans
    # would be judged — std.v11 therefore declares no compatible ancestors.
    imagined_viewpoint_offsets_deg: tuple[int, ...] = (
        0,
        45,
        90,
        135,
        180,
        -45,
        -90,
        -135,
    )

    # --- cross-view landmark binding (std.v7) ---
    chain_min_covisible_frames: int = 2
    closer_min_distance_ratio: float = 1.25
    landmark_chain_lengths: tuple[int, ...] = (1, 2, 3)

    # --- snapshot cross-view integration (std.v11) ---
    snapshot_frames_per_station: int = 2
    snapshot_yaw_jitter_deg: float = 5.0

    # --- evidence coverage and absence calibration (std.v8) ---
    # Lower bounds for analysis buckets; the highest tier licenses absence.
    coverage_ratio_levels: tuple[float, ...] = (0.50, 0.70, 0.85)
    # Uniform conservative proxy from decision 5: if an unseen connected
    # region can contain even this 1 m footprint, absence is not licensed.
    coverage_hidden_object_size_m: float = 1.0

    # --- search phase tightening ---
    # Search-phase margins are tightened by this factor so that most candidates
    # survive the authoritative post-render re-check.
    search_tighten_factor: float = 1.2


# ``CompileStandard`` remains one superset data shape so predicates can stay
# simple, but these instances freeze the values that are allowed at each
# supported compile boundary.  Fields introduced after a legacy version are
# inert because a legacy compile is paired with its frozen legacy script.
STD_V3 = CompileStandard(
    standard_version="std.v3",
    imagined_viewpoint_offsets_deg=(0, 45, 90, 135, 180),
)
STD_V10 = CompileStandard(
    standard_version="std.v10",
    imagined_viewpoint_offsets_deg=(0, 45, 90, 135, 180),
)
STD_V11 = CompileStandard()

# Historical public name retained for callers.  New planning and rendering
# must always use the current standard through this alias.
STD_V1 = STD_V11

FROZEN_STANDARDS = MappingProxyType(
    {
        STD_V3.standard_version: STD_V3,
        STD_V10.standard_version: STD_V10,
        STD_V11.standard_version: STD_V11,
    }
)


def standard_for_version(
    version: str, *, allow_legacy: bool = False
) -> CompileStandard:
    """Return a frozen standard without silently opting into old semantics.

    Current callers need no flag.  A caller replaying an immutable old render
    must make the legacy choice explicit and pair the result with a versioned
    legacy script.  This is intentionally separate from plan-lineage
    compatibility: selecting ``std.v3`` does not make a v3 plan compatible
    with the current std.v11 compiler.
    """

    if version == STD_V11.standard_version:
        return STD_V11
    if not allow_legacy:
        raise ValueError(
            f"legacy standard {version!r} requires explicit allow_legacy=True"
        )
    try:
        return FROZEN_STANDARDS[version]
    except KeyError as error:
        raise ValueError(f"unsupported compile standard: {version!r}") from error


# The lineage is deliberately non-transitive. A prior plan version belongs
# here only when the current standard is a pure extension of it.
COMPATIBLE_PLAN_STANDARDS: dict[str, tuple[str, ...]] = {
    # std.v4 only adds the three answer thresholds below; every v3-era search
    # promise is judged by unchanged fields and values.
    "std.v4": ("std.v3",),
    # std.v5 only adds subtype fields. It is independently a pure extension
    # of both v4 and the wallfix plans authored under v3.
    "std.v5": ("std.v4", "std.v3"),
    # std.v6 adds only reference-frame fields.
    "std.v6": ("std.v5", "std.v4", "std.v3"),
    # std.v7 adds only cross-view binding fields.
    "std.v7": ("std.v6", "std.v5", "std.v4", "std.v3"),
    # std.v8 adds only evidence-coverage fields.
    "std.v8": ("std.v7", "std.v6", "std.v5", "std.v4", "std.v3"),
    # std.v9 removes the false +/-180-degree boundary at the center of the
    # back sector. This only relaxes false rejections: every plan accepted by
    # v3-v8 remains valid under the corrected margin geometry.
    "std.v9": ("std.v8", "std.v7", "std.v6", "std.v5", "std.v4", "std.v3"),
    # std.v10 strengthens the meaning of an occlusion clause with rendered
    # instance/depth attribution. Older plans must be regenerated explicitly.
    "std.v10": (),
    # std.v11 extends imagined_viewpoint_offsets_deg with negative offsets.
    # The imagined-curve predicates iterate that tuple, so plans qualified
    # under the five-point curve would be re-judged over eight points: not a
    # pure extension, hence no compatible ancestors.
    "std.v11": (),
}

STD_V4_ADDED_FIELDS = (
    "view_side_margin_deg",
    "net_turn_margin_deg",
    "homing_min_distance_m",
)

STD_V5_ADDED_FIELDS = (
    "pure_rotation_max_displacement_m",
    "pure_translation_max_turn_deg",
    "net_turn_magnitude_deg",
    "turn_segment_min_step_deg",
    "multi_turn_min_segments",
    "multi_turn_max_segments",
    "camera_height_m",
)

STD_V6_ADDED_FIELDS = (
    "landmark_min_visible_frames",
    "imagined_min_anchor_distance_m",
    "imagined_viewpoint_offsets_deg",
)

STD_V7_ADDED_FIELDS = (
    "chain_min_covisible_frames",
    "closer_min_distance_ratio",
    "landmark_chain_lengths",
)

STD_V8_ADDED_FIELDS = (
    "coverage_ratio_levels",
    "coverage_hidden_object_size_m",
)

STD_V10_ADDED_FIELDS = (
    "occlusion_center_patch_radius_px",
    "occlusion_min_support_pixels",
    "occlusion_min_depth_margin_m",
    "occlusion_min_dominance_ratio",
    "geom_occlusion_vertical_margin_m",
    "geom_occlusion_footprint_margin_ratio",
)

STD_V11_ADDED_FIELDS = (
    "snapshot_frames_per_station",
    "snapshot_yaw_jitter_deg",
)
