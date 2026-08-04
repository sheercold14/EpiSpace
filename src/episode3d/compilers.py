"""Trajectory-aware question and episode-family compilation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import replace
from itertools import combinations
from typing import Any

from episode3d.bundles import Bundle, stable_id
from episode3d.language import (
    entity_name,
    localize_categories,
    relation_name,
    view_number,
)
from episode3d.models import QuestionSpec, Rejection
from episode3d.programs import program_for

STRUCTURAL_LABELS = {
    "ceilings",
    "ceiling",
    "floor",
    "floors",
    "wall",
    "walls",
    "room",
    "background",
    "rail_fence",
}

MIN_REFERENT_PIXELS = 1_000
MIN_REFERENT_BBOX_SIDE_PX = 28
MIN_MASK_BBOX_FILL_RATIO = 0.10
SEVERE_MULTI_BORDER_OUTER_BAND_PIXEL_FRACTION = 0.35
SEVERE_SINGLE_BORDER_OUTER_BAND_PIXEL_FRACTION = 0.30
MAX_SEVERELY_CLIPPED_AXIS_FRACTION = 0.20
MAX_RGB_DOMINANT_COLOR_FRACTION = 0.60
MIN_RGB_QUANTIZED_ENTROPY_BITS = 2.75
MAX_NEAR_NONSTRUCTURAL_PIXEL_FRACTION = 0.35
MAX_NEAR_ANY_GEOMETRY_PIXEL_FRACTION = 0.90
MAX_CLOSE_GEOMETRY_PIXEL_FRACTION = 0.90
MIN_NEAR_ENCLOSURE_PIXEL_FRACTION = 0.95
MAX_NEAR_ENCLOSURE_MEDIAN_DEPTH_M = 0.90
MAX_NEAR_ENCLOSURE_DEPTH_SPREAD_M = 0.60
MIN_DOMINANT_FOREGROUND_PIXEL_FRACTION = 0.60
MAX_DOMINANT_FOREGROUND_MEDIAN_DEPTH_M = 0.75
DOMINANT_FOREGROUND_REQUIRED_BORDER_SIDES = frozenset({"left", "right", "top"})
MIN_RAW_NEGATIVE_CATEGORY_PIXELS = 64
MIN_RAW_NEGATIVE_CATEGORY_BBOX_SIDE_PX = 8
MIN_T10_ANCHOR_PIXELS = 2_000
MIN_T10_ANCHOR_BBOX_SIDE_PX = 40
MAX_T10_ANCHOR_OUTER_BAND_PIXEL_FRACTION = 0.80
MIN_RELATION_DOMINANCE_RATIO = 1.25

# These groups are used only when asserting visual *absence*.  Exact simulator
# labels such as bookcase/shelf or fixed/openable window are often not reliably
# separable from RGB, so seeing a confusable sibling invalidates a negative
# evidence claim even when the oracle category token is absent.
VISUAL_CONFUSABILITY_GROUPS = (
    frozenset(
        {
            "bookcase",
            "shelf",
            "shelving_unit",
            "grocery_shelf",
            "commercial_kitchen_shelf",
            "dry_food_dispenser_shelf",
            "candy_dispenser_shelf",
        }
    ),
    frozenset({"fixed_window", "openable_window", "window"}),
    frozenset(
        {
            "coffee_table",
            "breakfast_table",
            "conference_table",
            "console_table",
            "lab_table",
            "commercial_kitchen_table",
            "pedestal_table",
            "table",
        }
    ),
    # The simulator ontology distinguishes these subclasses, but an RGB-only
    # observer cannot reliably do so without a logo, vent or readable control
    # panel.  They therefore form one visual category for absence/change gates.
    frozenset({"washer", "clothes_dryer"}),
    frozenset(
        {
            "cedar_chest",
            "bottom_cabinet",
            "bottom_cabinet_no_top",
            "metal_bottom_cabinet",
            "cabinet",
            "dresser",
            "chest_of_drawers",
        }
    ),
    frozenset({"fire_alarm", "fire_sprinkler", "smoke_detector"}),
    frozenset(
        {
            "downlight",
            "spotlight",
            "rectangular_light",
            "square_light",
            "room_light",
            "track_light",
        }
    ),
)

# Thin or glass openings are frequently absent from the instance channel even
# while a human can see a window-like opening.  Their simulator subtype is also
# not identifiable from RGB.  We therefore never manufacture absence/unknown
# supervision for this visual family; positive spatial questions remain usable.
VISUAL_ABSENCE_UNSAFE_LABELS = frozenset({"fixed_window", "openable_window", "window"})

# A coffee table taller than 65 cm is visually a dining/meeting table for the
# model-facing ontology, regardless of a noisy asset tag.  This is a semantic
# geometry sanity rule, not a question-specific blacklist.
CATEGORY_MAX_HEIGHT_M = {"coffee_table": 0.65}

SOURCE_ALLOWED_BY_CLASS = {
    "T1": {
        "grounding_presence",
        "last_seen_memory",
        "metric_distance",
        "egocentric_relation",
        "cross_view_relation",
        "counterfactual_verification",
        "object_centric_perspective",
        "unknown_abstention",
    },
    "T3": {
        "grounding_presence",
        "last_seen_memory",
        "egocentric_relation",
        "cross_view_relation",
        "counterfactual_verification",
        "object_centric_perspective",
        "unknown_abstention",
    },
    "T4": {
        "grounding_presence",
        "last_seen_memory",
        "metric_distance",
        "egocentric_relation",
        "cross_view_relation",
        "counterfactual_verification",
        "object_centric_perspective",
        "unknown_abstention",
    },
    "T7": {
        "grounding_presence",
        "metric_distance",
        "egocentric_relation",
        "object_centric_perspective",
        "unknown_abstention",
    },
    "T8": {
        "grounding_presence",
        "last_seen_memory",
        "metric_distance",
        "egocentric_relation",
        "cross_view_relation",
        "counterfactual_verification",
        "object_centric_perspective",
        "unknown_abstention",
    },
}

INSTANCE_BOUND_TASKS = {
    "last_seen_memory",
    "metric_distance",
    "egocentric_relation",
    "cross_view_relation",
    "counterfactual_verification",
    "object_centric_perspective",
    "orbit_identity",
}

# These tasks ask the model to preserve one physical identity across images.
# A unique category mention in the first image is not enough when a later image
# contains two equally valid category-level matches: the answer would then
# depend on the simulator instance id hidden from the model.
IDENTITY_TRACKING_TASKS = frozenset({"last_seen_memory", "orbit_identity"})


def _model_input_frame_quality(bundle: Bundle, view_id: str) -> dict[str, Any]:
    """Return oracle-only admission for one actual model frame.

    RGB entropy catches wall/mesh degeneration, while raw metric depth catches
    a different failure mode: a nominally diverse image rendered from inside a
    nearby object.  Both channels are used only to reject data.
    """

    rgb_stats = bundle.rgb_visual_stats(view_id)
    collision_stats = bundle.frame_collision_stats(view_id)
    context_stats = bundle.frame_context_stats(view_id)
    degenerate_rgb = (
        float(rgb_stats["dominant_quantized_color_fraction"])
        >= MAX_RGB_DOMINANT_COLOR_FRACTION
        and float(rgb_stats["quantized_color_entropy_bits"])
        < MIN_RGB_QUANTIZED_ENTROPY_BITS
    )
    legacy_camera_collision = (
        float(collision_stats["near_nonstructural_pixel_fraction"])
        >= MAX_NEAR_NONSTRUCTURAL_PIXEL_FRACTION
        or float(collision_stats["near_geometry_pixel_fraction"])
        >= MAX_NEAR_ANY_GEOMETRY_PIXEL_FRACTION
    )
    near_surface_saturation = (
        float(context_stats["close_geometry_pixel_fraction"])
        >= MAX_CLOSE_GEOMETRY_PIXEL_FRACTION
    )
    near_enclosure = (
        float(context_stats["enclosure_pixel_fraction"])
        >= MIN_NEAR_ENCLOSURE_PIXEL_FRACTION
        and float(context_stats["depth_median_m"])
        <= MAX_NEAR_ENCLOSURE_MEDIAN_DEPTH_M
        and float(context_stats["depth_p90_minus_p10_m"])
        <= MAX_NEAR_ENCLOSURE_DEPTH_SPREAD_M
    )
    foreground = context_stats["dominant_nonstructural_instance"]
    foreground_depth = foreground.get("median_depth_m")
    foreground_occlusion = (
        float(foreground["pixel_fraction"])
        >= MIN_DOMINANT_FOREGROUND_PIXEL_FRACTION
        and DOMINANT_FOREGROUND_REQUIRED_BORDER_SIDES.issubset(
            set(foreground["border_sides"])
        )
        and isinstance(foreground_depth, int | float)
        and not isinstance(foreground_depth, bool)
        and float(foreground_depth) <= MAX_DOMINANT_FOREGROUND_MEDIAN_DEPTH_M
    )
    camera_collision = (
        legacy_camera_collision
        or near_surface_saturation
        or near_enclosure
        or foreground_occlusion
    )
    return {
        **rgb_stats,
        "frame_collision_stats": collision_stats,
        "frame_context_stats": context_stats,
        "passed": not (degenerate_rgb or camera_collision),
        "failure_flags": {
            "degenerate_rgb": degenerate_rgb,
            "camera_collision": camera_collision,
            "legacy_camera_collision": legacy_camera_collision,
            "near_surface_saturation": near_surface_saturation,
            "near_enclosure": near_enclosure,
            "foreground_occlusion": foreground_occlusion,
        },
        "thresholds": {
            "maximum_rgb_dominant_color_fraction": (
                MAX_RGB_DOMINANT_COLOR_FRACTION
            ),
            "minimum_rgb_quantized_entropy_bits": MIN_RGB_QUANTIZED_ENTROPY_BITS,
            "maximum_near_nonstructural_pixel_fraction": (
                MAX_NEAR_NONSTRUCTURAL_PIXEL_FRACTION
            ),
            "maximum_near_any_geometry_pixel_fraction": (
                MAX_NEAR_ANY_GEOMETRY_PIXEL_FRACTION
            ),
            "maximum_close_geometry_pixel_fraction": (
                MAX_CLOSE_GEOMETRY_PIXEL_FRACTION
            ),
            "minimum_near_enclosure_pixel_fraction": (
                MIN_NEAR_ENCLOSURE_PIXEL_FRACTION
            ),
            "maximum_near_enclosure_median_depth_m": (
                MAX_NEAR_ENCLOSURE_MEDIAN_DEPTH_M
            ),
            "maximum_near_enclosure_depth_spread_m": (
                MAX_NEAR_ENCLOSURE_DEPTH_SPREAD_M
            ),
            "minimum_dominant_foreground_pixel_fraction": (
                MIN_DOMINANT_FOREGROUND_PIXEL_FRACTION
            ),
            "maximum_dominant_foreground_median_depth_m": (
                MAX_DOMINANT_FOREGROUND_MEDIAN_DEPTH_M
            ),
            "dominant_foreground_required_border_sides": sorted(
                DOMINANT_FOREGROUND_REQUIRED_BORDER_SIDES
            ),
        },
    }


def model_input_frame_is_admissible(bundle: Bundle, view_id: str) -> bool:
    """Return whether one frame may appear anywhere in a serialized model input."""

    return bool(_model_input_frame_quality(bundle, view_id)["passed"])


def _degenerate_model_input_views(
    bundle: Bundle, spec: QuestionSpec
) -> tuple[str, ...]:
    model_views = spec.model_view_ids or tuple(view.view_id for view in bundle.views)
    return tuple(
        view_id
        for view_id in model_views
        if not _model_input_frame_quality(bundle, view_id)["passed"]
    )


def _apply_model_input_frame_gate(
    bundle: Bundle,
    specs: list[QuestionSpec],
    rejected: list[Rejection],
) -> list[QuestionSpec]:
    """Fail closed on bad context frames and keep sibling families atomic."""

    invalid_by_fact = {
        spec.fact_id: _degenerate_model_input_views(bundle, spec) for spec in specs
    }
    invalid_groups = {
        spec.consistency_group
        for spec in specs
        if spec.consistency_group and invalid_by_fact[spec.fact_id]
    }
    retained: list[QuestionSpec] = []
    for spec in specs:
        invalid_views = invalid_by_fact[spec.fact_id]
        family_invalid = bool(
            spec.consistency_group and spec.consistency_group in invalid_groups
        )
        if not invalid_views and not family_invalid:
            retained.append(spec)
            continue
        detail = (
            f"actual model input contains inadmissible RGB/depth views={invalid_views}"
            if invalid_views
            else (
                "a sibling in consistency group "
                f"{spec.consistency_group} contains an inadmissible model input frame"
            )
        )
        rejected.append(
            Rejection(
                "question",
                spec.fact_id,
                "inadmissible_model_input_frame",
                detail,
                str(bundle.root),
            )
        )
    return retained


def _referent_visual_quality(
    bundle: Bundle, entity_id: str, view_id: str
) -> dict[str, Any]:
    """Return a strict oracle-only recognizability gate for one referent mask."""

    stats = bundle.instance_visual_stats(entity_id, view_id)
    pixels = int(stats["visible_pixels"])
    bbox_width = int(stats["bbox_width_px"])
    bbox_height = int(stats["bbox_height_px"])
    undersized_bbox = min(bbox_width, bbox_height) < MIN_REFERENT_BBOX_SIDE_PX
    fragmented_mask = float(stats["mask_bbox_fill_ratio"]) < MIN_MASK_BBOX_FILL_RATIO
    clipped_axis_fraction = min(
        bbox_width / max(int(stats["image_width_px"]), 1),
        bbox_height / max(int(stats["image_height_px"]), 1),
    )
    outer_fraction = float(stats["outer_five_percent_pixel_fraction"])
    border_count = len(stats["border_sides"])
    border_sides = set(stats["border_sides"])
    severe_border_crop = (
        (border_count >= 3 and {"top", "bottom"}.issubset(border_sides))
        or (
            border_count >= 2
            and outer_fraction
            >= SEVERE_MULTI_BORDER_OUTER_BAND_PIXEL_FRACTION
        )
        or (
            bool(stats["touches_image_border"])
            and outer_fraction
            >= SEVERE_SINGLE_BORDER_OUTER_BAND_PIXEL_FRACTION
            and clipped_axis_fraction < MAX_SEVERELY_CLIPPED_AXIS_FRACTION
        )
    )
    frame_quality = _model_input_frame_quality(bundle, view_id)
    rgb_stats = {
        key: value
        for key, value in frame_quality.items()
        if key not in {"passed", "failure_flags", "thresholds"}
    }
    degenerate_rgb = bool(frame_quality["failure_flags"]["degenerate_rgb"])
    camera_collision = bool(frame_quality["failure_flags"]["camera_collision"])
    entity = bundle.entity(entity_id)
    maximum_height = CATEGORY_MAX_HEIGHT_M.get(entity.label)
    implausible_category_geometry = (
        maximum_height is not None and float(entity.extent_m[2]) > maximum_height
    )
    passed = pixels >= MIN_REFERENT_PIXELS and not (
        undersized_bbox
        or fragmented_mask
        or severe_border_crop
        or degenerate_rgb
        or camera_collision
        or implausible_category_geometry
    )
    return {
        **stats,
        "rgb_visual_stats": rgb_stats,
        "category_geometry": {
            "raw_label": entity.label,
            "extent_m": list(entity.extent_m),
            "maximum_height_m": maximum_height,
        },
        "passed": passed,
        "failure_flags": {
            "below_minimum_pixels": pixels < MIN_REFERENT_PIXELS,
            "undersized_bbox": undersized_bbox,
            "fragmented_mask": fragmented_mask,
            "severe_border_crop": severe_border_crop,
            "degenerate_rgb": degenerate_rgb,
            "camera_collision": camera_collision,
            "implausible_category_geometry": implausible_category_geometry,
        },
        "thresholds": {
            "minimum_visible_pixels": MIN_REFERENT_PIXELS,
            "minimum_bbox_side_px": MIN_REFERENT_BBOX_SIDE_PX,
            "minimum_mask_bbox_fill_ratio": MIN_MASK_BBOX_FILL_RATIO,
            "severe_multi_border_outer_band_pixel_fraction": (
                SEVERE_MULTI_BORDER_OUTER_BAND_PIXEL_FRACTION
            ),
            "severe_single_border_outer_band_pixel_fraction": (
                SEVERE_SINGLE_BORDER_OUTER_BAND_PIXEL_FRACTION
            ),
            "maximum_severely_clipped_axis_fraction": (
                MAX_SEVERELY_CLIPPED_AXIS_FRACTION
            ),
            "maximum_rgb_dominant_color_fraction": (
                MAX_RGB_DOMINANT_COLOR_FRACTION
            ),
            "minimum_rgb_quantized_entropy_bits": (
                MIN_RGB_QUANTIZED_ENTROPY_BITS
            ),
        },
    }


def _referent_selection_score(quality: dict[str, Any]) -> tuple[float, ...]:
    """Rank already-admissible witnesses by semantic visibility, not time."""

    rgb = quality["rgb_visual_stats"]
    return (
        float(not quality["touches_image_border"]),
        -float(quality["outer_five_percent_pixel_fraction"]),
        float(min(quality["bbox_width_px"], quality["bbox_height_px"])),
        float(quality["visible_pixels"]),
        float(rgb["quantized_color_entropy_bits"]),
    )


def _t10_anchor_visual_quality(
    bundle: Bundle, entity_id: str, view_id: str
) -> dict[str, Any]:
    """Require a salient source-frame anchor for a held-out-view question."""

    quality = _referent_visual_quality(bundle, entity_id, view_id)
    salient = (
        int(quality["visible_pixels"]) >= MIN_T10_ANCHOR_PIXELS
        and min(int(quality["bbox_width_px"]), int(quality["bbox_height_px"]))
        >= MIN_T10_ANCHOR_BBOX_SIDE_PX
        and float(quality["outer_five_percent_pixel_fraction"])
        < MAX_T10_ANCHOR_OUTER_BAND_PIXEL_FRACTION
    )
    return {
        **quality,
        "passed": bool(quality["passed"] and salient),
        "t10_anchor_salience": {
            "passed": salient,
            "minimum_visible_pixels": MIN_T10_ANCHOR_PIXELS,
            "minimum_bbox_side_px": MIN_T10_ANCHOR_BBOX_SIDE_PX,
            "maximum_outer_five_percent_pixel_fraction": (
                MAX_T10_ANCHOR_OUTER_BAND_PIXEL_FRACTION
            ),
        },
    }


def _referent_is_recognizable(bundle: Bundle, entity_id: str, view_id: str) -> bool:
    return bool(_referent_visual_quality(bundle, entity_id, view_id)["passed"])


def recognizable_witnesses(bundle: Bundle) -> dict[str, tuple[str, ...]]:
    """Return model-usable evidence views for state auxiliary supervision."""

    result: dict[str, tuple[str, ...]] = {}
    for entity_id, entity in sorted(bundle.entities.items()):
        if entity.label in STRUCTURAL_LABELS:
            continue
        witnesses = tuple(
            view.view_id
            for view in bundle.views
            if entity_id in view.visible_entity_ids
            and _referent_is_recognizable(bundle, entity_id, view.view_id)
        )
        if witnesses:
            result[entity_id] = witnesses
    return result


def visual_quality_policy() -> dict[str, Any]:
    """Expose the frozen oracle-only admission policy for manifests and audits."""

    return {
        "policy_id": "epispace.referent_recognizability.v6",
        "minimum_visible_pixels": MIN_REFERENT_PIXELS,
        "minimum_bbox_side_px": MIN_REFERENT_BBOX_SIDE_PX,
        "minimum_mask_bbox_fill_ratio": MIN_MASK_BBOX_FILL_RATIO,
        "severe_multi_border_outer_band_pixel_fraction": (
            SEVERE_MULTI_BORDER_OUTER_BAND_PIXEL_FRACTION
        ),
        "severe_single_border_outer_band_pixel_fraction": (
            SEVERE_SINGLE_BORDER_OUTER_BAND_PIXEL_FRACTION
        ),
        "maximum_severely_clipped_axis_fraction": (
            MAX_SEVERELY_CLIPPED_AXIS_FRACTION
        ),
        "multi_border_severe_crop": True,
        "vertical_full_span_severe_crop": True,
        "maximum_rgb_dominant_color_fraction": MAX_RGB_DOMINANT_COLOR_FRACTION,
        "minimum_rgb_quantized_entropy_bits": MIN_RGB_QUANTIZED_ENTROPY_BITS,
        "maximum_near_nonstructural_pixel_fraction": (
            MAX_NEAR_NONSTRUCTURAL_PIXEL_FRACTION
        ),
        "maximum_near_any_geometry_pixel_fraction": (
            MAX_NEAR_ANY_GEOMETRY_PIXEL_FRACTION
        ),
        "maximum_close_geometry_pixel_fraction": (
            MAX_CLOSE_GEOMETRY_PIXEL_FRACTION
        ),
        "minimum_near_enclosure_pixel_fraction": (
            MIN_NEAR_ENCLOSURE_PIXEL_FRACTION
        ),
        "maximum_near_enclosure_median_depth_m": (
            MAX_NEAR_ENCLOSURE_MEDIAN_DEPTH_M
        ),
        "maximum_near_enclosure_depth_spread_m": (
            MAX_NEAR_ENCLOSURE_DEPTH_SPREAD_M
        ),
        "minimum_dominant_foreground_pixel_fraction": (
            MIN_DOMINANT_FOREGROUND_PIXEL_FRACTION
        ),
        "maximum_dominant_foreground_median_depth_m": (
            MAX_DOMINANT_FOREGROUND_MEDIAN_DEPTH_M
        ),
        "dominant_foreground_required_border_sides": sorted(
            DOMINANT_FOREGROUND_REQUIRED_BORDER_SIDES
        ),
        "minimum_raw_negative_category_pixels": MIN_RAW_NEGATIVE_CATEGORY_PIXELS,
        "minimum_raw_negative_category_bbox_side_px": (
            MIN_RAW_NEGATIVE_CATEGORY_BBOX_SIDE_PX
        ),
        "minimum_t10_anchor_pixels": MIN_T10_ANCHOR_PIXELS,
        "minimum_t10_anchor_bbox_side_px": MIN_T10_ANCHOR_BBOX_SIDE_PX,
        "maximum_t10_anchor_outer_band_pixel_fraction": (
            MAX_T10_ANCHOR_OUTER_BAND_PIXEL_FRACTION
        ),
        "category_max_height_m": dict(CATEGORY_MAX_HEIGHT_M),
        "visually_confusable_category_groups": [
            sorted(group) for group in VISUAL_CONFUSABILITY_GROUPS
        ],
        "identity_binding_uses_model_facing_categories": True,
        "identity_tracking_requires_unique_candidate_per_model_view": True,
        "visual_absence_unsafe_labels": sorted(VISUAL_ABSENCE_UNSAFE_LABELS),
        "minimum_relation_dominance_ratio": MIN_RELATION_DOMINANCE_RATIO,
        "model_input_frame_gate": (
            "every actual model_view_id must pass RGB-degeneration plus raw-depth/"
            "instance camera-collision, near-enclosure and foreground-obstruction rules"
        ),
        "model_visibility": "oracle-only admission; never serialized into model input",
    }


def _relation_is_unambiguous(primary: float, secondary: float) -> bool:
    """Reject dominant-axis labels too close to a diagonal quadrant boundary."""

    dominant = max(abs(primary), abs(secondary))
    minor = min(abs(primary), abs(secondary))
    return minor == 0.0 or dominant / minor >= MIN_RELATION_DOMINANCE_RATIO


def _source_relation_components(certificate: dict[str, Any]) -> tuple[float, float] | None:
    oracle = certificate.get("oracle")
    if not isinstance(oracle, dict):
        return None
    for right_key, front_key in (
        ("ego_right_delta_m", "ego_front_delta_m"),
        ("canonical_right_delta_m", "canonical_front_delta_m"),
        ("query_right_m", "query_front_m"),
        ("target_ego_right_m", "target_ego_front_m"),
    ):
        right = oracle.get(right_key)
        front = oracle.get(front_key)
        if (
            isinstance(right, int | float)
            and not isinstance(right, bool)
            and isinstance(front, int | float)
            and not isinstance(front, bool)
        ):
            return float(right), float(front)
    return None


def compile_bundle_questions(bundle: Bundle) -> tuple[list[QuestionSpec], list[Rejection]]:
    """Compile generic tasks and add questions specific to trajectory semantics."""

    if bundle.trajectory_class == "T10":
        return [], [
            Rejection(
                "bundle",
                bundle.episode_id,
                "held_out_verifier_only",
                "T10 target renders are forbidden from standalone training input",
                str(bundle.root),
            )
        ]

    accepted: list[QuestionSpec] = []
    rejected: list[Rejection] = []
    evidence_family_labels: set[str] = set()
    allowed = SOURCE_ALLOWED_BY_CLASS[bundle.trajectory_class]
    all_view_ids = {view.view_id for view in bundle.views}
    cross_view_proofs: dict[frozenset[str], str] = {}
    for candidate in bundle.tasks:
        if candidate.get("task_type") != "cross_view_relation":
            continue
        candidate_certificate = candidate.get("certificate") or {}
        checks = {
            item.get("name"): item
            for item in candidate_certificate.get("checks", ())
        }
        relation_components = _source_relation_components(candidate_certificate)
        candidate_views = tuple(
            str(value) for value in candidate.get("evidence_view_ids", ())
        )
        candidate_entities = tuple(
            str(value) for value in candidate.get("evidence_entity_ids", ())
        )
        if (
            checks.get("never_co_visible", {}).get("passed") is True
            and relation_components is not None
            and _relation_is_unambiguous(*relation_components)
            and _bindings_unambiguous(
                bundle,
                candidate_entities,
                candidate_views,
                surface_question=localize_categories(
                    str(candidate.get("question_zh", "")).strip()
                ),
            )
        ):
            cross_view_proofs[
                frozenset(candidate_entities)
            ] = str(candidate.get("task_id", "unknown"))
    for task in bundle.tasks:
        task_id = str(task.get("task_id", "unknown"))
        task_type = str(task.get("task_type", "unknown"))
        if task_type not in allowed:
            rejected.append(
                Rejection(
                    "task",
                    task_id,
                    "trajectory_illegal_task",
                    f"{task_type} is not legal for {bundle.trajectory_class}",
                    str(bundle.root),
                )
            )
            continue
        if task.get("status") not in {"accepted", "unknown"}:
            rejected.append(
                Rejection(
                    "task",
                    task_id,
                    "source_task_not_accepted",
                    f"source status={task.get('status')}",
                    str(bundle.root),
                )
            )
            continue
        certificate = task.get("certificate") or {}
        if certificate.get("result") not in {"pass", "unknown"}:
            rejected.append(
                Rejection(
                    "task",
                    task_id,
                    "certificate_not_valid",
                    f"certificate result={certificate.get('result')}",
                    str(bundle.root),
                )
            )
            continue
        evidence_views = tuple(str(value) for value in task.get("evidence_view_ids", ()))
        if not set(evidence_views).issubset(all_view_ids):
            rejected.append(
                Rejection(
                    "task",
                    task_id,
                    "evidence_outside_model_views",
                    f"evidence={evidence_views}",
                    str(bundle.root),
                )
            )
            continue
        evidence_entities = tuple(str(value) for value in task.get("evidence_entity_ids", ()))
        localized_question = localize_categories(str(task.get("question_zh", "")).strip())
        if task_type == "unknown_abstention":
            unknown_label = _source_unknown_target_label(bundle, task)
            if unknown_label is None or _confusable_category_visible(
                bundle, unknown_label, evidence_views
            ):
                rejected.append(
                    Rejection(
                        "task",
                        task_id,
                        "visually_confusable_negative_category",
                        (
                            "the unknown target has no recoverable ontology label or "
                            "a visually confusable category is present in model RGB"
                        ),
                        str(bundle.root),
                    )
                )
                continue
        if (
            task_type == "grounding_presence"
            and task.get("answer") is True
            and (
                not evidence_entities
                or not any(
                    _referent_is_recognizable(
                        bundle, bundle.entity(entity_id).entity_id, view_id
                    )
                    for entity_id in evidence_entities
                    for view_id in evidence_views
                )
            )
        ):
            rejected.append(
                Rejection(
                    "task",
                    task_id,
                    "referent_not_visually_recognizable",
                    "no grounding witness passes the mask recognizability gate",
                    str(bundle.root),
                )
            )
            continue
        if task_type == "last_seen_memory" and (
            len(evidence_entities) != 1
            or any(
                evidence_entities[0] not in bundle.view_by_id[view_id].visible_entity_ids
                or not _referent_is_recognizable(
                    bundle, evidence_entities[0], view_id
                )
                for view_id in evidence_views
            )
        ):
            rejected.append(
                Rejection(
                    "task",
                    task_id,
                    "temporal_track_contains_unrecognizable_witness",
                    "every reported visible step must pass the visual gate",
                    str(bundle.root),
                )
            )
            continue
        binding_views = (
            tuple(view.view_id for view in bundle.views)
            if task_type == "last_seen_memory"
            else evidence_views
        )
        if task_type in INSTANCE_BOUND_TASKS and not _bindings_unambiguous(
            bundle,
            evidence_entities,
            binding_views,
            surface_question=localized_question,
            require_unique_track=task_type in IDENTITY_TRACKING_TASKS,
        ):
            rejected.append(
                Rejection(
                    "task",
                    task_id,
                    "unresolvable_or_unrecognizable_entity_reference",
                    "an instance-bound category is duplicated or lacks a recognizable mask witness",
                    str(bundle.root),
                )
            )
            continue
        if task_type in INSTANCE_BOUND_TASKS:
            certificate = {
                **certificate,
                "checks": [
                    *certificate.get("checks", ()),
                    {
                        "name": "surface_referents_unique_and_resolvable",
                        "passed": True,
                        "binding_mode": (
                            "unique_model_facing_identity_track"
                            if task_type in IDENTITY_TRACKING_TASKS
                            else "unique_surface_anchor"
                        ),
                        "binding_view_ids": list(binding_views),
                        "uses_visually_confusable_category_groups": True,
                        "minimum_visible_pixels": MIN_REFERENT_PIXELS,
                        "minimum_bbox_side_px": MIN_REFERENT_BBOX_SIDE_PX,
                        "minimum_mask_bbox_fill_ratio": MIN_MASK_BBOX_FILL_RATIO,
                        "severe_multi_border_outer_band_pixel_fraction": (
                            SEVERE_MULTI_BORDER_OUTER_BAND_PIXEL_FRACTION
                        ),
                        "severe_single_border_outer_band_pixel_fraction": (
                            SEVERE_SINGLE_BORDER_OUTER_BAND_PIXEL_FRACTION
                        ),
                        "maximum_severely_clipped_axis_fraction": (
                            MAX_SEVERELY_CLIPPED_AXIS_FRACTION
                        ),
                        "maximum_rgb_dominant_color_fraction": (
                            MAX_RGB_DOMINANT_COLOR_FRACTION
                        ),
                        "minimum_rgb_quantized_entropy_bits": (
                            MIN_RGB_QUANTIZED_ENTROPY_BITS
                        ),
                    },
                ],
            }
        relation_components = _source_relation_components(certificate)
        if relation_components is not None and not _relation_is_unambiguous(
            *relation_components
        ):
            rejected.append(
                Rejection(
                    "task",
                    task_id,
                    "relation_quadrant_ambiguous",
                    (
                        "dominant-axis ratio is below "
                        f"{MIN_RELATION_DOMINANCE_RATIO:.2f}: "
                        f"right={relation_components[0]:.3f}, "
                        f"front={relation_components[1]:.3f}"
                    ),
                    str(bundle.root),
                )
            )
            continue
        if relation_components is not None:
            dominant = max(abs(value) for value in relation_components)
            minor = min(abs(value) for value in relation_components)
            certificate = {
                **certificate,
                "checks": [
                    *certificate.get("checks", ()),
                    {
                        "name": "relation_dominance_ratio",
                        "passed": True,
                        "measured": None if minor == 0 else dominant / minor,
                        "threshold": MIN_RELATION_DOMINANCE_RATIO,
                    },
                ],
            }
        if task_type == "cross_view_relation":
            checks = {item.get("name"): item for item in certificate.get("checks", ())}
            check = checks.get("never_co_visible")
            if not check or check.get("passed") is not True:
                rejected.append(
                    Rejection(
                        "task",
                        task_id,
                        "cross_view_not_proven",
                        "missing a passing never_co_visible certificate check",
                        str(bundle.root),
                    )
                )
                continue
        if task_type == "counterfactual_verification":
            entity_key = frozenset(str(value) for value in task.get("evidence_entity_ids", ()))
            linked_task = cross_view_proofs.get(entity_key)
            if not linked_task:
                rejected.append(
                    Rejection(
                        "task",
                        task_id,
                        "counterfactual_cross_view_not_proven",
                        "no matching cross-view task proves never-co-visible evidence",
                        str(bundle.root),
                    )
                )
                continue
            certificate = {
                **certificate,
                "linked_cross_view_proof_task_id": linked_task,
                "checks": [
                    *certificate.get("checks", ()),
                    {
                        "name": "never_co_visible_via_linked_task",
                        "passed": True,
                        "linked_task_id": linked_task,
                    },
                ],
            }
        question = localized_question
        answer = localize_categories(str(task.get("surface_answer_zh", "")).strip())
        if (
            task_type == "grounding_presence"
            and len(evidence_views) == 1
            and len(evidence_entities) == 1
        ):
            # The source generator may bind a category through another frame
            # ("the sofa seen in view 1") even though presence is tested in a
            # single clean frame.  Serialize the atom as a genuinely view-local
            # question so omitted/corrupt context is neither referenced nor
            # silently re-indexed.
            name = entity_name(bundle.entity(evidence_entities[0]).label)
            question = f"在这张图里，能看到{name}吗？"
            certificate = {
                **certificate,
                "checks": [
                    *certificate.get("checks", ()),
                    {
                        "name": "view_local_grounding_surface",
                        "passed": True,
                        "model_view_id": evidence_views[0],
                    },
                ],
            }
        if task_type == "egocentric_relation":
            question = question.replace("哪一侧？", "哪个方向？")
        if not question or not answer:
            rejected.append(
                Rejection(
                    "task",
                    task_id,
                    "missing_language_surface",
                    "question or answer is empty",
                    str(bundle.root),
                )
            )
            continue
        spec = QuestionSpec(
            fact_id=stable_id("fact", bundle.scene_id, bundle.episode_id, task_id),
            task_type=task_type,
            question_zh=question,
            answer_zh=answer,
            answer_value=task.get("answer"),
            answer_status=str(task["status"]),
            program=program_for(task_type),
            evidence_view_ids=evidence_views,
            evidence_entity_ids=evidence_entities,
            certificate=certificate,
            rationale_zh=_rationale(
                task_type,
                {**task, "surface_answer_zh": answer},
                evidence_views,
            ),
            source="omnigibson.reasoning_tasks.v1",
            # View-local grounding should not inherit unrelated bad context
            # frames.  Sequential/stateful tasks keep the complete trajectory.
            model_view_ids=(
                evidence_views
                if task_type == "grounding_presence"
                else tuple(view.view_id for view in bundle.views)
            ),
            tags=("source_generic", bundle.trajectory_class),
        )
        if task_type == "cross_view_relation":
            accepted.extend(_cross_view_evidence_family(bundle, spec))
        elif task_type == "counterfactual_verification":
            accepted.extend(_counterfactual_claim_pair(bundle, spec))
        else:
            accepted.append(spec)
            if task_type == "grounding_presence" and task.get("answer") is True:
                negative = _grounding_negative(bundle, spec)
                if negative is not None:
                    accepted.append(negative)
                label = bundle.entity(evidence_entities[0]).label
                if label not in evidence_family_labels:
                    family = _presence_evidence_family(bundle, spec)
                    if family:
                        accepted.extend(family)
                        evidence_family_labels.add(label)

    if bundle.trajectory_class in {"T3", "T4"}:
        accepted.extend(_compile_frame_equivariance(bundle))
    if bundle.trajectory_class == "T3":
        derived = _compile_t3(bundle)
    elif bundle.trajectory_class == "T4":
        derived = _compile_t4(bundle)
    elif bundle.trajectory_class == "T7":
        derived = _compile_t7(bundle)
    elif bundle.trajectory_class == "T8":
        derived = _compile_t8(bundle)
    else:
        derived = []
    if bundle.trajectory_class == "T8" and not derived:
        rejected.append(
            Rejection(
                "derived_family",
                bundle.episode_id,
                "t8_occlusion_family_unavailable_after_quality_gating",
                "no clean post-occlusion target witness satisfies the family contract",
                str(bundle.root),
            )
        )
    accepted.extend(derived)
    accepted = _apply_model_input_frame_gate(bundle, accepted, rejected)
    return accepted, rejected


def _rationale(
    task_type: str,
    task: dict[str, Any],
    evidence_views: tuple[str, ...],
) -> str:
    if len(evidence_views) > 3:
        view_text = f"第{view_number(evidence_views[0])}至第{view_number(evidence_views[-1])}个视角"
    else:
        view_text = "、".join(f"第{view_number(value)}个视角" for value in evidence_views)
    answer = str(task["surface_answer_zh"])
    if task_type == "cross_view_relation":
        return (
            f"两个关键物体分别由{view_text}提供，且没有共同可见的决定帧。"
            f"先把两段观察注册到同一锚定平面，再比较其位置；{answer}"
        )
    if task_type == "counterfactual_verification":
        return f"先用{view_text}重建共同参照下的真实关系，再与题中说法比较；{answer}"
    if task_type == "object_centric_perspective":
        return f"先用{view_text}绑定站位、朝向和目标，再建立新的自我坐标系；{answer}"
    if task_type == "last_seen_memory":
        return f"沿观察顺序更新该物体的可见状态，并检查最后出现后的帧；{answer}"
    if task_type == "metric_distance":
        return f"先在{view_text}中稳定绑定两个物体，再比较其空间中心距离；{answer}"
    if task_type == "unknown_abstention":
        return f"只检查已经给出的视角；未观察到不能推出场景中不存在。{answer}"
    return answer


def _eligible_entities(bundle: Bundle, entity_ids: Iterable[str]) -> list[str]:
    return sorted(
        (
            entity_id
            for entity_id in entity_ids
            if entity_id in bundle.entities
            and bundle.entities[entity_id].label not in STRUCTURAL_LABELS
        ),
        key=lambda entity_id: (bundle.entities[entity_id].label, entity_id),
    )


def _visually_confusable_labels(label: str) -> frozenset[str]:
    return next(
        (group for group in VISUAL_CONFUSABILITY_GROUPS if label in group),
        frozenset({label}),
    )


def _confusable_category_visible(
    bundle: Bundle, label: str, view_ids: Iterable[str]
) -> bool:
    """Return whether raw pixels contradict a category-absence claim.

    Acquisition summaries intentionally omit tiny instances, so using only
    ``visible_entity_ids`` creates false negative/unknown labels.  Re-scan the
    complete instance channel and aggregate all visually confusable labels.
    """

    if label in VISUAL_ABSENCE_UNSAFE_LABELS:
        return True
    alternatives = _visually_confusable_labels(label)
    return any(
        (
            (stats := bundle.raw_category_visual_stats(view_id, alternatives))[
                "matching_pixel_count"
            ]
            >= MIN_RAW_NEGATIVE_CATEGORY_PIXELS
            or stats["max_instance_bbox_min_side_px"]
            >= MIN_RAW_NEGATIVE_CATEGORY_BBOX_SIDE_PX
        )
        for view_id in view_ids
    )


def _source_unknown_target_label(bundle: Bundle, task: dict[str, Any]) -> str | None:
    """Recover the raw ontology token from deterministic source text."""

    question = str(task.get("question_zh", ""))
    labels = sorted({entity.label for entity in bundle.entities.values()}, key=len, reverse=True)
    return next(
        (
            label
            for label in labels
            if label in question or label.replace("_", " ") in question
        ),
        None,
    )


def _bindings_unambiguous(
    bundle: Bundle,
    entity_ids: tuple[str, ...],
    view_ids: tuple[str, ...],
    *,
    surface_question: str | None = None,
    require_unique_track: bool = False,
) -> bool:
    """Check that category-and-view language identifies visible, usable instances.

    Source questions sometimes name an exact witness (for example, ``the door
    seen in view 1``) while also supplying later views.  A clean mask in a later
    image must not rescue an unreadable surface anchor.  Bare category names may
    use any clean evidence view because no particular image is asserted by the
    language.

    Identity-tracking questions are stricter.  Every model-visible view must
    contain either the one bound instance or no recognizable member of its
    model-facing category.  Otherwise a category/view phrase cannot say which
    repeated instance is being followed.  Visually confusable ontology labels
    (for example ``downlight`` and ``square_light``) count as one category here;
    simulator-only subclasses cannot act as language qualifiers.
    """

    if not entity_ids or not view_ids:
        return False
    for raw_id in entity_ids:
        try:
            entity = bundle.entity(raw_id)
        except ValueError:
            return False
        visible_candidates: list[str] = []
        for view_id in view_ids:
            view = bundle.view_by_id[view_id]
            if entity.entity_id not in view.visible_entity_ids:
                continue
            visible_candidates.append(view_id)
        if not visible_candidates:
            return False
        explicit_anchors = _surface_anchor_views(
            bundle,
            entity.entity_id,
            tuple(visible_candidates),
            surface_question,
        )
        required_candidates = (
            view_ids
            if require_unique_track
            else explicit_anchors or tuple(visible_candidates)
        )
        for view_id in required_candidates:
            recognizable = _recognizable_category_candidates(
                bundle, entity.label, view_id
            )
            target_visible = (
                entity.entity_id in bundle.view_by_id[view_id].visible_entity_ids
            )
            if target_visible:
                if recognizable != (entity.entity_id,):
                    return False
            elif require_unique_track and recognizable:
                # A same-looking instance after the reported last-seen frame is
                # not RGB evidence that the originally bound object disappeared.
                return False
    return True


def _recognizable_category_candidates(
    bundle: Bundle, label: str, view_id: str
) -> tuple[str, ...]:
    """Return visually usable candidates for one model-facing category.

    Only candidates that pass the same mask/RGB admission as the target count.
    Tiny raw-mask fragments therefore do not create false ambiguity, while a
    simulator subclass hidden from natural language cannot manufacture a unique
    binding.
    """

    labels = _visually_confusable_labels(label)
    return tuple(
        sorted(
            entity_id
            for entity_id in bundle.view_by_id[view_id].visible_entity_ids
            if entity_id in bundle.entities
            and bundle.entities[entity_id].label in labels
            and _referent_is_recognizable(bundle, entity_id, view_id)
        )
    )


def _surface_anchor_views(
    bundle: Bundle,
    entity_id: str,
    visible_view_ids: tuple[str, ...],
    surface_question: str | None,
) -> tuple[str, ...]:
    """Recover explicit category/view bindings from deterministic source text."""

    if not surface_question:
        return ()
    name = entity_name(bundle.entities[entity_id].label)
    matches: list[str] = []
    for view_id in visible_view_ids:
        number = view_number(view_id)
        prefixes = (
            f"第{number}个视角里看到的",
            f"第{number}个视角中看到的",
            f"第{number}个视角看到的",
        )
        if _has_exact_surface_anchor(bundle, surface_question, prefixes, name):
            matches.append(view_id)
    return tuple(matches)


def _has_exact_surface_anchor(
    bundle: Bundle,
    question: str,
    prefixes: tuple[str, ...],
    name: str,
) -> bool:
    """Match a Chinese category mention without accepting a longer label.

    Chinese text has no regex word boundary.  A plain substring lookup would,
    for example, treat ``第4个视角里看到的床头柜`` as an explicit
    anchor for both ``床头柜`` and ``床``.  Longest ontology-surface matching
    keeps the model-facing reference and the oracle instance binding aligned.
    """

    longer_names = {
        entity_name(entity.label)
        for entity in bundle.entities.values()
        if len(entity_name(entity.label)) > len(name)
        and entity_name(entity.label).startswith(name)
    }
    for prefix in prefixes:
        needle = f"{prefix}{name}"
        start = question.find(needle)
        while start >= 0:
            name_start = start + len(prefix)
            if not any(question.startswith(longer, name_start) for longer in longer_names):
                return True
            start = question.find(needle, start + 1)
    return False


def _unique_anchor_view(
    bundle: Bundle, entity_id: str, *, t10_salient: bool = False
):
    entity = bundle.entities[entity_id]
    candidates: list[tuple[tuple[float, ...], int, Any]] = []
    for view in bundle.views:
        if entity_id not in view.visible_entity_ids:
            continue
        same_label = sum(
            bundle.entities[candidate_id].label == entity.label
            for candidate_id in view.visible_entity_ids
            if candidate_id in bundle.entities
        )
        if same_label != 1:
            continue
        quality = (
            _t10_anchor_visual_quality(bundle, entity_id, view.view_id)
            if t10_salient
            else _referent_visual_quality(bundle, entity_id, view.view_id)
        )
        if quality["passed"]:
            candidates.append(
                (_referent_selection_score(quality), -view.step, view)
            )
    return max(candidates)[2] if candidates else None


def _grounding_negative(bundle: Bundle, positive: QuestionSpec) -> QuestionSpec | None:
    """Add a view-local absent category to balance presence questions."""

    if len(positive.evidence_view_ids) != 1:
        return None
    anchor_id = positive.evidence_view_ids[0]
    anchor = bundle.view_by_id[anchor_id]
    visible_labels = {
        bundle.entities[entity_id].label
        for entity_id in anchor.visible_entity_ids
        if entity_id in bundle.entities
    }
    candidates = _eligible_entities(
        bundle,
        set().union(*(set(view.visible_entity_ids) for view in bundle.views))
        - set(anchor.visible_entity_ids),
    )
    candidate_id = next(
        (
            entity_id
            for entity_id in candidates
            if not _confusable_category_visible(
                bundle, bundle.entities[entity_id].label, (anchor_id,)
            )
        ),
        None,
    )
    if candidate_id is None:
        return None
    candidate = bundle.entities[candidate_id]
    name = entity_name(candidate.label)
    group = stable_id("contrast", bundle.scene_id, bundle.episode_id, positive.fact_id)
    return QuestionSpec(
        fact_id=stable_id("fact", group, "absent"),
        task_type="grounding_presence",
        question_zh=f"在这张图里，能看到{name}吗？",
        answer_zh="不能。",
        answer_value=False,
        answer_status="accepted",
        program=program_for("grounding_presence"),
        evidence_view_ids=(anchor_id,),
        evidence_entity_ids=(candidate.entity_id,),
        certificate={
            "schema_version": "epispace.certificate.v1",
            "result": "pass",
            "checks": [
                {
                    "name": "category_absent_in_anchor_view",
                    "passed": candidate.label not in visible_labels,
                    "anchor_view_id": anchor_id,
                    "category": candidate.label,
                },
                {
                    "name": "category_observed_elsewhere",
                    "passed": True,
                    "visible_view_ids": list(bundle.visible_views(candidate.entity_id)),
                },
            ],
        },
        rationale_zh=f"只检查这张图；其中没有{name}。",
        source="epispace.contrast_compiler.grounding.v1",
        model_view_ids=positive.model_view_ids,
        tags=("balanced_contrast", "view_local_absence"),
    )


def _presence_evidence_family(bundle: Bundle, positive: QuestionSpec) -> list[QuestionSpec]:
    """Build a same-question, same-image-count presence evidence intervention."""

    if len(positive.evidence_entity_ids) != 1:
        return []
    entity = bundle.entity(positive.evidence_entity_ids[0])
    decisive_candidates = []
    for view in bundle.views:
        if entity.entity_id not in view.visible_entity_ids:
            continue
        quality = _referent_visual_quality(bundle, entity.entity_id, view.view_id)
        if quality["passed"]:
            decisive_candidates.append(
                (_referent_selection_score(quality), -view.step, view)
            )
    if not decisive_candidates:
        return []
    decisive = max(decisive_candidates)[2]

    def category_visible(view_id: str) -> bool:
        return any(
            candidate_id in bundle.entities
            and bundle.entities[candidate_id].label == entity.label
            for candidate_id in bundle.view_by_id[view_id].visible_entity_ids
        )

    def confusable_visible(view_id: str) -> bool:
        return _confusable_category_visible(bundle, entity.label, (view_id,))

    safe = [view for view in bundle.views if not confusable_visible(view.view_id)]
    matched: tuple[Any, Any, Any] | None = None
    for common_view in safe:
        for direction in (-1, 1):
            decisive_offset = direction * (decisive.step - common_view.step)
            replacements = [
                view
                for view in safe
                if view.view_id != common_view.view_id
                and direction * (view.step - common_view.step) > 0
            ]
            if decisive_offset > 0 and len(replacements) >= 2:
                matched = (common_view, replacements[0], replacements[-1])
                break
        if matched is not None:
            break
    if matched is None:
        return []
    common_view, prefix_distractor, deleted_distractor = matched

    def ordered(*views: Any) -> tuple[str, ...]:
        by_id = {view.view_id: view for view in views}
        return tuple(
            view.view_id for view in sorted(by_id.values(), key=lambda view: view.step)
        )

    prefix_views = ordered(common_view, prefix_distractor)
    revealed_views = ordered(common_view, decisive)
    deleted_views = ordered(common_view, deleted_distractor)
    if not all(len(view_ids) == 2 for view_ids in (prefix_views, revealed_views, deleted_views)):
        return []

    name = entity_name(entity.label)
    question = f"仅根据当前给出的两张有序观察，能否确定这个场景里存在{name}？"
    group = stable_id(
        "consistency",
        bundle.scene_id,
        bundle.episode_id,
        entity.label,
        "presence_evidence",
    )
    common = {
        "question_zh": question,
        "evidence_entity_ids": (entity.entity_id,),
        "consistency_group": group,
        "source": "epispace.evidence_family_compiler.presence.v1",
    }
    common_checks = [
        {
            "name": "matched_visual_input_count",
            "passed": True,
            "image_count": 2,
        },
        {
            "name": "category_level_existence_query",
            "passed": True,
            "category": entity.label,
        },
        {
            "name": "one_shared_view_one_evidence_replacement",
            "passed": (
                set(prefix_views) & set(revealed_views) & set(deleted_views)
                == {common_view.view_id}
            ),
            "shared_view_id": common_view.view_id,
        },
    ]

    def unknown_member(variant: str, view_ids: tuple[str, ...]) -> QuestionSpec:
        return QuestionSpec(
            fact_id=stable_id("fact", group, variant),
            task_type="evidence_presence_unknown",
            answer_zh=f"还不能确定；当前两张图里没有{name}的视觉证据。",
            answer_value=None,
            answer_status="unknown",
            program=program_for("evidence_presence_unknown"),
            evidence_view_ids=view_ids,
            certificate={
                "schema_version": "epispace.certificate.v1",
                "result": "unknown",
                "checks": [
                    *common_checks,
                    {
                        "name": "target_category_absent_from_actual_input",
                        "passed": not any(
                            confusable_visible(view_id) for view_id in view_ids
                        ),
                    },
                ],
            },
            rationale_zh=f"当前两张图都没有{name}的视觉证据，因此不能断言场景里存在它。",
            family_variant=variant,
            model_view_ids=view_ids,
            tags=("evidence_intervention", "matched_two_view", "epistemic"),
            **common,
        )

    revealed = QuestionSpec(
        fact_id=stable_id("fact", group, "revealed"),
        task_type="evidence_presence_reveal",
        answer_zh=f"能确定；当前观察中能看到{name}。",
        answer_value={"status": "present", "category": entity.label},
        answer_status="accepted",
        program=program_for("evidence_presence_reveal"),
        evidence_view_ids=revealed_views,
        certificate={
            "schema_version": "epispace.certificate.v1",
            "result": "pass",
            "checks": [
                *common_checks,
                {
                    "name": "target_category_visible_in_actual_input",
                    "passed": any(category_visible(view_id) for view_id in revealed_views),
                    "decisive_view_id": decisive.view_id,
                    "visible_pixels": bundle.visible_pixel_count(
                        entity.entity_id, decisive.view_id
                    ),
                    "minimum_visible_pixels": MIN_REFERENT_PIXELS,
                    "referent_visual_quality": _referent_visual_quality(
                        bundle, entity.entity_id, decisive.view_id
                    ),
                },
            ],
        },
        rationale_zh=f"其中一张图提供了{name}的直接视觉证据，因此可以确认场景里存在它。",
        family_variant="revealed",
        model_view_ids=revealed_views,
        tags=("evidence_intervention", "matched_two_view", "evidence_reveal"),
        **common,
    )
    return [
        unknown_member("prefix_unknown", prefix_views),
        revealed,
        unknown_member("decisive_deleted", deleted_views),
    ]


def _camera_yaw_rad(bundle: Bundle, view_id: str) -> float:
    rotation = bundle.view_by_id[view_id].world_from_camera["rotation_xyzw"]
    x, y, z, w = (float(value) for value in rotation)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _camera_relation(bundle: Bundle, subject: Any, reference: Any, view_id: str) -> str:
    relation, _, _ = _camera_relation_components(bundle, subject, reference, view_id)
    return relation


def _camera_relation_components(
    bundle: Bundle, subject: Any, reference: Any, view_id: str
) -> tuple[str, float, float]:
    yaw = _camera_yaw_rad(bundle, view_id)
    world_dx = subject.center_world_m[0] - reference.center_world_m[0]
    world_dy = subject.center_world_m[1] - reference.center_world_m[1]
    right = world_dx * math.cos(yaw) + world_dy * math.sin(yaw)
    front = -world_dx * math.sin(yaw) + world_dy * math.cos(yaw)
    if abs(right) >= abs(front):
        relation = "right_of" if right > 0 else "left_of"
    else:
        relation = "in_front_of" if front > 0 else "behind"
    return relation, right, front


def _compile_frame_equivariance(bundle: Bundle) -> list[QuestionSpec]:
    """Search the whole bundle for one high-quality frame intervention.

    Restricting this family to the object pair chosen by an upstream generic
    task silently erased the entire frame axis once recognizability gating was
    tightened.  Bundle-level search keeps the semantic contract fixed while
    selecting a different, fully visible pair when necessary.
    """

    observed = set().union(*(set(view.visible_entity_ids) for view in bundle.views))
    candidate_families: list[tuple[tuple[float, ...], list[QuestionSpec]]] = []
    for subject_id, reference_id in combinations(_eligible_entities(bundle, observed), 2):
        subject = bundle.entities[subject_id]
        reference = bundle.entities[reference_id]
        if subject.label == reference.label:
            continue
        family = _frame_equivariance_family_for_pair(bundle, subject, reference)
        if not family:
            continue
        quality_check = next(
            check
            for check in family[0].certificate["checks"]
            if check["name"] == "frame_pair_quality"
        )
        score = (
            float(quality_check["minimum_bbox_side_px"]),
            float(quality_check["minimum_visible_pixels"]),
            float(quality_check["minimum_dominance_ratio"]),
            float(quality_check["yaw_separation_deg"]),
        )
        candidate_families.append((score, family))
    if not candidate_families:
        return []
    return max(
        candidate_families,
        key=lambda item: (item[0], item[1][0].fact_id),
    )[1]


def _frame_equivariance_family_for_pair(
    bundle: Bundle,
    subject: Any,
    reference: Any,
) -> list[QuestionSpec]:
    """Create two same-image reads whose answers transform with the frame."""

    eligible = [
        view
        for view in bundle.views
        if _bindings_unambiguous(
            bundle,
            (subject.entity_id, reference.entity_id),
            (view.view_id,),
        )
    ]
    candidates: list[tuple[Any, ...]] = []
    for left, right in combinations(eligible, 2):
        left_relation, left_dx, left_dy = _camera_relation_components(
            bundle, subject, reference, left.view_id
        )
        right_relation, right_dx, right_dy = _camera_relation_components(
            bundle, subject, reference, right.view_id
        )
        if not _relation_is_unambiguous(
            left_dx, left_dy
        ) or not _relation_is_unambiguous(right_dx, right_dy):
            continue
        if left_relation == right_relation:
            continue
        yaw_gap = abs(
            math.degrees(_camera_yaw_rad(bundle, left.view_id) - _camera_yaw_rad(bundle, right.view_id))
        ) % 360.0
        yaw_gap = min(yaw_gap, 360.0 - yaw_gap)
        if yaw_gap < 30.0:
            continue
        qualities = [
            _referent_visual_quality(bundle, entity.entity_id, view.view_id)
            for entity in (subject, reference)
            for view in (left, right)
        ]
        minimum_pixels = min(int(item["visible_pixels"]) for item in qualities)
        minimum_bbox_side = min(
            min(int(item["bbox_width_px"]), int(item["bbox_height_px"]))
            for item in qualities
        )
        left_ratio = max(abs(left_dx), abs(left_dy)) / max(
            min(abs(left_dx), abs(left_dy)), 1e-12
        )
        right_ratio = max(abs(right_dx), abs(right_dy)) / max(
            min(abs(right_dx), abs(right_dy)), 1e-12
        )
        minimum_ratio = min(left_ratio, right_ratio)
        score = (minimum_bbox_side, minimum_pixels, minimum_ratio, yaw_gap)
        candidates.append(
            (
                score,
                left,
                right,
                left_relation,
                right_relation,
                minimum_pixels,
                minimum_bbox_side,
                minimum_ratio,
                yaw_gap,
            )
        )
    if not candidates:
        return []
    (
        _,
        first,
        second,
        first_relation,
        second_relation,
        minimum_pixels,
        minimum_bbox_side,
        minimum_ratio,
        yaw_gap,
    ) = max(
        candidates,
        key=lambda item: (item[0], -item[1].step, -item[2].step),
    )
    if first.step > second.step:
        first, second = second, first
        first_relation, second_relation = second_relation, first_relation
    model_views = (first.view_id, second.view_id)
    subject_name = entity_name(subject.label)
    reference_name = entity_name(reference.label)
    group = stable_id(
        "consistency",
        bundle.scene_id,
        bundle.episode_id,
        subject.entity_id,
        reference.entity_id,
        "frame_equivariance",
    )

    def member(
        variant: str,
        display_index: int,
        anchor_view_id: str,
        relation: str,
    ) -> QuestionSpec:
        relation_zh = relation_name(relation)
        return QuestionSpec(
            fact_id=stable_id("fact", group, variant),
            task_type="egocentric_relation",
            question_zh=(
                f"以第{display_index}张图的相机朝向为正前方，"
                f"该图里看到的{subject_name}位于{reference_name}的哪个方向？"
            ),
            answer_zh=f"{subject_name}位于{reference_name}的{relation_zh}。",
            answer_value=relation,
            answer_status="accepted",
            program=program_for("egocentric_relation"),
            evidence_view_ids=(anchor_view_id,),
            evidence_entity_ids=(subject.entity_id, reference.entity_id),
            certificate={
                "schema_version": "epispace.certificate.v1",
                "result": "pass",
                "checks": [
                    {
                        "name": "same_two_images_across_frame_siblings",
                        "passed": True,
                        "model_view_ids": list(model_views),
                    },
                    {
                        "name": "both_referents_unique_and_resolvable_in_anchor",
                        "passed": _bindings_unambiguous(
                            bundle,
                            (subject.entity_id, reference.entity_id),
                            (anchor_view_id,),
                        ),
                        "minimum_visible_pixels": MIN_REFERENT_PIXELS,
                    },
                    {
                        "name": "camera_frame_relation_recomputed",
                        "passed": True,
                        "anchor_view_id": anchor_view_id,
                        "relation": relation,
                        "yaw_separation_deg": yaw_gap,
                    },
                    {
                        "name": "frame_pair_quality",
                        "passed": True,
                        "minimum_visible_pixels": minimum_pixels,
                        "minimum_bbox_side_px": minimum_bbox_side,
                        "minimum_dominance_ratio": minimum_ratio,
                        "yaw_separation_deg": yaw_gap,
                        "recognizability_threshold_pixels": MIN_REFERENT_PIXELS,
                        "recognizability_threshold_bbox_side_px": (
                            MIN_REFERENT_BBOX_SIDE_PX
                        ),
                        "dominance_ratio_threshold": MIN_RELATION_DOMINANCE_RATIO,
                    },
                ],
            },
            rationale_zh=(
                f"先把两者的位置转换到第{display_index}张图的自我坐标系；"
                f"在该朝向下，{subject_name}位于{reference_name}的{relation_zh}。"
            ),
            source="epispace.frame_family_compiler.v2",
            family_variant=variant,
            consistency_group=group,
            model_view_ids=model_views,
            tags=("frame_equivariance", "same_visual_input", "answer_transforms"),
        )

    return [
        member("frame_a", 1, first.view_id, first_relation),
        member("frame_b", 2, second.view_id, second_relation),
    ]


def _counterfactual_claim_pair(bundle: Bundle, false_claim: QuestionSpec) -> list[QuestionSpec]:
    """Pair a contradicted relation claim with its certified true form."""

    oracle = false_claim.certificate.get("oracle", {})
    rejected = str(oracle.get("rejected_relation", ""))
    verified = str(oracle.get("verified_relation", ""))
    rejected_surface = relation_name(rejected)
    verified_surface = relation_name(verified)
    if (
        not rejected
        or not verified
        or rejected == verified
        or rejected_surface not in false_claim.question_zh
    ):
        return [false_claim]
    group = stable_id(
        "consistency", bundle.scene_id, bundle.episode_id, false_claim.fact_id, "claim"
    )
    false_structured = replace(
        false_claim,
        answer_value={"claim_correct": False, "relation": verified},
        family_variant="claim_false",
        consistency_group=group,
        tags=(*false_claim.tags, "balanced_claim_pair"),
    )
    question_prefix, separator, question_suffix = false_claim.question_zh.rpartition(
        rejected_surface
    )
    if not separator:
        return [false_claim]
    true_question = question_prefix + verified_surface + question_suffix
    true_certificate = {
        **false_claim.certificate,
        "checks": [
            {
                "name": "claim_matches_verified_relation",
                "passed": True,
                "measured_value": verified,
                "threshold": verified,
            },
            *[
                check
                for check in false_claim.certificate.get("checks", ())
                if check.get("name") != "claim_contradicted"
            ],
        ],
        "oracle": {
            **oracle,
            "claimed_relation": verified,
            "verified_relation": verified,
        },
    }
    true_claim = replace(
        false_claim,
        fact_id=stable_id("fact", group, "claim_true"),
        question_zh=true_question,
        answer_zh=f"对；它实际就在参照物体的{verified_surface}。",
        answer_value={"claim_correct": True, "relation": verified},
        certificate=true_certificate,
        rationale_zh=(
            "先将两个不同时出现的物体注册到共同锚定坐标；"
            f"几何关系为{verified_surface}，与题中说法一致。"
        ),
        family_variant="claim_true",
        consistency_group=group,
        tags=(*false_claim.tags, "balanced_claim_pair"),
    )
    return [false_structured, true_claim]


def _cross_view_evidence_family(bundle: Bundle, revealed_spec: QuestionSpec) -> list[QuestionSpec]:
    """Turn a certified cross-view relation into a matched evidence intervention.

    Every member contains three images and keeps the first-camera reference frame
    fixed.  The revealed member contains a resolvable view of both referents;
    the two unknown members replace the decisive referent view with different
    distractors.  This makes the answer change because of visual evidence rather
    than prompt wording, view count, or a hidden production-history token.
    """

    if len(revealed_spec.evidence_entity_ids) != 2 or len(bundle.views) < 4:
        return [revealed_spec]
    subject, reference = (
        bundle.entity(entity_id) for entity_id in revealed_spec.evidence_entity_ids
    )
    # Bare category names are only safe when they denote one scene entity.
    for entity in (subject, reference):
        if sum(candidate.label == entity.label for candidate in bundle.entities.values()) != 1:
            return [revealed_spec]

    subject_anchor = _unique_anchor_view(bundle, subject.entity_id)
    reference_anchor = _unique_anchor_view(bundle, reference.entity_id)
    if subject_anchor is None or reference_anchor is None:
        return [revealed_spec]

    anchor = bundle.views[0]
    candidates = []
    for missing, preserved, preserved_anchor, missing_anchor in (
        (subject, reference, reference_anchor, subject_anchor),
        (reference, subject, subject_anchor, reference_anchor),
    ):
        if missing.entity_id in anchor.visible_entity_ids:
            continue
        if _confusable_category_visible(bundle, missing.label, (anchor.view_id,)):
            continue
        safe_views = [
            view
            for view in bundle.views
            if missing.entity_id not in view.visible_entity_ids
            and not _confusable_category_visible(
                bundle, missing.label, (view.view_id,)
            )
        ]
        base_common_ids = {anchor.view_id, preserved_anchor.view_id}
        if len(base_common_ids) == 2:
            common_candidates = [base_common_ids]
        else:
            common_candidates = [
                {anchor.view_id, view.view_id}
                for view in safe_views
                if view.view_id != anchor.view_id
                and view.view_id != missing_anchor.view_id
            ]
        for common_ids in common_candidates:
            if len(common_ids) != 2 or missing_anchor.view_id in common_ids:
                continue
            common_steps = sorted(bundle.view_by_id[view_id].step for view_id in common_ids)

            def insertion_slot(
                view: Any, boundaries: tuple[int, ...] = tuple(common_steps)
            ) -> int:
                return sum(step < view.step for step in boundaries)

            decisive_slot = insertion_slot(missing_anchor)
            distractors = [
                view
                for view in safe_views
                if view.view_id not in common_ids
                and view.view_id != missing_anchor.view_id
                and insertion_slot(view) == decisive_slot
            ]
            if len(distractors) < 2:
                continue

            def with_extra(
                extra_view: Any, base_view_ids: tuple[str, ...] = tuple(common_ids)
            ) -> tuple[str, ...]:
                selected = [bundle.view_by_id[view_id] for view_id in base_view_ids]
                selected.append(extra_view)
                return tuple(
                    view.view_id for view in sorted(selected, key=lambda view: view.step)
                )

            prefix_views = with_extra(distractors[0])
            deleted_views = with_extra(distractors[-1])
            revealed_views = with_extra(missing_anchor)
            shared_slots_fixed = all(
                prefix_views[index] == deleted_views[index] == revealed_views[index]
                for index in range(3)
                if index != decisive_slot
            )
            if (
                shared_slots_fixed
                and all(
                    view_ids[0] == anchor.view_id
                    for view_ids in (prefix_views, deleted_views, revealed_views)
                )
            ):
                candidates.append(
                    (
                        missing,
                        preserved,
                        tuple(sorted(common_ids)),
                        prefix_views,
                        deleted_views,
                        revealed_views,
                        decisive_slot,
                    )
                )
    if not candidates:
        return [revealed_spec]

    # Prefer the intervention with the largest visual separation between its two
    # distractor choices, then fall back to stable entity order.
    (
        missing,
        preserved,
        common_view_ids,
        prefix_views,
        deleted_views,
        revealed_views,
        intervention_slot,
    ) = max(
        candidates,
        key=lambda item: (
            len(set(item[3]) | set(item[4]) | set(item[5])),
            item[0].entity_id,
        ),
    )
    if (
        len(revealed_views) != 3
        or set(prefix_views) & set(deleted_views) & set(revealed_views)
        != set(common_view_ids)
    ):
        return [revealed_spec]

    relation, margin, dx, dy = bundle.canonical_relation(subject, reference)
    if margin < 0.4 or not _relation_is_unambiguous(dx, dy):
        return [revealed_spec]
    subject_name = entity_name(subject.label)
    reference_name = entity_name(reference.label)
    missing_name = entity_name(missing.label)
    question = (
        "综合当前给出的三张有序观察，以第一张图的相机朝向为正前方，"
        f"{subject_name}位于{reference_name}的哪个方向？"
    )
    group = stable_id(
        "consistency",
        bundle.scene_id,
        bundle.episode_id,
        subject.entity_id,
        reference.entity_id,
        "cross_view_evidence",
    )
    common_checks = [
        {
            "name": "matched_visual_input_count",
            "passed": True,
            "image_count": 3,
        },
        {
            "name": "anchor_reference_frame_fixed",
            "passed": True,
            "anchor_view_id": anchor.view_id,
        },
        {
            "name": "scene_category_referents_unique",
            "passed": True,
            "entity_ids": [subject.entity_id, reference.entity_id],
        },
        {
            "name": "two_shared_views_one_evidence_replacement",
            "passed": True,
            "shared_view_ids": list(common_view_ids),
            "intervention_slot_zero_based": intervention_slot,
        },
    ]
    revealed = replace(
        revealed_spec,
        fact_id=stable_id("fact", group, "revealed"),
        question_zh=question,
        answer_zh=(
            f"可以确定；{subject_name}位于{reference_name}的{relation_name(relation)}。"
        ),
        answer_value=relation,
        evidence_view_ids=revealed_views,
        certificate={
            "schema_version": "epispace.certificate.v1",
            "result": "pass",
            "checks": [
                *common_checks,
                {
                    "name": "both_referents_resolvable_in_actual_input",
                    "passed": all(
                        any(
                            entity.entity_id in bundle.view_by_id[view_id].visible_entity_ids
                            and _referent_is_recognizable(
                                bundle, entity.entity_id, view_id
                            )
                            for view_id in revealed_views
                        )
                        for entity in (subject, reference)
                    ),
                },
                {
                    "name": "canonical_relation_recomputed",
                    "passed": margin > 0,
                    "anchor_view_id": anchor.view_id,
                    "relation": relation,
                    "dx_m": dx,
                    "dy_m": dy,
                    "margin_m": margin,
                },
            ],
        },
        rationale_zh=(
            f"三张图中分别出现了{subject_name}与{reference_name}；"
            "把它们注册到第一张图锚定的坐标系后，"
            f"{subject_name}位于{reference_name}的{relation_name(relation)}。"
        ),
        source="epispace.evidence_family_compiler.cross_view.v1",
        family_variant="revealed",
        consistency_group=group,
        model_view_ids=revealed_views,
        tags=(*revealed_spec.tags, "evidence_intervention", "matched_three_view"),
    )

    def unknown_member(variant: str, view_ids: tuple[str, ...]) -> QuestionSpec:
        visible_entities = set().union(
            *(set(bundle.view_by_id[view_id].visible_entity_ids) for view_id in view_ids)
        )
        return QuestionSpec(
            fact_id=stable_id("fact", group, variant),
            task_type="cross_view_unknown",
            question_zh=question,
            answer_zh=(
                f"还不能确定；当前观察缺少{missing_name}的视觉位置证据。"
            ),
            answer_value=None,
            answer_status="unknown",
            program=program_for("cross_view_unknown"),
            evidence_view_ids=view_ids,
            evidence_entity_ids=(subject.entity_id, reference.entity_id),
            certificate={
                "schema_version": "epispace.certificate.v1",
                "result": "unknown",
                "checks": [
                    *common_checks,
                    {
                        "name": "one_referent_removed_from_actual_input",
                        "passed": (
                            missing.entity_id not in visible_entities
                            and preserved.entity_id in visible_entities
                        ),
                        "missing_entity_id": missing.entity_id,
                        "preserved_entity_id": preserved.entity_id,
                    },
                ],
            },
            rationale_zh=(
                f"当前三张图只提供了{entity_name(preserved.label)}的位置，"
                f"没有{missing_name}的位置证据，因此无法比较两者方向。"
            ),
            source="epispace.evidence_family_compiler.cross_view.v1",
            family_variant=variant,
            consistency_group=group,
            model_view_ids=view_ids,
            tags=("evidence_intervention", "matched_three_view", "epistemic"),
        )

    return [
        unknown_member("prefix_unknown", prefix_views),
        revealed,
        unknown_member("decisive_deleted", deleted_views),
    ]


def _compile_t3(bundle: Bundle) -> list[QuestionSpec]:
    yaws = [float(value) for value in bundle.plan.get("yaw_sequence_deg", ())]
    if len(yaws) != len(bundle.views) or len(bundle.views) < 2:
        return []
    for index, (current, nxt) in enumerate(zip(bundle.views, bundle.views[1:], strict=False)):
        recognizable_before = {
            entity_id
            for entity_id in current.visible_entity_ids
            if entity_id in bundle.entities
            and bundle.entities[entity_id].label not in STRUCTURAL_LABELS
            and _referent_is_recognizable(bundle, entity_id, current.view_id)
        }
        recognizable_after = {
            entity_id
            for entity_id in nxt.visible_entity_ids
            if entity_id in bundle.entities
            and bundle.entities[entity_id].label not in STRUCTURAL_LABELS
            and _referent_is_recognizable(bundle, entity_id, nxt.view_id)
        }
        entered = _eligible_entities(
            bundle, recognizable_after - recognizable_before
        )
        if len(entered) != 1:
            continue
        entity = bundle.entities[entered[0]]
        # A subclass transition such as washer -> clothes dryer is not a
        # visually observable category change when the two appliances share
        # the same appearance.  Do not teach the model to recover hidden asset
        # labels from an RGB pair.
        if _confusable_category_visible(
            bundle, entity.label, (current.view_id,)
        ):
            continue
        if not _bindings_unambiguous(bundle, (entity.entity_id,), (nxt.view_id,)):
            continue
        delta = (yaws[index + 1] - yaws[index]) % 360.0
        name = entity_name(entity.label)
        return [
            QuestionSpec(
                fact_id=stable_id(
                    "fact", bundle.scene_id, bundle.episode_id, "rotation", str(index)
                ),
                task_type="rotation_change_detection",
                question_zh=(
                    "比较前后两个视角："
                    f"保持站位不动并原地旋转约{delta:.0f}°后，"
                    "唯一新进入视野的主要物体是什么？"
                ),
                answer_zh=f"{name}会进入视野。",
                answer_value=entity.label,
                answer_status="accepted",
                program=program_for("rotation_change_detection"),
                evidence_view_ids=(current.view_id, nxt.view_id),
                evidence_entity_ids=(entity.entity_id,),
                certificate={
                    "schema_version": "epispace.certificate.v1",
                    "result": "pass",
                    "checks": [
                        {
                            "name": "fixed_camera_position",
                            "passed": bundle.quality["gates"]["T3"]["checks"]["zero_parallax"],
                        },
                        {"name": "yaw_delta_deg", "passed": True, "measured": delta},
                        {
                            "name": "entity_unrecognizable_before_recognizable_after",
                            "passed": (
                                entity.entity_id not in recognizable_before
                                and entity.entity_id in recognizable_after
                            ),
                        },
                        {
                            "name": "recognizable_entered_entity_count",
                            "passed": len(entered) == 1,
                            "measured": len(entered),
                            "entity_ids": entered,
                            "recognizability_policy_id": (
                                visual_quality_policy()["policy_id"]
                            ),
                        },
                    ],
                },
                rationale_zh=(
                    f"站位不变，只更新朝向；沿采集方向旋转约{delta:.0f}°后，下一视角新增了{name}。"
                ),
                source="epispace.trajectory_compiler.t3.v1",
                model_view_ids=(current.view_id, nxt.view_id),
                tags=("trajectory_specific", "pure_rotation", "change_detection"),
            )
        ]
    return []


def _angular_separation(a: float, b: float) -> float:
    difference = abs((a - b) % 360.0)
    return min(difference, 360.0 - difference)


def _compile_t4(bundle: Bundle) -> list[QuestionSpec]:
    azimuths = [float(value) for value in bundle.plan.get("azimuth_deg_per_view", ())]
    if len(azimuths) != len(bundle.views) or len(bundle.views) < 2:
        return []
    try:
        focus = bundle.entity(str(bundle.plan["focus_entity_id"]))
    except (KeyError, ValueError):
        return []
    candidates: list[tuple[tuple[float, ...], float, int, int, int]] = []
    for left in range(len(azimuths)):
        for right in range(left + 1, len(azimuths)):
            separation = _angular_separation(azimuths[left], azimuths[right])
            left_view, right_view = bundle.views[left], bundle.views[right]
            if (
                separation < 120.0
                or focus.entity_id not in left_view.visible_entity_ids
                or focus.entity_id not in right_view.visible_entity_ids
            ):
                continue
            if not _bindings_unambiguous(
                bundle,
                (focus.entity_id,),
                (left_view.view_id, right_view.view_id),
                require_unique_track=True,
            ):
                continue
            left_quality = _referent_visual_quality(
                bundle, focus.entity_id, left_view.view_id
            )
            right_quality = _referent_visual_quality(
                bundle, focus.entity_id, right_view.view_id
            )
            if left_quality["passed"] and right_quality["passed"]:
                pair_score = min(
                    _referent_selection_score(left_quality),
                    _referent_selection_score(right_quality),
                )
                candidates.append(
                    (pair_score, separation, -(left + right), left, right)
                )
    if not candidates:
        return []
    _, separation, _, left, right = max(candidates)
    left_view, right_view = bundle.views[left], bundle.views[right]
    left_quality = _referent_visual_quality(bundle, focus.entity_id, left_view.view_id)
    right_quality = _referent_visual_quality(bundle, focus.entity_id, right_view.view_id)
    name = entity_name(focus.label)
    return [
        QuestionSpec(
            fact_id=stable_id("fact", bundle.scene_id, bundle.episode_id, "orbit_identity"),
            task_type="orbit_identity",
            question_zh=(f"这两个视角相隔约{separation:.0f}°，画面中的{name}是同一个物体吗？"),
            answer_zh=f"是同一个{name}，只是观察方位发生了变化。",
            answer_value=True,
            answer_status="accepted",
            program=program_for("orbit_identity"),
            evidence_view_ids=(left_view.view_id, right_view.view_id),
            evidence_entity_ids=(focus.entity_id,),
            certificate={
                "schema_version": "epispace.certificate.v1",
                "result": "pass",
                "checks": [
                    {
                        "name": "stable_focus_identity",
                        "passed": True,
                        "entity_id": focus.entity_id,
                    },
                    {
                        "name": "azimuth_separation_deg",
                        "passed": separation >= 120.0,
                        "measured": round(separation, 3),
                        "threshold": 120.0,
                    },
                    {
                        "name": "focus_visible_in_both",
                        "passed": True,
                        "left_referent_visual_quality": left_quality,
                        "right_referent_visual_quality": right_quality,
                    },
                    {
                        "name": "surface_identity_unique_in_both_views",
                        "passed": True,
                        "model_facing_category_labels": sorted(
                            _visually_confusable_labels(focus.label)
                        ),
                    },
                ],
            },
            rationale_zh=(
                f"两帧都绑定到同一稳定实体；相机围绕它移动约{separation:.0f}°，物体身份没有变化。"
            ),
            source="epispace.trajectory_compiler.t4.v1",
            model_view_ids=(left_view.view_id, right_view.view_id),
            tags=(
                "trajectory_specific",
                "object_centric",
                "partial_arc_safe",
                "training_only",
            ),
        )
    ]


def _compile_t7(bundle: Bundle) -> list[QuestionSpec]:
    if len(bundle.views) < 2:
        return []
    low = min(
        bundle.views,
        key=lambda view: float(bundle.plan["views"][view.step].get("camera_height_m", math.inf)),
    )
    high = max(
        bundle.views,
        key=lambda view: float(bundle.plan["views"][view.step].get("camera_height_m", -math.inf)),
    )
    common = _eligible_entities(bundle, set(low.visible_entity_ids) & set(high.visible_entity_ids))
    for left_index, left_id in enumerate(common):
        for right_id in common[left_index + 1 :]:
            left = bundle.entities[left_id]
            right = bundle.entities[right_id]
            if left.label == right.label:
                continue
            if not _bindings_unambiguous(
                bundle,
                (left.entity_id, right.entity_id),
                (low.view_id, high.view_id),
            ) or not all(
                _referent_is_recognizable(bundle, entity.entity_id, view.view_id)
                for entity in (left, right)
                for view in (low, high)
            ):
                continue
            relation, margin, delta_right, delta_front = bundle.canonical_relation(left, right)
            if margin < 0.4 or not _relation_is_unambiguous(
                delta_right, delta_front
            ):
                continue
            left_name, right_name = entity_name(left.label), entity_name(right.label)
            relation_zh = relation_name(relation)
            return [
                QuestionSpec(
                    fact_id=stable_id(
                        "fact",
                        bundle.scene_id,
                        bundle.episode_id,
                        "elevation_relation_transfer",
                    ),
                    task_type="elevation_relation_transfer",
                    question_zh=(
                        "以低视角的朝向为正前方。"
                        f"相机从低视角升到高视角后，{left_name}位于"
                        f"{right_name}的哪个方向？"
                    ),
                    answer_zh=f"{left_name}仍在{right_name}的{relation_zh}。",
                    answer_value=relation,
                    answer_status="accepted",
                    program=program_for("elevation_relation_transfer"),
                    evidence_view_ids=(low.view_id, high.view_id),
                    evidence_entity_ids=(left.entity_id, right.entity_id),
                    certificate={
                        "schema_version": "epispace.certificate.v1",
                        "result": "pass",
                        "checks": [
                            {
                                "name": "same_station",
                                "passed": True,
                                "station_id": bundle.plan.get("station_id"),
                            },
                            {
                                "name": "entities_visible_both_heights",
                                "passed": True,
                            },
                            {
                                "name": "canonical_relation_margin_m",
                                "passed": margin >= 0.4,
                                "measured": round(margin, 3),
                                "threshold": 0.4,
                            },
                            {
                                "name": "relation_dominance_ratio",
                                "passed": True,
                                "measured": (
                                    max(abs(delta_right), abs(delta_front))
                                    / min(abs(delta_right), abs(delta_front))
                                    if min(abs(delta_right), abs(delta_front)) > 0
                                    else None
                                ),
                                "threshold": MIN_RELATION_DOMINANCE_RATIO,
                            },
                            {
                                "name": "canonical_frame",
                                "passed": True,
                                "anchor_view_id": bundle.views[0].view_id,
                                "contract": "+X camera right, +Y camera forward",
                            },
                            {
                                "name": "canonical_relation",
                                "passed": True,
                                "relation": relation,
                                "delta_right_m": round(delta_right, 3),
                                "delta_front_m": round(delta_front, 3),
                            },
                        ],
                    },
                    rationale_zh=(
                        "高度和俯仰改变了投影外观，但没有移动场景实体；"
                        f"回到共同水平坐标后，{left_name}仍在{right_name}的{relation_zh}。"
                    ),
                    source="epispace.trajectory_compiler.t7.v2",
                    model_view_ids=(low.view_id, high.view_id),
                    tags=("trajectory_specific", "elevation", "frame_transfer"),
                )
            ]
    return []


def _compile_t8(bundle: Bundle) -> list[QuestionSpec]:
    annotation = bundle.plan.get("occlusion_annotation")
    if not annotation:
        return []
    try:
        target = bundle.entity(str(annotation["target_id"]))
    except (KeyError, ValueError):
        return []
    annotated_decisive = str(annotation["decisive_view"])
    if annotated_decisive not in bundle.view_by_id:
        return []
    occluded_view_ids = tuple(
        sorted(
            (
                str(view_id)
                for view_id in annotation.get("occluded_views", ())
                if str(view_id) in bundle.view_by_id
            ),
            key=lambda view_id: bundle.view_by_id[view_id].step,
        )
    )
    if not occluded_view_ids:
        return []
    last_occluded_step = max(bundle.view_by_id[view_id].step for view_id in occluded_view_ids)
    prefix = tuple(view.view_id for view in bundle.views if view.step <= last_occluded_step)
    target_category_ids = {
        entity.entity_id for entity in bundle.entities.values() if entity.label == target.label
    }

    def category_visible(view_id: str) -> bool:
        return bool(target_category_ids & set(bundle.view_by_id[view_id].visible_entity_ids))

    clean_witnesses: list[tuple[tuple[float, ...], str, dict[str, Any]]] = []
    for view in bundle.views:
        if (
            view.step <= last_occluded_step
            or target.entity_id not in view.visible_entity_ids
        ):
            continue
        quality = _referent_visual_quality(bundle, target.entity_id, view.view_id)
        if not quality["passed"]:
            continue
        score = (*_referent_selection_score(quality), -float(view.step))
        clean_witnesses.append((score, view.view_id, quality))
    if not clean_witnesses:
        return []
    _, decisive, decisive_quality = max(
        clean_witnesses, key=lambda item: (item[0], item[1])
    )
    decisive_step = bundle.view_by_id[decisive].step

    non_target_views = [
        view.view_id
        for view in bundle.views
        if not _confusable_category_visible(bundle, target.label, (view.view_id,))
    ]
    if len(non_target_views) < 2:
        return []
    anchor_view = non_target_views[0]
    prefix = (anchor_view, non_target_views[1])
    deleted_views = (
        anchor_view,
        non_target_views[2] if len(non_target_views) > 2 else non_target_views[1],
    )
    revealed_views = (anchor_view, decisive)
    target_visible_prefix = any(
        _confusable_category_visible(bundle, target.label, (view_id,))
        for view_id in prefix
    )
    target_visible_decisive = category_visible(decisive)
    decisive_pixels = int(decisive_quality["visible_pixels"])
    if (
        not prefix
        or decisive_step <= last_occluded_step
        or target_visible_prefix
        or not target_visible_decisive
        or not decisive_quality["passed"]
    ):
        return []

    target_name = entity_name(target.label)
    group = stable_id(
        "consistency",
        bundle.scene_id,
        bundle.episode_id,
        "evidence",
        decisive,
    )
    question = f"仅根据当前给出的视角，能否确定这个场景里存在{target_name}？"
    visible_target_views = {view.view_id for view in bundle.views if category_visible(view.view_id)}
    common = {
        "question_zh": question,
        "evidence_entity_ids": (target.entity_id,),
        "consistency_group": group,
        "source": "epispace.trajectory_compiler.t8.v3",
    }
    return [
        QuestionSpec(
            fact_id=stable_id("fact", group, "prefix_unknown"),
            task_type="occlusion_unknown",
            answer_zh=f"还不能确定；当前画面里没有观察到{target_name}。",
            answer_value=None,
            answer_status="unknown",
            program=program_for("occlusion_unknown"),
            evidence_view_ids=prefix,
            certificate={
                "schema_version": "epispace.certificate.v1",
                "result": "unknown",
                "checks": [
                    {
                        "name": "target_category_absent_from_prefix",
                        "passed": not target_visible_prefix,
                    },
                    {
                        "name": "decisive_view_withheld",
                        "passed": decisive not in prefix,
                        "decisive_view": decisive,
                        "annotated_decisive_view": annotated_decisive,
                        "quality_selected_fallback": decisive != annotated_decisive,
                        "visible_pixels": decisive_pixels,
                        "minimum_pixels": MIN_REFERENT_PIXELS,
                        "referent_visual_quality": decisive_quality,
                    },
                ],
            },
            rationale_zh=(f"当前画面没有{target_name}的视觉证据，因此必须暂时弃答。"),
            family_variant="prefix_unknown",
            model_view_ids=prefix,
            tags=("trajectory_specific", "epistemic", "unknown_before_reveal"),
            **common,
        ),
        QuestionSpec(
            fact_id=stable_id("fact", group, "revealed"),
            task_type="occlusion_reveal",
            answer_zh=f"能确定；后一个视角看到了{target_name}。",
            answer_value={"status": "present", "category": target.label},
            answer_status="accepted",
            program=program_for("occlusion_reveal"),
            evidence_view_ids=revealed_views,
            certificate={
                "schema_version": "epispace.certificate.v1",
                "result": "pass",
                "checks": [
                    {
                        "name": "target_category_absent_before_reveal",
                        "passed": not target_visible_prefix,
                    },
                    {
                        "name": "target_category_visible_in_decisive_view",
                        "passed": target_visible_decisive,
                        "decisive_view": decisive,
                        "annotated_decisive_view": annotated_decisive,
                        "quality_selected_fallback": decisive != annotated_decisive,
                        "referent_visual_quality": decisive_quality,
                    },
                ],
            },
            rationale_zh=(
                f"前一个视角中没有出现{target_name}；后一个决定性视角里出现了{target_name}。"
            ),
            family_variant="revealed",
            model_view_ids=revealed_views,
            tags=("trajectory_specific", "epistemic", "unknown_to_known"),
            **common,
        ),
        QuestionSpec(
            fact_id=stable_id("fact", group, "decisive_deleted"),
            task_type="occlusion_unknown",
            answer_zh=f"还不能确定；当前画面里没有观察到{target_name}。",
            answer_value=None,
            answer_status="unknown",
            program=program_for("occlusion_unknown"),
            evidence_view_ids=deleted_views,
            certificate={
                "schema_version": "epispace.certificate.v1",
                "result": "unknown",
                "checks": [
                    {
                        "name": "all_target_category_views_deleted",
                        "passed": not any(category_visible(view_id) for view_id in deleted_views),
                        "deleted_view_ids": sorted(visible_target_views),
                    }
                ],
            },
            rationale_zh=(f"当前画面没有{target_name}的视觉证据，因此必须暂时弃答。"),
            family_variant="decisive_deleted",
            model_view_ids=deleted_views,
            tags=("trajectory_specific", "epistemic", "evidence_deletion"),
            **common,
        ),
    ]


def compile_t10_predictions(
    target_bundle: Bundle,
    source_bundle: Bundle,
) -> list[QuestionSpec]:
    """Attach held-out T10 renders to a source trajectory from the same scene."""

    if target_bundle.trajectory_class != "T10":
        raise ValueError("target_bundle must be T10")
    if target_bundle.scene_id != source_bundle.scene_id:
        raise ValueError("T10 and source bundle must share scene_id")
    if any(
        not _model_input_frame_quality(source_bundle, view.view_id)["passed"]
        for view in source_bundle.views
    ):
        # Every generated T10 item below exposes the complete source trajectory.
        # A bad context frame therefore invalidates the source prefix itself.
        return []
    observed_source = set().union(*(set(view.visible_entity_ids) for view in source_bundle.views))
    anchors = target_bundle.plan.get("target_anchors", ())
    specs: list[QuestionSpec] = []
    for anchor in anchors:
        target_view_id = str(anchor["view_id"])
        if target_view_id not in target_bundle.view_by_id:
            continue
        try:
            origin = source_bundle.entity(str(anchor["origin_entity_id"]))
            facing = source_bundle.entity(str(anchor["facing_entity_id"]))
        except ValueError:
            continue
        origin_anchor = _unique_anchor_view(
            source_bundle, origin.entity_id, t10_salient=True
        )
        facing_anchor = _unique_anchor_view(
            source_bundle, facing.entity_id, t10_salient=True
        )
        if (
            origin.entity_id not in observed_source
            or facing.entity_id not in observed_source
            or origin_anchor is None
            or facing_anchor is None
        ):
            continue
        target_visible = target_bundle.view_by_id[target_view_id].visible_entity_ids
        excluded = {origin.entity_id, facing.entity_id}
        positive = _eligible_entities(
            source_bundle, (set(target_visible) & observed_source) - excluded
        )
        negative = _eligible_entities(
            source_bundle, observed_source - set(target_visible) - excluded
        )
        candidates: list[tuple[str, bool, Any]] = []
        for entity_ids, visible in ((positive, True), (negative, False)):
            # Stable category order is useful for reproducibility, but the first
            # category may only be a thin/cropped mask.  Search until a genuine
            # model-visible anchor is found instead of discarding the signature.
            for candidate_id in entity_ids:
                if visible and not _referent_is_recognizable(
                    target_bundle, candidate_id, target_view_id
                ):
                    continue
                candidate_anchor = _unique_anchor_view(
                    source_bundle, candidate_id, t10_salient=True
                )
                if candidate_anchor is not None:
                    candidates.append((candidate_id, visible, candidate_anchor))
                    break
        for candidate_id, visible, candidate_anchor in candidates:
            candidate = source_bundle.entities[candidate_id]
            target_quality = (
                _referent_visual_quality(target_bundle, candidate_id, target_view_id)
                if visible
                else None
            )
            origin_name = entity_name(origin.label)
            facing_name = entity_name(facing.label)
            candidate_name = entity_name(candidate.label)
            answer = "能看到。" if visible else "看不到。"
            fact_id = stable_id(
                "fact",
                source_bundle.scene_id,
                target_bundle.episode_id,
                target_view_id,
                candidate_id,
            )
            specs.append(
                QuestionSpec(
                    fact_id=fact_id,
                    task_type="target_view_prediction",
                    question_zh=(
                        f"假设站在第{view_number(origin_anchor.view_id)}个视角里看到的"
                        f"{origin_name}的位置，并面向第{view_number(facing_anchor.view_id)}个"
                        f"视角里看到的{facing_name}，此时能看到第"
                        f"{view_number(candidate_anchor.view_id)}个视角里看到的"
                        f"{candidate_name}吗？"
                    ),
                    answer_zh=answer,
                    answer_value=visible,
                    answer_status="accepted",
                    program=program_for("target_view_prediction"),
                    evidence_view_ids=tuple(view.view_id for view in source_bundle.views),
                    evidence_entity_ids=(
                        origin.entity_id,
                        facing.entity_id,
                        candidate.entity_id,
                    ),
                    certificate={
                        "schema_version": "epispace.certificate.v1",
                        "result": "pass",
                        "checks": [
                            {
                                "name": "target_render_held_out",
                                "passed": True,
                                "target_view_id": target_view_id,
                            },
                            {
                                "name": "candidate_visibility_in_target_render",
                                "passed": True,
                                "measured": visible,
                                "positive_referent_visual_quality": target_quality,
                            },
                            {
                                "name": "candidate_observed_in_write_prefix",
                                "passed": candidate.entity_id in observed_source,
                            },
                            {
                                "name": "surface_referents_uniquely_anchored",
                                "passed": True,
                                "origin_anchor_view_id": origin_anchor.view_id,
                                "facing_anchor_view_id": facing_anchor.view_id,
                                "candidate_anchor_view_id": candidate_anchor.view_id,
                                "minimum_visible_pixels": MIN_T10_ANCHOR_PIXELS,
                                "minimum_bbox_side_px": MIN_T10_ANCHOR_BBOX_SIDE_PX,
                                "maximum_outer_five_percent_pixel_fraction": (
                                    MAX_T10_ANCHOR_OUTER_BAND_PIXEL_FRACTION
                                ),
                            },
                        ],
                        "oracle_target_bundle": str(target_bundle.root),
                        "oracle_target_view_id": target_view_id,
                        "origin_entity_id": origin.entity_id,
                        "facing_entity_id": facing.entity_id,
                        "candidate_entity_id": candidate.entity_id,
                        "oracle_target_rgb": str(target_bundle.view_by_id[target_view_id].rgb_path),
                    },
                    rationale_zh=(
                        f"先从观察序列建立共享布局，再以{origin_name}为原点、"
                        f"朝向{facing_name}重建目标视角；渲染真值表明{answer}"
                    ),
                    source="epispace.trajectory_compiler.t10.v1",
                    model_view_ids=tuple(view.view_id for view in source_bundle.views),
                    oracle_held_out_view_ids=(target_view_id,),
                    tags=("trajectory_specific", "held_out_render", "composition_holdout"),
                )
            )
    if specs:
        return specs
    # Some target poses are defined by objects that are geometrically valid but
    # not uniquely nameable in any source frame.  Do not weaken grounding.  A
    # relative camera-motion instruction is an equally typed target-view input
    # and preserves the held-out-render verification contract.
    for anchor in anchors:
        target_view_id = str(anchor["view_id"])
        if target_view_id in target_bundle.view_by_id:
            specs.extend(
                _compile_t10_pose_instruction(
                    target_bundle,
                    source_bundle,
                    target_view_id,
                    observed_source,
                )
            )
    return specs


def _compile_t10_pose_instruction(
    target_bundle: Bundle,
    source_bundle: Bundle,
    target_view_id: str,
    observed_source: set[str],
) -> list[QuestionSpec]:
    target_view = target_bundle.view_by_id[target_view_id]
    target_translation = target_view.world_from_camera["translation_m"]
    source_view = min(
        source_bundle.views,
        key=lambda view: (
            math.hypot(
                float(target_translation[0])
                - float(view.world_from_camera["translation_m"][0]),
                float(target_translation[1])
                - float(view.world_from_camera["translation_m"][1]),
            ),
            view.step,
        ),
    )
    source_translation = source_view.world_from_camera["translation_m"]
    world_dx = float(target_translation[0]) - float(source_translation[0])
    world_dy = float(target_translation[1]) - float(source_translation[1])
    source_yaw = _camera_yaw_rad(source_bundle, source_view.view_id)
    right_m = world_dx * math.cos(source_yaw) + world_dy * math.sin(source_yaw)
    front_m = -world_dx * math.sin(source_yaw) + world_dy * math.cos(source_yaw)
    target_yaw = _camera_yaw_rad(target_bundle, target_view_id)
    yaw_delta_deg = math.degrees(target_yaw - source_yaw)
    yaw_delta_deg = (yaw_delta_deg + 180.0) % 360.0 - 180.0

    target_visible = set(target_view.visible_entity_ids)
    candidates: list[tuple[str, bool, Any, dict[str, Any] | None]] = []
    for entity_ids, visible in (
        (_eligible_entities(source_bundle, target_visible & observed_source), True),
        (_eligible_entities(source_bundle, observed_source - target_visible), False),
    ):
        for candidate_id in entity_ids:
            candidate_anchor = _unique_anchor_view(
                source_bundle, candidate_id, t10_salient=True
            )
            if candidate_anchor is None:
                continue
            target_quality = (
                _referent_visual_quality(target_bundle, candidate_id, target_view_id)
                if visible
                else None
            )
            if visible and not target_quality["passed"]:
                continue
            candidates.append((candidate_id, visible, candidate_anchor, target_quality))
            break

    def translation_phrase(
        value: float, positive: str, negative: str, stationary: str
    ) -> str:
        if abs(value) < 0.05:
            return stationary
        direction = positive if value >= 0 else negative
        return f"向{direction}{abs(value):.1f}米"

    def rotation_phrase(value: float) -> str:
        if abs(value) < 0.5:
            return "保持朝向"
        direction = "左" if value > 0 else "右"
        return f"再原地向{direction}转{abs(value):.0f}度"

    instruction = (
        f"从第{view_number(source_view.view_id)}个视角的相机位置和朝向出发，"
        f"先{translation_phrase(right_m, '右', '左', '保持横向位置')}、"
        f"{translation_phrase(front_m, '前', '后', '保持前后位置')}，"
        f"{rotation_phrase(yaw_delta_deg)}"
    )
    specs: list[QuestionSpec] = []
    for candidate_id, visible, candidate_anchor, target_quality in candidates:
        candidate = source_bundle.entities[candidate_id]
        candidate_name = entity_name(candidate.label)
        answer = "能看到。" if visible else "看不到。"
        fact_id = stable_id(
            "fact",
            source_bundle.scene_id,
            target_bundle.episode_id,
            target_view_id,
            candidate_id,
            "pose_instruction",
        )
        specs.append(
            QuestionSpec(
                fact_id=fact_id,
                task_type="target_view_prediction",
                question_zh=(
                    f"{instruction}。到达这个目标视角后，能看到第"
                    f"{view_number(candidate_anchor.view_id)}个视角里看到的"
                    f"{candidate_name}吗？"
                ),
                answer_zh=answer,
                answer_value=visible,
                answer_status="accepted",
                program=program_for("target_view_prediction"),
                evidence_view_ids=tuple(view.view_id for view in source_bundle.views),
                evidence_entity_ids=(candidate.entity_id,),
                certificate={
                    "schema_version": "epispace.certificate.v1",
                    "result": "pass",
                    "checks": [
                        {
                            "name": "target_render_held_out",
                            "passed": True,
                            "target_view_id": target_view_id,
                        },
                        {
                            "name": "relative_pose_instruction_exposed",
                            "passed": True,
                            "source_view_id": source_view.view_id,
                            "right_m": round(right_m, 6),
                            "front_m": round(front_m, 6),
                            "yaw_delta_deg": round(yaw_delta_deg, 6),
                            "surface_translation_precision_m": 0.1,
                            "surface_rotation_precision_deg": 1.0,
                        },
                        {
                            "name": "candidate_visibility_in_target_render",
                            "passed": True,
                            "measured": visible,
                            "positive_referent_visual_quality": target_quality,
                        },
                        {
                            "name": "candidate_uniquely_anchored_in_source",
                            "passed": True,
                            "candidate_anchor_view_id": candidate_anchor.view_id,
                            "minimum_visible_pixels": MIN_T10_ANCHOR_PIXELS,
                            "minimum_bbox_side_px": MIN_T10_ANCHOR_BBOX_SIDE_PX,
                            "maximum_outer_five_percent_pixel_fraction": (
                                MAX_T10_ANCHOR_OUTER_BAND_PIXEL_FRACTION
                            ),
                        },
                    ],
                    "oracle_target_bundle": str(target_bundle.root),
                    "oracle_target_view_id": target_view_id,
                    "candidate_entity_id": candidate.entity_id,
                    "oracle_target_rgb": str(target_view.rgb_path),
                },
                rationale_zh=(
                    "先把题面给出的相对平移和旋转复合到观察轨迹建立的布局中，"
                    f"再预测目标视锥；隐藏渲染真值表明{answer}"
                ),
                source="epispace.trajectory_compiler.t10_pose_instruction.v1",
                model_view_ids=tuple(view.view_id for view in source_bundle.views),
                oracle_held_out_view_ids=(target_view_id,),
                tags=(
                    "trajectory_specific",
                    "held_out_render",
                    "composition_holdout",
                    "relative_pose_instruction",
                ),
            )
        )
    return specs
