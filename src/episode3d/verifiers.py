"""Independent replay of compiled targets from bundle truth.

The replay path does not trust a stored ``passed`` flag.  It recomputes the
surface target from entity geometry, actual model-visible views, or a held-out
target observation and is used as a fail-closed corpus gate.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from episode3d.bundles import Bundle, Entity, read_json
from episode3d.compilers import STRUCTURAL_LABELS, visual_quality_policy
from episode3d.language import entity_name, view_number
from episode3d.models import QuestionSpec


def replay_question(bundle: Bundle, spec: QuestionSpec) -> tuple[bool, str]:
    try:
        admission_passed, admission_detail = _replay_visual_admission(bundle, spec)
        if not admission_passed:
            return False, f"admission_replay_failed:{admission_detail}"
        actual = _recompute(bundle, spec)
    except (KeyError, ValueError, IndexError, OSError) as error:
        return False, f"replay_error:{error}"
    expected = spec.answer_value
    if spec.task_type == "metric_distance":
        passed = isinstance(expected, int | float) and abs(float(expected) - actual) <= 0.11
    else:
        passed = actual == expected
    return passed, (
        f"expected={expected!r};actual={actual!r};admission={admission_detail}"
    )


_INSTANCE_BOUND_REPLAY_TASKS = {
    "last_seen_memory",
    "metric_distance",
    "egocentric_relation",
    "cross_view_relation",
    "counterfactual_verification",
    "object_centric_perspective",
    "orbit_identity",
    "target_view_prediction",
}

_IDENTITY_TRACKING_REPLAY_TASKS = frozenset(
    {"last_seen_memory", "orbit_identity"}
)


def _replay_visual_admission(bundle: Bundle, spec: QuestionSpec) -> tuple[bool, str]:
    """Independently reapply visual and directional admission from raw masks."""

    entities = [bundle.entity(entity_id) for entity_id in spec.evidence_entity_ids]
    model_views = spec.model_view_ids or tuple(view.view_id for view in bundle.views)
    inadmissible_views = tuple(
        view_id for view_id in model_views if not _frame_rgb_quality_passes(bundle, view_id)
    )
    if inadmissible_views:
        return False, f"inadmissible actual model input views={inadmissible_views}"
    binding_views = (
        model_views
        if spec.task_type == "last_seen_memory"
        else spec.evidence_view_ids
    )
    if spec.task_type in _INSTANCE_BOUND_REPLAY_TASKS and not _replay_bindings(
        bundle,
        entities,
        binding_views,
        spec.question_zh,
        t10_salient=spec.task_type == "target_view_prediction",
        require_unique_track=spec.task_type in _IDENTITY_TRACKING_REPLAY_TASKS,
    ):
        return False, "surface binding has no unique recognizable witness"
    if (
        spec.task_type == "grounding_presence"
        and spec.answer_value is True
        and not any(
            _mask_quality_passes(bundle, entities[0].entity_id, view_id)
            for view_id in spec.evidence_view_ids
            if entities[0].entity_id in bundle.view_by_id[view_id].visible_entity_ids
        )
    ):
        return False, "positive grounding has no recognizable witness"
    if (
        spec.task_type in {"evidence_presence_reveal", "occlusion_reveal"}
        and not any(
            entities[0].entity_id in bundle.view_by_id[view_id].visible_entity_ids
            and _mask_quality_passes(bundle, entities[0].entity_id, view_id)
            for view_id in model_views
        )
    ):
        return False, "revealed target is not recognizable"
    if spec.task_type == "rotation_change_detection":
        before_view_id, after_view_id = spec.evidence_view_ids
        recognizable_before = _recognizable_entity_ids(bundle, before_view_id)
        recognizable_after = _recognizable_entity_ids(bundle, after_view_id)
        entered = recognizable_after - recognizable_before
        if entered != {entities[0].entity_id}:
            return False, (
                "rotation recognizable entrant set is not the claimed singleton: "
                f"{sorted(entered)}"
            )
    if (
        spec.task_type in {"orbit_identity", "elevation_relation_transfer"}
        and not all(
            _mask_quality_passes(bundle, entity.entity_id, view_id)
            for entity in entities
            for view_id in spec.evidence_view_ids
        )
    ):
        return False, "multi-view referent is not recognizable in every compared view"
    if spec.task_type == "target_view_prediction" and spec.answer_value is True:
        target_bundle = Bundle(
            Path(str(spec.certificate["oracle_target_bundle"])),
            trajectory_class="T10",
            source_sweep="certificate_replay",
            job_status="passed",
        )
        target_view_id = str(spec.certificate["oracle_target_view_id"])
        candidate_id = str(spec.certificate["candidate_entity_id"])
        if not _mask_quality_passes(target_bundle, candidate_id, target_view_id):
            return False, "positive held-out target is not recognizable"
    if spec.task_type == "last_seen_memory" and not all(
        entities[0].entity_id in bundle.view_by_id[view_id].visible_entity_ids
        and _mask_quality_passes(bundle, entities[0].entity_id, view_id)
        for view_id in spec.evidence_view_ids
    ):
        return False, "temporal track contains an unrecognizable witness"
    if (
        spec.task_type == "grounding_presence"
        and spec.answer_value is False
        and any(
            _confusable_category_visible(bundle, entities[0].label, view_id)
            for view_id in spec.evidence_view_ids
        )
    ):
        return False, "negative grounding contains a confusable category"

    relation_components = _replay_relation_components(bundle, spec, entities)
    if relation_components is not None:
        dominant = max(abs(value) for value in relation_components)
        minor = min(abs(value) for value in relation_components)
        ratio = math.inf if minor == 0.0 else dominant / minor
        threshold = float(visual_quality_policy()["minimum_relation_dominance_ratio"])
        if ratio < threshold:
            return False, f"direction ratio {ratio:.6f} < {threshold:.6f}"
    return True, "visual_and_directional_policy_reexecuted"


def _replay_bindings(
    bundle: Bundle,
    entities: list[Entity],
    evidence_view_ids: tuple[str, ...],
    question: str,
    *,
    t10_salient: bool = False,
    require_unique_track: bool = False,
) -> bool:
    for entity in entities:
        visible = [
            view_id
            for view_id in evidence_view_ids
            if entity.entity_id in bundle.view_by_id[view_id].visible_entity_ids
        ]
        if not visible:
            return False
        name = entity_name(entity.label)
        longer_names = {
            entity_name(candidate.label)
            for candidate in bundle.entities.values()
            if len(entity_name(candidate.label)) > len(name)
            and entity_name(candidate.label).startswith(name)
        }
        explicit = [
            view_id
            for view_id in visible
            if _has_exact_replay_anchor(
                question,
                view_number(view_id),
                name,
                longer_names,
            )
        ]
        candidates = evidence_view_ids if require_unique_track else explicit or visible
        for view_id in candidates:
            recognizable = _replay_recognizable_category_candidates(
                bundle,
                entity.label,
                view_id,
                t10_salient=t10_salient,
            )
            target_visible = (
                entity.entity_id in bundle.view_by_id[view_id].visible_entity_ids
            )
            if target_visible:
                if recognizable != (entity.entity_id,):
                    return False
            elif require_unique_track and recognizable:
                return False
    return True


def _replay_recognizable_category_candidates(
    bundle: Bundle,
    label: str,
    view_id: str,
    *,
    t10_salient: bool = False,
) -> tuple[str, ...]:
    """Independently recover usable candidates for a model-facing category."""

    policy = visual_quality_policy()
    labels = next(
        (
            set(group)
            for group in policy["visually_confusable_category_groups"]
            if label in group
        ),
        {label},
    )
    return tuple(
        sorted(
            entity_id
            for entity_id in bundle.view_by_id[view_id].visible_entity_ids
            if entity_id in bundle.entities
            and bundle.entities[entity_id].label in labels
            and _mask_quality_passes(
                bundle,
                entity_id,
                view_id,
                t10_salient=t10_salient,
            )
        )
    )


def _has_exact_replay_anchor(
    question: str,
    number: int,
    name: str,
    longer_names: set[str],
) -> bool:
    """Independent longest-surface replay for Chinese instance references."""

    prefixes = (
        f"第{number}个视角里看到的",
        f"第{number}个视角中看到的",
        f"第{number}个视角看到的",
    )
    for prefix in prefixes:
        needle = f"{prefix}{name}"
        start = question.find(needle)
        while start >= 0:
            name_start = start + len(prefix)
            if not any(question.startswith(longer, name_start) for longer in longer_names):
                return True
            start = question.find(needle, start + 1)
    return False


def _mask_quality_passes(
    bundle: Bundle,
    entity_id: str,
    view_id: str,
    *,
    t10_salient: bool = False,
) -> bool:
    stats = bundle.instance_visual_stats(entity_id, view_id)
    policy = visual_quality_policy()
    pixels = int(stats["visible_pixels"])
    minimum_side = min(int(stats["bbox_width_px"]), int(stats["bbox_height_px"]))
    undersized = minimum_side < int(policy["minimum_bbox_side_px"])
    fragmented = float(stats["mask_bbox_fill_ratio"]) < float(
        policy["minimum_mask_bbox_fill_ratio"]
    )
    clipped_axis_fraction = min(
        int(stats["bbox_width_px"]) / max(int(stats["image_width_px"]), 1),
        int(stats["bbox_height_px"]) / max(int(stats["image_height_px"]), 1),
    )
    outer_fraction = float(stats["outer_five_percent_pixel_fraction"])
    border_count = len(stats["border_sides"])
    border_sides = set(stats["border_sides"])
    severe_crop = (
        (border_count >= 3 and {"top", "bottom"}.issubset(border_sides))
        or (
            border_count >= 2
            and outer_fraction
            >= float(policy["severe_multi_border_outer_band_pixel_fraction"])
        )
        or (
            bool(stats["touches_image_border"])
            and outer_fraction
            >= float(policy["severe_single_border_outer_band_pixel_fraction"])
            and clipped_axis_fraction
            < float(policy["maximum_severely_clipped_axis_fraction"])
        )
    )
    inadmissible_frame = not _frame_rgb_quality_passes(bundle, view_id)
    entity = bundle.entity(entity_id)
    maximum_height = policy["category_max_height_m"].get(entity.label)
    implausible_geometry = (
        maximum_height is not None and entity.extent_m[2] > float(maximum_height)
    )
    base_passed = (
        pixels >= int(policy["minimum_visible_pixels"])
        and not undersized
        and not fragmented
        and not severe_crop
        and not inadmissible_frame
        and not implausible_geometry
    )
    if not t10_salient:
        return base_passed
    return (
        base_passed
        and pixels >= int(policy["minimum_t10_anchor_pixels"])
        and minimum_side >= int(policy["minimum_t10_anchor_bbox_side_px"])
        and outer_fraction
        < float(policy["maximum_t10_anchor_outer_band_pixel_fraction"])
    )


def _frame_rgb_quality_passes(bundle: Bundle, view_id: str) -> bool:
    """Independently replay RGB degeneration and camera-collision gates."""

    policy = visual_quality_policy()
    rgb = bundle.rgb_visual_stats(view_id)
    degenerate_rgb = (
        float(rgb["dominant_quantized_color_fraction"])
        >= float(policy["maximum_rgb_dominant_color_fraction"])
        and float(rgb["quantized_color_entropy_bits"])
        < float(policy["minimum_rgb_quantized_entropy_bits"])
    )
    collision = bundle.frame_collision_stats(view_id)
    legacy_camera_collision = (
        float(collision["near_nonstructural_pixel_fraction"])
        >= float(policy["maximum_near_nonstructural_pixel_fraction"])
        or float(collision["near_geometry_pixel_fraction"])
        >= float(policy["maximum_near_any_geometry_pixel_fraction"])
    )
    context = bundle.frame_context_stats(view_id)
    near_surface_saturation = (
        float(context["close_geometry_pixel_fraction"])
        >= float(policy["maximum_close_geometry_pixel_fraction"])
    )
    near_enclosure = (
        float(context["enclosure_pixel_fraction"])
        >= float(policy["minimum_near_enclosure_pixel_fraction"])
        and float(context["depth_median_m"])
        <= float(policy["maximum_near_enclosure_median_depth_m"])
        and float(context["depth_p90_minus_p10_m"])
        <= float(policy["maximum_near_enclosure_depth_spread_m"])
    )
    foreground = context["dominant_nonstructural_instance"]
    foreground_depth = foreground.get("median_depth_m")
    foreground_occlusion = (
        float(foreground["pixel_fraction"])
        >= float(policy["minimum_dominant_foreground_pixel_fraction"])
        and set(policy["dominant_foreground_required_border_sides"]).issubset(
            set(foreground["border_sides"])
        )
        and isinstance(foreground_depth, int | float)
        and not isinstance(foreground_depth, bool)
        and float(foreground_depth)
        <= float(policy["maximum_dominant_foreground_median_depth_m"])
    )
    camera_collision = (
        legacy_camera_collision
        or near_surface_saturation
        or near_enclosure
        or foreground_occlusion
    )
    return not (degenerate_rgb or camera_collision)


def _recognizable_entity_ids(bundle: Bundle, view_id: str) -> set[str]:
    """Return non-structural instances that pass one shared visual threshold."""

    view = bundle.view_by_id[view_id]
    return {
        entity_id
        for entity_id in view.visible_entity_ids
        if entity_id in bundle.entities
        and bundle.entities[entity_id].label not in STRUCTURAL_LABELS
        and _mask_quality_passes(bundle, entity_id, view_id)
    }


def _replay_relation_components(
    bundle: Bundle,
    spec: QuestionSpec,
    entities: list[Entity],
) -> tuple[float, float] | None:
    if spec.task_type == "egocentric_relation":
        return _relation_components_in_camera(
            bundle, entities[0], entities[1], spec.evidence_view_ids[0]
        )
    if spec.task_type in {
        "cross_view_relation",
        "counterfactual_verification",
        "elevation_relation_transfer",
    }:
        _, _, right, front = bundle.canonical_relation(entities[0], entities[1])
        return right, front
    if spec.task_type == "object_centric_perspective":
        return _object_components(entities[0], entities[1], entities[2])
    return None


def _recompute(bundle: Bundle, spec: QuestionSpec) -> Any:
    task = spec.task_type
    entities = [bundle.entity(entity_id) for entity_id in spec.evidence_entity_ids]
    model_views = spec.model_view_ids or tuple(view.view_id for view in bundle.views)

    if task == "grounding_presence":
        label = entities[0].label
        return any(_category_visible(bundle, label, view_id) for view_id in spec.evidence_view_ids)
    if task == "last_seen_memory":
        visible_steps = [
            bundle.view_by_id[view_id].step
            for view_id in model_views
            if entities[0].entity_id in bundle.view_by_id[view_id].visible_entity_ids
        ]
        if not visible_steps:
            raise ValueError("tracked entity is never model-visible")
        return max(visible_steps) + 1
    if task == "metric_distance":
        return round(math.dist(entities[0].center_world_m, entities[1].center_world_m), 1)
    if task == "egocentric_relation":
        return _relation_in_camera(bundle, entities[0], entities[1], spec.evidence_view_ids[0])
    if task == "cross_view_relation":
        return bundle.canonical_relation(entities[0], entities[1])[0]
    if task == "cross_view_unknown":
        observed = {
            entity_id
            for view_id in model_views
            for entity_id in bundle.view_by_id[view_id].visible_entity_ids
        }
        presence = [entity.entity_id in observed for entity in entities]
        if sum(presence) != 1:
            raise ValueError(
                f"cross-view unknown requires exactly one observed referent, got {presence}"
            )
        missing = entities[presence.index(False)]
        if any(
            _confusable_category_visible(bundle, missing.label, view_id)
            for view_id in model_views
        ):
            raise ValueError("cross-view unknown contains a confusable missing category")
        return None
    if task == "evidence_presence_unknown":
        label = entities[0].label
        if any(
            _confusable_category_visible(bundle, label, view_id)
            for view_id in model_views
        ):
            raise ValueError("unknown-family target category is visible")
        return None
    if task == "evidence_presence_reveal":
        label = entities[0].label
        if not any(_category_visible(bundle, label, view_id) for view_id in model_views):
            raise ValueError("revealed-family target category is absent")
        return {"status": "present", "category": label}
    if task == "counterfactual_verification":
        relation = bundle.canonical_relation(entities[0], entities[1])[0]
        oracle = spec.certificate.get("oracle", {})
        claimed = oracle.get("claimed_relation", oracle.get("rejected_relation"))
        return {"claim_correct": claimed == relation, "relation": relation}
    if task == "object_centric_perspective":
        return _object_quadrant(entities[0], entities[1], entities[2])
    if task == "unknown_abstention":
        category_surface = _question_category_surface(spec.question_zh)
        target_labels = {
            entity.label
            for entity in bundle.entities.values()
            if entity_name(entity.label) == category_surface
        }
        if not target_labels:
            raise ValueError("unknown category surface has no scene ontology binding")
        if any(
            _confusable_category_visible(bundle, label, view_id)
            for label in target_labels
            for view_id in model_views
        ):
            raise ValueError("unknown category is visible in actual model input")
        return None
    if task == "rotation_change_detection":
        before, after = (bundle.view_by_id[view_id] for view_id in spec.evidence_view_ids)
        entered = sorted(
            _recognizable_entity_ids(bundle, after.view_id)
            - _recognizable_entity_ids(bundle, before.view_id)
        )
        if len(entered) != 1:
            raise ValueError(f"entered entity cardinality is {len(entered)}")
        entrant_label = bundle.entities[entered[0]].label
        if _confusable_category_visible(bundle, entrant_label, before.view_id):
            raise ValueError("rotation entrant is visually confusable with a prior object")
        return entrant_label
    if task == "orbit_identity":
        focus = entities[0]
        return all(
            focus.entity_id in bundle.view_by_id[view_id].visible_entity_ids
            for view_id in spec.evidence_view_ids
        )
    if task == "elevation_relation_transfer":
        return bundle.canonical_relation(entities[0], entities[1])[0]
    if task == "occlusion_unknown":
        label = entities[0].label
        if any(
            _confusable_category_visible(bundle, label, view_id)
            for view_id in model_views
        ):
            raise ValueError("unknown-family target category is visible")
        return None
    if task == "occlusion_reveal":
        label = entities[0].label
        if not any(_category_visible(bundle, label, view_id) for view_id in model_views):
            raise ValueError("revealed category is absent")
        return {"status": "present", "category": label}
    if task == "target_view_prediction":
        root = Path(str(spec.certificate["oracle_target_bundle"]))
        target_view_id = str(spec.certificate["oracle_target_view_id"])
        candidate_id = str(spec.certificate["candidate_entity_id"])
        episode = read_json(root / "spatial_episode.json")
        observation = next(
            item for item in episode["observations"] if item["view_id"] == target_view_id
        )
        return candidate_id in set(observation["visible_entity_ids"])
    raise ValueError(f"unsupported replay task: {task}")


def _category_visible(bundle: Bundle, label: str, view_id: str) -> bool:
    return any(
        entity_id in bundle.entities and bundle.entities[entity_id].label == label
        for entity_id in bundle.view_by_id[view_id].visible_entity_ids
    )


def _confusable_category_visible(bundle: Bundle, label: str, view_id: str) -> bool:
    policy = visual_quality_policy()
    if label in set(policy["visual_absence_unsafe_labels"]):
        return True
    groups = policy["visually_confusable_category_groups"]
    alternatives = next(
        (set(group) for group in groups if label in group),
        {label},
    )
    stats = bundle.raw_category_visual_stats(view_id, alternatives)
    return (
        int(stats["matching_pixel_count"])
        >= int(policy["minimum_raw_negative_category_pixels"])
        or int(stats["max_instance_bbox_min_side_px"])
        >= int(policy["minimum_raw_negative_category_bbox_side_px"])
    )


def _question_category_surface(question: str) -> str:
    marker = "存在"
    if marker not in question:
        raise ValueError("unknown question has no existence marker")
    surface = question.rsplit(marker, 1)[1].rstrip("？?。 ")
    if not surface:
        raise ValueError("unknown question has an empty category")
    return surface


def _yaw_rad(rotation_xyzw: list[float] | tuple[float, ...]) -> float:
    x, y, z, w = (float(value) for value in rotation_xyzw)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _dominant_relation(dx: float, dy: float) -> str:
    if abs(dx) >= abs(dy):
        return "right_of" if dx > 0 else "left_of"
    return "in_front_of" if dy > 0 else "behind"


def _relation_in_camera(bundle: Bundle, subject: Entity, reference: Entity, view_id: str) -> str:
    right, front = _relation_components_in_camera(bundle, subject, reference, view_id)
    return _dominant_relation(right, front)


def _relation_components_in_camera(
    bundle: Bundle, subject: Entity, reference: Entity, view_id: str
) -> tuple[float, float]:
    yaw = _yaw_rad(bundle.view_by_id[view_id].world_from_camera["rotation_xyzw"])
    world_dx = subject.center_world_m[0] - reference.center_world_m[0]
    world_dy = subject.center_world_m[1] - reference.center_world_m[1]
    right = world_dx * math.cos(yaw) + world_dy * math.sin(yaw)
    front = -world_dx * math.sin(yaw) + world_dy * math.cos(yaw)
    return right, front


def _object_quadrant(origin: Entity, facing: Entity, target: Entity) -> str:
    right, front = _object_components(origin, facing, target)
    return ("front_" if front >= 0 else "back_") + ("right" if right >= 0 else "left")


def _object_components(
    origin: Entity, facing: Entity, target: Entity
) -> tuple[float, float]:
    forward_x = facing.center_world_m[0] - origin.center_world_m[0]
    forward_y = facing.center_world_m[1] - origin.center_world_m[1]
    norm = math.hypot(forward_x, forward_y)
    if norm < 1e-6:
        raise ValueError("origin and facing points coincide")
    forward_x, forward_y = forward_x / norm, forward_y / norm
    right_x, right_y = forward_y, -forward_x
    target_x = target.center_world_m[0] - origin.center_world_m[0]
    target_y = target.center_world_m[1] - origin.center_world_m[1]
    right = target_x * right_x + target_y * right_y
    front = target_x * forward_x + target_y * forward_y
    return right, front
