"""Registered declarative answer algorithms.

Answer modes mirror the predicate registry: specs name a pure function and
provide declarative arguments, while the compiler resolves those arguments
and records the returned witness in the authoritative certificate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .coverage import measure_coverage
from .geometry import (
    azimuth_deg,
    bearing_deg,
    distance_m,
    net_turn_deg,
    sector_margin_deg,
    sector_of,
    wrap_deg,
)
from .sceneview import GeometrySceneView, Pose2D, SceneView
from .standards import CompileStandard

AnswerWitnessValue = float | int | str


@dataclass(frozen=True)
class AnswerResult:
    """One closed-option label and the values used to derive it."""

    label: str
    witness: dict[str, AnswerWitnessValue]


AnswerFn = Callable[..., AnswerResult]
_REGISTRY: dict[str, AnswerFn] = {}


def answer_mode(name: str) -> Callable[[AnswerFn], AnswerFn]:
    """Register an answer mode under the stable name used by specs."""

    def register(fn: AnswerFn) -> AnswerFn:
        if name in _REGISTRY:
            raise ValueError(f"duplicate answer mode name: {name}")
        _REGISTRY[name] = fn
        return fn

    return register


def get_answer_mode(name: str) -> AnswerFn:
    try:
        return _REGISTRY[name]
    except KeyError as error:
        raise KeyError(f"unknown answer mode: {name}; known: {sorted(_REGISTRY)}") from error


def registered_answer_modes() -> list[str]:
    return sorted(_REGISTRY)


@answer_mode("target_sector")
def target_sector(
    view: SceneView,
    std: CompileStandard,
    *,
    obj: str,
    frame: int,
) -> AnswerResult:
    """Four-sector direction of an object at the declared camera frame."""
    del std  # signature is uniform across modes; this mode has no own threshold.
    pose = view.camera_pose(frame)
    azimuth = azimuth_deg(pose.xy, pose.yaw_deg, view.object(obj).xy)
    label = sector_of(azimuth)
    return AnswerResult(
        label=label,
        witness={
            "azimuth_deg": round(azimuth, 1),
            "sector": label,
            "margin_deg": round(sector_margin_deg(azimuth), 1),
            "question_frame": frame,
        },
    )


@answer_mode("occluder_category_at")
def occluder_category_at(
    view: SceneView,
    std: CompileStandard,
    *,
    obj: str,
    frame: int,
) -> AnswerResult:
    """Semantic category of the authority-attributed blocker at an event."""

    del std
    from .occlusion import DISAPPEARANCE_POLICY_VERSION, occluder_category_allowed

    observation = view.occlusion(obj, frame)
    category = observation.witness.get("occluder_category")
    if observation.kind == "geometry_ray_3d" and observation.occluder_ids:
        try:
            category = view.object(observation.occluder_ids[0]).category
        except KeyError:
            category = observation.occluder_ids[0]
    if observation.status != "occluded" or (
        observation.kind != "geometry_ray_3d"
        and not occluder_category_allowed(category if isinstance(category, str) else None)
    ):
        raise ValueError("occluder category answer requires an eligible occlusion clause")
    entity_id = observation.witness.get("occluder_entity_id") or (
        observation.occluder_ids[0] if observation.occluder_ids else ""
    )
    return AnswerResult(
        label=str(category),
        witness={
            "event_frame": frame,
            "occluder_category": str(category),
            "occluder_entity_id": str(entity_id),
            "event_policy_version": DISAPPEARANCE_POLICY_VERSION,
        },
    )


@answer_mode("disappearance_cause_at")
def disappearance_cause_at(
    view: SceneView,
    std: CompileStandard,
    *,
    obj: str,
    frame: int,
) -> AnswerResult:
    """Whether the decisive disappearance was occlusion or leaving the view."""

    del std
    observation = view.occlusion(obj, frame)
    if observation.status == "occluded":
        label = "occluded"
    elif observation.status == "clear" and observation.witness.get("reason") in {
        "target_behind_camera",
        "target_projection_outside_image",
    }:
        label = "out_of_view"
    else:
        raise ValueError("disappearance cause answer requires a decisive event clause")
    return AnswerResult(
        label=label,
        witness={
            "event_frame": frame,
            "cause": label,
            "authority_reason": str(observation.witness.get("reason") or "occluder_attributed"),
        },
    )


@answer_mode("view_side")
def view_side(
    view: SceneView,
    std: CompileStandard,
    *,
    obj: str,
    frame: int,
) -> AnswerResult:
    """Left or right image half at one declared sighting frame."""
    del std
    pose = view.camera_pose(frame)
    azimuth = azimuth_deg(pose.xy, pose.yaw_deg, view.object(obj).xy)
    label = "left_half" if azimuth > 0.0 else "right_half"
    return AnswerResult(
        label=label,
        witness={
            "azimuth_deg": round(azimuth, 1),
            "margin_deg": round(abs(azimuth), 1),
            "frame": frame,
        },
    )


@answer_mode("net_turn")
def net_turn(
    view: SceneView,
    std: CompileStandard,
    *,
    frames: list[int],
) -> AnswerResult:
    """Direction of signed, wrapped heading change accumulated over frames."""
    del std
    turn = net_turn_deg([view.camera_pose(t).yaw_deg for t in frames])
    return AnswerResult(
        label="left" if turn > 0.0 else "right",
        witness={"net_turn_deg": round(turn, 1)},
    )


@answer_mode("net_turn_magnitude")
def net_turn_magnitude(
    view: SceneView,
    std: CompileStandard,
    *,
    frames: list[int],
) -> AnswerResult:
    """Whether absolute signed net turn exceeds the declared magnitude tier."""
    turn = net_turn_deg([view.camera_pose(t).yaw_deg for t in frames])
    magnitude = abs(turn)
    return AnswerResult(
        label="over_90" if magnitude > std.net_turn_magnitude_deg else "at_most_90",
        witness={
            "net_turn_deg": round(turn, 1),
            "magnitude_deg": round(magnitude, 1),
            "threshold_deg": std.net_turn_magnitude_deg,
        },
    )


@answer_mode("start_sector")
def start_sector(
    view: SceneView,
    std: CompileStandard,
    *,
    frame: int,
) -> AnswerResult:
    """Four-sector direction of the frame-0 position from a later pose."""
    del std
    start = view.camera_pose(0)
    pose = view.camera_pose(frame)
    azimuth = azimuth_deg(pose.xy, pose.yaw_deg, start.xy)
    label = sector_of(azimuth)
    return AnswerResult(
        label=label,
        witness={
            "azimuth_deg": round(azimuth, 1),
            "sector": label,
            "margin_deg": round(sector_margin_deg(azimuth), 1),
            "start_distance_m": round(distance_m(start.xy, pose.xy), 2),
            "question_frame": frame,
        },
    )


def _imagined_pose(
    view: SceneView,
    std: CompileStandard,
    viewpoint: str,
    facing: str,
    yaw_offset_deg: float,
) -> Pose2D:
    from .motifs import imagined_station

    origin = imagined_station(view.layout, viewpoint, facing, std.camera_height_m)
    if origin is None:
        raise ValueError(
            f"no imagined station for viewpoint={viewpoint} facing={facing}; "
            "imagined_pose_valid should have rejected this binding"
        )
    yaw = wrap_deg(bearing_deg(origin, view.object(facing).xy) + yaw_offset_deg)
    return Pose2D(origin[0], origin[1], yaw)


@answer_mode("imagined_sector")
def imagined_sector(
    view: SceneView,
    std: CompileStandard,
    *,
    viewpoint: str,
    facing: str,
    obj: str,
    yaw_offset_deg: float = 0.0,
) -> AnswerResult:
    """Target sector from a constructed position-and-facing reference frame."""
    pose = _imagined_pose(view, std, viewpoint, facing, yaw_offset_deg)
    azimuth = azimuth_deg(pose.xy, pose.yaw_deg, view.object(obj).xy)
    label = sector_of(azimuth)
    return AnswerResult(
        label=label,
        witness={
            "imagined_x": round(pose.x, 3),
            "imagined_y": round(pose.y, 3),
            "imagined_yaw_deg": round(pose.yaw_deg, 1),
            "yaw_offset_deg": round(yaw_offset_deg, 1),
            "azimuth_deg": round(azimuth, 1),
            "margin_deg": round(sector_margin_deg(azimuth), 1),
        },
    )


@answer_mode("imagined_visibility")
def imagined_visibility(
    view: SceneView,
    std: CompileStandard,
    *,
    viewpoint: str,
    facing: str,
    obj: str,
    yaw_offset_deg: float = 0.0,
) -> AnswerResult:
    """Geometric visibility from a constructed pose, independent of sequence frames."""
    pose = _imagined_pose(view, std, viewpoint, facing, yaw_offset_deg)
    probe = GeometrySceneView(layout=view.layout, poses=(pose,), std=std)
    observation = probe.visibility(obj, 0)
    state = observation.tristate(std)
    if state is None:
        raise ValueError("imagined visibility is ambiguous; qualification clause missing")
    return AnswerResult(
        label="visible" if state else "not_visible",
        witness={
            "imagined_x": round(pose.x, 3),
            "imagined_y": round(pose.y, 3),
            "imagined_yaw_deg": round(pose.yaw_deg, 1),
            "yaw_offset_deg": round(yaw_offset_deg, 1),
            "visibility_value": round(observation.value, 4),
            "unoccluded_ratio": round(observation.unoccluded_ratio, 3),
        },
    )


@answer_mode("pair_relation")
def pair_relation(
    view: SceneView,
    std: CompileStandard,
    *,
    obj: str,
    reference: str,
) -> AnswerResult:
    """Object sector in the reference object's intrinsic OBB orientation."""
    del std
    source = view.object(obj)
    anchor = view.object(reference)
    azimuth = azimuth_deg(anchor.xy, anchor.yaw_deg, source.xy)
    label = sector_of(azimuth)
    return AnswerResult(
        label=label,
        witness={
            "reference_yaw_deg": round(anchor.yaw_deg, 1),
            "azimuth_deg": round(azimuth, 1),
            "margin_deg": round(sector_margin_deg(azimuth), 1),
        },
    )


@answer_mode("closer_of")
def closer_of(
    view: SceneView,
    std: CompileStandard,
    *,
    first: str,
    second: str,
    anchor: str,
) -> AnswerResult:
    """Which declared object is closer to an anchor in world geometry."""
    del std
    anchor_xy = view.object(anchor).xy
    first_distance = distance_m(view.object(first).xy, anchor_xy)
    second_distance = distance_m(view.object(second).xy, anchor_xy)
    label = "first" if first_distance < second_distance else "second"
    smaller, larger = sorted((first_distance, second_distance))
    return AnswerResult(
        label=label,
        witness={
            "first_distance_m": round(first_distance, 3),
            "second_distance_m": round(second_distance, 3),
            "distance_ratio": round(larger / max(smaller, 1e-9), 3),
        },
    )


@answer_mode("existence_sufficiency")
def existence_sufficiency(
    view: SceneView,
    std: CompileStandard,
    *,
    category: str,
    frames: list[int],
) -> AnswerResult:
    """Present, absent, or unknowable from sightings plus free-space coverage."""
    matching = [obj for obj in view.objects() if obj.category == category]
    visible_instances = 0
    first_visible_frame = -1
    for obj in matching:
        visible_frames = [
            frame
            for frame in frames
            if view.visibility(obj.name, frame).tristate(std) is True
        ]
        if visible_frames:
            visible_instances += 1
            if first_visible_frame < 0:
                first_visible_frame = min(visible_frames)
            else:
                first_visible_frame = min(first_visible_frame, *visible_frames)

    coverage = measure_coverage(view, std, frames=frames)
    tier_names = ("low", "medium", "high")
    coverage_tier = "below_low"
    for threshold, tier_name in zip(std.coverage_ratio_levels, tier_names, strict=True):
        if coverage.coverage_ratio >= threshold:
            coverage_tier = tier_name
    sufficient = (
        coverage.coverage_ratio >= std.coverage_ratio_levels[-1]
        and coverage.max_hidden_diameter_m < std.coverage_hidden_object_size_m
    )
    if visible_instances:
        label = "present"
    elif not matching and sufficient:
        label = "absent"
    else:
        label = "无法判断"
    return AnswerResult(
        label=label,
        witness={
            "category": category,
            "scene_instance_count": len(matching),
            "visible_instance_count": visible_instances,
            "first_visible_frame": first_visible_frame,
            "coverage_ratio": round(coverage.coverage_ratio, 4),
            "coverage_tier": coverage_tier,
            "covered_cell_count": coverage.covered_cell_count,
            "total_free_cell_count": coverage.total_free_cell_count,
            "max_hidden_diameter_m": round(coverage.max_hidden_diameter_m, 3),
            "required_coverage_ratio": std.coverage_ratio_levels[-1],
            "hidden_object_size_m": std.coverage_hidden_object_size_m,
        },
    )
