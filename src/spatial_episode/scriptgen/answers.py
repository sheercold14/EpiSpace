"""Registered declarative answer algorithms.

Answer modes mirror the predicate registry: specs name a pure function and
provide declarative arguments, while the compiler resolves those arguments
and records the returned witness in the authoritative certificate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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


def _imagined_pose(view: SceneView, viewpoint: str, facing: str, yaw_offset_deg: float) -> Pose2D:
    origin = view.object(viewpoint).xy
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
    del std
    pose = _imagined_pose(view, viewpoint, facing, yaw_offset_deg)
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
    pose = _imagined_pose(view, viewpoint, facing, yaw_offset_deg)
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
