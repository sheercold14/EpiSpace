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
    distance_m,
    net_turn_deg,
    sector_margin_deg,
    sector_of,
)
from .sceneview import SceneView
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
