"""Registered declarative answer algorithms.

Answer modes mirror the predicate registry: specs name a pure function and
provide declarative arguments, while the compiler resolves those arguments
and records the returned witness in the authoritative certificate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .geometry import azimuth_deg, sector_margin_deg, sector_of
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
        raise KeyError(
            f"unknown answer mode: {name}; known: {sorted(_REGISTRY)}"
        ) from error


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
