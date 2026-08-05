"""Predicate library: named, pure checks over a SceneView.

Every predicate returns a :class:`Verdict` carrying both the tri-state result
and a *witness* — the concrete numbers the decision was based on. Witnesses
flow into trajectory plans and certificates so any decision can be replayed
and audited later.

Predicates are registered by name; script clauses reference them by that name
only. Adding a capability must never require editing this module unless a
genuinely new concept is needed — in that case the new predicate is added
here once and becomes available to every script.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .geometry import azimuth_deg, cumulative_turn_deg, sector_margin_deg, sector_of
from .sceneview import SceneView
from .standards import CompileStandard


@dataclass(frozen=True)
class Verdict:
    """Tri-state decision plus its evidence.

    ``holds`` is True/False for a definite decision and None when the check is
    ambiguous (e.g. a frame falls in the double-threshold gap). Ambiguity is
    never silently coerced: candidates containing ambiguous evidence frames
    are rejected upstream.
    """

    holds: bool | None
    witness: dict[str, Any]


PredicateFn = Callable[..., Verdict]
_REGISTRY: dict[str, PredicateFn] = {}


def predicate(name: str) -> Callable[[PredicateFn], PredicateFn]:
    def register(fn: PredicateFn) -> PredicateFn:
        if name in _REGISTRY:
            raise ValueError(f"duplicate predicate name: {name}")
        _REGISTRY[name] = fn
        return fn

    return register


def get_predicate(name: str) -> PredicateFn:
    try:
        return _REGISTRY[name]
    except KeyError as error:
        raise KeyError(f"unknown predicate: {name}; known: {sorted(_REGISTRY)}") from error


def registered_predicates() -> list[str]:
    return sorted(_REGISTRY)


def _tristates(
    view: SceneView, obj: str, frames: Sequence[int], std: CompileStandard
) -> list[tuple[int, bool | None, float]]:
    rows = []
    for t in frames:
        observation = view.visibility(obj, t)
        rows.append((t, observation.tristate(std), round(observation.value, 4)))
    return rows


@predicate("visible_in_range")
def visible_in_range(
    view: SceneView, std: CompileStandard, *, obj: str, frames: Sequence[int]
) -> Verdict:
    """Object is definitely visible in EVERY frame of the range."""
    rows = _tristates(view, obj, frames, std)
    if any(state is None for _, state, _ in rows):
        return Verdict(None, {"obj": obj, "frames": rows, "reason": "ambiguous_frame"})
    return Verdict(all(state is True for _, state, _ in rows), {"obj": obj, "frames": rows})


@predicate("visible_somewhere")
def visible_somewhere(
    view: SceneView, std: CompileStandard, *, obj: str, frames: Sequence[int]
) -> Verdict:
    """Object is definitely visible in AT LEAST one frame of the range."""
    rows = _tristates(view, obj, frames, std)
    if any(state is True for _, state, _ in rows):
        return Verdict(True, {"obj": obj, "frames": rows})
    if any(state is None for _, state, _ in rows):
        return Verdict(None, {"obj": obj, "frames": rows, "reason": "ambiguous_frame"})
    return Verdict(False, {"obj": obj, "frames": rows})


@predicate("invisible_in_range")
def invisible_in_range(
    view: SceneView, std: CompileStandard, *, obj: str, frames: Sequence[int]
) -> Verdict:
    """Object is definitely invisible in EVERY frame of the range."""
    rows = _tristates(view, obj, frames, std)
    if any(state is None for _, state, _ in rows):
        return Verdict(None, {"obj": obj, "frames": rows, "reason": "ambiguous_frame"})
    return Verdict(all(state is False for _, state, _ in rows), {"obj": obj, "frames": rows})


@predicate("disappeared_after")
def disappeared_after(view: SceneView, std: CompileStandard, *, obj: str, t: int) -> Verdict:
    """Object stays invisible for ``absence_min_frames`` frames after ``t``."""
    span = range(t + 1, min(t + 1 + std.absence_min_frames, view.frame_count))
    if len(span) < std.absence_min_frames:
        return Verdict(False, {"obj": obj, "t": t, "reason": "trajectory_too_short"})
    inner = invisible_in_range(view, std, obj=obj, frames=span)
    return Verdict(inner.holds, {"obj": obj, "t": t, **inner.witness})


@predicate("cum_turn_ge")
def cum_turn_ge(
    view: SceneView, std: CompileStandard, *, frames: Sequence[int], deg: float
) -> Verdict:
    """Cumulative absolute heading change across the range reaches ``deg``."""
    yaws = [view.camera_pose(t).yaw_deg for t in frames]
    total = cumulative_turn_deg(yaws)
    return Verdict(total >= deg, {"cum_turn_deg": round(total, 1), "required_deg": deg})


@predicate("sector_margin_ge")
def sector_margin_ge(
    view: SceneView,
    std: CompileStandard,
    *,
    obj: str,
    frame: int,
    tighten: bool = False,
) -> Verdict:
    """The object's camera-frame azimuth clears the sector boundary margin.

    With ``tighten=True`` (search phase) the required margin is multiplied by
    ``search_tighten_factor`` so most candidates also pass the authoritative
    post-render re-check.
    """
    pose = view.camera_pose(frame)
    azimuth = azimuth_deg(pose.xy, pose.yaw_deg, view.object(obj).xy)
    margin = sector_margin_deg(azimuth)
    required = std.sector_margin_deg * (std.search_tighten_factor if tighten else 1.0)
    return Verdict(
        margin >= required,
        {
            "obj": obj,
            "frame": frame,
            "azimuth_deg": round(azimuth, 1),
            "sector": sector_of(azimuth),
            "margin_deg": round(margin, 1),
            "required_deg": round(required, 1),
        },
    )
