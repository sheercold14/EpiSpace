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

from .geometry import (
    azimuth_deg,
    cumulative_turn_deg,
    point_in_rotated_rect,
    rotated_rect_penetration_depth,
    sector_margin_deg,
    sector_of,
    segment_intersects_rotated_rect,
)
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


def _body_band_obstacles(view: SceneView, std: CompileStandard):
    return (
        obstacle
        for obstacle in view.layout.obstacles
        if obstacle.z_low <= std.clearance_z_high_m
        and obstacle.z_high >= std.clearance_z_low_m
    )


@predicate("poses_clear")
def poses_clear(view: SceneView, std: CompileStandard, *, frames: Sequence[int]) -> Verdict:
    """Every sampled body centre clears all body-height obstacle footprints."""
    worst: dict[str, Any] | None = None
    collisions = 0
    for t in frames:
        position = view.camera_pose(t).xy
        for obstacle in _body_band_obstacles(view, std):
            expanded = tuple(value + std.body_radius_m for value in obstacle.half_extents_xy)
            if not point_in_rotated_rect(
                position, obstacle.center_xy, expanded, obstacle.yaw_deg
            ):
                continue
            collisions += 1
            depth = rotated_rect_penetration_depth(
                position, obstacle.center_xy, expanded, obstacle.yaw_deg
            )
            candidate = {
                "frame": t,
                "obstacle": obstacle.label,
                "penetration_depth_m": round(depth, 4),
            }
            if worst is None or depth > float(worst["penetration_depth_m"]):
                worst = candidate
    return Verdict(
        worst is None,
        {
            "body_radius_m": std.body_radius_m,
            "collision_count": collisions,
            "worst_collision": worst,
        },
    )


@predicate("path_clear")
def path_clear(view: SceneView, std: CompileStandard, *, frames: Sequence[int]) -> Verdict:
    """Every consecutive pose segment clears body-height obstacle footprints."""
    from itertools import pairwise

    collisions: list[dict[str, Any]] = []
    for a, b in pairwise(frames):
        start, end = view.camera_pose(a).xy, view.camera_pose(b).xy
        for obstacle in _body_band_obstacles(view, std):
            expanded = tuple(value + std.body_radius_m for value in obstacle.half_extents_xy)
            if segment_intersects_rotated_rect(
                start, end, obstacle.center_xy, expanded, obstacle.yaw_deg
            ):
                collisions.append({"frames": [a, b], "obstacle": obstacle.label})
    return Verdict(
        not collisions,
        {
            "body_radius_m": std.body_radius_m,
            "collision_count": len(collisions),
            "worst_collision": collisions[0] if collisions else None,
        },
    )


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


@predicate("cum_turn_between")
def cum_turn_between(
    view: SceneView,
    std: CompileStandard,
    *,
    frames: Sequence[int],
    deg_min: float,
    deg_max: float,
) -> Verdict:
    """Cumulative heading change lies in [deg_min, deg_max].

    The lower bound makes the remembered viewpoint stale; the upper bound
    keeps the transformation inside a range a tracker could accumulate — far
    beyond it the answer collapses to a guess rather than a mental rotation.
    """
    yaws = [view.camera_pose(t).yaw_deg for t in frames]
    total = cumulative_turn_deg(yaws)
    return Verdict(
        deg_min <= total <= deg_max,
        {"cum_turn_deg": round(total, 1), "deg_min": deg_min, "deg_max": deg_max},
    )


@predicate("step_motion_bounded")
def step_motion_bounded(view: SceneView, std: CompileStandard, *, frames: Sequence[int]) -> Verdict:
    """Every consecutive frame pair keeps yaw and translation within bounds.

    This is the ego-motion trackability contract: bounded per-frame rotation
    preserves visual overlap between frames, so the model CAN in principle
    estimate its own motion from the image stream. Snap turns make the
    self-motion question unanswerable and must reject the candidate.
    """
    from itertools import pairwise

    from .geometry import distance_m, wrap_deg

    worst_turn, worst_step = 0.0, 0.0
    for a, b in pairwise(frames):
        pa, pb = view.camera_pose(a), view.camera_pose(b)
        worst_turn = max(worst_turn, abs(wrap_deg(pb.yaw_deg - pa.yaw_deg)))
        worst_step = max(worst_step, distance_m(pa.xy, pb.xy))
    holds = worst_turn <= std.max_step_turn_deg and worst_step <= std.max_step_translation_m
    return Verdict(
        holds,
        {
            "worst_step_turn_deg": round(worst_turn, 1),
            "max_step_turn_deg": std.max_step_turn_deg,
            "worst_step_translation_m": round(worst_step, 2),
            "max_step_translation_m": std.max_step_translation_m,
        },
    )


@predicate("frame_gap_ge")
def frame_gap_ge(
    view: SceneView, std: CompileStandard, *, later: int, earlier: int, gap: int
) -> Verdict:
    """At least ``gap`` frames separate two frame variables."""
    actual = later - earlier
    return Verdict(actual >= gap, {"later": later, "earlier": earlier, "gap": actual})


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
