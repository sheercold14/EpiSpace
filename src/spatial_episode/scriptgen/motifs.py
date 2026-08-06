"""Candidate trajectory motifs.

A motif is a heuristic that proposes camera pose sequences likely to satisfy a
family of scripts. Motifs generate candidates ONLY — every requirement is
judged by the checker, never inside the motif. This keeps constraint logic in
exactly one place (the predicate library).

Existing acquisition samplers (T1..T10 in the OmniGibson backend) are natural
future motifs: stripped of their inline constraint checks, they plug in behind
the same ``Motif`` protocol.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

from .geometry import bearing_deg, wrap_deg
from .sceneview import Pose2D, SceneLayout

Motif = Callable[[SceneLayout, dict[str, str], int, random.Random], tuple[Pose2D, ...]]

_MOTIFS: dict[str, Motif] = {}


def motif(name: str) -> Callable[[Motif], Motif]:
    def register(fn: Motif) -> Motif:
        if name in _MOTIFS:
            raise ValueError(f"duplicate motif name: {name}")
        _MOTIFS[name] = fn
        return fn

    return register


def get_motif(name: str) -> Motif:
    try:
        return _MOTIFS[name]
    except KeyError as error:
        raise KeyError(f"unknown motif: {name}; known: {sorted(_MOTIFS)}") from error


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _walk_polyline(
    waypoints: list[tuple[float, float]], headings: list[float], frame_count: int
) -> tuple[Pose2D, ...]:
    """Distribute frames evenly over polyline segments, yaw following headings.

    Heading changes are concentrated in the first quarter of each segment
    (people turn on the spot, then walk). Decisive turns also keep targets
    from lingering at the field-of-view edge, where extent-aware visibility
    is ambiguous and candidates would be rejected.
    """
    segments = len(waypoints) - 1
    poses: list[Pose2D] = []
    for i in range(frame_count):
        progress = i / max(frame_count - 1, 1) * segments
        seg = min(int(progress), segments - 1)
        t = progress - seg
        turn = min(t / 0.25, 1.0)
        yaw = headings[seg] + wrap_deg(headings[min(seg + 1, segments - 1)] - headings[seg]) * turn
        x = _lerp(waypoints[seg][0], waypoints[seg + 1][0], t)
        y = _lerp(waypoints[seg][1], waypoints[seg + 1][1], t)
        poses.append(Pose2D(x, y, wrap_deg(yaw)))
    return tuple(poses)


@motif("walk_and_turn")
def walk_and_turn(
    layout: SceneLayout, binding: dict[str, str], frame_count: int, rng: random.Random
) -> tuple[Pose2D, ...]:
    """Start facing the target, then walk away through one or two turns.

    The walk begins at a point with clear line of sight preference (near the
    target, facing it), then leaves towards a far corner of the walkable area
    so that later frames tend to lose sight of the target and accumulate
    heading change. Whether that actually happens is for the checker to judge.
    """
    target = layout.object(binding["target"])
    (min_x, min_y), (max_x, max_y) = layout.walkable_min, layout.walkable_max

    def jitter(low: float, high: float) -> float:
        return rng.uniform(low, high)

    # Start: a spot 2-4 m from the target, roughly facing it.
    angle = math.radians(jitter(0.0, 360.0))
    radius = jitter(2.0, 4.0)
    start = (
        min(max(target.xy[0] + radius * math.cos(angle), min_x + 0.5), max_x - 0.5),
        min(max(target.xy[1] + radius * math.sin(angle), min_y + 0.5), max_y - 0.5),
    )
    # Leave via a mid waypoint towards the corner farthest from the target.
    corners = [
        (min_x + 0.5, min_y + 0.5),
        (min_x + 0.5, max_y - 0.5),
        (max_x - 0.5, min_y + 0.5),
        (max_x - 0.5, max_y - 0.5),
    ]
    far_corner = max(corners, key=lambda c: (c[0] - target.xy[0]) ** 2 + (c[1] - target.xy[1]) ** 2)
    mid = (
        _lerp(start[0], far_corner[0], jitter(0.35, 0.65)) + jitter(-1.0, 1.0),
        _lerp(start[1], far_corner[1], jitter(0.35, 0.65)) + jitter(-1.0, 1.0),
    )
    mid = (min(max(mid[0], min_x + 0.5), max_x - 0.5), min(max(mid[1], min_y + 0.5), max_y - 0.5))

    waypoints = [start, mid, far_corner]
    headings = [
        bearing_deg(start, target.xy),  # first look at the target
        bearing_deg(start, mid),
        bearing_deg(mid, far_corner),
    ]
    # Polyline has 2 segments; drop the extra heading entry used for the gaze.
    return _walk_polyline(waypoints, [headings[0], headings[2]], frame_count)
