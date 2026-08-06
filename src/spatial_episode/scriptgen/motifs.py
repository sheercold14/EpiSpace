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


MAX_TURN_PER_FRAME_DEG = 32.0  # below std.v2's 40-degree trackability cap


def _walk_polyline(
    waypoints: list[tuple[float, float]], headings: list[float], frame_count: int
) -> tuple[Pose2D, ...]:
    """Distribute frames evenly over polyline segments with rate-limited yaw.

    Yaw chases the current segment's desired heading but never changes by
    more than MAX_TURN_PER_FRAME_DEG between frames: consecutive frames keep
    visual overlap, so the camera's own motion stays trackable from images
    (the std.v2 ego-motion contract). Partial-visibility frames during the
    turn are expected; the script's transition zone accommodates them.
    """
    segments = len(waypoints) - 1
    poses: list[Pose2D] = []
    yaw = headings[0]
    for i in range(frame_count):
        progress = i / max(frame_count - 1, 1) * segments
        seg = min(int(progress), segments - 1)
        t = progress - seg
        desired = headings[min(seg + 1, segments - 1)] if t > 0.05 or seg > 0 else headings[0]
        delta = wrap_deg(desired - yaw)
        yaw = wrap_deg(yaw + max(-MAX_TURN_PER_FRAME_DEG, min(MAX_TURN_PER_FRAME_DEG, delta)))
        x = _lerp(waypoints[seg][0], waypoints[seg + 1][0], t)
        y = _lerp(waypoints[seg][1], waypoints[seg + 1][1], t)
        poses.append(Pose2D(x, y, yaw))
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
    # Leave via a mid waypoint towards a RANDOM far-ish corner: varying the
    # exit direction varies the final target bearing, so gold answers spread
    # over left/right/back instead of collapsing onto "back".
    corners = [
        (min_x + 0.5, min_y + 0.5),
        (min_x + 0.5, max_y - 0.5),
        (max_x - 0.5, min_y + 0.5),
        (max_x - 0.5, max_y - 0.5),
    ]
    corners.sort(key=lambda c: (c[0] - start[0]) ** 2 + (c[1] - start[1]) ** 2, reverse=True)
    exit_corner = corners[rng.randrange(3)]  # any of the three farther corners
    mid = (
        _lerp(start[0], exit_corner[0], jitter(0.35, 0.65)) + jitter(-1.0, 1.0),
        _lerp(start[1], exit_corner[1], jitter(0.35, 0.65)) + jitter(-1.0, 1.0),
    )
    mid = (min(max(mid[0], min_x + 0.5), max_x - 0.5), min(max(mid[1], min_y + 0.5), max_y - 0.5))

    waypoints = [start, mid, exit_corner]
    headings = [
        bearing_deg(start, target.xy),  # initial gaze at the target
        bearing_deg(start, mid),
        bearing_deg(mid, exit_corner),
    ]
    look_frames = max(2, frame_count // 4)
    walk = list(_walk_polyline(waypoints, [headings[0], headings[2]], frame_count - look_frames))

    # Arrive, then look around: rotate in place (rate-limited) towards a
    # random final gaze so the target's final bearing spreads over
    # left/right/back instead of collapsing onto "behind the walker".
    end = walk[-1]
    offset = rng.choice([90.0, -90.0, 180.0]) + rng.uniform(-20.0, 20.0)
    final_yaw = wrap_deg(bearing_deg(end.xy, target.xy) + offset)
    yaw = end.yaw_deg
    for _ in range(look_frames):
        delta = wrap_deg(final_yaw - yaw)
        step = max(-MAX_TURN_PER_FRAME_DEG, min(MAX_TURN_PER_FRAME_DEG, delta))
        yaw = wrap_deg(yaw + step)
        walk.append(Pose2D(end.x, end.y, yaw))
    return tuple(walk)
