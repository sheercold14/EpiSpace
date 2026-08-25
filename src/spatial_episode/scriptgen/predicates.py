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
from functools import lru_cache
from itertools import pairwise
from typing import Any

from .coverage import measure_coverage
from .geometry import (
    azimuth_deg,
    bearing_deg,
    cumulative_turn_deg,
    distance_m,
    net_turn_deg,
    sector_margin_deg,
    sector_of,
    segment_intersects_rect,
    wrap_deg,
)
from .sceneview import GeometrySceneView, Pose2D, ReindexedSceneView, SceneLayout, SceneView
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


@lru_cache(maxsize=64)
def _expanded_body_obstacles(
    layout: SceneLayout, body_radius_m: float, z_low_m: float, z_high_m: float
) -> tuple[tuple[str, float, float, float, float, float, float, float, float], ...]:
    """Precompute expanded local frames shared by both clearance predicates."""
    import math

    rows = []
    for obstacle in layout.obstacles:
        if obstacle.z_low > z_high_m or obstacle.z_high < z_low_m:
            continue
        angle = math.radians(obstacle.yaw_deg)
        cos_yaw, sin_yaw = math.cos(angle), math.sin(angle)
        hx = obstacle.half_extents_xy[0] + body_radius_m
        hy = obstacle.half_extents_xy[1] + body_radius_m
        rows.append(
            (
                obstacle.label,
                obstacle.center_xy[0],
                obstacle.center_xy[1],
                hx,
                hy,
                cos_yaw,
                sin_yaw,
                abs(cos_yaw) * hx + abs(sin_yaw) * hy,
                abs(sin_yaw) * hx + abs(cos_yaw) * hy,
            )
        )
    return tuple(rows)


@lru_cache(maxsize=256)
def _clearance_audit(
    layout: SceneLayout,
    samples: tuple[tuple[int, float, float], ...],
    body_radius_m: float,
    z_low_m: float,
    z_high_m: float,
) -> tuple[int, int, int, dict[str, Any] | None, int, dict[str, Any] | None]:
    obstacles = _expanded_body_obstacles(layout, body_radius_m, z_low_m, z_high_m)
    pose_count = 0
    pose_worst: dict[str, Any] | None = None
    pose_worst_depth = -1.0
    for frame, x, y in samples:
        for label, cx, cy, hx, hy, cos_yaw, sin_yaw, extent_x, extent_y in obstacles:
            dx, dy = x - cx, y - cy
            if abs(dx) > extent_x or abs(dy) > extent_y:
                continue
            local_x = cos_yaw * dx + sin_yaw * dy
            local_y = -sin_yaw * dx + cos_yaw * dy
            if abs(local_x) > hx or abs(local_y) > hy:
                continue
            pose_count += 1
            depth = max(0.0, min(hx - abs(local_x), hy - abs(local_y)))
            if depth > pose_worst_depth:
                pose_worst_depth = depth
                pose_worst = {
                    "frame": frame,
                    "obstacle": label,
                    "penetration_depth_m": round(depth, 4),
                }

    path_count = 0
    path_worst: dict[str, Any] | None = None
    from itertools import pairwise

    for (frame_a, x0, y0), (frame_b, x1, y1) in pairwise(samples):
        for label, cx, cy, hx, hy, cos_yaw, sin_yaw, extent_x, extent_y in obstacles:
            if (
                max(x0, x1) < cx - extent_x
                or min(x0, x1) > cx + extent_x
                or max(y0, y1) < cy - extent_y
                or min(y0, y1) > cy + extent_y
            ):
                continue
            dx0, dy0 = x0 - cx, y0 - cy
            dx1, dy1 = x1 - cx, y1 - cy
            local_start = (
                cos_yaw * dx0 + sin_yaw * dy0,
                -sin_yaw * dx0 + cos_yaw * dy0,
            )
            local_end = (
                cos_yaw * dx1 + sin_yaw * dy1,
                -sin_yaw * dx1 + cos_yaw * dy1,
            )
            if not segment_intersects_rect(local_start, local_end, (-hx, -hy), (hx, hy)):
                continue
            path_count += 1
            if path_worst is None:
                path_worst = {"frames": [frame_a, frame_b], "obstacle": label}
    return (
        len(obstacles),
        sum(row[0] == "walls" for row in obstacles),
        pose_count,
        pose_worst,
        path_count,
        path_worst,
    )


def _clearance_results(view: SceneView, std: CompileStandard, frames: Sequence[int]):
    samples = tuple((t, *view.camera_pose(t).xy) for t in frames)
    return _clearance_audit(
        view.layout,
        samples,
        std.body_radius_m,
        std.clearance_z_low_m,
        std.clearance_z_high_m,
    )


@predicate("poses_clear")
def poses_clear(view: SceneView, std: CompileStandard, *, frames: Sequence[int]) -> Verdict:
    """Every sampled body centre clears all body-height obstacle footprints."""
    obstacle_count, wall_count, collisions, worst, _, _ = _clearance_results(view, std, frames)
    return Verdict(
        worst is None,
        {
            "body_radius_m": std.body_radius_m,
            "checked_obstacle_count": obstacle_count,
            "checked_wall_count": wall_count,
            "collision_count": collisions,
            "worst_collision": worst,
        },
    )


@predicate("path_clear")
def path_clear(view: SceneView, std: CompileStandard, *, frames: Sequence[int]) -> Verdict:
    """Every consecutive pose segment clears body-height obstacle footprints."""
    obstacle_count, wall_count, _, _, collisions, worst = _clearance_results(view, std, frames)
    return Verdict(
        collisions == 0,
        {
            "body_radius_m": std.body_radius_m,
            "checked_obstacle_count": obstacle_count,
            "checked_wall_count": wall_count,
            "collision_count": collisions,
            "worst_collision": worst,
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


@predicate("consecutive_visible_frames_ge")
def consecutive_visible_frames_ge(
    view: SceneView,
    std: CompileStandard,
    *,
    obj: str,
    frames: Sequence[int],
    required: int,
) -> Verdict:
    """The target supplies a consecutive run of clear identity evidence."""

    if required < 1:
        raise ValueError("required must be positive")
    rows = _tristates(view, obj, frames, std)
    longest = 0
    current = 0
    for _, state, _ in rows:
        current = current + 1 if state is True else 0
        longest = max(longest, current)
    ambiguous = [frame for frame, state, _ in rows if state is None]
    holds: bool | None
    if longest >= required:
        holds = True
    elif ambiguous:
        holds = None
    else:
        holds = False
    return Verdict(
        holds,
        {
            "obj": obj,
            "frames": rows,
            "longest_consecutive_visible_frames": longest,
            "required_frames": required,
            "ambiguous_frames": ambiguous,
        },
    )


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


@predicate("view_side_margin_ge")
def view_side_margin_ge(view: SceneView, std: CompileStandard, *, obj: str, frame: int) -> Verdict:
    """An object's image-side azimuth clears the vertical midline."""
    pose = view.camera_pose(frame)
    azimuth = azimuth_deg(pose.xy, pose.yaw_deg, view.object(obj).xy)
    margin = abs(azimuth)
    return Verdict(
        margin >= std.view_side_margin_deg,
        {
            "obj": obj,
            "frame": frame,
            "azimuth_deg": round(azimuth, 1),
            "margin_deg": round(margin, 1),
            "required_deg": std.view_side_margin_deg,
        },
    )


@predicate("net_turn_margin_ge")
def net_turn_margin_ge(view: SceneView, std: CompileStandard, *, frames: Sequence[int]) -> Verdict:
    """Signed net turn clears the zero-degree left/right boundary."""
    turn = net_turn_deg([view.camera_pose(t).yaw_deg for t in frames])
    margin = abs(turn)
    return Verdict(
        margin >= std.net_turn_margin_deg,
        {
            "net_turn_deg": round(turn, 1),
            "margin_deg": round(margin, 1),
            "required_deg": std.net_turn_margin_deg,
        },
    )


@predicate("net_turn_magnitude_margin_ge")
def net_turn_magnitude_margin_ge(
    view: SceneView, std: CompileStandard, *, frames: Sequence[int]
) -> Verdict:
    """Net-turn magnitude stays clear of the declared 90-degree tier boundary."""
    turn = net_turn_deg([view.camera_pose(t).yaw_deg for t in frames])
    magnitude = abs(turn)
    margin = abs(magnitude - std.net_turn_magnitude_deg)
    return Verdict(
        margin >= std.net_turn_margin_deg,
        {
            "net_turn_deg": round(turn, 1),
            "magnitude_deg": round(magnitude, 1),
            "threshold_deg": std.net_turn_magnitude_deg,
            "margin_deg": round(margin, 1),
            "required_margin_deg": std.net_turn_margin_deg,
        },
    )


@predicate("displacement_below")
def displacement_below(view: SceneView, std: CompileStandard, *, frames: Sequence[int]) -> Verdict:
    """Every pose remains within the pure-rotation radius of the first pose."""
    frame_list = list(frames)
    origin = view.camera_pose(frame_list[0]).xy
    displacement = max(distance_m(origin, view.camera_pose(t).xy) for t in frame_list)
    return Verdict(
        displacement <= std.pure_rotation_max_displacement_m,
        {
            "max_displacement_m": round(displacement, 3),
            "allowed_m": std.pure_rotation_max_displacement_m,
        },
    )


@predicate("turn_below")
def turn_below(view: SceneView, std: CompileStandard, *, frames: Sequence[int]) -> Verdict:
    """Cumulative heading change stays inside the pure-translation envelope."""
    turn = cumulative_turn_deg([view.camera_pose(t).yaw_deg for t in frames])
    return Verdict(
        turn <= std.pure_translation_max_turn_deg,
        {
            "cum_turn_deg": round(turn, 1),
            "allowed_deg": std.pure_translation_max_turn_deg,
        },
    )


def _turn_segment_count(view: SceneView, frames: Sequence[int], threshold: float) -> int:
    from .geometry import wrap_deg

    count = 0
    active_run = False
    for left, right in pairwise(frames):
        # Repeating the exact same source frame is the explicit delay
        # intervention. It adds elapsed time but no new motion evidence, so it
        # is neutral and must not split one continuous turn into two segments.
        left_source = view.frames[left] if isinstance(view, ReindexedSceneView) else left
        right_source = view.frames[right] if isinstance(view, ReindexedSceneView) else right
        if left_source == right_source:
            continue
        active = (
            abs(
                wrap_deg(
                    view.camera_pose(right).yaw_deg
                    - view.camera_pose(left).yaw_deg
                )
            )
            >= threshold
        )
        if active and not active_run:
            count += 1
        active_run = active
    return count


@predicate("turn_segments_between")
def turn_segments_between(
    view: SceneView, std: CompileStandard, *, frames: Sequence[int]
) -> Verdict:
    """Heading changes form two or three disjoint segments separated by straight motion."""
    count = _turn_segment_count(view, frames, std.turn_segment_min_step_deg)
    return Verdict(
        std.multi_turn_min_segments <= count <= std.multi_turn_max_segments,
        {
            "turn_segment_count": count,
            "min_segments": std.multi_turn_min_segments,
            "max_segments": std.multi_turn_max_segments,
            "active_step_deg": std.turn_segment_min_step_deg,
        },
    )


@predicate("occluded_in_view")
def occluded_in_view(view: SceneView, std: CompileStandard, *, obj: str, frame: int) -> Verdict:
    """Target is hidden by an identified blocker in the active backend."""
    if hasattr(view, "occlusion"):
        observation = view.occlusion(obj, frame)
        return Verdict(
            True if observation.status == "occluded" else None if observation.status == "ambiguous" else False,
            dict(observation.witness),
        )
    pose = view.camera_pose(frame)
    target = view.object(obj)
    azimuth = abs(azimuth_deg(pose.xy, pose.yaw_deg, target.xy))
    in_frustum = azimuth <= std.fov_half_angle_deg
    blocked_by = view.occluders_between(obj, frame) if hasattr(view, "occluders_between") else ()
    return Verdict(
        in_frustum and bool(blocked_by),
        {
            "obj": obj,
            "frame": frame,
            "azimuth_deg": round(azimuth, 1),
            "fov_half_angle_deg": std.fov_half_angle_deg,
            "blocked_by": ",".join(blocked_by),
        },
    )


@predicate("disappearance_event")
def disappearance_event(
    view: SceneView,
    std: CompileStandard,
    *,
    obj: str,
    frame: int,
    cause: str = "either",
) -> Verdict:
    """A decisive, semantically eligible disappearance event at ``frame``."""

    if not hasattr(view, "occlusion"):
        return Verdict(None, {"obj": obj, "frame": frame, "reason": "backend_unsupported"})
    from .occlusion import occluder_category_allowed

    observation = view.occlusion(obj, frame)
    witness = dict(observation.witness)
    witness.update({"obj": obj, "frame": frame, "required_cause": cause})
    if observation.status == "ambiguous":
        return Verdict(None, witness)
    if observation.status == "occluded":
        # Search-time OBB rays nominate plausible blockers but do not carry a
        # render instance category.  Geometry therefore passes conservatively;
        # authoritative render compilation applies the semantic allow-list.
        if observation.kind == "geometry_ray_3d":
            witness["semantic_filter_phase"] = "render_compile"
            return Verdict(cause in {"either", "occluded"}, witness)
        category = witness.get("occluder_category")
        allowed = occluder_category_allowed(category if isinstance(category, str) else None)
        witness["occluder_category_allowed"] = allowed
        if not allowed:
            witness["semantic_reason"] = "disallowed_occluder_category"
            return Verdict(False, witness)
        return Verdict(cause in {"either", "occluded"}, witness)
    reason = witness.get("reason")
    is_out = reason in {"target_behind_camera", "target_projection_outside_image"}
    return Verdict(is_out and cause in {"either", "out_of_view"}, witness)


@predicate("all_landmarks_visible")
def all_landmarks_visible(
    view: SceneView,
    std: CompileStandard,
    *,
    viewpoint: str,
    facing: str,
    obj: str,
    frames: Sequence[int],
) -> Verdict:
    """Each imagined-frame landmark has enough independently clear sightings."""
    counts: dict[str, int] = {}
    ambiguous: dict[str, int] = {}
    for name in (viewpoint, facing, obj):
        rows = _tristates(view, name, frames, std)
        counts[name] = sum(state is True for _, state, _ in rows)
        ambiguous[name] = sum(state is None for _, state, _ in rows)
    holds = all(count >= std.landmark_min_visible_frames for count in counts.values())
    decision: bool | None = None if not holds and any(ambiguous.values()) else holds
    return Verdict(
        decision,
        {
            "visible_counts": counts,
            "ambiguous_counts": ambiguous,
            "required_frames": std.landmark_min_visible_frames,
        },
    )


@predicate("never_all_covisible")
def never_all_covisible(
    view: SceneView,
    std: CompileStandard,
    *,
    viewpoint: str,
    facing: str,
    obj: str,
    frames: Sequence[int],
) -> Verdict:
    """No single frame exposes all three landmarks as a static shortcut."""
    all_visible_frames: list[int] = []
    ambiguous_frames: list[int] = []
    for frame in frames:
        states = [view.visibility(name, frame).tristate(std) for name in (viewpoint, facing, obj)]
        if all(state is True for state in states):
            all_visible_frames.append(frame)
        elif all(state is not False for state in states) and any(state is None for state in states):
            ambiguous_frames.append(frame)
    if all_visible_frames:
        holds: bool | None = False
    elif ambiguous_frames:
        holds = None
    else:
        holds = True
    return Verdict(
        holds,
        {
            "all_visible_frames": all_visible_frames,
            "ambiguous_frames": ambiguous_frames,
        },
    )


def _imagined_pose_values(
    view: SceneView,
    std: CompileStandard,
    viewpoint: str,
    facing: str,
    yaw_offset_deg: float,
) -> tuple[tuple[float, float], float]:
    from .motifs import imagined_station

    origin = imagined_station(view.layout, viewpoint, facing, std.camera_height_m)
    if origin is None:
        raise ValueError(
            f"no imagined station for viewpoint={viewpoint} facing={facing}; "
            "imagined_pose_valid should have rejected this binding"
        )
    yaw = wrap_deg(bearing_deg(origin, view.object(facing).xy) + yaw_offset_deg)
    return origin, yaw


@predicate("imagined_pose_valid")
def imagined_pose_valid(
    view: SceneView,
    std: CompileStandard,
    *,
    viewpoint: str,
    facing: str,
) -> Verdict:
    """A person can stand at P, look at Q, and have a stable heading doing it.

    Two things have to hold and both are about the camera, not the objects.
    There must be somewhere to stand: the reference object's own centre if it
    is clear at eye height, otherwise a point just outside its footprint on the
    side it faces.  And the station has to be far enough from the object being
    faced that the heading is well defined - close up, a few centimetres of
    station offset would swing the imagined heading through a large angle.

    The distance is measured from the station rather than from the reference
    object's centre, because the station is where the render puts the camera
    and where the answer is derived.
    """
    from .motifs import imagined_station

    origin = imagined_station(view.layout, viewpoint, facing, std.camera_height_m)
    if origin is None:
        return Verdict(
            False,
            {
                "reason": "no_clear_station",
                "required_m": std.imagined_min_anchor_distance_m,
            },
        )
    distance = distance_m(origin, view.object(facing).xy)
    centre = view.object(viewpoint).xy
    return Verdict(
        distance >= std.imagined_min_anchor_distance_m,
        {
            "anchor_distance_m": round(distance, 3),
            "required_m": std.imagined_min_anchor_distance_m,
            "station_offset_m": round(distance_m(centre, origin), 3),
        },
    )


@predicate("imagined_sector_margin_ge")
def imagined_sector_margin_ge(
    view: SceneView,
    std: CompileStandard,
    *,
    viewpoint: str,
    facing: str,
    obj: str,
    yaw_offset_deg: float = 0.0,
) -> Verdict:
    """Target direction clears sector boundaries in the constructed frame."""
    origin, yaw = _imagined_pose_values(view, std, viewpoint, facing, yaw_offset_deg)
    azimuth = azimuth_deg(origin, yaw, view.object(obj).xy)
    margin = sector_margin_deg(azimuth)
    return Verdict(
        margin >= std.sector_margin_deg,
        {
            "imagined_yaw_deg": round(yaw, 1),
            "yaw_offset_deg": round(yaw_offset_deg, 1),
            "azimuth_deg": round(azimuth, 1),
            "sector": sector_of(azimuth),
            "margin_deg": round(margin, 1),
            "required_deg": std.sector_margin_deg,
        },
    )


@predicate("imagined_curve_sector_margins_ge")
def imagined_curve_sector_margins_ge(
    view: SceneView,
    std: CompileStandard,
    *,
    viewpoint: str,
    facing: str,
    obj: str,
) -> Verdict:
    """Every preregistered imagined-yaw point clears a sector boundary."""
    margins: list[tuple[int, float]] = []
    for offset in std.imagined_viewpoint_offsets_deg:
        origin, yaw = _imagined_pose_values(view, std, viewpoint, facing, offset)
        azimuth = azimuth_deg(origin, yaw, view.object(obj).xy)
        margins.append((offset, sector_margin_deg(azimuth)))
    minimum = min(margin for _, margin in margins)
    return Verdict(
        minimum >= std.sector_margin_deg,
        {
            "offset_margins_deg": {str(offset): round(margin, 1) for offset, margin in margins},
            "minimum_margin_deg": round(minimum, 1),
            "required_deg": std.sector_margin_deg,
        },
    )


@predicate("imagined_visibility_decisive")
def imagined_visibility_decisive(
    view: SceneView,
    std: CompileStandard,
    *,
    viewpoint: str,
    facing: str,
    obj: str,
    yaw_offset_deg: float = 0.0,
) -> Verdict:
    """Constructed-pose visibility lies outside the geometry ambiguity band."""
    origin, yaw = _imagined_pose_values(view, std, viewpoint, facing, yaw_offset_deg)
    probe = GeometrySceneView(
        layout=view.layout,
        poses=(Pose2D(origin[0], origin[1], yaw),),
        std=std,
    )
    observation = probe.visibility(obj, 0)
    state = observation.tristate(std)
    return Verdict(
        state is not None,
        {
            "imagined_yaw_deg": round(yaw, 1),
            "yaw_offset_deg": round(yaw_offset_deg, 1),
            "visibility_state": (
                "visible" if state is True else "invisible" if state is False else "ambiguous"
            ),
            "visibility_value": round(observation.value, 4),
            "unoccluded_ratio": round(observation.unoccluded_ratio, 3),
        },
    )


@predicate("imagined_curve_visibility_decisive")
def imagined_curve_visibility_decisive(
    view: SceneView,
    std: CompileStandard,
    *,
    viewpoint: str,
    facing: str,
    obj: str,
) -> Verdict:
    """Every preregistered imagined-yaw point has a decisive visibility state."""
    states: list[tuple[int, str]] = []
    for offset in std.imagined_viewpoint_offsets_deg:
        origin, yaw = _imagined_pose_values(view, std, viewpoint, facing, offset)
        probe = GeometrySceneView(
            layout=view.layout,
            poses=(Pose2D(origin[0], origin[1], yaw),),
            std=std,
        )
        state = probe.visibility(obj, 0).tristate(std)
        states.append(
            (
                offset,
                "visible" if state is True else "invisible" if state is False else "ambiguous",
            )
        )
    return Verdict(
        all(state != "ambiguous" for _, state in states),
        {"offset_states": {str(offset): state for offset, state in states}},
    )


@predicate("never_covisible")
def never_covisible(
    view: SceneView,
    std: CompileStandard,
    *,
    first: str,
    second: str,
    frames: Sequence[int],
) -> Verdict:
    """Two queried objects are never jointly readable in a single frame."""
    covisible: list[int] = []
    ambiguous: list[int] = []
    for frame in frames:
        states = (
            view.visibility(first, frame).tristate(std),
            view.visibility(second, frame).tristate(std),
        )
        if all(state is True for state in states):
            covisible.append(frame)
        elif all(state is not False for state in states) and any(
            state is None for state in states
        ):
            ambiguous.append(frame)
    holds: bool | None = False if covisible else None if ambiguous else True
    return Verdict(
        holds,
        {"covisible_frames": covisible, "ambiguous_frames": ambiguous},
    )


@predicate("chain_connected")
def chain_connected(
    view: SceneView,
    std: CompileStandard,
    *,
    first: str,
    second: str,
    anchor1: str,
    frames: Sequence[int],
    anchor2: str = "",
    anchor3: str = "",
) -> Verdict:
    """Every adjacent pair in X-L1-...-Lk-Y is clearly co-visible enough."""
    names = [first, anchor1]
    names.extend(name for name in (anchor2, anchor3) if name)
    names.append(second)
    counts: dict[str, int] = {}
    ambiguous_counts: dict[str, int] = {}
    for left, right in pairwise(names):
        edge = f"{left}->{right}"
        count = 0
        ambiguous = 0
        for frame in frames:
            states = (
                view.visibility(left, frame).tristate(std),
                view.visibility(right, frame).tristate(std),
            )
            if all(state is True for state in states):
                count += 1
            elif all(state is not False for state in states) and any(
                state is None for state in states
            ):
                ambiguous += 1
        counts[edge] = count
        ambiguous_counts[edge] = ambiguous
    holds = all(count >= std.chain_min_covisible_frames for count in counts.values())
    failing = [
        edge for edge, count in counts.items() if count < std.chain_min_covisible_frames
    ]
    unresolved = bool(failing) and all(
        counts[edge] + ambiguous_counts[edge] >= std.chain_min_covisible_frames
        for edge in failing
    )
    return Verdict(
        None if unresolved else holds,
        {
            "covisible_counts": counts,
            "ambiguous_counts": ambiguous_counts,
            "required_frames": std.chain_min_covisible_frames,
        },
    )


@predicate("pair_relation_margin_ge")
def pair_relation_margin_ge(
    view: SceneView,
    std: CompileStandard,
    *,
    obj: str,
    reference: str,
) -> Verdict:
    """Object relation clears sector boundaries in Y's intrinsic frame."""
    source = view.object(obj)
    anchor = view.object(reference)
    azimuth = azimuth_deg(anchor.xy, anchor.yaw_deg, source.xy)
    margin = sector_margin_deg(azimuth)
    return Verdict(
        margin >= std.sector_margin_deg,
        {
            "reference_yaw_deg": round(anchor.yaw_deg, 1),
            "azimuth_deg": round(azimuth, 1),
            "sector": sector_of(azimuth),
            "margin_deg": round(margin, 1),
            "required_deg": std.sector_margin_deg,
        },
    )


@predicate("closer_ratio_ge")
def closer_ratio_ge(
    view: SceneView,
    std: CompileStandard,
    *,
    first: str,
    second: str,
    anchor: str,
) -> Verdict:
    """The closer comparison is separated by the declared distance ratio."""
    anchor_xy = view.object(anchor).xy
    distances = (
        distance_m(view.object(first).xy, anchor_xy),
        distance_m(view.object(second).xy, anchor_xy),
    )
    smaller, larger = sorted(distances)
    ratio = larger / max(smaller, 1e-9)
    return Verdict(
        ratio >= std.closer_min_distance_ratio,
        {
            "first_distance_m": round(distances[0], 3),
            "second_distance_m": round(distances[1], 3),
            "distance_ratio": round(ratio, 3),
            "required_ratio": std.closer_min_distance_ratio,
        },
    )


@predicate("category_absent")
def category_absent(
    view: SceneView,
    std: CompileStandard,
    *,
    category: str,
) -> Verdict:
    """The queried category is absent in simulator scene truth."""
    del std
    matches = [obj.name for obj in view.objects() if obj.category == category]
    return Verdict(
        not matches,
        {"category": category, "scene_instance_count": len(matches), "instances": matches},
    )


@predicate("coverage_ratio_ge")
def coverage_ratio_ge(
    view: SceneView,
    std: CompileStandard,
    *,
    frames: Sequence[int],
) -> Verdict:
    """Coverage and residual hiding space jointly license an absence claim."""
    required = std.coverage_ratio_levels[-1]
    report = measure_coverage(view, std, frames=list(frames))
    ratio_ok = report.coverage_ratio >= required
    hidden_region_ok = report.max_hidden_diameter_m < std.coverage_hidden_object_size_m
    return Verdict(
        ratio_ok and hidden_region_ok,
        {
            "tier": "high",
            "coverage_ratio": round(report.coverage_ratio, 4),
            "required_ratio": required,
            "ratio_ok": ratio_ok,
            "covered_cell_count": report.covered_cell_count,
            "total_free_cell_count": report.total_free_cell_count,
            "uncovered_component_count": report.uncovered_component_count,
            "largest_uncovered_component_cells": report.largest_uncovered_component_cells,
            "max_hidden_diameter_m": round(report.max_hidden_diameter_m, 3),
            "hidden_object_size_m": std.coverage_hidden_object_size_m,
            "hidden_region_ok": hidden_region_ok,
        },
    )


@predicate("no_single_frame_coverage_sufficient")
def no_single_frame_coverage_sufficient(
    view: SceneView,
    std: CompileStandard,
    *,
    frames: Sequence[int],
) -> Verdict:
    """No individual frame can license the canonical absence answer."""
    sufficient_frames: list[int] = []
    per_frame: dict[str, dict[str, float]] = {}
    for frame in frames:
        report = measure_coverage(view, std, frames=[frame])
        sufficient = (
            report.coverage_ratio >= std.coverage_ratio_levels[-1]
            and report.max_hidden_diameter_m < std.coverage_hidden_object_size_m
        )
        if sufficient:
            sufficient_frames.append(frame)
        per_frame[str(frame)] = {
            "coverage_ratio": round(report.coverage_ratio, 4),
            "max_hidden_diameter_m": round(report.max_hidden_diameter_m, 3),
        }
    return Verdict(
        not sufficient_frames,
        {"sufficient_frames": sufficient_frames, "per_frame": per_frame},
    )


@predicate("drop_target_breaks_coverage")
def drop_target_breaks_coverage(
    view: SceneView,
    std: CompileStandard,
    *,
    target: str,
    frames: Sequence[int],
) -> Verdict:
    """The generic drop-key intervention provably destroys coverage evidence."""
    kept = [
        frame
        for frame in frames
        if view.visibility(target, frame).tristate(std) is False
    ]
    removed = [frame for frame in frames if frame not in kept]
    report = measure_coverage(view, std, frames=kept)
    still_sufficient = (
        report.coverage_ratio >= std.coverage_ratio_levels[-1]
        and report.max_hidden_diameter_m < std.coverage_hidden_object_size_m
    )
    holds = bool(kept) and bool(removed) and not still_sufficient
    return Verdict(
        holds,
        {
            "kept_frames": kept,
            "removed_frames": removed,
            "residual_coverage_ratio": round(report.coverage_ratio, 4),
            "residual_max_hidden_diameter_m": round(report.max_hidden_diameter_m, 3),
            "still_sufficient": still_sufficient,
        },
    )


@predicate("start_far_enough")
def start_far_enough(view: SceneView, std: CompileStandard, *, frame: int) -> Verdict:
    """The question pose is far enough from frame 0 for homing to be stable."""
    distance = distance_m(view.camera_pose(0).xy, view.camera_pose(frame).xy)
    return Verdict(
        distance >= std.homing_min_distance_m,
        {
            "frame": frame,
            "start_distance_m": round(distance, 2),
            "required_m": std.homing_min_distance_m,
        },
    )


@predicate("start_sector_margin_ge")
def start_sector_margin_ge(view: SceneView, std: CompileStandard, *, frame: int) -> Verdict:
    """The frame-0 position clears a four-sector boundary at question time."""
    start = view.camera_pose(0)
    pose = view.camera_pose(frame)
    azimuth = azimuth_deg(pose.xy, pose.yaw_deg, start.xy)
    margin = sector_margin_deg(azimuth)
    return Verdict(
        margin >= std.sector_margin_deg,
        {
            "frame": frame,
            "azimuth_deg": round(azimuth, 1),
            "sector": sector_of(azimuth),
            "margin_deg": round(margin, 1),
            "required_deg": std.sector_margin_deg,
        },
    )
