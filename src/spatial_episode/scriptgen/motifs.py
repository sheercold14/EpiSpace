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

import hashlib
import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush
from itertools import pairwise

from .geometry import (
    azimuth_deg,
    bearing_deg,
    distance_m,
    point_in_rotated_rect,
    rotated_rect_penetration_depth,
    sector_margin_deg,
    segment_rotated_rect_interval,
    wrap_deg,
)
from .sceneview import GeometrySceneView, Obstacle, Pose2D, SceneLayout, blocking_occluders

Motif = Callable[[SceneLayout, dict[str, str], int, random.Random], tuple[Pose2D, ...] | None]

_MOTIFS: dict[str, Motif] = {}


class MotifUnavailable(RuntimeError):
    """A scene cannot supply the geometry required by a trajectory motif."""


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


MAX_TURN_PER_FRAME_DEG = 32.0  # below the standard's 40-degree trackability cap
GRID_RESOLUTION_M = 0.05
PROPOSAL_CLEARANCE_M = 0.35
PROPOSAL_Z_LOW_M = 0.10
PROPOSAL_Z_HIGH_M = 1.70


@dataclass(frozen=True)
class _OccupancyGrid:
    origin_xy: tuple[float, float]
    width: int
    height: int
    blocked: bytes
    free_cells: tuple[int, ...]
    neighbours: tuple[tuple[tuple[int, float], ...], ...]
    component_ids: tuple[int, ...]
    component_cells: tuple[tuple[int, ...], ...]

    def index(self, ix: int, iy: int) -> int:
        return iy * self.width + ix

    def cell(self, index: int) -> tuple[int, int]:
        return (index % self.width, index // self.width)

    def world_xy(self, index: int) -> tuple[float, float]:
        ix, iy = self.cell(index)
        return (
            self.origin_xy[0] + ix * GRID_RESOLUTION_M,
            self.origin_xy[1] + iy * GRID_RESOLUTION_M,
        )

    def nearest_index(self, xy: tuple[float, float]) -> int:
        ix = round((xy[0] - self.origin_xy[0]) / GRID_RESOLUTION_M)
        iy = round((xy[1] - self.origin_xy[1]) / GRID_RESOLUTION_M)
        ix = min(max(ix, 0), self.width - 1)
        iy = min(max(iy, 0), self.height - 1)
        return self.index(ix, iy)

    def is_free(self, index: int) -> bool:
        return self.blocked[index] == 0


def _proposal_height_overlap(z_low: float, z_high: float) -> bool:
    return z_low <= PROPOSAL_Z_HIGH_M and z_high >= PROPOSAL_Z_LOW_M


@lru_cache(maxsize=32)
def _occupancy_grid(layout: SceneLayout) -> _OccupancyGrid:
    """Rasterise a conservative proposal grid once per immutable layout."""
    min_x, min_y = layout.walkable_min
    max_x, max_y = layout.walkable_max
    width = max(1, math.floor((max_x - min_x) / GRID_RESOLUTION_M) + 1)
    height = max(1, math.floor((max_y - min_y) / GRID_RESOLUTION_M) + 1)
    blocked = bytearray(width * height)

    # Keep the proposed body centre away from the floor bounding-box edge.
    for iy in range(height):
        y = min_y + iy * GRID_RESOLUTION_M
        for ix in range(width):
            x = min_x + ix * GRID_RESOLUTION_M
            if (
                x < min_x + PROPOSAL_CLEARANCE_M
                or x > max_x - PROPOSAL_CLEARANCE_M
                or y < min_y + PROPOSAL_CLEARANCE_M
                or y > max_y - PROPOSAL_CLEARANCE_M
            ):
                blocked[iy * width + ix] = 1

    for obstacle in layout.obstacles:
        if not _proposal_height_overlap(obstacle.z_low, obstacle.z_high):
            continue
        hx = obstacle.half_extents_xy[0] + PROPOSAL_CLEARANCE_M
        hy = obstacle.half_extents_xy[1] + PROPOSAL_CLEARANCE_M
        angle = math.radians(obstacle.yaw_deg)
        extent_x = abs(math.cos(angle)) * hx + abs(math.sin(angle)) * hy
        extent_y = abs(math.sin(angle)) * hx + abs(math.cos(angle)) * hy
        ix_low = max(0, math.ceil((obstacle.center_xy[0] - extent_x - min_x) / GRID_RESOLUTION_M))
        ix_high = min(
            width - 1,
            math.floor((obstacle.center_xy[0] + extent_x - min_x) / GRID_RESOLUTION_M),
        )
        iy_low = max(0, math.ceil((obstacle.center_xy[1] - extent_y - min_y) / GRID_RESOLUTION_M))
        iy_high = min(
            height - 1,
            math.floor((obstacle.center_xy[1] + extent_y - min_y) / GRID_RESOLUTION_M),
        )
        expanded = (hx, hy)
        for iy in range(iy_low, iy_high + 1):
            for ix in range(ix_low, ix_high + 1):
                index = iy * width + ix
                if blocked[index]:
                    continue
                point = (min_x + ix * GRID_RESOLUTION_M, min_y + iy * GRID_RESOLUTION_M)
                if point_in_rotated_rect(point, obstacle.center_xy, expanded, obstacle.yaw_deg):
                    blocked[index] = 1

    free_cells = tuple(index for index, value in enumerate(blocked) if value == 0)
    neighbour_offsets = (
        (-1, -1, math.sqrt(2.0)),
        (0, -1, 1.0),
        (1, -1, math.sqrt(2.0)),
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (-1, 1, math.sqrt(2.0)),
        (0, 1, 1.0),
        (1, 1, math.sqrt(2.0)),
    )
    neighbours: list[tuple[tuple[int, float], ...]] = []
    for index, value in enumerate(blocked):
        if value:
            neighbours.append(())
            continue
        cx, cy = index % width, index // width
        row: list[tuple[int, float]] = []
        for dx, dy, cost in neighbour_offsets:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            neighbour = ny * width + nx
            if blocked[neighbour]:
                continue
            if dx and dy and (blocked[cy * width + nx] or blocked[ny * width + cx]):
                continue
            row.append((neighbour, cost))
        neighbours.append(tuple(row))

    # Walls split multi-room scenes into disconnected regions. Record the
    # components once so candidate endpoints are never sampled in an
    # unreachable room (which would otherwise trigger repeated failed A*).
    component_ids = [-1] * len(blocked)
    component_cells: list[tuple[int, ...]] = []
    for root in free_cells:
        if component_ids[root] >= 0:
            continue
        component_id = len(component_cells)
        component_ids[root] = component_id
        stack = [root]
        cells: list[int] = []
        while stack:
            current = stack.pop()
            cells.append(current)
            for neighbour, _ in neighbours[current]:
                if component_ids[neighbour] >= 0:
                    continue
                component_ids[neighbour] = component_id
                stack.append(neighbour)
        component_cells.append(tuple(cells))
    return _OccupancyGrid(
        origin_xy=(min_x, min_y),
        width=width,
        height=height,
        blocked=bytes(blocked),
        free_cells=free_cells,
        neighbours=tuple(neighbours),
        component_ids=tuple(component_ids),
        component_cells=tuple(component_cells),
    )


def proposal_occupancy_grid(layout: SceneLayout) -> _OccupancyGrid:
    """Public read-only access to the trajectory proposal free-space grid."""
    return _occupancy_grid(layout)


def _octile_heuristic(grid: _OccupancyGrid, left: int, right: int) -> float:
    lx, ly = left % grid.width, left // grid.width
    rx, ry = right % grid.width, right // grid.width
    dx, dy = abs(lx - rx), abs(ly - ry)
    return max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy)


def _astar(grid: _OccupancyGrid, start: int, goal: int) -> tuple[int, ...] | None:
    """Eight-connected A* without diagonal corner cutting."""
    if not grid.is_free(start) or not grid.is_free(goal):
        return None
    if start == goal:
        return (start,)

    frontier: list[tuple[float, float, int]] = []
    heappush(frontier, (_octile_heuristic(grid, start, goal), 0.0, start))
    came_from = [-1] * len(grid.blocked)
    g_score = [math.inf] * len(grid.blocked)
    g_score[start] = 0.0
    goal_x, goal_y = goal % grid.width, goal // grid.width
    while frontier:
        _, current_g, current = heappop(frontier)
        if current_g > g_score[current]:
            continue
        if current == goal:
            path = [current]
            while current != start:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return tuple(path)

        for neighbour, cost in grid.neighbours[current]:
            tentative = current_g + cost
            if tentative >= g_score[neighbour]:
                continue
            came_from[neighbour] = current
            g_score[neighbour] = tentative
            neighbour_x, neighbour_y = neighbour % grid.width, neighbour // grid.width
            dx, dy = abs(neighbour_x - goal_x), abs(neighbour_y - goal_y)
            heuristic = max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy)
            heappush(
                frontier,
                (tentative + heuristic, tentative, neighbour),
            )
    return None


def _grid_line_clear(grid: _OccupancyGrid, start: int, end: int) -> bool:
    """Conservative Bresenham visibility over the proposal occupancy grid."""
    x, y = grid.cell(start)
    goal_x, goal_y = grid.cell(end)
    delta_x, delta_y = abs(goal_x - x), -abs(goal_y - y)
    step_x = 1 if x < goal_x else -1
    step_y = 1 if y < goal_y else -1
    error = delta_x + delta_y
    while True:
        if grid.blocked[y * grid.width + x]:
            return False
        if x == goal_x and y == goal_y:
            return True
        doubled = 2 * error
        move_x = doubled >= delta_y
        move_y = doubled <= delta_x
        previous_x, previous_y = x, y
        if move_x:
            error += delta_y
            x += step_x
        if move_y:
            error += delta_x
            y += step_y
        if (
            move_x
            and move_y
            and (
                grid.blocked[previous_y * grid.width + x]
                or grid.blocked[y * grid.width + previous_x]
            )
        ):
            return False


def _simplify_route(grid: _OccupancyGrid, cells: tuple[int, ...]) -> list[int]:
    """Greedily shortcut an A* route without sacrificing proposal clearance."""
    simplified = [cells[0]]
    current = 0
    while current < len(cells) - 1:
        next_index = len(cells) - 1
        while next_index > current + 1 and not _grid_line_clear(
            grid, cells[current], cells[next_index]
        ):
            next_index -= 1
        simplified.append(cells[next_index])
        current = next_index
    return simplified


def _sample_start(grid: _OccupancyGrid, target_xy: tuple[float, float], rng: random.Random) -> int:
    for _ in range(64):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        radius = rng.uniform(2.0, 4.0)
        candidate = grid.nearest_index(
            (
                target_xy[0] + radius * math.cos(angle),
                target_xy[1] + radius * math.sin(angle),
            )
        )
        if grid.is_free(candidate):
            return candidate
    annulus = [
        index
        for index in grid.free_cells
        if 2.0 <= distance_m(grid.world_xy(index), target_xy) <= 4.0
    ]
    pool = annulus or list(grid.free_cells)
    if not pool:
        raise MotifUnavailable("occupancy_grid_has_no_free_cells")
    return pool[rng.randrange(len(pool))]


def _sample_end(grid: _OccupancyGrid, start: int, rng: random.Random) -> int:
    start_xy = grid.world_xy(start)
    component = grid.component_cells[grid.component_ids[start]]
    diagonal = math.hypot(grid.width, grid.height) * GRID_RESOLUTION_M
    desired = max(2.0, diagonal * 0.35)
    best, best_distance = start, -1.0
    for _ in range(64):
        candidate = component[rng.randrange(len(component))]
        distance = distance_m(start_xy, grid.world_xy(candidate))
        if distance >= desired:
            return candidate
        if distance > best_distance:
            best, best_distance = candidate, distance
    return best


def _connect_cells(grid: _OccupancyGrid, start: int, end: int) -> tuple[int, ...] | None:
    """Use the trivial straight route when clear, otherwise run grid A*."""
    if start == end:
        return (start,)
    if _grid_line_clear(grid, start, end):
        return (start, end)
    return _astar(grid, start, end)


def _propose_route(
    layout: SceneLayout,
    target_xy: tuple[float, float],
    frame_count: int,
    rng: random.Random,
) -> list[tuple[float, float]] | None:
    grid = _occupancy_grid(layout)
    if not grid.free_cells:
        raise MotifUnavailable("occupancy_grid_has_no_free_cells")
    start = _sample_start(grid, target_xy, rng)
    for _ in range(8):
        end = _sample_end(grid, start, rng)
        indices = _connect_cells(grid, start, end)
        if indices is None:
            continue
        # The route's middle cell is itself a sampled free waypoint; retaining
        # it before line-of-sight simplification keeps the start→mid→end
        # proposal structure without paying for two independent A* searches.
        if len(indices) > 2:
            fraction = rng.uniform(0.3, 0.7)
            middle = max(1, min(len(indices) - 2, round((len(indices) - 1) * fraction)))
            first = _simplify_route(grid, indices[: middle + 1])
            second = _simplify_route(grid, indices[middle:])
            route_cells = first + second[1:]
        else:
            route_cells = list(indices)
        route = [grid.world_xy(index) for index in route_cells]
        if len(route) > frame_count:
            route = route[:frame_count]
        if (
            len(route) >= 2
            and sum(distance_m(left, right) for left, right in pairwise(route)) >= 1.0
        ):
            return route
    return None


def _walk_polyline(
    waypoints: list[tuple[float, float]], initial_yaw: float, frame_count: int
) -> tuple[Pose2D, ...]:
    """Distribute frames evenly over polyline segments with rate-limited yaw.

    Yaw chases the current segment's desired heading but never changes by
    more than MAX_TURN_PER_FRAME_DEG between frames: consecutive frames keep
    visual overlap, so the camera's own motion stays trackable from images
    (the ego-motion contract introduced in std.v2). Partial-visibility frames during the
    turn are expected; the script's transition zone accommodates them.
    """
    if frame_count < 2:
        raise ValueError("walk polyline needs at least two frames")
    segments = len(waypoints) - 1
    if segments > frame_count - 1:
        waypoints = waypoints[:frame_count]
        segments = len(waypoints) - 1
    lengths = [distance_m(left, right) for left, right in pairwise(waypoints)]
    intervals = [1] * segments
    for _ in range(frame_count - 1 - segments):
        index = max(range(segments), key=lambda i: lengths[i] / intervals[i])
        intervals[index] += 1

    positions = [waypoints[0]]
    for (start, end), count in zip(pairwise(waypoints), intervals, strict=True):
        positions.extend(
            (_lerp(start[0], end[0], step / count), _lerp(start[1], end[1], step / count))
            for step in range(1, count + 1)
        )

    poses = [Pose2D(positions[0][0], positions[0][1], initial_yaw)]
    yaw = initial_yaw
    for previous, position in pairwise(positions):
        desired = bearing_deg(previous, position) if previous != position else yaw
        delta = wrap_deg(desired - yaw)
        yaw = wrap_deg(yaw + max(-MAX_TURN_PER_FRAME_DEG, min(MAX_TURN_PER_FRAME_DEG, delta)))
        poses.append(Pose2D(position[0], position[1], yaw))
    return tuple(poses)


def _turn_in_place(
    xy: tuple[float, float], start_yaw: float, final_yaw: float, frame_count: int
) -> list[Pose2D]:
    """Rate-limited in-place rotation, including exactly ``frame_count`` poses."""
    poses: list[Pose2D] = []
    yaw = start_yaw
    for _ in range(frame_count):
        delta = wrap_deg(final_yaw - yaw)
        step = max(-MAX_TURN_PER_FRAME_DEG, min(MAX_TURN_PER_FRAME_DEG, delta))
        yaw = wrap_deg(yaw + step)
        poses.append(Pose2D(xy[0], xy[1], yaw))
    return poses


def _target_final_yaw(
    camera_xy: tuple[float, float],
    target_xy: tuple[float, float],
    rng: random.Random,
    *,
    desired_azimuths: tuple[float, ...] = (100.0, 150.0),
) -> float:
    """Balanced left/right/back final gaze with extra back acceptance mass."""
    turn_direction = rng.choice((-1.0, 1.0))
    desired_azimuth = turn_direction * rng.choice(desired_azimuths)
    desired_azimuth += rng.uniform(-8.0, 8.0)
    return wrap_deg(bearing_deg(camera_xy, target_xy) - desired_azimuth)


def _straight_past_endpoints(
    layout: SceneLayout, target_xy: tuple[float, float], target_size_m: float, rng: random.Random
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Sample a collision-free straight line passing beside, never through, a target."""
    grid = _occupancy_grid(layout)
    base_lateral_offset = max(
        1.1,
        target_size_m / 2.0 + PROPOSAL_CLEARANCE_M + 0.25,
    )
    for _ in range(96):
        angle = rng.uniform(-math.pi, math.pi)
        forward = (math.cos(angle), math.sin(angle))
        left = (-forward[1], forward[0])
        sign = rng.choice((-1.0, 1.0))
        # Forty percent retain the long ``back`` endpoint.  The remaining
        # proposals stop shortly after passing the target, so its bearing is
        # decisively left/right while heading remains exactly constant.
        back_endpoint = rng.random() < 0.4
        # Keep enough temporal baseline that a non-trivial frame permutation
        # still violates the declared per-step trackability contract.
        start_reach = rng.uniform(4.0, 4.5)
        end_reach = rng.uniform(2.2, 3.0) if back_endpoint else rng.uniform(0.5, 0.7)
        lateral_offset = (
            base_lateral_offset
            if back_endpoint
            else max(rng.uniform(1.3, 1.6), base_lateral_offset)
        )
        start_xy = (
            target_xy[0] - forward[0] * start_reach + left[0] * lateral_offset * sign,
            target_xy[1] - forward[1] * start_reach + left[1] * lateral_offset * sign,
        )
        end_xy = (
            target_xy[0] + forward[0] * end_reach + left[0] * lateral_offset * sign,
            target_xy[1] + forward[1] * end_reach + left[1] * lateral_offset * sign,
        )
        start, end = grid.nearest_index(start_xy), grid.nearest_index(end_xy)
        if not grid.is_free(start) or not grid.is_free(end):
            continue
        if grid.component_ids[start] != grid.component_ids[end]:
            continue
        if not _grid_line_clear(grid, start, end):
            continue
        return grid.world_xy(start), grid.world_xy(end)
    return None


def _direct_multi_turn_route(
    layout: SceneLayout, target_xy: tuple[float, float], rng: random.Random
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Direct free route whose heading differs enough from the initial target gaze."""
    grid = _occupancy_grid(layout)
    for _ in range(96):
        start = _sample_start(grid, target_xy, rng)
        component = grid.component_cells[grid.component_ids[start]]
        end = component[rng.randrange(len(component))]
        distance = distance_m(grid.world_xy(start), grid.world_xy(end))
        if not 2.0 <= distance <= 3.5 or not _grid_line_clear(grid, start, end):
            continue
        start_xy, end_xy = grid.world_xy(start), grid.world_xy(end)
        gaze = bearing_deg(start_xy, target_xy)
        route_heading = bearing_deg(start_xy, end_xy)
        heading_change = abs(wrap_deg(route_heading - gaze))
        if 55.0 <= heading_change <= 120.0:
            return start_xy, end_xy
    return None


@motif("walk_and_turn")
def walk_and_turn(
    layout: SceneLayout, binding: dict[str, str], frame_count: int, rng: random.Random
) -> tuple[Pose2D, ...] | None:
    """Start facing the target, then walk away through one or two turns.

    The walk begins at a point with clear line of sight preference (near the
    target, facing it), then leaves towards a far corner of the walkable area
    so that later frames tend to lose sight of the target and accumulate
    heading change. Whether that actually happens is for the checker to judge.
    """
    target = layout.object(binding["target"])
    look_frames = max(2, frame_count // 4)
    waypoints = _propose_route(layout, target.xy, frame_count - look_frames, rng)
    if waypoints is None:
        return None
    initial_yaw = bearing_deg(waypoints[0], target.xy)
    walk = list(_walk_polyline(waypoints, initial_yaw, frame_count - look_frames))

    # Arrive, then look around: rotate in place (rate-limited) towards a
    # random final gaze so the target's final bearing spreads over
    # left/right/back instead of collapsing onto "behind the walker".
    end = walk[-1]
    final_yaw = _target_final_yaw(end.xy, target.xy, rng)
    yaw = end.yaw_deg
    for _ in range(look_frames):
        delta = wrap_deg(final_yaw - yaw)
        step = max(-MAX_TURN_PER_FRAME_DEG, min(MAX_TURN_PER_FRAME_DEG, delta))
        yaw = wrap_deg(yaw + step)
        walk.append(Pose2D(end.x, end.y, yaw))
    return tuple(walk)


@motif("stand_and_turn")
def stand_and_turn(
    layout: SceneLayout, binding: dict[str, str], frame_count: int, rng: random.Random
) -> tuple[Pose2D, ...]:
    """T2: observe a target, then rotate in place until it leaves the view."""
    target = layout.object(binding["target"])
    grid = _occupancy_grid(layout)
    start = grid.world_xy(_sample_start(grid, target.xy, rng))
    initial_yaw = bearing_deg(start, target.xy)
    # A pure rotation retains more of a large target near the image edge than
    # the geometry proxy predicts. Use the existing 150-degree proposal tier
    # so rendered disappearance still leaves enough observable rotation.
    final_yaw = _target_final_yaw(start, target.xy, rng, desired_azimuths=(100.0, 150.0))
    poses = [Pose2D(start[0], start[1], initial_yaw)] * 2
    poses.extend(_turn_in_place(start, initial_yaw, final_yaw, frame_count - 2))
    return tuple(poses)


@motif("walk_straight_past")
def walk_straight_past(
    layout: SceneLayout, binding: dict[str, str], frame_count: int, rng: random.Random
) -> tuple[Pose2D, ...] | None:
    """T3: translate past a side-front target with exactly constant heading."""
    target = layout.object(binding["target"])
    endpoints = _straight_past_endpoints(layout, target.xy, target.size_m, rng)
    if endpoints is None:
        return None
    start, end = endpoints
    heading = bearing_deg(start, end) if start != end else bearing_deg(start, target.xy)
    return tuple(
        Pose2D(_lerp(start[0], end[0], t), _lerp(start[1], end[1], t), heading)
        for t in (index / (frame_count - 1) for index in range(frame_count))
    )


@motif("walk_multi_turn")
def walk_multi_turn(
    layout: SceneLayout, binding: dict[str, str], frame_count: int, rng: random.Random
) -> tuple[Pose2D, ...] | None:
    """T4: two or three separated rotations with straight motion between them."""
    target = layout.object(binding["target"])
    route = _direct_multi_turn_route(layout, target.xy, rng)
    if route is None:
        return None
    start, end = route
    initial_yaw = bearing_deg(start, target.xy)
    route_yaw = bearing_deg(start, end) if start != end else initial_yaw
    requested_segments = rng.choice((2, 3))
    first_turn_frames = 3
    middle_turn_frames = 2 if requested_segments == 3 else 0
    final_turn_frames = 4
    straight_frames = frame_count - 2 - first_turn_frames - middle_turn_frames - final_turn_frames
    poses = [Pose2D(start[0], start[1], initial_yaw)] * 2
    poses.extend(_turn_in_place(start, initial_yaw, route_yaw, first_turn_frames))
    first_straight = straight_frames // 2 if middle_turn_frames else straight_frames
    second_straight = straight_frames - first_straight
    poses.extend(
        Pose2D(
            _lerp(start[0], end[0], step / straight_frames),
            _lerp(start[1], end[1], step / straight_frames),
            route_yaw,
        )
        for step in range(1, first_straight + 1)
    )
    gaze_yaw = route_yaw
    if middle_turn_frames:
        midpoint = (
            _lerp(start[0], end[0], first_straight / straight_frames),
            _lerp(start[1], end[1], first_straight / straight_frames),
        )
        gaze_yaw = wrap_deg(route_yaw + rng.choice((-45.0, 45.0)))
        poses.extend(_turn_in_place(midpoint, route_yaw, gaze_yaw, middle_turn_frames))
        poses.extend(
            Pose2D(
                _lerp(midpoint[0], end[0], step / second_straight),
                _lerp(midpoint[1], end[1], step / second_straight),
                gaze_yaw,
            )
            for step in range(1, second_straight + 1)
        )
    final_yaw = _target_final_yaw(end, target.xy, rng)
    poses.extend(_turn_in_place(end, gaze_yaw, final_yaw, final_turn_frames))
    return tuple(poses)


@lru_cache(maxsize=128)
def _occluded_endpoints(layout: SceneLayout, target_name: str) -> tuple[int, ...]:
    """Free cells from which the bound target is geometrically invisible.

    The expensive scan is cached per immutable scene/target pair.  It uses
    the same height-aware rays and visibility thresholds as the checker, so a
    motif cannot nominate an endpoint that later turns out to be on the
    occluder's near side after grid snapping.
    """
    grid = _occupancy_grid(layout)
    target = layout.object(target_name)
    from .standards import STD_V1

    candidates: list[int] = []
    for index in grid.free_cells:
        xy = grid.world_xy(index)
        distance = distance_m(xy, target.xy)
        if not 0.75 <= distance <= STD_V1.max_view_distance_m:
            continue
        pose = Pose2D(xy[0], xy[1], bearing_deg(xy, target.xy))
        from .sceneview import GeometrySceneView

        observation = GeometrySceneView(layout, (pose,), STD_V1).visibility(target_name, 0)
        if observation.tristate(STD_V1) is False and blocking_occluders(layout, xy, target):
            candidates.append(index)
    return tuple(candidates)


@lru_cache(maxsize=256)
def _decisive_occluded_routes(
    layout: SceneLayout, target_name: str
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Cache direct routes whose first disappearance is the blocker event.

    Coverage asks for up to 30 initial candidates per binding.  Re-running a
    full endpoint/start search for every attempt made no-route bindings scale
    30x worse.  This deterministic pool pays the scene/target scan once; the
    motif still uses its seeded RNG to choose a route and final direction.
    """

    from .occlusion import first_decisive_disappearance
    from .sceneview import GeometrySceneView
    from .standards import STD_V1

    grid = _occupancy_grid(layout)
    target = layout.object(target_name)
    endpoints = list(_occluded_endpoints(layout, target_name))
    obstacle_by_id = {
        obstacle.entity_id: obstacle
        for obstacle in layout.occlusion_obstacles
        if obstacle.entity_id is not None
    }

    def endpoint_score(index: int) -> float:
        xy = grid.world_xy(index)
        target_distance = distance_m(xy, target.xy)
        scores: list[float] = []
        for blocker_id in blocking_occluders(layout, xy, target):
            obstacle = obstacle_by_id.get(blocker_id)
            if obstacle is None:
                continue
            interval = segment_rotated_rect_interval(
                xy,
                target.xy,
                obstacle.center_xy,
                obstacle.half_extents_xy,
                obstacle.yaw_deg,
            )
            if interval is None:
                continue
            enter, leave = interval
            midpoint = (enter + leave) / 2.0
            crossing = (
                xy[0] + (target.xy[0] - xy[0]) * midpoint,
                xy[1] + (target.xy[1] - xy[1]) * midpoint,
            )
            penetration = rotated_rect_penetration_depth(
                crossing,
                obstacle.center_xy,
                obstacle.half_extents_xy,
                obstacle.yaw_deg,
            ) / max(min(obstacle.half_extents_xy), 1e-9)
            chord_m = (leave - enter) * target_distance
            angular_cover = (
                (2.0 * min(obstacle.half_extents_xy))
                / max(distance_m(xy, obstacle.center_xy), 1e-9)
            ) / (target.size_m / max(target_distance, 1e-9))
            scores.append(4.0 * penetration + angular_cover + chord_m)
        return max(scores, default=0.0)

    endpoints.sort(key=endpoint_score, reverse=True)
    route_rng = random.Random(
        int(
            hashlib.sha256(f"{layout.scene_id}:{target_name}:decisive".encode()).hexdigest()[:16],
            16,
        )
    )
    routes: list[tuple[tuple[float, float], ...]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for end_index in endpoints[:128]:
        end_xy = grid.world_xy(end_index)
        for _ in range(64):
            start_index = _sample_start(grid, target.xy, route_rng)
            if grid.component_ids[start_index] != grid.component_ids[end_index]:
                continue
            if (start_index, end_index) in seen_pairs:
                continue
            seen_pairs.add((start_index, end_index))
            start_xy = grid.world_xy(start_index)
            if distance_m(start_xy, end_xy) < 1.0:
                continue
            if not _grid_line_clear(grid, start_index, end_index):
                continue
            # Replay a representative translated prefix.  This rejects a
            # route if distance or an earlier viewing transition makes the
            # target disappear before the endpoint blocker, or if it
            # reappears immediately afterwards.
            probe_count = 12
            probe_poses = tuple(
                Pose2D(
                    _lerp(start_xy[0], end_xy[0], step / (probe_count - 1)),
                    _lerp(start_xy[1], end_xy[1], step / (probe_count - 1)),
                    bearing_deg(
                        (
                            _lerp(start_xy[0], end_xy[0], step / (probe_count - 1)),
                            _lerp(start_xy[1], end_xy[1], step / (probe_count - 1)),
                        ),
                        target.xy,
                    ),
                )
                for step in range(probe_count)
            )
            view = GeometrySceneView(layout, probe_poses, STD_V1)
            event = first_decisive_disappearance(view, STD_V1, target_name)
            if event is None or event.cause != "occluded":
                continue
            states = [
                view.visibility(target_name, frame).tristate(STD_V1) for frame in range(probe_count)
            ]
            if sum(state is True for state in states[: event.frame]) < 2:
                continue
            if any(state is not False for state in states[event.frame :]):
                continue
            routes.append((start_xy, end_xy))
            if len(routes) >= 10:
                return tuple(routes)
    return tuple(routes)


def _occluded_route(
    layout: SceneLayout,
    target_name: str,
    rng: random.Random,
    *,
    require_decisive_boundary: bool = False,
) -> list[tuple[float, float]] | None:
    if require_decisive_boundary:
        routes = _decisive_occluded_routes(layout, target_name)
        return list(rng.choice(routes)) if routes else None
    grid = _occupancy_grid(layout)
    target = layout.object(target_name)
    endpoints = list(_occluded_endpoints(layout, target_name))
    obstacle_by_id = {
        obstacle.entity_id: obstacle
        for obstacle in layout.occlusion_obstacles
        if obstacle.entity_id is not None
    }

    def endpoint_score(index: int) -> float:
        xy = grid.world_xy(index)
        target_distance = distance_m(xy, target.xy)
        scores: list[float] = []
        for blocker_id in blocking_occluders(layout, xy, target):
            obstacle = obstacle_by_id.get(blocker_id)
            if obstacle is None:
                continue
            interval = segment_rotated_rect_interval(
                xy,
                target.xy,
                obstacle.center_xy,
                obstacle.half_extents_xy,
                obstacle.yaw_deg,
            )
            if interval is None:
                continue
            enter, leave = interval
            midpoint = (enter + leave) / 2.0
            crossing = (
                xy[0] + (target.xy[0] - xy[0]) * midpoint,
                xy[1] + (target.xy[1] - xy[1]) * midpoint,
            )
            penetration = rotated_rect_penetration_depth(
                crossing,
                obstacle.center_xy,
                obstacle.half_extents_xy,
                obstacle.yaw_deg,
            ) / max(min(obstacle.half_extents_xy), 1e-9)
            chord_m = (leave - enter) * target_distance
            # Conservative angular cover: use the blocker's smaller footprint
            # dimension, so a long but edge-on bed is not overrated.
            angular_cover = (
                (2.0 * min(obstacle.half_extents_xy))
                / max(distance_m(xy, obstacle.center_xy), 1e-9)
            ) / (target.size_m / max(target_distance, 1e-9))
            scores.append(4.0 * penetration + angular_cover + chord_m)
        return max(scores, default=0.0)

    # Geometry offers many valid cells; start with the one whose blocker is
    # deepest on the target ray and has the strongest angular coverage.
    endpoints.sort(key=endpoint_score, reverse=True)
    for end_index in endpoints[:128]:
        end_xy = grid.world_xy(end_index)
        if not blocking_occluders(layout, end_xy, target):
            continue
        for _ in range(64):
            start_index = _sample_start(grid, target.xy, rng)
            start_xy = grid.world_xy(start_index)
            if grid.component_ids[start_index] != grid.component_ids[end_index]:
                continue
            if distance_m(start_xy, end_xy) < 1.0:
                continue
            if blocking_occluders(layout, start_xy, target):
                continue
            # The causal motif prefers a direct camera path.  Falling back to
            # a long A* detour is both expensive and semantically risky: the
            # target can become too small or leave the view well before the
            # eventual blocker.  Other (legacy) occlusion motifs retain A*.
            if require_decisive_boundary:
                if not _grid_line_clear(grid, start_index, end_index):
                    continue
                cells = (start_index, end_index)
            else:
                cells = _connect_cells(grid, start_index, end_index)
            if cells is None:
                continue
            # Stop at the first robustly blocked cell on the connected route.
            # Continuing all the way to the OBB-score maximum can walk around
            # a non-convex mesh (notably a swivel-chair back) and expose the
            # target again even though the fitted box still intersects the
            # ray.  The first clear->blocked transition is also the intended
            # causal event for an occlusion episode.
            boundary = next(
                (
                    position
                    for position, cell in enumerate(cells[1:], start=1)
                    if blocking_occluders(layout, grid.world_xy(cell), target)
                    and distance_m(start_xy, grid.world_xy(cell)) >= 1.0
                ),
                None,
            )
            if boundary is None:
                continue
            if require_decisive_boundary:
                # A causal occlusion route may not let the target disappear
                # from distance before reaching its blocker.  With target-
                # tracked gaze and a clear ray, geometry visibility reduces
                # to size / distance, so check every dense path cell before
                # simplifying.  The boundary itself must already be a
                # definitely invisible, in-frustum blocker observation.
                from .standards import STD_V1

                pre_boundary = cells[:boundary]
                maximum_visible_distance = target.size_m / STD_V1.geom_min_visible_ratio
                if not pre_boundary or any(
                    distance_m(grid.world_xy(cell), target.xy) > maximum_visible_distance
                    for cell in pre_boundary
                ):
                    continue
                from .sceneview import GeometrySceneView

                boundary_xy = grid.world_xy(cells[boundary])
                boundary_pose = Pose2D(
                    boundary_xy[0],
                    boundary_xy[1],
                    bearing_deg(boundary_xy, target.xy),
                )
                boundary_view = GeometrySceneView(layout, (boundary_pose,), STD_V1)
                if (
                    boundary_view.visibility(target_name, 0).tristate(STD_V1) is not False
                    or boundary_view.occlusion(target_name, 0).status != "occluded"
                ):
                    continue
            route_cells = _simplify_route(grid, cells[: boundary + 1])
            route = [grid.world_xy(index) for index in route_cells]
            if len(route) >= 2:
                return route
    return None


@motif("walk_to_occlusion")
def walk_to_occlusion(
    layout: SceneLayout, binding: dict[str, str], frame_count: int, rng: random.Random
) -> tuple[Pose2D, ...] | None:
    """Track a target while walking behind a verified occluder.

    Unlike the generic walk motif, this camera keeps its optical axis on the
    target throughout the translation.  The target therefore disappears
    because an obstacle enters the line of sight, not because the camera first
    looks away and later swings back.  A short terminal hold supplies a clean,
    monotonic invisible suffix for the memory question.
    """
    target = layout.object(binding["target"])
    route = _occluded_route(layout, binding["target"], rng)
    if route is None:
        return None

    # Once hidden, retain a small, trackable left/right head sweep.  This
    # records non-trivial ego-motion after disappearance (the capability being
    # tested) while keeping the occluded target inside the 90-degree frustum.
    # Its ordered 21-degree steps are valid; suitable permutations create a
    # 42-degree discontinuity and are therefore machine-verifiable negatives.
    sweep_sign = rng.choice((-1.0, 1.0))
    sweep_offsets = (0.0, 21.0 * sweep_sign, 0.0, -21.0 * sweep_sign, 0.0)
    initial_yaw = bearing_deg(route[0], target.xy)
    translated = _walk_polyline(route, initial_yaw, frame_count - len(sweep_offsets))
    tracked = [Pose2D(pose.x, pose.y, bearing_deg(pose.xy, target.xy)) for pose in translated]
    end = tracked[-1]
    final_bearing = bearing_deg(end.xy, target.xy)
    tracked.extend(
        Pose2D(end.x, end.y, wrap_deg(final_bearing + offset)) for offset in sweep_offsets
    )
    return tuple(tracked)


@motif("walk_through_occlusion")
def walk_through_occlusion(
    layout: SceneLayout, binding: dict[str, str], frame_count: int, rng: random.Random
) -> tuple[Pose2D, ...] | None:
    """Acquire a real blocker, then continue trackable ego-rotation behind it.

    The route ends at the first geometric clear -> blocked boundary.  Its
    translated prefix keeps the target centred, establishing both identity
    evidence and the causal occlusion event.  The remaining frames rotate in
    place in at most 32-degree increments, spreading final target direction
    across all four sectors while preserving a 3+ frame hidden suffix.
    """

    target = layout.object(binding["target"])
    route = _occluded_route(layout, binding["target"], rng, require_decisive_boundary=True)
    if route is None:
        return None

    # Seven post-event poses are enough for the longest 160-degree turn (or
    # the 84 -> 0 front excursion) while reserving at least nine poses for a
    # long room-crossing route at the 1.2 m translation-step contract.
    post_frames = min(7, frame_count - 6)
    translated_frames = frame_count - post_frames
    initial_yaw = bearing_deg(route[0], target.xy)
    translated = _walk_polyline(route, initial_yaw, translated_frames)
    tracked = [Pose2D(pose.x, pose.y, bearing_deg(pose.xy, target.xy)) for pose in translated]
    end = tracked[-1]
    target_bearing = bearing_deg(end.xy, target.xy)
    desired_sector = rng.choice(("front", "left", "right", "back"))
    desired_azimuth = (
        rng.choice((-160.0, 160.0))
        if desired_sector == "back"
        else {"front": 0.0, "left": 90.0, "right": -90.0}[desired_sector]
    )

    # A front answer would otherwise require zero post-event motion.  Make a
    # bounded 84-degree excursion and return, yielding 168 degrees cumulative
    # rotation without changing the final front label.
    yaw_targets = (
        (wrap_deg(target_bearing + rng.choice((-84.0, 84.0))), target_bearing)
        if desired_azimuth == 0.0
        else (wrap_deg(target_bearing - desired_azimuth),)
    )
    yaws: list[float] = []
    current = target_bearing
    for target_yaw in yaw_targets:
        while len(yaws) < post_frames:
            delta = wrap_deg(target_yaw - current)
            if abs(delta) <= 1e-6:
                break
            current = wrap_deg(
                current + max(-MAX_TURN_PER_FRAME_DEG, min(MAX_TURN_PER_FRAME_DEG, delta))
            )
            yaws.append(current)
        if len(yaws) >= post_frames:
            break
    yaws.extend([current] * (post_frames - len(yaws)))
    tracked.extend(Pose2D(end.x, end.y, yaw) for yaw in yaws)
    return tuple(tracked)


MIN_STATION_RANGE_M = 0.75
# A 90-degree frustum only contains an object of extent ``s`` from about 1.2*s
# away; closer than that the object is clipped by the frame edge and reads as an
# anonymous surface rather than as its category, so a question naming it has no
# groundable referent however many pixels it covers. Pilot frames put the
# boundary just past 1.4*s, and the factor keeps margin beyond it so the object
# is framed with some surrounding context.
LANDMARK_STANDOFF_FACTOR = 1.5


def _landmark_view_cap_m(size_m: float) -> float:
    """Farthest distance at which an object of this size stays clearly visible."""
    from .standards import STD_V1

    return min(STD_V1.max_view_distance_m, size_m / STD_V1.geom_min_visible_ratio)


def _landmark_view_floor_m(size_m: float) -> float:
    """Closest distance at which an object of this size still frames whole."""
    return max(MIN_STATION_RANGE_M, LANDMARK_STANDOFF_FACTOR * size_m)


def _covering_arc_deg(bearings: list[float]) -> tuple[float, float]:
    """Minimal circular arc containing every bearing: (start_deg, spread_deg)."""
    ordered = sorted(bearing % 360.0 for bearing in bearings)
    if len(ordered) == 1:
        return ordered[0], 0.0
    gaps = [
        (ordered[(index + 1) % len(ordered)] - ordered[index]) % 360.0
        for index in range(len(ordered))
    ]
    largest = max(range(len(gaps)), key=gaps.__getitem__)
    return ordered[(largest + 1) % len(ordered)], 360.0 - gaps[largest]


def _survey_station(
    layout: SceneLayout,
    landmark_names: tuple[str, ...],
    rng: random.Random,
    *,
    min_spread_deg: float = 0.0,
    max_spread_deg: float = 360.0,
) -> tuple[float, float]:
    """Return the binding's cached deterministic survey station."""
    del rng  # trajectory randomness belongs to the scan, not station viability
    return _survey_station_cached(
        layout,
        landmark_names,
        min_spread_deg,
        max_spread_deg,
    )


@lru_cache(maxsize=65536)
def _survey_station_cached(
    layout: SceneLayout,
    landmark_names: tuple[str, ...],
    min_spread_deg: float,
    max_spread_deg: float,
) -> tuple[float, float]:
    """Best free point with unobstructed, close-enough rays to all landmarks.

    A candidate must keep every landmark inside its clearly-visible distance
    band and, when a spread window is requested over three or more landmarks,
    keep the minimal bearing arc inside that window.  Among valid candidates
    the station maximising the worst apparent angular size wins; scenes with
    no valid candidate raise ``MotifUnavailable`` instead of silently
    degrading to an arbitrary free cell.
    """
    grid = _occupancy_grid(layout)
    candidates = list(grid.free_cells)
    rng = random.Random(
        "survey_station|" + "|".join(landmark_names) + f"|{min_spread_deg:.6f}|{max_spread_deg:.6f}"
    )
    rng.shuffle(candidates)
    landmarks = tuple(layout.object(name) for name in landmark_names)
    caps = tuple(_landmark_view_cap_m(landmark.size_m) for landmark in landmarks)
    floors = tuple(_landmark_view_floor_m(landmark.size_m) for landmark in landmarks)
    check_spread = (min_spread_deg > 0.0 or max_spread_deg < 360.0) and len(landmarks) >= 3
    best_xy: tuple[float, float] | None = None
    best_score = -math.inf
    occlusion_checks = 0
    valid_count = 0
    for index in candidates:
        if occlusion_checks >= 1024 or valid_count >= 16:
            break
        xy = grid.world_xy(index)
        distances = tuple(distance_m(xy, landmark.xy) for landmark in landmarks)
        if any(
            not floor <= distance <= cap
            for distance, floor, cap in zip(distances, floors, caps, strict=True)
        ):
            continue
        if check_spread:
            _, spread = _covering_arc_deg([bearing_deg(xy, landmark.xy) for landmark in landmarks])
            if not min_spread_deg <= spread <= max_spread_deg:
                continue
        occlusion_checks += 1
        if any(blocking_occluders(layout, xy, landmark) for landmark in landmarks):
            continue
        valid_count += 1
        score = min(
            landmark.size_m / distance
            for landmark, distance in zip(landmarks, distances, strict=True)
        )
        if score > best_score:
            best_score, best_xy = score, xy
    if best_xy is None:
        raise MotifUnavailable("survey_station:no_valid_station")
    return best_xy


@motif("survey")
def survey(
    layout: SceneLayout, binding: dict[str, str], frame_count: int, rng: random.Random
) -> tuple[Pose2D, ...]:
    """Stationary, rate-limited panorama exposing every bound landmark."""
    names = tuple(binding[name] for name in ("viewpoint", "facing", "target") if name in binding)
    station = _survey_station(layout, names, rng)
    start_yaw = rng.uniform(-180.0, 180.0)
    direction = rng.choice((-1.0, 1.0))
    step = direction * 360.0 / (frame_count - 1)
    return tuple(
        Pose2D(station[0], station[1], wrap_deg(start_yaw + index * step))
        for index in range(frame_count)
    )


SURVEY_ARC_MIN_SPREAD_DEG = 100.0
SURVEY_ARC_MARGIN_DEG = 25.0
SURVEY_ARC_MIN_MARGIN_DEG = 8.0


@motif("survey_arc")
def survey_arc(
    layout: SceneLayout, binding: dict[str, str], frame_count: int, rng: random.Random
) -> tuple[Pose2D, ...]:
    """Stationary rate-limited scan over the minimal arc covering all landmarks.

    Unlike ``survey`` (a full panorama), the sweep covers only the landmark
    bearing arc plus a small margin on each side.  The station must offer a
    bearing spread above 100 degrees, so no single 90-degree frustum frame can
    expose all three landmarks (the never_all_covisible contract), while the
    whole sweep still fits the per-frame turn cap within the frame budget.
    """
    names = tuple(binding[name] for name in ("viewpoint", "facing", "target") if name in binding)
    max_sweep = (frame_count - 1) * MAX_TURN_PER_FRAME_DEG
    max_spread = max_sweep - 2.0 * SURVEY_ARC_MIN_MARGIN_DEG
    if max_spread <= SURVEY_ARC_MIN_SPREAD_DEG:
        raise MotifUnavailable("survey_arc:frame_budget_too_small")
    station = _survey_station(
        layout,
        names,
        rng,
        min_spread_deg=SURVEY_ARC_MIN_SPREAD_DEG,
        max_spread_deg=max_spread,
    )
    start, spread = _covering_arc_deg(
        [bearing_deg(station, layout.object(name).xy) for name in names]
    )
    margin = min(SURVEY_ARC_MARGIN_DEG, (max_sweep - spread) / 2.0)
    sweep = spread + 2.0 * margin
    low = start - margin
    step = sweep / (frame_count - 1)
    forward = rng.random() < 0.5
    yaws = (
        [low + index * step for index in range(frame_count)]
        if forward
        else [low + sweep - index * step for index in range(frame_count)]
    )
    return tuple(Pose2D(station[0], station[1], wrap_deg(yaw)) for yaw in yaws)


def _landmark_chain(binding: dict[str, str]) -> tuple[str, ...]:
    anchors = tuple(
        binding[name]
        for name in sorted(
            (name for name in binding if name.startswith("anchor")),
            key=lambda name: int(name.removeprefix("anchor")),
        )
    )
    return (binding["target"], *anchors, binding["other"])


def pair_framable_station_exists(layout: SceneLayout, left_name: str, right_name: str) -> bool:
    """Whether some free station frames both objects clearly at jitter extremes.

    Pair-only form of the snapshot station admission: no target/other
    exclusivity and no final-edge margin.  Passing here is necessary for any
    chain edge over this pair, so the result can prune bindings without
    excluding a chain the full station search could still realise.  Sampling
    density mirrors ``_edge_station_pool`` and the result is deterministic in
    the pair, so it is safe to cache across capabilities.
    """
    if left_name > right_name:
        left_name, right_name = right_name, left_name
    return _pair_framable(layout, left_name, right_name)


@lru_cache(maxsize=65536)
def _pair_framable(layout: SceneLayout, left_name: str, right_name: str) -> bool:
    from .standards import STD_V1 as std

    grid = _occupancy_grid(layout)
    left, right = layout.object(left_name), layout.object(right_name)
    cap_left = _landmark_view_cap_m(left.size_m)
    cap_right = _landmark_view_cap_m(right.size_m)
    floor_left = _landmark_view_floor_m(left.size_m)
    floor_right = _landmark_view_floor_m(right.size_m)
    edge = distance_m(left.xy, right.xy)
    half = edge / 2.0
    reach = min(cap_left, cap_right)
    if half >= reach:
        return False
    jitter = std.snapshot_yaw_jitter_deg
    rng = random.Random(f"pair_framable|{left_name}|{right_name}")

    def admit(index: int) -> bool:
        xy = grid.world_xy(index)
        left_distance = distance_m(xy, left.xy)
        right_distance = distance_m(xy, right.xy)
        if not floor_left <= left_distance <= cap_left:
            return False
        if not floor_right <= right_distance <= cap_right:
            return False
        left_yaw = bearing_deg(xy, left.xy)
        separation = wrap_deg(bearing_deg(xy, right.xy) - left_yaw)
        if abs(separation) > 70.0:
            return False
        yaw = wrap_deg(left_yaw + separation / 2.0)
        probes = GeometrySceneView(
            layout,
            tuple(Pose2D(xy[0], xy[1], wrap_deg(yaw + delta)) for delta in (-jitter, 0.0, jitter)),
            std,
        )
        return all(
            probes.visibility(name, frame).tristate(std) is True
            for frame in range(3)
            for name in (left_name, right_name)
        )

    mid = ((left.xy[0] + right.xy[0]) / 2.0, (left.xy[1] + right.xy[1]) / 2.0)
    if edge > 1e-6:
        axis = ((right.xy[0] - left.xy[0]) / edge, (right.xy[1] - left.xy[1]) / edge)
    else:
        angle = rng.uniform(0.0, 2.0 * math.pi)
        axis = (math.cos(angle), math.sin(angle))
    normal = (-axis[1], axis[0])
    height_min = max(half / math.tan(math.radians(35.0)), 0.75)
    height_max = math.sqrt(max(reach * reach - half * half, 0.0))
    tried: set[int] = set()
    if height_min < height_max:
        for _ in range(96):
            side = rng.choice((-1.0, 1.0))
            height = rng.uniform(height_min, height_max)
            lateral = rng.uniform(-0.6, 0.6) * half
            proposal = (
                mid[0] + normal[0] * height * side + axis[0] * lateral,
                mid[1] + normal[1] * height * side + axis[1] * lateral,
            )
            index = grid.nearest_index(proposal)
            if index in tried or not grid.is_free(index):
                continue
            tried.add(index)
            if admit(index):
                return True
    for _ in range(min(256, len(grid.free_cells))):
        index = grid.free_cells[rng.randrange(len(grid.free_cells))]
        if index in tried:
            continue
        tried.add(index)
        if admit(index):
            return True
    return False


IMAGINED_CAMERA_PROBE_RADIUS_M = 0.05
# How far past its own footprint the imagined camera may stand.  About one
# step: far enough to clear a wardrobe's depth, close enough that "stand at the
# wardrobe" still describes where the camera is.
IMAGINED_STATION_STANDOFF_M = 0.6
# A station around the side of the object is still an honest realisation of
# "stand at P and face Q" only while it preserves substantially the same
# heading as the object-centre reference frame.  This is deliberately the same
# size as the registered answer-boundary margin: a placement must not consume
# the safety margin that makes the downstream sector label decisive.
IMAGINED_STATION_MAX_HEADING_SHIFT_DEG = 15.0


@dataclass(frozen=True)
class ImaginedStationPlacement:
    """One deterministic physical realisation of an imagined object frame."""

    xy: tuple[float, float]
    method: str
    offset_m: float
    heading_shift_deg: float
    surface_standoff_m: float


def _camera_blocked(layout: SceneLayout, xy: tuple[float, float], height_m: float) -> bool:
    """Some obstacle occupies this point at camera height."""
    return any(
        obstacle.z_low <= height_m <= obstacle.z_high
        and point_in_rotated_rect(
            xy,
            obstacle.center_xy,
            (
                obstacle.half_extents_xy[0] + IMAGINED_CAMERA_PROBE_RADIUS_M,
                obstacle.half_extents_xy[1] + IMAGINED_CAMERA_PROBE_RADIUS_M,
            ),
            obstacle.yaw_deg,
        )
        for obstacle in layout.obstacles
    )


def _footprint_reach_m(
    layout: SceneLayout, entity_id: str, unit: tuple[float, float], fallback_m: float
) -> float:
    """How far the object's own footprint extends from its centre along ``unit``."""
    reach = 0.0
    for obstacle in layout.obstacles:
        if obstacle.entity_id != entity_id:
            continue
        angle = math.radians(obstacle.yaw_deg)
        local_x = unit[0] * math.cos(angle) + unit[1] * math.sin(angle)
        local_y = -unit[0] * math.sin(angle) + unit[1] * math.cos(angle)
        reach = max(
            reach,
            abs(local_x) * obstacle.half_extents_xy[0] + abs(local_y) * obstacle.half_extents_xy[1],
        )
    return reach or fallback_m


def _point_rotated_rect_distance_m(xy: tuple[float, float], obstacle: Obstacle) -> float:
    """Horizontal distance from a point to one oriented obstacle footprint."""
    angle = math.radians(obstacle.yaw_deg)
    dx = xy[0] - obstacle.center_xy[0]
    dy = xy[1] - obstacle.center_xy[1]
    local_x = dx * math.cos(angle) + dy * math.sin(angle)
    local_y = -dx * math.sin(angle) + dy * math.cos(angle)
    outside_x = max(abs(local_x) - obstacle.half_extents_xy[0], 0.0)
    outside_y = max(abs(local_y) - obstacle.half_extents_xy[1], 0.0)
    return math.hypot(outside_x, outside_y)


def _surface_standoff_m(
    layout: SceneLayout,
    entity_id: str,
    xy: tuple[float, float],
    fallback_radius_m: float,
) -> float:
    """Distance from ``xy`` to the entity's actual compound footprint."""
    own = tuple(obstacle for obstacle in layout.obstacles if obstacle.entity_id == entity_id)
    if own:
        return min(_point_rotated_rect_distance_m(xy, obstacle) for obstacle in own)
    return max(distance_m(xy, layout.object(entity_id).xy) - fallback_radius_m, 0.0)


def _footprint_search_radius_m(
    layout: SceneLayout, entity_id: str, fallback_radius_m: float
) -> float:
    """Bounding radius containing the entity footprint and allowed standoff."""
    centre = layout.object(entity_id).xy
    radii = [fallback_radius_m]
    for obstacle in layout.obstacles:
        if obstacle.entity_id != entity_id:
            continue
        radii.append(distance_m(centre, obstacle.center_xy) + math.hypot(*obstacle.half_extents_xy))
    return max(radii) + IMAGINED_STATION_STANDOFF_M


def _placement(
    xy: tuple[float, float],
    *,
    method: str,
    origin: tuple[float, float],
    facing_xy: tuple[float, float],
    surface_standoff_m: float,
) -> ImaginedStationPlacement:
    base_yaw = bearing_deg(origin, facing_xy)
    actual_yaw = bearing_deg(xy, facing_xy)
    return ImaginedStationPlacement(
        xy=xy,
        method=method,
        offset_m=distance_m(origin, xy),
        heading_shift_deg=abs(wrap_deg(actual_yaw - base_yaw)),
        surface_standoff_m=surface_standoff_m,
    )


@lru_cache(maxsize=65536)
def imagined_station_placement(
    layout: SceneLayout,
    viewpoint_name: str,
    facing_name: str,
    camera_height_m: float,
) -> ImaginedStationPlacement | None:
    """Where a person standing at ``viewpoint`` and looking at ``facing`` stands.

    The imagined viewpoint used to be the reference object's horizontal centre
    at eye height, which is only inhabitable for furniture shorter than a
    person.  Anything whose own vertical extent contains eye height - a
    wardrobe, a locker, a fridge, a door - put the camera inside itself, and the
    binding was dropped.  That is a limit of where the code put the camera, not
    of whether the question is answerable: you stand *at* the wardrobe and look
    out from it, not inside it.

    So when the centre is occupied, step out along the ray toward the object
    being faced until the camera is clear of the furniture and standing on
    walkable floor.  Pushing along that ray leaves the imagined heading
    numerically unchanged - centre, station and facing object stay collinear -
    so only the position moves and the answer is still derived at the pose the
    render will use.

    Returns ``None`` when no honest station exists: the object is walled in, or
    every clear point on its facing side is beyond the standoff, which would
    make "stand at this object" a false description of the camera.
    """
    viewpoint = layout.object(viewpoint_name)
    facing = layout.object(facing_name)
    origin = viewpoint.xy
    span = distance_m(origin, facing.xy)
    if span < 1e-6:
        return None
    if not _camera_blocked(layout, origin, camera_height_m):
        return _placement(
            origin,
            method="centre",
            origin=origin,
            facing_xy=facing.xy,
            surface_standoff_m=0.0,
        )

    unit = ((facing.xy[0] - origin[0]) / span, (facing.xy[1] - origin[1]) / span)
    limit = min(
        _footprint_reach_m(layout, viewpoint_name, unit, 0.5 * viewpoint.size_m)
        + IMAGINED_STATION_STANDOFF_M,
        0.5 * span,
    )
    grid = _occupancy_grid(layout)
    for step in range(1, int(limit / GRID_RESOLUTION_M) + 1):
        push = step * GRID_RESOLUTION_M
        xy = (origin[0] + unit[0] * push, origin[1] + unit[1] * push)
        if _camera_blocked(layout, xy, camera_height_m):
            continue
        if grid.is_free(grid.nearest_index(xy)):
            return _placement(
                xy,
                method="facing_ray",
                origin=origin,
                facing_xy=facing.xy,
                surface_standoff_m=_surface_standoff_m(
                    layout, viewpoint_name, xy, 0.5 * viewpoint.size_m
                ),
            )

    # The preferred ray can be occupied by a neighbouring cabinet even when a
    # person can stand immediately beside the reference object.  Search only
    # the facing-side half of the object's real footprint perimeter, rather
    # than accepting the first point on an unconstrained centre-radius ring.
    # The placement is independent of the target/answer and is ranked by how
    # faithfully it preserves the centre-defined reference direction.
    search_radius = _footprint_search_radius_m(layout, viewpoint_name, 0.5 * viewpoint.size_m)
    half_span = 0.5 * span
    max_offset = min(search_radius, half_span)
    min_ix, min_iy = grid.cell(grid.nearest_index((origin[0] - max_offset, origin[1] - max_offset)))
    max_ix, max_iy = grid.cell(grid.nearest_index((origin[0] + max_offset, origin[1] + max_offset)))
    candidates: list[tuple[tuple[float | int, ...], ImaginedStationPlacement]] = []
    for iy in range(min_iy, max_iy + 1):
        for ix in range(min_ix, max_ix + 1):
            index = grid.index(ix, iy)
            if not grid.is_free(index):
                continue
            xy = grid.world_xy(index)
            dx, dy = xy[0] - origin[0], xy[1] - origin[1]
            offset = math.hypot(dx, dy)
            if offset > half_span + 1e-9:
                continue
            # Do not stand behind the reference object relative to Q.  Side
            # placements remain available, which is precisely the false
            # rejection the perimeter fallback is intended to recover.
            if dx * unit[0] + dy * unit[1] < -1e-9:
                continue
            standoff = _surface_standoff_m(layout, viewpoint_name, xy, 0.5 * viewpoint.size_m)
            if standoff > IMAGINED_STATION_STANDOFF_M + 1e-9:
                continue
            if _camera_blocked(layout, xy, camera_height_m):
                continue
            candidate = _placement(
                xy,
                method="footprint_perimeter",
                origin=origin,
                facing_xy=facing.xy,
                surface_standoff_m=standoff,
            )
            if candidate.heading_shift_deg > IMAGINED_STATION_MAX_HEADING_SHIFT_DEG + 1e-9:
                continue
            candidates.append(
                (
                    (
                        round(candidate.heading_shift_deg, 6),
                        round(candidate.surface_standoff_m, 6),
                        round(candidate.offset_m, 6),
                        -len(grid.neighbours[index]),
                        round(xy[0], 6),
                        round(xy[1], 6),
                    ),
                    candidate,
                )
            )
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def imagined_station(
    layout: SceneLayout,
    viewpoint_name: str,
    facing_name: str,
    camera_height_m: float,
) -> tuple[float, float] | None:
    """Return the selected station coordinate, preserving the legacy API."""
    result = imagined_station_placement(layout, viewpoint_name, facing_name, camera_height_m)
    return None if result is None else result.xy


def chain_edge_station_exists(
    layout: SceneLayout,
    left_name: str,
    right_name: str,
    target_name: str,
    other_name: str,
    *,
    final_edge: bool,
) -> bool:
    """Some station realises this chain edge under the full motif constraints.

    Unlike ``pair_framable_station_exists`` this admits stations with the same
    judge the motif uses, including the never-covisible exclusion of the
    queried pair and the final-edge target sector margin, so a binding passing
    every edge here is one ``snapshot_landmarks`` can actually realise.
    """
    if not pair_framable_station_exists(layout, left_name, right_name):
        return False
    return _edge_station_exists(layout, left_name, right_name, target_name, other_name, final_edge)


@lru_cache(maxsize=65536)
def _edge_station_exists(
    layout: SceneLayout,
    left_name: str,
    right_name: str,
    target_name: str,
    other_name: str,
    final_edge: bool,
) -> bool:
    grid = _occupancy_grid(layout)
    rng = random.Random(
        f"edge_station|{left_name}|{right_name}|{target_name}|{other_name}|{final_edge}"
    )
    return bool(
        _edge_station_pool(
            layout,
            grid,
            left_name,
            right_name,
            target_name,
            other_name,
            rng,
            None,
            final_edge=final_edge,
            pool_size=1,
        )
    )


def _edge_station_pool(
    layout: SceneLayout,
    grid: _OccupancyGrid,
    left_name: str,
    right_name: str,
    target_name: str,
    other_name: str,
    rng: random.Random,
    component_id: int | None,
    *,
    final_edge: bool = False,
    pool_size: int = 8,
) -> tuple[tuple[int, float], ...]:
    """Ranked stations that frame one chain edge but never X and Y together.

    Stations are constructed geometrically first: near the perpendicular
    bisector of the edge, far enough out that both endpoints fit one
    90-degree frustum under the 70-degree separation cap, and inside both
    endpoints' clearly-visible distance bands.  Random free cells supplement
    the construction in cluttered scenes where the bisector strip is blocked.
    Ranking prefers the station whose worse endpoint has the largest apparent
    angular size.

    Admission judges every candidate with the checker's own visibility model
    at both yaw-jitter extremes, so an admitted station survives the chain
    evidence and never-covisible clauses instead of a looser angular proxy.
    ``final_edge`` marks the station that hosts the question frame: there the
    target must additionally clear the ego sector-margin requirement.
    """
    from .standards import STD_V1 as std

    left, right = layout.object(left_name), layout.object(right_name)
    target = layout.object(target_name)
    cap_left = _landmark_view_cap_m(left.size_m)
    cap_right = _landmark_view_cap_m(right.size_m)
    floor_left = _landmark_view_floor_m(left.size_m)
    floor_right = _landmark_view_floor_m(right.size_m)
    edge = distance_m(left.xy, right.xy)
    half = edge / 2.0
    reach = min(cap_left, cap_right)
    if half >= reach:
        return ()
    jitter = std.snapshot_yaw_jitter_deg
    required_target_margin = std.sector_margin_deg * std.search_tighten_factor + jitter

    def admit(index: int) -> tuple[float, int, float] | None:
        if component_id is not None and grid.component_ids[index] != component_id:
            return None
        xy = grid.world_xy(index)
        left_distance = distance_m(xy, left.xy)
        right_distance = distance_m(xy, right.xy)
        if not floor_left <= left_distance <= cap_left:
            return None
        if not floor_right <= right_distance <= cap_right:
            return None
        left_yaw = bearing_deg(xy, left.xy)
        separation = wrap_deg(bearing_deg(xy, right.xy) - left_yaw)
        if abs(separation) > 70.0:
            return None
        yaw = wrap_deg(left_yaw + separation / 2.0)
        if final_edge and (
            sector_margin_deg(azimuth_deg(xy, yaw, target.xy)) < required_target_margin
        ):
            return None
        probes = GeometrySceneView(
            layout,
            tuple(Pose2D(xy[0], xy[1], wrap_deg(yaw + delta)) for delta in (-jitter, 0.0, jitter)),
            std,
        )
        for frame in range(3):
            if probes.visibility(left_name, frame).tristate(std) is not True:
                return None
            if probes.visibility(right_name, frame).tristate(std) is not True:
                return None
            target_state = probes.visibility(target_name, frame).tristate(std)
            other_state = probes.visibility(other_name, frame).tristate(std)
            if target_state is not False and other_state is not False:
                return None
        score = min(left.size_m / left_distance, right.size_m / right_distance)
        return score, index, yaw

    admitted: dict[int, tuple[float, int, float]] = {}
    mid = ((left.xy[0] + right.xy[0]) / 2.0, (left.xy[1] + right.xy[1]) / 2.0)
    if edge > 1e-6:
        axis = ((right.xy[0] - left.xy[0]) / edge, (right.xy[1] - left.xy[1]) / edge)
    else:
        angle = rng.uniform(0.0, 2.0 * math.pi)
        axis = (math.cos(angle), math.sin(angle))
    normal = (-axis[1], axis[0])
    # A station on the bisector at height h sees the endpoints separated by
    # 2*atan(half/h); h >= half/tan(35 deg) keeps that separation under 70.
    height_min = max(half / math.tan(math.radians(35.0)), 0.75)
    height_max = math.sqrt(max(reach * reach - half * half, 0.0))
    if height_min < height_max:
        for _ in range(96):
            if len(admitted) >= pool_size * 2:
                break
            side = rng.choice((-1.0, 1.0))
            height = rng.uniform(height_min, height_max)
            lateral = rng.uniform(-0.6, 0.6) * half
            proposal = (
                mid[0] + normal[0] * height * side + axis[0] * lateral,
                mid[1] + normal[1] * height * side + axis[1] * lateral,
            )
            index = grid.nearest_index(proposal)
            if index in admitted or not grid.is_free(index):
                continue
            entry = admit(index)
            if entry is not None:
                admitted[index] = entry
    if len(admitted) < pool_size:
        candidates = (
            grid.component_cells[component_id] if component_id is not None else grid.free_cells
        )
        for _ in range(min(256, len(candidates))):
            if len(admitted) >= pool_size * 2:
                break
            index = candidates[rng.randrange(len(candidates))]
            if index in admitted:
                continue
            entry = admit(index)
            if entry is not None:
                admitted[index] = entry
    ranked = sorted(admitted.values(), key=lambda entry: entry[0], reverse=True)
    return tuple((index, yaw) for _, index, yaw in ranked[:pool_size])


def _landmark_stations(
    layout: SceneLayout, binding: dict[str, str], rng: random.Random
) -> tuple[tuple[int, float], ...] | None:
    """Backtracking station assignment over ranked per-edge candidate pools.

    The first chosen station fixes the walkable component; later edges only
    consider stations in that component, so consecutive stations always stay
    connectable.  Backtracking recovers from an edge whose pool is empty in
    the chosen component by revisiting earlier choices.
    """
    grid = _occupancy_grid(layout)
    chain = _landmark_chain(binding)
    edges = tuple(pairwise(chain))
    pools: dict[tuple[int, int | None], tuple[tuple[int, float], ...]] = {}

    def pool(edge_index: int, component_id: int | None) -> tuple[tuple[int, float], ...]:
        key = (edge_index, component_id)
        if key not in pools:
            left, right = edges[edge_index]
            pools[key] = _edge_station_pool(
                layout,
                grid,
                left,
                right,
                binding["target"],
                binding["other"],
                rng,
                component_id,
                final_edge=edge_index == len(edges) - 1,
            )
        return pools[key]

    budget = 128

    def extend(prefix: list[tuple[int, float]]) -> tuple[tuple[int, float], ...] | None:
        nonlocal budget
        if len(prefix) == len(edges):
            return tuple(prefix)
        component_id = grid.component_ids[prefix[-1][0]] if prefix else None
        stations = list(pool(len(prefix), component_id))
        rng.shuffle(stations)
        for station in stations:
            if budget <= 0:
                return None
            budget -= 1
            if prefix and _connect_cells(grid, prefix[-1][0], station[0]) is None:
                continue
            prefix.append(station)
            result = extend(prefix)
            if result is not None:
                return result
            prefix.pop()
        return None

    return extend([])


@motif("visit_landmarks")
def visit_landmarks(
    layout: SceneLayout, binding: dict[str, str], frame_count: int, rng: random.Random
) -> tuple[Pose2D, ...]:
    """Visit edge-wise co-visibility stations along an X-anchor-Y chain."""
    grid = _occupancy_grid(layout)
    stations = _landmark_stations(layout, binding, rng)
    if stations is None:
        raise MotifUnavailable("visit_landmarks:no_station_assignment")

    transition_count = len(stations) - 1
    reserved = 2 + 8 * transition_count
    nav_total = frame_count - reserved
    nav_intervals = [nav_total // transition_count] * transition_count
    for index in range(nav_total % transition_count):
        nav_intervals[index] += 1

    first_xy = grid.world_xy(stations[0][0])
    poses = [Pose2D(first_xy[0], first_xy[1], stations[0][1])] * 2
    current_yaw = stations[0][1]
    for transition, ((start_index, _), (end_index, station_yaw)) in enumerate(pairwise(stations)):
        cells = _connect_cells(grid, start_index, end_index)
        if cells is None:
            raise MotifUnavailable("visit_landmarks:station_route_disconnected")
        route = [grid.world_xy(index) for index in _simplify_route(grid, cells)]
        if len(route) == 1:
            walk = tuple(
                Pose2D(route[0][0], route[0][1], current_yaw)
                for _ in range(nav_intervals[transition] + 1)
            )
        else:
            walk = _walk_polyline(route, current_yaw, nav_intervals[transition] + 1)
        poses.extend(walk[1:])
        end_xy = grid.world_xy(end_index)
        turned = _turn_in_place(end_xy, walk[-1].yaw_deg, station_yaw, 6)
        poses.extend(turned)
        poses.extend([Pose2D(end_xy[0], end_xy[1], station_yaw)] * 2)
        current_yaw = station_yaw
    return tuple(poses)


@motif("snapshot_landmarks")
def snapshot_landmarks(
    layout: SceneLayout, binding: dict[str, str], frame_count: int, rng: random.Random
) -> tuple[Pose2D, ...]:
    """Teleport-cut snapshot pairs framing each chain edge in order.

    Stations need no walkable connection between them: the sequence is a set
    of static snapshots, so consecutive frames may jump across rooms.  Each
    station contributes ``snapshot_frames_per_station`` frames whose yaws
    carry small independent jitter, keeping within-station frames
    near-duplicates without being pixel-identical.
    """
    from .standards import STD_V1

    grid = _occupancy_grid(layout)
    chain = _landmark_chain(binding)
    edges = tuple(pairwise(chain))
    if frame_count < STD_V1.snapshot_frames_per_station * len(edges):
        raise MotifUnavailable("snapshot_landmarks:frame_budget_below_station_count")
    stations: list[tuple[int, float]] = []
    for edge_index, (left, right) in enumerate(edges):
        pool = _edge_station_pool(
            layout,
            grid,
            left,
            right,
            binding["target"],
            binding["other"],
            rng,
            None,
            final_edge=edge_index == len(edges) - 1,
        )
        if not pool:
            raise MotifUnavailable(f"snapshot_landmarks:no_station:{left}->{right}")
        stations.append(pool[rng.randrange(min(len(pool), 3))])
    counts = [frame_count // len(stations)] * len(stations)
    for index in range(frame_count - sum(counts)):
        counts[index] += 1
    jitter = STD_V1.snapshot_yaw_jitter_deg
    poses: list[Pose2D] = []
    for (index, yaw), count in zip(stations, counts, strict=True):
        xy = grid.world_xy(index)
        poses.extend(
            Pose2D(xy[0], xy[1], wrap_deg(yaw + rng.uniform(-jitter, jitter))) for _ in range(count)
        )
    return tuple(poses)
