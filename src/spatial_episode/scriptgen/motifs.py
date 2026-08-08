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
from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush
from itertools import pairwise

from .geometry import (
    bearing_deg,
    distance_m,
    point_in_rotated_rect,
    wrap_deg,
)
from .sceneview import Pose2D, SceneLayout, blocking_occluders

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
) -> list[tuple[float, float]]:
    grid = _occupancy_grid(layout)
    if not grid.free_cells:
        return [layout.walkable_min, layout.walkable_min]
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
        if len(route) >= 2:
            return route
    start_xy = grid.world_xy(start)
    return [start_xy, start_xy]


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
    camera_xy: tuple[float, float], target_xy: tuple[float, float], rng: random.Random
) -> float:
    """Balanced left/right/back final gaze with extra back acceptance mass."""
    turn_direction = rng.choice((-1.0, 1.0))
    desired_azimuth = turn_direction * rng.choice((100.0, 150.0))
    desired_azimuth += rng.uniform(-8.0, 8.0)
    return wrap_deg(bearing_deg(camera_xy, target_xy) - desired_azimuth)


def _straight_past_endpoints(
    layout: SceneLayout, target_xy: tuple[float, float], target_size_m: float, rng: random.Random
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Sample a collision-free straight line passing beside, never through, a target."""
    grid = _occupancy_grid(layout)
    lateral_offset = max(1.1, target_size_m / 2.0 + PROPOSAL_CLEARANCE_M + 0.25)
    for _ in range(96):
        angle = rng.uniform(-math.pi, math.pi)
        forward = (math.cos(angle), math.sin(angle))
        left = (-forward[1], forward[0])
        sign = rng.choice((-1.0, 1.0))
        reach = rng.uniform(2.2, 3.0)
        start_xy = (
            target_xy[0] - forward[0] * reach + left[0] * lateral_offset * sign,
            target_xy[1] - forward[1] * reach + left[1] * lateral_offset * sign,
        )
        end_xy = (
            target_xy[0] + forward[0] * reach + left[0] * lateral_offset * sign,
            target_xy[1] + forward[1] * reach + left[1] * lateral_offset * sign,
        )
        start, end = grid.nearest_index(start_xy), grid.nearest_index(end_xy)
        if not grid.is_free(start) or not grid.is_free(end):
            continue
        if grid.component_ids[start] != grid.component_ids[end]:
            continue
        if not _grid_line_clear(grid, start, end):
            continue
        return grid.world_xy(start), grid.world_xy(end)
    fallback = grid.world_xy(_sample_start(grid, target_xy, rng))
    return fallback, fallback


def _direct_multi_turn_route(
    layout: SceneLayout, target_xy: tuple[float, float], rng: random.Random
) -> tuple[tuple[float, float], tuple[float, float]]:
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
    fallback = grid.world_xy(_sample_start(grid, target_xy, rng))
    return fallback, fallback


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
    look_frames = max(2, frame_count // 4)
    waypoints = _propose_route(layout, target.xy, frame_count - look_frames, rng)
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
    final_yaw = _target_final_yaw(start, target.xy, rng)
    poses = [Pose2D(start[0], start[1], initial_yaw)] * 2
    poses.extend(_turn_in_place(start, initial_yaw, final_yaw, frame_count - 2))
    return tuple(poses)


@motif("walk_straight_past")
def walk_straight_past(
    layout: SceneLayout, binding: dict[str, str], frame_count: int, rng: random.Random
) -> tuple[Pose2D, ...]:
    """T3: translate past a side-front target with exactly constant heading."""
    target = layout.object(binding["target"])
    start, end = _straight_past_endpoints(layout, target.xy, target.size_m, rng)
    heading = bearing_deg(start, end) if start != end else bearing_deg(start, target.xy)
    return tuple(
        Pose2D(_lerp(start[0], end[0], t), _lerp(start[1], end[1], t), heading)
        for t in (index / (frame_count - 1) for index in range(frame_count))
    )


@motif("walk_multi_turn")
def walk_multi_turn(
    layout: SceneLayout, binding: dict[str, str], frame_count: int, rng: random.Random
) -> tuple[Pose2D, ...]:
    """T4: two or three separated rotations with straight motion between them."""
    target = layout.object(binding["target"])
    start, end = _direct_multi_turn_route(layout, target.xy, rng)
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


def _occluded_endpoint(
    layout: SceneLayout, target_xy: tuple[float, float], rng: random.Random
) -> tuple[float, float] | None:
    grid = _occupancy_grid(layout)
    candidates: list[tuple[tuple[float, float], float]] = []
    if layout.occlusion_obstacles:
        candidates.extend(
            (obstacle.center_xy, math.hypot(*obstacle.half_extents_xy))
            for obstacle in layout.occlusion_obstacles
        )
    else:
        candidates.extend(
            (
                ((low[0] + high[0]) / 2.0, (low[1] + high[1]) / 2.0),
                math.hypot((high[0] - low[0]) / 2.0, (high[1] - low[1]) / 2.0),
            )
            for low, high in layout.occluders
        )
    rng.shuffle(candidates)
    for center, extent in candidates:
        length = distance_m(target_xy, center)
        if length < 0.5:
            continue
        unit = ((center[0] - target_xy[0]) / length, (center[1] - target_xy[1]) / length)
        for clearance in (0.8, 1.1, 1.4):
            proposed = (
                center[0] + unit[0] * (extent + clearance),
                center[1] + unit[1] * (extent + clearance),
            )
            index = grid.nearest_index(proposed)
            if grid.is_free(index):
                return grid.world_xy(index)
    return None


@motif("walk_to_occlusion")
def walk_to_occlusion(
    layout: SceneLayout, binding: dict[str, str], frame_count: int, rng: random.Random
) -> tuple[Pose2D, ...]:
    """T5: finish facing a target from behind an eye-height occluder."""
    target = layout.object(binding["target"])
    grid = _occupancy_grid(layout)
    endpoint = _occluded_endpoint(layout, target.xy, rng)
    if endpoint is None:
        start = grid.world_xy(_sample_start(grid, target.xy, rng))
        return tuple(
            Pose2D(start[0], start[1], bearing_deg(start, target.xy)) for _ in range(frame_count)
        )

    end_index = grid.nearest_index(endpoint)
    route: list[tuple[float, float]] | None = None
    for _ in range(64):
        start_index = _sample_start(grid, target.xy, rng)
        start_xy = grid.world_xy(start_index)
        if grid.component_ids[start_index] != grid.component_ids[end_index]:
            continue
        if blocking_occluders(layout, start_xy, target):
            continue
        cells = _connect_cells(grid, start_index, end_index)
        if cells is None:
            continue
        route = [grid.world_xy(index) for index in _simplify_route(grid, cells)]
        break
    if route is None:
        route = [endpoint, endpoint]

    look_frames = 5
    initial_yaw = bearing_deg(route[0], target.xy)
    walk = list(_walk_polyline(route, initial_yaw, frame_count - look_frames))
    end = walk[-1]
    final_yaw = bearing_deg(end.xy, target.xy)
    walk.extend(_turn_in_place(end.xy, end.yaw_deg, final_yaw, look_frames))
    return tuple(walk)


def _survey_station(
    layout: SceneLayout, landmark_names: tuple[str, ...], rng: random.Random
) -> tuple[float, float]:
    """Free point with unobstructed, sufficiently close rays to all landmarks."""
    grid = _occupancy_grid(layout)
    candidates = list(grid.free_cells)
    rng.shuffle(candidates)
    landmarks = tuple(layout.object(name) for name in landmark_names)
    for index in candidates[:512]:
        xy = grid.world_xy(index)
        if all(not blocking_occluders(layout, xy, landmark) for landmark in landmarks):
            return xy
    return grid.world_xy(candidates[0])


@motif("survey")
def survey(
    layout: SceneLayout, binding: dict[str, str], frame_count: int, rng: random.Random
) -> tuple[Pose2D, ...]:
    """Stationary, rate-limited panorama exposing every bound landmark."""
    names = tuple(
        binding[name]
        for name in ("viewpoint", "facing", "target")
        if name in binding
    )
    station = _survey_station(layout, names, rng)
    start_yaw = rng.uniform(-180.0, 180.0)
    direction = rng.choice((-1.0, 1.0))
    step = direction * 360.0 / (frame_count - 1)
    return tuple(
        Pose2D(station[0], station[1], wrap_deg(start_yaw + index * step))
        for index in range(frame_count)
    )


def _landmark_chain(binding: dict[str, str]) -> tuple[str, ...]:
    anchors = tuple(
        binding[name]
        for name in sorted(
            (name for name in binding if name.startswith("anchor")),
            key=lambda name: int(name.removeprefix("anchor")),
        )
    )
    return (binding["target"], *anchors, binding["other"])


def _pair_station(
    layout: SceneLayout,
    grid: _OccupancyGrid,
    left_name: str,
    right_name: str,
    target_name: str,
    other_name: str,
    rng: random.Random,
    component_id: int | None,
) -> tuple[int, float] | None:
    """Free observation station that frames one chain edge but not X and Y."""
    left, right = layout.object(left_name), layout.object(right_name)
    target, other = layout.object(target_name), layout.object(other_name)
    candidates = (
        grid.component_cells[component_id]
        if component_id is not None
        else grid.free_cells
    )
    for _ in range(min(512, len(candidates))):
        index = candidates[rng.randrange(len(candidates))]
        xy = grid.world_xy(index)
        if blocking_occluders(layout, xy, left) or blocking_occluders(layout, xy, right):
            continue
        left_yaw = bearing_deg(xy, left.xy)
        separation = wrap_deg(bearing_deg(xy, right.xy) - left_yaw)
        if abs(separation) > 70.0:
            continue
        yaw = wrap_deg(left_yaw + separation / 2.0)
        target_in_view = (
            abs(wrap_deg(bearing_deg(xy, target.xy) - yaw)) <= 45.0
            and not blocking_occluders(layout, xy, target)
        )
        other_in_view = (
            abs(wrap_deg(bearing_deg(xy, other.xy) - yaw)) <= 45.0
            and not blocking_occluders(layout, xy, other)
        )
        if target_in_view and other_in_view:
            continue
        return index, yaw
    return None


def _landmark_stations(
    layout: SceneLayout, binding: dict[str, str], rng: random.Random
) -> tuple[tuple[int, float], ...] | None:
    grid = _occupancy_grid(layout)
    chain = _landmark_chain(binding)
    for _ in range(8):
        stations: list[tuple[int, float]] = []
        component_id: int | None = None
        for left, right in pairwise(chain):
            station = _pair_station(
                layout,
                grid,
                left,
                right,
                binding["target"],
                binding["other"],
                rng,
                component_id,
            )
            if station is None:
                break
            stations.append(station)
            component_id = grid.component_ids[station[0]]
        if len(stations) != len(chain) - 1:
            continue
        if all(
            _connect_cells(grid, left[0], right[0]) is not None
            for left, right in pairwise(stations)
        ):
            return tuple(stations)
    return None


@motif("visit_landmarks")
def visit_landmarks(
    layout: SceneLayout, binding: dict[str, str], frame_count: int, rng: random.Random
) -> tuple[Pose2D, ...]:
    """Visit edge-wise co-visibility stations along an X-anchor-Y chain."""
    grid = _occupancy_grid(layout)
    stations = _landmark_stations(layout, binding, rng)
    if stations is None:
        fallback = grid.world_xy(grid.free_cells[0])
        return tuple(Pose2D(fallback[0], fallback[1], 0.0) for _ in range(frame_count))

    transition_count = len(stations) - 1
    reserved = 2 + 8 * transition_count
    nav_total = frame_count - reserved
    nav_intervals = [nav_total // transition_count] * transition_count
    for index in range(nav_total % transition_count):
        nav_intervals[index] += 1

    first_xy = grid.world_xy(stations[0][0])
    poses = [Pose2D(first_xy[0], first_xy[1], stations[0][1])] * 2
    current_yaw = stations[0][1]
    for transition, ((start_index, _), (end_index, station_yaw)) in enumerate(
        pairwise(stations)
    ):
        cells = _connect_cells(grid, start_index, end_index)
        if cells is None:
            return tuple(poses[-1] for _ in range(frame_count))
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
