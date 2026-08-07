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
                if point_in_rotated_rect(
                    point, obstacle.center_xy, expanded, obstacle.yaw_deg
                ):
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
    return _OccupancyGrid(
        origin_xy=(min_x, min_y),
        width=width,
        height=height,
        blocked=bytes(blocked),
        free_cells=free_cells,
        neighbours=tuple(neighbours),
    )


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
        if move_x and move_y and (
            grid.blocked[previous_y * grid.width + x]
            or grid.blocked[y * grid.width + previous_x]
        ):
            return False


def _simplify_route(
    grid: _OccupancyGrid, cells: tuple[int, ...]
) -> list[int]:
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


def _sample_start(
    grid: _OccupancyGrid, target_xy: tuple[float, float], rng: random.Random
) -> int:
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
    diagonal = math.hypot(grid.width, grid.height) * GRID_RESOLUTION_M
    desired = max(2.0, diagonal * 0.35)
    best, best_distance = start, -1.0
    for _ in range(64):
        candidate = grid.free_cells[rng.randrange(len(grid.free_cells))]
        distance = distance_m(start_xy, grid.world_xy(candidate))
        if distance >= desired:
            return candidate
        if distance > best_distance:
            best, best_distance = candidate, distance
    return best


def _connect_cells(
    grid: _OccupancyGrid, start: int, end: int
) -> tuple[int, ...] | None:
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
    (the std.v2 ego-motion contract). Partial-visibility frames during the
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
        yaw = wrap_deg(
            yaw + max(-MAX_TURN_PER_FRAME_DEG, min(MAX_TURN_PER_FRAME_DEG, delta))
        )
        poses.append(Pose2D(position[0], position[1], yaw))
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
    look_frames = max(2, frame_count // 4)
    waypoints = _propose_route(layout, target.xy, frame_count - look_frames, rng)
    initial_yaw = bearing_deg(waypoints[0], target.xy)
    walk = list(_walk_polyline(waypoints, initial_yaw, frame_count - look_frames))

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
