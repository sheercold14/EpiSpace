"""Free-space evidence coverage for existence-sufficiency questions.

Coverage is geometry truth, not a claim that an RGB model saw every object.
Each trajectory frustum is rasterised over the same conservative 5 cm free
grid used by trajectory proposals, with the scene view's ray blockers cutting
off cells behind eye-height obstacles.  The report also measures the largest
disk that can fit in any connected unseen region; a high scalar ratio alone
must never license an absence claim while one room-sized blind spot remains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush

from .geometry import azimuth_deg, distance_m
from .motifs import GRID_RESOLUTION_M, proposal_occupancy_grid
from .sceneview import Pose2D, SceneLayout, SceneObject, SceneView, blocking_occluders
from .standards import CompileStandard


@dataclass(frozen=True)
class CoverageReport:
    """Auditable union of observed free cells and residual hiding space."""

    covered_cell_count: int
    total_free_cell_count: int
    uncovered_cell_count: int
    uncovered_component_count: int
    largest_uncovered_component_cells: int
    coverage_ratio: float
    max_hidden_diameter_m: float


def measure_coverage(
    view: SceneView,
    std: CompileStandard,
    *,
    frames: list[int] | range,
) -> CoverageReport:
    """Rasterise the union of declared frame frustums over traversable space."""
    # Frustum coverage is a set property. Canonicalising also makes permute and
    # delay variants share the same cached world-truth computation.
    poses = tuple(
        sorted(
            {view.camera_pose(frame) for frame in frames},
            key=lambda pose: (pose.x, pose.y, pose.yaw_deg),
        )
    )
    return _measure_coverage(view.layout, poses, std)


@lru_cache(maxsize=128)
def _measure_coverage(
    layout: SceneLayout,
    poses: tuple[Pose2D, ...],
    std: CompileStandard,
) -> CoverageReport:
    grid = proposal_occupancy_grid(layout)
    if not grid.free_cells:
        raise ValueError(f"scene {layout.scene_id!r} has no traversable coverage cells")

    covered: set[int] = set()
    for index in grid.free_cells:
        xy = grid.world_xy(index)
        cell = SceneObject(
            name=f"coverage_cell_{index}",
            category="__coverage_cell__",
            xy=xy,
            size_m=GRID_RESOLUTION_M,
            uid=f"coverage_cell_{index}",
        )
        for pose in poses:
            distance = distance_m(pose.xy, xy)
            if distance <= GRID_RESOLUTION_M / 2.0:
                covered.add(index)
                break
            if distance > std.max_view_distance_m:
                continue
            if abs(azimuth_deg(pose.xy, pose.yaw_deg, xy)) > std.fov_half_angle_deg:
                continue
            if not blocking_occluders(layout, pose.xy, cell):
                covered.add(index)
                break

    unseen = set(grid.free_cells) - covered
    component_count, largest_component = _unseen_components(grid, unseen)
    max_hidden_diameter = _max_hidden_diameter_m(grid, unseen)
    total = len(grid.free_cells)
    return CoverageReport(
        covered_cell_count=len(covered),
        total_free_cell_count=total,
        uncovered_cell_count=len(unseen),
        uncovered_component_count=component_count,
        largest_uncovered_component_cells=largest_component,
        coverage_ratio=len(covered) / total,
        max_hidden_diameter_m=max_hidden_diameter,
    )


def _unseen_components(grid, unseen: set[int]) -> tuple[int, int]:
    remaining = set(unseen)
    component_count = 0
    largest = 0
    while remaining:
        component_count += 1
        root = remaining.pop()
        stack = [root]
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbour, _ in grid.neighbours[current]:
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)
        largest = max(largest, size)
    return component_count, largest


def _max_hidden_diameter_m(grid, unseen: set[int]) -> float:
    """Approximate the largest inscribed disk with an 8-neighbour distance map."""
    if not unseen:
        return 0.0

    neighbour_offsets = (
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    )
    distances: dict[int, float] = {}
    frontier: list[tuple[float, int]] = []
    for index in unseen:
        x, y = grid.cell(index)
        boundary = False
        for dx, dy in neighbour_offsets:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                boundary = True
                break
            if grid.index(nx, ny) not in unseen:
                boundary = True
                break
        if boundary:
            distances[index] = 0.5
            heappush(frontier, (0.5, index))

    while frontier:
        current_distance, current = heappop(frontier)
        if current_distance > distances[current]:
            continue
        for neighbour, step_cost in grid.neighbours[current]:
            if neighbour not in unseen:
                continue
            candidate = current_distance + step_cost
            if candidate >= distances.get(neighbour, math.inf):
                continue
            distances[neighbour] = candidate
            heappush(frontier, (candidate, neighbour))

    radius_cells = max(distances.values(), default=0.0)
    return 2.0 * radius_cells * GRID_RESOLUTION_M
