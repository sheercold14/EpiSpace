"""Canonical path and camera geometry independent of OmniGibson."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Any
from uuid import UUID, uuid5

Vec3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]

PROJECT_NAMESPACE = UUID("2b74debb-f2c4-5ca6-9563-fbe800f3f724")


def stable_scene_id(source_version: str, source_scene_id: str) -> str:
    value = f"scene:omnigibson:{source_version}:{source_scene_id}"
    return str(uuid5(PROJECT_NAMESPACE, value))


def distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(left[i]) - float(right[i])) ** 2 for i in range(3)))


def normalize_quaternion(values: Sequence[float]) -> Quaternion:
    norm = math.sqrt(sum(float(value) ** 2 for value in values))
    if norm <= 1e-12:
        raise ValueError("quaternion cannot be zero")
    return tuple(float(value) / norm for value in values)  # type: ignore[return-value]


def quaternion_multiply(left: Sequence[float], right: Sequence[float]) -> Quaternion:
    """Hamilton product for xyzw quaternions: apply right, then left."""
    lx, ly, lz, lw = (float(value) for value in left)
    rx, ry, rz, rw = (float(value) for value in right)
    return normalize_quaternion(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )
    )


def yaw_quaternion(start: Sequence[float], target: Sequence[float]) -> Quaternion:
    dx = float(target[0]) - float(start[0])
    dy = float(target[1]) - float(start[1])
    if math.hypot(dx, dy) <= 1e-9:
        raise ValueError("cannot orient along a zero-length segment")
    yaw = math.atan2(-dx, dy)
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def clockwise_yaw_quaternion(yaw_deg: float) -> Quaternion:
    """Rotate canonical agent-forward (+Y) clockwise by ``yaw_deg``."""

    radians = math.radians(float(yaw_deg))
    return (0.0, 0.0, -math.sin(radians / 2.0), math.cos(radians / 2.0))


def yaw_pitch_quaternion(yaw_deg: float, pitch_down_deg: float) -> Quaternion:
    """Return agent orientation with clockwise yaw and local downward pitch."""

    if not 0.0 <= float(pitch_down_deg) < 90.0:
        raise ValueError("downward pitch must be in [0, 90) degrees")
    pitch = math.radians(-float(pitch_down_deg))
    local_pitch = (math.sin(pitch / 2.0), 0.0, 0.0, math.cos(pitch / 2.0))
    return quaternion_multiply(clockwise_yaw_quaternion(yaw_deg), local_pitch)


def omnigibson_camera_quaternion(agent_yaw_xyzw: Sequence[float]) -> Quaternion:
    """Map canonical camera (+X right,+Y forward,+Z up) to USD camera (-Z optical)."""
    root_half = math.sqrt(0.5)
    camera_from_canonical = (root_half, 0.0, 0.0, root_half)
    return quaternion_multiply(agent_yaw_xyzw, camera_from_canonical)


def planar_deltas_in_yaw_frame(
    subject: Sequence[float],
    reference: Sequence[float],
    world_from_frame_xyzw: Sequence[float],
) -> tuple[float, float]:
    """Return subject-reference displacement as frame-right and frame-forward."""

    _, _, z, w = normalize_quaternion(world_from_frame_xyzw)
    cosine = 1.0 - 2.0 * z * z
    sine = 2.0 * z * w
    dx = float(subject[0]) - float(reference[0])
    dy = float(subject[1]) - float(reference[1])
    return dx * cosine + dy * sine, dx * (-sine) + dy * cosine


def planar_delta_to_world(
    right: float,
    forward: float,
    world_from_frame_xyzw: Sequence[float],
) -> tuple[float, float]:
    """Map a planar right/forward displacement back into world X/Y."""

    _, _, z, w = normalize_quaternion(world_from_frame_xyzw)
    cosine = 1.0 - 2.0 * z * z
    sine = 2.0 * z * w
    return right * cosine - forward * sine, right * sine + forward * cosine


def planar_aabb_half_extents_in_yaw_frame(
    half_extents_xy: Sequence[float], world_from_frame_xyzw: Sequence[float]
) -> tuple[float, float]:
    """Project a world-axis-aligned box onto frame-right and frame-forward."""

    _, _, z, w = normalize_quaternion(world_from_frame_xyzw)
    cosine = 1.0 - 2.0 * z * z
    sine = 2.0 * z * w
    half_x, half_y = float(half_extents_xy[0]), float(half_extents_xy[1])
    return (
        abs(cosine) * half_x + abs(sine) * half_y,
        abs(sine) * half_x + abs(cosine) * half_y,
    )


def dominant_planar_relation(
    subject: Sequence[float],
    reference: Sequence[float],
    world_from_frame_xyzw: Sequence[float],
) -> tuple[str, str, float, float, float]:
    """Classify a relation in an explicit observer frame with dominance margin."""

    right, forward = planar_deltas_in_yaw_frame(
        subject, reference, world_from_frame_xyzw
    )
    if abs(right) >= abs(forward):
        relation = "right_of" if right > 0 else "left_of"
        axis = "right"
        margin = abs(right) - abs(forward)
    else:
        relation = "in_front_of" if forward > 0 else "behind"
        axis = "forward"
        margin = abs(forward) - abs(right)
    return relation, axis, margin, right, forward


def resample_polyline(points: Sequence[Sequence[float]], count: int) -> tuple[list[Vec3], float]:
    """Return arc-length samples that lie on the source geodesic polyline."""
    if count < 2:
        raise ValueError("sample count must be at least two")
    vertices: list[Vec3] = [tuple(float(value) for value in point) for point in points]  # type: ignore[misc]
    if len(vertices) < 2:
        raise ValueError("polyline must contain at least two points")
    cumulative = [0.0]
    for previous, current in pairwise(vertices):
        cumulative.append(cumulative[-1] + distance(previous, current))
    total = cumulative[-1]
    if total <= 1e-9:
        raise ValueError("polyline cannot have zero length")

    samples: list[Vec3] = []
    segment = 0
    for index in range(count):
        target = total * index / (count - 1)
        while segment + 1 < len(cumulative) - 1 and cumulative[segment + 1] < target:
            segment += 1
        start_distance = cumulative[segment]
        end_distance = cumulative[segment + 1]
        ratio = (target - start_distance) / (end_distance - start_distance)
        start, end = vertices[segment], vertices[segment + 1]
        samples.append(
            tuple(start[axis] + ratio * (end[axis] - start[axis]) for axis in range(3))
        )
    return samples, total


def build_closed_trajectory(
    *,
    scene_id: str,
    recipe_id: str,
    seed: int,
    source_points: Sequence[Sequence[float]],
    outbound_view_count: int,
    floor_area_m2: float,
    backend_version: str,
) -> dict[str, Any]:
    outbound, geodesic = resample_polyline(source_points, outbound_view_count)
    forward = [
        yaw_quaternion(outbound[index], outbound[index + 1])
        for index in range(len(outbound) - 1)
    ]
    forward.append(forward[-1])
    route = outbound + list(reversed(outbound[:-1]))
    views = []
    cumulative = 0.0
    step_distance = geodesic / (outbound_view_count - 1)
    for step, point in enumerate(route):
        if step == 0:
            role = "initial"
            delta = 0.0
            rotation = forward[0]
        elif step < outbound_view_count - 1:
            role = "explore"
            delta = step_distance
            rotation = forward[step]
        elif step == outbound_view_count - 1:
            role = "turnaround"
            delta = step_distance
            rotation = yaw_quaternion(route[step], route[step + 1])
        elif step == len(route) - 1:
            role = "loop_closure"
            delta = step_distance
            rotation = forward[0]
        else:
            role = "revisit"
            delta = step_distance
            rotation = yaw_quaternion(route[step], route[step + 1])
        cumulative += delta
        view_id = f"view-{step:03d}"
        views.append(
            {
                "step": step,
                "view_id": view_id,
                "role": role,
                "world_from_agent": {
                    "parent_frame": "world",
                    "child_frame": f"agent:{view_id}",
                    "translation_m": [round(value, 6) for value in point],
                    "rotation_xyzw": [round(value, 9) for value in rotation],
                    "convention": "active_child_to_parent",
                },
                "distance_from_previous_m": round(delta, 6),
                "cumulative_distance_m": round(cumulative, 6),
                "backend_position_m": [round(value, 6) for value in point],
            }
        )
    return {
        "schema_version": "trajectory_plan.v1",
        "scene_id": scene_id,
        "recipe_id": recipe_id,
        "seed": seed,
        "backend_name": "omnigibson",
        "backend_version": backend_version,
        "backend_frame": "omnigibson_world_x_right_y_forward_z_up",
        "canonical_from_backend": {
            "parent_frame": "world",
            "child_frame": "omnigibson_world",
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "convention": "active_child_to_parent",
        },
        "primary_island": 0,
        "primary_island_area_m2": round(float(floor_area_m2), 6),
        "outbound_geodesic_distance_m": round(geodesic, 6),
        "closed_loop": True,
        "trajectory_class": "T1",
        "views": views,
    }


def build_rotation_station_trajectory(
    *,
    scene_id: str,
    recipe_id: str,
    seed: int,
    station_position: Sequence[float],
    station_id: str,
    yaw_step_deg: float,
    view_count: int,
    floor_area_m2: float,
    backend_version: str,
) -> dict[str, object]:
    """Build a zero-baseline T3 panorama with one exact camera position."""

    if len(station_position) != 3:
        raise ValueError("rotation station position must be a three-vector")
    if view_count < 4:
        raise ValueError("rotation station requires at least four views")
    if not math.isclose(yaw_step_deg * view_count, 360.0, abs_tol=1e-6):
        raise ValueError("rotation station yaws must cover exactly 360 degrees")
    position = tuple(round(float(value), 6) for value in station_position)
    yaws = [round(index * float(yaw_step_deg), 6) for index in range(view_count)]
    views = []
    for step, yaw_deg in enumerate(yaws):
        view_id = f"view-{step:03d}"
        rotation = clockwise_yaw_quaternion(yaw_deg)
        views.append(
            {
                "step": step,
                "view_id": view_id,
                "role": "initial" if step == 0 else "rotation_station",
                "world_from_agent": {
                    "parent_frame": "world",
                    "child_frame": f"agent:{view_id}",
                    "translation_m": position,
                    "rotation_xyzw": tuple(round(value, 9) for value in rotation),
                    "convention": "active_child_to_parent",
                },
                "distance_from_previous_m": 0.0,
                "cumulative_distance_m": 0.0,
                "backend_position_m": position,
            }
        )
    return {
        "schema_version": "trajectory_plan.v1",
        "scene_id": scene_id,
        "recipe_id": recipe_id,
        "seed": seed,
        "backend_name": "omnigibson",
        "backend_version": backend_version,
        "backend_frame": "omnigibson_world_x_right_y_forward_z_up",
        "canonical_from_backend": {
            "parent_frame": "world",
            "child_frame": "omnigibson_world",
            "translation_m": (0.0, 0.0, 0.0),
            "rotation_xyzw": (0.0, 0.0, 0.0, 1.0),
            "convention": "active_child_to_parent",
        },
        "primary_island": 0,
        "primary_island_area_m2": round(float(floor_area_m2), 3),
        "outbound_geodesic_distance_m": 0.0,
        "closed_loop": False,
        "trajectory_class": "T3",
        "station_id": station_id,
        "yaw_sequence_deg": yaws,
        "depth_questions_forbidden": True,
        "views": views,
    }


def build_among5_trajectory(
    *,
    scene_id: str,
    recipe_id: str,
    seed: int,
    camera_positions: Sequence[Sequence[float]],
    focus_position: Sequence[float],
    camera_pitch_down_deg: float,
    floor_area_m2: float,
    backend_version: str,
) -> dict[str, object]:
    """Build four controlled views that look at one exact Among-5 layout.

    Layout truth intentionally lives in ``trajectory_selection.json`` rather
    than the model-facing trajectory contract.  The trajectory only carries
    camera evidence and therefore remains valid under ``trajectory_plan.v1``.
    """

    if len(camera_positions) != 4:
        raise ValueError("Among-5 requires exactly four camera positions")
    positions = [tuple(float(value) for value in point) for point in camera_positions]
    if any(len(point) != 3 for point in positions):
        raise ValueError("Among-5 camera positions must be three-vectors")
    focus = tuple(float(value) for value in focus_position)
    if len(focus) != 3:
        raise ValueError("Among-5 focus position must be a three-vector")
    if not 0.0 <= float(camera_pitch_down_deg) <= 60.0:
        raise ValueError("Among-5 camera pitch must be between 0 and 60 degrees")

    views = []
    cumulative = 0.0
    for step, point in enumerate(positions):
        delta = 0.0 if step == 0 else distance(positions[step - 1], point)
        cumulative += delta
        dx = focus[0] - point[0]
        dy = focus[1] - point[1]
        if math.hypot(dx, dy) <= 1e-9:
            raise ValueError("Among-5 camera cannot coincide with its focus")
        clockwise_yaw_deg = math.degrees(math.atan2(dx, dy))
        rotation = yaw_pitch_quaternion(clockwise_yaw_deg, camera_pitch_down_deg)
        view_id = f"view-{step:03d}"
        views.append(
            {
                "step": step,
                "view_id": view_id,
                "role": "initial" if step == 0 else "orbit",
                "world_from_agent": {
                    "parent_frame": "world",
                    "child_frame": f"agent:{view_id}",
                    "translation_m": tuple(round(value, 6) for value in point),
                    "rotation_xyzw": tuple(round(value, 9) for value in rotation),
                    "convention": "active_child_to_parent",
                },
                "distance_from_previous_m": round(delta, 6),
                "cumulative_distance_m": round(cumulative, 6),
                "backend_position_m": tuple(round(value, 6) for value in point),
            }
        )
    return {
        "schema_version": "trajectory_plan.v1",
        "scene_id": scene_id,
        "recipe_id": recipe_id,
        "seed": seed,
        "backend_name": "omnigibson",
        "backend_version": backend_version,
        "backend_frame": "omnigibson_world_x_right_y_forward_z_up",
        "canonical_from_backend": {
            "parent_frame": "world",
            "child_frame": "omnigibson_world",
            "translation_m": (0.0, 0.0, 0.0),
            "rotation_xyzw": (0.0, 0.0, 0.0, 1.0),
            "convention": "active_child_to_parent",
        },
        "primary_island": 0,
        "primary_island_area_m2": round(float(floor_area_m2), 3),
        "outbound_geodesic_distance_m": round(cumulative, 6),
        "closed_loop": False,
        "trajectory_class": "T2",
        "views": views,
    }


def build_object_orbit_trajectory(
    *,
    scene_id: str,
    recipe_id: str,
    seed: int,
    orbit_positions: Sequence[Sequence[float]],
    focus_position: Sequence[float],
    focus_entity_id: str,
    azimuth_deg_per_view: Sequence[float],
    floor_area_m2: float,
    backend_version: str,
) -> dict[str, object]:
    """Build a T4 object-centred orbit from certified traversable stations."""

    if len(orbit_positions) < 6 or len(orbit_positions) % 2:
        raise ValueError("object orbit requires an even number of at least six views")
    if len(orbit_positions) != len(azimuth_deg_per_view):
        raise ValueError("orbit positions and azimuths must align")
    focus = tuple(float(value) for value in focus_position)
    if len(focus) != 3:
        raise ValueError("focus position must be a three-vector")
    positions = [tuple(float(value) for value in point) for point in orbit_positions]
    radii = [math.hypot(point[0] - focus[0], point[1] - focus[1]) for point in positions]
    if min(radii) <= 0.0:
        raise ValueError("orbit camera cannot coincide with the focus")
    views = []
    cumulative = 0.0
    for step, point in enumerate(positions):
        delta = 0.0 if step == 0 else distance(positions[step - 1], point)
        cumulative += delta
        view_id = f"view-{step:03d}"
        rotation = yaw_quaternion(point, focus)
        views.append(
            {
                "step": step,
                "view_id": view_id,
                "role": "initial" if step == 0 else "orbit",
                "world_from_agent": {
                    "parent_frame": "world",
                    "child_frame": f"agent:{view_id}",
                    "translation_m": tuple(round(value, 6) for value in point),
                    "rotation_xyzw": tuple(round(value, 9) for value in rotation),
                    "convention": "active_child_to_parent",
                },
                "distance_from_previous_m": round(delta, 6),
                "cumulative_distance_m": round(cumulative, 6),
                "backend_position_m": tuple(round(value, 6) for value in point),
            }
        )
    return {
        "schema_version": "trajectory_plan.v1",
        "scene_id": scene_id,
        "recipe_id": recipe_id,
        "seed": seed,
        "backend_name": "omnigibson",
        "backend_version": backend_version,
        "backend_frame": "omnigibson_world_x_right_y_forward_z_up",
        "canonical_from_backend": {
            "parent_frame": "world",
            "child_frame": "omnigibson_world",
            "translation_m": (0.0, 0.0, 0.0),
            "rotation_xyzw": (0.0, 0.0, 0.0, 1.0),
            "convention": "active_child_to_parent",
        },
        "primary_island": 0,
        "primary_island_area_m2": round(float(floor_area_m2), 3),
        "outbound_geodesic_distance_m": round(cumulative, 6),
        "closed_loop": False,
        "trajectory_class": "T4",
        "focus_entity_id": focus_entity_id,
        "azimuth_deg_per_view": [round(float(value) % 360.0, 6) for value in azimuth_deg_per_view],
        "radius_m": round(float(sum(radii) / len(radii)), 6),
        # Current scene IR deliberately does not claim category-level canonical fronts.
        "orientation_questions_allowed": False,
        "views": views,
    }


def build_elevation_trajectory(
    *,
    scene_id: str,
    recipe_id: str,
    seed: int,
    station_position: Sequence[float],
    station_id: str,
    heights_m: Sequence[float],
    pitch_down_deg: Sequence[float],
    floor_area_m2: float,
    backend_version: str,
) -> dict[str, object]:
    """Build a T7 same-XY height and pitch intervention trajectory."""

    if len(heights_m) < 3 or len(heights_m) != len(pitch_down_deg):
        raise ValueError("T7 requires aligned height and pitch sequences")
    position = tuple(round(float(value), 6) for value in station_position)
    views = []
    for step, (height, pitch) in enumerate(zip(heights_m, pitch_down_deg, strict=True)):
        view_id = f"view-{step:03d}"
        rotation = yaw_pitch_quaternion(0.0, float(pitch))
        views.append(
            {
                "step": step,
                "view_id": view_id,
                "role": "initial" if step == 0 else "elevation",
                "world_from_agent": {
                    "parent_frame": "world",
                    "child_frame": f"agent:{view_id}",
                    "translation_m": position,
                    "rotation_xyzw": tuple(round(value, 9) for value in rotation),
                    "convention": "active_child_to_parent",
                },
                "distance_from_previous_m": 0.0,
                "cumulative_distance_m": 0.0,
                "backend_position_m": position,
                "camera_height_m": round(float(height), 6),
            }
        )
    return {
        "schema_version": "trajectory_plan.v1",
        "scene_id": scene_id,
        "recipe_id": recipe_id,
        "seed": seed,
        "backend_name": "omnigibson",
        "backend_version": backend_version,
        "backend_frame": "omnigibson_world_x_right_y_forward_z_up",
        "canonical_from_backend": {
            "parent_frame": "world",
            "child_frame": "omnigibson_world",
            "translation_m": (0.0, 0.0, 0.0),
            "rotation_xyzw": (0.0, 0.0, 0.0, 1.0),
            "convention": "active_child_to_parent",
        },
        "primary_island": 0,
        "primary_island_area_m2": round(float(floor_area_m2), 3),
        "outbound_geodesic_distance_m": 0.0,
        "closed_loop": False,
        "trajectory_class": "T7",
        "station_id": station_id,
        "height_sequence_m": [round(float(value), 6) for value in heights_m],
        "pitch_sequence_deg": [round(float(value), 6) for value in pitch_down_deg],
        "views": views,
    }


def build_occlusion_reveal_trajectory(
    *,
    scene_id: str,
    recipe_id: str,
    seed: int,
    source_points: Sequence[Sequence[float]],
    view_count: int,
    target_position: Sequence[float],
    target_entity_id: str,
    occluder_entity_id: str,
    floor_area_m2: float,
    backend_version: str,
) -> dict[str, object]:
    """Build a T8 path expected to turn an occluded target into visible evidence."""

    samples, geodesic = resample_polyline(source_points, view_count)
    target = tuple(float(value) for value in target_position)
    views = []
    cumulative = 0.0
    for step, point in enumerate(samples):
        delta = 0.0 if step == 0 else distance(samples[step - 1], point)
        cumulative += delta
        view_id = f"view-{step:03d}"
        if step == 0:
            role = "initial"
        elif step == len(samples) - 1:
            role = "reveal"
        elif step == 1:
            role = "occlusion_pass"
        else:
            role = "explore"
        rotation = yaw_quaternion(point, target)
        views.append(
            {
                "step": step,
                "view_id": view_id,
                "role": role,
                "world_from_agent": {
                    "parent_frame": "world",
                    "child_frame": f"agent:{view_id}",
                    "translation_m": tuple(round(value, 6) for value in point),
                    "rotation_xyzw": tuple(round(value, 9) for value in rotation),
                    "convention": "active_child_to_parent",
                },
                "distance_from_previous_m": round(delta, 6),
                "cumulative_distance_m": round(cumulative, 6),
                "backend_position_m": tuple(round(value, 6) for value in point),
            }
        )
    return {
        "schema_version": "trajectory_plan.v1",
        "scene_id": scene_id,
        "recipe_id": recipe_id,
        "seed": seed,
        "backend_name": "omnigibson",
        "backend_version": backend_version,
        "backend_frame": "omnigibson_world_x_right_y_forward_z_up",
        "canonical_from_backend": {
            "parent_frame": "world",
            "child_frame": "omnigibson_world",
            "translation_m": (0.0, 0.0, 0.0),
            "rotation_xyzw": (0.0, 0.0, 0.0, 1.0),
            "convention": "active_child_to_parent",
        },
        "primary_island": 0,
        "primary_island_area_m2": round(float(floor_area_m2), 3),
        "outbound_geodesic_distance_m": round(geodesic, 6),
        "closed_loop": False,
        "trajectory_class": "T8",
        "occlusion_annotation": {
            "target_id": target_entity_id,
            "occluder_id": occluder_entity_id,
            "occluded_views": ["view-000"],
            "decisive_view": f"view-{len(samples) - 1:03d}",
        },
        "views": views,
    }


def build_target_view_trajectory(
    *,
    scene_id: str,
    recipe_id: str,
    seed: int,
    camera_positions: Sequence[Sequence[float]],
    anchor_pairs: Sequence[tuple[str, str]],
    facing_positions: Sequence[Sequence[float]],
    floor_area_m2: float,
    backend_version: str,
) -> dict[str, object]:
    """Build independent T10 held-out views grounded by origin/facing anchors."""

    if len(camera_positions) < 3 or not (
        len(camera_positions) == len(anchor_pairs) == len(facing_positions)
    ):
        raise ValueError("T10 positions, anchor pairs and facing points must align")
    positions = [tuple(float(value) for value in point) for point in camera_positions]
    views = []
    anchors = []
    cumulative = 0.0
    for step, (point, pair, facing) in enumerate(
        zip(positions, anchor_pairs, facing_positions, strict=True)
    ):
        delta = 0.0 if step == 0 else distance(positions[step - 1], point)
        cumulative += delta
        view_id = f"view-{step:03d}"
        rotation = yaw_quaternion(point, facing)
        views.append(
            {
                "step": step,
                "view_id": view_id,
                "role": "target_holdout",
                "world_from_agent": {
                    "parent_frame": "world",
                    "child_frame": f"agent:{view_id}",
                    "translation_m": tuple(round(value, 6) for value in point),
                    "rotation_xyzw": tuple(round(value, 9) for value in rotation),
                    "convention": "active_child_to_parent",
                },
                "distance_from_previous_m": round(delta, 6),
                "cumulative_distance_m": round(cumulative, 6),
                "backend_position_m": tuple(round(value, 6) for value in point),
            }
        )
        anchors.append(
            {
                "view_id": view_id,
                "origin_entity_id": pair[0],
                "facing_entity_id": pair[1],
            }
        )
    return {
        "schema_version": "trajectory_plan.v1",
        "scene_id": scene_id,
        "recipe_id": recipe_id,
        "seed": seed,
        "backend_name": "omnigibson",
        "backend_version": backend_version,
        "backend_frame": "omnigibson_world_x_right_y_forward_z_up",
        "canonical_from_backend": {
            "parent_frame": "world",
            "child_frame": "omnigibson_world",
            "translation_m": (0.0, 0.0, 0.0),
            "rotation_xyzw": (0.0, 0.0, 0.0, 1.0),
            "convention": "active_child_to_parent",
        },
        "primary_island": 0,
        "primary_island_area_m2": round(float(floor_area_m2), 3),
        "outbound_geodesic_distance_m": round(cumulative, 6),
        "closed_loop": False,
        "trajectory_class": "T10",
        "target_anchors": anchors,
        "held_out": True,
        "views": views,
    }
