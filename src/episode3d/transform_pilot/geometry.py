"""Horizontal-frame geometry shared by Transform Pilot compilers.

The simulator uses active child-to-parent quaternions.  In the acquired T3
sequence, a positive declared turn is a clockwise/right turn and therefore a
negative mathematical camera-yaw delta.  Keeping that convention in one module
prevents the silent sign flips that commonly invalidate mental-rotation data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DirectionMeasurement:
    label: str
    right_m: float
    front_m: float
    bearing_deg: float
    center_margin_deg: float
    obb_angular_radius_deg: float
    effective_margin_deg: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "label": self.label,
            "right_m": round(self.right_m, 6),
            "front_m": round(self.front_m, 6),
            "bearing_deg": round(self.bearing_deg, 6),
            "center_margin_deg": round(self.center_margin_deg, 6),
            "obb_angular_radius_deg": round(self.obb_angular_radius_deg, 6),
            "effective_margin_deg": round(self.effective_margin_deg, 6),
        }


def wrap_degrees(angle_deg: float) -> float:
    """Map an angle to [-180, 180)."""

    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def yaw_from_xyzw(rotation_xyzw: list[float] | tuple[float, ...]) -> float:
    x, y, z, w = (float(value) for value in rotation_xyzw)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def camera_components(
    point_xy: tuple[float, float],
    origin_xy: tuple[float, float],
    camera_yaw_rad: float,
) -> tuple[float, float]:
    dx = float(point_xy[0]) - float(origin_xy[0])
    dy = float(point_xy[1]) - float(origin_xy[1])
    right = dx * math.cos(camera_yaw_rad) + dy * math.sin(camera_yaw_rad)
    front = -dx * math.sin(camera_yaw_rad) + dy * math.cos(camera_yaw_rad)
    return right, front


def yaw_facing(
    origin_xy: tuple[float, float], target_xy: tuple[float, float]
) -> float:
    """Return the simulator yaw that makes ``target_xy`` straight ahead.

    In this camera convention yaw=0 faces world +Y and positive yaw rotates
    the forward vector toward world -X.  The minus sign on ``dx`` is therefore
    essential for object-anchored perspective questions.
    """

    dx = float(target_xy[0]) - float(origin_xy[0])
    dy = float(target_xy[1]) - float(origin_xy[1])
    if math.hypot(dx, dy) <= 1e-9:
        raise ValueError("origin and target must be distinct to define a facing yaw")
    return math.atan2(-dx, dy)


def direction_measurement(
    *,
    point_xy: tuple[float, float],
    origin_xy: tuple[float, float],
    camera_yaw_rad: float,
    horizontal_radius_m: float = 0.0,
) -> DirectionMeasurement:
    """Measure a four-sector bearing with an OBB-aware boundary margin."""

    right, front = camera_components(point_xy, origin_xy, camera_yaw_rad)
    distance = math.hypot(right, front)
    bearing = wrap_degrees(math.degrees(math.atan2(right, front)))
    centers = {
        "front": 0.0,
        "right": 90.0,
        "back": -180.0,
        "left": -90.0,
    }
    label, center = min(
        centers.items(), key=lambda item: abs(wrap_degrees(bearing - item[1]))
    )
    center_offset = abs(wrap_degrees(bearing - center))
    center_margin = 45.0 - center_offset
    angular_radius = (
        math.degrees(math.atan2(max(float(horizontal_radius_m), 0.0), distance))
        if distance > 1e-9
        else 90.0
    )
    return DirectionMeasurement(
        label=label,
        right_m=right,
        front_m=front,
        bearing_deg=bearing,
        center_margin_deg=center_margin,
        obb_angular_radius_deg=angular_radius,
        effective_margin_deg=center_margin - angular_radius,
    )


def horizontal_obb_radius(extent_m: tuple[float, float, float]) -> float:
    """Conservative radius of an axis-aligned horizontal OBB projection."""

    return 0.5 * math.hypot(float(extent_m[0]), float(extent_m[1]))
