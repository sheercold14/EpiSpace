"""Pure 2-D geometry helpers shared by predicates and answer compilation.

Conventions (frozen; changing any of them is a standards version bump):

* World frame: x to the right, y forward, angles in degrees, counter-clockwise,
  measured from the +x axis. ``yaw_deg`` of a camera is the world heading of
  its optical axis.
* Camera-frame azimuth of an object: ``wrap_deg(bearing - yaw)``. Zero means
  straight ahead, positive means to the LEFT, negative to the RIGHT.
* Four sectors on azimuth: front (-45, 45], left (45, 135],
  back (135, 180] plus (-180, -135], right (-135, -45].
"""

from __future__ import annotations

import math
from itertools import pairwise

SECTOR_NAMES_4 = ("front", "left", "back", "right")
_SECTOR_BOUNDARIES_4 = (-135.0, -45.0, 45.0, 135.0)


def wrap_deg(angle: float) -> float:
    """Map any angle to the half-open interval (-180, 180]."""
    wrapped = math.fmod(angle, 360.0)
    if wrapped <= -180.0:
        wrapped += 360.0
    elif wrapped > 180.0:
        wrapped -= 360.0
    return wrapped


def bearing_deg(from_xy: tuple[float, float], to_xy: tuple[float, float]) -> float:
    """World-frame direction from one point towards another."""
    return math.degrees(math.atan2(to_xy[1] - from_xy[1], to_xy[0] - from_xy[0]))


def azimuth_deg(
    camera_xy: tuple[float, float], camera_yaw_deg: float, target_xy: tuple[float, float]
) -> float:
    """Camera-frame azimuth of ``target_xy`` (positive = left)."""
    return wrap_deg(bearing_deg(camera_xy, target_xy) - camera_yaw_deg)


def sector_of(azimuth: float) -> str:
    """Discretise an azimuth into one of the four sector names."""
    a = wrap_deg(azimuth)
    if -45.0 < a <= 45.0:
        return "front"
    if 45.0 < a <= 135.0:
        return "left"
    if -135.0 < a <= -45.0:
        return "right"
    return "back"


def sector_margin_deg(azimuth: float) -> float:
    """Distance in degrees from an azimuth to the nearest sector boundary."""
    a = wrap_deg(azimuth)
    margins = [abs(wrap_deg(a - boundary)) for boundary in _SECTOR_BOUNDARIES_4]
    margins.append(abs(wrap_deg(a - 180.0)))
    return min(margins)


def cumulative_turn_deg(yaws_deg: list[float]) -> float:
    """Total absolute heading change along a yaw sequence."""
    return sum(abs(wrap_deg(b - a)) for a, b in pairwise(yaws_deg))


def distance_m(a_xy: tuple[float, float], b_xy: tuple[float, float]) -> float:
    return math.hypot(b_xy[0] - a_xy[0], b_xy[1] - a_xy[1])


def segment_intersects_rect(
    p0: tuple[float, float],
    p1: tuple[float, float],
    rect_min: tuple[float, float],
    rect_max: tuple[float, float],
) -> bool:
    """Whether segment p0-p1 intersects an axis-aligned rectangle (slab test)."""
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    t_min, t_max = 0.0, 1.0
    for start, delta, low, high in (
        (x0, dx, rect_min[0], rect_max[0]),
        (y0, dy, rect_min[1], rect_max[1]),
    ):
        if abs(delta) < 1e-12:
            if start < low or start > high:
                return False
            continue
        t0 = (low - start) / delta
        t1 = (high - start) / delta
        if t0 > t1:
            t0, t1 = t1, t0
        t_min = max(t_min, t0)
        t_max = min(t_max, t1)
        if t_min > t_max:
            return False
    return True


def _point_in_rect_frame(
    point: tuple[float, float], center: tuple[float, float], yaw_deg: float
) -> tuple[float, float]:
    """Transform a world point into an oriented rectangle's local frame."""
    angle = math.radians(yaw_deg)
    cos_yaw, sin_yaw = math.cos(angle), math.sin(angle)
    dx, dy = point[0] - center[0], point[1] - center[1]
    return (cos_yaw * dx + sin_yaw * dy, -sin_yaw * dx + cos_yaw * dy)


def point_in_rotated_rect(
    point: tuple[float, float],
    center: tuple[float, float],
    half_extents: tuple[float, float],
    yaw_deg: float,
) -> bool:
    """Whether a point lies inside or on an oriented rectangle."""
    x, y = _point_in_rect_frame(point, center, yaw_deg)
    return abs(x) <= half_extents[0] and abs(y) <= half_extents[1]


def segment_intersects_rotated_rect(
    p0: tuple[float, float],
    p1: tuple[float, float],
    center: tuple[float, float],
    half_extents: tuple[float, float],
    yaw_deg: float,
) -> bool:
    """Whether a segment intersects an oriented rectangle."""
    local_p0 = _point_in_rect_frame(p0, center, yaw_deg)
    local_p1 = _point_in_rect_frame(p1, center, yaw_deg)
    hx, hy = half_extents
    return segment_intersects_rect(local_p0, local_p1, (-hx, -hy), (hx, hy))


def rotated_rect_penetration_depth(
    point: tuple[float, float],
    center: tuple[float, float],
    half_extents: tuple[float, float],
    yaw_deg: float,
) -> float:
    """Shortest local-axis distance from an interior point to the boundary."""
    x, y = _point_in_rect_frame(point, center, yaw_deg)
    return max(0.0, min(half_extents[0] - abs(x), half_extents[1] - abs(y)))
