"""Scene access layer for predicates.

Predicates never touch a simulator or an image directly. They see the world
through a :class:`SceneView`, which answers three questions:

* where is the camera at frame ``t`` (pose),
* where is a named object (static world position),
* how visible is a named object at frame ``t`` (a backend-tagged observation).

Two backends implement the visibility part with the *same* predicate code on
top:

* geometry backend (search phase): frustum + ray occlusion estimate, no
  rendering; provided here by :class:`GeometrySceneView`.
* render backend (compile phase): instance-mask pixel counts from a rendered
  bundle; an adapter over acquisition bundles implements the same protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Protocol

from .geometry import (
    azimuth_deg,
    distance_m,
    segment_intersects_rect,
    segment_intersects_rotated_rect,
)
from .standards import CompileStandard

VisibilityKind = Literal["geom_ratio", "render_pixels"]


@dataclass(frozen=True)
class Pose2D:
    """Camera pose on the floor plane: position plus heading."""

    x: float
    y: float
    yaw_deg: float

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class VisibilityObservation:
    """Backend-tagged visibility measurement for one object at one frame."""

    kind: VisibilityKind
    value: float
    unoccluded_ratio: float

    def tristate(self, std: CompileStandard) -> bool | None:
        """Double-threshold verdict: True / False / None (ambiguous)."""
        if self.kind == "geom_ratio":
            hi, lo = std.geom_min_visible_ratio, std.geom_max_invisible_ratio
            min_unoccluded = std.min_unoccluded_ratio
        else:
            hi, lo = float(std.render_min_visible_pixels), float(std.render_max_invisible_pixels)
            min_unoccluded = std.render_min_unoccluded_ratio
        if self.value <= lo:
            return False
        if self.value >= hi and self.unoccluded_ratio >= min_unoccluded:
            return True
        return None


@dataclass(frozen=True)
class SceneObject:
    """Static object ground truth exposed to the engine."""

    name: str
    category: str
    xy: tuple[float, float]
    size_m: float
    uid: str


@dataclass(frozen=True)
class Obstacle:
    """One oriented obstacle footprint with its conservative vertical span."""

    label: str
    center_xy: tuple[float, float]
    half_extents_xy: tuple[float, float]
    yaw_deg: float
    z_low: float
    z_high: float
    entity_id: str | None = None


class SceneView(Protocol):
    """What predicates are allowed to know about a candidate trajectory."""

    @property
    def frame_count(self) -> int: ...

    @property
    def layout(self) -> SceneLayout: ...

    def camera_pose(self, t: int) -> Pose2D: ...

    def object(self, name: str) -> SceneObject: ...

    def objects(self) -> list[SceneObject]: ...

    def visibility(self, name: str, t: int) -> VisibilityObservation: ...

    def occluders_between(self, name: str, t: int) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class SceneLayout:
    """Simulator-independent static description of a scene.

    ``occluders`` are axis-aligned rectangles (walls, partitions) given as
    ``(min_xy, max_xy)``; ``walkable`` is the traversable bounding box.
    """

    scene_id: str
    objects: tuple[SceneObject, ...]
    obstacles: tuple[Obstacle, ...] = ()
    occluders: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = ()
    occlusion_obstacles: tuple[Obstacle, ...] = ()
    walkable_min: tuple[float, float] = (0.0, 0.0)
    walkable_max: tuple[float, float] = (10.0, 10.0)

    def object(self, name: str) -> SceneObject:
        for obj in self.objects:
            if obj.name == name:
                return obj
        raise KeyError(f"unknown object: {name}")


@dataclass(frozen=True)
class ReindexedSceneView:
    """SceneView over a reordering / subset / repetition of another view's frames.

    ``frames[i]`` is the base frame presented at new index ``i``. This is the
    substrate for interventions: a variant is nothing but an index sequence
    over already-rendered frames (permute reorders, drop omits, delay repeats),
    so every variant is judged by the same predicates on the same rendered
    evidence — never on synthetic or re-rendered imagery.
    """

    base: SceneView
    frames: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("frame sequence must not be empty")
        bad = [t for t in self.frames if not 0 <= t < self.base.frame_count]
        if bad:
            raise ValueError(f"frame indices out of range: {bad}")

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def layout(self) -> SceneLayout:
        return self.base.layout

    def camera_pose(self, t: int) -> Pose2D:
        return self.base.camera_pose(self.frames[t])

    def object(self, name: str) -> SceneObject:
        return self.base.object(name)

    def objects(self) -> list[SceneObject]:
        return self.base.objects()

    def visibility(self, name: str, t: int) -> VisibilityObservation:
        return self.base.visibility(name, self.frames[t])

    def occluders_between(self, name: str, t: int) -> tuple[str, ...]:
        return self.base.occluders_between(name, self.frames[t])


@dataclass(frozen=True)
class GeometrySceneView:
    """Render-free SceneView over a layout and a candidate pose sequence."""

    layout: SceneLayout
    poses: tuple[Pose2D, ...]
    std: CompileStandard
    ray_samples: int = field(default=5)

    @property
    def frame_count(self) -> int:
        return len(self.poses)

    def camera_pose(self, t: int) -> Pose2D:
        return self.poses[t]

    def object(self, name: str) -> SceneObject:
        return self.layout.object(name)

    def objects(self) -> list[SceneObject]:
        return list(self.layout.objects)

    def visibility(self, name: str, t: int) -> VisibilityObservation:
        """Extent-aware frustum estimate, conservative in BOTH claim directions.

        The object is an arc of angular half-width ``atan(size/2 / dist)``, not
        a point. Fully outside the field of view means definitely invisible;
        fully inside falls back to the size/distance ratio; PARTIALLY inside is
        reported in the ambiguous band on purpose — a rendered image may show
        an edge sliver (verified against real renders), so neither a visible
        nor an invisible claim is safe and the candidate must be rejected.
        Distance imposes no hard cutoff: the ratio itself decays with range,
        and a large object far away can still render above threshold.
        """
        obj = self.layout.object(name)
        pose = self.poses[t]
        dist = distance_m(pose.xy, obj.xy)
        if dist < 1e-6:
            return VisibilityObservation("geom_ratio", 0.0, 0.0)
        azimuth = abs(azimuth_deg(pose.xy, pose.yaw_deg, obj.xy))
        half_width_deg = math.degrees(math.atan2(obj.size_m / 2.0, dist))
        fov = self.std.fov_half_angle_deg
        if azimuth >= fov + half_width_deg:
            return VisibilityObservation("geom_ratio", 0.0, 0.0)
        if azimuth > fov - half_width_deg:
            ambiguous = (self.std.geom_max_invisible_ratio + self.std.geom_min_visible_ratio) / 2.0
            return VisibilityObservation("geom_ratio", ambiguous, 1.0)
        unoccluded = self._unoccluded_ratio(pose.xy, obj)
        if unoccluded == 0.0:
            return VisibilityObservation("geom_ratio", 0.0, 0.0)
        return VisibilityObservation("geom_ratio", obj.size_m / dist, unoccluded)

    def occluders_between(self, name: str, t: int) -> tuple[str, ...]:
        target = self.layout.object(name)
        return blocking_occluders(self.layout, self.poses[t].xy, target)

    def _unoccluded_ratio(self, camera_xy: tuple[float, float], obj: SceneObject) -> float:
        """Fraction of sample rays towards the object that clear all occluders."""
        half = obj.size_m / 2.0
        offsets = [
            (i / max(self.ray_samples - 1, 1) - 0.5) * 2.0 * half for i in range(self.ray_samples)
        ]
        clear = 0
        for offset in offsets:
            target = (obj.xy[0] + offset, obj.xy[1])
            if not blocking_occluders(self.layout, camera_xy, obj, target_xy=target):
                clear += 1
        return clear / self.ray_samples


def blocking_occluders(
    layout: SceneLayout,
    camera_xy: tuple[float, float],
    target: SceneObject,
    *,
    target_xy: tuple[float, float] | None = None,
) -> tuple[str, ...]:
    """Eye-height obstacles intersecting the ray before the target's near face."""
    destination = target.xy if target_xy is None else target_xy
    distance = distance_m(camera_xy, destination)
    if distance < 1e-6:
        return ()
    stop = max(0.0, distance - target.size_m / 2.0)
    fraction = stop / distance
    ray_end = (
        camera_xy[0] + (destination[0] - camera_xy[0]) * fraction,
        camera_xy[1] + (destination[1] - camera_xy[1]) * fraction,
    )

    blocked: list[str] = []
    if layout.occlusion_obstacles:
        for obstacle in layout.occlusion_obstacles:
            if obstacle.entity_id == target.uid:
                continue
            if segment_intersects_rotated_rect(
                camera_xy,
                ray_end,
                obstacle.center_xy,
                obstacle.half_extents_xy,
                obstacle.yaw_deg,
            ):
                blocked.append(obstacle.entity_id or obstacle.label)
    else:
        for index, (rect_min, rect_max) in enumerate(layout.occluders):
            if segment_intersects_rect(camera_xy, ray_end, rect_min, rect_max):
                blocked.append(f"legacy_occluder:{index}")
    return tuple(blocked)
