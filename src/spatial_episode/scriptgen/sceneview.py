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

from dataclasses import dataclass, field
from typing import Literal, Protocol

from .geometry import azimuth_deg, distance_m, segment_intersects_rect
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


class SceneView(Protocol):
    """What predicates are allowed to know about a candidate trajectory."""

    @property
    def frame_count(self) -> int: ...

    def camera_pose(self, t: int) -> Pose2D: ...

    def object(self, name: str) -> SceneObject: ...

    def objects(self) -> list[SceneObject]: ...

    def visibility(self, name: str, t: int) -> VisibilityObservation: ...


@dataclass(frozen=True)
class SceneLayout:
    """Simulator-independent static description of a scene.

    ``occluders`` are axis-aligned rectangles (walls, partitions) given as
    ``(min_xy, max_xy)``; ``walkable`` is the traversable bounding box.
    """

    scene_id: str
    objects: tuple[SceneObject, ...]
    occluders: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = ()
    walkable_min: tuple[float, float] = (0.0, 0.0)
    walkable_max: tuple[float, float] = (10.0, 10.0)

    def object(self, name: str) -> SceneObject:
        for obj in self.objects:
            if obj.name == name:
                return obj
        raise KeyError(f"unknown object: {name}")


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
        obj = self.layout.object(name)
        pose = self.poses[t]
        dist = distance_m(pose.xy, obj.xy)
        if dist < 1e-6 or dist > self.std.max_view_distance_m:
            return VisibilityObservation("geom_ratio", 0.0, 0.0)
        if abs(azimuth_deg(pose.xy, pose.yaw_deg, obj.xy)) > self.std.fov_half_angle_deg:
            return VisibilityObservation("geom_ratio", 0.0, 0.0)
        unoccluded = self._unoccluded_ratio(pose.xy, obj)
        return VisibilityObservation("geom_ratio", obj.size_m / dist, unoccluded)

    def _unoccluded_ratio(self, camera_xy: tuple[float, float], obj: SceneObject) -> float:
        """Fraction of sample rays towards the object that clear all occluders."""
        half = obj.size_m / 2.0
        offsets = [
            (i / max(self.ray_samples - 1, 1) - 0.5) * 2.0 * half for i in range(self.ray_samples)
        ]
        clear = 0
        for offset in offsets:
            target = (obj.xy[0] + offset, obj.xy[1])
            blocked = any(
                segment_intersects_rect(camera_xy, target, rect_min, rect_max)
                for rect_min, rect_max in self.layout.occluders
            )
            if not blocked:
                clear += 1
        return clear / self.ray_samples
