"""Canonical frame and rigid-transform value objects.

The canonical world is right-handed, measured in meters and uses +Z as up.
Quaternions are always stored in xyzw order. ``RigidTransform`` maps points from
its child frame into its parent frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, isclose, sqrt

from spatial_episode.domain.errors import FrameMismatchError

_EPSILON = 1e-12


@dataclass(frozen=True, slots=True, order=True)
class FrameId:
    """Stable name of a coordinate frame."""

    value: str

    def __post_init__(self) -> None:
        if not self.value or self.value.strip() != self.value:
            raise ValueError("frame ID must be non-empty and have no surrounding whitespace")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Vec3:
    """Three-dimensional vector with explicit arithmetic."""

    x: float
    y: float
    z: float

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def scale(self, factor: float) -> Vec3:
        return Vec3(self.x * factor, self.y * factor, self.z * factor)

    def dot(self, other: Vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self) -> float:
        return sqrt(self.dot(self))

    def almost_equal(self, other: Vec3, *, atol: float = 1e-9) -> bool:
        return all(
            isclose(left, right, rel_tol=0.0, abs_tol=atol)
            for left, right in zip(self.as_tuple(), other.as_tuple(), strict=True)
        )

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True, slots=True)
class Quaternion:
    """Hamilton quaternion in xyzw order."""

    x: float
    y: float
    z: float
    w: float

    @classmethod
    def identity(cls) -> Quaternion:
        return cls(0.0, 0.0, 0.0, 1.0)

    def norm(self) -> float:
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w)

    def normalized(self) -> Quaternion:
        magnitude = self.norm()
        if magnitude < _EPSILON:
            raise ValueError("zero quaternion cannot represent a rotation")
        return Quaternion(
            self.x / magnitude,
            self.y / magnitude,
            self.z / magnitude,
            self.w / magnitude,
        )

    def conjugate(self) -> Quaternion:
        return Quaternion(-self.x, -self.y, -self.z, self.w)

    def multiply(self, other: Quaternion) -> Quaternion:
        """Return the Hamilton product ``self * other``."""

        return Quaternion(
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        )

    def rotate(self, vector: Vec3) -> Vec3:
        unit = self.normalized()
        pure = Quaternion(vector.x, vector.y, vector.z, 0.0)
        rotated = unit.multiply(pure).multiply(unit.conjugate())
        return Vec3(rotated.x, rotated.y, rotated.z)

    def angular_distance(self, other: Quaternion) -> float:
        left = self.normalized()
        right = other.normalized()
        dot = abs(left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w)
        return 2.0 * acos(min(1.0, max(-1.0, dot)))

    def as_xyzw(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.w)


@dataclass(frozen=True, slots=True)
class RigidTransform:
    """Active transform from ``child_frame`` coordinates to ``parent_frame``."""

    parent_frame: FrameId
    child_frame: FrameId
    translation_m: Vec3
    rotation_xyzw: Quaternion

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation_xyzw", self.rotation_xyzw.normalized())

    @classmethod
    def identity(cls, frame: FrameId) -> RigidTransform:
        return cls(frame, frame, Vec3(0.0, 0.0, 0.0), Quaternion.identity())

    def apply_point(self, point_in_child: Vec3) -> Vec3:
        return self.rotation_xyzw.rotate(point_in_child) + self.translation_m

    def inverse(self) -> RigidTransform:
        inverse_rotation = self.rotation_xyzw.conjugate().normalized()
        inverse_translation = inverse_rotation.rotate(self.translation_m.scale(-1.0))
        return RigidTransform(
            parent_frame=self.child_frame,
            child_frame=self.parent_frame,
            translation_m=inverse_translation,
            rotation_xyzw=inverse_rotation,
        )

    def compose(self, child_transform: RigidTransform) -> RigidTransform:
        """Compose ``T_parent_child`` with ``T_child_grandchild``."""

        if self.child_frame != child_transform.parent_frame:
            raise FrameMismatchError(
                f"cannot compose {self.parent_frame}<-{self.child_frame} with "
                f"{child_transform.parent_frame}<-{child_transform.child_frame}"
            )
        return RigidTransform(
            parent_frame=self.parent_frame,
            child_frame=child_transform.child_frame,
            translation_m=self.apply_point(child_transform.translation_m),
            rotation_xyzw=self.rotation_xyzw.multiply(child_transform.rotation_xyzw),
        )

    def almost_equal(
        self,
        other: RigidTransform,
        *,
        translation_atol_m: float = 1e-9,
        rotation_atol_rad: float = 1e-9,
    ) -> bool:
        return (
            self.parent_frame == other.parent_frame
            and self.child_frame == other.child_frame
            and self.translation_m.almost_equal(other.translation_m, atol=translation_atol_m)
            and self.rotation_xyzw.angular_distance(other.rotation_xyzw) <= rotation_atol_rad
        )
