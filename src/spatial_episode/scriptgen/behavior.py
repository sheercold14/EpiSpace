"""Adapters between scriptgen and BEHAVIOR/OmniGibson bundle artifacts.

Three adapters, all render-free at import time (they read JSON/NPZ files that
acquisition already produced; no Isaac Sim dependency):

* :func:`layout_from_scene_ir` — scene_ir.json -> :class:`SceneLayout`, so the
  engine plans trajectories on real BEHAVIOR scenes;
* :func:`poses_from_trajectory_plan` — trajectory_plan.json -> pose sequence,
  so existing acquired trajectories can be replayed through the checker;
* :class:`RenderSceneView` — the authoritative render backend: visibility is
  counted from rendered instance masks (``views/*.sensors.npz``), implementing
  the same :class:`SceneView` protocol as the geometry backend.

Conventions bridged here (single place, never elsewhere):

* scene_ir canonical frame is world/+Y-forward/+Z-up/right-handed/xyzw, so the
  floor plane is x-y and matches scriptgen's 2-D world directly;
* agent heading: scriptgen ``yaw_deg`` is the world angle of the camera's
  optical axis, counter-clockwise from +x. The agent's forward axis is +Y in
  its own frame, so yaw is the world direction of the rotated +Y axis.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .sceneview import Obstacle, Pose2D, SceneLayout, SceneObject, VisibilityObservation
from .standards import CompileStandard

# Structural categories: never question targets.
STRUCTURAL_LABELS = frozenset(
    {"ceilings", "floors", "walls", "background", "roof", "lawn", "driveway"}
)
# Structural geometry that blocks locomotion even though it is never a target.
STRUCTURAL_OBSTACLE_LABELS = frozenset({"walls"})
# Categories whose OBB blocks line of sight at eye level.
OCCLUDER_LABELS = frozenset({"walls", "pillar"})


def yaw_deg_from_quaternion_xyzw(q: tuple[float, float, float, float]) -> float:
    """World heading (degrees, CCW from +x) of the agent's +Y forward axis."""
    x, y, z, w = q
    forward_x = 2.0 * (x * y - w * z)
    forward_y = 1.0 - 2.0 * (x * x + z * z)
    return math.degrees(math.atan2(forward_y, forward_x))


def quaternion_xyzw_from_yaw_deg(yaw_deg: float) -> tuple[float, float, float, float]:
    """Inverse of :func:`yaw_deg_from_quaternion_xyzw` (pure z rotation).

    The agent's forward axis is +Y, which points at world angle 90 when the
    rotation is identity, so the z rotation angle is ``yaw - 90`` degrees.
    """
    half = math.radians(yaw_deg - 90.0) / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


def plan_to_agent_views(plan: Any, *, camera_height_m: float = 1.5) -> list[dict[str, Any]]:
    """Export a scriptgen TrajectoryPlan as acquisition camera-schedule views.

    Output items mirror the ``world_from_agent`` view records the OmniGibson
    backend already writes into ``trajectory_plan.json``, so the renderer can
    replay a planned trajectory with the same sensor pipeline it uses for its
    own sampled trajectories.
    """
    views: list[dict[str, Any]] = []
    for pose in plan.poses:
        views.append(
            {
                "view_id": f"view-{pose.frame:03d}",
                "step": pose.frame,
                "camera_height_m": camera_height_m,
                "world_from_agent": {
                    "parent_frame": "world",
                    "child_frame": f"agent:view-{pose.frame:03d}",
                    "convention": "active_child_to_parent",
                    "translation_m": [pose.x, pose.y, 0.0],
                    "rotation_xyzw": list(quaternion_xyzw_from_yaw_deg(pose.yaw_deg)),
                },
            }
        )
    return views


def _footprint_aabb(obb: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis-aligned x-y footprint of a possibly rotated OBB (conservative)."""
    cx, cy, _ = obb["center_m"]
    hx, hy, hz = obb["half_extents_m"]
    qx, qy, qz, qw = obb["world_from_obb"]["rotation_xyzw"]
    # Absolute-value rotation trick: world half-extent along each axis is
    # |R| @ half_extents. Only the x-y rows matter for the footprint.
    r00 = abs(1.0 - 2.0 * (qy * qy + qz * qz))
    r01 = abs(2.0 * (qx * qy - qw * qz))
    r02 = abs(2.0 * (qx * qz + qw * qy))
    r10 = abs(2.0 * (qx * qy + qw * qz))
    r11 = abs(1.0 - 2.0 * (qx * qx + qz * qz))
    r12 = abs(2.0 * (qy * qz - qw * qx))
    ex = r00 * hx + r01 * hy + r02 * hz
    ey = r10 * hx + r11 * hy + r12 * hz
    return (cx - ex, cy - ey), (cx + ex, cy + ey)


def _z_span(obb: dict[str, Any]) -> tuple[float, float]:
    center_z = obb["center_m"][2]
    # Conservative: use the raw half extent (rotation can only shrink z reach
    # of the z axis but other axes may contribute; walls are near-upright).
    hz = max(obb["half_extents_m"])
    return center_z - hz, center_z + hz


def _yaw_deg(obb: dict[str, Any]) -> float:
    """Planar heading of the OBB's local x axis in the canonical world frame."""
    qx, qy, qz, qw = obb["world_from_obb"]["rotation_xyzw"]
    sin_yaw = 2.0 * (qw * qz + qx * qy)
    cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.degrees(math.atan2(sin_yaw, cos_yaw))


def layout_from_scene_ir(
    scene_ir: dict[str, Any] | str | Path,
    *,
    camera_height_m: float = 1.5,
) -> SceneLayout:
    """Build a planning layout from a bundle's ``scene_ir.json``.

    Objects are non-structural entities; obstacles contain those objects plus
    structural walls; occluders are wall/pillar footprints that span the
    camera height; walkable bounds are the union of floor footprints. Walls
    are deliberately collision-only, never question targets. The raw label is
    used as the category, and entity UUIDs carry through so the render backend
    and certificates reference the same ids.
    """
    ir = scene_ir
    if not isinstance(ir, dict):
        ir = json.loads(Path(ir).read_text(encoding="utf-8"))

    objects: list[SceneObject] = []
    obstacles: list[Obstacle] = []
    occluders: list[tuple[tuple[float, float], tuple[float, float]]] = []
    floor_boxes: list[tuple[tuple[float, float], tuple[float, float]]] = []

    for entity in ir["entities"]:
        label = entity["raw_label"]
        obb = entity["obb"]
        if label == "floors":
            floor_boxes.append(_footprint_aabb(obb))
        if label in OCCLUDER_LABELS:
            z_low, z_high = _z_span(obb)
            if z_low <= camera_height_m <= z_high:
                occluders.append(_footprint_aabb(obb))
        if label not in STRUCTURAL_LABELS or label in STRUCTURAL_OBSTACLE_LABELS:
            hx, hy, _ = obb["half_extents_m"]
            z_low, z_high = _z_span(obb)
            obstacles.append(
                Obstacle(
                    label=label,
                    center_xy=(obb["center_m"][0], obb["center_m"][1]),
                    half_extents_xy=(hx, hy),
                    yaw_deg=_yaw_deg(obb),
                    z_low=z_low,
                    z_high=z_high,
                )
            )
        if label in STRUCTURAL_LABELS:
            continue
        hx, hy, _ = obb["half_extents_m"]
        objects.append(
            SceneObject(
                name=entity["entity_id"],
                category=label,
                xy=(obb["center_m"][0], obb["center_m"][1]),
                size_m=2.0 * max(hx, hy),
                uid=entity["entity_id"],
            )
        )

    if floor_boxes:
        walkable_min = (
            min(box[0][0] for box in floor_boxes),
            min(box[0][1] for box in floor_boxes),
        )
        walkable_max = (
            max(box[1][0] for box in floor_boxes),
            max(box[1][1] for box in floor_boxes),
        )
    else:
        walkable_min, walkable_max = (0.0, 0.0), (0.0, 0.0)

    return SceneLayout(
        scene_id=ir["scene_id"],
        objects=tuple(objects),
        obstacles=tuple(obstacles),
        occluders=tuple(occluders),
        walkable_min=walkable_min,
        walkable_max=walkable_max,
    )


def poses_from_trajectory_plan(plan: dict[str, Any] | str | Path) -> tuple[Pose2D, ...]:
    """Convert an acquired ``trajectory_plan.json`` into checker poses."""
    payload = plan
    if not isinstance(payload, dict):
        payload = json.loads(Path(payload).read_text(encoding="utf-8"))
    poses: list[Pose2D] = []
    for view in payload["views"]:
        transform = view["world_from_agent"]
        tx, ty, _ = transform["translation_m"]
        yaw = yaw_deg_from_quaternion_xyzw(tuple(transform["rotation_xyzw"]))
        poses.append(Pose2D(tx, ty, yaw))
    return tuple(poses)


@dataclass(frozen=True)
class RenderSceneView:
    """Authoritative SceneView backed by rendered instance masks.

    Visibility value is the pixel count of the entity's runtime ids in the
    view's ``instance_id`` array. Despite its name, ``runtime_semantic_id_map``
    in scene_ir keys on instance ids (verified against render_report's visible
    id lists). Rendered masks only contain pixels that survived occlusion, so
    ``unoccluded_ratio`` is reported as 1.0 and the double pixel threshold
    does all the work.
    """

    layout: SceneLayout
    poses: tuple[Pose2D, ...]
    std: CompileStandard
    bundle_root: Path
    entity_runtime_ids: dict[str, tuple[int, ...]]
    _mask_cache: dict[int, np.ndarray] = field(default_factory=dict, compare=False)

    @classmethod
    def from_bundle(
        cls,
        bundle_root: str | Path,
        std: CompileStandard,
        *,
        scene_ir: str | Path | None = None,
    ) -> RenderSceneView:
        """Build the render backend for a bundle.

        ``scene_ir`` overrides the geometry-truth file location for bundles
        that do not embed their own copy (scripted_plan batch renders reuse
        the scene_ir of the scene they were planned on).
        """
        root = Path(bundle_root)
        ir_path = Path(scene_ir) if scene_ir is not None else root / "scene_ir.json"
        ir = json.loads(ir_path.read_text(encoding="utf-8"))
        layout = layout_from_scene_ir(ir)
        poses = poses_from_trajectory_plan(root / "trajectory_plan.json")
        runtime_map: dict[str, list[int]] = {}
        for runtime_id, entity_id in ir["runtime_semantic_id_map"].items():
            runtime_map.setdefault(entity_id, []).append(int(runtime_id))
        return cls(
            layout=layout,
            poses=poses,
            std=std,
            bundle_root=root,
            entity_runtime_ids={k: tuple(v) for k, v in runtime_map.items()},
        )

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
        runtime_ids = self.entity_runtime_ids.get(name, ())
        if not runtime_ids:
            return VisibilityObservation("render_pixels", 0.0, 1.0)
        instance = self._instance_mask(t)
        pixels = int(np.isin(instance, runtime_ids).sum())
        return VisibilityObservation("render_pixels", float(pixels), 1.0)

    def _instance_mask(self, t: int) -> np.ndarray:
        if t not in self._mask_cache:
            path = self.bundle_root / "views" / f"view-{t:03d}.sensors.npz"
            with np.load(path) as arrays:
                self._mask_cache[t] = arrays["instance_id"]
        return self._mask_cache[t]
