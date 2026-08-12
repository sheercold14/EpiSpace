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

from .sceneview import (
    Obstacle,
    OcclusionObservation,
    Pose2D,
    SceneLayout,
    SceneObject,
    VisibilityObservation,
    blocking_occluders,
)
from .standards import STD_V1, CompileStandard

# Structural categories: never question targets.
STRUCTURAL_LABELS = frozenset(
    {"ceilings", "floors", "walls", "background", "roof", "lawn", "driveway"}
)
# Structural geometry that blocks locomotion even though it is never a target.
STRUCTURAL_OBSTACLE_LABELS = frozenset({"walls"})


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


def plan_to_agent_views(plan: Any, *, camera_height_m: float | None = None) -> list[dict[str, Any]]:
    """Export a scriptgen TrajectoryPlan as acquisition camera-schedule views.

    Output items mirror the ``world_from_agent`` view records the OmniGibson
    backend already writes into ``trajectory_plan.json``, so the renderer can
    replay a planned trajectory with the same sensor pipeline it uses for its
    own sampled trajectories.
    """
    camera_height_m = STD_V1.camera_height_m if camera_height_m is None else camera_height_m
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
    hx, hy, hz = obb["half_extents_m"]
    qx, qy, qz, qw = obb["world_from_obb"]["rotation_xyzw"]
    r20 = abs(2.0 * (qx * qz - qw * qy))
    r21 = abs(2.0 * (qy * qz + qw * qx))
    r22 = abs(1.0 - 2.0 * (qx * qx + qy * qy))
    extent_z = r20 * hx + r21 * hy + r22 * hz
    return center_z - extent_z, center_z + extent_z


def _yaw_deg(obb: dict[str, Any]) -> float:
    """Planar heading of the OBB's local x axis in the canonical world frame."""
    qx, qy, qz, qw = obb["world_from_obb"]["rotation_xyzw"]
    sin_yaw = 2.0 * (qw * qz + qx * qy)
    cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.degrees(math.atan2(sin_yaw, cos_yaw))


def layout_from_scene_ir(
    scene_ir: dict[str, Any] | str | Path,
    *,
    std: CompileStandard = STD_V1,
) -> SceneLayout:
    """Build a planning layout from a bundle's ``scene_ir.json``.

    Objects are non-structural entities; obstacles contain those objects plus
    structural walls; occluders are obstacle footprints whose top reaches
    the declared eye height; walkable bounds are the union of floor footprints. Walls
    are deliberately collision-only, never question targets. The raw label is
    used as the category, and entity UUIDs carry through so the render backend
    and certificates reference the same ids.
    """
    ir = scene_ir
    if not isinstance(ir, dict):
        ir = json.loads(Path(ir).read_text(encoding="utf-8"))

    objects: list[SceneObject] = []
    obstacles: list[Obstacle] = []
    occlusion_obstacles: list[Obstacle] = []
    occluders: list[tuple[tuple[float, float], tuple[float, float]]] = []
    floor_boxes: list[tuple[tuple[float, float], tuple[float, float]]] = []

    for entity in ir["entities"]:
        label = entity["raw_label"]
        obb = entity["obb"]
        if label == "floors":
            floor_boxes.append(_footprint_aabb(obb))
        if label not in STRUCTURAL_LABELS or label in STRUCTURAL_OBSTACLE_LABELS:
            hx, hy, _ = obb["half_extents_m"]
            z_low, z_high = _z_span(obb)
            obstacle = Obstacle(
                label=label,
                center_xy=(obb["center_m"][0], obb["center_m"][1]),
                half_extents_xy=(hx, hy),
                yaw_deg=_yaw_deg(obb),
                z_low=z_low,
                z_high=z_high,
                entity_id=entity["entity_id"],
            )
            obstacles.append(obstacle)
            # Keep every potential blocker with its full vertical span. The
            # target-specific 3-D ray test decides whether it actually reaches
            # the sightline; a global eye-height cutoff loses valid downward
            # occlusions by beds and bookcases.
            occlusion_obstacles.append(obstacle)
            if z_high >= std.camera_height_m:
                occluders.append(_footprint_aabb(obb))
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
                yaw_deg=_yaw_deg(obb),
                center_z=float(obb["center_m"][2]),
                half_height=float(obb["half_extents_m"][2]),
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
        occlusion_obstacles=tuple(occlusion_obstacles),
        walkable_min=walkable_min,
        walkable_max=walkable_max,
        camera_height_m=std.camera_height_m,
        geom_occlusion_vertical_margin_m=std.geom_occlusion_vertical_margin_m,
        geom_occlusion_footprint_margin_ratio=std.geom_occlusion_footprint_margin_ratio,
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
    runtime_entity_ids: dict[int, str] = field(default_factory=dict)
    entity_categories: dict[str, str] = field(default_factory=dict)
    entity_source_ids: dict[str, str] = field(default_factory=dict)
    sensor_contract: dict[str, Any] = field(default_factory=dict)
    camera_heights_m: tuple[float, ...] = ()
    _mask_cache: dict[int, np.ndarray] = field(default_factory=dict, compare=False)
    _depth_cache: dict[int, np.ndarray] = field(default_factory=dict, compare=False)

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
        layout = layout_from_scene_ir(ir, std=std)
        poses = poses_from_trajectory_plan(root / "trajectory_plan.json")
        runtime_map: dict[str, list[int]] = {}
        # Runtime instance ids are assigned afresh when a scene is replayed.
        # Resolve the current bundle registry through scene_ir's stable
        # source_entity_id. The scene_ir registry remains a legacy fallback.
        snapshot_path = root / "scene_snapshot.json"
        if snapshot_path.is_file():
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            stable_entities = {
                str(entity["source_entity_id"]): str(entity["entity_id"])
                for entity in ir["entities"]
            }
            for runtime_id, source_entity_id in snapshot.get(
                "runtime_instance_registry", {}
            ).items():
                entity_id = stable_entities.get(str(source_entity_id))
                if entity_id is not None:
                    runtime_map.setdefault(entity_id, []).append(int(runtime_id))
        replayed_entities = set(runtime_map)
        for runtime_id, entity_id in ir["runtime_semantic_id_map"].items():
            if entity_id not in replayed_entities:
                runtime_map.setdefault(entity_id, []).append(int(runtime_id))
        runtime_to_entity = {
            runtime_id: entity_id
            for entity_id, runtime_ids in runtime_map.items()
            for runtime_id in runtime_ids
        }
        entity_categories = {
            str(entity["entity_id"]): str(entity["raw_label"]) for entity in ir["entities"]
        }
        entity_source_ids = {
            str(entity["entity_id"]): str(entity.get("source_entity_id", entity["entity_id"]))
            for entity in ir["entities"]
        }
        report_path = root / "render_report.json"
        report = (
            json.loads(report_path.read_text(encoding="utf-8"))
            if report_path.is_file()
            else {}
        )
        return cls(
            layout=layout,
            poses=poses,
            std=std,
            bundle_root=root,
            entity_runtime_ids={k: tuple(v) for k, v in runtime_map.items()},
            runtime_entity_ids=runtime_to_entity,
            entity_categories=entity_categories,
            entity_source_ids=entity_source_ids,
            sensor_contract=dict(report.get("sensor_contract", {})),
            camera_heights_m=tuple(
                float(row.get("camera_height_m", std.camera_height_m))
                for row in report.get("views", ())
            ),
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

    def occluders_between(self, name: str, t: int) -> tuple[str, ...]:
        return self.occlusion(name, t).occluder_ids

    def occlusion(self, name: str, t: int) -> OcclusionObservation:
        """Attribute an invisible target to a nearer rendered instance.

        The target centre is projected into the calibrated image. A small
        central patch must contain one dominant, resolvable foreground entity
        whose linear depth is safely in front of the target centre. Ambiguous
        or unmapped evidence is never promoted to an occlusion claim.
        """
        target_visibility = self.visibility(name, t)
        target_state = target_visibility.tristate(self.std)
        base: dict[str, Any] = {
            "backend": "render_instance_depth",
            "target_entity_id": name,
            "target_source_entity_id": self.entity_source_ids.get(name, name),
            "frame": t,
            "target_pixels": int(target_visibility.value),
        }
        if target_state is True:
            return OcclusionObservation(
                "render_instance_depth", "clear", (), {**base, "reason": "target_visible"}
            )
        if target_state is None:
            return OcclusionObservation(
                "render_instance_depth",
                "ambiguous",
                (),
                {**base, "reason": "target_visibility_ambiguous"},
            )

        instance = self._instance_mask(t)
        depth = self._depth_map(t)
        height, width = instance.shape
        pose = self.poses[t]
        target = self.layout.object(name)
        dx, dy = target.xy[0] - pose.x, target.xy[1] - pose.y
        yaw = math.radians(pose.yaw_deg)
        forward = math.cos(yaw) * dx + math.sin(yaw) * dy
        right = math.sin(yaw) * dx - math.cos(yaw) * dy
        camera_height = (
            self.camera_heights_m[t]
            if t < len(self.camera_heights_m)
            else self.layout.camera_height_m
        )
        up = target.center_z - camera_height
        if forward <= 1e-6:
            return OcclusionObservation(
                "render_instance_depth",
                "clear",
                (),
                {**base, "reason": "target_behind_camera", "forward_depth_m": forward},
            )
        horizontal_fov = float(
            self.sensor_contract.get("horizontal_fov_deg", 2.0 * self.std.fov_half_angle_deg)
        )
        focal_px = width / (2.0 * math.tan(math.radians(horizontal_fov) / 2.0))
        u = (width - 1) / 2.0 + focal_px * right / forward
        v = (height - 1) / 2.0 - focal_px * up / forward
        radius = self.std.occlusion_center_patch_radius_px
        center_x, center_y = int(round(u)), int(round(v))
        base.update(
            {
                "projected_target_center_px": [round(u, 3), round(v, 3)],
                "expected_target_depth_m": round(forward, 4),
                "patch_radius_px": radius,
            }
        )
        if not (
            radius <= center_x < width - radius and radius <= center_y < height - radius
        ):
            return OcclusionObservation(
                "render_instance_depth",
                "clear",
                (),
                {**base, "reason": "target_projection_outside_image"},
            )

        patch_ids = instance[
            center_y - radius : center_y + radius + 1,
            center_x - radius : center_x + radius + 1,
        ]
        patch_depth = depth[
            center_y - radius : center_y + radius + 1,
            center_x - radius : center_x + radius + 1,
        ]
        target_runtime_ids = set(self.entity_runtime_ids.get(name, ()))
        foreground = (
            np.isfinite(patch_depth)
            & (patch_depth > 0.0)
            & (patch_depth <= forward - self.std.occlusion_min_depth_margin_m)
            & (patch_ids > 1)
            & ~np.isin(patch_ids, tuple(target_runtime_ids))
        )
        supported_ids, supported_counts = np.unique(patch_ids[foreground], return_counts=True)
        ranked = sorted(
            (
                (int(count), int(runtime_id))
                for runtime_id, count in zip(supported_ids, supported_counts, strict=True)
            ),
            reverse=True,
        )
        if not ranked or ranked[0][0] < self.std.occlusion_min_support_pixels:
            return OcclusionObservation(
                "render_instance_depth",
                "ambiguous",
                (),
                {
                    **base,
                    "reason": "insufficient_foreground_support",
                    "support_pixels": ranked[0][0] if ranked else 0,
                    "minimum_support_pixels": self.std.occlusion_min_support_pixels,
                },
            )
        winner_count, winner_runtime_id = ranked[0]
        runner_count = ranked[1][0] if len(ranked) > 1 else 0
        dominance = float("inf") if runner_count == 0 else winner_count / runner_count
        if runner_count and dominance < self.std.occlusion_min_dominance_ratio:
            return OcclusionObservation(
                "render_instance_depth",
                "ambiguous",
                (),
                {
                    **base,
                    "reason": "competing_foreground_instances",
                    "winner_support_pixels": winner_count,
                    "runner_support_pixels": runner_count,
                    "dominance_ratio": round(dominance, 4),
                },
            )
        occluder = self.runtime_entity_ids.get(winner_runtime_id)
        if occluder is None:
            return OcclusionObservation(
                "render_instance_depth",
                "ambiguous",
                (),
                {
                    **base,
                    "reason": "foreground_runtime_id_unresolved",
                    "occluder_runtime_instance_id": winner_runtime_id,
                },
            )
        winner_depths = patch_depth[foreground & (patch_ids == winner_runtime_id)]
        observed_depth = float(np.median(winner_depths))
        witness = {
            **base,
            "reason": None,
            "occluder_entity_id": occluder,
            "occluder_source_entity_id": self.entity_source_ids.get(occluder, occluder),
            "occluder_category": self.entity_categories.get(occluder, "unknown"),
            "occluder_runtime_instance_id": winner_runtime_id,
            "observed_occluder_depth_m": round(observed_depth, 4),
            "depth_margin_m": round(forward - observed_depth, 4),
            "minimum_depth_margin_m": self.std.occlusion_min_depth_margin_m,
            "support_pixels": winner_count,
            "runner_support_pixels": runner_count,
            "dominance_ratio": None if runner_count == 0 else round(dominance, 4),
            "minimum_dominance_ratio": self.std.occlusion_min_dominance_ratio,
        }
        return OcclusionObservation("render_instance_depth", "occluded", (occluder,), witness)

    def _instance_mask(self, t: int) -> np.ndarray:
        if t not in self._mask_cache:
            path = self.bundle_root / "views" / f"view-{t:03d}.sensors.npz"
            with np.load(path) as arrays:
                self._mask_cache[t] = arrays["instance_id"]
        return self._mask_cache[t]

    def _depth_map(self, t: int) -> np.ndarray:
        if t not in self._depth_cache:
            path = self.bundle_root / "views" / f"view-{t:03d}.sensors.npz"
            with np.load(path) as arrays:
                self._depth_cache[t] = arrays["depth_m"]
        return self._depth_cache[t]
