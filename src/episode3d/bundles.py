"""Read-only adapters for OmniGibson geometry bundles."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from episode3d.language import entity_name

# Keep this oracle-only vocabulary local.  Importing the compiler's copy would
# create a bundles -> compiler -> bundles cycle, while the acquisition labels
# themselves are the source of truth for these admission statistics.
_STRUCTURAL_RAW_LABELS = frozenset(
    {
        "ceilings",
        "ceiling",
        "floor",
        "floors",
        "wall",
        "walls",
        "room",
        "background",
        "rail_fence",
    }
)
_FRAME_COLLISION_DEPTH_THRESHOLD_M = 0.4
_FRAME_CONTEXT_CLOSE_DEPTH_THRESHOLD_M = 0.75
_FRAME_CONTEXT_ENCLOSURE_RAW_LABELS = _STRUCTURAL_RAW_LABELS | frozenset(
    {"door", "fixed_window", "openable_window", "window"}
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def stable_id(namespace: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join((namespace, *parts)).encode()).hexdigest()[:20]
    return f"{namespace}-{digest}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_raw_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


@dataclass(frozen=True)
class Entity:
    entity_id: str
    source_entity_id: str
    label: str
    center_world_m: tuple[float, float, float]
    extent_m: tuple[float, float, float]
    region_id: str | None


@dataclass(frozen=True)
class View:
    view_id: str
    step: int
    role: str
    rgb_path: Path
    sensor_path: Path
    camera_height_m: float
    horizontal_fov_deg: float
    visible_entity_ids: frozenset[str]
    world_from_camera: dict[str, Any]


class Bundle:
    """Validated, immutable view over one compiled acquisition bundle."""

    REQUIRED_FILES = (
        "scene_ir.json",
        "spatial_episode.json",
        "trajectory_plan.json",
        "quality_report.json",
    )

    def __init__(
        self,
        root: Path,
        *,
        trajectory_class: str,
        source_sweep: str,
        job_status: str,
    ) -> None:
        self.root = root.resolve()
        for filename in self.REQUIRED_FILES:
            if not (self.root / filename).is_file():
                raise ValueError(f"{self.root}: missing {filename}")
        self.scene_ir = read_json(self.root / "scene_ir.json")
        self.episode = read_json(self.root / "spatial_episode.json")
        self.plan = read_json(self.root / "trajectory_plan.json")
        self.quality = read_json(self.root / "quality_report.json")
        render_report_path = self.root / "render_report.json"
        self.render_report = read_json(render_report_path) if render_report_path.is_file() else {}
        reasoning_path = self.root / "reasoning_tasks.json"
        self.reasoning_tasks_present = reasoning_path.is_file()
        self.reasoning = (
            read_json(reasoning_path)
            if self.reasoning_tasks_present
            else {"schema_version": "missing", "tasks": []}
        )
        self.trajectory_class = trajectory_class
        self.source_sweep = source_sweep
        self.job_status = job_status
        self.scene_id = str(self.episode["scene_id"])
        self.episode_id = str(self.episode["episode_id"])
        self.split_group = str(self.episode["split_group"])

        if self.quality.get("integrity_status") != "pass":
            raise ValueError(f"{self.root}: integrity_status is not pass")
        class_status = self.quality.get("trajectory_status")
        if class_status not in (None, "pass"):
            raise ValueError(f"{self.root}: trajectory_status is {class_status}")
        declared_class = self.plan.get("trajectory_class")
        if declared_class and declared_class != trajectory_class:
            raise ValueError(
                f"{self.root}: plan class {declared_class} != configured {trajectory_class}"
            )

        self.entities = self._load_entities()
        self.entities_by_source = {
            entity.source_entity_id: entity for entity in self.entities.values()
        }
        self._runtime_to_entity = {
            int(runtime_id): str(entity_id)
            for runtime_id, entity_id in self.scene_ir.get("runtime_semantic_id_map", {}).items()
        }
        self._entity_to_runtime = {
            entity_id: runtime_id for runtime_id, entity_id in self._runtime_to_entity.items()
        }
        self._view_pixel_counts: dict[str, dict[str, int]] = {}
        self._view_visual_stats: dict[str, dict[str, dict[str, Any]]] = {}
        self._view_rgb_stats: dict[str, dict[str, Any]] = {}
        self._view_frame_collision_stats: dict[str, dict[str, Any]] = {}
        self._view_frame_context_stats: dict[str, dict[str, Any]] = {}
        self._view_raw_category_visual_stats: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
        self.views = self._load_views()
        self.view_by_id = {view.view_id: view for view in self.views}
        if len(self.view_by_id) != len(self.views):
            raise ValueError(f"{self.root}: duplicate view_id")
        self.tasks = tuple(self.reasoning.get("tasks", ()))

    def _load_entities(self) -> dict[str, Entity]:
        entities: dict[str, Entity] = {}
        for raw in self.scene_ir.get("entities", []):
            center = raw["world_from_entity"]["translation_m"]
            half = raw["obb"]["half_extents_m"]
            entity = Entity(
                entity_id=str(raw["entity_id"]),
                source_entity_id=str(raw["source_entity_id"]),
                label=str(raw["raw_label"]),
                center_world_m=tuple(float(value) for value in center),
                extent_m=tuple(2.0 * float(value) for value in half),
                region_id=str(raw["region_id"]) if raw.get("region_id") else None,
            )
            entities[entity.entity_id] = entity
        if not entities:
            raise ValueError(f"{self.root}: scene has no entities")
        return entities

    def _load_views(self) -> tuple[View, ...]:
        plan_by_id = {str(item["view_id"]): item for item in self.plan["views"]}
        sensor_contract = self.render_report.get("sensor_contract", {})
        default_height = float(sensor_contract.get("sensor_height_m", 1.5))
        horizontal_fov = float(sensor_contract.get("horizontal_fov_deg", 90.0))
        views: list[View] = []
        for observation in self.episode["observations"]:
            view_id = str(observation["view_id"])
            plan_view = plan_by_id[view_id]
            sensor_path = self.root / "views" / f"{view_id}.sensors.npz"
            if not sensor_path.is_file():
                raise ValueError(f"{self.root}: missing sensor payload {sensor_path.name}")
            views.append(
                View(
                    view_id=view_id,
                    step=int(observation["step"]),
                    role=str(plan_view["role"]),
                    # A sensor archive is never a valid MLLM image.  The pipeline
                    # must call materialize_model_rgb() before export.
                    rgb_path=sensor_path.resolve(),
                    sensor_path=sensor_path.resolve(),
                    camera_height_m=float(plan_view.get("camera_height_m", default_height)),
                    horizontal_fov_deg=horizontal_fov,
                    visible_entity_ids=frozenset(
                        str(value) for value in observation["visible_entity_ids"]
                    ),
                    world_from_camera=dict(observation["world_from_camera"]),
                )
            )
        views.sort(key=lambda view: view.step)
        if [view.step for view in views] != list(range(len(views))):
            raise ValueError(f"{self.root}: observation steps are not contiguous")
        return tuple(views)

    def materialize_model_rgb(self, media_root: Path) -> None:
        """Extract only the raw RGB channel from sensor archives.

        Acquisition previews intentionally contain RGB, metric depth and an
        instance-ID panel.  They are useful for human QA but are forbidden as
        model input.  This method creates a content-faithful RGB-only cache and
        rewires the immutable view records to those files.
        """

        bundle_key = stable_id("rgb", self.source_sweep, self.root.name)
        bundle_root = media_root.resolve() / bundle_key
        bundle_root.mkdir(parents=True, exist_ok=True)
        materialized: list[View] = []
        for view in self.views:
            destination = bundle_root / f"{view.view_id}.png"
            metadata_path = bundle_root / f"{view.view_id}.rgb.json"
            metadata = _read_valid_rgb_metadata(metadata_path, destination, view.sensor_path)
            if metadata is None:
                with np.load(view.sensor_path, allow_pickle=False) as payload:
                    if "rgb" not in payload.files:
                        raise ValueError(f"{view.sensor_path}: missing rgb channel")
                    rgb = payload["rgb"]
                    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                        raise ValueError(
                            f"{view.sensor_path}: invalid rgb array {rgb.shape}/{rgb.dtype}"
                        )
                    _write_rgb_png_atomic(destination, rgb)
                    instance = payload.get("instance_id")
                    if instance is None or instance.shape != rgb.shape[:2]:
                        raise ValueError(
                            f"{view.sensor_path}: invalid or missing instance_id channel"
                        )
                    runtime_ids, counts = np.unique(instance, return_counts=True)
                    pixel_counts = {
                        entity_id: int(count)
                        for runtime_id, count in zip(runtime_ids, counts, strict=True)
                        if (entity_id := self._runtime_to_entity.get(int(runtime_id)))
                    }
                    source_stat = view.sensor_path.stat()
                    metadata = {
                        "schema_version": "epispace.raw_rgb.v1",
                        "source_sensor": str(view.sensor_path),
                        "source_size": source_stat.st_size,
                        "source_mtime_ns": source_stat.st_mtime_ns,
                        "width": int(rgb.shape[1]),
                        "height": int(rgb.shape[0]),
                        "mode": "RGB",
                        "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                        "instance_pixel_counts": pixel_counts,
                    }
                    _write_json_atomic(metadata_path, metadata)
            self._view_pixel_counts[view.view_id] = {
                str(entity_id): int(count)
                for entity_id, count in metadata["instance_pixel_counts"].items()
            }
            materialized.append(replace(view, rgb_path=destination))
        self.views = tuple(materialized)
        self.view_by_id = {view.view_id: view for view in self.views}
        self._view_rgb_stats.clear()

    def entity(self, entity_or_source_id: str) -> Entity:
        if entity_or_source_id in self.entities:
            return self.entities[entity_or_source_id]
        try:
            return self.entities_by_source[entity_or_source_id]
        except KeyError as error:
            raise ValueError(
                f"{self.root}: unknown entity/source id {entity_or_source_id}"
            ) from error

    def visible_views(self, entity_id: str) -> tuple[str, ...]:
        return tuple(view.view_id for view in self.views if entity_id in view.visible_entity_ids)

    def common_visible(self, entity_ids: set[str]) -> tuple[str, ...]:
        return tuple(
            view.view_id for view in self.views if entity_ids.issubset(view.visible_entity_ids)
        )

    def canonical_belief(
        self,
        *,
        quantization_m: float = 0.5,
        evidence_view_ids_by_entity: dict[str, tuple[str, ...]] | None = None,
    ) -> dict[str, Any]:
        """Build the query-agnostic target state from observed entities only."""

        anchor = self.views[0]
        basis = _anchor_basis(anchor.world_from_camera)
        observed_ids = (
            set(evidence_view_ids_by_entity)
            if evidence_view_ids_by_entity is not None
            else set().union(*(set(view.visible_entity_ids) for view in self.views))
        )
        entities: list[dict[str, Any]] = []
        final_visible = self.views[-1].visible_entity_ids
        for entity_id in sorted(observed_ids):
            entity = self.entities.get(entity_id)
            if entity is None:
                continue
            evidence = list(
                evidence_view_ids_by_entity[entity_id]
                if evidence_view_ids_by_entity is not None
                else self.visible_views(entity_id)
            )
            if not evidence:
                continue
            canonical = _canonical_point(entity.center_world_m, basis)
            entities.append(
                {
                    "state_id": stable_id("ent", self.scene_id, entity.entity_id),
                    "category": entity_name(entity.label),
                    "center_bin": [round(value / quantization_m) for value in canonical],
                    "extent_bin": [round(value / quantization_m) for value in entity.extent_m],
                    "status": (
                        "visible"
                        if self.views[-1].view_id in evidence and entity_id in final_visible
                        else "seen_not_current"
                    ),
                    "first_seen_view": evidence[0],
                    "last_seen_view": evidence[-1],
                    "evidence_views": evidence,
                }
            )
        return {
            "schema_version": "epispace.observable_belief.v1",
            "frame_id": f"anchor_camera@{anchor.view_id}",
            "frame_contract": {
                "origin": "anchor camera ground projection",
                "+X": "anchor camera right",
                "+Y": "anchor horizontal forward",
                "+Z": "gravity up",
                "anchor_view_id": anchor.view_id,
            },
            "quantization_m": quantization_m,
            "entity_count": len(entities),
            "entities": entities,
        }

    def media(self, view_ids: list[str] | tuple[str, ...] | None = None) -> list[str]:
        selected = (
            self.views
            if view_ids is None
            else tuple(self.view_by_id[view_id] for view_id in view_ids)
        )
        return [str(view.rgb_path) for view in selected]

    def rgb_hashes(self) -> dict[str, str]:
        return {view.view_id: file_sha256(view.rgb_path) for view in self.views}

    def visible_pixel_count(self, entity_id: str, view_id: str) -> int:
        return self._view_pixel_counts.get(view_id, {}).get(entity_id, 0)

    def instance_visual_stats(self, entity_id: str, view_id: str) -> dict[str, Any]:
        """Measure whether a visible mask is a usable natural-language referent.

        Pixel count alone accepts thin slivers, heavily occluded masks, and
        objects cut by an image boundary.  These statistics are computed from
        the oracle instance channel only for dataset gating; they are never
        serialized into model-visible input.
        """

        if view_id not in self._view_visual_stats:
            view = self.view_by_id[view_id]
            with np.load(view.sensor_path, allow_pickle=False) as payload:
                instance = payload.get("instance_id")
                if instance is None or instance.ndim != 2:
                    raise ValueError(f"{view.sensor_path}: invalid instance_id channel")
                height, width = instance.shape
                per_entity: dict[str, dict[str, Any]] = {}
                runtime_ids, inverse, pixel_counts = np.unique(
                    instance.reshape(-1), return_inverse=True, return_counts=True
                )
                linear = np.arange(instance.size, dtype=np.int32)
                pixel_x = linear % width
                pixel_y = linear // width
                left_by_runtime = np.full(runtime_ids.size, width, dtype=np.int32)
                right_by_runtime = np.full(runtime_ids.size, -1, dtype=np.int32)
                top_by_runtime = np.full(runtime_ids.size, height, dtype=np.int32)
                bottom_by_runtime = np.full(runtime_ids.size, -1, dtype=np.int32)
                np.minimum.at(left_by_runtime, inverse, pixel_x)
                np.maximum.at(right_by_runtime, inverse, pixel_x)
                np.minimum.at(top_by_runtime, inverse, pixel_y)
                np.maximum.at(bottom_by_runtime, inverse, pixel_y)
                outer_band_x = max(1, round(width * 0.05))
                outer_band_y = max(1, round(height * 0.05))
                in_outer_band = (
                    (pixel_x < outer_band_x)
                    | (pixel_x >= width - outer_band_x)
                    | (pixel_y < outer_band_y)
                    | (pixel_y >= height - outer_band_y)
                )
                outer_counts = np.bincount(
                    inverse,
                    weights=in_outer_band.astype(np.uint8),
                    minlength=runtime_ids.size,
                )
                for index, runtime_id in enumerate(runtime_ids):
                    candidate_id = self._runtime_to_entity.get(int(runtime_id))
                    if candidate_id not in view.visible_entity_ids:
                        continue
                    pixels = int(pixel_counts[index])
                    left = int(left_by_runtime[index])
                    right = int(right_by_runtime[index])
                    top = int(top_by_runtime[index])
                    bottom = int(bottom_by_runtime[index])
                    bbox_width = right - left + 1
                    bbox_height = bottom - top + 1
                    bbox_area = bbox_width * bbox_height
                    border_sides = [
                        side
                        for side, touched in (
                            ("left", left == 0),
                            ("top", top == 0),
                            ("right", right == width - 1),
                            ("bottom", bottom == height - 1),
                        )
                        if touched
                    ]
                    per_entity[candidate_id] = {
                        "visible_pixels": pixels,
                        "image_width_px": width,
                        "image_height_px": height,
                        "bbox_xyxy_px": [left, top, right, bottom],
                        "bbox_width_px": bbox_width,
                        "bbox_height_px": bbox_height,
                        "mask_bbox_fill_ratio": pixels / bbox_area,
                        "outer_five_percent_pixel_fraction": float(outer_counts[index] / pixels),
                        "touches_image_border": bool(border_sides),
                        "border_sides": border_sides,
                    }
            self._view_visual_stats[view_id] = per_entity
        return dict(
            self._view_visual_stats[view_id].get(
                entity_id,
                {
                    "visible_pixels": 0,
                    "image_width_px": 0,
                    "image_height_px": 0,
                    "bbox_xyxy_px": None,
                    "bbox_width_px": 0,
                    "bbox_height_px": 0,
                    "mask_bbox_fill_ratio": 0.0,
                    "outer_five_percent_pixel_fraction": 0.0,
                    "touches_image_border": False,
                    "border_sides": [],
                },
            )
        )

    def rgb_visual_stats(self, view_id: str) -> dict[str, Any]:
        """Measure gross RGB degeneration without consulting scene semantics.

        A valid instance mask can coexist with an unusable render when the
        camera clips into a wall or another mesh.  Quantized color dominance
        and entropy expose that failure mode from the exact RGB channel seen by
        the model.  The statistics are dataset-admission metadata only.
        """

        if view_id not in self._view_rgb_stats:
            view = self.view_by_id[view_id]
            if view.rgb_path == view.sensor_path or view.rgb_path.suffix == ".npz":
                with np.load(view.sensor_path, allow_pickle=False) as payload:
                    rgb = np.asarray(payload["rgb"], dtype=np.uint8)
            else:
                with Image.open(view.rgb_path) as image:
                    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(f"{view.rgb_path}: invalid RGB image")
            height, width = rgb.shape[:2]
            # Sixteen bins per channel retain coarse scene diversity while
            # making the metric stable to small renderer/color perturbations.
            quantized = rgb.astype(np.uint16) // 16
            encoded = quantized[:, :, 0] * 256 + quantized[:, :, 1] * 16 + quantized[:, :, 2]
            counts = np.bincount(encoded.reshape(-1), minlength=4096)
            probabilities = counts[counts > 0].astype(np.float64) / encoded.size
            entropy_bits = float(-np.sum(probabilities * np.log2(probabilities)))
            gray = rgb.astype(np.float32).mean(axis=2) / 255.0
            self._view_rgb_stats[view_id] = {
                "image_width_px": int(width),
                "image_height_px": int(height),
                "quantization_bins_per_channel": 16,
                "dominant_quantized_color_fraction": float(counts.max() / encoded.size),
                "quantized_color_entropy_bits": entropy_bits,
                "mean_luminance": float(gray.mean()),
                "luminance_std": float(gray.std()),
            }
        return dict(self._view_rgb_stats[view_id])

    def frame_collision_stats(self, view_id: str) -> dict[str, Any]:
        """Measure near-field non-structural geometry in the upper image half.

        This is an oracle-only dataset-admission statistic.  In particular it
        reads the raw ``depth_m`` and ``instance_id`` channels and deliberately
        does *not* trust ``visible_entity_ids``: a camera can collide with an
        object whose instance was omitted from the observation summary.

        Zero and non-finite depths are invalid sensor samples.  The reported
        fraction therefore uses the number of finite, positive top-half depth
        samples as its denominator rather than silently treating invalid depth
        as clear space.
        """

        if view_id not in self._view_frame_collision_stats:
            view = self.view_by_id[view_id]
            with np.load(view.sensor_path, allow_pickle=False) as payload:
                depth = payload.get("depth_m")
                instance = payload.get("instance_id")
                if depth is None or depth.ndim != 2 or not np.issubdtype(depth.dtype, np.number):
                    raise ValueError(f"{view.sensor_path}: invalid depth_m channel")
                if (
                    instance is None
                    or instance.ndim != 2
                    or instance.shape != depth.shape
                    or not np.issubdtype(instance.dtype, np.integer)
                ):
                    raise ValueError(f"{view.sensor_path}: invalid instance_id channel")
                height, width = depth.shape
                top_height = height // 2
                if top_height == 0 or width == 0:
                    raise ValueError(f"{view.sensor_path}: empty top-half sensor region")

                top_depth = depth[:top_height]
                top_instance = instance[:top_height]
                valid_depth = np.isfinite(top_depth) & (top_depth > 0.0)
                valid_count = int(np.count_nonzero(valid_depth))
                nonstructural_runtime_ids = tuple(
                    runtime_id
                    for runtime_id, entity_id in self._runtime_to_entity.items()
                    if (
                        (entity := self.entities.get(entity_id)) is not None
                        and _normalize_raw_label(entity.label) not in _STRUCTURAL_RAW_LABELS
                    )
                )
                belongs_to_nonstructural_entity = np.isin(
                    top_instance,
                    nonstructural_runtime_ids,
                    assume_unique=False,
                )
                near_nonstructural = (
                    valid_depth
                    & (top_depth < _FRAME_COLLISION_DEPTH_THRESHOLD_M)
                    & belongs_to_nonstructural_entity
                )
                near_geometry = valid_depth & (
                    top_depth < _FRAME_COLLISION_DEPTH_THRESHOLD_M
                )
                near_count = int(np.count_nonzero(near_nonstructural))
                near_geometry_count = int(np.count_nonzero(near_geometry))
                self._view_frame_collision_stats[view_id] = {
                    "near_nonstructural_pixel_count": near_count,
                    "valid_top_half_depth_pixel_count": valid_count,
                    "near_nonstructural_pixel_fraction": (
                        near_count / valid_count if valid_count else 0.0
                    ),
                    "near_geometry_pixel_count": near_geometry_count,
                    "near_geometry_pixel_fraction": (
                        near_geometry_count / valid_count if valid_count else 0.0
                    ),
                    "top_half_pixel_count": int(top_depth.size),
                    "depth_threshold_m": _FRAME_COLLISION_DEPTH_THRESHOLD_M,
                }
        return dict(self._view_frame_collision_stats[view_id])

    def frame_context_stats(self, view_id: str) -> dict[str, Any]:
        """Measure whole-frame enclosure and foreground-obstruction failure modes.

        The older collision statistic intentionally examines only geometry
        closer than 40 cm in the upper half of the image.  Two unusable camera
        poses evade that narrow test: looking squarely into a nearby wall, and
        placing a large thin object (for example a sports net) around 60--70 cm
        from the camera.  This oracle-only admission helper therefore combines
        raw metric depth with the complete instance channel.  It is never
        serialized into model input or used as a target.
        """

        if view_id not in self._view_frame_context_stats:
            view = self.view_by_id[view_id]
            with np.load(view.sensor_path, allow_pickle=False) as payload:
                depth = payload.get("depth_m")
                instance = payload.get("instance_id")
                if depth is None or depth.ndim != 2 or not np.issubdtype(depth.dtype, np.number):
                    raise ValueError(f"{view.sensor_path}: invalid depth_m channel")
                if (
                    instance is None
                    or instance.ndim != 2
                    or instance.shape != depth.shape
                    or not np.issubdtype(instance.dtype, np.integer)
                ):
                    raise ValueError(f"{view.sensor_path}: invalid instance_id channel")
                if depth.size == 0:
                    raise ValueError(f"{view.sensor_path}: empty sensor frame")

                valid_depth = np.isfinite(depth) & (depth > 0.0)
                valid_count = int(np.count_nonzero(valid_depth))
                if valid_count:
                    depth_values = depth[valid_depth]
                    depth_p10_m, depth_median_m, depth_p90_m = (
                        float(value)
                        for value in np.quantile(depth_values, (0.10, 0.50, 0.90))
                    )
                else:
                    depth_p10_m = depth_median_m = depth_p90_m = 0.0
                close_count = int(
                    np.count_nonzero(
                        valid_depth
                        & (depth < _FRAME_CONTEXT_CLOSE_DEPTH_THRESHOLD_M)
                    )
                )

                enclosure_runtime_ids = tuple(
                    runtime_id
                    for runtime_id, entity_id in self._runtime_to_entity.items()
                    if (
                        (entity := self.entities.get(entity_id)) is not None
                        and _normalize_raw_label(entity.label)
                        in _FRAME_CONTEXT_ENCLOSURE_RAW_LABELS
                    )
                )
                enclosure_mask = np.isin(
                    instance,
                    enclosure_runtime_ids,
                    assume_unique=False,
                )

                dominant: dict[str, Any] = {
                    "entity_id": None,
                    "raw_label": None,
                    "pixel_count": 0,
                    "pixel_fraction": 0.0,
                    "border_sides": [],
                    "median_depth_m": None,
                }
                for runtime_id, entity_id in self._runtime_to_entity.items():
                    entity = self.entities.get(entity_id)
                    if (
                        entity is None
                        or _normalize_raw_label(entity.label) in _STRUCTURAL_RAW_LABELS
                    ):
                        continue
                    mask = instance == runtime_id
                    pixel_count = int(np.count_nonzero(mask))
                    if pixel_count <= int(dominant["pixel_count"]):
                        continue
                    border_sides = []
                    if np.any(mask[:, 0]):
                        border_sides.append("left")
                    if np.any(mask[:, -1]):
                        border_sides.append("right")
                    if np.any(mask[0, :]):
                        border_sides.append("top")
                    if np.any(mask[-1, :]):
                        border_sides.append("bottom")
                    valid_mask = mask & valid_depth
                    median_depth = (
                        float(np.median(depth[valid_mask]))
                        if np.any(valid_mask)
                        else None
                    )
                    dominant = {
                        "entity_id": entity_id,
                        "raw_label": entity.label,
                        "pixel_count": pixel_count,
                        "pixel_fraction": pixel_count / instance.size,
                        "border_sides": border_sides,
                        "median_depth_m": median_depth,
                    }

                self._view_frame_context_stats[view_id] = {
                    "valid_depth_pixel_count": valid_count,
                    "frame_pixel_count": int(depth.size),
                    "close_geometry_pixel_count": close_count,
                    "close_geometry_pixel_fraction": (
                        close_count / valid_count if valid_count else 0.0
                    ),
                    "close_depth_threshold_m": (
                        _FRAME_CONTEXT_CLOSE_DEPTH_THRESHOLD_M
                    ),
                    "depth_p10_m": depth_p10_m,
                    "depth_median_m": depth_median_m,
                    "depth_p90_m": depth_p90_m,
                    "depth_p90_minus_p10_m": depth_p90_m - depth_p10_m,
                    "enclosure_pixel_count": int(np.count_nonzero(enclosure_mask)),
                    "enclosure_pixel_fraction": float(
                        np.count_nonzero(enclosure_mask) / instance.size
                    ),
                    "dominant_nonstructural_instance": dominant,
                }
        result = dict(self._view_frame_context_stats[view_id])
        result["dominant_nonstructural_instance"] = dict(
            result["dominant_nonstructural_instance"]
        )
        return result

    def raw_category_visual_stats(self, view_id: str, labels: Iterable[str]) -> dict[str, Any]:
        """Aggregate raw instance evidence for one or more category labels.

        Unlike :meth:`instance_visual_stats`, this oracle admission helper
        scans the complete ``instance_id`` channel.  Runtime instances omitted
        from ``visible_entity_ids`` remain part of the result, which is
        essential when validating negative visual claims.  The values are
        cached in memory only and are not added to any model-facing record.
        """

        normalized_labels = tuple(sorted({_normalize_raw_label(str(label)) for label in labels}))
        cache_key = (view_id, normalized_labels)
        if cache_key not in self._view_raw_category_visual_stats:
            view = self.view_by_id[view_id]
            with np.load(view.sensor_path, allow_pickle=False) as payload:
                instance = payload.get("instance_id")
                if (
                    instance is None
                    or instance.ndim != 2
                    or not np.issubdtype(instance.dtype, np.integer)
                ):
                    raise ValueError(f"{view.sensor_path}: invalid instance_id channel")
                height, width = instance.shape
                matching_runtime_ids = tuple(
                    runtime_id
                    for runtime_id, entity_id in self._runtime_to_entity.items()
                    if (
                        (entity := self.entities.get(entity_id)) is not None
                        and _normalize_raw_label(entity.label) in normalized_labels
                    )
                )
                matching_mask = np.isin(
                    instance,
                    matching_runtime_ids,
                    assume_unique=False,
                )
                matching_pixels = int(np.count_nonzero(matching_mask))
                matching_instance_count = 0
                max_bbox_min_side = 0
                if matching_pixels:
                    pixel_y, pixel_x = np.nonzero(matching_mask)
                    selected_runtime_ids = instance[matching_mask]
                    present_runtime_ids, inverse = np.unique(
                        selected_runtime_ids, return_inverse=True
                    )
                    matching_instance_count = int(present_runtime_ids.size)
                    left = np.full(present_runtime_ids.size, width, dtype=np.int32)
                    right = np.full(present_runtime_ids.size, -1, dtype=np.int32)
                    top = np.full(present_runtime_ids.size, height, dtype=np.int32)
                    bottom = np.full(present_runtime_ids.size, -1, dtype=np.int32)
                    np.minimum.at(left, inverse, pixel_x)
                    np.maximum.at(right, inverse, pixel_x)
                    np.minimum.at(top, inverse, pixel_y)
                    np.maximum.at(bottom, inverse, pixel_y)
                    bbox_min_sides = np.minimum(right - left + 1, bottom - top + 1)
                    max_bbox_min_side = int(bbox_min_sides.max())
                self._view_raw_category_visual_stats[cache_key] = {
                    "labels": normalized_labels,
                    "matching_pixel_count": matching_pixels,
                    "matching_instance_count": matching_instance_count,
                    "max_instance_bbox_min_side_px": max_bbox_min_side,
                    "image_width_px": int(width),
                    "image_height_px": int(height),
                }
        return dict(self._view_raw_category_visual_stats[cache_key])

    def canonical_relation(
        self, subject: Entity, reference: Entity
    ) -> tuple[str, float, float, float]:
        """Return a horizontal relation in the first-camera anchor frame."""

        basis = _anchor_basis(self.views[0].world_from_camera)
        subject_xyz = _canonical_point(subject.center_world_m, basis)
        reference_xyz = _canonical_point(reference.center_world_m, basis)
        dx = subject_xyz[0] - reference_xyz[0]
        dy = subject_xyz[1] - reference_xyz[1]
        if abs(dx) >= abs(dy):
            relation = "right_of" if dx > 0 else "left_of"
            margin = abs(dx) - abs(dy)
        else:
            relation = "in_front_of" if dy > 0 else "behind"
            margin = abs(dy) - abs(dx)
        return relation, margin, dx, dy


def _yaw_rad(rotation_xyzw: list[float] | tuple[float, ...]) -> float:
    x, y, z, w = (float(value) for value in rotation_xyzw)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _anchor_basis(frame: dict[str, Any]) -> dict[str, tuple[float, ...]]:
    yaw = _yaw_rad(frame["rotation_xyzw"])
    translation = tuple(float(value) for value in frame["translation_m"])
    return {
        "origin": translation,
        "right": (math.cos(yaw), math.sin(yaw)),
        "forward": (-math.sin(yaw), math.cos(yaw)),
    }


def _canonical_point(
    point_world_m: tuple[float, float, float],
    basis: dict[str, tuple[float, ...]],
) -> tuple[float, float, float]:
    origin = basis["origin"]
    dx = point_world_m[0] - origin[0]
    dy = point_world_m[1] - origin[1]
    right = basis["right"]
    forward = basis["forward"]
    return (
        dx * right[0] + dy * right[1],
        dx * forward[0] + dy * forward[1],
        point_world_m[2],
    )


def dominant_relation(subject: Entity, reference: Entity) -> tuple[str, float]:
    dx = subject.center_world_m[0] - reference.center_world_m[0]
    dy = subject.center_world_m[1] - reference.center_world_m[1]
    if abs(dx) >= abs(dy):
        return ("right_of" if dx > 0 else "left_of", abs(dx) - abs(dy))
    return ("in_front_of" if dy > 0 else "behind", abs(dy) - abs(dx))


def _read_valid_rgb_metadata(
    metadata_path: Path, image_path: Path, sensor_path: Path
) -> dict[str, Any] | None:
    if not metadata_path.is_file() or not image_path.is_file():
        return None
    try:
        metadata = read_json(metadata_path)
        source_stat = sensor_path.stat()
        if (
            metadata.get("schema_version") != "epispace.raw_rgb.v1"
            or metadata.get("source_size") != source_stat.st_size
            or metadata.get("source_mtime_ns") != source_stat.st_mtime_ns
            or not isinstance(metadata.get("instance_pixel_counts"), dict)
        ):
            return None
        with Image.open(image_path) as image:
            image.load()
            if (
                image.mode != "RGB"
                or image.width != metadata.get("width")
                or image.height != metadata.get("height")
                or hashlib.sha256(image.tobytes()).hexdigest() != metadata.get("rgb_sha256")
            ):
                return None
        return metadata
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _write_rgb_png_atomic(path: Path, rgb: np.ndarray) -> None:
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".png"
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            Image.fromarray(rgb).save(stream, format="PNG", optimize=False, compress_level=1)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".json"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
