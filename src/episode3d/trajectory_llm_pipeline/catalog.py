"""Geometry and perception access for one verified trajectory bundle.

The critical invariant is instance identity.  Questions may use natural category
names, but every calculation is bound through source id -> scene UUID -> runtime
semantic id -> instance mask.  This prevents repeated categories from silently
changing referent between views.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from episode3d.scene_llm_pipeline.lexicon import CATEGORY_NAMES_ZH, STRUCTURAL_LABELS

MIN_PIXELS = 256
HARD_DIRECTION_ANGLE_DEG = 20.0


class CatalogError(RuntimeError):
    """A source bundle violates the compiler's input contract."""


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CatalogError(f"expected JSON object: {path}")
    return value


def quat_yaw_deg(xyzw: list[float]) -> float:
    x, y, z, w = (float(v) for v in xyzw)
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def circular_delta_deg(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


@dataclass(frozen=True)
class Entity:
    source_id: str
    entity_id: str
    label: str
    name: str
    center_world: np.ndarray
    half_extents: np.ndarray
    views: tuple[str, ...]


class TrajectoryCatalog:
    """Typed, mask-aware view of one trajectory acquisition."""

    def __init__(self, bundle_root: Path) -> None:
        self.root = Path(bundle_root)
        self.scene = read_json(self.root / "scene_ir.json")
        self.episode = read_json(self.root / "spatial_episode.json")
        self.plan = read_json(self.root / "trajectory_plan.json")
        self.selection = read_json(self.root / "trajectory_selection.json")
        self.quality = read_json(self.root / "quality_report.json")
        raw_class = self.plan.get("trajectory_class") or self.selection.get("trajectory_class")
        if raw_class is None and len(self.episode.get("observations", [])) == 11:
            raw_class = "T1"  # legacy static-m2 bundles predate the explicit class field
        if raw_class is None:
            raise CatalogError("trajectory class is missing")
        self.trajectory_class = str(raw_class)
        if self.quality.get("integrity_status") != "pass":
            raise CatalogError("integrity_status is not pass")
        gate = self.quality.get("gates", {}).get(self.trajectory_class, {})
        if self.trajectory_class == "T1" and self.quality.get("visual_status") != "pass":
            raise CatalogError("T1 source visual_status is not pass")
        if self.trajectory_class != "T1" and gate.get("status") != "pass":
            raise CatalogError(f"{self.trajectory_class} class gate is not pass")

        self.view_ids = [o["view_id"] for o in self.episode["observations"]]
        self.observations = {o["view_id"]: o for o in self.episode["observations"]}
        self.plan_views = {o["view_id"]: o for o in self.plan["views"]}
        if set(self.view_ids) != set(self.plan_views):
            raise CatalogError("trajectory plan and episode disagree on view ids")
        self.visible_uuid = {
            o["view_id"]: set(o["visible_entity_ids"]) for o in self.episode["observations"]
        }

        runtime_map = self.scene.get("runtime_semantic_id_map", {})
        self.runtime_ids: dict[str, list[int]] = {}
        for runtime_id, entity_id in runtime_map.items():
            self.runtime_ids.setdefault(entity_id, []).append(int(runtime_id))
        self._mask_cache: dict[str, np.ndarray] = {}
        self._runtime_count_cache: dict[str, dict[int, int]] = {}
        self._pixel_cache: dict[tuple[str, str], int] = {}
        self._position_cache: dict[tuple[str, str], str] = {}

        self.entities_by_source: dict[str, Entity] = {}
        self.entities_by_uuid: dict[str, Entity] = {}
        self.label_counts: dict[str, int] = {}
        raw_entities = [
            raw
            for raw in self.scene["entities"]
            if raw["raw_label"] not in STRUCTURAL_LABELS and raw["raw_label"] in CATEGORY_NAMES_ZH
        ]
        for raw in raw_entities:
            label = raw["raw_label"]
            self.label_counts[label] = self.label_counts.get(label, 0) + 1
            entity = Entity(
                source_id=raw["source_entity_id"],
                entity_id=raw["entity_id"],
                label=label,
                name=CATEGORY_NAMES_ZH[label],
                center_world=np.array(raw["obb"]["center_m"], dtype=float),
                half_extents=np.array(raw["obb"]["half_extents_m"], dtype=float),
                views=tuple(v for v in self.view_ids if raw["entity_id"] in self.visible_uuid[v]),
            )
            self.entities_by_source[entity.source_id] = entity
            self.entities_by_uuid[entity.entity_id] = entity

    @property
    def scene_id(self) -> str:
        return str(self.episode["scene_id"])

    @property
    def family_id(self) -> str:
        return str(self.episode["family_id"])

    @property
    def split_group(self) -> str:
        return str(self.episode["split_group"])

    def entity(self, source_id: str) -> Entity:
        try:
            return self.entities_by_source[source_id]
        except KeyError as error:
            raw = next(
                (e for e in self.scene["entities"] if e.get("source_entity_id") == source_id),
                None,
            )
            if raw is None:
                raise CatalogError(f"unknown source entity: {source_id}") from error
            label = raw["raw_label"]
            if label not in CATEGORY_NAMES_ZH:
                raise CatalogError(f"selected category lacks Chinese lexicon: {label}") from error
            raise CatalogError(f"selected entity is never observed: {source_id}") from error

    def _mask(self, view_id: str) -> np.ndarray:
        if view_id not in self._mask_cache:
            path = self.root / "views" / f"{view_id}.sensors.npz"
            with np.load(path) as arrays:
                self._mask_cache[view_id] = arrays["instance_id"].copy()
        return self._mask_cache[view_id]

    def pixels(self, entity: Entity | str, view_id: str) -> int:
        item = self.entity(entity) if isinstance(entity, str) else entity
        key = (item.entity_id, view_id)
        if key in self._pixel_cache:
            return self._pixel_cache[key]
        ids = self.runtime_ids.get(item.entity_id, [])
        if not ids:
            return 0
        if view_id not in self._runtime_count_cache:
            values, counts = np.unique(self._mask(view_id), return_counts=True)
            self._runtime_count_cache[view_id] = {
                int(value): int(count) for value, count in zip(values, counts, strict=True)
            }
        count = sum(self._runtime_count_cache[view_id].get(runtime_id, 0) for runtime_id in ids)
        self._pixel_cache[key] = count
        return count

    def well_visible(
        self, entity: Entity | str, view_ids: list[str] | tuple[str, ...]
    ) -> str | None:
        item = self.entity(entity) if isinstance(entity, str) else entity
        return next((v for v in view_ids if self.pixels(item, v) >= MIN_PIXELS), None)

    def image_position(self, entity: Entity | str, view_id: str) -> str:
        item = self.entity(entity) if isinstance(entity, str) else entity
        key = (item.entity_id, view_id)
        if key in self._position_cache:
            return self._position_cache[key]
        ids = self.runtime_ids.get(item.entity_id, [])
        mask = np.isin(self._mask(view_id), ids)
        if not mask.any():
            return "中部"
        xs = np.nonzero(mask)[1]
        ratio = float(np.median(xs)) / max(mask.shape[1] - 1, 1)
        position = "左侧" if ratio < 0.38 else "右侧" if ratio > 0.62 else "中部"
        self._position_cache[key] = position
        return position

    def reference(self, entity: Entity | str, view_id: str | None = None) -> str:
        item = self.entity(entity) if isinstance(entity, str) else entity
        if self.label_counts.get(item.label, 0) == 1:
            return item.name
        evidence = view_id or self.well_visible(item, item.views)
        if evidence is None:
            raise CatalogError(f"ambiguous entity has no visible grounding view: {item.source_id}")
        index = self.view_ids.index(evidence) + 1
        return f"第{index}个视角画面{self.image_position(item, evidence)}的那{_classifier(item.label)}{item.name}"

    def named_visible(self, view_id: str, minimum_pixels: int = MIN_PIXELS) -> list[Entity]:
        rows = [
            entity
            for entity in self.entities_by_source.values()
            if view_id in entity.views and self.pixels(entity, view_id) >= minimum_pixels
        ]
        return sorted(rows, key=lambda e: (-self.pixels(e, view_id), e.source_id))

    def visibility_timeline(self, entity: Entity | str) -> list[str]:
        item = self.entity(entity) if isinstance(entity, str) else entity
        return [v for v in self.view_ids if self.pixels(item, v) >= MIN_PIXELS]

    def camera_xy_frame(self, view_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pose = self.observations[view_id]["world_from_camera"]
        yaw = math.radians(quat_yaw_deg(pose["rotation_xyzw"]))
        forward = np.array([-math.sin(yaw), math.cos(yaw)])
        right = np.array([forward[1], -forward[0]])
        origin = np.array(pose["translation_m"][:2], dtype=float)
        return origin, right, forward

    def relation(self, subject: Entity, reference: Entity, frame_view: str) -> dict[str, Any]:
        _, right, forward = self.camera_xy_frame(frame_view)
        delta = subject.center_world[:2] - reference.center_world[:2]
        dx, dy = float(delta @ right), float(delta @ forward)
        if abs(dx) >= abs(dy):
            relation = "right_of" if dx > 0 else "left_of"
            dominant, margin = abs(dx), abs(dx) - abs(dy)
        else:
            relation = "in_front_of" if dy > 0 else "behind"
            dominant, margin = abs(dy), abs(dy) - abs(dx)
        return {
            "subject_source_id": subject.source_id,
            "reference_source_id": reference.source_id,
            "frame_view_id": frame_view,
            "relation": relation,
            "delta_right_forward_m": [round(dx, 4), round(dy, 4)],
            "dominant_delta_m": round(dominant, 4),
            "margin_m": round(margin, 4),
        }

    def object_frame_relation(
        self, origin: Entity, facing: Entity, target: Entity
    ) -> dict[str, Any]:
        heading = facing.center_world[:2] - origin.center_world[:2]
        norm = float(np.linalg.norm(heading))
        if norm < 1e-6:
            raise CatalogError("object-frame anchors coincide")
        forward = heading / norm
        right = np.array([forward[1], -forward[0]])
        delta = target.center_world[:2] - origin.center_world[:2]
        dx, dy = float(delta @ right), float(delta @ forward)
        angle_margin = math.degrees(
            min(math.atan2(abs(dx), max(abs(dy), 1e-9)), math.atan2(abs(dy), max(abs(dx), 1e-9)))
        )
        quadrant = f"{'front' if dy >= 0 else 'back'}-{'right' if dx >= 0 else 'left'}"
        # Conservative OBB check.  Rotation is omitted in current scene IR, so
        # use the largest horizontal half extent as an upper bound on either axis.
        extent = float(max(target.half_extents[:2]))
        hard_ok = angle_margin >= HARD_DIRECTION_ANGLE_DEG and abs(dx) > extent and abs(dy) > extent
        return {
            "origin_source_id": origin.source_id,
            "facing_source_id": facing.source_id,
            "target_source_id": target.source_id,
            "query_xy_m": [round(dx, 4), round(dy, 4)],
            "quadrant": quadrant,
            "angle_margin_deg": round(angle_margin, 3),
            "target_extent_upper_bound_m": round(extent, 4),
            "hard_quadrant_ok": hard_ok,
        }

    def vertical_relation(
        self, a: Entity, b: Entity, margin_m: float = 0.1
    ) -> dict[str, Any] | None:
        a_low, a_high = a.center_world[2] - a.half_extents[2], a.center_world[2] + a.half_extents[2]
        b_low, b_high = b.center_world[2] - b.half_extents[2], b.center_world[2] + b.half_extents[2]
        if a_low - b_high >= margin_m:
            return {
                "subject_source_id": a.source_id,
                "reference_source_id": b.source_id,
                "relation": "above",
                "clearance_m": round(float(a_low - b_high), 4),
            }
        if b_low - a_high >= margin_m:
            return {
                "subject_source_id": a.source_id,
                "reference_source_id": b.source_id,
                "relation": "below",
                "clearance_m": round(float(b_low - a_high), 4),
            }
        return None


def _classifier(label: str) -> str:
    if label in {"pot_plant", "painting", "picture", "portrait"}:
        return "个"
    if label in {"straight_chair", "armchair", "swivel_chair", "eames_chair"}:
        return "把"
    if label in {"door"}:
        return "扇"
    if label in {"carpet"}:
        return "块"
    if label in {"sofa", "bed", "desk", "breakfast_table", "coffee_table", "bench"}:
        return "张"
    return "个"
