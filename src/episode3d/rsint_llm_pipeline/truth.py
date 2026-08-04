"""Deterministic truth compiler for the Rs_int incremental dialogue pilot.

The language model is deliberately absent from this module.  It freezes the
11-view evidence schedule, executes spatial operations against simulator truth,
and emits node-level claim sheets that downstream language agents may only
verbalize.  A source premise that is not re-executable causes compilation to
fail rather than becoming a natural-language training example.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "epispace.rsint_truth_compilation.v2"
FRAME_ID = "canonical_room_xy"
FRAME_SURFACE_ZH = "按房间平面图约定：+X 为右，+Y 为前"
HARD_QUADRANT_MIN_ANGLE_MARGIN_DEG = 20.0
MINIMUM_VISUAL_CUE_GAP_PX = 96.0

ENTITY_NAMES = {
    "fridge": "冰箱",
    "oven": "烤箱",
    "microwave": "微波炉",
    "dishwasher": "洗碗机",
    "sofa": "沙发",
    "coffee_table": "咖啡桌",
    "standing_tv": "电视",
    "laptop": "笔记本电脑",
    "swivel_chair": "转椅",
    "bed": "床",
}

RELATION_ZH = {
    "left_of": "左侧",
    "right_of": "右侧",
    "in_front_of": "前方",
    "behind": "后方",
}

QUADRANT_ZH = {
    "front-left": "左前方",
    "front-right": "右前方",
    "back-left": "左后方",
    "back-right": "右后方",
}


class TruthCompilationError(RuntimeError):
    """The bundle cannot support the requested dialogue claim."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TruthCompilationError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise TruthCompilationError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _content_id(namespace: str, value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:20]
    return f"{namespace}-{digest}"


@dataclass(frozen=True)
class EntityTruth:
    key: str
    name_zh: str
    entity_id: str
    center_m: tuple[float, float, float]
    half_extents_m: tuple[float, float, float]
    obb_rotation_xyzw: tuple[float, float, float, float]
    region_id: str | None
    source_path: str


class RsIntTruthCompiler:
    """Read-only adapter and seven-round deterministic program executor."""

    REQUIRED_FILES = (
        "scene_ir.json",
        "spatial_episode.json",
        "trajectory_plan.json",
        "quality_report.json",
        "render_report.json",
        "relation_oracle.json",
    )

    def __init__(self, bundle_root: str | Path) -> None:
        self.root = Path(bundle_root).resolve()
        for filename in self.REQUIRED_FILES:
            if not (self.root / filename).is_file():
                raise TruthCompilationError(f"missing source file: {self.root / filename}")
        self.scene = _read_json(self.root / "scene_ir.json")
        self.episode = _read_json(self.root / "spatial_episode.json")
        self.plan = _read_json(self.root / "trajectory_plan.json")
        self.quality = _read_json(self.root / "quality_report.json")
        self.render_report = _read_json(self.root / "render_report.json")
        self.oracle = _read_json(self.root / "relation_oracle.json")
        if self.quality.get("integrity_status") != "pass":
            raise TruthCompilationError("source bundle integrity_status is not pass")
        if self.quality.get("visual_status") != "pass":
            raise TruthCompilationError("source bundle visual_status is not pass")
        sensor = self.render_report.get("sensor_contract", {})
        if (
            self.render_report.get("status") != "success"
            or self.render_report.get("high_quality_rendering") is not True
            or sensor.get("rgb_render_mode") != "RealTimePathTracing"
            or int(sensor.get("width_px", 0)) != 1024
            or int(sensor.get("height_px", 0)) != 1024
            or int(sensor.get("minimum_visible_instance_pixels", 0)) != 256
        ):
            raise TruthCompilationError("source render contract is not the frozen 1024px HQ pilot")

        observations = self.episode.get("observations")
        if not isinstance(observations, list) or len(observations) != 11:
            raise TruthCompilationError("Rs_int flagship requires exactly 11 observations")
        self.observations = observations
        self.view_ids = tuple(str(value["view_id"]) for value in observations)
        expected = tuple(f"view-{index:03d}" for index in range(11))
        if self.view_ids != expected:
            raise TruthCompilationError(f"unexpected view order: {self.view_ids}")
        self.observation_by_id = {str(value["view_id"]): value for value in observations}
        self.visible_by_view = {
            str(value["view_id"]): frozenset(str(item) for item in value["visible_entity_ids"])
            for value in observations
        }

        by_label: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for index, raw in enumerate(self.scene.get("entities", [])):
            by_label.setdefault(str(raw["raw_label"]), []).append((index, raw))
        self.entities: dict[str, EntityTruth] = {}
        for key, name_zh in ENTITY_NAMES.items():
            matches = by_label.get(key, [])
            if len(matches) != 1:
                raise TruthCompilationError(f"expected one curated {key}, found {len(matches)}")
            index, raw = matches[0]
            center = raw["world_from_entity"]["translation_m"]
            half = raw["obb"]["half_extents_m"]
            obb_rotation = raw["obb"]["world_from_obb"]["rotation_xyzw"]
            self.entities[key] = EntityTruth(
                key=key,
                name_zh=name_zh,
                entity_id=str(raw["entity_id"]),
                center_m=tuple(float(value) for value in center),
                half_extents_m=tuple(float(value) for value in half),
                obb_rotation_xyzw=tuple(float(value) for value in obb_rotation),
                region_id=str(raw["region_id"]) if raw.get("region_id") else None,
                source_path=f"scene_ir.json#/entities/{index}",
            )

        runtime_map = self.scene.get("runtime_semantic_id_map")
        if not isinstance(runtime_map, dict):
            raise TruthCompilationError("scene has no runtime semantic-id map")
        self.runtime_id_by_entity_id = {
            str(entity_id): int(runtime_id) for runtime_id, entity_id in runtime_map.items()
        }

        self.regions = {
            str(value["region_id"]): str(value["raw_label"])
            for value in self.scene.get("regions", [])
        }
        plan_views = self.plan.get("views", [])
        self.plan_by_view = {str(value["view_id"]): value for value in plan_views}
        if set(self.plan_by_view) != set(self.view_ids):
            raise TruthCompilationError("trajectory plan and observation view ids differ")

        media = self.root / "oral_demo" / "media"
        self.media_by_view: dict[str, Path] = {}
        for view_id in self.view_ids:
            path = media / f"{view_id}-rgb.webp"
            if not path.is_file():
                raise TruthCompilationError(f"missing RGB frame: {path}")
            self.media_by_view[view_id] = path.resolve()

    def visible_views(self, entity_key: str) -> tuple[str, ...]:
        entity_id = self.entities[entity_key].entity_id
        return tuple(
            view_id for view_id in self.view_ids if entity_id in self.visible_by_view[view_id]
        )

    def co_visible_views(self, a: str, b: str) -> tuple[str, ...]:
        a_id, b_id = self.entities[a].entity_id, self.entities[b].entity_id
        return tuple(
            view_id
            for view_id in self.view_ids
            if {a_id, b_id}.issubset(self.visible_by_view[view_id])
        )

    def relation(self, subject: str, reference: str) -> dict[str, Any]:
        ps, pr = self.entities[subject].center_m, self.entities[reference].center_m
        dx, dy = ps[0] - pr[0], ps[1] - pr[1]
        if abs(dx) >= abs(dy):
            relation = "right_of" if dx > 0 else "left_of"
            dominant = abs(dx)
            margin = abs(dx) - abs(dy)
            axis = "x"
        else:
            relation = "in_front_of" if dy > 0 else "behind"
            dominant = abs(dy)
            margin = abs(dy) - abs(dx)
            axis = "y"
        if dominant < 0.25 or margin < 0.4:
            raise TruthCompilationError(
                f"ambiguous relation {subject}/{reference}: dominant={dominant:.3f}, margin={margin:.3f}"
            )
        return {
            "subject": subject,
            "reference": reference,
            "frame_id": FRAME_ID,
            "relation": relation,
            "signed_delta_xy_m": [round(dx, 6), round(dy, 6)],
            "dominant_axis": axis,
            "dominant_delta_m": round(dominant, 6),
            "decision_margin_m": round(margin, 6),
        }

    def distance(self, a: str, b: str) -> float:
        pa, pb = self.entities[a].center_m, self.entities[b].center_m
        return math.sqrt(sum((pa[index] - pb[index]) ** 2 for index in range(3)))

    @staticmethod
    def _rotation_matrix_xyzw(
        rotation_xyzw: tuple[float, float, float, float],
    ) -> tuple[tuple[float, float, float], ...]:
        x, y, z, w = rotation_xyzw
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-8:
            raise TruthCompilationError("OBB rotation quaternion is degenerate")
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        return (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        )

    def _obb_projected_half_extent_xy(
        self, entity_key: str, axis_xy: tuple[float, float]
    ) -> float:
        entity = self.entities[entity_key]
        rotation = self._rotation_matrix_xyzw(entity.obb_rotation_xyzw)
        # Rotation columns are the three local OBB axes expressed in world.
        return sum(
            entity.half_extents_m[index]
            * abs(axis_xy[0] * rotation[0][index] + axis_xy[1] * rotation[1][index])
            for index in range(3)
        )

    def object_anchored_relation(
        self,
        origin: str,
        facing: str,
        target: str,
        *,
        minimum_angle_margin_deg: float = HARD_QUADRANT_MIN_ANGLE_MARGIN_DEG,
    ) -> dict[str, Any]:
        """Classify a target in an object-anchored ego frame with OBB-aware margins.

        A hard quadrant is accepted only when the center direction is at least
        ``minimum_angle_margin_deg`` away from either quadrant boundary and the
        target OBB does not cross either query axis.  Boundary cases remain
        useful diagnostics, but they must be verbalized with a dominant
        direction plus a qualified secondary direction rather than a hard
        four-way label.
        """

        po = self.entities[origin].center_m
        pf = self.entities[facing].center_m
        pt = self.entities[target].center_m
        fx, fy = pf[0] - po[0], pf[1] - po[1]
        norm = math.hypot(fx, fy)
        if norm < 1e-6:
            raise TruthCompilationError("query-frame anchors are coincident")
        forward = (fx / norm, fy / norm)
        right = (forward[1], -forward[0])
        tx, ty = pt[0] - po[0], pt[1] - po[1]
        qx = tx * right[0] + ty * right[1]
        qy = tx * forward[0] + ty * forward[1]
        horizontal = "right" if qx >= 0 else "left"
        vertical = "front" if qy >= 0 else "back"
        quadrant = f"{vertical}-{horizontal}"
        right_extent = self._obb_projected_half_extent_xy(target, right)
        forward_extent = self._obb_projected_half_extent_xy(target, forward)
        right_clearance = abs(qx) - right_extent
        forward_clearance = abs(qy) - forward_extent
        angle_from_forward = math.degrees(math.atan2(abs(qx), abs(qy)))
        angle_from_side = 90.0 - angle_from_forward
        angle_margin = min(angle_from_forward, angle_from_side)
        crosses_left_right = right_clearance <= 0.0
        crosses_front_back = forward_clearance <= 0.0
        hard_quadrant_valid = (
            angle_margin >= minimum_angle_margin_deg
            and not crosses_left_right
            and not crosses_front_back
        )
        primary = horizontal if abs(qx) >= abs(qy) else vertical
        secondary = vertical if primary == horizontal else horizontal
        secondary_ratio = min(abs(qx), abs(qy)) / max(abs(qx), abs(qy), 1e-8)
        secondary_degree = "slight" if secondary_ratio < math.tan(math.radians(22.5)) else "clear"
        return {
            "origin": origin,
            "facing": facing,
            "target": target,
            "query_frame": "object_anchored_ego_xy",
            "quadrant": quadrant,
            "query_xy_m": [round(qx, 6), round(qy, 6)],
            "target_obb_half_extent_query_xy_m": [
                round(right_extent, 6),
                round(forward_extent, 6),
            ],
            "axis_clearance_m": {
                "left_right": round(right_clearance, 6),
                "front_back": round(forward_clearance, 6),
            },
            "axis_crossing": {
                "left_right": crosses_left_right,
                "front_back": crosses_front_back,
            },
            "angle_to_boundaries_deg": {
                "front_back_axis": round(angle_from_forward, 6),
                "left_right_axis": round(angle_from_side, 6),
                "minimum": round(angle_margin, 6),
            },
            "minimum_angle_margin_deg": float(minimum_angle_margin_deg),
            "hard_quadrant_valid": hard_quadrant_valid,
            "qualified_relation": {
                "primary": primary,
                "secondary": secondary,
                "secondary_degree": secondary_degree,
            },
        }

    def ego_quadrant(self, origin: str, facing: str, target: str) -> dict[str, Any]:
        """Backward-compatible name for the OBB-aware relation executor."""

        return self.object_anchored_relation(origin, facing, target)

    def instance_mask_pair_cue(self, view_id: str, a: str, b: str) -> dict[str, Any]:
        """Extract an auditable same-frame horizontal cue from instance masks."""

        path = self.root / "views" / f"{view_id}.sensors.npz"
        if not path.is_file():
            raise TruthCompilationError(f"missing sensor bundle for visual cue: {path}")
        expected_hash = str(self.observation_by_id[view_id]["artifacts"]["instance"]["sha256"])
        if _sha256(path) != expected_hash:
            raise TruthCompilationError(f"sensor bundle hash drift at {view_id}")
        with np.load(path) as sensors:
            instance = np.asarray(sensors["instance_id"])
        height, width = instance.shape
        centroids: dict[str, list[float]] = {}
        pixel_counts: dict[str, int] = {}
        for entity_key in (a, b):
            entity_id = self.entities[entity_key].entity_id
            runtime_id = self.runtime_id_by_entity_id.get(entity_id)
            if runtime_id is None:
                raise TruthCompilationError(f"no runtime id for {entity_key}")
            ys, xs = np.where(instance == runtime_id)
            pixel_count = int(xs.size)
            if pixel_count < 256:
                raise TruthCompilationError(
                    f"visual cue {view_id}/{entity_key} has only {pixel_count} pixels"
                )
            centroids[entity_key] = [round(float(xs.mean()), 3), round(float(ys.mean()), 3)]
            pixel_counts[entity_key] = pixel_count
        signed_gap = centroids[a][0] - centroids[b][0]
        if abs(signed_gap) < MINIMUM_VISUAL_CUE_GAP_PX:
            raise TruthCompilationError(
                f"visual cue {view_id}/{a}/{b} is horizontally ambiguous: {signed_gap:.1f}px"
            )
        return {
            "view_id": view_id,
            "image_size_px": [width, height],
            "entities": [a, b],
            "centroids_px": centroids,
            "visible_pixels": pixel_counts,
            "horizontal_order": f"{a}_{'right_of' if signed_gap > 0 else 'left_of'}_{b}",
            "horizontal_gap_px": round(abs(signed_gap), 3),
        }

    @staticmethod
    def _yaw_degrees(rotation_xyzw: list[float] | tuple[float, ...]) -> float:
        x, y, z, w = (float(value) for value in rotation_xyzw)
        return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def camera_turn_event(self, start_view: str, end_view: str) -> dict[str, Any]:
        start = self.observation_by_id[start_view]["world_from_camera"]
        end = self.observation_by_id[end_view]["world_from_camera"]
        start_yaw = self._yaw_degrees(start["rotation_xyzw"])
        end_yaw = self._yaw_degrees(end["rotation_xyzw"])
        delta = (end_yaw - start_yaw + 180.0) % 360.0 - 180.0
        is_turnaround = abs(abs(delta) - 180.0) <= 5.0
        return {
            "start_view_id": start_view,
            "end_view_id": end_view,
            "yaw_change_deg": round(delta, 6),
            "event": "turnaround" if is_turnaround else "heading_change",
            "verified": is_turnaround,
        }

    def camera_pose_equal(self, a: str, b: str, tolerance: float = 1e-8) -> bool:
        ta = self.observation_by_id[a]["world_from_camera"]
        tb = self.observation_by_id[b]["world_from_camera"]
        va = [*ta["translation_m"], *ta["rotation_xyzw"]]
        vb = [*tb["translation_m"], *tb["rotation_xyzw"]]
        return all(abs(float(x) - float(y)) <= tolerance for x, y in zip(va, vb, strict=True))

    def _claim(
        self,
        turn_id: str,
        claim_key: str,
        *,
        node_id: str,
        kind: str,
        statement_zh: str,
        value: Any,
        required: bool = True,
        views: tuple[str, ...] = (),
        entities: tuple[str, ...] = (),
        source_paths: tuple[str, ...] = (),
        allowed_directions: tuple[str, ...] = (),
        numeric_surfaces: tuple[dict[str, Any], ...] = (),
        epistemic_scope: str = "observed_prefix",
        reasoning_role: str | None = None,
    ) -> dict[str, Any]:
        if reasoning_role is None:
            if kind in {
                "grounding",
                "visible_set",
                "visibility",
                "last_seen",
                "co_visibility",
                "never_observed",
                "visual_cue",
                "route_event",
            }:
                reasoning_role = "cue"
            elif kind in {"frame", "belief_bridge", "perspective_projection"}:
                reasoning_role = "transform"
            elif kind in {"epistemic_status"}:
                reasoning_role = "calibration"
            else:
                reasoning_role = "conclusion"
        if reasoning_role not in {"cue", "transform", "conclusion", "calibration"}:
            raise TruthCompilationError(f"invalid claim reasoning role: {reasoning_role}")
        return {
            "claim_id": f"{turn_id}-{claim_key}",
            "node_id": node_id,
            "kind": kind,
            "statement_zh": statement_zh,
            "value": value,
            "required": required,
            "evidence_view_ids": list(views),
            "evidence_entity_ids": [self.entities[key].entity_id for key in entities],
            "source_paths": list(source_paths),
            "frame_id": FRAME_ID if kind in {"frame", "relation", "metric", "layout"} else None,
            "allowed_directions": list(allowed_directions),
            "numeric_surfaces": list(numeric_surfaces),
            "epistemic_scope": epistemic_scope,
            "reasoning_role": reasoning_role,
        }

    @staticmethod
    def _node(
        node_id: str,
        operation: str,
        inputs: tuple[str, ...],
        input_types: tuple[str, ...],
        output_type: str,
        value: Any,
        variant: str,
    ) -> dict[str, Any]:
        if len(inputs) != len(input_types):
            raise TruthCompilationError(f"typed input mismatch at node {node_id}")
        return {
            "node_id": node_id,
            "operation": operation,
            "inputs": list(inputs),
            "input_types": list(input_types),
            "output_type": output_type,
            "variant": variant,
            "value": value,
        }

    def _program(
        self,
        program_id: str,
        signature: str,
        external_inputs: dict[str, str],
        nodes: list[dict[str, Any]],
        answer_node: str,
    ) -> dict[str, Any]:
        available = dict(external_inputs)
        for node in nodes:
            for source, expected in zip(node["inputs"], node["input_types"], strict=True):
                if available.get(source) != expected:
                    raise TruthCompilationError(
                        f"{program_id}/{node['node_id']}: {source} is {available.get(source)!r}, expected {expected!r}"
                    )
            if node["node_id"] in available:
                raise TruthCompilationError(f"duplicate typed output {node['node_id']}")
            available[node["node_id"]] = node["output_type"]
        if answer_node not in {node["node_id"] for node in nodes}:
            raise TruthCompilationError(f"missing answer node {answer_node}")
        return {
            "schema_version": "epispace.executed_typed_program.v1",
            "program_id": program_id,
            "semantic_signature": signature,
            "external_inputs": external_inputs,
            "nodes": nodes,
            "answer_node": answer_node,
        }

    def _question_blueprint(
        self,
        turn_id: str,
        *,
        intent_zh: str,
        slots: dict[str, str],
        strategies: tuple[str, ...],
        subquestions: int = 1,
        surface_constraints_zh: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        # This payload is the complete question-agent input.  It contains no
        # direction, distance, final answer, claim, or certificate field.
        return {
            "schema_version": "epispace.rsint_answer_blind_blueprint.v1",
            "request_id": f"{turn_id}-question",
            "answer_blind": True,
            "intent_zh": intent_zh,
            "slot_values": slots,
            "required_slots": list(slots),
            "allowed_strategies": list(strategies),
            "required_subquestion_count": subquestions,
            "surface_constraints_zh": list(surface_constraints_zh),
            "frame_surface_zh": FRAME_SURFACE_ZH,
            "forbidden_semantics": [
                "answer",
                "direction_value",
                "distance_value",
                "claim",
                "certificate",
                "oracle",
            ],
        }

    def _turn_common(
        self,
        index: int,
        slug: str,
        new_views: tuple[str, ...],
        capability: dict[str, Any],
        skill_ids: tuple[str, ...],
        program: dict[str, Any],
        blueprint: dict[str, Any],
        claims: list[dict[str, Any]],
        canonical_answer_zh: str,
        answer_key: Any,
        reasoning_contract: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        turn_id = f"rsint17-r{index:02d}-{slug}"
        last = max(self.view_ids.index(view_id) for view_id in new_views) if new_views else 10
        available = self.view_ids[: last + 1]
        if index == 7:
            available = self.view_ids
        required = [claim["claim_id"] for claim in claims if claim["required"]]
        return {
            "turn_id": turn_id,
            "round_index": index,
            "new_view_ids": list(new_views),
            "available_view_ids": list(available),
            "oracle_held_out_view_ids": [
                value for value in self.view_ids if value not in available
            ],
            "evidence_view_ids": sorted(
                {view for claim in claims for view in claim["evidence_view_ids"]},
                key=self.view_ids.index,
            ),
            "capability": capability,
            "answer_skill_ids": list(skill_ids),
            "response_profile": "+".join(skill_ids),
            "program": program,
            "question_blueprint": blueprint,
            "claim_sheet": {
                "schema_version": "epispace.node_claim_sheet.v2",
                "claim_sheet_id": _content_id("rsint-claims", claims),
                "claims": claims,
                "required_claim_ids": required,
                "allowed_claim_ids": [claim["claim_id"] for claim in claims],
                "answer_contract": {
                    "answer_key": _canonical_json(answer_key),
                    "answer_value": answer_key,
                    "canonical_answer_zh": canonical_answer_zh,
                    "status": "unknown" if "无法确定" in canonical_answer_zh else "accepted",
                },
                "reasoning_contract": reasoning_contract,
            },
        }

    def compile_turns(self) -> list[dict[str, Any]]:
        """Execute the frozen seven-turn pilot against source geometry."""

        turns: list[dict[str, Any]] = []

        # Round 1: direct visual inventory.
        turn_id = "rsint17-r01-inventory"
        core = tuple(
            key
            for key in ENTITY_NAMES
            if key != "bed" and self.entities[key].entity_id in self.visible_by_view["view-000"]
        )
        expected_core = ("fridge", "microwave", "dishwasher", "sofa", "coffee_table", "laptop")
        if core != expected_core:
            raise TruthCompilationError(f"round-1 visible inventory changed: {core}")
        nodes = [
            self._node(
                "ground_visible",
                "G",
                ("view",),
                ("view",),
                "entity_set",
                list(core),
                "visible_inventory",
            ),
            self._node(
                "verify_inventory",
                "V",
                ("ground_visible",),
                ("entity_set",),
                "boolean",
                True,
                "instance_mask",
            ),
        ]
        program = self._program(
            "rsint.visual_inventory.v1",
            "G(view)->V_set",
            {"view": "view"},
            nodes,
            "verify_inventory",
        )
        claims = [
            self._claim(
                turn_id,
                "visible-set",
                node_id="ground_visible",
                kind="visible_set",
                statement_zh="第1个视角可见的主要家具和电器为冰箱、微波炉、洗碗机、沙发、咖啡桌和笔记本电脑。",
                value=list(core),
                views=("view-000",),
                entities=core,
                source_paths=("spatial_episode.json#/observations/0/visible_entity_ids",),
            )
        ]
        turns.append(
            self._turn_common(
                1,
                "inventory",
                ("view-000",),
                {
                    "primary": "SR",
                    "supporting": ["grounding"],
                    "sense_nova_subtask": "visual grounding",
                },
                ("visual_inventory",),
                program,
                self._question_blueprint(
                    turn_id,
                    intent_zh="询问第1个视角中能看见的主要家具和电器，不要求穷举小物体。",
                    slots={"view_ref": "第1个视角"},
                    strategies=("direct_observation", "guided_walkthrough"),
                ),
                claims,
                "我能看到冰箱、微波炉、洗碗机、沙发、咖啡桌和笔记本电脑。",
                {"visible_entities": list(core)},
            )
        )

        # Round 2: co-visibility plus allocentric relation and metric.
        turn_id = "rsint17-r02-covis-relation"
        rel = self.relation("oven", "fridge")
        covis = self.co_visible_views("oven", "fridge")
        if covis != ("view-002",) or rel["relation"] != "left_of":
            raise TruthCompilationError(f"round-2 premise changed: covis={covis}, rel={rel}")
        nodes = [
            self._node(
                "ground_oven",
                "G",
                ("view_set", "oven_hint"),
                ("view_set", "entity_hint"),
                "entity",
                "oven",
                "cross_view_grounding",
            ),
            self._node(
                "ground_fridge",
                "G",
                ("view_set", "fridge_hint"),
                ("view_set", "entity_hint"),
                "entity",
                "fridge",
                "cross_view_grounding",
            ),
            self._node(
                "align_room",
                "F",
                ("camera_frames", "room_frame"),
                ("frame_set", "frame"),
                "transform_set",
                FRAME_ID,
                "ego_to_allocentric",
            ),
            self._node(
                "pair_belief",
                "B",
                ("ground_oven", "ground_fridge", "align_room"),
                ("entity", "entity", "transform_set"),
                "belief",
                {"co_visible": list(covis)},
                "pair_update",
            ),
            self._node(
                "metric_delta",
                "M",
                ("ground_oven", "ground_fridge", "align_room"),
                ("entity", "entity", "transform_set"),
                "metric",
                rel["dominant_delta_m"],
                "axis_delta",
            ),
            self._node(
                "relation",
                "R",
                ("pair_belief", "room_frame"),
                ("belief", "frame"),
                "relation",
                rel["relation"],
                "canonical_xy",
            ),
            self._node(
                "verify",
                "V",
                ("relation", "metric_delta"),
                ("relation", "metric"),
                "boolean",
                True,
                "margin_and_visibility",
            ),
        ]
        program = self._program(
            "rsint.covis_metric_relation.v1",
            "G+G->F->B->M+R->V",
            {
                "view_set": "view_set",
                "oven_hint": "entity_hint",
                "fridge_hint": "entity_hint",
                "camera_frames": "frame_set",
                "room_frame": "frame",
            },
            nodes,
            "verify",
        )
        claims = [
            self._claim(
                turn_id,
                "bind-oven",
                node_id="ground_oven",
                kind="grounding",
                statement_zh="烤箱在第3个视角中被可靠定位。",
                value="oven",
                views=("view-002",),
                entities=("oven",),
                source_paths=("spatial_episode.json#/observations/2/visible_entity_ids",),
            ),
            self._claim(
                turn_id,
                "bind-fridge",
                node_id="ground_fridge",
                kind="grounding",
                statement_zh="冰箱在第3个视角中被可靠定位。",
                value="fridge",
                views=("view-002",),
                entities=("fridge",),
                source_paths=("spatial_episode.json#/observations/2/visible_entity_ids",),
            ),
            self._claim(
                turn_id,
                "co-visible",
                node_id="pair_belief",
                kind="co_visibility",
                statement_zh="烤箱和冰箱只在第3个视角同时可见。",
                value=list(covis),
                views=covis,
                entities=("oven", "fridge"),
                source_paths=("spatial_episode.json#/observations/2/visible_entity_ids",),
            ),
            self._claim(
                turn_id,
                "frame",
                node_id="align_room",
                kind="frame",
                statement_zh=FRAME_SURFACE_ZH + "。",
                value={"right": "+X", "front": "+Y"},
                views=("view-001", "view-002"),
                source_paths=("scene_ir.json#/canonical_frame",),
            ),
            self._claim(
                turn_id,
                "left",
                node_id="relation",
                kind="relation",
                statement_zh="烤箱在冰箱左侧。",
                value=rel,
                views=("view-002",),
                entities=("oven", "fridge"),
                source_paths=(
                    self.entities["oven"].source_path,
                    self.entities["fridge"].source_path,
                ),
                allowed_directions=("左侧",),
            ),
            self._claim(
                turn_id,
                "delta",
                node_id="metric_delta",
                kind="metric",
                statement_zh="两者在主导横向上的间隔约为1.7米。",
                value=rel["dominant_delta_m"],
                views=("view-002",),
                entities=("oven", "fridge"),
                source_paths=(
                    self.entities["oven"].source_path,
                    self.entities["fridge"].source_path,
                ),
                numeric_surfaces=(
                    {
                        "value": rel["dominant_delta_m"],
                        "unit": "m",
                        "tolerance": 0.1,
                        "surface": "约1.7米",
                    },
                ),
            ),
        ]
        turns.append(
            self._turn_common(
                2,
                "covis-relation",
                ("view-001", "view-002"),
                {
                    "primary": "SR",
                    "supporting": ["MM"],
                    "sense_nova_subtask": "co-visibility + allocentric relation + measurement",
                },
                ("co_visibility_then_metric",),
                program,
                self._question_blueprint(
                    turn_id,
                    intent_zh="询问烤箱和冰箱是否在新增画面中同时出现；若同时出现，再询问烤箱相对冰箱的平面图方向。",
                    slots={"subject": "烤箱", "reference": "冰箱", "view_refs": "第2、3个视角"},
                    strategies=("guided_walkthrough", "co_visibility_probe"),
                    subquestions=2,
                ),
                claims,
                "能，第3个视角里两者同时可见。烤箱在冰箱左侧，横向相距约1.7米。",
                {
                    "co_visible_view_number": 3,
                    "relation": "left_of",
                    "dominant_delta_m": round(rel["dominant_delta_m"], 3),
                },
            )
        )

        # Round 3: temporal belief update and last-seen readout.
        turn_id = "rsint17-r03-last-seen"
        fridge_views = self.visible_views("fridge")
        prefix = self.view_ids[:5]
        prefix_seen = tuple(value for value in prefix if value in fridge_views)
        last_seen = prefix_seen[-1]
        if last_seen != "view-002" or any(
            value in fridge_views for value in ("view-003", "view-004")
        ):
            raise TruthCompilationError("round-3 last-seen premise changed")
        nodes = [
            self._node(
                "track_fridge",
                "G",
                ("sequence", "fridge_hint"),
                ("view_sequence", "entity_hint"),
                "entity_track",
                {"visible": list(prefix_seen), "absent_after": ["view-003", "view-004"]},
                "temporal_grounding",
            ),
            self._node(
                "update_memory",
                "B",
                ("belief_0", "track_fridge"),
                ("belief", "entity_track"),
                "belief",
                {"status": "seen_not_current", "last_seen": last_seen},
                "incremental_memory",
            ),
            self._node(
                "verify_last_seen",
                "V",
                ("update_memory",),
                ("belief",),
                "view_index",
                3,
                "last_seen",
            ),
        ]
        program = self._program(
            "rsint.last_seen_memory.v1",
            "G(sequence)->B_temporal->V_last_seen",
            {"sequence": "view_sequence", "fridge_hint": "entity_hint", "belief_0": "belief"},
            nodes,
            "verify_last_seen",
        )
        claims = [
            self._claim(
                turn_id,
                "absent-now",
                node_id="track_fridge",
                kind="visibility",
                statement_zh="冰箱在当前新增的第4、5个视角中都不可见。",
                value={"view-003": False, "view-004": False},
                views=("view-003", "view-004"),
                entities=("fridge",),
                source_paths=(
                    "spatial_episode.json#/observations/3/visible_entity_ids",
                    "spatial_episode.json#/observations/4/visible_entity_ids",
                ),
                numeric_surfaces=(
                    {
                        "value": 4,
                        "unit": "view_index",
                        "tolerance": 0,
                        "surface": "第4个视角",
                    },
                    {
                        "value": 5,
                        "unit": "view_index",
                        "tolerance": 0,
                        "surface": "第5个视角",
                    },
                ),
            ),
            self._claim(
                turn_id,
                "last-seen",
                node_id="update_memory",
                kind="last_seen",
                statement_zh="截至第5个视角，冰箱最后一次出现是在第3个视角。",
                value={"view_id": last_seen, "display_index": 3},
                views=prefix_seen,
                entities=("fridge",),
                source_paths=tuple(
                    f"spatial_episode.json#/observations/{self.view_ids.index(value)}/visible_entity_ids"
                    for value in prefix
                ),
                numeric_surfaces=(
                    {
                        "value": 3,
                        "unit": "view_index",
                        "tolerance": 0,
                        "surface": "第3个视角",
                    },
                ),
            ),
        ]
        turns.append(
            self._turn_common(
                3,
                "last-seen",
                ("view-003", "view-004"),
                {
                    "primary": "CR",
                    "supporting": ["temporal_memory"],
                    "sense_nova_subtask": "episodic last-seen recall",
                },
                ("temporal_recall",),
                program,
                self._question_blueprint(
                    turn_id,
                    intent_zh="询问当前新增画面里是否还能看到冰箱，并追问它最后一次出现在哪个已给出的视角。",
                    slots={"target": "冰箱", "view_refs": "第4、5个视角"},
                    strategies=("temporal_recall", "guided_walkthrough"),
                    subquestions=2,
                ),
                claims,
                "现在看不到冰箱了；最后一次看到它是在第3个视角。",
                {"currently_visible": False, "last_seen_view_number": 3},
            )
        )

        # Round 4: the decisive non-co-visible, cross-view map read.
        turn_id = "rsint17-r04-crossview"
        rel = self.relation("standing_tv", "fridge")
        distance = self.distance("standing_tv", "fridge")
        covis = self.co_visible_views("standing_tv", "fridge")
        if covis or rel["relation"] != "behind":
            raise TruthCompilationError(f"round-4 cross-view premise changed: {covis}/{rel}")
        tv_views = tuple(
            value for value in self.visible_views("standing_tv") if self.view_ids.index(value) <= 7
        )
        fridge_prefix_views = tuple(
            value for value in self.visible_views("fridge") if self.view_ids.index(value) <= 7
        )
        outbound_bridge_cue = self.instance_mask_pair_cue(
            "view-001", "fridge", "sofa"
        )
        return_bridge_cue = self.instance_mask_pair_cue(
            "view-005", "standing_tv", "sofa"
        )
        turnaround = self.camera_turn_event("view-004", "view-005")
        if turnaround["verified"] is not True:
            raise TruthCompilationError(f"round-4 turnaround is not certified: {turnaround}")
        nodes = [
            self._node(
                "ground_fridge_track",
                "G",
                ("outbound_views", "fridge_hint"),
                ("view_set", "entity_hint"),
                "entity_track",
                {"entity": "fridge", "views": list(fridge_prefix_views)},
                "route_segment",
            ),
            self._node(
                "ground_tv_track",
                "G",
                ("return_views", "tv_hint"),
                ("view_set", "entity_hint"),
                "entity_track",
                {"entity": "standing_tv", "views": list(tv_views)},
                "route_segment",
            ),
            self._node(
                "ground_bridge_cues",
                "G",
                ("outbound_views", "return_views", "bridge_hint"),
                ("view_set", "view_set", "entity_hint"),
                "visual_cue_set",
                {
                    "bridge_entity": "sofa",
                    "outbound": outbound_bridge_cue,
                    "return": return_bridge_cue,
                },
                "instance_mask_bridge",
            ),
            self._node(
                "register_route",
                "F",
                ("trajectory_poses", "room_frame", "ground_bridge_cues"),
                ("transform_chain", "frame", "visual_cue_set"),
                "transform_chain",
                {
                    "registered_views": list(self.view_ids[:8]),
                    "closed_frame": FRAME_ID,
                    "turnaround": turnaround,
                    "bridge_entity": "sofa",
                },
                "cross_view_chain",
            ),
            self._node(
                "global_belief",
                "B",
                ("belief_0", "ground_fridge_track", "ground_tv_track", "register_route"),
                ("belief", "entity_track", "entity_track", "transform_chain"),
                "belief",
                {"entities": ["fridge", "standing_tv"], "co_visible": []},
                "global_register",
            ),
            self._node(
                "metric_delta",
                "M",
                ("global_belief",),
                ("belief",),
                "metric",
                {"dominant_delta_m": rel["dominant_delta_m"], "distance_m": round(distance, 6)},
                "cross_view_metric",
            ),
            self._node(
                "relation",
                "R",
                ("global_belief", "room_frame"),
                ("belief", "frame"),
                "relation",
                {"primary": "behind", "secondary": "right_of"},
                "canonical_xy",
            ),
            self._node(
                "verify",
                "V",
                ("relation", "metric_delta"),
                ("relation", "metric"),
                "boolean",
                True,
                "non_covis_reexecution",
            ),
        ]
        program = self._program(
            "rsint.crossview_relation.v2",
            "G+G+G->F*->B->R+M->V",
            {
                "outbound_views": "view_set",
                "return_views": "view_set",
                "fridge_hint": "entity_hint",
                "tv_hint": "entity_hint",
                "bridge_hint": "entity_hint",
                "trajectory_poses": "transform_chain",
                "room_frame": "frame",
                "belief_0": "belief",
            },
            nodes,
            "verify",
        )
        claims = [
            self._claim(
                turn_id,
                "bind-fridge",
                node_id="ground_fridge_track",
                kind="grounding",
                statement_zh="冰箱在去程画面中被观察到。",
                value=list(fridge_prefix_views),
                views=fridge_prefix_views,
                entities=("fridge",),
                source_paths=tuple(
                    f"spatial_episode.json#/observations/{self.view_ids.index(value)}/visible_entity_ids"
                    for value in fridge_prefix_views
                ),
                required=False,
            ),
            self._claim(
                turn_id,
                "bind-tv",
                node_id="ground_tv_track",
                kind="grounding",
                statement_zh="电视在返程画面中被观察到。",
                value=list(tv_views),
                views=tv_views,
                entities=("standing_tv",),
                source_paths=tuple(
                    f"spatial_episode.json#/observations/{self.view_ids.index(value)}/visible_entity_ids"
                    for value in tv_views
                ),
                required=False,
            ),
            self._claim(
                turn_id,
                "no-covis",
                node_id="global_belief",
                kind="co_visibility",
                statement_zh="截至第8个视角，电视和冰箱从未在同一视角中同时出现。",
                value=[],
                views=fridge_prefix_views + tv_views,
                entities=("standing_tv", "fridge"),
                source_paths=tuple(
                    f"spatial_episode.json#/observations/{index}/visible_entity_ids"
                    for index in range(8)
                ),
            ),
            self._claim(
                turn_id,
                "visual-bridge",
                node_id="ground_bridge_cues",
                kind="visual_cue",
                statement_zh="第2个视角中冰箱与沙发同框；掉头后的第6个视角里，同一张沙发又与电视同框。",
                value={"outbound": outbound_bridge_cue, "return": return_bridge_cue},
                views=("view-001", "view-005"),
                entities=("fridge", "sofa", "standing_tv"),
                source_paths=(
                    "views/view-001.sensors.npz#/instance_id",
                    "views/view-005.sensors.npz#/instance_id",
                ),
                numeric_surfaces=(
                    {"value": 2, "unit": "view_index", "tolerance": 0, "surface": "第2个视角"},
                    {"value": 6, "unit": "view_index", "tolerance": 0, "surface": "第6个视角"},
                ),
            ),
            self._claim(
                turn_id,
                "turnaround",
                node_id="register_route",
                kind="route_event",
                statement_zh="从第5个到第6个视角之间发生掉头，随后进入返程。",
                value=turnaround,
                views=("view-004", "view-005"),
                source_paths=(
                    "spatial_episode.json#/observations/4/world_from_camera",
                    "spatial_episode.json#/observations/5/world_from_camera",
                    "trajectory_plan.json#/views/5/role",
                ),
                numeric_surfaces=(
                    {"value": 5, "unit": "view_index", "tolerance": 0, "surface": "第5个视角"},
                    {"value": 6, "unit": "view_index", "tolerance": 0, "surface": "第6个视角"},
                ),
            ),
            self._claim(
                turn_id,
                "route-align",
                node_id="register_route",
                kind="belief_bridge",
                statement_zh="借助重复出现的沙发和中途掉头，可把去程的冰箱观察与返程的电视观察接到同一空间布局中。",
                value={
                    "registered_view_ids": list(self.view_ids[:8]),
                    "bridge_entity": "sofa",
                    "turnaround": turnaround,
                },
                views=self.view_ids[:8],
                entities=("fridge", "sofa", "standing_tv"),
                source_paths=("trajectory_plan.json#/views", "spatial_episode.json#/observations"),
            ),
            self._claim(
                turn_id,
                "behind",
                node_id="relation",
                kind="relation",
                statement_zh="在房间平面图中，电视位于冰箱后方，并略偏右。",
                value=rel,
                views=("view-002", "view-005", "view-006", "view-007"),
                entities=("standing_tv", "fridge"),
                source_paths=(
                    self.entities["standing_tv"].source_path,
                    self.entities["fridge"].source_path,
                ),
                allowed_directions=("后方", "右侧"),
            ),
            self._claim(
                turn_id,
                "delta-y",
                node_id="metric_delta",
                kind="metric",
                statement_zh="电视相对冰箱的纵向间隔约为4.7米。",
                value=rel["dominant_delta_m"],
                views=("view-002", "view-005", "view-006", "view-007"),
                entities=("standing_tv", "fridge"),
                source_paths=(
                    self.entities["standing_tv"].source_path,
                    self.entities["fridge"].source_path,
                ),
                numeric_surfaces=(
                    {
                        "value": rel["dominant_delta_m"],
                        "unit": "m",
                        "tolerance": 0.1,
                        "surface": "约4.7米",
                    },
                ),
            ),
            self._claim(
                turn_id,
                "distance",
                node_id="metric_delta",
                kind="metric",
                statement_zh="电视与冰箱的直线距离约为4.9米。",
                value=round(distance, 6),
                required=False,
                views=("view-002", "view-005", "view-006", "view-007"),
                entities=("standing_tv", "fridge"),
                source_paths=(
                    self.entities["standing_tv"].source_path,
                    self.entities["fridge"].source_path,
                ),
                numeric_surfaces=(
                    {"value": distance, "unit": "m", "tolerance": 0.1, "surface": "约4.9米"},
                ),
            ),
        ]
        turns.append(
            self._turn_common(
                4,
                "crossview",
                ("view-005", "view-006", "view-007"),
                {
                    "primary": "CR",
                    "supporting": ["PT", "SR", "MM"],
                    "sense_nova_subtask": "non-co-visible cross-view reconstruction",
                },
                ("route_replay", "landmark_hierarchy"),
                program,
                self._question_blueprint(
                    turn_id,
                    intent_zh="先询问电视和冰箱截至目前是否曾经同框，再要求综合此前所有观察判断电视相对冰箱的平面图方向。",
                    slots={"subject": "电视", "reference": "冰箱", "view_refs": "第6、7、8个视角"},
                    strategies=("route_replay", "evidence_contrast"),
                    subquestions=2,
                ),
                claims,
                "电视和冰箱没有同框过。第2个视角里冰箱和沙发同框；掉头后的第6个视角里，同一张沙发又和电视同框，所以可以用沙发把前后观察接起来。接起来看，电视在冰箱后方约4.7米，并略偏右。",
                {
                    "co_visible": False,
                    "relation": "behind",
                    "secondary": "right_of",
                    "dominant_delta_m": round(rel["dominant_delta_m"], 3),
                },
                {
                    "required_sentence_roles": ["evidence", "transform", "conclusion"],
                    "enforce_claim_role_alignment": True,
                },
            )
        )

        # Round 5: robust object-anchored perspective taking.  The old sofa
        # target is retained only as a rejected boundary diagnostic: its OBB
        # crosses the front/back axis and its center is only 10 degrees from
        # that boundary.  The fridge target has a clean OBB and angular margin
        # and is never co-visible with either query-frame anchor.
        turn_id = "rsint17-r05-perspective"
        perspective = self.object_anchored_relation(
            "swivel_chair", "standing_tv", "fridge"
        )
        sofa_boundary_probe = self.object_anchored_relation(
            "swivel_chair", "standing_tv", "sofa"
        )
        if perspective["quadrant"] != "front-left" or not perspective["hard_quadrant_valid"]:
            raise TruthCompilationError(f"round-5 robust perspective premise changed: {perspective}")
        if sofa_boundary_probe["hard_quadrant_valid"]:
            raise TruthCompilationError(
                f"round-5 boundary diagnostic unexpectedly became valid: {sofa_boundary_probe}"
            )
        self.perspective_sampling_diagnostics = {
            "accepted_training_target": perspective,
            "rejected_boundary_probe": {
                "disposition": "qualified_language_only_not_hard_quadrant",
                "relation": sofa_boundary_probe,
            },
        }
        fridge_views = tuple(
            view_id
            for view_id in self.visible_views("fridge")
            if self.view_ids.index(view_id) <= 9
        )
        anchor_views = tuple(
            view_id
            for view_id in self.view_ids[:10]
            if {
                self.entities["swivel_chair"].entity_id,
                self.entities["standing_tv"].entity_id,
            }.issubset(self.visible_by_view[view_id])
        )
        if fridge_views != ("view-000", "view-001", "view-002") or not anchor_views:
            raise TruthCompilationError(
                f"round-5 evidence tracks changed: fridge={fridge_views}, anchors={anchor_views}"
            )
        no_anchor_target_covis = not self.co_visible_views("fridge", "swivel_chair") and not self.co_visible_views(
            "fridge", "standing_tv"
        )
        if not no_anchor_target_covis:
            raise TruthCompilationError("round-5 target is co-visible with a query-frame anchor")
        anchor_pair_cue = self.instance_mask_pair_cue(
            "view-005", "standing_tv", "swivel_chair"
        )
        nodes = [
            self._node(
                "ground_target_track",
                "G",
                ("view_prefix", "target_hint"),
                ("view_set", "entity_hint"),
                "entity_track",
                {"entity": "fridge", "views": list(fridge_views)},
                "early_route_segment",
            ),
            self._node(
                "ground_query_anchors",
                "G",
                ("view_prefix", "anchor_hints"),
                ("view_set", "entity_hint_set"),
                "visual_cue_set",
                {
                    "entities": ["swivel_chair", "standing_tv"],
                    "joint_views": list(anchor_views),
                    "decisive_mask_cue": anchor_pair_cue,
                },
                "same_frame_anchor_pair",
            ),
            self._node(
                "global_belief",
                "B",
                ("belief_0", "ground_target_track", "ground_query_anchors", "route_chain"),
                ("belief", "entity_track", "visual_cue_set", "transform_chain"),
                "belief",
                {
                    "entities": ["swivel_chair", "standing_tv", "fridge"],
                    "target_anchor_co_visible": False,
                    "route_bridge": "outbound_to_return",
                },
                "persistent_cross_segment_map",
            ),
            self._node(
                "query_frame",
                "F",
                ("global_belief",),
                ("belief",),
                "transform",
                {"origin": "swivel_chair", "forward_target": "standing_tv"},
                "object_anchored_ego",
            ),
            self._node(
                "predict_view",
                "P",
                ("global_belief", "query_frame"),
                ("belief", "transform"),
                "perspective_projection",
                perspective,
                "obb_aware_perspective",
            ),
            self._node(
                "relation",
                "R",
                ("predict_view",),
                ("perspective_projection",),
                "relation",
                perspective["quadrant"],
                "ego_quadrant",
            ),
            self._node(
                "verify",
                "V",
                ("relation", "predict_view"),
                ("relation", "perspective_projection"),
                "boolean",
                perspective["hard_quadrant_valid"],
                "obb_and_angular_margin",
            ),
        ]
        program = self._program(
            "rsint.object_anchored_perspective.v2",
            "G+G->B->F_query->P->R->V",
            {
                "view_prefix": "view_set",
                "target_hint": "entity_hint",
                "anchor_hints": "entity_hint_set",
                "belief_0": "belief",
                "route_chain": "transform_chain",
            },
            nodes,
            "verify",
        )
        claims = [
            self._claim(
                turn_id,
                "target-track",
                node_id="ground_target_track",
                kind="grounding",
                statement_zh="冰箱出现在最初三个去程视角中，之后的返程画面不再直接显示它。",
                value={"entity": "fridge", "views": list(fridge_views)},
                views=fridge_views,
                entities=("fridge",),
                source_paths=tuple(
                    f"spatial_episode.json#/observations/{self.view_ids.index(view_id)}/visible_entity_ids"
                    for view_id in self.view_ids[:10]
                ),
            ),
            self._claim(
                turn_id,
                "anchor-cue",
                node_id="ground_query_anchors",
                kind="visual_cue",
                statement_zh="第6个视角中转椅和电视同时可见，可确定假想视角的站位与面向。",
                value=anchor_pair_cue,
                views=("view-005",),
                entities=("swivel_chair", "standing_tv"),
                source_paths=("views/view-005.sensors.npz#/instance_id",),
                numeric_surfaces=(
                    {"value": 6, "unit": "view_index", "tolerance": 0, "surface": "第6个视角"},
                ),
            ),
            self._claim(
                turn_id,
                "no-single-frame",
                node_id="global_belief",
                kind="co_visibility",
                statement_zh="截至当前，没有任何一个视角让冰箱与转椅或电视同时出现。",
                value={"fridge_chair": [], "fridge_tv": []},
                views=fridge_views + anchor_views,
                entities=("fridge", "swivel_chair", "standing_tv"),
                source_paths=("spatial_episode.json#/observations",),
            ),
            self._claim(
                turn_id,
                "shared-map",
                node_id="global_belief",
                kind="belief_bridge",
                statement_zh="冰箱和转椅、电视分属前后两段观察，需要沿连续路线接到同一空间布局中。",
                value={
                    "outbound_entity": "fridge",
                    "return_entities": ["swivel_chair", "standing_tv"],
                    "route_bridge": "continuous_outbound_return",
                },
                views=self.view_ids[:10],
                entities=("fridge", "swivel_chair", "standing_tv"),
                source_paths=("trajectory_plan.json#/views", "spatial_episode.json#/observations"),
            ),
            self._claim(
                turn_id,
                "query-frame",
                node_id="query_frame",
                kind="frame",
                statement_zh="假想坐在转椅上并转身面向电视，以指向电视的方向作为新的正前方。",
                value={"origin": "swivel_chair", "facing": "standing_tv"},
                views=("view-005",),
                entities=("swivel_chair", "standing_tv"),
                source_paths=(
                    self.entities["swivel_chair"].source_path,
                    self.entities["standing_tv"].source_path,
                ),
                allowed_directions=("正前方",),
            ),
            self._claim(
                turn_id,
                "projection",
                node_id="predict_view",
                kind="perspective_projection",
                statement_zh="换到这个朝向后，冰箱明确落在左侧和前方。",
                value=perspective,
                views=(*fridge_views, "view-005"),
                entities=("fridge", "swivel_chair", "standing_tv"),
                source_paths=tuple(
                    self.entities[key].source_path
                    for key in ("fridge", "swivel_chair", "standing_tv")
                ),
                allowed_directions=("左侧", "前方"),
            ),
            self._claim(
                turn_id,
                "quadrant",
                node_id="relation",
                kind="relation",
                statement_zh="该位置可稳健归为左前方。",
                value=perspective,
                views=(*fridge_views, "view-005"),
                entities=("fridge", "swivel_chair", "standing_tv"),
                source_paths=tuple(
                    self.entities[key].source_path
                    for key in ("fridge", "swivel_chair", "standing_tv")
                ),
                allowed_directions=("左前方",),
            ),
        ]
        turns.append(
            self._turn_common(
                5,
                "perspective",
                ("view-008", "view-009"),
                {
                    "primary": "PT",
                    "supporting": ["SR", "CR"],
                    "sense_nova_subtask": "object-anchored perspective taking",
                },
                ("mental_simulation", "query_frame"),
                program,
                self._question_blueprint(
                    turn_id,
                    intent_zh="要求依据截至当前的全部观察，把视点切换到坐在转椅并面向电视的位置，判断此前看到的冰箱在该视角中的方向。",
                    slots={
                        "memory_scope": "到目前为止的观察",
                        "origin": "转椅",
                        "facing_target": "电视",
                        "query_target": "冰箱",
                    },
                    strategies=("embodied_imagining", "object_anchored_query"),
                    surface_constraints_zh=(
                        "用坐在转椅上并转身面向电视的自然说法，不得写成站在物体中心。",
                        "不得声称当前新增图像直接显示冰箱；问题应明确依赖此前空间记忆。",
                    ),
                ),
                claims,
                "第6个视角里转椅和电视同框，但冰箱只在较早的去程画面中出现，没有任何一帧把它们放在一起。先沿路线把这些位置接起来，再想象坐到转椅上转身面向电视；换到这个朝向后，冰箱明确位于左侧且在前方。因此，它在你的左前方。",
                {"quadrant": "front-left", "hard_quadrant_valid": True},
                {
                    "required_sentence_roles": ["evidence", "transform", "conclusion"],
                    "enforce_claim_role_alignment": True,
                },
            )
        )

        # Round 6: calibrated absence plus exact loop closure and state persistence.
        turn_id = "rsint17-r06-loop-memory"
        bed_views = self.visible_views("bed")
        if bed_views:
            raise TruthCompilationError(f"bed unexpectedly observed: {bed_views}")
        if not self.camera_pose_equal("view-000", "view-010"):
            raise TruthCompilationError("view-010 is not an exact loop closure to view-000")
        fridge_final = self.entities["fridge"].entity_id in self.visible_by_view["view-010"]
        if not fridge_final:
            raise TruthCompilationError("fridge is not visible after loop closure")
        nodes = [
            self._node(
                "track_bed",
                "G",
                ("full_sequence", "bed_hint"),
                ("view_sequence", "entity_hint"),
                "entity_track",
                {"observed_views": []},
                "negative_evidence",
            ),
            self._node(
                "bed_belief",
                "B",
                ("belief_0", "track_bed"),
                ("belief", "entity_track"),
                "belief",
                {"status": "unobserved_unknown"},
                "epistemic_update",
            ),
            self._node(
                "verify_unknown",
                "V",
                ("bed_belief",),
                ("belief",),
                "epistemic_status",
                "unknown",
                "observed_scope",
            ),
            self._node(
                "align_loop",
                "F",
                ("start_pose", "end_pose"),
                ("camera_pose", "camera_pose"),
                "transform",
                {"translation_error_m": 0.0, "rotation_error": 0.0},
                "loop_closure",
            ),
            self._node(
                "fridge_belief",
                "B",
                ("belief_0", "align_loop"),
                ("belief", "transform"),
                "belief",
                {"entity": "fridge", "persistent": True, "visible_at_end": True},
                "reobservation",
            ),
            self._node(
                "verify_persistence",
                "V",
                ("fridge_belief",),
                ("belief",),
                "boolean",
                True,
                "loop_revisit",
            ),
        ]
        program = self._program(
            "rsint.loop_unknown_persistence.v1",
            "G->B->V_unknown + F_loop->B->V_persist",
            {
                "full_sequence": "view_sequence",
                "bed_hint": "entity_hint",
                "belief_0": "belief",
                "start_pose": "camera_pose",
                "end_pose": "camera_pose",
            },
            nodes,
            "verify_persistence",
        )
        claims = [
            self._claim(
                turn_id,
                "bed-unobserved",
                node_id="track_bed",
                kind="never_observed",
                statement_zh="床在已经提供的11个视角中从未出现。",
                value={"observed_view_ids": []},
                views=self.view_ids,
                source_paths=tuple(
                    f"spatial_episode.json#/observations/{index}/visible_entity_ids"
                    for index in range(11)
                ),
                numeric_surfaces=(
                    {
                        "value": 11,
                        "unit": "view_count",
                        "tolerance": 0,
                        "surface": "11个视角",
                    },
                ),
                epistemic_scope="provided_views_only",
            ),
            self._claim(
                turn_id,
                "bed-unknown",
                node_id="verify_unknown",
                kind="epistemic",
                statement_zh="未观察到床只能说明给定画面证据不足，不能据此断言整间住宅没有床。",
                value="unknown",
                views=self.view_ids,
                source_paths=("spatial_episode.json#/observations",),
                epistemic_scope="provided_views_only",
            ),
            self._claim(
                turn_id,
                "loop",
                node_id="align_loop",
                kind="frame",
                statement_zh="第11个视角回到了与第1个视角相同的观察位置和朝向。",
                value={"start": "view-000", "end": "view-010", "pose_equal": True},
                views=("view-000", "view-010"),
                source_paths=(
                    "spatial_episode.json#/observations/0/world_from_camera",
                    "spatial_episode.json#/observations/10/world_from_camera",
                ),
                numeric_surfaces=(
                    {
                        "value": 1,
                        "unit": "view_index",
                        "tolerance": 0,
                        "surface": "第1个视角",
                    },
                    {
                        "value": 11,
                        "unit": "view_index",
                        "tolerance": 0,
                        "surface": "第11个视角",
                    },
                ),
            ),
            self._claim(
                turn_id,
                "fridge-return",
                node_id="fridge_belief",
                kind="state_persistence",
                statement_zh="闭环回到起点后再次看到了同一位置的冰箱。",
                value={
                    "visible_at_view_000": True,
                    "visible_at_view_010": True,
                    "static_world_position": list(self.entities["fridge"].center_m),
                },
                views=("view-000", "view-010"),
                entities=("fridge",),
                source_paths=(
                    "spatial_episode.json#/observations/0/visible_entity_ids",
                    "spatial_episode.json#/observations/10/visible_entity_ids",
                    self.entities["fridge"].source_path,
                ),
                epistemic_scope="static_simulator_episode",
            ),
        ]
        turns.append(
            self._turn_common(
                6,
                "loop-memory",
                ("view-010",),
                {
                    "primary": "CR",
                    "supporting": ["PT", "epistemic_calibration"],
                    "sense_nova_subtask": "loop persistence + evidence status",
                },
                ("calibration", "temporal_recall"),
                program,
                self._question_blueprint(
                    turn_id,
                    intent_zh="在回到起点后，先询问此前的观察中是否见过床，再询问冰箱是否仍在最初的位置。",
                    slots={
                        "target_entity": "床",
                        "persistent_target": "冰箱",
                        "view_ref": "第11个视角",
                    },
                    strategies=("calibrated_recall", "loop_revisit"),
                    subquestions=2,
                ),
                claims,
                "我走过的视角里从未看到床，因此无法确定整间住宅里有没有床。闭环回到起点后，冰箱仍在最初的位置。",
                {"bed_status": "unknown", "fridge_persistent": True, "loop_closed": True},
            )
        )

        # Round 7: natural-language state commit from accepted layout claims.
        turn_id = "rsint17-r07-layout"
        fridge_mw = self.relation("fridge", "microwave")
        oven_mw = self.relation("oven", "microwave")
        sofa_coffee = self.relation("sofa", "coffee_table")
        tv_sofa = self.relation("standing_tv", "sofa")
        tv_fridge_distance = self.distance("standing_tv", "fridge")
        mw, dw = self.entities["microwave"], self.entities["dishwasher"]
        stack_horizontal = math.hypot(
            mw.center_m[0] - dw.center_m[0], mw.center_m[1] - dw.center_m[1]
        )
        stack_vertical = mw.center_m[2] - dw.center_m[2]
        coffee = self.entities["coffee_table"].center_m
        laptop = self.entities["laptop"].center_m
        near_horizontal = math.hypot(coffee[0] - laptop[0], coffee[1] - laptop[1])
        table_top = coffee[2] + self.entities["coffee_table"].half_extents_m[2]
        laptop_bottom = laptop[2] - self.entities["laptop"].half_extents_m[2]
        support_gap = laptop_bottom - table_top
        inside_table_footprint = (
            abs(laptop[0] - coffee[0]) <= self.entities["coffee_table"].half_extents_m[0]
            and abs(laptop[1] - coffee[1])
            <= self.entities["coffee_table"].half_extents_m[1]
        )
        if not (fridge_mw["relation"] == "right_of" and oven_mw["relation"] == "behind"):
            raise TruthCompilationError("round-7 kitchen layout changed")
        if not (sofa_coffee["relation"] == "left_of" and tv_sofa["relation"] == "right_of"):
            raise TruthCompilationError("round-7 living layout changed")
        if not (
            stack_horizontal < 0.05
            and stack_vertical > 0.25
            and near_horizontal < 0.5
            and inside_table_footprint
            and 0 <= support_gap <= 0.05
        ):
            raise TruthCompilationError("round-7 stack/near premises changed")
        sofa_x = self.entities["sofa"].center_m[0]
        tv_x = self.entities["standing_tv"].center_m[0]
        if not min(sofa_x, tv_x) < coffee[0] < max(sofa_x, tv_x):
            raise TruthCompilationError("coffee table no longer lies between sofa and TV along X")
        nodes = [
            self._node(
                "ground_landmarks",
                "G",
                ("full_sequence", "landmark_hints"),
                ("view_sequence", "entity_hint_set"),
                "entity_set",
                [key for key in ENTITY_NAMES if key != "bed"],
                "episode_grounding",
            ),
            self._node(
                "align_all",
                "F",
                ("trajectory_poses", "room_frame"),
                ("transform_chain", "frame"),
                "transform_chain",
                {"registered_views": list(self.view_ids)},
                "full_episode",
            ),
            self._node(
                "commit_state",
                "B",
                ("belief_0", "ground_landmarks", "align_all"),
                ("belief", "entity_set", "transform_chain"),
                "canonical_state",
                {"observed_entities": 9, "regions": ["kitchen_0", "living_room_0"]},
                "state_commit",
            ),
            self._node(
                "layout_metrics",
                "M",
                ("commit_state",),
                ("canonical_state",),
                "metric_set",
                {"tv_fridge_distance_m": round(tv_fridge_distance, 6)},
                "layout_distance",
            ),
            self._node(
                "layout_relations",
                "R",
                ("commit_state", "room_frame"),
                ("canonical_state", "frame"),
                "relation_set",
                {
                    "fridge_microwave": "right_of",
                    "oven_microwave": "behind",
                    "sofa_coffee": "left_of",
                    "tv_sofa": "right_of",
                },
                "layout_graph",
            ),
            self._node(
                "verify_summary",
                "V",
                ("layout_relations", "layout_metrics"),
                ("relation_set", "metric_set"),
                "boolean",
                True,
                "claim_coverage",
            ),
        ]
        program = self._program(
            "rsint.layout_state_commit.v1",
            "G*->F*->B_state->R*+M->V",
            {
                "full_sequence": "view_sequence",
                "landmark_hints": "entity_hint_set",
                "trajectory_poses": "transform_chain",
                "room_frame": "frame",
                "belief_0": "belief",
            },
            nodes,
            "verify_summary",
        )
        kitchen_entities = ("microwave", "dishwasher", "oven", "fridge")
        living_entities = ("sofa", "coffee_table", "laptop", "standing_tv", "swivel_chair")
        claims = [
            self._claim(
                turn_id,
                "zones",
                node_id="commit_state",
                kind="layout",
                statement_zh="已观察地标分属厨房区和客厅区两个功能区。",
                value={"kitchen": list(kitchen_entities), "living_room": list(living_entities)},
                views=self.view_ids,
                entities=kitchen_entities + living_entities,
                source_paths=tuple(
                    self.entities[key].source_path for key in kitchen_entities + living_entities
                ),
                epistemic_scope="observed_landmarks",
            ),
            self._claim(
                turn_id,
                "stack",
                node_id="layout_relations",
                kind="vertical_relation",
                statement_zh="微波炉和洗碗机上下叠放，微波炉在上。",
                value={
                    "upper": "microwave",
                    "lower": "dishwasher",
                    "horizontal_offset_m": round(stack_horizontal, 6),
                    "vertical_delta_m": round(stack_vertical, 6),
                },
                views=("view-000", "view-001", "view-002", "view-003", "view-004", "view-010"),
                entities=("microwave", "dishwasher"),
                source_paths=(mw.source_path, dw.source_path),
                allowed_directions=("上方",),
            ),
            self._claim(
                turn_id,
                "oven-behind",
                node_id="layout_relations",
                kind="relation",
                statement_zh="烤箱在微波炉后方。",
                value=oven_mw,
                views=("view-002", "view-003", "view-004"),
                entities=("oven", "microwave"),
                source_paths=(
                    self.entities["oven"].source_path,
                    self.entities["microwave"].source_path,
                ),
                allowed_directions=("后方",),
            ),
            self._claim(
                turn_id,
                "fridge-right",
                node_id="layout_relations",
                kind="relation",
                statement_zh="冰箱在微波炉右侧约1.9米。",
                value=fridge_mw,
                views=("view-000", "view-001", "view-002", "view-010"),
                entities=("fridge", "microwave"),
                source_paths=(
                    self.entities["fridge"].source_path,
                    self.entities["microwave"].source_path,
                ),
                allowed_directions=("右侧",),
                numeric_surfaces=(
                    {
                        "value": fridge_mw["dominant_delta_m"],
                        "unit": "m",
                        "tolerance": 0.1,
                        "surface": "约1.9米",
                    },
                ),
            ),
            self._claim(
                turn_id,
                "sofa-left",
                node_id="layout_relations",
                kind="relation",
                statement_zh="沙发在咖啡桌左侧。",
                value=sofa_coffee,
                views=("view-000", "view-001", "view-005", "view-006", "view-007", "view-010"),
                entities=("sofa", "coffee_table"),
                source_paths=(
                    self.entities["sofa"].source_path,
                    self.entities["coffee_table"].source_path,
                ),
                allowed_directions=("左侧",),
            ),
            self._claim(
                turn_id,
                "tv-right",
                node_id="layout_relations",
                kind="relation",
                statement_zh="电视在沙发右侧。",
                value=tv_sofa,
                views=("view-005", "view-006", "view-007"),
                entities=("standing_tv", "sofa"),
                source_paths=(
                    self.entities["standing_tv"].source_path,
                    self.entities["sofa"].source_path,
                ),
                allowed_directions=("右侧",),
            ),
            self._claim(
                turn_id,
                "coffee-between",
                node_id="layout_relations",
                kind="between",
                statement_zh="沿房间横向看，咖啡桌大致位于沙发与电视之间。",
                value={"middle": "coffee_table", "ends": ["sofa", "standing_tv"], "axis": "x"},
                views=("view-005", "view-006", "view-007"),
                entities=("coffee_table", "sofa", "standing_tv"),
                source_paths=tuple(
                    self.entities[key].source_path
                    for key in ("coffee_table", "sofa", "standing_tv")
                ),
            ),
            self._claim(
                turn_id,
                "laptop-on",
                node_id="layout_relations",
                kind="support_relation",
                statement_zh="笔记本电脑放在咖啡桌上。",
                value={
                    "horizontal_distance_m": round(near_horizontal, 6),
                    "support_gap_m": round(support_gap, 6),
                    "inside_table_footprint": inside_table_footprint,
                },
                views=("view-000", "view-001", "view-005", "view-006", "view-007", "view-010"),
                entities=("laptop", "coffee_table"),
                source_paths=(
                    self.entities["laptop"].source_path,
                    self.entities["coffee_table"].source_path,
                ),
            ),
            self._claim(
                turn_id,
                "chair-behind",
                node_id="layout_relations",
                kind="axis_order",
                statement_zh="沿房间平面图的前后方向，转椅比沙发更靠后。",
                value={
                    "axis": "y",
                    "swivel_chair_y_m": self.entities["swivel_chair"].center_m[1],
                    "sofa_y_m": self.entities["sofa"].center_m[1],
                    "ordered": self.entities["swivel_chair"].center_m[1]
                    < self.entities["sofa"].center_m[1],
                },
                views=("view-005", "view-006", "view-007", "view-008"),
                entities=("swivel_chair", "sofa"),
                source_paths=(
                    self.entities["swivel_chair"].source_path,
                    self.entities["sofa"].source_path,
                ),
                allowed_directions=("更靠后",),
            ),
            self._claim(
                turn_id,
                "tv-fridge-distance",
                node_id="layout_metrics",
                kind="metric",
                statement_zh="电视到冰箱的直线距离约为4.9米。",
                value=round(tv_fridge_distance, 6),
                views=("view-002", "view-005", "view-006", "view-007"),
                entities=("standing_tv", "fridge"),
                source_paths=(
                    self.entities["standing_tv"].source_path,
                    self.entities["fridge"].source_path,
                ),
                numeric_surfaces=(
                    {
                        "value": tv_fridge_distance,
                        "unit": "m",
                        "tolerance": 0.1,
                        "surface": "约4.9米",
                    },
                ),
            ),
        ]
        turns.append(
            self._turn_common(
                7,
                "layout",
                (),
                {
                    "primary": "CR",
                    "supporting": ["SR", "MM"],
                    "sense_nova_subtask": "state sufficiency / layout synthesis",
                },
                ("landmark_hierarchy", "layout_summary"),
                program,
                self._question_blueprint(
                    turn_id,
                    intent_zh="要求根据全部观察，用几句话概括已观察空间的主要布局和关键地标关系。",
                    slots={"scope": "这个空间"},
                    strategies=("layout_summary", "landmark_hierarchy"),
                ),
                claims,
                "这个空间可分为厨房和客厅两个已观察功能区。厨房里微波炉在洗碗机上方，烤箱在其后方，冰箱在微波炉右侧约1.9米；客厅里沙发在咖啡桌左侧、电视在沙发右侧，咖啡桌大致位于二者之间，笔记本电脑在桌上，更靠后还有转椅。电视到冰箱约4.9米。",
                {
                    "layout_claim_count": len(claims),
                    "tv_fridge_distance_m": round(tv_fridge_distance, 3),
                },
            )
        )

        self._validate_schedule(turns)
        return turns

    def _validate_schedule(self, turns: list[dict[str, Any]]) -> None:
        if len(turns) != 7:
            raise TruthCompilationError("dialogue must contain seven rounds")
        released: list[str] = []
        for expected_index, turn in enumerate(turns, 1):
            if turn["round_index"] != expected_index:
                raise TruthCompilationError("round indices are not contiguous")
            for view_id in turn["new_view_ids"]:
                if view_id in released:
                    raise TruthCompilationError(f"view released twice: {view_id}")
                released.append(view_id)
            if turn["available_view_ids"] != released:
                raise TruthCompilationError(
                    f"turn {expected_index} has a non-causal prefix: {turn['available_view_ids']} vs {released}"
                )
            if set(turn["evidence_view_ids"]) - set(released):
                raise TruthCompilationError(f"turn {expected_index} uses future evidence")
        if tuple(released) != self.view_ids:
            raise TruthCompilationError(f"not all views were released exactly once: {released}")

    def compile(self) -> dict[str, Any]:
        turns = self.compile_turns()
        source_hashes = {
            filename: _sha256(self.root / filename) for filename in self.REQUIRED_FILES
        }
        observations = [
            {
                "display_index": index,
                "view_id": view_id,
                "role": self.plan_by_view[view_id].get("role"),
                "rgb": str(self.media_by_view[view_id]),
                "rgb_sha256": _sha256(self.media_by_view[view_id]),
                "world_from_camera": self.observation_by_id[view_id]["world_from_camera"],
            }
            for index, view_id in enumerate(self.view_ids, 1)
        ]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": "ep3d-rsint17-dialogue-canonical-auto-v2",
            "family_id": self.episode.get("family_id"),
            "split_lock": self.episode.get("split_group"),
            "scene_id": self.episode.get("scene_id"),
            "status": "development_only",
            "presentation_mode": "incremental_dialogue",
            "frame_contract": {"frame_id": FRAME_ID, "surface_zh": FRAME_SURFACE_ZH},
            "source": {"bundle": str(self.root), "bundle_files_sha256": source_hashes},
            "capability_coverage": {
                "covered": ["MM", "SR", "PT", "CR"],
                "supporting": ["grounding", "temporal_memory", "epistemic_calibration"],
                "unsupported": {
                    "MR": "该轨迹没有 certified canonical object-front 或旋转干预，不能作为 SenseNova Mental Rotation 监督。"
                },
            },
            "sampling_diagnostics": {
                "perspective": self.perspective_sampling_diagnostics,
            },
            "observations": observations,
            "rounds": turns,
            "validation": {
                "source_integrity": True,
                "typed_programs": True,
                "node_claims_reexecuted": True,
                "causal_view_release": True,
                "future_view_leakage": False,
                "question_blueprints_answer_blind": True,
            },
        }
        payload["truth_compilation_sha256"] = hashlib.sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest()
        return payload


def compile_rsint_truth(bundle_root: str | Path) -> dict[str, Any]:
    """Convenience API used by the CLI and tests."""

    return RsIntTruthCompiler(bundle_root).compile()
