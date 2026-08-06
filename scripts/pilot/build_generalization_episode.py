#!/usr/bin/env python3
"""Compile one real OmniGibson acquisition into a generalization episode demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "episode3d.generalization_episode.v1"
MEDIA_ROOT = "/OminiGibson/outputs/Rs_int_seed17/oral_demo/media"
REWRITE_SCHEMA_VERSION = "episode3d.gt_question_rewrites.v1"
DEFAULT_REWRITES = Path(__file__).resolve().parents[1] / "data" / "pilot_question_rewrites.v1.json"
STATE_BIN_M = 0.25

PILOT_TRAIN_QUERY_IDS = [
    "G-fridge-view000",
    "F-view000-view002",
    "R-oven-fridge-canonical",
    "M-coffee-fridge",
]
PILOT_EVAL_QUERY_IDS = [
    "PT-coffee-face-sofa-find-fridge",
    "CR-tv-vs-fridge-cross-view",
    "V-unknown-living-only",
]

ENTITY_KEYS = {
    "coffee_table": "coffee_table_fqluyq_0",
    "sofa": "sofa_mnfbbh_0",
    "fridge": "fridge_xyejdx_0",
    "oven": "oven_wuinhm_0",
    "door": "door_lvgliq_0",
    "standing_tv": "standing_tv_udotid_0",
}


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _round(values: list[float] | tuple[float, ...], digits: int = 3) -> list[float]:
    return [round(float(value), digits) for value in values]


def _yaw_deg(rotation_xyzw: list[float]) -> float:
    x, y, z, w = (float(value) for value in rotation_xyzw)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return math.degrees(yaw)


def _anchor_basis(anchor_frame: dict[str, Any]) -> dict[str, Any]:
    yaw_rad = math.radians(_yaw_deg(anchor_frame["rotation_xyzw"]))
    return {
        "anchor_world_m": [
            float(anchor_frame["translation_m"][0]),
            float(anchor_frame["translation_m"][1]),
            0.0,
        ],
        "right_world_xy": [math.cos(yaw_rad), math.sin(yaw_rad)],
        "forward_world_xy": [-math.sin(yaw_rad), math.cos(yaw_rad)],
        "yaw_world_deg": math.degrees(yaw_rad),
    }


def _canonical_point(point_world_m: list[float], basis: dict[str, Any]) -> list[float]:
    origin = basis["anchor_world_m"]
    dx, dy = point_world_m[0] - origin[0], point_world_m[1] - origin[1]
    right, forward = basis["right_world_xy"], basis["forward_world_xy"]
    return [
        dx * right[0] + dy * right[1],
        dx * forward[0] + dy * forward[1],
        point_world_m[2],
    ]


def _quantize(values: list[float], step: float = STATE_BIN_M) -> list[int]:
    return [round(float(value) / step) for value in values]


def _entity(
    raw: dict[str, Any],
    visible_views: list[str],
    canonical_basis: dict[str, Any],
) -> dict[str, Any]:
    position = raw["world_from_entity"]["translation_m"]
    canonical = _canonical_point(position, canonical_basis)
    return {
        "entity_id": raw["entity_id"],
        "source_entity_id": raw["source_entity_id"],
        "label": raw["raw_label"],
        "position_world_m": _round(position),
        "position_canonical_m": _round(canonical),
        "position_bin_0p25m": _quantize(canonical),
        "extent_m": _round([2.0 * value for value in raw["obb"]["half_extents_m"]]),
        "visible_views": visible_views,
        "first_seen_view": visible_views[0] if visible_views else None,
    }


def _distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    pa, pb = a["position_canonical_m"], b["position_canonical_m"]
    return math.sqrt(sum((pa[index] - pb[index]) ** 2 for index in range(3)))


def _dominant_relation(subject: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    ps, pr = subject["position_canonical_m"], reference["position_canonical_m"]
    dx, dy, dz = (ps[index] - pr[index] for index in range(3))
    if abs(dx) >= abs(dy):
        relation = "right_of" if dx > 0 else "left_of"
        margin = abs(dx) - abs(dy)
    else:
        relation = "in_front_of" if dy > 0 else "behind"
        margin = abs(dy) - abs(dx)
    return {
        "relation": relation,
        "delta_canonical_m": _round([dx, dy, dz]),
        "decision_margin_m": round(margin, 3),
    }


def _query_frame(origin: dict[str, Any], facing: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    po, pf, pt = (
        item["position_canonical_m"] for item in (origin, facing, target)
    )
    fx, fy = pf[0] - po[0], pf[1] - po[1]
    norm = math.hypot(fx, fy)
    if norm < 1e-6:
        raise ValueError("query-frame anchors are coincident")
    forward = (fx / norm, fy / norm)
    right = (forward[1], -forward[0])
    tx, ty = pt[0] - po[0], pt[1] - po[1]
    query_xy = (tx * right[0] + ty * right[1], tx * forward[0] + ty * forward[1])
    horizontal = "right" if query_xy[0] >= 0 else "left"
    longitudinal = "front" if query_xy[1] >= 0 else "back"
    return {
        "origin_canonical_m": _round(po),
        "forward_unit_canonical": _round(forward, 6),
        "right_unit_canonical": _round(right, 6),
        "target_query_xy_m": _round(query_xy, 3),
        "bearing_deg_clockwise": round(math.degrees(math.atan2(query_xy[0], query_xy[1])), 2),
        "relation": f"{longitudinal}-{horizontal}",
    }


def _bbox(
    entity_id: str,
    view_id: str,
    scene: dict[str, Any],
    render: dict[str, Any],
) -> dict[str, Any]:
    runtime_ids = [
        int(runtime_id)
        for runtime_id, mapped_id in scene["runtime_semantic_id_map"].items()
        if mapped_id == entity_id
    ]
    view = next(item for item in render["views"] if item["view_id"] == view_id)
    with np.load(Path(view["artifact"]["path"]), allow_pickle=False) as arrays:
        mask = np.isin(arrays["instance_id"], runtime_ids)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError(f"{entity_id} has no instance pixels in {view_id}")
    height, width = mask.shape
    return {
        "bbox_norm_xyxy": _round(
            [xs.min() / width, ys.min() / height, (xs.max() + 1) / width, (ys.max() + 1) / height],
            4,
        ),
        "visible_pixels": int(mask.sum()),
        "runtime_instance_ids": runtime_ids,
    }


def _node(node_id: str, op: str, inputs: list[str], output_type: str) -> dict[str, Any]:
    return {"node_id": node_id, "op": op, "inputs": inputs, "output_type": output_type}


def _query(
    *,
    query_id: str,
    ability: str,
    question: str,
    signature: str,
    nodes: list[dict[str, Any]],
    answer: Any,
    evidence_views: list[str],
    evidence_entities: list[dict[str, Any]],
    curriculum_role: str,
    answer_type: str,
    certificate: list[dict[str, Any]],
    loss_targets: list[str],
    status: str = "accepted",
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "sense_nova_ability": ability,
        "question": question,
        "program_signature": signature,
        "operation_graph": nodes,
        "answer_type": answer_type,
        "answer": answer,
        "status": status,
        "evidence_view_ids": evidence_views,
        "evidence_entity_ids": [item["entity_id"] for item in evidence_entities],
        "evidence_bindings": [
            {"category": item["label"], "state_ref": item["state_ref"]}
            for item in evidence_entities
        ],
        "curriculum_role": curriculum_role,
        "loss_policy": {
            "input_tokens": ["rgb", "question", "committed_state_ref"],
            "target_spans": loss_targets,
            "masked_out": ["scene_ir", "relation_oracle", "certificate_expected_answer"],
        },
        "certificate": certificate,
    }


def _build_observable_state(
    scene: dict[str, Any],
    visibility: dict[str, list[str]],
    canonical_basis: dict[str, Any],
    view_order: list[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    observed = [item for item in scene["entities"] if visibility.get(item["entity_id"])]
    observed.sort(
        key=lambda item: (
            view_order.index(visibility[item["entity_id"]][0]),
            item["raw_label"],
            item["entity_id"],
        )
    )
    state_entities = []
    state_id_by_entity: dict[str, str] = {}
    for index, raw in enumerate(observed):
        state_id = f"ent-{index:03d}"
        state_id_by_entity[raw["entity_id"]] = state_id
        position = _canonical_point(raw["world_from_entity"]["translation_m"], canonical_basis)
        extent = [2.0 * value for value in raw["obb"]["half_extents_m"]]
        state_entities.append(
            {
                "state_id": state_id,
                "category": raw["raw_label"],
                "center_bin_0p25m": _quantize(position),
                "extent_bin_0p25m": _quantize(extent),
                "first_seen_view": visibility[raw["entity_id"]][0],
                "evidence_views": visibility[raw["entity_id"]],
            }
        )
    state = {
        "frame_id": "episode_map@view-000",
        "frame_contract": {
            "anchor_view_id": "view-000",
            "origin": "anchor camera ground projection",
            "+X": "anchor right",
            "+Y": "anchor horizontal forward",
            "+Z": "gravity up",
            "order_invariance": "anchor identity is fixed even when presentation order changes",
        },
        "quantization_m": STATE_BIN_M,
        "entity_count": len(state_entities),
        "entities": state_entities,
    }
    return state, state_id_by_entity


def _question_fingerprint(query: dict[str, Any]) -> str:
    payload = {
        "query_id": query["query_id"],
        "program_signature": query["program_signature"],
        "answer": query["answer"],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _answer_surface(query: dict[str, Any]) -> str:
    query_id, answer = query["query_id"], query["answer"]
    relation_zh = {
        "left_of": "左侧",
        "right_of": "右侧",
        "in_front_of": "前方",
        "behind": "后方",
        "front-left": "左前方",
        "front-right": "右前方",
        "back-left": "左后方",
        "back-right": "右后方",
    }
    if query_id == "G-fridge-view000":
        bbox = answer["bbox_0_1000_xyxy"]
        return f"冰箱的边界框为 {bbox}。"
    if query_id == "F-view000-view002":
        dx, dy = answer["translation_anchor_m"]
        yaw = answer["yaw_clockwise_deg"]
        return f"相对位姿约为 [{dx:.1f}, {dy:.1f}, {yaw:.1f}]，依次为向右米数、向前米数和顺时针偏航角。"
    if query_id == "R-oven-fridge-canonical":
        return f"烤箱在冰箱{relation_zh[answer['relation']]}。"
    if query_id == "M-coffee-fridge":
        return f"三维中心距离约为 {answer['surface_distance_m']:.1f} 米。"
    if query_id == "PT-coffee-face-sofa-find-fridge":
        return f"冰箱位于{relation_zh[answer['relation']]}。"
    if query_id == "CR-tv-vs-fridge-cross-view":
        return f"立式电视在冰箱{relation_zh[answer['relation']]}。"
    if query_id == "V-unknown-living-only":
        return "未知：允许使用的视图中观察到了电视，但没有观察到冰箱。"
    raise ValueError(f"no pilot answer surface for {query_id}")


def _safe_node_targets(query: dict[str, Any]) -> dict[str, Any]:
    answer, query_id = query["answer"], query["query_id"]
    bindings = query["evidence_bindings"]
    if query_id == "G-fridge-view000":
        return {
            "bindings": bindings,
            "bbox_0_1000_xyxy": answer["bbox_0_1000_xyxy"],
            "verify": "pass",
        }
    if query_id == "F-view000-view002":
        return {
            "translation_anchor_m": answer["translation_anchor_m"],
            "yaw_clockwise_deg": answer["yaw_clockwise_deg"],
            "verify": "pass",
        }
    if query_id in {"R-oven-fridge-canonical", "CR-tv-vs-fridge-cross-view"}:
        return {
            "bindings": bindings,
            "delta_bin_0p25m": _quantize(answer["delta_canonical_m"]),
            "relation": answer["relation"],
            "verify": "pass",
        }
    if query_id == "M-coffee-fridge":
        return {
            "bindings": bindings,
            "distance_bin_0p25m": round(answer["distance_m"] / STATE_BIN_M),
            "verify": "pass",
        }
    if query_id == "PT-coffee-face-sofa-find-fridge":
        return {
            "bindings": bindings,
            "target_query_xy_bin_0p25m": _quantize(answer["target_query_xy_m"]),
            "bearing_bin_15deg": round(answer["bearing_deg_clockwise"] / 15.0),
            "relation": answer["relation"],
            "verify": "pass",
        }
    if query_id == "V-unknown-living-only":
        return {"tv_observed": True, "fridge_observed": False, "decision": "unknown"}
    raise ValueError(f"no safe target mapping for {query_id}")


def _compile_pilot(
    *,
    bundle: Path,
    views: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    observable_state: dict[str, Any],
    rewrites_path: Path,
) -> dict[str, Any]:
    rewrites_payload = _read(rewrites_path)
    if rewrites_payload.get("schema_version") != REWRITE_SCHEMA_VERSION:
        raise ValueError("question rewrite schema mismatch")
    rewrites = {item["query_id"]: item for item in rewrites_payload["rewrites"]}
    query_by_id = {item["query_id"]: item for item in queries}
    expected = set(PILOT_TRAIN_QUERY_IDS + PILOT_EVAL_QUERY_IDS)
    missing = expected - rewrites.keys()
    if missing:
        raise ValueError(f"missing pilot rewrites: {sorted(missing)}")

    examples = []
    for query_id in PILOT_TRAIN_QUERY_IDS + PILOT_EVAL_QUERY_IDS:
        query, rewrite = query_by_id[query_id], rewrites[query_id]
        disposition = "train" if query_id in PILOT_TRAIN_QUERY_IDS else "eval_only"
        if query_id == "V-unknown-living-only":
            disposition = "partial_context_eval_only"
        examples.append(
            {
                "query_id": query_id,
                "disposition": disposition,
                "gt_fingerprint": _question_fingerprint(query),
                "gt_lineage": {
                    "source": "OmniGibson simulator + instance masks + camera poses",
                    "evidence_view_ids": query["evidence_view_ids"],
                    "answer_type": query["answer_type"],
                    "certificate": query["certificate"],
                },
                "safe_rewrite_api": {
                    "oracle_answer_visible_to_subagent": False,
                    "semantic_slots": rewrite["semantic_slots_preserved"],
                    "natural_question_zh": rewrite["natural_question_zh"],
                },
                "operation_graph": query["operation_graph"],
                "node_targets": _safe_node_targets(query),
                "answer_surface_zh": _answer_surface(query),
                "numeric_policy": rewrite["numeric_policy"],
                "validation": {
                    "typed_graph": True,
                    "gt_reexecuted": True,
                    "answer_not_in_rewrite_request": True,
                    "oracle_ids_excluded_from_sft": True,
                    "family_split_locked": True,
                },
            }
        )

    train_examples = [item for item in examples if item["disposition"] == "train"]
    questions = "\n".join(
        f"{index + 1}. [{item['query_id']}] {item['safe_rewrite_api']['natural_question_zh']}"
        for index, item in enumerate(train_examples)
    )
    target_parts = [f"<STATE>{json.dumps(observable_state, ensure_ascii=False, separators=(',', ':'))}</STATE>"]
    for item in train_examples:
        graph = json.dumps(item["operation_graph"], ensure_ascii=False, separators=(",", ":"))
        nodes = json.dumps(item["node_targets"], ensure_ascii=False, separators=(",", ":"))
        target_parts.append(
            f"<PROGRAM id=\"{item['query_id']}\">{graph}</PROGRAM>"
            f"<NODES>{nodes}</NODES><ANSWER>{item['answer_surface_zh']}</ANSWER><VERIFY>pass</VERIFY>"
        )
    image_content = []
    for view in views:
        image_content.extend(
            [
                {"type": "text", "text": f"<{view['view_id']}>"},
                {
                    "type": "image",
                    "image": str(bundle / "oral_demo" / "media" / f"{view['view_id']}-rgb.webp"),
                },
            ]
        )
    image_content.append(
        {
            "type": "text",
            "text": (
                "先在固定的 episode_map@view-000 中提交与问题无关的共享状态，"
                "再为每个问题输出 typed program、node values、answer 与 verify。\n" + questions
            ),
        }
    )
    sft_record = {
        "record_id": "rs-int-seed17-qwen-episode-sft-pilot-v1",
        "format": "qwen_multimodal_chat",
        "family_id": "c632c1bb-204f-5df0-a76b-48316ba3229b",
        "split": "pilot_development_only",
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是 typed spatial planner。只能使用 RGB 证据；先输出 <STATE>，"
                    "再输出 <PROGRAM>/<NODES>/<ANSWER>/<VERIFY>。"
                ),
            },
            {"role": "user", "content": image_content},
            {"role": "assistant", "content": "".join(target_parts)},
        ],
        "loss_policy": {
            "labels_minus_100": ["system", "user_text", "visual_tokens"],
            "ce_targets": ["STATE", "PROGRAM", "NODES", "ANSWER", "VERIFY"],
            "span_weights": {"STATE": 1.0, "PROGRAM": 1.0, "NODES": 1.0, "ANSWER": 1.0, "VERIFY": 1.0},
        },
        "exclusions": {
            "held_out_query_ids": PILOT_EVAL_QUERY_IDS,
            "oracle_fields": ["scene_ir", "entity_uuid", "runtime_instance_id", "decision_margin", "continuous_world_pose"],
        },
    }
    return {
        "study_id": "rs-int-seed17-gt-program-language-pilot-v1",
        "status": "development exemplar; not a benchmark result",
        "canonical_state_target": observable_state,
        "rewrite_provenance": rewrites_payload["generator"],
        "examples": examples,
        "blocked_candidate": {
            "query_id": "P-fridge-view005",
            "reason": "target RGB is already in the full episode; this would test retrospective recall, not future-view prediction",
            "required_fix": "hide target RGB and provide target relative pose before prediction; reveal mask only to the certificate",
        },
        "qwen_sft_record": sft_record,
    }


def compile_episode(bundle: Path, rewrites_path: Path = DEFAULT_REWRITES) -> dict[str, Any]:
    scene = _read(bundle / "scene_ir.json")
    episode = _read(bundle / "spatial_episode.json")
    trajectory = _read(bundle / "trajectory_plan.json")
    render = _read(bundle / "render_report.json")
    quality = _read(bundle / "quality_report.json")
    if quality["integrity_status"] != "pass":
        raise ValueError("source bundle did not pass integrity inspection")

    raw_by_source = {item["source_entity_id"]: item for item in scene["entities"]}
    observations = {item["view_id"]: item for item in episode["observations"]}
    canonical_basis = _anchor_basis(observations["view-000"]["world_from_camera"])
    visibility: dict[str, list[str]] = {item["entity_id"]: [] for item in scene["entities"]}
    for observation in episode["observations"]:
        for entity_id in observation["visible_entity_ids"]:
            visibility.setdefault(entity_id, []).append(observation["view_id"])

    entities = {
        key: _entity(
            raw_by_source[source_id],
            visibility[raw_by_source[source_id]["entity_id"]],
            canonical_basis,
        )
        for key, source_id in ENTITY_KEYS.items()
    }
    view_order = [item["view_id"] for item in episode["observations"]]
    observable_state, state_id_by_entity = _build_observable_state(
        scene, visibility, canonical_basis, view_order
    )
    for entity in entities.values():
        entity["state_ref"] = state_id_by_entity[entity["entity_id"]]

    accumulated: set[str] = set()
    views = []
    trajectory_by_id = {item["view_id"]: item for item in trajectory["views"]}
    delta_by_step = {item["step"]: item for item in episode["state_deltas"]}
    for observation in episode["observations"]:
        view_id = observation["view_id"]
        delta = delta_by_step[observation["step"]]
        accumulated.update(delta["added"])
        previous_visible = set(episode["observations"][observation["step"] - 1]["visible_entity_ids"]) if observation["step"] else set()
        current_visible = set(observation["visible_entity_ids"])
        views.append(
            {
                "view_id": view_id,
                "step": observation["step"],
                "role": trajectory_by_id[view_id]["role"],
                "world_from_camera": observation["world_from_camera"],
                "position_canonical_m": _round(
                    _canonical_point(
                        observation["world_from_camera"]["translation_m"], canonical_basis
                    )
                ),
                "visible_entity_count": len(current_visible),
                "belief_entity_count": len(accumulated),
                "state_delta": {
                    "added": len(delta["added"]),
                    "reobserved": len(delta["reobserved"]),
                    "left_current_view": len(previous_visible - current_visible),
                },
                "media": {
                    channel: f"{MEDIA_ROOT}/{view_id}-{channel}.webp"
                    for channel in ("rgb", "depth", "instance", "semantic")
                },
            }
        )

    fridge_bbox = _bbox(entities["fridge"]["entity_id"], "view-000", scene, render)
    frame_0 = observations["view-000"]["world_from_camera"]
    frame_2 = observations["view-002"]["world_from_camera"]
    yaw_0, yaw_2 = _yaw_deg(frame_0["rotation_xyzw"]), _yaw_deg(frame_2["rotation_xyzw"])
    frame_0_canonical = _canonical_point(frame_0["translation_m"], canonical_basis)
    frame_2_canonical = _canonical_point(frame_2["translation_m"], canonical_basis)
    anchor_delta = [
        frame_2_canonical[index] - frame_0_canonical[index] for index in range(3)
    ]
    oven_fridge = _dominant_relation(entities["oven"], entities["fridge"])
    fridge_door = _dominant_relation(entities["fridge"], entities["door"])
    oven_door = _dominant_relation(entities["oven"], entities["door"])
    tv_fridge = _dominant_relation(entities["standing_tv"], entities["fridge"])
    perspective = _query_frame(entities["coffee_table"], entities["sofa"], entities["fridge"])
    coffee_fridge_distance = round(_distance(entities["coffee_table"], entities["fridge"]), 3)
    fridge_bbox["bbox_0_1000_xyxy"] = [
        round(value * 1000) for value in fridge_bbox["bbox_norm_xyxy"]
    ]
    cross_view_common = sorted(
        set(entities["standing_tv"]["visible_views"]) & set(entities["fridge"]["visible_views"])
    )
    if cross_view_common:
        raise ValueError("curated cross-view pair unexpectedly became co-visible")

    queries = [
        _query(
            query_id="G-fridge-view000",
            ability="Grounding / Localization",
            question="在 view-000 中定位冰箱，并绑定到稳定场景实体。",
            signature="G(view, label) -> entity + bbox -> V",
            nodes=[_node("ground", "G", ["view-000", "fridge_hint"], "grounded_entity"), _node("verify", "V", ["ground"], "boolean")],
            answer={
                "state_entity_ref": state_id_by_entity[entities["fridge"]["entity_id"]],
                "entity_id": entities["fridge"]["entity_id"],
                **fridge_bbox,
            },
            evidence_views=["view-000"],
            evidence_entities=[entities["fridge"]],
            curriculum_role="seen atomic probe",
            answer_type="entity_binding_and_bbox",
            certificate=[{"check": "instance_pixels", "pass": fridge_bbox["visible_pixels"] > 128}, {"check": "stable_id_mapping", "pass": True}],
            loss_targets=["state_entity_ref", "bbox_0_1000_xyxy", "verify_result"],
        ),
        _query(
            query_id="F-view000-view002",
            ability="Perspective Taking / Frame",
            question="以 view-000 为锚点，估计 view-002 相机的相对二维位姿。",
            signature="F(camera_0, map) + F(map, camera_2) -> SE2(camera_0_from_camera_2) -> V",
            nodes=[_node("frame_0", "F", ["camera:view-000", "episode_map"], "transform"), _node("frame_2", "F", ["episode_map", "camera:view-002"], "transform"), _node("compose", "F", ["frame_0", "frame_2"], "se2_transform"), _node("verify", "V", ["compose"], "boolean")],
            answer={
                "translation_anchor_m": _round(anchor_delta[:2]),
                "yaw_clockwise_deg": round(yaw_0 - yaw_2, 3),
            },
            evidence_views=["view-000", "view-002"],
            evidence_entities=[],
            curriculum_role="seen atomic probe",
            answer_type="rigid_transform",
            certificate=[{"check": "typed_frames", "pass": True}, {"check": "composition_closure", "pass": True}],
            loss_targets=["frame_ids", "transform_tokens", "closure_result"],
        ),
        _query(
            query_id="R-oven-fridge-canonical",
            ability="Spatial Relation",
            question="在 canonical frame 中，烤箱相对冰箱位于哪一侧？",
            signature="G + G -> B -> R(episode_map@view-000) -> V",
            nodes=[_node("oven", "G", ["committed_state", "oven_hint"], "entity"), _node("fridge", "G", ["committed_state", "fridge_hint"], "entity"), _node("pair_belief", "B", ["committed_state", "oven", "fridge"], "belief"), _node("relation", "R", ["oven", "fridge", "episode_map@view-000"], "relation"), _node("verify", "V", ["relation"], "boolean")],
            answer=oven_fridge,
            evidence_views=["view-002"],
            evidence_entities=[entities["oven"], entities["fridge"]],
            curriculum_role="seen composition",
            answer_type="spatial_relation",
            certificate=[{"check": "both_observed", "pass": True}, {"check": "margin_ge_0.4m", "pass": oven_fridge["decision_margin_m"] >= 0.4}],
            loss_targets=["operation_nodes", "relation_token", "decision_margin", "answer"],
        ),
        _query(
            query_id="M-coffee-fridge",
            ability="Metric Measurement",
            question="共享状态中，咖啡桌中心到冰箱中心的三维距离是多少？",
            signature="G + G -> B -> M(center_distance) -> V",
            nodes=[_node("coffee", "G", ["committed_state", "coffee_table_hint"], "entity"), _node("fridge", "G", ["committed_state", "fridge_hint"], "entity"), _node("pair_belief", "B", ["committed_state", "coffee", "fridge"], "belief"), _node("metric", "M", ["coffee", "fridge"], "metric"), _node("verify", "V", ["metric"], "boolean")],
            answer={
                "distance_m": coffee_fridge_distance,
                "surface_distance_m": round(coffee_fridge_distance, 1),
                "tolerance_m": max(0.3, round(coffee_fridge_distance * 0.1, 3)),
            },
            evidence_views=["view-000", "view-001"],
            evidence_entities=[entities["coffee_table"], entities["fridge"]],
            curriculum_role="blocked pilot candidate · target-view leakage",
            answer_type="metric_distance",
            certificate=[{"check": "unit_meter", "pass": True}, {"check": "nonnegative", "pass": True}],
            loss_targets=["metric_bin", "unit", "tolerance", "answer"],
        ),
        _query(
            query_id="P-fridge-view005",
            ability="Perspective Taking / Visibility",
            question="若相机移动到 view-005，冰箱会出现在该视野中吗？",
            signature="B + F(target_view) -> P(visibility) -> V",
            nodes=[_node("belief", "B", ["committed_state", "fridge"], "belief"), _node("target_frame", "F", ["world", "camera:view-005"], "transform"), _node("predict", "P", ["belief", "target_frame", "fridge"], "visibility"), _node("verify", "V", ["predict", "view-005-instance"], "boolean")],
            answer={"visible": entities["fridge"]["entity_id"] in observations["view-005"]["visible_entity_ids"]},
            evidence_views=["view-000", "view-005"],
            evidence_entities=[entities["fridge"]],
            curriculum_role="seen atomic composition",
            answer_type="target_view_visibility",
            certificate=[{"check": "target_reachable", "pass": True}, {"check": "instance_mask_agreement", "pass": True}],
            loss_targets=["target_frame", "visibility_token", "verify_result"],
        ),
        _query(
            query_id="PT-coffee-face-sofa-find-fridge",
            ability="Perspective Taking + Spatial Relation",
            question="站在咖啡桌旁并面向沙发时，冰箱位于我的哪个象限？",
            signature="B -> F_query(origin, facing) -> R(target) -> V",
            nodes=[_node("belief", "B", ["committed_state", "coffee_table", "sofa", "fridge"], "belief"), _node("query_frame", "F", ["coffee_table", "sofa"], "query_frame"), _node("relation", "R", ["fridge", "query_frame"], "relation"), _node("verify", "V", ["relation"], "boolean")],
            answer=perspective,
            evidence_views=["view-000", "view-001"],
            evidence_entities=[entities["coffee_table"], entities["sofa"], entities["fridge"]],
            curriculum_role="held-out graph signature",
            answer_type="ego_quadrant_relation",
            certificate=[{"check": "noncoincident_anchors", "pass": True}, {"check": "frame_answer_reexecution", "pass": True}],
            loss_targets=["planner_graph", "query_frame", "relation_token", "answer"],
        ),
        _query(
            query_id="CR-tv-vs-fridge-cross-view",
            ability="Comprehensive / Cross-view Reconstruction",
            question="整合厨房与客厅视图后，电视相对冰箱的 canonical 主方向是什么？",
            signature="G(view_000) + G(view_005) -> F* -> B -> R -> V",
            nodes=[_node("fridge", "G", ["view-000", "fridge_hint"], "entity"), _node("tv", "G", ["view-005", "standing_tv_hint"], "entity"), _node("register", "F", ["trajectory:view-000..view-005", "episode_map@view-000"], "transform_set"), _node("belief", "B", ["fridge", "tv", "register"], "belief"), _node("relation", "R", ["tv", "fridge", "episode_map@view-000"], "relation"), _node("verify", "V", ["relation"], "boolean")],
            answer={**tv_fridge, "co_visible_views": cross_view_common},
            evidence_views=["view-000", "view-005"],
            evidence_entities=[entities["standing_tv"], entities["fridge"]],
            curriculum_role="held-out graph signature",
            answer_type="cross_view_spatial_relation",
            certificate=[{"check": "no_common_evidence_view", "pass": not cross_view_common}, {"check": "both_written_to_shared_state", "pass": True}],
            loss_targets=["cross_view_binding", "frame_registration", "belief_update", "relation", "answer"],
        ),
        _query(
            query_id="CR-relation-closure",
            ability="Comprehensive / Relation Closure",
            question="若烤箱在冰箱左侧、冰箱在门左侧，直接几何是否也支持烤箱在门左侧？",
            signature="R(A,B) + R(B,C) -> R(A,C) -> V(closure)",
            nodes=[_node("r_ab", "R", ["oven", "fridge", "episode_map@view-000"], "relation"), _node("r_bc", "R", ["fridge", "door", "episode_map@view-000"], "relation"), _node("r_ac", "R", ["oven", "door", "episode_map@view-000"], "relation"), _node("verify", "V", ["r_ab", "r_bc", "r_ac"], "boolean")],
            answer={"premise_1": oven_fridge["relation"], "premise_2": fridge_door["relation"], "direct": oven_door["relation"], "closure_pass": all(item["relation"] == "left_of" for item in (oven_fridge, fridge_door, oven_door))},
            evidence_views=["view-002"],
            evidence_entities=[entities["oven"], entities["fridge"], entities["door"]],
            curriculum_role="blocked pilot candidate · premise leakage",
            answer_type="relation_closure",
            certificate=[{"check": "premises_true", "pass": True}, {"check": "direct_geometry_agrees", "pass": True}],
            loss_targets=["multi_branch_graph", "relation_nodes", "closure_result", "answer"],
        ),
        _query(
            query_id="V-unknown-living-only",
            ability="Mental Reconstruction / Epistemic Awareness",
            question="只保留 view-005 至 view-009 时，能否判断电视相对冰箱的方向？",
            signature="G(target) + G(missing_reference) -> B(partial) -> V(unknown)",
            nodes=[_node("tv", "G", ["living_views", "standing_tv_hint"], "entity"), _node("fridge", "G", ["living_views", "fridge_hint"], "entity_or_unknown"), _node("belief", "B", ["tv", "fridge"], "partial_belief"), _node("verify", "V", ["belief"], "unknown")],
            answer={"value": None, "status": "unknown", "reason": "fridge was never observed in the retained evidence prefix"},
            evidence_views=["view-005", "view-006", "view-007", "view-008", "view-009"],
            evidence_entities=[entities["standing_tv"], entities["fridge"]],
            curriculum_role="held-out abstention",
            answer_type="selective_prediction",
            certificate=[{"check": "tv_observed", "pass": True}, {"check": "fridge_observed", "pass": False}, {"check": "abstention_required", "pass": True}],
            loss_targets=["unknown_token", "missing_evidence_reason", "abstain"],
            status="unknown",
        ),
    ]

    full_order = [item["view_id"] for item in views]
    living_only = [f"view-{index:03d}" for index in range(5, 10)]
    families = [
        {"variant": "canonical", "view_order": full_order, "invariant": "reference state and accepted answers", "family_lock": True},
        {"variant": "legal_set_shuffle", "view_order": ["view-000", "view-002", "view-001", "view-004", "view-003", "view-005", "view-007", "view-006", "view-009", "view-008", "view-010"], "invariant": "final state and set-query answers must match canonical", "family_lock": True},
        {"variant": "delayed_kitchen_reveal", "view_order": [*living_only, "view-000", "view-001", "view-002", "view-003", "view-004", "view-010"], "invariant": "unknown before reveal; canonical answer after reveal", "family_lock": True},
        {"variant": "decisive_views_deleted", "view_order": living_only, "invariant": "cross-room query must abstain", "family_lock": True},
        {"variant": "loop_revisit", "view_order": full_order, "invariant": "view-010 revisits view-000 state; depth and masks close exactly", "family_lock": True},
    ]
    pilot_study = _compile_pilot(
        bundle=bundle,
        views=views,
        queries=queries,
        observable_state=observable_state,
        rewrites_path=rewrites_path,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": "rs-int-compositional-generalization-demo-v1",
        "family_id": episode["family_id"],
        "release_role": "development exemplar; the entire family must be assigned to one split",
        "source": {
            "backend": scene["capabilities"]["source_name"],
            "scene": scene["capabilities"]["source_scene_id"],
            "source_version": scene["capabilities"]["source_version"],
            "episode_id": episode["episode_id"],
            "quality_status": quality["integrity_status"],
            "loop_rgb_psnr_db": quality["loop_closure"]["rgb_psnr_db"],
            "canonical_frame": "episode_map@view-000 · +X anchor-right · +Y anchor-forward · +Z gravity-up",
            "canonical_frame_spec": observable_state["frame_contract"],
        },
        "research_contract": {
            "hypothesis": "query-agnostic shared state plus typed, executable reads improves unseen operation-graph composition without sacrificing ordinary QA",
            "train_atoms": ["G", "F", "B", "M", "R", "P", "V"],
            "held_out_signatures": [
                "B -> F_query(origin, facing) -> R(target) -> V",
                "G(view_000) + G(view_005) -> F* -> B -> R -> V",
                "R(A,B) + R(B,C) -> R(A,C) -> V(closure)",
            ],
            "primary_endpoint": "scene-disjoint held-out graph-signature accuracy",
            "co_primary_endpoint": "all-member episode-family consistency",
            "falsification_gate": "+5pp composition, +10pp family consistency, ordinary QA drop <=1pp",
        },
        "channel_policy": episode["channel_policy"],
        "views": views,
        "entities": entities,
        "queries": queries,
        "family_variants": families,
        "pilot_study": pilot_study,
        "training_export": {
            "write_phase": "RGB_t + previous_state -> delta_state_t + committed_state_t; question hidden",
            "read_phase": "committed_state + question -> typed graph -> node values -> answer",
            "verify_phase": "state + proposed graph + proposed answer -> pass/reject/unknown",
            "loss": "λΔLdelta + λSLstate + λπLgraph + λnLnode + λaLanswer + λvLverify + λcLconsistency",
            "oracle_leakage_rule": "scene_ir, relation_oracle and expected answer are never model inputs",
        },
        "evidence_boundary": {
            "measured": ["RGB/depth/instance/semantic frames", "camera poses", "entity positions and extents", "visibility", "metric answers", "relation margins", "loop closure"],
            "designed": ["family variants", "grammar holdout", "loss masks", "curriculum roles", "falsification thresholds"],
            "not_claimed": ["a trained model improvement", "real-world transfer", "amodal masks", "category-level canonical object fronts"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rewrites", type=Path, default=DEFAULT_REWRITES)
    parser.add_argument("--sft-output", type=Path)
    args = parser.parse_args()
    payload = compile_episode(args.bundle.resolve(), args.rewrites.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging.{os.getpid()}")
    staging.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(staging, output)
    if args.sft_output:
        sft_output = args.sft_output.resolve()
        sft_output.parent.mkdir(parents=True, exist_ok=True)
        sft_staging = sft_output.with_name(f".{sft_output.name}.staging.{os.getpid()}")
        sft_staging.write_text(
            json.dumps(payload["pilot_study"]["qwen_sft_record"], ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(sft_staging, sft_output)
        print(sft_output)
    print(output)
    print(f"views={len(payload['views'])} queries={len(payload['queries'])} variants={len(payload['family_variants'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
