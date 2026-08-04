"""Compile Transform Pilot facts into paired multimodal SFT and evaluation data.

The compiler keeps three layers deliberately separate:

1. model-visible RGB, action language and questions;
2. model-visible grounded traces with cue -> transform -> conclusion roles;
3. oracle geometry and executable certificates used only by the verifier.

Answer-only and grounded-CoT records are produced from the same question record,
so a training comparison cannot accidentally change images, wording or labels.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from episode3d.transform_pilot.geometry import (
    direction_measurement,
    horizontal_obb_radius,
    yaw_facing,
    yaw_from_xyzw,
)

SCHEMA_VERSION = "epispace.transform_dataset.v1"
COT_SCHEMA_VERSION = "epispace.grounded_cot.v1"
DIRECTION_CN = {
    "front": "前方",
    "right": "右侧",
    "back": "后方",
    "left": "左侧",
}
OBJECT_CN = {
    "footstool": "脚凳",
    "straight_chair": "椅子",
    "backpack": "背包",
    "public_trash_can": "垃圾桶",
    "pot_plant": "盆栽",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _prune_unreferenced_media(
    output_root: Path, records: list[dict[str, Any]]
) -> int:
    """Remove compiler-owned media that no accepted record references."""

    media_root = output_root / "media"
    if not media_root.is_dir():
        return 0
    referenced = {
        (output_root / relative).resolve()
        for row in records
        for relative in row["model_input"]["images"]
    }
    removed = 0
    for path in media_root.rglob("*"):
        if path.is_file() and path.resolve() not in referenced:
            path.unlink()
            removed += 1
    for path in sorted(media_root.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    return removed


def _stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:20]
    return f"{prefix}-{digest}"


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def _split_name(value: str) -> str:
    return "validation" if value == "val" else value


def _turn_phrase(turns: list[int]) -> str:
    fragments = []
    for value in turns:
        side = "右" if value > 0 else "左"
        fragments.append(f"向{side}转{abs(value)}度")
    if len(fragments) == 1:
        return fragments[0]
    return "先" + "，再".join(fragments)


def _bearing_phrase(bearing_deg: float) -> str:
    bearing = (float(bearing_deg) + 180.0) % 360.0 - 180.0
    sectors = [
        (0.0, "大致正前方"),
        (45.0, "右前方"),
        (90.0, "大致右侧"),
        (135.0, "右后方"),
        (-180.0, "大致后方"),
        (-135.0, "左后方"),
        (-90.0, "大致左侧"),
        (-45.0, "左前方"),
    ]
    _, phrase = min(
        sectors,
        key=lambda item: abs((bearing - item[0] + 180.0) % 360.0 - 180.0),
    )
    return phrase


def _options(record_id: str, correct_label: str) -> dict[str, Any]:
    labels = list(DIRECTION_CN)
    labels.sort(
        key=lambda label: hashlib.sha256(f"{record_id}:{label}".encode()).hexdigest()
    )
    letters = "ABCD"
    choices = [
        {"key": letter, "label": label, "text": DIRECTION_CN[label]}
        for letter, label in zip(letters, labels, strict=True)
    ]
    correct = next(item for item in choices if item["label"] == correct_label)
    return {
        "choices": choices,
        "surface": " ".join(f"{item['key']}. {item['text']}" for item in choices),
        "answer_key": correct["key"],
        "answer_text": correct["text"],
        "answer_surface": f"{correct['key']}. {correct['text']}",
    }


def _copy_media(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or source.stat().st_size != destination.stat().st_size:
        shutil.copy2(source, destination)
    return destination.as_posix()


def _extract_rgb_media(source: Path, destination: Path) -> str:
    """Materialize only the model-visible RGB channel from a sensor bundle."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as sensors:
        rgb = np.asarray(sensors["rgb"], dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"invalid RGB channel in {source}: {rgb.shape}")
    rewrite = True
    if destination.is_file():
        with Image.open(destination) as existing:
            rewrite = existing.size != (rgb.shape[1], rgb.shape[0])
    if rewrite:
        with tempfile.NamedTemporaryFile(
            suffix=".png", dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        Image.fromarray(rgb, mode="RGB").save(temporary)
        temporary.replace(destination)
    return destination.as_posix()


def _assistant_target(record: dict[str, Any], arm: str) -> str:
    answer = record["answer"]["surface"]
    if arm == "answer_only":
        return f"<answer>{answer}</answer>"
    if arm != "grounded_cot":
        raise ValueError(f"unknown supervision arm: {arm}")
    trace = record["grounded_trace"]
    return (
        f"<cue>{trace['cue']['surface']}</cue>"
        f"<transform>{trace['transform']['surface']}</transform>"
        f"<answer>{answer}</answer>"
    )


def _sft_record(record: dict[str, Any], arm: str) -> dict[str, Any]:
    images = [
        {"type": "image", "image": path} for path in record["model_input"]["images"]
    ]
    return {
        "record_id": record["record_id"],
        "format": "qwen_multimodal_chat",
        "task_id": record["task_id"],
        "split": record["split"],
        "scene_id": record["scene_id"],
        "base_fact_id": record["base_fact_id"],
        "supervision_arm": arm,
        "messages": [
            {"role": "system", "content": record["model_input"]["system"]},
            {
                "role": "user",
                "content": [
                    *images,
                    {"type": "text", "text": record["model_input"]["question"]},
                ],
            },
            {"role": "assistant", "content": _assistant_target(record, arm)},
        ],
        "loss_policy": {
            "train_on": "assistant_only",
            "role_spans": (
                ["answer"] if arm == "answer_only" else ["cue", "transform", "answer"]
            ),
        },
    }


def _self_rotation_records(
    *, facts_path: Path, output_root: Path, surfaces_per_fact: int
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fact in _read_jsonl(facts_path):
        view_ids = [str(value) for value in fact["model_view_ids"]]
        cue_view = str(fact["cue"]["selected_cue_view_id"])
        cue_index = view_ids.index(cue_view)
        net_right = int(fact["turn"]["net_right_deg"])
        history_turn = (1 if net_right > 0 else -1) * 60 * (
            len(view_ids) - 1 - cue_index
        )
        programs = fact["turn"]["surface_programs_deg"][:surfaces_per_fact]
        for surface_index, raw_program in enumerate(programs):
            program = [int(value) for value in raw_program]
            record_id = _stable_id(
                "transform-sft", fact["base_fact_id"], "surface", surface_index
            )
            option = _options(record_id, str(fact["answer"]["label"]))
            side = "右" if net_right > 0 else "左"
            copied_images = []
            for view_id, raw_path in zip(view_ids, fact["model_rgb"], strict=True):
                relative = Path("media") / "self_rotation" / str(
                    fact["episode_id"]
                ) / f"{view_id}.png"
                _copy_media(Path(raw_path), output_root / relative)
                copied_images.append(relative.as_posix())
            source_mode = str(fact["source_mode"])
            history_note = (
                "当前最后一图就是这条视觉线索所在的朝向。"
                if history_turn == 0
                else (
                    f"从第{cue_index + 1}张图到当前最后一图，我已经原地向{side}"
                    f"转了{abs(history_turn)}度。"
                )
            )
            memory_note = (
                "目标在最后一图中已经不可见，必须保留先前视角中的位置。"
                if source_mode == "episodic_memory"
                else "目标在当前图中仍可直接定位。"
            )
            category = str(fact["cue"]["category"])
            question = (
                f"你现在站在最后一张图的原地朝向。假设接下来{_turn_phrase(program)}，"
                f"转身后的画面不会提供。{category}会位于你的哪个方向？ "
                f"{option['surface']}"
            )
            cue_surface = (
                f"第{cue_index + 1}张图中可以确认{category}位于我的"
                f"{_bearing_phrase(fact['cue']['observed_direction']['bearing_deg'])}；"
                f"{memory_note}"
            )
            transform_surface = (
                f"{history_note}我保持{category}在场景中的位置不动，只更新自身朝向；"
                f"然后执行题设的“{_turn_phrase(program)}”。人向一侧转动时，"
                "同一物体在自我坐标中按相反方向等角度更新。"
            )
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "record_id": record_id,
                    "task_id": "self_rotation_query.v1",
                    "task_tags": [
                        "perspective_taking",
                        "mental_simulation",
                        source_mode,
                    ],
                    "base_fact_id": fact["base_fact_id"],
                    "family_id": fact["episode_id"],
                    "scene_id": fact["scene_id"],
                    "split": _split_name(str(fact["split"])),
                    "surface_variant_index": surface_index,
                    "model_input": {
                        "images": copied_images,
                        "system": (
                            f"这些图按时间顺序拍于同一站位；相邻两图之间，观察者原地向{side}"
                            "转60度。请只依据图像和已知动作更新物体相对你的方向。"
                        ),
                        "question": question,
                    },
                    "answer": {
                        "key": option["answer_key"],
                        "label": fact["answer"]["label"],
                        "text": option["answer_text"],
                        "surface": option["answer_surface"],
                    },
                    "grounded_trace": {
                        "schema_version": COT_SCHEMA_VERSION,
                        "cue": {
                            "surface": cue_surface,
                            "claims": [
                                {
                                    "predicate": "observed_in",
                                    "entity": fact["cue"]["entity_id"],
                                    "view_id": cue_view,
                                    "bearing_deg": fact["cue"]["observed_direction"][
                                        "bearing_deg"
                                    ],
                                },
                                {
                                    "predicate": "visible_pixels",
                                    "value": fact["cue"]["visible_pixels"],
                                },
                            ],
                        },
                        "transform": {
                            "surface": transform_surface,
                            "claims": [
                                {
                                    "predicate": "history_turn_right_deg",
                                    "value": history_turn,
                                },
                                {
                                    "predicate": "query_turn_right_deg",
                                    "value": sum(program),
                                },
                                {
                                    "predicate": "egocentric_update",
                                    "equation": "bearing_after = bearing_before - observer_turn",
                                },
                            ],
                        },
                        "conclusion": {"label": fact["answer"]["label"]},
                    },
                    "operation_graph": {
                        "nodes": ["G", "B_view_memory", "F_self", "R_bearing", "V"],
                        "typed_signature": (
                            "G(ordered_rgb,entity_hint)->B_view_memory->"
                            "F_self(turn_sequence)->R_bearing->V_direction"
                        ),
                    },
                    "certificate": {
                        "source_bundle": fact["source_bundle"],
                        "answer_measurement": fact["answer"],
                        "cue_measurement": fact["cue"]["observed_direction"],
                        "current_measurement": fact["cue"][
                            "current_direction_before_query"
                        ],
                        "target_orientation_view_withheld": fact["oracle"][
                            "target_orientation_view_id"
                        ],
                        "program_minimality": fact["program_minimality"],
                    },
                }
            )
    return records


def _entity_geometry(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(entity["name"]): entity
        for entity in snapshot["entities"]
        if str(entity.get("name", "")).startswith("among_")
    }


def _among_model_name(category: str) -> str:
    return OBJECT_CN.get(category, category.replace("_", " "))


def _among_base_layout(job: dict[str, Any], output_root: Path) -> dict[str, Any] | None:
    bundle = Path(job["bundle"])
    required = [
        bundle / "quality_report.json",
        bundle / "trajectory_selection.json",
        bundle / "trajectory_plan.json",
        bundle / "scene_snapshot.json",
    ]
    if not all(path.is_file() for path in required):
        return None
    quality = _read_json(required[0])
    gate = quality.get("gates", {}).get("T2", {})
    if not (
        quality.get("integrity_status") == "pass"
        and quality.get("visual_status") == "pass"
        and quality.get("trajectory_status") == "pass"
        and gate.get("status") == "pass"
    ):
        return None
    selection = _read_json(required[1])
    if selection.get("among5_protocol_version") != "omnigibson_mindcube_among_layout.v2":
        return None
    trajectory = _read_json(required[2])
    snapshot = _read_json(required[3])
    entities = _entity_geometry(snapshot)
    placements = selection["placements"]
    anchor = next(item for item in placements if item["role"] == "anchor")
    satellites = [item for item in placements if item["role"] == "satellite"]
    first_view = trajectory["views"][0]
    frame_yaw = yaw_from_xyzw(first_view["world_from_agent"]["rotation_xyzw"])
    view_target = {
        str(item["expected_target_name"]): f"view-{index:03d}"
        for index, item in enumerate(selection["camera_checks"])
    }
    copied_images = []
    for view in trajectory["views"]:
        view_id = str(view["view_id"])
        source = bundle / "views" / f"{view_id}.sensors.npz"
        if not source.is_file():
            return None
        relative = (
            Path("media")
            / "among5"
            / str(job["scene_model"])
            / str(job["variant_id"])
            / f"{view_id}.png"
        )
        _extract_rgb_media(source, output_root / relative)
        copied_images.append(relative.as_posix())
    return {
        "job": job,
        "bundle": bundle,
        "quality": quality,
        "selection": selection,
        "trajectory": trajectory,
        "entities": entities,
        "anchor": anchor,
        "satellites": satellites,
        "frame_yaw": frame_yaw,
        "view_target": view_target,
        "images": copied_images,
    }


def _measurement(
    layout: dict[str, Any], *, subject: str, reference: str, frame_yaw: float
) -> dict[str, Any]:
    entities = layout["entities"]
    subject_entity = entities[subject]
    reference_entity = entities[reference]
    point = subject_entity["aabb_center_m"]
    origin = reference_entity["aabb_center_m"]
    radius = horizontal_obb_radius(tuple(subject_entity["aabb_extent_m"]))
    radius += horizontal_obb_radius(tuple(reference_entity["aabb_extent_m"]))
    return direction_measurement(
        point_xy=(point[0], point[1]),
        origin_xy=(origin[0], origin[1]),
        camera_yaw_rad=frame_yaw,
        horizontal_radius_m=radius,
    ).as_dict()


def _among_question_record(
    *,
    layout: dict[str, Any],
    task_kind: str,
    subject: dict[str, Any],
    reference: dict[str, Any],
    query_index: int,
) -> dict[str, Any] | None:
    anchor = layout["anchor"]
    if task_kind == "cross_view_relation":
        measurement = _measurement(
            layout,
            subject=subject["name"],
            reference=reference["name"],
            frame_yaw=float(layout["frame_yaw"]),
        )
        question_core = (
            "以第1张图拍摄者面向脚凳的方向为“前”，"
            f"{_among_model_name(subject['category'])}在"
            f"{_among_model_name(reference['category'])}的哪个方向？"
        )
        transform_surface = (
            "相邻拍摄点沿同一方向绕脚凳移动90度并始终面向脚凳。"
            "我用四图中反复出现的脚凳对齐两个证据视角，把两件物体登记到"
            "以第1图朝向为“前”的同一平面，再比较它们的位置。"
        )
        graph_nodes = ["G", "F_anchor_register", "B_layout", "R", "V"]
        frame = "camera@view-000"
    elif task_kind == "object_anchored_perspective":
        origin = layout["entities"][reference["name"]]["aabb_center_m"]
        anchor_center = layout["entities"][anchor["name"]]["aabb_center_m"]
        query_yaw = yaw_facing(
            (origin[0], origin[1]),
            (anchor_center[0], anchor_center[1]),
        )
        measurement = _measurement(
            layout,
            subject=subject["name"],
            reference=reference["name"],
            frame_yaw=query_yaw,
        )
        question_core = (
            f"想象你站在{_among_model_name(reference['category'])}的位置，"
            f"正面朝向脚凳。{_among_model_name(subject['category'])}"
            "会位于你的哪个方向？"
        )
        transform_surface = (
            "我先借助各图共同出现的脚凳和90度绕行顺序恢复四件物体的环形布局；"
            f"再把坐标原点移到{_among_model_name(reference['category'])}，"
            "以它指向脚凳的方向作为新的正前方，换算目标物体的方位。"
        )
        graph_nodes = ["G", "F_anchor_register", "B_layout", "P", "R", "V"]
        frame = f"object@{reference['name']}:facing_anchor"
    else:
        raise ValueError(f"unknown Among task kind: {task_kind}")
    if (
        float(measurement["center_margin_deg"]) < 15.0
        or float(measurement["effective_margin_deg"]) < 8.0
    ):
        return None
    job = layout["job"]
    subject_view = layout["view_target"][subject["name"]]
    reference_view = layout["view_target"][reference["name"]]
    if subject_view == reference_view:
        return None
    base_fact_id = _stable_id(
        "among-fact",
        job["family_id"],
        task_kind,
        subject["category"],
        reference["category"],
    )
    record_id = _stable_id("transform-sft", base_fact_id, job["variant_id"])
    # Keep wording and option order identical across counterfactual siblings;
    # only the rendered evidence and the correct label may change.
    option = _options(base_fact_id, str(measurement["label"]))
    subject_index = int(subject_view.split("-")[-1]) + 1
    reference_index = int(reference_view.split("-")[-1]) + 1
    cue_surface = (
        f"第{subject_index}张图中，脚凳位于"
        f"{_among_model_name(subject['category'])}前方；"
        f"第{reference_index}张图中，脚凳位于"
        f"{_among_model_name(reference['category'])}前方。"
        "这两件物体没有在同一张图中出现。"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id,
        "task_id": f"among5_{task_kind}.v1",
        "task_tags": [
            "cognitive_mapping",
            "cross_view_integration",
            (
                "perspective_taking"
                if task_kind == "object_anchored_perspective"
                else "spatial_relation"
            ),
        ],
        "base_fact_id": base_fact_id,
        "counterfactual_group_id": base_fact_id,
        "family_id": job["family_id"],
        "scene_id": job["scene_model"],
        "split": _split_name(str(job["split"])),
        "variant_id": job["variant_id"],
        "query_index": query_index,
        "query_signature": {
            "task_kind": task_kind,
            "subject_category": subject["category"],
            "reference_category": reference["category"],
        },
        "model_input": {
            "images": layout["images"],
            "system": (
                "四张图来自同一布局：相机位于中心脚凳与外围物体之间，"
                "相邻拍摄点沿逆时针方向绕脚凳移动90度，并始终面向脚凳。"
                "每张图只给出脚凳和一个外围物体的局部证据。"
            ),
            "question": f"{question_core} {option['surface']}",
        },
        "answer": {
            "key": option["answer_key"],
            "label": measurement["label"],
            "text": option["answer_text"],
            "surface": option["answer_surface"],
        },
        "grounded_trace": {
            "schema_version": COT_SCHEMA_VERSION,
            "cue": {
                "surface": cue_surface,
                "claims": [
                    {
                        "predicate": "anchor_before_satellite",
                        "view_id": subject_view,
                        "satellite": subject["name"],
                    },
                    {
                        "predicate": "anchor_before_satellite",
                        "view_id": reference_view,
                        "satellite": reference["name"],
                    },
                    {
                        "predicate": "not_co_visible",
                        "entities": [subject["name"], reference["name"]],
                    },
                ],
            },
            "transform": {
                "surface": transform_surface,
                "claims": [
                    {
                        "predicate": "register_views",
                        "anchor": anchor["name"],
                        "ordered_view_step_deg": 90,
                    },
                    {"predicate": "query_frame", "frame": frame},
                ],
            },
            "conclusion": {"label": measurement["label"]},
        },
        "operation_graph": {
            "nodes": graph_nodes,
            "typed_signature": (
                "G(partial_views,anchor)->F_register(anchor,ordered_motion)->"
                "B_layout5->R/P(query_frame)->V_direction"
            ),
        },
        "certificate": {
            "source_bundle": str(layout["bundle"]),
            "protocol_version": layout["selection"]["among5_protocol_version"],
            "visibility_contract": layout["selection"]["visibility_contract"],
            "subject_view_id": subject_view,
            "reference_view_id": reference_view,
            "co_visible_view_ids": [],
            "measurement": measurement,
            "frame": frame,
            "quality_gate": layout["quality"]["gates"]["T2"],
        },
    }


def _among_records(
    *,
    sweep_plan_path: Path,
    output_root: Path,
    questions_per_layout: int,
    required_variants: list[str],
    task_kinds: list[str],
) -> list[dict[str, Any]]:
    supported_task_kinds = {
        "cross_view_relation",
        "object_anchored_perspective",
    }
    unsupported = set(task_kinds) - supported_task_kinds
    if unsupported:
        raise ValueError(f"unsupported Among task kinds: {sorted(unsupported)}")
    plan = _read_json(sweep_plan_path)
    candidates_by_family: dict[
        str, dict[str, dict[tuple[str, str, str], dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(dict))
    for job in plan["jobs"]:
        if str(job["variant_id"]) not in required_variants:
            continue
        layout = _among_base_layout(job, output_root)
        if layout is None:
            continue
        satellites = sorted(layout["satellites"], key=lambda item: item["category"])
        candidates = []
        for task_kind in task_kinds:
            for subject in satellites:
                for reference in satellites:
                    if subject["name"] == reference["name"]:
                        continue
                    row = _among_question_record(
                        layout=layout,
                        task_kind=task_kind,
                        subject=subject,
                        reference=reference,
                        query_index=len(candidates),
                    )
                    if row is not None:
                        candidates.append(row)
        family_id = str(job["family_id"])
        variant_id = str(job["variant_id"])
        for row in candidates:
            signature = row["query_signature"]
            key = (
                str(signature["task_kind"]),
                str(signature["subject_category"]),
                str(signature["reference_category"]),
            )
            candidates_by_family[family_id][variant_id][key] = row

    records: list[dict[str, Any]] = []
    for family_id, variants in sorted(candidates_by_family.items()):
        if any(variant not in variants for variant in required_variants):
            continue
        common_keys = set(variants[required_variants[0]])
        for variant in required_variants[1:]:
            common_keys &= set(variants[variant])
        identity = variants["identity"]
        buckets: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
        for key in common_keys:
            row = identity[key]
            buckets[(row["task_id"], row["answer"]["label"])].append(key)
        for keys in buckets.values():
            keys.sort(key=lambda key: _stable_id("query-menu", family_id, *key))
        selected_keys: list[tuple[str, str, str]] = []
        task_ids = sorted({task_id for task_id, _ in buckets})
        if not task_ids:
            continue
        task_budget = {
            task_id: questions_per_layout // len(task_ids) for task_id in task_ids
        }
        for task_id in task_ids[: questions_per_layout % len(task_ids)]:
            task_budget[task_id] += 1
        family_offset = int(hashlib.sha256(family_id.encode()).hexdigest()[:8], 16)
        for task_id in task_ids:
            labels = sorted(label for candidate_task, label in buckets if candidate_task == task_id)
            if labels:
                offset = family_offset % len(labels)
                labels = labels[offset:] + labels[:offset]
            while task_budget[task_id] > 0 and any(
                buckets[(task_id, label)] for label in labels
            ):
                for label in labels:
                    bucket = buckets[(task_id, label)]
                    if bucket and task_budget[task_id] > 0:
                        selected_keys.append(bucket.pop(0))
                        task_budget[task_id] -= 1
        if len(selected_keys) < questions_per_layout:
            continue
        for query_index, key in enumerate(selected_keys):
            for variant in required_variants:
                row = variants[variant][key]
                row["query_index"] = query_index
                row["counterfactual_variants"] = required_variants
                records.append(row)
    return records


def _episode_records(
    isolated_records: list[dict[str, Any]], arm: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in isolated_records:
        if row["task_id"].startswith("among5_"):
            grouped[(row["family_id"], row["variant_id"])].append(row)
    episodes = []
    for (family_id, variant_id), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["query_index"]))
        used_pairs: set[frozenset[str]] = set()
        selected = []
        for row in rows:
            signature = row["query_signature"]
            pair = frozenset(
                [signature["subject_category"], signature["reference_category"]]
            )
            if pair in used_pairs:
                continue
            used_pairs.add(pair)
            selected.append(row)
        if len(selected) < 2:
            continue
        first = selected[0]
        image_content = [
            {"type": "image", "image": path}
            for path in first["model_input"]["images"]
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": first["model_input"]["system"]}
        ]
        for index, row in enumerate(selected):
            content: Any
            if index == 0:
                content = [
                    *image_content,
                    {"type": "text", "text": row["model_input"]["question"]},
                ]
            else:
                content = row["model_input"]["question"]
            messages.extend(
                [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": _assistant_target(row, arm)},
                ]
            )
        episodes.append(
            {
                "record_id": _stable_id("transform-episode", family_id, variant_id, arm),
                "format": "qwen_multimodal_chat",
                "sample_type": "shared_state_multi_read_episode",
                "scene_id": first["scene_id"],
                "family_id": family_id,
                "variant_id": variant_id,
                "split": first["split"],
                "supervision_arm": arm,
                "member_record_ids": [row["record_id"] for row in selected],
                "messages": messages,
                "loss_policy": {
                    "train_on": "assistant_only",
                    "role_spans": (
                        ["answer"]
                        if arm == "answer_only"
                        else ["cue", "transform", "answer"]
                    ),
                },
            }
        )
    return episodes


def _balanced_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Balance self-rotation semantics without counting paraphrases as facts."""

    selected = [row for row in records if row["task_id"].startswith("among5_")]
    self_rows = [row for row in records if row["task_id"] == "self_rotation_query.v1"]
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in self_rows:
        by_split[row["split"]].append(row)
    for split, rows in by_split.items():
        facts_by_label: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in rows:
            facts_by_label[row["answer"]["label"]][row["base_fact_id"]].append(row)
        if set(facts_by_label) != set(DIRECTION_CN):
            raise ValueError(f"self-rotation split lacks a direction label: {split}")
        fact_floor = min(len(facts) for facts in facts_by_label.values())
        for label in sorted(facts_by_label):
            fact_ids = sorted(
                facts_by_label[label],
                key=lambda fact_id: _stable_id("balanced-fact", split, label, fact_id),
            )[:fact_floor]
            for fact_id in fact_ids:
                # A paraphrase/program surface is not an independent fact.
                # The balanced causal arm keeps exactly one surface per fact;
                # the full export remains available for language robustness.
                selected.append(
                    min(
                        facts_by_label[label][fact_id],
                        key=lambda row: int(row.get("surface_variant_index", 0)),
                    )
                )
    return sorted(selected, key=lambda row: row["record_id"])


def _extract_answer(target: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", target)
    if not match:
        raise ValueError("assistant target has no answer tag")
    return match.group(1)


def _evaluation_input(row: dict[str, Any]) -> dict[str, Any]:
    axes: dict[str, Any] = {"task_tags": row["task_tags"]}
    if row["task_id"] == "self_rotation_query.v1":
        transform_claims = {
            claim["predicate"]: claim.get("value")
            for claim in row["grounded_trace"]["transform"]["claims"]
        }
        axes.update(
            {
                "source_mode": row["task_tags"][-1],
                "query_turn_deg": abs(
                    int(transform_claims["query_turn_right_deg"])
                ),
                "history_turn_deg": abs(
                    int(transform_claims["history_turn_right_deg"])
                ),
            }
        )
    else:
        axes.update(
            {
                "variant_id": row["variant_id"],
                "counterfactual_group_id": row["counterfactual_group_id"],
                "query_signature": row["query_signature"],
            }
        )
    return {
        "record_id": row["record_id"],
        "task_id": row["task_id"],
        "scene_id": row["scene_id"],
        "split": row["split"],
        "images": row["model_input"]["images"],
        "system": row["model_input"]["system"],
        "question": row["model_input"]["question"],
        "evaluation_axes": axes,
    }


def _audit(records: list[dict[str, Any]], exports: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    scenes_by_split: dict[str, set[str]] = defaultdict(set)
    for row in records:
        scenes_by_split[row["split"]].add(row["scene_id"])
    split_pairs = list(scenes_by_split)
    split_disjoint = all(
        not (scenes_by_split[left] & scenes_by_split[right])
        for index, left in enumerate(split_pairs)
        for right in split_pairs[index + 1 :]
    )
    paired_ok = True
    for split in scenes_by_split:
        answer_rows = {
            row["record_id"]: row
            for row in exports[f"isolated_answer_only_{split}"]
        }
        cot_rows = {
            row["record_id"]: row
            for row in exports[f"isolated_grounded_cot_{split}"]
        }
        if set(answer_rows) != set(cot_rows):
            paired_ok = False
            continue
        for record_id in answer_rows:
            left = answer_rows[record_id]
            right = cot_rows[record_id]
            if left["messages"][:-1] != right["messages"][:-1]:
                paired_ok = False
            if _extract_answer(left["messages"][-1]["content"]) != _extract_answer(
                right["messages"][-1]["content"]
            ):
                paired_ok = False
    cot_roles_ok = all(
        row["grounded_trace"]["cue"]["claims"]
        and row["grounded_trace"]["transform"]["claims"]
        and row["grounded_trace"]["conclusion"]["label"] == row["answer"]["label"]
        for row in records
    )
    among_groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        if row["task_id"].startswith("among5_"):
            among_groups[row["counterfactual_group_id"]][row["variant_id"]] = row
    required_variants = {"identity", "rotate90", "mirror", "permute1", "radius185"}
    complete_counterfactual_groups = bool(among_groups) and all(
        set(variants) == required_variants for variants in among_groups.values()
    )
    counterfactual_visible_contract_ok = bool(among_groups) and all(
        len(
            {
                (
                    row["model_input"]["system"],
                    row["model_input"]["question"],
                    tuple(sorted(row["query_signature"].items())),
                )
                for row in variants.values()
            }
        )
        == 1
        for variants in among_groups.values()
    )

    def rate(predicate: Any, expectation: Any) -> float:
        relevant = [variants for variants in among_groups.values() if predicate(variants)]
        if not relevant:
            return 0.0
        return sum(bool(expectation(variants)) for variants in relevant) / len(relevant)

    def label(variants: dict[str, dict[str, Any]], variant: str) -> str:
        return str(variants[variant]["answer"]["label"])

    relation_group = lambda variants: variants["identity"]["task_id"].endswith(  # noqa: E731
        "cross_view_relation.v1"
    )
    relation_chirality_group = lambda variants: (  # noqa: E731
        relation_group(variants)
        and label(variants, "identity") in {"left", "right"}
    )
    relation_depth_group = lambda variants: (  # noqa: E731
        relation_group(variants)
        and label(variants, "identity") in {"front", "back"}
    )
    counterfactual_rates = {
        "radius_invariance": rate(
            lambda variants: True,
            lambda variants: label(variants, "identity") == label(variants, "radius185"),
        ),
        "rotate90_changes_fixed_frame_relation": rate(
            relation_group,
            lambda variants: label(variants, "identity") != label(variants, "rotate90"),
        ),
        "permutation_changes_fixed_frame_relation": rate(
            relation_group,
            lambda variants: label(variants, "identity") != label(variants, "permute1"),
        ),
        "mirror_flips_fixed_frame_chirality": rate(
            relation_chirality_group,
            lambda variants: label(variants, "identity") != label(variants, "mirror"),
        ),
        "mirror_preserves_fixed_frame_depth": rate(
            relation_depth_group,
            lambda variants: label(variants, "identity") == label(variants, "mirror"),
        ),
    }
    counterfactual_semantics_ok = bool(among_groups) and (
        counterfactual_rates["radius_invariance"] >= 0.98
        and counterfactual_rates["rotate90_changes_fixed_frame_relation"] >= 0.95
        and counterfactual_rates["permutation_changes_fixed_frame_relation"] >= 0.95
        and counterfactual_rates["mirror_flips_fixed_frame_chirality"] >= 0.95
        and counterfactual_rates["mirror_preserves_fixed_frame_depth"] >= 0.95
    )
    counts = Counter((row["task_id"], row["split"]) for row in records)
    answers = Counter((row["task_id"], row["answer"]["label"]) for row in records)
    return {
        "schema_version": "epispace.transform_dataset_audit.v1",
        "status": (
            "pass"
            if split_disjoint
            and paired_ok
            and cot_roles_ok
            and complete_counterfactual_groups
            and counterfactual_visible_contract_ok
            and counterfactual_semantics_ok
            else "fail"
        ),
        "checks": {
            "scene_disjoint_splits": split_disjoint,
            "answer_and_cot_arms_have_identical_inputs_and_labels": paired_ok,
            "every_cot_has_grounded_cue_transform_conclusion": cot_roles_ok,
            "every_among_query_has_all_five_counterfactual_variants": (
                complete_counterfactual_groups
            ),
            "counterfactual_siblings_share_question_options_and_query": (
                counterfactual_visible_contract_ok
            ),
            "counterfactual_answers_follow_declared_semantics": (
                counterfactual_semantics_ok
            ),
        },
        "counterfactual_rates": {
            key: round(value, 6) for key, value in counterfactual_rates.items()
        },
        "complete_counterfactual_query_groups": len(among_groups),
        "counts_by_task_split": {
            f"{task}/{split}": count for (task, split), count in sorted(counts.items())
        },
        "answer_distribution": {
            f"{task}/{answer}": count
            for (task, answer), count in sorted(answers.items())
        },
        "scene_counts": {
            split: len(scenes) for split, scenes in sorted(scenes_by_split.items())
        },
    }


def build_transform_dataset(config_path: Path) -> dict[str, Any]:
    """Build portable, paired training arms and hidden evaluation certificates."""

    config_path = config_path.resolve()
    config = _read_json(config_path)
    if config.get("schema_version") != "epispace.transform_dataset_config.v1":
        raise ValueError("unsupported Transform dataset config")
    output_root = _resolve(config_path, str(config["output_dir"]))
    output_root.mkdir(parents=True, exist_ok=True)
    self_records = _self_rotation_records(
        facts_path=_resolve(config_path, str(config["self_rotation_facts"])),
        output_root=output_root,
        surfaces_per_fact=int(config["self_rotation_surfaces_per_fact"]),
    )
    among_records = _among_records(
        sweep_plan_path=_resolve(config_path, str(config["among5_sweep_plan"])),
        output_root=output_root,
        questions_per_layout=int(config["among5_questions_per_layout"]),
        required_variants=[str(value) for value in config["among5_required_variants"]],
        task_kinds=[str(value) for value in config["among5_task_kinds"]],
    )
    records = sorted(
        [*self_records, *among_records], key=lambda row: str(row["record_id"])
    )
    pruned_media_files = _prune_unreferenced_media(output_root, records)
    balanced_records = _balanced_records(records)
    _write_jsonl(output_root / "teacher_records.jsonl", records)
    exports: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation", "test"):
        split_records = [row for row in records if row["split"] == split]
        for arm in ("answer_only", "grounded_cot"):
            key = f"isolated_{arm}_{split}"
            exports[key] = [_sft_record(row, arm) for row in split_records]
            _write_jsonl(output_root / f"{key}.jsonl", exports[key])
            episodes = _episode_records(split_records, arm)
            _write_jsonl(output_root / f"episode_{arm}_{split}.jsonl", episodes)
            balanced_key = f"balanced_isolated_{arm}_{split}"
            balanced_split = [row for row in balanced_records if row["split"] == split]
            balanced_export = [_sft_record(row, arm) for row in balanced_split]
            _write_jsonl(output_root / f"{balanced_key}.jsonl", balanced_export)
            if split == "train":
                atomic_composition_rows = [
                    row
                    for row in balanced_split
                    if row["task_id"].startswith("among5_")
                    or abs(
                        int(
                            next(
                                claim["value"]
                                for claim in row["grounded_trace"]["transform"]["claims"]
                                if claim["predicate"] == "query_turn_right_deg"
                            )
                        )
                    )
                    == 60
                ]
                _write_jsonl(
                    output_root / f"atomic_composition_{arm}_train.jsonl",
                    [_sft_record(row, arm) for row in atomic_composition_rows],
                )
    def evaluation_oracle_row(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "record_id": row["record_id"],
            "split": row["split"],
            "answer": row["answer"],
            "certificate": row["certificate"],
        }

    evaluation_inputs = []
    evaluation_oracle = []
    _write_jsonl(
        output_root / "eval_inputs_all.jsonl",
        [_evaluation_input(row) for row in records],
    )
    _write_jsonl(
        output_root / "eval_oracle_all.jsonl",
        [evaluation_oracle_row(row) for row in records],
    )
    _write_jsonl(
        output_root / "eval_inputs_balanced_all.jsonl",
        [_evaluation_input(row) for row in balanced_records],
    )
    _write_jsonl(
        output_root / "eval_oracle_balanced_all.jsonl",
        [evaluation_oracle_row(row) for row in balanced_records],
    )
    _write_jsonl(
        output_root / "eval_inputs_full_test.jsonl",
        [
            _evaluation_input(row)
            for row in records
            if row["split"] == "test"
        ],
    )
    _write_jsonl(
        output_root / "eval_oracle_full_test.jsonl",
        [
            evaluation_oracle_row(row)
            for row in records
            if row["split"] == "test"
        ],
    )
    for row in balanced_records:
        if row["split"] != "test":
            continue
        evaluation_inputs.append(
            _evaluation_input(row)
        )
        evaluation_oracle.append(evaluation_oracle_row(row))
    _write_jsonl(output_root / "eval_inputs_test.jsonl", evaluation_inputs)
    _write_jsonl(output_root / "eval_oracle_test.jsonl", evaluation_oracle)
    audit = _audit(records, exports)
    complete_among_families = len(
        {row["family_id"] for row in among_records if row["variant_id"] == "identity"}
    )
    data_scale_ok = (
        len({row["base_fact_id"] for row in self_records})
        >= int(config["minimum_self_rotation_base_facts"])
        and complete_among_families >= int(config["minimum_complete_among5_families"])
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": config["dataset_id"],
        "status": (
            "data_ready"
            if records and audit["status"] == "pass" and data_scale_ok
            else "blocked"
        ),
        "counts": {
            "teacher_records": len(records),
            "self_rotation_records": len(self_records),
            "among5_records": len(among_records),
            "complete_among5_families": complete_among_families,
            "unique_base_facts": len({row["base_fact_id"] for row in records}),
            "pruned_unreferenced_media_files": pruned_media_files,
            "by_split": dict(Counter(row["split"] for row in records)),
            "balanced_by_split": dict(
                Counter(row["split"] for row in balanced_records)
            ),
        },
        "audit": audit,
        "readiness": {
            "scale_gate": data_scale_ok,
            "minimum_self_rotation_base_facts": int(
                config["minimum_self_rotation_base_facts"]
            ),
            "minimum_complete_among5_families": int(
                config["minimum_complete_among5_families"]
            ),
        },
        "sources": {
            "self_rotation_facts": str(
                _resolve(config_path, str(config["self_rotation_facts"]))
            ),
            "among5_sweep_plan": str(
                _resolve(config_path, str(config["among5_sweep_plan"]))
            ),
        },
    }
    _write_json(output_root / "corpus_report.json", report)
    return report
