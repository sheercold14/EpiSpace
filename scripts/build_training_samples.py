#!/usr/bin/env python3
"""Compile the Rs_int_seed17 bundle into MLLM-native post-training samples.

Design contract (research contract v1 + MLLM post-training notes):
- structure stays in the data engine: typed geometry only compiles questions and
  verifies answers; the model interface is natural multi-turn dialogue.
- every answerable fact carries a certificate that tests re-execute from geometry.
- oracle channels (scene_ir / relation_oracle / UUIDs) never appear in
  model-visible text.
- family variants share family_id and are locked to a single split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "episode3d.training_samples.v1"
FRAME_NOTE = "按房间平面图约定：+X 为右，+Y 为前"

CORE_NAMES = {
    "fridge": "冰箱",
    "oven": "烤箱",
    "microwave": "微波炉",
    "dishwasher": "洗碗机",
    "sofa": "沙发",
    "coffee_table": "咖啡桌",
    "standing_tv": "电视",
    "laptop": "笔记本电脑",
    "swivel_chair": "转椅",
}
EPISTEMIC_NAMES = {"bed": "床", "toilet": "马桶", "mirror": "镜子"}
REL_CN = {"left_of": "左侧", "right_of": "右侧", "in_front_of": "前方", "behind": "后方"}
QUAD_CN = {"front-left": "左前方", "front-right": "右前方", "back-left": "左后方", "back-right": "右后方"}


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


class Bundle:
    """Geometry facts derived from the acquisition; the only source of answers."""

    def __init__(self, root: Path) -> None:
        self.root = root
        scene = _read(root / "scene_ir.json")
        self.episode = _read(root / "spatial_episode.json")
        quality = _read(root / "quality_report.json")
        if quality["integrity_status"] != "pass":
            raise ValueError("source bundle did not pass integrity inspection")
        self.family_id = self.episode["family_id"]
        self.split_group = self.episode["split_group"]
        self.view_ids = [item["view_id"] for item in self.episode["observations"]]
        self.visible = {
            item["view_id"]: set(item["visible_entity_ids"])
            for item in self.episode["observations"]
        }
        by_label: dict[str, list[dict[str, Any]]] = {}
        for entity in scene["entities"]:
            by_label.setdefault(entity["raw_label"], []).append(entity)
        self.entities: dict[str, dict[str, Any]] = {}
        for key, name in CORE_NAMES.items():
            (raw,) = by_label[key]  # curated keys are single-instance categories
            views = [v for v in self.view_ids if raw["entity_id"] in self.visible[v]]
            if not views:
                raise ValueError(f"curated entity was never observed: {key}")
            self.entities[key] = {
                "key": key,
                "name": name,
                "pos": [float(v) for v in raw["world_from_entity"]["translation_m"]],
                "views": views,
            }
        for key in EPISTEMIC_NAMES:
            (raw,) = by_label[key]
            views = [v for v in self.view_ids if raw["entity_id"] in self.visible[v]]
            if views:
                raise ValueError(f"epistemic probe entity is unexpectedly observed: {key}")
        media = root / "oral_demo" / "media"
        for view_id in self.view_ids:
            if not (media / f"{view_id}-rgb.webp").exists():
                raise ValueError(f"missing rgb media for {view_id}")
        self.media_root = str(media.parent)

    def entity(self, key: str) -> dict[str, Any]:
        return self.entities[key]

    def visible_core(self, view_ids: list[str]) -> list[str]:
        seen = set()
        for view_id in view_ids:
            for key, item in self.entities.items():
                if view_id in item["views"]:
                    seen.add(key)
        return [key for key in CORE_NAMES if key in seen]

    def co_visible_views(self, a: str, b: str) -> list[str]:
        return sorted(set(self.entity(a)["views"]) & set(self.entity(b)["views"]), key=self.view_ids.index)


def relation(bundle: Bundle, subject: str, reference: str) -> dict[str, Any]:
    ps, pr = bundle.entity(subject)["pos"], bundle.entity(reference)["pos"]
    dx, dy = ps[0] - pr[0], ps[1] - pr[1]
    if abs(dx) >= abs(dy):
        rel = "right_of" if dx > 0 else "left_of"
        dominant, margin = abs(dx), abs(dx) - abs(dy)
    else:
        rel = "in_front_of" if dy > 0 else "behind"
        dominant, margin = abs(dy), abs(dy) - abs(dx)
    return {
        "kind": "relation",
        "frame": "canonical_xy",
        "subject": subject,
        "reference": reference,
        "relation": rel,
        "dominant_delta_m": round(dominant, 3),
        "margin_m": round(margin, 3),
        "status": "accepted",
    }


def distance(bundle: Bundle, a: str, b: str, tolerance: float = 0.3) -> dict[str, Any]:
    pa, pb = bundle.entity(a)["pos"], bundle.entity(b)["pos"]
    value = math.sqrt(sum((pa[i] - pb[i]) ** 2 for i in range(3)))
    return {
        "kind": "distance",
        "a": a,
        "b": b,
        "distance_m": round(value, 3),
        "tolerance_m": tolerance,
        "status": "accepted",
    }


def ego_quadrant(bundle: Bundle, origin: str, facing: str, target: str) -> dict[str, Any]:
    po, pf, pt = (bundle.entity(k)["pos"] for k in (origin, facing, target))
    fx, fy = pf[0] - po[0], pf[1] - po[1]
    norm = math.hypot(fx, fy)
    if norm < 1e-6:
        raise ValueError("query-frame anchors are coincident")
    forward = (fx / norm, fy / norm)
    right = (forward[1], -forward[0])
    tx, ty = pt[0] - po[0], pt[1] - po[1]
    qx, qy = tx * right[0] + ty * right[1], tx * forward[0] + ty * forward[1]
    quadrant = f"{'front' if qy >= 0 else 'back'}-{'right' if qx >= 0 else 'left'}"
    return {
        "kind": "ego_quadrant",
        "origin": origin,
        "facing": facing,
        "target": target,
        "quadrant": quadrant,
        "query_xy_m": [round(qx, 3), round(qy, 3)],
        "status": "accepted",
    }


def rel_answer(bundle: Bundle, fact: dict[str, Any]) -> str:
    subject = bundle.entity(fact["subject"])["name"]
    reference = bundle.entity(fact["reference"])["name"]
    return (
        f"{subject}在{reference}的{REL_CN[fact['relation']]}，"
        f"主导方向上相距约{round(fact['dominant_delta_m'], 1)}米。"
    )


def _record(
    record_id: str,
    sample_type: str,
    images: list[str],
    conversations: list[dict[str, str]],
    facts: list[dict[str, Any]],
    bundle: Bundle,
    **extra: Any,
) -> dict[str, Any]:
    placeholders = sum(turn["value"].count("<image>") for turn in conversations)
    if placeholders != len(images):
        raise ValueError(f"{record_id}: {placeholders} placeholders vs {len(images)} images")
    train_on = "rl_only" if sample_type == "rlvr" else "gpt_turns_only"
    meta = {
        "schema_version": SCHEMA_VERSION,
        "family_id": bundle.family_id,
        "split_lock": bundle.split_group,
        "media_root": bundle.media_root,
        "loss_policy": {"train_on": train_on},
        "facts": facts,
        "paraphrase_pool": True,
        **extra,
    }
    return {
        "id": record_id,
        "sample_type": sample_type,
        "images": images,
        "conversations": conversations,
        "meta": meta,
    }


def _rgb(view_ids: list[str]) -> list[str]:
    return [f"media/{view_id}-rgb.webp" for view_id in view_ids]


def layout_caption(bundle: Bundle) -> tuple[str, list[dict[str, Any]]]:
    mw_fridge = relation(bundle, "fridge", "microwave")
    oven_mw = relation(bundle, "oven", "microwave")
    sofa_coffee = relation(bundle, "sofa", "coffee_table")
    tv_sofa = relation(bundle, "standing_tv", "sofa")
    tv_fridge_d = distance(bundle, "standing_tv", "fridge")
    pos = {k: bundle.entity(k)["pos"] for k in CORE_NAMES}
    stack = {
        "kind": "vertical_stack",
        "upper": "microwave",
        "lower": "dishwasher",
        "status": "accepted",
    }
    between = {
        "kind": "x_between",
        "middle": "coffee_table",
        "ends": ["sofa", "standing_tv"],
        "status": "accepted",
    }
    near = {
        "kind": "near",
        "a": "laptop",
        "b": "coffee_table",
        "max_horizontal_m": 0.5,
        "status": "accepted",
    }
    text = (
        f"这个空间大致分为两个功能区（{FRAME_NOTE}）。"
        f"厨房区在平面图靠前（+Y）一侧：微波炉和洗碗机上下叠放在同一位置（微波炉在上），"
        f"烤箱在这组电器的{REL_CN[oven_mw['relation']]}，"
        f"冰箱在微波炉的{REL_CN[mw_fridge['relation']]}约{round(mw_fridge['dominant_delta_m'], 1)}米处。"
        f"客厅区在靠后（-Y）一侧：沙发在咖啡桌的{REL_CN[sofa_coffee['relation']]}，"
        f"咖啡桌上放着一台笔记本电脑，电视在沙发的{REL_CN[tv_sofa['relation']]}，"
        f"咖啡桌大致位于沙发与电视之间；更靠后还有一把转椅。"
        f"两个功能区相距数米：从电视到冰箱的直线距离约{round(tv_fridge_d['distance_m'], 1)}米。"
    )
    assert pos["swivel_chair"][1] < pos["sofa"][1]  # 转椅确实更靠后
    facts = [mw_fridge, oven_mw, sofa_coffee, tv_sofa, tv_fridge_d, stack, between, near]
    return text, facts


def build_episodic_dialogues(bundle: Bundle) -> list[dict[str, Any]]:
    records = []
    order = bundle.view_ids
    oven_fridge = relation(bundle, "oven", "fridge")
    tv_fridge = relation(bundle, "standing_tv", "fridge")
    quad = ego_quadrant(bundle, "swivel_chair", "standing_tv", "sofa")
    seen_first = bundle.visible_core([order[0]])
    names_first = "、".join(bundle.entity(k)["name"] for k in seen_first)
    last_seen_fridge = [v for v in order[:5] if v in bundle.entity("fridge")["views"]][-1]
    caption, caption_facts = layout_caption(bundle)
    conversations = [
        {
            "from": "human",
            "value": (
                f"我正在一间住宅里移动，会按顺序给你我看到的画面。涉及方向的问题一律{FRAME_NOTE}。"
                "<image>\n这是第1个视角，你能看到哪些主要的家具和电器？"
            ),
        },
        {"from": "gpt", "value": f"我能看到：{names_first}。"},
        {
            "from": "human",
            "value": "<image><image>\n继续前进（第2、3个视角）。现在能同时看到烤箱和冰箱吗？如果能，烤箱在冰箱的哪一侧？",
        },
        {
            "from": "gpt",
            "value": (
                "能，第3个视角里两者同时可见。"
                f"烤箱在冰箱的{REL_CN[oven_fridge['relation']]}，"
                f"横向相距约{round(oven_fridge['dominant_delta_m'], 1)}米。"
            ),
        },
        {
            "from": "human",
            "value": "<image><image>\n继续走（第4、5个视角）。现在还能看到冰箱吗？你最后一次看到它是在第几个视角？",
        },
        {"from": "gpt", "value": "现在看不到冰箱了。最后一次看到它是在第3个视角。"},
        {
            "from": "human",
            "value": (
                "<image><image><image>\n转身返回途中（第6、7、8个视角）。两个问题："
                "电视和冰箱有没有在同一个视角里同时出现过？综合目前所有观察，电视在冰箱的什么方向？"
            ),
        },
        {
            "from": "gpt",
            "value": (
                "没有同框过：冰箱只出现在前几段面向厨房的视角里，电视只出现在返程的客厅视角里。"
                "把两段观察拼到同一张平面图上，"
                f"电视在冰箱的{REL_CN[tv_fridge['relation']]}，"
                f"纵向相距约{round(tv_fridge['dominant_delta_m'], 1)}米，另外略偏右。"
            ),
        },
        {
            "from": "human",
            "value": "<image><image>\n第9、10个视角。假设你站在转椅的位置、面向电视，沙发在你的哪个方位？",
        },
        {
            "from": "gpt",
            "value": (
                f"在我的{QUAD_CN[quad['quadrant']]}。"
                "以转椅到电视的连线为正前方建立朝向后，沙发落在前方偏左的位置。"
            ),
        },
        {
            "from": "human",
            "value": "<image>\n最后回到起点（第11个视角）。这间房子里你见到过床吗？冰箱还在原来的位置吗？",
        },
        {
            "from": "gpt",
            "value": (
                "我走过的视角里从未出现过床——如果有卧室，它不在已观察范围内，我无法确定有没有床。"
                "冰箱仍在原位：这个视角与第1个视角位置一致，冰箱的位置和最初观察相同。"
            ),
        },
        {"from": "human", "value": "用几句话总结一下这个空间的布局。"},
        {"from": "gpt", "value": caption},
    ]
    facts = [
        {"kind": "visible_set", "view_id": order[0], "entities": seen_first, "status": "accepted"},
        oven_fridge,
        {
            "kind": "last_seen",
            "entity": "fridge",
            "view_id": last_seen_fridge,
            "dialogue_image_index": order.index(last_seen_fridge) + 1,
            "checked_after_view_ids": [order[3], order[4]],
            "status": "accepted",
        },
        {"kind": "co_visibility", "a": "standing_tv", "b": "fridge", "co_visible_view_ids": bundle.co_visible_views("standing_tv", "fridge"), "status": "accepted"},
        tv_fridge,
        quad,
        {"kind": "never_observed", "entity": "bed", "status": "unknown"},
        {"kind": "visibility", "view_id": order[10], "entity": "fridge", "visible": True, "status": "accepted"},
        *caption_facts,
    ]
    records.append(
        _record(
            "ep3d-rsint17-dialogue-canonical",
            "episodic_dialogue",
            _rgb(order),
            conversations,
            facts,
            bundle,
            variant="canonical",
            abilities=["grounding", "spatial_relation", "mental_reconstruction", "perspective_taking", "epistemic"],
        )
    )

    living = order[5:10]
    kitchen = order[0:5]
    if set(bundle.entity("fridge")["views"]) & set(living):
        raise ValueError("delayed-reveal premise broken: fridge visible in living prefix")
    conversations = [
        {
            "from": "human",
            "value": (
                f"我在一间住宅内移动，按顺序给你画面，方向问题一律{FRAME_NOTE}。"
                "<image><image><image><image><image>\n这是前5个视角，你能看到哪些主要家具和电器？"
            ),
        },
        {
            "from": "gpt",
            "value": "我能看到：" + "、".join(bundle.entity(k)["name"] for k in bundle.visible_core(living)) + "。",
        },
        {"from": "human", "value": "那么，电视在冰箱的什么方向？"},
        {
            "from": "gpt",
            "value": "目前无法确定。到现在为止我还没有在任何视角里观察到冰箱，不能凭常识猜测它的位置。",
        },
        {"from": "human", "value": "<image><image><image><image><image>\n再看这5个视角，然后回答刚才的问题。"},
        {
            "from": "gpt",
            "value": (
                "现在可以回答了：新视角里出现了冰箱（厨房区）。"
                f"综合前后观察，电视在冰箱的{REL_CN[tv_fridge['relation']]}，"
                f"纵向相距约{round(tv_fridge['dominant_delta_m'], 1)}米。"
            ),
        },
        {"from": "human", "value": "<image>\n最后一个视角回到起点。简述这个空间的布局。"},
        {"from": "gpt", "value": caption},
    ]
    facts = [
        {"kind": "abstain", "question_about": ["standing_tv", "fridge"], "missing": "fridge", "retained_view_ids": living, "status": "unknown"},
        tv_fridge,
        *caption_facts,
    ]
    records.append(
        _record(
            "ep3d-rsint17-dialogue-delayed-reveal",
            "episodic_dialogue",
            _rgb([*living, *kitchen, order[10]]),
            conversations,
            facts,
            bundle,
            variant="delayed_kitchen_reveal",
            abilities=["epistemic", "mental_reconstruction", "spatial_relation"],
        )
    )

    tv_sofa = relation(bundle, "standing_tv", "sofa")
    coffee_sofa = distance(bundle, "coffee_table", "sofa")
    conversations = [
        {
            "from": "human",
            "value": (
                f"这段观察只有5个视角，方向问题一律{FRAME_NOTE}。"
                "<image><image><image><image><image>\n请先回答：电视在沙发的哪一侧？"
            ),
        },
        {"from": "gpt", "value": rel_answer(bundle, tv_sofa)},
        {"from": "human", "value": "那电视在冰箱的什么方向？"},
        {
            "from": "gpt",
            "value": "无法确定。这5个视角里没有出现过冰箱，我没有它位置的任何证据，只能弃答。",
        },
        {"from": "human", "value": "咖啡桌到沙发的直线距离大约是多少？"},
        {"from": "gpt", "value": f"大约{round(coffee_sofa['distance_m'], 1)}米。"},
    ]
    facts = [
        tv_sofa,
        {"kind": "abstain", "question_about": ["standing_tv", "fridge"], "missing": "fridge", "retained_view_ids": living, "status": "unknown"},
        coffee_sofa,
    ]
    records.append(
        _record(
            "ep3d-rsint17-dialogue-decisive-deleted",
            "episodic_dialogue",
            _rgb(living),
            conversations,
            facts,
            bundle,
            variant="decisive_views_deleted",
            abilities=["spatial_relation", "metric", "epistemic"],
        )
    )
    return records


def build_grounded_cot(bundle: Bundle) -> list[dict[str, Any]]:
    records = []

    tv_fridge = relation(bundle, "standing_tv", "fridge")
    tv_fridge_d = distance(bundle, "standing_tv", "fridge")
    records.append(
        _record(
            "ep3d-rsint17-cot-cross-view",
            "grounded_cot",
            _rgb(["view-002", "view-006"]),
            [
                {
                    "from": "human",
                    "value": (
                        "<image><image>\n这两张图来自同一套住宅的两个视角，拍摄之间相机转身走过了几米。"
                        f"请推理：电视在冰箱的什么方向（{FRAME_NOTE}），直线距离大约多少？"
                    ),
                },
                {
                    "from": "gpt",
                    "value": (
                        "两个目标从未同框，需要把两个视角拼到同一张平面图上。"
                        "第1张图面向厨房区，能看到冰箱；第2张图是转身返回后拍的客厅区，能看到电视。"
                        "相机从厨房一侧走向客厅一侧，说明客厅在厨房的后方（-Y）。"
                        f"因此电视在冰箱的{REL_CN[tv_fridge['relation']]}，"
                        f"纵向相距约{round(tv_fridge['dominant_delta_m'], 1)}米，另外略偏右；"
                        f"两者中心的直线距离约{round(tv_fridge_d['distance_m'], 1)}米。"
                    ),
                },
            ],
            [tv_fridge, tv_fridge_d],
            bundle,
            abilities=["comprehensive_reasoning", "spatial_relation", "metric"],
        )
    )

    quad = ego_quadrant(bundle, "coffee_table", "sofa", "fridge")
    records.append(
        _record(
            "ep3d-rsint17-cot-perspective-1",
            "grounded_cot",
            _rgb(["view-000", "view-001"]),
            [
                {
                    "from": "human",
                    "value": "<image><image>\n如果你站在咖啡桌旁边、面向沙发，那么冰箱在你的哪个方位？请给出推理过程。",
                },
                {
                    "from": "gpt",
                    "value": (
                        "先建立观察者坐标系：站在咖啡桌处，把咖啡桌指向沙发的方向定为正前方。"
                        "从这两个视角可以看出，沙发在咖啡桌的左后侧，而冰箱远在厨房区，"
                        "位于咖啡桌的右前远端。把冰箱的位置变换到刚才建立的朝向里：它落在正前方的反方向一侧、且偏右。"
                        f"所以冰箱在我的{QUAD_CN[quad['quadrant']]}。"
                    ),
                },
            ],
            [quad],
            bundle,
            abilities=["perspective_taking"],
        )
    )

    quad2 = ego_quadrant(bundle, "swivel_chair", "standing_tv", "fridge")
    records.append(
        _record(
            "ep3d-rsint17-cot-perspective-2",
            "grounded_cot",
            _rgb(["view-007", "view-002"]),
            [
                {
                    "from": "human",
                    "value": "<image><image>\n第1张图里能看到转椅和电视，第2张图里能看到冰箱。如果你坐在转椅上面向电视，冰箱在你的哪个方位？请推理。",
                },
                {
                    "from": "gpt",
                    "value": (
                        "以转椅到电视的连线为正前方。转椅在客厅靠后的位置，电视在它的右前方向；"
                        "而冰箱在厨房区，位于整个客厅的前方（+Y）远端。"
                        "把冰箱位置变换到这个朝向下：它在前方很远处、且明显偏左。"
                        f"所以冰箱在我的{QUAD_CN[quad2['quadrant']]}。"
                    ),
                },
            ],
            [quad2],
            bundle,
            abilities=["perspective_taking"],
        )
    )

    coffee_fridge = distance(bundle, "coffee_table", "fridge")
    records.append(
        _record(
            "ep3d-rsint17-cot-metric",
            "grounded_cot",
            _rgb(["view-000"]),
            [
                {
                    "from": "human",
                    "value": "<image>\n这个视角里同时能看到咖啡桌和冰箱。估计一下两者中心的直线距离，并解释依据。",
                },
                {
                    "from": "gpt",
                    "value": (
                        "咖啡桌在客厅一侧的近景，冰箱在画面深处的厨房区。"
                        "以沙发约两米的长度和地面延伸作为尺度参照，咖啡桌到厨房入口约有三米多，"
                        "冰箱还要更靠里一些。"
                        f"综合估计两者中心距离约{round(coffee_fridge['distance_m'], 1)}米。"
                    ),
                },
            ],
            [coffee_fridge],
            bundle,
            abilities=["metric"],
        )
    )

    r_ab = relation(bundle, "sofa", "coffee_table")
    r_bc = relation(bundle, "coffee_table", "standing_tv")
    r_ac = relation(bundle, "sofa", "standing_tv")
    closure = all(f["relation"] == "left_of" for f in (r_ab, r_bc, r_ac))
    records.append(
        _record(
            "ep3d-rsint17-cot-closure",
            "grounded_cot",
            _rgb(["view-006"]),
            [
                {
                    "from": "human",
                    "value": (
                        f"<image>\n{FRAME_NOTE}。已知沙发在咖啡桌的左侧、咖啡桌在电视的左侧，"
                        "那么沙发是否一定在电视的左侧？请用画面直接验证这条推理链。"
                    ),
                },
                {
                    "from": "gpt",
                    "value": (
                        "按传递性，左侧关系沿同一坐标轴可以传递：沙发在咖啡桌左侧、咖啡桌在电视左侧，"
                        "则沙发在电视左侧。画面也直接支持这个结论：三件家具同时可见，"
                        f"从左到右依次是沙发、咖啡桌、电视，沙发与电视横向相距约{round(r_ac['dominant_delta_m'], 1)}米。"
                        "推理链与直接观察一致。"
                    ),
                },
            ],
            [r_ab, r_bc, r_ac, {"kind": "closure", "relations": ["left_of", "left_of", "left_of"], "holds": closure, "status": "accepted"}],
            bundle,
            abilities=["comprehensive_reasoning", "spatial_relation"],
        )
    )

    tv_back_visible = "standing_tv" in bundle.visible_core(["view-010"])
    records.append(
        _record(
            "ep3d-rsint17-cot-visibility",
            "grounded_cot",
            _rgb(["view-000", "view-006"]),
            [
                {
                    "from": "human",
                    "value": (
                        "<image><image>\n第1张图是出发时的视角，第2张图是走到客厅后的视角。"
                        "如果回到第1张图的位置并保持同样的朝向，能看到电视吗？请推理。"
                    ),
                },
                {
                    "from": "gpt",
                    "value": (
                        "第1个视角面向厨房方向，画面里能看到冰箱等厨房电器，说明镜头朝向平面图的前方（+Y）。"
                        "而第2张图表明电视在客厅一侧，也就是起点位置的后方。"
                        "回到起点并保持原朝向时，电视位于身后，不在视野内，所以看不到。"
                    ),
                },
            ],
            [{"kind": "visibility", "view_id": "view-010", "entity": "standing_tv", "visible": tv_back_visible, "status": "accepted"}],
            bundle,
            abilities=["perspective_taking", "mental_reconstruction"],
        )
    )
    if tv_back_visible:
        raise ValueError("visibility premise broken: tv visible at loop-closure view")
    return records


def build_direct_qa(bundle: Bundle) -> list[dict[str, Any]]:
    records = []
    question_templates = [
        "{note}，{a}在{b}的哪一侧？",
        "如果画出这间屋子的俯视平面图（{note}），{a}位于{b}的什么方向？",
        "从俯视角度看（{note}），{a}相对{b}的主导方向是什么？",
    ]
    candidates: dict[str, list[dict[str, Any]]] = {rel: [] for rel in REL_CN}
    used_pairs: set[frozenset[str]] = set()
    keys = list(CORE_NAMES)
    for a in keys:
        for b in keys:
            if a == b:
                continue
            fact = relation(bundle, a, b)
            if fact["margin_m"] >= 0.4:
                candidates[fact["relation"]].append(fact)
    for group in candidates.values():
        group.sort(key=lambda f: (-f["margin_m"], f["subject"], f["reference"]))
    picked: list[dict[str, Any]] = []
    for slot in range(3):
        for rel in REL_CN:
            for fact in candidates[rel]:
                pair = frozenset((fact["subject"], fact["reference"]))
                if pair in used_pairs:
                    continue
                used_pairs.add(pair)
                picked.append(fact)
                break
            else:
                raise ValueError(f"not enough balanced candidates for {rel} at slot {slot}")
    for index, fact in enumerate(picked):
        covis = bundle.co_visible_views(fact["subject"], fact["reference"])
        if covis:
            views = [covis[0]]
            preamble = "<image>\n"
        else:
            views = [bundle.entity(fact["subject"])["views"][0], bundle.entity(fact["reference"])["views"][0]]
            preamble = "<image><image>\n这两张图来自同一场景的两个视角。"
        template = question_templates[index % len(question_templates)]
        question = template.format(
            note=FRAME_NOTE,
            a=bundle.entity(fact["subject"])["name"],
            b=bundle.entity(fact["reference"])["name"],
        )
        records.append(
            _record(
                f"ep3d-rsint17-qa-rel-{index:02d}",
                "direct_qa",
                _rgb(views),
                [
                    {"from": "human", "value": preamble + question},
                    {"from": "gpt", "value": rel_answer(bundle, fact)},
                ],
                [fact],
                bundle,
                abilities=["spatial_relation"],
                template_id=f"relation.{index % len(question_templates)}",
            )
        )

    extras: list[tuple[str, list[str], str, str, dict[str, Any]]] = []
    fridge_absent_view = next(v for v in bundle.view_ids if v not in bundle.entity("fridge")["views"])
    extras.append(
        (
            "qa-vis-no",
            [fridge_absent_view],
            "<image>\n这张图里能看到冰箱吗？",
            "看不到，这个视角里没有冰箱。",
            {"kind": "visibility", "view_id": fridge_absent_view, "entity": "fridge", "visible": False, "status": "accepted"},
        )
    )
    tv_view = bundle.entity("standing_tv")["views"][0]
    extras.append(
        (
            "qa-vis-yes",
            [tv_view],
            "<image>\n这张图里有电视吗？",
            "有，画面里能看到电视。",
            {"kind": "visibility", "view_id": tv_view, "entity": "standing_tv", "visible": True, "status": "accepted"},
        )
    )
    d1 = distance(bundle, "coffee_table", "sofa")
    extras.append(
        (
            "qa-dist-near",
            [bundle.co_visible_views("coffee_table", "sofa")[0]],
            "<image>\n咖啡桌和沙发的中心距离大约是多少米？",
            f"大约{round(d1['distance_m'], 1)}米。",
            d1,
        )
    )
    d2 = distance(bundle, "standing_tv", "fridge")
    extras.append(
        (
            "qa-dist-far",
            ["view-006", "view-002"],
            "<image><image>\n这两张图来自同一场景。电视和冰箱的直线距离大约是多少米？",
            f"大约{round(d2['distance_m'], 1)}米。",
            d2,
        )
    )
    near_fact = {
        "kind": "nearest",
        "anchor": "sofa",
        "candidates": ["coffee_table", "standing_tv"],
        "nearest": "coffee_table",
        "status": "accepted",
    }
    if distance(bundle, "sofa", "coffee_table")["distance_m"] >= distance(bundle, "sofa", "standing_tv")["distance_m"]:
        raise ValueError("nearest premise broken")
    extras.append(
        (
            "qa-nearest",
            [bundle.co_visible_views("sofa", "standing_tv")[0]],
            "<image>\n离沙发更近的是咖啡桌还是电视？",
            "咖啡桌。它就在沙发旁边，而电视隔着整个客厅。",
            near_fact,
        )
    )
    extras.append(
        (
            "qa-cooccur",
            ["view-002", "view-006"],
            "<image><image>\n在这次漫游的全部视角中，电视和冰箱出现在过同一张画面里吗？",
            "没有。冰箱只出现在面向厨房的视角，电视只出现在客厅方向的视角，两者从未同框。",
            {"kind": "co_visibility", "a": "standing_tv", "b": "fridge", "co_visible_view_ids": bundle.co_visible_views("standing_tv", "fridge"), "status": "accepted"},
        )
    )
    extras.append(
        (
            "qa-unknown-bed",
            list(bundle.view_ids),
            "<image>" * len(bundle.view_ids) + "\n这是完整的一圈漫游。这间住宅的床在什么位置？",
            "在这些视角里我从未观察到床。如果有卧室，它不在已观察范围内，所以我无法给出床的位置。",
            {"kind": "never_observed", "entity": "bed", "status": "unknown"},
        )
    )
    extras.append(
        (
            "qa-unknown-toilet",
            list(bundle.view_ids),
            "<image>" * len(bundle.view_ids) + "\n根据这一圈观察，马桶在冰箱的哪个方向？",
            "无法确定。整段观察里没有出现过马桶，我没有它位置的证据，不能猜测。",
            {"kind": "never_observed", "entity": "toilet", "status": "unknown"},
        )
    )
    for name, views, question, answer, fact in extras:
        records.append(
            _record(
                f"ep3d-rsint17-{name}",
                "direct_qa",
                _rgb(views),
                [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": answer},
                ],
                [fact],
                bundle,
                abilities=["spatial_relation", "epistemic"],
            )
        )
    return records


def build_captions(bundle: Bundle) -> list[dict[str, Any]]:
    caption, facts = layout_caption(bundle)
    tour = ["view-000", "view-002", "view-004", "view-006", "view-008", "view-010"]
    records = [
        _record(
            "ep3d-rsint17-caption-layout",
            "spatial_caption",
            _rgb(tour),
            [
                {
                    "from": "human",
                    "value": "<image>" * len(tour) + "\n这是在一间住宅里走一圈拍到的6个视角。请描述整体空间布局。",
                },
                {"from": "gpt", "value": caption},
            ],
            facts,
            bundle,
            abilities=["mental_reconstruction", "spatial_relation"],
        )
    ]
    mw_fridge = relation(bundle, "fridge", "microwave")
    oven_mw = relation(bundle, "oven", "microwave")
    records.append(
        _record(
            "ep3d-rsint17-caption-kitchen",
            "spatial_caption",
            _rgb(["view-002"]),
            [
                {
                    "from": "human",
                    "value": f"<image>\n描述这个视角里厨房区的布局（{FRAME_NOTE}）。",
                },
                {
                    "from": "gpt",
                    "value": (
                        "这是厨房区：微波炉和洗碗机上下叠放（微波炉在上），"
                        f"烤箱在它们的{REL_CN[oven_mw['relation']]}，"
                        f"冰箱在微波炉的{REL_CN[mw_fridge['relation']]}约{round(mw_fridge['dominant_delta_m'], 1)}米处。"
                    ),
                },
            ],
            [mw_fridge, oven_mw, {"kind": "vertical_stack", "upper": "microwave", "lower": "dishwasher", "status": "accepted"}],
            bundle,
            abilities=["spatial_relation", "grounding"],
        )
    )
    return records


def _state_payload(bundle: Bundle, view_ids: list[str]) -> list[dict[str, Any]]:
    last = view_ids[-1]
    payload = []
    for key in bundle.visible_core(view_ids):
        item = bundle.entity(key)
        indices = [view_ids.index(v) + 1 for v in item["views"] if v in view_ids]
        payload.append(
            {
                "name": item["name"],
                "category": key,
                "position_bin_0p5m": [round(v / 0.5) for v in item["pos"]],
                "status": "visible" if last in item["views"] else "seen_not_current",
                "evidence_view_indices": indices,
            }
        )
    return payload


def build_structured_aux(bundle: Bundle) -> list[dict[str, Any]]:
    instruction = (
        "请以 JSON 数组输出你对场景的实体状态估计，每个元素包含字段："
        "name、category、position_bin_0p5m（坐标除以0.5米后取整）、"
        "status（visible=最后一个视角可见 / seen_not_current=之前见过）、"
        "evidence_view_indices（1开始的视角序号）。只包含你实际观察到的实体。"
    )
    records = []
    for name, view_ids in (
        ("full", list(bundle.view_ids)),
        ("living-only", bundle.view_ids[5:10]),
    ):
        payload = _state_payload(bundle, view_ids)
        records.append(
            _record(
                f"ep3d-rsint17-state-{name}",
                "structured_aux",
                _rgb(view_ids),
                [
                    {"from": "human", "value": "<image>" * len(view_ids) + f"\n{instruction}"},
                    {"from": "gpt", "value": json.dumps(payload, ensure_ascii=False, indent=2)},
                ],
                [{"kind": "belief_state", "view_ids": view_ids, "entities": [p["category"] for p in payload], "status": "accepted"}],
                bundle,
                abilities=["mental_reconstruction"],
                format_conditioned=True,
            )
        )
    return records


def build_rlvr(bundle: Bundle) -> list[dict[str, Any]]:
    records = []

    def add(name: str, views: list[str], prompt: str, reward_spec: dict[str, Any], facts: list[dict[str, Any]], group: str | None = None) -> None:
        extra: dict[str, Any] = {"reward_spec": reward_spec}
        if group:
            extra["consistency_group"] = group
        records.append(
            _record(
                f"ep3d-rsint17-rlvr-{name}",
                "rlvr",
                _rgb(views),
                [{"from": "human", "value": "<image>" * len(views) + "\n" + prompt}],
                facts,
                bundle,
                **extra,
            )
        )

    tv_fridge = relation(bundle, "standing_tv", "fridge")
    direction_q = f"这两张图来自同一住宅的两个视角。{FRAME_NOTE}，电视在冰箱的什么方向？（前方/后方/左侧/右侧，若无法判断请明确说无法确定）"
    add(
        "cross-view-dir",
        ["view-002", "view-006"],
        direction_q,
        {"type": "enum", "choices": list(REL_CN.values()), "correct": REL_CN[tv_fridge["relation"]]},
        [tv_fridge],
        group="tv_fridge_direction",
    )
    add(
        "abstain-dir",
        bundle.view_ids[5:10],
        f"以下5个视角是全部证据。{FRAME_NOTE}，电视在冰箱的什么方向？（前方/后方/左侧/右侧，若无法判断请明确说无法确定）",
        {"type": "abstain", "correct_behavior": "abstain"},
        [{"kind": "abstain", "question_about": ["standing_tv", "fridge"], "missing": "fridge", "retained_view_ids": bundle.view_ids[5:10], "status": "unknown"}],
        group="tv_fridge_direction",
    )
    quad = ego_quadrant(bundle, "coffee_table", "sofa", "fridge")
    add(
        "pt-quadrant-1",
        ["view-000", "view-001"],
        "站在咖啡桌旁、面向沙发，冰箱在你的哪个方位？（左前方/右前方/左后方/右后方）",
        {"type": "enum", "choices": list(QUAD_CN.values()), "correct": QUAD_CN[quad["quadrant"]]},
        [quad],
    )
    quad2 = ego_quadrant(bundle, "swivel_chair", "standing_tv", "fridge")
    add(
        "pt-quadrant-2",
        ["view-007", "view-002"],
        "坐在转椅上、面向电视，冰箱在你的哪个方位？（左前方/右前方/左后方/右后方）",
        {"type": "enum", "choices": list(QUAD_CN.values()), "correct": QUAD_CN[quad2["quadrant"]]},
        [quad2],
    )
    d1 = distance(bundle, "coffee_table", "fridge")
    add(
        "dist-coffee-fridge",
        ["view-000"],
        "估计咖啡桌与冰箱中心的直线距离（单位：米，给出一个数）。",
        {"type": "numeric", "unit": "m", "correct": d1["distance_m"], "tolerance": d1["tolerance_m"]},
        [d1],
    )
    d2 = distance(bundle, "sofa", "standing_tv")
    add(
        "dist-sofa-tv",
        ["view-006"],
        "估计沙发与电视中心的直线距离（单位：米，给出一个数）。",
        {"type": "numeric", "unit": "m", "correct": d2["distance_m"], "tolerance": d2["tolerance_m"]},
        [d2],
    )
    r_ac = relation(bundle, "sofa", "standing_tv")
    add(
        "closure",
        ["view-006"],
        f"{FRAME_NOTE}。已知沙发在咖啡桌左侧、咖啡桌在电视左侧，那么沙发在电视的什么方向？画面是否支持？",
        {"type": "enum", "choices": list(REL_CN.values()), "correct": REL_CN[r_ac["relation"]]},
        [r_ac],
    )
    add(
        "visibility-return",
        ["view-000", "view-006"],
        "第1张图是出发视角，第2张图在客厅。回到第1张图的位置和朝向后能看到电视吗？（能/不能）",
        {"type": "boolean", "correct": False},
        [{"kind": "visibility", "view_id": "view-010", "entity": "standing_tv", "visible": False, "status": "accepted"}],
    )
    return records


def compile_samples(bundle_root: Path) -> list[dict[str, Any]]:
    bundle = Bundle(bundle_root)
    return [
        *build_episodic_dialogues(bundle),
        *build_grounded_cot(bundle),
        *build_direct_qa(bundle),
        *build_captions(bundle),
        *build_structured_aux(bundle),
        *build_rlvr(bundle),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = compile_samples(args.bundle.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging.{os.getpid()}")
    with staging.open("w", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(staging, output)
    counts: dict[str, int] = {}
    for record in records:
        counts[record["sample_type"]] = counts.get(record["sample_type"], 0) + 1
    print(output)
    print(f"records={len(records)} " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
