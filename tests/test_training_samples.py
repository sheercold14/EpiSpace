from __future__ import annotations

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "training_samples.v1.jsonl"
BUNDLE = ROOT.parent / "OminiGibson" / "outputs" / "Rs_int_seed17"

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}")
ORACLE_MARKERS = ("scene_ir", "relation_oracle", "source_entity_id", "expected_relation")


def _records() -> list[dict]:
    return [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line]


def _geometry() -> tuple[dict[str, list[float]], dict[str, list[str]], list[str]]:
    scene = json.loads((BUNDLE / "scene_ir.json").read_text(encoding="utf-8"))
    episode = json.loads((BUNDLE / "spatial_episode.json").read_text(encoding="utf-8"))
    view_ids = [item["view_id"] for item in episode["observations"]]
    visible = {item["view_id"]: set(item["visible_entity_ids"]) for item in episode["observations"]}
    positions: dict[str, list[float]] = {}
    views: dict[str, list[str]] = {}
    by_label: dict[str, list[dict]] = {}
    for entity in scene["entities"]:
        by_label.setdefault(entity["raw_label"], []).append(entity)
    for label, group in by_label.items():
        if len(group) != 1:
            continue
        (raw,) = group
        positions[label] = [float(v) for v in raw["world_from_entity"]["translation_m"]]
        views[label] = [v for v in view_ids if raw["entity_id"] in visible[v]]
    return positions, views, view_ids


def _relation(pa: list[float], pb: list[float]) -> tuple[str, float, float]:
    dx, dy = pa[0] - pb[0], pa[1] - pb[1]
    if abs(dx) >= abs(dy):
        return ("right_of" if dx > 0 else "left_of", abs(dx), abs(dx) - abs(dy))
    return ("in_front_of" if dy > 0 else "behind", abs(dy), abs(dy) - abs(dx))


def test_corpus_shape_and_family_lock() -> None:
    records = _records()
    assert len(records) >= 40
    types = {record["sample_type"] for record in records}
    assert types == {
        "episodic_dialogue",
        "grounded_cot",
        "direct_qa",
        "spatial_caption",
        "structured_aux",
        "rlvr",
    }
    families = {record["meta"]["family_id"] for record in records}
    splits = {record["meta"]["split_lock"] for record in records}
    assert len(families) == 1 and len(splits) == 1
    for record in records:
        expected = "rl_only" if record["sample_type"] == "rlvr" else "gpt_turns_only"
        assert record["meta"]["loss_policy"]["train_on"] == expected


def test_no_oracle_leakage_and_image_placeholders() -> None:
    for record in _records():
        placeholders = 0
        for turn in record["conversations"]:
            text = turn["value"]
            assert not UUID_RE.search(text), record["id"]
            for marker in ORACLE_MARKERS:
                assert marker not in text, (record["id"], marker)
            placeholders += text.count("<image>")
        assert placeholders == len(record["images"]), record["id"]
        if record["sample_type"] == "rlvr":
            assert all(turn["from"] == "human" for turn in record["conversations"])
            assert "reward_spec" in record["meta"]
        else:
            assert record["conversations"][-1]["from"] == "gpt"


def test_images_exist() -> None:
    for record in _records():
        media_root = Path(record["meta"]["media_root"])
        for image in record["images"]:
            assert (media_root / image).exists(), (record["id"], image)


def test_facts_reexecute_from_geometry() -> None:
    positions, views, _ = _geometry()
    checked = 0
    for record in _records():
        for fact in record["meta"]["facts"]:
            kind = fact["kind"]
            if kind == "relation":
                rel, dominant, margin = _relation(positions[fact["subject"]], positions[fact["reference"]])
                assert rel == fact["relation"], record["id"]
                assert math.isclose(dominant, fact["dominant_delta_m"], abs_tol=0.01)
                assert math.isclose(margin, fact["margin_m"], abs_tol=0.01)
                assert margin >= 0.4, record["id"]
                checked += 1
            elif kind == "distance":
                pa, pb = positions[fact["a"]], positions[fact["b"]]
                value = math.sqrt(sum((pa[i] - pb[i]) ** 2 for i in range(3)))
                assert math.isclose(value, fact["distance_m"], abs_tol=0.01)
                checked += 1
            elif kind == "ego_quadrant":
                po, pf, pt = (positions[fact[k]] for k in ("origin", "facing", "target"))
                fx, fy = pf[0] - po[0], pf[1] - po[1]
                norm = math.hypot(fx, fy)
                forward = (fx / norm, fy / norm)
                right = (forward[1], -forward[0])
                tx, ty = pt[0] - po[0], pt[1] - po[1]
                qx = tx * right[0] + ty * right[1]
                qy = tx * forward[0] + ty * forward[1]
                quadrant = f"{'front' if qy >= 0 else 'back'}-{'right' if qx >= 0 else 'left'}"
                assert quadrant == fact["quadrant"], record["id"]
                checked += 1
            elif kind == "visibility":
                observed = fact["view_id"] in views[fact["entity"]]
                assert observed == fact["visible"], record["id"]
                checked += 1
            elif kind == "co_visibility":
                common = sorted(set(views[fact["a"]]) & set(views[fact["b"]]))
                assert common == sorted(fact["co_visible_view_ids"]), record["id"]
                checked += 1
            elif kind == "abstain":
                retained = set(fact["retained_view_ids"])
                assert not retained & set(views[fact["missing"]]), record["id"]
                checked += 1
            elif kind == "never_observed":
                assert views[fact["entity"]] == [], record["id"]
                checked += 1
            elif kind == "last_seen":
                entity_views = views[fact["entity"]]
                assert fact["view_id"] in entity_views
                for later in fact["checked_after_view_ids"]:
                    assert later not in entity_views, record["id"]
                checked += 1
            elif kind == "vertical_stack":
                upper, lower = positions[fact["upper"]], positions[fact["lower"]]
                assert upper[2] > lower[2]
                assert math.hypot(upper[0] - lower[0], upper[1] - lower[1]) < 0.3
                checked += 1
            elif kind == "x_between":
                middle = positions[fact["middle"]][0]
                left, right_x = sorted(positions[end][0] for end in fact["ends"])
                assert left < middle < right_x
                checked += 1
            elif kind == "nearest":
                anchor = positions[fact["anchor"]]
                dists = {
                    key: math.dist(anchor, positions[key]) for key in fact["candidates"]
                }
                assert min(dists, key=dists.get) == fact["nearest"]  # type: ignore[arg-type]
                checked += 1
    assert checked >= 40


def test_answer_balance_and_abstention_pressure() -> None:
    records = _records()
    directions: dict[str, int] = {}
    for record in records:
        if record["sample_type"] != "direct_qa":
            continue
        for fact in record["meta"]["facts"]:
            if fact["kind"] == "relation":
                directions[fact["relation"]] = directions.get(fact["relation"], 0) + 1
    total = sum(directions.values())
    assert total >= 8
    assert max(directions.values()) <= math.ceil(0.4 * total)
    unknown = sum(
        1
        for record in records
        for fact in record["meta"]["facts"]
        if fact.get("status") == "unknown"
    )
    assert unknown >= 4


def test_family_variants_behave() -> None:
    records = {record["id"]: record for record in _records()}
    delayed = records["ep3d-rsint17-dialogue-delayed-reveal"]
    statuses = [fact.get("status") for fact in delayed["meta"]["facts"]]
    assert "unknown" in statuses and "accepted" in statuses
    text = " ".join(turn["value"] for turn in delayed["conversations"] if turn["from"] == "gpt")
    assert "无法确定" in text and "现在可以回答" in text
    decisive = records["ep3d-rsint17-dialogue-decisive-deleted"]
    kinds = {fact["kind"]: fact for fact in decisive["meta"]["facts"]}
    assert kinds["abstain"]["status"] == "unknown"
    assert sum(1 for fact in decisive["meta"]["facts"] if fact.get("status") == "accepted") >= 2
    groups: dict[str, list[str]] = {}
    for record in records.values():
        group = record["meta"].get("consistency_group")
        if group:
            groups.setdefault(group, []).append(record["id"])
    assert any(len(members) >= 2 for members in groups.values())


def test_structured_aux_is_format_conditioned_and_parseable() -> None:
    positions, views, _ = _geometry()
    for record in _records():
        if record["sample_type"] != "structured_aux":
            continue
        assert record["meta"].get("format_conditioned") is True
        assert "JSON" in record["conversations"][0]["value"]
        payload = json.loads(record["conversations"][-1]["value"])
        assert isinstance(payload, list) and payload
        (belief_fact,) = record["meta"]["facts"]
        retained = belief_fact["view_ids"]
        for item in payload:
            entity_views = views[item["category"]]
            assert set(entity_views) & set(retained), record["id"]
            expected_bins = [round(v / 0.5) for v in positions[item["category"]]]
            assert item["position_bin_0p5m"] == expected_bins
