from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data" / "trajectory_dialogues_v2_pilot"
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}")

pytestmark = pytest.mark.skipif(not PILOT.exists(), reason="trajectory dialogue pilot not built")


def _artifacts() -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((PILOT / "scenes").glob("*/*.dialogue.json"))
    ]


def test_one_pilot_per_controlled_class() -> None:
    report = json.loads((PILOT / "corpus_report.json").read_text(encoding="utf-8"))
    assert report["requested"] == {"T10": 1, "T3": 1, "T4": 1, "T7": 1, "T8": 1}
    assert report["compiled_counts"] == report["requested"]
    assert not report["rejected"]


def test_evidence_is_prefix_bounded_and_programs_end_in_verify() -> None:
    for artifact in _artifacts():
        released: list[str] = []
        for round_ in artifact["rounds"]:
            released.extend(round_["new_view_ids"])
            assert set(round_["evidence_view_ids"]) <= set(released)
            nodes = round_["program"]["nodes"]
            assert nodes[-1]["operation"] == "V"
            seen: set[str] = set()
            for node in nodes:
                assert set(node["depends_on"]) <= seen
                seen.add(node["node_id"])
        assert set(released) == set(artifact["view_ids"])


def test_language_has_no_hidden_channel_leakage() -> None:
    forbidden = ("source_entity_id", "entity_id", "scene_ir", "OBB", "+X", "+Y")
    for artifact in _artifacts():
        for round_ in artifact["rounds"]:
            surface = round_["question_zh"] + round_["answer_zh"]
            assert not UUID_RE.search(surface)
            assert not any(token in surface for token in forbidden)


def test_class_specific_semantic_boundaries() -> None:
    by_class = {artifact["trajectory_class"]: artifact for artifact in _artifacts()}
    t3_surface = json.dumps(by_class["T3"]["rounds"], ensure_ascii=False)
    assert "深度" not in t3_surface and "视差" not in t3_surface
    t4_surface = "".join(
        round_["question_zh"] + round_["answer_zh"] for round_ in by_class["T4"]["rounds"]
    )
    assert "不定义物体的正面或背面" in t4_surface
    assert by_class["T10"]["task_scope"] == "target_view_read_not_prediction"


def test_t8_updates_unknown_only_after_new_evidence() -> None:
    t8 = next(x for x in _artifacts() if x["trajectory_class"] == "T8")
    assert t8["rounds"][0]["answer_key"]["target_status"] == "unknown"
    assert "没看到不等于不存在" in t8["rounds"][0]["answer_zh"]
    assert t8["rounds"][-1]["answer_key"]["target_exists"] is True
    assert t8["rounds"][-1]["answer_key"]["initial_unknown_was_correct"] is True


def test_sft_is_assistant_only_and_all_images_exist() -> None:
    rows = [
        json.loads(line)
        for line in (PILOT / "all.dialogue_episode_sft.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert len(rows) == 5
    for row in rows:
        assert row["loss_policy"] == {"train_on": "assistant_only"}
        for message in row["messages"]:
            if isinstance(message["content"], list):
                for part in message["content"]:
                    if part["type"] == "image":
                        assert Path(part["image"]).exists()
