from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "scene_dialogues_v1"
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}")
FORBIDDEN = ("+X", "+Y", "entity_id", "scene_ir", "relation_oracle", "坐标系")

pytestmark = pytest.mark.skipif(not OUT.exists(), reason="scene dialogues not compiled")


def _episodes() -> list[dict]:
    path = OUT / "train.dialogue_episode_sft.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_corpus_report_is_honest() -> None:
    report = json.loads((OUT / "corpus_report.json").read_text(encoding="utf-8"))
    assert len(report["compiled"]) >= 20
    assert report["rejected"], "fail-closed rejection list should not be empty for this sweep"
    for entry in report["compiled"]:
        turn_ids = set(entry["rounds"])
        assert turn_ids & {"r04-crossview-relation", "r05-object-perspective"}, entry


def test_direction_balance_is_bounded() -> None:
    report = json.loads((OUT / "corpus_report.json").read_text(encoding="utf-8"))
    balance = report["direction_balance"]
    assert set(balance) <= {"left_of", "right_of", "in_front_of", "behind"}
    assert max(balance.values()) <= 2 * min(balance.values()) + 3


def test_no_leakage_and_images_exist() -> None:
    for record in _episodes():
        for message in record["messages"]:
            content = message["content"]
            parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
            for part in parts:
                if part["type"] == "image":
                    assert Path(part["image"]).exists(), part["image"]
                else:
                    assert not UUID_RE.search(part["text"])
                    for token in FORBIDDEN:
                        assert token not in part["text"], (record["record_id"], token)
        assert record["loss_policy"]["train_on"] == "assistant_only"


def test_isolated_arm_pairs_with_episode_arm() -> None:
    isolated = [
        json.loads(line)
        for line in (OUT / "train.dialogue_isolated_sft.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    episode_ids = {record["record_id"] for record in _episodes()}
    assert isolated
    for record in isolated:
        assert record["hidden_meta"]["comparison_of"] in episode_ids
