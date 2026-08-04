from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "t10_compositions_v2"

pytestmark = pytest.mark.skipif(not OUT.exists(), reason="T10 composition corpus not built")


def _artifacts() -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((OUT / "scenes").glob("*.dialogue.json"))
    ]


def test_strict_source_to_heldout_yield_is_reported() -> None:
    report = json.loads((OUT / "corpus_report.json").read_text(encoding="utf-8"))
    assert report["requested"] == 9
    assert len(report["compiled"]) == 5
    assert len(report["rejected"]) == 4
    assert sum(row["cross_view_required"] for row in report["compiled"]) >= 3


def test_prediction_precedes_target_render_release() -> None:
    for artifact in _artifacts():
        assert artifact["task_scope"] == "source_to_heldout_perspective_prediction"
        assert artifact["composition_certificate"]["prediction_before_target_release"] is True
        prediction = artifact["rounds"][2]
        verification = artifact["rounds"][3]
        assert prediction["turn_id"] == "t10c-r03-heldout-prediction"
        assert prediction["new_view_ids"] == []
        assert all(view.startswith("source-") for view in prediction["evidence_view_ids"])
        assert verification["new_view_ids"] == [
            f"heldout-{artifact['composition_certificate']['target_view_id']}"
        ]


def test_all_composition_images_and_assistant_only_loss() -> None:
    rows = [
        json.loads(line)
        for line in (OUT / "all.dialogue_episode_sft.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert len(rows) == 5
    for row in rows:
        assert row["loss_policy"] == {"train_on": "assistant_only"}
        assert row["task_scope"] == "source_to_heldout_perspective_prediction"
        for message in row["messages"]:
            if isinstance(message["content"], list):
                for part in message["content"]:
                    if part["type"] == "image":
                        assert Path(part["image"]).exists()
