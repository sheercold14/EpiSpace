from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "trajectory_dialogues_v2"

pytestmark = pytest.mark.skipif(
    not OUT.exists(), reason="full trajectory dialogue corpus not built"
)


def _artifacts() -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((OUT / "scenes").glob("*/*.dialogue.json"))
    ]


def test_all_passed_jobs_are_compiled_without_silent_drop() -> None:
    report = json.loads((OUT / "corpus_report.json").read_text(encoding="utf-8"))
    expected = {"T10": 21, "T3": 83, "T4": 10, "T7": 43, "T8": 8}
    assert report["requested"] == expected
    assert report["compiled_counts"] == expected
    assert report["rejected"] == []
    assert len(report["compiled"]) == 165
    assert sum(report["round_counts"].values()) == 588


def test_full_artifact_and_image_accounting() -> None:
    artifacts = _artifacts()
    assert len(artifacts) == 165
    assert Counter(a["trajectory_class"] for a in artifacts) == Counter(
        {"T3": 83, "T4": 10, "T7": 43, "T8": 8, "T10": 21}
    )
    image_paths = [Path(path) for artifact in artifacts for path in artifact["rgb_paths"].values()]
    assert len(image_paths) == 818
    assert all(path.exists() for path in image_paths)
    assert sum(len(artifact["rounds"]) for artifact in artifacts) == 588


def test_every_round_has_prefix_safe_evidence_and_verbalized_claims() -> None:
    for artifact in _artifacts():
        released: set[str] = set()
        for round_ in artifact["rounds"]:
            released.update(round_["new_view_ids"])
            assert set(round_["evidence_view_ids"]) <= released
            assert round_["program"]["nodes"][-1]["operation"] == "V"
            claims = round_["claim_sheet"]["claims"]
            required = {claim["claim_id"] for claim in claims if claim["required"]}
            verbalized = {
                claim_id
                for sentence in round_["answer_sentences"]
                for claim_id in sentence["claim_ids"]
            }
            assert required <= verbalized
        assert released == set(artifact["view_ids"])


def test_primary_and_fallback_templates_remain_auditable() -> None:
    turns = Counter(round_["turn_id"] for artifact in _artifacts() for round_ in artifact["rounds"])
    assert turns["t3-r04-panorama-relation"] == 77
    assert turns["t3-r04-rotation-closure"] == 6
    assert turns["t7-r03-world-vertical-relation"] == 16
    assert turns["t7-r03-high-downward-view"] == 27
    assert turns["t8-r01-occluded-unknown"] == 8
    assert turns["t10-r01-target-frame-read"] == 21


def test_sft_record_counts_and_loss_mask() -> None:
    episode_rows = [
        json.loads(line)
        for line in (OUT / "all.dialogue_episode_sft.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    isolated_rows = [
        json.loads(line)
        for line in (OUT / "all.dialogue_isolated_sft.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert len(episode_rows) == 165
    assert len(isolated_rows) == 588
    assert all(row["loss_policy"] == {"train_on": "assistant_only"} for row in episode_rows)
    assert sum(row["task_scope"] == "target_view_read_not_prediction" for row in episode_rows) == 21
