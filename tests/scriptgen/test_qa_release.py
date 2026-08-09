from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "scriptgen_qa_v1"

pytestmark = pytest.mark.skipif(not OUT.exists(), reason="scriptgen QA release not built")


def _rows(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (OUT / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_release_counts_and_scope() -> None:
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["family_count"] == 287
    assert manifest["raw_record_count"] == 1394
    assert manifest["streaming_record_count"] == 238
    assert manifest["streaming_turn_count"] == 534
    assert manifest["streaming_skip_count"] == 0
    assert manifest["excluded_capabilities"] == ["existence_sufficiency_bed"]


def test_raw_records_are_rgb_only_and_oracle_separated() -> None:
    raw = _rows("raw_qa.jsonl")
    inputs = {row["record_id"]: row for row in _rows("raw_eval_inputs.jsonl")}
    oracles = {row["record_id"]: row for row in _rows("raw_eval_oracle.jsonl")}
    assert len(raw) == len(inputs) == len(oracles) == 1394
    assert Counter(row["variant"] for row in raw) == Counter(
        {"canonical": 287, "permute": 287, "drop_key": 246, "drop_filler": 287, "delay": 287}
    )
    for row in raw:
        assert row["capability"] != "existence_sufficiency_bed"
        assert all(image["path"].endswith(".rgb.png") for image in row["images"])
        assert row["input_sha256"] == inputs[row["record_id"]]["input_sha256"]
        assert row["input_sha256"] == oracles[row["record_id"]]["input_sha256"]
        assert "answer" not in inputs[row["record_id"]]
        assert "certificate" not in inputs[row["record_id"]]
        assert row["answer"] == oracles[row["record_id"]]["answer"]


def test_streaming_prefixes_are_monotonic_and_compiler_valid() -> None:
    rows = _rows("streaming_qa.jsonl")
    assert Counter(row["stream_kind"] for row in rows) == Counter(
        {"evidence_reveal": 220, "trajectory_multi_question": 18}
    )
    for row in rows:
        prefixes = [turn["prefix_length"] for turn in row["turns"]]
        assert prefixes == sorted(prefixes)
        assert all(turn["status"] in {"answerable", "abstain"} for turn in row["turns"])
        assert sum(len(turn["new_images"]) for turn in row["turns"]) == len(row["images"])
        if row["tier"] == "P1":
            assert min(prefixes) >= 2
        else:
            assert len(row["turns"]) == 2
            assert row["turns"][0]["status"] == "abstain"
            assert row["turns"][0]["answer"] == "无法判断"
            assert row["turns"][1]["status"] == "answerable"


def test_batches_are_at_most_one_hundred_and_stratified_across_tiers() -> None:
    for filename in ("raw_qa.jsonl", "streaming_qa.jsonl"):
        by_batch: dict[str, list[dict]] = {}
        for row in _rows(filename):
            by_batch.setdefault(row["batch_id"], []).append(row)
        assert all(1 <= len(rows) <= 100 for rows in by_batch.values())
        if filename == "raw_qa.jsonl":
            assert all(len({row["tier"] for row in rows}) >= 2 for rows in by_batch.values())


def test_full_model_evaluation_is_complete_when_summary_exists() -> None:
    summary_path = OUT / "sensenova_summary.json"
    if not summary_path.is_file():
        pytest.skip("SenseNova full evaluation has not completed")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["prediction_record_count"] != 3740:
        pytest.skip("SenseNova summary is currently a limited smoke run")
    assert summary["error_count"] == 0
    predictions = _rows("sensenova_predictions.jsonl")
    assert len(predictions) == 3740
    assert {row["prompt_version"] for row in predictions} == {"scriptgen-qa-eval.v2"}
    input_hashes = {
        row["record_id"]: row["input_sha256"]
        for filename in ("raw_eval_inputs.jsonl", "streaming_eval_inputs.jsonl")
        for row in _rows(filename)
    }
    assert all(row["input_sha256"] == input_hashes[row["record_id"]] for row in predictions)
    assert set(summary["raw"]["conditions"]) == {"multimodal", "vision_free"}
    assert set(summary["streaming"]["conditions"]) == {
        "multimodal_teacher_forced",
        "multimodal_free_running",
        "vision_free_teacher_forced",
        "vision_free_free_running",
    }
