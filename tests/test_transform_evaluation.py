from __future__ import annotations

import json
from pathlib import Path

import pytest

import episode3d.transform_pilot.evaluation as evaluation
from episode3d.transform_pilot.evaluation import (
    _score_response,
    parse_direction_label,
    parse_multiple_choice,
    trace_format,
)


def test_answer_parser_prefers_explicit_answer_tag() -> None:
    text = "<cue>A不是答案。</cue><answer>C. 左侧</answer>"
    assert parse_multiple_choice(text) == "C"


def test_answer_parser_rejects_ambiguous_prose() -> None:
    assert parse_multiple_choice("A和B都可能，但我不确定。") is None


def test_trace_format_requires_all_three_roles() -> None:
    full = "<cue>线索</cue><transform>变换</transform><answer>D. 后方</answer>"
    assert all(trace_format(full).values())
    assert trace_format("<answer>D</answer>")["has_transform"] is False


def test_direction_parser_ignores_cue_and_reads_answer_span() -> None:
    response = "<cue>物体原来在右侧</cue><answer>C. 左侧</answer>"
    assert parse_direction_label(response) == "left"


def test_scoring_exposes_option_text_contradiction() -> None:
    scores = _score_response(
        response="<answer>D. 左侧</answer>",
        item={"question": "A. 前方 B. 右侧 C. 左侧 D. 后方"},
        oracle={"answer": {"key": "C", "label": "left"}},
    )
    assert scores["correct"] is False
    assert scores["semantic_correct"] is True
    assert scores["option_text_consistent"] is False


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_evaluator_range_is_exact_and_reports_original_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = [
        {
            "record_id": f"record-{index}",
            "task_id": "self_rotation_query.v1",
            "scene_id": f"scene-{index}",
            "split": split,
            "images": ["unused.png"],
            "system": "system",
            "question": "方向？ A. 前方 B. 右侧 C. 后方 D. 左侧",
            "evaluation_axes": {},
        }
        for index, split in enumerate(("train", "validation", "test"))
    ]
    oracle = [
        {
            "record_id": row["record_id"],
            "split": row["split"],
            "answer": {"key": "A", "label": "front"},
            "certificate": {},
        }
        for row in inputs
    ]
    _write_jsonl(tmp_path / "eval_inputs_all.jsonl", inputs)
    _write_jsonl(tmp_path / "eval_oracle_all.jsonl", oracle)

    class FakeBackend:
        def __init__(self, _model_path: Path) -> None:
            pass

        def generate(self, **_kwargs: object) -> str:
            return "<answer>A. 前方</answer>"

    monkeypatch.setattr(evaluation, "InternVLBackend", FakeBackend)
    output = tmp_path / "predictions.jsonl"
    summary = evaluation.evaluate_sensenova(
        dataset_root=tmp_path,
        model_path=tmp_path / "model",
        output_path=output,
        prompt_mode="answer_only",
        evaluation_scope="all",
        start_index=1,
        stop_index=2,
    )

    predictions = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["record_id"] for row in predictions] == ["record-1"]
    assert summary["evaluated"] == 1
    assert summary["record_index_range"] == {"start": 1, "stop": 2}
    assert summary["by_split"] == {
        "validation": {
            "count": 1,
            "accuracy": 1.0,
            "semantic_accuracy": 1.0,
            "trace_format_rate": 0.0,
            "joint_correct_trace_rate": 0.0,
        }
    }
