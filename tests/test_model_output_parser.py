from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from episode3d.benchmark_evaluator import evaluate_records
from episode3d.model_output_parser import INVALID_ANSWER_VALUE, parse_model_output

PROGRAM_BY_TASK = {
    "counterfactual_verification": "counterfactual_cross_view.v1",
    "egocentric_relation": "egocentric_relation.v1",
    "evidence_presence_reveal": "evidence_presence_reveal.v1",
    "evidence_presence_unknown": "evidence_presence_unknown.v1",
    "target_view_prediction": "target_view_prediction.v1",
}


def _row(task: str, answer_value: object = "POISON") -> dict:
    return {
        "record_id": f"record-{task}",
        "schema_version": "epispace.benchmark.v1",
        "target": {
            "task_type": task,
            "answer_value": answer_value,
            "answer_status": "POISON",
            "answer_zh": "POISON",
        },
        "program": {"program_id": PROGRAM_BY_TASK[task]},
        "model_input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "仅根据当前观察，能否确定这个场景里存在钢琴？",
                    }
                ],
            }
        ],
    }


def _answer(row: Mapping, text: str) -> object:
    result = parse_model_output(row, text)
    assert result["parse_status"] == "parsed", result
    return result["answer_value"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("餐桌位于书架的右侧。", "right_of"),
        ("最终答案：left_of", "left_of"),
        ("不是后方，而是前方。", "in_front_of"),
        ('```json\n{"relation":"behind"}\n```', "behind"),
    ],
)
def test_egocentric_relation_parses_closed_relation_ontology(
    text: str, expected: str
) -> None:
    assert _answer(_row("egocentric_relation"), text) == expected


def test_relation_parser_fails_closed_on_diagonal_or_missing_answer() -> None:
    row = _row("egocentric_relation")
    for output in ("它在左前方。", "我无法判断。", "左边或右边都有可能。"):
        parsed = parse_model_output(row, output)
        assert parsed["parse_status"] == "invalid"
        assert parsed["answer_value"] == INVALID_ANSWER_VALUE

    malformed_json = parse_model_output(row, '{"relation":"diagonal","note":"右侧"}')
    assert malformed_json["parse_error"] == "invalid_structured_relation"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "不对；不是在后方，而是实际位于前方。",
            {"claim_correct": False, "relation": "in_front_of"},
        ),
        (
            "对，这个说法成立，它确实在左侧。",
            {"claim_correct": True, "relation": "left_of"},
        ),
        (
            "是，它位于右侧。",
            {"claim_correct": True, "relation": "right_of"},
        ),
        (
            '{"claim_correct": false, "relation": "right_of"}',
            {"claim_correct": False, "relation": "right_of"},
        ),
    ],
)
def test_claim_parser_requires_correctness_and_relation_from_output(
    text: str, expected: dict
) -> None:
    assert _answer(_row("counterfactual_verification"), text) == expected


def test_claim_parser_does_not_fill_relation_from_target() -> None:
    row = _row(
        "counterfactual_verification",
        {"claim_correct": True, "relation": "behind"},
    )
    parsed = parse_model_output(row, "对。")
    assert parsed["parse_status"] == "invalid"
    assert parsed["parse_error"] == "missing_or_ambiguous_relation"


@pytest.mark.parametrize(
    "text",
    [
        "还不能确定；当前证据不足。",
        "根据这些图无法判断这个场景里是否存在钢琴。",
        "null",
        '{"status":"unknown"}',
    ],
)
def test_presence_unknown_requires_explicit_epistemic_uncertainty(text: str) -> None:
    assert _answer(_row("evidence_presence_unknown", None), text) is None


def test_presence_unknown_rejects_absence_or_bare_non_observation() -> None:
    row = _row("evidence_presence_unknown", None)
    for output in ("不存在钢琴。", "没有看到钢琴。", "看不到钢琴。", "否。"):
        parsed = parse_model_output(row, output)
        assert parsed["parse_status"] == "invalid"
        assert parsed["parse_error"] == "missing_explicit_epistemic_unknown"


@pytest.mark.parametrize(
    "text",
    [
        "能确定，当前画面中能看到钢琴。",
        "确实存在。",
        "画面里有钢琴。",
        "true",
        '{"status":"present","category":"钢琴"}',
        '{"answer_value":{"status":"present","category":"piano"}}',
    ],
)
def test_presence_reveal_uses_query_category_constant(text: str) -> None:
    row = _row(
        "evidence_presence_reveal",
        {"status": "POISON_DECISION", "category": "piano"},
    )
    assert _answer(row, text) == {"status": "present", "category": "piano"}


def test_presence_reveal_prefers_explicit_query_arguments_and_checks_json_category() -> None:
    row = _row(
        "evidence_presence_reveal",
        {"status": "present", "category": "target_poison"},
    )
    row["query_arguments"] = {"category": "piano"}
    assert _answer(row, '{"status":"present","category":"piano"}') == {
        "status": "present",
        "category": "piano",
    }
    invalid = parse_model_output(row, '{"status":"present","category":"sofa"}')
    assert invalid["parse_status"] == "invalid"
    assert invalid["parse_error"] == "presence_category_mismatch"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("能看到。", True),
        ("看不到。", False),
        ("最终答案：不会看到目标。", False),
        ("```json\nfalse\n```", False),
        ('{"answer_value": true}', True),
    ],
)
def test_target_view_prediction_parses_visibility_from_output(
    text: str, expected: bool
) -> None:
    assert _answer(_row("target_view_prediction", not expected), text) is expected


def test_target_view_prediction_rejects_uncertainty_and_contradiction() -> None:
    row = _row("target_view_prediction", True)
    for output in ("无法确定能不能看到。", "能看到，也可能看不到。", "也许。"):
        parsed = parse_model_output(row, output)
        assert parsed["parse_status"] == "invalid"

    malformed_json = parse_model_output(row, '{"visible":"看不到"}')
    assert malformed_json["parse_error"] == "invalid_structured_visibility"


class _TargetWithoutDecisionAccess(Mapping[str, object]):
    """Raise if a parser attempts to read any hidden target decision."""

    def __getitem__(self, key: str) -> object:
        if key == "task_type":
            return "egocentric_relation"
        raise AssertionError(f"target decision field was read: {key}")

    def __iter__(self) -> Iterator[str]:
        yield "task_type"

    def __len__(self) -> int:
        return 1


def test_non_presence_tasks_never_read_target_decision_fields() -> None:
    row = _row("egocentric_relation")
    row["target"] = _TargetWithoutDecisionAccess()
    assert _answer(row, "它在右侧。") == "right_of"


def test_claim_and_visibility_outputs_are_independent_of_poisoned_target_values() -> None:
    claim = _row(
        "counterfactual_verification",
        {"claim_correct": True, "relation": "left_of"},
    )
    assert _answer(claim, "不对，实际在后方。") == {
        "claim_correct": False,
        "relation": "behind",
    }

    visibility = _row("target_view_prediction", True)
    assert _answer(visibility, "看不到。") is False


@pytest.mark.parametrize(
    ("mutator", "reason"),
    [
        (lambda row: row.pop("record_id"), "invalid_record_id"),
        (lambda row: row.update(schema_version="other"), "unsupported_benchmark_schema"),
        (
            lambda row: row["program"].update(program_id="wrong.v1"),
            "task_program_schema_mismatch",
        ),
    ],
)
def test_malformed_contracts_are_explicitly_invalid(mutator, reason: str) -> None:
    row = _row("egocentric_relation")
    mutator(row)
    parsed = parse_model_output(row, "右侧。")
    assert parsed["parse_status"] == "invalid"
    assert parsed["parse_error"] == reason
    assert parsed["answer_value"] == INVALID_ANSWER_VALUE


def test_empty_and_non_string_outputs_are_explicitly_invalid() -> None:
    row = _row("egocentric_relation")
    assert parse_model_output(row, "  ")["parse_error"] == "empty_model_output"
    assert parse_model_output(row, None)["parse_error"] == "model_output_not_string"  # type: ignore[arg-type]


def test_frozen_core_reference_language_round_trips_all_five_schemas() -> None:
    benchmark_path = Path("data/epispace_pilot_v1/benchmark.core.jsonl")
    rows = [
        json.loads(line)
        for line in benchmark_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 115
    assert {row["target"]["task_type"] for row in rows} == set(PROGRAM_BY_TASK)

    predictions = []
    for row in rows:
        parsed = parse_model_output(row, row["target"]["answer_zh"])
        assert parsed["parse_status"] == "parsed", (row["record_id"], parsed)
        assert parsed["answer_value"] == row["target"]["answer_value"], row["record_id"]
        predictions.append(parsed)

    report = evaluate_records(rows, predictions)
    assert report["record_accuracy"]["accuracy"] == 1.0
    assert report["family_exact_match"]["accuracy"] == 1.0
