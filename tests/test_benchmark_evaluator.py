from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from episode3d.benchmark_evaluator import (
    EvaluationInputError,
    evaluate_files,
    evaluate_records,
    main,
    oracle_predictions,
    render_markdown,
    write_reports,
)


def _benchmark_row(
    record_id: str,
    *,
    answer_value: object,
    task_type: str,
    program_id: str,
    family_id: str,
    family_variant: str,
    consistency_group: str | None = None,
    split: str = "test",
) -> dict:
    return {
        "record_id": record_id,
        "target": {"answer_value": answer_value, "task_type": task_type},
        "program": {"program_id": program_id},
        "split": split,
        "family_id": family_id,
        "family_variant": family_variant,
        "consistency_group": consistency_group,
        "certificate": {},
    }


def _benchmark_fixture() -> list[dict]:
    rows = [
        _benchmark_row(
            "claim-false",
            answer_value={"claim_correct": False, "relation": "left"},
            task_type="counterfactual_verification",
            program_id="counterfactual_cross_view.v1",
            family_id="family-claim",
            family_variant="claim_false",
            consistency_group="group-claim",
        ),
        _benchmark_row(
            "claim-true",
            answer_value={"claim_correct": True, "relation": "left"},
            task_type="counterfactual_verification",
            program_id="counterfactual_cross_view.v1",
            family_id="family-claim",
            family_variant="claim_true",
            consistency_group="group-claim",
        ),
    ]
    for group_number in (1, 2):
        group_id = f"group-evidence-{group_number}"
        family_id = f"family-evidence-{group_number}"
        rows.extend(
            [
                _benchmark_row(
                    f"e{group_number}-prefix",
                    answer_value=None,
                    task_type="evidence_presence_unknown",
                    program_id="evidence_presence_unknown.v1",
                    family_id=family_id,
                    family_variant="prefix_unknown",
                    consistency_group=group_id,
                    split="validation" if group_number == 2 else "test",
                ),
                _benchmark_row(
                    f"e{group_number}-revealed",
                    answer_value={"status": "present", "category": "piano"},
                    task_type="evidence_presence_reveal",
                    program_id="evidence_presence_reveal.v1",
                    family_id=family_id,
                    family_variant="revealed",
                    consistency_group=group_id,
                    split="validation" if group_number == 2 else "test",
                ),
                _benchmark_row(
                    f"e{group_number}-deleted",
                    answer_value=None,
                    task_type="evidence_presence_unknown",
                    program_id="evidence_presence_unknown.v1",
                    family_id=family_id,
                    family_variant="decisive_deleted",
                    consistency_group=group_id,
                    split="validation" if group_number == 2 else "test",
                ),
            ]
        )
    rows.append(
        _benchmark_row(
            "single",
            answer_value=True,
            task_type="grounding_presence",
            program_id="grounding_presence.v1",
            family_id="family-single",
            family_variant="canonical",
        )
    )
    return rows


def _frame_fixture() -> list[dict]:
    rows = [
        _benchmark_row(
            "frame-a",
            answer_value="right_of",
            task_type="egocentric_relation",
            program_id="egocentric_relation.v1",
            family_id="family-frame",
            family_variant="frame_a",
            consistency_group="group-frame",
        ),
        _benchmark_row(
            "frame-b",
            answer_value="behind",
            task_type="egocentric_relation",
            program_id="egocentric_relation.v1",
            family_id="family-frame",
            family_variant="frame_b",
            consistency_group="group-frame",
        ),
    ]
    shared_images = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/rgb/view-000.png"},
                {"type": "image", "image": "/rgb/view-005.png"},
            ],
        }
    ]
    for row in rows:
        row["model_input"] = shared_images
    return rows


def test_full_metrics_condition_on_revealed_correctness_and_track_coverage() -> None:
    benchmark = _benchmark_fixture()
    oracle = oracle_predictions(benchmark)
    predictions = [row for row in oracle if row["record_id"] != "single"]
    predictions.append({"record_id": "extra", "answer_value": True})

    # First evidence triple remains jointly correct. In the second triple, the
    # revealed answer is wrong, so it is excluded from the sensitivity denominator.
    next(
        row for row in predictions if row["record_id"] == "e2-revealed"
    )["answer_value"] = {"status": "absent", "category": "piano"}
    # One claim-pair member wrong makes claim-pair exact match fail.
    next(row for row in predictions if row["record_id"] == "claim-false")[
        "answer_value"
    ] = {"claim_correct": True, "relation": "left"}

    report = evaluate_records(benchmark, predictions)

    assert report["record_accuracy"] == {
        "correct": 6,
        "total": 9,
        "accuracy": 2 / 3,
        "status": "defined",
    }
    assert report["predictions"]["missing_record_ids"] == ["single"]
    assert report["predictions"]["extra_record_ids"] == ["extra"]
    assert report["per_task"]["evidence_presence_reveal"]["accuracy"] == 0.5
    assert report["per_program"]["counterfactual_cross_view.v1"]["accuracy"] == 0.5
    assert report["per_split"]["validation"]["accuracy"] == pytest.approx(2 / 3)
    assert report["family_exact_match"]["correct"] == 1
    assert report["family_exact_match"]["total"] == 3
    assert report["benchmark"]["family_id_values"] == 4
    assert report["benchmark"]["families"] == 3
    assert report["claim_pair_exact_match"]["accuracy"] == 0.0

    evidence = report["evidence_triples"]
    assert evidence["revealed_accuracy"]["accuracy"] == 0.5
    assert evidence["exact_match"]["accuracy"] == 0.5
    assert evidence["accuracy_conditioned_evidence_sensitivity"] == {
        "numerator_joint_correct": 1,
        "denominator_revealed_correct": 1,
        "value": 1.0,
        "status": "defined",
        "definition": "all three variants correct / revealed variant correct",
    }


def test_zero_revealed_correct_denominator_is_explicit_na() -> None:
    benchmark = _benchmark_fixture()
    predictions = oracle_predictions(benchmark)
    for row in predictions:
        if "revealed" in row["record_id"]:
            row["answer_value"] = "wrong"

    report = evaluate_records(benchmark, predictions)
    sensitivity = report["evidence_triples"][
        "accuracy_conditioned_evidence_sensitivity"
    ]
    assert sensitivity["denominator_revealed_correct"] == 0
    assert sensitivity["value"] is None
    assert sensitivity["status"] == "NA_ZERO_DENOMINATOR"
    assert "NA (zero revealed-correct denominator)" in render_markdown(report)


def test_frame_pair_reports_exact_equivariance_and_rejects_wrong_label_changes() -> None:
    benchmark = _frame_fixture()
    oracle_report = evaluate_records(benchmark, oracle_predictions(benchmark))
    frame = oracle_report["frame_equivariance"]
    assert frame["frame_pair_exact_match"]["accuracy"] == 1.0
    assert frame["equivariance_obligation_satisfaction"]["accuracy"] == 1.0
    assert frame["oracle_relation_change_required"]["accuracy"] == 1.0
    assert frame["prediction_change_rate_when_required"]["accuracy"] == 1.0
    assert frame["same_visual_input_contract"]["accuracy"] == 1.0
    assert frame["pairs"][0]["expected_by_frame"] == {
        "frame_a": "right_of",
        "frame_b": "behind",
    }

    wrong_but_different = [
        {"record_id": "frame-a", "answer_value": "left_of"},
        {"record_id": "frame-b", "answer_value": "in_front_of"},
    ]
    wrong_report = evaluate_records(benchmark, wrong_but_different)
    wrong_frame = wrong_report["frame_equivariance"]
    assert wrong_frame["prediction_change_rate_when_required"]["accuracy"] == 1.0
    assert wrong_frame["frame_pair_exact_match"]["accuracy"] == 0.0
    assert wrong_frame["equivariance_obligation_satisfaction"]["accuracy"] == 0.0
    assert (
        wrong_frame["prediction_change_rate_when_required"]["credit_policy"]
        == "diagnostic_only; changing between two wrong labels earns no equivariance credit"
    )


def test_duplicate_or_malformed_predictions_are_incorrect_not_silently_selected() -> None:
    benchmark = _benchmark_fixture()
    predictions = oracle_predictions(benchmark)
    predictions.append({"record_id": "single", "answer_value": True})
    next(row for row in predictions if row["record_id"] == "claim-true").pop(
        "answer_value"
    )
    predictions.append({"answer_value": True})

    report = evaluate_records(benchmark, predictions)
    assert report["predictions"]["duplicate_record_ids"] == ["single"]
    assert report["predictions"]["invalid_answer_record_ids"] == ["claim-true"]
    assert report["predictions"]["rows_without_valid_record_id"] == [11]
    result_by_id = {row["record_id"]: row for row in report["record_results"]}
    assert result_by_id["single"]["reason"] == "duplicate_prediction"
    assert result_by_id["claim-true"]["reason"] == "missing_answer_value"


def test_oracle_cli_and_reports_form_an_end_to_end_smoke_test(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.core.jsonl"
    benchmark_path.write_text(
        "".join(json.dumps(row) + "\n" for row in _benchmark_fixture()),
        encoding="utf-8",
    )
    oracle_path = tmp_path / "oracle.jsonl"
    report_dir = tmp_path / "report"

    assert (
        main(
            [
                "oracle",
                "--benchmark",
                str(benchmark_path),
                "--output",
                str(oracle_path),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "evaluate",
                "--benchmark",
                str(benchmark_path),
                "--predictions",
                str(oracle_path),
                "--output-dir",
                str(report_dir),
            ]
        )
        == 0
    )

    report = evaluate_files(benchmark_path, oracle_path)
    assert report["record_accuracy"]["accuracy"] == 1.0
    assert report["family_exact_match"]["accuracy"] == 1.0
    assert report["claim_pair_exact_match"]["accuracy"] == 1.0
    assert report["evidence_triples"]["exact_match"]["accuracy"] == 1.0
    assert report["inputs"]["benchmark_sha256"] == hashlib.sha256(
        benchmark_path.read_bytes()
    ).hexdigest()
    assert report["inputs"]["predictions_sha256"] == hashlib.sha256(
        oracle_path.read_bytes()
    ).hexdigest()
    assert (report_dir / "evaluation.json").is_file()
    assert (report_dir / "evaluation.md").is_file()
    assert "Benchmark SHA-256" in (report_dir / "evaluation.md").read_text()

    second_dir = tmp_path / "second-report"
    json_path, markdown_path = write_reports(report, second_dir)
    assert json.loads(json_path.read_text())["record_accuracy"]["correct"] == 9
    assert "Per task" in markdown_path.read_text()


def test_duplicate_benchmark_ids_fail_closed() -> None:
    benchmark = _benchmark_fixture()
    benchmark.append(dict(benchmark[0]))
    with pytest.raises(EvaluationInputError, match="must be unique"):
        evaluate_records(benchmark, [])
