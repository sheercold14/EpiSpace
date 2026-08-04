"""Deterministic evaluator for the EpiSpace structured-answer benchmark.

The evaluator intentionally consumes only the public benchmark target and a
minimal prediction schema (``record_id`` plus ``answer_value``).  It does not
depend on the simulator/compiler, so a frozen benchmark can be scored without
installing OmniGibson.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "epispace.benchmark_evaluation.v2"
DEFAULT_BENCHMARK = Path("data/epispace_pilot_v1/benchmark.core.jsonl")
CLAIM_PAIR_VARIANTS = frozenset({"claim_false", "claim_true"})
EVIDENCE_TRIPLE_VARIANTS = frozenset(
    {"prefix_unknown", "revealed", "decisive_deleted"}
)
FRAME_PAIR_VARIANTS = frozenset({"frame_a", "frame_b"})
COMPLETE_FAMILY_VARIANT_SETS = frozenset(
    {CLAIM_PAIR_VARIANTS, EVIDENCE_TRIPLE_VARIANTS, FRAME_PAIR_VARIANTS}
)


class EvaluationInputError(ValueError):
    """Raised when an input file violates the evaluator contract."""


def _reject_non_finite(value: str) -> None:
    raise EvaluationInputError(f"non-finite JSON number is not allowed: {value}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read object-only JSONL and report the precise malformed line."""

    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvaluationInputError(f"cannot read {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=_reject_non_finite)
        except (json.JSONDecodeError, EvaluationInputError) as exc:
            raise EvaluationInputError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise EvaluationInputError(
                f"{path}:{line_number}: each JSONL row must be an object"
            )
        rows.append(row)
    return rows


def _validate_benchmark(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise EvaluationInputError("benchmark is empty")
    ids: list[str] = []
    for index, row in enumerate(rows, 1):
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise EvaluationInputError(
                f"benchmark row {index} has no non-empty string record_id"
            )
        ids.append(record_id)
        target = row.get("target")
        if not isinstance(target, Mapping) or "answer_value" not in target:
            raise EvaluationInputError(
                f"benchmark record {record_id} has no target.answer_value"
            )
        if not isinstance(target.get("task_type"), str):
            raise EvaluationInputError(
                f"benchmark record {record_id} has no target.task_type"
            )
        program = row.get("program")
        if not isinstance(program, Mapping) or not isinstance(
            program.get("program_id"), str
        ):
            raise EvaluationInputError(
                f"benchmark record {record_id} has no program.program_id"
            )
        for key in ("split", "family_id", "family_variant"):
            if not isinstance(row.get(key), str):
                raise EvaluationInputError(
                    f"benchmark record {record_id} has no string {key}"
                )
    duplicate_ids = sorted(record_id for record_id, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        raise EvaluationInputError(
            "benchmark record_id values must be unique; duplicates: "
            + ", ".join(duplicate_ids)
        )


def _prediction_index(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[Mapping[str, Any]]], list[int]]:
    by_id: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    rows_without_valid_id: list[int] = []
    for index, row in enumerate(rows, 1):
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            rows_without_valid_id.append(index)
            continue
        by_id[record_id].append(row)
    return dict(by_id), rows_without_valid_id


def _numeric_tolerance(record: Mapping[str, Any]) -> float:
    certificate = record.get("certificate")
    if isinstance(certificate, Mapping):
        candidate = certificate.get("answer_tolerance_m")
        if (
            isinstance(candidate, int | float)
            and not isinstance(candidate, bool)
            and math.isfinite(float(candidate))
            and candidate >= 0
        ):
            return float(candidate)
    return 1e-6


def structured_equal(expected: Any, predicted: Any, *, tolerance: float) -> bool:
    """Compare JSON values recursively, with tolerance only for finite numbers."""

    if isinstance(expected, bool) or isinstance(predicted, bool):
        return type(expected) is bool and type(predicted) is bool and expected == predicted
    if isinstance(expected, int | float) and isinstance(predicted, int | float):
        expected_number = float(expected)
        predicted_number = float(predicted)
        return (
            math.isfinite(expected_number)
            and math.isfinite(predicted_number)
            and math.isclose(
                expected_number,
                predicted_number,
                rel_tol=0.0,
                abs_tol=tolerance,
            )
        )
    if expected is None or predicted is None:
        return expected is None and predicted is None
    if isinstance(expected, str) or isinstance(predicted, str):
        return type(expected) is str and type(predicted) is str and expected == predicted
    if isinstance(expected, Mapping) and isinstance(predicted, Mapping):
        if set(expected) != set(predicted):
            return False
        return all(
            structured_equal(expected[key], predicted[key], tolerance=tolerance)
            for key in expected
        )
    if isinstance(expected, list) and isinstance(predicted, list):
        return len(expected) == len(predicted) and all(
            structured_equal(left, right, tolerance=tolerance)
            for left, right in zip(expected, predicted, strict=True)
        )
    return type(expected) is type(predicted) and expected == predicted


def _metric(correct: int, total: int) -> dict[str, int | float | str | None]:
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
        "status": "defined" if total else "NA_ZERO_DENOMINATOR",
    }


def _group_metrics(
    rows: Sequence[Mapping[str, Any]],
    correctness: Mapping[str, bool],
    key_fn: Any,
) -> dict[str, dict[str, int | float | str | None]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        key = str(key_fn(row))
        counts[key][1] += 1
        counts[key][0] += int(correctness[row["record_id"]])
    return {key: _metric(*counts[key]) for key in sorted(counts)}


def _consistency_groups(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        group = row.get("consistency_group")
        if isinstance(group, str) and group:
            result[group].append(row)
    return dict(result)


def _model_image_sequence(row: Mapping[str, Any]) -> tuple[str, ...] | None:
    """Return the public image sequence, or None for minimal test fixtures."""

    model_input = row.get("model_input")
    if not isinstance(model_input, list):
        return None
    images: list[str] = []
    for message in model_input:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if (
                isinstance(item, Mapping)
                and item.get("type") == "image"
                and isinstance(item.get("image"), str)
            ):
                images.append(str(item["image"]))
    return tuple(images)


def _variant_family_metric(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    correctness: Mapping[str, bool],
    expected_variants: frozenset[str],
) -> tuple[dict[str, int | float | str | None], list[dict[str, Any]], list[str]]:
    eligible: list[tuple[str, Sequence[Mapping[str, Any]]]] = []
    malformed: list[str] = []
    for group_id, group_rows in groups.items():
        variants = [str(row["family_variant"]) for row in group_rows]
        variant_set = frozenset(variants)
        if not (variant_set & expected_variants):
            continue
        if variant_set != expected_variants or len(variants) != len(expected_variants):
            malformed.append(group_id)
            continue
        eligible.append((group_id, group_rows))

    detail: list[dict[str, Any]] = []
    exact = 0
    for group_id, group_rows in sorted(eligible):
        record_correctness = {
            str(row["family_variant"]): correctness[row["record_id"]]
            for row in group_rows
        }
        group_correct = all(record_correctness.values())
        exact += int(group_correct)
        detail.append(
            {
                "consistency_group": group_id,
                "exact_match": group_correct,
                "variant_correctness": record_correctness,
            }
        )
    return _metric(exact, len(eligible)), detail, sorted(malformed)


def _frame_pair_metrics(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    correctness: Mapping[str, bool],
    record_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Score paired ego-frame queries and expose each equivariance obligation.

    A prediction satisfies the geometric obligation only when both frame labels
    match their frame-conditioned targets.  Merely changing the label across
    frames is reported as a diagnostic and never counted as correctness.
    """

    eligible: list[tuple[str, Sequence[Mapping[str, Any]]]] = []
    malformed: list[str] = []
    for group_id, group_rows in groups.items():
        variants = [str(row["family_variant"]) for row in group_rows]
        variant_set = frozenset(variants)
        if not (variant_set & FRAME_PAIR_VARIANTS):
            continue
        if variant_set != FRAME_PAIR_VARIANTS or len(variants) != len(FRAME_PAIR_VARIANTS):
            malformed.append(group_id)
            continue
        eligible.append((group_id, group_rows))

    exact_count = 0
    oracle_change_count = 0
    predicted_change_on_required_count = 0
    same_visual_input_count = 0
    details: list[dict[str, Any]] = []
    for group_id, group_rows in sorted(eligible):
        by_variant = {str(row["family_variant"]): row for row in group_rows}
        frame_a = by_variant["frame_a"]
        frame_b = by_variant["frame_b"]
        result_a = record_results[str(frame_a["record_id"])]
        result_b = record_results[str(frame_b["record_id"])]
        expected_a = frame_a["target"]["answer_value"]
        expected_b = frame_b["target"]["answer_value"]
        predicted_a = result_a["predicted_answer_value"]
        predicted_b = result_b["predicted_answer_value"]
        valid_prediction_pair = result_a["reason"] in {"correct", "answer_mismatch"} and result_b[
            "reason"
        ] in {"correct", "answer_mismatch"}
        oracle_requires_change = not structured_equal(expected_a, expected_b, tolerance=1e-6)
        prediction_changes = (
            not structured_equal(predicted_a, predicted_b, tolerance=1e-6)
            if valid_prediction_pair
            else None
        )
        pair_exact = correctness[str(frame_a["record_id"])] and correctness[
            str(frame_b["record_id"])
        ]
        image_sequence_a = _model_image_sequence(frame_a)
        image_sequence_b = _model_image_sequence(frame_b)
        same_visual_input = (
            image_sequence_a == image_sequence_b
            if image_sequence_a is not None and image_sequence_b is not None
            else None
        )
        exact_count += int(pair_exact)
        oracle_change_count += int(oracle_requires_change)
        predicted_change_on_required_count += int(
            oracle_requires_change and prediction_changes is True
        )
        same_visual_input_count += int(same_visual_input is True)
        details.append(
            {
                "consistency_group": group_id,
                "family_id": frame_a["family_id"],
                "record_ids": {
                    "frame_a": frame_a["record_id"],
                    "frame_b": frame_b["record_id"],
                },
                "expected_by_frame": {
                    "frame_a": expected_a,
                    "frame_b": expected_b,
                },
                "predicted_by_frame": {
                    "frame_a": predicted_a,
                    "frame_b": predicted_b,
                },
                "record_correctness": {
                    "frame_a": correctness[str(frame_a["record_id"])],
                    "frame_b": correctness[str(frame_b["record_id"])],
                },
                "same_visual_input": same_visual_input,
                "oracle_requires_relation_change": oracle_requires_change,
                "prediction_changes_with_frame": prediction_changes,
                "frame_pair_exact_match": pair_exact,
                "equivariance_obligation_satisfied": pair_exact,
            }
        )

    change_required_total = oracle_change_count
    return {
        "frame_pair_exact_match": _metric(exact_count, len(eligible)),
        "equivariance_obligation_satisfaction": {
            **_metric(exact_count, len(eligible)),
            "definition": (
                "both frame-conditioned relation labels exactly match their targets"
            ),
        },
        "oracle_relation_change_required": _metric(
            oracle_change_count,
            len(eligible),
        ),
        "prediction_change_rate_when_required": {
            **_metric(predicted_change_on_required_count, change_required_total),
            "credit_policy": (
                "diagnostic_only; changing between two wrong labels earns no "
                "equivariance credit"
            ),
        },
        "same_visual_input_contract": _metric(
            same_visual_input_count,
            sum(
                detail["same_visual_input"] is not None
                for detail in details
            ),
        ),
        "malformed_or_incomplete_groups": sorted(malformed),
        "pairs": details,
    }


def evaluate_records(
    benchmark_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate structured predictions against an in-memory benchmark."""

    _validate_benchmark(benchmark_rows)
    benchmark_ids = {str(row["record_id"]) for row in benchmark_rows}
    predictions_by_id, rows_without_valid_id = _prediction_index(prediction_rows)
    prediction_ids = set(predictions_by_id)

    missing_ids = sorted(benchmark_ids - prediction_ids)
    extra_ids = sorted(prediction_ids - benchmark_ids)
    duplicate_ids = sorted(
        record_id for record_id, rows in predictions_by_id.items() if len(rows) > 1
    )
    invalid_answer_ids = sorted(
        record_id
        for record_id, rows in predictions_by_id.items()
        if len(rows) == 1 and "answer_value" not in rows[0]
    )

    correctness: dict[str, bool] = {}
    record_results: list[dict[str, Any]] = []
    for benchmark in benchmark_rows:
        record_id = str(benchmark["record_id"])
        candidates = predictions_by_id.get(record_id, [])
        if not candidates:
            reason = "missing_prediction"
            predicted: Any = None
            correct = False
        elif len(candidates) > 1:
            reason = "duplicate_prediction"
            predicted = None
            correct = False
        elif "answer_value" not in candidates[0]:
            reason = "missing_answer_value"
            predicted = None
            correct = False
        else:
            predicted = candidates[0]["answer_value"]
            expected = benchmark["target"]["answer_value"]
            correct = structured_equal(
                expected,
                predicted,
                tolerance=_numeric_tolerance(benchmark),
            )
            reason = "correct" if correct else "answer_mismatch"
        correctness[record_id] = correct
        record_results.append(
            {
                "record_id": record_id,
                "correct": correct,
                "reason": reason,
                "expected_answer_value": benchmark["target"]["answer_value"],
                "predicted_answer_value": predicted,
            }
        )

    total_correct = sum(correctness.values())
    consistency_groups = _consistency_groups(benchmark_rows)
    family_rows: dict[str, list[Mapping[str, Any]]] = {}
    malformed_family_groups: list[str] = []
    for group_id, rows in consistency_groups.items():
        variants = frozenset(str(row["family_variant"]) for row in rows)
        family_ids = {str(row["family_id"]) for row in rows}
        if variants not in COMPLETE_FAMILY_VARIANT_SETS or len(family_ids) != 1:
            malformed_family_groups.append(group_id)
            continue
        family_rows[group_id] = rows
    exact_families = sum(
        all(correctness[row["record_id"]] for row in rows)
        for rows in family_rows.values()
    )
    family_details = [
        {
            "consistency_group": group_id,
            "family_id": str(rows[0]["family_id"]),
            "record_count": len(rows),
            "exact_match": all(correctness[row["record_id"]] for row in rows),
        }
        for group_id, rows in sorted(family_rows.items())
    ]

    results_by_id = {row["record_id"]: row for row in record_results}
    claim_metric, claim_details, malformed_claim = _variant_family_metric(
        consistency_groups,
        correctness,
        CLAIM_PAIR_VARIANTS,
    )
    evidence_exact, evidence_details, malformed_evidence = _variant_family_metric(
        consistency_groups,
        correctness,
        EVIDENCE_TRIPLE_VARIANTS,
    )
    evidence_groups = [
        rows
        for rows in consistency_groups.values()
        if len(rows) == len(EVIDENCE_TRIPLE_VARIANTS)
        and {str(row["family_variant"]) for row in rows} == EVIDENCE_TRIPLE_VARIANTS
    ]
    revealed_correct = 0
    joint_correct = 0
    for rows in evidence_groups:
        by_variant = {str(row["family_variant"]): row for row in rows}
        is_revealed_correct = correctness[by_variant["revealed"]["record_id"]]
        revealed_correct += int(is_revealed_correct)
        joint_correct += int(
            is_revealed_correct
            and correctness[by_variant["prefix_unknown"]["record_id"]]
            and correctness[by_variant["decisive_deleted"]["record_id"]]
        )
    sensitivity_value = (
        joint_correct / revealed_correct if revealed_correct else None
    )
    frame_pairs = _frame_pair_metrics(
        consistency_groups,
        correctness,
        results_by_id,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "comparison_policy": {
            "structured_json": "recursive_exact_types_and_keys",
            "numeric": (
                "absolute tolerance from certificate.answer_tolerance_m; "
                "otherwise 1e-6"
            ),
            "missing_duplicate_or_malformed_prediction": "incorrect",
        },
        "benchmark": {
            "records": len(benchmark_rows),
            "families": len(family_rows),
            "family_id_values": len(
                {str(row["family_id"]) for row in benchmark_rows}
            ),
            "consistency_groups": len(consistency_groups),
        },
        "predictions": {
            "rows": len(prediction_rows),
            "unique_record_ids": len(prediction_ids),
            "matched_unique_ids": len(benchmark_ids & prediction_ids),
            "missing_count": len(missing_ids),
            "missing_record_ids": missing_ids,
            "extra_count": len(extra_ids),
            "extra_record_ids": extra_ids,
            "duplicate_id_count": len(duplicate_ids),
            "duplicate_record_ids": duplicate_ids,
            "invalid_answer_count": len(invalid_answer_ids),
            "invalid_answer_record_ids": invalid_answer_ids,
            "rows_without_valid_record_id": rows_without_valid_id,
        },
        "record_accuracy": _metric(total_correct, len(benchmark_rows)),
        "per_task": _group_metrics(
            benchmark_rows,
            correctness,
            lambda row: row["target"]["task_type"],
        ),
        "per_program": _group_metrics(
            benchmark_rows,
            correctness,
            lambda row: row["program"]["program_id"],
        ),
        "per_split": _group_metrics(
            benchmark_rows,
            correctness,
            lambda row: row["split"],
        ),
        "family_exact_match": {
            **_metric(exact_families, len(family_rows)),
            "definition": (
                "exact match over complete claim, evidence, or frame sibling "
                "families; canonical non-sibling records are excluded"
            ),
            "malformed_or_incomplete_groups": sorted(malformed_family_groups),
            "families": family_details,
        },
        "claim_pair_exact_match": {
            **claim_metric,
            "malformed_or_incomplete_groups": malformed_claim,
            "pairs": claim_details,
        },
        "frame_equivariance": frame_pairs,
        "evidence_triples": {
            "revealed_accuracy": _metric(revealed_correct, len(evidence_groups)),
            "exact_match": evidence_exact,
            "accuracy_conditioned_evidence_sensitivity": {
                "numerator_joint_correct": joint_correct,
                "denominator_revealed_correct": revealed_correct,
                "value": sensitivity_value,
                "status": (
                    "defined" if revealed_correct else "NA_ZERO_DENOMINATOR"
                ),
                "definition": (
                    "all three variants correct / revealed variant correct"
                ),
            },
            "malformed_or_incomplete_groups": malformed_evidence,
            "triples": evidence_details,
        },
        "record_results": record_results,
    }


def evaluate_files(benchmark_path: Path, predictions_path: Path) -> dict[str, Any]:
    """Load and evaluate a benchmark/prediction JSONL pair."""

    report = evaluate_records(read_jsonl(benchmark_path), read_jsonl(predictions_path))
    report["inputs"] = {
        "benchmark": str(benchmark_path.resolve()),
        "benchmark_sha256": _file_sha256(benchmark_path),
        "predictions": str(predictions_path.resolve()),
        "predictions_sha256": _file_sha256(predictions_path),
    }
    return report


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_accuracy(metric: Mapping[str, Any]) -> str:
    accuracy = metric.get("accuracy")
    return "NA" if accuracy is None else f"{float(accuracy) * 100:.2f}%"


def _markdown_table(title: str, rows: Mapping[str, Mapping[str, Any]]) -> list[str]:
    result = [f"## {title}", "", "| Group | Correct | Total | Accuracy |", "|---|---:|---:|---:|"]
    for key, metric in rows.items():
        safe_key = key.replace("|", "\\|")
        result.append(
            f"| `{safe_key}` | {metric['correct']} | {metric['total']} | "
            f"{_format_accuracy(metric)} |"
        )
    result.append("")
    return result


def _render_id_list(ids: Sequence[Any], *, limit: int = 100) -> str:
    if not ids:
        return "None"
    rendered = ", ".join(f"`{value}`" for value in ids[:limit])
    if len(ids) > limit:
        rendered += f" … (+{len(ids) - limit} more; see JSON report)"
    return rendered


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise human-readable companion to the lossless JSON report."""

    record = report["record_accuracy"]
    family = report["family_exact_match"]
    claim = report["claim_pair_exact_match"]
    frame = report["frame_equivariance"]
    frame_exact = frame["frame_pair_exact_match"]
    equivariance = frame["equivariance_obligation_satisfaction"]
    evidence = report["evidence_triples"]
    revealed = evidence["revealed_accuracy"]
    evidence_exact = evidence["exact_match"]
    sensitivity = evidence["accuracy_conditioned_evidence_sensitivity"]
    prediction = report["predictions"]
    sensitivity_text = (
        "NA (zero revealed-correct denominator)"
        if sensitivity["value"] is None
        else f"{float(sensitivity['value']) * 100:.2f}%"
    )

    lines = [
        "# EpiSpace benchmark evaluation",
        "",
        "## Summary",
        "",
        "| Metric | Correct | Total | Accuracy |",
        "|---|---:|---:|---:|",
        f"| Record accuracy | {record['correct']} | {record['total']} | {_format_accuracy(record)} |",
        f"| Family exact match | {family['correct']} | {family['total']} | {_format_accuracy(family)} |",
        f"| Claim-pair exact match | {claim['correct']} | {claim['total']} | {_format_accuracy(claim)} |",
        f"| Frame-pair exact match | {frame_exact['correct']} | {frame_exact['total']} | {_format_accuracy(frame_exact)} |",
        f"| Frame-equivariance obligation | {equivariance['correct']} | {equivariance['total']} | {_format_accuracy(equivariance)} |",
        f"| Evidence revealed accuracy | {revealed['correct']} | {revealed['total']} | {_format_accuracy(revealed)} |",
        f"| Evidence-triple exact match | {evidence_exact['correct']} | {evidence_exact['total']} | {_format_accuracy(evidence_exact)} |",
        "",
        "Accuracy-conditioned evidence sensitivity is **"
        + sensitivity_text
        + "** (joint correct / revealed-correct = "
        + f"{sensitivity['numerator_joint_correct']}/"
        + f"{sensitivity['denominator_revealed_correct']}).",
        "",
        "## Prediction coverage",
        "",
        f"- Prediction rows: {prediction['rows']}; unique IDs: {prediction['unique_record_ids']}.",
        f"- Missing ({prediction['missing_count']}): "
        + _render_id_list(prediction["missing_record_ids"]),
        f"- Extra ({prediction['extra_count']}): "
        + _render_id_list(prediction["extra_record_ids"]),
        f"- Duplicate IDs ({prediction['duplicate_id_count']}): "
        + _render_id_list(prediction["duplicate_record_ids"]),
        f"- Missing `answer_value` ({prediction['invalid_answer_count']}): "
        + _render_id_list(prediction["invalid_answer_record_ids"]),
        "- Rows without a valid `record_id`: "
        + _render_id_list(prediction["rows_without_valid_record_id"]),
        "",
    ]
    lines.extend(_markdown_table("Per task", report["per_task"]))
    lines.extend(_markdown_table("Per program", report["per_program"]))
    lines.extend(_markdown_table("Per split", report["per_split"]))

    inputs = report.get("inputs")
    if isinstance(inputs, Mapping):
        lines.extend(
            [
                "## Source binding",
                "",
                f"- Benchmark SHA-256: `{inputs['benchmark_sha256']}`",
                f"- Predictions SHA-256: `{inputs['predictions_sha256']}`",
                "",
            ]
        )

    malformed_claim = claim["malformed_or_incomplete_groups"]
    malformed_evidence = evidence["malformed_or_incomplete_groups"]
    malformed_frame = frame["malformed_or_incomplete_groups"]
    lines.extend(
        [
            "## Structural diagnostics",
            "",
            f"- Malformed/incomplete claim groups: {_render_id_list(malformed_claim)}",
            f"- Malformed/incomplete evidence groups: {_render_id_list(malformed_evidence)}",
            f"- Malformed/incomplete frame groups: {_render_id_list(malformed_frame)}",
            f"- Same-visual-input frame contract: {_format_accuracy(frame['same_visual_input_contract'])}",
            f"- Oracle requires a relation change: {_format_accuracy(frame['oracle_relation_change_required'])}",
            "- Predicted change rate where required (diagnostic only): "
            + _format_accuracy(frame["prediction_change_rate_when_required"]),
            "",
            "The JSON report contains per-record, per-family, per-claim-pair, "
            "per-frame-pair, and per-evidence-triple details.",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(report: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write lossless JSON plus a Markdown summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evaluation.json"
    markdown_path = output_dir / "evaluation.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def oracle_predictions(
    benchmark_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Create the minimal prediction rows expected from a perfect oracle."""

    rows = list(benchmark_rows)
    _validate_benchmark(rows)
    return [
        {
            "record_id": row["record_id"],
            "answer_value": row["target"]["answer_value"],
        }
        for row in rows
    ]


def write_oracle_predictions(benchmark_path: Path, output_path: Path) -> Path:
    """Generate an oracle JSONL file for evaluator smoke testing."""

    rows = oracle_predictions(read_jsonl(benchmark_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return output_path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Evaluate EpiSpace benchmark.core structured answers."
    )
    commands = result.add_subparsers(dest="command", required=True)

    evaluate = commands.add_parser("evaluate", help="write JSON and Markdown reports")
    evaluate.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)

    oracle = commands.add_parser(
        "oracle", help="write perfect structured predictions for a smoke test"
    )
    oracle.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    oracle.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "oracle":
            output = write_oracle_predictions(args.benchmark, args.output)
            print(json.dumps({"oracle_predictions": str(output)}, ensure_ascii=False))
            return 0
        report = evaluate_files(args.benchmark, args.predictions)
        json_path, markdown_path = write_reports(report, args.output_dir)
        print(
            json.dumps(
                {
                    "record_accuracy": report["record_accuracy"]["accuracy"],
                    "json_report": str(json_path),
                    "markdown_report": str(markdown_path),
                },
                ensure_ascii=False,
            )
        )
        return 0
    except EvaluationInputError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
