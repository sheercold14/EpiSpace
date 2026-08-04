"""Independent, read-only audit for an exported EpiSpace corpus.

The builder has its own release gates.  This module deliberately does not call
the builder, compiler, verifier, or exporter: it re-reads the serialized corpus
and measures what a downstream researcher actually receives.

Run it with::

    python -m episode3d.corpus_audit data/epispace_pilot_v1

By default, ``corpus_audit.json`` and ``corpus_audit.md`` are written inside
the release directory.  A model prediction file is optional; without one the
report states which family metrics are structurally evaluable, but never
pretends that model accuracy has been measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from episode3d.language import CATEGORY_ZH, RELATION_ZH

AUDIT_SCHEMA_VERSION = "epispace.corpus_audit.v1"
UTC = timezone.utc

IR_NAME = "episodes.ir.jsonl"
EPISODE_NAME = "train.episode_sft.jsonl"
ISOLATED_NAME = "train.isolated_sft.jsonl"
STATE_AUX_NAME = "train.state_aux_sft.jsonl"
RLVR_NAME = "train.rlvr.jsonl"
BENCHMARK_NAME = "benchmark.jsonl"
BENCHMARK_FAMILY_NAME = "benchmark.family.jsonl"
BENCHMARK_COMPOSITION_NAME = "benchmark.composition.jsonl"
BENCHMARK_CORE_NAME = "benchmark.core.jsonl"
BENCHMARK_NAMES = (
    BENCHMARK_NAME,
    BENCHMARK_FAMILY_NAME,
    BENCHMARK_COMPOSITION_NAME,
    BENCHMARK_CORE_NAME,
)

MODEL_INPUT_ARTIFACT_NAMES = (
    EPISODE_NAME,
    ISOLATED_NAME,
    STATE_AUX_NAME,
    RLVR_NAME,
    *BENCHMARK_NAMES,
)
FROZEN_MAX_RGB_DOMINANT_COLOR_FRACTION = 0.60
FROZEN_MIN_RGB_QUANTIZED_ENTROPY_BITS = 2.75

CLAIM_VARIANTS = frozenset({"claim_false", "claim_true"})
EVIDENCE_VARIANTS = frozenset({"prefix_unknown", "revealed", "decisive_deleted"})
FRAME_VARIANTS = frozenset({"frame_a", "frame_b"})


@dataclass(frozen=True)
class LoadedArtifact:
    """A JSONL artifact plus parse errors that must remain visible in the report."""

    name: str
    path: Path
    rows: list[dict[str, Any]]
    errors: list[str]

    @property
    def available(self) -> bool:
        return self.path.is_file() and not self.errors


def _json_canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _target_key(status: Any, value: Any) -> str:
    return _json_canonical({"status": status, "value": value})


def _display_target(key: str, *, limit: int = 100) -> str:
    if len(key) <= limit:
        return key
    return key[: limit - 1] + "…"


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def _read_jsonl(path: Path) -> LoadedArtifact:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.is_file():
        return LoadedArtifact(path.name, path, rows, [f"missing artifact: {path}"])
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON ({exc.msg})")
                continue
            if not isinstance(value, dict):
                errors.append(f"line {line_number}: expected JSON object")
                continue
            rows.append(value)
    return LoadedArtifact(path.name, path, rows, errors)


def _missing_field_summary(
    rows: Sequence[Mapping[str, Any]], required: Sequence[str]
) -> dict[str, Any]:
    counts = {field: 0 for field in required}
    samples: dict[str, list[int]] = {field: [] for field in required}
    for index, row in enumerate(rows):
        for field in required:
            if field not in row:
                counts[field] += 1
                if len(samples[field]) < 5:
                    samples[field].append(index)
    missing = {
        field: {"record_count": count, "row_indices": samples[field]}
        for field, count in counts.items()
        if count
    }
    return {
        "record_count": len(rows),
        "required_fields": list(required),
        "complete": not missing,
        "missing": missing,
    }


def _iter_ir_questions(
    episodes: Sequence[Mapping[str, Any]],
) -> Iterator[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    for episode in episodes:
        questions = episode.get("questions")
        if not isinstance(questions, list):
            continue
        for question in questions:
            if isinstance(question, dict):
                yield episode, question


def _distribution_entries(counter: Counter[str]) -> list[dict[str, Any]]:
    total = sum(counter.values())
    return [
        {
            "target": label,
            "count": count,
            "fraction": _safe_ratio(count, total),
        }
        for label, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _flatten_structured_target(status: Any, value: Any) -> dict[str, str]:
    """Expose marginal fields without losing the exact joint target distribution."""

    flattened = {"status": _json_canonical(status)}

    def visit(item: Any, path: str) -> None:
        if isinstance(item, dict):
            if not item:
                flattened[path] = _json_canonical(item)
            for key, child in sorted(item.items()):
                visit(child, f"{path}.{key}")
        else:
            # Lists remain atomic because positional list marginals are rarely meaningful
            # for spatial answers and would silently conflate different output schemas.
            flattened[path] = _json_canonical(item)

    visit(value, "value")
    return flattened


def _majority(counter: Counter[str]) -> dict[str, Any]:
    total = sum(counter.values())
    if not counter:
        return {
            "status": "not_evaluable",
            "reason": "no valid structured targets",
            "count": 0,
            "accuracy": None,
            "labels": [],
        }
    count = max(counter.values())
    labels = sorted(label for label, label_count in counter.items() if label_count == count)
    return {
        "status": "evaluable",
        "count": count,
        "accuracy": _safe_ratio(count, total),
        "labels": labels,
        "tie_count": len(labels),
    }


def _target_and_baseline_audit(episodes: LoadedArtifact) -> dict[str, Any]:
    if not episodes.path.is_file():
        return {
            "status": "not_evaluable",
            "reason": f"{IR_NAME} is missing",
            "groups": [],
            "task_conditioned_majority": {},
        }

    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    marginal_grouped: dict[tuple[str, str], dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    missing: Counter[str] = Counter()
    question_count = 0
    for episode, question in _iter_ir_questions(episodes.rows):
        question_count += 1
        split = episode.get("split")
        task = question.get("task_type")
        if split is None:
            missing["episode.split"] += 1
        if task is None:
            missing["question.task_type"] += 1
        if "answer_status" not in question:
            missing["question.answer_status"] += 1
        if "answer_value" not in question:
            missing["question.answer_value"] += 1
        if split is None or task is None or "answer_status" not in question or "answer_value" not in question:
            continue
        group_key = (str(split), str(task))
        grouped[group_key][
            _target_key(question["answer_status"], question["answer_value"])
        ] += 1
        for field, field_value in _flatten_structured_target(
            question["answer_status"], question["answer_value"]
        ).items():
            marginal_grouped[group_key][field][field_value] += 1

    groups: list[dict[str, Any]] = []
    split_task_counters: dict[str, dict[str, Counter[str]]] = defaultdict(dict)
    for (split, task), counter in sorted(grouped.items()):
        split_task_counters[split][task] = counter
        groups.append(
            {
                "split": split,
                "task_type": task,
                "count": sum(counter.values()),
                "unique_target_count": len(counter),
                "distribution": _distribution_entries(counter),
                "field_distributions": {
                    field: _distribution_entries(field_counter)
                    for field, field_counter in sorted(marginal_grouped[(split, task)].items())
                },
                "within_split_majority": _majority(counter),
            }
        )

    within_split: dict[str, Any] = {}
    for split, task_counters in sorted(split_task_counters.items()):
        correct = sum(max(counter.values()) for counter in task_counters.values() if counter)
        total = sum(sum(counter.values()) for counter in task_counters.values())
        within_split[split] = {
            "correct": correct,
            "total": total,
            "accuracy": _safe_ratio(correct, total),
            "task_count": len(task_counters),
            "interpretation": (
                "oracle label-prior upper bound: each task predicts its most frequent target "
                "measured on this same split"
            ),
        }

    train_counters = split_task_counters.get("train", {})
    train_labels: dict[str, str] = {}
    train_ties: dict[str, list[str]] = {}
    for task, counter in train_counters.items():
        majority = _majority(counter)
        labels = majority.get("labels", [])
        if labels:
            # Stable tie breaking is part of the baseline definition.
            train_labels[task] = labels[0]
            if len(labels) > 1:
                train_ties[task] = labels

    train_fitted: dict[str, Any] = {}
    for split, task_counters in sorted(split_task_counters.items()):
        if split == "train":
            continue
        per_task: list[dict[str, Any]] = []
        covered_correct = 0
        covered_total = 0
        all_total = sum(sum(counter.values()) for counter in task_counters.values())
        for task, counter in sorted(task_counters.items()):
            if task not in train_labels:
                per_task.append(
                    {
                        "task_type": task,
                        "status": "not_evaluable",
                        "reason": "task has no valid structured target in train",
                        "count": sum(counter.values()),
                    }
                )
                continue
            predicted = train_labels[task]
            correct = counter[predicted]
            total = sum(counter.values())
            covered_correct += correct
            covered_total += total
            per_task.append(
                {
                    "task_type": task,
                    "status": "evaluated",
                    "predicted_target": predicted,
                    "correct": correct,
                    "total": total,
                    "accuracy": _safe_ratio(correct, total),
                }
            )
        train_fitted[split] = {
            "status": "evaluated" if covered_total else "not_evaluable",
            "correct": covered_correct,
            "evaluated_count": covered_total,
            "total_count": all_total,
            "coverage": _safe_ratio(covered_total, all_total),
            "accuracy_on_covered": _safe_ratio(covered_correct, covered_total),
            "per_task": per_task,
        }

    status = "complete" if not missing and not episodes.errors else "partial"
    return {
        "status": status,
        "source": IR_NAME,
        "question_count": question_count,
        "valid_target_count": sum(sum(counter.values()) for counter in grouped.values()),
        "missing_fields": dict(sorted(missing.items())),
        "parse_errors": episodes.errors,
        "groups": groups,
        "task_conditioned_majority": {
            "definition": (
                "A no-image label-prior baseline conditioned only on task_type. This is not a "
                "question-only VLM/LLM run."
            ),
            "within_split_oracle": within_split,
            "train_fitted": {
                "tie_break": "lexicographically smallest canonical JSON target",
                "train_ties": train_ties,
                "evaluation": train_fitted,
            },
            "question_only_model": {
                "status": "not_evaluated",
                "reason": "the release contains no question-only model predictions",
            },
        },
    }


def _benchmark_partition_target_audit(
    artifacts: Mapping[str, LoadedArtifact],
) -> dict[str, Any]:
    """Measure distributions of the published benchmark views, not just the superset IR."""

    result: dict[str, Any] = {}
    for name in BENCHMARK_NAMES:
        artifact = artifacts[name]
        if not artifact.path.is_file():
            result[name] = {
                "status": "not_evaluable",
                "reason": f"missing artifact: {artifact.path}",
                "record_count": 0,
                "groups": [],
            }
            continue
        grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        missing: Counter[str] = Counter()
        for row in artifact.rows:
            split = row.get("split")
            target = row.get("target")
            if split is None:
                missing["split"] += 1
            if not isinstance(target, dict):
                missing["target"] += 1
                continue
            task = target.get("task_type")
            if task is None:
                missing["target.task_type"] += 1
            if "answer_status" not in target:
                missing["target.answer_status"] += 1
            if "answer_value" not in target:
                missing["target.answer_value"] += 1
            if (
                split is None
                or task is None
                or "answer_status" not in target
                or "answer_value" not in target
            ):
                continue
            grouped[(str(split), str(task))][
                _target_key(target["answer_status"], target["answer_value"])
            ] += 1
        groups = [
            {
                "split": split,
                "task_type": task,
                "count": sum(counter.values()),
                "unique_target_count": len(counter),
                "distribution": _distribution_entries(counter),
                "within_split_majority": _majority(counter),
            }
            for (split, task), counter in sorted(grouped.items())
        ]
        correct = sum(
            group["within_split_majority"]["count"]
            for group in groups
            if group["within_split_majority"]["status"] == "evaluable"
        )
        valid = sum(group["count"] for group in groups)
        result[name] = {
            "status": "complete" if not missing and not artifact.errors else "partial",
            "record_count": len(artifact.rows),
            "valid_target_count": valid,
            "missing_fields": dict(sorted(missing.items())),
            "parse_errors": artifact.errors,
            "task_conditioned_within_split_majority_accuracy": _safe_ratio(correct, valid),
            "groups": groups,
        }
    return result


def _family_contract(variants: set[str]) -> tuple[str | None, frozenset[str] | None]:
    if variants & CLAIM_VARIANTS:
        return "claim_verification_pair", CLAIM_VARIANTS
    if variants & EVIDENCE_VARIANTS:
        return "evidence_intervention_triple", EVIDENCE_VARIANTS
    if variants & FRAME_VARIANTS:
        return "frame_reference_pair", FRAME_VARIANTS
    return None, None


def _single_replacement_check(
    by_variant: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Check the matched visual intervention promised by an evidence triple.

    The primary contract is set based: all three inputs share N-1 images and
    each contributes one distinct replacement.  Ordered-slot stability is
    reported separately because reordering a shared image can itself expose a
    position shortcut even when the set-level intervention is correct.
    """

    view_lists = {
        variant: member.get("model_view_ids") for variant, member in by_variant.items()
    }
    if any(not isinstance(views, list) for views in view_lists.values()):
        return {
            "status": "not_evaluable",
            "passed": False,
            "reason": "model_view_ids is missing or not a list",
            "view_ids_by_variant": view_lists,
        }
    lengths = {len(views) for views in view_lists.values()}
    if len(lengths) != 1:
        return {
            "status": "evaluated",
            "passed": False,
            "reason": "family members expose different image counts",
            "view_ids_by_variant": view_lists,
        }
    image_count = lengths.pop()
    ordered_lists = list(view_lists.values())
    if any(len(set(views)) != image_count for views in ordered_lists):
        return {
            "status": "evaluated",
            "passed": False,
            "reason": "one or more family inputs repeat an image path",
            "image_count": image_count,
            "view_ids_by_variant": view_lists,
        }
    shared_views = set(ordered_lists[0]).intersection(*(set(views) for views in ordered_lists[1:]))
    replacement_views = {
        variant: sorted(set(views) - shared_views)
        for variant, views in view_lists.items()
    }
    set_replacements_distinct = (
        len(shared_views) == image_count - 1
        and all(len(values) == 1 for values in replacement_views.values())
        and len({values[0] for values in replacement_views.values()}) == len(replacement_views)
    )
    set_level_passed = image_count > 0 and set_replacements_distinct

    shared_slots = [
        index
        for index in range(image_count)
        if len({views[index] for views in ordered_lists}) == 1
    ]
    replaced_slots = [index for index in range(image_count) if index not in shared_slots]
    ordered_replacements_distinct = (
        len(replaced_slots) == 1
        and len({views[replaced_slots[0]] for views in ordered_lists}) == len(ordered_lists)
    )
    ordered_slot_stable = (
        image_count > 0
        and len(shared_slots) == image_count - 1
        and len(replaced_slots) == 1
        and ordered_replacements_distinct
    )
    return {
        "status": "evaluated",
        "passed": set_level_passed,
        "reason": None
        if set_level_passed
        else "expected N-1 shared images and one distinct replacement per family member",
        "image_count": image_count,
        "shared_view_count": len(shared_views),
        "shared_view_ids": sorted(shared_views),
        "replacement_view_ids_by_variant": replacement_views,
        "set_replacements_distinct": set_replacements_distinct,
        "ordered_slot_stable": ordered_slot_stable,
        "shared_slot_count": len(shared_slots),
        "shared_slot_indices": shared_slots,
        "replaced_slot_indices": replaced_slots,
        "ordered_replacements_distinct": ordered_replacements_distinct,
        "view_ids_by_variant": view_lists,
    }


def _named_certificate_check(member: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    certificate = member.get("certificate")
    if not isinstance(certificate, dict):
        return None
    checks = certificate.get("checks")
    if not isinstance(checks, list):
        return None
    for check in checks:
        if isinstance(check, dict) and check.get("name") == name:
            return check
    return None


def _frame_pair_check(by_variant: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Audit a controlled ego-frame transformation without trusting its prose label alone."""

    reasons: list[str] = []
    if set(by_variant) != set(FRAME_VARIANTS):
        return {
            "status": "not_evaluable",
            "passed": False,
            "reasons": ["family does not contain exactly frame_a and frame_b"],
        }
    frame_a = by_variant["frame_a"]
    frame_b = by_variant["frame_b"]
    views_a = frame_a.get("model_view_ids")
    views_b = frame_b.get("model_view_ids")
    same_input = (
        isinstance(views_a, list)
        and isinstance(views_b, list)
        and len(views_a) >= 2
        and views_a == views_b
        and len(set(views_a)) == len(views_a)
    )
    if not same_input:
        reasons.append("siblings do not expose the same ordered, non-duplicated image input")

    question_a = frame_a.get("question")
    question_b = frame_b.get("question")
    normalized_a = None
    normalized_b = None
    if isinstance(question_a, str) and isinstance(question_b, str):
        normalized_a = question_a.replace("第1张图", "第<FRAME>张图")
        normalized_b = question_b.replace("第2张图", "第<FRAME>张图")
        if "第1张图" not in question_a or "第2张图" not in question_b:
            reasons.append("questions do not explicitly declare frame 1 versus frame 2")
        if normalized_a != normalized_b:
            reasons.append("question content changes beyond the declared reference frame")
    else:
        reasons.append("question text is missing")

    targets = {
        variant: member.get("answer_value") for variant, member in by_variant.items()
    }
    if any(member.get("answer_status") != "accepted" for member in by_variant.values()):
        reasons.append("one or more frame targets are not accepted answers")
    if any(not isinstance(target, str) or target not in RELATION_ZH for target in targets.values()):
        reasons.append("frame target is not a registered spatial relation")
    if targets["frame_a"] == targets["frame_b"]:
        reasons.append("answer does not transform when the reference frame changes")

    if len({member.get("task_type") for member in by_variant.values()}) != 1:
        reasons.append("task_type changes across frame siblings")
    if len({member.get("program_id") for member in by_variant.values()}) != 1:
        reasons.append("program_id changes across frame siblings")
    if len({member.get("semantic_signature") for member in by_variant.values()}) != 1:
        reasons.append("semantic signature changes across frame siblings")

    anchor_ids: dict[str, Any] = {}
    yaw_values: dict[str, Any] = {}
    certificate_relations: dict[str, Any] = {}
    for index, variant in enumerate(("frame_a", "frame_b")):
        member = by_variant[variant]
        same_input_check = _named_certificate_check(
            member, "same_two_images_across_frame_siblings"
        )
        relation_check = _named_certificate_check(member, "camera_frame_relation_recomputed")
        if (
            not isinstance(same_input_check, Mapping)
            or same_input_check.get("passed") is not True
            or same_input_check.get("model_view_ids") != views_a
        ):
            reasons.append(f"{variant} lacks a matching same-input certificate check")
        if not isinstance(relation_check, Mapping) or relation_check.get("passed") is not True:
            reasons.append(f"{variant} lacks a passing frame-recomputation certificate check")
            continue
        anchor_ids[variant] = relation_check.get("anchor_view_id")
        yaw_values[variant] = relation_check.get("yaw_separation_deg")
        certificate_relations[variant] = relation_check.get("relation")
        expected_anchor = views_a[index] if same_input else None
        if relation_check.get("anchor_view_id") != expected_anchor:
            reasons.append(f"{variant} certificate anchor does not match its declared image frame")
        if relation_check.get("relation") != targets[variant]:
            reasons.append(f"{variant} certificate relation does not match its target")

    numeric_yaws = [value for value in yaw_values.values() if isinstance(value, (int, float))]
    if len(numeric_yaws) != 2:
        reasons.append("yaw separation is missing from a frame certificate")
    elif numeric_yaws[0] <= 0 or not math.isclose(
        numeric_yaws[0], numeric_yaws[1], rel_tol=1e-6, abs_tol=1e-6
    ):
        reasons.append("siblings do not share the same non-zero frame rotation")

    return {
        "status": "evaluated",
        "passed": not reasons,
        "reasons": sorted(set(reasons)),
        "same_ordered_model_input": same_input,
        "model_view_ids": views_a if same_input else {"frame_a": views_a, "frame_b": views_b},
        "normalized_question_equal": normalized_a is not None and normalized_a == normalized_b,
        "answer_values": targets,
        "answer_transformed": targets["frame_a"] != targets["frame_b"],
        "certificate_anchor_view_ids": anchor_ids,
        "certificate_relations": certificate_relations,
        "yaw_separation_deg": yaw_values,
    }


def _load_predictions(path: Path | None) -> tuple[dict[str, bool], dict[str, Any]]:
    if path is None:
        return {}, {
            "status": "not_provided",
            "reason": "pass --predictions with JSONL rows containing fact_id and boolean correct",
        }
    artifact = _read_jsonl(path)
    predictions: dict[str, bool] = {}
    missing = 0
    duplicate = 0
    for row in artifact.rows:
        fact_id = row.get("fact_id")
        correct = row.get("correct")
        if not isinstance(fact_id, str) or not isinstance(correct, bool):
            missing += 1
            continue
        if fact_id in predictions:
            duplicate += 1
        predictions[fact_id] = correct
    return predictions, {
        "status": "loaded" if artifact.path.is_file() else "not_evaluable",
        "path": str(path),
        "valid_prediction_count": len(predictions),
        "invalid_row_count": missing,
        "duplicate_fact_id_count": duplicate,
        "parse_errors": artifact.errors,
        "schema": {"fact_id": "string", "correct": "boolean"},
    }


def _family_audit(
    episodes: LoadedArtifact, predictions_path: Path | None
) -> dict[str, Any]:
    if not episodes.path.is_file():
        return {
            "status": "not_evaluable",
            "reason": f"{IR_NAME} is missing",
            "families": [],
        }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_group_fields: Counter[str] = Counter()
    for episode, question in _iter_ir_questions(episodes.rows):
        group = question.get("consistency_group")
        if not group:
            continue
        program = question.get("program")
        member = {
            "fact_id": question.get("fact_id"),
            "variant": question.get("family_variant"),
            "task_type": question.get("task_type"),
            "split": episode.get("split"),
            "scene_id": episode.get("scene_id"),
            "question": question.get("question_zh"),
            "answer_status": question.get("answer_status"),
            "answer_value": question.get("answer_value"),
            "model_view_ids": question.get("model_view_ids"),
            "program_id": program.get("program_id") if isinstance(program, dict) else None,
            "semantic_signature": (
                program.get("semantic_signature") if isinstance(program, dict) else None
            ),
            "certificate": question.get("certificate"),
        }
        for field in ("fact_id", "variant", "task_type", "split", "scene_id"):
            if member[field] is None:
                missing_group_fields[field] += 1
        grouped[str(group)].append(member)

    predictions, prediction_summary = _load_predictions(predictions_path)
    families: list[dict[str, Any]] = []
    contract_counts: Counter[str] = Counter()
    complete_counts: Counter[str] = Counter()
    evidence_eligible: list[dict[str, Any]] = []
    actual_joint: list[dict[str, Any]] = []
    replacement_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    replacement_failures: list[dict[str, Any]] = []
    ordered_replacement_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    ordered_replacement_failures: list[dict[str, Any]] = []
    frame_pass_count = 0
    frame_failures: list[dict[str, Any]] = []

    for group, members in sorted(grouped.items()):
        variants_list = [member["variant"] for member in members if member["variant"] is not None]
        variants = set(variants_list)
        contract, expected = _family_contract(variants)
        if contract is None or expected is None:
            completeness_status = "not_evaluable"
            reasons = ["no registered family contract can be inferred from observed variants"]
        else:
            contract_counts[contract] += 1
            reasons = []
            if variants != set(expected):
                reasons.append(
                    f"expected variants {sorted(expected)}, observed {sorted(variants)}"
                )
            duplicates = sorted(
                variant for variant, count in Counter(variants_list).items() if count != 1
            )
            if duplicates:
                reasons.append(f"variants do not occur exactly once: {duplicates}")
            if len({member["split"] for member in members}) != 1:
                reasons.append("members cross split boundaries")
            if len({member["scene_id"] for member in members}) != 1:
                reasons.append("members cross scene boundaries")
            completeness_status = "complete" if not reasons else "incomplete"
            if completeness_status == "complete":
                complete_counts[contract] += 1

        evidence_checks: dict[str, Any] | None = None
        frame_checks: dict[str, Any] | None = None
        if contract == "evidence_intervention_triple":
            by_variant = {
                str(member["variant"]): member
                for member in members
                if member["variant"] is not None
            }
            evidence_reasons = list(reasons)
            replacement_check: dict[str, Any]
            if set(by_variant) == set(EVIDENCE_VARIANTS):
                questions = {member["question"] for member in by_variant.values()}
                if None in questions or len(questions) != 1:
                    evidence_reasons.append("natural-language question is not invariant")
                view_lists = [member["model_view_ids"] for member in by_variant.values()]
                if any(not isinstance(views, list) for views in view_lists):
                    evidence_reasons.append("model_view_ids is missing or not a list")
                elif len({len(views) for views in view_lists}) != 1:
                    evidence_reasons.append("family members expose different image counts")
                unknown_members = [
                    by_variant["prefix_unknown"],
                    by_variant["decisive_deleted"],
                ]
                if any(member["answer_status"] != "unknown" for member in unknown_members):
                    evidence_reasons.append("deletion siblings are not labeled unknown")
                if by_variant["revealed"]["answer_status"] == "unknown":
                    evidence_reasons.append("revealed sibling is still labeled unknown")
                targets = {
                    variant: _target_key(member["answer_status"], member["answer_value"])
                    for variant, member in by_variant.items()
                }
                if targets["revealed"] in {
                    targets["prefix_unknown"],
                    targets["decisive_deleted"],
                }:
                    evidence_reasons.append("revealed target does not change with evidence")
                replacement_check = _single_replacement_check(by_variant)
                task_signature = "+".join(
                    sorted({str(member["task_type"]) for member in by_variant.values()})
                )
                replacement_by_task[task_signature][
                    "passed" if replacement_check["passed"] else "failed"
                ] += 1
                ordered_replacement_by_task[task_signature][
                    "passed" if replacement_check.get("ordered_slot_stable") else "failed"
                ] += 1
                if replacement_check["passed"] and not replacement_check.get(
                    "ordered_slot_stable"
                ):
                    ordered_replacement_failures.append(
                        {
                            "consistency_group": group,
                            "task_signature": task_signature,
                            **replacement_check,
                        }
                    )
                if not replacement_check["passed"]:
                    evidence_reasons.append(
                        "visual intervention is not shared-view single-replacement matched"
                    )
                    replacement_failures.append(
                        {
                            "consistency_group": group,
                            "task_signature": task_signature,
                            **replacement_check,
                        }
                    )
            else:
                replacement_check = {
                    "status": "not_evaluable",
                    "passed": False,
                    "reason": "family does not contain the three expected variants",
                }
            structurally_evaluable = not evidence_reasons
            evidence_checks = {
                "structurally_evaluable": structurally_evaluable,
                "reasons": evidence_reasons,
                "shared_view_single_replacement": replacement_check,
                "required_metric_inputs": [
                    "boolean correctness for prefix_unknown",
                    "boolean correctness for revealed",
                    "boolean correctness for decisive_deleted",
                ],
            }
            if structurally_evaluable:
                evidence_eligible.append({"group": group, "members": by_variant})

        if contract == "frame_reference_pair":
            by_variant = {
                str(member["variant"]): member
                for member in members
                if member["variant"] is not None
            }
            frame_checks = _frame_pair_check(by_variant)
            if frame_checks["passed"]:
                frame_pass_count += 1
            else:
                frame_failures.append(
                    {
                        "consistency_group": group,
                        **frame_checks,
                    }
                )

        families.append(
            {
                "consistency_group": group,
                "contract": contract,
                "completeness_status": completeness_status,
                "reasons": reasons,
                "variants": sorted(variants),
                "member_count": len(members),
                "splits": sorted({str(member["split"]) for member in members}),
                "scene_ids": sorted({str(member["scene_id"]) for member in members}),
                "fact_ids": [member["fact_id"] for member in members],
                "evidence_sensitivity": evidence_checks,
                "frame_transform": frame_checks,
            }
        )

    conditioned_denominator = 0
    conditioned_numerator = 0
    fully_predicted = 0
    for eligible in evidence_eligible:
        by_variant = eligible["members"]
        fact_ids = {
            variant: member["fact_id"] for variant, member in by_variant.items()
        }
        if any(not isinstance(fact_id, str) or fact_id not in predictions for fact_id in fact_ids.values()):
            continue
        fully_predicted += 1
        correctness = {
            variant: predictions[fact_id] for variant, fact_id in fact_ids.items()
        }
        joint = all(correctness.values())
        if correctness["revealed"]:
            conditioned_denominator += 1
            conditioned_numerator += int(
                correctness["prefix_unknown"] and correctness["decisive_deleted"]
            )
        actual_joint.append(
            {
                "consistency_group": eligible["group"],
                "correctness": correctness,
                "joint_correct": joint,
            }
        )

    if fully_predicted == len(evidence_eligible) and evidence_eligible:
        metric_status = "evaluated"
    elif fully_predicted:
        metric_status = "partial"
    else:
        metric_status = "not_evaluated"
    if conditioned_denominator:
        aces_score = _safe_ratio(conditioned_numerator, conditioned_denominator)
    else:
        aces_score = None

    return {
        "status": "complete" if not missing_group_fields else "partial",
        "group_count": len(grouped),
        "missing_member_fields": dict(sorted(missing_group_fields.items())),
        "contract_counts": dict(sorted(contract_counts.items())),
        "complete_contract_counts": dict(sorted(complete_counts.items())),
        "unregistered_contract_count": sum(
            family["contract"] is None for family in families
        ),
        "families": families,
        "shared_view_single_replacement": {
            "definition": (
                "All three evidence-family inputs have equal length, share an N-1 image set, "
                "and each contributes one distinct replacement image. Ordered-slot stability "
                "is a separately reported stronger anti-shortcut condition."
            ),
            "evaluated_family_count": sum(
                sum(counts.values()) for counts in replacement_by_task.values()
            ),
            "pass_count": sum(counts["passed"] for counts in replacement_by_task.values()),
            "fail_count": sum(counts["failed"] for counts in replacement_by_task.values()),
            "all_passed": bool(replacement_by_task)
            and all(not counts["failed"] for counts in replacement_by_task.values()),
            "by_task_signature": {
                task: {
                    "family_count": sum(counts.values()),
                    "pass_count": counts["passed"],
                    "fail_count": counts["failed"],
                    "all_passed": counts["failed"] == 0,
                }
                for task, counts in sorted(replacement_by_task.items())
            },
            "failure_samples": replacement_failures[:50],
            "samples_truncated": len(replacement_failures) > 50,
            "ordered_slot_stability": {
                "pass_count": sum(
                    counts["passed"] for counts in ordered_replacement_by_task.values()
                ),
                "fail_count": sum(
                    counts["failed"] for counts in ordered_replacement_by_task.values()
                ),
                "all_passed": bool(ordered_replacement_by_task)
                and all(
                    not counts["failed"] for counts in ordered_replacement_by_task.values()
                ),
                "by_task_signature": {
                    task: {
                        "family_count": sum(counts.values()),
                        "pass_count": counts["passed"],
                        "fail_count": counts["failed"],
                        "all_passed": counts["failed"] == 0,
                    }
                    for task, counts in sorted(ordered_replacement_by_task.items())
                },
                "failure_samples": ordered_replacement_failures[:50],
                "samples_truncated": len(ordered_replacement_failures) > 50,
            },
        },
        "frame_same_input_answer_transform": {
            "definition": (
                "The pair receives the same ordered images and asks the same entity relation; "
                "only the explicitly named camera frame changes. The structured relation target "
                "must change, and each target/camera anchor must agree with its serialized "
                "frame-recomputation certificate."
            ),
            "evaluated_family_count": frame_pass_count + len(frame_failures),
            "pass_count": frame_pass_count,
            "fail_count": len(frame_failures),
            "all_passed": bool(frame_pass_count + len(frame_failures))
            and not frame_failures,
            "failure_samples": frame_failures[:50],
            "samples_truncated": len(frame_failures) > 50,
        },
        "accuracy_conditioned_evidence_sensitivity": {
            "definition": (
                "Among structurally valid evidence families for which the revealed member is "
                "answered correctly, measure the fraction whose two evidence-deleted members "
                "are also answered correctly (with the required unknown targets)."
            ),
            "structurally_evaluable_family_count": len(evidence_eligible),
            "prediction_input": prediction_summary,
            "metric_status": metric_status,
            "fully_predicted_family_count": fully_predicted,
            "conditioning_family_count": conditioned_denominator,
            "conditioned_success_count": conditioned_numerator,
            "score": aces_score,
            "family_joint_accuracy": (
                _safe_ratio(sum(item["joint_correct"] for item in actual_joint), len(actual_joint))
                if actual_joint
                else None
            ),
            "per_family_predictions": actual_joint,
        },
    }


def _cluster_stats(values: Iterable[str]) -> dict[str, Any]:
    counts = Counter(values)
    sizes = sorted(counts.values())
    if not sizes:
        return {
            "cluster_count": 0,
            "record_count": 0,
            "min_size": None,
            "median_size": None,
            "max_size": None,
            "mean_size": None,
        }
    return {
        "cluster_count": len(counts),
        "record_count": sum(sizes),
        "min_size": min(sizes),
        "median_size": statistics.median(sizes),
        "max_size": max(sizes),
        "mean_size": round(statistics.fmean(sizes), 6),
    }


def _split_overlap(cluster_to_splits: Mapping[str, set[str]]) -> dict[str, Any]:
    leaked = {
        cluster: sorted(splits)
        for cluster, splits in cluster_to_splits.items()
        if len(splits) > 1
    }
    return {
        "cross_split_cluster_count": len(leaked),
        "cross_split_clusters": dict(list(sorted(leaked.items()))[:50]),
        "truncated": len(leaked) > 50,
    }


def _cluster_audit(artifacts: Mapping[str, LoadedArtifact]) -> dict[str, Any]:
    ir = artifacts[IR_NAME]
    if not ir.path.is_file():
        return {"status": "not_evaluable", "reason": f"{IR_NAME} is missing"}

    fields = ("scene_id", "scene_family_id", "trajectory_family_id", "episode_id")
    overall: dict[str, Any] = {}
    per_split: dict[str, Any] = defaultdict(dict)
    overlap_maps: dict[str, dict[str, set[str]]] = {
        field: defaultdict(set) for field in fields[:-1]
    }
    missing: Counter[str] = Counter()
    for field in fields:
        present = [str(row[field]) for row in ir.rows if row.get(field) is not None]
        missing[field] = len(ir.rows) - len(present)
        overall[field] = _cluster_stats(present)
        split_values: dict[str, list[str]] = defaultdict(list)
        for row in ir.rows:
            if row.get("split") is None or row.get(field) is None:
                continue
            split = str(row["split"])
            value = str(row[field])
            split_values[split].append(value)
            if field in overlap_maps:
                overlap_maps[field][value].add(split)
        for split, values in split_values.items():
            per_split[split][field] = _cluster_stats(values)

    consistency_values: list[str] = []
    consistency_splits: dict[str, set[str]] = defaultdict(set)
    fact_values: list[str] = []
    for episode, question in _iter_ir_questions(ir.rows):
        group = question.get("consistency_group")
        if group:
            consistency_values.append(str(group))
            if episode.get("split") is not None:
                consistency_splits[str(group)].add(str(episode["split"]))
        fact_id = question.get("fact_id")
        if fact_id:
            fact_values.append(str(fact_id))
    overall["consistency_group"] = _cluster_stats(consistency_values)
    overall["fact_id"] = _cluster_stats(fact_values)

    export_family_by_split: dict[str, set[str]] = defaultdict(set)
    export_family_values: list[str] = []
    seen_export_records: set[str] = set()
    for name in (
        EPISODE_NAME,
        ISOLATED_NAME,
        STATE_AUX_NAME,
        *BENCHMARK_NAMES,
        RLVR_NAME,
    ):
        for row_index, row in enumerate(artifacts[name].rows):
            record_key = str(row.get("record_id", f"{name}:{row_index}"))
            if record_key in seen_export_records:
                continue
            seen_export_records.add(record_key)
            family_id = row.get("family_id")
            if not family_id:
                continue
            export_family_values.append(str(family_id))
            split = row.get("split")
            if split is not None:
                export_family_by_split[str(family_id)].add(str(split))
    overall["export_family_id"] = _cluster_stats(export_family_values)

    split_leakage = {
        field: _split_overlap(mapping) for field, mapping in overlap_maps.items()
    }
    split_leakage["consistency_group"] = _split_overlap(consistency_splits)
    split_leakage["export_family_id"] = _split_overlap(export_family_by_split)
    return {
        "status": "complete" if not any(missing.values()) else "partial",
        "episode_record_count": len(ir.rows),
        "missing_fields": {field: count for field, count in missing.items() if count},
        "overall": overall,
        "per_split": dict(sorted(per_split.items())),
        "split_leakage": split_leakage,
    }


def _message_texts(messages: Any, prefix: str) -> Iterator[tuple[str, str]]:
    if not isinstance(messages, list):
        return
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        location = f"{prefix}.messages[{message_index}]"
        if isinstance(content, str):
            yield location, content
        elif isinstance(content, list):
            for content_index, item in enumerate(content):
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(
                    item.get("text"), str
                ):
                    yield f"{location}.content[{content_index}]", item["text"]


def _raw_ontology_hits(text: str) -> list[str]:
    inventory = sorted(set(CATEGORY_ZH) | set(RELATION_ZH), key=lambda item: (-len(item), item))
    hits: set[str] = set()
    for token in inventory:
        surfaces = {token, token.replace("_", " ")}
        for surface in surfaces:
            if re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(surface)}(?![A-Za-z0-9_])",
                text,
                flags=re.IGNORECASE,
            ):
                hits.add(token)
                break
    return sorted(hits)


def _iter_strings(value: Any, prefix: str = "$") -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_strings(item, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_strings(item, f"{prefix}[{index}]")


def _ontology_audit(artifacts: Mapping[str, LoadedArtifact]) -> dict[str, Any]:
    visible: list[tuple[str, str]] = []
    ir = artifacts[IR_NAME]
    for episode_index, (_episode, question) in enumerate(_iter_ir_questions(ir.rows)):
        fact = question.get("fact_id", episode_index)
        for field in ("question_zh", "answer_zh", "rationale_zh"):
            text = question.get(field)
            if isinstance(text, str):
                visible.append((f"{IR_NAME}:{fact}.{field}", text))

    for name in (EPISODE_NAME, ISOLATED_NAME, STATE_AUX_NAME):
        for row_index, row in enumerate(artifacts[name].rows):
            record = row.get("record_id", row_index)
            visible.extend(_message_texts(row.get("messages"), f"{name}:{record}"))
    for name in BENCHMARK_NAMES:
        for row_index, row in enumerate(artifacts[name].rows):
            record = row.get("record_id", row_index)
            visible.extend(_message_texts(row.get("model_input"), f"{name}:{record}"))
    for row_index, row in enumerate(artifacts[RLVR_NAME].rows):
        record = row.get("record_id", row_index)
        visible.extend(_message_texts(row.get("prompt"), f"{RLVR_NAME}:{record}"))

    occurrences: list[dict[str, Any]] = []
    hit_counts: Counter[str] = Counter()
    hit_field_count = 0
    for location, text in visible:
        hits = _raw_ontology_hits(text)
        if not hits:
            continue
        hit_field_count += 1
        hit_counts.update(hits)
        if len(occurrences) < 100:
            occurrences.append(
                {
                    "location": location,
                    "tokens": hits,
                    "text_excerpt": text[:240],
                }
            )

    hidden_counts: Counter[str] = Counter()
    hidden_samples: list[dict[str, Any]] = []
    for _, question in _iter_ir_questions(ir.rows):
        for location, text in _iter_strings(question.get("answer_value"), "answer_value"):
            hits = _raw_ontology_hits(text)
            if hits:
                hidden_counts.update(hits)
                if len(hidden_samples) < 20:
                    hidden_samples.append(
                        {
                            "fact_id": question.get("fact_id"),
                            "location": location,
                            "tokens": hits,
                        }
                    )

    return {
        "status": "pass" if not hit_field_count else "fail",
        "definition": (
            "Exact raw category/relation tokens from the authoritative ontology are scanned in "
            "model-visible natural-language inputs and supervised outputs. Protocol tokens such "
            "as image, camera_height, and JSON field names are not ontology leakage."
        ),
        "scanned_text_field_count": len(visible),
        "model_visible_hit_field_count": hit_field_count,
        "model_visible_hit_counts_by_token": dict(sorted(hit_counts.items())),
        "model_visible_occurrence_samples": occurrences,
        "samples_truncated": hit_field_count > len(occurrences),
        "hidden_structured_target_tokens": {
            "interpretation": "allowed compiler/evaluator metadata; never shown as model input",
            "hit_counts_by_token": dict(sorted(hidden_counts.items())),
            "samples": hidden_samples,
        },
    }


def _message_images(messages: Any) -> Iterator[str]:
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if (
                isinstance(item, dict)
                and item.get("type") == "image"
                and isinstance(item.get("image"), str)
            ):
                yield item["image"]


def _collect_image_references(
    artifacts: Mapping[str, LoadedArtifact]
) -> dict[str, list[str]]:
    references: dict[str, list[str]] = defaultdict(list)
    for row in artifacts[IR_NAME].rows:
        observations = row.get("observations")
        if not isinstance(observations, list):
            continue
        for observation in observations:
            if isinstance(observation, dict) and isinstance(observation.get("rgb"), str):
                references[IR_NAME].append(observation["rgb"])
    for name in (EPISODE_NAME, ISOLATED_NAME, STATE_AUX_NAME):
        for row in artifacts[name].rows:
            references[name].extend(_message_images(row.get("messages")))
    for name in BENCHMARK_NAMES:
        for row in artifacts[name].rows:
            references[name].extend(_message_images(row.get("model_input")))
    for row in artifacts[RLVR_NAME].rows:
        references[RLVR_NAME].extend(_message_images(row.get("prompt")))
    return dict(references)


def _collect_model_input_image_references(
    artifacts: Mapping[str, LoadedArtifact],
) -> dict[str, list[str]]:
    """Collect only images that are serialized into an actual model input.

    In particular, IR observations are compiler/evaluator evidence.  They are
    intentionally outside this gate unless an exporter also references the
    image from a training prompt or benchmark ``model_input``.
    """

    references = {name: [] for name in MODEL_INPUT_ARTIFACT_NAMES}
    for name in (EPISODE_NAME, ISOLATED_NAME, STATE_AUX_NAME):
        for row in artifacts[name].rows:
            references[name].extend(_message_images(row.get("messages")))
    for row in artifacts[RLVR_NAME].rows:
        references[RLVR_NAME].extend(_message_images(row.get("prompt")))
    for name in BENCHMARK_NAMES:
        for row in artifacts[name].rows:
            references[name].extend(_message_images(row.get("model_input")))
    return references


def _resolve_image_path(raw_path: str, release_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (release_dir / path).resolve()


def _model_input_rgb_policy(manifest_path: Path) -> dict[str, Any]:
    default = {
        "maximum_rgb_dominant_color_fraction": (
            FROZEN_MAX_RGB_DOMINANT_COLOR_FRACTION
        ),
        "minimum_rgb_quantized_entropy_bits": (
            FROZEN_MIN_RGB_QUANTIZED_ENTROPY_BITS
        ),
        "source": "frozen_default",
        "policy_id": "epispace.corpus_audit.frozen_rgb_gate.v1",
    }
    if not manifest_path.is_file():
        return {**default, "fallback_reason": "release_manifest.json is missing"}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {**default, "fallback_reason": f"release manifest is unreadable: {exc}"}
    if not isinstance(manifest, dict):
        return {
            **default,
            "fallback_reason": "release manifest root is not a JSON object",
        }
    research_contract = manifest.get("research_contract")
    policy = (
        research_contract.get("visual_quality_policy")
        if isinstance(research_contract, dict)
        else None
    )
    if not isinstance(policy, dict):
        return {
            **default,
            "fallback_reason": (
                "research_contract.visual_quality_policy is missing"
            ),
        }
    maximum = policy.get("maximum_rgb_dominant_color_fraction")
    minimum = policy.get("minimum_rgb_quantized_entropy_bits")
    valid_maximum = (
        isinstance(maximum, int | float)
        and not isinstance(maximum, bool)
        and math.isfinite(float(maximum))
        and 0.0 <= float(maximum) <= 1.0
    )
    valid_minimum = (
        isinstance(minimum, int | float)
        and not isinstance(minimum, bool)
        and math.isfinite(float(minimum))
        and float(minimum) >= 0.0
    )
    if not valid_maximum or not valid_minimum:
        return {
            **default,
            "fallback_reason": (
                "release visual-quality RGB thresholds are missing or invalid"
            ),
        }
    return {
        "maximum_rgb_dominant_color_fraction": float(maximum),
        "minimum_rgb_quantized_entropy_bits": float(minimum),
        "source": "release_manifest.research_contract.visual_quality_policy",
        "policy_id": policy.get("policy_id"),
    }


def _quantized_rgb_statistics(path: Path) -> dict[str, float | int]:
    """Independently recompute the frozen 16-bin-per-channel RGB statistics."""

    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"expected exact RGB input, got {image.mode}")
        rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise ValueError(f"invalid RGB array shape: {rgb.shape}")
    quantized = rgb.astype(np.uint16) // 16
    encoded = (
        quantized[:, :, 0] * 256
        + quantized[:, :, 1] * 16
        + quantized[:, :, 2]
    )
    counts = np.bincount(encoded.reshape(-1), minlength=4096)
    probabilities = counts[counts > 0].astype(np.float64) / encoded.size
    entropy_bits = float(-np.sum(probabilities * np.log2(probabilities)))
    return {
        "pixel_count": int(encoded.size),
        "dominant_quantized_color_fraction": float(counts.max() / encoded.size),
        "quantized_color_entropy_bits": entropy_bits,
    }


def _model_input_rgb_audit(
    artifacts: Mapping[str, LoadedArtifact], release_dir: Path
) -> dict[str, Any]:
    references = _collect_model_input_image_references(artifacts)
    manifest_path = release_dir / "release_manifest.json"
    policy = _model_input_rgb_policy(manifest_path)
    maximum_dominant = float(policy["maximum_rgb_dominant_color_fraction"])
    minimum_entropy = float(policy["minimum_rgb_quantized_entropy_bits"])

    resolved_for_raw: dict[str, str] = {}
    raw_paths_for_resolved: dict[str, set[str]] = defaultdict(set)
    occurrences_for_resolved: Counter[str] = Counter()
    artifacts_for_resolved: dict[str, set[str]] = defaultdict(set)
    for artifact_name, paths in references.items():
        for raw_path in paths:
            resolved = str(_resolve_image_path(raw_path, release_dir).resolve())
            resolved_for_raw[raw_path] = resolved
            raw_paths_for_resolved[resolved].add(raw_path)
            occurrences_for_resolved[resolved] += 1
            artifacts_for_resolved[resolved].add(artifact_name)

    # Cache by resolved filesystem path so repeated exposures and path aliases
    # never trigger repeated image decoding or histogram computation.
    measurements: dict[str, dict[str, float | int]] = {}
    errors: list[dict[str, str]] = []
    for resolved in sorted(raw_paths_for_resolved):
        path = Path(resolved)
        try:
            measurements[resolved] = _quantized_rgb_statistics(path)
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            errors.append({"resolved_path": resolved, "error": str(exc)})

    hard_degenerate = {
        resolved
        for resolved, measurement in measurements.items()
        if float(measurement["dominant_quantized_color_fraction"])
        >= maximum_dominant
        and float(measurement["quantized_color_entropy_bits"]) < minimum_entropy
    }
    by_artifact: dict[str, dict[str, int]] = {}
    for artifact_name in MODEL_INPUT_ARTIFACT_NAMES:
        paths = references[artifact_name]
        resolved_paths = [resolved_for_raw[path] for path in paths]
        bad_paths = [path for path in resolved_paths if path in hard_degenerate]
        by_artifact[artifact_name] = {
            "reference_count": len(paths),
            "unique_path_count": len(set(resolved_paths)),
            "hard_degenerate_occurrence_count": len(bad_paths),
            "hard_degenerate_unique_path_count": len(set(bad_paths)),
        }

    samples: list[dict[str, Any]] = []
    for resolved in sorted(hard_degenerate)[:50]:
        measurement = measurements[resolved]
        samples.append(
            {
                "path": sorted(raw_paths_for_resolved[resolved])[0],
                "path_aliases": sorted(raw_paths_for_resolved[resolved]),
                "resolved_path": resolved,
                "artifacts": sorted(artifacts_for_resolved[resolved]),
                "occurrence_count": occurrences_for_resolved[resolved],
                "dominant_quantized_color_fraction": round(
                    float(measurement["dominant_quantized_color_fraction"]), 9
                ),
                "quantized_color_entropy_bits": round(
                    float(measurement["quantized_color_entropy_bits"]), 9
                ),
                "pixel_count": measurement["pixel_count"],
            }
        )

    total_occurrences = sum(len(paths) for paths in references.values())
    bad_occurrences = sum(occurrences_for_resolved[path] for path in hard_degenerate)
    return {
        "status": "fail" if hard_degenerate else ("partial" if errors else "pass"),
        "definition": (
            "Independent 16-bin-per-channel statistics over exact exported RGB files. "
            "An actual model input is hard-degenerate iff dominant fraction >= maximum "
            "and quantized entropy < minimum. IR observations are excluded unless an "
            "exported model input also references them."
        ),
        "scope_artifacts": list(MODEL_INPUT_ARTIFACT_NAMES),
        "excluded_artifacts": [IR_NAME],
        "policy": policy,
        "model_input_reference_count": total_occurrences,
        "model_input_unique_path_count": len(raw_paths_for_resolved),
        "measured_unique_path_count": len(measurements),
        "measurement_error_count": len(errors),
        "measurement_error_samples": errors[:50],
        "hard_degenerate_occurrence_count": bad_occurrences,
        "hard_degenerate_unique_path_count": len(hard_degenerate),
        "hard_degenerate_samples": samples,
        "samples_truncated": len(hard_degenerate) > len(samples),
        "by_artifact": by_artifact,
    }


def _image_audit(
    artifacts: Mapping[str, LoadedArtifact], release_dir: Path
) -> tuple[dict[str, Any], dict[str, tuple[int, int]]]:
    references = _collect_image_references(artifacts)
    all_references = [path for paths in references.values() for path in paths]
    unique_raw = sorted(set(all_references))
    dimensions: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    missing: list[str] = []
    unreadable: list[dict[str, str]] = []
    resolved_dimensions: dict[str, tuple[int, int]] = {}
    absolute_count = 0
    outside_release_count = 0
    release_resolved = release_dir.resolve()

    for raw_path in unique_raw:
        if Path(raw_path).is_absolute():
            absolute_count += 1
        resolved = _resolve_image_path(raw_path, release_dir)
        try:
            resolved.relative_to(release_resolved)
        except ValueError:
            outside_release_count += 1
        if not resolved.is_file():
            missing.append(raw_path)
            continue
        try:
            with Image.open(resolved) as image:
                width, height = image.size
                mode = image.mode
                image_format = image.format or "unknown"
                image.verify()
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            unreadable.append({"path": raw_path, "error": str(exc)})
            continue
        resolved_dimensions[raw_path] = (width, height)
        dimensions[f"{width}x{height}"] += 1
        modes[mode] += 1
        formats[image_format] += 1

    artifact_counts = {
        name: {
            "reference_count": len(paths),
            "unique_path_count": len(set(paths)),
        }
        for name, paths in sorted(references.items())
    }
    bad_modes = {mode: count for mode, count in modes.items() if mode != "RGB"}
    status = "pass" if not missing and not unreadable and not bad_modes else "fail"
    return (
        {
            "status": status,
            "reference_count": len(all_references),
            "unique_path_count": len(unique_raw),
            "existing_readable_unique_count": len(resolved_dimensions),
            "absolute_path_count": absolute_count,
            "relative_path_count": len(unique_raw) - absolute_count,
            "outside_release_directory_count": outside_release_count,
            "portability_note": (
                "Absolute paths can be valid on this host but require path remapping when the "
                "release is moved."
            ),
            "dimensions": dict(sorted(dimensions.items())),
            "uniform_dimensions": len(dimensions) == 1 and bool(dimensions),
            "modes": dict(sorted(modes.items())),
            "formats": dict(sorted(formats.items())),
            "non_rgb_modes": bad_modes,
            "missing_path_count": len(missing),
            "missing_path_samples": missing[:50],
            "unreadable_path_count": len(unreadable),
            "unreadable_path_samples": unreadable[:50],
            "by_artifact": artifact_counts,
        },
        resolved_dimensions,
    )


def _training_record_images(row: Mapping[str, Any]) -> list[str]:
    return list(_message_images(row.get("messages")))


def _pixel_exposure(
    paths: Sequence[str], dimensions: Mapping[str, tuple[int, int]]
) -> int | None:
    if any(path not in dimensions for path in paths):
        return None
    return sum(dimensions[path][0] * dimensions[path][1] for path in paths)


def _arm_comparison_audit(
    episode_artifact: LoadedArtifact,
    isolated_artifact: LoadedArtifact,
    dimensions: Mapping[str, tuple[int, int]],
) -> dict[str, Any]:
    if not episode_artifact.path.is_file() or not isolated_artifact.path.is_file():
        missing = [
            artifact.name
            for artifact in (episode_artifact, isolated_artifact)
            if not artifact.path.is_file()
        ]
        return {
            "status": "not_evaluable",
            "reason": f"missing training arm artifacts: {missing}",
        }

    groups: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: {"episode": [], "isolated": []}
    )
    missing_contract: Counter[str] = Counter()
    for arm, artifact in (
        ("episode", episode_artifact),
        ("isolated", isolated_artifact),
    ):
        for row in artifact.rows:
            contract = row.get("comparison_contract")
            if not isinstance(contract, dict):
                missing_contract[f"{arm}.comparison_contract"] += 1
                continue
            comparison_id = contract.get("comparison_id")
            if not isinstance(comparison_id, str):
                missing_contract[f"{arm}.comparison_id"] += 1
                continue
            groups[comparison_id][arm].append(row)

    comparisons: list[dict[str, Any]] = []
    global_episode_facts: Counter[str] = Counter()
    global_isolated_facts: Counter[str] = Counter()
    episode_refs_total: list[str] = []
    isolated_refs_total: list[str] = []
    for comparison_id, arms in sorted(groups.items()):
        episode_rows = arms["episode"]
        isolated_rows = arms["isolated"]
        episode_facts: list[str] = []
        isolated_facts: list[str] = []
        declared_errors: list[str] = []
        for arm, rows, facts in (
            ("episode", episode_rows, episode_facts),
            ("isolated", isolated_rows, isolated_facts),
        ):
            for row in rows:
                contract = row.get("comparison_contract", {})
                raw_facts = contract.get("fact_ids")
                if not isinstance(raw_facts, list) or not all(
                    isinstance(item, str) for item in raw_facts
                ):
                    declared_errors.append(f"{arm} row has invalid fact_ids")
                    continue
                facts.extend(raw_facts)
                actual_image_count = len(set(_training_record_images(row)))
                if contract.get("unique_image_count") != actual_image_count:
                    declared_errors.append(
                        f"{arm} unique_image_count declaration differs from serialized images"
                    )
                if contract.get("question_count") != len(raw_facts):
                    declared_errors.append(
                        f"{arm} question_count declaration differs from fact_ids count"
                    )
        episode_counter = Counter(episode_facts)
        isolated_counter = Counter(isolated_facts)
        global_episode_facts.update(episode_counter)
        global_isolated_facts.update(isolated_counter)
        episode_images = [
            path for row in episode_rows for path in _training_record_images(row)
        ]
        isolated_images = [
            path for row in isolated_rows for path in _training_record_images(row)
        ]
        episode_refs_total.extend(episode_images)
        isolated_refs_total.extend(isolated_images)
        episode_pixels = _pixel_exposure(episode_images, dimensions)
        isolated_pixels = _pixel_exposure(isolated_images, dimensions)
        comparisons.append(
            {
                "comparison_id": comparison_id,
                "episode_record_count": len(episode_rows),
                "isolated_record_count": len(isolated_rows),
                "episode_fact_count": len(episode_facts),
                "isolated_fact_count": len(isolated_facts),
                "fact_multiset_equal": episode_counter == isolated_counter,
                "episode_image_reference_count": len(episode_images),
                "isolated_image_reference_count": len(isolated_images),
                "episode_unique_image_count": len(set(episode_images)),
                "isolated_unique_image_count": len(set(isolated_images)),
                "image_set_equal": set(episode_images) == set(isolated_images),
                "episode_pixel_exposure": episode_pixels,
                "isolated_pixel_exposure": isolated_pixels,
                "isolated_to_episode_reference_ratio": _safe_ratio(
                    len(isolated_images), len(episode_images)
                ),
                "isolated_to_episode_pixel_ratio": (
                    round(isolated_pixels / episode_pixels, 6)
                    if isolated_pixels is not None and episode_pixels
                    else None
                ),
                "declaration_errors": sorted(set(declared_errors)),
            }
        )

    episode_pixels_total = _pixel_exposure(episode_refs_total, dimensions)
    isolated_pixels_total = _pixel_exposure(isolated_refs_total, dimensions)
    pair_complete = sum(
        bool(item["episode_record_count"] and item["isolated_record_count"])
        for item in comparisons
    )
    fact_equal = sum(item["fact_multiset_equal"] for item in comparisons)
    image_set_equal = sum(item["image_set_equal"] for item in comparisons)
    declaration_error_count = sum(bool(item["declaration_errors"]) for item in comparisons)
    all_valid = (
        not missing_contract
        and comparisons
        and pair_complete == len(comparisons)
        and fact_equal == len(comparisons)
        and image_set_equal == len(comparisons)
        and not declaration_error_count
    )
    return {
        "status": "pass" if all_valid else "fail",
        "comparison_count": len(comparisons),
        "paired_comparison_count": pair_complete,
        "fact_multiset_equal_comparison_count": fact_equal,
        "image_set_equal_comparison_count": image_set_equal,
        "declaration_error_comparison_count": declaration_error_count,
        "missing_contract_fields": dict(sorted(missing_contract.items())),
        "global_fact_multiset_equal": global_episode_facts == global_isolated_facts,
        "global_episode_fact_count": sum(global_episode_facts.values()),
        "global_isolated_fact_count": sum(global_isolated_facts.values()),
        "global_unique_fact_count": len(global_episode_facts | global_isolated_facts),
        "image_exposure": {
            "episode_reference_count": len(episode_refs_total),
            "isolated_reference_count": len(isolated_refs_total),
            "isolated_to_episode_reference_ratio": _safe_ratio(
                len(isolated_refs_total), len(episode_refs_total)
            ),
            "episode_unique_image_count": len(set(episode_refs_total)),
            "isolated_unique_image_count": len(set(isolated_refs_total)),
            "episode_pixel_exposure": episode_pixels_total,
            "isolated_pixel_exposure": isolated_pixels_total,
            "isolated_to_episode_pixel_ratio": (
                round(isolated_pixels_total / episode_pixels_total, 6)
                if isolated_pixels_total is not None and episode_pixels_total
                else None
            ),
            "compute_match_status": (
                "matched"
                if Counter(episode_refs_total) == Counter(isolated_refs_total)
                else "not_matched_in_raw_exports"
            ),
            "interpretation": (
                "Fact multisets and image sets test supervision comparability. Serialized image "
                "reference/pixel exposure measures raw input compute; a sampler is still required "
                "when these exposures differ."
            ),
        },
        "mismatch_samples": [
            item
            for item in comparisons
            if not item["fact_multiset_equal"]
            or not item["image_set_equal"]
            or item["declaration_errors"]
            or not item["episode_record_count"]
            or not item["isolated_record_count"]
        ][:50],
        "comparisons": comparisons,
    }


def _artifact_schema_audit(artifacts: Mapping[str, LoadedArtifact]) -> dict[str, Any]:
    benchmark_requirements = (
        "record_id",
        "family_id",
        "scene_id",
        "split",
        "model_input",
        "target",
        "program",
    )
    requirements = {
        IR_NAME: (
            "episode_id",
            "scene_id",
            "scene_family_id",
            "trajectory_family_id",
            "split",
            "observations",
            "questions",
        ),
        EPISODE_NAME: (
            "record_id",
            "family_id",
            "scene_id",
            "split",
            "messages",
            "comparison_contract",
        ),
        ISOLATED_NAME: (
            "record_id",
            "family_id",
            "scene_id",
            "split",
            "messages",
            "comparison_contract",
        ),
        STATE_AUX_NAME: ("record_id", "family_id", "scene_id", "split", "messages"),
        RLVR_NAME: ("record_id", "family_id", "scene_id", "prompt", "reward_spec"),
        **{name: benchmark_requirements for name in BENCHMARK_NAMES},
    }
    result: dict[str, Any] = {}
    for name, required in requirements.items():
        artifact = artifacts[name]
        result[name] = {
            "path": str(artifact.path),
            "exists": artifact.path.is_file(),
            "parse_errors": artifact.errors,
            **_missing_field_summary(artifact.rows, required),
        }
    return result


def _overall_quality(sections: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    if sections["ontology_leakage"]["status"] == "fail":
        failures.append("model-visible raw English ontology tokens found")
    if sections["images"]["status"] == "fail":
        failures.append("one or more referenced images are missing, unreadable, or non-RGB")
    rgb_gate = sections["model_input_rgb_quality"]
    if rgb_gate["status"] == "fail":
        failures.append(
            f"{rgb_gate['hard_degenerate_occurrence_count']} actual model-input image "
            "occurrences fail the exported-RGB hard-degeneration gate"
        )
    elif rgb_gate["status"] == "partial":
        warnings.append("model-input RGB degeneration audit is partial")
    if sections["training_arm_comparison"]["status"] == "fail":
        failures.append("episode/isolated pairing or declared matching contract failed")
    target_status = sections["targets_and_baselines"]["status"]
    if target_status != "complete":
        warnings.append(f"target audit is {target_status}")
    family_section = sections["families"]
    incomplete = sum(
        family["completeness_status"] == "incomplete"
        for family in family_section.get("families", [])
    )
    if incomplete:
        failures.append(f"{incomplete} registered families are incomplete")
    unregistered = family_section.get("unregistered_contract_count", 0)
    if unregistered:
        warnings.append(f"{unregistered} consistency groups have no registered audit contract")
    replacement = family_section.get("shared_view_single_replacement", {})
    if replacement.get("fail_count"):
        warnings.append(
            f"{replacement['fail_count']} evidence families are not shared-view "
            "single-replacement matched"
        )
    ordered_stability = replacement.get("ordered_slot_stability", {})
    if ordered_stability.get("fail_count"):
        warnings.append(
            f"{ordered_stability['fail_count']} set-matched evidence families reorder a "
            "shared image across input slots"
        )
    frame_transform = family_section.get("frame_same_input_answer_transform", {})
    if frame_transform.get("fail_count"):
        failures.append(
            f"{frame_transform['fail_count']} frame families fail the same-input "
            "answer-transform contract"
        )
    metric = family_section.get("accuracy_conditioned_evidence_sensitivity", {})
    if metric.get("metric_status") == "not_evaluated":
        warnings.append("evidence-sensitivity model score not evaluated (predictions absent)")
    schema_partial = [
        name
        for name, value in sections["artifact_schema"].items()
        if not value["exists"] or value["parse_errors"] or not value["complete"]
    ]
    if schema_partial:
        failures.append(f"missing/malformed required artifact schema: {schema_partial}")
    return {
        "status": "fail" if failures else ("warn" if warnings else "pass"),
        "failures": failures,
        "warnings": warnings,
        "interpretation": (
            "This status describes serialized corpus integrity and shortcut diagnostics; it is "
            "independent of release_manifest.json and is not evidence of model quality."
        ),
    }


def audit_release(
    release_dir: Path | str, *, predictions_path: Path | str | None = None
) -> dict[str, Any]:
    """Audit a release directory and return a JSON-serializable report."""

    root = Path(release_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"release directory does not exist: {root}")
    artifacts = {
        name: _read_jsonl(root / name)
        for name in (
            IR_NAME,
            EPISODE_NAME,
            ISOLATED_NAME,
            STATE_AUX_NAME,
            RLVR_NAME,
            *BENCHMARK_NAMES,
        )
    }
    prediction_path = Path(predictions_path).expanduser().resolve() if predictions_path else None
    image_report, dimensions = _image_audit(artifacts, root)
    model_input_rgb_report = _model_input_rgb_audit(artifacts, root)
    target_report = _target_and_baseline_audit(artifacts[IR_NAME])
    target_report["benchmark_partitions"] = _benchmark_partition_target_audit(artifacts)
    sections: dict[str, Any] = {
        "artifact_schema": _artifact_schema_audit(artifacts),
        "targets_and_baselines": target_report,
        "families": _family_audit(artifacts[IR_NAME], prediction_path),
        "clusters": _cluster_audit(artifacts),
        "ontology_leakage": _ontology_audit(artifacts),
        "images": image_report,
        "model_input_rgb_quality": model_input_rgb_report,
        "training_arm_comparison": _arm_comparison_audit(
            artifacts[EPISODE_NAME], artifacts[ISOLATED_NAME], dimensions
        ),
    }
    manifest_path = root / "release_manifest.json"
    manifest_sha256 = None
    if manifest_path.is_file():
        digest = hashlib.sha256()
        with manifest_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest_sha256 = digest.hexdigest()
    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "release_dir": str(root),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": manifest_sha256,
        "sections": sections,
    }
    report["summary"] = _overall_quality(sections)
    return report


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    def cell(value: Any) -> str:
        if value is None:
            return "—"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the compact, human-facing view of a full JSON audit."""

    sections = report["sections"]
    summary = report["summary"]
    lines = [
        "# EpiSpace corpus audit",
        "",
        f"- Release: `{report['release_dir']}`",
        f"- Generated: `{report['generated_at']}`",
        f"- Independent audit status: **{summary['status']}**",
        "- This audit reads exported files only; it does not trust or rerun the builder.",
        "",
    ]
    if summary["failures"]:
        lines.extend(["## Failures", ""])
        lines.extend(f"- {item}" for item in summary["failures"])
        lines.append("")
    if summary["warnings"]:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {item}" for item in summary["warnings"])
        lines.append("")

    target = sections["targets_and_baselines"]
    lines.extend(["## Structured targets and no-image priors", ""])
    lines.extend(
        _markdown_table(
            ("Split", "Task", "N", "Targets", "Oracle majority", "Majority target"),
            (
                (
                    group["split"],
                    group["task_type"],
                    group["count"],
                    group["unique_target_count"],
                    group["within_split_majority"]["accuracy"],
                    _display_target(group["within_split_majority"]["labels"][0])
                    if group["within_split_majority"]["labels"]
                    else None,
                )
                for group in target.get("groups", [])
            ),
        )
    )
    lines.extend(
        [
            "",
            "`Oracle majority` is a same-split task-conditioned label-prior upper bound, not a "
            "VLM result. Train-fitted evaluation is reported separately below.",
            "",
        ]
    )
    train_eval = target.get("task_conditioned_majority", {}).get("train_fitted", {}).get(
        "evaluation", {}
    )
    lines.extend(
        _markdown_table(
            ("Evaluation split", "Coverage", "Accuracy on covered", "Evaluated / total"),
            (
                (
                    split,
                    value.get("coverage"),
                    value.get("accuracy_on_covered"),
                    f"{value.get('evaluated_count', 0)} / {value.get('total_count', 0)}",
                )
                for split, value in sorted(train_eval.items())
            ),
        )
    )
    lines.append("")
    partitions = target.get("benchmark_partitions", {})
    lines.extend(["### Published benchmark partitions", ""])
    lines.extend(
        _markdown_table(
            ("Artifact", "Records", "Valid targets", "Task-prior oracle", "Status"),
            (
                (
                    name,
                    value.get("record_count"),
                    value.get("valid_target_count"),
                    value.get("task_conditioned_within_split_majority_accuracy"),
                    value.get("status"),
                )
                for name, value in partitions.items()
            ),
        )
    )
    lines.append("")

    family = sections["families"]
    evidence = family.get("accuracy_conditioned_evidence_sensitivity", {})
    replacement = family.get("shared_view_single_replacement", {})
    ordered_stability = replacement.get("ordered_slot_stability", {})
    frame_transform = family.get("frame_same_input_answer_transform", {})
    lines.extend(
        [
            "## Counterfactual families",
            "",
            f"- Consistency groups: **{family.get('group_count', 0)}**",
            f"- Registered contracts: `{json.dumps(family.get('contract_counts', {}), ensure_ascii=False)}`",
            f"- Complete contracts: `{json.dumps(family.get('complete_contract_counts', {}), ensure_ascii=False)}`",
            f"- Shared-view single-replacement: **{replacement.get('pass_count', 0)} / {replacement.get('evaluated_family_count', 0)}** families pass.",
            f"- By task signature: `{json.dumps(replacement.get('by_task_signature', {}), ensure_ascii=False)}`",
            f"- Stronger ordered-slot stability: **{ordered_stability.get('pass_count', 0)} / {replacement.get('evaluated_family_count', 0)}** families pass.",
            f"- Same-input frame answer-transform: **{frame_transform.get('pass_count', 0)} / {frame_transform.get('evaluated_family_count', 0)}** families pass.",
            f"- Structurally evaluable evidence families: **{evidence.get('structurally_evaluable_family_count', 0)}**",
            f"- Accuracy-conditioned evidence sensitivity: **{evidence.get('metric_status', 'not_evaluable')}**, score = `{evidence.get('score')}`",
            "- A score is intentionally absent unless per-fact model correctness is supplied via `--predictions`.",
            "",
        ]
    )

    clusters = sections["clusters"]
    lines.extend(["## Scene and family clusters", ""])
    lines.extend(
        _markdown_table(
            ("Cluster key", "Clusters", "Records", "Min", "Median", "Max"),
            (
                (
                    key,
                    value.get("cluster_count"),
                    value.get("record_count"),
                    value.get("min_size"),
                    value.get("median_size"),
                    value.get("max_size"),
                )
                for key, value in clusters.get("overall", {}).items()
            ),
        )
    )
    leakage_counts = {
        key: value["cross_split_cluster_count"]
        for key, value in clusters.get("split_leakage", {}).items()
    }
    lines.extend(
        [
            "",
            f"Cross-split cluster counts: `{json.dumps(leakage_counts, ensure_ascii=False)}`",
            "",
        ]
    )

    ontology = sections["ontology_leakage"]
    images = sections["images"]
    rgb_gate = sections["model_input_rgb_quality"]
    rgb_policy = rgb_gate["policy"]
    lines.extend(
        [
            "## Ontology and image integrity",
            "",
            f"- English ontology leakage: **{ontology['status']}**; visible hit fields = {ontology['model_visible_hit_field_count']}.",
            f"- Images: **{images['status']}**; {images['existing_readable_unique_count']} / {images['unique_path_count']} unique paths readable.",
            f"- Dimension distribution: `{json.dumps(images['dimensions'], ensure_ascii=False)}`",
            f"- Mode distribution: `{json.dumps(images['modes'], ensure_ascii=False)}`",
            f"- Absolute paths: {images['absolute_path_count']}; paths outside release directory: {images['outside_release_directory_count']}.",
            f"- Actual model-input RGB gate: **{rgb_gate['status']}**; hard-degenerate occurrences = {rgb_gate['hard_degenerate_occurrence_count']}, unique files = {rgb_gate['hard_degenerate_unique_path_count']}.",
            f"- RGB gate thresholds: dominant >= {rgb_policy['maximum_rgb_dominant_color_fraction']} AND entropy < {rgb_policy['minimum_rgb_quantized_entropy_bits']} bits; source = `{rgb_policy['source']}`.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ("Model-input artifact", "References", "Unique", "Bad occurrences", "Bad unique"),
            (
                (
                    name,
                    value["reference_count"],
                    value["unique_path_count"],
                    value["hard_degenerate_occurrence_count"],
                    value["hard_degenerate_unique_path_count"],
                )
                for name, value in rgb_gate["by_artifact"].items()
            ),
        )
    )
    lines.append("")
    if rgb_gate["hard_degenerate_samples"]:
        lines.extend(["Hard-degenerate model-input samples:", ""])
        for sample in rgb_gate["hard_degenerate_samples"]:
            lines.append(
                "- `{path}`: dominant={dominant}, entropy={entropy} bits, "
                "occurrences={occurrences}, artifacts={artifacts}".format(
                    path=sample["path"],
                    dominant=sample["dominant_quantized_color_fraction"],
                    entropy=sample["quantized_color_entropy_bits"],
                    occurrences=sample["occurrence_count"],
                    artifacts=", ".join(sample["artifacts"]),
                )
            )
        lines.append("")

    arms = sections["training_arm_comparison"]
    exposure = arms.get("image_exposure", {})
    lines.extend(
        [
            "## Episode versus isolated control",
            "",
            f"- Pair audit: **{arms.get('status', 'not_evaluable')}**; comparisons = {arms.get('comparison_count', 0)}.",
            f"- Global fact multiset equal: `{arms.get('global_fact_multiset_equal')}` ({arms.get('global_episode_fact_count')} vs {arms.get('global_isolated_fact_count')}).",
            f"- Image references: episode = {exposure.get('episode_reference_count')}, isolated = {exposure.get('isolated_reference_count')}, ratio = {exposure.get('isolated_to_episode_reference_ratio')}.",
            f"- Pixel exposure ratio (isolated / episode): `{exposure.get('isolated_to_episode_pixel_ratio')}`.",
            f"- Raw-export compute match: **{exposure.get('compute_match_status', 'not_evaluable')}**.",
            "",
            "Full target distributions, per-family diagnostics, per-comparison exposure, missing fields, and path samples are retained in the JSON report.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    report: Mapping[str, Any], *, json_path: Path, markdown_path: Path
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a serialized EpiSpace release without invoking its build pipeline."
    )
    parser.add_argument("release_dir", type=Path, help="directory containing exported JSONL files")
    parser.add_argument(
        "--predictions",
        type=Path,
        help="optional JSONL with {fact_id: string, correct: boolean}",
    )
    parser.add_argument("--json", type=Path, help="JSON output path")
    parser.add_argument("--markdown", type=Path, help="Markdown output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    release_dir = args.release_dir.expanduser().resolve()
    report = audit_release(release_dir, predictions_path=args.predictions)
    json_path = args.json or release_dir / "corpus_audit.json"
    markdown_path = args.markdown or release_dir / "corpus_audit.md"
    write_report(report, json_path=json_path, markdown_path=markdown_path)
    print(
        json.dumps(
            {
                "status": report["summary"]["status"],
                "json": str(json_path),
                "markdown": str(markdown_path),
            },
            ensure_ascii=False,
        )
    )
    return 1 if report["summary"]["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
