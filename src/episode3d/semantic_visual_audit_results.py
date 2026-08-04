"""Fail-closed adjudication for independent semantic visual review files.

The packet builder intentionally leaves every review field empty.  This module
is the separate closure step: it verifies the immutable packet evidence, binds
independent reviewer files by SHA-256, checks exact item coverage, and emits an
explicit pass/fail gate.  It never upgrades a missing, malformed, or internally
inconsistent review into a passing result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from episode3d.semantic_visual_audit import PACKET_SCHEMA

RESULT_SCHEMA = "epispace.semantic_visual_audit_result.v1"
OPTIONAL_REVIEW_SCHEMA = "epispace.semantic_visual_audit_review.v1"
ALLOWED_REVIEWER_TYPES = frozenset(
    {"model_assisted_independent", "human_independent"}
)
ALLOWED_OVERALL_STATUSES = frozenset(
    {"pass", "minor_issue", "major_issue", "unreviewable"}
)
ALLOWED_SEVERITIES = frozenset({"none", "minor", "major"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVIEW_INDEX_CONTRACT = "zero_based_position_in_packet_items"

_EMPTY_REVIEWER_FIELDS = {
    "reviewer_type": None,
    "reviewer_id": "",
    "reviewer_system": "",
    "reviewer_model": "",
    "review_protocol_id": "",
    "review_prompt_sha256": "",
    "reviewed_at": "",
    "overall_status": None,
    "referents_recognizable": None,
    "answer_supported_by_model_rgb": None,
    "family_intervention_valid": None,
    "severity": None,
    "reason_codes": [],
    "notes_zh": "",
}


class SemanticVisualAuditResultsError(ValueError):
    """Raised when audit inputs cannot support a trustworthy decision."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SemanticVisualAuditResultsError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is absent: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: _reject_non_finite(token, label),
        )
    except SemanticVisualAuditResultsError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticVisualAuditResultsError(
            f"cannot read valid {label} JSON: {path}"
        ) from exc
    _require(isinstance(value, dict), f"{label} JSON root must be an object: {path}")
    return value


def _reject_non_finite(token: str, label: str) -> None:
    raise SemanticVisualAuditResultsError(
        f"{label} contains a non-finite JSON value: {token}"
    )


def _nonempty_string(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{label} must be non-empty")
    return value


def _valid_reviewed_at(value: Any, label: str) -> str:
    text = _nonempty_string(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SemanticVisualAuditResultsError(
            f"{label} must be an ISO-8601 timestamp"
        ) from exc
    _require(parsed.tzinfo is not None, f"{label} must include a timezone")
    return text


def _validate_packet(packet: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    _require(packet.get("schema_version") == PACKET_SCHEMA, "unsupported audit packet schema")
    _nonempty_string(packet.get("packet_id"), "packet.packet_id")
    _require(
        packet.get("status") == "awaiting_independent_review",
        "audit packet status must be awaiting_independent_review",
    )
    _require(
        packet.get("independent_review_completed") is False,
        "audit packet must still be unreviewed",
    )
    provenance_policy = _mapping(
        packet.get("reviewer_provenance_policy"), "packet.reviewer_provenance_policy"
    )
    allowed_types = provenance_policy.get("allowed_reviewer_types")
    _require(
        isinstance(allowed_types, list)
        and all(isinstance(value, str) for value in allowed_types)
        and set(allowed_types) == ALLOWED_REVIEWER_TYPES
        and len(allowed_types) == len(ALLOWED_REVIEWER_TYPES),
        "packet reviewer provenance policy has unsupported reviewer types",
    )

    items = packet.get("items")
    _require(isinstance(items, list) and bool(items), "audit packet items must be non-empty")
    fact_ids: set[str] = set()
    audit_indices: set[int] = set()
    for position, raw_item in enumerate(items):
        item = _mapping(raw_item, f"packet.items[{position}]")
        fact_id = _nonempty_string(item.get("fact_id"), f"packet.items[{position}].fact_id")
        _require(fact_id not in fact_ids, f"duplicate packet fact_id: {fact_id}")
        fact_ids.add(fact_id)
        audit_index = item.get("audit_index")
        _require(
            isinstance(audit_index, int) and not isinstance(audit_index, bool),
            f"packet.items[{position}].audit_index must be an integer",
        )
        _require(audit_index == position + 1, "packet audit_index sequence is not one-based")
        _require(audit_index not in audit_indices, f"duplicate packet audit_index: {audit_index}")
        audit_indices.add(audit_index)
        _require(
            item.get("reviewer_fields") == _EMPTY_REVIEWER_FIELDS,
            f"packet item {fact_id} contains filled or unsupported reviewer_fields",
        )
        siblings = item.get("family_siblings_for_intervention_review")
        _require(
            isinstance(siblings, list),
            f"packet item {fact_id} family siblings must be a list",
        )

    evidence_binding = _mapping(
        packet.get("review_evidence_binding"), "packet.review_evidence_binding"
    )
    expected_evidence_sha = evidence_binding.get("sha256")
    _require(
        isinstance(expected_evidence_sha, str) and SHA256_RE.fullmatch(expected_evidence_sha),
        "packet review evidence SHA-256 is invalid",
    )
    evidence_payload = [
        {key: value for key, value in item.items() if key != "reviewer_fields"}
        for item in items
    ]
    actual_evidence_sha = _text_sha256(_canonical_json(evidence_payload))
    _require(
        actual_evidence_sha == expected_evidence_sha,
        "packet review evidence SHA-256 does not match packet items",
    )
    return [_mapping(item, f"packet.items[{index}]") for index, item in enumerate(items)]


def _validate_provenance(
    raw: Any,
    *,
    reviewer_label: str,
) -> dict[str, Any]:
    provenance = _mapping(raw, f"{reviewer_label}.reviewer_provenance")
    reviewer_type = provenance.get("reviewer_type")
    _require(
        reviewer_type in ALLOWED_REVIEWER_TYPES,
        f"{reviewer_label} has unsupported reviewer_type",
    )
    reviewer_id = _nonempty_string(
        provenance.get("reviewer_id"), f"{reviewer_label}.reviewer_id"
    )
    protocol_id = _nonempty_string(
        provenance.get("review_protocol_id"),
        f"{reviewer_label}.review_protocol_id",
    )
    reviewed_at = _valid_reviewed_at(
        provenance.get("reviewed_at"), f"{reviewer_label}.reviewed_at"
    )
    reviewer_system = provenance.get("reviewer_system", "")
    reviewer_model = provenance.get("reviewer_model", "")
    prompt = provenance.get("review_prompt")
    prompt_sha = provenance.get("review_prompt_sha256")
    prompt_verification = "not_applicable_for_human_without_recorded_prompt"
    if reviewer_type == "model_assisted_independent":
        reviewer_system = _nonempty_string(
            reviewer_system, f"{reviewer_label}.reviewer_system"
        )
        reviewer_model = _nonempty_string(
            reviewer_model, f"{reviewer_label}.reviewer_model"
        )
        prompt = _nonempty_string(prompt, f"{reviewer_label}.review_prompt")
        _require(
            isinstance(prompt_sha, str) and SHA256_RE.fullmatch(prompt_sha),
            f"{reviewer_label}.review_prompt_sha256 is invalid",
        )
        _require(
            _text_sha256(prompt) == prompt_sha,
            f"{reviewer_label} review prompt SHA-256 mismatch",
        )
        prompt_verification = "verified"
    else:
        _require(
            isinstance(reviewer_system, str) and isinstance(reviewer_model, str),
            f"{reviewer_label} human reviewer system/model fields must be strings",
        )
        if prompt is not None or prompt_sha is not None:
            prompt = _nonempty_string(prompt, f"{reviewer_label}.review_prompt")
            _require(
                isinstance(prompt_sha, str) and SHA256_RE.fullmatch(prompt_sha),
                f"{reviewer_label}.review_prompt_sha256 is invalid",
            )
            _require(
                _text_sha256(prompt) == prompt_sha,
                f"{reviewer_label} review prompt SHA-256 mismatch",
            )
            prompt_verification = "verified"
    return {
        "reviewer_type": reviewer_type,
        "reviewer_id": reviewer_id,
        "reviewer_system": reviewer_system,
        "reviewer_model": reviewer_model,
        "review_protocol_id": protocol_id,
        "review_prompt_sha256": prompt_sha,
        "review_prompt_sha256_verification": prompt_verification,
        "reviewed_at": reviewed_at,
    }


def _validate_review(
    raw: Any,
    *,
    packet_item: Mapping[str, Any],
    position: int,
    reviewer_label: str,
) -> dict[str, Any]:
    review = _mapping(raw, f"{reviewer_label}.reviews entry")
    index = review.get("index")
    _require(
        isinstance(index, int) and not isinstance(index, bool),
        f"{reviewer_label} review index must be an integer",
    )
    _require(
        index == position,
        f"{reviewer_label} review index {index} does not match packet position {position}",
    )
    fact_id = _nonempty_string(review.get("fact_id"), f"{reviewer_label} review fact_id")
    _require(
        fact_id == packet_item.get("fact_id"),
        f"{reviewer_label} review index/fact_id disagrees with packet: {index}/{fact_id}",
    )
    overall_status = review.get("overall_status")
    _require(
        overall_status in ALLOWED_OVERALL_STATUSES,
        f"review {fact_id} has unsupported overall_status",
    )
    recognizable = review.get("referents_recognizable")
    answer_supported = review.get("answer_supported")
    _require(
        isinstance(recognizable, bool),
        f"review {fact_id} referents_recognizable must be boolean",
    )
    _require(
        isinstance(answer_supported, bool),
        f"review {fact_id} answer_supported must be boolean",
    )
    family_valid = review.get("family_intervention_valid")
    siblings = packet_item.get("family_siblings_for_intervention_review")
    if siblings:
        _require(
            isinstance(family_valid, bool),
            f"review {fact_id} family_intervention_valid must be boolean for a family item",
        )
    else:
        _require(
            family_valid is None,
            f"review {fact_id} family_intervention_valid must be null without siblings",
        )
    severity = review.get("severity")
    _require(severity in ALLOWED_SEVERITIES, f"review {fact_id} has unsupported severity")
    reason_codes = review.get("reason_codes")
    _require(
        isinstance(reason_codes, list)
        and all(isinstance(code, str) and bool(code.strip()) for code in reason_codes)
        and len(reason_codes) == len(set(reason_codes)),
        f"review {fact_id} reason_codes must be unique non-empty strings",
    )
    notes = review.get("notes_zh")
    _require(isinstance(notes, str), f"review {fact_id} notes_zh must be a string")

    if overall_status == "pass":
        _require(severity == "none", f"review {fact_id} pass must have severity none")
        _require(not reason_codes, f"review {fact_id} pass must not contain reason codes")
        _require(
            recognizable and answer_supported and family_valid is not False,
            f"review {fact_id} pass conflicts with a failed semantic check",
        )
    elif overall_status == "minor_issue":
        _require(severity == "minor", f"review {fact_id} minor_issue must have severity minor")
        _require(bool(reason_codes), f"review {fact_id} minor_issue requires a reason code")
        _require(
            recognizable and answer_supported and family_valid is not False,
            f"review {fact_id} minor_issue conflicts with a failed semantic check",
        )
    else:
        _require(
            severity == "major",
            f"review {fact_id} {overall_status} must have severity major",
        )
        _require(
            bool(reason_codes),
            f"review {fact_id} {overall_status} requires a reason code",
        )
    return {
        "index": index,
        "audit_index": int(packet_item["audit_index"]),
        "fact_id": fact_id,
        "overall_status": overall_status,
        "referents_recognizable": recognizable,
        "answer_supported": answer_supported,
        "family_intervention_valid": family_valid,
        "severity": severity,
        "reason_codes": list(reason_codes),
        "notes_zh": notes,
    }


def adjudicate_semantic_visual_audit(
    packet_path: Path,
    reviewer_paths: Sequence[Path],
    *,
    allow_minor: bool = True,
) -> dict[str, Any]:
    """Validate and aggregate reviewer files into an evidence-bound decision."""

    packet_path = packet_path.expanduser().resolve()
    normalized_reviewers = sorted(
        (path.expanduser().resolve() for path in reviewer_paths), key=str
    )
    _require(bool(normalized_reviewers), "at least one reviewer JSON is required")
    _require(
        len(normalized_reviewers) == len(set(normalized_reviewers)),
        "reviewer paths must be unique",
    )
    packet = _read_json(packet_path, "audit packet")
    packet_items = _validate_packet(packet)
    packet_fact_to_position = {
        str(item["fact_id"]): position for position, item in enumerate(packet_items)
    }

    reviewer_bindings: list[dict[str, Any]] = []
    reviews_by_position: dict[int, dict[str, Any]] = {}
    reviewer_ids: set[str] = set()
    for reviewer_number, reviewer_path in enumerate(normalized_reviewers, 1):
        label = f"reviewer[{reviewer_number}]"
        payload = _read_json(reviewer_path, label)
        if "schema_version" in payload:
            _require(
                payload["schema_version"] == OPTIONAL_REVIEW_SCHEMA,
                f"{label} has unsupported schema_version",
            )
        if "packet_id" in payload:
            _require(
                payload["packet_id"] == packet.get("packet_id"),
                f"{label} is bound to a different packet_id",
            )
        provenance = _validate_provenance(
            payload.get("reviewer_provenance"), reviewer_label=label
        )
        reviewer_id = str(provenance["reviewer_id"])
        _require(reviewer_id not in reviewer_ids, f"duplicate reviewer_id: {reviewer_id}")
        reviewer_ids.add(reviewer_id)
        raw_reviews = payload.get("reviews")
        _require(isinstance(raw_reviews, list) and bool(raw_reviews), f"{label}.reviews is empty")
        positions_in_file: set[int] = set()
        for raw_review in raw_reviews:
            review_map = _mapping(raw_review, f"{label}.reviews entry")
            fact_id = _nonempty_string(
                review_map.get("fact_id"), f"{label}.reviews fact_id"
            )
            _require(fact_id in packet_fact_to_position, f"{label} has unknown fact_id: {fact_id}")
            position = packet_fact_to_position[fact_id]
            _require(
                position not in positions_in_file,
                f"{label} contains duplicate packet position {position}",
            )
            _require(
                position not in reviews_by_position,
                f"packet position {position} is covered by more than one reviewer",
            )
            positions_in_file.add(position)
            validated = _validate_review(
                review_map,
                packet_item=packet_items[position],
                position=position,
                reviewer_label=label,
            )
            validated["reviewer_id"] = reviewer_id
            reviews_by_position[position] = validated
        reviewer_bindings.append(
            {
                "path": str(reviewer_path),
                "sha256": _sha256(reviewer_path),
                "bytes": reviewer_path.stat().st_size,
                "review_count": len(raw_reviews),
                "covered_index_min": min(positions_in_file),
                "covered_index_max": max(positions_in_file),
                "provenance": provenance,
            }
        )

    expected_positions = set(range(len(packet_items)))
    actual_positions = set(reviews_by_position)
    missing = sorted(expected_positions - actual_positions)
    extra = sorted(actual_positions - expected_positions)
    _require(not extra, f"reviewers contain unexpected packet positions: {extra}")
    _require(not missing, f"reviewers do not cover packet positions: {missing}")
    ordered_reviews = [reviews_by_position[position] for position in range(len(packet_items))]

    status_counts = Counter(str(review["overall_status"]) for review in ordered_reviews)
    reason_counts = Counter(
        str(code) for review in ordered_reviews for code in review["reason_codes"]
    )
    failed_reviews = [
        review
        for review in ordered_reviews
        if review["overall_status"] in {"major_issue", "unreviewable"}
        or review["referents_recognizable"] is False
        or review["answer_supported"] is False
        or review["family_intervention_valid"] is False
    ]
    minor_reviews = [
        review for review in ordered_reviews if review["overall_status"] == "minor_issue"
    ]
    failure_reasons: list[str] = []
    if failed_reviews:
        failure_reasons.append("major_unreviewable_or_failed_semantic_check")
    if minor_reviews and not allow_minor:
        failure_reasons.append("minor_issue_disallowed_by_policy")
    status = "pass" if not failure_reasons else "fail"
    policy = {
        "decision_policy_id": "epispace.semantic_visual_audit_decision.v1",
        "coverage": "every packet item exactly once by index/fact_id",
        "review_index_contract": REVIEW_INDEX_CONTRACT,
        "allowed_reviewer_types": sorted(ALLOWED_REVIEWER_TYPES),
        "fail_on_overall_status": ["major_issue", "unreviewable"],
        "fail_on_false_fields": [
            "referents_recognizable",
            "answer_supported",
            "family_intervention_valid",
        ],
        "minor_issue_policy": "allow" if allow_minor else "fail",
        "invalid_or_incomplete_input_policy": "raise_error_and_emit_no_decision",
    }
    packet_sha = _sha256(packet_path)
    result_id = "semantic-audit-result-" + _text_sha256(
        _canonical_json(
            {
                "packet_sha256": packet_sha,
                "reviewer_sha256": [item["sha256"] for item in reviewer_bindings],
                "policy": policy,
            }
        )
    )[:20]
    return {
        "schema_version": RESULT_SCHEMA,
        "result_id": result_id,
        "status": status,
        "decision": {
            "semantic_visual_audit_gate": status == "pass",
            "failure_reasons": failure_reasons,
            "failed_item_count": len(failed_reviews),
            "minor_item_count": len(minor_reviews),
        },
        "policy": policy,
        "packet_binding": {
            "path": str(packet_path),
            "sha256": packet_sha,
            "bytes": packet_path.stat().st_size,
            "packet_id": packet["packet_id"],
            "review_evidence_sha256": packet["review_evidence_binding"]["sha256"],
            "item_count": len(packet_items),
        },
        "reviewer_bindings": reviewer_bindings,
        "summary": {
            "reviewed_items": len(ordered_reviews),
            "reviewer_count": len(reviewer_bindings),
            "overall_status_counts": {
                key: status_counts.get(key, 0)
                for key in ("pass", "minor_issue", "major_issue", "unreviewable")
            },
            "reason_code_counts": dict(sorted(reason_counts.items())),
            "failed_fact_ids": [review["fact_id"] for review in failed_reviews],
            "minor_fact_ids": [review["fact_id"] for review in minor_reviews],
        },
        "item_decisions": ordered_reviews,
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    """Render a concise Chinese report with the same bindings and decision."""

    decision = _mapping(result["decision"], "decision")
    policy = _mapping(result["policy"], "policy")
    packet = _mapping(result["packet_binding"], "packet_binding")
    summary = _mapping(result["summary"], "summary")
    status_label = "通过" if result["status"] == "pass" else "不通过"
    lines = [
        "# EpiSpace 独立语义视觉审计裁决",
        "",
        f"> **最终裁决：{status_label}**（machine status=`{result['status']}`）",
        "",
        "## 证据绑定",
        "",
        f"- Result ID：`{result['result_id']}`",
        f"- Packet ID：`{packet['packet_id']}`",
        f"- Packet SHA-256：`{packet['sha256']}`",
        f"- Review evidence SHA-256：`{packet['review_evidence_sha256']}`",
        f"- Packet 条目：{packet['item_count']}",
        "",
        "| Reviewer | Type | Reviews | Prompt SHA | File SHA |",
        "|---|---|---:|---|---|",
    ]
    for binding in result["reviewer_bindings"]:
        provenance = binding["provenance"]
        prompt_sha = provenance.get("review_prompt_sha256") or "N/A (human)"
        lines.append(
            f"| `{provenance['reviewer_id']}` | `{provenance['reviewer_type']}` | "
            f"{binding['review_count']} | `{prompt_sha}` | `{binding['sha256']}` |"
        )
    counts = summary["overall_status_counts"]
    lines.extend(
        (
            "",
            "## 覆盖与结论",
            "",
            f"- 覆盖：{summary['reviewed_items']} / {packet['item_count']}（每个零基 packet index 与 fact_id 恰好一次）",
            (
                "- 判定计数："
                f"pass={counts['pass']}，minor={counts['minor_issue']}，"
                f"major={counts['major_issue']}，unreviewable={counts['unreviewable']}"
            ),
            f"- Minor policy：`{policy['minor_issue_policy']}`",
            f"- Fail-closed gate：`{str(decision['semantic_visual_audit_gate']).lower()}`",
        )
    )
    if decision["failure_reasons"]:
        lines.append("- 失败原因：" + ", ".join(decision["failure_reasons"]))
    if summary["failed_fact_ids"]:
        lines.extend(("", "### 失败条目", ""))
        for fact_id in summary["failed_fact_ids"]:
            review = next(item for item in result["item_decisions"] if item["fact_id"] == fact_id)
            lines.append(
                f"- `{fact_id}`：`{review['overall_status']}`；"
                f"reason={', '.join(review['reason_codes'])}；{review['notes_zh']}"
            )
    if summary["minor_fact_ids"]:
        lines.extend(("", "### Minor 条目", ""))
        for fact_id in summary["minor_fact_ids"]:
            review = next(item for item in result["item_decisions"] if item["fact_id"] == fact_id)
            lines.append(
                f"- `{fact_id}`：reason={', '.join(review['reason_codes'])}；{review['notes_zh']}"
            )
    return "\n".join(lines).rstrip() + "\n"


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def write_semantic_visual_audit_result(
    packet_path: Path,
    reviewer_paths: Sequence[Path],
    *,
    json_output: Path,
    markdown_output: Path,
    allow_minor: bool = True,
) -> dict[str, Any]:
    result = adjudicate_semantic_visual_audit(
        packet_path, reviewer_paths, allow_minor=allow_minor
    )
    _write_text_atomic(
        json_output,
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(markdown_output, render_markdown(result))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed aggregation of independent semantic visual reviews."
    )
    parser.add_argument("packet", type=Path)
    parser.add_argument("reviewers", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument(
        "--fail-on-minor",
        action="store_true",
        help="Treat otherwise valid minor_issue reviews as a failed gate.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    packet_path = args.packet.expanduser().resolve()
    json_output = args.json_output or packet_path.with_name(
        f"{packet_path.stem}.result.json"
    )
    markdown_output = args.markdown_output or packet_path.with_name(
        f"{packet_path.stem}.result.md"
    )
    try:
        result = write_semantic_visual_audit_result(
            packet_path,
            args.reviewers,
            json_output=json_output,
            markdown_output=markdown_output,
            allow_minor=not args.fail_on_minor,
        )
    except SemanticVisualAuditResultsError as exc:
        print(f"semantic visual audit input error: {exc}", file=sys.stderr)
        return 2
    print(
        f"{result['result_id']}: {result['status']} -> {json_output}, {markdown_output}"
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
