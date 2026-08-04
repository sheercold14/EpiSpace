from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from episode3d.semantic_visual_audit import PACKET_SCHEMA
from episode3d.semantic_visual_audit_results import (
    RESULT_SCHEMA,
    SemanticVisualAuditResultsError,
    adjudicate_semantic_visual_audit,
    write_semantic_visual_audit_result,
)


def _canonical_sha(value: object) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _empty_reviewer_fields() -> dict[str, object]:
    return {
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


def _packet(tmp_path: Path) -> Path:
    items = [
        {
            "audit_index": 1,
            "fact_id": "fact-a",
            "episode_id": "episode-a",
            "family_siblings_for_intervention_review": [{"fact_id": "fact-a-sibling"}],
            "actual_model_rgb": [{"path": "/rgb/a.png", "sha256": "a" * 64}],
            "reviewer_fields": _empty_reviewer_fields(),
        },
        {
            "audit_index": 2,
            "fact_id": "fact-b",
            "episode_id": "episode-b",
            "family_siblings_for_intervention_review": [],
            "actual_model_rgb": [{"path": "/rgb/b.png", "sha256": "b" * 64}],
            "reviewer_fields": _empty_reviewer_fields(),
        },
    ]
    evidence = [
        {key: value for key, value in item.items() if key != "reviewer_fields"}
        for item in items
    ]
    return _write(
        tmp_path / "packet.json",
        {
            "schema_version": PACKET_SCHEMA,
            "packet_id": "semantic-audit-fixture",
            "status": "awaiting_independent_review",
            "independent_review_completed": False,
            "reviewer_provenance_policy": {
                "allowed_reviewer_types": [
                    "model_assisted_independent",
                    "human_independent",
                ]
            },
            "review_evidence_binding": {"sha256": _canonical_sha(evidence)},
            "items": items,
        },
    )


def _provenance(reviewer_id: str) -> dict[str, str]:
    prompt = f"Independently review the assigned RGB items as {reviewer_id}."
    return {
        "reviewer_type": "model_assisted_independent",
        "reviewer_id": reviewer_id,
        "reviewer_system": "OpenAI Codex",
        "reviewer_model": "GPT-5",
        "review_protocol_id": "epispace.semantic_rgb_review.v1",
        "review_prompt": prompt,
        "review_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "reviewed_at": "2026-07-17T01:02:03Z",
    }


def _review(
    index: int,
    fact_id: str,
    *,
    status: str = "pass",
    family_valid: bool | None = None,
) -> dict[str, object]:
    if status == "pass":
        severity = "none"
        reasons: list[str] = []
    elif status == "minor_issue":
        severity = "minor"
        reasons = ["minor_template_wording"]
    else:
        severity = "major"
        reasons = ["referent_not_recognizable"]
    return {
        "fact_id": fact_id,
        "index": index,
        "overall_status": status,
        "referents_recognizable": status not in {"major_issue", "unreviewable"},
        "answer_supported": status not in {"major_issue", "unreviewable"},
        "family_intervention_valid": family_valid,
        "severity": severity,
        "reason_codes": reasons,
        "notes_zh": "独立查看了实际 RGB。",
    }


def _review_files(tmp_path: Path) -> tuple[Path, Path]:
    first = _write(
        tmp_path / "review-a.json",
        {
            "reviewer_provenance": _provenance("agent-a"),
            "reviews": [_review(0, "fact-a", family_valid=True)],
        },
    )
    second = _write(
        tmp_path / "review-b.json",
        {
            "reviewer_provenance": _provenance("agent-b"),
            "reviews": [_review(1, "fact-b", status="minor_issue")],
        },
    )
    return first, second


def test_complete_reviews_are_bound_and_minor_is_allowed_by_default(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    reviewers = _review_files(tmp_path)
    json_output = tmp_path / "result.json"
    markdown_output = tmp_path / "result.md"

    result = write_semantic_visual_audit_result(
        packet,
        reviewers,
        json_output=json_output,
        markdown_output=markdown_output,
    )

    assert result["schema_version"] == RESULT_SCHEMA
    assert result["status"] == "pass"
    assert result["decision"] == {
        "semantic_visual_audit_gate": True,
        "failure_reasons": [],
        "failed_item_count": 0,
        "minor_item_count": 1,
    }
    assert result["policy"]["minor_issue_policy"] == "allow"
    assert result["policy"]["review_index_contract"] == "zero_based_position_in_packet_items"
    assert result["packet_binding"]["sha256"] == hashlib.sha256(packet.read_bytes()).hexdigest()
    assert [item["review_count"] for item in result["reviewer_bindings"]] == [1, 1]
    assert all(
        item["provenance"]["review_prompt_sha256_verification"] == "verified"
        for item in result["reviewer_bindings"]
    )
    assert json.loads(json_output.read_text(encoding="utf-8")) == result
    markdown = markdown_output.read_text(encoding="utf-8")
    assert "最终裁决：通过" in markdown
    assert "2 / 2" in markdown


def test_major_or_false_semantic_check_makes_valid_report_fail(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    first, second = _review_files(tmp_path)
    payload = json.loads(first.read_text(encoding="utf-8"))
    payload["reviews"][0] = _review(
        0, "fact-a", status="major_issue", family_valid=False
    )
    _write(first, payload)

    result = adjudicate_semantic_visual_audit(packet, [first, second])

    assert result["status"] == "fail"
    assert result["decision"]["semantic_visual_audit_gate"] is False
    assert result["decision"]["failure_reasons"] == [
        "major_unreviewable_or_failed_semantic_check"
    ]
    assert result["summary"]["failed_fact_ids"] == ["fact-a"]


def test_minor_can_be_rejected_by_explicit_policy(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    reviewers = _review_files(tmp_path)

    result = adjudicate_semantic_visual_audit(packet, reviewers, allow_minor=False)

    assert result["status"] == "fail"
    assert result["policy"]["minor_issue_policy"] == "fail"
    assert result["decision"]["failure_reasons"] == [
        "minor_issue_disallowed_by_policy"
    ]


@pytest.mark.parametrize("mutation", ["schema", "status", "filled", "evidence_sha"])
def test_packet_contract_fails_closed(tmp_path: Path, mutation: str) -> None:
    packet = _packet(tmp_path)
    reviewers = _review_files(tmp_path)
    payload = json.loads(packet.read_text(encoding="utf-8"))
    if mutation == "schema":
        payload["schema_version"] = "wrong"
    elif mutation == "status":
        payload["status"] = "pass"
    elif mutation == "filled":
        payload["items"][0]["reviewer_fields"]["overall_status"] = "pass"
    else:
        payload["review_evidence_binding"]["sha256"] = "0" * 64
    _write(packet, payload)

    with pytest.raises(SemanticVisualAuditResultsError):
        adjudicate_semantic_visual_audit(packet, reviewers)


@pytest.mark.parametrize("mutation", ["type", "prompt", "time", "duplicate_id"])
def test_reviewer_provenance_fails_closed(tmp_path: Path, mutation: str) -> None:
    packet = _packet(tmp_path)
    first, second = _review_files(tmp_path)
    payload = json.loads(first.read_text(encoding="utf-8"))
    if mutation == "type":
        payload["reviewer_provenance"]["reviewer_type"] = "unspecified"
    elif mutation == "prompt":
        payload["reviewer_provenance"]["review_prompt"] += " tampered"
    elif mutation == "time":
        payload["reviewer_provenance"]["reviewed_at"] = "2026-07-17T01:02:03"
    else:
        other = json.loads(second.read_text(encoding="utf-8"))
        payload["reviewer_provenance"]["reviewer_id"] = other[
            "reviewer_provenance"
        ]["reviewer_id"]
    _write(first, payload)

    with pytest.raises(SemanticVisualAuditResultsError):
        adjudicate_semantic_visual_audit(packet, [first, second])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_index", "wrong_fact"])
def test_exact_index_fact_coverage_fails_closed(tmp_path: Path, mutation: str) -> None:
    packet = _packet(tmp_path)
    first, second = _review_files(tmp_path)
    if mutation == "missing":
        with pytest.raises(SemanticVisualAuditResultsError, match="do not cover"):
            adjudicate_semantic_visual_audit(packet, [first])
        return
    payload = json.loads(second.read_text(encoding="utf-8"))
    if mutation == "duplicate":
        payload["reviews"][0] = _review(0, "fact-a", family_valid=True)
    elif mutation == "wrong_index":
        payload["reviews"][0]["index"] = 0
    else:
        payload["reviews"][0]["fact_id"] = "unknown-fact"
    _write(second, payload)

    with pytest.raises(SemanticVisualAuditResultsError):
        adjudicate_semantic_visual_audit(packet, [first, second])


@pytest.mark.parametrize("mutation", ["pass_false", "minor_none", "family_null"])
def test_internally_inconsistent_fields_fail_closed(tmp_path: Path, mutation: str) -> None:
    packet = _packet(tmp_path)
    first, second = _review_files(tmp_path)
    payload = json.loads(first.read_text(encoding="utf-8"))
    review = payload["reviews"][0]
    if mutation == "pass_false":
        review["answer_supported"] = False
    elif mutation == "minor_none":
        review["overall_status"] = "minor_issue"
    else:
        review["family_intervention_valid"] = None
    _write(first, payload)

    with pytest.raises(SemanticVisualAuditResultsError):
        adjudicate_semantic_visual_audit(packet, [first, second])
