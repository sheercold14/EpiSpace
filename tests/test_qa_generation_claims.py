from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from episode3d.qa_generation.claims import (
    ClaimCompilationError,
    compile_claim_sheet,
)

ROOT = Path(__file__).resolve().parents[1]
IR_PATH = ROOT / "data" / "epispace_pilot_v1" / "episodes.ir.jsonl"

EXPECTED_TASKS = {
    "counterfactual_verification",
    "cross_view_relation",
    "egocentric_relation",
    "elevation_relation_transfer",
    "evidence_presence_reveal",
    "evidence_presence_unknown",
    "grounding_presence",
    "last_seen_memory",
    "metric_distance",
    "object_centric_perspective",
    "occlusion_reveal",
    "occlusion_unknown",
    "orbit_identity",
    "rotation_change_detection",
    "target_view_prediction",
    "unknown_abstention",
}


def _episodes() -> list[dict[str, object]]:
    with IR_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _find(task_type: str, *, family_variant: str | None = None) -> tuple[dict, dict]:
    for episode in _episodes():
        for question in episode["questions"]:
            if question["task_type"] != task_type:
                continue
            if family_variant is not None and question.get("family_variant") != family_variant:
                continue
            return episode, question
    raise AssertionError(f"missing fixture task={task_type}, variant={family_variant}")


def test_all_current_episode_ir_questions_compile_fail_closed() -> None:
    task_types: set[str] = set()
    count = 0
    for episode in _episodes():
        for question in episode["questions"]:
            sheet = compile_claim_sheet(episode, question)
            task_types.add(sheet.task_type)
            count += 1
            assert sheet.required_claim_ids
            assert set(sheet.required_claim_ids) <= set(sheet.allowed_claim_ids)
            assert all(claim.source_paths for claim in sheet.claims)
            assert all(set(claim.evidence_view_ids) <= set(sheet.exposure_view_ids) for claim in sheet.claims)
    assert count == 884
    assert task_types == EXPECTED_TASKS


def test_metric_geometry_tamper_is_rejected() -> None:
    episode, question = _find("metric_distance")
    tampered = copy.deepcopy(question)
    for check in tampered["certificate"]["checks"]:
        if check["name"] == "center_distance_m":
            check["measured_value"] += 1.0
            break
    with pytest.raises(ClaimCompilationError, match="distance"):
        compile_claim_sheet(episode, tampered)


@pytest.mark.parametrize(
    ("task_type", "variant"),
    [
        ("unknown_abstention", "canonical"),
        ("evidence_presence_unknown", "prefix_unknown"),
        ("evidence_presence_unknown", "decisive_deleted"),
        ("occlusion_unknown", "prefix_unknown"),
        ("occlusion_unknown", "decisive_deleted"),
    ],
)
def test_unknown_claims_serialize_no_hidden_supervision(
    task_type: str, variant: str
) -> None:
    episode, question = _find(task_type, family_variant=variant)
    sheet = compile_claim_sheet(episode, question)
    serialized_claims = json.dumps(
        [claim.as_dict() for claim in sheet.claims], ensure_ascii=False
    ).lower()
    assert "oracle" not in serialized_claims
    assert "world_truth" not in serialized_claims
    assert "decisive_view" not in serialized_claims
    assert "deleted_view" not in serialized_claims
    assert all(not claim.evidence_entity_ids for claim in sheet.claims)


def test_unknown_profile_cannot_be_expanded_to_a_decisive_view() -> None:
    episode, question = _find("evidence_presence_unknown", family_variant="prefix_unknown")
    expanded = (*tuple(question["model_view_ids"]), "view-002")
    with pytest.raises(ClaimCompilationError, match="cannot be expanded or reordered"):
        compile_claim_sheet(episode, question, exposure_view_ids=expanded)


def test_counterfactual_requires_linked_cross_view_geometry() -> None:
    episode, question = _find("counterfactual_verification", family_variant="claim_false")
    broken_episode = copy.deepcopy(episode)
    broken_episode["questions"] = [
        item for item in broken_episode["questions"] if item["task_type"] != "cross_view_relation"
    ]
    with pytest.raises(ClaimCompilationError, match="linked cross_view_relation"):
        compile_claim_sheet(broken_episode, question)
