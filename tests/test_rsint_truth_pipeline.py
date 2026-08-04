from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from episode3d.qwen_training import supervised_assistant_turns
from episode3d.rsint_llm_pipeline.pipeline import (
    _episode_sft,
    _isolated_sft,
    _observable_state_delta,
    _state_aux_sft,
)
from episode3d.rsint_llm_pipeline.truth import RsIntTruthCompiler

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT.parent / "OminiGibson" / "outputs" / "Rs_int_seed17"


@pytest.fixture(scope="module")
def truth() -> dict[str, object]:
    return RsIntTruthCompiler(BUNDLE).compile()


def test_truth_compiler_freezes_causal_seven_round_schedule(
    truth: dict[str, object],
) -> None:
    rounds = truth["rounds"]
    assert isinstance(rounds, list)
    assert [value["new_view_ids"] for value in rounds] == [
        ["view-000"],
        ["view-001", "view-002"],
        ["view-003", "view-004"],
        ["view-005", "view-006", "view-007"],
        ["view-008", "view-009"],
        ["view-010"],
        [],
    ]
    assert rounds[3]["program"]["semantic_signature"] == "G+G+G->F*->B->R+M->V"
    assert rounds[4]["program"]["semantic_signature"] == "G+G->B->F_query->P->R->V"
    assert truth["capability_coverage"]["covered"] == ["MM", "SR", "PT", "CR"]
    assert "MR" in truth["capability_coverage"]["unsupported"]


def test_claims_bind_to_executed_nodes_and_reproduce_flagship_values(
    truth: dict[str, object],
) -> None:
    rounds = truth["rounds"]
    relation_claim = next(
        claim for claim in rounds[1]["claim_sheet"]["claims"] if claim["kind"] == "relation"
    )
    assert relation_claim["value"]["relation"] == "left_of"
    assert relation_claim["value"]["dominant_delta_m"] == pytest.approx(1.724095)
    crossview = rounds[3]
    assert (
        next(
            claim
            for claim in crossview["claim_sheet"]["claims"]
            if claim["kind"] == "co_visibility"
        )["value"]
        == []
    )
    behind = next(
        claim for claim in crossview["claim_sheet"]["claims"] if claim["kind"] == "relation"
    )
    assert behind["value"]["relation"] == "behind"
    assert behind["value"]["dominant_delta_m"] == pytest.approx(4.658407)
    perspective = next(
        claim for claim in rounds[4]["claim_sheet"]["claims"] if claim["kind"] == "relation"
    )
    assert perspective["value"]["target"] == "fridge"
    assert perspective["value"]["quadrant"] == "front-left"
    assert perspective["value"]["query_xy_m"] == pytest.approx([-3.956367, 5.189577])
    assert perspective["value"]["angle_to_boundaries_deg"]["minimum"] == pytest.approx(
        37.320733
    )
    assert perspective["value"]["axis_crossing"] == {
        "left_right": False,
        "front_back": False,
    }
    assert perspective["value"]["hard_quadrant_valid"] is True

    for turn in rounds:
        node_ids = {node["node_id"] for node in turn["program"]["nodes"]}
        assert all(claim["node_id"] in node_ids for claim in turn["claim_sheet"]["claims"])


def test_perspective_sampling_rejects_boundary_label_and_computes_real_verifier(
    truth: dict[str, object],
) -> None:
    diagnostic = truth["sampling_diagnostics"]["perspective"]
    rejected = diagnostic["rejected_boundary_probe"]["relation"]
    assert rejected["target"] == "sofa"
    assert rejected["query_xy_m"] == pytest.approx([-2.344437, 0.416677])
    assert rejected["axis_crossing"]["front_back"] is True
    assert rejected["angle_to_boundaries_deg"]["minimum"] == pytest.approx(10.077954)
    assert rejected["hard_quadrant_valid"] is False
    assert rejected["qualified_relation"] == {
        "primary": "left",
        "secondary": "front",
        "secondary_degree": "slight",
    }

    turn = truth["rounds"][4]
    verify = next(node for node in turn["program"]["nodes"] if node["operation"] == "V")
    assert verify["variant"] == "obb_and_angular_margin"
    assert verify["value"] is True


def test_compositional_rounds_require_cue_transform_conclusion(
    truth: dict[str, object],
) -> None:
    crossview, perspective = truth["rounds"][3:5]
    for turn in (crossview, perspective):
        assert turn["claim_sheet"]["reasoning_contract"] == {
            "required_sentence_roles": ["evidence", "transform", "conclusion"],
            "enforce_claim_role_alignment": True,
        }
        roles = {claim["reasoning_role"] for claim in turn["claim_sheet"]["claims"]}
        assert {"cue", "transform", "conclusion"} <= roles

    visual_bridge = next(
        claim
        for claim in crossview["claim_sheet"]["claims"]
        if claim["kind"] == "visual_cue"
    )
    assert visual_bridge["value"]["outbound"]["horizontal_gap_px"] > 500
    assert visual_bridge["value"]["return"]["horizontal_gap_px"] > 400
    turnaround = next(
        claim
        for claim in crossview["claim_sheet"]["claims"]
        if claim["kind"] == "route_event"
    )
    assert abs(turnaround["value"]["yaw_change_deg"]) == pytest.approx(180.0)

    no_single_frame = next(
        claim
        for claim in perspective["claim_sheet"]["claims"]
        if claim["kind"] == "co_visibility"
    )
    assert no_single_frame["value"] == {"fridge_chair": [], "fridge_tv": []}


def test_unknown_claim_does_not_reveal_world_truth_entity_identity(
    truth: dict[str, object],
) -> None:
    turn = truth["rounds"][5]
    bed_claim = next(
        claim for claim in turn["claim_sheet"]["claims"] if claim["kind"] == "never_observed"
    )
    assert bed_claim["evidence_entity_ids"] == []
    assert bed_claim["epistemic_scope"] == "provided_views_only"
    assert "不存在" not in bed_claim["statement_zh"]
    blueprint = json.dumps(turn["question_blueprint"], ensure_ascii=False)
    assert "observed_view_ids" not in blueprint
    assert "unknown" not in blueprint.casefold()


def test_episode_export_contract_supervises_all_seven_assistant_turns(
    truth: dict[str, object],
) -> None:
    generated = json.loads(json.dumps(truth))
    for turn in generated["rounds"]:
        answer = turn["claim_sheet"]["answer_contract"]["canonical_answer_zh"]
        turn["generated"] = {
            "question": {"question_zh": "请回答当前空间问题？"},
            "answer": {"surface_answer_zh": answer},
        }
    generated["artifact_sha256"] = "0" * 64
    row = _episode_sft(generated)
    turns = supervised_assistant_turns(row)
    assert len(turns) == 7
    assert [value.message_index for value in turns] == [2, 4, 6, 8, 10, 12, 14]
    assert row["loss_policy"]["normalization"] == "mean_tokens_per_turn_then_mean_turns"
    isolated = _isolated_sft(generated)
    episode_pairs = {value["unit_id"]: value for value in row["comparison_contracts"]}
    isolated_pairs = {value["comparison_contract"]["unit_id"]: value["comparison_contract"] for value in isolated}
    assert episode_pairs.keys() == isolated_pairs.keys()
    for turn_id in episode_pairs:
        assert episode_pairs[turn_id]["exposure_sha256"] == isolated_pairs[turn_id]["exposure_sha256"]
        assert episode_pairs[turn_id]["answer_sha256"] == isolated_pairs[turn_id]["answer_sha256"]


def test_state_delta_uses_observable_categories_without_unknown_probe(
    truth: dict[str, object],
) -> None:
    first = _observable_state_delta(truth, 0)
    assert first["added"] == [
        "coffee_table",
        "dishwasher",
        "fridge",
        "laptop",
        "microwave",
        "sofa",
    ]
    crossview = _observable_state_delta(truth, 3)
    assert "standing_tv" in crossview["added"]
    perspective = _observable_state_delta(truth, 4)
    assert "swivel_chair" in perspective["added"]
    unknown = _observable_state_delta(truth, 5)
    assert "bed" not in unknown["added"]


def test_state_auxiliary_target_excludes_oracle_identifiers_and_raw_poses(
    truth: dict[str, object],
) -> None:
    generated = json.loads(json.dumps(truth))
    for turn in generated["rounds"]:
        turn["generated"] = {"question": {"question_zh": "当前问题？"}}
    serialized = json.dumps(_state_aux_sft(generated), ensure_ascii=False)
    assert not re.search(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        serialized,
    )
    for forbidden in ("world_from_camera", "decision_margin_m", "source_paths"):
        assert forbidden not in serialized
