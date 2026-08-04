from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from episode3d.qa_generation.backends import BackendOutputError, validate_json_schema
from episode3d.qa_generation.blueprints import build_question_blueprint
from episode3d.qa_generation.claims import compile_claim_sheet
from episode3d.qa_generation.episode_planner import (
    EpisodeBatchPlanner,
    PlannerContract,
    question_by_fact,
    read_episode_ir,
)
from episode3d.qa_generation.pipeline import (
    _optimizer_decision,
    _training_records,
    _validate_blueprint_claim_alignment,
)
from episode3d.qa_generation.prompts import (
    assert_answer_blind_payload,
    build_answer_batch_request,
)
from episode3d.qa_generation.schemas import EpisodeBatchPlan, canonical_json
from episode3d.qa_generation.validators import validate_answer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "qa_generation_pilot_v1.json"
EPISODE_IR_PATH = PROJECT_ROOT / "data" / "epispace_pilot_v1" / "episodes.ir.jsonl"
AUDIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "epispace_pilot_v1"
    / "semantic_visual_audit"
    / "result.json"
)


@pytest.fixture(scope="module")
def pilot_inputs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return config, read_episode_ir(EPISODE_IR_PATH)


@pytest.fixture(scope="module")
def pilot_plans(
    pilot_inputs: tuple[dict[str, Any], list[dict[str, Any]]],
) -> tuple[EpisodeBatchPlan, ...]:
    config, episodes = pilot_inputs
    planner = EpisodeBatchPlanner(
        heldout_program_ids=config["source"]["heldout_program_ids"]
    )
    return tuple(
        planner.plan(
            episodes,
            target_question_count=config["target_question_count"],
            semantic_audit_path=AUDIT_PATH,
            split="train",
            include_eval_only=True,
            curated=config["curated_episode_facts"],
        )
    )


def _episode_index(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(episode["episode_id"]): episode for episode in episodes}


def _fake_generation_records(
    episode: Mapping[str, Any],
    plan: EpisodeBatchPlan,
    *,
    heldout_program_ids: set[str],
) -> list[dict[str, Any]]:
    sources = question_by_fact(episode)
    records: list[dict[str, Any]] = []
    for planned in plan.questions:
        source = sources[planned.fact_id]
        disposition = (
            "composition_heldout"
            if planned.program_id in heldout_program_ids
            else "train_candidate"
        )
        records.append(
            {
                "fact_id": planned.fact_id,
                "split": plan.split,
                "disposition": disposition,
                "capability": planned.capability.as_dict(),
                "validation": {"passed": True},
                "program": dict(source["program"]),
                "claim_sheet": {"claim_sheet_id": f"sheet-{planned.fact_id}"},
                "generation_provenance": {
                    "answer": {"request_hash": f"answer-{planned.fact_id}"}
                },
                "generated": {
                    "question_zh": f"请回答与 {planned.fact_id} 对应的空间问题。",
                    "answer": {"surface_answer_zh": "依据给定观察，可以作答。"},
                },
            }
        )
    return records


def test_all_twenty_curated_facts_compile_answer_blind_blueprints(
    pilot_inputs: tuple[dict[str, Any], list[dict[str, Any]]],
) -> None:
    config, episodes = pilot_inputs
    episodes_by_id = _episode_index(episodes)
    compiled = []

    for episode_id, fact_ids in config["curated_episode_facts"].items():
        episode = episodes_by_id[episode_id]
        sources = question_by_fact(episode)
        exposure = tuple(item["view_id"] for item in episode["observations"])
        for fact_id in fact_ids:
            source = sources[fact_id]
            baseline = build_question_blueprint(
                episode, source, exposure_view_ids=exposure
            ).answer_blind_dict()

            mutated = copy.deepcopy(source)
            mutated["answer_zh"] = "SENTINEL_ANSWER_ZH"
            mutated["answer_value"] = {"sentinel": "SENTINEL_ANSWER_VALUE"}
            mutated["answer_status"] = "SENTINEL_ANSWER_STATUS"
            mutated["certificate"] = {"oracle": "SENTINEL_CERTIFICATE"}
            mutated["rationale_zh"] = "SENTINEL_RATIONALE"
            after_mutation = build_question_blueprint(
                episode, mutated, exposure_view_ids=exposure
            ).answer_blind_dict()

            assert after_mutation == baseline
            assert "fact_id" not in baseline
            assert "SENTINEL" not in canonical_json(baseline)
            assert_answer_blind_payload(baseline)
            compiled.append((episode_id, fact_id, baseline))

    assert len(compiled) == 20
    assert len({item[2]["request_id"] for item in compiled}) == 20


def test_curated_planner_preserves_declared_episode_and_fact_order(
    pilot_inputs: tuple[dict[str, Any], list[dict[str, Any]]],
    pilot_plans: tuple[EpisodeBatchPlan, ...],
) -> None:
    config, _ = pilot_inputs
    declared = config["curated_episode_facts"]

    assert [plan.episode_id for plan in pilot_plans] == list(declared)
    assert sum(len(plan.questions) for plan in pilot_plans) == 20
    for plan in pilot_plans:
        assert [question.fact_id for question in plan.questions] == declared[plan.episode_id]


def test_answer_batch_schema_accepts_the_requested_strategy_and_rejects_others(
    pilot_inputs: tuple[dict[str, Any], list[dict[str, Any]]],
) -> None:
    config, episodes = pilot_inputs
    episode_id, fact_ids = next(iter(config["curated_episode_facts"].items()))
    episode = _episode_index(episodes)[episode_id]
    source = question_by_fact(episode)[fact_ids[0]]
    exposure = tuple(item["view_id"] for item in episode["observations"])
    sheet = compile_claim_sheet(episode, source, exposure_view_ids=exposure)
    strategy_id = "scale_anchor"
    request = build_answer_batch_request(
        [(sheet, "这两个物体相距多远？", strategy_id)]
    )
    answer = {
        "request_id": sheet.claim_sheet_id,
        "strategy_id": strategy_id,
        "answer_key": sheet.answer_key,
        "surface_answer_zh": sheet.canonical_answer_zh,
        "sentences": [
            {
                "role": "conclusion",
                "text": sheet.canonical_answer_zh,
                "claim_ids": list(sheet.required_claim_ids),
            }
        ],
    }

    validate_json_schema({"responses": [answer]}, request.output_schema)
    with pytest.raises(BackendOutputError, match="allowed enum"):
        validate_json_schema(
            {"responses": [answer | {"strategy_id": "unrequested_strategy"}]},
            request.output_schema,
        )


def test_curated_unknown_calibration_does_not_trigger_false_nonexistence_gate(
    pilot_inputs: tuple[dict[str, Any], list[dict[str, Any]]],
) -> None:
    _, episodes = pilot_inputs
    episode = _episode_index(episodes)["c58ce8f9-ade1-5cea-8c8b-a750d22e7c8e"]
    source = question_by_fact(episode)["fact-2431df1fab3917fb8956"]
    exposure = tuple(item["view_id"] for item in episode["observations"])
    sheet = compile_claim_sheet(episode, source, exposure_view_ids=exposure)
    answer = {
        "request_id": sheet.claim_sheet_id,
        "strategy_id": "calibration",
        "answer_key": sheet.answer_key,
        "surface_answer_zh": sheet.canonical_answer_zh,
        "sentences": [
            {
                "role": "calibration",
                "text": sheet.canonical_answer_zh,
                "claim_ids": list(sheet.required_claim_ids),
            }
        ],
    }

    report = validate_answer(sheet, answer, expected_strategy_id="calibration")
    assert report.passed, report.errors


def test_counterfactual_parser_ignores_reference_frame_direction_token(
    pilot_inputs: tuple[dict[str, Any], list[dict[str, Any]]],
) -> None:
    _, episodes = pilot_inputs
    episode = _episode_index(episodes)["c58ce8f9-ade1-5cea-8c8b-a750d22e7c8e"]
    source = question_by_fact(episode)["fact-08a2291c443c8e7ee315"]
    exposure = tuple(item["view_id"] for item in episode["observations"])
    blueprint = build_question_blueprint(episode, source, exposure_view_ids=exposure)
    sheet = compile_claim_sheet(episode, source, exposure_view_ids=exposure)

    # The source first says the reference frame's positive direction is front,
    # then tests the *different* relation "behind".  The latter is the premise.
    assert blueprint.slot_values["tested_relation"] == "后方"
    _validate_blueprint_claim_alignment(blueprint, sheet)


def test_paired_view_indices_are_valid_question_premises_in_an_answer(
    pilot_inputs: tuple[dict[str, Any], list[dict[str, Any]]],
) -> None:
    _, episodes = pilot_inputs
    episode = _episode_index(episodes)["a2f1fe74-da7e-546b-aa9d-b1a2f62f9904"]
    source = question_by_fact(episode)["fact-ae9e47410179f06a107d"]
    exposure = tuple(item["view_id"] for item in episode["observations"])
    sheet = compile_claim_sheet(episode, source, exposure_view_ids=exposure)
    identity_claim = next(
        claim for claim in sheet.claims if claim.kind == "stable_object_identity"
    )
    orbit_claim = next(claim for claim in sheet.claims if claim.kind == "orbit_viewpoint_change")
    conclusion = "第3个和第7个视角中看到的是同一个餐桌。"
    evidence = "两个观察方位相隔约180度。"
    answer = {
        "request_id": sheet.claim_sheet_id,
        "strategy_id": "mental_simulation",
        "answer_key": sheet.answer_key,
        "surface_answer_zh": conclusion + evidence,
        "sentences": [
            {
                "role": "conclusion",
                "text": conclusion,
                "claim_ids": [identity_claim.claim_id],
            },
            {
                "role": "evidence",
                "text": evidence,
                "claim_ids": [orbit_claim.claim_id],
            },
        ],
    }

    report = validate_answer(sheet, answer, expected_strategy_id="mental_simulation")
    assert report.passed, report.errors


def test_heldout_facts_and_t1_shortfall_do_not_enter_optimizer_exports(
    pilot_inputs: tuple[dict[str, Any], list[dict[str, Any]]],
    pilot_plans: tuple[EpisodeBatchPlan, ...],
) -> None:
    config, episodes = pilot_inputs
    episodes_by_id = _episode_index(episodes)
    heldout_program_ids = set(config["source"]["heldout_program_ids"])
    contract = PlannerContract()
    expected_train_counts = {
        "c58ce8f9-ade1-5cea-8c8b-a750d22e7c8e": 3,
        "94ae917a-b0ff-51fa-949c-39452aba2248": 5,
        "a2f1fe74-da7e-546b-aa9d-b1a2f62f9904": 4,
        "b1a433ea-e1c0-5284-bbb0-62362974f841": 4,
    }

    for plan in pilot_plans:
        episode = episodes_by_id[plan.episode_id]
        records = _fake_generation_records(
            episode, plan, heldout_program_ids=heldout_program_ids
        )
        decision = _optimizer_decision(records, contract, heldout_program_ids)
        episode_rows, isolated_rows = _training_records(
            episode, plan, records, decision
        )

        assert decision["train_question_count"] == expected_train_counts[plan.episode_id]
        assert decision["checks"]["heldout_disposition_partition_valid"] is True
        heldout_records = [
            record
            for record in records
            if record["program"]["program_id"] in heldout_program_ids
        ]
        if heldout_records:
            incorrectly_partitioned = copy.deepcopy(records)
            next(
                record
                for record in incorrectly_partitioned
                if record["program"]["program_id"] in heldout_program_ids
            )["disposition"] = "train_candidate"
            invalid_decision = _optimizer_decision(
                incorrectly_partitioned, contract, heldout_program_ids
            )
            assert invalid_decision["optimizer_eligible"] is False
            assert (
                invalid_decision["checks"]["heldout_disposition_partition_valid"]
                is False
            )
        if plan.episode_id == "c58ce8f9-ade1-5cea-8c8b-a750d22e7c8e":
            assert decision["optimizer_eligible"] is False
            assert decision["checks"]["question_count_4_to_6"] is False
            assert episode_rows == []
            assert isolated_rows == []
            continue

        assert decision["optimizer_eligible"] is True
        assert len(episode_rows) == 1
        assert len(isolated_rows) == expected_train_counts[plan.episode_id]
        exported_fact_ids = set(episode_rows[0]["comparison_contract"]["fact_ids"])
        assert len(exported_fact_ids) == expected_train_counts[plan.episode_id]
        assert all(
            question.program_id not in heldout_program_ids
            for question in plan.questions
            if question.fact_id in exported_fact_ids
        )
        assert not any(
            question.fact_id in exported_fact_ids
            for question in plan.questions
            if question.program_id in heldout_program_ids
        )
        assert {row["comparison_contract"]["fact_ids"][0] for row in isolated_rows} == (
            exported_fact_ids
        )
