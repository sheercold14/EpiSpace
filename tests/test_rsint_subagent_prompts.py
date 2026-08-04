from __future__ import annotations

import json

import pytest

from episode3d.qa_generation.backends import BackendOutputError, validate_json_schema
from episode3d.qa_generation.schemas import CapabilitySpec, Claim, ClaimSheet, NumericSurface
from episode3d.rsint_llm_pipeline import (
    NARRATION_SKILLS,
    build_answer_narrator_request,
    build_critic_request,
    build_question_editor_request,
)


def _episode_context() -> dict:
    return {
        "episode_id": "ep3d-rsint17-dialogue-canonical",
        "scene_id": "Rs_int",
        "turn_index": 4,
        "released_view_ids": [f"view-{index:03d}" for index in range(8)],
        "route_phase_zh": "转身返回途中",
        "coordinate_convention_zh": "+X为右，+Y为前",
        "prior_user_turns_zh": ["我会按顺序给你住宅漫游画面。", "继续向前走。"],
    }


def _capability_contract() -> dict:
    return {
        "primary": "CR",
        "supporting": ["SR", "PT"],
        "official_subtask": "跨视图空间关系整合",
        "typed_operation_sequence": ["G", "F", "B", "R", "M", "V"],
        "learning_intent_zh": "整合从未同框的电视和冰箱并判断全局方向",
        "response_profile": "evidence_transform_conclusion",
    }


def _claim_sheet() -> ClaimSheet:
    claims = (
        Claim(
            claim_id="claim-covisibility",
            kind="co_visibility",
            statement_zh="电视和冰箱没有在同一视角中同时出现",
            value=[],
            required=True,
            evidence_view_ids=("view-002", "view-006"),
            evidence_entity_ids=("tv", "fridge"),
            source_paths=("facts[0].certificate.co_visible_view_ids",),
        ),
        Claim(
            claim_id="claim-relation",
            kind="relation",
            statement_zh="电视在冰箱后方，另外略偏右",
            value="behind_right",
            required=True,
            evidence_view_ids=("view-002", "view-006"),
            evidence_entity_ids=("tv", "fridge"),
            source_paths=("facts[0].certificate.relation",),
            frame_id="room_xy",
            allowed_directions=("后方", "略偏右"),
        ),
        Claim(
            claim_id="claim-distance",
            kind="metric",
            statement_zh="电视与冰箱纵向相距约4.7米",
            value=4.658,
            required=True,
            evidence_entity_ids=("tv", "fridge"),
            source_paths=("facts[0].certificate.delta_y_m",),
            numeric_surfaces=(NumericSurface(4.658, "米", 0.2, "约4.7米"),),
        ),
    )
    return ClaimSheet(
        claim_sheet_id="sheet-rsint-turn-4",
        fact_id="fact-rsint-turn-4",
        episode_id="ep3d-rsint17-dialogue-canonical",
        task_type="cross_view_relation",
        program_id="program-rsint-turn-4",
        semantic_signature="G-F-B-R-M-V",
        capability=CapabilitySpec(
            "CR", ("SR", "PT"), "跨视图空间关系整合", "read", "compositional"
        ),
        exposure_view_ids=tuple(f"view-{index:03d}" for index in range(8)),
        claims=claims,
        answer_key="behind_right",
        answer_value={"relation": "behind", "delta_y_m": 4.658},
        canonical_answer_zh="电视在冰箱后方约4.7米，另外略偏右。",
        answer_status="accepted",
        source_episode_ir_sha256="a" * 64,
        source_bundle="/data/rsint/bundle.json",
    )


def _candidate_answer(skill_id: str = "route_replay") -> dict:
    first = "电视和冰箱没有在同一个视角里同时出现。"
    second = "综合前后观察，电视在冰箱后方约4.7米，另外略偏右。"
    return {
        "request_id": "sheet-rsint-turn-4",
        "strategy_id": skill_id,
        "answer_key": "behind_right",
        "surface_answer_zh": first + second,
        "sentences": [
            {
                "role": "evidence",
                "text": first,
                "claim_ids": ["claim-covisibility"],
            },
            {
                "role": "conclusion",
                "text": second,
                "claim_ids": ["claim-relation", "claim-distance"],
            },
        ],
    }


def test_question_editor_payload_is_structurally_answer_blind() -> None:
    request = build_question_editor_request(
        request_id="question-rsint-turn-4",
        episode_context=_episode_context(),
        capability_contract=_capability_contract(),
        intent_zh="询问两个从未同框物体的共视历史和全局方向",
        protected_slots={"subject": "电视", "reference": "冰箱"},
        required_slots=("subject", "reference"),
        surface_constraints_zh=("承接转身返回的叙事", "保留两个连续子问题"),
    )

    serialized = json.dumps(request.payload, ensure_ascii=False).casefold()
    assert request.role == "question_editor"
    assert request.skill_id is None
    assert "answer" not in serialized
    assert "claim" not in serialized
    assert "certificate" not in serialized
    assert "behind" not in serialized
    assert set(request.backend_kwargs()) == {"stage", "prompt", "payload", "output_schema"}


def test_question_editor_rejects_accidental_answer_field_before_provider_call() -> None:
    context = _episode_context() | {"answer_zh": "后方"}
    with pytest.raises(ValueError, match="forbidden field"):
        build_question_editor_request(
            request_id="question-rsint-turn-4",
            episode_context=context,
            capability_contract=_capability_contract(),
            intent_zh="询问方向",
            protected_slots={"subject": "电视", "reference": "冰箱"},
            required_slots=("subject", "reference"),
        )


def test_each_narration_skill_has_distinct_context_and_frozen_provider_contract() -> None:
    requests = {
        skill_id: build_answer_narrator_request(
            episode_context=_episode_context(),
            question_zh="电视和冰箱同框过吗？综合观察，电视在冰箱的什么方向？",
            claim_sheet=_claim_sheet(),
            skill_id=skill_id,
        )
        for skill_id in NARRATION_SKILLS
    }

    assert len(requests) == 8
    assert len({request.prompt for request in requests.values()}) == 8
    assert len({request.stage for request in requests.values()}) == 8
    for skill_id, request in requests.items():
        assert request.role == "answer_narrator"
        assert request.skill_id == skill_id
        assert request.payload["narration_skill"]["skill_id"] == skill_id
        assert request.output_schema["properties"]["strategy_id"]["enum"] == [skill_id]


def test_three_roles_have_distinct_contexts_and_contract_versions() -> None:
    question_request = build_question_editor_request(
        request_id="question-rsint-turn-4",
        episode_context=_episode_context(),
        capability_contract=_capability_contract(),
        intent_zh="询问跨视图方向",
        protected_slots={"subject": "电视", "reference": "冰箱"},
        required_slots=("subject", "reference"),
    )
    answer_request = build_answer_narrator_request(
        episode_context=_episode_context(),
        question_zh="电视相对冰箱在哪个方向？",
        claim_sheet=_claim_sheet(),
        skill_id="route_replay",
    )
    critic_request = build_critic_request(
        episode_context=_episode_context(),
        question_zh="电视相对冰箱在哪个方向？",
        claim_sheet=_claim_sheet(),
        skill_id="route_replay",
        candidate_answer=_candidate_answer(),
    )

    assert len({question_request.prompt, answer_request.prompt, critic_request.prompt}) == 3
    assert question_request.payload["schema_version"] == "epispace.rsint.question_editor.v1"
    assert answer_request.payload["schema_version"] == "epispace.rsint.answer_narrator.v1"
    assert critic_request.payload["schema_version"] == "epispace.rsint.critic.v1"


def test_answer_and_critic_outputs_have_strict_schemas() -> None:
    answer_request = build_answer_narrator_request(
        episode_context=_episode_context(),
        question_zh="电视和冰箱同框过吗？综合观察，电视在冰箱的什么方向？",
        claim_sheet=_claim_sheet(),
        skill_id="route_replay",
    )
    candidate = _candidate_answer()
    validate_json_schema(candidate, answer_request.output_schema)
    with pytest.raises(BackendOutputError, match="additional fields"):
        validate_json_schema(candidate | {"private_cot": "先建图再计算"}, answer_request.output_schema)

    critic_request = build_critic_request(
        episode_context=_episode_context(),
        question_zh="电视和冰箱同框过吗？综合观察，电视在冰箱的什么方向？",
        claim_sheet=_claim_sheet(),
        skill_id="route_replay",
        candidate_answer=candidate,
    )
    accepted = {
        "request_id": "sheet-rsint-turn-4",
        "verdict": "accept",
        "supported": True,
        "unsupported_sentence_indices": [],
        "missing_required_claim_ids": [],
        "reason_codes": [],
        "repair_brief_zh": "",
    }
    validate_json_schema(accepted, critic_request.output_schema)
    with pytest.raises(BackendOutputError, match="additional fields"):
        validate_json_schema(accepted | {"rewritten_answer_zh": "不要由 critic 改写"}, critic_request.output_schema)


def test_critic_rejects_malformed_candidate_before_provider_call() -> None:
    candidate = _candidate_answer() | {"strategy_id": "calibration"}
    with pytest.raises(BackendOutputError, match="allowed enum"):
        build_critic_request(
            episode_context=_episode_context(),
            question_zh="电视和冰箱同框过吗？综合观察，电视在冰箱的什么方向？",
            claim_sheet=_claim_sheet(),
            skill_id="route_replay",
            candidate_answer=candidate,
        )
