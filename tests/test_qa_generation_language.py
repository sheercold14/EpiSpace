from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from episode3d.qa_generation.backends import (
    BackendCacheError,
    BackendOutputError,
    CodexExecBackend,
    FixtureBackend,
    ReplayBackend,
    request_hash,
)
from episode3d.qa_generation.prompts import (
    ANSWER_STAGE,
    QUESTION_STAGE,
    assert_answer_blind_payload,
    build_answer_request,
    build_question_request,
)
from episode3d.qa_generation.schemas import (
    CapabilitySpec,
    Claim,
    ClaimSheet,
    NumericSurface,
    QuestionBlueprint,
)
from episode3d.qa_generation.validators import (
    corpus_dedup_result,
    render_question,
    validate_answer,
    validate_question,
)


def _capability(profile: str = "compositional") -> CapabilitySpec:
    return CapabilitySpec("SR", ("PT",), "cross-view relation", "read", profile)


def _blueprint() -> QuestionBlueprint:
    return QuestionBlueprint(
        request_id="question-1",
        fact_id="fact-1",
        task_type="cross_view_relation",
        capability=_capability(),
        intent_zh="询问两个不同视角物体的左右关系",
        slot_values={"subject": "椅子", "reference": "桌子"},
        required_slots=("subject", "reference"),
        allowed_strategies=("landmark_hierarchy", "mental_simulation"),
        output_contract="一个简短的中文问句",
        family_key="family-1",
    )


def _sheet(*, unknown: bool = False) -> ClaimSheet:
    claims = (
        Claim(
            claim_id="claim-relation",
            kind="relation",
            statement_zh="椅子在桌子左侧",
            value="left_of",
            required=True,
            evidence_view_ids=("view-000", "view-002"),
            evidence_entity_ids=("chair", "table"),
            source_paths=("facts[0].answer",),
            frame_id="room",
            allowed_directions=("left_of",),
        ),
        Claim(
            claim_id="claim-distance",
            kind="metric",
            statement_zh="两者相距约1.7米",
            value=1.724,
            required=False,
            numeric_surfaces=(NumericSurface(1.724, "米", 0.2, "约1.7米"),),
        ),
    )
    if unknown:
        claims = (
            Claim(
                claim_id="claim-unknown",
                kind="epistemic",
                statement_zh="给定画面没有观察到床，证据不足",
                value=None,
                required=True,
                epistemic_scope="observed_evidence_only",
            ),
        )
    return ClaimSheet(
        claim_sheet_id="sheet-1",
        fact_id="fact-1",
        episode_id="episode-1",
        task_type="unknown_abstention" if unknown else "cross_view_relation",
        program_id="program-1",
        semantic_signature="signature-1",
        capability=_capability("epistemic" if unknown else "compositional"),
        exposure_view_ids=("view-000", "view-001", "view-002"),
        claims=claims,
        answer_key="unknown" if unknown else "left_of",
        answer_value=None if unknown else "left_of",
        canonical_answer_zh="无法确定" if unknown else "左侧",
        answer_status="unknown" if unknown else "known",
        source_episode_ir_sha256="a" * 64,
        source_bundle="/bundle",
    )


def _valid_question() -> dict:
    return {
        "request_id": "question-1",
        "strategy_id": "mental_simulation",
        "template_zh": "把前后看到的内容联系起来，{{subject}}在{{reference}}的哪一侧？",
        "used_slots": ["subject", "reference"],
    }


def _valid_answer() -> dict:
    text = "综合前后画面，椅子在桌子左侧，约相距1.7米。"
    return {
        "request_id": "sheet-1",
        "strategy_id": "mental_simulation",
        "answer_key": "left_of",
        "surface_answer_zh": text,
        "sentences": [
            {
                "role": "conclusion",
                "text": text,
                "claim_ids": ["claim-relation", "claim-distance"],
            }
        ],
    }


def test_question_prompt_is_structurally_answer_blind_and_uses_protected_slots() -> None:
    request = build_question_request(_blueprint())

    assert request.stage == QUESTION_STAGE
    serialized = json.dumps(request.payload, ensure_ascii=False)
    assert "answer_key" not in serialized
    assert "certificate" not in serialized
    assert "claims" not in request.payload
    report = validate_question(_blueprint(), _valid_question())
    assert report.passed, report.errors
    assert render_question(_blueprint(), _valid_question()).endswith("椅子在桌子的哪一侧？")


def test_answer_blind_gate_rejects_nested_leakage() -> None:
    with pytest.raises(ValueError, match="forbidden field"):
        assert_answer_blind_payload({"intent": "natural", "nested": {"answer_zh": "左侧"}})


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ({"template_zh": "{{subject}}和{{subject}}相对{{reference}}在哪边？"}, "required_slots_once"),
        ({"template_zh": "椅子相对{{reference}}在哪边？"}, "required_slots_once"),
        ({"used_slots": ["subject"]}, "used_slots_exact"),
        ({"strategy_id": "free_form_guess"}, "strategy_allowlist"),
    ],
)
def test_question_validator_fails_closed(mutation: dict, failed_check: str) -> None:
    response = _valid_question() | mutation
    report = validate_question(_blueprint(), response)

    assert not report.passed
    assert report.checks[failed_check] is False


def test_answer_prompt_contains_claim_contract_but_not_simulator_certificate() -> None:
    request = build_answer_request(
        _sheet(), question_zh="椅子在桌子的哪一侧？", strategy_id="mental_simulation"
    )

    assert request.stage == ANSWER_STAGE
    assert request.payload["claim_sheet"]["answer_contract"]["answer_key"] == "left_of"
    assert "certificate" not in json.dumps(request.payload, ensure_ascii=False).casefold()


def test_claim_grounded_answer_passes_all_deterministic_gates() -> None:
    report = validate_answer(
        _sheet(), _valid_answer(), expected_strategy_id="mental_simulation"
    )

    assert report.passed, report.errors
    assert all(report.checks.values())


@pytest.mark.parametrize("surface", ["任何一个视角", "任何单一视角", "同一视角"])
def test_view_identity_phrases_are_not_misparsed_as_chinese_counts(surface: str) -> None:
    text = f"没有{surface}同时显示这两个物体。"
    candidate = _valid_answer() | {
        "surface_answer_zh": text,
        "sentences": [
            {"role": "evidence", "text": text, "claim_ids": ["claim-relation"]}
        ],
    }
    report = validate_answer(_sheet(), candidate)

    assert report.checks["numeric_unit_allowlist"] is True, report.errors


@pytest.mark.parametrize(
    ("text", "claim_ids", "answer_key", "failed_check"),
    [
        ("椅子在桌子右侧。", ["claim-relation"], "left_of", "direction_allowlist"),
        ("椅子在桌子左侧，相距1.9米。", ["claim-relation"], "left_of", "numeric_unit_allowlist"),
        ("椅子在桌子左侧。", ["claim-distance"], "left_of", "required_claim_coverage"),
        ("椅子在桌子左侧。", ["claim-not-allowed"], "left_of", "claim_id_allowlist"),
        ("椅子在桌子左侧。", ["claim-relation"], "right_of", "answer_key"),
        ("canonical 坐标中椅子在桌子左侧。", ["claim-relation"], "left_of", "no_geometry_jargon"),
    ],
)
def test_answer_validator_rejects_wrong_claim_number_direction_or_jargon(
    text: str, claim_ids: list[str], answer_key: str, failed_check: str
) -> None:
    candidate = _valid_answer() | {
        "answer_key": answer_key,
        "surface_answer_zh": text,
        "sentences": [{"role": "conclusion", "text": text, "claim_ids": claim_ids}],
    }
    report = validate_answer(_sheet(), candidate)

    assert not report.passed
    assert report.checks[failed_check] is False


def test_unknown_must_report_observation_limit_not_scene_nonexistence() -> None:
    bad_text = "这间房里不存在床，所以无法确定。"
    bad = {
        "request_id": "sheet-1",
        "strategy_id": "temporal_recall",
        "answer_key": "unknown",
        "surface_answer_zh": bad_text,
        "sentences": [
            {
                "role": "calibration",
                "text": bad_text,
                "claim_ids": ["claim-unknown"],
            }
        ],
    }
    good_text = "给定画面里没有观察到床，现有证据不足，无法确定。"
    good = bad | {
        "surface_answer_zh": good_text,
        "sentences": [
            {
                "role": "calibration",
                "text": good_text,
                "claim_ids": ["claim-unknown"],
            }
        ],
    }

    bad_report = validate_answer(_sheet(unknown=True), bad)
    good_report = validate_answer(_sheet(unknown=True), good)
    assert not bad_report.passed
    assert bad_report.checks["unknown_epistemic_scope"] is False
    assert good_report.passed, good_report.errors


def test_character_ngram_dedup_scrubs_entities_numbers_and_directions() -> None:
    result = corpus_dedup_result(
        "沙发在桌子左侧约1.7米吗？",
        ["椅子在柜子右侧约2.3米吗？"],
        variable_terms=("沙发", "桌子", "椅子", "柜子"),
    )

    assert not result["passed"]
    assert result["max_similarity"] == 1.0


def _simple_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["value"],
        "properties": {"value": {"type": "string", "enum": ["ok"]}},
    }


def test_request_hash_covers_stage_provider_model_prompt_schema_and_payload() -> None:
    base = {
        "stage": "question",
        "provider": "provider-a",
        "model": "model-a",
        "prompt": "prompt-a",
        "payload": {"input": "a"},
        "output_schema": _simple_schema(),
    }
    original = request_hash(**base)
    mutations = [
        {"stage": "answer"},
        {"provider": "provider-b"},
        {"model": "model-b"},
        {"prompt": "prompt-b"},
        {"payload": {"input": "b"}},
        {"output_schema": _simple_schema() | {"title": "different"}},
    ]

    assert all(request_hash(**(base | mutation)) != original for mutation in mutations)


def test_fixture_and_replay_fail_closed_on_missing_or_invalid_output(tmp_path: Path) -> None:
    fixture = FixtureBackend({"question": "```json\n{\"value\": \"ok\"}\n```"})
    with pytest.raises(BackendOutputError, match="strict JSON"):
        fixture.complete(
            stage="question", prompt="p", payload={"x": 1}, output_schema=_simple_schema()
        )

    replay = ReplayBackend(tmp_path)
    with pytest.raises(BackendCacheError, match="cache miss"):
        replay.complete(
            stage="question", prompt="p", payload={"x": 1}, output_schema=_simple_schema()
        )


def test_codex_exec_uses_safe_argv_isolated_cwd_timeout_and_frozen_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        output_path = Path(command[command.index("-o") + 1])
        output_path.write_text('{"value":"ok"}\n', encoding="utf-8")
        calls.append({"command": command, **kwargs})
        assert Path(str(kwargs["cwd"])).is_dir()
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("episode3d.qa_generation.backends.shutil.which", lambda _: "/bin/codex")
    monkeypatch.setattr("episode3d.qa_generation.backends.subprocess.run", fake_run)
    backend = CodexExecBackend(tmp_path / "cache", timeout_seconds=17)

    first = backend.complete(
        stage="question", prompt="strict", payload={"x": 1}, output_schema=_simple_schema()
    )
    second = backend.complete(
        stage="question", prompt="strict", payload={"x": 1}, output_schema=_simple_schema()
    )

    assert first.response == {"value": "ok"}
    assert not first.cache_hit
    assert second.cache_hit
    assert len(calls) == 1
    call = calls[0]
    command = call["command"]
    assert isinstance(command, list)
    assert command[:2] == ["/bin/codex", "exec"]
    assert "--ephemeral" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--skip-git-repo-check" in command
    assert "--output-schema" in command
    assert command[-1] == "-"
    assert call["shell"] is False
    assert call["timeout"] == 17
    assert "epispace-codex-" in str(call["cwd"])

    replay = ReplayBackend(tmp_path / "cache")
    replayed = replay.complete(
        stage="question", prompt="strict", payload={"x": 1}, output_schema=_simple_schema()
    )
    assert replayed.cache_hit
    assert replayed.response == first.response
