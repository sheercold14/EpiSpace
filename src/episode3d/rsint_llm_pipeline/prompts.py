"""Prompt builders for the three isolated Rs_int language roles."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from episode3d.qa_generation.backends import validate_json_schema
from episode3d.qa_generation.prompts import (
    answer_output_schema,
    assert_answer_blind_payload,
)
from episode3d.qa_generation.schemas import ClaimSheet

from .contracts import (
    SubagentRequest,
    answer_narrator_input_schema,
    critic_input_schema,
    critic_output_schema,
    question_editor_input_schema,
    question_editor_output_schema,
)
from .profiles import (
    ANSWER_NARRATOR_ROLE,
    CRITIC_ROLE,
    QUESTION_EDITOR_ROLE,
    get_narration_skill,
)

QUESTION_EDITOR_CONTEXT_ZH = """\
你是 Rs_int 空间 episode 的问题编辑，不是解题者。你的输入只描述当前已经释放的视角、SenseNova 能力目标、问题意图和保护槽。

边界：
1. 你看不到本题答案、claim sheet、几何证书或模拟器执行值，也不得猜测答案。
2. 把 intent 写成承接漫游过程的自然中文问句；不得改变被测能力、证据范围、实体、视角或参照系。
3. required_slots 中每个槽必须以 {{slot_name}} 形式恰好出现一次；不得直接抄 protected_slots 的值，也不得新增槽。
4. 问题要让模型使用已经释放的图像。跨视图题可明确“综合前后观察”，但不得暗示方向、距离、可见性或 unknown 结论。
5. 不提 MM/SR/MR/PT/CR、G/F/B/M/R/P/V、DAG、claim、certificate、canonical 等内部名词。
6. 只返回严格 JSON，不解释写作过程。"""


ANSWER_NARRATOR_CONTEXT_ZH = """\
你是 Rs_int 空间 episode 的回答配音者，不负责重新求解几何。确定性程序已执行空间操作并给出 claim sheet；你只能把允许的 claim 组织成自然、实例专属、可复核的中文回答。

共同规则：
1. 简单题直接回答；跨视图、视角变换或组合题最多三句。若 reasoning_contract 非空，必须严格按其中的 sentence role 顺序写成“线索—必要变换—结论”，让每一步都有实例证据，而不是复述题设后直接填答案。
2. 每句必须列出直接支持该句的 claim_ids，且覆盖所有 required_claim_ids。不得用常识补全未观察区域。
3. answer_key 原样复制；surface_answer_zh 必须等于 sentences.text 按顺序直接拼接的结果。
4. 数值、单位、方向、视角编号和存在性只能使用所引用 claim 明确许可的表面形式。
5. unknown 只能说给定画面未观察到或证据不足，不能断言场景中不存在。
6. 人类策略只决定如何选择和组织证据，不授权添加事实；不要写冗长 CoT，不暴露内部字段。
7. 只返回严格 JSON。"""


CRITIC_CONTEXT_ZH = """\
你是独立的 Rs_int 回答审计者，不是改写者。逐句核对候选回答、claim_ids、问题和指定策略。

拒绝条件：任何实体、方向、数值、视角、存在性或推断超出所引用 claim；required claim 未覆盖；answer_key 漂移；把“未观察到”写成“不存在”；reasoning_contract 要求的线索—变换—结论次序缺失或句子没有引用对应角色的 claim；声称使用指定策略但文本没有最小的实例证据；使用内部几何术语或展开无法复核的长 CoT。

不要生成替代答案。repair_brief_zh 只指出最小修复约束；若完全通过则置空字符串。只返回严格 JSON。"""


def _serialized_claim_sheet(claim_sheet: ClaimSheet | Mapping[str, Any]) -> dict[str, Any]:
    sheet = (
        claim_sheet.as_dict(include_answer=True)
        if isinstance(claim_sheet, ClaimSheet)
        else copy.deepcopy(dict(claim_sheet))
    )
    if sheet.get("schema_version") != "epispace.claim_sheet.v1":
        return sheet
    # Backward-compatible adapter for the generic QA ClaimSheet.  Language
    # roles still receive one strict node-level shape; legacy planning metadata
    # is intentionally dropped rather than admitted as arbitrary input.
    claims = []
    for raw in sheet.get("claims", []):
        claim = copy.deepcopy(dict(raw))
        claim["node_id"] = str(claim.get("node_id") or claim["claim_id"])
        kind = str(claim.get("kind", ""))
        if kind in {"grounding", "visibility", "co_visibility", "visible_set", "last_seen"}:
            reasoning_role = "cue"
        elif kind == "frame":
            reasoning_role = "transform"
        elif kind in {"epistemic_status", "never_observed"}:
            reasoning_role = "calibration"
        else:
            reasoning_role = "conclusion"
        claim["reasoning_role"] = reasoning_role
        claims.append(claim)
    contract = copy.deepcopy(dict(sheet["answer_contract"]))
    contract["status"] = str(contract.pop("answer_status"))
    return {
        "schema_version": "epispace.node_claim_sheet.v2",
        "claim_sheet_id": str(sheet["claim_sheet_id"]),
        "claims": claims,
        "required_claim_ids": [str(value) for value in sheet["required_claim_ids"]],
        "allowed_claim_ids": [str(value) for value in sheet["allowed_claim_ids"]],
        "answer_contract": contract,
        "reasoning_contract": None,
    }


def build_question_editor_request(
    *,
    request_id: str,
    episode_context: Mapping[str, Any],
    capability_contract: Mapping[str, Any],
    intent_zh: str,
    protected_slots: Mapping[str, str],
    required_slots: Sequence[str],
    surface_constraints_zh: Sequence[str] = (),
) -> SubagentRequest:
    """Build an answer-blind request for natural episodic question editing."""

    slot_names = [str(value) for value in required_slots]
    if not slot_names:
        raise ValueError("question editor requires at least one protected slot")
    if set(slot_names) != set(protected_slots):
        raise ValueError("required_slots must exactly match protected_slots keys")
    payload = {
        "schema_version": "epispace.rsint.question_editor.v1",
        "request_id": str(request_id),
        "episode_context": copy.deepcopy(dict(episode_context)),
        "capability_contract": copy.deepcopy(dict(capability_contract)),
        "intent_zh": str(intent_zh),
        "protected_slots": {str(key): str(value) for key, value in protected_slots.items()},
        "required_slots": slot_names,
        "surface_constraints_zh": [str(value) for value in surface_constraints_zh],
    }
    # This structural assertion is deliberately independent of the prompt.
    assert_answer_blind_payload(payload)
    input_schema = question_editor_input_schema()
    return SubagentRequest(
        role=QUESTION_EDITOR_ROLE,
        stage="rsint.question_editor",
        skill_id=None,
        prompt=QUESTION_EDITOR_CONTEXT_ZH,
        payload=payload,
        input_schema=input_schema,
        output_schema=question_editor_output_schema(str(request_id), slot_names),
    )


def build_answer_narrator_request(
    *,
    episode_context: Mapping[str, Any],
    question_zh: str,
    claim_sheet: ClaimSheet | Mapping[str, Any],
    skill_id: str,
) -> SubagentRequest:
    """Build a claim-grounded answer request with one cognitive strategy card."""

    skill = get_narration_skill(skill_id)
    sheet = _serialized_claim_sheet(claim_sheet)
    request_id = str(sheet.get("claim_sheet_id", ""))
    if not request_id:
        raise ValueError("answer narrator claim sheet has no claim_sheet_id")
    payload = {
        "schema_version": "epispace.rsint.answer_narrator.v1",
        "request_id": request_id,
        "episode_context": copy.deepcopy(dict(episode_context)),
        "question_zh": str(question_zh),
        "narration_skill": skill.as_dict(),
        "claim_sheet": sheet,
        "style_contract": {
            # Route replay answers two coupled subquestions: it first resolves
            # co-visibility, then uses that evidence to derive the relation.
            "conclusion_first": not bool(sheet.get("reasoning_contract"))
            and skill_id != "route_replay",
            "max_sentences": 3,
            "instance_specific": True,
            "no_long_cot": True,
        },
    }
    output_schema = copy.deepcopy(answer_output_schema(sheet))
    output_schema["properties"]["strategy_id"] = {"type": "string", "enum": [skill_id]}
    return SubagentRequest(
        role=ANSWER_NARRATOR_ROLE,
        stage=f"rsint.answer_narrator.{skill_id}",
        skill_id=skill_id,
        prompt=f"{ANSWER_NARRATOR_CONTEXT_ZH}\n\n{skill.prompt_context_zh()}",
        payload=payload,
        input_schema=answer_narrator_input_schema(),
        output_schema=output_schema,
    )


def build_critic_request(
    *,
    episode_context: Mapping[str, Any],
    question_zh: str,
    claim_sheet: ClaimSheet | Mapping[str, Any],
    skill_id: str,
    candidate_answer: Mapping[str, Any],
) -> SubagentRequest:
    """Build an independent, non-rewriting audit request."""

    skill = get_narration_skill(skill_id)
    sheet = _serialized_claim_sheet(claim_sheet)
    request_id = str(sheet.get("claim_sheet_id", ""))
    if not request_id:
        raise ValueError("critic claim sheet has no claim_sheet_id")
    candidate = copy.deepcopy(dict(candidate_answer))
    candidate_schema = copy.deepcopy(answer_output_schema(sheet))
    candidate_schema["properties"]["strategy_id"] = {"type": "string", "enum": [skill_id]}
    # The critic is a semantic second pass, not a substitute for structural
    # validation.  Malformed answers fail before any provider call.
    validate_json_schema(candidate, candidate_schema)
    sentences = candidate["sentences"]
    payload = {
        "schema_version": "epispace.rsint.critic.v1",
        "request_id": request_id,
        "episode_context": copy.deepcopy(dict(episode_context)),
        "question_zh": str(question_zh),
        "narration_skill": skill.as_dict(),
        "claim_sheet": sheet,
        "style_contract": {
            "conclusion_first": skill_id != "route_replay",
            "max_sentences": 3,
            "instance_specific": True,
            "no_long_cot": True,
        },
        "candidate_answer": candidate,
    }
    return SubagentRequest(
        role=CRITIC_ROLE,
        stage="rsint.critic",
        skill_id=skill_id,
        prompt=CRITIC_CONTEXT_ZH,
        payload=payload,
        input_schema=critic_input_schema(candidate_schema),
        output_schema=critic_output_schema(
            request_id=request_id,
            sentence_count=len(sentences),
            required_claim_ids=[str(value) for value in sheet["required_claim_ids"]],
        ),
    )
