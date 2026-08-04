"""Prompt contracts for the replaceable QA language backend.

The question writer is deliberately answer blind.  It receives an intent and
protected surface slots, but never a claim sheet, answer key, simulator value,
or geometry certificate.  The answer writer is a separate call and may only
verbalize an allow-listed claim sheet.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from episode3d.qa_generation.schemas import ClaimSheet, QuestionBlueprint

QUESTION_STAGE = "question"
ANSWER_STAGE = "answer"
CRITIC_STAGE = "critic"


@dataclass(frozen=True)
class PromptRequest:
    """One fully specified structured-generation request."""

    stage: str
    prompt: str
    payload: dict[str, Any]
    output_schema: dict[str, Any]


_ANSWER_BEARING_KEYS = {
    "answer",
    "answerkey",
    "answervalue",
    "answercontract",
    "canonicalanswerzh",
    "certificate",
    "certificates",
    "claim",
    "claims",
    "claimsheet",
    "oracle",
    "groundtruth",
    "expectedanswer",
    "rationale",
}


def _normalized_key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def assert_answer_blind_payload(payload: Mapping[str, Any]) -> None:
    """Reject accidental answer/certificate leakage before invoking a model.

    This is a structural gate rather than a prompt promise: a future API
    backend cannot receive answer-bearing fields even if a caller constructs a
    malformed payload by hand.
    """

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized = _normalized_key(key)
                if normalized in _ANSWER_BEARING_KEYS or any(
                    marker in normalized
                    for marker in (
                        "answer",
                        "certificate",
                        "claimsheet",
                        "groundtruth",
                        "oracle",
                    )
                ):
                    raise ValueError(f"answer-blind payload contains forbidden field {path}.{key}")
                walk(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(payload, "$")


def question_output_schema(blueprint: QuestionBlueprint | Mapping[str, Any]) -> dict[str, Any]:
    payload = (
        blueprint.answer_blind_dict()
        if isinstance(blueprint, QuestionBlueprint)
        else dict(blueprint)
    )
    request_id = str(payload["request_id"])
    strategies = [str(value) for value in payload.get("allowed_strategies", [])]
    if not strategies:
        raise ValueError("question blueprint must allow at least one strategy")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["request_id", "strategy_id", "template_zh", "used_slots"],
        "properties": {
            "request_id": {"type": "string", "enum": [request_id]},
            "strategy_id": {"type": "string", "enum": strategies},
            "template_zh": {"type": "string", "minLength": 4, "maxLength": 240},
            "used_slots": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    }


def answer_output_schema(claim_sheet: ClaimSheet | Mapping[str, Any]) -> dict[str, Any]:
    payload = (
        claim_sheet.as_dict(include_answer=True)
        if isinstance(claim_sheet, ClaimSheet)
        else dict(claim_sheet)
    )
    answer_contract = payload.get("answer_contract")
    if not isinstance(answer_contract, Mapping):
        raise ValueError("answer-stage payload is missing answer_contract")
    request_id = str(payload.get("claim_sheet_id", ""))
    answer_key = str(answer_contract.get("answer_key", ""))
    allowed_claim_ids = [str(value) for value in payload.get("allowed_claim_ids", [])]
    if not request_id or not answer_key or not allowed_claim_ids:
        raise ValueError("claim sheet must provide an id, answer key, and allowed claims")
    sentence_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["role", "text", "claim_ids"],
        "properties": {
            "role": {
                "type": "string",
                "enum": ["evidence", "transform", "conclusion", "calibration"],
            },
            "text": {"type": "string", "minLength": 1, "maxLength": 180},
            "claim_ids": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "enum": allowed_claim_ids},
            },
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "request_id",
            "strategy_id",
            "answer_key",
            "surface_answer_zh",
            "sentences",
        ],
        "properties": {
            "request_id": {"type": "string", "enum": [request_id]},
            "strategy_id": {"type": "string", "minLength": 1, "maxLength": 80},
            # Some deterministic answer keys are canonical JSON strings and
            # therefore contain quotes. Codex strict structured outputs reject
            # quoted enum literals, so exact equality is enforced by the local
            # answer validator after generation rather than provider schema.
            "answer_key": {"type": "string", "minLength": 1, "maxLength": 512},
            "surface_answer_zh": {"type": "string", "minLength": 1, "maxLength": 420},
            "sentences": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": sentence_schema,
            },
        },
    }


def critic_output_schema(sentence_count: int) -> dict[str, Any]:
    if sentence_count < 1:
        raise ValueError("critic requires at least one sentence")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["supported", "unsupported_sentence_indices", "reason_codes"],
        "properties": {
            "supported": {"type": "boolean"},
            "unsupported_sentence_indices": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": sentence_count - 1},
            },
            "reason_codes": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "unsupported_claim",
                        "wrong_direction",
                        "wrong_number",
                        "wrong_epistemic_scope",
                        "geometry_jargon",
                        "unverifiable_inference",
                    ],
                },
            },
        },
    }


_QUESTION_PROMPT = """\
你是空间 episode 数据的“问题编辑”，只负责把结构化意图写成自然、简洁的中文提问。

严格边界：
1. 你没有答案，也不得猜答案；输入特意不含答案、几何真值、claim 或 certificate。
2. template_zh 必须保留 {{slot_name}} 形式的保护槽。required_slots 中每个槽恰好出现一次；不得新增槽。
3. 不得把 slot_values 的值直接抄进 template_zh；值由确定性程序在模型返回后填入。
4. 只问 intent_zh 指定的一个连贯问题，不添加新的事实前提，不暗示答案。
5. 用生活化表达，不提 operation graph、DAG、canonical state、坐标注册、证书或内部字段。
6. strategy_id 只能从 allowed_strategies 选择，used_slots 必须准确列出模板中使用的槽名。

只返回符合给定 JSON Schema 的 JSON。"""


_ANSWER_PROMPT = """\
你是空间 episode 数据的“回答编辑”。几何程序已经算出事实；你的唯一工作是把给定 claim sheet 写成自然、短而有信息量的中文回答。

严格边界：
1. 只能陈述 claims 中的内容，不能用常识补全房间，不能引入未列出的物体、方向、距离或视角。
2. 每个句子都必须列出直接支撑它的 claim_ids；不得挂靠无关 claim。所有 required_claim_ids 都必须被覆盖。
3. answer_key 必须原样复制 answer_contract.answer_key；它是机器校验字段，不要在正文解释。
4. surface_answer_zh 必须与 sentences.text 按顺序直接拼接后的文字完全一致。
5. 总共 1–3 个短句：先给结论；只有跨视图组合题才补一句关键观察或变换。不要写冗长文本 CoT。
6. unknown 时只能说“给定画面没有观察到/证据不足，无法确定”，不得断言整个场景中不存在该物体。
7. 数值、单位和方向只能使用各句所引用 claim 明确许可的形式。
8. 不提 canonical、世界坐标、OBB/AABB、operation graph、claim、certificate、oracle、真值、坐标注册等内部术语。

只返回符合给定 JSON Schema 的 JSON。"""


_CRITIC_PROMPT = """\
你是严格的事实审计器。逐句对照 claim sheet：若一个句子的任何实体、方向、数值、存在性或推断没有被它列出的 claim_ids 直接支持，就把该句索引列为 unsupported。unknown 回答若把“没观察到”说成“场景不存在”，也必须拒绝。不要改写答案，只返回 JSON 审计结果。"""


def build_question_request(blueprint: QuestionBlueprint | Mapping[str, Any]) -> PromptRequest:
    payload = (
        blueprint.answer_blind_dict()
        if isinstance(blueprint, QuestionBlueprint)
        else dict(blueprint)
    )
    assert_answer_blind_payload(payload)
    return PromptRequest(
        stage=QUESTION_STAGE,
        prompt=_QUESTION_PROMPT,
        payload=payload,
        output_schema=question_output_schema(payload),
    )


def build_answer_request(
    claim_sheet: ClaimSheet | Mapping[str, Any],
    *,
    question_zh: str,
    strategy_id: str,
) -> PromptRequest:
    sheet_payload = (
        claim_sheet.as_dict(include_answer=True)
        if isinstance(claim_sheet, ClaimSheet)
        else dict(claim_sheet)
    )
    payload = {
        "schema_version": "epispace.answer_realization_request.v1",
        "question_zh": str(question_zh),
        "strategy_id": str(strategy_id),
        "claim_sheet": sheet_payload,
    }
    return PromptRequest(
        stage=ANSWER_STAGE,
        prompt=_ANSWER_PROMPT,
        payload=payload,
        output_schema=answer_output_schema(sheet_payload),
    )


def build_critic_request(
    claim_sheet: ClaimSheet | Mapping[str, Any],
    answer: Mapping[str, Any],
) -> PromptRequest:
    sheet_payload = (
        claim_sheet.as_dict(include_answer=True)
        if isinstance(claim_sheet, ClaimSheet)
        else dict(claim_sheet)
    )
    sentences = answer.get("sentences")
    if not isinstance(sentences, list) or not sentences:
        raise ValueError("critic input must contain non-empty sentences")
    payload = {
        "schema_version": "epispace.answer_critic_request.v1",
        "claim_sheet": sheet_payload,
        "candidate_answer": dict(answer),
    }
    return PromptRequest(
        stage=CRITIC_STAGE,
        prompt=_CRITIC_PROMPT,
        payload=payload,
        output_schema=critic_output_schema(len(sentences)),
    )


def _batch_schema(
    item_schemas: Sequence[Mapping[str, Any]],
    *,
    expected_strategy_ids: Sequence[str] = (),
) -> dict[str, Any]:
    if not item_schemas:
        raise ValueError("structured batch must contain at least one item")
    # Provider-side validation uses the union of per-item enums.  Exact
    # request→response matching remains a deterministic per-item gate after
    # the batch is unpacked; this keeps the schema compatible with Codex's
    # strict structured-output subset without trusting array order.
    request_ids: set[str] = set()
    strategy_ids: set[str] = {str(value) for value in expected_strategy_ids}
    claim_ids: set[str] = set()
    answer_mode = "answer_key" in item_schemas[0].get("properties", {})
    for schema in item_schemas:
        properties = schema["properties"]
        request_ids.update(str(value) for value in properties["request_id"].get("enum", []))
        strategy_ids.update(str(value) for value in properties["strategy_id"].get("enum", []))
        if answer_mode:
            sentence = properties["sentences"]["items"]
            claim_ids.update(
                str(value)
                for value in sentence["properties"]["claim_ids"]["items"].get("enum", [])
            )
    if answer_mode:
        item = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "request_id",
                "strategy_id",
                "answer_key",
                "surface_answer_zh",
                "sentences",
            ],
            "properties": {
                "request_id": {"type": "string", "enum": sorted(request_ids)},
                "strategy_id": {"type": "string", "enum": sorted(strategy_ids)},
                "answer_key": {"type": "string", "minLength": 1, "maxLength": 512},
                "surface_answer_zh": {"type": "string", "minLength": 1, "maxLength": 420},
                "sentences": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["role", "text", "claim_ids"],
                        "properties": {
                            "role": {
                                "type": "string",
                                "enum": ["evidence", "transform", "conclusion", "calibration"],
                            },
                            "text": {"type": "string", "minLength": 1, "maxLength": 180},
                            "claim_ids": {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "string", "enum": sorted(claim_ids)},
                            },
                        },
                    },
                },
            },
        }
    else:
        item = {
            "type": "object",
            "additionalProperties": False,
            "required": ["request_id", "strategy_id", "template_zh", "used_slots"],
            "properties": {
                "request_id": {"type": "string", "enum": sorted(request_ids)},
                "strategy_id": {"type": "string", "enum": sorted(strategy_ids)},
                "template_zh": {"type": "string", "minLength": 4, "maxLength": 240},
                "used_slots": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["responses"],
        "properties": {
            "responses": {
                "type": "array",
                "minItems": len(item_schemas),
                "maxItems": len(item_schemas),
                "items": item,
            }
        },
    }


def build_question_batch_request(
    blueprints: Sequence[QuestionBlueprint | Mapping[str, Any]],
) -> PromptRequest:
    payloads = [
        value.answer_blind_dict() if isinstance(value, QuestionBlueprint) else dict(value)
        for value in blueprints
    ]
    for payload in payloads:
        assert_answer_blind_payload(payload)
    schemas = [question_output_schema(payload) for payload in payloads]
    return PromptRequest(
        stage=f"{QUESTION_STAGE}_batch",
        prompt=(
            _QUESTION_PROMPT
            + "\n对 INPUT_JSON.items 中每项各生成一个 response；request_id 一一对应，"
            "不得遗漏、重复或合并。"
        ),
        payload={
            "schema_version": "epispace.question_batch_request.answer_blind.v1",
            "items": payloads,
        },
        output_schema=_batch_schema(schemas),
    )


def build_answer_batch_request(
    items: Sequence[tuple[ClaimSheet | Mapping[str, Any], str, str]],
) -> PromptRequest:
    payloads: list[dict[str, Any]] = []
    schemas: list[dict[str, Any]] = []
    for claim_sheet, question_zh, strategy_id in items:
        sheet_payload = (
            claim_sheet.as_dict(include_answer=True)
            if isinstance(claim_sheet, ClaimSheet)
            else dict(claim_sheet)
        )
        payloads.append(
            {
                "question_zh": str(question_zh),
                "strategy_id": str(strategy_id),
                "claim_sheet": sheet_payload,
            }
        )
        schemas.append(answer_output_schema(sheet_payload))
    return PromptRequest(
        stage=f"{ANSWER_STAGE}_batch",
        prompt=(
            _ANSWER_PROMPT
            + "\n对 INPUT_JSON.items 中每项各生成一个 response；request_id 必须等于该项"
            "claim_sheet.claim_sheet_id，顺序不限但不得遗漏、重复或合并。"
        ),
        payload={"schema_version": "epispace.answer_batch_request.v1", "items": payloads},
        output_schema=_batch_schema(
            schemas,
            expected_strategy_ids=[strategy_id for _, _, strategy_id in items],
        ),
    )


def build_critic_batch_request(
    items: Sequence[tuple[ClaimSheet | Mapping[str, Any], Mapping[str, Any]]],
) -> PromptRequest:
    """Build one independently addressable critic request for an episode batch."""

    if not items:
        raise ValueError("critic batch must contain at least one item")
    payloads: list[dict[str, Any]] = []
    request_ids: list[str] = []
    maximum_sentence_index = 0
    for claim_sheet, answer in items:
        sheet_payload = (
            claim_sheet.as_dict(include_answer=True)
            if isinstance(claim_sheet, ClaimSheet)
            else dict(claim_sheet)
        )
        sentences = answer.get("sentences")
        if not isinstance(sentences, list) or not sentences:
            raise ValueError("critic input must contain non-empty sentences")
        request_id = str(sheet_payload.get("claim_sheet_id", ""))
        if not request_id:
            raise ValueError("critic claim sheet has no id")
        request_ids.append(request_id)
        maximum_sentence_index = max(maximum_sentence_index, len(sentences) - 1)
        payloads.append(
            {
                "request_id": request_id,
                "claim_sheet": sheet_payload,
                "candidate_answer": dict(answer),
            }
        )
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("critic batch contains duplicate request ids")
    item_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "request_id",
            "supported",
            "unsupported_sentence_indices",
            "reason_codes",
        ],
        "properties": {
            "request_id": {"type": "string", "enum": sorted(request_ids)},
            "supported": {"type": "boolean"},
            "unsupported_sentence_indices": {
                "type": "array",
                "items": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": maximum_sentence_index,
                },
            },
            "reason_codes": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "unsupported_claim",
                        "wrong_direction",
                        "wrong_number",
                        "wrong_epistemic_scope",
                        "geometry_jargon",
                        "unverifiable_inference",
                    ],
                },
            },
        },
    }
    output_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["responses"],
        "properties": {
            "responses": {
                "type": "array",
                "minItems": len(items),
                "maxItems": len(items),
                "items": item_schema,
            }
        },
    }
    return PromptRequest(
        stage=f"{CRITIC_STAGE}_batch",
        prompt=(
            _CRITIC_PROMPT
            + "\n对 INPUT_JSON.items 中每项各审计一次；request_id 必须原样返回，"
            "不得遗漏或重复。"
        ),
        payload={"schema_version": "epispace.critic_batch_request.v1", "items": payloads},
        output_schema=output_schema,
    )
