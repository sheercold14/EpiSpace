"""Leakage-resistant parsing of free-form model answers for the core benchmark.

The parser converts a benchmark row and a free-form Chinese answer into the
structured ``answer_value`` consumed by :mod:`episode3d.benchmark_evaluator`.
It uses only the public task/program contract and query constants.  In
particular, target booleans, relations, answer status, and reference answers are
never used to decide a prediction.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Final

PARSER_SCHEMA_VERSION: Final = "epispace.model_output_parser.v1"
BENCHMARK_SCHEMA_VERSION: Final = "epispace.benchmark.v1"
INVALID_ANSWER_VALUE: Final = {"status": "invalid"}

_MISSING: Final = object()

_TASK_PROGRAMS: Final = {
    "counterfactual_verification": "counterfactual_cross_view.v1",
    "egocentric_relation": "egocentric_relation.v1",
    "evidence_presence_reveal": "evidence_presence_reveal.v1",
    "evidence_presence_unknown": "evidence_presence_unknown.v1",
    "target_view_prediction": "target_view_prediction.v1",
}

_RELATION_PATTERNS: Final = {
    "left_of": (
        r"\bleft_of\b",
        r"\bleft\b",
        r"左侧",
        r"左边",
        r"左方",
        r"左(?=[前后])",
    ),
    "right_of": (
        r"\bright_of\b",
        r"\bright\b",
        r"右侧",
        r"右边",
        r"右方",
        r"右(?=[前后])",
    ),
    "in_front_of": (
        r"\bin_front_of\b",
        r"\bfront\b",
        r"前方",
        r"前面",
    ),
    "behind": (
        r"\bbehind\b",
        r"\bback\b",
        r"后方",
        r"后面",
    ),
}

_UNKNOWN_PATTERNS: Final = (
    r"(?:还|目前|现在)?(?:无法|不能|不可以|难以)(?:确定|判断|得知)",
    r"(?:证据|信息|观察)不足",
    r"无法得知",
    r"无法确认",
    r"不清楚",
    r"不知道",
    r"未知",
    r"不能断言",
)

_VISIBLE_NEGATIVE_PATTERNS: Final = (
    r"看不(?:到|见)",
    r"(?:无法|不能|不会|不可以|不可能)(?:够)?看(?:到|见)",
    r"不可见",
    r"不在(?:视野|画面|视线)(?:中|里)?",
    r"不会出现",
)

_VISIBLE_POSITIVE_PATTERNS: Final = (
    r"(?<![不未没])(?:能|会|可以|能够)(?:够)?看(?:到|见)",
    r"(?<![不未没])看(?:到|见)了?",
    r"(?<!不)可见",
    r"会出现在(?:视野|画面|视线)(?:中|里)?",
)


def parse_model_output(
    benchmark_row: Mapping[str, Any], model_output: str
) -> dict[str, Any]:
    """Parse one free-form answer into an evaluator-ready prediction row.

    Parsing is fail-closed.  Unsupported/malformed rows, empty output,
    contradictory cues, missing claim relations, and out-of-schema answers all
    produce ``parse_status == "invalid"`` and the impossible sentinel
    ``{"status": "invalid"}`` as ``answer_value``.  The evaluator therefore
    scores them incorrect instead of silently dropping them.
    """

    record_id = benchmark_row.get("record_id")
    safe_record_id = record_id if isinstance(record_id, str) else ""
    if not isinstance(record_id, str) or not record_id:
        return _invalid(safe_record_id, "invalid_record_id")
    if not isinstance(model_output, str):
        return _invalid(record_id, "model_output_not_string")
    if not model_output.strip():
        return _invalid(record_id, "empty_model_output")

    task, contract_error = _task_contract(benchmark_row)
    if contract_error is not None:
        return _invalid(record_id, contract_error)

    parsed: Any = _MISSING
    reason = "unparseable_output"
    if task == "counterfactual_verification":
        parsed, reason = _parse_counterfactual(model_output)
    elif task == "egocentric_relation":
        parsed, reason = _parse_egocentric_relation(model_output)
    elif task == "evidence_presence_unknown":
        parsed, reason = _parse_presence_unknown(model_output)
    elif task == "evidence_presence_reveal":
        category = _presence_query_category(benchmark_row)
        if category is None:
            return _invalid(record_id, "missing_presence_query_category")
        parsed, reason = _parse_presence_reveal(
            model_output,
            category=category,
            query_surface=_presence_query_surface(benchmark_row),
        )
    elif task == "target_view_prediction":
        parsed, reason = _parse_target_view(model_output)

    if parsed is _MISSING:
        return _invalid(record_id, reason)
    return {
        "record_id": record_id,
        "answer_value": parsed,
        "parse_status": "parsed",
        "parser_schema_version": PARSER_SCHEMA_VERSION,
    }


def _task_contract(row: Mapping[str, Any]) -> tuple[str, str | None]:
    schema_version = row.get("schema_version")
    if schema_version != BENCHMARK_SCHEMA_VERSION:
        return "", "unsupported_benchmark_schema"

    target = row.get("target")
    if not isinstance(target, Mapping):
        return "", "missing_task_contract"
    task = target.get("task_type")
    if not isinstance(task, str) or task not in _TASK_PROGRAMS:
        return "", "unsupported_task"

    program = row.get("program")
    program_id = program.get("program_id") if isinstance(program, Mapping) else None
    if program_id != _TASK_PROGRAMS[task]:
        return "", "task_program_schema_mismatch"
    return task, None


def _invalid(record_id: str, reason: str) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "answer_value": dict(INVALID_ANSWER_VALUE),
        "parse_status": "invalid",
        "parse_error": reason,
        "parser_schema_version": PARSER_SCHEMA_VERSION,
    }


def _parse_counterfactual(text: str) -> tuple[Any, str]:
    value = _structured_value(text)
    if isinstance(value, Mapping):
        claim_correct = value.get("claim_correct")
        relation = _relation_from_value(value.get("relation", _MISSING))
        if type(claim_correct) is bool and relation is not None:
            return {"claim_correct": claim_correct, "relation": relation}, ""
        return _MISSING, "invalid_structured_claim"

    answer_text = value if isinstance(value, str) else _answer_segment(text)
    claim_correct = _claim_correctness(answer_text)
    if claim_correct is None:
        return _MISSING, "missing_or_ambiguous_claim_correctness"
    relation = _relation_from_text(answer_text)
    if relation is None:
        return _MISSING, "missing_or_ambiguous_relation"
    return {"claim_correct": claim_correct, "relation": relation}, ""


def _parse_egocentric_relation(text: str) -> tuple[Any, str]:
    value = _structured_value(text)
    if value is not _MISSING:
        if isinstance(value, Mapping):
            value = value.get("relation", _MISSING)
        relation = _relation_from_value(value)
        if relation is None:
            return _MISSING, "invalid_structured_relation"
        return relation, ""

    relation = _relation_from_text(_answer_segment(text))
    if relation is None:
        return _MISSING, "missing_or_ambiguous_relation"
    return relation, ""


def _parse_presence_unknown(text: str) -> tuple[Any, str]:
    value = _structured_value(text)
    if value is None:
        return None, ""
    if isinstance(value, Mapping) and value.get("status") == "unknown":
        return None, ""
    answer_text = value if isinstance(value, str) else _answer_segment(text)
    if _has_any(answer_text, _UNKNOWN_PATTERNS):
        return None, ""
    return _MISSING, "missing_explicit_epistemic_unknown"


def _parse_presence_reveal(
    text: str, *, category: str, query_surface: str | None
) -> tuple[Any, str]:
    value = _structured_value(text)
    if isinstance(value, Mapping):
        status = value.get("status")
        supplied_category = value.get("category")
        if status != "present":
            return _MISSING, "presence_status_not_present"
        if supplied_category is not None and supplied_category not in {
            category,
            query_surface,
        }:
            return _MISSING, "presence_category_mismatch"
        return {"status": "present", "category": category}, ""
    if type(value) is bool:
        if value:
            return {"status": "present", "category": category}, ""
        return _MISSING, "presence_status_not_present"

    answer_text = value if isinstance(value, str) else _answer_segment(text)
    if _has_any(answer_text, _UNKNOWN_PATTERNS):
        return _MISSING, "presence_answer_is_unknown"
    visibility = _visibility_value(answer_text)
    presence_asserted = _positive_presence_assertion(answer_text) or any(
        surface
        and re.search(rf"(?<!没)有\s*{re.escape(surface)}", answer_text, re.IGNORECASE)
        for surface in (query_surface, category)
    )
    if visibility is False or _negative_presence_assertion(answer_text):
        return _MISSING, "presence_answer_is_negative"
    if visibility is True or presence_asserted:
        return {"status": "present", "category": category}, ""
    return _MISSING, "missing_present_assertion"


def _parse_target_view(text: str) -> tuple[Any, str]:
    value = _structured_value(text)
    if value is not _MISSING:
        if type(value) is bool:
            return value, ""
        if isinstance(value, Mapping) and type(value.get("visible")) is bool:
            return value["visible"], ""
        if not isinstance(value, str):
            return _MISSING, "invalid_structured_visibility"
    answer_text = value if isinstance(value, str) else _answer_segment(text)
    if _has_any(answer_text, _UNKNOWN_PATTERNS):
        return _MISSING, "target_view_answer_is_unknown"
    visible = _visibility_value(answer_text)
    if visible is None:
        return _MISSING, "missing_or_ambiguous_visibility"
    return visible, ""


def _structured_value(text: str) -> Any:
    candidates = [text.strip()]
    fenced = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(reversed(fenced))
    final_segment = _answer_segment(text)
    if final_segment != text.strip():
        candidates.append(final_segment)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, Mapping) and "answer_value" in value:
            return value["answer_value"]
        return value
    return _MISSING


def _answer_segment(text: str) -> str:
    # Preserve underscores because they are semantic inside canonical enum
    # values such as ``in_front_of`` and ``left_of``.
    stripped = re.sub(r"[*`]", "", text).strip()
    markers = list(
        re.finditer(
            r"(?:最终答案|答案|结论)\s*(?:是|为)?\s*[:：]\s*",
            stripped,
            flags=re.IGNORECASE,
        )
    )
    return stripped[markers[-1].end() :].strip() if markers else stripped


def _relation_from_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return _relation_from_text(value)


def _relation_from_text(text: str) -> str | None:
    candidate = text
    corrections = list(
        re.finditer(
            r"(?:而是|实际(?:上)?|其实|正确(?:的)?(?:空间)?关系\s*(?:是|为)|"
            r"应(?:该)?\s*(?:是|为))",
            candidate,
        )
    )
    if corrections:
        corrected = candidate[corrections[-1].end() :]
        corrected_relations = _relations_in(corrected)
        if len(corrected_relations) == 1:
            return next(iter(corrected_relations))
        if len(corrected_relations) > 1:
            return None
    relations = _relations_in(candidate)
    return next(iter(relations)) if len(relations) == 1 else None


def _relations_in(text: str) -> set[str]:
    lowered = text.lower()
    return {
        relation
        for relation, patterns in _RELATION_PATTERNS.items()
        if any(re.search(pattern, lowered) for pattern in patterns)
    }


def _claim_correctness(text: str) -> bool | None:
    false_patterns = (
        r"^\s*(?:不对|错(?:误|的)?|不正确|不是|否)(?:\s|[，,。.;；！!]|$)",
        r"(?:这个|该|此)?说法[^。；，,]{0,8}(?:不对|错误|有误|不正确|不成立)",
    )
    true_patterns = (
        r"^\s*(?:对(?:的)?|正确(?!关系|方向|位置)|是(?:的)?|没错|属实)"
        r"(?:\s|[，,。.;；！!]|$)",
        r"(?:这个|该|此)?说法[^。；，,]{0,8}(?<!不)(?:正确|成立|属实)",
    )
    negative = _has_any(text, false_patterns)
    positive = _has_any(text, true_patterns)
    if negative == positive:
        return None
    return not negative


def _visibility_value(text: str) -> bool | None:
    lowered = text.lower().strip()
    negative_pattern = "|".join(f"(?:{pattern})" for pattern in _VISIBLE_NEGATIVE_PATTERNS)
    negative = bool(re.search(negative_pattern, lowered, flags=re.IGNORECASE)) or bool(
        re.fullmatch(r"(?:不|否|不能|不会|不可以|没有|false|no)[。.!！]?", lowered)
    )
    # Do not count the nested ``看到`` inside ``不会看到`` as a simultaneous
    # positive cue.  Genuine contradictions survive outside the removed span.
    positive_search_text = re.sub(negative_pattern, "", lowered, flags=re.IGNORECASE)
    positive = _has_any(positive_search_text, _VISIBLE_POSITIVE_PATTERNS) or bool(
        re.fullmatch(r"(?:是的?|能|可以|会|有|对|true|yes)[。.!！]?", lowered)
    )
    if negative == positive:
        return None
    return not negative


def _positive_presence_assertion(text: str) -> bool:
    patterns = (
        r"(?<![不未没])(?:能|可以|能够)确定",
        r"(?<![不未没])(?:确实|明确)?存在",
        r"(?<![不未没])有(?:视觉)?证据",
        r"^\s*(?:是的?|有|对)[。.!！]?\s*$",
    )
    return _has_any(text, patterns)


def _negative_presence_assertion(text: str) -> bool:
    return _has_any(text, (r"不(?:存在|成立)", r"没有(?:这个|该)?(?:物体|目标|东西)?"))


def _has_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _presence_query_category(row: Mapping[str, Any]) -> str | None:
    for container_name in ("query_arguments", "query_args"):
        container = row.get(container_name)
        if isinstance(container, Mapping):
            category = container.get("category")
            if isinstance(category, str) and category:
                return category

    query = row.get("query")
    if isinstance(query, Mapping):
        arguments = query.get("arguments")
        if isinstance(arguments, Mapping):
            category = arguments.get("category")
            if isinstance(category, str) and category:
                return category

    # Frozen benchmark v1 does not duplicate the canonical query category in a
    # public query_arguments field.  For reveal rows only, target.answer_value's
    # category is the non-decision schema constant already named in the prompt.
    target = row.get("target")
    answer_value = target.get("answer_value") if isinstance(target, Mapping) else None
    category = answer_value.get("category") if isinstance(answer_value, Mapping) else None
    return category if isinstance(category, str) and category else None


def _presence_query_surface(row: Mapping[str, Any]) -> str | None:
    model_input = row.get("model_input")
    if not isinstance(model_input, list):
        return None
    texts: list[str] = []
    for message in model_input:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(
                str(item["text"])
                for item in content
                if isinstance(item, Mapping)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            )
    matches = re.findall(r"存在\s*([^？?。；;]+)", "\n".join(texts))
    return matches[-1].strip() if matches else None
