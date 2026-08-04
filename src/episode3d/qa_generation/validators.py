"""Fail-closed validators for model-written spatial questions and answers."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from episode3d.qa_generation.schemas import ClaimSheet, QuestionBlueprint, ValidationReport


class QAValidationError(ValueError):
    """Raised when generated language fails a deterministic release gate."""


_SLOT_RE = re.compile(r"\{\{([A-Za-z][A-Za-z0-9_]*)\}\}")
_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")
_VIEW_NUMBER_RE = re.compile(r"第\s*([-+]?\d+)\s*(?:个)?(?:视角|画面|帧)")
_PAIRED_VIEW_NUMBER_RE = re.compile(
    r"第\s*([-+]?\d+)\s*(?:个)?\s*(?:和|与|、)\s*"
    r"第\s*([-+]?\d+)\s*(?:个)?(?:视角|画面|帧)"
)
_UNIT_RE = re.compile(r"平方厘米|平方米|毫米|厘米|千米|米|度|°")
_CHINESE_COUNT_RE = re.compile(
    r"(?<![同单])(?<!任何)[零〇一二两三四五六七八九十百千半]+\s*(?:个)?"
    r"(?:视角|画面|帧|毫米|厘米|千米|米|度)"
)

_GEOMETRY_JARGON = (
    "canonical",
    "规范坐标",
    "世界坐标",
    "全局坐标",
    "obb",
    "aabb",
    "operation graph",
    "操作图",
    "typed graph",
    "dag",
    "claim_id",
    "claim sheet",
    "claim",
    "certificate",
    "几何证书",
    "oracle",
    "ground truth",
    "真值",
    "注册到锚定",
    "坐标注册",
    "锚定平面",
    "g→",
    "f→",
    "b→",
)

# Greedy matching is important: ``左前方`` is one relation, not two unrelated
# tokens (``左`` and ``前方``).
_DIRECTION_SURFACES: dict[str, tuple[str, ...]] = {
    "front-left": ("左前方", "左前侧", "前方偏左", "左前"),
    "front-right": ("右前方", "右前侧", "前方偏右", "右前"),
    "back-left": ("左后方", "左后侧", "后方偏左", "左后"),
    "back-right": ("右后方", "右后侧", "后方偏右", "右后"),
    "left": ("左侧", "左边", "偏左"),
    "right": ("右侧", "右边", "偏右"),
    "front": ("正前方", "前方", "前侧", "前面"),
    "back": ("正后方", "后方", "后侧", "后面"),
    "above": ("正上方", "上方", "上侧", "上面"),
    "below": ("正下方", "下方", "下侧", "下面"),
    "near": ("更近", "较近", "附近"),
    "far": ("更远", "较远", "远处"),
    "between": ("两者之间", "中间"),
}

_DIRECTION_ALIASES: dict[str, str] = {
    "left": "left",
    "left_of": "left",
    "right": "right",
    "right_of": "right",
    "front": "front",
    "in_front_of": "front",
    "front_of": "front",
    "behind": "back",
    "back": "back",
    "above": "above",
    "over": "above",
    "below": "below",
    "under": "below",
    "near": "near",
    "far": "far",
    "front_left": "front-left",
    "left_front": "front-left",
    "front-left": "front-left",
    "front_right": "front-right",
    "right_front": "front-right",
    "front-right": "front-right",
    "back_left": "back-left",
    "left_back": "back-left",
    "back-left": "back-left",
    "back_right": "back-right",
    "right_back": "back-right",
    "back-right": "back-right",
    "between": "between",
}
for _canonical, _surfaces in _DIRECTION_SURFACES.items():
    for _surface in _surfaces:
        _DIRECTION_ALIASES[_surface.casefold()] = _canonical

_SORTED_DIRECTION_SURFACES = tuple(
    sorted(
        {surface for surfaces in _DIRECTION_SURFACES.values() for surface in surfaces},
        key=len,
        reverse=True,
    )
)
_DIRECTION_RE = re.compile("|".join(re.escape(value) for value in _SORTED_DIRECTION_SURFACES))


def _sheet_dict(claim_sheet: ClaimSheet | Mapping[str, Any]) -> dict[str, Any]:
    return (
        claim_sheet.as_dict(include_answer=True)
        if isinstance(claim_sheet, ClaimSheet)
        else dict(claim_sheet)
    )


def _blueprint_dict(blueprint: QuestionBlueprint | Mapping[str, Any]) -> dict[str, Any]:
    return (
        blueprint.answer_blind_dict()
        if isinstance(blueprint, QuestionBlueprint)
        else dict(blueprint)
    )


def _record(
    checks: dict[str, bool], errors: list[str], name: str, condition: bool, message: str
) -> None:
    checks[name] = bool(condition)
    if not condition:
        errors.append(message)


def _contains_jargon(text: str) -> list[str]:
    folded = text.casefold()
    return [term for term in _GEOMETRY_JARGON if term in folded]


def validate_question_response(
    blueprint: QuestionBlueprint | Mapping[str, Any],
    response: Mapping[str, Any] | Any,
    *,
    corpus: Sequence[str] = (),
    dedup_variable_terms: Iterable[str] = (),
    dedup_threshold: float = 0.85,
) -> ValidationReport:
    """Validate a question template before protected slots are rendered."""

    checks: dict[str, bool] = {}
    errors: list[str] = []
    try:
        spec = _blueprint_dict(blueprint)
        if not isinstance(response, Mapping):
            raise TypeError("question response is not an object")
        template = response.get("template_zh")
        used_slots = response.get("used_slots")
        if not isinstance(template, str) or not isinstance(used_slots, list):
            raise TypeError("question response lacks template_zh or used_slots")
        if not all(isinstance(value, str) for value in used_slots):
            raise TypeError("used_slots must contain strings")

        _record(
            checks,
            errors,
            "request_id",
            response.get("request_id") == spec.get("request_id"),
            "question request_id does not match the blueprint",
        )
        allowed_strategies = set(spec.get("allowed_strategies", []))
        _record(
            checks,
            errors,
            "strategy_allowlist",
            response.get("strategy_id") in allowed_strategies,
            "question strategy_id is not allow-listed",
        )

        slot_values = spec.get("slot_values", {})
        required_slots = list(spec.get("required_slots", []))
        if not isinstance(slot_values, Mapping):
            raise TypeError("blueprint slot_values is not an object")
        placeholders = _SLOT_RE.findall(template)
        unknown_slots = sorted(set(placeholders) - set(slot_values))
        slot_counts = {slot: placeholders.count(slot) for slot in required_slots}
        slots_once = not unknown_slots and all(count == 1 for count in slot_counts.values())
        _record(
            checks,
            errors,
            "required_slots_once",
            slots_once,
            f"protected slots must appear exactly once; counts={slot_counts}, unknown={unknown_slots}",
        )
        _record(
            checks,
            errors,
            "used_slots_exact",
            len(used_slots) == len(set(used_slots))
            and set(used_slots) == set(placeholders)
            and set(required_slots) <= set(used_slots),
            "used_slots does not exactly describe the protected placeholders",
        )
        leaked_values = [
            str(value)
            for value in slot_values.values()
            if str(value).strip() and str(value).strip() in template
        ]
        _record(
            checks,
            errors,
            "protected_values_not_inlined",
            not leaked_values,
            f"question writer inlined protected slot values: {leaked_values}",
        )
        jargon = _contains_jargon(template)
        _record(
            checks,
            errors,
            "no_geometry_jargon",
            not jargon,
            f"question contains internal geometry jargon: {jargon}",
        )
        dedup = corpus_dedup_result(
            template,
            corpus,
            variable_terms=dedup_variable_terms,
            threshold=dedup_threshold,
        )
        _record(
            checks,
            errors,
            "corpus_ngram_dedup",
            dedup["passed"],
            f"question is too similar to corpus item {dedup['match_index']} "
            f"(score={dedup['max_similarity']:.3f})",
        )
        checks["well_formed"] = True
    except (KeyError, TypeError, ValueError) as error:
        checks["well_formed"] = False
        errors.append(f"malformed question response: {error}")
    return ValidationReport(
        passed=all(checks.values()) and not errors, checks=checks, errors=errors
    )


def render_question_template(
    blueprint: QuestionBlueprint | Mapping[str, Any], response: Mapping[str, Any]
) -> str:
    """Validate, then deterministically fill protected question slots."""

    report = validate_question_response(blueprint, response)
    require_valid(report)
    spec = _blueprint_dict(blueprint)
    rendered = str(response["template_zh"])
    for slot, value in spec["slot_values"].items():
        rendered = rendered.replace(f"{{{{{slot}}}}}", str(value))
    if _SLOT_RE.search(rendered):
        raise QAValidationError("rendered question still contains an unresolved slot")
    return rendered


def _claim_lookup(sheet: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    claims = sheet.get("claims")
    if not isinstance(claims, list):
        raise TypeError("claim sheet claims is not an array")
    lookup: dict[str, Mapping[str, Any]] = {}
    for claim in claims:
        if not isinstance(claim, Mapping) or not isinstance(claim.get("claim_id"), str):
            raise TypeError("claim sheet contains a malformed claim")
        claim_id = str(claim["claim_id"])
        if claim_id in lookup:
            raise ValueError(f"duplicate claim id {claim_id}")
        lookup[claim_id] = claim
    return lookup


def _canonical_direction(value: str) -> str | None:
    folded = value.strip().casefold().replace(" ", "_")
    return _DIRECTION_ALIASES.get(folded) or _DIRECTION_ALIASES.get(value.strip().casefold())


def _directions_in_text(text: str) -> list[tuple[str, str]]:
    return [
        (match.group(0), _DIRECTION_ALIASES[match.group(0).casefold()])
        for match in _DIRECTION_RE.finditer(text)
    ]


def _allowed_direction_codes(claims: Iterable[Mapping[str, Any]]) -> set[str]:
    result: set[str] = set()
    for claim in claims:
        values = claim.get("allowed_directions", [])
        if not isinstance(values, list):
            continue
        for value in values:
            canonical = _canonical_direction(str(value))
            if canonical:
                result.add(canonical)
    return result


def _normalized_surface(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).casefold()


def _numeric_permissions(
    claims: Iterable[Mapping[str, Any]],
) -> tuple[set[str], set[str], set[str]]:
    numeric_tokens: set[str] = set()
    units: set[str] = set()
    literal_counts: set[str] = set()
    for claim in claims:
        surfaces = claim.get("numeric_surfaces", [])
        if isinstance(surfaces, list):
            for item in surfaces:
                if not isinstance(item, Mapping):
                    continue
                surface = str(item.get("surface", ""))
                numeric_tokens.update(_NUMBER_RE.findall(surface))
                units.update(_UNIT_RE.findall(surface))
                unit = str(item.get("unit", "")).strip()
                if unit:
                    units.add(unit)
                literal_counts.update(_CHINESE_COUNT_RE.findall(surface))
        statement = str(claim.get("statement_zh", ""))
        # View/order counts can be naturally verbalized and are still directly
        # grounded by this claim. Metric numbers must use numeric_surfaces.
        literal_counts.update(_CHINESE_COUNT_RE.findall(statement))
    return numeric_tokens, units, {_normalized_surface(value) for value in literal_counts}


def _sentence_numeric_errors(
    text: str,
    referenced_claims: Sequence[Mapping[str, Any]],
    *,
    exposure_count: int,
) -> list[str]:
    errors: list[str] = []
    allowed_numbers, allowed_units, allowed_chinese = _numeric_permissions(referenced_claims)
    view_spans: list[tuple[int, int]] = []
    for match in _PAIRED_VIEW_NUMBER_RE.finditer(text):
        for group_index in (1, 2):
            index = int(match.group(group_index))
            if index < 1 or index > exposure_count:
                errors.append(f"view index {index} is outside the exposed episode")
            view_spans.append(match.span(group_index))
    for match in _VIEW_NUMBER_RE.finditer(text):
        index = int(match.group(1))
        if index < 1 or index > exposure_count:
            errors.append(f"view index {index} is outside the exposed episode")
        view_spans.append(match.span(1))

    for match in _NUMBER_RE.finditer(text):
        is_view_index = any(
            start <= match.start() and match.end() <= end for start, end in view_spans
        )
        if not is_view_index and match.group(0) not in allowed_numbers:
            errors.append(f"number {match.group(0)!r} is not allow-listed by referenced claims")
    for unit in _UNIT_RE.findall(text):
        if unit not in allowed_units:
            errors.append(f"unit {unit!r} is not allow-listed by referenced claims")
    for count in _CHINESE_COUNT_RE.findall(text):
        if _normalized_surface(count) not in allowed_chinese:
            errors.append(f"count surface {count!r} is not allow-listed by referenced claims")
    return errors


def _unknown_scope_errors(text: str) -> list[str]:
    errors: list[str] = []
    safe_nonexistence_context = re.compile(
        r"(?:不等于|不代表|并不意味(?:着)?|不能说明|无法说明|不说明)不存在"
    )
    assertion_probe = safe_nonexistence_context.sub("", text)
    if "不存在" in assertion_probe or "根本没有" in assertion_probe:
        errors.append("unknown answer asserts object non-existence")
    scene_absence = re.compile(
        r"(?:场景|房间|住宅|屋子|屋里|空间)(?:中|里|内)?[^。！？]{0,10}"
        r"(?<!有)没有(?!\s*(?:观察到|看到|足够|充分|证据))"
    )
    if scene_absence.search(text):
        errors.append("unknown answer turns observed absence into scene-level absence")
    if not re.search(r"无法确定|不能确定|尚不能确定|无法判断|不能判断|证据不足|信息不足", text):
        errors.append("unknown answer does not explicitly calibrate uncertainty")
    return errors


def validate_answer_response(
    claim_sheet: ClaimSheet | Mapping[str, Any],
    response: Mapping[str, Any] | Any,
    *,
    expected_strategy_id: str | None = None,
    corpus: Sequence[str] = (),
    dedup_variable_terms: Iterable[str] = (),
    dedup_threshold: float = 0.85,
) -> ValidationReport:
    """Validate claim-grounded natural language without trusting the backend."""

    checks: dict[str, bool] = {}
    errors: list[str] = []
    try:
        sheet = _sheet_dict(claim_sheet)
        if not isinstance(response, Mapping):
            raise TypeError("answer response is not an object")
        lookup = _claim_lookup(sheet)
        answer_contract = sheet.get("answer_contract")
        sentences = response.get("sentences")
        surface = response.get("surface_answer_zh")
        if not isinstance(answer_contract, Mapping):
            raise TypeError("claim sheet has no answer contract")
        if not isinstance(sentences, list) or not 1 <= len(sentences) <= 3:
            raise TypeError("answer must contain one to three sentences")
        if not isinstance(surface, str):
            raise TypeError("surface_answer_zh is not a string")

        _record(
            checks,
            errors,
            "request_id",
            response.get("request_id") == sheet.get("claim_sheet_id"),
            "answer request_id does not match claim_sheet_id",
        )
        _record(
            checks,
            errors,
            "answer_key",
            response.get("answer_key") == answer_contract.get("answer_key"),
            "answer_key does not match deterministic geometry output",
        )
        if expected_strategy_id is not None:
            _record(
                checks,
                errors,
                "strategy_id",
                response.get("strategy_id") == expected_strategy_id,
                "answer strategy_id differs from the selected question strategy",
            )

        all_claim_ids: list[str] = []
        malformed_sentence = False
        unanchored_sentence = False
        unallowlisted: set[str] = set()
        direction_errors: list[str] = []
        numeric_errors: list[str] = []
        texts: list[str] = []
        exposure_count = len(sheet.get("exposure_view_ids", []))
        for index, sentence in enumerate(sentences):
            if not isinstance(sentence, Mapping):
                malformed_sentence = True
                continue
            text = sentence.get("text")
            claim_ids = sentence.get("claim_ids")
            if not isinstance(text, str) or not text.strip() or not isinstance(claim_ids, list):
                malformed_sentence = True
                continue
            texts.append(text)
            if not claim_ids or not all(isinstance(value, str) for value in claim_ids):
                unanchored_sentence = True
                continue
            if len(claim_ids) != len(set(claim_ids)):
                unanchored_sentence = True
            all_claim_ids.extend(claim_ids)
            unallowlisted.update(value for value in claim_ids if value not in lookup)
            referenced = [lookup[value] for value in claim_ids if value in lookup]
            allowed_directions = _allowed_direction_codes(referenced)
            for surface_term, direction in _directions_in_text(text):
                if direction not in allowed_directions:
                    direction_errors.append(
                        f"sentence {index} direction {surface_term!r} is not licensed by its claims"
                    )
            numeric_errors.extend(
                f"sentence {index}: {message}"
                for message in _sentence_numeric_errors(
                    text, referenced, exposure_count=exposure_count
                )
            )

        _record(
            checks,
            errors,
            "sentence_claim_anchoring",
            not malformed_sentence and not unanchored_sentence,
            "every sentence must be non-empty and cite at least one unique claim id",
        )
        _record(
            checks,
            errors,
            "claim_id_allowlist",
            not unallowlisted,
            f"answer cites claims outside the sheet: {sorted(unallowlisted)}",
        )
        required = set(sheet.get("required_claim_ids", []))
        if not required:
            required = {
                claim_id for claim_id, claim in lookup.items() if bool(claim.get("required"))
            }
        covered = set(all_claim_ids)
        _record(
            checks,
            errors,
            "required_claim_coverage",
            required <= covered,
            f"answer omits required claims: {sorted(required - covered)}",
        )
        _record(
            checks,
            errors,
            "surface_matches_sentences",
            surface == "".join(texts),
            "surface_answer_zh must exactly equal the concatenated sentence text",
        )
        _record(
            checks,
            errors,
            "direction_allowlist",
            not direction_errors,
            "; ".join(direction_errors) or "direction allow-list violation",
        )
        _record(
            checks,
            errors,
            "numeric_unit_allowlist",
            not numeric_errors,
            "; ".join(numeric_errors) or "number/unit allow-list violation",
        )

        answer_status = str(answer_contract.get("answer_status", ""))
        unknown_errors = _unknown_scope_errors(surface) if answer_status == "unknown" else []
        _record(
            checks,
            errors,
            "unknown_epistemic_scope",
            not unknown_errors,
            "; ".join(unknown_errors) or "unknown answer violates epistemic scope",
        )
        jargon = _contains_jargon(surface)
        _record(
            checks,
            errors,
            "no_geometry_jargon",
            not jargon,
            f"answer exposes internal geometry jargon: {jargon}",
        )
        dedup = corpus_dedup_result(
            surface,
            corpus,
            variable_terms=dedup_variable_terms,
            threshold=dedup_threshold,
        )
        _record(
            checks,
            errors,
            "corpus_ngram_dedup",
            dedup["passed"],
            f"answer is too similar to corpus item {dedup['match_index']} "
            f"(score={dedup['max_similarity']:.3f})",
        )
        checks["well_formed"] = not malformed_sentence
    except (KeyError, TypeError, ValueError) as error:
        checks["well_formed"] = False
        errors.append(f"malformed answer response: {error}")
    return ValidationReport(
        passed=all(checks.values()) and not errors, checks=checks, errors=errors
    )


def validate_critic_response(response: Mapping[str, Any] | Any) -> ValidationReport:
    checks: dict[str, bool] = {}
    errors: list[str] = []
    try:
        if not isinstance(response, Mapping):
            raise TypeError("critic response is not an object")
        indices = response.get("unsupported_sentence_indices")
        reasons = response.get("reason_codes")
        if not isinstance(indices, list) or not isinstance(reasons, list):
            raise TypeError("critic response arrays are missing")
        _record(
            checks,
            errors,
            "critic_arrays_unique",
            len(indices) == len(set(indices)) and len(reasons) == len(set(reasons)),
            "critic response contains duplicate indices or reason codes",
        )
        _record(
            checks,
            errors,
            "critic_support",
            response.get("supported") is True and not indices and not reasons,
            f"critic rejected generated answer: indices={indices}, reasons={reasons}",
        )
        checks["well_formed"] = True
    except (TypeError, ValueError) as error:
        checks["well_formed"] = False
        errors.append(f"malformed critic response: {error}")
    return ValidationReport(
        passed=all(checks.values()) and not errors, checks=checks, errors=errors
    )


def normalize_for_dedup(text: str, *, variable_terms: Iterable[str] = ()) -> str:
    """Normalize template variables before corpus-level character n-gram checks."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    terms = sorted(
        {str(value).strip().casefold() for value in variable_terms if str(value).strip()},
        key=len,
        reverse=True,
    )
    for term in terms:
        normalized = normalized.replace(term, "¤")
    normalized = _DIRECTION_RE.sub("方", normalized)
    normalized = _NUMBER_RE.sub("数", normalized)
    normalized = re.sub(r"\{\{[A-Za-z][A-Za-z0-9_]*\}\}", "槽", normalized)
    return "".join(
        character for character in normalized if character.isalnum() or character in "¤方数槽"
    )


def character_ngrams(text: str, *, n: int = 4) -> set[str]:
    if n < 1:
        raise ValueError("n must be positive")
    if not text:
        return set()
    if len(text) <= n:
        return {text}
    return {text[index : index + n] for index in range(len(text) - n + 1)}


def ngram_similarity(left: str, right: str, *, n: int = 4) -> float:
    left_ngrams = character_ngrams(left, n=n)
    right_ngrams = character_ngrams(right, n=n)
    if not left_ngrams and not right_ngrams:
        return 1.0
    if not left_ngrams or not right_ngrams:
        return 0.0
    return 2.0 * len(left_ngrams & right_ngrams) / (len(left_ngrams) + len(right_ngrams))


def corpus_dedup_result(
    candidate: str,
    corpus: Sequence[str],
    *,
    variable_terms: Iterable[str] = (),
    threshold: float = 0.85,
    n: int = 4,
) -> dict[str, Any]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("dedup threshold must be between zero and one")
    candidate_normalized = normalize_for_dedup(candidate, variable_terms=variable_terms)
    maximum = 0.0
    match_index: int | None = None
    for index, existing in enumerate(corpus):
        existing_normalized = normalize_for_dedup(existing, variable_terms=variable_terms)
        score = ngram_similarity(candidate_normalized, existing_normalized, n=n)
        if score > maximum:
            maximum = score
            match_index = index
    return {
        "passed": match_index is None or maximum < threshold,
        "max_similarity": maximum,
        "match_index": match_index,
        "threshold": threshold,
        "ngram_n": n,
    }


def require_valid(report: ValidationReport) -> None:
    if not report.passed:
        raise QAValidationError("; ".join(report.errors) or "QA validation failed")


# Short aliases keep pipeline call sites readable.
validate_question = validate_question_response
validate_answer = validate_answer_response
validate_critic = validate_critic_response
render_question = render_question_template
