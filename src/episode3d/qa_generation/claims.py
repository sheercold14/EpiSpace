"""Compile simulator-certified question facts into model-facing claim sheets.

The language model is deliberately downstream of this module.  It may choose
how to express a claim, but it cannot create spatial truth: every claim below
is replayed from ``answer_value``, a passed certificate check, certificate
oracle geometry, or the exact observation/evidence bindings in the Episode
IR.  Unsupported or internally inconsistent records fail closed.

Unknown/abstention records receive stricter treatment.  Their claim sheets
contain only facts about the *exposed observations* and the epistemic status;
world truth, held-out decisive views, and hidden entity bindings are never
copied into a claim or its provenance paths.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from episode3d.language import entity_name, relation_name
from episode3d.qa_generation.capabilities import capability_for
from episode3d.qa_generation.schemas import (
    Claim,
    ClaimSheet,
    NumericSurface,
    canonical_json,
    content_id,
)


class ClaimCompilationError(ValueError):
    """A question cannot be converted into a trustworthy claim sheet."""


_KNOWN_STATUS = "accepted"
_UNKNOWN_STATUS = "unknown"
_UNKNOWN_TASKS = frozenset(
    {
        "unknown_abstention",
        "evidence_presence_unknown",
        "occlusion_unknown",
    }
)

_DIRECTION_SURFACES: dict[str, tuple[str, ...]] = {
    "left_of": ("left_of", "左侧", "左边"),
    "right_of": ("right_of", "右侧", "右边"),
    "in_front_of": ("in_front_of", "前方", "前面"),
    "behind": ("behind", "后方", "后面"),
    "front_left": ("front_left", "左前方", "前方偏左"),
    "front_right": ("front_right", "右前方", "前方偏右"),
    "back_left": ("back_left", "左后方", "后方偏左"),
    "back_right": ("back_right", "右后方", "后方偏右"),
}


def _fail(context: str, message: str) -> ClaimCompilationError:
    return ClaimCompilationError(f"{context}: {message}")


def _mapping(value: Any, path: str, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(context, f"{path} must be an object")
    return value


def _sequence(value: Any, path: str, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _fail(context, f"{path} must be an array")
    return value


def _string(value: Any, path: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(context, f"{path} must be a non-empty string")
    return value.strip()


def _boolean(value: Any, path: str, context: str) -> bool:
    if not isinstance(value, bool):
        raise _fail(context, f"{path} must be a boolean")
    return value


def _number(value: Any, path: str, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(context, f"{path} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise _fail(context, f"{path} must be a finite number")
    return number


def _integer(value: Any, path: str, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(context, f"{path} must be an integer")
    return value


def _string_tuple(value: Any, path: str, context: str) -> tuple[str, ...]:
    values = _sequence(value, path, context)
    result = tuple(_string(item, f"{path}[{index}]", context) for index, item in enumerate(values))
    if len(set(result)) != len(result):
        raise _fail(context, f"{path} contains duplicate values")
    return result


def _numeric_vector(
    value: Any, path: str, context: str, *, length: int
) -> tuple[float, ...]:
    values = _sequence(value, path, context)
    if len(values) != length:
        raise _fail(context, f"{path} must contain exactly {length} numbers")
    return tuple(_number(item, f"{path}[{index}]", context) for index, item in enumerate(values))


def _close(left: float, right: float, *, tolerance: float = 1e-5) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def _direction_surfaces(relation: str, context: str) -> tuple[str, ...]:
    try:
        return _DIRECTION_SURFACES[relation]
    except KeyError as error:
        raise _fail(context, f"unsupported spatial relation {relation!r}") from error


def _relation_from_deltas(right: float, front: float, context: str) -> str:
    if _close(abs(right), abs(front), tolerance=1e-9):
        raise _fail(context, "right/front deltas are directionally ambiguous")
    if abs(right) > abs(front):
        return "right_of" if right > 0 else "left_of"
    return "in_front_of" if front > 0 else "behind"


def _quadrant_from_deltas(right: float, front: float, context: str) -> str:
    if _close(right, 0.0) or _close(front, 0.0):
        raise _fail(context, "object-centric offset lies on a quadrant boundary")
    horizontal = "right" if right > 0 else "left"
    vertical = "front" if front > 0 else "back"
    return f"{vertical}_{horizontal}"


def _source_hash(episode: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(episode).encode("utf-8")).hexdigest()


def _answer_key(value: Any, status: str) -> str:
    if status == _UNKNOWN_STATUS:
        return "unknown"
    if isinstance(value, str):
        return value
    return canonical_json(value)


def _metre_surface(value: float, *, digits: int = 1, prefix: str = "约") -> NumericSurface:
    rounded = round(value, digits)
    return NumericSurface(
        value=rounded,
        unit="m",
        tolerance=0.5 * 10 ** (-digits) + 1e-9,
        surface=f"{prefix}{rounded:.{digits}f}米",
    )


def _degree_surface(value: float, *, digits: int = 0, prefix: str = "约") -> NumericSurface:
    rounded = round(value, digits)
    rendered = f"{rounded:.{digits}f}" if digits else f"{rounded:.0f}"
    return NumericSurface(
        value=rounded,
        unit="deg",
        tolerance=0.5 * 10 ** (-digits) + 1e-9,
        surface=f"{prefix}{rendered}度",
    )


@dataclass
class _Context:
    episode: Mapping[str, Any]
    question: Mapping[str, Any]
    exposure_view_ids: tuple[str, ...]
    episode_id: str
    fact_id: str
    task_type: str
    program_id: str
    semantic_signature: str
    answer_status: str
    answer_value: Any
    answer_zh: str
    evidence_view_ids: tuple[str, ...]
    evidence_entity_ids: tuple[str, ...]
    certificate: Mapping[str, Any]
    frame_id: str
    observation_by_id: dict[str, Mapping[str, Any]]
    checks: dict[str, Mapping[str, Any]]

    @property
    def label(self) -> str:
        return f"episode {self.episode_id}, fact {self.fact_id}, task {self.task_type}"

    def check(self, name: str) -> Mapping[str, Any]:
        try:
            check = self.checks[name]
        except KeyError as error:
            raise _fail(self.label, f"certificate check {name!r} is required") from error
        if check.get("passed") is not True:
            raise _fail(self.label, f"certificate check {name!r} did not pass")
        return check

    def observation(self, view_id: str) -> Mapping[str, Any]:
        try:
            return self.observation_by_id[view_id]
        except KeyError as error:
            raise _fail(self.label, f"unknown observation view {view_id!r}") from error

    def known(self) -> None:
        if self.answer_status != _KNOWN_STATUS:
            raise _fail(self.label, f"known task requires answer_status={_KNOWN_STATUS!r}")
        if self.certificate.get("result") != "pass":
            raise _fail(self.label, "known task requires certificate.result='pass'")

    def unknown(self) -> None:
        if self.answer_status != _UNKNOWN_STATUS:
            raise _fail(self.label, f"unknown task requires answer_status={_UNKNOWN_STATUS!r}")
        if self.answer_value is not None:
            raise _fail(self.label, "unknown task must have answer_value=null")
        if self.certificate.get("result") != "unknown":
            raise _fail(self.label, "unknown task requires certificate.result='unknown'")
        source_views = _string_tuple(
            self.question.get("model_view_ids"), "question.model_view_ids", self.label
        )
        if self.exposure_view_ids != source_views:
            raise _fail(
                self.label,
                "unknown evidence profile cannot be expanded or reordered; "
                "exposure_view_ids must exactly equal question.model_view_ids",
            )


def _make_context(
    episode: Mapping[str, Any],
    question: Mapping[str, Any],
    exposure_view_ids: Sequence[str] | None,
) -> _Context:
    episode_id = _string(episode.get("episode_id"), "episode.episode_id", "episode")
    fact_id = _string(question.get("fact_id"), "question.fact_id", f"episode {episode_id}")
    task_type = _string(
        question.get("task_type"), "question.task_type", f"episode {episode_id}, fact {fact_id}"
    )
    label = f"episode {episode_id}, fact {fact_id}, task {task_type}"

    program = _mapping(question.get("program"), "question.program", label)
    program_id = _string(program.get("program_id"), "question.program.program_id", label)
    signature = _string(
        program.get("semantic_signature"), "question.program.semantic_signature", label
    )
    certificate = _mapping(question.get("certificate"), "question.certificate", label)
    answer_status = _string(question.get("answer_status"), "question.answer_status", label)
    answer_zh = _string(question.get("answer_zh"), "question.answer_zh", label)
    evidence_views = _string_tuple(
        question.get("evidence_view_ids"), "question.evidence_view_ids", label
    )
    evidence_entities = _string_tuple(
        question.get("evidence_entity_ids"), "question.evidence_entity_ids", label
    )
    model_views = _string_tuple(question.get("model_view_ids"), "question.model_view_ids", label)
    exposed = (
        model_views
        if exposure_view_ids is None
        else _string_tuple(exposure_view_ids, "exposure_view_ids", label)
    )

    observations = _sequence(episode.get("observations"), "episode.observations", label)
    observation_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(observations):
        observation = _mapping(raw, f"episode.observations[{index}]", label)
        view_id = _string(
            observation.get("view_id"), f"episode.observations[{index}].view_id", label
        )
        if view_id in observation_by_id:
            raise _fail(label, f"duplicate observation view_id {view_id!r}")
        observation_by_id[view_id] = observation
    if not exposed:
        raise _fail(label, "exposure_view_ids must not be empty")
    missing_exposure = sorted(set(exposed) - set(observation_by_id))
    if missing_exposure:
        raise _fail(label, f"exposure contains unknown views: {missing_exposure}")
    missing_evidence = sorted(set(evidence_views) - set(exposed))
    if missing_evidence:
        raise _fail(label, f"evidence views are not exposed: {missing_evidence}")

    belief = _mapping(episode.get("observable_belief"), "episode.observable_belief", label)
    frame_id = _string(belief.get("frame_id"), "episode.observable_belief.frame_id", label)

    check_values = _sequence(certificate.get("checks"), "question.certificate.checks", label)
    checks: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(check_values):
        check = _mapping(raw, f"question.certificate.checks[{index}]", label)
        name = _string(check.get("name"), f"question.certificate.checks[{index}].name", label)
        if name in checks:
            raise _fail(label, f"duplicate certificate check {name!r}")
        checks[name] = check

    # If the full episode carries its question inventory, make accidental
    # cross-episode pairing impossible while still permitting minimal fixtures.
    raw_questions = episode.get("questions")
    if raw_questions is not None:
        episode_questions = _sequence(raw_questions, "episode.questions", label)
        matches = [
            item
            for item in episode_questions
            if isinstance(item, Mapping) and item.get("fact_id") == fact_id
        ]
        if len(matches) != 1:
            raise _fail(label, "fact_id must occur exactly once in episode.questions")

    return _Context(
        episode=episode,
        question=question,
        exposure_view_ids=exposed,
        episode_id=episode_id,
        fact_id=fact_id,
        task_type=task_type,
        program_id=program_id,
        semantic_signature=signature,
        answer_status=answer_status,
        answer_value=question.get("answer_value"),
        answer_zh=answer_zh,
        evidence_view_ids=evidence_views,
        evidence_entity_ids=evidence_entities,
        certificate=certificate,
        frame_id=frame_id,
        observation_by_id=observation_by_id,
        checks=checks,
    )


def _claim(
    context: _Context,
    *,
    kind: str,
    statement_zh: str,
    value: Any,
    required: bool,
    source_paths: Sequence[str],
    evidence_view_ids: Sequence[str] | None = None,
    evidence_entity_ids: Sequence[str] | None = None,
    frame_id: str | None = None,
    allowed_directions: Sequence[str] = (),
    numeric_surfaces: Sequence[NumericSurface] = (),
    epistemic_scope: str = "observed_evidence",
) -> Claim:
    paths = tuple(_string(path, "claim.source_paths", context.label) for path in source_paths)
    if not paths:
        raise _fail(context.label, f"claim {kind!r} has no provenance source_paths")
    views = tuple(context.evidence_view_ids if evidence_view_ids is None else evidence_view_ids)
    entities = tuple(
        context.evidence_entity_ids if evidence_entity_ids is None else evidence_entity_ids
    )
    claim_payload = {
        "fact_id": context.fact_id,
        "kind": kind,
        "value": value,
        "required": required,
        "source_paths": paths,
        "views": views,
        "entities": entities,
        "frame_id": frame_id,
        "epistemic_scope": epistemic_scope,
    }
    return Claim(
        claim_id=content_id("claim", claim_payload),
        kind=kind,
        statement_zh=_string(statement_zh, "claim.statement_zh", context.label),
        value=value,
        required=required,
        evidence_view_ids=views,
        evidence_entity_ids=entities,
        source_paths=paths,
        frame_id=frame_id,
        allowed_directions=tuple(allowed_directions),
        numeric_surfaces=tuple(numeric_surfaces),
        epistemic_scope=epistemic_scope,
    )


def _relation_claim(
    context: _Context,
    *,
    relation: str,
    kind: str,
    statement_prefix: str,
    source_paths: Sequence[str],
    frame_id: str,
    required: bool = True,
) -> Claim:
    surfaces = _direction_surfaces(relation, context.label)
    return _claim(
        context,
        kind=kind,
        statement_zh=f"{statement_prefix}{relation_name(relation)}。",
        value=relation,
        required=required,
        source_paths=source_paths,
        frame_id=frame_id,
        allowed_directions=surfaces,
        epistemic_scope="geometry_certificate",
    )


def _build_metric(context: _Context) -> list[Claim]:
    context.known()
    answer = _number(context.answer_value, "question.answer_value", context.label)
    measured_check = context.check("center_distance_m")
    measured = _number(
        measured_check.get("measured_value"),
        "certificate.checks[center_distance_m].measured_value",
        context.label,
    )
    if not _close(answer, round(measured, 1), tolerance=1e-9):
        raise _fail(context.label, "answer distance is not the 0.1 m rounding of certificate distance")
    oracle = _mapping(context.certificate.get("oracle"), "question.certificate.oracle", context.label)
    subject = _numeric_vector(
        oracle.get("subject_center_m"), "certificate.oracle.subject_center_m", context.label, length=3
    )
    reference = _numeric_vector(
        oracle.get("reference_center_m"),
        "certificate.oracle.reference_center_m",
        context.label,
        length=3,
    )
    replayed = math.dist(subject, reference)
    if not _close(replayed, measured, tolerance=1e-5):
        raise _fail(context.label, "certificate center distance does not replay from oracle centers")
    if len(context.evidence_entity_ids) != 2:
        raise _fail(context.label, "metric_distance requires exactly two evidence entities")
    return [
        _claim(
            context,
            kind="metric_center_distance",
            statement_zh=f"两件证据物体的三维中心距离约为{answer:.1f}米。",
            value=answer,
            required=True,
            source_paths=(
                "question.answer_value",
                "question.certificate.checks[name=center_distance_m].measured_value",
                "question.certificate.oracle.subject_center_m",
                "question.certificate.oracle.reference_center_m",
            ),
            frame_id=context.frame_id,
            numeric_surfaces=(_metre_surface(answer),),
            epistemic_scope="geometry_certificate",
        )
    ]


def _build_egocentric(context: _Context) -> list[Claim]:
    context.known()
    relation = _string(context.answer_value, "question.answer_value", context.label)
    _direction_surfaces(relation, context.label)
    if len(context.evidence_entity_ids) != 2:
        raise _fail(context.label, "egocentric_relation requires exactly two evidence entities")

    claims: list[Claim] = []
    if "camera_frame_relation_recomputed" in context.checks:
        check = context.check("camera_frame_relation_recomputed")
        certified = _string(
            check.get("relation"),
            "certificate.checks[camera_frame_relation_recomputed].relation",
            context.label,
        )
        anchor = _string(
            check.get("anchor_view_id"),
            "certificate.checks[camera_frame_relation_recomputed].anchor_view_id",
            context.label,
        )
        context.observation(anchor)
        if certified != relation:
            raise _fail(context.label, "answer relation disagrees with recomputed camera-frame relation")
        if anchor not in context.evidence_view_ids:
            raise _fail(context.label, "camera-frame anchor is not an evidence view")
        yaw = _number(
            check.get("yaw_separation_deg"),
            "certificate.checks[camera_frame_relation_recomputed].yaw_separation_deg",
            context.label,
        )
        frame_id = f"camera@{anchor}"
        display_ordinal = context.exposure_view_ids.index(anchor) + 1
        claims.append(
            _claim(
                context,
                kind="camera_frame_selection",
                statement_zh=f"本题方向以第{display_ordinal}个视角的相机朝向为正前方。",
                value={
                    "anchor_view_id": anchor,
                    "display_ordinal": display_ordinal,
                    "paired_yaw_separation_deg": yaw,
                },
                required=False,
                source_paths=(
                    "question.certificate.checks[name=camera_frame_relation_recomputed].anchor_view_id",
                    "question.certificate.checks[name=camera_frame_relation_recomputed].yaw_separation_deg",
                ),
                frame_id=frame_id,
                allowed_directions=_direction_surfaces("in_front_of", context.label),
                numeric_surfaces=(_degree_surface(yaw),),
                epistemic_scope="geometry_certificate",
            )
        )
        relation_paths = (
            "question.answer_value",
            "question.certificate.checks[name=camera_frame_relation_recomputed].relation",
        )
    else:
        context.check("ego_axis_margin_m")
        oracle = _mapping(
            context.certificate.get("oracle"), "question.certificate.oracle", context.label
        )
        right = _number(
            oracle.get("ego_right_delta_m"), "certificate.oracle.ego_right_delta_m", context.label
        )
        front = _number(
            oracle.get("ego_front_delta_m"), "certificate.oracle.ego_front_delta_m", context.label
        )
        replayed = _relation_from_deltas(right, front, context.label)
        if replayed != relation:
            raise _fail(context.label, "answer relation does not replay from egocentric deltas")
        if len(context.evidence_view_ids) != 1:
            raise _fail(context.label, "canonical egocentric relation requires one evidence view")
        anchor = context.evidence_view_ids[0]
        frame_id = f"camera@{anchor}"
        relation_paths = (
            "question.answer_value",
            "question.certificate.oracle.ego_right_delta_m",
            "question.certificate.oracle.ego_front_delta_m",
        )

    claims.append(
        _relation_claim(
            context,
            relation=relation,
            kind="egocentric_relation",
            statement_prefix="在指定相机朝向下，主体位于参照物的",
            source_paths=relation_paths,
            frame_id=frame_id,
        )
    )
    return claims


def _cross_view_oracle(
    context: _Context, question: Mapping[str, Any] | None = None
) -> tuple[Mapping[str, Any], str, float, float, float]:
    source = context.question if question is None else question
    certificate = _mapping(source.get("certificate"), "cross-view question.certificate", context.label)
    oracle = _mapping(certificate.get("oracle"), "cross-view certificate.oracle", context.label)
    right = _number(
        oracle.get("canonical_right_delta_m"),
        "cross-view certificate.oracle.canonical_right_delta_m",
        context.label,
    )
    front = _number(
        oracle.get("canonical_front_delta_m"),
        "cross-view certificate.oracle.canonical_front_delta_m",
        context.label,
    )
    distance = _number(
        oracle.get("center_distance_m"),
        "cross-view certificate.oracle.center_distance_m",
        context.label,
    )
    relation = _relation_from_deltas(right, front, context.label)
    # Full 3-D center distance may include height; it can exceed the planar
    # norm, but never be materially smaller than it.
    if (
        not _close(math.hypot(right, front), distance, tolerance=0.25)
        and distance + 1e-5 < math.hypot(right, front)
    ):
        raise _fail(context.label, "cross-view distance is smaller than its planar displacement")
    return oracle, relation, right, front, distance


def _build_cross_view(context: _Context) -> list[Claim]:
    context.known()
    relation = _string(context.answer_value, "question.answer_value", context.label)
    context.check("never_co_visible")
    context.check("canonical_axis_margin_m")
    if len(context.evidence_entity_ids) != 2:
        raise _fail(context.label, "cross_view_relation requires exactly two evidence entities")
    oracle, replayed, _right, _front, distance = _cross_view_oracle(context)
    if relation != replayed:
        raise _fail(context.label, "answer relation does not replay from canonical deltas")
    subject_steps = tuple(
        _integer(value, "certificate.oracle.subject_visible_steps", context.label)
        for value in _sequence(
            oracle.get("subject_visible_steps"),
            "certificate.oracle.subject_visible_steps",
            context.label,
        )
    )
    reference_steps = tuple(
        _integer(value, "certificate.oracle.reference_visible_steps", context.label)
        for value in _sequence(
            oracle.get("reference_visible_steps"),
            "certificate.oracle.reference_visible_steps",
            context.label,
        )
    )
    if not subject_steps or not reference_steps or set(subject_steps) & set(reference_steps):
        raise _fail(context.label, "cross-view visible-step sets must be non-empty and disjoint")
    return [
        _claim(
            context,
            kind="cross_view_evidence_partition",
            statement_zh="两个目标从未在同一画面里同时出现，需要整合不同视角的观察。",
            value={
                "never_co_visible": True,
                "subject_visible_steps": list(subject_steps),
                "reference_visible_steps": list(reference_steps),
            },
            required=True,
            source_paths=(
                "question.certificate.checks[name=never_co_visible].measured_value",
                "question.certificate.oracle.subject_visible_steps",
                "question.certificate.oracle.reference_visible_steps",
            ),
            frame_id=context.frame_id,
            epistemic_scope="geometry_certificate",
        ),
        _relation_claim(
            context,
            relation=relation,
            kind="canonical_cross_view_relation",
            statement_prefix="注册到第一个视角定义的共同平面后，主体位于参照物的",
            source_paths=(
                "question.answer_value",
                "question.certificate.oracle.canonical_right_delta_m",
                "question.certificate.oracle.canonical_front_delta_m",
            ),
            frame_id=context.frame_id,
        ),
        _claim(
            context,
            kind="cross_view_center_distance",
            statement_zh=f"两件物体的三维中心距离约为{distance:.1f}米。",
            value=round(distance, 1),
            required=False,
            source_paths=("question.certificate.oracle.center_distance_m",),
            frame_id=context.frame_id,
            numeric_surfaces=(_metre_surface(distance),),
            epistemic_scope="geometry_certificate",
        ),
    ]


def _linked_cross_view_question(context: _Context) -> Mapping[str, Any]:
    questions = _sequence(context.episode.get("questions"), "episode.questions", context.label)
    matches = [
        item
        for item in questions
        if isinstance(item, Mapping)
        and item.get("task_type") == "cross_view_relation"
        and tuple(item.get("evidence_entity_ids", ())) == context.evidence_entity_ids
    ]
    if len(matches) != 1:
        raise _fail(
            context.label,
            "counterfactual_verification requires exactly one linked cross_view_relation proof",
        )
    return matches[0]


def _build_counterfactual(context: _Context) -> list[Claim]:
    context.known()
    answer = _mapping(context.answer_value, "question.answer_value", context.label)
    correct = _boolean(answer.get("claim_correct"), "question.answer_value.claim_correct", context.label)
    relation = _string(answer.get("relation"), "question.answer_value.relation", context.label)
    _direction_surfaces(relation, context.label)
    oracle = _mapping(context.certificate.get("oracle"), "question.certificate.oracle", context.label)
    verified = _string(
        oracle.get("verified_relation"), "certificate.oracle.verified_relation", context.label
    )
    if relation != verified:
        raise _fail(context.label, "answer relation disagrees with verified_relation")

    if correct:
        check = context.check("claim_matches_verified_relation")
        claimed = _string(
            oracle.get("claimed_relation"), "certificate.oracle.claimed_relation", context.label
        )
        if claimed != verified or check.get("measured_value") != verified:
            raise _fail(context.label, "true counterfactual claim does not match verified relation")
        tested = claimed
        verdict_path = "question.certificate.checks[name=claim_matches_verified_relation]"
    else:
        check = context.check("claim_contradicted")
        rejected = _string(
            oracle.get("rejected_relation"), "certificate.oracle.rejected_relation", context.label
        )
        if rejected == verified:
            raise _fail(context.label, "false counterfactual must test a different relation")
        if check.get("measured_value") != verified or check.get("threshold") != rejected:
            raise _fail(context.label, "contradiction check disagrees with oracle relations")
        tested = rejected
        verdict_path = "question.certificate.checks[name=claim_contradicted]"

    context.check("never_co_visible_via_linked_task")
    linked = _linked_cross_view_question(context)
    linked_fact = _string(linked.get("fact_id"), "linked_cross_view.fact_id", context.label)
    linked_answer = _string(
        linked.get("answer_value"), "linked_cross_view.answer_value", context.label
    )
    linked_oracle, replayed, _right, _front, distance = _cross_view_oracle(context, linked)
    if linked_answer != verified or replayed != verified:
        raise _fail(context.label, "linked cross-view geometry disagrees with verified relation")
    if tuple(linked.get("evidence_view_ids", ())) != context.evidence_view_ids:
        raise _fail(context.label, "counterfactual and linked proof must share evidence views")

    related_prefix = f"episode.questions[fact_id={linked_fact}]"
    return [
        _claim(
            context,
            kind="counterfactual_verdict",
            statement_zh=(
                f"题中“{relation_name(tested)}”的说法{'成立' if correct else '不成立'}。"
            ),
            value={"claim_correct": correct, "tested_relation": tested},
            required=True,
            source_paths=("question.answer_value.claim_correct", verdict_path),
            frame_id=context.frame_id,
            allowed_directions=_direction_surfaces(tested, context.label),
            epistemic_scope="geometry_certificate",
        ),
        _relation_claim(
            context,
            relation=verified,
            kind="verified_canonical_relation",
            statement_prefix="跨视图几何验证得到的实际关系是主体位于参照物的",
            source_paths=(
                "question.answer_value.relation",
                "question.certificate.oracle.verified_relation",
                f"{related_prefix}.certificate.oracle.canonical_right_delta_m",
                f"{related_prefix}.certificate.oracle.canonical_front_delta_m",
            ),
            frame_id=context.frame_id,
        ),
        _claim(
            context,
            kind="linked_cross_view_distance",
            statement_zh=f"链接的跨视图证明中，两件物体中心相距约{distance:.1f}米。",
            value=round(distance, 1),
            required=False,
            source_paths=(f"{related_prefix}.certificate.oracle.center_distance_m",),
            frame_id=context.frame_id,
            numeric_surfaces=(_metre_surface(distance),),
            epistemic_scope="geometry_certificate",
        ),
        _claim(
            context,
            kind="linked_cross_view_evidence",
            statement_zh="该判定依赖没有同框的两段观察，而不是单帧关系。",
            value={
                "linked_fact_id": linked_fact,
                "subject_visible_steps": linked_oracle.get("subject_visible_steps"),
                "reference_visible_steps": linked_oracle.get("reference_visible_steps"),
            },
            required=False,
            source_paths=(
                "question.certificate.checks[name=never_co_visible_via_linked_task]",
                f"{related_prefix}.certificate.oracle.subject_visible_steps",
                f"{related_prefix}.certificate.oracle.reference_visible_steps",
            ),
            frame_id=context.frame_id,
            epistemic_scope="geometry_certificate",
        ),
    ]


def _build_last_seen(context: _Context) -> list[Claim]:
    context.known()
    answer = _integer(context.answer_value, "question.answer_value", context.label)
    if answer < 1:
        raise _fail(context.label, "last-seen answer must be a one-based positive view ordinal")
    last_check = context.check("last_visible_step")
    absent_check = context.check("absent_after_last_seen")
    last_step = _integer(
        last_check.get("measured_value"),
        "certificate.checks[last_visible_step].measured_value",
        context.label,
    )
    if answer != last_step + 1:
        raise _fail(context.label, "last-seen answer must equal zero-based step plus one")
    if absent_check.get("measured_value") is not True:
        raise _fail(context.label, "absent_after_last_seen must certify true")
    oracle = _mapping(context.certificate.get("oracle"), "question.certificate.oracle", context.label)
    visible_steps = tuple(
        _integer(value, "certificate.oracle.visible_steps", context.label)
        for value in _sequence(
            oracle.get("visible_steps"), "certificate.oracle.visible_steps", context.label
        )
    )
    if not visible_steps or max(visible_steps) != last_step:
        raise _fail(context.label, "last-visible step does not match oracle visibility history")
    matching_views = [
        view_id
        for view_id, observation in context.observation_by_id.items()
        if observation.get("step") == last_step
    ]
    if len(matching_views) != 1:
        raise _fail(context.label, "last-visible step must map to exactly one observation")
    last_view = matching_views[0]
    if last_view not in context.evidence_view_ids:
        raise _fail(context.label, "last-visible view is not bound as evidence")
    if len(context.evidence_entity_ids) != 1:
        raise _fail(context.label, "last_seen_memory requires exactly one evidence entity")
    ordinal = NumericSurface(
        value=float(answer),
        unit="view_ordinal",
        tolerance=0.0,
        surface=f"第{answer}个视角",
    )
    return [
        _claim(
            context,
            kind="last_seen_view",
            statement_zh=f"目标最后一次出现于第{answer}个视角。",
            value={"view_id": last_view, "step": last_step, "display_ordinal": answer},
            required=True,
            source_paths=(
                "question.answer_value",
                "question.certificate.checks[name=last_visible_step].measured_value",
                "question.certificate.oracle.visible_steps",
            ),
            frame_id="episode_order",
            numeric_surfaces=(ordinal,),
            epistemic_scope="observed_evidence",
        ),
        _claim(
            context,
            kind="absent_after_last_seen",
            statement_zh="在该视角之后的已给观察中，目标没有再次出现。",
            value=True,
            required=True,
            source_paths=(
                "question.certificate.checks[name=absent_after_last_seen].measured_value",
            ),
            frame_id="episode_order",
            epistemic_scope="observed_evidence",
        ),
    ]


def _build_orbit(context: _Context) -> list[Claim]:
    context.known()
    if context.answer_value is not True:
        raise _fail(context.label, "orbit_identity currently requires a certified positive identity")
    if len(context.evidence_view_ids) != 2 or len(context.evidence_entity_ids) != 1:
        raise _fail(context.label, "orbit_identity requires two views and one stable entity")
    identity = context.check("stable_focus_identity")
    entity_id = _string(
        identity.get("entity_id"),
        "certificate.checks[stable_focus_identity].entity_id",
        context.label,
    )
    if entity_id != context.evidence_entity_ids[0]:
        raise _fail(context.label, "stable orbit entity disagrees with evidence binding")
    separation_check = context.check("azimuth_separation_deg")
    separation = _number(
        separation_check.get("measured"),
        "certificate.checks[azimuth_separation_deg].measured",
        context.label,
    )
    context.check("focus_visible_in_both")
    frame = f"object_centric@{entity_id}"
    return [
        _claim(
            context,
            kind="orbit_viewpoint_change",
            statement_zh=f"两个观察方位围绕同一焦点相隔约{separation:.0f}度。",
            value={"azimuth_separation_deg": separation},
            required=True,
            source_paths=(
                "question.certificate.checks[name=azimuth_separation_deg].measured",
                "question.certificate.checks[name=focus_visible_in_both]",
            ),
            frame_id=frame,
            numeric_surfaces=(_degree_surface(separation),),
            epistemic_scope="geometry_certificate",
        ),
        _claim(
            context,
            kind="stable_object_identity",
            statement_zh="方位变化前后绑定的是同一个场景实体。",
            value=True,
            required=True,
            source_paths=(
                "question.answer_value",
                "question.certificate.checks[name=stable_focus_identity].entity_id",
            ),
            frame_id=frame,
            epistemic_scope="geometry_certificate",
        ),
    ]


def _build_rotation(context: _Context) -> list[Claim]:
    context.known()
    category = _string(context.answer_value, "question.answer_value", context.label)
    if len(context.evidence_view_ids) != 2 or len(context.evidence_entity_ids) != 1:
        raise _fail(context.label, "rotation_change_detection requires two views and one entered entity")
    context.check("fixed_camera_position")
    context.check("entity_absent_before_present_after")
    yaw_check = context.check("yaw_delta_deg")
    yaw = _number(
        yaw_check.get("measured"),
        "certificate.checks[yaw_delta_deg].measured",
        context.label,
    )
    entered = context.check("eligible_entered_entity_count")
    count = _integer(
        entered.get("measured"),
        "certificate.checks[eligible_entered_entity_count].measured",
        context.label,
    )
    entity_ids = _string_tuple(
        entered.get("entity_ids"),
        "certificate.checks[eligible_entered_entity_count].entity_ids",
        context.label,
    )
    if count != 1 or entity_ids != context.evidence_entity_ids:
        raise _fail(context.label, "rotation change must have exactly the bound entered entity")
    before, after = context.evidence_view_ids
    frame = f"camera_motion@{before}->{after}"
    return [
        _claim(
            context,
            kind="pure_camera_rotation",
            statement_zh=f"相机站位保持不变，只原地旋转了约{yaw:.0f}度。",
            value={"fixed_position": True, "yaw_delta_deg": yaw},
            required=True,
            source_paths=(
                "question.certificate.checks[name=fixed_camera_position]",
                "question.certificate.checks[name=yaw_delta_deg].measured",
            ),
            frame_id=frame,
            numeric_surfaces=(_degree_surface(yaw),),
            epistemic_scope="geometry_certificate",
        ),
        _claim(
            context,
            kind="unique_entered_entity",
            statement_zh=f"旋转后唯一新进入主要视野的物体是{entity_name(category)}。",
            value=category,
            required=True,
            source_paths=(
                "question.answer_value",
                "question.certificate.checks[name=entity_absent_before_present_after]",
                "question.certificate.checks[name=eligible_entered_entity_count].entity_ids",
            ),
            frame_id=frame,
            epistemic_scope="observed_evidence",
        ),
    ]


def _build_elevation(context: _Context) -> list[Claim]:
    context.known()
    relation = _string(context.answer_value, "question.answer_value", context.label)
    if len(context.evidence_view_ids) != 2 or len(context.evidence_entity_ids) != 2:
        raise _fail(context.label, "elevation transfer requires two views and two entities")
    station = context.check("same_station")
    context.check("entities_visible_both_heights")
    canonical = context.check("canonical_relation")
    certified = _string(
        canonical.get("relation"),
        "certificate.checks[canonical_relation].relation",
        context.label,
    )
    right = _number(
        canonical.get("delta_right_m"),
        "certificate.checks[canonical_relation].delta_right_m",
        context.label,
    )
    front = _number(
        canonical.get("delta_front_m"),
        "certificate.checks[canonical_relation].delta_front_m",
        context.label,
    )
    if certified != relation or _relation_from_deltas(right, front, context.label) != relation:
        raise _fail(context.label, "elevation relation does not replay from canonical deltas")
    anchor_check = context.check("canonical_frame")
    anchor = _string(
        anchor_check.get("anchor_view_id"),
        "certificate.checks[canonical_frame].anchor_view_id",
        context.label,
    )
    if anchor not in context.evidence_view_ids:
        raise _fail(context.label, "canonical elevation anchor is not exposed")
    heights = [
        _number(
            context.observation(view_id).get("camera_height_m"),
            f"episode.observations[view_id={view_id}].camera_height_m",
            context.label,
        )
        for view_id in context.evidence_view_ids
    ]
    low, high = min(heights), max(heights)
    if high - low < 0.1:
        raise _fail(context.label, "elevation views do not have a meaningful height change")
    frame = f"camera@{anchor}"
    station_id = _string(
        station.get("station_id"), "certificate.checks[same_station].station_id", context.label
    )
    return [
        _claim(
            context,
            kind="same_station_elevation_change",
            statement_zh=f"相机在同一站位由约{low:.1f}米升到约{high:.1f}米。",
            value={"station_id": station_id, "low_height_m": low, "high_height_m": high},
            required=True,
            source_paths=(
                "question.certificate.checks[name=same_station].station_id",
                f"episode.observations[view_id={context.evidence_view_ids[0]}].camera_height_m",
                f"episode.observations[view_id={context.evidence_view_ids[1]}].camera_height_m",
            ),
            frame_id=frame,
            numeric_surfaces=(_metre_surface(low), _metre_surface(high)),
            epistemic_scope="observed_evidence",
        ),
        _relation_claim(
            context,
            relation=relation,
            kind="elevation_invariant_relation",
            statement_prefix="回到低视角定义的共同水平坐标后，主体仍位于参照物的",
            source_paths=(
                "question.answer_value",
                "question.certificate.checks[name=canonical_relation].relation",
                "question.certificate.checks[name=canonical_relation].delta_right_m",
                "question.certificate.checks[name=canonical_relation].delta_front_m",
            ),
            frame_id=frame,
        ),
    ]


def _build_object_perspective(context: _Context) -> list[Claim]:
    context.known()
    quadrant = _string(context.answer_value, "question.answer_value", context.label)
    _direction_surfaces(quadrant, context.label)
    if len(context.evidence_entity_ids) != 3:
        raise _fail(context.label, "object-centric perspective requires origin, facing, and target")
    oracle = _mapping(context.certificate.get("oracle"), "question.certificate.oracle", context.label)
    right = _number(
        oracle.get("target_ego_right_m"),
        "certificate.oracle.target_ego_right_m",
        context.label,
    )
    front = _number(
        oracle.get("target_ego_front_m"),
        "certificate.oracle.target_ego_front_m",
        context.label,
    )
    if _quadrant_from_deltas(right, front, context.label) != quadrant:
        raise _fail(context.label, "object-centric quadrant does not replay from target offsets")
    right_check = context.check("ego_right_abs_m")
    front_check = context.check("ego_front_abs_m")
    if not _close(
        abs(right),
        _number(
            right_check.get("measured_value"),
            "certificate.checks[ego_right_abs_m].measured_value",
            context.label,
        ),
    ) or not _close(
        abs(front),
        _number(
            front_check.get("measured_value"),
            "certificate.checks[ego_front_abs_m].measured_value",
            context.label,
        ),
    ):
        raise _fail(context.label, "object-centric offset disagrees with certificate checks")
    origin, facing, _target = context.evidence_entity_ids
    frame = f"object_anchored@{origin}->{facing}"
    horizontal = "向右" if right > 0 else "向左"
    vertical = "向前" if front > 0 else "向后"
    horizontal_relation = "right_of" if right > 0 else "left_of"
    vertical_relation = "in_front_of" if front > 0 else "behind"
    offset_directions = tuple(
        dict.fromkeys(
            (
                *_direction_surfaces(quadrant, context.label),
                *_direction_surfaces(horizontal_relation, context.label),
                *_direction_surfaces(vertical_relation, context.label),
                horizontal,
                vertical,
            )
        )
    )
    return [
        _claim(
            context,
            kind="object_anchored_offset",
            statement_zh=(
                f"以起点指向朝向物体为正前方后，目标{horizontal}约{abs(right):.1f}米、"
                f"{vertical}约{abs(front):.1f}米。"
            ),
            value={"right_m": right, "front_m": front},
            required=True,
            source_paths=(
                "question.certificate.oracle.target_ego_right_m",
                "question.certificate.oracle.target_ego_front_m",
            ),
            frame_id=frame,
            allowed_directions=offset_directions,
            numeric_surfaces=(
                _metre_surface(abs(right), prefix=f"{horizontal}约"),
                _metre_surface(abs(front), prefix=f"{vertical}约"),
            ),
            epistemic_scope="geometry_certificate",
        ),
        _relation_claim(
            context,
            relation=quadrant,
            kind="object_anchored_quadrant",
            statement_prefix="因此，目标位于该假想观察者的",
            source_paths=(
                "question.answer_value",
                "question.certificate.oracle.target_ego_right_m",
                "question.certificate.oracle.target_ego_front_m",
            ),
            frame_id=frame,
        ),
    ]


def _translation_surface(axis: str, value: float, precision: float, context: str) -> NumericSurface:
    if precision <= 0:
        raise _fail(context, "translation surface precision must be positive")
    rounded = round(value / precision) * precision
    digits = max(0, round(-math.log10(precision)))
    magnitude = abs(rounded)
    rendered = f"{magnitude:.{digits}f}"
    if abs(rounded) < precision / 2:
        surface = "横向位置不变" if axis == "right" else "前后位置不变"
    elif axis == "right":
        surface = f"向{'右' if rounded > 0 else '左'}{rendered}米"
    else:
        surface = f"向{'前' if rounded > 0 else '后'}{rendered}米"
    return NumericSurface(
        value=rounded,
        unit="m",
        tolerance=precision / 2 + 1e-9,
        surface=surface,
    )


def _yaw_surface(value: float, precision: float, context: str) -> NumericSurface:
    if precision <= 0:
        raise _fail(context, "rotation surface precision must be positive")
    rounded = round(value / precision) * precision
    digits = max(0, round(-math.log10(precision)))
    rendered = f"{abs(rounded):.{digits}f}" if digits else f"{abs(rounded):.0f}"
    if abs(rounded) < precision / 2:
        surface = "朝向不变"
    else:
        surface = f"向{'左' if rounded > 0 else '右'}转{rendered}度"
    return NumericSurface(
        value=rounded,
        unit="deg",
        tolerance=precision / 2 + 1e-9,
        surface=surface,
    )


def _build_target_view(context: _Context) -> list[Claim]:
    context.known()
    answer = _boolean(context.answer_value, "question.answer_value", context.label)
    visibility = context.check("candidate_visibility_in_target_render")
    measured = _boolean(
        visibility.get("measured"),
        "certificate.checks[candidate_visibility_in_target_render].measured",
        context.label,
    )
    if measured != answer:
        raise _fail(context.label, "target-view answer disagrees with held-out render visibility")
    held_out = context.check("target_render_held_out")
    target_view = _string(
        held_out.get("target_view_id"),
        "certificate.checks[target_render_held_out].target_view_id",
        context.label,
    )
    oracle_target_view = _string(
        context.certificate.get("oracle_target_view_id"),
        "question.certificate.oracle_target_view_id",
        context.label,
    )
    if target_view != oracle_target_view:
        raise _fail(context.label, "held-out target-view identifiers disagree")
    candidate_id = _string(
        context.certificate.get("candidate_entity_id"),
        "question.certificate.candidate_entity_id",
        context.label,
    )
    if candidate_id not in context.evidence_entity_ids:
        raise _fail(context.label, "candidate entity is not present in the evidence binding")

    claims: list[Claim] = []
    if "relative_pose_instruction_exposed" in context.checks:
        pose = context.check("relative_pose_instruction_exposed")
        source_view = _string(
            pose.get("source_view_id"),
            "certificate.checks[relative_pose_instruction_exposed].source_view_id",
            context.label,
        )
        if source_view not in context.exposure_view_ids:
            raise _fail(context.label, "relative-pose source view is not exposed")
        right = _number(
            pose.get("right_m"),
            "certificate.checks[relative_pose_instruction_exposed].right_m",
            context.label,
        )
        front = _number(
            pose.get("front_m"),
            "certificate.checks[relative_pose_instruction_exposed].front_m",
            context.label,
        )
        yaw = _number(
            pose.get("yaw_delta_deg"),
            "certificate.checks[relative_pose_instruction_exposed].yaw_delta_deg",
            context.label,
        )
        translation_precision = _number(
            pose.get("surface_translation_precision_m"),
            "certificate.checks[relative_pose_instruction_exposed].surface_translation_precision_m",
            context.label,
        )
        rotation_precision = _number(
            pose.get("surface_rotation_precision_deg"),
            "certificate.checks[relative_pose_instruction_exposed].surface_rotation_precision_deg",
            context.label,
        )
        context.check("candidate_uniquely_anchored_in_source")
        numeric = (
            _translation_surface("right", right, translation_precision, context.label),
            _translation_surface("front", front, translation_precision, context.label),
            _yaw_surface(yaw, rotation_precision, context.label),
        )
        frame = f"relative_camera@{source_view}"
        claims.append(
            _claim(
                context,
                kind="relative_target_pose",
                statement_zh="目标相机位姿由题面给出的横向、前后平移和原地旋转共同确定。",
                value={"source_view_id": source_view, "right_m": right, "front_m": front, "yaw_delta_deg": yaw},
                required=True,
                source_paths=(
                    "question.certificate.checks[name=relative_pose_instruction_exposed].source_view_id",
                    "question.certificate.checks[name=relative_pose_instruction_exposed].right_m",
                    "question.certificate.checks[name=relative_pose_instruction_exposed].front_m",
                    "question.certificate.checks[name=relative_pose_instruction_exposed].yaw_delta_deg",
                ),
                frame_id=frame,
                numeric_surfaces=numeric,
                epistemic_scope="geometry_certificate",
            )
        )
    else:
        context.check("candidate_observed_in_write_prefix")
        anchors = context.check("surface_referents_uniquely_anchored")
        origin = _string(
            context.certificate.get("origin_entity_id"),
            "question.certificate.origin_entity_id",
            context.label,
        )
        facing = _string(
            context.certificate.get("facing_entity_id"),
            "question.certificate.facing_entity_id",
            context.label,
        )
        expected_entities = (origin, facing, candidate_id)
        if context.evidence_entity_ids != expected_entities:
            raise _fail(context.label, "object-anchored target-view entity roles are inconsistent")
        origin_anchor = _string(
            anchors.get("origin_anchor_view_id"),
            "certificate.checks[surface_referents_uniquely_anchored].origin_anchor_view_id",
            context.label,
        )
        facing_anchor = _string(
            anchors.get("facing_anchor_view_id"),
            "certificate.checks[surface_referents_uniquely_anchored].facing_anchor_view_id",
            context.label,
        )
        if origin_anchor not in context.exposure_view_ids or facing_anchor not in context.exposure_view_ids:
            raise _fail(context.label, "object-anchored target-view referents are not exposed")
        frame = f"object_anchored@{origin}->{facing}"
        claims.append(
            _claim(
                context,
                kind="object_anchored_target_frame",
                statement_zh="目标视角位于起点物体处，并以起点指向朝向物体的方向为正前方。",
                value={
                    "origin_entity_id": origin,
                    "facing_entity_id": facing,
                    "origin_anchor_view_id": origin_anchor,
                    "facing_anchor_view_id": facing_anchor,
                },
                required=True,
                source_paths=(
                    "question.certificate.origin_entity_id",
                    "question.certificate.facing_entity_id",
                    "question.certificate.checks[name=surface_referents_uniquely_anchored].origin_anchor_view_id",
                    "question.certificate.checks[name=surface_referents_uniquely_anchored].facing_anchor_view_id",
                ),
                frame_id=frame,
                allowed_directions=_direction_surfaces("in_front_of", context.label),
                epistemic_scope="geometry_certificate",
            )
        )

    claims.append(
        _claim(
            context,
            kind="target_view_visibility",
            statement_zh=f"在目标视角中，候选物体{'可见' if answer else '不可见'}。",
            value=answer,
            required=True,
            source_paths=(
                "question.answer_value",
                "question.certificate.checks[name=candidate_visibility_in_target_render].measured",
                "question.certificate.candidate_entity_id",
            ),
            frame_id=frame,
            epistemic_scope="held_out_render_supervision",
        )
    )
    return claims


def _build_grounding(context: _Context) -> list[Claim]:
    context.known()
    answer = _boolean(context.answer_value, "question.answer_value", context.label)
    if len(context.evidence_view_ids) != 1:
        raise _fail(context.label, "grounding_presence requires exactly one evidence view")
    view_id = context.evidence_view_ids[0]
    frame = f"image@{view_id}"
    if answer:
        visible = context.check("entity_visible")
        measured_view = _string(
            visible.get("measured_value"),
            "certificate.checks[entity_visible].measured_value",
            context.label,
        )
        surface = context.check("view_local_grounding_surface")
        if measured_view != view_id or surface.get("model_view_id") != view_id:
            raise _fail(context.label, "positive grounding view bindings disagree")
        if len(context.evidence_entity_ids) != 1:
            raise _fail(context.label, "positive grounding requires one visible entity")
        statement = "查询目标在这张图中可见。"
        paths = (
            "question.answer_value",
            "question.certificate.checks[name=entity_visible].measured_value",
            "question.certificate.checks[name=view_local_grounding_surface].model_view_id",
        )
        entities: Sequence[str] = context.evidence_entity_ids
    else:
        absent = context.check("category_absent_in_anchor_view")
        anchor = _string(
            absent.get("anchor_view_id"),
            "certificate.checks[category_absent_in_anchor_view].anchor_view_id",
            context.label,
        )
        category = _string(
            absent.get("category"),
            "certificate.checks[category_absent_in_anchor_view].category",
            context.label,
        )
        if anchor != view_id:
            raise _fail(context.label, "negative grounding anchor disagrees with evidence view")
        # category_observed_elsewhere is a construction gate only.  Its hidden
        # views and entity binding are intentionally absent from this claim.
        context.check("category_observed_elsewhere")
        statement = f"这张图中没有观察到{entity_name(category)}。"
        paths = (
            "question.answer_value",
            "question.certificate.checks[name=category_absent_in_anchor_view].anchor_view_id",
            "question.certificate.checks[name=category_absent_in_anchor_view].category",
        )
        entities = ()
    return [
        _claim(
            context,
            kind="view_local_presence",
            statement_zh=statement,
            value=answer,
            required=True,
            source_paths=paths,
            evidence_entity_ids=entities,
            frame_id=frame,
            epistemic_scope="observed_evidence",
        )
    ]


def _unknown_claims(
    context: _Context,
    *,
    absence_statement: str,
    absence_paths: Sequence[str],
) -> list[Claim]:
    return [
        _claim(
            context,
            kind="observed_evidence_absence",
            statement_zh=absence_statement,
            value={"positive_evidence_in_exposure": False},
            required=True,
            source_paths=absence_paths,
            evidence_entity_ids=(),
            frame_id="observation_set",
            epistemic_scope="observed_evidence_only",
        ),
        _claim(
            context,
            kind="epistemic_abstention",
            statement_zh="当前视角只覆盖部分场景；没有观察到目标，因此仍无法确定场景里是否有目标。",
            value="unknown",
            required=True,
            source_paths=("question.answer_status", "question.model_view_ids"),
            evidence_entity_ids=(),
            frame_id="observation_set",
            epistemic_scope="observed_evidence_only",
        ),
    ]


def _build_unknown_abstention(context: _Context) -> list[Claim]:
    context.unknown()
    absent = context.check("category_absent_from_all_observations")
    if absent.get("measured_value") is not True:
        raise _fail(context.label, "unknown task must certify absence from exposed observations")
    # This check is a non-leakage gate.  Its name/value is deliberately not
    # copied to a claim source path, and certificate.oracle is never read.
    context.check("world_truth_not_model_visible")
    return _unknown_claims(
        context,
        absence_statement="给出的所有视角都没有提供查询目标的正视觉证据。",
        absence_paths=(
            "question.certificate.checks[name=category_absent_from_all_observations].measured_value",
            "question.evidence_view_ids",
        ),
    )


def _build_evidence_unknown(context: _Context) -> list[Claim]:
    context.unknown()
    count_check = context.check("matched_visual_input_count")
    count = _integer(
        count_check.get("image_count"),
        "certificate.checks[matched_visual_input_count].image_count",
        context.label,
    )
    if count != len(context.exposure_view_ids):
        raise _fail(context.label, "matched evidence-family image count is inconsistent")
    category_check = context.check("category_level_existence_query")
    category = _string(
        category_check.get("category"),
        "certificate.checks[category_level_existence_query].category",
        context.label,
    )
    context.check("target_category_absent_from_actual_input")
    return _unknown_claims(
        context,
        absence_statement=f"当前输入中没有{entity_name(category)}的正视觉证据。",
        absence_paths=(
            "question.certificate.checks[name=category_level_existence_query].category",
            "question.certificate.checks[name=target_category_absent_from_actual_input]",
            "question.evidence_view_ids",
        ),
    )


def _build_evidence_reveal(context: _Context) -> list[Claim]:
    context.known()
    answer = _mapping(context.answer_value, "question.answer_value", context.label)
    status = _string(answer.get("status"), "question.answer_value.status", context.label)
    category = _string(answer.get("category"), "question.answer_value.category", context.label)
    if status != "present":
        raise _fail(context.label, "evidence reveal answer must certify status='present'")
    count_check = context.check("matched_visual_input_count")
    count = _integer(
        count_check.get("image_count"),
        "certificate.checks[matched_visual_input_count].image_count",
        context.label,
    )
    if count != len(context.exposure_view_ids):
        raise _fail(context.label, "matched evidence-family image count is inconsistent")
    category_check = context.check("category_level_existence_query")
    if category_check.get("category") != category:
        raise _fail(context.label, "reveal answer category disagrees with existence query")
    visible = context.check("target_category_visible_in_actual_input")
    decisive = _string(
        visible.get("decisive_view_id"),
        "certificate.checks[target_category_visible_in_actual_input].decisive_view_id",
        context.label,
    )
    if decisive not in context.exposure_view_ids:
        raise _fail(context.label, "decisive reveal view is not exposed")
    pixels = _integer(
        visible.get("visible_pixels"),
        "certificate.checks[target_category_visible_in_actual_input].visible_pixels",
        context.label,
    )
    minimum = _integer(
        visible.get("minimum_visible_pixels"),
        "certificate.checks[target_category_visible_in_actual_input].minimum_visible_pixels",
        context.label,
    )
    if pixels < minimum:
        raise _fail(context.label, "decisive reveal does not meet visibility threshold")
    return [
        _claim(
            context,
            kind="evidence_reveal_presence",
            statement_zh=f"加入决定性视角后，画面提供了{entity_name(category)}的直接证据，可以确认其存在。",
            value={"status": "present", "category": category, "decisive_view_id": decisive},
            required=True,
            source_paths=(
                "question.answer_value",
                "question.certificate.checks[name=category_level_existence_query].category",
                "question.certificate.checks[name=target_category_visible_in_actual_input].decisive_view_id",
                "question.certificate.checks[name=target_category_visible_in_actual_input].visible_pixels",
            ),
            evidence_view_ids=(decisive,),
            frame_id="observation_set",
            epistemic_scope="observed_evidence",
        )
    ]


def _build_occlusion_unknown(context: _Context) -> list[Claim]:
    context.unknown()
    variant = _string(context.question.get("family_variant"), "question.family_variant", context.label)
    if variant == "prefix_unknown":
        context.check("target_category_absent_from_prefix")
        # decisive_view_withheld contains hidden supervision.  It is permitted
        # as a compiler-side gate but intentionally never serialized below.
        context.check("decisive_view_withheld")
        path = "question.certificate.checks[name=target_category_absent_from_prefix]"
    elif variant == "decisive_deleted":
        # This gate proves the family construction, but its deleted-view list
        # is hidden supervision and must not become model-authorized language.
        context.check("all_target_category_views_deleted")
        path = "question.answer_status"
    else:
        raise _fail(context.label, f"unsupported occlusion unknown variant {variant!r}")
    return _unknown_claims(
        context,
        absence_statement="当前暴露的视角中没有查询目标的正视觉证据。",
        absence_paths=(path, "question.evidence_view_ids"),
    )


def _build_occlusion_reveal(context: _Context) -> list[Claim]:
    context.known()
    answer = _mapping(context.answer_value, "question.answer_value", context.label)
    status = _string(answer.get("status"), "question.answer_value.status", context.label)
    category = _string(answer.get("category"), "question.answer_value.category", context.label)
    if status != "present":
        raise _fail(context.label, "occlusion reveal answer must certify status='present'")
    context.check("target_category_absent_before_reveal")
    visible = context.check("target_category_visible_in_decisive_view")
    decisive = _string(
        visible.get("decisive_view"),
        "certificate.checks[target_category_visible_in_decisive_view].decisive_view",
        context.label,
    )
    if decisive not in context.exposure_view_ids:
        raise _fail(context.label, "occlusion decisive view is not exposed")
    return [
        _claim(
            context,
            kind="occlusion_reveal_presence",
            statement_zh=(
                f"前段观察没有{entity_name(category)}证据；移动到决定性视角后看到了它，"
                "因此可以确认存在。"
            ),
            value={"status": "present", "category": category, "decisive_view_id": decisive},
            required=True,
            source_paths=(
                "question.answer_value",
                "question.certificate.checks[name=target_category_absent_before_reveal]",
                "question.certificate.checks[name=target_category_visible_in_decisive_view].decisive_view",
            ),
            frame_id="observation_set",
            epistemic_scope="observed_evidence",
        )
    ]


_BUILDERS: dict[str, Callable[[_Context], list[Claim]]] = {
    "metric_distance": _build_metric,
    "egocentric_relation": _build_egocentric,
    "cross_view_relation": _build_cross_view,
    "counterfactual_verification": _build_counterfactual,
    "last_seen_memory": _build_last_seen,
    "orbit_identity": _build_orbit,
    "rotation_change_detection": _build_rotation,
    "elevation_relation_transfer": _build_elevation,
    "object_centric_perspective": _build_object_perspective,
    "target_view_prediction": _build_target_view,
    "grounding_presence": _build_grounding,
    "unknown_abstention": _build_unknown_abstention,
    "evidence_presence_unknown": _build_evidence_unknown,
    "evidence_presence_reveal": _build_evidence_reveal,
    "occlusion_unknown": _build_occlusion_unknown,
    "occlusion_reveal": _build_occlusion_reveal,
}


def _related_fact_ids(context: _Context) -> tuple[str, ...]:
    questions = context.episode.get("questions")
    if questions is None:
        return ()
    rows = _sequence(questions, "episode.questions", context.label)
    related: set[str] = set()
    consistency_group = context.question.get("consistency_group")
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        fact_id = item.get("fact_id")
        if not isinstance(fact_id, str) or fact_id == context.fact_id:
            continue
        if consistency_group and item.get("consistency_group") == consistency_group:
            related.add(fact_id)
        if (
            context.task_type in {"cross_view_relation", "counterfactual_verification"}
            and item.get("task_type") in {"cross_view_relation", "counterfactual_verification"}
            and tuple(item.get("evidence_entity_ids", ())) == context.evidence_entity_ids
        ):
            related.add(fact_id)
    return tuple(sorted(related))


def _validate_compiled_claims(context: _Context, claims: Sequence[Claim]) -> None:
    if not claims:
        raise _fail(context.label, "claim builder returned no claims")
    if not any(claim.required for claim in claims):
        raise _fail(context.label, "claim sheet must contain at least one required claim")
    ids = [claim.claim_id for claim in claims]
    if len(set(ids)) != len(ids):
        raise _fail(context.label, "claim IDs are not unique")
    exposed = set(context.exposure_view_ids)
    for claim in claims:
        unknown_views = sorted(set(claim.evidence_view_ids) - exposed)
        if unknown_views:
            raise _fail(context.label, f"claim {claim.kind!r} cites unexposed views {unknown_views}")
        if not claim.source_paths:
            raise _fail(context.label, f"claim {claim.kind!r} has no source paths")
        if context.task_type in _UNKNOWN_TASKS:
            leaked_paths = [
                path
                for path in claim.source_paths
                if "oracle" in path.lower()
                or "world_truth" in path.lower()
                or "decisive_view" in path.lower()
                or "deleted_view" in path.lower()
            ]
            if leaked_paths:
                raise _fail(
                    context.label,
                    f"unknown claim {claim.kind!r} leaks hidden supervision paths {leaked_paths}",
                )
            if claim.evidence_entity_ids:
                raise _fail(
                    context.label,
                    f"unknown claim {claim.kind!r} must not expose hidden entity bindings",
                )


def compile_claim_sheet(
    episode: Mapping[str, Any],
    question: Mapping[str, Any],
    *,
    exposure_view_ids: Sequence[str] | None = None,
    source_episode_ir_sha256: str | None = None,
) -> ClaimSheet:
    """Compile one Episode IR question into a deterministic :class:`ClaimSheet`.

    ``exposure_view_ids`` may expand safe, fully determined questions to a
    shared episode context.  Unknown/evidence-deletion questions reject any
    expansion or reordering because it could change their answer.

    ``source_episode_ir_sha256`` can carry the raw JSONL-line digest supplied
    by a loader.  When omitted, a stable digest of the parsed canonical Episode
    IR object is used.
    """

    episode_map = _mapping(episode, "episode", "claim compiler")
    question_map = _mapping(question, "question", "claim compiler")
    context = _make_context(episode_map, question_map, exposure_view_ids)
    try:
        builder = _BUILDERS[context.task_type]
    except KeyError as error:
        raise _fail(context.label, f"unsupported task_type {context.task_type!r}") from error
    # This call also verifies that the task has an explicit SenseNova-SI
    # capability contract; the ClaimSheet stores the immutable result.
    capability = capability_for(context.task_type)
    claims = tuple(builder(context))
    _validate_compiled_claims(context, claims)

    digest = source_episode_ir_sha256 or _source_hash(episode_map)
    if not isinstance(digest, str) or len(digest) != 64:
        raise _fail(context.label, "source_episode_ir_sha256 must be a 64-character hex digest")
    try:
        int(digest, 16)
    except ValueError as error:
        raise _fail(context.label, "source_episode_ir_sha256 is not hexadecimal") from error
    source_bundle = _string(
        episode_map.get("source_bundle"), "episode.source_bundle", context.label
    )
    answer_key = _answer_key(context.answer_value, context.answer_status)
    identity_payload = {
        "fact_id": context.fact_id,
        "episode_id": context.episode_id,
        "task_type": context.task_type,
        "program_id": context.program_id,
        "exposure_view_ids": context.exposure_view_ids,
        "claim_ids": [claim.claim_id for claim in claims],
        "answer_key": answer_key,
        "source_episode_ir_sha256": digest,
    }
    return ClaimSheet(
        claim_sheet_id=content_id("claim-sheet", identity_payload),
        fact_id=context.fact_id,
        episode_id=context.episode_id,
        task_type=context.task_type,
        program_id=context.program_id,
        semantic_signature=context.semantic_signature,
        capability=capability,
        exposure_view_ids=context.exposure_view_ids,
        claims=claims,
        answer_key=answer_key,
        answer_value=context.answer_value,
        canonical_answer_zh=context.answer_zh,
        answer_status=context.answer_status,
        source_episode_ir_sha256=digest,
        source_bundle=source_bundle,
        related_fact_ids=_related_fact_ids(context),
    )


def compile_claim_sheets(
    episode: Mapping[str, Any],
    *,
    questions: Sequence[Mapping[str, Any]] | None = None,
    exposure_by_fact_id: Mapping[str, Sequence[str]] | None = None,
    source_episode_ir_sha256: str | None = None,
) -> tuple[ClaimSheet, ...]:
    """Compile several questions from one episode with shared provenance."""

    episode_map = _mapping(episode, "episode", "claim compiler")
    selected = (
        _sequence(episode_map.get("questions"), "episode.questions", "claim compiler")
        if questions is None
        else _sequence(questions, "questions", "claim compiler")
    )
    digest = source_episode_ir_sha256 or _source_hash(episode_map)
    sheets: list[ClaimSheet] = []
    for index, raw in enumerate(selected):
        question = _mapping(raw, f"questions[{index}]", "claim compiler")
        fact_id = _string(question.get("fact_id"), f"questions[{index}].fact_id", "claim compiler")
        exposure = None if exposure_by_fact_id is None else exposure_by_fact_id.get(fact_id)
        sheets.append(
            compile_claim_sheet(
                episode_map,
                question,
                exposure_view_ids=exposure,
                source_episode_ir_sha256=digest,
            )
        )
    return tuple(sheets)


# A readable alias for callers that think of compilation as a construction
# step.  Keep one implementation so IDs and fail-closed behavior stay identical.
build_claim_sheet = compile_claim_sheet


__all__ = [
    "ClaimCompilationError",
    "build_claim_sheet",
    "compile_claim_sheet",
    "compile_claim_sheets",
]
