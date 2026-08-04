"""Serializable contracts for the QA-generation pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_id(namespace: str, value: Any) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:20]
    return f"{namespace}-{digest}"


@dataclass(frozen=True)
class CapabilitySpec:
    primary: str
    supporting: tuple[str, ...]
    official_subtask: str
    episode_role: str
    response_profile: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary,
            "supporting": list(self.supporting),
            "official_subtask": self.official_subtask,
            "episode_role": self.episode_role,
            "response_profile": self.response_profile,
        }


@dataclass(frozen=True)
class NumericSurface:
    value: float
    unit: str
    tolerance: float
    surface: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "tolerance": self.tolerance,
            "surface": self.surface,
        }


@dataclass(frozen=True)
class Claim:
    claim_id: str
    kind: str
    statement_zh: str
    value: Any
    required: bool
    evidence_view_ids: tuple[str, ...] = ()
    evidence_entity_ids: tuple[str, ...] = ()
    source_paths: tuple[str, ...] = ()
    frame_id: str | None = None
    allowed_directions: tuple[str, ...] = ()
    numeric_surfaces: tuple[NumericSurface, ...] = ()
    epistemic_scope: str = "observed_evidence"

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "kind": self.kind,
            "statement_zh": self.statement_zh,
            "value": self.value,
            "required": self.required,
            "evidence_view_ids": list(self.evidence_view_ids),
            "evidence_entity_ids": list(self.evidence_entity_ids),
            "source_paths": list(self.source_paths),
            "frame_id": self.frame_id,
            "allowed_directions": list(self.allowed_directions),
            "numeric_surfaces": [item.as_dict() for item in self.numeric_surfaces],
            "epistemic_scope": self.epistemic_scope,
        }


@dataclass(frozen=True)
class ClaimSheet:
    claim_sheet_id: str
    fact_id: str
    episode_id: str
    task_type: str
    program_id: str
    semantic_signature: str
    capability: CapabilitySpec
    exposure_view_ids: tuple[str, ...]
    claims: tuple[Claim, ...]
    answer_key: str
    answer_value: Any
    canonical_answer_zh: str
    answer_status: str
    source_episode_ir_sha256: str
    source_bundle: str
    related_fact_ids: tuple[str, ...] = ()

    @property
    def required_claim_ids(self) -> tuple[str, ...]:
        return tuple(claim.claim_id for claim in self.claims if claim.required)

    @property
    def allowed_claim_ids(self) -> tuple[str, ...]:
        return tuple(claim.claim_id for claim in self.claims)

    def as_dict(self, *, include_answer: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": "epispace.claim_sheet.v1",
            "claim_sheet_id": self.claim_sheet_id,
            "fact_id": self.fact_id,
            "episode_id": self.episode_id,
            "task_type": self.task_type,
            "program_id": self.program_id,
            "semantic_signature": self.semantic_signature,
            "capability": self.capability.as_dict(),
            "exposure_view_ids": list(self.exposure_view_ids),
            "claims": [claim.as_dict() for claim in self.claims],
            "required_claim_ids": list(self.required_claim_ids),
            "allowed_claim_ids": list(self.allowed_claim_ids),
            "source_episode_ir_sha256": self.source_episode_ir_sha256,
            "source_bundle": self.source_bundle,
            "related_fact_ids": list(self.related_fact_ids),
        }
        if include_answer:
            payload["answer_contract"] = {
                "answer_key": self.answer_key,
                "answer_value": self.answer_value,
                "canonical_answer_zh": self.canonical_answer_zh,
                "answer_status": self.answer_status,
            }
        return payload


@dataclass(frozen=True)
class QuestionBlueprint:
    request_id: str
    fact_id: str
    task_type: str
    capability: CapabilitySpec
    intent_zh: str
    slot_values: dict[str, str]
    required_slots: tuple[str, ...]
    allowed_strategies: tuple[str, ...]
    output_contract: str
    family_key: str

    def answer_blind_dict(self) -> dict[str, Any]:
        """Return a payload whose schema cannot carry answer or certificate fields."""

        return {
            "schema_version": "epispace.question_blueprint.answer_blind.v1",
            "request_id": self.request_id,
            "task_type": self.task_type,
            "capability": self.capability.as_dict(),
            "intent_zh": self.intent_zh,
            "slot_values": dict(self.slot_values),
            "required_slots": list(self.required_slots),
            "allowed_strategies": list(self.allowed_strategies),
            "output_contract": self.output_contract,
            "family_key": self.family_key,
        }


@dataclass(frozen=True)
class PlannedQuestion:
    fact_id: str
    task_type: str
    program_id: str
    capability: CapabilitySpec
    exposure_view_ids: tuple[str, ...]
    source_model_view_ids: tuple[str, ...]
    expanded_to_episode_context: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "task_type": self.task_type,
            "program_id": self.program_id,
            "capability": self.capability.as_dict(),
            "exposure_view_ids": list(self.exposure_view_ids),
            "source_model_view_ids": list(self.source_model_view_ids),
            "expanded_to_episode_context": self.expanded_to_episode_context,
        }


@dataclass(frozen=True)
class EpisodeBatchPlan:
    batch_id: str
    episode_id: str
    scene_id: str
    split: str
    trajectory_class: str
    source_bundle: str
    exposure_view_ids: tuple[str, ...]
    questions: tuple[PlannedQuestion, ...]
    development_only: bool = True
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        capabilities = sorted(
            {
                value
                for question in self.questions
                for value in (question.capability.primary, *question.capability.supporting)
                if value != "AUX"
            }
        )
        return {
            "schema_version": "epispace.episode_batch_plan.v1",
            "batch_id": self.batch_id,
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "split": self.split,
            "trajectory_class": self.trajectory_class,
            "source_bundle": self.source_bundle,
            "exposure_view_ids": list(self.exposure_view_ids),
            "question_count": len(self.questions),
            "capabilities": capabilities,
            "questions": [question.as_dict() for question in self.questions],
            "development_only": self.development_only,
            "warnings": list(self.warnings),
        }


@dataclass
class ValidationReport:
    passed: bool
    checks: dict[str, bool]
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": self.checks, "errors": self.errors}

