"""Serializable intermediate records for the EpiSpace compiler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from episode3d.programs import TypedProgram


@dataclass(frozen=True)
class QuestionSpec:
    fact_id: str
    task_type: str
    question_zh: str
    answer_zh: str
    answer_value: Any
    answer_status: str
    program: TypedProgram
    evidence_view_ids: tuple[str, ...]
    evidence_entity_ids: tuple[str, ...]
    certificate: dict[str, Any]
    rationale_zh: str
    source: str
    family_variant: str = "canonical"
    consistency_group: str | None = None
    model_view_ids: tuple[str, ...] | None = None
    oracle_held_out_view_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def as_ir_dict(self) -> dict[str, Any]:
        payload = {
            "fact_id": self.fact_id,
            "task_type": self.task_type,
            "question_zh": self.question_zh,
            "answer_zh": self.answer_zh,
            "answer_value": self.answer_value,
            "answer_status": self.answer_status,
            "program": self.program.as_dict(),
            "evidence_view_ids": list(self.evidence_view_ids),
            "evidence_entity_ids": list(self.evidence_entity_ids),
            "certificate": self.certificate,
            "rationale_zh": self.rationale_zh,
            "source": self.source,
            "family_variant": self.family_variant,
            "oracle_held_out_view_ids": list(self.oracle_held_out_view_ids),
            "tags": list(self.tags),
        }
        if self.consistency_group:
            payload["consistency_group"] = self.consistency_group
        if self.model_view_ids is not None:
            payload["model_view_ids"] = list(self.model_view_ids)
        return payload


@dataclass
class Rejection:
    scope: str
    item_id: str
    reason_code: str
    detail: str
    source: str

    def as_dict(self) -> dict[str, str]:
        return {
            "scope": self.scope,
            "item_id": self.item_id,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "source": self.source,
        }


@dataclass
class CompiledBundle:
    bundle: Any
    split: str
    questions: list[QuestionSpec] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
