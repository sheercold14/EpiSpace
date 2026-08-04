"""Pre-render geometric oracle candidates with typed operation certificates."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from spatial_episode.contracts.base import ContractModel
from spatial_episode.contracts.operation_v1 import OperationGraphV1


class SpatialRelation(StrEnum):
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    ABOVE = "above"
    BELOW = "below"


class EvidenceStatus(StrEnum):
    PENDING_RENDER = "pending_render"
    OBSERVED = "observed"
    REJECTED = "rejected"


class RelationCandidateV1(ContractModel):
    candidate_id: UUID
    subject_entity_id: UUID
    reference_entity_id: UUID
    relation: SpatialRelation
    frame_id: Literal["world"] = "world"
    signed_axis_delta_m: float
    decision_margin_m: float = Field(gt=0.0)
    center_distance_m: float = Field(gt=0.0)
    evidence_status: EvidenceStatus = EvidenceStatus.PENDING_RENDER
    operation_graph: OperationGraphV1

    @model_validator(mode="after")
    def entities_are_distinct(self) -> RelationCandidateV1:
        if self.subject_entity_id == self.reference_entity_id:
            raise ValueError("relation candidate entities must be distinct")
        return self


class RelationOracleV1(ContractModel):
    schema_version: Literal["relation_oracle.v1"] = "relation_oracle.v1"
    oracle_version: Literal["axis_margin.v1"] = "axis_margin.v1"
    scene_id: UUID
    minimum_center_distance_m: float = Field(gt=0.0)
    maximum_center_distance_m: float = Field(gt=0.0)
    minimum_axis_margin_m: float = Field(gt=0.0)
    candidates: tuple[RelationCandidateV1, ...]

    @model_validator(mode="after")
    def candidate_ids_are_unique(self) -> RelationOracleV1:
        if self.maximum_center_distance_m <= self.minimum_center_distance_m:
            raise ValueError("oracle center-distance range is invalid")
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("relation candidate IDs must be unique")
        return self
