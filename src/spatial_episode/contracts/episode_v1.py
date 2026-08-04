"""Canonical spatial episode contract with strict channel separation."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from spatial_episode.contracts.base import ArtifactRefV1, ContractModel, ProvenanceV1, TransformV1
from spatial_episode.contracts.operation_v1 import OperationGraphV1


class SensorType(StrEnum):
    RGB = "rgb"
    DEPTH = "depth"
    INSTANCE = "instance"
    SEMANTIC = "semantic"


class QueryStatus(StrEnum):
    ACCEPTED = "accepted"
    UNKNOWN = "unknown"
    REJECTED = "rejected"


class CertificateResult(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class SensorSpecV1(ContractModel):
    sensor_id: str = Field(min_length=1)
    sensor_type: SensorType
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    horizontal_fov_deg: float = Field(gt=0.0, lt=180.0)
    near_m: float = Field(gt=0.0)
    far_m: float = Field(gt=0.0)
    depth_unit: Literal["meter"] | None = None

    @model_validator(mode="after")
    def range_is_valid(self) -> SensorSpecV1:
        if self.far_m <= self.near_m:
            raise ValueError("far plane must be greater than near plane")
        if self.sensor_type is SensorType.DEPTH and self.depth_unit != "meter":
            raise ValueError("depth sensor must declare meter units")
        if self.sensor_type is not SensorType.DEPTH and self.depth_unit is not None:
            raise ValueError("only depth sensors declare depth_unit")
        return self


class ChannelPolicyV1(ContractModel):
    model_visible: frozenset[str]
    supervision: frozenset[str]
    oracle_only: frozenset[str]

    @model_validator(mode="after")
    def channels_are_disjoint(self) -> ChannelPolicyV1:
        if self.model_visible & self.supervision:
            raise ValueError("model_visible and supervision channels overlap")
        if self.model_visible & self.oracle_only:
            raise ValueError("model_visible and oracle_only channels overlap")
        if self.supervision & self.oracle_only:
            raise ValueError("supervision and oracle_only channels overlap")
        return self


class ObservationV1(ContractModel):
    step: int = Field(ge=0)
    view_id: str = Field(min_length=1)
    world_from_camera: TransformV1
    artifacts: dict[SensorType, ArtifactRefV1]
    visible_entity_ids: tuple[UUID, ...]


class StateDeltaV1(ContractModel):
    step: int = Field(ge=0)
    added: tuple[UUID, ...] = ()
    reobserved: tuple[UUID, ...] = ()
    occluded: tuple[UUID, ...] = ()

    @model_validator(mode="after")
    def state_sets_are_disjoint(self) -> StateDeltaV1:
        groups = (set(self.added), set(self.reobserved), set(self.occluded))
        if any(
            groups[i] & groups[j] for i in range(len(groups)) for j in range(i + 1, len(groups))
        ):
            raise ValueError("state delta entity groups must be disjoint")
        return self


class VerificationCheckV1(ContractModel):
    name: str = Field(min_length=1)
    passed: bool
    measured_value: float | str | bool | None = None
    threshold: float | str | None = None


class CertificateV1(ContractModel):
    verifier_version: str = Field(min_length=1)
    result: CertificateResult
    checks: tuple[VerificationCheckV1, ...]


class QueryV1(ContractModel):
    query_id: str = Field(min_length=1)
    question_text: str | None = None
    answer_type: str = Field(min_length=1)
    answer: str | float | bool | tuple[str, ...] | None
    status: QueryStatus
    decision_margin: float | None = Field(default=None, ge=0.0)
    evidence_view_ids: tuple[str, ...]
    evidence_entity_ids: tuple[UUID, ...]
    operation_graph: OperationGraphV1
    certificate: CertificateV1
    rejection_reason: str | None = None

    @model_validator(mode="after")
    def answer_matches_status(self) -> QueryV1:
        if self.status is QueryStatus.ACCEPTED and self.answer is None:
            raise ValueError("accepted query must have an answer")
        if self.status is QueryStatus.REJECTED and not self.rejection_reason:
            raise ValueError("rejected query must have a rejection reason")
        if self.status is not QueryStatus.REJECTED and self.rejection_reason is not None:
            raise ValueError("only rejected queries may have rejection_reason")
        return self


class SpatialEpisodeV1(ContractModel):
    schema_version: Literal["spatial_episode.v1"] = "spatial_episode.v1"
    episode_id: UUID
    family_id: UUID
    family_variant: str = Field(min_length=1)
    scene_id: UUID
    split_group: str = Field(min_length=1)
    recipe_id: str = Field(min_length=1)
    sensor_specs: tuple[SensorSpecV1, ...]
    channel_policy: ChannelPolicyV1
    observations: tuple[ObservationV1, ...]
    state_deltas: tuple[StateDeltaV1, ...]
    queries: tuple[QueryV1, ...]
    provenance: ProvenanceV1

    @model_validator(mode="after")
    def episode_references_are_consistent(self) -> SpatialEpisodeV1:
        steps = [observation.step for observation in self.observations]
        if steps != list(range(len(steps))):
            raise ValueError("observation steps must be contiguous and ordered from zero")
        view_ids = [observation.view_id for observation in self.observations]
        if len(view_ids) != len(set(view_ids)):
            raise ValueError("view IDs must be unique within an episode")
        if [delta.step for delta in self.state_deltas] != steps:
            raise ValueError("state deltas must align one-to-one with observations")
        view_set = set(view_ids)
        for query in self.queries:
            missing = set(query.evidence_view_ids) - view_set
            if missing:
                raise ValueError(
                    f"query {query.query_id} references missing views: {sorted(missing)}"
                )
        sensor_ids = [sensor.sensor_id for sensor in self.sensor_specs]
        if len(sensor_ids) != len(set(sensor_ids)):
            raise ValueError("sensor IDs must be unique")
        return self
