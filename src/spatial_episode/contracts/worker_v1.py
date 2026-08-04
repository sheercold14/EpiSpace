"""Control-plane-only worker protocol."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from spatial_episode.contracts.base import ArtifactRefV1, ContractModel


class WorkerOperation(StrEnum):
    DOCTOR = "doctor"
    PREPARE = "prepare"
    RENDER = "render"
    RAYCAST = "raycast"
    NAVIGABILITY = "navigability"


class WorkerStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"


class WorkerRequestV1(ContractModel):
    protocol_version: Literal["worker.v1"] = "worker.v1"
    request_id: UUID
    operation: WorkerOperation
    input_artifacts: tuple[ArtifactRefV1, ...] = ()
    output_staging_uri: str
    config: dict[str, object]
    seed: int


class WorkerErrorV1(ContractModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool
    context: dict[str, object] = Field(default_factory=dict)


class WorkerResultV1(ContractModel):
    protocol_version: Literal["worker.v1"] = "worker.v1"
    request_id: UUID
    status: WorkerStatus
    outputs: tuple[ArtifactRefV1, ...] = ()
    metrics: dict[str, float | int | str | bool] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    error: WorkerErrorV1 | None = None

    @model_validator(mode="after")
    def status_matches_error(self) -> WorkerResultV1:
        if self.status is WorkerStatus.SUCCESS and self.error is not None:
            raise ValueError("successful worker result cannot contain an error")
        if self.status is WorkerStatus.FAILURE and self.error is None:
            raise ValueError("failed worker result must contain an error")
        return self
