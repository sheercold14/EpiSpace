"""Shared contract primitives."""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Vec3Value = tuple[float, float, float]
QuaternionValue = tuple[float, float, float, float]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class ArtifactRefV1(ContractModel):
    uri: str = Field(pattern=r"^artifact://sha256/[0-9a-f]{64}$")
    sha256: Sha256
    media_type: str
    byte_size: int = Field(ge=0)

    @model_validator(mode="after")
    def digest_matches_uri(self) -> ArtifactRefV1:
        if self.uri.rsplit("/", maxsplit=1)[-1] != self.sha256:
            raise ValueError("artifact URI digest must equal sha256")
        return self


class TransformV1(ContractModel):
    parent_frame: str = Field(min_length=1)
    child_frame: str = Field(min_length=1)
    translation_m: Vec3Value
    rotation_xyzw: QuaternionValue
    convention: Literal["active_child_to_parent"] = "active_child_to_parent"


class ProvenanceV1(ContractModel):
    source_name: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    source_scene_id: str = Field(min_length=1)
    source_digest: Sha256
    generator_version: str = Field(min_length=1)
    recipe_digest: Sha256
    license_identifier: str = Field(min_length=1)


class UUIDRefSetV1(ContractModel):
    values: tuple[UUID, ...] = ()
