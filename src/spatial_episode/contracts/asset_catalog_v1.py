"""Small, committable asset acquisition and license lock manifest."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from spatial_episode.contracts.base import ContractModel, Sha256
from spatial_episode.domain.capability import RedistributionPolicy


class MirrorPolicyV1(ContractModel):
    repository: str = Field(min_length=1)
    endpoint: str = Field(pattern=r"^https://")
    trust: Literal["transport_only"]
    note: str = Field(min_length=1)


class AssetCatalogFileV1(ContractModel):
    name: str = Field(min_length=1)
    byte_size: int = Field(gt=0)
    sha256: Sha256 | None


class AssetCatalogV1(ContractModel):
    catalog_version: Literal["asset_catalog.v1"] = "asset_catalog.v1"
    source: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    split: str = Field(min_length=1)
    authoritative_repository: str = Field(min_length=1)
    authoritative_revision: str = Field(min_length=1)
    license_identifier: str = Field(min_length=1)
    academic_only: bool
    redistribution: RedistributionPolicy
    terms_uri: str = Field(pattern=r"^https://")
    mirror_policy: MirrorPolicyV1 | None = None
    files: tuple[AssetCatalogFileV1, ...]

    @model_validator(mode="after")
    def files_are_unique(self) -> AssetCatalogV1:
        names = [file.name for file in self.files]
        if len(names) != len(set(names)):
            raise ValueError("asset catalog filenames must be unique")
        return self
