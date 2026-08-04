"""Serialized asset capability declaration."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from spatial_episode.contracts.base import ContractModel
from spatial_episode.domain.capability import Capability, RedistributionPolicy


class LicensePolicyV1(ContractModel):
    identifier: str = Field(min_length=1)
    academic_only: bool
    redistribution: RedistributionPolicy
    terms_uri: str | None = None


class CapabilityGapV1(ContractModel):
    capability: Capability
    reason: str = Field(min_length=1)


class CapabilityManifestV1(ContractModel):
    schema_version: Literal["capability_manifest.v1"] = "capability_manifest.v1"
    source_name: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    source_scene_id: str = Field(min_length=1)
    capabilities: frozenset[Capability]
    license_policy: LicensePolicyV1
    gaps: tuple[CapabilityGapV1, ...] = ()

    @model_validator(mode="after")
    def gaps_are_for_missing_capabilities(self) -> CapabilityManifestV1:
        duplicates: set[Capability] = set()
        seen: set[Capability] = set()
        for gap in self.gaps:
            if gap.capability in seen:
                duplicates.add(gap.capability)
            seen.add(gap.capability)
            if gap.capability in self.capabilities:
                raise ValueError(f"declared capability cannot also be a gap: {gap.capability}")
        if duplicates:
            raise ValueError(
                f"duplicate capability gaps: {sorted(item.value for item in duplicates)}"
            )
        return self
