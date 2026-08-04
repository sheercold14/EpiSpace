"""Capabilities are declared by assets and required by episode recipes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from spatial_episode.domain.errors import CapabilityError


class Capability(StrEnum):
    RGB = "rgb"
    METRIC_DEPTH = "metric_depth"
    INSTANCE_MASK = "instance_mask"
    SEMANTIC_MASK = "semantic_mask"
    STATIC_GEOMETRY = "static_geometry"
    NAVMESH = "navmesh"
    RAYCAST = "raycast"
    REGION_ANNOTATIONS = "region_annotations"
    ORIENTED_INSTANCES = "oriented_instances"
    EDITABLE_INSTANCES = "editable_instances"
    ARTICULATED_INSTANCES = "articulated_instances"
    AMODAL_MASK = "amodal_mask"
    DETERMINISTIC_FLAT_RENDER = "deterministic_flat_render"


class RedistributionPolicy(StrEnum):
    ALLOWED = "allowed"
    DERIVED_ONLY = "derived_only"
    METADATA_ONLY = "metadata_only"
    FORBIDDEN = "forbidden"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class LicensePolicy:
    identifier: str
    academic_only: bool
    redistribution: RedistributionPolicy
    terms_uri: str | None = None


@dataclass(frozen=True, slots=True)
class CapabilityGap:
    capability: Capability
    reason: str


@dataclass(frozen=True, slots=True)
class CapabilityManifest:
    source_name: str
    source_version: str
    source_scene_id: str
    capabilities: frozenset[Capability]
    license_policy: LicensePolicy
    gaps: tuple[CapabilityGap, ...] = field(default_factory=tuple)

    def supports(self, *required: Capability) -> bool:
        return set(required).issubset(self.capabilities)

    def require(self, *required: Capability) -> None:
        missing = sorted(set(required) - self.capabilities, key=lambda item: item.value)
        if not missing:
            return
        known_reasons = {gap.capability: gap.reason for gap in self.gaps}
        detail = ", ".join(
            f"{capability.value} ({known_reasons.get(capability, 'not declared')})"
            for capability in missing
        )
        raise CapabilityError(
            f"scene {self.source_name}:{self.source_scene_id} lacks capabilities: {detail}"
        )
