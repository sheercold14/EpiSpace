"""Canonical simulator-independent scene contract."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from spatial_episode.contracts.base import ArtifactRefV1, ContractModel, TransformV1, Vec3Value
from spatial_episode.contracts.capability_v1 import CapabilityManifestV1


class OrientationEvidence(StrEnum):
    NONE = "none"
    SOURCE_ANNOTATION = "source_annotation"
    GEOMETRIC_HEURISTIC = "geometric_heuristic"
    HUMAN_VERIFIED = "human_verified"


class CoordinateFrameV1(ContractModel):
    frame_id: Literal["world"] = "world"
    handedness: Literal["right"] = "right"
    unit: Literal["meter"] = "meter"
    right_axis: Literal["+X"] = "+X"
    forward_axis: Literal["+Y"] = "+Y"
    up_axis: Literal["+Z"] = "+Z"
    quaternion_order: Literal["xyzw"] = "xyzw"


class ObbV1(ContractModel):
    center_m: Vec3Value
    half_extents_m: Vec3Value
    world_from_obb: TransformV1

    @model_validator(mode="after")
    def extents_are_positive(self) -> ObbV1:
        if any(value <= 0.0 for value in self.half_extents_m):
            raise ValueError("OBB half extents must be positive")
        return self


class RegionV1(ContractModel):
    region_id: UUID
    source_region_id: str = Field(min_length=1)
    category_uri: str = Field(min_length=1)
    raw_label: str


class EntityV1(ContractModel):
    entity_id: UUID
    source_entity_id: str = Field(min_length=1)
    category_uri: str = Field(min_length=1)
    raw_label: str
    world_from_entity: TransformV1
    obb: ObbV1 | None = None
    canonical_front: Vec3Value | None = None
    orientation_evidence: OrientationEvidence = OrientationEvidence.NONE
    symmetry_group: str = "unknown"
    region_id: UUID | None = None
    geometry: ArtifactRefV1 | None = None

    @model_validator(mode="after")
    def front_requires_evidence(self) -> EntityV1:
        if (
            self.canonical_front is not None
            and self.orientation_evidence is OrientationEvidence.NONE
        ):
            raise ValueError("canonical_front requires orientation evidence")
        if (
            self.canonical_front is None
            and self.orientation_evidence is not OrientationEvidence.NONE
        ):
            raise ValueError("orientation evidence requires canonical_front")
        return self


class SceneIRV1(ContractModel):
    schema_version: Literal["scene_ir.v1"] = "scene_ir.v1"
    scene_id: UUID
    canonical_frame: CoordinateFrameV1 = CoordinateFrameV1()
    capabilities: CapabilityManifestV1
    regions: tuple[RegionV1, ...]
    entities: tuple[EntityV1, ...]
    geometry_assets: tuple[ArtifactRefV1, ...] = ()
    runtime_semantic_id_map: dict[int, UUID]
    source_to_canonical: TransformV1

    @model_validator(mode="after")
    def references_are_consistent(self) -> SceneIRV1:
        entity_ids = [entity.entity_id for entity in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("entity IDs must be unique")
        region_ids = [region.region_id for region in self.regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("region IDs must be unique")
        region_set = set(region_ids)
        missing_regions = {
            entity.region_id
            for entity in self.entities
            if entity.region_id is not None and entity.region_id not in region_set
        }
        if missing_regions:
            raise ValueError(
                f"entities reference missing regions: {sorted(map(str, missing_regions))}"
            )
        unknown_entities = set(self.runtime_semantic_id_map.values()) - set(entity_ids)
        if unknown_entities:
            identifiers = sorted(map(str, unknown_entities))
            raise ValueError(f"runtime semantic IDs reference missing entities: {identifiers}")
        if self.source_to_canonical.parent_frame != "world":
            raise ValueError("source_to_canonical parent frame must be world")
        return self
