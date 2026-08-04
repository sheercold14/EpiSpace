"""Strict semantic configuration contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from spatial_episode.contracts.base import ContractModel
from spatial_episode.contracts.episode_v1 import SensorSpecV1
from spatial_episode.domain.capability import Capability
from spatial_episode.domain.operations import OperationKind


class TrajectoryPattern(StrEnum):
    YAW = "yaw"
    TRANSLATION = "translation"
    LONG_BASELINE = "long_baseline"
    OCCLUSION_REVEAL = "occlusion_reveal"
    REVISIT = "revisit"
    LOOP = "loop"
    HELD_OUT_VIEW = "held_out_view"


class SensorPresetV1(ContractModel):
    schema_version: Literal["sensor_preset.v1"] = "sensor_preset.v1"
    name: str = Field(min_length=1)
    sensors: tuple[SensorSpecV1, ...]

    @model_validator(mode="after")
    def sensors_are_unique(self) -> SensorPresetV1:
        identifiers = [sensor.sensor_id for sensor in self.sensors]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("sensor IDs must be unique")
        return self


class EpisodeRecipeV1(ContractModel):
    schema_version: Literal["episode_recipe.v1"] = "episode_recipe.v1"
    recipe_id: str = Field(min_length=1)
    required_capabilities: frozenset[Capability]
    views_per_episode: int = Field(ge=2, le=64)
    queries_per_episode: int = Field(ge=1, le=256)
    trajectory_patterns: frozenset[TrajectoryPattern]
    allowed_operations: frozenset[OperationKind]
    forbidden_relations: tuple[str, ...] = ()
    model_visible_channels: frozenset[str]

    @model_validator(mode="after")
    def recipe_has_verification_and_observations(self) -> EpisodeRecipeV1:
        if OperationKind.VERIFY not in self.allowed_operations:
            raise ValueError("episode recipe must include V verification")
        if Capability.RGB not in self.required_capabilities:
            raise ValueError("M1 episode recipe must require RGB")
        return self
