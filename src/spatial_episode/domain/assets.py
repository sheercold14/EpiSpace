"""Asset inventory and probe results shared by source plugins."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from spatial_episode.domain.capability import CapabilityManifest


class AssetRole(StrEnum):
    SCENE_DATASET_CONFIG = "scene_dataset_config"
    RENDER_MESH = "render_mesh"
    NAVMESH = "navmesh"
    SEMANTIC_MESH = "semantic_mesh"
    SEMANTIC_DESCRIPTOR = "semantic_descriptor"


class ProbeSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class SourceSceneRef:
    source_name: str
    source_version: str
    source_scene_id: str
    split: str
    source_uri: str


@dataclass(frozen=True, slots=True)
class AssetFile:
    role: AssetRole
    relative_path: str
    byte_size: int
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ProbeIssue:
    code: str
    severity: ProbeSeverity
    message: str
    relative_path: str | None = None


@dataclass(frozen=True, slots=True)
class ProbeMeasurement:
    name: str
    value: int | float | str | bool


@dataclass(frozen=True, slots=True)
class AssetProbeReport:
    scene: SourceSceneRef
    files: tuple[AssetFile, ...]
    issues: tuple[ProbeIssue, ...]
    capability_manifest: CapabilityManifest
    measurements: tuple[ProbeMeasurement, ...] = ()

    @property
    def accepted(self) -> bool:
        return not any(issue.severity is ProbeSeverity.ERROR for issue in self.issues)
