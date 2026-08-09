"""Deterministic JSON Schema export for public contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias

from pydantic import BaseModel

from spatial_episode.contracts.asset_catalog_v1 import AssetCatalogV1
from spatial_episode.contracts.capability_v1 import CapabilityManifestV1
from spatial_episode.contracts.episode_v1 import SpatialEpisodeV1
from spatial_episode.contracts.oracle_v1 import RelationOracleV1
from spatial_episode.contracts.scene_ir_v1 import SceneIRV1
from spatial_episode.contracts.trajectory_v1 import TrajectoryPlanV1
from spatial_episode.contracts.worker_v1 import WorkerRequestV1, WorkerResultV1
from spatial_episode.scriptgen.dataset import ScriptgenDatasetV1
from spatial_episode.scriptgen.family import ScriptgenFamilyV4

ContractType: TypeAlias = type[BaseModel]

CONTRACTS: dict[str, ContractType] = {
    "asset_catalog.v1.schema.json": AssetCatalogV1,
    "capability_manifest.v1.schema.json": CapabilityManifestV1,
    "scene_ir.v1.schema.json": SceneIRV1,
    "spatial_episode.v1.schema.json": SpatialEpisodeV1,
    "relation_oracle.v1.schema.json": RelationOracleV1,
    "scriptgen_family.v4.schema.json": ScriptgenFamilyV4,
    "scriptgen_dataset.v1.schema.json": ScriptgenDatasetV1,
    "trajectory_plan.v1.schema.json": TrajectoryPlanV1,
    "worker_request.v1.schema.json": WorkerRequestV1,
    "worker_result.v1.schema.json": WorkerResultV1,
}


def export_schemas(output_directory: Path) -> tuple[Path, ...]:
    output_directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, model in CONTRACTS.items():
        path = output_directory / filename
        payload = json.dumps(
            model.model_json_schema(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        path.write_text(payload + "\n", encoding="utf-8")
        written.append(path)
    return tuple(written)
