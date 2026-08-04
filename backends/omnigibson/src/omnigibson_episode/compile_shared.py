"""Compile an OmniGibson acquisition bundle with shared Forge contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from omnigibson_episode.io import sha256_file, sha256_json, write_json_atomic


def _tuple3(values: Any) -> tuple[float, float, float]:
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("expected a three-element vector")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def compile_scene_snapshot(snapshot: dict[str, Any]) -> Any:
    from spatial_episode.contracts.base import TransformV1
    from spatial_episode.contracts.capability_v1 import (
        CapabilityGapV1,
        CapabilityManifestV1,
        LicensePolicyV1,
    )
    from spatial_episode.contracts.scene_ir_v1 import EntityV1, ObbV1, RegionV1, SceneIRV1
    from spatial_episode.domain.capability import Capability, RedistributionPolicy
    from spatial_episode.domain.ids import stable_entity_id, stable_region_id, stable_scene_id

    if snapshot.get("protocol_version") != "omnigibson_scene_snapshot.v1":
        raise ValueError("unsupported scene snapshot protocol")
    source_version = str(snapshot["source_version"])
    source_scene_id = str(snapshot["source_scene_id"])
    scene_id = stable_scene_id("omnigibson", source_version, source_scene_id)
    if str(scene_id) != str(snapshot.get("scene_id", scene_id)):
        raise ValueError("snapshot scene_id does not match stable source identity")

    raw_entities = snapshot.get("entities")
    if not isinstance(raw_entities, list) or not raw_entities:
        raise ValueError("scene snapshot contains no entities")
    def normalized_region(item: dict[str, Any]) -> str:
        return str(item.get("region") or "").strip() or "unassigned"

    region_names = sorted({normalized_region(item) for item in raw_entities})
    regions = tuple(
        RegionV1(
            region_id=stable_region_id(scene_id, name),
            source_region_id=name,
            category_uri="urn:spatialep:omnigibson:region",
            raw_label=name,
        )
        for name in region_names
    )
    region_map = {region.source_region_id: region.region_id for region in regions}
    entities = []
    name_to_id = {}
    any_articulated = False
    for item in sorted(raw_entities, key=lambda value: str(value["source_entity_id"])):
        source_entity_id = str(item["source_entity_id"])
        identity_digest = sha256_json(
            {
                "source_entity_id": source_entity_id,
                "category": item.get("category"),
                "prim_path": item.get("prim_path"),
            }
        )
        entity_id = stable_entity_id(scene_id, source_entity_id, identity_digest)
        center = _tuple3(item["aabb_center_m"])
        extent = _tuple3(item["aabb_extent_m"])
        half_extent = tuple(value / 2.0 for value in extent)
        if any(value <= 0.0 for value in half_extent):
            raise ValueError(f"entity {source_entity_id} has an invalid AABB")
        identity = (0.0, 0.0, 0.0, 1.0)
        category = str(item.get("category", "object"))
        region_name = normalized_region(item)
        entities.append(
            EntityV1(
                entity_id=entity_id,
                source_entity_id=source_entity_id,
                category_uri=(
                    "urn:spatialep:omnigibson:category:" + quote(category.casefold(), safe="")
                ),
                raw_label=category,
                world_from_entity=TransformV1(
                    parent_frame="world",
                    child_frame=f"entity:{entity_id}",
                    translation_m=center,
                    rotation_xyzw=identity,
                ),
                obb=ObbV1(
                    center_m=center,
                    half_extents_m=half_extent,
                    world_from_obb=TransformV1(
                        parent_frame="world",
                        child_frame=f"obb:{entity_id}",
                        translation_m=center,
                        rotation_xyzw=identity,
                    ),
                ),
                region_id=region_map[region_name],
            )
        )
        name_to_id[source_entity_id] = entity_id
        any_articulated = any_articulated or bool(item.get("articulated", False))

    runtime_registry = snapshot.get("runtime_instance_registry", {})
    if not isinstance(runtime_registry, dict):
        raise ValueError("runtime_instance_registry must be a mapping")
    runtime_map = {
        int(identifier): name_to_id[str(name)]
        for identifier, name in runtime_registry.items()
        if str(name) in name_to_id and int(identifier) > 1
    }
    if not runtime_map:
        raise ValueError("no rendered instance identifiers map to scene entities")

    capabilities = {
        Capability.RGB,
        Capability.METRIC_DEPTH,
        Capability.INSTANCE_MASK,
        Capability.SEMANTIC_MASK,
        Capability.STATIC_GEOMETRY,
        Capability.NAVMESH,
        Capability.REGION_ANNOTATIONS,
        Capability.EDITABLE_INSTANCES,
    }
    if any_articulated:
        capabilities.add(Capability.ARTICULATED_INSTANCES)
    license_raw = snapshot.get("license", {})
    manifest = CapabilityManifestV1(
        source_name="omnigibson",
        source_version=source_version,
        source_scene_id=source_scene_id,
        capabilities=frozenset(capabilities),
        license_policy=LicensePolicyV1(
            identifier=str(license_raw.get("identifier", "BEHAVIOR-1K-ASSET-LICENSE")),
            academic_only=bool(license_raw.get("academic_only", True)),
            redistribution=RedistributionPolicy(
                str(license_raw.get("redistribution", "metadata_only"))
            ),
            terms_uri=license_raw.get("terms_uri"),
        ),
        gaps=(
            CapabilityGapV1(
                capability=Capability.ORIENTED_INSTANCES,
                reason="asset pose is not evidence of a category-level canonical front",
            ),
            CapabilityGapV1(
                capability=Capability.AMODAL_MASK,
                reason="M1 records visible instance masks only",
            ),
        ),
    )
    return SceneIRV1(
        scene_id=scene_id,
        capabilities=manifest,
        regions=regions,
        entities=tuple(entities),
        geometry_assets=(),
        runtime_semantic_id_map=runtime_map,
        source_to_canonical=TransformV1(
            parent_frame="world",
            child_frame="omnigibson_world",
            translation_m=(0.0, 0.0, 0.0),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        ),
    )


def _verify_render_evidence(bundle_directory: Path, render_report: dict[str, Any]) -> None:
    if render_report.get("status") != "success":
        raise ValueError("render report is not successful")
    bundle_root = bundle_directory.resolve()
    views = render_report.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError("render report contains no views")
    for view in views:
        artifact = view.get("artifact")
        if not isinstance(artifact, dict):
            raise ValueError("rendered view has no artifact manifest")
        path = Path(str(artifact.get("path", ""))).resolve()
        try:
            path.relative_to(bundle_root)
        except ValueError as error:
            raise ValueError("render artifact must reside inside its bundle") from error
        if not path.is_file():
            raise FileNotFoundError(f"render artifact is missing: {path}")
        if int(artifact.get("byte_size", -1)) != path.stat().st_size:
            raise ValueError(f"render artifact size mismatch: {path.name}")
        if str(artifact.get("sha256")) != sha256_file(path):
            raise ValueError(f"render artifact digest mismatch: {path.name}")


def compile_bundle(
    *,
    bundle_directory: Path,
    output_path: Path,
    generator_version: str = "omnigibson-spatial-episode/0.1.0",
    maximum_queries_per_relation: int = 2,
) -> Any:
    from spatial_episode.application.services.episode_compiler import compile_episode
    from spatial_episode.application.services.relation_oracle import (
        RelationOracleConfig,
        generate_relation_oracle,
    )
    from spatial_episode.contracts.trajectory_v1 import TrajectoryPlanV1

    snapshot_path = bundle_directory / "scene_snapshot.json"
    trajectory_path = bundle_directory / "trajectory_plan.json"
    render_path = bundle_directory / "render_report.json"
    for path in (snapshot_path, trajectory_path, render_path):
        if not path.is_file():
            raise FileNotFoundError(f"bundle member is missing: {path.name}")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    trajectory = TrajectoryPlanV1.model_validate_json(trajectory_path.read_text(encoding="utf-8"))
    render_report = json.loads(render_path.read_text(encoding="utf-8"))
    _verify_render_evidence(bundle_directory, render_report)
    declared_trajectory_digest = render_report.get("trajectory_sha256")
    if (
        declared_trajectory_digest is not None
        and str(declared_trajectory_digest) != sha256_file(trajectory_path)
    ):
        raise ValueError("render report was generated from a different trajectory")
    scene = compile_scene_snapshot(snapshot)
    if trajectory.scene_id != scene.scene_id:
        raise ValueError("trajectory and scene snapshot identities do not match")
    oracle = generate_relation_oracle(
        scene,
        RelationOracleConfig(maximum_candidates_per_relation=100),
    )
    episode = compile_episode(
        scene=scene,
        trajectory=trajectory,
        oracle=oracle,
        render_report=render_report,
        generator_version=generator_version,
        maximum_queries_per_relation=maximum_queries_per_relation,
    )
    execution_path = bundle_directory / "intervention_execution.json"
    if execution_path.is_file():
        from spatial_episode.domain.ids import stable_episode_id

        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        if execution.get("status") != "success":
            raise ValueError("intervention execution is not successful")
        variant = str(execution.get("family_variant", "")).strip()
        if not variant or variant == "canonical":
            raise ValueError("intervention bundle must declare a non-canonical family variant")
        base = execution.get("base", {})
        if str(base.get("family_id")) != str(episode.family_id):
            raise ValueError("intervention and compiled episode family IDs disagree")
        if str(base.get("scene_id")) != str(episode.scene_id):
            raise ValueError("intervention and compiled scene IDs disagree")
        episode = episode.model_copy(
            update={
                "episode_id": stable_episode_id(episode.family_id, variant),
                "family_variant": variant,
            }
        )
    # A sensor-complete acquisition remains a valid audit unit even when no
    # relation pair has enough rendered evidence. Downstream reasoning audit
    # records the coverage gap and release policy keeps it out of the default
    # training tier; rejecting here would discard useful negative evidence and
    # make 46-scene acquisition coverage impossible to report honestly.
    write_json_atomic(bundle_directory / "scene_ir.json", scene.model_dump(mode="json"))
    write_json_atomic(
        bundle_directory / "relation_oracle.json", oracle.model_dump(mode="json")
    )
    write_json_atomic(output_path, episode.model_dump(mode="json"))
    return episode
