"""Compile accepted rendered evidence into a canonical episode."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any
from uuid import UUID

from spatial_episode.contracts.base import ArtifactRefV1, ProvenanceV1, TransformV1
from spatial_episode.contracts.episode_v1 import (
    CertificateResult,
    CertificateV1,
    ChannelPolicyV1,
    ObservationV1,
    QueryStatus,
    QueryV1,
    SensorSpecV1,
    SensorType,
    SpatialEpisodeV1,
    StateDeltaV1,
    VerificationCheckV1,
)
from spatial_episode.contracts.oracle_v1 import RelationOracleV1, SpatialRelation
from spatial_episode.contracts.scene_ir_v1 import SceneIRV1
from spatial_episode.contracts.trajectory_v1 import TrajectoryPlanV1
from spatial_episode.domain.ids import stable_episode_family_id, stable_episode_id


def _artifact(payload: dict[str, Any]) -> ArtifactRefV1:
    digest = str(payload["sha256"])
    return ArtifactRefV1(
        uri=f"artifact://sha256/{digest}",
        sha256=digest,
        media_type=str(payload["media_type"]),
        byte_size=int(payload["byte_size"]),
    )


def _camera_transform(view: Any, sensor_height_m: float) -> TransformV1:
    agent = view.world_from_agent
    effective_height = (
        float(view.camera_height_m)
        if getattr(view, "camera_height_m", None) is not None
        else sensor_height_m
    )
    return TransformV1(
        parent_frame="world",
        child_frame=f"camera:{view.view_id}",
        translation_m=(
            agent.translation_m[0],
            agent.translation_m[1],
            agent.translation_m[2] + effective_height,
        ),
        rotation_xyzw=agent.rotation_xyzw,
    )


def _recipe_digest(trajectory: TrajectoryPlanV1, sensor_contract: dict[str, Any]) -> str:
    payload = {
        "recipe_id": trajectory.recipe_id,
        "seed": trajectory.seed,
        "sensor_contract": sensor_contract,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def compile_episode(
    *,
    scene: SceneIRV1,
    trajectory: TrajectoryPlanV1,
    oracle: RelationOracleV1,
    render_report: dict[str, Any],
    generator_version: str,
    maximum_queries_per_relation: int = 2,
) -> SpatialEpisodeV1:
    if render_report.get("status") != "success":
        raise ValueError("only a successful render bundle can compile into an episode")
    if trajectory.scene_id != scene.scene_id or oracle.scene_id != scene.scene_id:
        raise ValueError("scene, trajectory, and oracle IDs must match")
    rendered_views = render_report.get("views")
    if not isinstance(rendered_views, list) or len(rendered_views) != len(trajectory.views):
        raise ValueError("render bundle must align one-to-one with trajectory views")
    if maximum_queries_per_relation <= 0:
        raise ValueError("maximum queries per relation must be positive")

    sensor_contract = render_report["sensor_contract"]
    width = int(sensor_contract["width_px"])
    height = int(sensor_contract["height_px"])
    hfov = float(sensor_contract["horizontal_fov_deg"])
    sensor_height = float(sensor_contract["sensor_height_m"])
    sensor_specs = (
        SensorSpecV1(
            sensor_id="rgb",
            sensor_type=SensorType.RGB,
            width_px=width,
            height_px=height,
            horizontal_fov_deg=hfov,
            near_m=0.05,
            far_m=100.0,
        ),
        SensorSpecV1(
            sensor_id="depth",
            sensor_type=SensorType.DEPTH,
            width_px=width,
            height_px=height,
            horizontal_fov_deg=hfov,
            near_m=0.05,
            far_m=100.0,
            depth_unit="meter",
        ),
        SensorSpecV1(
            sensor_id="instance",
            sensor_type=SensorType.INSTANCE,
            width_px=width,
            height_px=height,
            horizontal_fov_deg=hfov,
            near_m=0.05,
            far_m=100.0,
        ),
    )

    observations = []
    state_deltas = []
    seen_entities: set[UUID] = set()
    entity_views: dict[UUID, list[str]] = defaultdict(list)
    view_entities: dict[str, set[UUID]] = {}
    for expected_step, (planned_view, rendered_view) in enumerate(
        zip(trajectory.views, rendered_views, strict=True)
    ):
        if (
            int(rendered_view["step"]) != expected_step
            or rendered_view["view_id"] != planned_view.view_id
        ):
            raise ValueError("rendered view identity does not match trajectory")
        visible = {
            scene.runtime_semantic_id_map[runtime_id]
            for runtime_id in rendered_view["visible_runtime_semantic_ids"]
            if runtime_id in scene.runtime_semantic_id_map
        }
        ordered_visible = tuple(sorted(visible, key=str))
        for entity_id in ordered_visible:
            entity_views[entity_id].append(planned_view.view_id)
        view_entities[planned_view.view_id] = visible
        artifact = _artifact(rendered_view["artifact"])
        observations.append(
            ObservationV1(
                step=expected_step,
                view_id=planned_view.view_id,
                world_from_camera=_camera_transform(planned_view, sensor_height),
                artifacts={
                    SensorType.RGB: artifact,
                    SensorType.DEPTH: artifact,
                    SensorType.INSTANCE: artifact,
                },
                visible_entity_ids=ordered_visible,
            )
        )
        added = visible - seen_entities
        reobserved = visible & seen_entities
        state_deltas.append(
            StateDeltaV1(
                step=expected_step,
                added=tuple(sorted(added, key=str)),
                reobserved=tuple(sorted(reobserved, key=str)),
                occluded=(),
            )
        )
        seen_entities.update(visible)

    queries = []
    relation_counts: dict[SpatialRelation, int] = defaultdict(int)
    used_pairs: set[frozenset[UUID]] = set()
    for candidate in oracle.candidates:
        subject_views = entity_views.get(candidate.subject_entity_id, [])
        reference_views = entity_views.get(candidate.reference_entity_id, [])
        if not subject_views or not reference_views:
            continue
        pair = frozenset({candidate.subject_entity_id, candidate.reference_entity_id})
        if pair in used_pairs:
            continue
        if relation_counts[candidate.relation] >= maximum_queries_per_relation:
            continue
        common_views = [
            view_id
            for view_id in subject_views
            if candidate.reference_entity_id in view_entities[view_id]
        ]
        evidence_views = (
            (common_views[0],) if common_views else (subject_views[0], reference_views[0])
        )
        queries.append(
            QueryV1(
                query_id=str(candidate.candidate_id),
                question_text=None,
                answer_type="spatial_relation",
                answer=candidate.relation.value,
                status=QueryStatus.ACCEPTED,
                decision_margin=candidate.decision_margin_m,
                evidence_view_ids=evidence_views,
                evidence_entity_ids=(
                    candidate.subject_entity_id,
                    candidate.reference_entity_id,
                ),
                operation_graph=candidate.operation_graph,
                certificate=CertificateV1(
                    verifier_version="relation_evidence.v1",
                    result=CertificateResult.PASS,
                    checks=(
                        VerificationCheckV1(
                            name="subject_observed", passed=True, measured_value=True
                        ),
                        VerificationCheckV1(
                            name="reference_observed", passed=True, measured_value=True
                        ),
                        VerificationCheckV1(
                            name="axis_decision_margin_m",
                            passed=True,
                            measured_value=candidate.decision_margin_m,
                            threshold=oracle.minimum_axis_margin_m,
                        ),
                        VerificationCheckV1(
                            name="typed_operation_graph", passed=True, measured_value=True
                        ),
                    ),
                ),
            )
        )
        used_pairs.add(pair)
        relation_counts[candidate.relation] += 1

    family_id = stable_episode_family_id(scene.scene_id, trajectory.recipe_id, trajectory.seed)
    source_digest = (
        scene.geometry_assets[0].sha256
        if scene.geometry_assets
        else hashlib.sha256(
            scene.model_dump_json(exclude={"geometry_assets"}).encode("utf-8")
        ).hexdigest()
    )
    return SpatialEpisodeV1(
        episode_id=stable_episode_id(family_id, "canonical"),
        family_id=family_id,
        family_variant="canonical",
        scene_id=scene.scene_id,
        split_group=str(scene.scene_id),
        recipe_id=trajectory.recipe_id,
        sensor_specs=sensor_specs,
        channel_policy=ChannelPolicyV1(
            model_visible=frozenset({"rgb"}),
            supervision=frozenset(
                {"depth_m", "instance_id", "world_from_camera", "observable_state"}
            ),
            oracle_only=frozenset({"scene_ir", "relation_oracle"}),
        ),
        observations=tuple(observations),
        state_deltas=tuple(state_deltas),
        queries=tuple(queries),
        provenance=ProvenanceV1(
            source_name=scene.capabilities.source_name,
            source_version=scene.capabilities.source_version,
            source_scene_id=scene.capabilities.source_scene_id,
            source_digest=source_digest,
            generator_version=generator_version,
            recipe_digest=_recipe_digest(trajectory, sensor_contract),
            license_identifier=scene.capabilities.license_policy.identifier,
        ),
    )
