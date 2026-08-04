"""Generate conservative, balanced allocentric relation candidates."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from uuid import uuid5

from spatial_episode.contracts.operation_v1 import OperationGraphV1, OperationNodeV1
from spatial_episode.contracts.oracle_v1 import (
    RelationCandidateV1,
    RelationOracleV1,
    SpatialRelation,
)
from spatial_episode.contracts.scene_ir_v1 import EntityV1, SceneIRV1
from spatial_episode.domain.operations import OperationKind, ValueType

_EXCLUDED_LABELS = frozenset({"unknown", "wall", "floor", "ceiling", "clutter", "objects", "void"})
_INVERSE = {
    SpatialRelation.LEFT_OF: SpatialRelation.RIGHT_OF,
    SpatialRelation.RIGHT_OF: SpatialRelation.LEFT_OF,
    SpatialRelation.IN_FRONT_OF: SpatialRelation.BEHIND,
    SpatialRelation.BEHIND: SpatialRelation.IN_FRONT_OF,
    SpatialRelation.ABOVE: SpatialRelation.BELOW,
    SpatialRelation.BELOW: SpatialRelation.ABOVE,
}


@dataclass(frozen=True, slots=True)
class RelationOracleConfig:
    minimum_center_distance_m: float = 0.3
    maximum_center_distance_m: float = 4.5
    minimum_axis_margin_m: float = 0.4
    maximum_half_extent_m: float = 2.5
    maximum_vertical_horizontal_distance_m: float = 1.5
    maximum_candidates_per_relation: int = 100


def _eligible(entity: EntityV1, config: RelationOracleConfig) -> bool:
    return (
        entity.obb is not None
        and entity.raw_label.casefold() not in _EXCLUDED_LABELS
        and max(entity.obb.half_extents_m) <= config.maximum_half_extent_m
    )


def _graph(
    subject_entity_id: str,
    reference_entity_id: str,
    relation: SpatialRelation,
) -> OperationGraphV1:
    return OperationGraphV1(
        external_inputs={
            "view_context": ValueType.VIEW,
            "initial_belief": ValueType.BELIEF,
            "world_frame": ValueType.FRAME,
            "query_frame": ValueType.FRAME,
            "subject_hint": ValueType.ENTITY,
            "reference_hint": ValueType.ENTITY,
        },
        nodes=(
            OperationNodeV1(
                node_id="ground_subject",
                operation=OperationKind.GROUNDING,
                inputs=("view_context", "subject_hint"),
                output_type=ValueType.ENTITY,
                parameters={"entity_id": subject_entity_id},
            ),
            OperationNodeV1(
                node_id="ground_reference",
                operation=OperationKind.GROUNDING,
                inputs=("view_context", "reference_hint"),
                output_type=ValueType.ENTITY,
                parameters={"entity_id": reference_entity_id},
            ),
            OperationNodeV1(
                node_id="align_frame",
                operation=OperationKind.FRAME,
                inputs=("query_frame", "world_frame"),
                output_type=ValueType.TRANSFORM,
            ),
            OperationNodeV1(
                node_id="update_subject_belief",
                operation=OperationKind.BELIEF,
                inputs=("initial_belief", "ground_subject", "align_frame"),
                output_type=ValueType.BELIEF,
            ),
            OperationNodeV1(
                node_id="update_pair_belief",
                operation=OperationKind.BELIEF,
                inputs=("update_subject_belief", "ground_reference", "align_frame"),
                output_type=ValueType.BELIEF,
            ),
            OperationNodeV1(
                node_id="evaluate_relation",
                operation=OperationKind.RELATION,
                inputs=("ground_subject", "ground_reference", "query_frame"),
                output_type=ValueType.RELATION,
                parameters={"expected_relation": relation.value},
            ),
            OperationNodeV1(
                node_id="verify_relation",
                operation=OperationKind.VERIFY,
                inputs=("evaluate_relation",),
                output_type=ValueType.BOOLEAN,
                parameters={"expected_relation": relation.value},
            ),
        ),
        answer_node="verify_relation",
    )


def generate_relation_oracle(
    scene: SceneIRV1, config: RelationOracleConfig | None = None
) -> RelationOracleV1:
    config = config or RelationOracleConfig()
    if config.maximum_candidates_per_relation <= 0:
        raise ValueError("maximum candidates per relation must be positive")
    entities = sorted(
        (entity for entity in scene.entities if _eligible(entity, config)),
        key=lambda entity: str(entity.entity_id),
    )
    buckets: dict[SpatialRelation, list[RelationCandidateV1]] = defaultdict(list)

    def add_pair(
        subject: EntityV1,
        reference: EntityV1,
        relation: SpatialRelation,
        signed_delta: float,
        margin: float,
        distance: float,
    ) -> None:
        for left, right, item_relation, item_delta in (
            (subject, reference, relation, signed_delta),
            (reference, subject, _INVERSE[relation], -signed_delta),
        ):
            identifier = uuid5(
                scene.scene_id,
                f"relation:{left.entity_id}:{right.entity_id}:{item_relation.value}:world",
            )
            buckets[item_relation].append(
                RelationCandidateV1(
                    candidate_id=identifier,
                    subject_entity_id=left.entity_id,
                    reference_entity_id=right.entity_id,
                    relation=item_relation,
                    signed_axis_delta_m=item_delta,
                    decision_margin_m=margin,
                    center_distance_m=distance,
                    operation_graph=_graph(
                        str(left.entity_id), str(right.entity_id), item_relation
                    ),
                )
            )

    for index, subject in enumerate(entities):
        subject_center = subject.world_from_entity.translation_m
        for reference in entities[index + 1 :]:
            if subject.region_id != reference.region_id:
                continue
            reference_center = reference.world_from_entity.translation_m
            dx, dy, dz = (subject_center[axis] - reference_center[axis] for axis in range(3))
            distance = sqrt(dx * dx + dy * dy + dz * dz)
            if not (
                config.minimum_center_distance_m <= distance <= config.maximum_center_distance_m
            ):
                continue

            horizontal_margin = abs(dx) - abs(dy)
            longitudinal_margin = abs(dy) - abs(dx)
            if horizontal_margin >= config.minimum_axis_margin_m:
                relation = SpatialRelation.RIGHT_OF if dx > 0.0 else SpatialRelation.LEFT_OF
                add_pair(subject, reference, relation, dx, horizontal_margin, distance)
            elif longitudinal_margin >= config.minimum_axis_margin_m:
                relation = SpatialRelation.IN_FRONT_OF if dy > 0.0 else SpatialRelation.BEHIND
                add_pair(subject, reference, relation, dy, longitudinal_margin, distance)

            horizontal_distance = sqrt(dx * dx + dy * dy)
            vertical_margin = abs(dz) - horizontal_distance
            if (
                horizontal_distance <= config.maximum_vertical_horizontal_distance_m
                and vertical_margin >= config.minimum_axis_margin_m
            ):
                relation = SpatialRelation.ABOVE if dz > 0.0 else SpatialRelation.BELOW
                add_pair(subject, reference, relation, dz, vertical_margin, distance)

    selected = []
    for relation in SpatialRelation:
        ordered = sorted(buckets[relation], key=lambda candidate: str(candidate.candidate_id))
        selected.extend(ordered[: config.maximum_candidates_per_relation])
    return RelationOracleV1(
        scene_id=scene.scene_id,
        minimum_center_distance_m=config.minimum_center_distance_m,
        maximum_center_distance_m=config.maximum_center_distance_m,
        minimum_axis_margin_m=config.minimum_axis_margin_m,
        candidates=tuple(selected),
    )
