"""Typed, answer-free intermediate programs used by the dataset compiler.

Programs are compiler metadata.  They are never inserted into the main
RGB-and-language training prompt.  Keeping them answer-free lets us audit
composition splits without accidentally leaking an oracle label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SPATIAL_ATOMS = frozenset({"G", "F", "B", "M", "R", "P", "V"})


@dataclass(frozen=True)
class ProgramNode:
    node_id: str
    operation: str
    inputs: tuple[str, ...]
    input_types: tuple[str, ...]
    output_type: str
    variant: str | None = None

    def __post_init__(self) -> None:
        if self.operation not in SPATIAL_ATOMS:
            raise ValueError(f"unknown spatial operation: {self.operation}")
        if len(self.inputs) != len(self.input_types):
            raise ValueError(f"{self.node_id}: inputs and input_types differ in length")

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "node_id": self.node_id,
            "operation": self.operation,
            "inputs": list(self.inputs),
            "input_types": list(self.input_types),
            "output_type": self.output_type,
        }
        if self.variant:
            payload["variant"] = self.variant
        return payload


@dataclass(frozen=True)
class TypedProgram:
    program_id: str
    semantic_signature: str
    external_inputs: dict[str, str]
    nodes: tuple[ProgramNode, ...]
    answer_node: str

    @property
    def atoms(self) -> tuple[str, ...]:
        return tuple(sorted({node.operation for node in self.nodes}))

    def validate(self) -> None:
        available = dict(self.external_inputs)
        seen_nodes: set[str] = set()
        for node in self.nodes:
            if node.node_id in available or node.node_id in seen_nodes:
                raise ValueError(f"{self.program_id}: duplicate node {node.node_id}")
            for source, expected_type in zip(node.inputs, node.input_types, strict=True):
                if source not in available:
                    raise ValueError(
                        f"{self.program_id}/{node.node_id}: input {source} is not available"
                    )
                actual_type = available[source]
                if actual_type != expected_type:
                    raise ValueError(
                        f"{self.program_id}/{node.node_id}: {source} has type "
                        f"{actual_type}, expected {expected_type}"
                    )
            available[node.node_id] = node.output_type
            seen_nodes.add(node.node_id)
        if self.answer_node not in seen_nodes:
            raise ValueError(f"{self.program_id}: missing answer node {self.answer_node}")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": "epispace.typed_program.v1",
            "program_id": self.program_id,
            "semantic_signature": self.semantic_signature,
            "atoms": list(self.atoms),
            "external_inputs": dict(self.external_inputs),
            "nodes": [node.as_dict() for node in self.nodes],
            "answer_node": self.answer_node,
        }


def _n(
    node_id: str,
    operation: str,
    inputs: tuple[str, ...],
    input_types: tuple[str, ...],
    output_type: str,
    variant: str | None = None,
) -> ProgramNode:
    return ProgramNode(node_id, operation, inputs, input_types, output_type, variant)


def program_for(task_type: str) -> TypedProgram:
    """Return the semantic program for a source or trajectory-specific task."""

    factories = {
        "grounding_presence": _grounding,
        "last_seen_memory": _memory,
        "metric_distance": _metric,
        "egocentric_relation": _egocentric_relation,
        "cross_view_relation": _cross_view_relation,
        "cross_view_unknown": _cross_view_unknown,
        "evidence_presence_unknown": _evidence_presence_unknown,
        "evidence_presence_reveal": _evidence_presence_reveal,
        "counterfactual_verification": _counterfactual,
        "object_centric_perspective": _object_perspective,
        "unknown_abstention": _unknown,
        "rotation_change_detection": _rotation_change_detection,
        "orbit_identity": _orbit_identity,
        "elevation_relation_transfer": _elevation_relation_transfer,
        "occlusion_unknown": _occlusion_unknown,
        "occlusion_reveal": _occlusion_reveal,
        "target_view_prediction": _target_view_prediction,
    }
    try:
        program = factories[task_type]()
    except KeyError as error:
        raise ValueError(f"unsupported task type: {task_type}") from error
    program.validate()
    return program


def _grounding() -> TypedProgram:
    return TypedProgram(
        "grounding_presence.v1",
        "G(view,entity_hint)->V_presence",
        {"view": "view", "entity_hint": "entity_hint"},
        (
            _n("entity", "G", ("view", "entity_hint"), ("view", "entity_hint"), "entity"),
            _n("answer", "V", ("entity",), ("entity",), "boolean", "presence"),
        ),
        "answer",
    )


def _memory() -> TypedProgram:
    return TypedProgram(
        "last_seen_memory.v1",
        "G(sequence,entity_hint)->B_temporal->V_last_seen",
        {
            "sequence": "view_sequence",
            "entity_hint": "entity_hint",
            "belief_0": "belief",
        },
        (
            _n(
                "track",
                "G",
                ("sequence", "entity_hint"),
                ("view_sequence", "entity_hint"),
                "entity_track",
                "temporal",
            ),
            _n(
                "belief",
                "B",
                ("belief_0", "track"),
                ("belief", "entity_track"),
                "belief",
                "incremental",
            ),
            _n("answer", "V", ("belief",), ("belief",), "view_index", "last_seen"),
        ),
        "answer",
    )


def _metric() -> TypedProgram:
    return TypedProgram(
        "metric_distance.v1",
        "G(view_set,a)+G(view_set,b)->M_distance->V_tolerance",
        {"view_set": "view_set", "a_hint": "entity_hint", "b_hint": "entity_hint"},
        (
            _n("a", "G", ("view_set", "a_hint"), ("view_set", "entity_hint"), "entity"),
            _n("b", "G", ("view_set", "b_hint"), ("view_set", "entity_hint"), "entity"),
            _n("distance", "M", ("a", "b"), ("entity", "entity"), "metric", "distance"),
            _n("answer", "V", ("distance",), ("metric",), "metric", "tolerance"),
        ),
        "answer",
    )


def _egocentric_relation() -> TypedProgram:
    return TypedProgram(
        "egocentric_relation.v1",
        "G(view,a)+G(view,b)->F_ego(camera)->R->V",
        {
            "view": "view",
            "a_hint": "entity_hint",
            "b_hint": "entity_hint",
            "camera_frame": "frame",
            "world_frame": "frame",
        },
        (
            _n("a", "G", ("view", "a_hint"), ("view", "entity_hint"), "entity"),
            _n("b", "G", ("view", "b_hint"), ("view", "entity_hint"), "entity"),
            _n(
                "ego_transform",
                "F",
                ("world_frame", "camera_frame"),
                ("frame", "frame"),
                "transform",
                "ego",
            ),
            _n(
                "relation",
                "R",
                ("a", "b", "ego_transform"),
                ("entity", "entity", "transform"),
                "relation",
                "ego",
            ),
            _n("answer", "V", ("relation",), ("relation",), "relation", "relation"),
        ),
        "answer",
    )


def _cross_view_relation() -> TypedProgram:
    return TypedProgram(
        "cross_view_register_relation.v1",
        "G(view_a,a)+G(view_b,b)->F*_cross_view->B_global->R_canonical->V",
        {
            "view_a": "view",
            "view_b": "view",
            "a_hint": "entity_hint",
            "b_hint": "entity_hint",
            "belief_0": "belief",
        },
        (
            _n("a", "G", ("view_a", "a_hint"), ("view", "entity_hint"), "entity"),
            _n("b", "G", ("view_b", "b_hint"), ("view", "entity_hint"), "entity"),
            _n(
                "registration",
                "F",
                ("view_a", "view_b"),
                ("view", "view"),
                "transform_chain",
                "cross_view_chain",
            ),
            _n(
                "belief_a",
                "B",
                ("belief_0", "a", "registration"),
                ("belief", "entity", "transform_chain"),
                "belief",
                "global_register",
            ),
            _n(
                "belief_ab",
                "B",
                ("belief_a", "b", "registration"),
                ("belief", "entity", "transform_chain"),
                "belief",
                "global_register",
            ),
            _n(
                "relation",
                "R",
                ("belief_ab", "a", "b"),
                ("belief", "entity", "entity"),
                "relation",
                "canonical",
            ),
            _n("answer", "V", ("relation",), ("relation",), "relation", "relation"),
        ),
        "answer",
    )


def _counterfactual() -> TypedProgram:
    base = _cross_view_relation()
    nodes = (
        *base.nodes[:-1],
        _n(
            "answer",
            "V",
            ("relation", "claim"),
            ("relation", "relation_claim"),
            "claim_verification",
            "counterfactual_claim",
        ),
    )
    return TypedProgram(
        "counterfactual_cross_view.v1",
        "G(view_a,a)+G(view_b,b)->F*_cross_view->B_global->R_canonical->V_claim",
        {**base.external_inputs, "claim": "relation_claim"},
        nodes,
        "answer",
    )


def _cross_view_unknown() -> TypedProgram:
    return TypedProgram(
        "cross_view_evidence_unknown.v1",
        "G(view_set,a,b)->B_partial->V_unknown",
        {
            "view_set": "view_set",
            "a_hint": "entity_hint",
            "b_hint": "entity_hint",
            "belief_0": "belief",
        },
        (
            _n(
                "observed",
                "G",
                ("view_set", "a_hint", "b_hint"),
                ("view_set", "entity_hint", "entity_hint"),
                "partial_entity_set",
                "actual_input",
            ),
            _n(
                "belief",
                "B",
                ("belief_0", "observed"),
                ("belief", "partial_entity_set"),
                "belief",
                "partial",
            ),
            _n(
                "answer",
                "V",
                ("belief",),
                ("belief",),
                "epistemic_status",
                "unknown_relation",
            ),
        ),
        "answer",
    )


def _evidence_presence_unknown() -> TypedProgram:
    return TypedProgram(
        "evidence_presence_unknown.v1",
        "G(view_set,entity_hint)->B_partial->V_unknown",
        {
            "view_set": "view_set",
            "entity_hint": "entity_hint",
            "belief_0": "belief",
        },
        (
            _n(
                "observed",
                "G",
                ("view_set", "entity_hint"),
                ("view_set", "entity_hint"),
                "entity_evidence",
                "actual_input",
            ),
            _n(
                "belief",
                "B",
                ("belief_0", "observed"),
                ("belief", "entity_evidence"),
                "belief",
                "partial",
            ),
            _n(
                "answer",
                "V",
                ("belief",),
                ("belief",),
                "epistemic_status",
                "unknown_presence",
            ),
        ),
        "answer",
    )


def _evidence_presence_reveal() -> TypedProgram:
    return TypedProgram(
        "evidence_presence_reveal.v1",
        "G(view_set,entity_hint)->B_update->V_presence",
        {
            "view_set": "view_set",
            "entity_hint": "entity_hint",
            "belief_0": "belief",
        },
        (
            _n(
                "entity",
                "G",
                ("view_set", "entity_hint"),
                ("view_set", "entity_hint"),
                "entity",
                "actual_input",
            ),
            _n(
                "belief",
                "B",
                ("belief_0", "entity"),
                ("belief", "entity"),
                "belief",
                "evidence_update",
            ),
            _n(
                "answer",
                "V",
                ("belief", "entity"),
                ("belief", "entity"),
                "presence_status",
                "observed_presence",
            ),
        ),
        "answer",
    )


def _object_perspective() -> TypedProgram:
    return TypedProgram(
        "object_centric_perspective.v1",
        "G(view_set,o)+G(view_set,f)+G(view_set,t)->B_global->F_query(o,f)->P(t)->V",
        {
            "view_set": "view_set",
            "origin_hint": "entity_hint",
            "facing_hint": "entity_hint",
            "target_hint": "entity_hint",
            "belief_0": "belief",
        },
        (
            _n(
                "origin",
                "G",
                ("view_set", "origin_hint"),
                ("view_set", "entity_hint"),
                "entity",
            ),
            _n(
                "facing",
                "G",
                ("view_set", "facing_hint"),
                ("view_set", "entity_hint"),
                "entity",
            ),
            _n(
                "target",
                "G",
                ("view_set", "target_hint"),
                ("view_set", "entity_hint"),
                "entity",
            ),
            _n(
                "belief_origin",
                "B",
                ("belief_0", "origin"),
                ("belief", "entity"),
                "belief",
            ),
            _n(
                "belief_facing",
                "B",
                ("belief_origin", "facing"),
                ("belief", "entity"),
                "belief",
            ),
            _n(
                "belief",
                "B",
                ("belief_facing", "target"),
                ("belief", "entity"),
                "belief",
            ),
            _n(
                "query_frame",
                "F",
                ("origin", "facing"),
                ("entity", "entity"),
                "frame",
                "object_anchored",
            ),
            _n(
                "prediction",
                "P",
                ("belief", "query_frame", "target"),
                ("belief", "frame", "entity"),
                "view_prediction",
            ),
            _n(
                "answer",
                "V",
                ("prediction",),
                ("view_prediction",),
                "quadrant",
                "perspective",
            ),
        ),
        "answer",
    )


def _unknown() -> TypedProgram:
    return TypedProgram(
        "unknown_abstention.v1",
        "G(observed_views)->B_observed->V_unknown",
        {"observed_views": "view_set", "belief_0": "belief"},
        (
            _n(
                "observed",
                "G",
                ("observed_views",),
                ("view_set",),
                "entity_set",
                "observed_only",
            ),
            _n(
                "belief",
                "B",
                ("belief_0", "observed"),
                ("belief", "entity_set"),
                "belief",
                "epistemic",
            ),
            _n(
                "answer",
                "V",
                ("belief",),
                ("belief",),
                "epistemic_status",
                "unknown",
            ),
        ),
        "answer",
    )


def _rotation_change_detection() -> TypedProgram:
    return TypedProgram(
        "rotation_change_detection.v1",
        "G(view_before)+G(view_after)->F_rotate->V_entered",
        {"current_view": "view", "next_view": "view"},
        (
            _n(
                "before_entities",
                "G",
                ("current_view",),
                ("view",),
                "entity_set",
                "visible",
            ),
            _n(
                "after_entities",
                "G",
                ("next_view",),
                ("view",),
                "entity_set",
                "visible",
            ),
            _n(
                "rotated_frame",
                "F",
                ("current_view", "next_view"),
                ("view", "view"),
                "transform",
                "pure_rotation",
            ),
            _n(
                "answer",
                "V",
                ("before_entities", "after_entities", "rotated_frame"),
                ("entity_set", "entity_set", "transform"),
                "entity",
                "unique_entered_entity",
            ),
        ),
        "answer",
    )


def _orbit_identity() -> TypedProgram:
    return TypedProgram(
        "orbit_identity.v1",
        "G(view_a,focus)+G(view_b,focus)->F_object(azimuths)->V_identity",
        {
            "view_a": "view",
            "view_b": "view",
            "focus_hint": "entity_hint",
            "azimuth_pair": "rotation_pair",
        },
        (
            _n(
                "focus_a",
                "G",
                ("view_a", "focus_hint"),
                ("view", "entity_hint"),
                "entity",
            ),
            _n(
                "focus_b",
                "G",
                ("view_b", "focus_hint"),
                ("view", "entity_hint"),
                "entity",
            ),
            _n(
                "object_frame",
                "F",
                ("focus_a", "focus_b", "azimuth_pair"),
                ("entity", "entity", "rotation_pair"),
                "transform",
                "object_centric",
            ),
            _n(
                "answer",
                "V",
                ("focus_a", "focus_b", "object_frame"),
                ("entity", "entity", "transform"),
                "boolean",
                "identity",
            ),
        ),
        "answer",
    )


def _elevation_relation_transfer() -> TypedProgram:
    return TypedProgram(
        "elevation_relation_transfer.v1",
        "G(low,a,b)+G(high,a,b)->F_elevation->R_canonical->V_relation",
        {
            "low_view": "view",
            "high_view": "view",
            "a_hint": "entity_hint",
            "b_hint": "entity_hint",
        },
        (
            _n("a", "G", ("low_view", "a_hint"), ("view", "entity_hint"), "entity"),
            _n("b", "G", ("low_view", "b_hint"), ("view", "entity_hint"), "entity"),
            _n(
                "elevation_transform",
                "F",
                ("low_view", "high_view"),
                ("view", "view"),
                "transform",
                "elevation",
            ),
            _n(
                "relation",
                "R",
                ("a", "b", "elevation_transform"),
                ("entity", "entity", "transform"),
                "relation",
                "invariant",
            ),
            _n(
                "answer",
                "V",
                ("relation",),
                ("relation",),
                "relation",
                "relation_after_elevation",
            ),
        ),
        "answer",
    )


def _occlusion_unknown() -> TypedProgram:
    return TypedProgram(
        "occlusion_unknown.v1",
        "G(prefix)->B_occluded->V_unknown",
        {"prefix": "view_sequence", "belief_0": "belief"},
        (
            _n("observed", "G", ("prefix",), ("view_sequence",), "entity_set"),
            _n(
                "belief",
                "B",
                ("belief_0", "observed"),
                ("belief", "entity_set"),
                "belief",
                "occluded",
            ),
            _n(
                "answer",
                "V",
                ("belief",),
                ("belief",),
                "epistemic_status",
                "unknown",
            ),
        ),
        "answer",
    )


def _occlusion_reveal() -> TypedProgram:
    return TypedProgram(
        "occlusion_reveal.v1",
        "G(prefix)->B_occluded+G(decisive)->B_reveal->V",
        {"prefix": "view_sequence", "decisive_view": "view", "belief_0": "belief"},
        (
            _n("prefix_entities", "G", ("prefix",), ("view_sequence",), "entity_set"),
            _n(
                "occluded_belief",
                "B",
                ("belief_0", "prefix_entities"),
                ("belief", "entity_set"),
                "belief",
                "occluded",
            ),
            _n("revealed_entity", "G", ("decisive_view",), ("view",), "entity"),
            _n(
                "belief",
                "B",
                ("occluded_belief", "revealed_entity"),
                ("belief", "entity"),
                "belief",
                "reveal",
            ),
            _n("answer", "V", ("belief",), ("belief",), "entity", "reveal"),
        ),
        "answer",
    )


def _target_view_prediction() -> TypedProgram:
    return TypedProgram(
        "target_view_prediction.v1",
        "B_global->F_query(origin,facing)->P_visibility(target)->V_render",
        {
            "belief": "belief",
            "origin": "entity",
            "facing": "entity",
            "target": "entity",
            "target_render": "held_out_view",
        },
        (
            _n(
                "query_frame",
                "F",
                ("origin", "facing"),
                ("entity", "entity"),
                "frame",
                "object_anchored",
            ),
            _n(
                "prediction",
                "P",
                ("belief", "query_frame", "target"),
                ("belief", "frame", "entity"),
                "visibility_prediction",
                "novel_view",
            ),
            _n(
                "answer",
                "V",
                ("prediction", "target_render"),
                ("visibility_prediction", "held_out_view"),
                "boolean",
                "render_evidence",
            ),
        ),
        "answer",
    )
