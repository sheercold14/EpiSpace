"""Typed spatial operation graph and its static validator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from spatial_episode.domain.errors import GraphValidationError


class OperationKind(StrEnum):
    GROUNDING = "G"
    FRAME = "F"
    BELIEF = "B"
    METRIC = "M"
    RELATION = "R"
    PERSPECTIVE = "P"
    VERIFY = "V"


class ValueType(StrEnum):
    VIEW = "view"
    ENTITY = "entity"
    ENTITY_SET = "entity_set"
    FRAME = "frame"
    TRANSFORM = "transform"
    BELIEF = "belief"
    METRIC = "metric"
    RELATION = "relation"
    VIEW_PREDICTION = "view_prediction"
    BOOLEAN = "boolean"


@dataclass(frozen=True, slots=True)
class OperationSignature:
    inputs: tuple[ValueType, ...]
    output: ValueType


SIGNATURES: Mapping[OperationKind, tuple[OperationSignature, ...]] = MappingProxyType(
    {
        OperationKind.GROUNDING: (
            OperationSignature((ValueType.VIEW,), ValueType.ENTITY_SET),
            OperationSignature((ValueType.VIEW, ValueType.ENTITY), ValueType.ENTITY),
        ),
        OperationKind.FRAME: (
            OperationSignature((ValueType.FRAME, ValueType.FRAME), ValueType.TRANSFORM),
        ),
        OperationKind.BELIEF: (
            OperationSignature(
                (ValueType.BELIEF, ValueType.ENTITY_SET, ValueType.TRANSFORM),
                ValueType.BELIEF,
            ),
            OperationSignature(
                (ValueType.BELIEF, ValueType.ENTITY, ValueType.TRANSFORM),
                ValueType.BELIEF,
            ),
        ),
        OperationKind.METRIC: (
            OperationSignature(
                (ValueType.ENTITY, ValueType.ENTITY, ValueType.FRAME), ValueType.METRIC
            ),
        ),
        OperationKind.RELATION: (
            OperationSignature(
                (ValueType.ENTITY, ValueType.ENTITY, ValueType.FRAME), ValueType.RELATION
            ),
        ),
        OperationKind.PERSPECTIVE: (
            OperationSignature((ValueType.BELIEF, ValueType.VIEW), ValueType.VIEW_PREDICTION),
        ),
        OperationKind.VERIFY: tuple(
            OperationSignature((input_type,), ValueType.BOOLEAN)
            for input_type in (
                ValueType.ENTITY,
                ValueType.ENTITY_SET,
                ValueType.BELIEF,
                ValueType.METRIC,
                ValueType.RELATION,
                ValueType.VIEW_PREDICTION,
            )
        ),
    }
)


@dataclass(frozen=True, slots=True)
class OperationNode:
    node_id: str
    operation: OperationKind
    inputs: tuple[str, ...]
    output_type: ValueType
    parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id:
            raise GraphValidationError("operation node ID cannot be empty")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True, slots=True)
class OperationGraph:
    external_inputs: Mapping[str, ValueType]
    nodes: tuple[OperationNode, ...]
    answer_node: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "external_inputs", MappingProxyType(dict(self.external_inputs)))

    def validate(self) -> None:
        nodes_by_id: dict[str, OperationNode] = {}
        for node in self.nodes:
            if node.node_id in nodes_by_id or node.node_id in self.external_inputs:
                raise GraphValidationError(f"duplicate graph value ID: {node.node_id}")
            nodes_by_id[node.node_id] = node

        if self.answer_node not in nodes_by_id:
            raise GraphValidationError("answer node must reference an operation node")

        for node in self.nodes:
            for input_id in node.inputs:
                if input_id not in self.external_inputs and input_id not in nodes_by_id:
                    raise GraphValidationError(
                        f"node {node.node_id} references missing input {input_id}"
                    )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visited:
                return
            if node_id in visiting:
                raise GraphValidationError(f"operation graph contains a cycle at {node_id}")
            visiting.add(node_id)
            for input_id in nodes_by_id[node_id].inputs:
                if input_id in nodes_by_id:
                    visit(input_id)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in nodes_by_id:
            visit(node_id)

        output_types = dict(self.external_inputs)

        def resolve(node_id: str) -> ValueType:
            if node_id in output_types:
                return output_types[node_id]
            node = nodes_by_id[node_id]
            input_types = tuple(resolve(input_id) for input_id in node.inputs)
            candidates = SIGNATURES[node.operation]
            if not any(
                signature.inputs == input_types and signature.output == node.output_type
                for signature in candidates
            ):
                expected = " or ".join(
                    f"({', '.join(item.value for item in signature.inputs)})"
                    f" -> {signature.output.value}"
                    for signature in candidates
                )
                actual = (
                    f"({', '.join(item.value for item in input_types)}) -> {node.output_type.value}"
                )
                raise GraphValidationError(
                    f"invalid signature for {node.node_id}/{node.operation.value}: "
                    f"got {actual}; expected {expected}"
                )
            output_types[node_id] = node.output_type
            return node.output_type

        for node_id in nodes_by_id:
            resolve(node_id)

        answer = nodes_by_id[self.answer_node]
        if (
            answer.operation is not OperationKind.VERIFY
            or answer.output_type is not ValueType.BOOLEAN
        ):
            raise GraphValidationError("answer node must be a V operation returning boolean")
