"""Serializable typed operation graph."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from spatial_episode.contracts.base import ContractModel
from spatial_episode.domain.operations import (
    OperationGraph,
    OperationKind,
    OperationNode,
    ValueType,
)


class OperationNodeV1(ContractModel):
    node_id: str = Field(min_length=1)
    operation: OperationKind
    inputs: tuple[str, ...]
    output_type: ValueType
    parameters: dict[str, object] = Field(default_factory=dict)


class OperationGraphV1(ContractModel):
    schema_version: Literal["operation_graph.v1"] = "operation_graph.v1"
    external_inputs: dict[str, ValueType]
    nodes: tuple[OperationNodeV1, ...]
    answer_node: str = Field(min_length=1)

    @model_validator(mode="after")
    def graph_is_well_typed(self) -> OperationGraphV1:
        graph = OperationGraph(
            external_inputs=self.external_inputs,
            nodes=tuple(
                OperationNode(
                    node_id=node.node_id,
                    operation=node.operation,
                    inputs=node.inputs,
                    output_type=node.output_type,
                    parameters=node.parameters,
                )
                for node in self.nodes
            ),
            answer_node=self.answer_node,
        )
        graph.validate()
        return self
