"""Dialogue schema builders and fail-closed validators."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "epispace.trajectory_dialogue_truth.v2"
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}")
FORBIDDEN_SURFACE = (
    "entity_id",
    "source_entity_id",
    "scene_ir",
    "runtime_semantic",
    "instance_id",
    "OBB",
    "oracle",
    "+X",
    "+Y",
)
ALLOWED_OUTPUT_PREFIX = {
    "G": ("entity", "grounding", "visibility", "track"),
    "F": ("transform", "frame", "pose", "motion"),
    "B": ("belief", "scene_state", "identity"),
    "M": ("scalar", "metric", "distance", "angle", "height"),
    "R": ("relation", "occlusion", "closure", "visibility_relation"),
    "P": ("prediction", "query_relation", "query_coordinates"),
    "V": ("boolean", "unknown", "verification"),
}
ROLE_ORDER = {"cue": 0, "evidence": 0, "transform": 1, "calibration": 2, "conclusion": 3}


class ContractError(RuntimeError):
    """A compiled round violates a typed or language contract."""


@dataclass
class RoundBuilder:
    turn_id: str
    new_view_ids: list[str]
    evidence_view_ids: list[str]
    capability: dict[str, Any]
    semantic_signature: str
    question_zh: str
    answer_key: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    nodes: list[dict[str, Any]] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    sentences: list[dict[str, Any]] = field(default_factory=list)

    def node(
        self,
        node_id: str,
        operation: str,
        input_types: list[str],
        output_type: str,
        value: Any,
        depends_on: list[str] | None = None,
    ) -> str:
        self.nodes.append(
            {
                "node_id": node_id,
                "operation": operation,
                "input_types": input_types,
                "output_type": output_type,
                "depends_on": depends_on or [],
                "value": value,
            }
        )
        return node_id

    def claim(
        self,
        claim_id: str,
        node_id: str,
        kind: str,
        statement_zh: str,
        value: Any,
        evidence_view_ids: list[str],
        reasoning_role: str,
        required: bool = True,
    ) -> str:
        self.claims.append(
            {
                "claim_id": claim_id,
                "node_id": node_id,
                "kind": kind,
                "statement_zh": statement_zh,
                "value": value,
                "evidence_view_ids": evidence_view_ids,
                "reasoning_role": reasoning_role,
                "required": required,
            }
        )
        return claim_id

    def sentence(self, role: str, text: str, claim_ids: list[str]) -> None:
        self.sentences.append({"role": role, "text": text, "claim_ids": claim_ids})

    def payload(self, round_index: int) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "round_index": round_index,
            "new_view_ids": self.new_view_ids,
            "evidence_view_ids": self.evidence_view_ids,
            "capability": self.capability,
            "program": {"semantic_signature": self.semantic_signature, "nodes": self.nodes},
            "claim_sheet": {"claims": self.claims},
            "question_zh": self.question_zh,
            "answer_sentences": self.sentences,
            "answer_zh": "".join(row["text"] for row in self.sentences),
            "answer_key": self.answer_key,
            "diagnostics": self.diagnostics,
        }


def validate_artifact(artifact: dict[str, Any]) -> None:
    all_views = artifact["view_ids"]
    released: list[str] = []
    for round_ in artifact["rounds"]:
        for view_id in round_["new_view_ids"]:
            if view_id not in all_views:
                raise ContractError(f"{round_['turn_id']}: unknown released view {view_id}")
            if view_id in released:
                raise ContractError(f"{round_['turn_id']}: view released twice: {view_id}")
            released.append(view_id)
        for view_id in round_["evidence_view_ids"]:
            if view_id not in released:
                raise ContractError(f"{round_['turn_id']}: future evidence leaked: {view_id}")
        _validate_program(round_)
        _validate_claims_and_language(round_)
    if set(released) != set(all_views):
        missing = sorted(set(all_views) - set(released))
        raise ContractError(f"unreleased views: {missing}")


def _validate_program(round_: dict[str, Any]) -> None:
    seen: set[str] = set()
    for node in round_["program"]["nodes"]:
        node_id, op = node["node_id"], node["operation"]
        if node_id in seen:
            raise ContractError(f"{round_['turn_id']}: duplicate node {node_id}")
        if op not in ALLOWED_OUTPUT_PREFIX:
            raise ContractError(f"{round_['turn_id']}: unknown operation {op}")
        if not str(node["output_type"]).startswith(ALLOWED_OUTPUT_PREFIX[op]):
            raise ContractError(f"{round_['turn_id']}: {op} cannot output {node['output_type']}")
        missing = set(node["depends_on"]) - seen
        if missing:
            raise ContractError(f"{round_['turn_id']}: non-topological dependencies {missing}")
        if op == "P" and not (
            any("belief" in t or "state" in t for t in node["input_types"])
            and any("frame" in t or "view" in t for t in node["input_types"])
        ):
            raise ContractError(f"{round_['turn_id']}: P requires belief/state and frame/view")
        if op == "R" and not any(
            "frame" in t or "view" in t or "belief" in t or "entity_pair" in t or "pose" in t
            for t in node["input_types"]
        ):
            raise ContractError(f"{round_['turn_id']}: R lacks spatial context")
        seen.add(node_id)
    if not round_["program"]["nodes"] or round_["program"]["nodes"][-1]["operation"] != "V":
        raise ContractError(f"{round_['turn_id']}: program must end in V")


def _validate_claims_and_language(round_: dict[str, Any]) -> None:
    node_ids = {n["node_id"] for n in round_["program"]["nodes"]}
    claims = round_["claim_sheet"]["claims"]
    claim_ids = {c["claim_id"] for c in claims}
    if len(claim_ids) != len(claims):
        raise ContractError(f"{round_['turn_id']}: duplicate claim ids")
    for claim in claims:
        if claim["node_id"] not in node_ids:
            raise ContractError(f"{round_['turn_id']}: claim references missing node")
        if set(claim["evidence_view_ids"]) - set(round_["evidence_view_ids"]):
            raise ContractError(f"{round_['turn_id']}: claim evidence omitted from round evidence")
    roles = [ROLE_ORDER[s["role"]] for s in round_["answer_sentences"]]
    if roles != sorted(roles):
        raise ContractError(
            f"{round_['turn_id']}: answer does not follow cue->transform->conclusion"
        )
    used_claims: set[str] = set()
    for sentence in round_["answer_sentences"]:
        missing = set(sentence["claim_ids"]) - claim_ids
        if missing:
            raise ContractError(f"{round_['turn_id']}: sentence uses unknown claims {missing}")
        used_claims.update(sentence["claim_ids"])
    required = {c["claim_id"] for c in claims if c["required"]}
    if required - used_claims:
        raise ContractError(f"{round_['turn_id']}: required claims not verbalized")
    surface = round_["question_zh"] + round_["answer_zh"]
    if UUID_RE.search(surface):
        raise ContractError(f"{round_['turn_id']}: UUID leaked into language")
    for token in FORBIDDEN_SURFACE:
        if token in surface:
            raise ContractError(f"{round_['turn_id']}: hidden token leaked: {token}")
    if not round_["answer_sentences"]:
        raise ContractError(f"{round_['turn_id']}: empty answer")
