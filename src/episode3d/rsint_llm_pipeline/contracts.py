"""Provider-neutral structured request contracts for the Rs_int subagents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from episode3d.qa_generation.backends import validate_json_schema
from episode3d.qa_generation.prompts import PromptRequest

from .profiles import SENSENOVA_CAPABILITIES, TYPED_OPERATIONS


@dataclass(frozen=True)
class SubagentRequest:
    """A provider-neutral request with locally validated input and output schemas.

    `backend_kwargs` can be passed unchanged to any `StructuredLLMBackend`.
    The role/skill metadata remains outside the natural-language prompt so it
    is also available for provenance and routing.
    """

    role: str
    stage: str
    skill_id: str | None
    prompt: str
    payload: dict[str, Any]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]

    def __post_init__(self) -> None:
        validate_json_schema(self.payload, self.input_schema)

    def as_prompt_request(self) -> PromptRequest:
        return PromptRequest(
            stage=self.stage,
            prompt=self.prompt,
            payload=self.payload,
            output_schema=self.output_schema,
        )

    def backend_kwargs(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "prompt": self.prompt,
            "payload": self.payload,
            "output_schema": self.output_schema,
        }


STRING_ARRAY = {"type": "array", "items": {"type": "string"}}

EPISODE_CONTEXT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "episode_id",
        "scene_id",
        "turn_index",
        "released_view_ids",
        "route_phase_zh",
        "coordinate_convention_zh",
        "prior_user_turns_zh",
    ],
    "properties": {
        "episode_id": {"type": "string", "minLength": 1},
        "scene_id": {"type": "string", "minLength": 1},
        "turn_index": {"type": "integer", "minimum": 1},
        "released_view_ids": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "route_phase_zh": {"type": "string", "minLength": 1, "maxLength": 100},
        "coordinate_convention_zh": {"type": "string", "maxLength": 200},
        # Only previous *user* turns are available to the question editor.  It
        # never receives the current answer or prior answer-bearing records.
        "prior_user_turns_zh": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 300},
        },
    },
}

CAPABILITY_CONTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "primary",
        "supporting",
        "official_subtask",
        "typed_operation_sequence",
        "learning_intent_zh",
        "response_profile",
    ],
    "properties": {
        "primary": {"type": "string", "enum": list(SENSENOVA_CAPABILITIES)},
        "supporting": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "enum": list(SENSENOVA_CAPABILITIES)},
        },
        "official_subtask": {"type": "string", "minLength": 1, "maxLength": 120},
        "typed_operation_sequence": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "enum": list(TYPED_OPERATIONS)},
        },
        "learning_intent_zh": {"type": "string", "minLength": 1, "maxLength": 240},
        "response_profile": {
            "type": "string",
            "enum": ["short", "evidence_conclusion", "evidence_transform_conclusion", "calibrated"],
        },
    },
}


def question_editor_input_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "request_id",
            "episode_context",
            "capability_contract",
            "intent_zh",
            "protected_slots",
            "required_slots",
            "surface_constraints_zh",
        ],
        "properties": {
            "schema_version": {"type": "string", "const": "epispace.rsint.question_editor.v1"},
            "request_id": {"type": "string", "minLength": 1},
            "episode_context": EPISODE_CONTEXT_SCHEMA,
            "capability_contract": CAPABILITY_CONTRACT_SCHEMA,
            "intent_zh": {"type": "string", "minLength": 1, "maxLength": 300},
            "protected_slots": {
                "type": "object",
                "additionalProperties": {"type": "string", "minLength": 1},
            },
            "required_slots": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "surface_constraints_zh": STRING_ARRAY,
        },
    }


def question_editor_output_schema(request_id: str, required_slots: list[str]) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["request_id", "question_template_zh", "used_slots"],
        "properties": {
            "request_id": {"type": "string", "enum": [request_id]},
            "question_template_zh": {"type": "string", "minLength": 4, "maxLength": 300},
            "used_slots": {
                "type": "array",
                "minItems": len(required_slots),
                "maxItems": len(required_slots),
                "items": {"type": "string", "enum": required_slots},
            },
        },
    }


def narration_skill_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "skill_id",
            "name_zh",
            "objective_zh",
            "required_moves_zh",
            "forbidden_moves_zh",
            "preferred_sentence_roles",
            "applicable_capabilities",
        ],
        "properties": {
            "skill_id": {"type": "string", "minLength": 1},
            "name_zh": {"type": "string", "minLength": 1},
            "objective_zh": {"type": "string", "minLength": 1},
            "required_moves_zh": STRING_ARRAY,
            "forbidden_moves_zh": STRING_ARRAY,
            "preferred_sentence_roles": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["evidence", "transform", "conclusion", "calibration"],
                },
            },
            "applicable_capabilities": {
                "type": "array",
                "items": {"type": "string", "enum": list(SENSENOVA_CAPABILITIES)},
            },
        },
    }


def strict_claim_sheet_schema() -> dict[str, Any]:
    """Schema for the Rs_int node-level claim sheet accepted by language roles."""

    numeric_surface = {
        "type": "object",
        "additionalProperties": False,
        "required": ["value", "unit", "tolerance", "surface"],
        "properties": {
            "value": {"type": "number"},
            "unit": {"type": "string"},
            "tolerance": {"type": "number", "minimum": 0},
            "surface": {"type": "string"},
        },
    }
    claim = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "claim_id",
            "node_id",
            "kind",
            "statement_zh",
            "value",
            "required",
            "evidence_view_ids",
            "evidence_entity_ids",
            "source_paths",
            "frame_id",
            "allowed_directions",
            "numeric_surfaces",
            "epistemic_scope",
            "reasoning_role",
        ],
        "properties": {
            "claim_id": {"type": "string", "minLength": 1},
            "node_id": {"type": "string", "minLength": 1},
            "kind": {"type": "string", "minLength": 1},
            "statement_zh": {"type": "string", "minLength": 1},
            "value": {},
            "required": {"type": "boolean"},
            "evidence_view_ids": STRING_ARRAY,
            "evidence_entity_ids": STRING_ARRAY,
            "source_paths": STRING_ARRAY,
            "frame_id": {"type": ["string", "null"]},
            "allowed_directions": STRING_ARRAY,
            "numeric_surfaces": {"type": "array", "items": numeric_surface},
            "epistemic_scope": {"type": "string", "minLength": 1},
            "reasoning_role": {
                "type": "string",
                "enum": ["cue", "transform", "conclusion", "calibration"],
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "claim_sheet_id",
            "claims",
            "required_claim_ids",
            "allowed_claim_ids",
            "answer_contract",
            "reasoning_contract",
        ],
        "properties": {
            "schema_version": {
                "type": "string",
                "const": "epispace.node_claim_sheet.v2",
            },
            "claim_sheet_id": {"type": "string", "minLength": 1},
            "claims": {"type": "array", "minItems": 1, "items": claim},
            "required_claim_ids": STRING_ARRAY,
            "allowed_claim_ids": STRING_ARRAY,
            "answer_contract": {
                "type": "object",
                "additionalProperties": False,
                "required": ["answer_key", "answer_value", "canonical_answer_zh", "status"],
                "properties": {
                    "answer_key": {"type": "string", "minLength": 1},
                    "answer_value": {},
                    "canonical_answer_zh": {"type": "string", "minLength": 1},
                    "status": {"type": "string", "minLength": 1},
                },
            },
            "reasoning_contract": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "required": ["required_sentence_roles", "enforce_claim_role_alignment"],
                "properties": {
                    "required_sentence_roles": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {
                            "type": "string",
                            "enum": ["evidence", "transform", "conclusion", "calibration"],
                        },
                    },
                    "enforce_claim_role_alignment": {"type": "boolean"},
                },
            },
        },
    }


def answer_narrator_input_schema(candidate_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "schema_version": {"type": "string", "const": "epispace.rsint.answer_narrator.v1"},
        "request_id": {"type": "string", "minLength": 1},
        "episode_context": EPISODE_CONTEXT_SCHEMA,
        "question_zh": {"type": "string", "minLength": 4, "maxLength": 400},
        "narration_skill": narration_skill_schema(),
        "claim_sheet": strict_claim_sheet_schema(),
        "style_contract": {
            "type": "object",
            "additionalProperties": False,
            "required": ["conclusion_first", "max_sentences", "instance_specific", "no_long_cot"],
            "properties": {
                "conclusion_first": {"type": "boolean"},
                "max_sentences": {"type": "integer", "minimum": 1, "maximum": 3},
                "instance_specific": {"type": "boolean"},
                "no_long_cot": {"type": "boolean"},
            },
        },
    }
    required = [
        "schema_version",
        "request_id",
        "episode_context",
        "question_zh",
        "narration_skill",
        "claim_sheet",
        "style_contract",
    ]
    if candidate_schema is not None:
        properties["candidate_answer"] = candidate_schema
        required.append("candidate_answer")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def critic_input_schema(candidate_schema: dict[str, Any]) -> dict[str, Any]:
    """Return the critic's role-specific, top-level strict input contract."""

    schema = answer_narrator_input_schema(candidate_schema)
    schema["properties"]["schema_version"] = {
        "type": "string",
        "const": "epispace.rsint.critic.v1",
    }
    return schema


def critic_output_schema(
    *, request_id: str, sentence_count: int, required_claim_ids: list[str]
) -> dict[str, Any]:
    if sentence_count < 1:
        raise ValueError("critic requires at least one candidate sentence")
    missing_claim_items: dict[str, Any] = {"type": "string"}
    if required_claim_ids:
        missing_claim_items["enum"] = required_claim_ids
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "request_id",
            "verdict",
            "supported",
            "unsupported_sentence_indices",
            "missing_required_claim_ids",
            "reason_codes",
            "repair_brief_zh",
        ],
        "properties": {
            "request_id": {"type": "string", "enum": [request_id]},
            "verdict": {"type": "string", "enum": ["accept", "revise", "reject"]},
            "supported": {"type": "boolean"},
            "unsupported_sentence_indices": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": sentence_count - 1},
            },
            "missing_required_claim_ids": {
                "type": "array",
                "items": missing_claim_items,
            },
            "reason_codes": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "unsupported_claim",
                        "missing_required_claim",
                        "wrong_direction",
                        "wrong_number",
                        "wrong_epistemic_scope",
                        "strategy_mismatch",
                        "geometry_jargon",
                        "long_cot",
                        "unverifiable_inference",
                    ],
                },
            },
            "repair_brief_zh": {"type": "string", "maxLength": 240},
        },
    }
