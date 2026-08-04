"""Automated geometry -> subagent -> verified incremental-dialogue pipeline."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from episode3d.qa_generation.backends import (
    CodexExecBackend,
    ReplayBackend,
    StructuredLLMBackend,
)
from episode3d.qa_generation.schemas import canonical_json
from episode3d.qa_generation.validators import validate_answer_response

from .prompts import (
    build_answer_narrator_request,
    build_critic_request,
    build_question_editor_request,
)
from .truth import FRAME_SURFACE_ZH, RsIntTruthCompiler


class RsIntPipelineError(RuntimeError):
    """The pilot failed a generation, geometry, or export gate."""


_ROUTE_PHASE = {
    1: "起点观察",
    2: "沿去程继续前进",
    3: "抵达厨房端并更新记忆",
    4: "转身返程并连接非共视地标",
    5: "返程中的假想视角读取",
    6: "闭环回到起点",
    7: "无新图的全局状态总结",
}

_RESPONSE_PROFILE = {
    1: "short",
    2: "evidence_conclusion",
    3: "evidence_conclusion",
    4: "evidence_transform_conclusion",
    5: "evidence_transform_conclusion",
    6: "calibrated",
    7: "evidence_conclusion",
}

_OFFICIAL = frozenset({"MM", "SR", "MR", "PT", "CR"})
_INTERNAL_JARGON = (
    "canonical",
    "DAG",
    "operation graph",
    "claim",
    "certificate",
    "oracle",
    "OBB",
    "AABB",
    "真值",
    "坐标注册",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RsIntPipelineError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise RsIntPipelineError(f"expected JSON object: {path}")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(canonical_json(value).encode("utf-8"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(dict(row)) + "\n" for row in rows), encoding="utf-8")


def _resolve(base: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RsIntPipelineError(f"{field} must be a non-empty path")
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _backend(config: Mapping[str, Any], cache_dir: Path) -> StructuredLLMBackend:
    kind = str(config.get("kind", "codex_exec"))
    model = str(config.get("model", "default"))
    if kind == "codex_exec":
        return CodexExecBackend(
            cache_dir,
            model=None if model == "default" else model,
            codex_binary=str(config.get("codex_binary", "codex")),
            timeout_seconds=float(config.get("timeout_seconds", 300)),
        )
    if kind == "replay":
        return ReplayBackend(
            cache_dir,
            provider=str(config.get("provider", "codex-cli")),
            model=model,
        )
    raise RsIntPipelineError(f"unsupported backend kind {kind!r}")


def _episode_context(
    truth: Mapping[str, Any], turn: Mapping[str, Any], prior_questions: Sequence[str]
) -> dict[str, Any]:
    return {
        "episode_id": str(truth["episode_id"]),
        "scene_id": str(truth["scene_id"]),
        "turn_index": int(turn["round_index"]),
        "released_view_ids": list(turn["available_view_ids"]),
        "route_phase_zh": _ROUTE_PHASE[int(turn["round_index"])],
        "coordinate_convention_zh": FRAME_SURFACE_ZH,
        "prior_user_turns_zh": list(prior_questions),
    }


def _capability_contract(turn: Mapping[str, Any]) -> dict[str, Any]:
    capability = turn["capability"]
    nodes = turn["program"]["nodes"]
    return {
        "primary": str(capability["primary"]),
        "supporting": [str(value) for value in capability["supporting"] if str(value) in _OFFICIAL],
        "official_subtask": str(capability["sense_nova_subtask"]),
        "typed_operation_sequence": [str(node["operation"]) for node in nodes],
        "learning_intent_zh": str(turn["question_blueprint"]["intent_zh"]),
        "response_profile": _RESPONSE_PROFILE[int(turn["round_index"])],
    }


def _render_question(blueprint: Mapping[str, Any], response: Mapping[str, Any]) -> str:
    template = response.get("question_template_zh")
    used = response.get("used_slots")
    if not isinstance(template, str) or not isinstance(used, list):
        raise RsIntPipelineError("question editor returned malformed fields")
    required = [str(value) for value in blueprint["required_slots"]]
    slots = {str(key): str(value) for key, value in blueprint["slot_values"].items()}
    if set(used) != set(required) or len(used) != len(set(used)):
        raise RsIntPipelineError("question editor did not preserve the protected-slot set")
    for slot in required:
        marker = "{{" + slot + "}}"
        if template.count(marker) != 1:
            raise RsIntPipelineError(f"question slot {slot!r} must occur exactly once")
        if slots[slot] in template:
            raise RsIntPipelineError(f"question editor inlined protected slot {slot!r}")
    if any(term.casefold() in template.casefold() for term in _INTERNAL_JARGON):
        raise RsIntPipelineError("question exposes internal geometry terminology")
    rendered = template
    for slot, value in slots.items():
        rendered = rendered.replace("{{" + slot + "}}", value)
    if "{{" in rendered or "}}" in rendered:
        raise RsIntPipelineError("question contains unresolved protected slots")
    if len(rendered) > 400 or not rendered.endswith(("？", "。")):
        raise RsIntPipelineError("rendered question is not a concise Chinese question/instruction")
    return rendered


def _validate_answer(
    turn: Mapping[str, Any], answer: Mapping[str, Any], skill_id: str
) -> dict[str, Any]:
    sheet = copy.deepcopy(dict(turn["claim_sheet"]))
    sheet["exposure_view_ids"] = list(turn["available_view_ids"])
    sheet["answer_contract"]["answer_status"] = sheet["answer_contract"]["status"]
    report = validate_answer_response(
        sheet,
        answer,
        expected_strategy_id=skill_id,
        dedup_threshold=1.0,
    )
    if not report.passed:
        raise RsIntPipelineError("answer validator rejected output: " + "; ".join(report.errors))
    checks = dict(report.checks)
    reasoning_contract = sheet.get("reasoning_contract")
    if isinstance(reasoning_contract, Mapping):
        sentences = answer.get("sentences")
        if not isinstance(sentences, list):
            raise RsIntPipelineError("reasoning contract requires structured answer sentences")
        expected_roles = [
            str(value) for value in reasoning_contract.get("required_sentence_roles", [])
        ]
        actual_roles = [str(sentence.get("role", "")) for sentence in sentences]
        role_order_valid = actual_roles == expected_roles
        checks["reasoning_role_order"] = role_order_valid
        if not role_order_valid:
            raise RsIntPipelineError(
                f"answer reasoning roles {actual_roles} do not match {expected_roles}"
            )
        if reasoning_contract.get("enforce_claim_role_alignment") is True:
            claim_lookup = {
                str(claim["claim_id"]): claim for claim in sheet.get("claims", [])
            }
            expected_claim_role = {
                "evidence": "cue",
                "transform": "transform",
                "conclusion": "conclusion",
                "calibration": "calibration",
            }
            aligned = True
            for sentence, sentence_role in zip(sentences, expected_roles, strict=True):
                required_claim_role = expected_claim_role[sentence_role]
                cited = [
                    claim_lookup.get(str(claim_id))
                    for claim_id in sentence.get("claim_ids", [])
                ]
                if not any(
                    claim is not None and claim.get("reasoning_role") == required_claim_role
                    for claim in cited
                ):
                    aligned = False
                    break
            checks["claim_reasoning_role_alignment"] = aligned
            if not aligned:
                raise RsIntPipelineError(
                    "answer sentence roles are not anchored to cue/transform/conclusion claims"
                )
    surface = str(answer.get("surface_answer_zh", ""))
    if any(term in surface for term in ("相机轨迹", "相机位姿")):
        raise RsIntPipelineError("answer exposes implementation-flavored camera wording")
    return {"passed": True, "checks": checks, "errors": []}


def _validate_critic(response: Mapping[str, Any]) -> dict[str, Any]:
    accepted = (
        response.get("verdict") == "accept"
        and response.get("supported") is True
        and response.get("unsupported_sentence_indices") == []
        and response.get("missing_required_claim_ids") == []
        and response.get("reason_codes") == []
    )
    if not accepted:
        raise RsIntPipelineError(
            "independent critic rejected answer: " + json.dumps(response, ensure_ascii=False)
        )
    return {"passed": True, "checks": {"independent_critic_accept": True}, "errors": []}


def _provenance(
    result: Any,
    *,
    role: str,
    prompt_profile: str,
    skill_ids: Sequence[str] = (),
    input_sha256: str,
) -> dict[str, Any]:
    response = result.response
    generated_at = datetime.now(timezone.utc)
    if result.cache_path:
        generated_at = datetime.fromtimestamp(
            Path(result.cache_path).stat().st_mtime, tz=timezone.utc
        )
    return {
        "stage": result.stage,
        "role": role,
        "provider": result.provider,
        "model": result.model,
        "model_revision": "provider_default",
        "prompt_profile": prompt_profile,
        "prompt_version": "epispace.rsint.subagent.v2",
        "context_policy": "role_isolated_structured_json",
        "skill_ids": list(skill_ids),
        "input_sha256": input_sha256,
        "request_hash": result.request_hash,
        "response_hash": _sha256_json(response),
        "cache_hit": bool(result.cache_hit),
        "cache_path": result.cache_path,
        "attempt": 1,
        "agent_task_id": f"subagent-{result.request_hash[:20]}",
        "generated_at": generated_at.isoformat(),
    }


def _content_items_for_views(
    observations: Mapping[str, Mapping[str, Any]], view_ids: Sequence[str]
) -> list[dict[str, str]]:
    content: list[dict[str, str]] = []
    for view_id in view_ids:
        observation = observations[view_id]
        content.extend(
            [
                {
                    "type": "text",
                    "text": f"<图{observation['display_index']}：{view_id}>",
                },
                {"type": "image", "image": str(observation["rgb"])},
            ]
        )
    return content


def _episode_sft(artifact: Mapping[str, Any]) -> dict[str, Any]:
    observations = {str(value["view_id"]): value for value in artifact["observations"]}
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "你会按真实移动顺序逐轮收到住宅内的RGB画面。只依据当前已释放画面和对话历史回答；"
                + FRAME_SURFACE_ZH
                + "。证据不足时明确说无法确定。"
            ),
        }
    ]
    spans: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for turn in artifact["rounds"]:
        user_content = _content_items_for_views(observations, turn["new_view_ids"])
        user_content.append({"type": "text", "text": turn["generated"]["question"]["question_zh"]})
        messages.append({"role": "user", "content": user_content})
        answer = str(turn["generated"]["answer"]["surface_answer_zh"])
        messages.append({"role": "assistant", "content": answer})
        spans.append(
            {
                "message_index": len(messages) - 1,
                "turn_id": turn["turn_id"],
                "claim_sheet_id": turn["claim_sheet"]["claim_sheet_id"],
                "source_claim_ids": list(turn["claim_sheet"]["required_claim_ids"]),
                "answer_sha256": _sha256_bytes(answer.encode("utf-8")),
                "supervise_eos": True,
            }
        )
        comparisons.append(
            {
                "comparison_id": f"rsint17-{turn['turn_id']}",
                "unit_id": turn["turn_id"],
                "arm": "episode",
                "exposure_view_ids": list(turn["available_view_ids"]),
                "exposure_sha256": _sha256_json(
                    [
                        observations[value]["rgb_sha256"]
                        for value in turn["available_view_ids"]
                    ]
                ),
                "answer_sha256": _sha256_bytes(answer.encode("utf-8")),
                "supervision_matched": True,
                "context_treatment": {
                    "episode": "prior_qa_history",
                    "isolated": "none",
                },
            }
        )
    return {
        "record_id": artifact["episode_id"] + "-sft",
        "format": "qwen_multimodal_chat",
        "sample_type": "incremental_episodic_dialogue",
        "messages": messages,
        "loss_policy": {
            "train_on": "all_assistant_turns",
            "mask_system_user_visual": True,
            "supervise_eos": True,
            "normalization": "mean_tokens_per_turn_then_mean_turns",
        },
        "assistant_span_contract": spans,
        "comparison_contracts": comparisons,
        "hidden_meta": {
            "dialogue_artifact_sha256": artifact["artifact_sha256"],
            "claim_sheet_ids": [
                turn["claim_sheet"]["claim_sheet_id"] for turn in artifact["rounds"]
            ],
            "program_signatures": [
                turn["program"]["semantic_signature"] for turn in artifact["rounds"]
            ],
            "development_only": True,
        },
    }


def _isolated_sft(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    observations = {str(value["view_id"]): value for value in artifact["observations"]}
    rows: list[dict[str, Any]] = []
    for turn in artifact["rounds"]:
        user_content = _content_items_for_views(observations, turn["available_view_ids"])
        user_content.append({"type": "text", "text": turn["generated"]["question"]["question_zh"]})
        answer = str(turn["generated"]["answer"]["surface_answer_zh"])
        exposure_hash = _sha256_json(
            [observations[value]["rgb_sha256"] for value in turn["available_view_ids"]]
        )
        rows.append(
            {
                "record_id": f"{artifact['episode_id']}-{turn['turn_id']}-isolated",
                "format": "qwen_multimodal_chat",
                "sample_type": "isolated_prefix_matched",
                "messages": [
                    {
                        "role": "system",
                        "content": "只依据这些按顺序给出的RGB画面回答；" + FRAME_SURFACE_ZH + "。",
                    },
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": answer},
                ],
                "loss_policy": {
                    "train_on": "all_assistant_turns",
                    "mask_system_user_visual": True,
                    "supervise_eos": True,
                    "normalization": "mean_tokens_per_turn_then_mean_turns",
                },
                "assistant_span_contract": [
                    {
                        "message_index": 2,
                        "turn_id": turn["turn_id"],
                        "claim_sheet_id": turn["claim_sheet"]["claim_sheet_id"],
                        "source_claim_ids": list(
                            turn["claim_sheet"]["required_claim_ids"]
                        ),
                        "answer_sha256": _sha256_bytes(answer.encode("utf-8")),
                        "supervise_eos": True,
                    }
                ],
                "comparison_contract": {
                    "comparison_id": f"rsint17-{turn['turn_id']}",
                    "unit_id": turn["turn_id"],
                    "arm": "isolated",
                    "exposure_view_ids": list(turn["available_view_ids"]),
                    "exposure_sha256": exposure_hash,
                    "answer_sha256": _sha256_bytes(answer.encode("utf-8")),
                    "supervision_matched": True,
                    "context_treatment": {"episode": "prior_qa_history", "isolated": "none"},
                },
            }
        )
    return rows


def _observable_state_delta(artifact: Mapping[str, Any], turn_index: int) -> dict[str, list[str]]:
    turns = artifact["rounds"]

    def grounded_categories(turn: Mapping[str, Any]) -> set[str]:
        categories: set[str] = set()
        for node in turn["program"]["nodes"]:
            if node["operation"] != "G":
                continue
            value = node["value"]
            candidates: list[str] = []
            if isinstance(value, str):
                candidates = [value]
            elif isinstance(value, list):
                candidates = [str(item) for item in value]
            elif isinstance(value, Mapping):
                if isinstance(value.get("entity"), str):
                    candidates = [str(value["entity"])]
                elif isinstance(value.get("entities"), list):
                    candidates = [str(item) for item in value["entities"]]
            # An unobserved query target is not an observable state addition.
            categories.update(
                item for item in candidates if item != "bed" and not item.startswith("view-")
            )
        return categories

    current_ids = grounded_categories(turns[turn_index])
    previous_ids = set().union(*(grounded_categories(prior) for prior in turns[:turn_index]))
    return {
        "added": sorted(current_ids - previous_ids),
        "reobserved": sorted(current_ids & previous_ids),
        "seen_not_current": sorted(previous_ids - current_ids),
    }


def _state_aux_sft(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    observations = {str(value["view_id"]): value for value in artifact["observations"]}
    rows: list[dict[str, Any]] = []
    for index, turn in enumerate(artifact["rounds"]):
        user_content = _content_items_for_views(observations, turn["available_view_ids"])
        user_content.append({"type": "text", "text": turn["generated"]["question"]["question_zh"]})
        claim_ids_by_node: dict[str, list[str]] = {}
        for claim in turn["claim_sheet"]["claims"]:
            claim_ids_by_node.setdefault(str(claim["node_id"]), []).append(str(claim["claim_id"]))
        nodes = [
            {
                "node_id": node["node_id"],
                "operation": node["operation"],
                "output_type": node["output_type"],
                "value": node["value"],
                "claim_ids": claim_ids_by_node.get(str(node["node_id"]), []),
            }
            for node in turn["program"]["nodes"]
        ]
        target = {
            "state_delta": _observable_state_delta(artifact, index),
            "program_id": turn["program"]["program_id"],
            "nodes": nodes,
            "answer_key": turn["claim_sheet"]["answer_contract"]["answer_key"],
            "verify": {"passed": True},
        }
        answer = canonical_json(target)
        rows.append(
            {
                "record_id": f"{artifact['episode_id']}-{turn['turn_id']}-state-op",
                "format": "qwen_multimodal_chat",
                "sample_type": "state_operation_auxiliary",
                "messages": [
                    {
                        "role": "system",
                        "content": "根据已给RGB和当前问题，输出严格JSON的可观察状态增量、空间操作及其执行值。",
                    },
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": answer},
                ],
                "loss_policy": {
                    "train_on": "all_assistant_turns",
                    "mask_system_user_visual": True,
                    "supervise_eos": True,
                    "normalization": "mean_tokens_per_turn_then_mean_turns",
                },
                "assistant_span_contract": [
                    {
                        "message_index": 2,
                        "turn_id": turn["turn_id"],
                        "claim_sheet_id": turn["claim_sheet"]["claim_sheet_id"],
                        "answer_sha256": _sha256_bytes(answer.encode("utf-8")),
                        "supervise_eos": True,
                    }
                ],
                "hidden_meta": {
                    "auxiliary_only": True,
                    "must_not_mix_with_surface_answer": True,
                    "development_only": True,
                },
            }
        )
    return rows


def _web_payload(artifact: Mapping[str, Any], media_dir: Path, web_output: Path) -> dict[str, Any]:
    media_dir.mkdir(parents=True, exist_ok=True)
    web_artifact = copy.deepcopy(dict(artifact))
    for observation in web_artifact["observations"]:
        source = Path(str(observation["rgb"]))
        destination = media_dir / source.name
        if not destination.is_file() or _file_sha256(destination) != observation["rgb_sha256"]:
            shutil.copy2(source, destination)
        observation["rgb"] = os.path.relpath(destination, web_output.parent.parent)
        observation.pop("world_from_camera", None)
    return {
        "schema_version": "epispace.rsint_dialogue_web.v1",
        "title": "INCREMENTAL FLAGSHIP V2 · Rs_int 11图7轮",
        "subtitle": "图像按 1/2/2/3/2/1/0 逐轮释放；组合题强制线索→变换→结论，PT采用OBB与角度margin双重验证。",
        "artifact": web_artifact,
        "training_exports": [
            "../data/rsint_dialogue_pilot_v2/train.dialogue_episode_sft.jsonl",
            "../data/rsint_dialogue_pilot_v2/train.dialogue_isolated_sft.jsonl",
            "../data/rsint_dialogue_pilot_v2/train.dialogue_state_op_aux_sft.jsonl",
        ],
    }


def build_rsint_dialogue(config_path: str | Path) -> dict[str, Any]:
    """Run all three subagent roles and emit only fully accepted artifacts."""

    path = Path(config_path).resolve()
    config = _read_json(path)
    if config.get("schema_version") != "epispace.rsint_llm_pipeline_config.v1":
        raise RsIntPipelineError("unsupported Rs_int pipeline config schema")
    base = path.parent
    source = config.get("source")
    output = config.get("output")
    backend_config = config.get("backend")
    if not all(isinstance(value, Mapping) for value in (source, output, backend_config)):
        raise RsIntPipelineError("config requires source, output, and backend objects")
    bundle_root = _resolve(base, source["bundle"], "source.bundle")
    output_dir = _resolve(base, output["directory"], "output.directory")
    cache_dir = _resolve(base, output["cache_directory"], "output.cache_directory")
    web_output = _resolve(base, output["web_payload"], "output.web_payload")
    web_media_dir = _resolve(base, output["web_media_directory"], "output.web_media_directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    compiler = RsIntTruthCompiler(bundle_root)
    artifact = compiler.compile()
    provider = _backend(backend_config, cache_dir)
    prior_questions: list[str] = []
    generated_turns: list[dict[str, Any]] = []

    for turn in artifact["rounds"]:
        context = _episode_context(artifact, turn, prior_questions)
        blueprint = turn["question_blueprint"]
        question_request = build_question_editor_request(
            request_id=str(blueprint["request_id"]),
            episode_context=context,
            capability_contract=_capability_contract(turn),
            intent_zh=str(blueprint["intent_zh"]),
            protected_slots=blueprint["slot_values"],
            required_slots=blueprint["required_slots"],
            surface_constraints_zh=(
                f"保持{int(blueprint['required_subquestion_count'])}个意图部分，不得删减。",
                "必须是承接当前漫游进度的自然问句。",
                *tuple(str(value) for value in blueprint.get("surface_constraints_zh", [])),
            ),
        )
        question_result = provider.complete(**question_request.backend_kwargs())
        question_zh = _render_question(blueprint, question_result.response)

        skill_id = str(turn["answer_skill_ids"][0])
        answer_request = build_answer_narrator_request(
            episode_context=context,
            question_zh=question_zh,
            claim_sheet=turn["claim_sheet"],
            skill_id=skill_id,
        )
        answer_result = provider.complete(**answer_request.backend_kwargs())
        answer_validation = _validate_answer(turn, answer_result.response, skill_id)

        critic_request = build_critic_request(
            episode_context=context,
            question_zh=question_zh,
            claim_sheet=turn["claim_sheet"],
            skill_id=skill_id,
            candidate_answer=answer_result.response,
        )
        critic_result = provider.complete(**critic_request.backend_kwargs())
        critic_validation = _validate_critic(critic_result.response)

        compiled_turn = copy.deepcopy(dict(turn))
        compiled_turn["generated"] = {
            "question": {
                **question_result.response,
                "question_zh": question_zh,
            },
            "answer": dict(answer_result.response),
            "critic": dict(critic_result.response),
        }
        compiled_turn["generation_provenance"] = {
            "question": _provenance(
                question_result,
                role="question_editor",
                prompt_profile="answer_blind_episode_question",
                input_sha256=_sha256_json(question_request.payload),
            ),
            "answer": _provenance(
                answer_result,
                role="answer_narrator",
                prompt_profile="claim_grounded_cognitive_voiceover",
                skill_ids=(skill_id,),
                input_sha256=_sha256_json(answer_request.payload),
            ),
            "critic": _provenance(
                critic_result,
                role="critic",
                prompt_profile="independent_claim_audit",
                skill_ids=(skill_id,),
                input_sha256=_sha256_json(critic_request.payload),
            ),
        }
        compiled_turn["validation"] = {
            "question": {"passed": True, "checks": {"answer_blind": True, "slots_preserved": True}},
            "answer": answer_validation,
            "critic": critic_validation,
            "passed": True,
        }
        generated_turns.append(compiled_turn)
        prior_questions.append(question_zh)

    artifact["schema_version"] = "epispace.incremental_dialogue.v1"
    artifact["rounds"] = generated_turns
    artifact["generation_summary"] = {
        "question_editor_calls": 7,
        "answer_narrator_calls": 7,
        "critic_calls": 7,
        "all_rounds_accepted": True,
        "backend": {"provider": provider.provider, "model": provider.model},
        "cache_hits": sum(
            int(stage["cache_hit"])
            for turn in generated_turns
            for stage in turn["generation_provenance"].values()
        ),
    }
    artifact["validation"].update(
        {
            "question_language": True,
            "answer_claim_coverage": True,
            "independent_critic": True,
            "all_rounds_passed": True,
        }
    )
    artifact["artifact_sha256"] = _sha256_json(artifact)

    artifact_path = output_dir / "dialogue.compiled.json"
    episode_path = output_dir / "train.dialogue_episode_sft.jsonl"
    isolated_path = output_dir / "train.dialogue_isolated_sft.jsonl"
    aux_path = output_dir / "train.dialogue_state_op_aux_sft.jsonl"
    _write_json(artifact_path, artifact)
    _write_jsonl(episode_path, [_episode_sft(artifact)])
    _write_jsonl(isolated_path, _isolated_sft(artifact))
    _write_jsonl(aux_path, _state_aux_sft(artifact))
    web_payload = _web_payload(artifact, web_media_dir, web_output)
    _write_json(web_output, web_payload)

    manifest_files = [artifact_path, episode_path, isolated_path, aux_path, web_output]
    manifest = {
        "schema_version": "epispace.rsint_dialogue_manifest.v1",
        "artifact_sha256": artifact["artifact_sha256"],
        "files": {
            str(value.resolve()): {"sha256": _file_sha256(value), "bytes": value.stat().st_size}
            for value in manifest_files
        },
        "source_bundle": str(bundle_root),
        "source_bundle_hashes": artifact["source"]["bundle_files_sha256"],
        "development_only": True,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "artifact": str(artifact_path),
        "episode_sft": str(episode_path),
        "isolated_sft": str(isolated_path),
        "state_op_aux_sft": str(aux_path),
        "web_payload": str(web_output),
        "manifest": str(manifest_path),
        "artifact_sha256": artifact["artifact_sha256"],
        "rounds": len(generated_turns),
        "cache_hits": artifact["generation_summary"]["cache_hits"],
    }
