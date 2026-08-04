"""End-to-end, claim-grounded language generation for episode QA.

The simulator remains the source of truth.  A structured language backend is
used only to naturalize answer-blind question blueprints and deterministic
claim sheets.  Every generated item is rejected unless local validators and an
independent model critic accept it.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from episode3d.exporters import SYSTEM_PROMPT, assert_no_model_input_leakage
from episode3d.qa_generation.backends import (
    CodexExecBackend,
    ReplayBackend,
    StructuredLLMBackend,
    complete_request,
    validate_json_schema,
)
from episode3d.qa_generation.blueprints import build_question_blueprint, strategy_library
from episode3d.qa_generation.claims import compile_claim_sheet
from episode3d.qa_generation.episode_planner import (
    EpisodeBatchPlanner,
    PlannerContract,
    question_by_fact,
    read_episode_ir,
)
from episode3d.qa_generation.prompts import (
    answer_output_schema,
    build_answer_batch_request,
    build_critic_batch_request,
    build_question_batch_request,
    question_output_schema,
)
from episode3d.qa_generation.schemas import (
    ClaimSheet,
    EpisodeBatchPlan,
    QuestionBlueprint,
    canonical_json,
    content_id,
)
from episode3d.qa_generation.validators import (
    QAValidationError,
    require_valid,
    validate_answer,
    validate_critic,
    validate_question,
)


class QAGenerationPipelineError(RuntimeError):
    """The pilot cannot be emitted without violating a fail-closed contract."""


@dataclass(frozen=True)
class PipelinePaths:
    config_path: Path
    episode_ir: Path
    semantic_audit: Path
    source_pipeline_config: Path
    output_dir: Path
    cache_dir: Path
    web_output: Path


def _resolve(base: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise QAGenerationPipelineError(f"configuration field {label} must be a path")
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _load_config(config_path: Path) -> tuple[dict[str, Any], PipelinePaths]:
    path = config_path.resolve()
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QAGenerationPipelineError(f"cannot read QA generation config {path}: {error}") from error
    if config.get("schema_version") != "epispace.qa_generation_config.v1":
        raise QAGenerationPipelineError("unsupported QA generation configuration schema")
    source = config.get("source")
    output = config.get("output")
    if not isinstance(source, Mapping) or not isinstance(output, Mapping):
        raise QAGenerationPipelineError("configuration requires source and output objects")
    base = path.parent
    paths = PipelinePaths(
        config_path=path,
        episode_ir=_resolve(base, source.get("episode_ir"), "source.episode_ir"),
        semantic_audit=_resolve(
            base, source.get("semantic_visual_audit"), "source.semantic_visual_audit"
        ),
        source_pipeline_config=_resolve(
            base, source.get("pipeline_config"), "source.pipeline_config"
        ),
        output_dir=_resolve(base, output.get("directory"), "output.directory"),
        cache_dir=_resolve(base, output.get("cache_directory"), "output.cache_directory"),
        web_output=_resolve(base, output.get("web_payload"), "output.web_payload"),
    )
    for required in (paths.episode_ir, paths.semantic_audit, paths.source_pipeline_config):
        if not required.is_file():
            raise QAGenerationPipelineError(f"required source artifact does not exist: {required}")
    return config, paths


def _sha256(path: Path) -> str:
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
    path.write_text(
        "".join(canonical_json(dict(row)) + "\n" for row in rows), encoding="utf-8"
    )


def _backend_from_config(
    config: Mapping[str, Any], paths: PipelinePaths
) -> StructuredLLMBackend:
    backend = config.get("backend", {})
    if not isinstance(backend, Mapping):
        raise QAGenerationPipelineError("backend configuration must be an object")
    kind = str(backend.get("kind", "codex_exec"))
    provider = str(backend.get("provider", "codex-cli"))
    model = str(backend.get("model", "default"))
    if kind == "replay":
        return ReplayBackend(paths.cache_dir, provider=provider, model=model)
    if kind == "codex_exec":
        return CodexExecBackend(
            paths.cache_dir,
            model=None if model == "default" else model,
            codex_binary=str(backend.get("codex_binary", "codex")),
            timeout_seconds=float(backend.get("timeout_seconds", 300)),
        )
    raise QAGenerationPipelineError(f"unsupported backend kind {kind!r}")


def _response_index(
    result: Mapping[str, Any], expected_ids: Sequence[str], *, stage: str
) -> dict[str, dict[str, Any]]:
    raw = result.get("responses")
    if not isinstance(raw, list):
        raise QAGenerationPipelineError(f"{stage} batch response has no responses array")
    indexed: dict[str, dict[str, Any]] = {}
    for value in raw:
        if not isinstance(value, Mapping):
            raise QAGenerationPipelineError(f"{stage} batch contains a non-object response")
        request_id = str(value.get("request_id", ""))
        if not request_id or request_id in indexed:
            raise QAGenerationPipelineError(
                f"{stage} batch contains an empty or duplicate request_id {request_id!r}"
            )
        indexed[request_id] = dict(value)
    expected = set(expected_ids)
    if set(indexed) != expected:
        raise QAGenerationPipelineError(
            f"{stage} batch request ids differ: missing={sorted(expected-set(indexed))}, "
            f"extra={sorted(set(indexed)-expected)}"
        )
    return indexed


def _render_question(blueprint: QuestionBlueprint, response: Mapping[str, Any]) -> str:
    rendered = str(response["template_zh"])
    for name, value in blueprint.slot_values.items():
        rendered = rendered.replace(f"{{{{{name}}}}}", value)
    if "{{" in rendered or "}}" in rendered:
        raise QAValidationError("rendered question contains an unresolved protected slot")
    return rendered


def _validate_blueprint_claim_alignment(
    blueprint: QuestionBlueprint, claim_sheet: ClaimSheet
) -> None:
    """Cross-check question premises against independently compiled claims."""

    if blueprint.task_type != "counterfactual_verification":
        return
    tested_surface = blueprint.slot_values.get("tested_relation")
    verdict_claims = [
        claim for claim in claim_sheet.claims if claim.kind == "counterfactual_verdict"
    ]
    if len(verdict_claims) != 1 or tested_surface not in verdict_claims[0].allowed_directions:
        raise QAGenerationPipelineError(
            f"counterfactual premise/claim mismatch for {blueprint.fact_id}: "
            f"tested={tested_surface!r}, allowed={verdict_claims[0].allowed_directions if verdict_claims else ()}"
        )


def _capabilities(records: Sequence[Mapping[str, Any]]) -> list[str]:
    capabilities = {
        str(value)
        for record in records
        for value in (
            record["capability"]["primary"],
            *record["capability"].get("supporting", []),
        )
        if value != "AUX"
    }
    return sorted(capabilities)


def _optimizer_decision(
    records: Sequence[Mapping[str, Any]],
    contract: PlannerContract,
    heldout_program_ids: set[str],
) -> dict[str, Any]:
    candidates = [record for record in records if record["disposition"] == "train_candidate"]
    capabilities = _capabilities(candidates)
    aux_count = sum(record["capability"]["primary"] == "AUX" for record in candidates)
    checks = {
        "all_generation_gates_pass": all(record["validation"]["passed"] for record in candidates),
        "question_count_4_to_6": contract.min_questions
        <= len(candidates)
        <= contract.max_questions,
        "at_least_three_sensenova_capabilities": len(capabilities)
        >= contract.min_capabilities,
        "pt_backbone_present": (not contract.require_pt_backbone) or "PT" in capabilities,
        "auxiliary_limit": aux_count <= contract.max_auxiliary_questions,
        "source_split_train": all(record["split"] == "train" for record in candidates),
        "heldout_disposition_partition_valid": all(
            (record["program"]["program_id"] in heldout_program_ids)
            == (record["disposition"] == "composition_heldout")
            for record in records
        ),
    }
    return {
        "optimizer_eligible": bool(candidates) and all(checks.values()),
        "release_eligible": False,
        "train_question_count": len(candidates),
        "capabilities": capabilities,
        "checks": checks,
        "note": (
            "仅表示通过当前 QA pilot 的优化器准入；源语义视觉审计整体未通过，"
            "因此不是正式 release 资格。"
        ),
    }


def _observation_payload(
    episode: Mapping[str, Any], exposure_view_ids: Sequence[str], *, web_root: Path | None = None
) -> list[dict[str, Any]]:
    by_view = {str(value["view_id"]): value for value in episode["observations"]}
    result: list[dict[str, Any]] = []
    for display_index, view_id in enumerate(exposure_view_ids, 1):
        observation = by_view[view_id]
        rgb = Path(str(observation["rgb"])).resolve()
        rgb_value = os.path.relpath(rgb, web_root) if web_root is not None else str(rgb)
        result.append(
            {
                "display_index": display_index,
                "view_id": view_id,
                "role": observation.get("role"),
                "rgb": rgb_value,
                "camera_height_m": observation.get("camera_height_m"),
                "horizontal_fov_deg": observation.get("horizontal_fov_deg"),
            }
        )
    return result


def _user_content(observations: Sequence[Mapping[str, Any]], questions: Sequence[str]) -> list[dict[str, str]]:
    content: list[dict[str, str]] = []
    for observation in observations:
        content.extend(
            (
                {
                    "type": "text",
                    "text": (
                        f"<image-{observation['display_index']}; "
                        f"camera_height={float(observation['camera_height_m']):.2f}m; "
                        f"horizontal_fov={float(observation['horizontal_fov_deg']):.1f}deg>"
                    ),
                },
                {"type": "image", "image": str(observation["rgb"])},
            )
        )
    numbered = "\n".join(f"{index}. {question}" for index, question in enumerate(questions, 1))
    content.append(
        {
            "type": "text",
            "text": "观察阶段到此结束。请依次回答下列问题：\n" + numbered,
        }
    )
    return content


def _training_records(
    episode: Mapping[str, Any],
    plan: EpisodeBatchPlan,
    records: Sequence[Mapping[str, Any]],
    optimizer: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not optimizer["optimizer_eligible"]:
        return [], []
    selected = [record for record in records if record["disposition"] == "train_candidate"]
    observations = _observation_payload(episode, plan.exposure_view_ids)
    questions = [str(record["generated"]["question_zh"]) for record in selected]
    answers = [str(record["generated"]["answer"]["surface_answer_zh"]) for record in selected]
    comparison_id = content_id(
        "qa-comparison", {"batch_id": plan.batch_id, "facts": [r["fact_id"] for r in selected]}
    )
    family_id = content_id("qa-family", {"episode_id": plan.episode_id, "batch": plan.batch_id})
    common_hidden = {
        "qa_generation_schema": "epispace.claim_grounded_qa.v1",
        "trajectory_class": plan.trajectory_class,
        "source_bundle": plan.source_bundle,
        "program_ids": [record["program"]["program_id"] for record in selected],
        "semantic_signatures": [record["program"]["semantic_signature"] for record in selected],
        "claim_sheet_ids": [record["claim_sheet"]["claim_sheet_id"] for record in selected],
        "generation_request_hashes": [
            record["generation_provenance"]["answer"]["request_hash"] for record in selected
        ],
        "development_only": True,
        "source_release_eligible": False,
    }
    episode_record = {
        "record_id": content_id("record", {"arm": "episode", "comparison": comparison_id}),
        "format": "qwen_multimodal_chat",
        "sample_type": "claim_grounded_observe_then_batch_qa",
        "arrangement": "B_observe_then_ask",
        "family_id": family_id,
        "scene_id": plan.scene_id,
        "split": "train",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_content(observations, questions)},
            {
                "role": "assistant",
                "content": "\n".join(
                    f"{index}. {answer}" for index, answer in enumerate(answers, 1)
                ),
            },
        ],
        "loss_policy": {"train_on": "assistant_only"},
        "comparison_contract": {
            "comparison_id": comparison_id,
            "arm": "episode",
            "fact_ids": [record["fact_id"] for record in selected],
            "unique_image_count": len(observations),
            "question_count": len(selected),
            "supervision_matched": True,
            "compute_matching_requires_sampler": True,
            "surface_format": "numbered_answer_list.v1",
        },
        "hidden_meta": common_hidden,
    }
    isolated: list[dict[str, Any]] = []
    for record in selected:
        isolated.append(
            {
                "record_id": content_id(
                    "record",
                    {"arm": "isolated", "comparison": comparison_id, "fact": record["fact_id"]},
                ),
                "format": "qwen_multimodal_chat",
                "sample_type": "claim_grounded_isolated_qa_control",
                "arrangement": "C_isolated_format_matched",
                "family_id": family_id,
                "scene_id": plan.scene_id,
                "split": "train",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _user_content(
                            observations, [str(record["generated"]["question_zh"])]
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": f"1. {record['generated']['answer']['surface_answer_zh']}",
                    },
                ],
                "loss_policy": {"train_on": "assistant_only"},
                "comparison_contract": {
                    "comparison_id": comparison_id,
                    "arm": "isolated",
                    "fact_ids": [record["fact_id"]],
                    "unique_image_count": len(observations),
                    "question_count": 1,
                    "supervision_matched": True,
                    "compute_matching_requires_sampler": True,
                    "surface_format": "numbered_answer_list.v1",
                },
                "hidden_meta": {
                    **common_hidden,
                    "program_ids": [record["program"]["program_id"]],
                    "semantic_signatures": [record["program"]["semantic_signature"]],
                    "claim_sheet_ids": [record["claim_sheet"]["claim_sheet_id"]],
                    "generation_request_hashes": [
                        record["generation_provenance"]["answer"]["request_hash"]
                    ],
                },
            }
        )
    for value in (episode_record, *isolated):
        assert_no_model_input_leakage(value["messages"][:2])
    return [episode_record], isolated


def _generate_episode(
    *,
    episode: Mapping[str, Any],
    plan: EpisodeBatchPlan,
    backend: StructuredLLMBackend,
    heldout_program_ids: set[str],
    episode_ir_sha256: str,
    question_corpus: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    questions = question_by_fact(episode)
    blueprints: list[QuestionBlueprint] = []
    sheets: list[ClaimSheet] = []
    sources: list[dict[str, Any]] = []
    for planned in plan.questions:
        source = questions[planned.fact_id]
        blueprint = build_question_blueprint(
            episode, source, exposure_view_ids=planned.exposure_view_ids
        )
        sheet = compile_claim_sheet(
            episode,
            source,
            exposure_view_ids=planned.exposure_view_ids,
            source_episode_ir_sha256=episode_ir_sha256,
        )
        _validate_blueprint_claim_alignment(blueprint, sheet)
        blueprints.append(blueprint)
        sheets.append(sheet)
        sources.append(source)

    question_call = complete_request(backend, build_question_batch_request(blueprints))
    question_by_id = _response_index(
        question_call.response,
        [blueprint.request_id for blueprint in blueprints],
        stage="question",
    )
    natural_questions: list[str] = []
    question_responses: list[dict[str, Any]] = []
    question_reports: list[dict[str, Any]] = []
    for blueprint in blueprints:
        response = question_by_id[blueprint.request_id]
        validate_json_schema(response, question_output_schema(blueprint))
        report = validate_question(
            blueprint,
            response,
            corpus=question_corpus,
            dedup_variable_terms=blueprint.slot_values.values(),
        )
        require_valid(report)
        rendered = _render_question(blueprint, response)
        natural_questions.append(rendered)
        question_responses.append(response)
        question_reports.append(report.as_dict())
        question_corpus.append(rendered)

    answer_call = complete_request(
        backend,
        build_answer_batch_request(
            [
                (sheet, question, response["strategy_id"])
                for sheet, question, response in zip(
                    sheets, natural_questions, question_responses, strict=True
                )
            ]
        ),
    )
    answer_by_id = _response_index(
        answer_call.response,
        [sheet.claim_sheet_id for sheet in sheets],
        stage="answer",
    )
    answer_responses: list[dict[str, Any]] = []
    answer_reports: list[dict[str, Any]] = []
    for sheet, question_response in zip(sheets, question_responses, strict=True):
        response = answer_by_id[sheet.claim_sheet_id]
        validate_json_schema(response, answer_output_schema(sheet))
        variable_terms = [
            str(entity)
            for claim in sheet.claims
            for entity in (claim.statement_zh, *claim.allowed_directions)
        ]
        report = validate_answer(
            sheet,
            response,
            expected_strategy_id=str(question_response["strategy_id"]),
            dedup_variable_terms=variable_terms,
        )
        require_valid(report)
        answer_responses.append(response)
        answer_reports.append(report.as_dict())

    critic_call = complete_request(
        backend,
        build_critic_batch_request(list(zip(sheets, answer_responses, strict=True))),
    )
    critic_by_id = _response_index(
        critic_call.response,
        [sheet.claim_sheet_id for sheet in sheets],
        stage="critic",
    )

    records: list[dict[str, Any]] = []
    for index, (planned, source, blueprint, sheet) in enumerate(
        zip(plan.questions, sources, blueprints, sheets, strict=True)
    ):
        critic_response = critic_by_id[sheet.claim_sheet_id]
        critic_report = validate_critic(critic_response)
        require_valid(critic_report)
        disposition = (
            "composition_heldout"
            if planned.program_id in heldout_program_ids
            else "train_candidate"
        )
        validation = {
            "question": question_reports[index],
            "answer": answer_reports[index],
            "critic": critic_report.as_dict(),
        }
        validation["passed"] = all(
            validation[name]["passed"] for name in ("question", "answer", "critic")
        )
        records.append(
            {
                "schema_version": "epispace.claim_grounded_qa.v1",
                "qa_id": content_id(
                    "qa", {"fact_id": planned.fact_id, "batch_id": plan.batch_id}
                ),
                "batch_id": plan.batch_id,
                "episode_id": plan.episode_id,
                "scene_id": plan.scene_id,
                "split": plan.split,
                "trajectory_class": plan.trajectory_class,
                "fact_id": planned.fact_id,
                "task_type": planned.task_type,
                "program": dict(source["program"]),
                "capability": planned.capability.as_dict(),
                "source_model_view_ids": list(planned.source_model_view_ids),
                "exposure_view_ids": list(planned.exposure_view_ids),
                "expanded_to_episode_context": planned.expanded_to_episode_context,
                "original": {
                    "question_zh": source.get("question_zh"),
                    "answer_zh": source.get("answer_zh"),
                    "rationale_zh": source.get("rationale_zh"),
                },
                "question_blueprint": blueprint.answer_blind_dict(),
                "generated": {
                    "question_zh": natural_questions[index],
                    "question_template": question_responses[index],
                    "answer": answer_responses[index],
                    "strategy_description_zh": strategy_library()[
                        str(question_responses[index]["strategy_id"])
                    ],
                },
                "claim_sheet": sheet.as_dict(include_answer=True),
                "disposition": disposition,
                "validation": validation,
                "generation_provenance": {
                    "question": {
                        "provider": question_call.provider,
                        "model": question_call.model,
                        "request_hash": question_call.request_hash,
                        "cache_hit": question_call.cache_hit,
                    },
                    "answer": {
                        "provider": answer_call.provider,
                        "model": answer_call.model,
                        "request_hash": answer_call.request_hash,
                        "cache_hit": answer_call.cache_hit,
                    },
                    "critic": {
                        "provider": critic_call.provider,
                        "model": critic_call.model,
                        "request_hash": critic_call.request_hash,
                        "cache_hit": critic_call.cache_hit,
                    },
                },
            }
        )
    generation = {
        "question": question_call.as_dict() | {"response": None},
        "answer": answer_call.as_dict() | {"response": None},
        "critic": critic_call.as_dict() | {"response": None},
    }
    return records, generation


def _web_payload(
    *,
    config: Mapping[str, Any],
    paths: PipelinePaths,
    episodes: Mapping[str, Mapping[str, Any]],
    plans: Sequence[EpisodeBatchPlan],
    records_by_batch: Mapping[str, Sequence[Mapping[str, Any]]],
    decisions: Mapping[str, Mapping[str, Any]],
    semantic_audit: Mapping[str, Any],
    manifest_summary: Mapping[str, Any],
) -> dict[str, Any]:
    web_root = paths.web_output.parent.parent
    episode_payloads: list[dict[str, Any]] = []
    for plan in plans:
        episode = episodes[plan.episode_id]
        episode_payloads.append(
            {
                "batch_plan": plan.as_dict(),
                "optimizer_decision": decisions[plan.batch_id],
                "observations": _observation_payload(
                    episode, plan.exposure_view_ids, web_root=web_root
                ),
                "qa_records": list(records_by_batch[plan.batch_id]),
            }
        )
    return {
        "schema_version": "epispace.qa_generation_web.v1",
        "dataset_id": config["dataset_id"],
        "title": "从几何真值到自然语言 Episode QA",
        "status": "development_only",
        "release_eligible": False,
        "summary": dict(manifest_summary),
        "source_semantic_visual_audit": {
            "status": semantic_audit.get("status"),
            "gate": semantic_audit.get("decision", {}).get("semantic_visual_audit_gate"),
            "counts": semantic_audit.get("summary", {}).get("overall_status_counts", {}),
            "reviewed_items": semantic_audit.get("summary", {}).get("reviewed_items"),
            "interpretation_zh": (
                "已排除抽审中的 major/unreviewable fact 及其污染 family；但源 release 的"
                "整体语义视觉审计仍为 fail，因此本页只证明管线可运行，不代表正式可发布。"
            ),
        },
        "contracts": {
            "observe_once_read_many": "一次按序看完整轨迹，随后回答 4–6 个共享状态问题。",
            "answer_blind_question_writer": True,
            "claim_grounded_answer_writer": True,
            "model_critic_required": True,
            "heldout_programs": list(config["source"]["heldout_program_ids"]),
            "unsupported_capability": {
                "MR": "当前资产没有可靠 canonical object front，不能伪造物体旋转/正面监督。"
            },
        },
        "episodes": episode_payloads,
    }


def build_qa_generation_pilot(
    config_path: str | Path,
    *,
    backend: StructuredLLMBackend | None = None,
    output_dir: str | Path | None = None,
    web_output: str | Path | None = None,
) -> dict[str, Any]:
    """Generate, validate and export the curated claim-grounded QA pilot."""

    config, original_paths = _load_config(Path(config_path))
    paths = PipelinePaths(
        **{
            **original_paths.__dict__,
            "output_dir": Path(output_dir).resolve() if output_dir is not None else original_paths.output_dir,
            "web_output": Path(web_output).resolve() if web_output is not None else original_paths.web_output,
        }
    )
    source_pipeline_config = json.loads(paths.source_pipeline_config.read_text(encoding="utf-8"))
    heldout_program_ids = set(source_pipeline_config["composition_holdout"]["program_ids"])
    configured_heldout = set(config["source"]["heldout_program_ids"])
    if heldout_program_ids != configured_heldout:
        raise QAGenerationPipelineError(
            "QA config heldout programs do not match the immutable source pipeline config"
        )
    episodes = read_episode_ir(paths.episode_ir)
    episode_index = {str(value["episode_id"]): value for value in episodes}
    curated = config.get("curated_episode_facts")
    if not isinstance(curated, Mapping):
        raise QAGenerationPipelineError("curated_episode_facts must be an object")
    curated_map = {
        str(episode_id): tuple(str(fact_id) for fact_id in fact_ids)
        for episode_id, fact_ids in curated.items()
    }
    contract = PlannerContract()
    planner = EpisodeBatchPlanner(
        heldout_program_ids=heldout_program_ids,
        contract=contract,
    )
    plans = planner.plan(
        episodes,
        target_question_count=int(config.get("target_question_count", 20)),
        semantic_audit_path=paths.semantic_audit,
        split="train",
        include_eval_only=True,
        curated=curated_map,
    )
    selected_count = sum(len(plan.questions) for plan in plans)
    if selected_count != int(config.get("target_question_count", 20)):
        raise QAGenerationPipelineError(
            f"curated pilot contains {selected_count} facts, expected target exactly"
        )
    language_backend = backend or _backend_from_config(config, paths)
    episode_ir_sha256 = _sha256(paths.episode_ir)
    question_corpus: list[str] = []
    all_records: list[dict[str, Any]] = []
    records_by_batch: dict[str, list[dict[str, Any]]] = {}
    generation_batches: dict[str, Any] = {}
    decisions: dict[str, dict[str, Any]] = {}
    episode_sft: list[dict[str, Any]] = []
    isolated_sft: list[dict[str, Any]] = []

    for plan in plans:
        episode = episode_index[plan.episode_id]
        records, generation = _generate_episode(
            episode=episode,
            plan=plan,
            backend=language_backend,
            heldout_program_ids=heldout_program_ids,
            episode_ir_sha256=episode_ir_sha256,
            question_corpus=question_corpus,
        )
        decision = _optimizer_decision(records, contract, heldout_program_ids)
        if not decision["optimizer_eligible"]:
            for record in records:
                if record["disposition"] == "train_candidate":
                    record["disposition"] = "development_only_batch_shortfall"
        records_by_batch[plan.batch_id] = records
        all_records.extend(records)
        generation_batches[plan.batch_id] = generation
        decisions[plan.batch_id] = decision
        episode_rows, isolated_rows = _training_records(
            episode, plan, records, decision
        )
        episode_sft.extend(episode_rows)
        isolated_sft.extend(isolated_rows)

    if not all(record["validation"]["passed"] for record in all_records):
        raise QAGenerationPipelineError("one or more QA records failed a validation gate")
    semantic_audit = json.loads(paths.semantic_audit.read_text(encoding="utf-8"))
    summary = {
        "episode_count": len(plans),
        "qa_count": len(all_records),
        "train_candidate_count": sum(
            record["disposition"] == "train_candidate" for record in all_records
        ),
        "heldout_count": sum(
            record["disposition"] == "composition_heldout" for record in all_records
        ),
        "batch_shortfall_count": sum(
            record["disposition"] == "development_only_batch_shortfall"
            for record in all_records
        ),
        "optimizer_episode_records": len(episode_sft),
        "optimizer_isolated_records": len(isolated_sft),
        "optimizer_supervised_facts": sum(
            row["comparison_contract"]["question_count"] for row in episode_sft
        ),
        "capability_counts": {
            capability: sum(
                capability
                in (record["capability"]["primary"], *record["capability"]["supporting"])
                for record in all_records
            )
            for capability in ("MM", "SR", "MR", "PT", "CR")
        },
    }
    manifest = {
        "schema_version": "epispace.qa_generation_manifest.v1",
        "dataset_id": config["dataset_id"],
        "status": "development_only",
        "release_eligible": False,
        "summary": summary,
        "backend": {
            "provider": language_backend.provider,
            "model": language_backend.model,
            "cache_directory": str(paths.cache_dir),
        },
        "source_integrity": {
            "episode_ir": {"path": str(paths.episode_ir), "sha256": episode_ir_sha256},
            "semantic_visual_audit": {
                "path": str(paths.semantic_audit),
                "sha256": _sha256(paths.semantic_audit),
                "status": semantic_audit.get("status"),
                "gate": semantic_audit.get("decision", {}).get("semantic_visual_audit_gate"),
            },
            "pipeline_config": {
                "path": str(paths.source_pipeline_config),
                "sha256": _sha256(paths.source_pipeline_config),
            },
        },
        "episode_plans": [plan.as_dict() for plan in plans],
        "optimizer_decisions": decisions,
        "generation_batches": generation_batches,
        "artifacts": {
            "qa_records": "qa_records.jsonl",
            "claim_sheets": "claim_sheets.jsonl",
            "episode_sft": "train.episode_sft.jsonl",
            "isolated_sft": "train.isolated_sft.jsonl",
            "web_payload": str(paths.web_output),
        },
    }
    _write_jsonl(paths.output_dir / "qa_records.jsonl", all_records)
    _write_jsonl(
        paths.output_dir / "claim_sheets.jsonl",
        [record["claim_sheet"] for record in all_records],
    )
    _write_jsonl(paths.output_dir / "train.episode_sft.jsonl", episode_sft)
    _write_jsonl(paths.output_dir / "train.isolated_sft.jsonl", isolated_sft)
    _write_json(paths.output_dir / "episode_plans.json", [plan.as_dict() for plan in plans])
    _write_json(paths.output_dir / "manifest.json", manifest)
    web_payload = _web_payload(
        config=config,
        paths=paths,
        episodes=episode_index,
        plans=plans,
        records_by_batch=records_by_batch,
        decisions=decisions,
        semantic_audit=semantic_audit,
        manifest_summary=summary,
    )
    _write_json(paths.web_output, web_payload)
    return manifest


__all__ = ["QAGenerationPipelineError", "build_qa_generation_pilot"]
