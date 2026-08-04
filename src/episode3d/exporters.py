"""Export EpiSpace IR into MLLM-native SFT, baseline, benchmark and RLVR data."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from episode3d.bundles import Bundle, stable_id
from episode3d.compilers import model_input_frame_is_admissible
from episode3d.models import CompiledBundle, QuestionSpec

SYSTEM_PROMPT = (
    "你正在观察同一真实场景中的一组有序视角。请只依据给出的画面维护空间信息并回答问题；"
    "证据不足时必须明确说无法确定。方向问题严格使用题目声明的参照系。"
)

AUX_SYSTEM_PROMPT = (
    "只有在用户明确要求时才输出可审计的场景状态。状态只能包含已观察证据，"
    "不得把未见物体或完整世界真值写入状态。"
)

HARD_REASONING_TYPES = {
    "last_seen_memory",
    "cross_view_relation",
    "cross_view_unknown",
    "evidence_presence_unknown",
    "evidence_presence_reveal",
    "counterfactual_verification",
    "object_centric_perspective",
    "rotation_change_detection",
    "elevation_relation_transfer",
    "occlusion_unknown",
    "occlusion_reveal",
    "target_view_prediction",
}

TASK_PRIORITY = {
    "occlusion_unknown": 0,
    "occlusion_reveal": 0,
    "rotation_change_detection": 1,
    "elevation_relation_transfer": 1,
    "orbit_identity": 1,
    "last_seen_memory": 2,
    "cross_view_unknown": 2,
    "evidence_presence_unknown": 0,
    "evidence_presence_reveal": 0,
    "object_centric_perspective": 3,
    "unknown_abstention": 4,
    "egocentric_relation": 5,
    "metric_distance": 6,
    "grounding_presence": 7,
    "cross_view_relation": 8,
    "counterfactual_verification": 9,
    "target_view_prediction": 10,
}

FORBIDDEN_INPUT_PATTERNS = (
    re.compile(r"expected_relation", re.IGNORECASE),
    re.compile(r"oracle", re.IGNORECASE),
    re.compile(r"certificate", re.IGNORECASE),
    re.compile(r"scene_ir", re.IGNORECASE),
    re.compile(r"relation_oracle", re.IGNORECASE),
    re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
        re.IGNORECASE,
    ),
)


def _content(bundle: Bundle, view_ids: tuple[str, ...], text: str) -> list[dict[str, str]]:
    content: list[dict[str, str]] = []
    for display_index, view_id in enumerate(view_ids, 1):
        view = bundle.view_by_id[view_id]
        content.extend(
            (
                {
                    "type": "text",
                    "text": (
                        f"<image-{display_index}; camera_height={view.camera_height_m:.2f}m; "
                        f"horizontal_fov={view.horizontal_fov_deg:.1f}deg>"
                    ),
                },
                {
                    "type": "image",
                    "image": str(view.rgb_path),
                },
            )
        )
    content.append({"type": "text", "text": text})
    return content


def _answer(spec: QuestionSpec) -> str:
    if spec.task_type in HARD_REASONING_TYPES:
        return spec.rationale_zh
    return spec.answer_zh


def _family_id(
    bundle: Bundle, variant: str, specs: list[QuestionSpec] | tuple[QuestionSpec, ...]
) -> str:
    groups = {spec.consistency_group for spec in specs if spec.consistency_group}
    if len(groups) > 1:
        raise ValueError(f"mixed consistency groups in one export group: {groups}")
    if groups:
        return stable_id("family", groups.pop())
    return stable_id("family", bundle.scene_id, bundle.episode_id, variant)


def _group_questions(
    compiled: CompiledBundle,
    *,
    excluded_program_ids: set[str],
    max_questions: int,
) -> list[tuple[tuple[str, ...], str, list[QuestionSpec]]]:
    groups: dict[tuple[tuple[str, ...], str], list[QuestionSpec]] = defaultdict(list)
    full_views = tuple(view.view_id for view in compiled.bundle.views)
    for spec in compiled.questions:
        if spec.program.program_id in excluded_program_ids:
            continue
        views = spec.model_view_ids or full_views
        groups[(views, spec.family_variant)].append(spec)
    result = []
    for (views, variant), specs in groups.items():
        specs.sort(key=lambda item: (TASK_PRIORITY.get(item.task_type, 100), item.fact_id))
        result.append((views, variant, specs[:max_questions]))
    result.sort(key=lambda item: (item[1] != "canonical", item[1], item[0]))
    return result


def export_training_arms(
    compiled: CompiledBundle,
    *,
    heldout_program_ids: set[str],
    max_questions: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[QuestionSpec]]:
    """Produce episode and paired isolated arms from exactly the same facts."""

    if compiled.split != "train":
        return [], [], []
    episode_records: list[dict[str, Any]] = []
    isolated_records: list[dict[str, Any]] = []
    selected: list[QuestionSpec] = []
    bundle = compiled.bundle
    for group_index, (view_ids, variant, specs) in enumerate(
        _group_questions(
            compiled,
            excluded_program_ids=heldout_program_ids,
            max_questions=max_questions,
        )
    ):
        if not specs:
            continue
        selected.extend(specs)
        comparison_id = stable_id(
            "comparison", bundle.scene_id, bundle.episode_id, variant, str(group_index)
        )
        questions = "\n".join(f"{index}. {spec.question_zh}" for index, spec in enumerate(specs, 1))
        answers = "\n".join(f"{index}. {_answer(spec)}" for index, spec in enumerate(specs, 1))
        record_id = stable_id("record", "episode", bundle.episode_id, variant, str(group_index))
        family_id = _family_id(bundle, variant, specs)
        episode_records.append(
            {
                "record_id": record_id,
                "format": "qwen_multimodal_chat",
                "sample_type": "observe_then_batch_qa",
                "arrangement": "B_observe_then_ask",
                "family_id": family_id,
                "scene_id": bundle.scene_id,
                "split": "train",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _content(
                            bundle,
                            view_ids,
                            "观察阶段到此结束。请依次回答下列问题：\n" + questions,
                        ),
                    },
                    {"role": "assistant", "content": answers},
                ],
                "loss_policy": {"train_on": "assistant_only"},
                "comparison_contract": {
                    "comparison_id": comparison_id,
                    "arm": "episode",
                    "fact_ids": [spec.fact_id for spec in specs],
                    "unique_image_count": len(view_ids),
                    "question_count": len(specs),
                    "supervision_matched": True,
                    "compute_matching_requires_sampler": True,
                    "surface_format": "numbered_answer_list.v1",
                },
                "hidden_meta": {
                    "trajectory_class": bundle.trajectory_class,
                    "program_ids": [spec.program.program_id for spec in specs],
                    "semantic_signatures": [spec.program.semantic_signature for spec in specs],
                    "source_bundle": str(bundle.root),
                },
            }
        )
        for spec in specs:
            isolated_id = stable_id("record", "isolated", bundle.episode_id, spec.fact_id)
            isolated_records.append(
                {
                    "record_id": isolated_id,
                    "format": "qwen_multimodal_chat",
                    "sample_type": "isolated_qa_control",
                    "arrangement": "C_isolated_format_matched",
                    "family_id": family_id,
                    "scene_id": bundle.scene_id,
                    "split": "train",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": _content(
                                bundle,
                                view_ids,
                                "观察阶段到此结束。请依次回答下列问题：\n"
                                f"1. {spec.question_zh}",
                            ),
                        },
                        {"role": "assistant", "content": f"1. {_answer(spec)}"},
                    ],
                    "loss_policy": {"train_on": "assistant_only"},
                    "comparison_contract": {
                        "comparison_id": comparison_id,
                        "arm": "isolated",
                        "fact_ids": [spec.fact_id],
                        "unique_image_count": len(view_ids),
                        "question_count": 1,
                        "supervision_matched": True,
                        "compute_matching_requires_sampler": True,
                        "surface_format": "numbered_answer_list.v1",
                    },
                    "hidden_meta": {
                        "trajectory_class": bundle.trajectory_class,
                        "program_id": spec.program.program_id,
                        "semantic_signature": spec.program.semantic_signature,
                        "source_bundle": str(bundle.root),
                    },
                }
            )
    for record in (*episode_records, *isolated_records):
        assert_no_model_input_leakage(record["messages"])
    return episode_records, isolated_records, selected


def export_state_aux(
    compiled: CompiledBundle,
    *,
    observable_belief: dict[str, Any],
) -> dict[str, Any] | None:
    if compiled.split != "train" or compiled.bundle.trajectory_class == "T10":
        return None
    bundle = compiled.bundle
    # The state target is tied to the complete ordered observation history.  Do
    # not silently delete/reindex a bad frame: that would make first/last-seen
    # evidence IDs and the canonical view-000 anchor disagree with the images.
    # Omit this optional record instead; question-specific exports remain valid
    # because their exact model_view_ids are independently gated and replayed.
    if any(
        not model_input_frame_is_admissible(bundle, view.view_id)
        for view in bundle.views
    ):
        return None
    view_ids = tuple(view.view_id for view in bundle.views)
    state = observable_belief
    record = {
        "record_id": stable_id("record", "state", bundle.episode_id),
        "format": "qwen_multimodal_chat",
        "sample_type": "observable_state_aux",
        "arrangement": "D_state_commit",
        "family_id": stable_id("family", bundle.scene_id, bundle.episode_id, "state"),
        "scene_id": bundle.scene_id,
        "split": "train",
        "messages": [
            {"role": "system", "content": AUX_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _content(
                    bundle,
                    view_ids,
                    "请在以第1个视角相机为锚的坐标系中，提交与具体问题无关的"
                    "已观察场景状态。使用给定 JSON schema；不要写入未观察物体。",
                ),
            },
            {
                "role": "assistant",
                "content": "<STATE>" + json.dumps(state, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "loss_policy": {"train_on": "assistant_only"},
        "hidden_meta": {
            "trajectory_class": bundle.trajectory_class,
            "source_bundle": str(bundle.root),
            "format_conditioned": True,
        },
    }
    assert_no_model_input_leakage(record["messages"][:2])
    return record


def export_benchmark(compiled: CompiledBundle) -> list[dict[str, Any]]:
    if compiled.split == "train":
        return []
    records: list[dict[str, Any]] = []
    bundle = compiled.bundle
    full_views = tuple(view.view_id for view in bundle.views)
    for spec in compiled.questions:
        if "training_only" in spec.tags:
            continue
        view_ids = spec.model_view_ids or full_views
        model_input = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _content(
                    bundle,
                    view_ids,
                    "观察阶段到此结束。请依次回答下列问题：\n"
                    f"1. {spec.question_zh}",
                ),
            },
        ]
        assert_no_model_input_leakage(model_input)
        records.append(
            {
                "record_id": stable_id("benchmark", bundle.episode_id, spec.fact_id),
                "schema_version": "epispace.benchmark.v1",
                "scene_id": bundle.scene_id,
                "family_id": _family_id(bundle, spec.family_variant, (spec,)),
                "split": compiled.split,
                "trajectory_class": bundle.trajectory_class,
                "family_variant": spec.family_variant,
                "consistency_group": spec.consistency_group,
                "model_input": model_input,
                "target": {
                    "answer_zh": spec.answer_zh,
                    "answer_value": spec.answer_value,
                    "answer_status": spec.answer_status,
                    "task_type": spec.task_type,
                    "fact_id": spec.fact_id,
                },
                "program": spec.program.as_dict(),
                "certificate": spec.certificate,
                "evidence_view_ids": list(spec.evidence_view_ids),
                "presentation_contract": "numbered_answer_list.v1",
                "oracle_held_out_view_ids": list(spec.oracle_held_out_view_ids),
                "source_bundle": str(bundle.root),
            }
        )
    return records


def export_rlvr(
    compiled: CompiledBundle, selected_specs: list[QuestionSpec]
) -> list[dict[str, Any]]:
    if compiled.split != "train":
        return []
    bundle = compiled.bundle
    full_views = tuple(view.view_id for view in bundle.views)
    records: list[dict[str, Any]] = []
    for spec in selected_specs:
        view_ids = spec.model_view_ids or full_views
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _content(bundle, view_ids, spec.question_zh)},
        ]
        assert_no_model_input_leakage(prompt)
        records.append(
            {
                "record_id": stable_id("rlvr", bundle.episode_id, spec.fact_id),
                "schema_version": "epispace.rlvr.v1",
                "scene_id": bundle.scene_id,
                "family_id": _family_id(bundle, spec.family_variant, (spec,)),
                "prompt": prompt,
                "reward_spec": _reward_spec(spec),
                "consistency_group": spec.consistency_group,
                "program_id": spec.program.program_id,
                "source_bundle": str(bundle.root),
            }
        )
    return records


def _reward_spec(spec: QuestionSpec) -> dict[str, Any]:
    if spec.answer_status == "unknown":
        reward_type = "abstain"
    elif isinstance(spec.answer_value, bool):
        reward_type = "boolean"
    elif isinstance(spec.answer_value, int | float):
        reward_type = "numeric"
    else:
        reward_type = "enum_or_text"
    reward: dict[str, Any] = {
        "type": reward_type,
        "expected": spec.answer_value,
        "certificate": spec.certificate,
    }
    if reward_type == "numeric":
        tolerance = spec.certificate.get("answer_tolerance_m")
        if tolerance is None:
            tolerance = 0.2
        reward["tolerance"] = tolerance
    return reward


def assert_no_model_input_leakage(messages: list[dict[str, Any]]) -> None:
    """Fail closed if hidden geometry identifiers appear in model input."""

    serialized = json.dumps(messages, ensure_ascii=False)
    for pattern in FORBIDDEN_INPUT_PATTERNS:
        match = pattern.search(serialized)
        if match:
            raise ValueError(f"model input leakage matched {pattern.pattern}: {match.group(0)}")
