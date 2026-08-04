"""Build answer-blind question-writing requests from executable Episode IR.

This module is the firewall between geometry and the question-writing model.
It is allowed to expose entities, reference frames, view indices, and a claim
being *tested* because those are part of the question.  It never copies an
answer, certificate, oracle value, or rationale into :class:`QuestionBlueprint`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from episode3d.bundles import Bundle
from episode3d.language import entity_name, view_number
from episode3d.qa_generation.capabilities import capability_for
from episode3d.qa_generation.schemas import QuestionBlueprint, content_id


class BlueprintError(ValueError):
    """A fact cannot be expressed without crossing the answer-blind boundary."""


STRATEGIES = {
    "route_replay": "按观察路线回想关键视角",
    "landmark_hierarchy": "先用稳定地标分区，再比较局部位置",
    "mental_simulation": "在脑中切换观察者位置或朝向",
    "elimination": "排除与给定观察冲突的候选",
    "temporal_recall": "按观察顺序检索出现和消失",
    "calibration": "区分没有看到与能够断言不存在",
    "scale_anchor": "用画面中的已绑定物体形成尺度判断",
}


_TASK_STRATEGIES: dict[str, tuple[str, ...]] = {
    "metric_distance": ("scale_anchor", "landmark_hierarchy"),
    "egocentric_relation": ("mental_simulation", "elimination"),
    "cross_view_relation": ("route_replay", "landmark_hierarchy", "mental_simulation"),
    "counterfactual_verification": ("route_replay", "elimination"),
    "last_seen_memory": ("temporal_recall",),
    "orbit_identity": ("mental_simulation", "landmark_hierarchy"),
    "rotation_change_detection": ("mental_simulation", "temporal_recall"),
    "elevation_relation_transfer": ("mental_simulation", "landmark_hierarchy"),
    "object_centric_perspective": ("mental_simulation",),
    "target_view_prediction": ("mental_simulation", "elimination"),
    "grounding_presence": ("temporal_recall",),
    "unknown_abstention": ("temporal_recall", "calibration"),
    "evidence_presence_unknown": ("temporal_recall", "calibration"),
    "evidence_presence_reveal": ("temporal_recall",),
    "occlusion_unknown": ("temporal_recall", "calibration"),
    "occlusion_reveal": ("temporal_recall",),
}

_RELATION_SURFACES = (
    "左前方",
    "右前方",
    "左后方",
    "右后方",
    "前方偏左",
    "前方偏右",
    "后方偏左",
    "后方偏右",
    "左侧",
    "右侧",
    "前方",
    "后方",
    "上方",
    "下方",
)


def _view_index(view_id: str, exposure_view_ids: Sequence[str]) -> int:
    try:
        return tuple(exposure_view_ids).index(view_id) + 1
    except ValueError as error:
        raise BlueprintError(f"view {view_id!r} is outside the model exposure") from error


def _view_surface(view_id: str, exposure_view_ids: Sequence[str]) -> str:
    return f"第{_view_index(view_id, exposure_view_ids)}个视角"


def _explicit_anchor_view(question_zh: str, name: str, exposure_view_ids: Sequence[str]) -> str | None:
    for view_id in exposure_view_ids:
        original_number = view_number(view_id)
        candidates = (
            f"第{original_number}个视角里看到的{name}",
            f"第{original_number}个视角中看到的{name}",
            f"第{original_number}个视角看到的{name}",
        )
        if any(surface in question_zh for surface in candidates):
            return view_id
    return None


def _entity_surface(
    bundle: Bundle,
    entity_id: str,
    *,
    source_question_zh: str,
    exposure_view_ids: Sequence[str],
    evidence_view_ids: Sequence[str],
) -> str:
    try:
        entity = bundle.entities[entity_id]
    except KeyError as error:
        raise BlueprintError(f"question references unknown entity {entity_id}") from error
    name = entity_name(entity.label)
    explicit = _explicit_anchor_view(source_question_zh, name, exposure_view_ids)
    if explicit is not None:
        return f"{_view_surface(explicit, exposure_view_ids)}里看到的{name}"

    # A bare name in compiler-approved source text means the category/track is
    # already unique over its binding views.  Preserve that natural surface.
    if name in source_question_zh:
        return name

    candidates = [
        view_id
        for view_id in (*evidence_view_ids, *exposure_view_ids)
        if view_id in bundle.view_by_id
        and entity_id in bundle.view_by_id[view_id].visible_entity_ids
    ]
    if not candidates:
        raise BlueprintError(f"entity {entity_id} has no exposed visual anchor")
    anchor = candidates[0]
    return f"{_view_surface(anchor, exposure_view_ids)}里看到的{name}"


def _target_category(question_zh: str) -> str:
    patterns = (
        r"存在([^？?，。]+)[？?]?$",
        r"看到([^？?，。]+)[？?]?$",
        r"能看到([^？?，。]+)[吗么]?[？?]?$",
    )
    for pattern in patterns:
        match = re.search(pattern, question_zh)
        if match:
            value = match.group(1).strip().removesuffix("吗")
            if value:
                return value
    raise BlueprintError(f"cannot recover answer-free target category from {question_zh!r}")


def _tested_relation(question_zh: str) -> str:
    # Reference-frame wording itself commonly contains “正前方”.  The tested
    # claim appears later in the sentence (usually after “有人说”), so select
    # the final relation mention rather than the first direction token.
    matches = [
        (question_zh.rfind(surface), surface)
        for surface in _RELATION_SURFACES
        if surface in question_zh
    ]
    if matches:
        return max(matches, key=lambda value: (value[0], len(value[1])))[1]
    raise BlueprintError("counterfactual question has no explicit tested relation")


def _numbered_motion(question_zh: str) -> str:
    # Relative motion is a question premise, not an answer.  Keep it as one
    # protected slot so a language model cannot change its signs or values.
    match = re.search(r"(先.+?)(?:到达这个目标视角后|此时)", question_zh)
    if not match:
        raise BlueprintError("target-view question has no exposed motion instruction")
    return match.group(1).strip("，。； ")


def _entity_slots(
    episode: Mapping[str, Any],
    question: Mapping[str, Any],
    exposure_view_ids: Sequence[str],
) -> tuple[Bundle, list[str]]:
    bundle = Bundle(
        Path(str(episode["source_bundle"])),
        trajectory_class=str(episode["trajectory_class"]),
        source_sweep=str(episode["source_sweep"]),
        job_status="passed",
    )
    entity_ids = [str(value) for value in question.get("evidence_entity_ids", [])]
    surfaces = [
        _entity_surface(
            bundle,
            entity_id,
            source_question_zh=str(question["question_zh"]),
            exposure_view_ids=exposure_view_ids,
            evidence_view_ids=tuple(str(value) for value in question.get("evidence_view_ids", [])),
        )
        for entity_id in entity_ids
    ]
    return bundle, surfaces


def _specification(
    episode: Mapping[str, Any],
    question: Mapping[str, Any],
    exposure_view_ids: Sequence[str],
) -> tuple[str, dict[str, str], tuple[str, ...], str]:
    task = str(question["task_type"])
    source_views = tuple(str(value) for value in question.get("model_view_ids", exposure_view_ids))
    _, entities = _entity_slots(episode, question, exposure_view_ids)

    if task == "metric_distance":
        intent = "综合有序观察，询问两个已绑定物体中心之间的米制距离。"
        slots = {"subject": entities[0], "reference": entities[1]}
        contract = "一个自然的距离问题；不得提供或暗示距离值。"
    elif task == "egocentric_relation":
        variant = str(question.get("family_variant", "canonical"))
        frame_view_id = source_views[-1] if variant == "frame_b" else source_views[0]
        intent = "严格按指定相机朝向，询问两个已绑定物体的前后或左右关系。"
        slots = {
            "frame_view": _view_surface(frame_view_id, exposure_view_ids),
            "subject": entities[0],
            "reference": entities[1],
        }
        contract = "必须明确参照朝向；只问一个方向关系。"
    elif task == "cross_view_relation":
        intent = "先综合不同视角，再按首个观察的朝向询问两个非共视物体的方向关系。"
        slots = {
            "frame_view": _view_surface(exposure_view_ids[0], exposure_view_ids),
            "subject": entities[0],
            "reference": entities[1],
        }
        contract = "体现跨视角整合，但不得使用坐标注册等内部术语。"
    elif task == "counterfactual_verification":
        intent = "判断一个跨视角方向陈述是否与完整观察一致。"
        slots = {
            "frame_view": _view_surface(exposure_view_ids[0], exposure_view_ids),
            "subject": entities[0],
            "reference": entities[1],
            "tested_relation": _tested_relation(str(question["question_zh"])),
        }
        contract = "完整复述待验证的陈述；不得暗示它为真或为假。"
    elif task == "last_seen_memory":
        intent = "在有序观察结束后，询问目标最后一次出现在哪个视角。"
        slots = {"target": entities[0]}
        contract = "要求回忆最后一次出现；不得提供视角编号。"
    elif task == "orbit_identity":
        intent = "只比较指定的两个环绕视角，询问画面目标是否为同一实例。"
        slots = {
            "view_a": _view_surface(source_views[0], exposure_view_ids),
            "view_b": _view_surface(source_views[-1], exposure_view_ids),
            "target": entities[0],
        }
        contract = "明确限定两个视角；不要暗示是否同一物体。"
    elif task == "rotation_change_detection":
        rotation = re.search(r"约\s*\d+(?:\.\d+)?\s*°?度?", str(question["question_zh"]))
        if not rotation:
            raise BlueprintError("rotation question lacks an exposed rotation amount")
        intent = "只比较指定的原地旋转前后两帧，询问唯一新进入主要视野的物体类别。"
        slots = {
            "view_a": _view_surface(source_views[0], exposure_view_ids),
            "view_b": _view_surface(source_views[-1], exposure_view_ids),
            "rotation": rotation.group(0),
        }
        contract = "不能向问题编辑器暴露将进入视野的实体。"
    elif task == "elevation_relation_transfer":
        intent = "比较同一站位的低、高视角，按低视角朝向询问两个物体的稳定方向关系。"
        slots = {
            "low_view": _view_surface(source_views[0], exposure_view_ids),
            "high_view": _view_surface(source_views[-1], exposure_view_ids),
            "subject": entities[0],
            "reference": entities[1],
        }
        contract = "明确低、高视角和参照朝向；不得暗示方向答案。"
    elif task == "object_centric_perspective":
        intent = "让观察者站到一个物体的位置并面向第二个物体，询问第三个物体的方位。"
        slots = {"origin": entities[0], "facing": entities[1], "target": entities[2]}
        contract = "用生活化的假想站位表达，不得给出目标方位。"
    elif task == "target_view_prediction":
        if len(entities) == 3:
            intent = "构造一个由站位和朝向定义的新视角，询问该视角能否看到指定目标。"
            slots = {"origin": entities[0], "facing": entities[1], "target": entities[2]}
        elif len(entities) == 1:
            intent = "按给定相机相对运动构造新视角，询问能否看到指定目标。"
            slots = {"motion": _numbered_motion(str(question["question_zh"])), "target": entities[0]}
        else:
            raise BlueprintError("target-view question needs one or three exposed entities")
        contract = "不得透露隐藏目标渲染或可见性答案。"
    elif task in {"unknown_abstention", "evidence_presence_unknown", "occlusion_unknown"}:
        intent = "只依据给定画面，询问是否有足够证据确认一个目标类别存在。"
        slots = {"target_category": _target_category(str(question["question_zh"]))}
        contract = "问题必须允许回答证据不足；不得暗示场景中存在或不存在。"
    elif task in {"evidence_presence_reveal", "occlusion_reveal"}:
        intent = "只依据当前有序画面，询问是否能够确认目标类别存在。"
        slots = {"target_category": _target_category(str(question["question_zh"]))}
        contract = "不得暗示哪一帧提供了决定证据。"
    elif task == "grounding_presence":
        intent = "在指定单帧中询问能否看到目标类别。"
        target = entities[0] if entities else _target_category(str(question["question_zh"]))
        slots = {"view": _view_surface(source_views[0], exposure_view_ids), "target": target}
        contract = "只问该帧的可见性，不得暗示是或否。"
    else:
        raise BlueprintError(f"unsupported task type {task!r}")
    required = tuple(slots)
    return intent, slots, required, contract


def build_question_blueprint(
    episode: Mapping[str, Any],
    question: Mapping[str, Any],
    *,
    exposure_view_ids: Sequence[str] | None = None,
) -> QuestionBlueprint:
    """Compile one question-only language request.

    The implementation deliberately never reads ``answer_zh``,
    ``answer_value``, ``answer_status``, ``certificate``, or ``rationale_zh``.
    Tests place sentinels in those fields to enforce this contract.
    """

    task = str(question.get("task_type", ""))
    if task not in _TASK_STRATEGIES:
        raise BlueprintError(f"task {task!r} has no human strategy contract")
    model_views = tuple(str(value) for value in question.get("model_view_ids", ()))
    exposure = model_views if exposure_view_ids is None else tuple(exposure_view_ids)
    if not exposure:
        raise BlueprintError("question blueprint has no model-visible exposure")
    intent, slots, required, output_contract = _specification(episode, question, exposure)
    family_key = str(
        question.get("consistency_group")
        or f"{episode.get('episode_id')}:{question.get('task_type')}:{question.get('family_variant', 'canonical')}"
    )
    identity = {
        "fact_id": str(question["fact_id"]),
        "task_type": task,
        "exposure": exposure,
        "intent": intent,
        "slots": slots,
    }
    return QuestionBlueprint(
        request_id=content_id("question-request", identity),
        fact_id=str(question["fact_id"]),
        task_type=task,
        capability=capability_for(task),
        intent_zh=intent,
        slot_values=slots,
        required_slots=required,
        allowed_strategies=_TASK_STRATEGIES[task],
        output_contract=output_contract,
        family_key=family_key,
    )


def strategy_library() -> dict[str, str]:
    return dict(STRATEGIES)


__all__ = ["BlueprintError", "build_question_blueprint", "strategy_library"]
