"""Batch-level research audit and browser review artifacts for Scriptgen QA."""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .qa_dataset import (
    _json_sha,
    _write_json,
    _write_jsonl,
    read_jsonl,
    verify_source_snapshot,
)

REFLECTION_SCHEMA = "scriptgen.qa_batch_reflection.v1"
REVIEW_SCHEMA = "scriptgen.qa_review.v1"
REVIEW_TEMPLATE = Path(__file__).resolve().parents[3] / "web" / "scriptgen_qa_review.html"


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _raw_case(predictions: dict[str, dict[str, Any]]) -> tuple[str, str]:
    multimodal = predictions.get("multimodal")
    vision_free = predictions.get("vision_free")
    if multimodal is None or vision_free is None:
        return "protocol_failure", "missing_required_condition"
    if multimodal.get("error") or vision_free.get("error"):
        return "protocol_failure", "model_runtime_error"
    if not multimodal.get("parseable"):
        return "valid_hard", "multimodal_answer_outside_declared_labels"
    if vision_free.get("correct"):
        return "shortcut_risk", "gold_recovered_without_images"
    if not multimodal.get("correct"):
        return "valid_hard", "compiler_valid_but_frozen_model_wrong"
    return "eligible", "visual_condition_correct_and_vision_free_wrong"


def _stream_case(predictions: dict[str, dict[str, Any]]) -> tuple[str, str]:
    required = {
        "multimodal_teacher_forced",
        "multimodal_free_running",
        "vision_free_teacher_forced",
        "vision_free_free_running",
    }
    if not required <= predictions.keys():
        return "protocol_failure", "missing_required_condition"
    selected = [predictions[name] for name in sorted(required)]
    if any(row.get("error") for row in selected):
        return "protocol_failure", "model_runtime_error"
    teacher = predictions["multimodal_teacher_forced"]
    free = predictions["multimodal_free_running"]
    no_vision = predictions["vision_free_teacher_forced"]
    if not all(turn.get("parseable") for turn in teacher.get("turns", [])):
        return "valid_hard", "teacher_forced_answer_outside_declared_labels"
    if not all(turn.get("parseable") for turn in free.get("turns", [])):
        return "valid_hard", "free_running_answer_outside_declared_labels"
    teacher_all = all(turn["correct"] for turn in teacher["turns"])
    free_all = all(turn["correct"] for turn in free["turns"])
    no_vision_all = all(turn["correct"] for turn in no_vision["turns"])
    if no_vision_all:
        return "shortcut_risk", "all_turns_recovered_without_images"
    if teacher_all and not free_all:
        return "valid_hard", "free_running_history_compounds_errors"
    if not teacher_all:
        return "valid_hard", "compiler_valid_but_teacher_forced_model_wrong"
    return "eligible", "visual_teacher_and_free_running_succeed"


def _counter(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def _capability_assessment(
    rows: list[dict[str, Any]], cases: dict[str, tuple[str, str]], *, streaming: bool
) -> dict[str, Any]:
    tiers = Counter(row["tier"] for row in rows)
    scene_count = len({row["cluster_ids"]["scene"] for row in rows})
    hard = sum(cases[row["record_id"]][0] == "valid_hard" for row in rows)
    shortcuts = sum(cases[row["record_id"]][0] == "shortcut_risk" for row in rows)
    statements = []
    if tiers.get("P1"):
        statements.append("P1 样本要求根据第一人称时序更新自运动、净转向、回指方向或历史画面位置。")
    if tiers.get("P2"):
        statements.append(
            "P2 样本要求把目标变换到假想站位和朝向；同一轨迹的角度兄弟题未串进同一历史。"
        )
    if tiers.get("P3"):
        statements.append("P3 样本要求通过跨视图锚链绑定非共视对象，并含删关键证据的弃答对照。")
    if streaming:
        statements.append("流式记录只在编译器认可的前缀提问，可测证据到达后的状态更新及错误累积。")
    verdict = "supported" if rows and scene_count >= 2 else "limited"
    return {
        "verdict": verdict,
        "scene_count": scene_count,
        "tier_counts": dict(sorted(tiers.items())),
        "valid_hard_count": hard,
        "shortcut_risk_count": shortcuts,
        "reasoning": statements,
        "boundary": (
            "这些记录能诊断所声明的空间程序，但当前是小规模 development pool，"
            "不能据此概括模型的全部空间智能。"
        ),
    }


def _generalization_assessment(
    rows: list[dict[str, Any]], cases: dict[str, tuple[str, str]], *, streaming: bool
) -> dict[str, Any]:
    capabilities = {
        capability
        for row in rows
        for capability in (row.get("capabilities", []) if streaming else [row["capability"]])
    }
    scenes = {row["cluster_ids"]["scene"] for row in rows}
    variants = {row.get("variant") for row in rows if row.get("variant")}
    shortcut_count = sum(cases[row["record_id"]][0] == "shortcut_risk" for row in rows)
    return {
        "supervision_mechanisms": [
            "同一几何事实在完整、删关键证据、删冗余证据、延迟和安全乱序条件下保持可执行一致性。"
            if not streaming
            else "同一轨迹按证据前缀监督弃答、修正和持续状态读取。",
            "scene、trajectory、family 三层聚类键允许后续 scaling 时做无泄漏切分。",
            "可执行 certificate 为 SFT 标签和 RL exact-label reward 提供同一真值来源。",
        ],
        "coverage": {
            "scene_count": len(scenes),
            "capability_count": len(capabilities),
            "variant_count": len(variants),
        },
        "shortcut_risk_count": shortcut_count,
        "conclusion": (
            "这些机制有理由给模型施加跨场景、跨视图和证据干预的一致性监督，"
            "但本轮没有训练或 held-out scaling 实验，因此只能判断监督设计是否支持泛化，"
            "不能声称已经提升空间泛化能力。"
        ),
    }


def _strategy_decision(
    rows: list[dict[str, Any]], cases: dict[str, tuple[str, str]]
) -> dict[str, Any]:
    categories = Counter(cases[row["record_id"]][0] for row in rows)
    protocol_rate = _ratio(categories["protocol_failure"], len(rows)) or 0.0
    shortcut_rate = _ratio(categories["shortcut_risk"], len(rows)) or 0.0
    actions = []
    status = "keep"
    if protocol_rate:
        status = "hold"
        actions.append("修复模型输入或答案解析协议，并按 input hash 重跑受影响记录。")
    if shortcut_rate >= 0.35:
        status = "review"
        actions.append("检查标签分布和问题文本捷径；保留 compiler 金标，不因模型猜中而改答案。")
    if not actions:
        actions.append("保持冻结问题表面和证据调度，继续分层交错生成下一批。")
    return {
        "status": status,
        "case_counts": dict(sorted(categories.items())),
        "actions": actions,
        "gold_change_allowed": False,
    }


def _batch_reflection(
    *,
    batch_id: str,
    rows: list[dict[str, Any]],
    cases: dict[str, tuple[str, str]],
    prediction_rows: dict[str, dict[str, dict[str, Any]]],
    streaming: bool,
) -> dict[str, Any]:
    condition_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for condition, prediction in prediction_rows.get(row["record_id"], {}).items():
            if prediction.get("error"):
                continue
            if streaming:
                condition_rows[condition].extend(prediction.get("turns", []))
            else:
                condition_rows[condition].append(prediction)
    return {
        "schema_version": REFLECTION_SCHEMA,
        "batch_id": batch_id,
        "sample_type": "streaming_qa" if streaming else "raw_qa",
        "record_count": len(rows),
        "distributions": {
            "tier": _counter(rows, "tier"),
            "capability": (
                dict(
                    sorted(
                        Counter(
                            capability for row in rows for capability in row["capabilities"]
                        ).items()
                    )
                )
                if streaming
                else _counter(rows, "capability")
            ),
            "variant": {} if streaming else _counter(rows, "variant"),
            "scene_count": len({row["cluster_ids"]["scene"] for row in rows}),
        },
        "model_evaluation": {
            condition: {
                "decision_count": len(values),
                "accuracy": (
                    sum(bool(value["correct"]) for value in values) / len(values)
                    if values
                    else None
                ),
                "semantic_parse_rate": (
                    sum(bool(value.get("parseable")) for value in values) / len(values)
                    if values
                    else None
                ),
                "strict_format_rate": (
                    sum(bool(value.get("format_compliant")) for value in values) / len(values)
                    if values
                    else None
                ),
            }
            for condition, values in sorted(condition_rows.items())
        },
        "capability_assessment": _capability_assessment(rows, cases, streaming=streaming),
        "generalization_assessment": _generalization_assessment(rows, cases, streaming=streaming),
        "strategy_decision": _strategy_decision(rows, cases),
        "case_examples": [
            {
                "record_id": row["record_id"],
                "category": cases[row["record_id"]][0],
                "reason": cases[row["record_id"]][1],
            }
            for row in rows
            if cases[row["record_id"]][0] != "eligible"
        ][:20],
    }


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def _reflection_markdown(
    reflections: list[dict[str, Any]],
    manifest: dict[str, Any],
    model_summary: dict[str, Any],
) -> str:
    lines = [
        "# Scriptgen Raw / Streaming QA 批次研究审计 v1",
        "",
        "## 数据边界",
        "",
        f"本轮扫描 {manifest['source_trajectory_count']} 条已接受轨迹，导出 "
        f"{manifest['raw_record_count']} 条 raw QA、{manifest['streaming_record_count']} 条 "
        f"streaming QA（{manifest['streaming_turn_count']} 个 turn）。",
        "床存在性按既定范围排除；模型输入只有 RGB。当前产物是 development pool，未划分训练集或测试集，也没有执行训练。",
        "",
        "SenseNova-SI-1.5-InternVL3-8B 只作为冻结诊断模型。模型输出不会生成、修改或否决 compiler 金标。",
        "",
        "## 冻结模型总体结果",
        "",
    ]
    for condition, values in model_summary.get("raw", {}).get("conditions", {}).items():
        lines.append(
            f"- raw `{condition}`：语义准确率 {_pct(values.get('accuracy'))}，"
            f"可解析率 {_pct(values.get('parse_rate'))}，严格格式率 "
            f"{_pct(values.get('format_rate'))}（n={values.get('count', 0)}）。"
        )
    for condition, values in model_summary.get("streaming", {}).get("conditions", {}).items():
        turns = values.get("turns", {})
        lines.append(
            f"- streaming `{condition}`：turn 语义准确率 {_pct(turns.get('accuracy'))}，"
            f"最终 turn 准确率 {_pct(values.get('final_turn_accuracy'))}，全 turn 正确率 "
            f"{_pct(values.get('all_turns_correct_rate'))}（records={values.get('records', 0)}，"
            f"turns={turns.get('count', 0)}）。"
        )
    lines.extend(
        [
            "",
            "总体均值只用于检查评测是否完整；能力结论必须结合下列逐批、分层和 vision-free 对照。",
            "",
        ]
    )
    for reflection in reflections:
        model = reflection["model_evaluation"]
        cases = reflection["strategy_decision"]["case_counts"]
        lines.extend(
            [
                f"## {reflection['batch_id']} · {reflection['record_count']} 条",
                "",
                f"- 层级分布：`{json.dumps(reflection['distributions']['tier'], ensure_ascii=False)}`；"
                f"场景数：{reflection['distributions']['scene_count']}。",
                "- 模型条件："
                + "；".join(
                    f"{condition}=语义{_pct(values['accuracy'])} / "
                    f"格式{_pct(values['strict_format_rate'])} (n={values['decision_count']})"
                    for condition, values in model.items()
                )
                + "。",
                f"- case 分类：`{json.dumps(cases, ensure_ascii=False)}`。",
                "",
                "空间心智能力判断："
                + "".join(reflection["capability_assessment"]["reasoning"])
                + reflection["capability_assessment"]["boundary"],
                "",
                "空间泛化判断：" + reflection["generalization_assessment"]["conclusion"],
                "",
                "下一批策略：" + "".join(reflection["strategy_decision"]["actions"]),
                "",
            ]
        )
        examples = reflection["case_examples"]
        if examples:
            lines.extend(
                [
                    "需 review 的代表 case：",
                    "",
                    *(
                        f"- `{row['record_id']}`：{row['category']} / {row['reason']}"
                        for row in examples
                    ),
                    "",
                ]
            )
    lines.extend(
        [
            "## 结论边界",
            "",
            "这批数据已经能以可执行金标测试 P1 自运动更新、P2 假想参照系变换和 P3 跨视图绑定，"
            "并通过删证据与 vision-free 条件检查模型是否依赖视觉。它尚不能证明训练收益：需要后续按 scene/plan/family 聚类扩量，"
            "再进行训练前后和 held-out composition 对照。",
            "",
        ]
    )
    return "\n".join(lines)


def build_research_review(*, dataset_root: Path) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    raw = read_jsonl(dataset_root / "raw_qa.jsonl")
    streaming = read_jsonl(dataset_root / "streaming_qa.jsonl")
    predictions = read_jsonl(dataset_root / "sensenova_predictions.jsonl")
    model_summary = json.loads(
        (dataset_root / "sensenova_summary.json").read_text(encoding="utf-8")
    )
    prediction_rows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in predictions:
        prediction_rows[row["record_id"]][row["condition"]] = row

    cases: dict[str, tuple[str, str]] = {}
    for row in raw:
        cases[row["record_id"]] = _raw_case(prediction_rows.get(row["record_id"], {}))
    for row in streaming:
        cases[row["record_id"]] = _stream_case(prediction_rows.get(row["record_id"], {}))

    by_batch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in [*raw, *streaming]:
        by_batch[row["batch_id"]].append(row)
    reflections = []
    reflection_dir = dataset_root / "batch_reflections"
    reflection_dir.mkdir(parents=True, exist_ok=True)
    for batch_id, rows in sorted(by_batch.items()):
        reflection = _batch_reflection(
            batch_id=batch_id,
            rows=rows,
            cases=cases,
            prediction_rows=prediction_rows,
            streaming=rows[0]["sample_type"] == "streaming_qa",
        )
        reflections.append(reflection)
        _write_json(reflection_dir / f"{batch_id}.json", reflection)

    quarantine = [
        {
            "record_id": record_id,
            "category": category,
            "reason": reason,
        }
        for record_id, (category, reason) in sorted(cases.items())
        if category in {"protocol_failure", "data_invalid", "out_of_scope"}
    ]
    _write_jsonl(dataset_root / "quarantine.jsonl", quarantine)
    source_check = verify_source_snapshot(
        source_root=Path(manifest["source_root"]),
        snapshot_path=dataset_root / "source_snapshot.jsonl",
    )
    _write_json(dataset_root / "source_integrity_after.json", source_check)

    raw_inputs = {
        row["record_id"]: row for row in read_jsonl(dataset_root / "raw_eval_inputs.jsonl")
    }
    raw_oracles = {
        row["record_id"]: row for row in read_jsonl(dataset_root / "raw_eval_oracle.jsonl")
    }
    stream_inputs = {
        row["record_id"]: row for row in read_jsonl(dataset_root / "streaming_eval_inputs.jsonl")
    }
    stream_oracles = {
        row["record_id"]: row for row in read_jsonl(dataset_root / "streaming_eval_oracle.jsonl")
    }
    review_rows = []
    for row in raw:
        review_rows.append(
            {
                "record_id": row["record_id"],
                "sample_type": "raw_qa",
                "batch_id": row["batch_id"],
                "tier": row["tier"],
                "capabilities": [row["capability"]],
                "role": row["role"],
                "variant": row["variant"],
                "cluster_ids": row["cluster_ids"],
                "input": raw_inputs[row["record_id"]],
                "oracle": {"answer": raw_oracles[row["record_id"]]["answer"]},
                "predictions": prediction_rows.get(row["record_id"], {}),
                "review_class": cases[row["record_id"]][0],
                "review_reason": cases[row["record_id"]][1],
            }
        )
    for row in streaming:
        review_rows.append(
            {
                "record_id": row["record_id"],
                "sample_type": "streaming_qa",
                "batch_id": row["batch_id"],
                "tier": row["tier"],
                "capabilities": row["capabilities"],
                "role": None,
                "variant": "canonical",
                "cluster_ids": row["cluster_ids"],
                "input": stream_inputs[row["record_id"]],
                "oracle": {
                    "turns": [
                        {
                            key: turn[key]
                            for key in (
                                "turn_id",
                                "prefix_length",
                                "answer",
                                "status",
                                "certificate_sha256",
                                "family_id",
                            )
                        }
                        for turn in stream_oracles[row["record_id"]]["turns"]
                    ]
                },
                "predictions": prediction_rows.get(row["record_id"], {}),
                "review_class": cases[row["record_id"]][0],
                "review_reason": cases[row["record_id"]][1],
            }
        )
    review = {
        "schema_version": REVIEW_SCHEMA,
        "prediction_snapshot_sha256": _json_sha(predictions),
        "manifest": manifest,
        "model_summary": model_summary,
        "source_integrity": source_check,
        "case_counts": dict(sorted(Counter(value[0] for value in cases.values()).items())),
        "reflections": reflections,
        "rows": review_rows,
    }
    _write_json(dataset_root / "review.json", review)
    (dataset_root / "research_report.md").write_text(
        _reflection_markdown(reflections, manifest, model_summary), encoding="utf-8"
    )
    if not REVIEW_TEMPLATE.is_file():
        raise FileNotFoundError(f"missing QA review template: {REVIEW_TEMPLATE}")
    shutil.copyfile(REVIEW_TEMPLATE, dataset_root / "index.html")
    return review
