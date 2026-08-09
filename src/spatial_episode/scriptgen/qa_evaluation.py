"""Frozen SenseNova evaluation for Scriptgen raw and streaming QA."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .qa_dataset import _write_json, _write_jsonl, read_jsonl

EVALUATION_SCHEMA = "scriptgen.sensenova_eval.v1"
PROMPT_VERSION = "scriptgen-qa-eval.v2"

LABEL_ALIASES_ZH = {
    "front": ("前方", "前面"),
    "left": ("左侧", "左边"),
    "back": ("后方", "后面"),
    "right": ("右侧", "右边"),
    "left_half": ("画面左半边", "左半边"),
    "right_half": ("画面右半边", "右半边"),
    "over_90": ("超过90度", "大于90度"),
    "at_most_90": ("不超过90度", "至多90度"),
    "visible": ("可见", "能看见"),
    "not_visible": ("不可见", "看不见"),
    "present": ("存在", "有"),
    "absent": ("不存在", "没有"),
    "first": ("第一个",),
    "second": ("第二个",),
    "无法判断": ("证据不足", "不能判断", "无法确定"),
}


def _normalized_answer_span(response: str) -> tuple[str, bool]:
    tagged = re.findall(r"<answer>\s*([^<>\n]+?)\s*</answer>", response, re.IGNORECASE)
    scope = tagged[-1] if tagged else response.strip()
    return scope.strip().strip("。.!！ `\"'"), bool(tagged)


def answer_format_compliant(response: str, choices: Iterable[str]) -> bool:
    scope, tagged = _normalized_answer_span(response)
    return tagged and any(scope.casefold() == choice.casefold() for choice in choices)


def parse_answer_label(
    response: str, choices: Iterable[str], *, question: str | None = None
) -> str | None:
    """Parse a semantic answer while measuring exact-label format separately."""

    choices = tuple(choices)
    normalized, tagged = _normalized_answer_span(response)
    scopes = [normalized] if tagged else [normalized, response.strip()]
    for scope in scopes:
        normalized = scope.strip().strip("。.!！ `\"'")
        exact = [choice for choice in choices if normalized.casefold() == choice.casefold()]
        if len(exact) == 1:
            return exact[0]
        matches = []
        for choice in choices:
            if choice == "无法判断":
                found = choice in scope
            else:
                found = bool(
                    re.search(
                        rf"(?<![A-Za-z0-9_]){re.escape(choice)}(?![A-Za-z0-9_])",
                        scope,
                        re.IGNORECASE,
                    )
                )
            if found:
                matches.append(choice)
        if len(matches) == 1:
            return matches[0]
        alias_matches = []
        for choice in choices:
            aliases = LABEL_ALIASES_ZH.get(choice, ())
            if any(normalized == alias or alias in scope for alias in aliases):
                alias_matches.append(choice)
        # Longer negative aliases such as “不可见” must not also match “可见”.
        alias_matches = [
            choice
            for choice in alias_matches
            if not any(
                choice != other
                and any(
                    alias in scope and len(alias) > len(candidate)
                    for alias in LABEL_ALIASES_ZH.get(other, ())
                    for candidate in LABEL_ALIASES_ZH.get(choice, ())
                )
                for other in alias_matches
            )
        ]
        if len(alias_matches) == 1:
            return alias_matches[0]
        if question and {"first", "second"} <= set(choices):
            pair = re.search(
                r"([A-Za-z0-9_]+)和([A-Za-z0-9_]+)中[，,]?哪一个",
                question,
            )
            if pair:
                object_matches = [
                    label
                    for label, entity in zip(("first", "second"), pair.groups(), strict=True)
                    if re.search(
                        rf"(?<![A-Za-z0-9_]){re.escape(entity)}(?![A-Za-z0-9_])",
                        scope,
                        re.IGNORECASE,
                    )
                ]
                if len(object_matches) == 1:
                    return object_matches[0]
    return None


def _prediction_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["record_id"]), str(row["condition"])


def _prediction_sha(input_sha256: str, condition: str, model_path: Path) -> str:
    text = "|".join((input_sha256, condition, str(model_path.resolve()), PROMPT_VERSION))
    return hashlib.sha256(text.encode()).hexdigest()


class SenseNovaEvaluator:
    """Thin adapter around the repository's already audited InternVL loader."""

    def __init__(self, model_path: Path) -> None:
        from episode3d.transform_pilot.evaluation import InternVLBackend

        self.model_path = model_path.resolve()
        started = time.monotonic()
        self.backend = InternVLBackend(self.model_path)
        self.load_seconds = round(time.monotonic() - started, 3)

    @property
    def model_id(self) -> str:
        return self.model_path.name

    def _text_only(self, prompt: str, *, maximum_new_tokens: int = 32) -> str:
        response = self.backend.model.chat(
            self.backend.tokenizer,
            pixel_values=None,
            question=prompt,
            history=None,
            num_patches_list=[],
            generation_config={
                "max_new_tokens": maximum_new_tokens,
                "do_sample": False,
                "pad_token_id": (
                    self.backend.tokenizer.pad_token_id or self.backend.tokenizer.eos_token_id
                ),
            },
        )
        return str(response)

    def raw(self, item: dict[str, Any], source_root: Path, *, vision: bool) -> str:
        prompt = f"{item['system']}\n{item['question']}"
        if not vision:
            return self._text_only("[本条件不提供图像]\n" + prompt)
        images = [source_root / image["path"] for image in item["images"]]
        return self.backend.generate(
            images=images,
            prompt=prompt,
            maximum_new_tokens=32,
        )

    def streaming(
        self,
        item: dict[str, Any],
        oracle: dict[str, Any],
        source_root: Path,
        *,
        vision: bool,
        teacher_forced: bool,
    ) -> list[dict[str, Any]]:
        cumulative_images: list[Path] = []
        history: list[tuple[str, str]] = []
        rows = []
        oracle_by_turn = {turn["turn_id"]: turn for turn in oracle["turns"]}
        global_image_index = 0
        for turn_index, turn in enumerate(item["turns"]):
            new_paths = [source_root / image["path"] for image in turn["new_images"]]
            image_prefix = ""
            if vision:
                markers = []
                for _ in new_paths:
                    global_image_index += 1
                    markers.append(f"Image-{global_image_index}: <image>\n")
                image_prefix = "".join(markers)
                cumulative_images.extend(new_paths)
            question = image_prefix + str(turn["question"])
            if turn_index == 0:
                question = f"{item['system']}\n{question}"
            started = time.monotonic()
            if vision:
                pixel_values, patch_counts = self.backend._pixels(cumulative_images)
                response = self.backend.model.chat(
                    self.backend.tokenizer,
                    pixel_values=pixel_values,
                    num_patches_list=patch_counts,
                    question=question,
                    history=list(history),
                    generation_config={
                        "max_new_tokens": 32,
                        "do_sample": False,
                        "pad_token_id": (
                            self.backend.tokenizer.pad_token_id
                            or self.backend.tokenizer.eos_token_id
                        ),
                    },
                )
            else:
                hidden_question = (
                    "[本条件不提供图像]\n" if turn_index == 0 else ""
                ) + question.replace("<image>", "[图像隐藏]")
                response = self.backend.model.chat(
                    self.backend.tokenizer,
                    pixel_values=None,
                    num_patches_list=[],
                    question=hidden_question,
                    history=list(history),
                    generation_config={
                        "max_new_tokens": 32,
                        "do_sample": False,
                        "pad_token_id": (
                            self.backend.tokenizer.pad_token_id
                            or self.backend.tokenizer.eos_token_id
                        ),
                    },
                )
                question = hidden_question
            elapsed = round(time.monotonic() - started, 3)
            response = str(response)
            target = oracle_by_turn[turn["turn_id"]]["answer"]
            prediction = parse_answer_label(response, turn["choices"], question=turn["question"])
            rows.append(
                {
                    "turn_id": turn["turn_id"],
                    "capability": turn["capability"],
                    "prefix_length": turn["prefix_length"],
                    "ground_truth": target,
                    "prediction": prediction,
                    "correct": prediction == target,
                    "parseable": prediction is not None,
                    "format_compliant": answer_format_compliant(response, turn["choices"]),
                    "response": response,
                    "elapsed_seconds": elapsed,
                    "cumulative_image_count": len(cumulative_images) if vision else 0,
                }
            )
            history_answer = f"<answer>{target}</answer>" if teacher_forced else response
            history.append((question, history_answer))
        return rows


def _load_done(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.is_file():
        return {}
    result = {}
    for row in read_jsonl(path):
        result[_prediction_key(row)] = row
    return result


def _aggregate_binary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "accuracy": None, "parse_rate": None}
    return {
        "count": len(rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
        "parse_rate": sum(bool(row["parseable"]) for row in rows) / len(rows),
        "format_rate": (sum(bool(row.get("format_compliant")) for row in rows) / len(rows)),
    }


def _raw_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "conditions": {},
        "by_capability": {},
        "by_tier": {},
        "by_role": {},
        "by_variant": {},
    }
    conditions = sorted({row["condition"] for row in rows})
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition and not row.get("error")]
        result["conditions"][condition] = _aggregate_binary(selected)
    for capability in sorted({row["capability"] for row in rows}):
        result["by_capability"][capability] = {
            condition: _aggregate_binary(
                [
                    row
                    for row in rows
                    if row["capability"] == capability
                    and row["condition"] == condition
                    and not row.get("error")
                ]
            )
            for condition in conditions
        }
    for tier in sorted({row["tier"] for row in rows}):
        result["by_tier"][tier] = {
            condition: _aggregate_binary(
                [
                    row
                    for row in rows
                    if row["tier"] == tier
                    and row["condition"] == condition
                    and not row.get("error")
                ]
            )
            for condition in conditions
        }
    for role in sorted({row["role"] for row in rows}):
        result["by_role"][role] = {
            condition: _aggregate_binary(
                [
                    row
                    for row in rows
                    if row["role"] == role
                    and row["condition"] == condition
                    and not row.get("error")
                ]
            )
            for condition in conditions
        }
    for variant in sorted({row["variant"] for row in rows}):
        result["by_variant"][variant] = {
            condition: _aggregate_binary(
                [
                    row
                    for row in rows
                    if row["variant"] == variant
                    and row["condition"] == condition
                    and not row.get("error")
                ]
            )
            for condition in conditions
        }
    result["macro_accuracy"] = {}
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition and not row.get("error")]
        capability_scores = []
        for capability in sorted({row["capability"] for row in selected}):
            group = [row for row in selected if row["capability"] == capability]
            capability_scores.append(sum(row["correct"] for row in group) / len(group))
        family_scores = []
        for family in sorted({row["cluster_ids"]["family"] for row in selected}):
            group = [row for row in selected if row["cluster_ids"]["family"] == family]
            family_scores.append(sum(row["correct"] for row in group) / len(group))
        trajectory_scores = []
        for trajectory in sorted({row["cluster_ids"]["trajectory"] for row in selected}):
            group = [row for row in selected if row["cluster_ids"]["trajectory"] == trajectory]
            trajectory_scores.append(sum(row["correct"] for row in group) / len(group))
        result["macro_accuracy"][condition] = {
            "capability_macro": (
                sum(capability_scores) / len(capability_scores) if capability_scores else None
            ),
            "family_macro": sum(family_scores) / len(family_scores) if family_scores else None,
            "trajectory_macro": (
                sum(trajectory_scores) / len(trajectory_scores) if trajectory_scores else None
            ),
        }
    family_consistency = {}
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition and not row.get("error")]
        groups = defaultdict(list)
        for row in selected:
            groups[row["cluster_ids"]["family"]].append(row)
        complete = [
            group
            for group in groups.values()
            if any(row["variant"] == "canonical" for row in group)
        ]
        family_consistency[condition] = {
            "family_count": len(complete),
            "all_variants_correct_rate": (
                sum(all(row["correct"] for row in group) for group in complete) / len(complete)
                if complete
                else None
            ),
        }
    result["family_consistency"] = family_consistency
    paired = defaultdict(dict)
    for row in rows:
        if not row.get("error"):
            paired[row["record_id"]][row["condition"]] = row
    comparable = [
        value for value in paired.values() if {"multimodal", "vision_free"} <= value.keys()
    ]
    result["vision_contribution"] = {
        "paired_count": len(comparable),
        "multimodal_accuracy": (
            sum(value["multimodal"]["correct"] for value in comparable) / len(comparable)
            if comparable
            else None
        ),
        "vision_free_accuracy": (
            sum(value["vision_free"]["correct"] for value in comparable) / len(comparable)
            if comparable
            else None
        ),
    }
    return result


def _stream_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"conditions": {}, "by_capability": {}}
    turn_rows = [
        {**turn, "condition": row["condition"], "record_id": row["record_id"]}
        for row in rows
        if not row.get("error")
        for turn in row.get("turns", [])
    ]
    conditions = sorted({row["condition"] for row in rows})
    for condition in conditions:
        records = [row for row in rows if row["condition"] == condition and not row.get("error")]
        turns = [row for row in turn_rows if row["condition"] == condition]
        result["conditions"][condition] = {
            "records": len(records),
            "turns": _aggregate_binary(turns),
            "final_turn_accuracy": (
                sum(bool(row["turns"][-1]["correct"]) for row in records) / len(records)
                if records
                else None
            ),
            "all_turns_correct_rate": (
                sum(all(turn["correct"] for turn in row["turns"]) for row in records) / len(records)
                if records
                else None
            ),
        }
    for capability in sorted({turn["capability"] for turn in turn_rows}):
        result["by_capability"][capability] = {
            condition: _aggregate_binary(
                [
                    turn
                    for turn in turn_rows
                    if turn["condition"] == condition and turn["capability"] == capability
                ]
            )
            for condition in conditions
        }
    reveal = [
        row
        for row in rows
        if row.get("stream_kind") == "evidence_reveal"
        and not row.get("error")
        and len(row.get("turns", [])) == 2
    ]
    result["evidence_update_success"] = {
        condition: (
            sum(
                all(turn["correct"] for turn in row["turns"])
                for row in reveal
                if row["condition"] == condition
            )
            / sum(row["condition"] == condition for row in reveal)
            if any(row["condition"] == condition for row in reveal)
            else None
        )
        for condition in conditions
    }
    return result


def _flush_predictions(path: Path, done: dict[tuple[str, str], dict[str, Any]]) -> None:
    _write_jsonl(
        path,
        [done[key] for key in sorted(done)],
    )


def evaluate_qa_dataset(
    *,
    dataset_root: Path,
    model_path: Path,
    limit_raw: int | None = None,
    limit_streaming: int | None = None,
) -> dict[str, Any]:
    """Evaluate all records in both visual and vision-free conditions, resumably."""

    dataset_root = dataset_root.resolve()
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    source_root = Path(manifest["source_root"])
    raw_inputs = read_jsonl(dataset_root / "raw_eval_inputs.jsonl")
    raw_oracles = {
        row["record_id"]: row for row in read_jsonl(dataset_root / "raw_eval_oracle.jsonl")
    }
    stream_inputs = read_jsonl(dataset_root / "streaming_eval_inputs.jsonl")
    stream_oracles = {
        row["record_id"]: row for row in read_jsonl(dataset_root / "streaming_eval_oracle.jsonl")
    }
    if limit_raw is not None:
        raw_inputs = raw_inputs[:limit_raw]
    if limit_streaming is not None:
        stream_inputs = stream_inputs[:limit_streaming]
    predictions_path = dataset_root / "sensenova_predictions.jsonl"
    done = _load_done(predictions_path)
    wanted: list[tuple[str, dict[str, Any], str]] = []
    for item in raw_inputs:
        for condition in ("multimodal", "vision_free"):
            wanted.append(("raw", item, condition))
    for item in stream_inputs:
        for condition in (
            "multimodal_teacher_forced",
            "multimodal_free_running",
            "vision_free_teacher_forced",
            "vision_free_free_running",
        ):
            wanted.append(("streaming", item, condition))
    pending = []
    for sample_type, item, condition in wanted:
        key = (item["record_id"], condition)
        prediction_sha = _prediction_sha(item["input_sha256"], condition, model_path)
        if key in done and done[key].get("prediction_sha256") == prediction_sha:
            continue
        pending.append((sample_type, item, condition, prediction_sha))
    evaluator = SenseNovaEvaluator(model_path) if pending else None
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    append_stream = predictions_path.open("a", encoding="utf-8") if pending else None
    try:
        for sample_type, item, condition, prediction_sha in pending:
            started = time.monotonic()
            base = {
                "schema_version": EVALUATION_SCHEMA,
                "model_id": model_path.name,
                "model_path": str(model_path.resolve()),
                "prompt_version": PROMPT_VERSION,
                "record_id": item["record_id"],
                "input_sha256": item["input_sha256"],
                "prediction_sha256": prediction_sha,
                "sample_type": sample_type,
                "condition": condition,
                "batch_id": item["batch_id"],
                "tier": item["tier"],
            }
            try:
                assert evaluator is not None
                if sample_type == "raw":
                    oracle = raw_oracles[item["record_id"]]
                    response = evaluator.raw(
                        item,
                        source_root,
                        vision=condition == "multimodal",
                    )
                    ground_truth = oracle["answer"]["label"]
                    prediction = parse_answer_label(
                        response, item["choices"], question=item["question"]
                    )
                    row = {
                        **base,
                        "capability": item["capability"],
                        "role": item["role"],
                        "variant": item["variant"],
                        "cluster_ids": item["cluster_ids"],
                        "ground_truth": ground_truth,
                        "prediction": prediction,
                        "correct": prediction == ground_truth,
                        "parseable": prediction is not None,
                        "format_compliant": answer_format_compliant(response, item["choices"]),
                        "response": response,
                    }
                else:
                    oracle = stream_oracles[item["record_id"]]
                    row = {
                        **base,
                        "stream_kind": item["stream_kind"],
                        "capabilities": item["capabilities"],
                        "cluster_ids": item["cluster_ids"],
                        "turns": evaluator.streaming(
                            item,
                            oracle,
                            source_root,
                            vision=condition.startswith("multimodal"),
                            teacher_forced=condition.endswith("teacher_forced"),
                        ),
                    }
                row["elapsed_seconds"] = round(time.monotonic() - started, 3)
            except (RuntimeError, ValueError, OSError) as error:
                row = {
                    **base,
                    "error": f"{type(error).__name__}: {error}",
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            done[(item["record_id"], condition)] = row
            assert append_stream is not None
            append_stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            append_stream.flush()
    finally:
        if append_stream is not None:
            append_stream.close()
    selected_keys = {(item["record_id"], condition) for _, item, condition in wanted}
    selected_done = {key: row for key, row in done.items() if key in selected_keys}
    _flush_predictions(predictions_path, selected_done)
    selected = list(selected_done.values())
    raw_rows = [row for row in selected if row["sample_type"] == "raw"]
    stream_rows = [row for row in selected if row["sample_type"] == "streaming"]
    errors = [row for row in selected if row.get("error")]
    summary = {
        "schema_version": EVALUATION_SCHEMA,
        "model_id": model_path.name,
        "model_path": str(model_path.resolve()),
        "prompt_version": PROMPT_VERSION,
        "load_seconds": evaluator.load_seconds if evaluator else None,
        "prediction_record_count": len(selected),
        "error_count": len(errors),
        "errors": [
            {
                "record_id": row["record_id"],
                "condition": row["condition"],
                "error": row["error"],
            }
            for row in errors
        ],
        "raw": _raw_summary(raw_rows),
        "streaming": _stream_summary(stream_rows),
    }
    _write_json(dataset_root / "sensenova_summary.json", summary)
    return summary
