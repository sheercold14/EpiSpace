#!/usr/bin/env python3
"""Build a compact standalone browser for Transform Pilot training records."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def target(row: dict[str, Any], arm: str) -> str:
    answer = row["answer"]["surface"]
    if arm == "answer_only":
        return f"<answer>{answer}</answer>"
    trace = row["grounded_trace"]
    return (
        f"<cue>{trace['cue']['surface']}</cue>\n"
        f"<transform>{trace['transform']['surface']}</transform>\n"
        f"<answer>{answer}</answer>"
    )


def select_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    self_rows = [row for row in rows if row["task_id"] == "self_rotation_query.v1"]
    seen = set()
    for row in self_rows:
        key = (row["task_tags"][-1], row["answer"]["label"])
        if key not in seen and row["split"] == "train" and row["surface_variant_index"] == 0:
            selected.append(row)
            seen.add(key)
    among = [row for row in rows if row["task_id"].startswith("among5_")]
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in among:
        by_family[row["family_id"]].append(row)
    complete = [
        family
        for family, values in by_family.items()
        if {row["variant_id"] for row in values}
        == {"identity", "rotate90", "mirror", "permute1", "radius185"}
    ]
    preferred = next(
        (family for family in complete if family.startswith("hall_conference_large")),
        complete[0] if complete else None,
    )
    if preferred:
        family_rows = by_family[preferred]
        identity_rows = [
            row
            for row in family_rows
            if row["variant_id"] == "identity"
            and row["task_id"] == "among5_cross_view_relation.v1"
        ]
        chosen_queries: list[int] = []
        for labels in ({"left", "right"}, {"front", "back"}):
            match = next(
                (
                    int(row["query_index"])
                    for row in identity_rows
                    if row["answer"]["label"] in labels
                ),
                None,
            )
            if match is not None:
                chosen_queries.append(match)
        selected.extend(
            row for row in family_rows if int(row["query_index"]) in chosen_queries
        )
    return selected


def load_canonical_predictions(
    dataset_root: Path,
    row_count: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Index complete full-corpus model outputs by record, model, and prompt arm."""
    indexed: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    evaluation_dir = dataset_root / "evaluation"
    if not evaluation_dir.is_dir():
        return indexed
    for summary_path in sorted(evaluation_dir.glob("*.all.predictions.summary.json")):
        summary = read_json(summary_path)
        if (
            summary.get("evaluation_scope") != "all"
            or int(summary.get("evaluated", -1)) != row_count
        ):
            continue
        predictions_path = summary_path.with_name(
            summary_path.name.removesuffix(".summary.json") + ".jsonl"
        )
        if not predictions_path.is_file():
            continue
        model_id = Path(str(summary["model_path"])).name
        prompt_mode = str(summary["prompt_mode"])
        for prediction in read_jsonl(predictions_path):
            record_id = str(prediction["record_id"])
            if prompt_mode in indexed[record_id][model_id]:
                raise ValueError(
                    f"duplicate model output: {record_id}/{model_id}/{prompt_mode}"
                )
            indexed[record_id][model_id][prompt_mode] = {
                "response": prediction.get("response", ""),
                "prediction": prediction.get("prediction"),
                "prediction_semantic_label": prediction.get(
                    "prediction_semantic_label"
                ),
                "ground_truth": prediction.get("ground_truth"),
                "ground_truth_label": prediction.get("ground_truth_label"),
                "correct": prediction.get("correct"),
                "semantic_correct": prediction.get("semantic_correct"),
                "trace_format": prediction.get("trace_format", {}),
            }
    return indexed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "transform_pilot_v1" / "dataset",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "web" / "data" / "transform_pilot.v1.json"
    )
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    rows = read_jsonl(dataset_root / "teacher_records.jsonl")
    report = read_json(dataset_root / "corpus_report.json")
    selected = select_records(rows)
    model_outputs = load_canonical_predictions(dataset_root, len(rows))
    media_root = args.output.parent / "transform_pilot_media"
    public_rows = []
    for row in selected:
        copied = []
        for index, relative in enumerate(row["model_input"]["images"]):
            source = dataset_root / relative
            destination = media_root / row["record_id"] / f"view-{index + 1:02d}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
                shutil.copy2(source, destination)
            copied.append(
                str(destination.relative_to(args.output.parent.parent)).replace("\\", "/")
            )
        public_rows.append(
            {
                "record_id": row["record_id"],
                "task_id": row["task_id"],
                "task_tags": row["task_tags"],
                "scene_id": row["scene_id"],
                "family_id": row["family_id"],
                "split": row["split"],
                "variant_id": row.get("variant_id"),
                "query_index": row.get("query_index"),
                "base_fact_id": row["base_fact_id"],
                "model_input": {**row["model_input"], "images": copied},
                "answer": row["answer"],
                "targets": {
                    "answer_only": target(row, "answer_only"),
                    "grounded_cot": target(row, "grounded_cot"),
                },
                "grounded_trace": row["grounded_trace"],
                "operation_graph": row["operation_graph"],
                "certificate": row["certificate"],
                "model_outputs": model_outputs.get(row["record_id"], {}),
            }
        )
    evaluation = {}
    evaluation_dir = dataset_root / "evaluation"
    if evaluation_dir.is_dir():
        for path in sorted(evaluation_dir.glob("*.summary.json")):
            summary = read_json(path)
            scope = summary.get("evaluation_scope")
            expected = (
                len(rows)
                if scope == "all"
                else sum(report["counts"]["balanced_by_split"].values())
                if scope == "balanced_all"
                else None
            )
            if expected is None or int(summary.get("evaluated", -1)) != expected:
                continue
            model_id = Path(str(summary["model_path"])).name
            summary["model_id"] = model_id
            key = f"{model_id}.{summary['prompt_mode']}.{scope}"
            if key in evaluation:
                raise ValueError(f"duplicate canonical evaluation summary: {key}")
            evaluation[key] = summary
    payload = {
        "schema_version": "epispace.transform_web.v2",
        "title": "EpiSpace Transform Pilot",
        "report": report,
        "showcase_records": public_rows,
        "catalog": {
            "record_count": len(rows),
            "task_counts": dict(Counter(row["task_id"] for row in rows)),
            "split_counts": dict(Counter(row["split"] for row in rows)),
            "complete_among_families": report["counts"].get("complete_among5_families", 0),
        },
        "evaluation": evaluation,
        "artifacts": {
            "teacher": str(dataset_root / "teacher_records.jsonl"),
            "balanced_sft": str(dataset_root / "balanced_isolated_grounded_cot_train.jsonl"),
            "episode_sft": str(dataset_root / "episode_grounded_cot_train.jsonl"),
            "eval_inputs_all": str(dataset_root / "eval_inputs_all.jsonl"),
            "eval_oracle_all": str(dataset_root / "eval_oracle_all.jsonl"),
            "eval_inputs_balanced_all": str(
                dataset_root / "eval_inputs_balanced_all.jsonl"
            ),
            "eval_oracle_balanced_all": str(
                dataset_root / "eval_oracle_balanced_all.jsonl"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"output={args.output.resolve()} records={len(rows)} showcase={len(public_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
