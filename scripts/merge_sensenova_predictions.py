#!/usr/bin/env python3
"""Merge resumable SenseNova ranges and rebuild one exact-coverage summary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row is not an object")
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--evaluation-scope",
        choices=("test", "full_test", "balanced_all", "all"),
        required=True,
    )
    parser.add_argument(
        "--prompt-mode", choices=("answer_only", "grounded_cot"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    expected_rows = read_jsonl(
        args.dataset_root / f"eval_inputs_{args.evaluation_scope}.jsonl"
    )
    expected_ids = [str(row["record_id"]) for row in expected_rows]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("evaluation input contains duplicate record IDs")
    expected = set(expected_ids)
    merged: dict[str, dict[str, Any]] = {}
    for path in args.inputs:
        for row in read_jsonl(path):
            record_id = str(row.get("record_id", ""))
            if record_id not in expected:
                raise ValueError(f"{path}: unexpected record_id {record_id!r}")
            if row.get("prompt_mode") != args.prompt_mode:
                raise ValueError(f"{path}: prompt mode mismatch for {record_id}")
            previous = merged.get(record_id)
            if previous is not None and previous.get("response") != row.get("response"):
                raise ValueError(f"conflicting duplicate prediction for {record_id}")
            merged[record_id] = row
    missing = [record_id for record_id in expected_ids if record_id not in merged]
    if missing:
        raise ValueError(
            f"prediction coverage incomplete: {len(missing)} missing; first={missing[:5]}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".merge.tmp")
    temporary.write_text(
        "".join(
            json.dumps(merged[record_id], ensure_ascii=False) + "\n"
            for record_id in sorted(expected_ids)
        ),
        encoding="utf-8",
    )
    temporary.replace(args.output)

    # No model is loaded here: exact coverage makes the evaluator take its
    # deterministic rescore-and-summarize path only.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from episode3d.transform_pilot.evaluation import evaluate_sensenova

    summary = evaluate_sensenova(
        dataset_root=args.dataset_root,
        model_path=args.model_path,
        output_path=args.output,
        prompt_mode=args.prompt_mode,
        evaluation_scope=args.evaluation_scope,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
