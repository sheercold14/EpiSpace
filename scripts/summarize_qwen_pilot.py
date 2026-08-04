#!/usr/bin/env python3
"""Validate and summarize the paired EpiSpace Qwen3-VL-4B pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from episode3d.pilot_summary import (  # noqa: E402
    PilotLayout,
    PilotSummaryError,
    summarize_pilot,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "epispace_qwen3vl4b",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=PROJECT_ROOT / "data" / "epispace_pilot_v1" / "benchmark.core.jsonl",
    )
    parser.add_argument(
        "--composition-benchmark",
        type=Path,
        default=PROJECT_ROOT / "data" / "epispace_pilot_v1" / "benchmark.composition.jsonl",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/data/shichao/data/dataV100/models/viewfusion/base_Qwen3-VL-4B-Instruct"),
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write a partial report for absent runs; integrity mismatches still fail",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = PilotLayout.defaults(
        args.experiment_root,
        args.benchmark,
        args.model,
        args.composition_benchmark,
    )
    output_json = args.output_json or layout.experiment_root / "pilot_results.json"
    output_markdown = args.output_markdown or layout.experiment_root / "pilot_results.md"
    try:
        summary = summarize_pilot(layout, allow_incomplete=args.allow_incomplete)
        write_summary(summary, output_json, output_markdown)
    except PilotSummaryError as exc:
        raise SystemExit(f"pilot summary failed closed: {exc}") from exc
    print(
        json.dumps(
            {
                "status": summary["status"],
                "complete_runs": summary["complete_runs"],
                "missing_runs": summary["missing_runs"],
                "output_json": str(output_json.resolve()),
                "output_markdown": str(output_markdown.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
