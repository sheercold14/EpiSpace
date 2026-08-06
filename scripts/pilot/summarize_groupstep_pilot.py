#!/usr/bin/env python3
"""Validate and report an explicit multi-seed EpiSpace group-step pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from episode3d.groupstep_pilot_summary import (  # noqa: E402
    GroupStepPilotLayout,
    GroupStepSummaryError,
    summarize_groupstep_pilot,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit JSON config containing every seed pair and artifact path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Defaults to groupstep_pilot_results.json beside the config.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        help="Defaults to groupstep_pilot_results.md beside the config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        layout = GroupStepPilotLayout.from_config(args.config)
        summary = summarize_groupstep_pilot(layout)
        output_json = (
            args.output_json.resolve()
            if args.output_json
            else layout.config_path.with_name("groupstep_pilot_results.json")
        )
        output_markdown = (
            args.output_markdown.resolve()
            if args.output_markdown
            else layout.config_path.with_name("groupstep_pilot_results.md")
        )
        write_summary(summary, output_json, output_markdown)
    except GroupStepSummaryError as exc:
        raise SystemExit(f"group-step pilot summary failed closed: {exc}") from exc
    print(
        json.dumps(
            {
                "artifact_status": summary["artifact_status"]["status"],
                "causal_validity": summary["causal_validity"]["status"],
                "seeds": summary["contract"]["seeds"],
                "output_json": str(output_json),
                "output_markdown": str(output_markdown),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
