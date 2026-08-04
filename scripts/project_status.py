#!/usr/bin/env python3
"""Render the current project snapshot from copied authoritative manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "manifests"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def percent(value: Any) -> str:
    return "—" if value is None else f"{100 * float(value):.2f}%"


def render() -> str:
    release = read_json(MANIFESTS / "datasets" / "epispace_pilot_v1.release_manifest.json")
    transform = read_json(MANIFESTS / "datasets" / "transform_pilot_v1.corpus_report.json")
    dialogues = read_json(MANIFESTS / "datasets" / "scene_dialogues_v1.corpus_report.json")

    acquisition = release["statistics"]["acquisition"]
    exports = release["statistics"]["exports"]
    counts = transform["counts"]

    evaluation_rows = []
    for path in sorted((MANIFESTS / "experiments").glob("*.all.predictions.summary.json")):
        summary = read_json(path)
        if summary.get("evaluation_scope") != "all" or summary.get("evaluated") != 2961:
            continue
        model = "SenseNova 1.5" if "1.5" in str(summary.get("model_path")) else "SenseNova 1.1"
        prompt = "Grounded-CoT" if summary.get("prompt_mode") == "grounded_cot" else "Answer-only"
        evaluation_rows.append(
            f"| {model} | {prompt} | {percent(summary.get('accuracy'))} | "
            f"{percent(summary.get('trace_format_rate'))} |"
        )

    lines = [
        "# Project Status",
        "",
        "> Generated from the copied manifests by `python scripts/project_status.py --write`.",
        "> It summarizes completed artifacts; it does not imply an episode-training gain.",
        "",
        "## Data construction",
        "",
        "| Asset | Current snapshot |",
        "|---|---:|",
        f"| Planned simulator jobs | {acquisition['planned_jobs']} |",
        f"| Strict simulator bundles | {acquisition['strict_bundles_loaded']} |",
        f"| Main episode SFT records | {exports['episode_sft_records']} |",
        f"| Main isolated SFT records | {exports['isolated_sft_records']} |",
        f"| Main benchmark records | {exports['benchmark_records']} |",
        f"| Scene dialogue episodes | {len(dialogues['compiled'])} |",
        f"| Transform teacher records | {counts['teacher_records']} |",
        f"| Transform geometric facts | {counts['unique_base_facts']} |",
        f"| Complete Among-5 families | {counts['complete_among5_families']} |",
        "",
        "## Frozen zero-shot evaluation",
        "",
        "| Model | Prompt | Accuracy | Complete trace |",
        "|---|---|---:|---:|",
        *evaluation_rows,
        "",
        "## Interpretation",
        "",
        "- SenseNova 1.5 is substantially more accurate than 1.1 on the Transform Pilot.",
        "- SenseNova 1.1 Grounded-CoT can produce the requested format but often changes a correct answer.",
        "- SenseNova 1.5 ignores the cue/transform contract, so its prompt gain is not evidence of explicit trace reasoning.",
        "- The paired episode-vs-isolated SFT experiment is still required before claiming episode-learning gains.",
        "",
        "## Immediate engineering work",
        "",
        "1. Replace copied legacy sibling paths with a workspace resolver.",
        "2. Add a self-contained golden bundle for bundle-to-episode replay.",
        "3. Freeze one paired training protocol and run three random seeds.",
        "4. Generate paper tables directly from versioned summary JSON.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write STATUS.md instead of stdout")
    args = parser.parse_args()
    content = render()
    if args.write:
        output = ROOT / "STATUS.md"
        output.write_text(content, encoding="utf-8")
        print(output)
    else:
        print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

