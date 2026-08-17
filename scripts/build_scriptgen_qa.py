#!/usr/bin/env python3
"""Build, evaluate, and review the Scriptgen raw/streaming QA release."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--source", type=Path, required=True)
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--batch-size", type=int, default=100)
    generate.add_argument(
        "--quarantine",
        type=Path,
        help="JSONL audit rows whose episode_id values are excluded from this QA release",
    )
    generate.add_argument(
        "--exclusions",
        type=Path,
        help=(
            "JSONL rows with episode_id and capability; excludes only that question "
            "while retaining other capabilities on the same rendered episode"
        ),
    )

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--dataset-root", type=Path, required=True)
    evaluate.add_argument("--model-path", type=Path, required=True)
    evaluate.add_argument("--limit-raw", type=int)
    evaluate.add_argument("--limit-streaming", type=int)

    report = subparsers.add_parser("report")
    report.add_argument("--dataset-root", type=Path, required=True)

    all_command = subparsers.add_parser("all")
    all_command.add_argument("--source", type=Path, required=True)
    all_command.add_argument("--out", type=Path, required=True)
    all_command.add_argument("--model-path", type=Path, required=True)
    all_command.add_argument("--model-python", type=Path, required=True)
    all_command.add_argument("--gpu-ids", default="1,2")
    all_command.add_argument("--batch-size", type=int, default=100)
    all_command.add_argument("--limit-raw", type=int)
    all_command.add_argument("--limit-streaming", type=int)
    return parser


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _quarantined_episode_ids(path: Path | None) -> frozenset[str]:
    if path is None:
        return frozenset()
    result: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        episode_id = row.get("episode_id") if isinstance(row, dict) else None
        if not isinstance(episode_id, str):
            raise ValueError(f"{path}:{line_number}: missing string episode_id")
        result.add(episode_id)
    return frozenset(result)


def _episode_capability_exclusions(path: Path | None) -> dict[str, frozenset[str]]:
    if path is None:
        return {}
    result: dict[str, set[str]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        episode_id = row.get("episode_id") if isinstance(row, dict) else None
        capability = row.get("capability") if isinstance(row, dict) else None
        if not isinstance(episode_id, str) or not isinstance(capability, str):
            raise ValueError(
                f"{path}:{line_number}: expected string episode_id and capability"
            )
        result.setdefault(episode_id, set()).add(capability)
    return {episode_id: frozenset(values) for episode_id, values in result.items()}


def main() -> int:
    args = _parser().parse_args()
    if args.command == "generate":
        from spatial_episode.scriptgen.qa_dataset import build_qa_dataset

        _print(
            build_qa_dataset(
                source_root=args.source,
                output_root=args.out,
                batch_size=args.batch_size,
                excluded_episode_ids=_quarantined_episode_ids(args.quarantine),
                excluded_episode_capabilities=_episode_capability_exclusions(args.exclusions),
            )
        )
        return 0
    if args.command == "evaluate":
        from spatial_episode.scriptgen.qa_evaluation import evaluate_qa_dataset

        _print(
            evaluate_qa_dataset(
                dataset_root=args.dataset_root,
                model_path=args.model_path,
                limit_raw=args.limit_raw,
                limit_streaming=args.limit_streaming,
            )
        )
        return 0
    if args.command == "report":
        from spatial_episode.scriptgen.qa_reporting import build_research_review

        review = build_research_review(dataset_root=args.dataset_root)
        _print(
            {
                "schema_version": review["schema_version"],
                "case_counts": review["case_counts"],
                "record_count": len(review["rows"]),
                "source_integrity": review["source_integrity"],
            }
        )
        return 0

    from spatial_episode.scriptgen.qa_dataset import build_qa_dataset

    manifest = build_qa_dataset(
        source_root=args.source,
        output_root=args.out,
        batch_size=args.batch_size,
    )
    project_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    environment["PYTHONUNBUFFERED"] = "1"
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(project_root / "src") + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    command = [
        str(args.model_python),
        str(Path(__file__).resolve()),
        "evaluate",
        "--dataset-root",
        str(args.out.resolve()),
        "--model-path",
        str(args.model_path.resolve()),
    ]
    if args.limit_raw is not None:
        command.extend(("--limit-raw", str(args.limit_raw)))
    if args.limit_streaming is not None:
        command.extend(("--limit-streaming", str(args.limit_streaming)))
    subprocess.run(command, check=True, env=environment, cwd=project_root)
    from spatial_episode.scriptgen.qa_reporting import build_research_review

    review = build_research_review(dataset_root=args.out)
    _print(
        {
            "manifest": manifest,
            "case_counts": review["case_counts"],
            "source_integrity": review["source_integrity"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
