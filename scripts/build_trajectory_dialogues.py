#!/usr/bin/env python3
"""Compile passed controlled trajectories into deterministic dialogue SFT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from episode3d.trajectory_llm_pipeline import compile_catalog


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=("T3", "T4", "T7", "T8", "T10"),
        default=("T3", "T4", "T7", "T8", "T10"),
    )
    args = parser.parse_args()
    report = compile_catalog(
        catalog_path=args.catalog,
        code_root=args.code_root,
        output_dir=args.output_dir,
        limit_per_class=args.limit_per_class,
        classes=tuple(args.classes),
    )
    print("requested", json.dumps(report["requested"], ensure_ascii=False))
    print("compiled", json.dumps(report["compiled_counts"], ensure_ascii=False))
    print("rejected", json.dumps(report["rejected_counts"], ensure_ascii=False))
    for row in report["rejected"]:
        print(f"  rejected {row['acquisition_id']}: {row['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
