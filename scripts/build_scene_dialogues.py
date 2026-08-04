#!/usr/bin/env python3
"""Compile scene-generic dialogue episodes from a sweep release index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from episode3d.scene_llm_pipeline import compile_sweep


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-index", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    report = compile_sweep(
        release_index=args.release_index.resolve(),
        sweep_root=args.sweep_root.resolve(),
        output_dir=args.output_dir.resolve(),
        limit=args.limit,
    )
    compiled = report["compiled"]
    print(f"compiled={len(compiled)} rejected={len(report['rejected'])}")
    print("direction_balance:", json.dumps(report["direction_balance"], ensure_ascii=False))
    for name, reason in report["rejected"].items():
        print(f"  rejected {name}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
