#!/usr/bin/env python3
"""Build strict T1-source -> T10-heldout perspective episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from episode3d.trajectory_llm_pipeline.composition import compile_composition_catalog


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = compile_composition_catalog(
        args.catalog.resolve(), args.code_root.resolve(), args.output_dir.resolve()
    )
    print(
        json.dumps(
            {
                "requested": report["requested"],
                "compiled": len(report["compiled"]),
                "rejected": len(report["rejected"]),
            },
            ensure_ascii=False,
        )
    )
    for row in report["rejected"]:
        print(f"  rejected {row['acquisition_id']}: {row['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
