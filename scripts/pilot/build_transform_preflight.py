#!/usr/bin/env python3
"""Build the geometry-only EpiSpace Transform Pilot census."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from episode3d.transform_pilot import build_preflight

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "transform_pilot_v1.json",
    )
    args = parser.parse_args()
    summary = build_preflight(args.config)
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    print(f"status={summary['status']}")


if __name__ == "__main__":
    main()
