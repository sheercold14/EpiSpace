#!/usr/bin/env python3
"""Build paired Answer-only / grounded-CoT Transform Pilot exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from episode3d.transform_pilot.dataset import build_transform_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "transform_dataset_v1.json",
    )
    args = parser.parse_args()
    report = build_transform_dataset(args.config)
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    print(f"status={report['status']} audit={report['audit']['status']}")
    return 0 if report["status"] == "data_ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
