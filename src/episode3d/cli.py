"""Command-line entry point for the EpiSpace data engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from episode3d.pipeline import build_dataset


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Compile strict simulator trajectories into EpiSpace training and benchmark data."
    )
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--latex-dir", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    manifest = build_dataset(
        args.config,
        output_dir=args.output_dir,
        latex_dir=args.latex_dir,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "dataset_id": manifest["dataset_id"],
                "statistics": manifest["statistics"]["exports"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
