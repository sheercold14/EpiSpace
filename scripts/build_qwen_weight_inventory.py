#!/usr/bin/env python3
"""Create a reusable inventory for weights and model runtime configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from episode3d.pilot_summary import write_inventory_artifact  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=("model", "directory"),
        required=True,
        help=(
            "model includes weights, tokenizer, chat template, processor and "
            "generation configuration; directory inventories every file"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    binding = write_inventory_artifact(
        args.root,
        args.output,
        kind=args.kind,
        overwrite=args.overwrite,
    )
    print(json.dumps(binding, ensure_ascii=False))


if __name__ == "__main__":
    main()
