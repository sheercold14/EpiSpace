#!/usr/bin/env python3
"""Build the verified Rs_int incremental-dialogue pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from episode3d.rsint_llm_pipeline.pipeline import build_rsint_dialogue  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/rsint_llm_pipeline_v2.json",
        help="pipeline configuration JSON",
    )
    args = parser.parse_args()
    print(json.dumps(build_rsint_dialogue(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
