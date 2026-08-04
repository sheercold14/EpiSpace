#!/usr/bin/env python3
"""Build deterministic compute-matched episode-vs-isolated schedules."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from episode3d.compute_matching import (  # noqa: E402
    SUPPORTED_REGIMES,
    build_compute_matched_schedules,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Build fact-matched and image-occurrence-matched training schedules "
            "from paired EpiSpace SFT JSONL files."
        )
    )
    result.add_argument("--episode", type=Path, required=True, help="episode SFT JSONL")
    result.add_argument("--isolated", type=Path, required=True, help="isolated SFT JSONL")
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument(
        "--regime",
        action="append",
        choices=SUPPORTED_REGIMES,
        dest="regimes",
        help="Regime to build; repeat this option. Defaults to both regimes.",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    manifest = build_compute_matched_schedules(
        args.episode,
        args.isolated,
        args.output_dir,
        seed=args.seed,
        regimes=args.regimes or SUPPORTED_REGIMES,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "seed": manifest["seed"],
                "comparison_count": manifest["pairing"]["comparison_count"],
                "regimes": {
                    name: {
                        "status": value["status"],
                        "episode_draws": value["arms"]["episode"]["schedule_draw_count"],
                        "isolated_draws": value["arms"]["isolated"]["schedule_draw_count"],
                    }
                    for name, value in manifest["regimes"].items()
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
