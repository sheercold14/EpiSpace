"""Command line entry point for the scriptgen engine.

Example (render-free, uses the built-in demo layout):

    python -m spatial_episode.scriptgen.cli \
        --capability self_motion_update --seed 17 --out plans.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .demo import DEMO_LAYOUT
from .generate import generate_plans
from .library import SCRIPT_LIBRARY
from .standards import STD_V1


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Script-driven trajectory generation (render-free search phase)."
    )
    result.add_argument(
        "--capability",
        required=True,
        choices=sorted(SCRIPT_LIBRARY),
        help="Capability key in the script library.",
    )
    result.add_argument("--seed", type=int, default=17, help="Deterministic sampling seed.")
    result.add_argument("--out", type=Path, default=Path("plans.json"), help="Output JSON path.")
    result.add_argument(
        "--attempts", type=int, default=40, help="Candidate attempts per slot binding."
    )
    result.add_argument("--max-plans", type=int, default=None, help="Stop after this many plans.")
    return result


def main() -> int:
    args = parser().parse_args()
    report = generate_plans(
        DEMO_LAYOUT,
        SCRIPT_LIBRARY[args.capability],
        STD_V1,
        seed=args.seed,
        attempts_per_binding=args.attempts,
        max_plans=args.max_plans,
    )
    args.out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "scene": report.scene_id,
                "capability": report.capability,
                "standard": report.standard_version,
                "plans": len(report.plans),
                "rejections": report.rejection_counts,
                "slot_rejections": report.slot_rejections,
                "out": str(args.out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
