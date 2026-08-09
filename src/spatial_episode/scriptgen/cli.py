"""Command line entry point for the scriptgen engine.

Examples (both render-free):

    python -m spatial_episode.scriptgen.cli \
        --capability self_motion_update --seed 17 --out plans.json

    python -m spatial_episode.scriptgen.cli \
        --scene-ir /path/to/scene_ir.json \
        --capability self_motion_update --out real_scene_plans.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .behavior import layout_from_scene_ir
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
    result.add_argument(
        "--scene-ir",
        type=Path,
        help="Real scene_ir.json input; omit to use the built-in demo layout.",
    )
    result.add_argument("--seed", type=int, default=17, help="Deterministic sampling seed.")
    result.add_argument("--out", type=Path, default=Path("plans.json"), help="Output JSON path.")
    result.add_argument(
        "--attempts", type=int, default=150, help="Total candidate attempts per slot binding."
    )
    result.add_argument(
        "--plans-per-binding",
        type=int,
        default=10,
        help="Successful plans retained for each object-slot binding.",
    )
    result.add_argument("--max-plans", type=int, default=None, help="Stop after this many plans.")
    return result


def main() -> int:
    args = parser().parse_args()
    layout = DEMO_LAYOUT if args.scene_ir is None else layout_from_scene_ir(args.scene_ir)
    report = generate_plans(
        layout,
        SCRIPT_LIBRARY[args.capability],
        STD_V1,
        seed=args.seed,
        attempts_per_binding=args.attempts,
        plans_per_binding=args.plans_per_binding,
        max_plans=args.max_plans,
    )
    args.out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "scene": report.scene_id,
                "scene_ir": str(args.scene_ir) if args.scene_ir is not None else None,
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
