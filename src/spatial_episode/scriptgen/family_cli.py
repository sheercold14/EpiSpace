"""One-command family builder.

From one rendered bundle to a complete, audited family site:

    .venv/bin/python -m spatial_episode.scriptgen.family_cli \
        --bundle <render dir> --plan-record <plan record json> \
        --scene-ir <scene_ir.json> --out <output dir>

Output: ``<out>/family.json`` (canonical + 4 variants + certificates, one
document), ``<out>/media/`` (per-frame rgb/depth/instance PNGs) and
``<out>/index.html`` (the family review page; serve with any static server).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .family import ScriptgenFamilyV1, build_family_site
from .standards import STD_V1


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Build a family (canonical + 4 recompiled variants) from a rendered bundle."
    )
    result.add_argument("--bundle", type=Path, required=True, help="Rendered bundle directory.")
    result.add_argument(
        "--plan-record", type=Path, required=True, help="Scriptgen plan record JSON."
    )
    result.add_argument(
        "--scene-ir", type=Path, required=True, help="scene_ir.json geometry truth."
    )
    result.add_argument("--out", type=Path, required=True, help="Output directory.")
    result.add_argument("--seed", type=int, default=17, help="Variant sampling seed.")
    result.add_argument(
        "--drop-count", type=int, default=2, help="Frames removed by drop_filler."
    )
    result.add_argument(
        "--delay-extra", type=int, default=4, help="Pause frames inserted by delay."
    )
    return result


def main() -> int:
    args = parser().parse_args()
    family_path = build_family_site(
        args.bundle,
        args.plan_record,
        args.scene_ir,
        args.out,
        STD_V1,
        seed=args.seed,
        drop_count=args.drop_count,
        delay_extra=args.delay_extra,
    )
    doc = ScriptgenFamilyV1.model_validate_json(family_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "family_id": doc.family_id,
                "capability": doc.capability,
                "standard": doc.standard_version,
                "episodes": {e.kind: e.gold for e in doc.episodes},
                "checks": doc.checks.model_dump(),
                "family": str(family_path),
                "review": str(family_path.parent / "index.html"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
