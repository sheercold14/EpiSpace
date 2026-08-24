#!/usr/bin/env python
"""Report how much of each capability's supply comes from its top few scenes.

A capability that only two rooms can produce trains a model on those rooms, not
on the relation the question asks about, and that is the most direct route from
this pipeline to polluted training data.  The per-scene binding cap is the lever
that controls it, so this evaluates candidate caps side by side: for each one,
how many scenes contribute, and what share the largest two still hold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CANDIDATE_CAPS = (8, 16, 32, 64)

# Above this share the capability is effectively a two-room dataset.
CONCENTRATION_ALARM = 0.5


def _rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else payload["scenes"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yield-report", type=Path, required=True)
    parser.add_argument("--top", type=int, default=2)
    args = parser.parse_args()

    rows = _rows(args.yield_report)
    capabilities = sorted({key for row in rows for key in row["counts"]})
    print(f"{len(rows)} scenes screened, top-{args.top} share per cap\n")
    header = "  ".join(f"cap{cap:<3}scenes  share" for cap in CANDIDATE_CAPS)
    print(f"{'capability':<32} {header}")
    for capability in capabilities:
        supply = sorted(
            ((row["counts"][capability], row["scene"]) for row in rows), reverse=True
        )
        cells = []
        for cap in CANDIDATE_CAPS:
            capped = [min(count, cap) for count, _ in supply if count > 0]
            total = sum(capped)
            share = sum(capped[: args.top]) / total if total else 0.0
            flag = "!" if share > CONCENTRATION_ALARM else " "
            cells.append(f"{total:>5} {len(capped):>4}  {share:>5.1%}{flag}")
        print(f"{capability:<32} " + "  ".join(cells))

    print(f"\n! = top {args.top} scenes hold over {CONCENTRATION_ALARM:.0%} of the supply")
    print("\ntop scenes per capability:")
    for capability in capabilities:
        supply = sorted(
            ((row["counts"][capability], row["scene"]) for row in rows), reverse=True
        )
        named = ", ".join(f"{scene}={count}" for count, scene in supply[:3] if count)
        print(f"  {capability:<32} {named or '(no yield)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
