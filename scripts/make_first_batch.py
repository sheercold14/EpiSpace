#!/usr/bin/env python
"""Choose which cells the first render batch covers.

The plan enumerates cells scene by scene, so rendering a prefix of it renders
the first few rooms exhaustively and leaves the rest untouched.  That is the
wrong shape for a batch whose job is to be reviewed before the rest is
committed: a reviewer looking at three rooms cannot tell whether the
trajectories are reasonable anywhere else, and a partial dataset skewed to
three rooms is exactly the concentration the per-scene cap exists to avoid.

This draws the batch by round-robin over scenes within each capability, so a
batch of any size spreads over every scene that supplies that capability and
the scene mix of the batch matches the scene mix of the finished collection.

Only producer cells are listed.  The deferred cells - the fifteen imagined
yaws, and the anchor and closer of each trio - are credited from the producer's
rendered episode, so listing them would ask for renders that already exist.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def _is_producer(capability: str) -> bool:
    """True for the capability whose render the rest of its group is credited from."""
    if capability.startswith("cross_view_"):
        return "_ego_" in capability
    return capability == "reference_frame_transform"


def _round_robin(cells: list[dict]) -> list[str]:
    """Interleave one capability's cells across the scenes that supply it."""
    by_scene: dict[str, list[str]] = collections.defaultdict(list)
    for cell in cells:
        by_scene[cell["scene_key"]].append(cell["cell_id"])
    queues = [sorted(ids) for _, ids in sorted(by_scene.items())]
    ordered: list[str] = []
    for index in range(max((len(queue) for queue in queues), default=0)):
        for queue in queues:
            if index < len(queue):
                ordered.append(queue[index])
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.2,
        help="share of each capability's producer cells to render in this batch",
    )
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    by_capability: dict[str, list[dict]] = collections.defaultdict(list)
    for cell in plan["cells"]:
        # A cell whose geometry search came back empty would spend the batch
        # searching rather than rendering, and it contributes no episode to
        # review.  Batch one is drawn from cells that already hold candidates
        # so its size predicts its cost; the empty cells are repaired later,
        # when the wider attempt budget is worth spending.
        if _is_producer(cell["capability"]) and cell["candidates"]:
            by_capability[cell["capability"]].append(cell)

    chosen: list[str] = []
    total = 0
    print(f"{args.plan.parent.name}: {args.fraction:.0%} of each capability")
    for capability, cells in sorted(by_capability.items()):
        ordered = _round_robin(cells)
        # At least one cell, so a capability with few bindings is still
        # represented in the review set rather than rounded out of it.
        take = max(1, round(len(ordered) * args.fraction))
        picked = ordered[:take]
        chosen.extend(picked)
        picked_set = set(picked)
        scenes = {cell["scene_key"] for cell in cells if cell["cell_id"] in picked_set}
        trajectories = sum(len(c["candidates"]) for c in cells if c["cell_id"] in picked_set)
        total += trajectories
        print(
            f"  {capability:<40} {take:>4}/{len(ordered):<4} cells  "
            f"{len(scenes):>3}/{len({c['scene_key'] for c in cells})} scenes  "
            f"{trajectories:>4} trajectories"
        )

    args.output.write_text(json.dumps(sorted(chosen), indent=2) + "\n", encoding="utf-8")
    print(f"  -> {len(chosen)} producer cells, {total} trajectories, {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
