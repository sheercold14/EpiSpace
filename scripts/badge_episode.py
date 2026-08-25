#!/usr/bin/env python
"""Badge the objects of an already-rendered episode and lay them out for review.

This exists to answer one question before anything downstream is built on
badges: does a badge actually land on the object it names, in every frame, for
the awkward objects?  The awkward ones are the point - a two-centimetre television
seen edge-on, a sink recessed into a counter, a cabinet under a worktop - because
those are exactly the objects the pipeline currently throws away, and badging
them is only worth doing if the badge is legible when it lands.

Runs on episodes already on disk; no GPU and no re-render.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image

from spatial_episode.scriptgen.behavior import RenderSceneView
from spatial_episode.scriptgen.markers import annotate_frame, assign_badges
from spatial_episode.scriptgen.standards import STD_V1

# The order badges are numbered in.  Slot order rather than alphabetical, so
# badge 1 is the viewpoint in every reference-frame episode and the chain runs
# target -> anchors -> other in every cross-view one; the reviewer does not
# have to re-learn the numbering per episode.  Anchors sort by their index, so
# anchor10 follows anchor9 rather than anchor1.
SLOT_ORDER = ("viewpoint", "facing", "target", "other")


def _slot_rank(slot: str) -> tuple[int, int, str]:
    if slot.startswith("anchor") and slot[6:].isdigit():
        return (SLOT_ORDER.index("target") + 1, int(slot[6:]), slot)
    if slot in SLOT_ORDER:
        rank = SLOT_ORDER.index(slot)
        # "other" closes the chain, so push it past the anchors.
        return (rank + 1 if slot == "other" else rank, 0, slot)
    return (len(SLOT_ORDER) + 1, 0, slot)


def _binding_entities(binding: dict) -> list[tuple[str, str]]:
    """(slot, entity_id) in a stable, human-predictable order."""
    return sorted(binding.items(), key=lambda item: _slot_rank(item[0]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--episode", required=True, help="group directory name")
    parser.add_argument(
        "--sources-root",
        type=Path,
        default=Path("outputs/p23_sources_all_v1"),
        help="holds scenes/<scene>/scene_ir.json for the runtime id map",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=900,
        help="render-resolution pixels below which an object is left unbadged",
    )
    args = parser.parse_args()

    group = args.collection_root / "groups" / args.episode
    bundle = args.collection_root / "bundles" / args.episode
    certificate = json.loads((group / "primary.certificate.json").read_text(encoding="utf-8"))
    binding = certificate["binding"]
    scene_key = args.episode.split("__", 1)[0]
    scene_ir_path = args.sources_root / "scenes" / scene_key / "scene_ir.json"
    scene_ir = json.loads(scene_ir_path.read_text(encoding="utf-8"))
    # The bundle's own registry, not scene_ir's: replaying a scene reassigns
    # every runtime instance id, so scene_ir's copy matches nothing in these
    # masks.  RenderSceneView already resolves this through source_entity_id.
    runtime_by_entity = RenderSceneView.from_bundle(
        bundle, STD_V1, scene_ir=scene_ir_path
    ).entity_runtime_ids
    labels = {
        entity["entity_id"]: entity.get("raw_label", entity["entity_id"])
        for entity in scene_ir["entities"]
    }

    slots = _binding_entities(binding)
    assignments = assign_badges([entity for _, entity in slots], runtime_by_entity)
    by_entity = {assignment.entity_id: assignment for assignment in assignments}
    missing = [slot for slot, entity in slots if entity not in by_entity]

    args.output.mkdir(parents=True, exist_ok=True)
    frames = sorted((bundle / "views").glob("view-*.sensors.npz"))
    seen: dict[int, int] = {}
    for index, path in enumerate(frames):
        with np.load(path) as arrays:
            image, placements = annotate_frame(
                arrays["rgb"],
                arrays["instance_id"],
                assignments,
                size=args.size,
                min_pixels=args.min_pixels,
            )
        Image.fromarray(image).save(args.output / f"view-{index:03d}.badged.png")
        for placement in placements:
            seen[placement.number] = seen.get(placement.number, 0) + 1

    legend = []
    for slot, entity in slots:
        assignment = by_entity.get(entity)
        number = assignment.number if assignment else None
        legend.append(
            {
                "slot": slot,
                "entity_id": entity,
                "label": labels.get(entity, "?"),
                "badge": number,
                "frames_badged": seen.get(number, 0) if number else 0,
            }
        )
    summary = {
        "episode": args.episode,
        "capability": certificate["capability"],
        "status": certificate["status"],
        "frames": len(frames),
        "min_pixels": args.min_pixels,
        "legend": legend,
        "unrenderable_slots": missing,
    }
    (args.output / "badges.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows = "".join(
        f"<tr><td>{entry['badge'] or '-'}</td><td>{html.escape(entry['slot'])}</td>"
        f"<td>{html.escape(entry['label'])}</td>"
        f"<td>{entry['frames_badged']}/{len(frames)}</td></tr>"
        for entry in legend
    )
    images = "".join(
        f'<figure><img src="view-{i:03d}.badged.png" width="{args.size}">'
        f"<figcaption>frame {i}</figcaption></figure>"
        for i in range(len(frames))
    )
    (args.output / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<style>body{font-family:sans-serif;background:#111;color:#eee}"
        "figure{display:inline-block;margin:6px}img{border:1px solid #444}"
        "table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #555;padding:4px 8px}"
        "</style>"
        f"<h2>{html.escape(args.episode)}</h2>"
        f"<p>{html.escape(certificate['capability'])} &middot; {certificate['status']} &middot; "
        f"badge threshold {args.min_pixels}px</p>"
        f"<table><tr><th>badge</th><th>slot</th><th>label</th><th>frames badged</th></tr>"
        f"{rows}</table>{images}",
        encoding="utf-8",
    )

    print(f"{args.episode}: {len(frames)} frames -> {args.output}")
    for entry in legend:
        print(
            f"  badge {entry['badge'] or '-'}  {entry['slot']:<10} {entry['label']:<28} "
            f"badged in {entry['frames_badged']}/{len(frames)} frames"
        )
    if missing:
        print(f"  slots with no runtime id (never renderable): {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
