#!/usr/bin/env python
"""Count the bindings each rendered scene can supply to the P2 and P3 lines.

The production question is how many trajectories a full-scene run will yield,
and the answer is decided long before any GPU work: a capability can only
produce episodes for bindings that survive its slot, evidence and geometry
gates.  This walks a directory of already-rendered static source bundles,
reproduces exactly the binding selection ``coverage-plan`` would perform, and
reports the surviving count per scene and capability.  No renderer is
involved, so a full sweep runs on CPU in minutes.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

from spatial_episode.scriptgen.behavior import layout_from_scene_ir
from spatial_episode.scriptgen.collection import (
    CHAIN_MOTIFS,
    REFERENCE_CAPABILITIES,
    SourceRenderEvidence,
    _chain_binding_filter,
    _reference_binding_eligible,
    _scene_can_bind,
    ranked_bindings,
)
from spatial_episode.scriptgen.library import SCRIPT_LIBRARY
from spatial_episode.scriptgen.slotting import iter_bindings
from spatial_episode.scriptgen.standards import STD_V1

# One representative per group of capabilities that share a binding search.
# The yaw-offset reference variants and the anchor/closer chain cells reuse
# the ego cell's bindings, so screening them separately measures nothing.
SCREENED = (
    "reference_frame_transform",
    "cross_view_snapshot_ego_k1",
    "cross_view_snapshot_ego_k2",
    "cross_view_snapshot_ego_k3",
    "cross_view_ego_k1",
    "cross_view_ego_k2",
    "cross_view_ego_k3",
)


def _evidence(bundle: Path, ir: dict) -> SourceRenderEvidence:
    """Visibility proven by the bundle's own static survey render."""
    import itertools

    from spatial_episode.scriptgen.behavior import RenderSceneView

    view = RenderSceneView.from_bundle(bundle, STD_V1)
    visible_frames = {
        obj.name: frozenset(
            frame
            for frame in range(view.frame_count)
            if view.visibility(obj.name, frame).tristate(STD_V1) is True
        )
        for obj in view.layout.objects
    }
    visible = frozenset(name for name, frames in visible_frames.items() if frames)
    pairs = frozenset(
        frozenset((left, right))
        for left, right in itertools.combinations(visible, 2)
        if len(visible_frames[left] & visible_frames[right])
        >= STD_V1.chain_min_covisible_frames
    )
    return SourceRenderEvidence(visible_entities=visible, covisible_pairs=pairs)


def screen(bundle: Path, maximum: int) -> dict:
    ir = json.loads((bundle / "scene_ir.json").read_text(encoding="utf-8"))
    layout = layout_from_scene_ir(ir, std=STD_V1)
    try:
        evidence = _evidence(bundle, ir)
    except Exception as error:  # noqa: BLE001 - a broken bundle is a datum
        return {"scene": bundle.name, "error": f"{type(error).__name__}: {error}"}
    counts: dict[str, int] = {}
    for capability in SCREENED:
        script = SCRIPT_LIBRARY[capability]
        if not _scene_can_bind(layout, script):
            counts[capability] = 0
            continue
        if len(script.slots) == 1:
            bindings, _ = iter_bindings(layout, script)
            counts[capability] = len(tuple(bindings))
        elif capability in REFERENCE_CAPABILITIES:
            counts[capability] = sum(
                1
                for binding in ranked_bindings(layout, script, maximum=maximum)
                if _reference_binding_eligible(layout, binding, STD_V1)
            )
        elif script.motifs in CHAIN_MOTIFS:
            counts[capability] = len(
                ranked_bindings(
                    layout,
                    script,
                    maximum=maximum,
                    allowed_entity_ids=evidence.visible_entities,
                    binding_filter=_chain_binding_filter(layout, script, evidence),
                    std=STD_V1,
                )
            )
        else:
            counts[capability] = len(ranked_bindings(layout, script, maximum=maximum))
    return {
        "scene": bundle.name,
        "entities": len(ir["entities"]),
        "objects": len(layout.objects),
        "visible": len(evidence.visible_entities),
        "counts": counts,
    }


def _worker(item: tuple[str, int]) -> dict:
    bundle, maximum = item
    try:
        return screen(Path(bundle), maximum)
    except Exception as error:  # noqa: BLE001
        return {"scene": Path(bundle).name, "error": f"{type(error).__name__}: {error}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-multislot-bindings", type=int, default=200)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--scenes", nargs="*", default=None)
    args = parser.parse_args()

    root = Path(args.bundles_root)
    bundles = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "scene_ir.json").is_file()
    )
    if args.scenes:
        wanted = set(args.scenes)
        bundles = [path for path in bundles if path.name in wanted]
    work = [(str(path), args.maximum_multislot_bindings) for path in bundles]
    with mp.Pool(args.workers) as pool:
        rows = []
        for row in pool.imap_unordered(_worker, work):
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    rows.sort(key=lambda row: row["scene"])
    Path(args.output).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
