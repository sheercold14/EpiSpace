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

from spatial_episode.scriptgen import collection as collection_module
from spatial_episode.scriptgen.behavior import layout_from_scene_ir
from spatial_episode.scriptgen.collection import (
    CHAIN_MOTIFS,
    REFERENCE_CAPABILITIES,
    SourceRenderEvidence,
    _chain_binding_filter,
    _reference_binding_eligible,
    _scene_can_bind,
    _stable_rank,
    ranked_bindings,
)
from spatial_episode.scriptgen.library import SCRIPT_LIBRARY
from spatial_episode.scriptgen.slotting import iter_bindings
from spatial_episode.scriptgen.spec import ScriptSpec
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

# Objects larger than this have no definite position for a question to ask
# about: Beechwood_0_garden's longest rail_fence spans 32 m, so "which
# direction is the fence" has no single answer.  The bound only bites once
# unique_referent is relaxed - a scene rarely holds exactly one of anything
# that big.
POOL_MAXIMUM_SIZE_M = 4.0


def relax_unique_referent(script: ScriptSpec) -> ScriptSpec:
    """The same script with the scene-unique-category requirement dropped.

    ``unique_referent`` exists because a question naming an object by its
    category is unanswerable when the scene holds two of them.  A badge drawn
    on the object refers without the category, so the requirement is what the
    badge is for: measuring the badged pipeline means measuring this script.

    Screening-only.  It builds a modified copy rather than editing the library
    so a screening run cannot leak a relaxed slot into a production plan.
    """
    return script.model_copy(
        update={
            "slots": {
                name: slot.model_copy(update={"unique_referent": False})
                for name, slot in script.slots.items()
            }
        }
    )


def unbiased_pool(
    layout,
    *,
    maximum_size_m: float,
    count: int,
    restrict_to: frozenset[str] | None = None,
) -> frozenset[str]:
    """An unbiased sample of the scene's objects to draw slot candidates from.

    ``_eligible_objects`` ranks by ``-min(size_m, 2.0)`` and breaks the tie on
    category name, which under ``unique_referent`` was harmless - the pool was
    a handful of scene-unique objects and the whole pool fitted.  Without that
    filter the clamp puts every object over two metres into one tie and the
    alphabet decides the pool: Beechwood_0_garden's eighteen candidates come
    out as six fence segments, bushes, carpets and gates, and not one binding
    of them survives the reference-frame geometry gate.

    Uniqueness was quietly doing a second job.  A category present exactly once
    is usually a discrete, well-placed thing - a piano, a fridge, a car - while
    a category present many times is usually repeated structure.  Dropping the
    filter for the badge drops that job too, so it has to be done directly:
    exclude objects too large to have a definite position, then order by a
    stable hash so the pool is an unbiased sample rather than an alphabetical
    one.

    ``restrict_to`` narrows the population *before* the sample is drawn, which
    is not the same as intersecting afterwards.  A chain slot must have been
    seen by the source render, and office_large has 668 objects of which the
    render saw 27; eighteen drawn from all 668 and then intersected leaves
    almost nothing, while eighteen drawn from the 27 is a usable pool.
    """
    population = layout.objects
    if restrict_to is not None:
        population = [obj for obj in population if obj.name in restrict_to]
    objects = [obj for obj in population if obj.size_m <= maximum_size_m]
    objects.sort(key=lambda obj: _stable_rank(obj.name))
    return frozenset(obj.name for obj in objects[:count])


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


def screen(
    bundle: Path,
    maximum: int,
    *,
    relax: bool = False,
    unbiased: bool = False,
) -> dict:
    ir = json.loads((bundle / "scene_ir.json").read_text(encoding="utf-8"))
    layout = layout_from_scene_ir(ir, std=STD_V1)
    try:
        evidence = _evidence(bundle, ir)
    except Exception as error:  # noqa: BLE001 - a broken bundle is a datum
        return {"scene": bundle.name, "error": f"{type(error).__name__}: {error}"}
    pool = (
        unbiased_pool(
            layout,
            maximum_size_m=POOL_MAXIMUM_SIZE_M,
            count=collection_module.MAX_RANKED_OBJECTS,
        )
        if unbiased
        else None
    )
    counts: dict[str, int] = {}
    for capability in SCREENED:
        script = SCRIPT_LIBRARY[capability]
        if relax:
            script = relax_unique_referent(script)
        if not _scene_can_bind(layout, script):
            counts[capability] = 0
            continue
        if len(script.slots) == 1:
            bindings, _ = iter_bindings(layout, script)
            counts[capability] = len(tuple(bindings))
        elif capability in REFERENCE_CAPABILITIES:
            counts[capability] = sum(
                1
                for binding in ranked_bindings(
                    layout, script, maximum=maximum, allowed_entity_ids=pool
                )
                if _reference_binding_eligible(layout, binding, STD_V1)
            )
        elif script.motifs in CHAIN_MOTIFS:
            allowed = (
                unbiased_pool(
                    layout,
                    maximum_size_m=POOL_MAXIMUM_SIZE_M,
                    count=collection_module.MAX_RANKED_OBJECTS,
                    restrict_to=evidence.visible_entities,
                )
                if unbiased
                else evidence.visible_entities
            )
            counts[capability] = len(
                ranked_bindings(
                    layout,
                    script,
                    maximum=maximum,
                    allowed_entity_ids=allowed,
                    binding_filter=_chain_binding_filter(layout, script, evidence),
                    std=STD_V1,
                )
            )
        else:
            counts[capability] = len(
                ranked_bindings(layout, script, maximum=maximum, allowed_entity_ids=pool)
            )
    return {
        "scene": bundle.name,
        "entities": len(ir["entities"]),
        "objects": len(layout.objects),
        "visible": len(evidence.visible_entities),
        "pool": len(pool) if pool is not None else None,
        "counts": counts,
    }


def _worker(item: tuple[str, int, bool, bool, int]) -> dict:
    bundle, maximum, relax, unbiased, ranked_objects = item
    # Set in the child rather than the parent so the run is identical whether
    # the pool forks or spawns.  _eligible_objects reads the module global.
    collection_module.MAX_RANKED_OBJECTS = ranked_objects
    try:
        return screen(Path(bundle), maximum, relax=relax, unbiased=unbiased)
    except Exception as error:  # noqa: BLE001
        return {"scene": Path(bundle).name, "error": f"{type(error).__name__}: {error}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-multislot-bindings", type=int, default=200)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument(
        "--relax-unique-referent",
        action="store_true",
        help="screen the badged pipeline: objects need not be the only one of their category",
    )
    parser.add_argument(
        "--unbiased-pool",
        action="store_true",
        help="draw slot candidates by stable hash instead of by clamped size then alphabet",
    )
    parser.add_argument(
        "--max-ranked-objects",
        type=int,
        default=collection_module.MAX_RANKED_OBJECTS,
        help=(
            "per-slot candidate cap.  The default only ever bound a handful of "
            "scene-unique objects, so it was never the limit; with the badge it is."
        ),
    )
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
    work = [
        (
            str(path),
            args.maximum_multislot_bindings,
            args.relax_unique_referent,
            args.unbiased_pool,
            args.max_ranked_objects,
        )
        for path in bundles
    ]
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
