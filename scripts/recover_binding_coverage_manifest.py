#!/usr/bin/env python3
"""Rebuild a clobbered coverage manifest from durable planning artifacts.

The output is always a separate file.  This script never replaces the input
manifest, status file, plans, recipes, bundles, or groups.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from spatial_episode.scriptgen.binding_coverage import (
    DEFERRED_INITIAL_CAPABILITIES,
    CoverageCandidate,
    CoverageCell,
    _binding_cache_key,
    _bindings_for_scene,
    _canonical_binding,
    _derived_seed,
    _scene_occupancy_proxy_rejection,
    _stable_hex,
    _write_json,
    load_coverage,
)
from spatial_episode.scriptgen.motifs import _occupancy_grid
from spatial_episode.scriptgen.plan import TrajectoryPlan
from spatial_episode.scriptgen.standards import STD_V1


def _candidate_artifacts(
    output_root: Path,
) -> dict[str, tuple[CoverageCandidate, ...]]:
    grouped: dict[str, list[CoverageCandidate]] = defaultdict(list)
    for plan_record in sorted((output_root / "plans").glob("*.record.json")):
        candidate_id = plan_record.name.removesuffix(".record.json")
        try:
            cell_id, attempt_text = candidate_id.rsplit("__a", 1)
        except ValueError as error:
            raise ValueError(f"invalid candidate artifact name: {plan_record}") from error
        if not attempt_text.isdigit():
            raise ValueError(f"invalid candidate attempt suffix: {plan_record}")
        plan = TrajectoryPlan.model_validate_json(plan_record.read_text(encoding="utf-8"))
        attempt_index = int(attempt_text)
        if not plan.plan_id.endswith(f".a{attempt_index}"):
            raise ValueError(
                f"candidate filename / plan attempt mismatch: {plan_record} / {plan.plan_id}"
            )
        render_plan = output_root / "plans" / f"{candidate_id}.views.json"
        recipe = output_root / "recipes" / f"{candidate_id}.yaml"
        if not render_plan.is_file() or not recipe.is_file():
            raise FileNotFoundError(f"candidate artifact set is incomplete: {candidate_id}")
        grouped[cell_id].append(
            CoverageCandidate(
                candidate_id=candidate_id,
                plan_id=plan.plan_id,
                attempt_index=attempt_index,
                plan_record=str(plan_record.resolve()),
                render_plan=str(render_plan.resolve()),
                recipe=str(recipe.resolve()),
                bundle=str((output_root / "bundles" / candidate_id).resolve()),
                group=str((output_root / "groups" / candidate_id).resolve()),
                log=str((output_root / "logs" / f"{candidate_id}.log").resolve()),
            )
        )
    return {
        cell_id: tuple(sorted(candidates, key=lambda item: item.attempt_index))
        for cell_id, candidates in grouped.items()
    }


def recover_manifest(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("recovery output must differ from the source manifest")
    manifest = load_coverage(source)
    output_root = Path(manifest.output_root).resolve()
    artifacts = _candidate_artifacts(output_root)
    artifact_ids = {
        candidate.candidate_id
        for candidates in artifacts.values()
        for candidate in candidates
    }

    cells = list(manifest.cells)
    cell_indices = {cell.cell_id: index for index, cell in enumerate(cells)}
    original_cell_ids = set(cell_indices)

    # Preserve audited metadata for existing cells, while also recovering any
    # complete candidate artifact written immediately before an interruption.
    for cell_id, index in tuple(cell_indices.items()):
        cell = cells[index]
        discovered = artifacts.get(cell_id, ())
        if not discovered:
            continue
        known = {candidate.candidate_id: candidate for candidate in cell.candidates}
        for candidate in discovered:
            known.setdefault(candidate.candidate_id, candidate)
        merged = tuple(sorted(known.values(), key=lambda item: item.attempt_index))
        cells[index] = cell.model_copy(update={"candidates": merged})

    scene_skips = dict(manifest.scene_skips)
    for scene_index, scene in enumerate(manifest.scenes, 1):
        scene_rejection = _scene_occupancy_proxy_rejection(scene, std=STD_V1)
        if scene_rejection is not None:
            scene_skips[scene.scene_key] = scene_rejection
            print(
                f"recover scene={scene.scene_key} skipped={scene_rejection}",
                flush=True,
            )
            _occupancy_grid.cache_clear()
            continue
        _occupancy_grid.cache_clear()
        binding_cache: dict[tuple[object, ...], tuple[dict[str, str], ...]] = {}
        for capability in manifest.capabilities:
            cache_key = _binding_cache_key(capability)
            bindings = binding_cache.get(cache_key)
            if bindings is None:
                bindings = _bindings_for_scene(
                    scene,
                    capability,
                    maximum_multislot_bindings=manifest.maximum_multislot_bindings,
                    std=STD_V1,
                )
                binding_cache[cache_key] = bindings
            if manifest.limit_bindings_per_capability is not None:
                bindings = bindings[: manifest.limit_bindings_per_capability]
            for binding in bindings:
                cell_id = (
                    f"{scene.scene_key}__{capability}__"
                    f"{_stable_hex(_canonical_binding(binding))[:12]}"
                )
                if cell_id in cell_indices:
                    continue
                candidates = artifacts.get(cell_id, ())
                for candidate in candidates:
                    plan = TrajectoryPlan.model_validate_json(
                        Path(candidate.plan_record).read_text(encoding="utf-8")
                    )
                    if plan.capability != capability or plan.binding != binding:
                        raise ValueError(
                            f"artifact does not match recovered cell {cell_id}: {candidate.plan_id}"
                        )
                deferred = capability in DEFERRED_INITIAL_CAPABILITIES
                cell = CoverageCell(
                    cell_id=cell_id,
                    scene_key=scene.scene_key,
                    scene_id=scene.scene_id,
                    capability=capability,
                    binding=binding,
                    seed=_derived_seed(
                        manifest.collection_id,
                        scene.scene_id,
                        capability,
                        binding,
                    ),
                    target_accepted=manifest.accepted_per_binding,
                    geometry_pool_size=len(candidates),
                    diverse_pool_size=len(candidates),
                    search_attempt_limit=(
                        0 if deferred else manifest.initial_attempts_per_binding
                    ),
                    raw_plan_limit=(
                        0
                        if deferred
                        else min(
                            manifest.initial_attempts_per_binding,
                            manifest.accepted_per_binding * 2,
                        )
                    ),
                    search_pool_exhausted=False,
                    rejection_counts=(
                        {"search:deferred_for_shared_credit": 1}
                        if deferred
                        else {"recovery:artifact_manifest_rebuild": 1}
                    ),
                    candidates=candidates,
                )
                cell_indices[cell_id] = len(cells)
                cells.append(cell)
        print(
            f"recover scenes={scene_index}/{len(manifest.scenes)} "
            f"scene={scene.scene_key} cells={len(cells)}",
            flush=True,
        )

    referenced_ids = {
        candidate.candidate_id for cell in cells for candidate in cell.candidates
    }
    unreferenced = sorted(artifact_ids - referenced_ids)
    if unreferenced:
        raise ValueError(
            f"{len(unreferenced)} candidate artifacts do not map to enumerated cells; "
            f"first={unreferenced[:3]}"
        )
    recovered = manifest.model_copy(
        update={"scene_skips": scene_skips, "cells": tuple(cells)}
    )
    _write_json(output, recovered)
    validated = load_coverage(output)
    validated_ids = {
        candidate.candidate_id
        for cell in validated.cells
        for candidate in cell.candidates
    }
    if validated_ids != artifact_ids:
        raise ValueError("recovered manifest candidate inventory failed validation")
    print(
        f"recovered source_cells={len(original_cell_ids)} cells={len(validated.cells)} "
        f"candidates={len(validated_ids)} output={output}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recover_manifest(args.manifest, args.output)


if __name__ == "__main__":
    main()
