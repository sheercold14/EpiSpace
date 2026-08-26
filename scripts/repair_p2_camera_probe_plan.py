#!/usr/bin/env python
"""Minimally repair a finalized P2 plan after camera-probe collision changes.

The finalized P3 plan is frozen.  Valid P2 selections are retained verbatim;
only selections whose imagined camera station no longer satisfies the current
geometry contract are removed.  Replacements come from the existing P2 search
pool and pass the same exact-pose and structural-diversity gates as the 7k
finalizer.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spatial_episode.scriptgen.behavior import layout_from_scene_ir
from spatial_episode.scriptgen.binding_coverage import (
    CoverageCell,
    CoverageManifest,
    _attempt_index,
    _candidate_from_plan,
    _geometry_pool,
    trajectories_are_diverse,
)
from spatial_episode.scriptgen.collection import _reference_binding_eligible
from spatial_episode.scriptgen.motifs import imagined_station_placement
from spatial_episode.scriptgen.plan import TrajectoryPlan
from spatial_episode.scriptgen.standards import STD_V1


@dataclass(frozen=True)
class Option:
    cell: dict[str, Any]
    candidate: dict[str, Any]
    plan: TrajectoryPlan
    fingerprint: str


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _canonical(binding: dict[str, str]) -> str:
    return json.dumps(binding, sort_keys=True, separators=(",", ":"))


def _key(cell: dict[str, Any]) -> tuple[str, str]:
    return cell["scene_key"], _canonical(cell["binding"])


def _pose_fingerprint(plan: TrajectoryPlan) -> str:
    poses = [
        (pose.frame, round(pose.x, 6), round(pose.y, 6), round(pose.yaw_deg, 6))
        for pose in plan.poses
    ]
    encoded = json.dumps(poses, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_plan(candidate: dict[str, Any]) -> TrajectoryPlan:
    return TrajectoryPlan.model_validate_json(
        Path(candidate["plan_record"]).read_text(encoding="utf-8")
    )


def _extended_geometry_pool(
    payload: tuple[CoverageCell, Any, int],
) -> tuple[str, tuple[TrajectoryPlan, ...]]:
    """Generate one binding's expanded pool in a worker process."""
    cell, scene, attempt_frontier = payload
    plans, _ = _geometry_pool(
        cell,
        scene,
        attempts_per_binding=attempt_frontier,
        plans_per_binding=attempt_frontier,
        std=STD_V1,
    )
    return cell.cell_id, plans


def _producer_cells(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        cell
        for cell in manifest["cells"]
        if cell["capability"] == "reference_frame_transform"
    ]


def _capabilities_by_key(manifest: dict[str, Any]) -> dict[tuple[str, str], set[str]]:
    result: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for cell in manifest["cells"]:
        result[_key(cell)].add(cell["capability"])
    return result


def _station_status(
    cell: dict[str, Any],
    *,
    scene_ir_by_key: dict[str, str],
    layout_by_scene: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    scene_key = cell["scene_key"]
    if scene_key not in layout_by_scene:
        layout_by_scene[scene_key] = layout_from_scene_ir(
            Path(scene_ir_by_key[scene_key]), std=STD_V1
        )
    layout = layout_by_scene[scene_key]
    binding = cell["binding"]
    placement = imagined_station_placement(
        layout,
        binding["viewpoint"],
        binding["facing"],
        STD_V1.camera_height_m,
    )
    eligible = _reference_binding_eligible(layout, binding, STD_V1)
    candidates = cell.get("candidates") or []
    old: dict[str, Any] | None = None
    if candidates:
        render_plan = _read(Path(candidates[0]["render_plan"]))
        auxiliary = render_plan.get("auxiliary_views") or []
        if auxiliary:
            view = auxiliary[0]
            position = view["world_from_agent"]["translation_m"]
            old = {
                "method": view.get("station_method"),
                "x": float(position[0]),
                "y": float(position[1]),
            }
    current = (
        None
        if placement is None
        else {
            "method": placement.method,
            "x": placement.xy[0],
            "y": placement.xy[1],
        }
    )
    matches = bool(
        old is not None
        and current is not None
        and old["method"] == current["method"]
        and abs(old["x"] - current["x"]) <= 1e-8
        and abs(old["y"] - current["y"]) <= 1e-8
    )
    return eligible and matches, {
        "cell_id": cell["cell_id"],
        "scene_key": scene_key,
        "binding": binding,
        "pool_candidates": len(candidates),
        "old_station": old,
        "current_station": current,
        "binding_eligible": eligible,
        "station_matches": matches,
    }


def _ordered_options(cells: list[dict[str, Any]], cap: int) -> list[Option]:
    """Match the scene/binding/candidate interleave used by the 7k finalizer."""
    by_scene: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for cell in cells:
        by_scene[cell["scene_key"]].append(cell)
    for rows in by_scene.values():
        rows.sort(key=lambda cell: cell["cell_id"])

    result: list[Option] = []
    maximum_scene_bindings = max((len(rows) for rows in by_scene.values()), default=0)
    for candidate_depth in range(cap):
        for binding_depth in range(maximum_scene_bindings):
            for scene_key in sorted(by_scene):
                rows = by_scene[scene_key]
                if binding_depth >= len(rows):
                    continue
                cell = rows[binding_depth]
                candidates = cell["candidates"][:cap]
                if candidate_depth >= len(candidates):
                    continue
                candidate = candidates[candidate_depth]
                plan = _load_plan(candidate)
                result.append(
                    Option(
                        cell=cell,
                        candidate=candidate,
                        plan=plan,
                        fingerprint=_pose_fingerprint(plan),
                    )
                )
    return result


def _apply_selection(
    manifest: dict[str, Any], selected: dict[str, list[dict[str, Any]]]
) -> None:
    producers = {
        _key(cell): cell["cell_id"]
        for cell in manifest["cells"]
        if cell["capability"] == "reference_frame_transform"
        and cell["cell_id"] in selected
    }
    retained: list[dict[str, Any]] = []
    for original in manifest["cells"]:
        producer_id = producers.get(_key(original))
        if producer_id is None:
            continue
        candidates = selected[producer_id]
        cell = dict(original)
        cell["target_accepted"] = len(candidates)
        cell["candidates"] = candidates if cell["cell_id"] == producer_id else []
        retained.append(cell)
    manifest["cells"] = retained


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-final", type=Path, required=True)
    parser.add_argument("--p2-pool", type=Path, required=True)
    parser.add_argument("--p3-final", type=Path, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--cap", type=int, required=True)
    parser.add_argument(
        "--extend-attempts",
        type=int,
        default=0,
        help="extend valid bindings to this deterministic attempt frontier if needed",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="CPU workers used only for independent extended geometry searches",
    )
    parser.add_argument(
        "--extension-scenes",
        nargs="+",
        help="optionally limit extended search to these scene keys",
    )
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--output-pool", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    final = _read(args.p2_final)
    pool = _read(args.p2_pool)
    p3 = _read(args.p3_final)
    if pool["standard_version"] != STD_V1.standard_version:
        raise RuntimeError(
            f"P2 pool standard {pool['standard_version']} != {STD_V1.standard_version}"
        )

    scene_ir_by_key = {row["scene_key"]: row["scene_ir"] for row in pool["scenes"]}
    layout_by_scene: dict[str, Any] = {}
    invalid_keys: set[tuple[str, str]] = set()
    invalid_details: list[dict[str, Any]] = []
    for cell in _producer_cells(pool):
        if not cell.get("candidates"):
            continue
        valid, detail = _station_status(
            cell,
            scene_ir_by_key=scene_ir_by_key,
            layout_by_scene=layout_by_scene,
        )
        if not valid:
            invalid_keys.add(_key(cell))
            invalid_details.append(detail)

    filtered_pool = dict(pool)
    filtered_pool["cells"] = [
        cell for cell in pool["cells"] if _key(cell) not in invalid_keys
    ]
    capabilities = _capabilities_by_key(filtered_pool)
    required_capabilities = set(filtered_pool["capabilities"])
    eligible_cells = [
        cell
        for cell in _producer_cells(filtered_pool)
        if cell.get("candidates")
        and capabilities[_key(cell)] >= required_capabilities
    ]

    selected: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    selected_ids: set[str] = set()
    removed: list[dict[str, Any]] = []
    plans_by_scene: dict[str, list[TrajectoryPlan]] = collections.defaultdict(list)
    global_fingerprints: set[tuple[str, str]] = set()

    for cell in _producer_cells(final):
        if _key(cell) in invalid_keys:
            removed.extend(cell["candidates"])
            continue
        for candidate in cell["candidates"]:
            plan = _load_plan(candidate)
            selected[cell["cell_id"]].append(candidate)
            selected_ids.add(candidate["candidate_id"])
            plans_by_scene[cell["scene_key"]].append(plan)
            global_fingerprints.add((cell["scene_key"], _pose_fingerprint(plan)))

    # P3 remains unchanged, but replacement P2 paths must not duplicate any of it.
    for cell in p3["cells"]:
        if not cell["capability"].startswith("cross_view_ego_k"):
            continue
        for candidate in cell["candidates"]:
            plan = _load_plan(candidate)
            global_fingerprints.add((cell["scene_key"], _pose_fingerprint(plan)))

    additions: list[dict[str, Any]] = []
    rejected_exact = 0
    rejected_near = 0
    for option in _ordered_options(eligible_cells, args.cap):
        if sum(map(len, selected.values())) >= args.target:
            break
        cell_id = option.cell["cell_id"]
        if option.candidate["candidate_id"] in selected_ids:
            continue
        if len(selected[cell_id]) >= args.cap:
            continue
        scene_fingerprint = (option.cell["scene_key"], option.fingerprint)
        if scene_fingerprint in global_fingerprints:
            rejected_exact += 1
            continue
        if not all(
            trajectories_are_diverse(option.plan, prior)
            for prior in plans_by_scene[option.cell["scene_key"]]
        ):
            rejected_near += 1
            continue
        selected[cell_id].append(option.candidate)
        selected_ids.add(option.candidate["candidate_id"])
        plans_by_scene[option.cell["scene_key"]].append(option.plan)
        global_fingerprints.add(scene_fingerprint)
        additions.append(
            {
                "source": "existing_search_pool",
                "cell_id": cell_id,
                "scene_key": option.cell["scene_key"],
                "binding": option.cell["binding"],
                "candidate_id": option.candidate["candidate_id"],
                "plan_id": option.candidate["plan_id"],
            }
        )

    selected_count = sum(map(len, selected.values()))
    extension_stats = {
        "attempt_frontier": args.extend_attempts,
        "cells_searched": 0,
        "geometry_plans_examined": 0,
        "existing_candidate_ids_skipped": 0,
        "exact_pose_rejected": 0,
        "near_duplicate_rejected": 0,
        "generated_candidates_selected": 0,
    }
    if selected_count < args.target and args.extend_attempts:
        if args.extend_attempts <= int(pool["attempts_per_binding"]):
            raise ValueError(
                "--extend-attempts must exceed the existing pool attempt frontier "
                f"{pool['attempts_per_binding']}"
            )
        expanded_payload = dict(pool)
        expanded_payload["attempts_per_binding"] = args.extend_attempts
        expanded_manifest = CoverageManifest.model_validate(expanded_payload)
        scenes = {scene.scene_key: scene for scene in expanded_manifest.scenes}
        existing_candidate_ids = {
            candidate["candidate_id"]
            for cell in _producer_cells(pool)
            for candidate in cell.get("candidates") or []
        }
        scene_counts = collections.Counter(
            {
                scene_key: len(plans)
                for scene_key, plans in plans_by_scene.items()
            }
        )
        search_cells = sorted(
            eligible_cells,
            key=lambda cell: (
                scene_counts[cell["scene_key"]],
                len(selected[cell["cell_id"]]),
                cell["scene_key"],
                cell["cell_id"],
            ),
        )
        searchable = [
            cell
            for cell in search_cells
            if len(selected[cell["cell_id"]]) < args.cap
            and (
                args.extension_scenes is None
                or cell["scene_key"] in set(args.extension_scenes)
            )
        ]
        workers = max(1, args.workers)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for start in range(0, len(searchable), workers):
                if sum(map(len, selected.values())) >= args.target:
                    break
                batch = searchable[start : start + workers]
                futures = {
                    executor.submit(
                        _extended_geometry_pool,
                        (
                            CoverageCell.model_validate(cell),
                            scenes[cell["scene_key"]],
                            args.extend_attempts,
                        ),
                    ): cell
                    for cell in batch
                }
                for future in concurrent.futures.as_completed(futures):
                    cell = futures[future]
                    cell_id, geometry_pool = future.result()
                    room = args.cap - len(selected[cell_id])
                    typed_cell = CoverageCell.model_validate(cell)
                    scene = scenes[cell["scene_key"]]
                    extension_stats["cells_searched"] += 1
                    extension_stats["geometry_plans_examined"] += len(geometry_pool)
                    layout = layout_by_scene[cell["scene_key"]]
                    for plan in geometry_pool:
                        if sum(map(len, selected.values())) >= args.target or room <= 0:
                            break
                        attempt_index = _attempt_index(plan.plan_id)
                        candidate_id = (
                            f"{cell_id}{typed_cell.candidate_namespace}__a"
                            f"{attempt_index:03d}"
                        )
                        if candidate_id in existing_candidate_ids:
                            extension_stats["existing_candidate_ids_skipped"] += 1
                            continue
                        fingerprint = _pose_fingerprint(plan)
                        scene_fingerprint = (cell["scene_key"], fingerprint)
                        if scene_fingerprint in global_fingerprints:
                            extension_stats["exact_pose_rejected"] += 1
                            continue
                        if not all(
                            trajectories_are_diverse(plan, prior)
                            for prior in plans_by_scene[cell["scene_key"]]
                        ):
                            extension_stats["near_duplicate_rejected"] += 1
                            continue
                        candidate_model = _candidate_from_plan(
                            plan,
                            typed_cell,
                            scene,
                            layout,
                            expanded_manifest,
                            STD_V1,
                        )
                        candidate = candidate_model.model_dump(mode="json")
                        cell["candidates"].append(candidate)
                        selected[cell_id].append(candidate)
                        selected_ids.add(candidate_id)
                        existing_candidate_ids.add(candidate_id)
                        plans_by_scene[cell["scene_key"]].append(plan)
                        global_fingerprints.add(scene_fingerprint)
                        scene_counts[cell["scene_key"]] += 1
                        room -= 1
                        extension_stats["generated_candidates_selected"] += 1
                        additions.append(
                            {
                                "source": "extended_geometry_search",
                                "cell_id": cell_id,
                                "scene_key": cell["scene_key"],
                                "binding": cell["binding"],
                                "candidate_id": candidate_id,
                                "plan_id": plan.plan_id,
                            }
                        )
                    cell["geometry_pool_size"] = max(
                        int(cell.get("geometry_pool_size", 0)), len(geometry_pool)
                    )
                    cell["diverse_pool_size"] = max(
                        int(cell.get("diverse_pool_size", 0)), len(cell["candidates"])
                    )
                    cell["search_attempt_limit"] = args.extend_attempts
                    cell["raw_plan_limit"] = args.extend_attempts
                    cell["search_pool_exhausted"] = False
                    print(
                        "repair extension "
                        f"cell={cell_id} geometry={len(geometry_pool)} "
                        f"generated_total={extension_stats['generated_candidates_selected']} "
                        f"selected={sum(map(len, selected.values()))}/{args.target}",
                        flush=True,
                    )

        filtered_pool["attempts_per_binding"] = args.extend_attempts

    selected_count = sum(map(len, selected.values()))

    repaired = dict(filtered_pool)
    _apply_selection(repaired, dict(selected))
    _write(args.output_plan, repaired)
    _write(args.output_pool, filtered_pool)
    report = {
        "schema_version": "p2_camera_probe_plan_repair.v1",
        "status": "pass" if selected_count == args.target else "fail",
        "p2_target": args.target,
        "original_selected": sum(
            len(cell["candidates"]) for cell in _producer_cells(final)
        ),
        "selected_after_repair": selected_count,
        "unchanged_selected": selected_count - len(additions),
        "removed_selected": len(removed),
        "added_selected": len(additions),
        "invalid_pool_bindings": len(invalid_keys),
        "invalid_pool_candidates": sum(row["pool_candidates"] for row in invalid_details),
        "eligible_pool_bindings_after_filter": len(eligible_cells),
        "replacement_rejections": {
            "exact_pose": rejected_exact,
            "near_duplicate": rejected_near,
        },
        "extension_search": extension_stats,
        "removed_candidate_ids": [row["candidate_id"] for row in removed],
        "additions": additions,
        "invalid_bindings": invalid_details,
        "output_plan": str(args.output_plan.resolve()),
        "output_filtered_pool": str(args.output_pool.resolve()),
    }
    _write(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if selected_count == args.target else 1


if __name__ == "__main__":
    raise SystemExit(main())
