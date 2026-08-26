#!/usr/bin/env python
"""Select exactly 7,000 balanced, diverse trajectories from P2/P3 search pools.

This is deliberately a hard gate: a final manifest is written only when every
quota can be met after per-scene exact-pose de-duplication and a second
pairwise structural-diversity check across every selected trajectory in the
same scene and tier.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spatial_episode.scriptgen.binding_coverage import trajectories_are_diverse
from spatial_episode.scriptgen.plan import TrajectoryPlan


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
    return hashlib.sha256(json.dumps(poses, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load_pool(path: Path) -> tuple[dict[str, Any], Path]:
    backup = path.with_name("coverage.search_pool.json")
    return _read(backup if backup.is_file() else path), backup


def _capabilities_by_key(manifest: dict[str, Any]) -> dict[tuple[str, str], set[str]]:
    result: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for cell in manifest["cells"]:
        result[_key(cell)].add(cell["capability"])
    return result


def _eligible_cells(
    manifest: dict[str, Any],
    *,
    producer_capability: str,
    required_capabilities: set[str],
) -> list[dict[str, Any]]:
    capabilities = _capabilities_by_key(manifest)
    return [
        cell
        for cell in manifest["cells"]
        if cell["capability"] == producer_capability
        and cell["candidates"]
        and capabilities[_key(cell)] >= required_capabilities
    ]


def _ordered_options(cells: list[dict[str, Any]], cap: int) -> list[Option]:
    """Interleave scenes and bindings, exhausting one candidate depth at a time."""
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
                plan = TrajectoryPlan.model_validate_json(
                    Path(candidate["plan_record"]).read_text(encoding="utf-8")
                )
                result.append(
                    Option(
                        cell=cell,
                        candidate=candidate,
                        plan=plan,
                        fingerprint=_pose_fingerprint(plan),
                    )
                )
    return result


def _select(
    cells: list[dict[str, Any]],
    *,
    target: int,
    cap: int,
    minimum_bindings: int,
    global_fingerprints: set[tuple[str, str]],
    tier_plans_by_scene: dict[str, list[TrajectoryPlan]],
    allow_shortfall: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    selected: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    scene_counts: collections.Counter[str] = collections.Counter()
    rejected_exact = 0
    rejected_near = 0
    for option in _ordered_options(cells, cap):
        cell_id = option.cell["cell_id"]
        if len(selected.get(cell_id, ())) >= cap:
            continue
        scene_fingerprint = (option.cell["scene_key"], option.fingerprint)
        if scene_fingerprint in global_fingerprints:
            rejected_exact += 1
            continue
        if not all(
            trajectories_are_diverse(option.plan, prior)
            for prior in tier_plans_by_scene.get(option.cell["scene_key"], ())
        ):
            rejected_near += 1
            continue
        selected[cell_id].append(option.candidate)
        tier_plans_by_scene.setdefault(option.cell["scene_key"], []).append(option.plan)
        global_fingerprints.add(scene_fingerprint)
        scene_counts[option.cell["scene_key"]] += 1
        if sum(map(len, selected.values())) == target:
            break

    count = sum(map(len, selected.values()))
    if count != target and not allow_shortfall:
        raise RuntimeError(
            f"quality gate retained {count}/{target} trajectories from "
            f"{len(cells)} eligible bindings"
        )
    binding_count = sum(bool(candidates) for candidates in selected.values())
    if binding_count < minimum_bindings:
        raise RuntimeError(
            f"binding diversity gate retained {binding_count}; needs {minimum_bindings}"
        )
    histogram = collections.Counter(len(candidates) for candidates in selected.values())
    return dict(selected), {
        "trajectories": count,
        "bindings": binding_count,
        "eligible_bindings": len(cells),
        "scenes": len(scene_counts),
        "trajectories_by_scene": dict(sorted(scene_counts.items())),
        "trajectories_per_binding": {
            str(size): frequency for size, frequency in sorted(histogram.items())
        },
        "exact_pose_duplicates_rejected": rejected_exact,
        "near_duplicates_rejected": rejected_near,
    }


def _ratio_targets(total: int, weights: dict[str, int]) -> dict[str, int]:
    denominator = sum(weights.values())
    floors = {key: total * weight // denominator for key, weight in weights.items()}
    remainder = total - sum(floors.values())
    priority = sorted(
        weights,
        key=lambda key: (-(total * weights[key] % denominator), key),
    )
    for key in priority[:remainder]:
        floors[key] += 1
    return floors


def _apply_selection(
    manifest: dict[str, Any],
    selected: dict[str, list[dict[str, Any]]],
    producer_capabilities: set[str],
) -> None:
    producers = {
        _key(cell): cell["cell_id"]
        for cell in manifest["cells"]
        if cell["capability"] in producer_capabilities and cell["cell_id"] in selected
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
    parser.add_argument("--p2", type=Path, required=True)
    parser.add_argument("--p3", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    p2, p2_backup = _load_pool(args.p2)
    p3, p3_backup = _load_pool(args.p3)
    contract = _read(args.contract)
    allocation = contract["allocation"]
    allocation_policy = contract["allocation_policy"]
    quality = contract["quality"]
    global_fingerprints: set[tuple[str, str]] = set()
    p2_plans_by_scene: dict[str, list[TrajectoryPlan]] = {}

    p2_required = set(p2["capabilities"])
    p2_cells = _eligible_cells(
        p2,
        producer_capability="reference_frame_transform",
        required_capabilities=p2_required,
    )
    p2_selected, p2_summary = _select(
        p2_cells,
        target=int(allocation["p2"]),
        cap=int(quality["p2_max_trajectories_per_binding"]),
        minimum_bindings=0,
        global_fingerprints=global_fingerprints,
        tier_plans_by_scene=p2_plans_by_scene,
        allow_shortfall=True,
    )
    p2_count = int(p2_summary["trajectories"])
    if p2_count < int(allocation_policy["p2_minimum"]):
        raise RuntimeError(f"P2 retained {p2_count}; minimum is {allocation_policy['p2_minimum']}")
    p3_targets = _ratio_targets(
        int(contract["physical_trajectory_target"]) - p2_count,
        {key: int(value) for key, value in allocation_policy["p3_k_ratio"].items()},
    )

    p3_selected: dict[str, list[dict[str, Any]]] = {}
    p3_summary: dict[str, Any] = {}
    p3_plans_by_scene: dict[str, list[TrajectoryPlan]] = {}
    for k in (1, 2, 3):
        producer = f"cross_view_ego_k{k}"
        required = {
            producer,
            f"cross_view_anchor_k{k}",
            f"cross_view_closer_k{k}",
        }
        cells = _eligible_cells(
            p3,
            producer_capability=producer,
            required_capabilities=required,
        )
        selected, summary = _select(
            cells,
            target=p3_targets[f"k{k}"],
            cap=int(quality["p3_max_trajectories_per_binding"]),
            minimum_bindings=(
                p3_targets[f"k{k}"] + int(quality["p3_max_trajectories_per_binding"]) - 1
            )
            // int(quality["p3_max_trajectories_per_binding"]),
            global_fingerprints=global_fingerprints,
            tier_plans_by_scene=p3_plans_by_scene,
        )
        p3_selected.update(selected)
        p3_summary[f"k{k}"] = summary

    _apply_selection(p2, p2_selected, {"reference_frame_transform"})
    _apply_selection(
        p3,
        p3_selected,
        {f"cross_view_ego_k{k}" for k in (1, 2, 3)},
    )

    for path, backup, manifest in (
        (args.p2, p2_backup, p2),
        (args.p3, p3_backup, p3),
    ):
        if not backup.exists():
            shutil.copy2(path, backup)
        _write(path, manifest)

    report = {
        "schema_version": "p23_7k_search_finalization.v1",
        "status": "pass",
        "contract": str(args.contract.resolve()),
        "physical_trajectories": int(contract["physical_trajectory_target"]),
        "actual_allocation": {"p2": p2_count, "p3": p3_targets},
        "unique_pose_fingerprints_per_scene": len(global_fingerprints),
        "p2": p2_summary,
        "p3": p3_summary,
    }
    _write(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
