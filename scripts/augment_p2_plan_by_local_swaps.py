#!/usr/bin/env python
"""Augment a quality-gated P2 plan with deterministic one-to-two local swaps."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spatial_episode.scriptgen.binding_coverage import trajectories_are_diverse
from spatial_episode.scriptgen.plan import TrajectoryPlan


@dataclass(frozen=True)
class Entry:
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


def _fingerprint(plan: TrajectoryPlan) -> str:
    poses = [
        (pose.frame, round(pose.x, 6), round(pose.y, 6), round(pose.yaw_deg, 6))
        for pose in plan.poses
    ]
    encoded = json.dumps(poses, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _entry(cell: dict[str, Any], candidate: dict[str, Any]) -> Entry:
    plan = TrajectoryPlan.model_validate_json(
        Path(candidate["plan_record"]).read_text(encoding="utf-8")
    )
    return Entry(cell=cell, candidate=candidate, plan=plan, fingerprint=_fingerprint(plan))


def _producer_cells(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        cell
        for cell in manifest["cells"]
        if cell["capability"] == "reference_frame_transform"
    ]


def _apply_selection(
    manifest: dict[str, Any], selected: dict[str, list[dict[str, Any]]]
) -> None:
    producers = {
        _key(cell): cell["cell_id"]
        for cell in manifest["cells"]
        if cell["capability"] == "reference_frame_transform"
        and cell["cell_id"] in selected
    }
    cells: list[dict[str, Any]] = []
    for original in manifest["cells"]:
        producer_id = producers.get(_key(original))
        if producer_id is None:
            continue
        candidates = selected[producer_id]
        cell = dict(original)
        cell["target_accepted"] = len(candidates)
        cell["candidates"] = candidates if cell["cell_id"] == producer_id else []
        cells.append(cell)
    manifest["cells"] = cells


def _pair_respects_cap(
    left: Entry,
    right: Entry,
    victim: Entry,
    counts: collections.Counter[str],
    cap: int,
) -> bool:
    after = counts.copy()
    after[victim.cell["cell_id"]] -= 1
    after[left.cell["cell_id"]] += 1
    after[right.cell["cell_id"]] += 1
    return max(after.values(), default=0) <= cap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--p3", type=Path, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--cap", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    current = _read(args.p2)
    pool = _read(args.pool)
    p3 = _read(args.p3)
    pool_cells = {cell["cell_id"]: cell for cell in _producer_cells(pool)}

    selected: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    entries_by_id: dict[str, Entry] = {}
    for cell in _producer_cells(current):
        pool_cell = pool_cells[cell["cell_id"]]
        for candidate in cell["candidates"]:
            item = _entry(pool_cell, candidate)
            selected[cell["cell_id"]].append(candidate)
            entries_by_id[candidate["candidate_id"]] = item

    p3_fingerprints: set[tuple[str, str]] = set()
    for cell in p3["cells"]:
        if not cell["capability"].startswith("cross_view_ego_k"):
            continue
        for candidate in cell["candidates"]:
            plan = TrajectoryPlan.model_validate_json(
                Path(candidate["plan_record"]).read_text(encoding="utf-8")
            )
            p3_fingerprints.add((cell["scene_key"], _fingerprint(plan)))

    pool_entries = [
        _entry(cell, candidate)
        for cell in sorted(pool_cells.values(), key=lambda row: row["cell_id"])
        for candidate in cell["candidates"]
    ]
    swaps: list[dict[str, Any]] = []
    while len(entries_by_id) < args.target:
        selected_by_scene: dict[str, list[Entry]] = collections.defaultdict(list)
        for item in entries_by_id.values():
            selected_by_scene[item.cell["scene_key"]].append(item)
        counts = collections.Counter(
            item.cell["cell_id"] for item in entries_by_id.values()
        )
        groups: dict[str, list[Entry]] = collections.defaultdict(list)
        for option in pool_entries:
            candidate_id = option.candidate["candidate_id"]
            if candidate_id in entries_by_id:
                continue
            scene_key = option.cell["scene_key"]
            if (scene_key, option.fingerprint) in p3_fingerprints:
                continue
            conflicts = [
                prior
                for prior in selected_by_scene[scene_key]
                if option.fingerprint == prior.fingerprint
                or not trajectories_are_diverse(option.plan, prior.plan)
            ]
            if counts[option.cell["cell_id"]] >= args.cap:
                same_cell = [
                    prior
                    for prior in selected_by_scene[scene_key]
                    if prior.cell["cell_id"] == option.cell["cell_id"]
                ]
                if not any(prior in conflicts for prior in same_cell):
                    # A one-removal move can make room by evicting any member
                    # of this full binding, provided it is the only conflict.
                    if not conflicts:
                        for prior in same_cell:
                            groups[prior.candidate["candidate_id"]].append(option)
                    continue
            if len(conflicts) == 1:
                groups[conflicts[0].candidate["candidate_id"]].append(option)

        chosen: tuple[Entry, Entry, Entry] | None = None
        for victim_id in sorted(groups):
            victim = entries_by_id[victim_id]
            options = groups[victim_id]
            for left_index, left in enumerate(options):
                for right in options[left_index + 1 :]:
                    if left.candidate["candidate_id"] == right.candidate["candidate_id"]:
                        continue
                    if left.fingerprint == right.fingerprint:
                        continue
                    if not trajectories_are_diverse(left.plan, right.plan):
                        continue
                    if not _pair_respects_cap(left, right, victim, counts, args.cap):
                        continue
                    chosen = victim, left, right
                    break
                if chosen is not None:
                    break
            if chosen is not None:
                break
        if chosen is None:
            break

        victim, left, right = chosen
        victim_id = victim.candidate["candidate_id"]
        selected[victim.cell["cell_id"]] = [
            candidate
            for candidate in selected[victim.cell["cell_id"]]
            if candidate["candidate_id"] != victim_id
        ]
        del entries_by_id[victim_id]
        for replacement in (left, right):
            replacement_id = replacement.candidate["candidate_id"]
            selected[replacement.cell["cell_id"]].append(replacement.candidate)
            entries_by_id[replacement_id] = replacement
        swaps.append(
            {
                "scene_key": victim.cell["scene_key"],
                "removed_candidate_id": victim_id,
                "added_candidate_ids": [
                    left.candidate["candidate_id"],
                    right.candidate["candidate_id"],
                ],
            }
        )
        print(
            f"local swap scene={victim.cell['scene_key']} "
            f"selected={len(entries_by_id)}/{args.target}",
            flush=True,
        )

    result = dict(pool)
    _apply_selection(result, dict(selected))
    _write(args.output, result)
    report = {
        "schema_version": "p2_local_swap_augmentation.v1",
        "status": "pass" if len(entries_by_id) == args.target else "fail",
        "initial_trajectories": sum(
            len(cell["candidates"]) for cell in _producer_cells(current)
        ),
        "final_trajectories": len(entries_by_id),
        "target": args.target,
        "swap_count": len(swaps),
        "swaps": swaps,
        "output": str(args.output.resolve()),
    }
    _write(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
