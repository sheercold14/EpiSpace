#!/usr/bin/env python3
"""Plan label-aware side-sector repairs for pure rotation / translation.

The immutable canonical collection supplies the binding inventory and the old
``back`` examples.  This overlay generates only new ``left`` and ``right``
plans.  Each capability / desired-label pair has an independent manifest so a
label cannot fill another label's quota.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from spatial_episode.scriptgen.binding_coverage import load_coverage, plan_coverage


TASKS = (
    ("self_motion_update_pure_rotation", "left"),
    ("self_motion_update_pure_rotation", "right"),
    ("self_motion_update_pure_translation", "left"),
    ("self_motion_update_pure_translation", "right"),
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _binding_key(binding: dict[str, str]) -> str:
    return json.dumps(binding, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _active_bindings(
    source_manifest: dict[str, Any],
    source_status: dict[str, Any],
) -> tuple[dict[str, dict[str, list[dict[str, str]]]], dict[str, Any]]:
    result: dict[str, dict[str, list[dict[str, str]]]] = {}
    summary: dict[str, Any] = {}
    episodes = source_status.get("episodes", {})
    for capability, _ in TASKS:
        if capability in result:
            continue
        by_scene: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
        source_counts: Counter[int] = Counter()
        source_episode_count = 0
        for cell in source_manifest["cells"]:
            if cell["capability"] != capability:
                continue
            accepted = source_status["cells"][cell["cell_id"]]["accepted_episode_ids"]
            source_accepted = [
                episode_id
                for episode_id in accepted
                if episodes.get(episode_id, {}).get("source_capability") == capability
            ]
            if not source_accepted:
                continue
            binding = dict(cell["binding"])
            by_scene[cell["scene_key"]][_binding_key(binding)] = binding
            source_counts[len(source_accepted)] += 1
            source_episode_count += len(source_accepted)
        bindings_by_scene = {
            scene: [mapping[key] for key in sorted(mapping)]
            for scene, mapping in sorted(by_scene.items())
        }
        result[capability] = bindings_by_scene
        summary[capability] = {
            "binding_count": sum(len(items) for items in bindings_by_scene.values()),
            "scene_count": len(bindings_by_scene),
            "old_back_source_episode_count": source_episode_count,
            "old_source_count_histogram": {
                str(count): cells for count, cells in sorted(source_counts.items())
            },
        }
    return result, summary


def build(
    *,
    source_root: Path,
    output_root: Path,
    accepted_per_label: int,
    attempts_per_binding: int,
    initial_attempts_per_binding: int,
) -> Path:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    manifest_path = source_root / "coverage.plan.json"
    status_path = source_root / "coverage.status.json"
    ledger_path = source_root / "coverage.shard_merge.json"
    ledger = _read(ledger_path)
    manifest_sha = _sha256(manifest_path)
    status_sha = _sha256(status_path)
    if ledger.get("current_manifest_sha256") != manifest_sha:
        raise ValueError("canonical manifest differs from immutable merge ledger")
    if ledger.get("current_status_sha256") != status_sha:
        raise ValueError("canonical status differs from immutable merge ledger")

    source_manifest = _read(manifest_path)
    source_status = _read(status_path)
    bindings, binding_summary = _active_bindings(source_manifest, source_status)
    output_root.mkdir(parents=True, exist_ok=True)

    allowlist_paths: dict[str, str] = {}
    for capability, bindings_by_scene in bindings.items():
        short = capability.removeprefix("self_motion_update_")
        path = output_root / f"{short}.bindings.json"
        _write(
            path,
            {
                "schema_version": "epispace.direction_balance_binding_allowlist.v1",
                "source_manifest_sha256": manifest_sha,
                "source_status_sha256": status_sha,
                "capability": capability,
                "bindings_by_scene": bindings_by_scene,
            },
        )
        allowlist_paths[capability] = str(path)

    task_rows = []
    for capability, label in TASKS:
        short = capability.removeprefix("self_motion_update_")
        task_root = output_root / f"{short}_{label}"
        collection_id = f"behavior51_direction_balance_v1__{short}__{label}"
        coverage_path = plan_coverage(
            source_index_path=Path(source_manifest["source_index"]),
            output_root=task_root,
            collection_id=collection_id,
            accepted_per_binding=accepted_per_label,
            attempts_per_binding=attempts_per_binding,
            initial_attempts_per_binding=initial_attempts_per_binding,
            capabilities=(capability,),
            scene_keys=tuple(bindings[capability]),
            binding_allowlist=bindings[capability],
            desired_answer_label=label,
        )
        planned = load_coverage(coverage_path)
        observed = Counter()
        zero_candidate_cells = 0
        for cell in planned.cells:
            if cell.desired_answer_label != label:
                raise ValueError(f"cell lost desired label: {cell.cell_id}")
            if not cell.candidates:
                zero_candidate_cells += 1
            for candidate in cell.candidates:
                plan_payload = _read(Path(candidate.plan_record))
                observed[plan_payload["provisional_answer"]["label"]] += 1
        if set(observed) - {label}:
            raise ValueError(
                f"task {capability}/{label} contains wrong geometry labels: {dict(observed)}"
            )
        task_rows.append(
            {
                "capability": capability,
                "desired_answer_label": label,
                "manifest": str(coverage_path),
                "output_root": str(task_root),
                "cell_count": len(planned.cells),
                "initial_candidate_count": sum(
                    len(cell.candidates) for cell in planned.cells
                ),
                "zero_candidate_cell_count": zero_candidate_cells,
                "search_attempt_limit": initial_attempts_per_binding,
                "maximum_attempts_per_binding": attempts_per_binding,
                "target_accepted_per_binding": accepted_per_label,
                "observed_geometry_labels": dict(sorted(observed.items())),
            }
        )

    index_path = output_root / "direction_balance.plan.json"
    _write(
        index_path,
        {
            "schema_version": "epispace.behavior51_direction_balance_plan.v1",
            "collection_id": "behavior51_direction_balance_v1",
            "source_root": str(source_root),
            "source_manifest_sha256": manifest_sha,
            "source_status_sha256": status_sha,
            "source_applied_shards": ledger.get("applied_shards", []),
            "old_pixels_mutated": False,
            "old_back_policy": (
                "retain at most four old back episodes per binding at final release; "
                "do not relabel or rerender them"
            ),
            "binding_summary": binding_summary,
            "allowlists": allowlist_paths,
            "tasks": task_rows,
        },
    )
    return index_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("outputs/behavior51_coverage_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/behavior51_direction_balance_v1"),
    )
    parser.add_argument("--accepted-per-label", type=int, default=3)
    parser.add_argument("--attempts-per-binding", type=int, default=150)
    parser.add_argument("--initial-attempts-per-binding", type=int, default=30)
    args = parser.parse_args()
    print(
        build(
            source_root=args.source,
            output_root=args.output,
            accepted_per_label=args.accepted_per_label,
            attempts_per_binding=args.attempts_per_binding,
            initial_attempts_per_binding=args.initial_attempts_per_binding,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
