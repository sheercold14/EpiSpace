#!/usr/bin/env python3
"""Rebase audited repair data onto the immutable full behavior51 v1 snapshot.

This prepares a *metadata overlay*.  It never edits v1 and it never copies or
rewrites rendered media.  Accepted v2r1 and direction-balance episodes are
reused by reference, credits are recomputed against the full-v1 quotas, and
only the deficits that remain are exposed to the v2r2 planner.

The redesigned occlusion collection is recorded as a separate import because
``self_motion_update_after_occlusion`` intentionally does not claim credit for
the deprecated ``self_motion_update_occluded`` cells.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from spatial_episode.scriptgen.binding_coverage import trajectories_are_diverse
from spatial_episode.scriptgen.plan import TrajectoryPlan


RECOMPILE_ACTIONS = frozenset(
    {
        "fix_delay_variant_then_recompile_bundle",
        "recompile_existing_success_bundle",
    }
)
RERENDER_ACTIONS = frozenset({"retry_same_candidate_after_runtime_cleanup"})
READY_ACTIONS = RECOMPILE_ACTIONS | RERENDER_ACTIONS
HELD_CAPABILITIES = frozenset({"self_motion_update_occluded"})


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_index(
    manifest: dict[str, Any],
) -> dict[str, tuple[str, dict[str, Any]]]:
    result: dict[str, tuple[str, dict[str, Any]]] = {}
    for cell in manifest["cells"]:
        for candidate in cell.get("candidates", []):
            candidate_id = candidate["candidate_id"]
            if candidate_id in result:
                raise ValueError(f"duplicate candidate id in manifest: {candidate_id}")
            result[candidate_id] = (cell["cell_id"], candidate)
    return result


def _load_collection(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = _read(root / "coverage.plan.json")
    status = _read(root / "coverage.status.json")
    episodes = status.get("episodes", {})
    return {
        "root": root,
        "manifest": manifest,
        "status": status,
        "episodes": episodes,
        "candidate_index": _candidate_index(manifest),
    }


def _plan(path: str, cache: dict[str, TrajectoryPlan]) -> TrajectoryPlan:
    if path not in cache:
        cache[path] = TrajectoryPlan.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    return cache[path]


def _append_unique_candidate(
    *,
    final_cell: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    candidate_id = candidate["candidate_id"]
    existing = {
        row["candidate_id"]: row for row in final_cell.get("candidates", [])
    }.get(candidate_id)
    if existing is not None:
        if existing["plan_id"] != candidate["plan_id"]:
            raise ValueError(
                f"candidate collision with different plan: {candidate_id}"
            )
        return
    final_cell.setdefault("candidates", []).append(candidate)


def _candidate_is_diverse(
    *,
    candidate_plan: TrajectoryPlan,
    accepted_ids: list[str],
    final_candidates: dict[str, tuple[str, dict[str, Any]]],
    plan_cache: dict[str, TrajectoryPlan],
) -> bool:
    for episode_id in accepted_ids:
        indexed = final_candidates.get(episode_id)
        if indexed is None:
            # Every imported accepted episode is inserted into the manifest.
            # A missing immutable-v1 candidate is an integrity error, not a
            # reason to silently weaken the diversity check.
            raise ValueError(f"accepted episode has no manifest candidate: {episode_id}")
        prior = _plan(indexed[1]["plan_record"], plan_cache)
        if not trajectories_are_diverse(candidate_plan, prior):
            return False
    return True


def _import_episodes(
    *,
    tag: str,
    collection: dict[str, Any],
    episode_ids: list[str],
    namespace_ids: bool,
    final_manifest: dict[str, Any],
    final_status: dict[str, Any],
    eligible_credit_ids: set[str],
    plan_cache: dict[str, TrajectoryPlan],
) -> dict[str, Any]:
    cells = {cell["cell_id"]: cell for cell in final_manifest["cells"]}
    final_candidates = _candidate_index(final_manifest)
    imported_ids: list[str] = []
    redundant_ids: list[str] = []
    skipped_missing_source: list[str] = []
    credits_by_capability: Counter[str] = Counter()

    for original_id in sorted(episode_ids):
        episode = collection["episodes"].get(original_id)
        indexed = collection["candidate_index"].get(original_id)
        if episode is None or indexed is None:
            skipped_missing_source.append(original_id)
            continue
        source_cell_id, source_candidate = indexed
        final_cell = cells.get(source_cell_id)
        if final_cell is None:
            skipped_missing_source.append(original_id)
            continue

        imported_id = (
            f"{original_id}__import__{tag}" if namespace_ids else original_id
        )
        candidate = copy.deepcopy(source_candidate)
        candidate["candidate_id"] = imported_id
        candidate_plan = _plan(candidate["plan_record"], plan_cache)

        credited: list[str] = []
        for cell_id in episode.get("credited_cells", []):
            if cell_id not in eligible_credit_ids or cell_id not in cells:
                continue
            target = cells[cell_id]
            row = final_status["cells"][cell_id]
            if len(row["accepted_episode_ids"]) >= int(target["target_accepted"]):
                continue
            if _candidate_is_diverse(
                candidate_plan=candidate_plan,
                accepted_ids=row["accepted_episode_ids"],
                final_candidates=final_candidates,
                plan_cache=plan_cache,
            ):
                row["accepted_episode_ids"].append(imported_id)
                row["status"] = (
                    "complete"
                    if len(row["accepted_episode_ids"])
                    >= int(target["target_accepted"])
                    else "planned"
                )
                credited.append(cell_id)
                credits_by_capability[target["capability"]] += 1

        if not credited:
            redundant_ids.append(original_id)
            continue

        _append_unique_candidate(final_cell=final_cell, candidate=candidate)
        final_candidates[imported_id] = (source_cell_id, candidate)
        source_row = final_status["cells"][source_cell_id]
        source_row["candidate_statuses"][imported_id] = {
            "status": "accepted",
            "reason": f"imported_repair:{tag}",
            "credited_cells": credited,
        }
        final_status["episodes"][imported_id] = {
            **copy.deepcopy(episode),
            "credited_cells": credited,
            "import_source": tag,
            "original_episode_id": original_id,
        }
        imported_ids.append(imported_id)

    return {
        "tag": tag,
        "source_root": str(collection["root"]),
        "eligible_episode_count": len(episode_ids),
        "credited_episode_count": len(imported_ids),
        "redundant_episode_count": len(redundant_ids),
        "missing_source_count": len(skipped_missing_source),
        "credited_ids": imported_ids,
        "credits_by_capability": dict(sorted(credits_by_capability.items())),
        "missing_source_ids": skipped_missing_source,
    }


def _missing_slots(
    cell_id: str,
    cells: dict[str, dict[str, Any]],
    status: dict[str, Any],
) -> int:
    return max(
        0,
        int(cells[cell_id]["target_accepted"])
        - len(status["cells"][cell_id]["accepted_episode_ids"]),
    )


def _producer_can_serve(
    producer: dict[str, Any], target: dict[str, Any]
) -> bool:
    return producer["scene_key"] == target["scene_key"] and all(
        producer["binding"].get(name) == value
        for name, value in target["binding"].items()
    )


def prepare(
    *,
    source_root: Path,
    audit_root: Path,
    prior_repair_root: Path,
    direction_roots: list[Path],
    occlusion_root: Path | None,
    output_root: Path,
    collection_id: str,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    audit_root = audit_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty overlay: {output_root}")

    manifest_path = source_root / "coverage.plan.json"
    status_path = source_root / "coverage.status.json"
    ledger = _read(source_root / "coverage.shard_merge.json")
    manifest_sha = _sha256(manifest_path)
    status_sha = _sha256(status_path)
    if ledger["current_manifest_sha256"] != manifest_sha:
        raise ValueError("source manifest differs from its immutable merge ledger")
    if ledger["current_status_sha256"] != status_sha:
        raise ValueError("source status differs from its immutable merge ledger")

    repair_cells = _read(audit_root / "local_repair_cell_ids.json")
    if repair_cells["source_manifest_sha256"] != manifest_sha:
        raise ValueError("repair-cell audit was built from another manifest")
    if repair_cells["source_status_sha256"] != status_sha:
        raise ValueError("repair-cell audit was built from another status")

    manifest = copy.deepcopy(_read(manifest_path))
    status = copy.deepcopy(_read(status_path))
    source_episode_ids = set(status["episodes"])
    cells = {cell["cell_id"]: cell for cell in manifest["cells"]}
    candidate_index = _candidate_index(manifest)

    quarantine = _read_jsonl(audit_root / "quarantine.jsonl")
    rejected = _read_jsonl(audit_root / "rejected_candidates.jsonl")
    quarantined_ids = {row["episode_id"] for row in quarantine}
    unknown_quarantine = quarantined_ids - set(candidate_index)
    if unknown_quarantine:
        raise ValueError(
            "quarantined episodes have no manifest candidate: "
            + ", ".join(sorted(unknown_quarantine)[:10])
        )

    # Start from full immutable v1 and revoke every direct and derived credit
    # carried by the final 891-episode quarantine.
    for cell_id, row in status["cells"].items():
        row["accepted_episode_ids"] = [
            item for item in row["accepted_episode_ids"] if item not in quarantined_ids
        ]
        row["status"] = (
            "complete"
            if len(row["accepted_episode_ids"]) >= int(cells[cell_id]["target_accepted"])
            else "planned"
        )
    for episode_id in quarantined_ids:
        status["episodes"].pop(episode_id, None)
        source_cell_id = candidate_index[episode_id][0]
        status["cells"][source_cell_id]["candidate_statuses"][episode_id] = {
            "status": "rejected",
            "reason": "v2_final_quarantine_revoke_all_derived_credit",
        }

    marginal_ids = {
        row["cell_id"] for row in repair_cells["quarantine_marginal_deficits"]
    }
    ready_credit_ids = {
        cell_id
        for cell_id in marginal_ids
        if cells[cell_id]["capability"] not in HELD_CAPABILITIES
    }

    plan_cache: dict[str, TrajectoryPlan] = {}
    imports: list[dict[str, Any]] = []

    prior = _load_collection(prior_repair_root)
    prior_episode_ids = sorted(set(prior["episodes"]) - source_episode_ids)
    imports.append(
        _import_episodes(
            tag="v2r1",
            collection=prior,
            episode_ids=prior_episode_ids,
            namespace_ids=False,
            final_manifest=manifest,
            final_status=status,
            eligible_credit_ids=ready_credit_ids,
            plan_cache=plan_cache,
        )
    )

    # Direction collections intentionally reuse raw candidate ids across
    # left/right tasks.  Collection-qualified ids make their final identities
    # unambiguous while retaining the original media paths.
    for root in direction_roots:
        collection = _load_collection(root)
        quarantine_ids = {
            row["episode_id"]
            for row in _read_jsonl(root.resolve() / "audit" / "quarantine.jsonl")
        }
        eligible = sorted(set(collection["episodes"]) - quarantine_ids)
        tag = f"direction_{root.name}"
        imports.append(
            _import_episodes(
                tag=tag,
                collection=collection,
                episode_ids=eligible,
                namespace_ids=True,
                final_manifest=manifest,
                final_status=status,
                eligible_credit_ids=ready_credit_ids,
                plan_cache=plan_cache,
            )
        )

    # Rebuild indexes after imports before selecting remaining producers.
    cells = {cell["cell_id"]: cell for cell in manifest["cells"]}
    candidate_index = _candidate_index(manifest)

    action_by_candidate = {
        row["candidate_id"]: row["repair_action"] for row in rejected
    }
    ready_candidate_ids = {
        row["candidate_id"]
        for row in rejected
        if row["repair_action"] in READY_ACTIONS
        and cells[row["cell_id"]]["capability"] not in HELD_CAPABILITIES
        and row["candidate_id"] not in status["episodes"]
    }
    ready_credit_ids.update(
        candidate_index[candidate_id][0]
        for candidate_id in ready_candidate_ids
        if candidate_id in candidate_index
    )
    remaining_credit_ids = {
        cell_id
        for cell_id in ready_credit_ids
        if _missing_slots(cell_id, cells, status) > 0
    }

    producer_pool = {
        candidate_index[row["episode_id"]][0]
        for row in quarantine
        if row["source_capability"] not in HELD_CAPABILITIES
    }
    producer_pool.update(
        candidate_index[candidate_id][0]
        for candidate_id in ready_candidate_ids
        if candidate_id in candidate_index
    )
    producer_cell_ids = {
        producer_id
        for producer_id in producer_pool
        if any(
            _producer_can_serve(cells[producer_id], cells[target_id])
            for target_id in remaining_credit_ids
        )
    }

    manifest["collection_id"] = collection_id
    manifest["output_root"] = str(output_root)
    status["collection_id"] = collection_id

    # Existing pending candidates came from the immutable v1 proposal round.
    # Fresh v2r2 candidates use a new seed and namespace, but imported accepted
    # candidates remain available for diversity checks.
    for cell_id in producer_cell_ids:
        row = status["cells"][cell_id]
        for candidate_id, candidate_status in row["candidate_statuses"].items():
            if candidate_status["status"] == "pending":
                candidate_status.update(
                    status="superseded",
                    reason="v2r2_incremental_requires_fresh_planner",
                )
        cell = cells[cell_id]
        repair_seed = int(
            hashlib.sha256(
                f"{collection_id}\0{cell_id}\0v2r2".encode("utf-8")
            ).hexdigest()[:8],
            16,
        ) & 0x7FFFFFFF
        cell.update(
            seed=repair_seed,
            candidate_namespace="__v2r2",
            geometry_pool_size=0,
            diverse_pool_size=0,
            search_attempt_limit=0,
            raw_plan_limit=0,
            search_pool_exhausted=False,
        )

    # Recompile candidates retain immutable pixels and write only overlay group
    # metadata. Runtime retries write all outputs under v2r2.
    for candidate_id in ready_candidate_ids:
        indexed = candidate_index.get(candidate_id)
        if indexed is None:
            continue
        cell_id, candidate = indexed
        action = action_by_candidate[candidate_id]
        candidate["group"] = str(output_root / "groups" / candidate_id)
        if action in RERENDER_ACTIONS:
            candidate["bundle"] = str(output_root / "bundles" / candidate_id)
            candidate["log"] = str(output_root / "logs" / f"{candidate_id}.log")
        status["cells"][cell_id]["candidate_statuses"][candidate_id] = {
            "status": "pending",
            "reason": f"v2r2_ready:{action}",
        }

    deficit_rows = [
        {
            "cell_id": cell_id,
            "scene_key": cells[cell_id]["scene_key"],
            "capability": cells[cell_id]["capability"],
            "binding": cells[cell_id]["binding"],
            "target_accepted": cells[cell_id]["target_accepted"],
            "accepted_after_import": len(
                status["cells"][cell_id]["accepted_episode_ids"]
            ),
            "missing_slots": _missing_slots(cell_id, cells, status),
        }
        for cell_id in sorted(remaining_credit_ids)
    ]
    held_ids = sorted(
        cell_id
        for cell_id in marginal_ids
        if cells[cell_id]["capability"] in HELD_CAPABILITIES
        and _missing_slots(cell_id, cells, status) > 0
    )

    occlusion_import: dict[str, Any] | None = None
    if occlusion_root is not None:
        occlusion_root = occlusion_root.resolve()
        dataset = _read(occlusion_root / "dataset.json")
        occlusion_import = {
            "tag": "redesigned_occlusion",
            "source_root": str(occlusion_root),
            "episode_count": len(dataset.get("episodes", [])),
            "credit_policy": (
                "external_training_collection; does_not_credit_deprecated_"
                "self_motion_update_occluded_cells"
            ),
        }

    output_root.mkdir(parents=True, exist_ok=True)
    _write(output_root / "coverage.plan.json", manifest)
    _write(output_root / "coverage.status.json", status)
    _write(
        output_root / "producer_cell_ids.json",
        {
            "schema_version": "epispace.behavior51_v2r2_producers.v1",
            "cell_ids": sorted(producer_cell_ids),
        },
    )
    _write(
        output_root / "credit_cell_ids.json",
        {
            "schema_version": "epispace.behavior51_v2r2_credit_targets.v1",
            "cell_ids": sorted(remaining_credit_ids),
        },
    )
    _write(
        output_root / "held_cell_ids.json",
        {
            "schema_version": "epispace.behavior51_v2r2_held_cells.v1",
            "cell_ids": held_ids,
            "reason": (
                "deprecated occluded terminal definition; replaced by the separate "
                "self_motion_update_after_occlusion collection"
            ),
        },
    )
    _write(
        output_root / "remaining_deficits.json",
        {
            "schema_version": "epispace.behavior51_v2r2_deficits.v1",
            "cell_count": len(deficit_rows),
            "missing_slot_count": sum(row["missing_slots"] for row in deficit_rows),
            "by_capability": dict(
                sorted(Counter(row["capability"] for row in deficit_rows).items())
            ),
            "cells": deficit_rows,
        },
    )
    _write(
        output_root / "imported_sources.json",
        {
            "schema_version": "epispace.behavior51_v2r2_imports.v1",
            "coverage_imports": imports,
            "occlusion_import": occlusion_import,
        },
    )

    provenance = {
        "schema_version": "epispace.behavior51_v2r2_overlay.v1",
        "collection_id": collection_id,
        "source_root": str(source_root),
        "source_manifest_sha256": manifest_sha,
        "source_status_sha256": status_sha,
        "source_applied_shards": ledger["applied_shards"],
        "quarantined_episode_count": len(quarantined_ids),
        "retained_v1_episode_count": len(source_episode_ids - quarantined_ids),
        "imported_credited_episode_count": sum(
            row["credited_episode_count"] for row in imports
        ),
        "producer_cell_count": len(producer_cell_ids),
        "remaining_credit_cell_count": len(remaining_credit_ids),
        "remaining_missing_slot_count": sum(
            _missing_slots(cell_id, cells, status)
            for cell_id in remaining_credit_ids
        ),
        "held_deprecated_occlusion_cell_count": len(held_ids),
        "recompile_candidate_count": sum(
            action_by_candidate[item] in RECOMPILE_ACTIONS
            for item in ready_candidate_ids
        ),
        "rerender_same_plan_candidate_count": sum(
            action_by_candidate[item] in RERENDER_ACTIONS
            for item in ready_candidate_ids
        ),
        "candidate_namespace": "__v2r2",
        "v1_mutated": False,
    }
    _write(output_root / "overlay.provenance.json", provenance)
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--prior-repair", type=Path, required=True)
    parser.add_argument("--direction-root", type=Path, action="append", default=[])
    parser.add_argument("--occlusion-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--collection-id", default="behavior51_coverage_v2r2_incremental"
    )
    args = parser.parse_args()
    provenance = prepare(
        source_root=args.source,
        audit_root=args.audit,
        prior_repair_root=args.prior_repair,
        direction_roots=args.direction_root,
        occlusion_root=args.occlusion_root,
        output_root=args.output,
        collection_id=args.collection_id,
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
