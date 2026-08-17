#!/usr/bin/env python3
"""Fork an immutable behavior51 v1 snapshot into a local-only v2 repair overlay.

The overlay copies only plan/status metadata.  Healthy accepted media and
already-rendered reusable candidates remain referenced from v1; every candidate
that may write pixels or question groups is redirected to the new output root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


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
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def prepare(
    *,
    source_root: Path,
    audit_root: Path,
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

    manifest = _read(manifest_path)
    status = _read(status_path)
    repair_cells = _read(audit_root / "local_repair_cell_ids.json")
    if repair_cells["source_manifest_sha256"] != manifest_sha:
        raise ValueError("repair-cell audit was built from another manifest")
    if repair_cells["source_status_sha256"] != status_sha:
        raise ValueError("repair-cell audit was built from another status")

    quarantine = _read_jsonl(audit_root / "quarantine.jsonl")
    rejected = _read_jsonl(audit_root / "rejected_candidates.jsonl")
    quarantined_ids = {row["episode_id"] for row in quarantine}
    action_by_candidate = {row["candidate_id"]: row["repair_action"] for row in rejected}
    cells = {cell["cell_id"]: cell for cell in manifest["cells"]}
    candidate_cell: dict[str, str] = {}
    candidate_rows: dict[str, dict[str, Any]] = {}
    for cell in manifest["cells"]:
        for candidate in cell.get("candidates", ()):
            candidate_cell[candidate["candidate_id"]] = cell["cell_id"]
            candidate_rows[candidate["candidate_id"]] = candidate

    unknown_quarantine = quarantined_ids - set(candidate_cell)
    if unknown_quarantine:
        raise ValueError(
            "quarantined episodes have no manifest candidate: "
            + ", ".join(sorted(unknown_quarantine)[:10])
        )

    marginal_targets = {
        row["cell_id"] for row in repair_cells["quarantine_marginal_deficits"]
    }
    ready_target_ids = {
        cell_id
        for cell_id in marginal_targets
        if cells[cell_id]["capability"] not in HELD_CAPABILITIES
    }
    ready_candidate_ids = {
        row["candidate_id"]
        for row in rejected
        if row["repair_action"] in READY_ACTIONS
        and cells[row["cell_id"]]["capability"] not in HELD_CAPABILITIES
    }
    ready_target_ids.update(candidate_cell[item] for item in ready_candidate_ids)

    producer_cell_ids = {
        candidate_cell[row["episode_id"]]
        for row in quarantine
        if row["source_capability"] not in HELD_CAPABILITIES
    }
    producer_cell_ids.update(candidate_cell[item] for item in ready_candidate_ids)

    manifest["collection_id"] = collection_id
    manifest["output_root"] = str(output_root)
    status["collection_id"] = collection_id

    # Revoke every derived credit carried by a quarantined source episode.
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
        source_cell = candidate_cell[episode_id]
        status["cells"][source_cell]["candidate_statuses"][episode_id] = {
            "status": "rejected",
            "reason": "v2_quarantine_revoke_all_derived_credit",
        }

    # Existing pending plans on repaired producer cells were generated under
    # the audited v1 proposal policy.  Do not spend GPU time on them; fresh
    # backfill uses the corrected non-degenerate and direction-aware motifs.
    for cell_id in producer_cell_ids:
        row = status["cells"][cell_id]
        for candidate_id, candidate_status in row["candidate_statuses"].items():
            if candidate_status["status"] == "pending":
                candidate_status.update(
                    status="superseded",
                    reason="v2_repair_requires_fresh_planner",
                )
        cell = cells[cell_id]
        repair_seed = int(
            hashlib.sha256(
                f"{collection_id}\0{cell_id}\0v2r1".encode("utf-8")
            ).hexdigest()[:8],
            16,
        ) & 0x7FFFFFFF
        cell.update(
            seed=repair_seed,
            candidate_namespace="__v2r1",
            geometry_pool_size=0,
            diverse_pool_size=0,
            search_attempt_limit=0,
            raw_plan_limit=0,
            search_pool_exhausted=False,
        )

    # Recompile candidates reuse immutable v1 pixels and write only new group
    # metadata. Runtime retries write their complete output into v2.
    for candidate_id in ready_candidate_ids:
        cell_id = candidate_cell[candidate_id]
        action = action_by_candidate[candidate_id]
        candidate = candidate_rows[candidate_id]
        candidate["group"] = str(output_root / "groups" / candidate_id)
        if action in RERENDER_ACTIONS:
            candidate["bundle"] = str(output_root / "bundles" / candidate_id)
            candidate["log"] = str(output_root / "logs" / f"{candidate_id}.log")
        status["cells"][cell_id]["candidate_statuses"][candidate_id] = {
            "status": "pending",
            "reason": f"v2_ready:{action}",
        }

    output_root.mkdir(parents=True, exist_ok=True)
    _write(output_root / "coverage.plan.json", manifest)
    _write(output_root / "coverage.status.json", status)
    _write(
        output_root / "producer_cell_ids.json",
        {
            "schema_version": "epispace.behavior51_v2_local_producers.v1",
            "cell_ids": sorted(producer_cell_ids),
        },
    )
    _write(
        output_root / "credit_cell_ids.json",
        {
            "schema_version": "epispace.behavior51_v2_local_credit_targets.v1",
            "cell_ids": sorted(ready_target_ids),
        },
    )
    all_repair_ids = set(repair_cells["cell_ids"])
    _write(
        output_root / "held_cell_ids.json",
        {
            "schema_version": "epispace.behavior51_v2_local_held_cells.v1",
            "cell_ids": sorted(all_repair_ids - ready_target_ids),
            "reason": (
                "requires occluded event redesign, authoritative traversability, binding "
                "replacement, or semantic reserve planner before rendering"
            ),
        },
    )
    provenance = {
        "schema_version": "epispace.behavior51_v2_local_overlay.v1",
        "collection_id": collection_id,
        "source_root": str(source_root),
        "source_collection_id": ledger.get("collection_id", "behavior51_coverage_v1"),
        "source_manifest_sha256": manifest_sha,
        "source_status_sha256": status_sha,
        "source_applied_shards": ledger["applied_shards"],
        "quarantined_episode_count": len(quarantined_ids),
        "producer_cell_count": len(producer_cell_ids),
        "ready_credit_cell_count": len(ready_target_ids),
        "held_repair_cell_count": len(all_repair_ids - ready_target_ids),
        "recompile_candidate_count": sum(
            action_by_candidate[item] in RECOMPILE_ACTIONS for item in ready_candidate_ids
        ),
        "rerender_same_plan_candidate_count": sum(
            action_by_candidate[item] in RERENDER_ACTIONS for item in ready_candidate_ids
        ),
        "v1_mutated": False,
    }
    _write(output_root / "overlay.provenance.json", provenance)
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collection-id", default="behavior51_coverage_v2_local_repair")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                source_root=args.source,
                audit_root=args.audit,
                output_root=args.output,
                collection_id=args.collection_id,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
