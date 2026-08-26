#!/usr/bin/env python3
"""Verify terminal P2/P3 shard artifacts and marker evidence before upload."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

EXPECTED_COLLECTIONS = {"p2": "p23_p2_7k_v1", "p3": "p23_p3stream_7k_v1"}
TERMINAL_STATUSES = {"accepted", "redundant", "rejected"}


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(phase: str, work_root: Path) -> dict[str, Any]:
    output = work_root.resolve() / "output"
    manifest_path = output / "coverage.plan.json"
    status_path = output / "coverage.status.json"
    complete_path = output / "shard.complete.json"
    manifest = _read(manifest_path)
    status = _read(status_path)
    complete = _read(complete_path)
    expected_collection = EXPECTED_COLLECTIONS[phase]
    if manifest.get("collection_id") != expected_collection:
        raise ValueError(f"wrong output collection: {manifest.get('collection_id')}")
    if status.get("collection_id") != expected_collection:
        raise ValueError(f"wrong status collection: {status.get('collection_id')}")
    if complete.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("final manifest hash differs from shard.complete.json")
    if complete.get("status_sha256") != _sha256(status_path):
        raise ValueError("final status hash differs from shard.complete.json")

    candidates = {
        candidate["candidate_id"]: (candidate, cell)
        for cell in manifest.get("cells", ())
        for candidate in cell.get("candidates", ())
    }
    observed: dict[str, str] = {}
    for row in status.get("cells", {}).values():
        for candidate_id, candidate_row in row.get("candidate_statuses", {}).items():
            observed[candidate_id] = str(candidate_row.get("status", "unknown"))
    if set(observed) != set(candidates):
        raise ValueError("final manifest and status candidate indexes differ")
    nonterminal = sorted(
        candidate_id
        for candidate_id, candidate_status in observed.items()
        if candidate_status not in TERMINAL_STATUSES
    )
    if nonterminal:
        raise ValueError(f"shard still has nonterminal candidates: {nonterminal[:8]}")

    verified_markers = 0
    verified_rendered = 0
    for candidate_id, candidate_status in observed.items():
        if candidate_status not in {"accepted", "redundant"}:
            continue
        candidate, cell = candidates[candidate_id]
        render_report = _read(Path(candidate["bundle"]) / "render_report.json")
        if render_report.get("status") != "success":
            raise ValueError(f"render report is not successful: {candidate_id}")
        group_root = Path(candidate["group"])
        _read(group_root / "group.json")
        marker = _read(group_root / "media" / "marker.audit.json")
        if marker.get("schema_version") != "scriptgen_marker_audit.v1":
            raise ValueError(f"wrong marker audit schema: {candidate_id}")
        assignments = marker.get("assignments", ())
        if len(assignments) != len(cell.get("binding", {})):
            raise ValueError(f"marker assignment count differs from binding: {candidate_id}")
        minimum = int(marker.get("min_badged_frames", 0))
        if minimum < 2 or any(
            int(assignment.get("badged_frames", 0)) < minimum
            for assignment in assignments
        ):
            raise ValueError(f"marker evidence is insufficient: {candidate_id}")
        verified_rendered += 1
        verified_markers += len(assignments)

    counts = Counter(observed.values())
    report = {
        "schema_version": "p23_shard_output_verification.v1",
        "status": "pass",
        "phase": phase,
        "collection_id": expected_collection,
        "candidate_status_counts": dict(sorted(counts.items())),
        "verified_rendered_candidates": verified_rendered,
        "verified_marker_assignments": verified_markers,
        # This verifier deliberately does not assert a global accepted target;
        # release readiness is decided only after both machines are merged.
        "accepted_target_checked": False,
    }
    (output / "p23.shard_verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=sorted(EXPECTED_COLLECTIONS), required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.phase, args.work_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
