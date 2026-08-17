#!/usr/bin/env python3
"""Index and validate the four distributed direction-balance shard tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


TASKS = (
    "pure_rotation_left",
    "pure_rotation_right",
    "pure_translation_left",
    "pure_translation_right",
)
SCHEMA_VERSION = "epispace.behavior51_direction_balance_distribution.v1"
MERGE_SCHEMA_VERSION = "epispace.behavior51_direction_balance_merge.v1"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


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


def build_index(
    *,
    direction_root: Path,
    shard_root: Path,
    oss_prefix: str,
    output: Path,
) -> Path:
    direction_root = direction_root.resolve()
    shard_root = shard_root.resolve()
    rows = []
    common_revisions: dict[str, Any] | None = None
    for task in TASKS:
        distribution_path = shard_root / task / "distribution.json"
        distribution = _read(distribution_path)
        shards = distribution.get("shards", [])
        archives = distribution.get("archives", [])
        if len(shards) != 1 or len(archives) != 1:
            raise ValueError(f"direction task must contain exactly one shard: {task}")
        shard = shards[0]
        archive = archives[0]
        revisions = shard.get("code_revisions", {})
        if common_revisions is None:
            common_revisions = revisions
        elif revisions != common_revisions:
            raise ValueError(f"code revisions differ across direction tasks: {task}")
        manifest_path = direction_root / task / "coverage.plan.json"
        status_path = direction_root / task / "coverage.status.json"
        if _sha256(manifest_path) != distribution["base_manifest_sha256"]:
            raise ValueError(f"local manifest differs from shard base: {task}")
        if _sha256(status_path) != distribution["base_status_sha256"]:
            raise ValueError(f"local status differs from shard base: {task}")
        archive_path = shard_root / task / archive["name"]
        if _sha256(archive_path) != archive["sha256"]:
            raise ValueError(f"local archive checksum differs: {archive_path}")
        rows.append(
            {
                "task": task,
                "collection_id": distribution["collection_id"],
                "base_manifest_sha256": distribution["base_manifest_sha256"],
                "base_status_sha256": distribution["base_status_sha256"],
                "shard_name": shard["shard_name"],
                "archive": archive,
                "scene_count": len(shard["scene_keys"]),
                "cell_count": len(shard["cell_ids"]),
                "initial_candidate_count": len(shard["candidate_ids"]),
            }
        )
    _write(
        output,
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "oss_prefix": oss_prefix.rstrip("/"),
            "direction_root": str(direction_root),
            "shard_root": str(shard_root),
            "code_revisions": common_revisions or {},
            "tasks": rows,
        },
    )
    return output


def _load_index(path: Path) -> dict[str, Any]:
    index = _read(path)
    if index.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported direction distribution: {path}")
    observed = tuple(row.get("task") for row in index.get("tasks", ()))
    if observed != TASKS:
        raise ValueError(f"direction task order differs: {observed}")
    return index


def verify_local(index_path: Path, direction_root: Path) -> None:
    index = _load_index(index_path)
    direction_root = direction_root.resolve()
    for row in index["tasks"]:
        task = row["task"]
        manifest = direction_root / task / "coverage.plan.json"
        status = direction_root / task / "coverage.status.json"
        manifest_sha = _sha256(manifest)
        status_sha = _sha256(status)
        if (
            manifest_sha == row["base_manifest_sha256"]
            and status_sha == row["base_status_sha256"]
        ):
            continue
        ledger_path = direction_root / task / "coverage.shard_merge.json"
        if not ledger_path.is_file():
            raise ValueError(f"merge base changed without a ledger: {task}")
        ledger = _read(ledger_path)
        if (
            manifest_sha != ledger.get("current_manifest_sha256")
            or status_sha != ledger.get("current_status_sha256")
            or row["shard_name"] not in ledger.get("applied_shards", ())
        ):
            raise ValueError(f"local merged state is not hash-bound: {task}")


def write_merge_summary(
    *, index_path: Path, direction_root: Path, output: Path
) -> Path:
    index = _load_index(index_path)
    direction_root = direction_root.resolve()
    rows = []
    for expected in index["tasks"]:
        task = expected["task"]
        task_root = direction_root / task
        ledger = _read(task_root / "coverage.shard_merge.json")
        status = _read(task_root / "coverage.status.json")
        if expected["shard_name"] not in ledger.get("applied_shards", ()):
            raise ValueError(f"expected shard was not merged: {task}")
        candidate_counts: dict[str, int] = {}
        for cell in status["cells"].values():
            for candidate in cell.get("candidate_statuses", {}).values():
                state = str(candidate.get("status", "unknown"))
                candidate_counts[state] = candidate_counts.get(state, 0) + 1
        rows.append(
            {
                "task": task,
                "shard_name": expected["shard_name"],
                "episode_count": len(status.get("episodes", {})),
                "candidate_status_counts": dict(sorted(candidate_counts.items())),
                "manifest_sha256": _sha256(task_root / "coverage.plan.json"),
                "status_sha256": _sha256(task_root / "coverage.status.json"),
                "dataset_sha256": (
                    _sha256(task_root / "dataset.json")
                    if (task_root / "dataset.json").is_file()
                    else None
                ),
            }
        )
    _write(
        output,
        {
            "schema_version": MERGE_SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "source_distribution_sha256": _sha256(index_path),
            "tasks": rows,
        },
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-index")
    build.add_argument("--direction-root", type=Path, required=True)
    build.add_argument("--shard-root", type=Path, required=True)
    build.add_argument("--oss-prefix", required=True)
    build.add_argument("--output", type=Path, required=True)
    tasks = commands.add_parser("list-tasks")
    tasks.add_argument("--index", type=Path, required=True)
    field = commands.add_parser("task-field")
    field.add_argument("--index", type=Path, required=True)
    field.add_argument("--task", choices=TASKS, required=True)
    field.add_argument("--field", required=True)
    verify = commands.add_parser("verify-local")
    verify.add_argument("--index", type=Path, required=True)
    verify.add_argument("--direction-root", type=Path, required=True)
    summary = commands.add_parser("merge-summary")
    summary.add_argument("--index", type=Path, required=True)
    summary.add_argument("--direction-root", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "build-index":
        print(
            build_index(
                direction_root=args.direction_root,
                shard_root=args.shard_root,
                oss_prefix=args.oss_prefix,
                output=args.output,
            )
        )
    elif args.command == "list-tasks":
        print("\n".join(row["task"] for row in _load_index(args.index)["tasks"]))
    elif args.command == "task-field":
        index = _load_index(args.index)
        row = next(row for row in index["tasks"] if row["task"] == args.task)
        value = row.get(args.field)
        if value is None:
            raise ValueError(f"unknown or null task field: {args.field}")
        print(value)
    elif args.command == "verify-local":
        verify_local(args.index, args.direction_root)
        print("ok")
    elif args.command == "merge-summary":
        print(
            write_merge_summary(
                index_path=args.index,
                direction_root=args.direction_root,
                output=args.output,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
