from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "direction_balance_shards.py"
SPEC = importlib.util.spec_from_file_location("direction_balance_shards", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SHARDS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SHARDS)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_direction_index_hash_binds_all_four_tasks_and_accepts_merged_resume(
    tmp_path: Path,
) -> None:
    direction_root = tmp_path / "direction"
    shard_root = tmp_path / "shards"
    revisions = {
        "epispace": {"commit": "a" * 40},
        "omnigibson_episode": {"commit": "b" * 40},
    }
    for task in SHARDS.TASKS:
        task_root = direction_root / task
        manifest = task_root / "coverage.plan.json"
        status = task_root / "coverage.status.json"
        _write(manifest, {"task": task, "kind": "manifest"})
        _write(status, {"task": task, "kind": "status"})
        archive = shard_root / task / f"{task}.tar.zst"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(task.encode("utf-8"))
        _write(
            shard_root / task / "distribution.json",
            {
                "collection_id": task,
                "base_manifest_sha256": SHARDS._sha256(manifest),
                "base_status_sha256": SHARDS._sha256(status),
                "shards": [
                    {
                        "shard_name": f"{task}__shard-000-of-001",
                        "code_revisions": revisions,
                        "scene_keys": ["scene"],
                        "cell_ids": ["cell"],
                        "candidate_ids": ["candidate"],
                    }
                ],
                "archives": [
                    {
                        "name": archive.name,
                        "sha256": SHARDS._sha256(archive),
                        "size_bytes": archive.stat().st_size,
                    }
                ],
            },
        )

    index_path = tmp_path / "direction.distribution.json"
    SHARDS.build_index(
        direction_root=direction_root,
        shard_root=shard_root,
        oss_prefix="oss://bucket/prefix",
        output=index_path,
    )
    index = SHARDS._load_index(index_path)
    assert tuple(row["task"] for row in index["tasks"]) == SHARDS.TASKS
    assert index["code_revisions"] == revisions
    SHARDS.verify_local(index_path, direction_root)

    first = index["tasks"][0]
    first_root = direction_root / first["task"]
    manifest = first_root / "coverage.plan.json"
    status = first_root / "coverage.status.json"
    _write(manifest, {"task": first["task"], "kind": "merged-manifest"})
    _write(status, {"task": first["task"], "kind": "merged-status"})
    _write(
        first_root / "coverage.shard_merge.json",
        {
            "current_manifest_sha256": SHARDS._sha256(manifest),
            "current_status_sha256": SHARDS._sha256(status),
            "applied_shards": [first["shard_name"]],
        },
    )

    # A partially completed merge can be resumed: already-applied tasks use
    # their ledger hashes while untouched tasks still use immutable base hashes.
    SHARDS.verify_local(index_path, direction_root)
