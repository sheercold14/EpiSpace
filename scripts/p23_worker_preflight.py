#!/usr/bin/env python3
"""Fail-closed environment and immutable-input checks for one P2/P3 worker."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

EXPECTED_COLLECTIONS = {
    "p2": "p23_p2_7k_v1",
    "p3": "p23_p3stream_7k_v1",
}


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


def _git_check(root: Path, expected: str, label: str) -> dict[str, str]:
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise RuntimeError(f"{label} commit mismatch: {actual} != {expected}")
    for arguments in (
        ("diff", "--quiet", "HEAD", "--"),
        ("diff", "--cached", "--quiet"),
    ):
        if subprocess.run(
            ["git", "-C", str(root), *arguments], check=False
        ).returncode:
            raise RuntimeError(f"{label} has uncommitted tracked changes: {root}")
    untracked_code = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "src",
            "scripts",
            "pyproject.toml",
        ],
        text=True,
    ).splitlines()
    if untracked_code:
        raise RuntimeError(f"{label} has untracked code: {untracked_code[:5]}")
    return {"root": str(root.resolve()), "commit": actual}


def _gpu_rows(gpu_ids: list[int]) -> list[dict[str, Any]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    available: dict[int, dict[str, Any]] = {}
    for line in output.splitlines():
        raw_index, name, raw_free = (part.strip() for part in line.split(",", 2))
        available[int(raw_index)] = {
            "index": int(raw_index),
            "name": name,
            "free_memory_mib": int(raw_free),
        }
    missing = sorted(set(gpu_ids) - set(available))
    if missing:
        raise RuntimeError(f"requested GPUs are unavailable: {missing}")
    return [available[index] for index in gpu_ids]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=sorted(EXPECTED_COLLECTIONS), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--distribution", type=Path, required=True)
    parser.add_argument("--quality-audit", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--backend-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--gpu-ids", type=int, nargs="+", required=True)
    parser.add_argument("--min-free-gib", type=int, default=100)
    parser.add_argument("--bytes-per-frame", type=int, default=4 * 1024**2)
    parser.add_argument("--scratch-reserve-gib", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if (
        args.shard_index < 0
        or args.min_free_gib < 0
        or args.bytes_per_frame < 1
        or args.scratch_reserve_gib < 0
    ):
        raise ValueError("shard/disk preflight values are outside their valid range")
    expected_collection = EXPECTED_COLLECTIONS[args.phase]
    package = _read(args.package / "shard.json")
    distribution = _read(args.distribution)
    audit = _read(args.quality_audit)
    manifest = _read(args.package / "coverage.plan.template.json")

    if package.get("collection_id") != expected_collection:
        raise ValueError(f"wrong package collection: {package.get('collection_id')}")
    if package.get("shard_index") != args.shard_index:
        raise ValueError(f"wrong package shard index: {package.get('shard_index')}")
    if manifest.get("collection_id") != expected_collection:
        raise ValueError(f"wrong manifest collection: {manifest.get('collection_id')}")
    if manifest.get("standard_version") != "std.v11":
        raise ValueError(f"P2/P3 worker requires std.v11: {manifest.get('standard_version')}")
    if audit.get("status") != "pass" or audit.get("physical_trajectories") != 7000:
        raise ValueError("P2/P3 quality audit is not a passing 7,000-plan audit")
    audited_manifest = audit.get("inputs", {}).get(args.phase, {}).get("sha256")
    base_manifest = distribution.get("base_manifest_sha256")
    if audited_manifest != base_manifest or package.get("base_manifest_sha256") != base_manifest:
        raise ValueError("quality audit, distribution and package manifest hashes differ")

    archive_sha = _sha256(args.archive)
    archive_rows = {
        row.get("name"): row.get("sha256")
        for row in distribution.get("archives", ())
        if isinstance(row, dict)
    }
    if archive_rows.get(args.archive.name) != archive_sha:
        raise ValueError("archive hash differs from distribution.json")

    revisions = package.get("code_revisions", {})
    for repository in ("epispace", "omnigibson_episode"):
        commit = revisions.get(repository, {}).get("commit", "")
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise ValueError(f"missing exact {repository} commit in shard package")
    repositories = {
        "epispace": _git_check(
            args.repo_root.resolve(), revisions["epispace"]["commit"], "EpiSpace"
        ),
        "omnigibson_episode": _git_check(
            args.backend_root.resolve(),
            revisions["omnigibson_episode"]["commit"],
            "OminiGibson",
        ),
    }
    if not (args.backend_root / "src/omnigibson_episode/cli.py").is_file():
        raise FileNotFoundError("custom omnigibson_episode backend is missing")
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"OmniGibson data root is missing: {args.data_root}")

    args.work_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(args.work_root).free
    estimated_frames = int(package.get("estimated_frames", 0))
    frame_estimate_bytes = estimated_frames * args.bytes_per_frame
    scratch_reserve_bytes = args.scratch_reserve_gib * 1024**3
    required_bytes = max(
        args.min_free_gib * 1024**3,
        frame_estimate_bytes + scratch_reserve_bytes,
    )
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"insufficient worker disk: {free_bytes / 1024**3:.1f} GiB "
            f"< {args.min_free_gib} GiB"
        )

    versions = {
        package_name: importlib.metadata.version(package_name)
        for package_name in ("numpy", "scipy", "Pillow", "pydantic")
    }
    report = {
        "schema_version": "p23_worker_preflight.v1",
        "status": "pass",
        "phase": args.phase,
        "collection_id": expected_collection,
        "shard_index": args.shard_index,
        "shard_name": package["shard_name"],
        "base_manifest_sha256": base_manifest,
        "quality_audit_sha256": _sha256(args.quality_audit),
        "archive": {"name": args.archive.name, "sha256": archive_sha},
        "repositories": repositories,
        "dependencies": versions,
        "gpus": _gpu_rows(args.gpu_ids),
        "disk": {
            "path": str(args.work_root.resolve()),
            "free_bytes": free_bytes,
            "minimum_free_bytes": required_bytes,
            "estimated_frames": estimated_frames,
            "bytes_per_frame_budget": args.bytes_per_frame,
            "frame_output_budget_bytes": frame_estimate_bytes,
            "scratch_reserve_bytes": scratch_reserve_bytes,
        },
        "remote_backfill": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
