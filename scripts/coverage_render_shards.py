#!/usr/bin/env python3
"""Build, prepare, finalize, and merge render-only coverage shards.

The shard boundary is a complete scene.  This is important because one
rendered question group can credit several capability cells in that scene.
Keeping a scene on one machine makes those status updates disjoint and makes
the final merge deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PACKAGE_TOKEN = "${EPISPACE_SHARD_PACKAGE}"
OUTPUT_TOKEN = "${EPISPACE_SHARD_OUTPUT}"
SHARD_SCHEMA = "epispace_coverage_render_shard.v1"
COMPLETE_SCHEMA = "epispace_coverage_render_shard_complete.v1"
MERGE_SCHEMA = "epispace_coverage_render_shard_merge.v1"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(source: Path, destination: Path, *, hardlink: bool = False) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if hardlink:
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def _candidate_state(status: dict[str, Any], cell_id: str, candidate_id: str) -> str:
    return str(
        status.get("cells", {})
        .get(cell_id, {})
        .get("candidate_statuses", {})
        .get(candidate_id, {})
        .get("status", "pending")
    )


def _running_rows(status: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for cell_id, row in status.get("cells", {}).items():
        if row.get("status") == "running":
            result.append(str(cell_id))
        for candidate_id, candidate_row in row.get("candidate_statuses", {}).items():
            if candidate_row.get("status") == "running":
                result.append(str(candidate_id))
    return result


def _manifest_processes(manifest_path: Path) -> list[int]:
    """Find other processes whose argv directly names this manifest."""
    result: list[int] = []
    expected = str(manifest_path.resolve())
    own_pid = os.getpid()
    proc = Path("/proc")
    if not proc.is_dir():
        return result
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            arguments = (entry / "cmdline").read_bytes().split(b"\0")
            decoded = [argument.decode(errors="replace") for argument in arguments if argument]
        except (OSError, PermissionError):
            continue
        if expected in decoded:
            result.append(int(entry.name))
    return sorted(result)


def _frame_count(candidate: dict[str, Any]) -> int:
    path = Path(candidate["render_plan"])
    try:
        payload = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return 1
    return max(1, len(payload.get("views", ())) + len(payload.get("auxiliary_views", ())))


def _scene_weights(
    manifest: dict[str, Any], status: dict[str, Any]
) -> tuple[dict[str, int], dict[str, int]]:
    weights: dict[str, int] = defaultdict(int)
    pending_counts: dict[str, int] = defaultdict(int)
    for cell in manifest["cells"]:
        cell_id = cell["cell_id"]
        row = status.get("cells", {}).get(cell_id, {})
        if len(row.get("accepted_episode_ids", ())) >= int(cell["target_accepted"]):
            continue
        for candidate in cell.get("candidates", ()):
            if _candidate_state(status, cell_id, candidate["candidate_id"]) != "pending":
                continue
            pending_counts[cell["scene_key"]] += 1
            weights[cell["scene_key"]] += _frame_count(candidate)
    return dict(weights), dict(pending_counts)


def _assign_scenes(
    weights: dict[str, int], pending_counts: dict[str, int], shard_count: int
) -> list[dict[str, Any]]:
    active = [scene_key for scene_key, count in pending_counts.items() if count > 0]
    if len(active) < shard_count:
        raise ValueError(
            f"only {len(active)} scenes have pending render candidates; "
            f"cannot make {shard_count} non-empty scene shards"
        )
    shards = [
        {"index": index, "scene_keys": [], "estimated_frames": 0, "pending_candidates": 0}
        for index in range(shard_count)
    ]
    for scene_key in sorted(active, key=lambda key: (-weights[key], key)):
        target = min(
            shards,
            key=lambda shard: (
                shard["estimated_frames"],
                shard["pending_candidates"],
                shard["index"],
            ),
        )
        target["scene_keys"].append(scene_key)
        target["estimated_frames"] += weights[scene_key]
        target["pending_candidates"] += pending_counts[scene_key]
    for shard in shards:
        shard["scene_keys"].sort()
    return shards


def _normalized_subset_status(
    manifest: dict[str, Any], status: dict[str, Any], scene_keys: set[str]
) -> dict[str, Any]:
    result = {
        "schema_version": status.get(
            "schema_version", "scriptgen_binding_coverage_status.v1"
        ),
        "collection_id": manifest["collection_id"],
        "cells": {},
        "episodes": {},
    }
    for cell in manifest["cells"]:
        if cell["scene_key"] not in scene_keys:
            continue
        row = json.loads(json.dumps(status.get("cells", {}).get(cell["cell_id"], {})))
        row.setdefault("status", "planned")
        row.setdefault("accepted_episode_ids", [])
        candidate_statuses = row.setdefault("candidate_statuses", {})
        for candidate in cell.get("candidates", ()):
            candidate_statuses.setdefault(
                candidate["candidate_id"], {"status": "pending", "reason": None}
            )
        result["cells"][cell["cell_id"]] = row
    for episode_id, episode in status.get("episodes", {}).items():
        if episode.get("scene_key") in scene_keys:
            result["episodes"][episode_id] = json.loads(json.dumps(episode))
    return result


def _runtime_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    candidate_id = candidate["candidate_id"]
    result = dict(candidate)
    result.update(
        {
            "plan_record": f"{PACKAGE_TOKEN}/planning/plans/{candidate_id}.record.json",
            "render_plan": f"{PACKAGE_TOKEN}/planning/plans/{candidate_id}.views.json",
            "recipe": f"{OUTPUT_TOKEN}/recipes/{candidate_id}.yaml",
            "bundle": f"{OUTPUT_TOKEN}/bundles/{candidate_id}",
            "group": f"{OUTPUT_TOKEN}/groups/{candidate_id}",
            "log": f"{OUTPUT_TOKEN}/logs/{candidate_id}.log",
        }
    )
    return result


def _subset_manifest(
    manifest: dict[str, Any], scene_keys: set[str]
) -> dict[str, Any]:
    result = json.loads(json.dumps(manifest))
    result["source_index"] = f"{PACKAGE_TOKEN}/sources/source.index.json"
    result["output_root"] = OUTPUT_TOKEN
    result["scenes"] = [
        {
            **scene,
            "scene_ir": f"{PACKAGE_TOKEN}/sources/scenes/{scene['scene_key']}/scene_ir.json",
            "source_recipe": f"{PACKAGE_TOKEN}/sources/recipes/{scene['scene_key']}.yaml",
        }
        for scene in manifest["scenes"]
        if scene["scene_key"] in scene_keys
    ]
    result["scene_skips"] = {
        key: value
        for key, value in manifest.get("scene_skips", {}).items()
        if key in scene_keys
    }
    result["cells"] = [
        {
            **cell,
            "candidates": [_runtime_candidate(candidate) for candidate in cell["candidates"]],
        }
        for cell in manifest["cells"]
        if cell["scene_key"] in scene_keys
    ]
    return result


def _git_revision(repo_root: Path) -> dict[str, str]:
    """Return a reproducible Git revision and reject uncommitted tracked code."""
    repo_root = repo_root.resolve()
    if not (repo_root / ".git").exists():
        raise ValueError(f"code root is not a Git checkout: {repo_root}")
    for arguments in (("diff", "--quiet", "HEAD", "--"), ("diff", "--cached", "--quiet")):
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"tracked code is not committed under {repo_root}; commit it before sharding"
            )

    def output(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), *arguments], text=True
        ).strip()

    return {
        "remote": output("remote", "get-url", "origin"),
        "commit": output("rev-parse", "HEAD"),
    }


def _build_one_package(
    *,
    package_root: Path,
    manifest: dict[str, Any],
    status: dict[str, Any],
    source_manifest_path: Path,
    source_status_path: Path,
    assignment: dict[str, Any],
    shard_count: int,
    hardlink: bool,
    code_revisions: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    scene_keys = set(assignment["scene_keys"])
    subset_manifest = _subset_manifest(manifest, scene_keys)
    subset_status = _normalized_subset_status(manifest, status, scene_keys)
    manifest_template = package_root / "coverage.plan.template.json"
    status_template = package_root / "coverage.status.template.json"

    _copy_file(
        Path(manifest["source_index"]),
        package_root / "sources" / "source.index.json",
        hardlink=hardlink,
    )
    scene_lookup = {scene["scene_key"]: scene for scene in manifest["scenes"]}
    for scene_key in sorted(scene_keys):
        scene = scene_lookup[scene_key]
        scene_ir = Path(scene["scene_ir"])
        _copy_file(
            scene_ir,
            package_root / "sources" / "scenes" / scene_key / "scene_ir.json",
            hardlink=hardlink,
        )
        _copy_file(
            scene_ir.parent / "scene_snapshot.json",
            package_root / "sources" / "scenes" / scene_key / "scene_snapshot.json",
            hardlink=hardlink,
        )
        _copy_file(
            Path(scene["source_recipe"]),
            package_root / "sources" / "recipes" / f"{scene_key}.yaml",
            hardlink=hardlink,
        )

    candidate_ids: list[str] = []
    for cell in manifest["cells"]:
        if cell["scene_key"] not in scene_keys:
            continue
        for candidate in cell["candidates"]:
            candidate_id = candidate["candidate_id"]
            candidate_ids.append(candidate_id)
            _copy_file(
                Path(candidate["plan_record"]),
                package_root / "planning" / "plans" / f"{candidate_id}.record.json",
                hardlink=hardlink,
            )
            _copy_file(
                Path(candidate["render_plan"]),
                package_root / "planning" / "plans" / f"{candidate_id}.views.json",
                hardlink=hardlink,
            )
            recipe_text = Path(candidate["recipe"]).read_text(encoding="utf-8")
            plan_path = f"{PACKAGE_TOKEN}/planning/plans/{candidate_id}.views.json"
            recipe_text, replacements = re.subn(
                r"(?m)^(\s*plan_path:\s*).*$",
                lambda match: match.group(1) + plan_path,
                recipe_text,
            )
            if replacements != 1:
                raise ValueError(
                    f"expected exactly one trajectory plan_path in {candidate['recipe']}"
                )
            recipe_path = package_root / "planning" / "recipes" / f"{candidate_id}.yaml"
            recipe_path.parent.mkdir(parents=True, exist_ok=True)
            recipe_path.write_text(recipe_text, encoding="utf-8")

    _write_json(manifest_template, subset_manifest)
    _write_json(status_template, subset_status)
    cell_ids = [cell["cell_id"] for cell in subset_manifest["cells"]]
    _write_json(package_root / "cell_ids.json", {"cell_ids": cell_ids})
    metadata = {
        "schema_version": SHARD_SCHEMA,
        "collection_id": manifest["collection_id"],
        "shard_index": assignment["index"],
        "shard_count": shard_count,
        "shard_name": package_root.name,
        "scene_keys": sorted(scene_keys),
        "cell_ids": cell_ids,
        "candidate_ids": sorted(candidate_ids),
        "estimated_frames": assignment["estimated_frames"],
        "pending_candidates": assignment["pending_candidates"],
        "base_manifest": str(source_manifest_path.resolve()),
        "base_manifest_sha256": _sha256(source_manifest_path),
        "base_status": str(source_status_path.resolve()),
        "base_status_sha256": _sha256(source_status_path),
        "manifest_template_sha256": _sha256(manifest_template),
        "status_template_sha256": _sha256(status_template),
        "source_index_sha256": _sha256(package_root / "sources" / "source.index.json"),
        "code_revisions": code_revisions or {},
    }
    _write_json(package_root / "shard.json", metadata)
    return metadata


def command_build(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.resolve()
    manifest = _read_json(manifest_path)
    status_path = (
        args.status.resolve()
        if args.status is not None
        else Path(manifest["output_root"]).resolve() / "coverage.status.json"
    )
    status = _read_json(status_path)
    running = _running_rows(status)
    live_processes = _manifest_processes(manifest_path)
    if not args.preview:
        if live_processes:
            raise RuntimeError(
                "coverage pipeline still names this manifest; stop it at a candidate boundary "
                f"before building shards (pids={live_processes})"
            )
        if running:
            raise RuntimeError(
                "coverage status contains running work; recover/finish it before sharding: "
                + ", ".join(running[:8])
            )

    weights, pending_counts = _scene_weights(manifest, status)
    assignments = _assign_scenes(weights, pending_counts, args.num_shards)
    preview = {
        "manifest": str(manifest_path),
        "status": str(status_path),
        "live_manifest_processes": live_processes,
        "running_status_rows": len(running),
        "shards": assignments,
    }
    if args.preview:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0

    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    backend_root = args.backend_root.resolve()
    code_revisions = {
        "epispace": _git_revision(repo_root),
        "omnigibson_episode": _git_revision(backend_root),
    }
    manifest_hash_before = _sha256(manifest_path)
    status_hash_before = _sha256(status_path)
    metadata_rows: list[dict[str, Any]] = []
    archive_rows: list[dict[str, Any]] = []
    for assignment in assignments:
        shard_label = f"shard-{assignment['index']:03d}-of-{args.num_shards:03d}"
        package_name = (
            f"{manifest['collection_id']}__{manifest_hash_before[:12]}__{shard_label}"
        )
        package_root = output_root / package_name
        archive_path = output_root / f"{package_name}.tar.zst"
        if package_root.exists() or archive_path.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"shard output already exists: {package_root}; pass --overwrite explicitly"
                )
            if package_root.exists():
                shutil.rmtree(package_root)
            if archive_path.exists():
                archive_path.unlink()
        metadata = _build_one_package(
            package_root=package_root,
            manifest=manifest,
            status=status,
            source_manifest_path=manifest_path,
            source_status_path=status_path,
            assignment=assignment,
            shard_count=args.num_shards,
            hardlink=args.hardlink,
            code_revisions=code_revisions,
        )
        metadata_rows.append(metadata)
        if not args.no_archive:
            subprocess.run(
                [
                    "tar",
                    "--zstd",
                    "-cf",
                    str(archive_path),
                    "-C",
                    str(output_root),
                    package_name,
                ],
                check=True,
            )
            archive_hash = _sha256(archive_path)
            checksum_path = Path(str(archive_path) + ".sha256")
            checksum_path.write_text(
                f"{archive_hash}  {archive_path.name}\n", encoding="utf-8"
            )
            archive_rows.append(
                {
                    "name": archive_path.name,
                    "sha256": archive_hash,
                    "size_bytes": archive_path.stat().st_size,
                }
            )

    if _sha256(manifest_path) != manifest_hash_before or _sha256(status_path) != status_hash_before:
        raise RuntimeError(
            "manifest or status changed while packages were built; discard these shards and retry"
        )
    distribution = {
        "schema_version": "epispace_coverage_render_distribution.v1",
        "collection_id": manifest["collection_id"],
        "base_manifest_sha256": manifest_hash_before,
        "base_status_sha256": status_hash_before,
        "shards": metadata_rows,
        "archives": archive_rows,
    }
    _write_json(output_root / "distribution.json", distribution)
    print(json.dumps(distribution, ensure_ascii=False, indent=2))
    return 0


def _replace_tokens(value: Any, package_root: Path, output_root: Path) -> Any:
    if isinstance(value, str):
        return value.replace(PACKAGE_TOKEN, str(package_root)).replace(
            OUTPUT_TOKEN, str(output_root)
        )
    if isinstance(value, list):
        return [_replace_tokens(item, package_root, output_root) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_tokens(item, package_root, output_root)
            for key, item in value.items()
        }
    return value


def command_prepare(args: argparse.Namespace) -> int:
    package_root = args.package.resolve()
    work_root = args.work_root.resolve()
    output_root = work_root / "output"
    metadata = _read_json(package_root / "shard.json")
    if metadata.get("schema_version") != SHARD_SCHEMA:
        raise ValueError(f"unsupported shard metadata: {metadata.get('schema_version')}")
    template_manifest = package_root / "coverage.plan.template.json"
    template_status = package_root / "coverage.status.template.json"
    checks = {
        template_manifest: metadata["manifest_template_sha256"],
        template_status: metadata["status_template_sha256"],
        package_root / "sources" / "source.index.json": metadata["source_index_sha256"],
    }
    for path, expected in checks.items():
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"shard input changed: {path} ({actual} != {expected})")

    existing_metadata = output_root / "shard.json"
    if existing_metadata.is_file():
        previous = _read_json(existing_metadata)
        if previous.get("base_manifest_sha256") != metadata["base_manifest_sha256"]:
            raise ValueError(
                f"work root belongs to a different shard snapshot: {output_root}"
            )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = _replace_tokens(_read_json(template_manifest), package_root, output_root)
    _write_json(output_root / "coverage.plan.json", manifest)
    status_path = output_root / "coverage.status.json"
    if not status_path.exists():
        status = _replace_tokens(_read_json(template_status), package_root, output_root)
        _write_json(status_path, status)

    recipe_root = output_root / "recipes"
    recipe_root.mkdir(parents=True, exist_ok=True)
    for recipe_template in (package_root / "planning" / "recipes").glob("*.yaml"):
        recipe_text = recipe_template.read_text(encoding="utf-8")
        recipe_text = recipe_text.replace(PACKAGE_TOKEN, str(package_root)).replace(
            OUTPUT_TOKEN, str(output_root)
        )
        (recipe_root / recipe_template.name).write_text(recipe_text, encoding="utf-8")
    shutil.copy2(package_root / "shard.json", existing_metadata)
    shutil.copy2(package_root / "cell_ids.json", output_root / "cell_ids.json")
    print(output_root / "coverage.plan.json")
    return 0


def command_finalize(args: argparse.Namespace) -> int:
    output_root = args.work_root.resolve() / "output"
    manifest_path = output_root / "coverage.plan.json"
    manifest = _read_json(manifest_path)
    status_path = output_root / "coverage.status.json"
    status = _read_json(status_path)
    running = _running_rows(status)
    if running:
        raise RuntimeError("cannot finalize a running shard: " + ", ".join(running[:8]))
    metadata = _read_json(output_root / "shard.json")
    candidate_counts: dict[str, int] = defaultdict(int)
    cell_counts: dict[str, int] = defaultdict(int)
    for row in status.get("cells", {}).values():
        cell_counts[str(row.get("status", "unknown"))] += 1
        for candidate_row in row.get("candidate_statuses", {}).values():
            candidate_counts[str(candidate_row.get("status", "unknown"))] += 1
    complete = {
        "schema_version": COMPLETE_SCHEMA,
        "shard_name": metadata["shard_name"],
        "base_manifest_sha256": metadata["base_manifest_sha256"],
        "base_status_sha256": metadata["base_status_sha256"],
        "shard_metadata_sha256": _sha256(output_root / "shard.json"),
        "manifest_sha256": _sha256(manifest_path),
        "status_sha256": _sha256(status_path),
        "cell_status_counts": dict(sorted(cell_counts.items())),
        "candidate_status_counts": dict(sorted(candidate_counts.items())),
        "episode_count": len(status.get("episodes", {})),
        "cell_count": len(manifest.get("cells", ())),
        "candidate_count": sum(
            len(cell.get("candidates", ())) for cell in manifest.get("cells", ())
        ),
    }
    _write_json(output_root / "shard.complete.json", complete)
    print(json.dumps(complete, ensure_ascii=False, indent=2))
    return 0


def _copy_result_artifacts(result_root: Path, master_output: Path) -> None:
    for directory in ("bundles", "groups"):
        source = result_root / directory
        if source.is_dir():
            shutil.copytree(source, master_output / directory, dirs_exist_ok=True)
    source_logs = result_root / "logs"
    if source_logs.is_dir():
        (master_output / "logs").mkdir(parents=True, exist_ok=True)
        for source in source_logs.iterdir():
            if source.is_file():
                shutil.copy2(source, master_output / "logs" / source.name)


def _localize_remote_candidate(
    candidate: dict[str, Any],
    *,
    result_root: Path,
    master_output: Path,
) -> dict[str, Any]:
    """Copy one remotely backfilled plan and rewrite all worker-local paths."""
    candidate_id = candidate["candidate_id"]
    source_record = result_root / "plans" / f"{candidate_id}.record.json"
    source_views = result_root / "plans" / f"{candidate_id}.views.json"
    source_recipe = result_root / "recipes" / f"{candidate_id}.yaml"
    destination_record = master_output / "plans" / source_record.name
    destination_views = master_output / "plans" / source_views.name
    destination_recipe = master_output / "recipes" / source_recipe.name
    _copy_file(source_record, destination_record)
    _copy_file(source_views, destination_views)
    recipe_text = source_recipe.read_text(encoding="utf-8")
    recipe_text, replacements = re.subn(
        r"(?m)^(\s*plan_path:\s*).*$",
        lambda match: match.group(1) + str(destination_views),
        recipe_text,
    )
    if replacements != 1:
        raise ValueError(f"expected one plan_path in remote recipe: {source_recipe}")
    destination_recipe.parent.mkdir(parents=True, exist_ok=True)
    destination_recipe.write_text(recipe_text, encoding="utf-8")
    return {
        **candidate,
        "plan_record": str(destination_record),
        "render_plan": str(destination_views),
        "recipe": str(destination_recipe),
        "bundle": str(master_output / "bundles" / candidate_id),
        "group": str(master_output / "groups" / candidate_id),
        "log": str(master_output / "logs" / f"{candidate_id}.log"),
    }


def command_merge(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.resolve()
    manifest = _read_json(manifest_path)
    master_output = Path(manifest["output_root"]).resolve()
    status_path = (
        args.status.resolve()
        if args.status is not None
        else master_output / "coverage.status.json"
    )
    status = _read_json(status_path)
    manifest_hash = _sha256(manifest_path)
    status_hash = _sha256(status_path)
    ledger_path = master_output / "coverage.shard_merge.json"
    if ledger_path.is_file():
        ledger = _read_json(ledger_path)
        if ledger.get("schema_version") != MERGE_SCHEMA:
            raise ValueError(f"unsupported merge ledger: {ledger_path}")
        if ledger["current_manifest_sha256"] != manifest_hash:
            raise ValueError("master manifest changed outside the recorded shard merges")
        if ledger["current_status_sha256"] != status_hash:
            raise ValueError("master status changed outside the recorded shard merges")
    else:
        manifest_backup = master_output / f"coverage.plan.shard-base-{manifest_hash[:12]}.json"
        if not manifest_backup.exists():
            shutil.copy2(manifest_path, manifest_backup)
        backup_path = master_output / f"coverage.status.shard-base-{status_hash[:12]}.json"
        if not backup_path.exists():
            shutil.copy2(status_path, backup_path)
        ledger = {
            "schema_version": MERGE_SCHEMA,
            "base_manifest_sha256": manifest_hash,
            "current_manifest_sha256": manifest_hash,
            "base_status_sha256": status_hash,
            "current_status_sha256": status_hash,
            "applied_shards": [],
            "applied_scene_keys": [],
        }
    if _running_rows(status):
        raise RuntimeError("master status is running; stop the local pipeline before merging")

    master_cells = {cell["cell_id"]: cell for cell in manifest["cells"]}
    master_candidates = {
        candidate["candidate_id"]: candidate
        for cell in manifest["cells"]
        for candidate in cell["candidates"]
    }
    applied = set(ledger["applied_shards"])
    applied_scenes = set(ledger["applied_scene_keys"])
    result_roots = [path.resolve() for path in args.results]
    for result_root in result_roots:
        complete = _read_json(result_root / "shard.complete.json")
        metadata = _read_json(result_root / "shard.json")
        shard_name = metadata["shard_name"]
        if shard_name in applied:
            print(f"merge skip already-applied shard={shard_name}", file=sys.stderr)
            continue
        if complete.get("schema_version") != COMPLETE_SCHEMA:
            raise ValueError(f"shard is not finalized: {result_root}")
        if _sha256(result_root / "shard.json") != complete["shard_metadata_sha256"]:
            raise ValueError(f"shard metadata changed after finalization: {shard_name}")
        if metadata["base_manifest_sha256"] != ledger["base_manifest_sha256"]:
            raise ValueError(f"shard manifest snapshot differs: {shard_name}")
        if metadata["base_status_sha256"] != ledger["base_status_sha256"]:
            raise ValueError(f"shard status snapshot differs: {shard_name}")
        overlap = set(metadata["scene_keys"]) & applied_scenes
        if overlap:
            raise ValueError(f"scene assigned to more than one shard: {sorted(overlap)}")
        shard_status_path = result_root / "coverage.status.json"
        if _sha256(shard_status_path) != complete["status_sha256"]:
            raise ValueError(f"final shard status changed: {shard_name}")
        shard_status = _read_json(shard_status_path)
        shard_manifest_path = result_root / "coverage.plan.json"
        if _sha256(shard_manifest_path) != complete["manifest_sha256"]:
            raise ValueError(f"final shard manifest changed: {shard_name}")
        shard_manifest = _read_json(shard_manifest_path)
        if _running_rows(shard_status):
            raise ValueError(f"shard still contains running work: {shard_name}")
        if set(shard_status.get("cells", {})) != set(metadata["cell_ids"]):
            raise ValueError(f"shard cell assignment changed: {shard_name}")
        unknown_cells = set(metadata["cell_ids"]) - set(master_cells)
        if unknown_cells:
            raise ValueError(f"unknown cells in shard {shard_name}: {sorted(unknown_cells)}")
        if {cell["cell_id"] for cell in shard_manifest.get("cells", ())} != set(
            metadata["cell_ids"]
        ):
            raise ValueError(f"shard manifest cell assignment changed: {shard_name}")

        localized_cells: dict[str, dict[str, Any]] = {}
        for remote_cell in shard_manifest["cells"]:
            localized_candidates: list[dict[str, Any]] = []
            for remote_candidate in remote_cell.get("candidates", ()):
                candidate_id = remote_candidate["candidate_id"]
                if candidate_id in master_candidates:
                    localized = master_candidates[candidate_id]
                else:
                    localized = _localize_remote_candidate(
                        remote_candidate,
                        result_root=result_root,
                        master_output=master_output,
                    )
                    master_candidates[candidate_id] = localized
                localized_candidates.append(localized)
            localized_cell = {**remote_cell, "candidates": localized_candidates}
            localized_cells[remote_cell["cell_id"]] = localized_cell

        for cell_id, row in shard_status["cells"].items():
            previous_candidates = status["cells"][cell_id].get("candidate_statuses", {})
            for candidate_id, candidate_row in row.get("candidate_statuses", {}).items():
                if candidate_row.get("status") not in {"accepted", "redundant"}:
                    continue
                if previous_candidates.get(candidate_id, {}).get("status") in {
                    "accepted",
                    "redundant",
                }:
                    continue
                report_path = result_root / "bundles" / candidate_id / "render_report.json"
                group_path = result_root / "groups" / candidate_id / "group.json"
                if not report_path.is_file() or not group_path.is_file():
                    raise ValueError(
                        f"new authoritative result is incomplete: {shard_name}/{candidate_id}"
                    )
                report = _read_json(report_path)
                if report.get("status") != "success":
                    raise ValueError(
                        f"new authoritative bundle did not render successfully: "
                        f"{shard_name}/{candidate_id}"
                    )

        _copy_result_artifacts(result_root, master_output)
        manifest["cells"] = [
            localized_cells.get(cell["cell_id"], cell) for cell in manifest["cells"]
        ]
        master_cells.update(localized_cells)
        for cell_id, row in shard_status["cells"].items():
            status["cells"][cell_id] = row
        shard_scenes = set(metadata["scene_keys"])
        status["episodes"] = {
            episode_id: episode
            for episode_id, episode in status.get("episodes", {}).items()
            if episode.get("scene_key") not in shard_scenes
        }
        for episode_id, episode in shard_status.get("episodes", {}).items():
            if episode_id not in master_candidates:
                raise ValueError(f"unknown episode candidate in {shard_name}: {episode_id}")
            local_episode = dict(episode)
            candidate = master_candidates[episode_id]
            local_episode["bundle"] = candidate["bundle"]
            local_episode["group"] = str(Path(candidate["group"]) / "group.json")
            status["episodes"][episode_id] = local_episode
        applied.add(shard_name)
        applied_scenes.update(shard_scenes)
        _write_json(manifest_path, manifest)
        _write_json(status_path, status)
        ledger["applied_shards"] = sorted(applied)
        ledger["applied_scene_keys"] = sorted(applied_scenes)
        ledger["current_manifest_sha256"] = _sha256(manifest_path)
        ledger["current_status_sha256"] = _sha256(status_path)
        _write_json(ledger_path, ledger)
        print(f"merge applied shard={shard_name} scenes={len(shard_scenes)}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="make immutable scene-level render shards")
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--status", type=Path)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--num-shards", type=int, default=2)
    build.add_argument(
        "--backend-root",
        type=Path,
        default=Path(
            os.environ.get("EPISPACE_BACKEND_ROOT", "/home/wmq/project/bench/OminiGibson")
        ),
        help="Git checkout containing src/omnigibson_episode; its commit is recorded",
    )
    build.add_argument("--preview", action="store_true")
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--no-archive", action="store_true")
    build.add_argument(
        "--hardlink",
        action="store_true",
        help="save local disk space with hard links; archives remain self-contained",
    )
    build.set_defaults(handler=command_build)

    prepare = commands.add_parser("prepare", help="materialize absolute paths on a worker")
    prepare.add_argument("--package", type=Path, required=True)
    prepare.add_argument("--work-root", type=Path, required=True)
    prepare.set_defaults(handler=command_prepare)

    finalize = commands.add_parser("finalize", help="write a result completion marker")
    finalize.add_argument("--work-root", type=Path, required=True)
    finalize.set_defaults(handler=command_finalize)

    merge = commands.add_parser("merge", help="merge completed, disjoint shard results")
    merge.add_argument("--manifest", type=Path, required=True)
    merge.add_argument("--status", type=Path)
    merge.add_argument("--results", type=Path, nargs="+", required=True)
    merge.set_defaults(handler=command_merge)
    return result


def main() -> int:
    args = parser().parse_args()
    if getattr(args, "num_shards", 1) < 1:
        raise ValueError("--num-shards must be positive")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
