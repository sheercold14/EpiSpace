"""Fail-closed P2/P3 worker verification and scratch ownership tests."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_verifier():
    path = ROOT / "scripts" / "verify_p23_shard_output.py"
    spec = importlib.util.spec_from_file_location("verify_p23_shard_output", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFY = _load_verifier()


def _load_preflight():
    path = ROOT / "scripts" / "p23_worker_preflight.py"
    spec = importlib.util.spec_from_file_location("p23_worker_preflight", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PREFLIGHT = _load_preflight()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _worker_fixture(tmp_path: Path, *, badged_frames: int = 2) -> Path:
    work = tmp_path / "work"
    output = work / "output"
    bundle = output / "bundles" / "candidate"
    group = output / "groups" / "candidate"
    manifest_path = output / "coverage.plan.json"
    status_path = output / "coverage.status.json"
    manifest = {
        "collection_id": "p23_p2_7k_v1",
        "cells": [
            {
                "cell_id": "cell",
                "binding": {"viewpoint": "one", "target": "two"},
                "candidates": [
                    {
                        "candidate_id": "candidate",
                        "bundle": str(bundle),
                        "group": str(group),
                    }
                ],
            }
        ],
    }
    status = {
        "collection_id": "p23_p2_7k_v1",
        "cells": {
            "cell": {
                "candidate_statuses": {"candidate": {"status": "accepted"}}
            }
        },
        "episodes": {"candidate": {}},
    }
    _write(manifest_path, manifest)
    _write(status_path, status)
    _write(bundle / "render_report.json", {"status": "success"})
    _write(group / "group.json", {"schema_version": "scriptgen_question_group.v1"})
    _write(
        group / "media" / "marker.audit.json",
        {
            "schema_version": "scriptgen_marker_audit.v1",
            "min_badged_frames": 2,
            "assignments": [
                {"entity_id": "one", "badged_frames": badged_frames},
                {"entity_id": "two", "badged_frames": 3},
            ],
        },
    )
    _write(
        output / "shard.complete.json",
        {
            "manifest_sha256": _sha256(manifest_path),
            "status_sha256": _sha256(status_path),
        },
    )
    return work


def test_p23_output_verifier_checks_terminal_marker_artifacts(tmp_path: Path) -> None:
    work = _worker_fixture(tmp_path)

    report = VERIFY.verify("p2", work)

    assert report["status"] == "pass"
    assert report["verified_rendered_candidates"] == 1
    assert report["verified_marker_assignments"] == 2
    assert report["accepted_target_checked"] is False


def test_p23_output_verifier_rejects_insufficient_marker_evidence(
    tmp_path: Path,
) -> None:
    work = _worker_fixture(tmp_path, badged_frames=1)

    with pytest.raises(ValueError, match="marker evidence is insufficient"):
        VERIFY.verify("p2", work)


def test_scratch_janitor_only_removes_owned_sentinel_directories(
    tmp_path: Path,
) -> None:
    scratch_root = tmp_path / "epispace-scratch"
    owned = scratch_root / "run-owned"
    unrelated = scratch_root / "run-unrelated"
    owned.mkdir(parents=True)
    unrelated.mkdir()
    (owned / ".epispace_scratch").write_text("epispace_scratch.v1\n", encoding="utf-8")
    (unrelated / "keep.txt").write_text("not EpiSpace scratch", encoding="utf-8")
    old = 1_700_000_000
    os.utime(owned, (old, old))

    subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "tmp_scratch_janitor.sh"),
            "--once",
            str(scratch_root),
        ],
        env={**os.environ, "GRACE_MINUTES": "0"},
        check=True,
    )

    assert not owned.exists()
    assert (unrelated / "keep.txt").is_file()


def test_worker_preflight_binds_audit_distribution_archive_and_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "package"
    backend = tmp_path / "backend"
    data = tmp_path / "data"
    work = tmp_path / "work"
    output = work / "output" / "p23.worker_preflight.json"
    archive = tmp_path / "shard.tar.zst"
    distribution = tmp_path / "distribution.json"
    audit = tmp_path / "quality.json"
    backend.joinpath("src/omnigibson_episode").mkdir(parents=True)
    backend.joinpath("src/omnigibson_episode/cli.py").write_text("", encoding="utf-8")
    data.mkdir()
    package.mkdir()
    archive.write_bytes(b"immutable archive")
    manifest_hash = "a" * 64
    archive_hash = _sha256(archive)
    revisions = {
        "epispace": {"commit": "b" * 40},
        "omnigibson_episode": {"commit": "c" * 40},
    }
    _write(
        package / "shard.json",
        {
            "collection_id": "p23_p2_7k_v1",
            "shard_index": 0,
            "shard_name": "shard",
            "estimated_frames": 25,
            "base_manifest_sha256": manifest_hash,
            "code_revisions": revisions,
        },
    )
    _write(
        package / "coverage.plan.template.json",
        {"collection_id": "p23_p2_7k_v1", "standard_version": "std.v11"},
    )
    _write(
        distribution,
        {
            "base_manifest_sha256": manifest_hash,
            "archives": [{"name": archive.name, "sha256": archive_hash}],
        },
    )
    _write(
        audit,
        {
            "status": "pass",
            "physical_trajectories": 7000,
            "inputs": {"p2": {"sha256": manifest_hash}},
        },
    )
    monkeypatch.setattr(
        PREFLIGHT,
        "_git_check",
        lambda root, expected, label: {"root": str(root), "commit": expected},
    )
    monkeypatch.setattr(
        PREFLIGHT,
        "_gpu_rows",
        lambda gpu_ids: [{"index": index} for index in gpu_ids],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "p23_worker_preflight.py",
            "--phase",
            "p2",
            "--shard-index",
            "0",
            "--package",
            str(package),
            "--distribution",
            str(distribution),
            "--quality-audit",
            str(audit),
            "--archive",
            str(archive),
            "--repo-root",
            str(tmp_path),
            "--backend-root",
            str(backend),
            "--data-root",
            str(data),
            "--work-root",
            str(work),
            "--gpu-ids",
            "0",
            "1",
            "--min-free-gib",
            "0",
            "--scratch-reserve-gib",
            "0",
            "--output",
            str(output),
        ],
    )

    assert PREFLIGHT.main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["base_manifest_sha256"] == manifest_hash
    assert report["archive"]["sha256"] == archive_hash
    assert report["disk"]["estimated_frames"] == 25
    assert report["disk"]["bytes_per_frame_budget"] == 4 * 1024**2
    assert set(report["dependencies"]) == {"numpy", "scipy", "Pillow", "pydantic"}
