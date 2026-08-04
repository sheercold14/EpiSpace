"""Cryptographically bind a release to the code that produced and verifies it.

The dataset artifacts and their source inputs are already content addressed.
Those hashes are insufficient when compiler semantics change without changing
the inputs: an older manifest could otherwise continue to look valid.  This
module defines a deliberately closed set of release-critical implementation
files and builds a deterministic binding over their relative paths, sizes, and
SHA-256 digests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

IMPLEMENTATION_BINDING_SCHEMA = "epispace.implementation_binding.v1"
IMPLEMENTATION_FILE_SET = "epispace.release_critical_files.v2"

# This is an explicit dependency contract, not a best-effort directory scan.
# Adding a new module or script to the release build requires adding it here;
# the verifier rejects manifests with either missing or additional entries.
RELEASE_CRITICAL_PATHS = (
    "Makefile",
    "episode3d/__init__.py",
    "episode3d/benchmark_evaluator.py",
    "episode3d/bundles.py",
    "episode3d/cli.py",
    "episode3d/compilers.py",
    "episode3d/compute_matching.py",
    "episode3d/corpus_audit.py",
    "episode3d/exporters.py",
    "episode3d/implementation_binding.py",
    "episode3d/language.py",
    "episode3d/models.py",
    "episode3d/pipeline.py",
    "episode3d/programs.py",
    "episode3d/release_verifier.py",
    "episode3d/semantic_visual_audit.py",
    "episode3d/semantic_visual_audit_results.py",
    "episode3d/verifiers.py",
    "scripts/adjudicate_semantic_visual_audit.py",
    "scripts/build_compute_matched_schedules.py",
    "scripts/build_semantic_visual_audit_packet.py",
    "scripts/build_web_release_catalog.py",
)


def project_root() -> Path:
    """Return the source root containing ``episode3d/`` and ``scripts/``."""

    return Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_sha256(files: dict[str, dict[str, Any]]) -> str:
    canonical = json.dumps(
        {
            "file_set": IMPLEMENTATION_FILE_SET,
            "files": files,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_implementation_binding(root: Path | None = None) -> dict[str, Any]:
    """Build the deterministic release-critical implementation binding.

    Paths stored in the result are always relative to the project root.  The
    absolute checkout path is intentionally excluded so that a release can be
    verified after the repository is relocated.
    """

    root = (root or project_root()).resolve()
    files: dict[str, dict[str, Any]] = {}
    for relative in RELEASE_CRITICAL_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"release-critical implementation file is absent: {path}")
        files[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return {
        "schema_version": IMPLEMENTATION_BINDING_SCHEMA,
        "file_set": IMPLEMENTATION_FILE_SET,
        "root_policy": "paths_relative_to_current_episode3d_project_root",
        "files": files,
        "aggregate_sha256": _aggregate_sha256(files),
    }
