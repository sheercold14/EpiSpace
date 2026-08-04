"""Verify freshness and closure of a complete EpiSpace research release.

Unlike a checksum-list reader, this verifier does not trust derived reports or
their manifests.  It replays the source inventory, schedule contract, corpus
audit, oracle evaluation, and release-critical implementation binding against
the current files before writing a final release index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

from PIL import Image

from episode3d.implementation_binding import (
    IMPLEMENTATION_BINDING_SCHEMA,
    IMPLEMENTATION_FILE_SET,
    RELEASE_CRITICAL_PATHS,
    build_implementation_binding,
)


class ReleaseVerificationError(ValueError):
    """Raised when a release or a derived report is missing or stale."""


UTC = timezone.utc
RELEASE_SCHEMA = "epispace.release_manifest.v1"
SOURCE_SCHEMA = "epispace.source_inventory.v1"
SCHEDULE_SCHEMA = "epispace.compute_matching.v1"
AUDIT_SCHEMA = "epispace.corpus_audit.v1"
EVALUATION_SCHEMA = "epispace.benchmark_evaluation.v2"
REQUIRED_REGIMES = frozenset({"fact_matched", "image_occurrence_matched"})
REQUIRED_BUNDLE_FILES = frozenset(
    {
        "scene_ir.json",
        "spatial_episode.json",
        "trajectory_plan.json",
        "quality_report.json",
    }
)
OPTIONAL_BUNDLE_FILES = frozenset({"render_report.json", "reasoning_tasks.json"})
SEMANTIC_AUDIT_DIR = "semantic_visual_audit"
SEMANTIC_AUDIT_PACKET = f"{SEMANTIC_AUDIT_DIR}/packet.json"
SEMANTIC_AUDIT_PACKET_MARKDOWN = f"{SEMANTIC_AUDIT_DIR}/packet.md"
SEMANTIC_AUDIT_REVIEWS = f"{SEMANTIC_AUDIT_DIR}/reviews"
SEMANTIC_AUDIT_RESULT = f"{SEMANTIC_AUDIT_DIR}/result.json"
SEMANTIC_AUDIT_RESULT_MARKDOWN = f"{SEMANTIC_AUDIT_DIR}/result.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_non_finite(value: str) -> None:
    raise ReleaseVerificationError(f"non-finite JSON number is forbidden: {value}")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_non_finite)
    except ReleaseVerificationError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseVerificationError(f"cannot read valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ReleaseVerificationError(f"JSON root is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ReleaseVerificationError(f"cannot read JSONL: {path}") from exc
    rows: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, parse_constant=_reject_non_finite)
        except ReleaseVerificationError:
            raise
        except json.JSONDecodeError as exc:
            raise ReleaseVerificationError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ReleaseVerificationError(f"JSONL row is not an object at {path}:{line_number}")
        rows.append((line_number, value))
    return rows


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseVerificationError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _release_path(base: Path, value: Any, label: str) -> Path:
    _require(isinstance(value, str) and value, f"{label} must be a relative path")
    raw = Path(value)
    _require(not raw.is_absolute(), f"{label} must not be absolute: {value}")
    path = (base / raw).resolve()
    try:
        path.relative_to(base.resolve())
    except ValueError as exc:
        raise ReleaseVerificationError(f"{label} escapes its release boundary: {value}") from exc
    return path


def _external_path(base: Path, value: Any, label: str) -> Path:
    _require(isinstance(value, str) and value, f"{label} must be a path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _verify_file_record(path: Path, expected: Mapping[str, Any], label: str) -> str:
    _require(path.is_file(), f"{label} is absent: {path}")
    expected_sha = expected.get("sha256")
    expected_bytes = expected.get("bytes")
    _require(isinstance(expected_sha, str), f"{label} has no SHA-256")
    _require(
        isinstance(expected_bytes, int)
        and not isinstance(expected_bytes, bool)
        and expected_bytes >= 0,
        f"{label} has invalid byte count",
    )
    actual_sha = sha256(path)
    _require(actual_sha == expected_sha, f"{label} SHA mismatch: {path}")
    _require(path.stat().st_size == expected_bytes, f"{label} size mismatch: {path}")
    return actual_sha


def _verify_implementation_binding(
    manifest: Mapping[str, Any], *, implementation_root: Path | None = None
) -> dict[str, Any]:
    """Fail closed unless the manifest matches the current release code exactly."""

    stored = _mapping(
        manifest.get("implementation_integrity"), "implementation_integrity"
    )
    _require(
        stored.get("schema_version") == IMPLEMENTATION_BINDING_SCHEMA,
        "unsupported or absent implementation binding schema",
    )
    _require(
        stored.get("file_set") == IMPLEMENTATION_FILE_SET,
        "unsupported or absent implementation file-set contract",
    )
    _require(
        stored.get("root_policy")
        == "paths_relative_to_current_episode3d_project_root",
        "unsupported implementation root policy",
    )
    stored_files = _mapping(stored.get("files"), "implementation_integrity.files")
    required_paths = set(RELEASE_CRITICAL_PATHS)
    _require(
        set(stored_files) == required_paths,
        "implementation binding file set is not an exact release-critical closure",
    )

    try:
        current = build_implementation_binding(implementation_root)
    except OSError as exc:
        raise ReleaseVerificationError(
            f"cannot read release-critical implementation: {exc}"
        ) from exc
    current_files = _mapping(current["files"], "current implementation files")
    for relative in RELEASE_CRITICAL_PATHS:
        recorded = _mapping(
            stored_files[relative], f"implementation_integrity.files.{relative}"
        )
        expected = _mapping(current_files[relative], f"current implementation {relative}")
        _require(
            recorded.get("path") == relative,
            f"release-critical implementation path mismatch: {relative}",
        )
        _require(
            recorded.get("sha256") == expected.get("sha256"),
            f"release-critical implementation SHA mismatch: {relative}",
        )
        _require(
            recorded.get("bytes") == expected.get("bytes"),
            f"release-critical implementation size mismatch: {relative}",
        )
    _require(
        stored.get("aggregate_sha256") == current.get("aggregate_sha256"),
        "release-critical implementation aggregate SHA mismatch",
    )
    return {
        "schema_version": IMPLEMENTATION_BINDING_SCHEMA,
        "file_set": IMPLEMENTATION_FILE_SET,
        "aggregate_sha256": current["aggregate_sha256"],
        "verified_files": len(RELEASE_CRITICAL_PATHS),
        "files": current["files"],
    }


def _verify_base(release: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require(manifest.get("schema_version") == RELEASE_SCHEMA, "unsupported release schema")
    _require(manifest.get("status") == "pass", "release manifest status is not pass")
    checks = _mapping(
        _mapping(manifest.get("corpus_gates"), "corpus_gates").get("checks"),
        "corpus_gates.checks",
    )
    _require(
        bool(checks) and all(value is True for value in checks.values()),
        "one or more corpus gates failed or are non-boolean",
    )
    artifacts = _mapping(manifest.get("artifacts"), "artifacts")
    integrity = _mapping(manifest.get("artifact_integrity"), "artifact_integrity")
    _require(
        set(artifacts) == set(integrity), "artifact_integrity keys do not exactly match artifacts"
    )

    verified: dict[str, Any] = {}
    for name in sorted(artifacts):
        expected = _mapping(integrity[name], f"artifact_integrity.{name}")
        _require(
            expected.get("path") == artifacts[name],
            f"artifact path declarations disagree: {name}",
        )
        path = _release_path(release, artifacts[name], f"artifact {name}")
        actual_sha = _verify_file_record(path, expected, f"artifact {name}")
        records = expected.get("records")
        if records is not None:
            _require(
                isinstance(records, int) and not isinstance(records, bool) and records >= 0,
                f"artifact {name} has invalid record count",
            )
            line_count = len(_read_jsonl(path))
            _require(line_count == records, f"record count mismatch: {name}")
        verified[name] = {
            "path": os.path.relpath(path, start=release),
            "sha256": actual_sha,
        }
    return verified


def _verify_source_inventory(
    release: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], set[Path], Path]:
    artifacts = _mapping(manifest.get("artifacts"), "artifacts")
    inventory_path = _release_path(release, artifacts.get("source_inventory"), "source inventory")
    inventory = read_json(inventory_path)
    _require(
        inventory.get("schema_version") == SOURCE_SCHEMA, "unsupported source inventory schema"
    )
    source_integrity = _mapping(manifest.get("source_integrity"), "source_integrity")
    inventory_sha = sha256(inventory_path)
    _require(
        inventory_sha == source_integrity.get("source_inventory_sha256"),
        "source inventory is stale",
    )

    config = _mapping(inventory.get("config"), "source inventory config")
    config_path = _external_path(inventory_path.parent, config.get("path"), "config path")
    _require(config_path.is_file(), f"pipeline config is absent: {config_path}")
    config_sha = sha256(config_path)
    _require(config_sha == config.get("sha256"), "pipeline config changed")
    _require(
        config_sha == source_integrity.get("config_sha256"),
        "manifest config SHA disagrees with source inventory",
    )
    config_data = read_json(config_path)
    config_sources_raw = config_data.get("sources")
    _require(isinstance(config_sources_raw, list), "pipeline config sources must be a list")
    config_sources: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(config_sources_raw):
        source = _mapping(raw, f"pipeline config source {index}")
        name = source.get("name")
        _require(
            isinstance(name, str) and name and name not in config_sources,
            f"invalid or duplicate pipeline source name: {name}",
        )
        config_sources[name] = source

    sweeps = inventory.get("sweeps")
    _require(isinstance(sweeps, list) and sweeps, "source inventory has no sweeps")
    sweep_hashes: dict[str, str] = {}
    sweep_plans: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(sweeps):
        sweep = _mapping(raw, f"sweeps[{index}]")
        name = sweep.get("name")
        _require(
            isinstance(name, str) and name and name not in sweep_hashes,
            f"invalid or duplicate sweep name: {name}",
        )
        path = _external_path(inventory_path.parent, sweep.get("path"), f"sweep {name}")
        _require(path.is_file(), f"sweep plan is absent: {path}")
        actual = sha256(path)
        _require(actual == sweep.get("sha256"), f"sweep plan changed: {path}")
        sweep_hashes[name] = actual
        sweep_plans[name] = read_json(path)
    _require(
        sweep_hashes
        == dict(
            _mapping(
                source_integrity.get("sweep_plan_sha256"), "source_integrity.sweep_plan_sha256"
            )
        ),
        "manifest sweep hashes disagree with source inventory",
    )
    _require(
        set(sweep_hashes) == set(config_sources),
        "source inventory sweeps disagree with pipeline config",
    )
    for name, source in config_sources.items():
        configured_path = _external_path(
            config_path.parent, source.get("sweep_plan"), f"configured sweep {name}"
        )
        _require(
            configured_path.is_file() and sha256(configured_path) == sweep_hashes[name],
            f"configured sweep path/hash disagrees: {name}",
        )

    acquisition = _mapping(
        _mapping(manifest.get("statistics"), "statistics").get("acquisition"),
        "statistics.acquisition",
    )
    media_root = _external_path(release, acquisition.get("model_rgb_root"), "model RGB root")
    _require(media_root.is_dir(), f"model RGB root is absent: {media_root}")

    bundles = inventory.get("bundles")
    _require(isinstance(bundles, list), "source inventory bundles must be a list")
    _require(
        inventory.get("bundle_count") == len(bundles), "source inventory bundle_count mismatch"
    )
    referenced_pngs: set[Path] = set()
    sensor_hash_cache: dict[Path, str] = {}
    view_keys: set[tuple[str, str]] = set()
    bundle_counts: Counter[str] = Counter()
    missing_reasoning_counts: Counter[str] = Counter()
    total_views = 0
    for bundle_index, raw_bundle in enumerate(bundles):
        bundle = _mapping(raw_bundle, f"bundles[{bundle_index}]")
        episode_id = bundle.get("episode_id")
        _require(
            isinstance(episode_id, str) and episode_id, f"bundle {bundle_index} has no episode_id"
        )
        source_sweep = bundle.get("source_sweep")
        _require(
            isinstance(source_sweep, str) and source_sweep in config_sources,
            f"bundle {episode_id} has unknown source_sweep",
        )
        _require(
            bundle.get("trajectory_class") == config_sources[source_sweep].get("trajectory_class"),
            f"bundle {episode_id} trajectory class disagrees with config",
        )
        bundle_counts[source_sweep] += 1
        root = _external_path(
            inventory_path.parent, bundle.get("bundle_root"), f"bundle {episode_id} root"
        )
        _require(root.is_dir(), f"bundle root is absent: {root}")
        source_files = _mapping(bundle.get("source_files"), f"bundle {episode_id} files")
        _require(
            set(source_files) >= REQUIRED_BUNDLE_FILES,
            f"bundle {episode_id} inventory omits required JSON files",
        )
        absent_optional = bundle.get("absent_optional_files")
        _require(
            isinstance(absent_optional, list)
            and all(isinstance(name, str) for name in absent_optional)
            and set(absent_optional) <= OPTIONAL_BUNDLE_FILES,
            f"bundle {episode_id} has invalid absent_optional_files",
        )
        _require(
            (set(source_files) & OPTIONAL_BUNDLE_FILES) | set(absent_optional)
            == OPTIONAL_BUNDLE_FILES,
            f"bundle {episode_id} does not close optional source-file presence",
        )
        _require(
            not ((set(source_files) & OPTIONAL_BUNDLE_FILES) & set(absent_optional)),
            f"bundle {episode_id} marks an optional file both present and absent",
        )
        for filename in absent_optional:
            _require(
                not (root / filename).exists(),
                f"previously absent optional source appeared: {root / filename}",
            )
        if "reasoning_tasks.json" in absent_optional:
            missing_reasoning_counts[source_sweep] += 1
        for filename, raw_expected in source_files.items():
            _require(
                isinstance(filename, str) and filename,
                f"bundle {episode_id} has invalid source filename",
            )
            path = _release_path(root, filename, f"bundle {episode_id} source file")
            _verify_file_record(
                path,
                _mapping(raw_expected, f"bundle {episode_id}/{filename}"),
                f"bundle source {episode_id}/{filename}",
            )

        views = bundle.get("views")
        _require(isinstance(views, list) and views, f"bundle {episode_id} has no inventoried views")
        total_views += len(views)
        for view_index, raw_view in enumerate(views):
            view = _mapping(raw_view, f"bundle {episode_id} view {view_index}")
            view_id = view.get("view_id")
            key = (episode_id, str(view_id))
            _require(
                isinstance(view_id, str) and view_id and key not in view_keys,
                f"invalid or duplicate view key: {key}",
            )
            view_keys.add(key)

            rgb = _external_path(inventory_path.parent, view.get("rgb"), "RGB path")
            try:
                rgb.relative_to(media_root)
            except ValueError as exc:
                raise ReleaseVerificationError(
                    f"inventoried RGB escapes model_rgb_root: {rgb}"
                ) from exc
            rgb_expected = {
                "sha256": view.get("rgb_file_sha256"),
                "bytes": view.get("rgb_bytes"),
            }
            _verify_file_record(rgb, rgb_expected, f"RGB {episode_id}/{view_id}")
            _require(rgb.suffix.lower() == ".png", f"model RGB is not PNG: {rgb}")
            _require(rgb not in referenced_pngs, f"RGB reused by multiple views: {rgb}")
            referenced_pngs.add(rgb)

            sidecar = _external_path(inventory_path.parent, view.get("rgb_sidecar"), "RGB sidecar")
            _require(sidecar.is_file(), f"RGB sidecar is absent: {sidecar}")
            _require(
                sha256(sidecar) == view.get("rgb_sidecar_sha256"), f"RGB sidecar changed: {sidecar}"
            )
            metadata = read_json(sidecar)
            _require(
                metadata.get("schema_version") == "epispace.raw_rgb.v1",
                f"invalid RGB sidecar schema: {sidecar}",
            )

            sensor = _external_path(
                inventory_path.parent, view.get("source_sensor"), "source sensor"
            )
            _require(sensor.is_file(), f"source sensor is absent: {sensor}")
            _require(
                sensor.stat().st_size == view.get("source_sensor_bytes"),
                f"source sensor size changed: {sensor}",
            )
            sensor_sha = sensor_hash_cache.get(sensor)
            if sensor_sha is None:
                sensor_sha = sha256(sensor)
                sensor_hash_cache[sensor] = sensor_sha
            _require(
                sensor_sha == view.get("source_sensor_sha256"),
                f"source sensor content changed: {sensor}",
            )
            _require(
                metadata.get("source_size") == view.get("source_sensor_bytes"),
                f"sidecar sensor size disagrees: {sidecar}",
            )
            _require(
                metadata.get("source_mtime_ns") == view.get("source_sensor_mtime_ns"),
                f"sidecar sensor mtime disagrees: {sidecar}",
            )
            metadata_sensor = _external_path(
                sidecar.parent, metadata.get("source_sensor"), "sidecar source sensor"
            )
            _require(metadata_sensor == sensor, f"sidecar sensor path disagrees: {sidecar}")

            with Image.open(rgb) as image:
                image.load()
                _require(image.mode == "RGB", f"model image is not RGB: {rgb}")
                _require(
                    metadata.get("mode") == "RGB"
                    and image.width == metadata.get("width")
                    and image.height == metadata.get("height"),
                    f"RGB dimensions disagree with sidecar: {rgb}",
                )
                pixel_sha = hashlib.sha256(image.tobytes()).hexdigest()
            _require(
                pixel_sha == metadata.get("rgb_sha256"), f"RGB pixels disagree with sidecar: {rgb}"
            )
            _require(
                pixel_sha == view.get("rgb_pixel_sha256"),
                f"RGB pixels disagree with inventory: {rgb}",
            )

    _require(inventory.get("view_count") == total_views, "source inventory view_count mismatch")
    acquisition_sweeps_raw = acquisition.get("sweeps")
    _require(isinstance(acquisition_sweeps_raw, list), "acquisition sweeps must be a list")
    acquisition_sweeps: dict[str, Mapping[str, Any]] = {}
    total_planned = 0
    for index, raw in enumerate(acquisition_sweeps_raw):
        item = _mapping(raw, f"acquisition sweep {index}")
        name = item.get("name")
        _require(
            isinstance(name, str) and name in config_sources and name not in acquisition_sweeps,
            f"invalid acquisition sweep name: {name}",
        )
        acquisition_sweeps[name] = item
        plan = sweep_plans[name]
        jobs = plan.get("jobs")
        _require(isinstance(jobs, list), f"sweep plan {name} has no jobs list")
        status_counts = Counter(str(job.get("status", "missing")) for job in jobs)
        total_planned += len(jobs)
        configured_path = _external_path(
            config_path.parent,
            config_sources[name].get("sweep_plan"),
            f"configured sweep {name}",
        )
        _require(
            item.get("sweep_plan_sha256") == sweep_hashes[name],
            f"acquisition sweep SHA is stale: {name}",
        )
        _require(
            _external_path(release, item.get("sweep_plan"), f"acquisition sweep {name}")
            == configured_path,
            f"acquisition sweep path is stale: {name}",
        )
        _require(
            item.get("trajectory_class") == config_sources[name].get("trajectory_class"),
            f"acquisition trajectory class is stale: {name}",
        )
        _require(
            item.get("role") == config_sources[name].get("role"),
            f"acquisition role is stale: {name}",
        )
        _require(
            item.get("planned_jobs") == len(jobs),
            f"acquisition planned-job count is stale: {name}",
        )
        _require(
            item.get("status_counts") == dict(sorted(status_counts.items())),
            f"acquisition status counts are stale: {name}",
        )
        _require(
            item.get("strict_bundles_loaded") == bundle_counts[name],
            f"acquisition strict-bundle count is stale: {name}",
        )
        _require(
            item.get("missing_optional_reasoning_task_manifests") == missing_reasoning_counts[name],
            f"acquisition missing-reasoning count is stale: {name}",
        )
    _require(
        set(acquisition_sweeps) == set(config_sources),
        "acquisition sweeps disagree with config",
    )
    _require(
        acquisition.get("planned_jobs") == total_planned,
        "acquisition total planned-job count is stale",
    )
    _require(
        acquisition.get("strict_bundles_loaded") == len(bundles),
        "acquisition total strict-bundle count is stale",
    )
    return (
        {
            "inventory_sha256": inventory_sha,
            "config_sha256": config_sha,
            "sweep_count": len(sweeps),
            "bundle_count": len(bundles),
            "view_count": total_views,
            "sensor_count": len(sensor_hash_cache),
        },
        referenced_pngs,
        media_root,
    )


def _message_images(messages: Any, source_path: Path | None = None) -> list[str]:
    _require(isinstance(messages, list), "training messages must be a list")
    paths: list[str] = []
    for message in messages:
        _require(isinstance(message, Mapping), "training message must be an object")
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            _require(isinstance(block, Mapping), "message block must be an object")
            if block.get("type") == "image":
                path = block.get("image")
                _require(isinstance(path, str) and path, "image block has no path")
                if source_path is not None:
                    paths.append(str(_external_path(source_path.parent, path, "model image")))
                else:
                    paths.append(path)
    return paths


def _load_schedule_sources(
    release: Path,
    schedule_manifest_path: Path,
    schedule_manifest: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    sources = _mapping(schedule_manifest.get("sources"), "schedule sources")
    integrity = _mapping(manifest.get("artifact_integrity"), "artifact_integrity")
    artifacts = _mapping(manifest.get("artifacts"), "artifacts")
    result: dict[str, dict[str, Any]] = {}
    for arm, artifact_name in (("episode", "episode_sft"), ("isolated", "isolated_sft")):
        source = _mapping(sources.get(arm), f"schedule source {arm}")
        source_path = _external_path(
            schedule_manifest_path.parent, source.get("path"), f"schedule source {arm}"
        )
        expected_path = _release_path(release, artifacts.get(artifact_name), artifact_name)
        _require(
            source_path == expected_path,
            f"schedule source does not resolve to release artifact: {arm}",
        )
        expected_sha = _mapping(integrity.get(artifact_name), artifact_name).get("sha256")
        _require(source_path.is_file(), f"schedule source is absent: {source_path}")
        actual_sha = sha256(source_path)
        _require(
            actual_sha == expected_sha == source.get("sha256"), f"schedule source is stale: {arm}"
        )
        rows = _read_jsonl(source_path)
        _require(
            source.get("record_count") == len(rows), f"schedule source record count changed: {arm}"
        )
        by_line = {line: row for line, row in rows}
        by_record: dict[str, tuple[int, dict[str, Any]]] = {}
        for line, row in rows:
            record_id = row.get("record_id")
            _require(
                isinstance(record_id, str) and record_id and record_id not in by_record,
                f"invalid or duplicate {arm} source record_id: {record_id}",
            )
            contract = _mapping(
                row.get("comparison_contract"), f"source record {record_id} comparison_contract"
            )
            _require(contract.get("arm") == arm, f"source record {record_id} has wrong arm")
            facts = contract.get("fact_ids")
            _require(
                isinstance(facts, list)
                and facts
                and all(isinstance(value, str) and value for value in facts),
                f"source record {record_id} has invalid fact_ids",
            )
            images = _message_images(row.get("messages"), source_path)
            _require(images, f"source record {record_id} has no model images")
            by_record[record_id] = (line, row)
        result[arm] = {
            "path": source_path,
            "sha256": actual_sha,
            "rows": rows,
            "by_line": by_line,
            "by_record": by_record,
        }
    return result


def _validate_source_pairing(sources: Mapping[str, dict[str, Any]]) -> dict[str, int]:
    episode_by_comparison: dict[str, dict[str, Any]] = {}
    isolated_by_comparison: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for _line, row in sources["episode"]["rows"]:
        contract = _mapping(row["comparison_contract"], "episode comparison contract")
        comparison = contract.get("comparison_id")
        _require(
            isinstance(comparison, str) and comparison not in episode_by_comparison,
            f"invalid or duplicate episode comparison_id: {comparison}",
        )
        episode_by_comparison[comparison] = row
    for _line, row in sources["isolated"]["rows"]:
        contract = _mapping(row["comparison_contract"], "isolated comparison contract")
        comparison = contract.get("comparison_id")
        _require(isinstance(comparison, str), "invalid isolated comparison_id")
        isolated_by_comparison[comparison].append(row)
    _require(
        set(episode_by_comparison) == set(isolated_by_comparison),
        "episode and isolated comparison groups differ",
    )
    repetitions: dict[str, int] = {}
    for comparison, episode in episode_by_comparison.items():
        isolated = isolated_by_comparison[comparison]
        episode_contract = _mapping(episode["comparison_contract"], "episode contract")
        episode_facts = Counter(episode_contract["fact_ids"])
        isolated_facts: Counter[str] = Counter()
        episode_images = _message_images(episode["messages"], sources["episode"]["path"])
        for row in isolated:
            contract = _mapping(row["comparison_contract"], "isolated contract")
            _require(
                len(contract["fact_ids"]) == 1,
                f"isolated comparison {comparison} has a multi-fact record",
            )
            isolated_facts.update(contract["fact_ids"])
            _require(
                _message_images(row["messages"], sources["isolated"]["path"]) == episode_images,
                f"comparison {comparison} does not reuse identical image sequence",
            )
        _require(
            episode_facts == isolated_facts,
            f"comparison {comparison} has a different fact multiset",
        )
        repetitions[comparison] = len(isolated)
    return repetitions


def _expected_schedule_order(
    sources: Mapping[str, dict[str, Any]],
    repetitions: Mapping[str, int],
    *,
    regime: str,
    arm: str,
    seed: int,
) -> list[tuple[str, int, Fraction]]:
    draws: list[tuple[str, int, Fraction]] = []
    for record_id, (_line, row) in sources[arm]["by_record"].items():
        comparison = row["comparison_contract"]["comparison_id"]
        count = (
            repetitions[comparison]
            if regime == "image_occurrence_matched" and arm == "episode"
            else 1
        )
        weight = Fraction(1, count)
        draws.extend((record_id, repeat_index, weight) for repeat_index in range(count))

    def rank(draw: tuple[str, int, Fraction]) -> tuple[str, str, int]:
        record_id, repeat_index, _weight = draw
        payload = (
            f"epispace.compute_schedule.v1\0{seed}\0{regime}\0{arm}\0{record_id}\0{repeat_index}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest(), record_id, repeat_index

    return sorted(draws, key=rank)


def _verify_schedules(release: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    from episode3d.compute_matching import build_compute_matched_schedules

    path = release / "compute_matching" / "compute_matching_manifest.json"
    data = read_json(path)
    _require(data.get("schema_version") == SCHEDULE_SCHEMA, "unsupported compute matching schema")
    _require(data.get("status") == "pass", "compute matching status is not pass")
    seed = data.get("seed")
    _require(isinstance(seed, int) and not isinstance(seed, bool), "invalid schedule seed")
    sources = _load_schedule_sources(release, path, data, manifest)
    repetitions = _validate_source_pairing(sources)
    regimes = _mapping(data.get("regimes"), "schedule regimes")
    _require(
        set(regimes) >= REQUIRED_REGIMES, "release is missing a required compute-matching regime"
    )
    _require(
        set(regimes) <= REQUIRED_REGIMES, "release contains an unsupported compute-matching regime"
    )

    image_size_cache: dict[str, int] = {}
    schedule_hashes: dict[str, str] = {}
    for regime in sorted(regimes):
        details = _mapping(regimes[regime], f"schedule regime {regime}")
        _require(details.get("status") == "pass", f"schedule regime failed: {regime}")
        files = _mapping(details.get("schedule_files"), f"schedule files {regime}")
        _require(
            set(files) == {"episode", "isolated"},
            f"schedule regime {regime} must contain exactly two arms",
        )
        arm_facts: dict[str, Counter[str]] = {}
        arm_effective_facts: dict[str, defaultdict[str, Fraction]] = {}
        arm_images: dict[str, Counter[str]] = {}
        arm_pixels: dict[str, int] = {}
        for arm in ("episode", "isolated"):
            entry = _mapping(files[arm], f"schedule file {regime}/{arm}")
            schedule = _release_path(path.parent, entry.get("filename"), f"schedule {regime}/{arm}")
            _require(schedule.is_file(), f"schedule is absent: {schedule}")
            actual_sha = sha256(schedule)
            _require(actual_sha == entry.get("sha256"), f"schedule SHA mismatch: {regime}/{arm}")
            rows = _read_jsonl(schedule)
            expected = _expected_schedule_order(
                sources, repetitions, regime=regime, arm=arm, seed=seed
            )
            _require(len(rows) == len(expected), f"schedule draw count mismatch: {regime}/{arm}")
            facts: Counter[str] = Counter()
            effective: defaultdict[str, Fraction] = defaultdict(Fraction)
            images: Counter[str] = Counter()
            pixels = 0
            for schedule_index, ((line_number, row), expected_draw) in enumerate(
                zip(rows, expected, strict=True)
            ):
                _require(
                    row.get("schedule_index") == schedule_index,
                    f"non-contiguous schedule_index at {schedule}:{line_number}",
                )
                record_id, repeat_index, expected_weight = expected_draw
                _require(
                    row.get("record_id") == record_id,
                    f"unexpected record order at {schedule}:{line_number}",
                )
                _require(
                    row.get("repeat_index") == repeat_index,
                    f"unexpected repeat_index at {schedule}:{line_number}",
                )
                _require(row.get("arm") == arm, f"wrong schedule arm at {schedule}:{line_number}")
                source_path = _external_path(
                    schedule.parent, row.get("source_jsonl"), "draw source_jsonl"
                )
                _require(
                    source_path == sources[arm]["path"],
                    f"draw points outside its declared source: {schedule}:{line_number}",
                )
                _require(
                    row.get("source_jsonl_sha256") == sources[arm]["sha256"],
                    f"draw source_jsonl_sha256 mismatch at {schedule}:{line_number}",
                )
                source_line = row.get("source_line")
                _require(
                    isinstance(source_line, int)
                    and not isinstance(source_line, bool)
                    and source_line in sources[arm]["by_line"],
                    f"invalid source_line at {schedule}:{line_number}",
                )
                source_row = sources[arm]["by_line"][source_line]
                _require(
                    source_row.get("record_id") == record_id,
                    f"source_line record mismatch at {schedule}:{line_number}",
                )
                contract = _mapping(source_row["comparison_contract"], "source contract")
                _require(
                    row.get("comparison_id") == contract.get("comparison_id"),
                    f"comparison_id mismatch at {schedule}:{line_number}",
                )
                _require(
                    row.get("fact_ids") == contract.get("fact_ids"),
                    f"fact_ids mismatch at {schedule}:{line_number}",
                )
                source_images = _message_images(source_row["messages"], sources[arm]["path"])
                _require(
                    row.get("image_paths") == source_images,
                    f"image path list mismatch at {schedule}:{line_number}",
                )
                ratio_raw = row.get("sample_weight_ratio")
                _require(
                    isinstance(ratio_raw, str),
                    f"missing exact sample weight at {schedule}:{line_number}",
                )
                try:
                    ratio = Fraction(ratio_raw)
                except (ValueError, ZeroDivisionError) as exc:
                    raise ReleaseVerificationError(
                        f"invalid sample weight ratio at {schedule}:{line_number}"
                    ) from exc
                weight = row.get("sample_weight")
                _require(
                    isinstance(weight, int | float)
                    and not isinstance(weight, bool)
                    and math.isfinite(float(weight))
                    and float(weight) == float(ratio)
                    and ratio == expected_weight,
                    f"sample weight mismatch at {schedule}:{line_number}",
                )
                facts.update(contract["fact_ids"])
                for fact_id in contract["fact_ids"]:
                    effective[fact_id] += ratio
                images.update(source_images)
                for image_path in source_images:
                    if image_path not in image_size_cache:
                        image = _external_path(
                            sources[arm]["path"].parent,
                            image_path,
                            "schedule image",
                        )
                        _require(image.is_file(), f"schedule image is absent: {image}")
                        with Image.open(image) as decoded:
                            image_size_cache[image_path] = decoded.width * decoded.height
                    pixels += image_size_cache[image_path]
            arms = _mapping(details.get("arms"), f"schedule summaries {regime}")
            summary = _mapping(arms.get(arm), f"schedule summary {regime}/{arm}")
            _require(
                summary.get("schedule_draw_count") == len(rows),
                f"declared draw count mismatch: {regime}/{arm}",
            )
            _require(
                summary.get("fact_occurrence_count") == sum(facts.values()),
                f"declared fact count mismatch: {regime}/{arm}",
            )
            _require(
                summary.get("image_occurrence_count") == sum(images.values()),
                f"declared image count mismatch: {regime}/{arm}",
            )
            _require(
                summary.get("pixel_exposure") == pixels,
                f"declared pixel exposure mismatch: {regime}/{arm}",
            )
            arm_facts[arm] = facts
            arm_effective_facts[arm] = effective
            arm_images[arm] = images
            arm_pixels[arm] = pixels
            schedule_hashes[f"{regime}.{arm}"] = actual_sha

        observed = {
            "actual_fact_multiset_equal": arm_facts["episode"] == arm_facts["isolated"],
            "effective_fact_weight_multiset_equal": (
                arm_effective_facts["episode"] == arm_effective_facts["isolated"]
            ),
            "actual_image_occurrence_multiset_equal": (
                arm_images["episode"] == arm_images["isolated"]
            ),
            "actual_pixel_exposure_equal": arm_pixels["episode"] == arm_pixels["isolated"],
        }
        expected_required = (
            {
                "actual_fact_multiset_equal": True,
                "effective_fact_weight_multiset_equal": True,
            }
            if regime == "fact_matched"
            else {
                "effective_fact_weight_multiset_equal": True,
                "actual_image_occurrence_multiset_equal": True,
                "actual_pixel_exposure_equal": True,
            }
        )
        _require(
            dict(_mapping(details.get("required_invariants"), "required invariants"))
            == expected_required,
            f"wrong required invariants: {regime}",
        )
        _require(
            dict(_mapping(details.get("observed_invariants"), "observed invariants")) == observed,
            f"stale observed invariants: {regime}",
        )
        _require(
            all(observed[key] is value for key, value in expected_required.items()),
            f"recomputed matching invariants failed: {regime}",
        )

    # Rebuild at the same directory depth so every relative source pointer and
    # resulting schedule SHA is byte-identical.  This additionally checks all
    # advertised summaries (including exact multisets and proxy exposures), not
    # only the critical invariants replayed above.
    with tempfile.TemporaryDirectory(prefix=".verify-compute-", dir=release) as temporary:
        rebuilt = build_compute_matched_schedules(
            sources["episode"]["path"],
            sources["isolated"]["path"],
            Path(temporary),
            seed=seed,
            regimes=tuple(regimes),
        )
    _require(rebuilt == data, "compute matching manifest does not match deterministic replay")
    return {"manifest_sha256": sha256(path), "schedules": schedule_hashes}


def _verify_audit(release: Path, manifest_path: Path) -> dict[str, Any]:
    from episode3d.corpus_audit import audit_release

    path = release / "corpus_audit.json"
    data = read_json(path)
    _require(data.get("schema_version") == AUDIT_SCHEMA, "unsupported corpus audit schema")
    summary = _mapping(data.get("summary"), "corpus audit summary")
    _require(summary.get("status") != "fail", "corpus audit failed")
    _require(
        data.get("source_manifest_sha256") == sha256(manifest_path),
        "corpus audit is bound to a different release manifest",
    )
    recomputed = audit_release(release)
    _require(
        data.get("sections") == recomputed.get("sections"),
        "stored corpus audit sections do not match independent replay",
    )
    _require(
        data.get("summary") == recomputed.get("summary"),
        "stored corpus audit summary does not match independent replay",
    )
    return {
        "sha256": sha256(path),
        "status": summary.get("status"),
        "warnings": summary.get("warnings", []),
        "replayed": True,
    }


def _verify_evaluator(release: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    from episode3d.benchmark_evaluator import evaluate_files

    path = release / "evaluation_smoke" / "evaluation.json"
    predictions = release / "evaluation_smoke" / "oracle.predictions.jsonl"
    benchmark = _release_path(
        release,
        _mapping(manifest.get("artifacts"), "artifacts").get("benchmark_core"),
        "benchmark core",
    )
    data = read_json(path)
    _require(data.get("schema_version") == EVALUATION_SCHEMA, "unsupported evaluator report schema")
    inputs = _mapping(data.get("inputs"), "evaluator inputs")
    expected_benchmark = _mapping(
        _mapping(manifest.get("artifact_integrity"), "artifact_integrity").get("benchmark_core"),
        "benchmark_core integrity",
    ).get("sha256")
    _require(
        inputs.get("benchmark_sha256") == expected_benchmark == sha256(benchmark),
        "evaluator report is bound to a stale benchmark",
    )
    _require(predictions.is_file(), "oracle predictions are absent")
    _require(
        inputs.get("predictions_sha256") == sha256(predictions),
        "evaluator report is bound to stale oracle predictions",
    )
    recomputed = evaluate_files(benchmark, predictions)
    stored_body = {key: value for key, value in data.items() if key != "inputs"}
    replay_body = {key: value for key, value in recomputed.items() if key != "inputs"}
    _require(stored_body == replay_body, "stored evaluator report does not match evaluator replay")
    accuracy = _mapping(data.get("record_accuracy"), "record_accuracy").get("accuracy")
    _require(accuracy == 1.0, "oracle evaluator smoke is not 100%")
    return {
        "sha256": sha256(path),
        "benchmark_sha256": expected_benchmark,
        "predictions_sha256": sha256(predictions),
        "record_accuracy": 1.0,
        "replayed": True,
    }


def _collect_exported_pngs(release: Path, manifest: Mapping[str, Any]) -> set[Path]:
    """Collect every PNG exposed by a serialized model/evaluator artifact."""

    artifacts = _mapping(manifest.get("artifacts"), "artifacts")
    integrity = _mapping(manifest.get("artifact_integrity"), "artifact_integrity")
    result: set[Path] = set()

    def visit(value: Any, *, parent: Path, key: str | None = None) -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, parent=parent, key=str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, parent=parent, key=key)
        elif (
            isinstance(value, str)
            and key in {"image", "rgb", "oracle_target_rgb"}
            and value.lower().endswith(".png")
        ):
            result.add(_external_path(parent, value, f"exported {key}"))

    for name, relative in artifacts.items():
        records = _mapping(integrity.get(name), f"artifact integrity {name}").get("records")
        if not isinstance(records, int):
            continue
        path = _release_path(release, relative, f"artifact {name}")
        for _line, row in _read_jsonl(path):
            visit(row, parent=path.parent)
    return result


def _verify_statistics(release: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the manifest counts most likely to be quoted in a paper."""

    artifacts = _mapping(manifest.get("artifacts"), "artifacts")
    statistics = _mapping(manifest.get("statistics"), "statistics")
    exports = _mapping(statistics.get("exports"), "statistics.exports")

    def rows(name: str) -> list[dict[str, Any]]:
        path = _release_path(release, artifacts.get(name), f"artifact {name}")
        return [row for _line, row in _read_jsonl(path)]

    artifact_rows = {
        name: rows(name)
        for name in (
            "episode_ir",
            "episode_sft",
            "isolated_sft",
            "state_aux_sft",
            "rlvr",
            "benchmark",
            "benchmark_family",
            "benchmark_composition",
            "benchmark_core",
            "rejections",
        )
    }
    export_fields = {
        "episode_sft_records": "episode_sft",
        "isolated_sft_records": "isolated_sft",
        "state_aux_records": "state_aux_sft",
        "rlvr_records": "rlvr",
        "benchmark_records": "benchmark",
        "benchmark_family_records": "benchmark_family",
        "benchmark_composition_records": "benchmark_composition",
        "benchmark_core_records": "benchmark_core",
    }
    for field, artifact in export_fields.items():
        _require(
            exports.get(field) == len(artifact_rows[artifact]),
            f"release statistic is stale: exports.{field}",
        )
    family_groups = {
        row.get("consistency_group")
        for row in artifact_rows["benchmark_family"]
        if isinstance(row.get("consistency_group"), str)
    }
    _require(
        exports.get("benchmark_family_groups") == len(family_groups),
        "release statistic is stale: exports.benchmark_family_groups",
    )
    episode_fact_count = 0
    for row in artifact_rows["episode_sft"]:
        contract = _mapping(row.get("comparison_contract"), "episode comparison contract")
        facts = contract.get("fact_ids")
        _require(isinstance(facts, list), "episode comparison fact_ids must be a list")
        episode_fact_count += len(facts)
    _require(
        exports.get("episode_supervised_facts") == episode_fact_count,
        "release statistic is stale: exports.episode_supervised_facts",
    )

    ir = artifact_rows["episode_ir"]
    compiled = _mapping(statistics.get("compiled"), "statistics.compiled")
    trajectory_counts = Counter(str(row.get("trajectory_class")) for row in ir)
    split_scenes: defaultdict[str, set[str]] = defaultdict(set)
    split_bundles: Counter[str] = Counter()
    source_views = 0
    question_count = 0
    for row in ir:
        split = row.get("split")
        scene = row.get("scene_id")
        _require(
            isinstance(split, str) and isinstance(scene, str), "episode IR has invalid split/scene"
        )
        split_scenes[split].add(scene)
        split_bundles[split] += 1
        observations = row.get("observations")
        questions = row.get("questions")
        _require(
            isinstance(observations, list) and isinstance(questions, list),
            "episode IR has invalid observations/questions",
        )
        source_views += len(observations)
        question_count += len(questions)
    recomputed_compiled = {
        "unique_scenes": len({row["scene_id"] for row in ir}),
        "source_trajectory_bundles": len(ir),
        "source_views": source_views,
        "questions": question_count,
        "trajectory_counts": dict(sorted(trajectory_counts.items())),
        "split_scenes": {key: len(value) for key, value in sorted(split_scenes.items())},
        "split_bundles": dict(sorted(split_bundles.items())),
    }
    for field, value in recomputed_compiled.items():
        _require(
            compiled.get(field) == value,
            f"release statistic is stale: compiled.{field}",
        )

    rejection_rows = artifact_rows["rejections"]
    rejection_stats = _mapping(statistics.get("rejections"), "statistics.rejections")
    rejection_reasons = Counter(str(row.get("reason_code")) for row in rejection_rows)
    _require(
        rejection_stats.get("count") == len(rejection_rows),
        "release statistic is stale: rejections.count",
    )
    _require(
        rejection_stats.get("reason_counts") == dict(sorted(rejection_reasons.items())),
        "release statistic is stale: rejections.reason_counts",
    )
    return {
        "source_trajectories": len(ir),
        "source_views": source_views,
        "compiled_questions": question_count,
        "episode_supervised_facts": episode_fact_count,
        "benchmark_core_records": len(artifact_rows["benchmark_core"]),
    }


def _fixed_release_file(release: Path, relative: str, label: str) -> Path:
    """Resolve a formal-release file while rejecting symlink indirection."""

    lexical = release / relative
    _require(not lexical.is_symlink(), f"{label} must not be a symlink: {lexical}")
    path = _release_path(release, relative, label)
    _require(path.is_file(), f"{label} is absent: {path}")
    return path


def _verify_semantic_visual_audit(
    release: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay the independent RGB review and close it into release authority.

    The audit is deliberately derived from, rather than declared by, the
    release manifest.  This avoids a circular hash dependency: the packet
    binds the immutable manifest/IR/source inventory, while the final release
    index binds the packet, reviews, and adjudication result.
    """

    from episode3d.semantic_visual_audit import render_markdown as render_packet
    from episode3d.semantic_visual_audit_results import (
        SemanticVisualAuditResultsError,
        adjudicate_semantic_visual_audit,
    )
    from episode3d.semantic_visual_audit_results import (
        render_markdown as render_result,
    )

    packet_path = _fixed_release_file(
        release, SEMANTIC_AUDIT_PACKET, "semantic visual audit packet"
    )
    packet_markdown_path = _fixed_release_file(
        release,
        SEMANTIC_AUDIT_PACKET_MARKDOWN,
        "semantic visual audit packet Markdown",
    )
    result_path = _fixed_release_file(
        release, SEMANTIC_AUDIT_RESULT, "semantic visual audit result"
    )
    result_markdown_path = _fixed_release_file(
        release,
        SEMANTIC_AUDIT_RESULT_MARKDOWN,
        "semantic visual audit result Markdown",
    )
    lexical_reviews_dir = release / SEMANTIC_AUDIT_REVIEWS
    _require(
        not lexical_reviews_dir.is_symlink(),
        f"semantic visual audit reviews directory must not be a symlink: {lexical_reviews_dir}",
    )
    reviews_dir = _release_path(
        release, SEMANTIC_AUDIT_REVIEWS, "semantic visual audit reviews directory"
    )
    _require(reviews_dir.is_dir(), f"semantic visual audit reviews are absent: {reviews_dir}")
    review_children = sorted(reviews_dir.iterdir(), key=lambda path: path.name)
    _require(review_children, "semantic visual audit has no independent review files")
    _require(
        all(
            not path.is_symlink()
            and path.is_file()
            and path.suffix == ".json"
            for path in review_children
        ),
        "semantic visual audit reviews must contain only regular .json files",
    )
    reviewer_paths = [path.resolve() for path in review_children]

    packet = read_json(packet_path)
    binding = _mapping(packet.get("release_binding"), "audit packet release_binding")
    _require(
        _external_path(packet_path.parent, binding.get("release_dir"), "packet release_dir")
        == release,
        "semantic audit packet release_dir is stale",
    )
    _require(
        binding.get("dataset_id") == manifest.get("dataset_id"),
        "semantic audit packet dataset_id is stale",
    )

    artifacts = _mapping(manifest.get("artifacts"), "artifacts")
    integrity = _mapping(manifest.get("artifact_integrity"), "artifact_integrity")
    episode_ir_path = _release_path(
        release, artifacts.get("episode_ir"), "semantic audit episode_ir"
    )
    source_inventory_path = _release_path(
        release, artifacts.get("source_inventory"), "semantic audit source_inventory"
    )

    def verify_packet_binding(
        name: str,
        expected_path: Path,
        expected_sha256: str,
    ) -> Mapping[str, Any]:
        record = _mapping(binding.get(name), f"audit packet release_binding.{name}")
        bound_path = _external_path(
            packet_path.parent, record.get("path"), f"audit packet {name} path"
        )
        _require(bound_path == expected_path, f"semantic audit packet {name} path is stale")
        actual_sha256 = sha256(expected_path)
        _require(
            actual_sha256 == expected_sha256,
            f"current {name} disagrees with release manifest integrity",
        )
        _require(
            record.get("sha256") == actual_sha256,
            f"semantic audit packet {name} SHA is stale",
        )
        return record

    release_manifest_sha = sha256(manifest_path)
    verify_packet_binding("release_manifest", manifest_path, release_manifest_sha)
    episode_integrity = _mapping(integrity.get("episode_ir"), "episode_ir integrity")
    episode_binding = verify_packet_binding(
        "episode_ir", episode_ir_path, str(episode_integrity.get("sha256"))
    )
    source_integrity = _mapping(
        integrity.get("source_inventory"), "source_inventory integrity"
    )
    verify_packet_binding(
        "source_inventory",
        source_inventory_path,
        str(source_integrity.get("sha256")),
    )
    expected_episode_records = episode_integrity.get("records")
    if expected_episode_records is not None:
        _require(
            episode_binding.get("records") == expected_episode_records,
            "semantic audit packet episode_ir record count is stale",
        )
    _require(
        binding.get("final_release_index") is None,
        "semantic audit packet must not bind a prior final release index",
    )
    _require(
        packet.get("source_release_status") == "pass",
        "semantic audit packet source release status is not pass",
    )
    _require(
        packet_markdown_path.read_text(encoding="utf-8") == render_packet(packet),
        "semantic visual audit packet Markdown does not match packet replay",
    )

    stored_result = read_json(result_path)
    stored_policy = _mapping(stored_result.get("policy"), "semantic audit result policy")
    minor_policy = stored_policy.get("minor_issue_policy")
    _require(
        minor_policy in {"allow", "fail"},
        "semantic audit result has unsupported minor_issue_policy",
    )
    try:
        recomputed = adjudicate_semantic_visual_audit(
            packet_path,
            reviewer_paths,
            allow_minor=minor_policy == "allow",
        )
    except SemanticVisualAuditResultsError as exc:
        raise ReleaseVerificationError(
            f"semantic visual audit adjudication failed: {exc}"
        ) from exc
    _require(
        stored_result == recomputed,
        "semantic visual audit result does not exactly match adjudication replay",
    )
    decision = _mapping(recomputed.get("decision"), "semantic audit decision")
    _require(
        recomputed.get("status") == "pass"
        and decision.get("semantic_visual_audit_gate") is True,
        "semantic visual audit gate is not pass",
    )
    _require(
        result_markdown_path.read_text(encoding="utf-8") == render_result(recomputed),
        "semantic visual audit result Markdown does not match adjudication replay",
    )

    recomputed_reviewers = recomputed.get("reviewer_bindings")
    _require(
        isinstance(recomputed_reviewers, list)
        and len(recomputed_reviewers) == len(reviewer_paths),
        "semantic audit reviewer bindings are incomplete",
    )
    reviewer_index: list[dict[str, Any]] = []
    expected_reviewer_paths = set(reviewer_paths)
    for index, raw_reviewer in enumerate(recomputed_reviewers):
        reviewer = _mapping(raw_reviewer, f"semantic audit reviewer {index}")
        reviewer_path = Path(str(reviewer.get("path"))).expanduser().resolve()
        _require(
            reviewer_path in expected_reviewer_paths,
            "semantic audit adjudication references an unexpected review file",
        )
        provenance = _mapping(
            reviewer.get("provenance"), f"semantic audit reviewer {index} provenance"
        )
        reviewer_index.append(
            {
                "path": reviewer_path.relative_to(release).as_posix(),
                "sha256": reviewer["sha256"],
                "bytes": reviewer["bytes"],
                "review_count": reviewer["review_count"],
                "reviewer_id": provenance["reviewer_id"],
                "reviewer_type": provenance["reviewer_type"],
                "review_protocol_id": provenance["review_protocol_id"],
                "review_prompt_sha256": provenance.get("review_prompt_sha256"),
            }
        )

    packet_binding = _mapping(recomputed.get("packet_binding"), "packet binding")
    summary = dict(_mapping(recomputed.get("summary"), "semantic audit summary"))
    policy = dict(_mapping(recomputed.get("policy"), "semantic audit policy"))
    return {
        "status": "pass",
        "gate": True,
        "packet": {
            "path": SEMANTIC_AUDIT_PACKET,
            "sha256": sha256(packet_path),
            "bytes": packet_path.stat().st_size,
            "packet_id": packet_binding["packet_id"],
            "review_evidence_sha256": packet_binding["review_evidence_sha256"],
            "item_count": packet_binding["item_count"],
        },
        "packet_markdown": {
            "path": SEMANTIC_AUDIT_PACKET_MARKDOWN,
            "sha256": sha256(packet_markdown_path),
            "bytes": packet_markdown_path.stat().st_size,
        },
        "reviews": reviewer_index,
        "result": {
            "path": SEMANTIC_AUDIT_RESULT,
            "sha256": sha256(result_path),
            "bytes": result_path.stat().st_size,
            "result_id": recomputed["result_id"],
            "status": recomputed["status"],
            "semantic_visual_audit_gate": decision["semantic_visual_audit_gate"],
        },
        "result_markdown": {
            "path": SEMANTIC_AUDIT_RESULT_MARKDOWN,
            "sha256": sha256(result_markdown_path),
            "bytes": result_markdown_path.stat().st_size,
        },
        "release_binding": {
            "release_manifest_sha256": release_manifest_sha,
            "episode_ir_sha256": episode_integrity["sha256"],
            "source_inventory_sha256": source_integrity["sha256"],
        },
        "policy": policy,
        "summary": summary,
    }


def _verify_release_impl(release: Path) -> dict[str, Any]:
    manifest_path = release / "release_manifest.json"
    manifest = read_json(manifest_path)
    implementation = _verify_implementation_binding(manifest)
    base = _verify_base(release, manifest)
    source, referenced_pngs, media_root = _verify_source_inventory(release, manifest)
    exported_pngs = _collect_exported_pngs(release, manifest)
    _require(
        exported_pngs <= referenced_pngs,
        "serialized artifacts reference PNGs outside source_inventory",
    )
    schedules = _verify_schedules(release, manifest)
    statistics = _verify_statistics(release, manifest)
    audit = _verify_audit(release, manifest_path)
    evaluator = _verify_evaluator(release, manifest)
    semantic_visual_audit = _verify_semantic_visual_audit(
        release, manifest_path, manifest
    )
    cached_pngs = {path.resolve() for path in media_root.rglob("*.png") if path.is_file()}
    report = {
        "schema_version": "epispace.final_release_index.v1",
        "status": "pass",
        "generated_at": datetime.now(UTC).isoformat(),
        "release_dir": str(release),
        "release_manifest_sha256": sha256(manifest_path),
        "implementation_integrity": implementation,
        "base_artifacts": base,
        "source_integrity": source,
        "recomputed_statistics": statistics,
        "compute_matching": schedules,
        "corpus_audit": audit,
        "evaluator_smoke": evaluator,
        "semantic_visual_audit": semantic_visual_audit,
        "media_closure": {
            "referenced_pngs": len(referenced_pngs),
            "exported_pngs": len(exported_pngs),
            "cached_pngs": len(cached_pngs),
            "unreferenced_cache_pngs": len(cached_pngs - referenced_pngs),
            "missing_referenced_pngs": len(referenced_pngs - cached_pngs),
            "policy": "only source_inventory closure is part of the release",
        },
    }
    _require(not (referenced_pngs - cached_pngs), "source inventory references missing RGB files")
    return report


def verify_release(release_dir: Path) -> dict[str, Any]:
    release = release_dir.expanduser().resolve()
    _require(release.is_dir(), f"release directory is absent: {release}")
    try:
        return _verify_release_impl(release)
    except ReleaseVerificationError:
        raise
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise ReleaseVerificationError(f"malformed or unreadable release: {exc}") from exc


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    release = args.release_dir.expanduser().resolve()
    default_output = release / "final_release_index.json"
    output = args.output.expanduser().resolve() if args.output else default_output
    if output != default_output:
        try:
            output.relative_to(release)
        except ValueError:
            pass
        else:
            print(
                json.dumps(
                    {
                        "status": "fail",
                        "error": "custom output inside the release directory is forbidden",
                    },
                    ensure_ascii=False,
                )
            )
            return 1
    # Never leave an older pass index behind after a failed verification.
    output.unlink(missing_ok=True)
    try:
        report = verify_release(release)
    except ReleaseVerificationError as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, ensure_ascii=False))
        return 1
    _write_json_atomic(output, report)
    print(json.dumps({"status": "pass", "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
