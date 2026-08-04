#!/usr/bin/env python3
"""Build the fail-closed web projection of a verified EpiSpace release.

The web catalog is deliberately not another source of truth.  It is emitted
only when the final release index is a passing, hash-consistent authority, and
contains no host-absolute paths.  Large episode details are split into lazy
JSON documents so the landing page remains small.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = PROJECT_ROOT / "data" / "epispace_pilot_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "web" / "data" / "release_catalog.v1.json"
DEFAULT_DETAILS = PROJECT_ROOT / "web" / "data" / "release_details"
CATALOG_SCHEMA = "epispace.web_release_catalog.v1"
DETAIL_SCHEMA = "epispace.web_release_detail.v1"
FINAL_INDEX_SCHEMA = "epispace.final_release_index.v1"
RELEASE_SCHEMA = "epispace.release_manifest.v1"
SEMANTIC_PACKET_SCHEMA = "epispace.semantic_visual_audit_packet.v1"
SEMANTIC_RESULT_SCHEMA = "epispace.semantic_visual_audit_result.v1"
SEMANTIC_PACKET_PATH = "semantic_visual_audit/packet.json"
SEMANTIC_PACKET_MARKDOWN_PATH = "semantic_visual_audit/packet.md"
SEMANTIC_RESULT_PATH = "semantic_visual_audit/result.json"
SEMANTIC_RESULT_MARKDOWN_PATH = "semantic_visual_audit/result.md"
SEMANTIC_REVIEW_PREFIX = "semantic_visual_audit/reviews/"


class WebReleaseBuildError(ValueError):
    """Raised when the verified release cannot safely be projected to the web."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WebReleaseBuildError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WebReleaseBuildError(f"cannot read valid JSON: {path}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise WebReleaseBuildError(f"cannot read JSONL: {path}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WebReleaseBuildError(f"invalid JSONL at {path}:{line_number}") from exc
        _require(isinstance(value, dict), f"JSONL row is not an object: {path}:{line_number}")
        rows.append(value)
    return rows


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _release_path(release: Path, value: Any, label: str) -> Path:
    _require(isinstance(value, str) and value, f"{label} must be a relative path")
    raw = Path(value)
    _require(not raw.is_absolute(), f"{label} must be relative")
    path = (release / raw).resolve()
    try:
        path.relative_to(release.resolve())
    except ValueError as exc:
        raise WebReleaseBuildError(f"{label} escapes release directory") from exc
    return path


def _web_url(path: Path, *, web_root: Path) -> str:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"web-linked file is absent: {resolved}")
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise WebReleaseBuildError("web-linked file is outside the project") from exc
    return Path(os.path.relpath(resolved, start=web_root.resolve())).as_posix()


def _assert_portable(value: Any, *, location: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_portable(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_portable(child, location=f"{location}[{index}]")
    elif isinstance(value, str):
        _require(not Path(value).is_absolute(), f"absolute path leaked at {location}")
        _require("/data/shichao/" not in value, f"host path leaked at {location}")


def _verified_index_file(
    release: Path,
    raw_record: Any,
    *,
    label: str,
    expected_path: str | None = None,
) -> tuple[Path, Mapping[str, Any]]:
    record = _mapping(raw_record, label)
    if expected_path is not None:
        _require(record.get("path") == expected_path, f"{label} path is not canonical")
    path = _release_path(release, record.get("path"), f"{label}.path")
    _require(path.is_file(), f"{label} is absent")
    _require(_sha256(path) == record.get("sha256"), f"{label} SHA mismatch")
    _require(path.stat().st_size == record.get("bytes"), f"{label} byte count mismatch")
    return path, record


def _semantic_visual_audit_projection(
    release: Path,
    index: Mapping[str, Any],
    *,
    web_root: Path,
) -> dict[str, Any]:
    """Verify and project the formal RGB semantic-quality gate.

    This deliberately re-opens every indexed audit input instead of copying a
    badge from ``final_release_index.json``.  A stale packet, review, result, or
    path therefore invalidates the entire web catalog.
    """

    authority = _mapping(
        index.get("semantic_visual_audit"), "final_index.semantic_visual_audit"
    )
    _require(authority.get("status") == "pass", "semantic visual audit is not passing")
    _require(authority.get("gate") is True, "semantic visual audit gate is not true")

    packet_path, packet_record = _verified_index_file(
        release,
        authority.get("packet"),
        label="semantic visual audit packet",
        expected_path=SEMANTIC_PACKET_PATH,
    )
    _verified_index_file(
        release,
        authority.get("packet_markdown"),
        label="semantic visual audit packet markdown",
        expected_path=SEMANTIC_PACKET_MARKDOWN_PATH,
    )
    result_path, result_record = _verified_index_file(
        release,
        authority.get("result"),
        label="semantic visual audit result",
        expected_path=SEMANTIC_RESULT_PATH,
    )
    _verified_index_file(
        release,
        authority.get("result_markdown"),
        label="semantic visual audit result markdown",
        expected_path=SEMANTIC_RESULT_MARKDOWN_PATH,
    )
    packet = _read_json(packet_path)
    result = _read_json(result_path)
    _require(packet.get("schema_version") == SEMANTIC_PACKET_SCHEMA, "unsupported semantic audit packet")
    _require(result.get("schema_version") == SEMANTIC_RESULT_SCHEMA, "unsupported semantic audit result")
    _require(result.get("status") == "pass", "semantic audit result is not passing")
    decision = _mapping(result.get("decision"), "semantic audit result.decision")
    _require(
        decision.get("semantic_visual_audit_gate") is True,
        "semantic audit result gate is not true",
    )
    _require(decision.get("failed_item_count") == 0, "semantic audit contains failed items")
    _require(not decision.get("failure_reasons"), "semantic audit has failure reasons")
    _require(
        result_record.get("status") == "pass"
        and result_record.get("semantic_visual_audit_gate") is True,
        "indexed semantic result decision disagrees",
    )
    _require(result_record.get("result_id") == result.get("result_id"), "semantic result ID mismatch")

    packet_binding = _mapping(result.get("packet_binding"), "semantic result.packet_binding")
    packet_id = packet.get("packet_id")
    evidence_binding = _mapping(
        packet.get("review_evidence_binding"), "semantic packet.review_evidence_binding"
    )
    item_count = len(packet.get("items", [])) if isinstance(packet.get("items"), list) else -1
    _require(item_count > 0, "semantic audit packet has no items")
    expected_packet_binding = {
        "sha256": packet_record.get("sha256"),
        "bytes": packet_record.get("bytes"),
        "packet_id": packet_id,
        "review_evidence_sha256": evidence_binding.get("sha256"),
        "item_count": item_count,
    }
    for key, expected in expected_packet_binding.items():
        _require(packet_binding.get(key) == expected, f"semantic result packet {key} mismatch")
        _require(packet_record.get(key) == expected, f"indexed semantic packet {key} mismatch")

    raw_reviews = authority.get("reviews")
    _require(isinstance(raw_reviews, list) and raw_reviews, "semantic audit reviews are absent")
    indexed_reviews: dict[str, Mapping[str, Any]] = {}
    review_types: set[str] = set()
    review_count = 0
    review_prompt_hashes: list[str] = []
    for position, raw_review in enumerate(raw_reviews):
        path, record = _verified_index_file(
            release, raw_review, label=f"semantic review[{position}]"
        )
        relative = path.relative_to(release).as_posix()
        _require(
            relative.startswith(SEMANTIC_REVIEW_PREFIX)
            and relative.endswith(".json")
            and relative.count("/") == 2,
            f"semantic review[{position}] path is not canonical",
        )
        _require(relative not in indexed_reviews, "duplicate semantic review path")
        reviewer_type = record.get("reviewer_type")
        _require(
            reviewer_type in {"model_assisted_independent", "human_independent"},
            f"semantic review[{position}] reviewer_type is unsupported",
        )
        indexed_reviews[relative] = record
        review_types.add(str(reviewer_type))
        count = record.get("review_count")
        _require(isinstance(count, int) and count > 0, f"semantic review[{position}] count is invalid")
        review_count += count
        prompt_sha = record.get("review_prompt_sha256")
        if reviewer_type == "model_assisted_independent":
            _require(
                isinstance(prompt_sha, str) and re.fullmatch(r"[0-9a-f]{64}", prompt_sha),
                f"semantic review[{position}] prompt SHA is invalid",
            )
            review_prompt_hashes.append(prompt_sha)

    reviewer_bindings = result.get("reviewer_bindings")
    _require(
        isinstance(reviewer_bindings, list) and len(reviewer_bindings) == len(indexed_reviews),
        "semantic result reviewer bindings disagree",
    )
    unmatched_records = list(indexed_reviews.values())
    for position, raw_binding in enumerate(reviewer_bindings):
        binding = _mapping(raw_binding, f"semantic result.reviewer_bindings[{position}]")
        provenance = _mapping(binding.get("provenance"), f"semantic reviewer provenance[{position}]")
        candidates = [
            record
            for record in unmatched_records
            if record.get("sha256") == binding.get("sha256")
            and record.get("bytes") == binding.get("bytes")
            and record.get("review_count") == binding.get("review_count")
            and record.get("reviewer_id") == provenance.get("reviewer_id")
            and record.get("reviewer_type") == provenance.get("reviewer_type")
            and record.get("review_protocol_id") == provenance.get("review_protocol_id")
            and record.get("review_prompt_sha256") == provenance.get("review_prompt_sha256")
        ]
        _require(len(candidates) == 1, "semantic result/index reviewer hash binding mismatch")
        unmatched_records.remove(candidates[0])
    _require(not unmatched_records, "indexed semantic reviews were not adjudicated")

    summary = _mapping(result.get("summary"), "semantic audit result.summary")
    status_counts = _mapping(
        summary.get("overall_status_counts"), "semantic audit overall status counts"
    )
    projected_counts: dict[str, int] = {}
    for status in ("pass", "minor_issue", "major_issue", "unreviewable"):
        value = status_counts.get(status)
        _require(isinstance(value, int) and value >= 0, f"semantic audit count {status} is invalid")
        projected_counts[status] = value
    _require(sum(projected_counts.values()) == item_count, "semantic audit status counts do not close")
    _require(summary.get("reviewed_items") == item_count == review_count, "semantic audit review coverage is incomplete")
    _require(summary.get("reviewer_count") == len(indexed_reviews), "semantic audit reviewer count mismatch")
    _require(projected_counts["major_issue"] == 0, "semantic audit contains major issues")
    _require(projected_counts["unreviewable"] == 0, "semantic audit contains unreviewable items")
    _require(authority.get("summary") == result.get("summary"), "indexed semantic audit summary is stale")
    _require(authority.get("policy") == result.get("policy"), "indexed semantic audit policy is stale")

    release_binding = _mapping(
        authority.get("release_binding"), "semantic audit release binding"
    )
    packet_release_binding = _mapping(
        packet.get("release_binding"), "semantic packet.release_binding"
    )
    packet_manifest = _mapping(packet_release_binding.get("release_manifest"), "packet release manifest")
    packet_episode_ir = _mapping(packet_release_binding.get("episode_ir"), "packet episode IR")
    packet_inventory = _mapping(packet_release_binding.get("source_inventory"), "packet source inventory")
    _require(
        release_binding.get("release_manifest_sha256") == index.get("release_manifest_sha256")
        == packet_manifest.get("sha256"),
        "semantic audit manifest binding is stale",
    )
    episode_ir_sha = _mapping(
        _mapping(index.get("base_artifacts"), "final_index.base_artifacts").get("episode_ir"),
        "final_index episode_ir",
    ).get("sha256")
    _require(
        release_binding.get("episode_ir_sha256") == episode_ir_sha == packet_episode_ir.get("sha256"),
        "semantic audit episode IR binding is stale",
    )
    source_inventory_sha = _mapping(
        _mapping(index.get("base_artifacts"), "final_index.base_artifacts").get(
            "source_inventory"
        ),
        "final_index source_inventory",
    ).get("sha256")
    _require(
        release_binding.get("source_inventory_sha256")
        == source_inventory_sha
        == packet_inventory.get("sha256"),
        "semantic audit source inventory binding is stale",
    )

    sampling = _mapping(packet.get("sampling_contract"), "semantic packet.sampling_contract")
    seed = sampling.get("seed")
    _require(isinstance(seed, int) and not isinstance(seed, bool), "semantic audit seed is invalid")
    reviewer_type = (
        next(iter(review_types)) if len(review_types) == 1 else "mixed_independent"
    )
    reviewer_label_zh = (
        "模型辅助独立复核（非人工审计）"
        if reviewer_type == "model_assisted_independent"
        else "独立复核（含真人与模型辅助，逐项见 provenance）"
    )
    return {
        "status": "pass",
        "gate": True,
        "reviewer_type": reviewer_type,
        "reviewer_label_zh": reviewer_label_zh,
        "sample_size": item_count,
        "counts": projected_counts,
        "reviewer_count": len(indexed_reviews),
        "seed": seed,
        "packet_id": packet_id,
        "result_id": result.get("result_id"),
        "decision_policy_id": _mapping(result.get("policy"), "semantic result.policy").get(
            "decision_policy_id"
        ),
        "hash_binding": {
            "packet_sha256": packet_record.get("sha256"),
            "review_evidence_sha256": evidence_binding.get("sha256"),
            "review_sha256": sorted(record.get("sha256") for record in indexed_reviews.values()),
            "review_prompt_sha256": sorted(review_prompt_hashes),
            "result_sha256": result_record.get("sha256"),
        },
        "packet_url": _web_url(packet_path, web_root=web_root),
        "result_url": _web_url(result_path, web_root=web_root),
        "claim_boundary": (
            "model_assisted_independent is model-assisted independent review, "
            "not a human audit."
        ),
    }


def _verify_authority(release: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    release = release.expanduser().resolve()
    index_path = release / "final_release_index.json"
    manifest_path = release / "release_manifest.json"
    _require(index_path.is_file(), "final_release_index.json is absent")
    _require(manifest_path.is_file(), "release_manifest.json is absent")
    index = _read_json(index_path)
    manifest = _read_json(manifest_path)
    _require(index.get("schema_version") == FINAL_INDEX_SCHEMA, "unsupported final index")
    _require(index.get("status") == "pass", "final release index is not passing")
    _require(manifest.get("schema_version") == RELEASE_SCHEMA, "unsupported release manifest")
    _require(manifest.get("status") == "pass", "release manifest is not passing")
    manifest_sha = _sha256(manifest_path)
    _require(
        index.get("release_manifest_sha256") == manifest_sha,
        "final release index is bound to a different manifest",
    )

    artifacts = _mapping(manifest.get("artifacts"), "manifest.artifacts")
    integrity = _mapping(manifest.get("artifact_integrity"), "manifest.artifact_integrity")
    indexed = _mapping(index.get("base_artifacts"), "final_index.base_artifacts")
    _require(set(artifacts) == set(integrity) == set(indexed), "artifact declarations disagree")
    for name in sorted(artifacts):
        expected = _mapping(integrity[name], f"integrity.{name}")
        final_record = _mapping(indexed[name], f"base_artifacts.{name}")
        _require(expected.get("path") == artifacts[name], f"artifact path mismatch: {name}")
        _require(final_record.get("path") == artifacts[name], f"final path mismatch: {name}")
        path = _release_path(release, artifacts[name], f"artifact {name}")
        _require(path.is_file(), f"artifact is absent: {name}")
        actual_sha = _sha256(path)
        _require(actual_sha == expected.get("sha256"), f"manifest SHA mismatch: {name}")
        _require(actual_sha == final_record.get("sha256"), f"final-index SHA mismatch: {name}")
        _require(path.stat().st_size == expected.get("bytes"), f"byte count mismatch: {name}")
        records = expected.get("records")
        if records is not None:
            _require(len(_read_jsonl(path)) == records, f"record count mismatch: {name}")

    checks = _mapping(
        _mapping(manifest.get("corpus_gates"), "corpus_gates").get("checks"),
        "corpus_gates.checks",
    )
    _require(bool(checks) and all(value is True for value in checks.values()), "corpus gate failed")
    audit_path = release / "corpus_audit.json"
    evaluation_path = release / "evaluation_smoke" / "evaluation.json"
    compute_path = release / "compute_matching" / "compute_matching_manifest.json"
    _require(
        audit_path.is_file()
        and _sha256(audit_path)
        == _mapping(index.get("corpus_audit"), "corpus_audit").get("sha256"),
        "corpus audit is absent or stale",
    )
    _require(
        evaluation_path.is_file()
        and _sha256(evaluation_path)
        == _mapping(index.get("evaluator_smoke"), "evaluator_smoke").get("sha256"),
        "evaluator smoke report is absent or stale",
    )
    _require(
        compute_path.is_file()
        and _sha256(compute_path)
        == _mapping(index.get("compute_matching"), "compute_matching").get(
            "manifest_sha256"
        ),
        "compute-matching manifest is absent or stale",
    )
    return index, manifest, {
        "final_release_index_sha256": _sha256(index_path),
        "release_manifest_sha256": manifest_sha,
    }


def _program_projection(program: Mapping[str, Any]) -> dict[str, Any]:
    nodes = program.get("nodes")
    _require(isinstance(nodes, list), "typed program nodes must be a list")
    return {
        "program_id": program.get("program_id"),
        "semantic_signature": program.get("semantic_signature"),
        "atoms": list(program.get("atoms", [])),
        "nodes": [
            {
                "node_id": node.get("node_id"),
                "operation": node.get("operation"),
                "input_types": list(node.get("input_types", [])),
                "output_type": node.get("output_type"),
                **({"variant": node["variant"]} if node.get("variant") else {}),
            }
            for node in nodes
            if isinstance(node, Mapping)
        ],
        "answer_node": program.get("answer_node"),
    }


def _certificate_projection(certificate: Mapping[str, Any]) -> dict[str, Any]:
    checks = certificate.get("checks", [])
    _require(isinstance(checks, list), "certificate checks must be a list")
    return {
        "result": certificate.get("result"),
        "verifier_version": certificate.get("verifier_version"),
        "checks": [
            {"name": check.get("name"), "passed": check.get("passed")}
            for check in checks
            if isinstance(check, Mapping)
        ],
    }


def _training_export_index(
    release: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    artifacts = _mapping(manifest["artifacts"], "artifacts")
    by_bundle: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(_release_path(release, artifacts["episode_sft"], "episode SFT")):
        bundle = _mapping(row.get("hidden_meta"), "episode hidden_meta").get("source_bundle")
        contract = _mapping(row.get("comparison_contract"), "episode comparison contract")
        _require(isinstance(bundle, str), "episode SFT source bundle is absent")
        by_bundle[bundle].append(
            {
                "record_id": row.get("record_id"),
                "comparison_id": contract.get("comparison_id"),
                "fact_count": len(contract.get("fact_ids", [])),
                "image_count": contract.get("unique_image_count"),
                "question_count": contract.get("question_count"),
                "surface_format": contract.get("surface_format"),
            }
        )

    benchmark_by_bundle: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    core_ids = {
        row["record_id"]
        for row in _read_jsonl(_release_path(release, artifacts["benchmark_core"], "benchmark core"))
    }
    composition_ids = {
        row["record_id"]
        for row in _read_jsonl(
            _release_path(release, artifacts["benchmark_composition"], "composition benchmark")
        )
    }
    for row in _read_jsonl(_release_path(release, artifacts["benchmark"], "benchmark")):
        bundle = row.get("source_bundle")
        _require(isinstance(bundle, str), "benchmark source bundle is absent")
        target = _mapping(row.get("target"), "benchmark target")
        program = _mapping(row.get("program"), "benchmark program")
        benchmark_by_bundle[bundle].append(
            {
                "record_id": row.get("record_id"),
                "fact_id": target.get("fact_id"),
                "program_id": program.get("program_id"),
                "family_variant": row.get("family_variant"),
                "consistency_group": row.get("consistency_group"),
                "in_core": row.get("record_id") in core_ids,
                "in_composition": row.get("record_id") in composition_ids,
            }
        )
    return dict(by_bundle), dict(benchmark_by_bundle)


def _episode_detail(
    episode: Mapping[str, Any],
    *,
    dataset_id: str,
    manifest_sha: str,
    episode_ir_sha: str,
    web_root: Path,
    training_rows: Sequence[Mapping[str, Any]],
    benchmark_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    observations = episode.get("observations")
    questions = episode.get("questions")
    _require(isinstance(observations, list) and observations, "episode observations are absent")
    _require(isinstance(questions, list) and questions, "episode questions are absent")
    visible_observations = []
    for observation in observations:
        _require(isinstance(observation, Mapping), "observation must be an object")
        rgb = observation.get("rgb")
        _require(isinstance(rgb, str), "observation RGB is absent")
        visible_observations.append(
            {
                "view_id": observation.get("view_id"),
                "step": observation.get("step"),
                "role": observation.get("role"),
                "rgb_url": _web_url(Path(rgb), web_root=web_root),
            }
        )

    model_questions: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    proofs: list[dict[str, Any]] = []
    for order, question in enumerate(questions, 1):
        _require(isinstance(question, Mapping), "question must be an object")
        fact_id = question.get("fact_id")
        model_questions.append({"order": order, "question_zh": question.get("question_zh")})
        targets.append(
            {
                "order": order,
                "fact_id": fact_id,
                "answer_zh": question.get("answer_zh"),
                "answer_status": question.get("answer_status"),
                "task_type": question.get("task_type"),
                "family_variant": question.get("family_variant"),
            }
        )
        proofs.append(
            {
                "order": order,
                "fact_id": fact_id,
                "evidence_view_ids": list(question.get("evidence_view_ids", [])),
                "program": _program_projection(_mapping(question.get("program"), "program")),
                "certificate": _certificate_projection(
                    _mapping(question.get("certificate"), "certificate")
                ),
            }
        )

    belief = _mapping(episode.get("observable_belief"), "observable_belief")
    detail = {
        "schema_version": DETAIL_SCHEMA,
        "release_binding": {
            "dataset_id": dataset_id,
            "release_manifest_sha256": manifest_sha,
            "episode_ir_sha256": episode_ir_sha,
        },
        "identity": {
            "episode_id": episode.get("episode_id"),
            "scene_id": episode.get("scene_id"),
            "scene_family_id": episode.get("scene_family_id"),
            "trajectory_family_id": episode.get("trajectory_family_id"),
            "trajectory_class": episode.get("trajectory_class"),
            "split": episode.get("split"),
        },
        "model_visible": {
            "policy": (
                "Ordered RGB observations and natural-language questions only; "
                "the target and compiler proof below are never appended to model input."
            ),
            "observations": visible_observations,
            "questions": model_questions,
        },
        "supervision_target": {
            "policy": "Surface answers are assistant-only SFT/evaluation targets.",
            "answers": targets,
        },
        "auxiliary_state_target": {
            "policy": "Optional separate state-aux export; not part of the main episode input.",
            "frame_id": belief.get("frame_id"),
            "quantization_m": belief.get("quantization_m"),
            "entity_count": belief.get("entity_count"),
        },
        "compiler_proof": {
            "policy": "Hidden typed IR and replayable check summary; never model-visible.",
            "questions": proofs,
        },
        "release_membership": {
            "episode_sft": list(training_rows),
            "benchmark": list(benchmark_rows),
        },
    }
    _assert_portable(detail)
    return detail


def _artifact_projection(
    release: Path, manifest: Mapping[str, Any], *, web_root: Path
) -> list[dict[str, Any]]:
    artifacts = _mapping(manifest["artifacts"], "artifacts")
    integrity = _mapping(manifest["artifact_integrity"], "artifact_integrity")
    result = []
    for name in sorted(artifacts):
        item = _mapping(integrity[name], f"artifact {name}")
        path = _release_path(release, artifacts[name], f"artifact {name}")
        result.append(
            {
                "name": name,
                "file": artifacts[name],
                "url": _web_url(path, web_root=web_root),
                "sha256": item.get("sha256"),
                "bytes": item.get("bytes"),
                "records": item.get("records"),
            }
        )
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _invalidate(output: Path, details_dir: Path) -> None:
    output.unlink(missing_ok=True)
    if details_dir.exists():
        shutil.rmtree(details_dir)


def build_web_release_catalog(
    release_dir: Path = DEFAULT_RELEASE,
    output: Path = DEFAULT_OUTPUT,
    details_dir: Path = DEFAULT_DETAILS,
) -> dict[str, Any]:
    """Verify and atomically build a portable release catalog plus lazy details."""

    release = release_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    details_dir = details_dir.expanduser().resolve()
    _invalidate(output, details_dir)
    index, manifest, hashes = _verify_authority(release)
    web_root = output.parents[1]
    _require(web_root.name == "web", "output must be under a web/data directory")
    semantic_visual_audit = _semantic_visual_audit_projection(
        release, index, web_root=web_root
    )
    artifacts = _mapping(manifest["artifacts"], "artifacts")
    episode_ir_path = _release_path(release, artifacts["episode_ir"], "episode IR")
    episodes = _read_jsonl(episode_ir_path)
    dataset_id = manifest.get("dataset_id")
    _require(isinstance(dataset_id, str) and dataset_id, "dataset_id is absent")
    episode_ir_sha = _mapping(
        _mapping(manifest["artifact_integrity"], "artifact_integrity")["episode_ir"],
        "episode_ir integrity",
    )["sha256"]
    training_by_bundle, benchmark_by_bundle = _training_export_index(release, manifest)

    temporary_parent = details_dir.parent
    temporary_parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=".release_details.", dir=temporary_parent)
    )
    catalog_episodes: list[dict[str, Any]] = []
    try:
        for episode in sorted(episodes, key=lambda row: str(row.get("episode_id"))):
            episode_id = episode.get("episode_id")
            source_bundle = episode.get("source_bundle")
            _require(isinstance(episode_id, str) and episode_id, "episode_id is absent")
            _require(
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", episode_id) is not None,
                "episode_id is not safe for a detail filename",
            )
            _require(isinstance(source_bundle, str), "episode source bundle is absent")
            detail = _episode_detail(
                episode,
                dataset_id=dataset_id,
                manifest_sha=hashes["release_manifest_sha256"],
                episode_ir_sha=episode_ir_sha,
                web_root=web_root,
                training_rows=training_by_bundle.get(source_bundle, []),
                benchmark_rows=benchmark_by_bundle.get(source_bundle, []),
            )
            detail_name = f"{episode_id}.json"
            _write_json(temporary / detail_name, detail)
            questions = episode["questions"]
            observations = episode["observations"]
            programs = Counter(question["program"]["program_id"] for question in questions)
            task_types = Counter(question["task_type"] for question in questions)
            variants = Counter(question.get("family_variant", "canonical") for question in questions)
            catalog_episodes.append(
                {
                    "episode_id": episode_id,
                    "scene_id": episode.get("scene_id"),
                    "split": episode.get("split"),
                    "trajectory_class": episode.get("trajectory_class"),
                    "view_count": len(observations),
                    "question_count": len(questions),
                    "unknown_count": sum(
                        question.get("answer_status") == "unknown" for question in questions
                    ),
                    "programs": dict(sorted(programs.items())),
                    "task_types": dict(sorted(task_types.items())),
                    "family_variants": dict(sorted(variants.items())),
                    "thumbnail_url": detail["model_visible"]["observations"][0]["rgb_url"],
                    "detail_url": f"data/release_details/{detail_name}",
                    "episode_sft_record_count": len(training_by_bundle.get(source_bundle, [])),
                    "benchmark_record_count": len(benchmark_by_bundle.get(source_bundle, [])),
                }
            )

        trajectory_priority = {"T1": 0, "T8": 1, "T4": 2, "T7": 3, "T3": 4}
        split_priority = {"test": 0, "val": 1, "train": 2}
        catalog_episodes.sort(
            key=lambda item: (
                split_priority.get(str(item["split"]), 9),
                trajectory_priority.get(str(item["trajectory_class"]), 9),
                -int(item["question_count"]),
                str(item["episode_id"]),
            )
        )

        statistics = _mapping(manifest.get("statistics"), "statistics")
        acquisition = _mapping(statistics.get("acquisition"), "statistics.acquisition")
        compiled = _mapping(statistics.get("compiled"), "statistics.compiled")
        exports = _mapping(statistics.get("exports"), "statistics.exports")
        corpus_gates = _mapping(manifest.get("corpus_gates"), "corpus_gates")
        source_integrity = _mapping(index.get("source_integrity"), "source_integrity")
        write_sources = sum(
            item.get("strict_bundles_loaded", 0)
            for item in acquisition.get("sweeps", [])
            if isinstance(item, Mapping) and item.get("role") == "write_source"
        )
        verifier_sources = acquisition.get("strict_bundles_loaded", 0) - write_sources
        catalog = {
            "schema_version": CATALOG_SCHEMA,
            "status": "verified",
            "provenance": {
                "dataset_id": dataset_id,
                "verified_at": index.get("generated_at"),
                **hashes,
                "episode_ir_sha256": episode_ir_sha,
                "authority": (
                    "final_release_index.pass + manifest/artifact hash closure + "
                    "semantic_visual_audit.gate"
                ),
            },
            "research_contract": dict(manifest.get("research_contract", {})),
            "funnel": {
                "planned_jobs": acquisition.get("planned_jobs"),
                "strict_passed_bundles": acquisition.get("strict_bundles_loaded"),
                "write_source_bundles": write_sources,
                "oracle_verifier_bundles": verifier_sources,
                "compiled_episodes": compiled.get("source_trajectory_bundles"),
                "compiled_views": compiled.get("source_views"),
                "compiled_questions": compiled.get("questions"),
                "episode_supervised_facts": exports.get("episode_supervised_facts"),
            },
            "splits": {
                "scenes": dict(compiled.get("split_scenes", {})),
                "episodes": dict(compiled.get("split_bundles", {})),
            },
            "exports": dict(exports),
            "coverage": {
                "trajectory_counts": dict(compiled.get("trajectory_counts", {})),
                "task_types": dict(compiled.get("task_types", {})),
                "programs": dict(compiled.get("programs", {})),
            },
            "families": {
                "count": compiled.get("consistency_families"),
                "shapes": dict(compiled.get("family_shapes", {})),
                "variants": dict(compiled.get("family_variants", {})),
            },
            "composition_contract": {
                "train_atoms": list(corpus_gates.get("train_atoms", [])),
                "train_program_ids": list(corpus_gates.get("train_program_ids", [])),
                "heldout_atoms": list(corpus_gates.get("heldout_atoms", [])),
                "heldout_program_ids": list(corpus_gates.get("heldout_program_ids", [])),
                "heldout_benchmark_records": corpus_gates.get("heldout_benchmark_records"),
            },
            "quality": {
                "semantic_visual_audit": semantic_visual_audit,
            },
            "verification": {
                "corpus_gate_status": corpus_gates.get("status"),
                "corpus_gates": [
                    {"name": name, "passed": passed}
                    for name, passed in sorted(
                        _mapping(corpus_gates.get("checks"), "corpus gate checks").items()
                    )
                ],
                "certificate_replay": dict(corpus_gates.get("certificate_replay", {})),
                "source_integrity": {
                    key: source_integrity.get(key)
                    for key in ("sweep_count", "bundle_count", "view_count", "sensor_count")
                },
                "media_closure": dict(index.get("media_closure", {})),
                "corpus_audit": dict(index.get("corpus_audit", {})),
                "evaluator_smoke": {
                    "record_accuracy": _mapping(
                        index.get("evaluator_smoke"), "evaluator smoke"
                    ).get("record_accuracy"),
                    "replayed": _mapping(
                        index.get("evaluator_smoke"), "evaluator smoke"
                    ).get("replayed"),
                    "interpretation": "Oracle pipeline smoke test; not a model result.",
                },
                "semantic_visual_audit": {
                    "status": semantic_visual_audit["status"],
                    "gate": semantic_visual_audit["gate"],
                    "reviewer_type": semantic_visual_audit["reviewer_type"],
                    "sample_size": semantic_visual_audit["sample_size"],
                    "counts": dict(semantic_visual_audit["counts"]),
                    "seed": semantic_visual_audit["seed"],
                    "packet_id": semantic_visual_audit["packet_id"],
                    "result_id": semantic_visual_audit["result_id"],
                    "result_sha256": semantic_visual_audit["hash_binding"][
                        "result_sha256"
                    ],
                    "interpretation": semantic_visual_audit["claim_boundary"],
                },
            },
            "artifacts": _artifact_projection(release, manifest, web_root=web_root),
            "license_boundary": manifest.get("license_boundary"),
            "evidence_boundary": {
                "verified": [
                    "Serialized release integrity and source closure were replayed before catalog build.",
                    "RGB/natural-language model inputs are separated from targets and hidden compiler proof.",
                    "Scene/family split locks, typed programs, certificates, and paired facts passed gates.",
                    "A hash-bound independent semantic RGB review passed its fail-closed gate.",
                ],
                "not_claimed": [
                    "The oracle smoke test is not model performance.",
                    "This release page does not claim episode training improves a model.",
                    "Incremental state deltas and state sufficiency are not implemented release results.",
                    "Model-assisted independent review is not claimed as a human audit.",
                ],
            },
            "episodes": catalog_episodes,
        }
        _assert_portable(catalog)
        if details_dir.exists():
            shutil.rmtree(details_dir)
        temporary.replace(details_dir)
        temporary = None
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        os.close(descriptor)
        temporary_output = Path(temporary_name)
        try:
            _write_json(temporary_output, catalog)
            temporary_output.replace(output)
        finally:
            temporary_output.unlink(missing_ok=True)
        return catalog
    except Exception:
        _invalidate(output, details_dir)
        raise
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--details-dir", type=Path, default=DEFAULT_DETAILS)
    return result


def main() -> int:
    args = parser().parse_args()
    catalog = build_web_release_catalog(args.release_dir, args.output, args.details_dir)
    print(
        json.dumps(
            {
                "status": catalog["status"],
                "dataset_id": catalog["provenance"]["dataset_id"],
                "episodes": len(catalog["episodes"]),
                "questions": catalog["funnel"]["compiled_questions"],
                "details_dir": str(args.details_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
