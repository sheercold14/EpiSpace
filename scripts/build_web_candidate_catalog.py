#!/usr/bin/env python3
"""Project compiled Episode IR into a clearly non-release web candidate.

This builder is intentionally independent from ``build_web_release_catalog``.
It does not inspect or weaken the formal release authority.  Its only claim is
that every projected row is readable, media-closed, and carries a passing
compiler certificate.  Independent semantic RGB review remains pending.
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
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "epispace_pilot_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "web" / "data" / "candidate_catalog.v1.json"
DEFAULT_DETAILS = PROJECT_ROOT / "web" / "data" / "candidate_details"

CATALOG_SCHEMA = "epispace.web_candidate_catalog.v1"
DETAIL_SCHEMA = "epispace.web_candidate_detail.v1"
EPISODE_SCHEMA = "epispace.episode_ir.v1"
CONFIG_SCHEMA = "epispace.pipeline_config.v1"
STATUS = "machine_verified_candidate"
SEMANTIC_PACKET_SCHEMA = "epispace.semantic_visual_audit_packet.v1"
SEMANTIC_RESULT_SCHEMA = "epispace.semantic_visual_audit_result.v1"
SEMANTIC_REVIEW_SCHEMA = "epispace.semantic_visual_audit_review.v1"


class WebCandidateBuildError(ValueError):
    """Raised when a compiled candidate cannot be projected safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WebCandidateBuildError(message)


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
        raise WebCandidateBuildError(f"cannot read valid JSON: {path}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise WebCandidateBuildError(f"cannot read JSONL: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WebCandidateBuildError(
                f"invalid JSONL at {path}:{line_number}"
            ) from exc
        _require(
            isinstance(value, dict),
            f"JSONL row is not an object: {path}:{line_number}",
        )
        rows.append(value)
    return rows


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


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


def _web_url(path: Path, *, web_root: Path, project_root: Path) -> str:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"web-linked file is absent: {resolved}")
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise WebCandidateBuildError("web-linked file is outside the project") from exc
    return Path(os.path.relpath(resolved, start=web_root.resolve())).as_posix()


def _bound_audit_file(
    raw_path: Any,
    *,
    canonical_parent: Path,
    label: str,
) -> Path:
    _require(isinstance(raw_path, str) and raw_path, f"{label} path is absent")
    path = Path(raw_path)
    if not path.is_absolute():
        path = canonical_parent / path
    path = path.expanduser().resolve()
    try:
        path.relative_to(canonical_parent.resolve())
    except ValueError as exc:
        raise WebCandidateBuildError(f"{label} escapes its audit directory") from exc
    _require(path.is_file(), f"{label} is absent")
    return path


def _pending_semantic_audit() -> dict[str, Any]:
    return {
        "status": "pending",
        "gate": False,
        "disposition": "audit_required",
        "reviewer_label_zh": "独立 RGB 语义复核待执行",
        "claim_boundary": (
            "No semantic visual audit result is projected or implied by this catalog."
        ),
        "counts": None,
        "sample_size": None,
        "reviewer_count": None,
        "seed": None,
        "packet_id": None,
        "result_id": None,
        "packet_url": None,
        "result_url": None,
        "hash_binding": {
            "packet_sha256": None,
            "result_sha256": None,
            "review_sha256": [],
            "review_evidence_sha256": None,
        },
        "failure_reasons": [],
        "failed_item_count": None,
        "artifacts": [],
        "_fact_reviews": [],
    }


def _semantic_audit_projection(
    release: Path,
    *,
    dataset_id: str,
    episode_ir_sha: str,
    episode_count: int,
    web_root: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Project a completed audit only after replaying its full hash binding.

    A missing result is a normal pending candidate.  Once a result exists,
    malformed or stale evidence is an error rather than a reason to fall back
    to the less informative pending badge.
    """

    audit_dir = release / "semantic_visual_audit"
    result_path = audit_dir / "result.json"
    if not result_path.is_file():
        return _pending_semantic_audit()

    packet_path = audit_dir / "packet.json"
    _require(packet_path.is_file(), "semantic audit result exists without packet.json")
    packet = _read_json(packet_path)
    result = _read_json(result_path)
    _require(
        packet.get("schema_version") == SEMANTIC_PACKET_SCHEMA,
        "unsupported semantic audit packet",
    )
    _require(
        result.get("schema_version") == SEMANTIC_RESULT_SCHEMA,
        "unsupported semantic audit result",
    )
    _require(result.get("status") in {"pass", "fail"}, "invalid semantic audit status")

    items = packet.get("items")
    _require(isinstance(items, list) and items, "semantic audit packet has no items")
    item_count = len(items)
    packet_sha = _sha256(packet_path)
    packet_binding = _mapping(result.get("packet_binding"), "result.packet_binding")
    bound_packet = _bound_audit_file(
        packet_binding.get("path"), canonical_parent=audit_dir, label="bound packet"
    )
    _require(bound_packet == packet_path.resolve(), "semantic result binds a noncanonical packet")
    _require(packet_binding.get("sha256") == packet_sha, "semantic packet SHA mismatch")
    _require(
        packet_binding.get("bytes") == packet_path.stat().st_size,
        "semantic packet byte count mismatch",
    )
    _require(packet_binding.get("packet_id") == packet.get("packet_id"), "packet ID mismatch")
    _require(packet_binding.get("item_count") == item_count, "packet item count mismatch")
    evidence = _mapping(packet.get("review_evidence_binding"), "review_evidence_binding")
    _require(
        packet_binding.get("review_evidence_sha256") == evidence.get("sha256"),
        "review evidence binding mismatch",
    )

    release_binding = _mapping(packet.get("release_binding"), "packet.release_binding")
    _require(release_binding.get("dataset_id") == dataset_id, "audit dataset_id is stale")
    bound_ir = _mapping(release_binding.get("episode_ir"), "packet episode_ir")
    _require(bound_ir.get("sha256") == episode_ir_sha, "audit episode IR SHA is stale")
    _require(bound_ir.get("records") == episode_count, "audit episode count is stale")
    for key, filename in (
        ("release_manifest", "release_manifest.json"),
        ("source_inventory", "source_inventory.json"),
    ):
        binding = _mapping(release_binding.get(key), f"packet {key}")
        current = release / filename
        _require(current.is_file(), f"audit-bound {filename} is absent")
        _require(binding.get("sha256") == _sha256(current), f"audit-bound {filename} is stale")

    packet_fact_ids = [item.get("fact_id") for item in items if isinstance(item, Mapping)]
    packet_episode_ids = [
        item.get("episode_id") for item in items if isinstance(item, Mapping)
    ]
    _require(len(packet_fact_ids) == item_count, "packet item is not an object")
    _require(
        len(packet_episode_ids) == item_count
        and all(isinstance(value, str) and value for value in packet_episode_ids),
        "packet episode_id is absent",
    )
    _require(len(set(packet_fact_ids)) == item_count, "packet fact_id is not unique")

    reviewer_bindings = result.get("reviewer_bindings")
    _require(
        isinstance(reviewer_bindings, list) and reviewer_bindings,
        "semantic audit reviewer bindings are absent",
    )
    reviews_dir = audit_dir / "reviews"
    covered: dict[int, dict[str, Any]] = {}
    review_artifacts: list[dict[str, Any]] = []
    review_hashes: list[str] = []
    reviewer_types: set[str] = set()
    for position, raw_binding in enumerate(reviewer_bindings):
        binding = _mapping(raw_binding, f"reviewer_bindings[{position}]")
        path = _bound_audit_file(
            binding.get("path"), canonical_parent=reviews_dir, label=f"review[{position}]"
        )
        review_sha = _sha256(path)
        _require(binding.get("sha256") == review_sha, f"review[{position}] SHA mismatch")
        _require(
            binding.get("bytes") == path.stat().st_size,
            f"review[{position}] byte count mismatch",
        )
        payload = _read_json(path)
        _require(
            payload.get("schema_version") == SEMANTIC_REVIEW_SCHEMA,
            f"review[{position}] schema is unsupported",
        )
        _require(payload.get("packet_id") == packet.get("packet_id"), "review packet ID mismatch")
        provenance = _mapping(payload.get("reviewer_provenance"), "review provenance")
        bound_provenance = _mapping(binding.get("provenance"), "bound review provenance")
        for key in (
            "reviewer_type",
            "reviewer_id",
            "reviewer_system",
            "reviewer_model",
            "review_protocol_id",
            "review_prompt_sha256",
            "reviewed_at",
        ):
            _require(
                provenance.get(key) == bound_provenance.get(key),
                f"review[{position}] provenance {key} mismatch",
            )
        prompt = provenance.get("review_prompt")
        prompt_sha = provenance.get("review_prompt_sha256")
        _require(isinstance(prompt, str) and prompt, "review prompt is absent")
        _require(
            isinstance(prompt_sha, str)
            and hashlib.sha256(prompt.encode("utf-8")).hexdigest() == prompt_sha,
            "review prompt SHA mismatch",
        )
        reviews = payload.get("reviews")
        _require(isinstance(reviews, list) and reviews, "review decisions are absent")
        _require(binding.get("review_count") == len(reviews), "review count mismatch")
        indices: list[int] = []
        for raw_review in reviews:
            review = _mapping(raw_review, "review decision")
            index = review.get("index")
            _require(
                isinstance(index, int) and not isinstance(index, bool),
                "review index is invalid",
            )
            _require(index not in covered, "semantic audit item reviewed more than once")
            _require(0 <= index < item_count, "semantic audit review index is out of range")
            _require(review.get("fact_id") == packet_fact_ids[index], "review fact_id mismatch")
            covered[index] = dict(review)
            indices.append(index)
        _require(binding.get("covered_index_min") == min(indices), "review range minimum mismatch")
        _require(binding.get("covered_index_max") == max(indices), "review range maximum mismatch")
        reviewer_types.add(str(provenance.get("reviewer_type")))
        review_hashes.append(review_sha)
        review_artifacts.append(
            {
                "name": f"semantic_review_{position + 1}",
                "url": _web_url(path, web_root=web_root, project_root=project_root),
                "sha256": review_sha,
                "bytes": path.stat().st_size,
                "records": len(reviews),
            }
        )
    _require(set(covered) == set(range(item_count)), "semantic audit review coverage is incomplete")

    decisions = result.get("item_decisions")
    _require(isinstance(decisions, list) and len(decisions) == item_count, "result decisions incomplete")
    status_counts: Counter[str] = Counter()
    failed_count = 0
    fact_reviews: list[dict[str, Any]] = []
    for raw_decision in decisions:
        decision = _mapping(raw_decision, "result item decision")
        index = decision.get("index")
        _require(isinstance(index, int) and index in covered, "result item index is invalid")
        review = covered[index]
        for key in (
            "fact_id",
            "overall_status",
            "referents_recognizable",
            "answer_supported",
            "family_intervention_valid",
        ):
            _require(decision.get(key) == review.get(key), f"result decision {key} mismatch")
        status = str(decision.get("overall_status"))
        _require(
            status in {"pass", "minor_issue", "major_issue", "unreviewable"},
            "result item status is invalid",
        )
        status_counts[status] += 1
        false_field = any(
            decision.get(key) is False
            for key in (
                "referents_recognizable",
                "answer_supported",
                "family_intervention_valid",
            )
        )
        fact_reviews.append(
            {
                "index": index,
                "episode_id": packet_episode_ids[index],
                "fact_id": decision.get("fact_id"),
                "overall_status": status,
                "fact_passed": status == "pass" and not false_field,
            }
        )
        if status in {"major_issue", "unreviewable"} or false_field:
            failed_count += 1

    summary = _mapping(result.get("summary"), "semantic audit summary")
    projected_counts = {
        status: status_counts.get(status, 0)
        for status in ("pass", "minor_issue", "major_issue", "unreviewable")
    }
    _require(summary.get("overall_status_counts") == projected_counts, "audit counts mismatch")
    _require(summary.get("reviewed_items") == item_count, "audit reviewed count mismatch")
    _require(
        summary.get("reviewer_count") == len(reviewer_bindings),
        "audit reviewer count mismatch",
    )
    final_decision = _mapping(result.get("decision"), "semantic audit decision")
    gate = final_decision.get("semantic_visual_audit_gate")
    _require(isinstance(gate, bool), "semantic audit gate is invalid")
    _require(final_decision.get("failed_item_count") == failed_count, "failed item count mismatch")
    _require(gate is (failed_count == 0), "semantic audit gate contradicts decisions")
    _require(
        result.get("status") == ("pass" if gate else "fail"),
        "semantic audit status contradicts gate",
    )

    sampling = _mapping(packet.get("sampling_contract"), "packet sampling_contract")
    seed = sampling.get("seed")
    _require(isinstance(seed, int) and not isinstance(seed, bool), "audit seed is invalid")
    reviewer_label = (
        "模型辅助独立复核（非人工审计）"
        if reviewer_types == {"model_assisted_independent"}
        else "独立复核（逐项 provenance 见 review 文件）"
    )
    result_sha = _sha256(result_path)
    semantic_status = "passed" if gate else "failed"
    disposition = "awaiting_formal_release" if gate else "rework_required"
    artifacts = [
        {
            "name": "semantic_audit_packet",
            "url": _web_url(packet_path, web_root=web_root, project_root=project_root),
            "sha256": packet_sha,
            "bytes": packet_path.stat().st_size,
            "records": item_count,
        },
        *review_artifacts,
        {
            "name": "semantic_audit_result",
            "url": _web_url(result_path, web_root=web_root, project_root=project_root),
            "sha256": result_sha,
            "bytes": result_path.stat().st_size,
            "records": item_count,
        },
    ]
    return {
        "status": semantic_status,
        "source_result_status": result.get("status"),
        "gate": gate,
        "disposition": disposition,
        "reviewer_label_zh": reviewer_label,
        "claim_boundary": (
            "Hash-bound model-assisted independent review, not a human audit; "
            "a passing result alone is not formal release authority."
        ),
        "counts": projected_counts,
        "sample_size": item_count,
        "reviewer_count": len(reviewer_bindings),
        "seed": seed,
        "packet_id": packet.get("packet_id"),
        "result_id": result.get("result_id"),
        "packet_url": _web_url(packet_path, web_root=web_root, project_root=project_root),
        "result_url": _web_url(result_path, web_root=web_root, project_root=project_root),
        "hash_binding": {
            "packet_sha256": packet_sha,
            "result_sha256": result_sha,
            "review_sha256": sorted(review_hashes),
            "review_evidence_sha256": evidence.get("sha256"),
        },
        "failure_reasons": list(final_decision.get("failure_reasons", [])),
        "failed_item_count": failed_count,
        "minor_item_count": final_decision.get("minor_item_count"),
        "artifacts": artifacts,
        "_fact_reviews": fact_reviews,
    }


def _program_projection(program: Mapping[str, Any]) -> dict[str, Any]:
    nodes = program.get("nodes")
    _require(isinstance(nodes, list) and nodes, "typed program nodes are absent")
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
    checks = certificate.get("checks")
    _require(isinstance(checks, list) and checks, "certificate checks are absent")
    _require(
        certificate.get("result") in {"pass", "unknown"},
        "compiler certificate result is not passing",
    )
    _require(
        all(isinstance(check, Mapping) and check.get("passed") is True for check in checks),
        "compiler certificate contains a failed check",
    )
    return {
        "result": certificate.get("result"),
        "verifier_version": certificate.get("verifier_version"),
        "checks": [
            {"name": check.get("name"), "passed": True}
            for check in checks
            if isinstance(check, Mapping)
        ],
    }


def _episode_detail(
    episode: Mapping[str, Any],
    *,
    dataset_id: str,
    config_sha: str,
    episode_ir_sha: str,
    release: Path,
    web_root: Path,
    project_root: Path,
    semantic_audit: Mapping[str, Any],
) -> dict[str, Any]:
    observations = episode.get("observations")
    questions = episode.get("questions")
    _require(isinstance(observations, list) and observations, "episode observations are absent")
    _require(isinstance(questions, list) and questions, "episode questions are absent")

    visible_observations: list[dict[str, Any]] = []
    view_ids: set[str] = set()
    for observation in observations:
        item = _mapping(observation, "observation")
        view_id = item.get("view_id")
        rgb = item.get("rgb")
        _require(isinstance(view_id, str) and view_id, "observation view_id is absent")
        _require(view_id not in view_ids, "duplicate observation view_id")
        _require(isinstance(rgb, str) and rgb, "observation RGB is absent")
        view_ids.add(view_id)
        rgb_path = Path(rgb) if Path(rgb).is_absolute() else release / rgb
        projected = {
            "view_id": view_id,
            "step": item.get("step"),
            "role": item.get("role"),
            "rgb_url": _web_url(
                rgb_path, web_root=web_root, project_root=project_root
            ),
        }
        if item.get("camera_height_m") is not None:
            projected["camera_height_m"] = item.get("camera_height_m")
        if item.get("horizontal_fov_deg") is not None:
            projected["horizontal_fov_deg"] = item.get("horizontal_fov_deg")
        visible_observations.append(projected)

    model_questions: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    proofs: list[dict[str, Any]] = []
    for order, raw_question in enumerate(questions, 1):
        question = _mapping(raw_question, "question")
        question_zh = question.get("question_zh")
        answer_zh = question.get("answer_zh")
        _require(isinstance(question_zh, str) and question_zh, "question text is absent")
        _require(isinstance(answer_zh, str) and answer_zh, "answer text is absent")
        for key in ("model_view_ids", "evidence_view_ids"):
            references = question.get(key, [])
            _require(isinstance(references, list), f"{key} must be a list")
            _require(set(references) <= view_ids, f"{key} references an absent view")
        fact_id = question.get("fact_id")
        model_questions.append({"order": order, "question_zh": question_zh})
        targets.append(
            {
                "order": order,
                "fact_id": fact_id,
                "answer_zh": answer_zh,
                "answer_status": question.get("answer_status"),
                "task_type": question.get("task_type"),
                "family_variant": question.get("family_variant", "canonical"),
            }
        )
        proofs.append(
            {
                "order": order,
                "fact_id": fact_id,
                "evidence_view_ids": list(question.get("evidence_view_ids", [])),
                "program": _program_projection(
                    _mapping(question.get("program"), "typed program")
                ),
                "certificate": _certificate_projection(
                    _mapping(question.get("certificate"), "certificate")
                ),
            }
        )

    belief = _mapping(episode.get("observable_belief"), "observable_belief")
    detail = {
        "schema_version": DETAIL_SCHEMA,
        "status": STATUS,
        "authority": {
            "level": STATUS,
            "semantic_visual_audit": semantic_audit["status"],
            "disposition": semantic_audit["disposition"],
            "formal_release": False,
            "claim_boundary": (
                "Candidate authority only; semantic audit state is hash-bound when present, "
                "and formal release authority is never implied."
            ),
        },
        "candidate_binding": {
            "dataset_id": dataset_id,
            "pipeline_config_sha256": config_sha,
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
                "Ordered RGB observations, declared camera calibration, and natural-language "
                "questions only; targets and compiler proof are never appended to model input."
            ),
            "observations": visible_observations,
            "questions": model_questions,
        },
        "supervision_target": {
            "policy": "Surface answers are assistant-only SFT/evaluation targets.",
            "answers": targets,
        },
        "auxiliary_state_target": {
            "policy": "Optional state target; not part of the main episode input.",
            "frame_id": belief.get("frame_id"),
            "quantization_m": belief.get("quantization_m"),
            "entity_count": belief.get("entity_count"),
        },
        "compiler_proof": {
            "policy": "Hidden typed IR and replayed checks; never model-visible.",
            "questions": proofs,
        },
    }
    _assert_portable(detail)
    return detail


def _trajectory_sort_key(value: str) -> tuple[int, str]:
    match = re.fullmatch(r"T(\d+)", value)
    return (int(match.group(1)), value) if match else (10_000, value)


def _featured_rank_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -int(bool(item.get("fully_sample_reviewed_pass"))),
        -int(item["cross_view_question_count"] > 0),
        -len(item["programs"]),
        -len(item["task_types"]),
        -int(item["view_count"]),
        -int(item["question_count"]),
        str(item["episode_id"]),
    )


def _select_featured(
    episodes: Sequence[Mapping[str, Any]], *, limit: int = 12
) -> list[str]:
    """Select a deterministic, class-diverse, cross-view-heavy front page."""

    if not episodes or limit <= 0:
        return []
    by_class: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_class[str(episode["trajectory_class"])].append(episode)
    for values in by_class.values():
        values.sort(key=_featured_rank_key)

    selected: list[Mapping[str, Any]] = []
    selected_ids: set[str] = set()
    trajectory_classes = sorted(by_class, key=_trajectory_sort_key)
    target = min(limit, len(episodes))

    def add(item: Mapping[str, Any]) -> None:
        episode_id = str(item["episode_id"])
        if episode_id not in selected_ids and len(selected) < target:
            selected.append(item)
            selected_ids.add(episode_id)

    for trajectory_class in trajectory_classes:
        add(by_class[trajectory_class][0])

    selected_splits = {str(item["split"]) for item in selected}
    ranked = sorted(episodes, key=_featured_rank_key)
    for split in ("test", "val", "train"):
        if split in selected_splits:
            continue
        class_counts = Counter(str(value["trajectory_class"]) for value in selected)
        split_candidates = [value for value in ranked if str(value["split"]) == split]
        item = min(
            split_candidates,
            key=lambda value: (
                class_counts[str(value["trajectory_class"])],
                _featured_rank_key(value),
            ),
            default=None,
        )
        if item is not None:
            add(item)
            selected_splits.add(split)

    while len(selected) < target:
        class_counts = Counter(str(value["trajectory_class"]) for value in selected)
        next_items = [
            (
                class_counts[trajectory_class],
                _trajectory_sort_key(trajectory_class),
                next(
                    (
                        value
                        for value in by_class[trajectory_class]
                        if str(value["episode_id"]) not in selected_ids
                    ),
                    None,
                ),
            )
            for trajectory_class in trajectory_classes
        ]
        available = [item for item in next_items if item[2] is not None]
        if not available:
            break
        _, _, item = min(available, key=lambda value: (value[0], value[1]))
        assert item is not None
        add(item)
    selected.sort(
        key=lambda item: (
            not bool(item.get("fully_sample_reviewed_pass")),
            -int(item.get("sample_reviewed_fact_count", 0))
            if item.get("fully_sample_reviewed_pass")
            else 0,
        )
    )
    return [str(item["episode_id"]) for item in selected]


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


def build_web_candidate_catalog(
    release_dir: Path = DEFAULT_RELEASE,
    config_path: Path = DEFAULT_CONFIG,
    output: Path = DEFAULT_OUTPUT,
    details_dir: Path = DEFAULT_DETAILS,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Build an atomic, exhaustive web projection of compiled candidate IR."""

    release = release_dir.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    output = output.expanduser().resolve()
    details_dir = details_dir.expanduser().resolve()
    project_root = project_root.expanduser().resolve()
    _invalidate(output, details_dir)
    config = _read_json(config_path)
    _require(config.get("schema_version") == CONFIG_SCHEMA, "unsupported pipeline config")
    dataset_id = config.get("dataset_id")
    _require(isinstance(dataset_id, str) and dataset_id, "dataset_id is absent")
    episode_ir_path = release / "episodes.ir.jsonl"
    _require(episode_ir_path.is_file(), "episodes.ir.jsonl is absent")
    episodes = _read_jsonl(episode_ir_path)
    _require(episodes, "episodes.ir.jsonl is empty")
    config_sha = _sha256(config_path)
    episode_ir_sha = _sha256(episode_ir_path)

    web_root = output.parents[1]
    _require(web_root.name == "web", "output must be under a web/data directory")
    _require(details_dir.parent == output.parent, "details and catalog must share web/data")
    semantic_audit = _semantic_audit_projection(
        release,
        dataset_id=dataset_id,
        episode_ir_sha=episode_ir_sha,
        episode_count=len(episodes),
        web_root=web_root,
        project_root=project_root,
    )
    fact_reviews = semantic_audit.pop("_fact_reviews")
    reviewed_by_episode: defaultdict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for review in fact_reviews:
        reviewed_by_episode[str(review["episode_id"])][str(review["fact_id"])] = review
    episode_review_summary: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        episode_id = str(episode.get("episode_id"))
        fact_ids = {
            str(question.get("fact_id"))
            for question in episode.get("questions", [])
            if isinstance(question, Mapping)
        }
        reviewed = reviewed_by_episode.get(episode_id, {})
        reviewed_fact_ids = fact_ids & set(reviewed)
        fully_pass = bool(fact_ids) and reviewed_fact_ids == fact_ids and all(
            reviewed[fact_id]["fact_passed"] is True for fact_id in fact_ids
        )
        episode_review_summary[episode_id] = {
            "fully_sample_reviewed_pass": fully_pass,
            "sample_reviewed_fact_count": len(reviewed_fact_ids),
            "sample_reviewed_fact_total": len(fact_ids),
            "sample_review_status_counts": dict(
                sorted(
                    Counter(
                        reviewed[fact_id]["overall_status"]
                        for fact_id in reviewed_fact_ids
                    ).items()
                )
            ),
        }
    fully_reviewed_ids = sorted(
        episode_id
        for episode_id, summary in episode_review_summary.items()
        if summary["fully_sample_reviewed_pass"]
    )
    semantic_audit["fully_sample_reviewed_pass_episode_ids"] = fully_reviewed_ids
    semantic_audit["fully_sample_reviewed_pass_episode_count"] = len(fully_reviewed_ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".candidate_details.", dir=details_dir.parent))
    catalog_episodes: list[dict[str, Any]] = []
    try:
        seen_episode_ids: set[str] = set()
        family_variants: defaultdict[str, set[str]] = defaultdict(set)
        for episode in sorted(episodes, key=lambda row: str(row.get("episode_id"))):
            _require(episode.get("schema_version") == EPISODE_SCHEMA, "unsupported episode IR")
            episode_id = episode.get("episode_id")
            _require(isinstance(episode_id, str) and episode_id, "episode_id is absent")
            _require(
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", episode_id) is not None,
                "episode_id is not safe for a detail filename",
            )
            _require(episode_id not in seen_episode_ids, "duplicate episode_id")
            seen_episode_ids.add(episode_id)
            detail = _episode_detail(
                episode,
                dataset_id=dataset_id,
                config_sha=config_sha,
                episode_ir_sha=episode_ir_sha,
                release=release,
                web_root=web_root,
                project_root=project_root,
                semantic_audit=semantic_audit,
            )
            detail_name = f"{episode_id}.json"
            _write_json(temporary / detail_name, detail)
            questions = episode["questions"]
            observations = episode["observations"]
            programs = Counter(question["program"]["program_id"] for question in questions)
            task_types = Counter(str(question["task_type"]) for question in questions)
            variants = Counter(
                str(question.get("family_variant", "canonical")) for question in questions
            )
            family_id = str(episode.get("trajectory_family_id") or episode_id)
            family_variants[family_id].update(
                variant for variant in variants if variant != "canonical"
            )
            cross_view_count = sum(
                len(set(question.get("evidence_view_ids", []))) > 1
                for question in questions
            )
            review_summary = episode_review_summary[episode_id]
            catalog_episodes.append(
                {
                    "episode_id": episode_id,
                    "scene_id": episode.get("scene_id"),
                    "split": episode.get("split"),
                    "trajectory_class": episode.get("trajectory_class"),
                    "view_count": len(observations),
                    "question_count": len(questions),
                    "cross_view_question_count": cross_view_count,
                    **review_summary,
                    "unknown_count": sum(
                        question.get("answer_status") == "unknown" for question in questions
                    ),
                    "programs": dict(sorted(programs.items())),
                    "task_types": dict(sorted(task_types.items())),
                    "family_variants": dict(sorted(variants.items())),
                    "thumbnail_url": detail["model_visible"]["observations"][0]["rgb_url"],
                    "detail_url": f"data/{details_dir.name}/{detail_name}",
                }
            )

        featured_ids = _select_featured(catalog_episodes)
        featured_ranks = {episode_id: rank for rank, episode_id in enumerate(featured_ids, 1)}
        for item in catalog_episodes:
            item["featured"] = item["episode_id"] in featured_ranks
            item["featured_rank"] = featured_ranks.get(item["episode_id"])
        catalog_episodes.sort(
            key=lambda item: (
                item["featured_rank"] is None,
                item["featured_rank"] or 10_000,
                _trajectory_sort_key(str(item["trajectory_class"])),
                str(item["episode_id"]),
            )
        )

        split_scene_ids: defaultdict[str, set[str]] = defaultdict(set)
        split_episode_counts: Counter[str] = Counter()
        trajectory_counts: Counter[str] = Counter()
        all_programs: Counter[str] = Counter()
        all_task_types: Counter[str] = Counter()
        all_atoms: set[str] = set()
        for episode in episodes:
            split = str(episode.get("split"))
            split_scene_ids[split].add(str(episode.get("scene_id")))
            split_episode_counts[split] += 1
            trajectory_counts[str(episode.get("trajectory_class"))] += 1
            for question in episode["questions"]:
                program = _mapping(question.get("program"), "typed program")
                all_programs[str(program.get("program_id"))] += 1
                all_task_types[str(question.get("task_type"))] += 1
                all_atoms.update(str(atom) for atom in program.get("atoms", []))
        family_shapes = Counter(
            "+".join(sorted(variants))
            for variants in family_variants.values()
            if variants
        )
        holdout_programs = list(
            _mapping(config.get("composition_holdout", {}), "composition_holdout").get(
                "program_ids", []
            )
        )
        catalog = {
            "schema_version": CATALOG_SCHEMA,
            "status": STATUS,
            "authority": {
                "level": STATUS,
                "semantic_visual_audit": semantic_audit["status"],
                "disposition": semantic_audit["disposition"],
                "formal_release": False,
                "claim_boundary_zh": (
                    "仅表示自动编译、证书重放与媒体闭合通过；语义 RGB 审查结果只决定候选是否返工，"
                    "绝不构成正式 release authority。"
                    if semantic_audit["status"] != "pending"
                    else "仅表示自动编译、证书重放与媒体闭合通过；独立 RGB 语义复核尚未完成。"
                ),
            },
            "provenance": {
                "dataset_id": dataset_id,
                "pipeline_config_sha256": config_sha,
                "episode_ir_sha256": episode_ir_sha,
                "authority": (
                    f"{STATUS}; semantic_visual_audit={semantic_audit['status']}; "
                    f"disposition={semantic_audit['disposition']}"
                ),
            },
            "funnel": {
                "compiled_episodes": len(episodes),
                "compiled_views": sum(len(episode["observations"]) for episode in episodes),
                "compiled_questions": sum(len(episode["questions"]) for episode in episodes),
                "featured_episodes": len(featured_ids),
            },
            "splits": {
                "scenes": {
                    split: len(split_scene_ids.get(split, set()))
                    for split in ("train", "val", "test")
                },
                "episodes": {
                    split: split_episode_counts.get(split, 0)
                    for split in ("train", "val", "test")
                },
            },
            "coverage": {
                "trajectory_counts": dict(sorted(trajectory_counts.items())),
                "task_types": dict(sorted(all_task_types.items())),
                "programs": dict(sorted(all_programs.items())),
            },
            "families": {
                "count": sum(bool(variants) for variants in family_variants.values()),
                "shapes": dict(sorted(family_shapes.items())),
            },
            "composition_contract": {
                "train_atoms": sorted(all_atoms),
                "heldout_program_ids": holdout_programs,
                "heldout_benchmark_records": None,
            },
            "quality": {"semantic_visual_audit": semantic_audit},
            "verification": {
                "candidate_checks": [
                    {"name": "episode_ir_parse", "passed": True},
                    {"name": "episode_ir_config_hash_binding", "passed": True},
                    {"name": "rgb_media_closure", "passed": True},
                    {"name": "compiler_certificate_replay", "passed": True},
                ],
                "semantic_visual_audit": {
                    "status": semantic_audit["status"],
                    "gate": semantic_audit["gate"],
                    "disposition": semantic_audit["disposition"],
                    "hash_binding": dict(semantic_audit["hash_binding"]),
                },
            },
            "artifacts": [
                {
                    "name": "pipeline_config",
                    "url": _web_url(
                        config_path, web_root=web_root, project_root=project_root
                    ),
                    "sha256": config_sha,
                    "bytes": config_path.stat().st_size,
                    "records": None,
                },
                {
                    "name": "episode_ir",
                    "url": _web_url(
                        episode_ir_path, web_root=web_root, project_root=project_root
                    ),
                    "sha256": episode_ir_sha,
                    "bytes": episode_ir_path.stat().st_size,
                    "records": len(episodes),
                },
                *semantic_audit["artifacts"],
            ],
            "license_boundary": (
                "候选预览仅用于内部检查；语义 RGB 审查失败，必须返工且不得发布。"
                if semantic_audit["status"] == "failed"
                else "候选预览仅用于内部检查；语义 RGB 复核与正式 release authority 尚未闭合。"
            ),
            "featured_episode_ids": featured_ids,
            "episodes": catalog_episodes,
        }
        _assert_portable(catalog)
        if details_dir.exists():
            shutil.rmtree(details_dir)
        temporary.replace(details_dir)
        temporary = None  # type: ignore[assignment]
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
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--details-dir", type=Path, default=DEFAULT_DETAILS)
    return result


def main() -> int:
    args = parser().parse_args()
    catalog = build_web_candidate_catalog(
        args.release_dir, args.config, args.output, args.details_dir
    )
    print(
        json.dumps(
            {
                "status": catalog["status"],
                "semantic_visual_audit": catalog["authority"][
                    "semantic_visual_audit"
                ],
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
