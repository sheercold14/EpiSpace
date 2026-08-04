"""Audit whether an acquisition is informative enough for episode reasoning.

This report distinguishes evidence that *could* support a task from operation
types already sampled by the current compiler. That prevents a rich simulator
bundle from being mistaken for a complete MLLM training episode.
"""

from __future__ import annotations

import json
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

from omnigibson_episode.io import write_json_atomic

TARGET_OPERATIONS = ("G", "F", "B", "M", "R", "P", "V")
STRUCTURE_LABELS = frozenset(
    {"floors", "walls", "ceilings", "lawn", "driveway", "fence", "roof", "background"}
)


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"bundle member is missing: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"bundle member must be an object: {path.name}")
    return payload


def audit_reasoning_bundle(bundle_directory: Path) -> dict[str, Any]:
    bundle = bundle_directory.resolve()
    episode = _read(bundle / "spatial_episode.json")
    scene = _read(bundle / "scene_ir.json")
    oracle = _read(bundle / "relation_oracle.json")
    quality = _read(bundle / "quality_report.json")

    entities = {item["entity_id"]: item for item in scene["entities"]}
    non_structure = {
        entity_id
        for entity_id, item in entities.items()
        if str(item.get("raw_label", "")).casefold() not in STRUCTURE_LABELS
    }
    observations = episode["observations"]
    appearances: dict[str, list[int]] = {entity_id: [] for entity_id in entities}
    first_seen_counts: list[int] = []
    seen: set[str] = set()
    for index, observation in enumerate(observations):
        current = set(observation["visible_entity_ids"])
        first_seen_counts.append(len(current - seen))
        seen.update(current)
        for entity_id in current:
            if entity_id in appearances:
                appearances[entity_id].append(index)

    observed = {entity_id for entity_id, steps in appearances.items() if steps}
    observed_non_structure = observed & non_structure
    observed_categories = Counter(entities[entity_id]["raw_label"] for entity_id in observed)
    observed_regions = {
        entities[entity_id]["region_id"]
        for entity_id in observed
        if entities[entity_id].get("region_id")
    }
    reappearing = {
        entity_id
        for entity_id, steps in appearances.items()
        if any(right - left > 1 for left, right in pairwise(steps))
    }
    last_seen_candidates = {
        entity_id
        for entity_id, steps in appearances.items()
        if steps and steps[-1] < len(observations) - 1
    }

    sampled_operations: Counter[str] = Counter()
    signatures: Counter[str] = Counter()
    answers: Counter[str] = Counter()
    cross_view_queries = 0
    for query in episode["queries"]:
        operations = [node["operation"] for node in query["operation_graph"]["nodes"]]
        sampled_operations.update(operations)
        signatures["->".join(operations)] += 1
        answers[str(query["answer"])] += 1
        left, right = query["evidence_entity_ids"][:2]
        if set(appearances.get(left, [])) and set(appearances.get(right, [])) and not (
            set(appearances[left]) & set(appearances[right])
        ):
            cross_view_queries += 1

    task_manifest_path = bundle / "reasoning_tasks.json"
    task_manifest = (
        _read(task_manifest_path) if task_manifest_path.is_file() else None
    )
    task_capabilities: Counter[str] = Counter()
    if task_manifest is not None:
        for task in task_manifest.get("tasks", []):
            task_capabilities.update(str(value) for value in task.get("capabilities", []))
            operations = [
                node["operation"] for node in task["operation_graph"]["nodes"]
            ]
            sampled_operations.update(operations)
            signatures["->".join(operations)] += 1
            answers[str(task.get("answer"))] += 1

    cross_view_pairs: set[tuple[str, str]] = set()
    metric_pairs: set[tuple[str, str]] = set()
    for candidate in oracle["candidates"]:
        left = str(candidate["subject_entity_id"])
        right = str(candidate["reference_entity_id"])
        if left not in observed_non_structure or right not in observed_non_structure:
            continue
        pair = tuple(sorted((left, right)))
        metric_pairs.add(pair)
        if not (set(appearances[left]) & set(appearances[right])):
            cross_view_pairs.add(pair)

    sampled = set(sampled_operations)
    evidence_support = {
        "G": len(observed_non_structure) >= 2,
        "F": len(observations) >= 3,
        "B": len(observed_non_structure) >= 2 and len(observations) >= 3,
        "M": bool(metric_pairs),
        "R": bool(oracle["candidates"]),
        "P": len(observations) >= 3,
        "V": all(query["certificate"]["result"] == "pass" for query in episode["queries"]),
        "memory": bool(last_seen_candidates or reappearing),
        "cross_view_integration": bool(cross_view_pairs),
        "unknown_abstention": True,
    }
    # A selected task with an executable certificate is itself a constructive
    # witness that the bundle supports that capability. The generic oracle
    # heuristics above are intentionally conservative and do not enumerate
    # every valid metric or non-co-visible pair used by the task compiler.
    for capability, count in task_capabilities.items():
        if count > 0 and capability in evidence_support:
            evidence_support[capability] = True
    sample_coverage = {operation: operation in sampled for operation in TARGET_OPERATIONS}
    sample_coverage.update(
        {
            "memory": task_capabilities["memory"] > 0,
            "cross_view_integration": (
                cross_view_queries > 0
                or task_capabilities["cross_view_integration"] > 0
            ),
            "unknown_abstention": task_capabilities["unknown_abstention"] > 0,
        }
    )
    evidence_gates = {
        "non_structure_entities_at_least_10": len(observed_non_structure) >= 10,
        "observed_categories_at_least_8": len(observed_categories) >= 8,
        "informative_views_at_least_3": sum(count > 0 for count in first_seen_counts) >= 3,
        "cross_view_relation_pair_available": bool(cross_view_pairs),
        "metric_pairs_at_least_10": len(metric_pairs) >= 10,
    }
    gaps = [
        {
            "capability": capability,
            "reason": (
                "evidence substrate supports this capability, "
                "but no current query samples it"
            ),
        }
        for capability, supported in evidence_support.items()
        if supported and not sample_coverage[capability]
    ]
    report = {
        "schema_version": "omnigibson_reasoning_audit.v1",
        "episode_id": episode["episode_id"],
        "scene_id": episode["scene_id"],
        "quality_status": {
            "integrity": quality["integrity_status"],
            "visual": quality["visual_status"],
        },
        "evidence_summary": {
            "view_count": len(observations),
            "scene_entity_count": len(entities),
            "observed_entity_count": len(observed),
            "observed_non_structure_entity_count": len(observed_non_structure),
            "observed_entity_fraction": round(len(observed) / len(entities), 4),
            "observed_category_count": len(observed_categories),
            "observed_region_count": len(observed_regions),
            "new_entities_per_view": first_seen_counts,
            "reappearing_entity_count": len(reappearing),
            "last_seen_candidate_count": len(last_seen_candidates),
            "cross_view_relation_pair_count": len(cross_view_pairs),
            "metric_pair_count": len(metric_pairs),
        },
        "sample_summary": {
            "query_count": len(episode["queries"]),
            "reasoning_task_count": (
                int(task_manifest.get("task_count", 0)) if task_manifest else 0
            ),
            "answer_distribution": dict(sorted(answers.items())),
            "operation_node_distribution": dict(sorted(sampled_operations.items())),
            "program_signature_distribution": dict(sorted(signatures.items())),
            "cross_view_required_query_count": cross_view_queries,
        },
        "capability_matrix": {
            capability: {
                "evidence_supported": evidence_support[capability],
                "sampled": sample_coverage[capability],
            }
            for capability in (
                *TARGET_OPERATIONS,
                "memory",
                "cross_view_integration",
                "unknown_abstention",
            )
        },
        "evidence_eligibility": "pass" if all(evidence_gates.values()) else "insufficient",
        "evidence_gates": evidence_gates,
        "sample_coverage_status": "complete" if not gaps else "incomplete",
        "reasoning_status": "complete" if not gaps else "incomplete",
        "gaps": gaps,
        "audit_notes": [
            (
                "coverage includes executable reasoning_tasks.json specs"
                if task_manifest
                else "coverage currently includes canonical relation queries only"
            ),
            (
                "task wording is model-facing, while operation graphs and "
                "certificates remain hidden supervision"
            ),
            (
                "a certificate-passing executable task is treated as a constructive "
                "witness of capability evidence support"
            ),
        ],
    }
    write_json_atomic(bundle / "reasoning_audit.json", report)
    return report
