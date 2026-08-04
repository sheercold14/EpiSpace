"""Certify that two rendered bundles differ by one declared spatial cause."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from omnigibson_episode.intervention_acquire import relation_from_centers
from omnigibson_episode.io import sha256_file, write_json_atomic


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _entities(scene: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["source_entity_id"]): item for item in scene["entities"]}


def _center(entity: dict[str, Any]) -> list[float]:
    return [float(value) for value in entity["obb"]["center_m"]]


def _visible_views(episode: dict[str, Any], entity_id: str) -> list[str]:
    return [
        str(observation["view_id"])
        for observation in episode["observations"]
        if entity_id in {str(value) for value in observation["visible_entity_ids"]}
    ]


def _check(name: str, passed: bool, measured: Any, threshold: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "measured_value": measured,
        "threshold": threshold,
    }


def certify_minimal_pair(
    *,
    base_bundle: Path,
    variant_bundle: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    base = base_bundle.resolve()
    variant = variant_bundle.resolve()
    required = (
        "trajectory_plan.json",
        "render_report.json",
        "scene_ir.json",
        "spatial_episode.json",
    )
    for root in (base, variant):
        for name in required:
            if not (root / name).is_file():
                raise FileNotFoundError(root / name)
    execution_path = variant / "intervention_execution.json"
    if not execution_path.is_file():
        raise FileNotFoundError(execution_path)
    execution = _read(execution_path)
    if execution.get("status") != "success":
        raise ValueError("variant intervention is not successful")
    proposal = execution["proposal"]
    intervention_type = str(
        execution.get("intervention_type", "single_object_relation_flip")
    )
    if intervention_type not in {
        "single_object_relation_flip",
        "single_object_model_swap",
    }:
        raise ValueError(f"unsupported intervention type: {intervention_type}")
    base_scene = _read(base / "scene_ir.json")
    variant_scene = _read(variant / "scene_ir.json")
    base_episode = _read(base / "spatial_episode.json")
    variant_episode = _read(variant / "spatial_episode.json")
    base_entities = _entities(base_scene)
    variant_entities = _entities(variant_scene)
    target_name = str(proposal["target_source_entity_id"])
    anchor_name = str(proposal["anchor_source_entity_id"])
    if set(base_entities) != set(variant_entities):
        raise ValueError("base and variant entity registries differ")
    if target_name not in base_entities or anchor_name not in base_entities:
        raise ValueError("declared target or anchor is missing")
    if str(base_entities[target_name]["entity_id"]) != str(
        variant_entities[target_name]["entity_id"]
    ):
        raise ValueError("target identity changed across the pair")

    target_before = _center(base_entities[target_name])
    target_after = _center(variant_entities[target_name])
    anchor_before = _center(base_entities[anchor_name])
    anchor_after = _center(variant_entities[anchor_name])
    relation_before, margin_before = relation_from_centers(
        proposal=proposal,
        target_center=target_before,
        anchor_center=anchor_before,
    )
    relation_after, margin_after = relation_from_centers(
        proposal=proposal,
        target_center=target_after,
        anchor_center=anchor_after,
    )
    non_target_drifts = {
        name: math.dist(_center(base_entities[name]), _center(variant_entities[name]))
        for name in base_entities
        if name != target_name
    }
    max_non_target_drift = max(non_target_drifts.values(), default=0.0)
    trajectory_equal = sha256_file(base / "trajectory_plan.json") == sha256_file(
        variant / "trajectory_plan.json"
    )
    target_entity_id = str(base_entities[target_name]["entity_id"])
    anchor_entity_id = str(base_entities[anchor_name]["entity_id"])
    evidence = {
        "base": {
            "target_view_ids": _visible_views(base_episode, target_entity_id),
            "anchor_view_ids": _visible_views(base_episode, anchor_entity_id),
        },
        "variant": {
            "target_view_ids": _visible_views(variant_episode, target_entity_id),
            "anchor_view_ids": _visible_views(variant_episode, anchor_entity_id),
        },
    }
    evidence_complete = all(
        evidence[side][role]
        for side in ("base", "variant")
        for role in ("target_view_ids", "anchor_view_ids")
    )
    checks = [
        _check(
            "same_scene_identity",
            base_episode["scene_id"] == variant_episode["scene_id"],
            variant_episode["scene_id"],
            base_episode["scene_id"],
        ),
        _check(
            "same_episode_family",
            base_episode["family_id"] == variant_episode["family_id"],
            variant_episode["family_id"],
            base_episode["family_id"],
        ),
        _check("same_camera_trajectory", trajectory_equal, trajectory_equal, True),
        _check(
            "single_mutable_entity",
            max_non_target_drift <= 0.02,
            round(max_non_target_drift, 6),
            0.02,
        ),
        _check("evidence_available_both_sides", evidence_complete, evidence_complete, True),
    ]
    checks.extend(
        [
            _check(
                "relation_before_matches_plan",
                relation_before == proposal["relation_before"],
                relation_before,
                proposal["relation_before"],
            ),
            _check(
                "relation_after_matches_plan",
                relation_after == proposal["relation_after"],
                relation_after,
                proposal["relation_after"],
            ),
            _check(
                "axis_margin_before_m",
                margin_before >= 0.4,
                round(margin_before, 6),
                0.4,
            ),
            _check(
                "axis_margin_after_m",
                margin_after >= 0.4,
                round(margin_after, 6),
                0.4,
            ),
        ]
    )
    model_mutation = None
    if intervention_type == "single_object_relation_flip":
        checks.append(
            _check(
                "answers_flip",
                relation_before != relation_after,
                f"{relation_before}->{relation_after}",
                "different",
            )
        )
        learning_signal = "answer_flip"
    else:
        base_snapshot = _read(base / "scene_snapshot.json")
        variant_snapshot = _read(variant / "scene_snapshot.json")
        base_snapshot_entities = {
            str(item["source_entity_id"]): item
            for item in base_snapshot["entities"]
        }
        variant_snapshot_entities = {
            str(item["source_entity_id"]): item
            for item in variant_snapshot["entities"]
        }
        source_model = str(base_snapshot_entities[target_name].get("model"))
        replacement_model = str(variant_snapshot_entities[target_name].get("model"))
        base_category = str(base_snapshot_entities[target_name].get("category"))
        variant_category = str(variant_snapshot_entities[target_name].get("category"))
        target_center_drift = math.dist(target_before, target_after)
        checks.extend(
            [
                _check(
                    "source_model_matches_plan",
                    source_model == proposal["source_model"],
                    source_model,
                    proposal["source_model"],
                ),
                _check(
                    "replacement_model_matches_plan",
                    replacement_model == proposal["replacement_model"],
                    replacement_model,
                    proposal["replacement_model"],
                ),
                _check(
                    "asset_model_changed",
                    source_model != replacement_model,
                    f"{source_model}->{replacement_model}",
                    "different",
                ),
                _check(
                    "semantic_category_invariant",
                    base_category == variant_category == proposal["target_category"],
                    f"{base_category}->{variant_category}",
                    proposal["target_category"],
                ),
                _check(
                    "target_center_invariance_m",
                    target_center_drift <= 0.08,
                    round(target_center_drift, 6),
                    0.08,
                ),
                _check(
                    "answers_invariant",
                    relation_before == relation_after,
                    f"{relation_before}->{relation_after}",
                    "same",
                ),
            ]
        )
        model_mutation = {
            "source_model": source_model,
            "replacement_model": replacement_model,
            "category": base_category,
        }
        learning_signal = "answer_invariance"
    failed = [item["name"] for item in checks if not item["passed"]]
    if failed:
        message = "minimal-pair certification failed: " + ", ".join(failed)
        write_json_atomic(
            variant / "minimal_pair_failure.json",
            {
                "schema_version": "spatial_minimal_pair_failure.v1",
                "status": "failed",
                "proposal_id": proposal["proposal_id"],
                "intervention_type": intervention_type,
                "failed_checks": failed,
                "checks": checks,
                "error": {"type": "ValueError", "message": message},
            },
        )
        raise ValueError(message)
    pair_key = {
        "family_id": base_episode["family_id"],
        "proposal_id": proposal["proposal_id"],
        "base_episode_id": base_episode["episode_id"],
        "variant_episode_id": variant_episode["episode_id"],
    }
    digest = hashlib.sha256(
        json.dumps(pair_key, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    certificate = {
        "schema_version": "spatial_minimal_pair.v1",
        "pair_id": f"ogpair-{digest[:20]}",
        "status": "certified",
        "scene_id": base_episode["scene_id"],
        "split_group": base_episode["split_group"],
        "family_id": base_episode["family_id"],
        "intervention_type": intervention_type,
        "learning_signal": learning_signal,
        "causal_contract": execution.get("checks", []),
        "episodes": {
            "base": {
                "episode_id": base_episode["episode_id"],
                "family_variant": base_episode["family_variant"],
                "bundle": str(base),
                "answer": relation_before,
            },
            "variant": {
                "episode_id": variant_episode["episode_id"],
                "family_variant": variant_episode["family_variant"],
                "bundle": str(variant),
                "answer": relation_after,
            },
        },
        "query_spec": {
            "intent": "first_view_canonical_relation",
            "subject": {
                "entity_id": target_entity_id,
                "source_entity_id": target_name,
                "category": base_entities[target_name]["raw_label"],
            },
            "reference": {
                "entity_id": anchor_entity_id,
                "source_entity_id": anchor_name,
                "category": base_entities[anchor_name]["raw_label"],
            },
            "frame": proposal.get("relation_frame", "legacy_canonical_world_xy"),
            "frame_view_id": proposal.get("relation_frame_view_id"),
            "program_signature": "G->G->F->B->B->R->V",
            "learning_signal": learning_signal,
            "natural_language_policy": (
                "realize the same unambiguous question on both sides and explicitly say "
                "that the first view defines right and forward; never expose IDs, "
                "coordinates, certificates, or the intervention"
            ),
        },
        "evidence": evidence,
        "geometry": {
            "target_center_before_m": target_before,
            "target_center_after_m": target_after,
            "anchor_center_m": anchor_before,
            "movement_m": round(math.dist(target_before, target_after), 6),
            "axis_margin_before_m": round(margin_before, 6),
            "axis_margin_after_m": round(margin_after, 6),
        },
        "model_mutation": model_mutation,
        "checks": checks,
        "provenance": {
            "base_render_sha256": sha256_file(base / "render_report.json"),
            "variant_render_sha256": sha256_file(variant / "render_report.json"),
            "execution_sha256": sha256_file(execution_path),
        },
    }
    output = output_path or variant / "minimal_pair.json"
    write_json_atomic(output, certificate)
    (variant / "minimal_pair_failure.json").unlink(missing_ok=True)
    return certificate
