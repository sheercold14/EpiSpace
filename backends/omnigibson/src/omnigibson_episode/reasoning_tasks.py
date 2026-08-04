"""Compile diverse, executable reasoning task specs from one verified episode."""

# Chinese model-facing text intentionally uses full-width punctuation.
# ruff: noqa: RUF001

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from omnigibson_episode.geometry import dominant_planar_relation
from omnigibson_episode.io import write_json_atomic
from omnigibson_episode.scene_inventory import STRUCTURE_CATEGORIES

_RELATION_ZH = {
    "left_of": "左侧",
    "right_of": "右侧",
    "in_front_of": "前方",
    "behind": "后方",
}
_QUADRANT_ZH = {
    "front_left": "左前方",
    "front_right": "右前方",
    "back_left": "左后方",
    "back_right": "右后方",
}
_REQUIRED_CAPABILITIES = (
    "G",
    "F",
    "B",
    "M",
    "R",
    "P",
    "V",
    "memory",
    "cross_view_integration",
    "unknown_abstention",
)
_MAX_CROSS_VIEW_CENTER_DISTANCE_M = 12.0


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _graph(
    external_inputs: dict[str, str], nodes: list[dict[str, Any]], answer_node: str
) -> dict[str, Any]:
    from spatial_episode.contracts.operation_v1 import OperationGraphV1

    return OperationGraphV1.model_validate(
        {
            "schema_version": "operation_graph.v1",
            "external_inputs": external_inputs,
            "nodes": nodes,
            "answer_node": answer_node,
        }
    ).model_dump(mode="json")


def _ground_graph(entity_id: str) -> dict[str, Any]:
    return _graph(
        {"view_context": "view", "subject_hint": "entity"},
        [
            {
                "node_id": "ground_subject",
                "operation": "G",
                "inputs": ["view_context", "subject_hint"],
                "output_type": "entity",
                "parameters": {"entity_id": entity_id},
            },
            {
                "node_id": "verify_grounding",
                "operation": "V",
                "inputs": ["ground_subject"],
                "output_type": "boolean",
            },
        ],
        "verify_grounding",
    )


def _belief_graph(entity_id: str) -> dict[str, Any]:
    return _graph(
        {
            "view_context": "view",
            "subject_hint": "entity",
            "initial_belief": "belief",
            "query_frame": "frame",
            "world_frame": "frame",
        },
        [
            {
                "node_id": "ground_subject",
                "operation": "G",
                "inputs": ["view_context", "subject_hint"],
                "output_type": "entity",
                "parameters": {"entity_id": entity_id},
            },
            {
                "node_id": "align_frame",
                "operation": "F",
                "inputs": ["query_frame", "world_frame"],
                "output_type": "transform",
            },
            {
                "node_id": "update_belief",
                "operation": "B",
                "inputs": ["initial_belief", "ground_subject", "align_frame"],
                "output_type": "belief",
            },
            {
                "node_id": "verify_memory",
                "operation": "V",
                "inputs": ["update_belief"],
                "output_type": "boolean",
            },
        ],
        "verify_memory",
    )


def _metric_graph(subject_id: str, reference_id: str) -> dict[str, Any]:
    return _graph(
        {
            "view_context": "view",
            "subject_hint": "entity",
            "reference_hint": "entity",
            "query_frame": "frame",
        },
        [
            {
                "node_id": "ground_subject",
                "operation": "G",
                "inputs": ["view_context", "subject_hint"],
                "output_type": "entity",
                "parameters": {"entity_id": subject_id},
            },
            {
                "node_id": "ground_reference",
                "operation": "G",
                "inputs": ["view_context", "reference_hint"],
                "output_type": "entity",
                "parameters": {"entity_id": reference_id},
            },
            {
                "node_id": "estimate_distance",
                "operation": "M",
                "inputs": ["ground_subject", "ground_reference", "query_frame"],
                "output_type": "metric",
            },
            {
                "node_id": "verify_metric",
                "operation": "V",
                "inputs": ["estimate_distance"],
                "output_type": "boolean",
            },
        ],
        "verify_metric",
    )


def _relation_graph(subject_id: str, reference_id: str, relation: str) -> dict[str, Any]:
    return _graph(
        {
            "view_context": "view",
            "subject_hint": "entity",
            "reference_hint": "entity",
            "query_frame": "frame",
            "world_frame": "frame",
            "initial_belief": "belief",
        },
        [
            {
                "node_id": "ground_subject",
                "operation": "G",
                "inputs": ["view_context", "subject_hint"],
                "output_type": "entity",
                "parameters": {"entity_id": subject_id},
            },
            {
                "node_id": "ground_reference",
                "operation": "G",
                "inputs": ["view_context", "reference_hint"],
                "output_type": "entity",
                "parameters": {"entity_id": reference_id},
            },
            {
                "node_id": "align_frame",
                "operation": "F",
                "inputs": ["query_frame", "world_frame"],
                "output_type": "transform",
            },
            {
                "node_id": "update_subject_belief",
                "operation": "B",
                "inputs": ["initial_belief", "ground_subject", "align_frame"],
                "output_type": "belief",
            },
            {
                "node_id": "update_pair_belief",
                "operation": "B",
                "inputs": ["update_subject_belief", "ground_reference", "align_frame"],
                "output_type": "belief",
            },
            {
                "node_id": "evaluate_relation",
                "operation": "R",
                "inputs": ["ground_subject", "ground_reference", "query_frame"],
                "output_type": "relation",
                "parameters": {"expected_relation": relation},
            },
            {
                "node_id": "verify_relation",
                "operation": "V",
                "inputs": ["evaluate_relation"],
                "output_type": "boolean",
                "parameters": {"expected_relation": relation},
            },
        ],
        "verify_relation",
    )


def _perspective_graph(ids: tuple[str, str, str], quadrant: str) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    for role, entity_id in zip(
        ("viewpoint", "facing", "target"), ids, strict=True
    ):
        nodes.append(
            {
                "node_id": f"ground_{role}",
                "operation": "G",
                "inputs": ["view_context", f"{role}_hint"],
                "output_type": "entity",
                "parameters": {"entity_id": entity_id},
            }
        )
    nodes.extend(
        [
            {
                "node_id": "align_frame",
                "operation": "F",
                "inputs": ["query_frame", "world_frame"],
                "output_type": "transform",
            },
            {
                "node_id": "belief_viewpoint",
                "operation": "B",
                "inputs": ["initial_belief", "ground_viewpoint", "align_frame"],
                "output_type": "belief",
            },
            {
                "node_id": "belief_facing",
                "operation": "B",
                "inputs": ["belief_viewpoint", "ground_facing", "align_frame"],
                "output_type": "belief",
            },
            {
                "node_id": "belief_target",
                "operation": "B",
                "inputs": ["belief_facing", "ground_target", "align_frame"],
                "output_type": "belief",
            },
            {
                "node_id": "predict_target_view",
                "operation": "P",
                "inputs": ["belief_target", "target_view"],
                "output_type": "view_prediction",
                "parameters": {"expected_quadrant": quadrant},
            },
            {
                "node_id": "verify_perspective",
                "operation": "V",
                "inputs": ["predict_target_view"],
                "output_type": "boolean",
                "parameters": {"expected_quadrant": quadrant},
            },
        ]
    )
    return _graph(
        {
            "view_context": "view",
            "viewpoint_hint": "entity",
            "facing_hint": "entity",
            "target_hint": "entity",
            "query_frame": "frame",
            "world_frame": "frame",
            "initial_belief": "belief",
            "target_view": "view",
        },
        nodes,
        "verify_perspective",
    )


def _unknown_graph() -> dict[str, Any]:
    return _graph(
        {
            "view_context": "view",
            "initial_belief": "belief",
            "query_frame": "frame",
            "world_frame": "frame",
        },
        [
            {
                "node_id": "ground_observed_set",
                "operation": "G",
                "inputs": ["view_context"],
                "output_type": "entity_set",
            },
            {
                "node_id": "align_frame",
                "operation": "F",
                "inputs": ["query_frame", "world_frame"],
                "output_type": "transform",
            },
            {
                "node_id": "update_observed_belief",
                "operation": "B",
                "inputs": ["initial_belief", "ground_observed_set", "align_frame"],
                "output_type": "belief",
            },
            {
                "node_id": "verify_underdetermined",
                "operation": "V",
                "inputs": ["update_observed_belief"],
                "output_type": "boolean",
                "parameters": {"expected_status": "unknown"},
            },
        ],
        "verify_underdetermined",
    )


def _task_id(kind: str, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {"kind": kind, **payload}, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return f"ogtask-{kind}-{digest[:14]}"


def _name(entity: dict[str, Any]) -> str:
    return str(entity.get("_display_name", entity["raw_label"])).replace("_", " ")


def _model_visible_entity_references(
    eligible: dict[str, dict[str, Any]], observations: list[dict[str, Any]]
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[int]],
    dict[str, dict[str, Any]],
]:
    """Build unambiguous references using only information present in RGB views.

    Globally unique category labels need no qualifier. Repeated labels are retained
    only when at least one observation contains exactly one visible instance of that
    label; the earliest such observation becomes a model-visible anchor. Simulator
    room identifiers are deliberately excluded because they are hidden oracle data.
    """

    appearances = {
        entity_id: [
            index
            for index, observation in enumerate(observations)
            if entity_id in {str(value) for value in observation["visible_entity_ids"]}
        ]
        for entity_id in eligible
    }
    label_counts = Counter(str(item["raw_label"]) for item in eligible.values())
    visible_label_counts = []
    for observation in observations:
        visible_ids = {str(value) for value in observation["visible_entity_ids"]}
        visible_label_counts.append(
            Counter(
                str(eligible[entity_id]["raw_label"])
                for entity_id in visible_ids
                if entity_id in eligible
            )
        )

    referenced: dict[str, dict[str, Any]] = {}
    reference_metadata: dict[str, dict[str, Any]] = {}
    for entity_id, item in eligible.items():
        label = str(item["raw_label"])
        if label_counts[label] == 1:
            referenced[entity_id] = {**item, "_display_name": label}
            reference_metadata[entity_id] = {
                "surface_name": label.replace("_", " "),
                "basis": "globally_unique_category",
                "anchor_view_id": None,
            }
            continue

        anchor_steps = [
            step
            for step in appearances[entity_id]
            if visible_label_counts[step][label] == 1
        ]
        if not anchor_steps:
            continue
        anchor_step = anchor_steps[0]
        anchor_view_id = observations[anchor_step]["view_id"]
        display_name = f"第{anchor_step + 1}个视角里看到的{label}"
        referenced[entity_id] = {
            **item,
            "_display_name": display_name,
            "_reference_anchor_step": anchor_step,
        }
        reference_metadata[entity_id] = {
            "surface_name": display_name.replace("_", " "),
            "basis": "single_instance_in_anchor_view",
            "anchor_view_id": anchor_view_id,
        }

    return referenced, appearances, reference_metadata


def _center(entity: dict[str, Any]) -> list[float]:
    return [float(value) for value in entity["obb"]["center_m"]]


def _view_relation(
    left: list[float], right: list[float], rotation_xyzw: list[float]
) -> tuple[str, float, float, float]:
    relation, _, margin, right_delta, front_delta = dominant_planar_relation(
        left, right, rotation_xyzw
    )
    return relation, margin, right_delta, front_delta


def _perspective_quadrant(
    viewpoint: list[float], facing: list[float], target: list[float]
) -> tuple[str, float, float] | None:
    fx, fy = facing[0] - viewpoint[0], facing[1] - viewpoint[1]
    norm = math.hypot(fx, fy)
    if not 0.8 <= norm <= 8.0:
        return None
    fx, fy = fx / norm, fy / norm
    rx, ry = fy, -fx
    tx, ty = target[0] - viewpoint[0], target[1] - viewpoint[1]
    right, front = tx * rx + ty * ry, tx * fx + ty * fy
    if abs(right) < 0.3 or abs(front) < 0.3:
        return None
    quadrant = ("front" if front > 0 else "back") + "_" + (
        "right" if right > 0 else "left"
    )
    return quadrant, right, front


def _certificate(checks: list[dict[str, Any]], oracle: dict[str, Any]) -> dict[str, Any]:
    return {
        "verifier_version": "omnigibson-reasoning-task/1.0",
        "result": "pass" if all(item["passed"] for item in checks) else "fail",
        "checks": checks,
        "oracle": oracle,
    }


def _check(name: str, passed: bool, measured: Any, threshold: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "measured_value": measured,
        "threshold": threshold,
    }


def compile_reasoning_tasks(
    bundle_directory: Path, output_path: Path | None = None
) -> dict[str, Any]:
    bundle = bundle_directory.resolve()
    scene = _read(bundle / "scene_ir.json")
    episode = _read(bundle / "spatial_episode.json")
    entities = {str(item["entity_id"]): item for item in scene["entities"]}
    eligible = {
        entity_id: item
        for entity_id, item in entities.items()
        if str(item["raw_label"]).casefold() not in STRUCTURE_CATEGORIES
    }
    observations = episode["observations"]
    unique, appearances, reference_metadata = _model_visible_entity_references(
        eligible, observations
    )
    observed_unique = [
        entity_id for entity_id in sorted(unique) if appearances.get(entity_id)
    ]
    tasks: list[dict[str, Any]] = []

    globally_unique_observed = [
        entity_id
        for entity_id in observed_unique
        if reference_metadata[entity_id]["basis"] == "globally_unique_category"
    ]
    repeated_with_non_anchor_evidence = [
        entity_id
        for entity_id in observed_unique
        if reference_metadata[entity_id]["basis"] == "single_instance_in_anchor_view"
        and any(
            observations[step]["view_id"]
            != reference_metadata[entity_id]["anchor_view_id"]
            for step in appearances[entity_id]
        )
    ]
    grounding_candidates = globally_unique_observed or repeated_with_non_anchor_evidence
    if grounding_candidates:
        subject_id = min(
            grounding_candidates, key=lambda value: (-len(appearances[value]), value)
        )
        if reference_metadata[subject_id]["basis"] == "globally_unique_category":
            step = appearances[subject_id][0]
        else:
            step = next(
                value
                for value in appearances[subject_id]
                if observations[value]["view_id"]
                != reference_metadata[subject_id]["anchor_view_id"]
            )
        entity = unique[subject_id]
        payload = {"subject": subject_id, "view": observations[step]["view_id"]}
        tasks.append(
            {
                "task_id": _task_id("grounding", payload),
                "task_type": "grounding_presence",
                "capabilities": ["G", "V"],
                "question_zh": f"在第{step + 1}个视角里，能看到{_name(entity)}吗？",
                "surface_answer_zh": "能。",
                "answer": True,
                "status": "accepted",
                "evidence_view_ids": [observations[step]["view_id"]],
                "evidence_entity_ids": [subject_id],
                "operation_graph": _ground_graph(subject_id),
                "certificate": _certificate(
                    [_check("entity_visible", True, observations[step]["view_id"], True)],
                    {"visible_steps": appearances[subject_id]},
                ),
            }
        )

    memory_candidates = [
        entity_id
        for entity_id in observed_unique
        if appearances[entity_id][-1] < len(observations) - 1
        and (
            reference_metadata[entity_id]["basis"] == "globally_unique_category"
            or reference_metadata[entity_id]["anchor_view_id"]
            != observations[appearances[entity_id][-1]]["view_id"]
        )
    ]
    if memory_candidates:
        subject_id = min(
            memory_candidates,
            key=lambda value: (-appearances[value][-1], value),
        )
        last = appearances[subject_id][-1]
        entity = unique[subject_id]
        payload = {"subject": subject_id, "last": last}
        tasks.append(
            {
                "task_id": _task_id("memory", payload),
                "task_type": "last_seen_memory",
                "capabilities": ["G", "F", "B", "V", "memory"],
                "question_zh": (
                    f"走到最后一个视角时，最后一次看到{_name(entity)}是第几个视角？"
                ),
                "surface_answer_zh": f"最后一次是在第{last + 1}个视角。",
                "answer": last + 1,
                "status": "accepted",
                "evidence_view_ids": [
                    observations[index]["view_id"]
                    for index in appearances[subject_id]
                ],
                "evidence_entity_ids": [subject_id],
                "operation_graph": _belief_graph(subject_id),
                "certificate": _certificate(
                    [
                        _check("last_visible_step", True, last, last),
                        _check(
                            "absent_after_last_seen",
                            all(
                                subject_id
                                not in {
                                    str(value)
                                    for value in observations[index]["visible_entity_ids"]
                                }
                                for index in range(last + 1, len(observations))
                            ),
                            True,
                            True,
                        ),
                    ],
                    {"visible_steps": appearances[subject_id]},
                ),
            }
        )

    pair_candidates = []
    for left_id, right_id in itertools.combinations(observed_unique, 2):
        common = sorted(set(appearances[left_id]) & set(appearances[right_id]))
        distance = math.dist(_center(unique[left_id]), _center(unique[right_id]))
        if common and 0.5 <= distance <= 8.0:
            pair_candidates.append((abs(distance - 2.0), left_id, right_id, common, distance))
    if pair_candidates:
        _, left_id, right_id, common, distance = min(pair_candidates)
        left, right = unique[left_id], unique[right_id]
        payload = {"subject": left_id, "reference": right_id, "distance": round(distance, 3)}
        tasks.append(
            {
                "task_id": _task_id("metric", payload),
                "task_type": "metric_distance",
                "capabilities": ["G", "M", "V"],
                "question_zh": f"综合这些视角，{_name(left)}和{_name(right)}中心相距约多少米？",
                "surface_answer_zh": f"约{distance:.1f}米。",
                "answer": round(distance, 1),
                "answer_tolerance_m": 0.2,
                "status": "accepted",
                "evidence_view_ids": [observations[index]["view_id"] for index in common],
                "evidence_entity_ids": [left_id, right_id],
                "operation_graph": _metric_graph(left_id, right_id),
                "certificate": _certificate(
                    [_check("center_distance_m", True, round(distance, 6), ">=0.5")],
                    {
                        "subject_center_m": _center(left),
                        "reference_center_m": _center(right),
                    },
                ),
            }
        )

        frame_options = []
        for _, candidate_left, candidate_right, candidate_common, _ in pair_candidates:
            for step in candidate_common:
                relation, margin, right_delta, front_delta = _view_relation(
                    _center(unique[candidate_left]),
                    _center(unique[candidate_right]),
                    observations[step]["world_from_camera"]["rotation_xyzw"],
                )
                if margin >= 0.4:
                    frame_options.append(
                        (
                            -margin,
                            candidate_left,
                            candidate_right,
                            step,
                            relation,
                            right_delta,
                            front_delta,
                        )
                    )
        if frame_options:
            _, left_id, right_id, step, relation, right_delta, front_delta = min(
                frame_options
            )
            left, right = unique[left_id], unique[right_id]
            payload = {
                "subject": left_id,
                "reference": right_id,
                "view": observations[step]["view_id"],
                "relation": relation,
            }
            tasks.append(
                {
                    "task_id": _task_id("frame", payload),
                    "task_type": "egocentric_relation",
                    "capabilities": ["G", "F", "B", "R", "V"],
                    "question_zh": (
                        f"只按第{step + 1}个视角的朝向，{_name(left)}在"
                        f"{_name(right)}的哪一侧？"
                    ),
                    "surface_answer_zh": f"在{_name(right)}的{_RELATION_ZH[relation]}。",
                    "answer": relation,
                    "status": "accepted",
                    "evidence_view_ids": [observations[step]["view_id"]],
                    "evidence_entity_ids": [left_id, right_id],
                    "operation_graph": _relation_graph(left_id, right_id, relation),
                    "certificate": _certificate(
                        [
                            _check(
                                "ego_axis_margin_m",
                                True,
                                round(abs(abs(right_delta) - abs(front_delta)), 6),
                                0.4,
                            )
                        ],
                        {
                            "ego_right_delta_m": round(right_delta, 6),
                            "ego_front_delta_m": round(front_delta, 6),
                            "view_rotation_xyzw": observations[step]["world_from_camera"][
                                "rotation_xyzw"
                            ],
                        },
                    ),
                }
            )

    canonical_step = 0
    canonical_observation = observations[canonical_step]
    canonical_rotation = canonical_observation["world_from_camera"]["rotation_xyzw"]
    cross_options = []
    for left_id, right_id in itertools.combinations(observed_unique, 2):
        if set(appearances[left_id]) & set(appearances[right_id]):
            continue
        relation, margin, right_delta, front_delta = _view_relation(
            _center(unique[left_id]),
            _center(unique[right_id]),
            canonical_rotation,
        )
        distance = math.dist(_center(unique[left_id]), _center(unique[right_id]))
        if margin >= 0.4 and distance <= _MAX_CROSS_VIEW_CENTER_DISTANCE_M:
            cross_options.append(
                (
                    -margin,
                    left_id,
                    right_id,
                    relation,
                    distance,
                    right_delta,
                    front_delta,
                )
            )
    if cross_options:
        (
            neg_margin,
            left_id,
            right_id,
            relation,
            distance,
            right_delta,
            front_delta,
        ) = min(cross_options)
        left, right = unique[left_id], unique[right_id]
        evidence_steps = sorted(set(appearances[left_id]) | set(appearances[right_id]))
        payload = {"subject": left_id, "reference": right_id, "relation": relation}
        tasks.append(
            {
                "task_id": _task_id("crossview", payload),
                "task_type": "cross_view_relation",
                "capabilities": ["G", "F", "B", "R", "V", "cross_view_integration"],
                "question_zh": (
                    f"以第{canonical_step + 1}个视角的朝向为正前方，综合全部观察，"
                    f"{_name(left)}在{_name(right)}的哪个方向？"
                ),
                "surface_answer_zh": f"在{_name(right)}的{_RELATION_ZH[relation]}。",
                "answer": relation,
                "status": "accepted",
                "evidence_view_ids": [observations[index]["view_id"] for index in evidence_steps],
                "evidence_entity_ids": [left_id, right_id],
                "operation_graph": _relation_graph(left_id, right_id, relation),
                "certificate": _certificate(
                    [
                        _check("never_co_visible", True, True, True),
                        _check(
                            "canonical_axis_margin_m",
                            True,
                            round(-neg_margin, 6),
                            0.4,
                        ),
                    ],
                    {
                        "subject_visible_steps": appearances[left_id],
                        "reference_visible_steps": appearances[right_id],
                        "center_distance_m": round(distance, 6),
                        "canonical_frame_view_id": canonical_observation["view_id"],
                        "canonical_frame_rotation_xyzw": canonical_rotation,
                        "canonical_right_delta_m": round(right_delta, 6),
                        "canonical_front_delta_m": round(front_delta, 6),
                    },
                ),
            }
        )
        opposite = {
            "left_of": "right_of",
            "right_of": "left_of",
            "in_front_of": "behind",
            "behind": "in_front_of",
        }[relation]
        verify_payload = {**payload, "claimed": opposite}
        tasks.append(
            {
                "task_id": _task_id("verify", verify_payload),
                "task_type": "counterfactual_verification",
                "capabilities": ["G", "F", "B", "R", "V", "cross_view_integration"],
                "question_zh": (
                    f"仍以第{canonical_step + 1}个视角的朝向为正前方。有人说"
                    f"{_name(left)}在{_name(right)}的{_RELATION_ZH[opposite]}，"
                    "这个说法对吗？"
                ),
                "surface_answer_zh": (
                    f"不对；它实际在{_name(right)}的{_RELATION_ZH[relation]}。"
                ),
                "answer": False,
                "status": "accepted",
                "evidence_view_ids": [observations[index]["view_id"] for index in evidence_steps],
                "evidence_entity_ids": [left_id, right_id],
                "operation_graph": _relation_graph(left_id, right_id, relation),
                "certificate": _certificate(
                    [_check("claim_contradicted", relation != opposite, relation, opposite)],
                    {"verified_relation": relation, "rejected_relation": opposite},
                ),
            }
        )

    perspective_pool = sorted(
        observed_unique, key=lambda value: (-len(appearances[value]), value)
    )[:20]
    perspective_options = []
    for viewpoint_id, facing_id, target_id in itertools.permutations(perspective_pool, 3):
        result = _perspective_quadrant(
            _center(unique[viewpoint_id]),
            _center(unique[facing_id]),
            _center(unique[target_id]),
        )
        if result is None:
            continue
        quadrant, right, front = result
        score = min(abs(right), abs(front))
        perspective_options.append(
            (-score, viewpoint_id, facing_id, target_id, quadrant, right, front)
        )
    if perspective_options:
        _, viewpoint_id, facing_id, target_id, quadrant, right, front = min(
            perspective_options
        )
        viewpoint, facing, target = (
            unique[viewpoint_id],
            unique[facing_id],
            unique[target_id],
        )
        evidence_steps = sorted(
            set(appearances[viewpoint_id])
            | set(appearances[facing_id])
            | set(appearances[target_id])
        )
        payload = {
            "viewpoint": viewpoint_id,
            "facing": facing_id,
            "target": target_id,
            "quadrant": quadrant,
        }
        tasks.append(
            {
                "task_id": _task_id("perspective", payload),
                "task_type": "object_centric_perspective",
                "capabilities": ["G", "F", "B", "P", "V"],
                "question_zh": (
                    f"假设站在{_name(viewpoint)}的位置并面向{_name(facing)}，"
                    f"{_name(target)}位于你的哪个方位？"
                ),
                "surface_answer_zh": f"位于你的{_QUADRANT_ZH[quadrant]}。",
                "answer": quadrant,
                "status": "accepted",
                "evidence_view_ids": [observations[index]["view_id"] for index in evidence_steps],
                "evidence_entity_ids": [viewpoint_id, facing_id, target_id],
                "operation_graph": _perspective_graph(
                    (viewpoint_id, facing_id, target_id), quadrant
                ),
                "certificate": _certificate(
                    [
                        _check("ego_right_abs_m", abs(right) >= 0.3, round(right, 6), 0.3),
                        _check("ego_front_abs_m", abs(front) >= 0.3, round(front, 6), 0.3),
                    ],
                    {
                        "viewpoint_center_m": _center(viewpoint),
                        "facing_center_m": _center(facing),
                        "target_center_m": _center(target),
                        "target_ego_right_m": round(right, 6),
                        "target_ego_front_m": round(front, 6),
                    },
                ),
            }
        )

    observed_ids = {entity_id for entity_id, steps in appearances.items() if steps}
    observed_labels = {
        str(eligible[entity_id]["raw_label"]) for entity_id in observed_ids
    }
    world_labels = {str(item["raw_label"]) for item in eligible.values()}
    unobserved_world_labels = sorted(world_labels - observed_labels)
    plausible_unseen_labels = [
        label
        for label in (
            "bed",
            "bathtub",
            "microwave",
            "piano",
            "treadmill",
            "fire_extinguisher",
        )
        if label not in observed_labels
    ]
    if unobserved_world_labels or plausible_unseen_labels:
        label = (unobserved_world_labels or plausible_unseen_labels)[0]
        world_contains = label in world_labels
        payload = {"unobserved_category": label}
        tasks.append(
            {
                "task_id": _task_id("unknown", payload),
                "task_type": "unknown_abstention",
                "capabilities": ["G", "F", "B", "V", "unknown_abstention"],
                "question_zh": (
                    f"根据目前看过的画面，能否确定这个场景里存在{label.replace('_', ' ')}？"
                ),
                "surface_answer_zh": (
                    "无法确定；给出的所有视角里都没有观察到它，未观察到不等于不存在。"
                ),
                "answer": None,
                "status": "unknown",
                "evidence_view_ids": [item["view_id"] for item in observations],
                "evidence_entity_ids": [],
                "operation_graph": _unknown_graph(),
                "certificate": {
                    "verifier_version": "omnigibson-reasoning-task/1.0",
                    "result": "unknown",
                    "checks": [
                        _check("category_absent_from_all_observations", True, True, True),
                        _check("world_truth_not_model_visible", True, True, True),
                    ],
                    "oracle": {
                        "world_truth_contains_category": world_contains,
                        "observed_contains_category": False,
                        "abstention_basis": (
                            "the model-visible views do not provide positive evidence, and "
                            "absence from partial views cannot certify scene-wide absence"
                        ),
                    },
                },
            }
        )

    capability_counts: Counter[str] = Counter()
    for task in tasks:
        capability_counts.update(task["capabilities"])
    coverage = {
        capability: {
            "sampled": capability_counts[capability] > 0,
            "task_count": capability_counts[capability],
        }
        for capability in _REQUIRED_CAPABILITIES
    }
    gaps = [
        {
            "capability": capability,
            "reason": "no unambiguous executable task could be selected from this trajectory",
        }
        for capability, item in coverage.items()
        if not item["sampled"]
    ]
    manifest = {
        "schema_version": "omnigibson_reasoning_tasks.v1",
        "episode_id": episode["episode_id"],
        "scene_id": episode["scene_id"],
        "model_input_contract": {
            "visible": [
                "ordered_rgb_views",
                "question_zh",
                "frame_convention_when_explicit_in_question",
            ],
            "hidden_supervision": ["operation_graph", "certificate", "oracle"],
            "forbidden_in_prompt": [
                "entity_ids",
                "coordinates",
                "depth",
                "instance_masks",
                "certificate",
            ],
        },
        "selection_policy": {
            "entity_language": (
                "globally unique category, otherwise category anchored to the earliest "
                "view where exactly one visible instance has that category"
            ),
            "oracle_region_names_in_prompt": False,
            "direction_margin_m": 0.4,
            "perspective_component_margin_m": 0.3,
            "metric_answer_tolerance_m": 0.2,
        },
        "entity_references": reference_metadata,
        "task_count": len(tasks),
        "coverage_status": "complete" if not gaps else "incomplete",
        "coverage": coverage,
        "gaps": gaps,
        "tasks": tasks,
    }
    output = output_path or bundle / "reasoning_tasks.json"
    write_json_atomic(output, manifest)
    return manifest
