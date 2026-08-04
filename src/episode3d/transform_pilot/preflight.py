"""Geometry-only census for Self-Rotation and Among-5 task families.

This stage deliberately emits no natural-language QA.  It counts independent
geometric facts after model-facing visual gates, keeping base facts separate
from counterfactual siblings and future surface paraphrases.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import tempfile
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from episode3d.bundles import Bundle, file_sha256, read_json, stable_id
from episode3d.compilers import (
    STRUCTURAL_LABELS,
    model_input_frame_is_admissible,
    recognizable_witnesses,
    visual_quality_policy,
)
from episode3d.language import entity_name
from episode3d.transform_pilot.geometry import (
    direction_measurement,
    horizontal_obb_radius,
    wrap_degrees,
    yaw_from_xyzw,
)
from episode3d.transform_pilot.schemas import TASK_SCHEMAS

SCHEMA_VERSION = "epispace.transform_preflight.v1"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _resolve(config_path: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (config_path.parent / candidate).resolve()


def _load_parent_index(release_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    episodes: dict[str, dict[str, Any]] = {}
    scene_splits: dict[str, str] = {}
    with (release_dir / "episodes.ir.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            episodes[str(row["episode_id"])] = row
            scene_id = str(row["scene_id"])
            split = str(row["split"])
            previous = scene_splits.setdefault(scene_id, split)
            if previous != split:
                raise ValueError(f"scene split conflict: {scene_id}: {previous} != {split}")
    return episodes, scene_splits


def _load_bundles(
    *,
    config_path: Path,
    config: dict[str, Any],
    parent_episodes: dict[str, dict[str, Any]],
) -> tuple[list[Bundle], list[dict[str, Any]], dict[str, Any]]:
    parent_config_path = _resolve(config_path, str(config["parent_pipeline_config"]))
    parent_config = read_json(parent_config_path)
    accepted = {str(value) for value in config["accepted_plan_statuses"]}
    wanted = {
        str(config["self_rotation"]["source_trajectory_class"]),
        str(config["among5"]["primary_source_trajectory_class"]),
        str(config["among5"]["auxiliary_source_trajectory_class"]),
    }
    bundles: list[Bundle] = []
    rejections: list[dict[str, Any]] = []
    source_stats: list[dict[str, Any]] = []
    for source in parent_config["sources"]:
        trajectory_class = str(source["trajectory_class"])
        if trajectory_class not in wanted:
            continue
        sweep_path = _resolve(parent_config_path, str(source["sweep_plan"]))
        sweep = read_json(sweep_path)
        counts: Counter[str] = Counter()
        loaded = 0
        for job in sweep["jobs"]:
            status = str(job.get("status", "missing"))
            counts[status] += 1
            if status not in accepted:
                continue
            try:
                bundle = Bundle(
                    Path(job["bundle"]),
                    trajectory_class=trajectory_class,
                    source_sweep=str(source["name"]),
                    job_status=status,
                )
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                rejections.append(
                    {
                        "scope": "bundle",
                        "item_id": str(job["job_id"]),
                        "reason_code": "bundle_validation_failed",
                        "detail": str(error),
                        "source": str(sweep_path),
                    }
                )
                continue
            if bundle.episode_id not in parent_episodes:
                rejections.append(
                    {
                        "scope": "bundle",
                        "item_id": bundle.episode_id,
                        "reason_code": "not_in_parent_compiled_release",
                        "detail": "strict source bundle has no parent IR/media/split binding",
                        "source": str(bundle.root),
                    }
                )
                continue
            bundles.append(bundle)
            loaded += 1
        source_stats.append(
            {
                "source_name": str(source["name"]),
                "trajectory_class": trajectory_class,
                "sweep_plan": str(sweep_path),
                "sweep_plan_sha256": file_sha256(sweep_path),
                "planned_jobs": len(sweep["jobs"]),
                "status_counts": dict(sorted(counts.items())),
                "parent_bound_bundles": loaded,
            }
        )
    bundles.sort(key=lambda item: (item.trajectory_class, item.scene_id, item.episode_id))
    return bundles, rejections, {
        "parent_pipeline_config": str(parent_config_path),
        "parent_pipeline_config_sha256": file_sha256(parent_config_path),
        "sources": source_stats,
    }


def _rgb_map(parent_episode: dict[str, Any]) -> dict[str, str]:
    return {str(row["view_id"]): str(row["rgb"]) for row in parent_episode["observations"]}


def _recognizable_by_view(bundle: Bundle) -> tuple[dict[str, tuple[str, ...]], dict[str, set[str]]]:
    witnesses = recognizable_witnesses(bundle)
    by_view = {view.view_id: set() for view in bundle.views}
    for entity_id, view_ids in witnesses.items():
        for view_id in view_ids:
            by_view[view_id].add(entity_id)
    return witnesses, by_view


def _surface_candidates(
    bundle: Bundle, by_view: dict[str, set[str]], view_ids: tuple[str, ...], surface: str
) -> set[str]:
    return {
        entity_id
        for view_id in view_ids
        for entity_id in by_view[view_id]
        if entity_name(bundle.entities[entity_id].label) == surface
    }


def _quality_score(bundle: Bundle, entity_id: str, view_id: str) -> tuple[float, ...]:
    stats = bundle.instance_visual_stats(entity_id, view_id)
    return (
        float(not stats["touches_image_border"]),
        float(min(stats["bbox_width_px"], stats["bbox_height_px"])),
        float(stats["visible_pixels"]),
        float(stats["mask_bbox_fill_ratio"]),
    )


def _best_witness(bundle: Bundle, entity_id: str, view_ids: tuple[str, ...]) -> str:
    candidates = [view_id for view_id in view_ids if entity_id in bundle.view_by_id[view_id].visible_entity_ids]
    if not candidates:
        raise ValueError("no candidate witness")
    return max(candidates, key=lambda view_id: _quality_score(bundle, entity_id, view_id))


def _turn_surface_programs(net_right_deg: int) -> list[list[int]]:
    sign = 1 if net_right_deg > 0 else -1
    amount = abs(net_right_deg)
    programs = [[net_right_deg]]
    if amount in {120, 180}:
        programs.append([sign * 60] * (amount // 60))
    return programs


def _balanced_take(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int, str], deque[dict[str, Any]]] = {}
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[
            (
                str(row["source_mode"]),
                abs(int(row["turn"]["net_right_deg"])),
                str(row["answer"]["label"]),
            )
        ].append(row)
    for key, values in grouped.items():
        values.sort(
            key=lambda row: (
                -float(row["answer"]["effective_margin_deg"]),
                -int(row["cue"]["visible_pixels"]),
                str(row["base_fact_id"]),
            )
        )
        buckets[key] = deque(values)
    selected: list[dict[str, Any]] = []
    while len(selected) < limit and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].popleft())
    return selected


def _compile_self_rotation(
    bundle: Bundle,
    *,
    parent_episode: dict[str, Any],
    task_config: dict[str, Any],
    direction_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    witnesses, by_view = _recognizable_by_view(bundle)
    rgb = _rgb_map(parent_episode)
    views = list(bundle.views)
    sequence_by_sign = {
        1: list(range(len(views))),
        -1: [0, *range(len(views) - 1, 0, -1)],
    }
    candidates: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    gate_counts: Counter[str] = Counter()
    min_distance = float(task_config["minimum_horizontal_distance_m"])
    min_center = float(direction_config["minimum_center_margin_deg"])
    min_effective = float(direction_config["minimum_obb_aware_margin_deg"])

    for raw_turns in task_config["ordered_turn_sequences_deg"]:
        turns = tuple(int(value) for value in raw_turns)
        if not turns or any(value == 0 for value in turns) or len({value > 0 for value in turns}) != 1:
            raise ValueError(f"invalid ordered turn sequence: {turns}")
        net_right = sum(turns)
        if net_right % 60 or abs(net_right) >= 360:
            raise ValueError(f"turn sequence is not aligned to T3 acquisition: {turns}")
        sign = 1 if net_right > 0 else -1
        step_count = abs(net_right) // 60
        order = sequence_by_sign[sign]
        for current_order_index in range(len(order) - step_count):
            current_index = order[current_order_index]
            target_index = order[current_order_index + step_count]
            current = views[current_index]
            target = views[target_index]
            current_yaw = yaw_from_xyzw(current.world_from_camera["rotation_xyzw"])
            target_yaw = yaw_from_xyzw(target.world_from_camera["rotation_xyzw"])
            hypothetical_yaw = current_yaw - math.radians(net_right)
            yaw_error = abs(wrap_degrees(math.degrees(target_yaw - hypothetical_yaw)))
            if yaw_error > 1.0:
                raise ValueError(
                    f"{bundle.root}: declared T3 turn convention disagrees with quaternion by {yaw_error:.3f}°"
                )
            for source_mode in task_config["source_modes"]:
                if source_mode == "view_local":
                    model_view_ids = (current.view_id,)
                    entity_ids = sorted(by_view[current.view_id])
                elif source_mode == "episodic_memory":
                    if current_order_index < 1:
                        continue
                    model_view_ids = tuple(views[index].view_id for index in order[: current_order_index + 1])
                    earlier = set().union(*(by_view[view_id] for view_id in model_view_ids[:-1]))
                    entity_ids = sorted(earlier - by_view[current.view_id])
                else:
                    raise ValueError(f"unsupported source_mode: {source_mode}")
                gate_counts["candidate_bindings"] += len(entity_ids)
                if target.view_id in model_view_ids:
                    raise ValueError("oracle target orientation leaked into model views")
                if not all(model_input_frame_is_admissible(bundle, view_id) for view_id in model_view_ids):
                    gate_counts["rejected_model_frame"] += len(entity_ids)
                    continue
                for entity_id in entity_ids:
                    entity = bundle.entities[entity_id]
                    surface = entity_name(entity.label)
                    if _surface_candidates(bundle, by_view, model_view_ids, surface) != {entity_id}:
                        gate_counts["rejected_surface_ambiguity"] += 1
                        continue
                    evidence_ids = tuple(
                        view_id for view_id in model_view_ids if view_id in witnesses[entity_id]
                    )
                    cue_view_id = _best_witness(bundle, entity_id, evidence_ids)
                    origin = current.world_from_camera["translation_m"]
                    point = entity.center_world_m
                    distance = math.hypot(point[0] - origin[0], point[1] - origin[1])
                    if distance < min_distance:
                        gate_counts["rejected_too_close"] += 1
                        continue
                    answer = direction_measurement(
                        point_xy=(point[0], point[1]),
                        origin_xy=(origin[0], origin[1]),
                        camera_yaw_rad=hypothetical_yaw,
                        horizontal_radius_m=horizontal_obb_radius(entity.extent_m),
                    )
                    if answer.center_margin_deg < min_center:
                        gate_counts["rejected_center_margin"] += 1
                        continue
                    if answer.effective_margin_deg < min_effective:
                        gate_counts["rejected_obb_margin"] += 1
                        continue
                    stats = bundle.instance_visual_stats(entity_id, cue_view_id)
                    cue_view = bundle.view_by_id[cue_view_id]
                    cue_yaw = yaw_from_xyzw(
                        cue_view.world_from_camera["rotation_xyzw"]
                    )
                    cue_origin = cue_view.world_from_camera["translation_m"]
                    cue_direction = direction_measurement(
                        point_xy=(point[0], point[1]),
                        origin_xy=(cue_origin[0], cue_origin[1]),
                        camera_yaw_rad=cue_yaw,
                        horizontal_radius_m=horizontal_obb_radius(entity.extent_m),
                    )
                    current_direction = direction_measurement(
                        point_xy=(point[0], point[1]),
                        origin_xy=(origin[0], origin[1]),
                        camera_yaw_rad=current_yaw,
                        horizontal_radius_m=horizontal_obb_radius(entity.extent_m),
                    )
                    fact_id = stable_id(
                        "transform-fact",
                        bundle.scene_id,
                        bundle.episode_id,
                        source_mode,
                        current.view_id,
                        target.view_id,
                        entity_id,
                    )
                    row = {
                        "schema_version": SCHEMA_VERSION,
                        "task_id": "self_rotation_query.v1",
                        "base_fact_id": fact_id,
                        "scene_id": bundle.scene_id,
                        "episode_id": bundle.episode_id,
                        "split": str(parent_episode["split"]),
                        "source_bundle": str(bundle.root),
                        "source_mode": source_mode,
                        "model_view_ids": list(model_view_ids),
                        "model_rgb": [rgb[view_id] for view_id in model_view_ids],
                        "cue": {
                            "entity_id": entity_id,
                            "category": surface,
                            "evidence_view_ids": list(evidence_ids),
                            "selected_cue_view_id": cue_view_id,
                            "visible_pixels": int(stats["visible_pixels"]),
                            "bbox_min_side_px": int(
                                min(stats["bbox_width_px"], stats["bbox_height_px"])
                            ),
                            # These two measurements turn the claim sheet from
                            # an answer-only certificate into an executable
                            # cue -> transform -> conclusion trace.  They are
                            # teacher/verifier facts; only their qualitative
                            # language rendering is model-visible.
                            "observed_direction": cue_direction.as_dict(),
                            "current_direction_before_query": current_direction.as_dict(),
                        },
                        "turn": {
                            "atomic_turns_deg": list(turns),
                            "net_right_deg": net_right,
                            "right_turn_positive": True,
                            "surface_programs_deg": _turn_surface_programs(net_right),
                        },
                        "answer": answer.as_dict(),
                        "oracle": {
                            "current_view_id": current.view_id,
                            "target_orientation_view_id": target.view_id,
                            "target_orientation_is_model_visible": False,
                            "quaternion_yaw_replay_error_deg": round(yaw_error, 6),
                            "entity_center_world_m": list(entity.center_world_m),
                            "entity_extent_m": list(entity.extent_m),
                        },
                        "program_minimality": {
                            "target_view_withheld": target.view_id not in model_view_ids,
                            "episodic_memory_required_by_construction": source_mode == "episodic_memory",
                            "entity_absent_from_current_view": entity_id not in by_view[current.view_id],
                        },
                    }
                    candidates.append(row)
                    gate_counts["accepted_before_balance"] += 1

    deduplicated = {str(row["base_fact_id"]): row for row in candidates}
    selected = _balanced_take(
        list(deduplicated.values()), int(task_config["maximum_base_facts_per_bundle"])
    )
    selected_ids = {str(row["base_fact_id"]) for row in selected}
    for row in deduplicated.values():
        if row["base_fact_id"] not in selected_ids:
            rejections.append(
                {
                    "scope": "self_rotation_fact",
                    "item_id": row["base_fact_id"],
                    "reason_code": "bundle_balance_cap",
                    "detail": "valid candidate retained only in census counts, not selected base facts",
                    "source": str(bundle.root),
                }
            )
    gate_counts["unique_before_balance"] = len(deduplicated)
    gate_counts["selected_base_facts"] = len(selected)
    return selected, rejections, dict(gate_counts)


def _view_graph_connected(
    selected_view_ids: tuple[str, ...],
    by_view: dict[str, set[str]],
    minimum_shared: int,
) -> tuple[bool, list[dict[str, Any]]]:
    edges: list[dict[str, Any]] = []
    adjacency = {view_id: set() for view_id in selected_view_ids}
    for left, right in itertools.combinations(selected_view_ids, 2):
        shared = sorted(by_view[left] & by_view[right])
        if len(shared) >= minimum_shared:
            adjacency[left].add(right)
            adjacency[right].add(left)
            edges.append({"view_a": left, "view_b": right, "shared_entity_ids": shared})
    visited = {selected_view_ids[0]}
    frontier = [selected_view_ids[0]]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current] - visited:
            visited.add(neighbor)
            frontier.append(neighbor)
    return len(visited) == len(selected_view_ids), edges


def _compile_among5(
    bundle: Bundle,
    *,
    parent_episode: dict[str, Any],
    task_config: dict[str, Any],
    direction_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    witnesses, by_view = _recognizable_by_view(bundle)
    rgb = _rgb_map(parent_episode)
    views = list(bundle.views)
    view0 = views[0]
    model_view_count = int(task_config["model_view_count"])
    minimum_anchor_views = int(task_config["minimum_anchor_witness_views"])
    min_shared = int(task_config["minimum_pairwise_shared_recognizable_entities"])
    min_distance = float(task_config["minimum_satellite_distance_m"])
    max_distance = float(task_config["maximum_satellite_distance_m"])
    min_center = float(direction_config["minimum_center_margin_deg"])
    min_effective = float(direction_config["minimum_obb_aware_margin_deg"])
    gate_counts: Counter[str] = Counter()
    rejections: list[dict[str, Any]] = []

    if bundle.trajectory_class == task_config["primary_source_trajectory_class"]:
        raw_focus = str(bundle.plan.get("focus_entity_id", ""))
        try:
            anchor_ids = [bundle.entity(raw_focus).entity_id]
        except ValueError:
            anchor_ids = []
    else:
        anchor_ids = sorted(by_view[view0.view_id])

    layout_candidates: list[dict[str, Any]] = []
    for anchor_id in anchor_ids:
        anchor = bundle.entities[anchor_id]
        if anchor.label in STRUCTURAL_LABELS:
            continue
        anchor_witnesses = tuple(witnesses.get(anchor_id, ()))
        if view0.view_id not in anchor_witnesses or len(anchor_witnesses) < minimum_anchor_views:
            gate_counts["rejected_anchor_witness_count"] += 1
            continue
        anchor_surface = entity_name(anchor.label)
        if _surface_candidates(bundle, by_view, anchor_witnesses, anchor_surface) != {anchor_id}:
            gate_counts["rejected_anchor_surface_ambiguity"] += 1
            continue
        other_anchor_views = [view_id for view_id in anchor_witnesses if view_id != view0.view_id]
        for tail in itertools.combinations(other_anchor_views, model_view_count - 1):
            selected_views = (view0.view_id, *tail)
            gate_counts["candidate_view_sets"] += 1
            if not all(model_input_frame_is_admissible(bundle, view_id) for view_id in selected_views):
                gate_counts["rejected_model_frame"] += 1
                continue
            connected, graph_edges = _view_graph_connected(selected_views, by_view, min_shared)
            if not connected:
                gate_counts["rejected_disconnected_view_graph"] += 1
                continue
            recognizable_union = set().union(*(by_view[view_id] for view_id in selected_views))
            origin = anchor.center_world_m
            frame_yaw = yaw_from_xyzw(view0.world_from_camera["rotation_xyzw"])
            by_direction: dict[str, list[tuple[tuple[float, ...], str, dict[str, Any]]]] = defaultdict(list)
            for entity_id in sorted(recognizable_union - {anchor_id}):
                entity = bundle.entities[entity_id]
                if entity.label in STRUCTURAL_LABELS:
                    continue
                surface = entity_name(entity.label)
                if _surface_candidates(bundle, by_view, selected_views, surface) != {entity_id}:
                    continue
                dx = entity.center_world_m[0] - origin[0]
                dy = entity.center_world_m[1] - origin[1]
                distance = math.hypot(dx, dy)
                if not min_distance <= distance <= max_distance:
                    continue
                measurement = direction_measurement(
                    point_xy=(entity.center_world_m[0], entity.center_world_m[1]),
                    origin_xy=(origin[0], origin[1]),
                    camera_yaw_rad=frame_yaw,
                    horizontal_radius_m=(
                        horizontal_obb_radius(anchor.extent_m)
                        + horizontal_obb_radius(entity.extent_m)
                    ),
                )
                if measurement.center_margin_deg < min_center or measurement.effective_margin_deg < min_effective:
                    continue
                evidence = tuple(view_id for view_id in selected_views if view_id in witnesses[entity_id])
                best_view = _best_witness(bundle, entity_id, evidence)
                stats = bundle.instance_visual_stats(entity_id, best_view)
                score = (
                    measurement.effective_margin_deg,
                    min(stats["bbox_width_px"], stats["bbox_height_px"]),
                    stats["visible_pixels"],
                    -distance,
                )
                by_direction[measurement.label].append(
                    (
                        score,
                        entity_id,
                        {
                            "entity_id": entity_id,
                            "category": surface,
                            "evidence_view_ids": list(evidence),
                            "selected_cue_view_id": best_view,
                            "relation_to_anchor": measurement.as_dict(),
                        },
                    )
                )
            if set(by_direction) != {"front", "right", "back", "left"}:
                gate_counts["rejected_missing_layout_sector"] += 1
                continue
            satellites: list[dict[str, Any]] = []
            used_surfaces = {anchor_surface}
            for direction in ("front", "right", "back", "left"):
                chosen = None
                for _, _, candidate in sorted(by_direction[direction], reverse=True):
                    if candidate["category"] not in used_surfaces:
                        chosen = candidate
                        break
                if chosen is None:
                    break
                satellites.append(chosen)
                used_surfaces.add(str(chosen["category"]))
            if len(satellites) != 4:
                gate_counts["rejected_nonunique_five_object_surface"] += 1
                continue
            satellite_ids = {str(item["entity_id"]) for item in satellites}
            if any(not (by_view[view_id] & satellite_ids) for view_id in selected_views):
                gate_counts["rejected_empty_satellite_view"] += 1
                continue
            decisive_views = {
                entity_id: [
                    view_id
                    for view_id in selected_views
                    if entity_id in by_view[view_id]
                    and not any(
                        entity_id in by_view[other]
                        for other in selected_views
                        if other != view_id
                    )
                ]
                for entity_id in satellite_ids
            }
            minimum_margin = min(
                float(item["relation_to_anchor"]["effective_margin_deg"])
                for item in satellites
            )
            record = {
                "schema_version": SCHEMA_VERSION,
                "task_id": "among5_layout.v1",
                "base_layout_id": stable_id(
                    "transform-layout",
                    bundle.scene_id,
                    bundle.episode_id,
                    anchor_id,
                    *selected_views,
                    *sorted(satellite_ids),
                ),
                "scene_id": bundle.scene_id,
                "episode_id": bundle.episode_id,
                "split": str(parent_episode["split"]),
                "source_bundle": str(bundle.root),
                "source_trajectory_class": bundle.trajectory_class,
                "protocol_alignment": (
                    "primary_orbit_among5"
                    if bundle.trajectory_class == task_config["primary_source_trajectory_class"]
                    else "auxiliary_coverage_among5"
                ),
                "model_view_ids": list(selected_views),
                "model_rgb": [rgb[view_id] for view_id in selected_views],
                "anchor": {
                    "entity_id": anchor_id,
                    "category": anchor_surface,
                    "evidence_view_ids": list(selected_views),
                },
                "satellites": satellites,
                "answer_frame": {
                    "frame_id": f"camera@{view0.view_id}",
                    "+X": "first-view right",
                    "+Y": "first-view forward",
                    "direction_order": ["front", "right", "back", "left"],
                },
                "view_registration_graph": {
                    "minimum_shared_recognizable_entities": min_shared,
                    "edges": graph_edges,
                    "connected": True,
                },
                "program_minimality": {
                    "decisive_view_ids_by_satellite": decisive_views,
                    "deletion_is_only_an_ablation": True,
                    "unknown_label_authorized": False,
                },
                "oracle": {
                    "canonical_layout_is_model_visible": False,
                    "anchor_center_world_m": list(anchor.center_world_m),
                    "minimum_obb_aware_direction_margin_deg": round(minimum_margin, 6),
                },
            }
            layout_candidates.append(record)
            gate_counts["accepted_before_bundle_cap"] += 1

    layout_candidates.sort(
        key=lambda row: (
            -float(row["oracle"]["minimum_obb_aware_direction_margin_deg"]),
            str(row["base_layout_id"]),
        )
    )
    selected = layout_candidates[: int(task_config["maximum_layouts_per_bundle"])]
    gate_counts["selected_base_layouts"] = len(selected)
    return selected, rejections, dict(gate_counts)


def _counts_by(rows: list[dict[str, Any]], *keys: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        label = "/".join(str(row[key]) for key in keys)
        counts[label] += 1
    return dict(sorted(counts.items()))


def _readiness(
    *,
    config: dict[str, Any],
    self_rows: list[dict[str, Any]],
    among_rows: list[dict[str, Any]],
    parent_audit: dict[str, Any],
) -> dict[str, Any]:
    policy = config["readiness"]
    self_scenes = {row["scene_id"] for row in self_rows}
    primary_among = [row for row in among_rows if row["protocol_alignment"] == "primary_orbit_among5"]
    primary_among_scenes = {row["scene_id"] for row in primary_among}
    gates = {
        "self_rotation_scene_floor": {
            "measured": len(self_scenes),
            "required": int(policy["minimum_self_rotation_scenes"]),
        },
        "self_rotation_base_fact_floor": {
            "measured": len(self_rows),
            "required": int(policy["minimum_self_rotation_base_facts"]),
        },
        "primary_among5_scene_floor": {
            "measured": len(primary_among_scenes),
            "required": int(policy["minimum_primary_among5_scenes"]),
        },
        "primary_among5_base_layout_floor": {
            "measured": len(primary_among),
            "required": int(policy["minimum_primary_among5_base_layouts"]),
        },
        "parent_semantic_visual_audit": {
            "measured": bool(parent_audit.get("decision", {}).get("semantic_visual_audit_gate")),
            "required": True,
        },
        "new_task_manual_audit": {
            "measured": False,
            "required": True,
        },
    }
    for value in gates.values():
        value["passed"] = value["measured"] >= value["required"] if isinstance(value["required"], int) else value["measured"] is value["required"]
    return {
        "status": "training_ready" if all(value["passed"] for value in gates.values()) else "review_required",
        "gates": gates,
        "counting_rule": (
            "Only unique geometry-backed base facts/layouts count toward scale; "
            "counterfactual siblings and language surfaces are reported separately."
        ),
    }


def build_preflight(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = read_json(config_path)
    if config.get("schema_version") != "epispace.transform_pilot_config.v1":
        raise ValueError("unsupported Transform Pilot config schema")
    output_dir = _resolve(config_path, str(config["output_dir"]))
    preflight_dir = output_dir / "preflight"
    parent_release = _resolve(config_path, str(config["parent_release_dir"]))
    parent_manifest_path = parent_release / "release_manifest.json"
    parent_audit_path = parent_release / "semantic_visual_audit" / "result.json"
    parent_manifest = read_json(parent_manifest_path)
    parent_audit = read_json(parent_audit_path)
    parent_episodes, _ = _load_parent_index(parent_release)
    bundles, rejections, source_binding = _load_bundles(
        config_path=config_path, config=config, parent_episodes=parent_episodes
    )

    self_rows: list[dict[str, Any]] = []
    among_rows: list[dict[str, Any]] = []
    per_bundle: list[dict[str, Any]] = []
    for bundle in bundles:
        parent_episode = parent_episodes[bundle.episode_id]
        if bundle.trajectory_class == config["self_rotation"]["source_trajectory_class"]:
            rows, rejected, gates = _compile_self_rotation(
                bundle,
                parent_episode=parent_episode,
                task_config=config["self_rotation"],
                direction_config=config["direction_contract"],
            )
            self_rows.extend(rows)
            rejections.extend(rejected)
            per_bundle.append(
                {
                    "episode_id": bundle.episode_id,
                    "scene_id": bundle.scene_id,
                    "trajectory_class": bundle.trajectory_class,
                    "gate_counts": gates,
                }
            )
        if bundle.trajectory_class in {
            config["among5"]["primary_source_trajectory_class"],
            config["among5"]["auxiliary_source_trajectory_class"],
        }:
            rows, rejected, gates = _compile_among5(
                bundle,
                parent_episode=parent_episode,
                task_config=config["among5"],
                direction_config=config["direction_contract"],
            )
            among_rows.extend(rows)
            rejections.extend(rejected)
            if not rows:
                rejections.append(
                    {
                        "scope": "among5_bundle",
                        "item_id": bundle.episode_id,
                        "reason_code": "no_among5_layout_after_gates",
                        "detail": json.dumps(gates, ensure_ascii=False, sort_keys=True),
                        "source": str(bundle.root),
                    }
                )
            per_bundle.append(
                {
                    "episode_id": bundle.episode_id,
                    "scene_id": bundle.scene_id,
                    "trajectory_class": bundle.trajectory_class,
                    "gate_counts": gates,
                }
            )

    self_rows.sort(key=lambda row: str(row["base_fact_id"]))
    among_rows.sort(key=lambda row: str(row["base_layout_id"]))
    rejections.sort(key=lambda row: (str(row["scope"]), str(row["item_id"]), str(row["reason_code"])))
    _write_jsonl(preflight_dir / "self_rotation.base_facts.jsonl", self_rows)
    _write_jsonl(preflight_dir / "among5.base_layouts.jsonl", among_rows)
    _write_jsonl(preflight_dir / "rejections.jsonl", rejections)
    _write_json(preflight_dir / "per_bundle_census.json", per_bundle)

    readiness = _readiness(
        config=config,
        self_rows=self_rows,
        among_rows=among_rows,
        parent_audit=parent_audit,
    )
    primary_among = [row for row in among_rows if row["protocol_alignment"] == "primary_orbit_among5"]
    auxiliary_among = [row for row in among_rows if row["protocol_alignment"] != "primary_orbit_among5"]
    surface_program_count = sum(len(row["turn"]["surface_programs_deg"]) for row in self_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": str(config["dataset_id"]),
        "status": readiness["status"],
        "stage": "geometry_only_preflight",
        "training_exports_present": False,
        "task_schemas": TASK_SCHEMAS,
        "counts": {
            "parent_bound_bundles": len(bundles),
            "unique_scenes": len({bundle.scene_id for bundle in bundles}),
            "self_rotation": {
                "unique_scenes": len({row["scene_id"] for row in self_rows}),
                "base_geometric_facts": len(self_rows),
                "counterfactual_or_program_surface_records_if_expanded": surface_program_count,
                "language_surface_variants": 0,
                "by_split": _counts_by(self_rows, "split"),
                "by_source_mode": _counts_by(self_rows, "source_mode"),
                "by_answer": dict(
                    sorted(Counter(str(row["answer"]["label"]) for row in self_rows).items())
                ),
            },
            "among5": {
                "primary_orbit_scenes": len({row["scene_id"] for row in primary_among}),
                "primary_orbit_base_layouts": len(primary_among),
                "auxiliary_coverage_scenes": len({row["scene_id"] for row in auxiliary_among}),
                "auxiliary_coverage_base_layouts": len(auxiliary_among),
                "language_surface_variants": 0,
                "by_split_and_alignment": _counts_by(among_rows, "split", "protocol_alignment"),
            },
            "rejections": len(rejections),
            "rejection_reasons": dict(sorted(Counter(row["reason_code"] for row in rejections).items())),
        },
        "readiness": readiness,
        "provenance": {
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "parent_release_manifest": str(parent_manifest_path),
            "parent_release_manifest_sha256": file_sha256(parent_manifest_path),
            "parent_release_status": parent_manifest.get("status"),
            "parent_semantic_visual_audit": str(parent_audit_path),
            "parent_semantic_visual_audit_sha256": file_sha256(parent_audit_path),
            "parent_semantic_visual_audit_gate": parent_audit.get("decision", {}).get(
                "semantic_visual_audit_gate"
            ),
            "source_binding": source_binding,
            "visual_quality_policy": visual_quality_policy(),
        },
        "artifacts": {},
    }
    for filename in (
        "self_rotation.base_facts.jsonl",
        "among5.base_layouts.jsonl",
        "rejections.jsonl",
        "per_bundle_census.json",
    ):
        path = preflight_dir / filename
        summary["artifacts"][filename] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    canonical = json.dumps(summary, ensure_ascii=False, sort_keys=True).encode()
    summary["content_digest"] = hashlib.sha256(canonical).hexdigest()
    _write_json(preflight_dir / "census.json", summary)
    return summary
