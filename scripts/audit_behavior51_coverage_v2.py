#!/usr/bin/env python3
"""Audit a merged coverage snapshot and write a non-destructive v2 overlay.

The source coverage directory is immutable input.  This tool never edits its
plan, status, bundles, groups, or merge ledger.  It emits dispositions and a
quarantine proposal that can later drive v2 credit recomputation/backfill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from spatial_episode.scriptgen.behavior import RenderSceneView
from spatial_episode.scriptgen.compiler import CapabilityCompiler
from spatial_episode.scriptgen.library import SCRIPT_LIBRARY
from spatial_episode.scriptgen.qa_dataset import (
    DEFAULT_EXCLUDED_CAPABILITIES,
    _compile_prefixes,
    _family_sources,
)
from spatial_episode.scriptgen.standards import STD_V1


SCHEMA = "epispace.behavior51_v2_partial_audit.v2"
SELF_MOTION = frozenset(
    {
        "self_motion_update",
        "self_motion_update_multi_turn",
        "self_motion_update_pure_rotation",
        "self_motion_update_pure_translation",
        "self_motion_update_occluded",
    }
)


def _failure_root(message: str | None) -> str | None:
    """Recover the useful renderer cause hidden by coarse status reasons."""
    if not message:
        return None
    if message.startswith("auxiliary camera failed physics clearance:"):
        return "auxiliary_camera_physics_clearance"
    if message.startswith("scripted path failed traversability clearance:"):
        return "scripted_path_traversability_clearance"
    if "Directory not empty" in message and ".staging." in message:
        return "atomic_bundle_publish_race"
    if "CUDA out of memory" in message:
        return "cuda_out_of_memory"
    return "other_renderer_failure"


def _certificate_root(certificate: dict[str, Any]) -> str:
    reason = str(certificate.get("reason") or "")
    if reason.startswith("frame_var_unresolvable:"):
        return reason
    if reason.startswith("clause:"):
        return reason
    if certificate.get("status") == "answerable" and certificate.get("mismatch") is None:
        return "primary_answerable"
    return reason or str(certificate.get("mismatch") or "primary_not_answerable")


def _repair_for_rejection(
    *,
    root_cause: str,
    primary_replay: dict[str, Any] | None,
) -> tuple[str, bool]:
    """Return the v2 repair action and whether unchanged pixels can be reused."""
    if root_cause == "auxiliary_camera_physics_clearance":
        return "rebind_after_aux_camera_clearance_prefilter", False
    if root_cause == "scripted_path_traversability_clearance":
        return "replan_with_authoritative_traversability", False
    if root_cause == "family_delay_turn_segments":
        return "fix_delay_variant_then_recompile_bundle", True
    if root_cause == "family_permute_unavailable":
        return "represent_unavailable_variant_then_recompile_bundle", True
    if primary_replay is not None:
        if primary_replay["status"] == "answerable" and primary_replay.get("mismatch") is None:
            return "recompile_existing_success_bundle", True
        return "replan_from_replayed_authority_failure", False
    if root_cause in {
        "render_timeout",
        "renderer_crash",
        "operator_stop",
        "cuda_out_of_memory",
    }:
        return "retry_same_candidate_after_runtime_cleanup", False
    if root_cause == "atomic_bundle_publish_race":
        return "revalidate_existing_bundle_or_retry_after_runtime_cleanup", True
    if root_cause == "frame_var_unresolvable:t_seen":
        return "replan_for_rendered_target_sighting", False
    if root_cause == "frame_var_unresolvable:t_gone":
        return "replan_for_rendered_disappearance", False
    if root_cause == "clause:turned":
        return "replan_with_post_sighting_turn_reserve", False
    if root_cause == "clause:gap":
        return "replan_with_post_disappearance_gap_reserve", False
    if root_cause == "clause:gone":
        return "replan_for_decisive_rendered_absence", False
    if root_cause == "clause:landmark_evidence":
        return "replan_survey_for_landmark_visibility", False
    if root_cause == "clause:no_single_frame_shortcut":
        return "replan_survey_without_covisibility_shortcut", False
    return "inspect_then_replan", False


def _audit_rejected_candidates(
    manifest: dict[str, Any],
    status: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audit every terminal rejection without mutating the canonical v1 tree."""
    scenes = {scene["scene_key"]: scene for scene in manifest["scenes"]}
    status_by_candidate = {
        candidate_id: candidate_status
        for cell in status["cells"].values()
        for candidate_id, candidate_status in cell.get("candidate_statuses", {}).items()
    }
    rows: list[dict[str, Any]] = []
    for cell in manifest["cells"]:
        for candidate in cell.get("candidates", ()):
            candidate_status = status_by_candidate[candidate["candidate_id"]]
            if candidate_status.get("status") != "rejected":
                continue
            bundle = Path(candidate["bundle"])
            group = Path(candidate["group"])
            render_report_path = bundle / "render_report.json"
            failure_report_path = bundle / "failure_report.json"
            certificate_path = group / "primary.certificate.json"
            failure_type = None
            failure_message = None
            if failure_report_path.is_file():
                failure = _read(failure_report_path).get("error") or {}
                failure_type = failure.get("type")
                failure_message = failure.get("message")
            renderer_root = _failure_root(failure_message)
            certificate = _read(certificate_path) if certificate_path.is_file() else None
            primary_replay = None
            # A renderer may publish a complete bundle and then hang during
            # shutdown. Replaying the primary authority is read-only and tells
            # us whether v2 can recover the pixels instead of rendering again.
            if render_report_path.is_file() and certificate is None:
                plan = _read(Path(candidate["plan_record"]))
                view = RenderSceneView.from_bundle(
                    bundle,
                    STD_V1,
                    scene_ir=Path(scenes[cell["scene_key"]]["scene_ir"]),
                )
                replay = CapabilityCompiler(
                    SCRIPT_LIBRARY[cell["capability"]], STD_V1
                ).compile(
                    view,
                    dict(plan["binding"]),
                    geometry_plan=plan,
                    with_essential=True,
                )
                primary_replay = replay.model_dump(mode="json")

            status_reason = str(candidate_status.get("reason") or "")
            if "variant delay:" in status_reason and "clause:turn_segments" in status_reason:
                root_cause = "family_delay_turn_segments"
                stage = "family_packaging"
            elif "no candidate sequences" in status_reason:
                root_cause = "family_permute_unavailable"
                stage = "family_packaging"
            elif certificate is not None:
                root_cause = _certificate_root(certificate)
                stage = (
                    "family_packaging"
                    if root_cause == "primary_answerable"
                    else "primary_authority"
                )
            elif primary_replay is not None:
                replay_root = _certificate_root(primary_replay)
                root_cause = (
                    "completed_bundle_primary_answerable"
                    if replay_root == "primary_answerable"
                    else replay_root
                )
                stage = "postrender_interrupted"
            elif renderer_root is not None:
                root_cause = renderer_root
                stage = "acquisition_preflight" if renderer_root in {
                    "auxiliary_camera_physics_clearance",
                    "scripted_path_traversability_clearance",
                } else "render_runtime"
            elif status_reason.startswith("render_timeout"):
                root_cause = "render_timeout"
                stage = "render_runtime"
            elif status_reason.startswith("renderer_exit"):
                root_cause = "renderer_crash"
                stage = "render_runtime"
            elif status_reason.startswith("manual_stop"):
                root_cause = "operator_stop"
                stage = "render_runtime"
            else:
                root_cause = status_reason or "unknown_rejection"
                stage = "unknown"

            repair_action, reuse_pixels = _repair_for_rejection(
                root_cause=root_cause,
                primary_replay=primary_replay,
            )
            failed_outcomes = []
            authority = certificate or primary_replay
            authority_diagnostics = None
            if authority is not None:
                failed_outcomes = [
                    outcome
                    for outcome in authority.get("clause_outcomes", ())
                    if outcome.get("holds") is not True
                ]
                visibility = authority.get("target_visibility") or []
                state_counts = Counter(row.get("tristate") for row in visibility)
                authority_diagnostics = {
                    "status": authority.get("status"),
                    "reason": authority.get("reason"),
                    "frame_vars": authority.get("frame_vars"),
                    "target_visibility_state_counts": dict(
                        sorted(
                            (str(key), value)
                            for key, value in state_counts.items()
                        )
                    ),
                    "maximum_target_pixels": (
                        max(float(row.get("value", 0.0)) for row in visibility)
                        if visibility
                        else None
                    ),
                    "terminal_target_state": (
                        visibility[-1].get("tristate") if visibility else None
                    ),
                    "terminal_target_pixels": (
                        float(visibility[-1].get("value", 0.0)) if visibility else None
                    ),
                }
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "cell_id": cell["cell_id"],
                    "scene_key": cell["scene_key"],
                    "source_capability": cell["capability"],
                    "attempt_index": candidate["attempt_index"],
                    "status_reason": candidate_status.get("reason"),
                    "failure_stage": stage,
                    "root_cause": root_cause,
                    "repair_action": repair_action,
                    "reuse_existing_pixels": reuse_pixels,
                    "artifacts": {
                        "render_report": render_report_path.is_file(),
                        "failure_report": failure_report_path.is_file(),
                        "primary_certificate": certificate_path.is_file(),
                        "bundle": str(bundle),
                    },
                    "renderer_failure": {
                        "type": failure_type,
                        "message": failure_message,
                        "normalized": renderer_root,
                    },
                    "primary_replay": (
                        {
                            "status": primary_replay["status"],
                            "reason": primary_replay.get("reason"),
                            "mismatch": primary_replay.get("mismatch"),
                            "answer": primary_replay.get("answer"),
                        }
                        if primary_replay is not None
                        else None
                    ),
                    "authority_diagnostics": authority_diagnostics,
                    "failed_clause_outcomes": failed_outcomes,
                }
            )

    root_counts = Counter(row["root_cause"] for row in rows)
    stage_counts = Counter(row["failure_stage"] for row in rows)
    action_counts = Counter(row["repair_action"] for row in rows)
    capability_counts = Counter(row["source_capability"] for row in rows)
    scene_counts = Counter(row["scene_key"] for row in rows)
    status_reason_counts = Counter(str(row["status_reason"]) for row in rows)
    renderer_failure_counts = Counter(
        row["renderer_failure"]["normalized"]
        for row in rows
        if row["renderer_failure"]["normalized"] is not None
    )
    action_scope = {
        action: {
            "candidate_count": count,
            "cell_count": len(
                {
                    row["cell_id"]
                    for row in rows
                    if row["repair_action"] == action
                }
            ),
            "scene_count": len(
                {
                    row["scene_key"]
                    for row in rows
                    if row["repair_action"] == action
                }
            ),
        }
        for action, count in sorted(action_counts.items())
    }
    t_seen_pixels = [
        row["authority_diagnostics"]["maximum_target_pixels"]
        for row in rows
        if row["root_cause"] == "frame_var_unresolvable:t_seen"
        and row["authority_diagnostics"] is not None
    ]
    turned_values = [
        float(outcome["witness"]["cum_turn_deg"])
        for row in rows
        if row["root_cause"] == "clause:turned"
        for outcome in row["failed_clause_outcomes"]
        if outcome.get("predicate") == "cum_turn_between"
    ]
    t_gone_terminal = Counter(
        row["authority_diagnostics"]["terminal_target_state"]
        for row in rows
        if row["root_cause"] == "frame_var_unresolvable:t_gone"
        and row["authority_diagnostics"] is not None
    )
    summary = {
        "candidate_count": len(rows),
        "by_failure_stage": dict(sorted(stage_counts.items())),
        "by_root_cause": dict(sorted(root_counts.items())),
        "by_repair_action": dict(sorted(action_counts.items())),
        "repair_action_scope": action_scope,
        "by_capability": dict(sorted(capability_counts.items())),
        "by_scene": dict(sorted(scene_counts.items())),
        "by_status_reason": dict(sorted(status_reason_counts.items())),
        "renderer_failure_reports": dict(sorted(renderer_failure_counts.items())),
        "render_success_bundle_count": sum(
            row["artifacts"]["render_report"] for row in rows
        ),
        "reusable_pixel_candidate_count": sum(
            row["reuse_existing_pixels"] for row in rows
        ),
        "replayed_primary": dict(
            sorted(
                Counter(
                    row["primary_replay"]["status"]
                    for row in rows
                    if row["primary_replay"] is not None
                ).items()
            )
        ),
        "authority_diagnostics": {
            "t_seen": {
                "count": len(t_seen_pixels),
                "zero_target_pixels": sum(value == 0.0 for value in t_seen_pixels),
                "positive_but_below_visible_threshold": sum(
                    value > 0.0 for value in t_seen_pixels
                ),
            },
            "turned": {
                "count": len(turned_values),
                "minimum_deg": min(turned_values) if turned_values else None,
                "median_deg": statistics.median(turned_values) if turned_values else None,
                "maximum_deg": max(turned_values) if turned_values else None,
                "below_80_deg": sum(value < 80.0 for value in turned_values),
                "above_200_deg": sum(value > 200.0 for value in turned_values),
            },
            "t_gone_terminal_state": dict(sorted(t_gone_terminal.items())),
        },
    }
    return rows, summary


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tier(capability: str) -> str:
    if capability.startswith("reference_frame_"):
        return "P2"
    if capability.startswith("cross_view_"):
        return "P3"
    return "P1"


def _angle_delta(left: float, right: float) -> float:
    return abs((right - left + 180.0) % 360.0 - 180.0)


def _source_family(group_path: Path, capability: str) -> tuple[dict[str, Any], dict[str, Any]]:
    group = _read(group_path)
    entry = next(
        (
            item
            for item in group.get("questions", ())
            if item.get("capability") == capability and item.get("family")
        ),
        None,
    )
    if entry is None:
        raise ValueError(f"missing source family: {group_path} {capability}")
    return group, _read(group_path.parent / entry["family"])


def _canonical(family: dict[str, Any]) -> dict[str, Any]:
    return next(episode for episode in family["episodes"] if episode["kind"] == "canonical")


def _streaming_audit(
    source_root: Path,
    dataset: dict[str, Any],
) -> dict[str, Any]:
    group_capability = {
        Path(episode["group"]).resolve(): episode["source_capability"]
        for episode in dataset["episodes"]
        if episode["source_capability"] in SELF_MOTION
    }
    sources = list(
        _family_sources(
            source_root,
            DEFAULT_EXCLUDED_CAPABILITIES,
            group_paths=tuple(sorted(group_capability)),
        )
    )
    by_group: dict[Path, list[Any]] = defaultdict(list)
    for source in sources:
        by_group[source.group_path].append(source)
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    affected: Counter[str] = Counter()
    old_turns: Counter[str] = Counter()
    consecutive_turns: Counter[str] = Counter()
    early: Counter[str] = Counter()
    skips: Counter[str] = Counter()
    missed = 0
    for group_path, rows in sorted(by_group.items(), key=lambda item: str(item[0])):
        first = rows[0]
        source_capability = group_capability[group_path]
        view = RenderSceneView.from_bundle(
            Path(first.group.trajectory.bundle),
            STD_V1,
            scene_ir=Path(first.group.trajectory.scene_ir),
        )
        binding = dict(first.plan["binding"])
        group_events: list[tuple[Any, Any]] = []
        for source in sorted(rows, key=lambda value: value.family.capability):
            states = _compile_prefixes(
                view,
                source.family,
                binding,
                stop_after_first_answerable=False,
            )
            seen: set[str] = set()
            last_label: str | None = None
            old = []
            consecutive = []
            for state in states:
                if (
                    state.prefix_length < 2
                    or state.certificate.status != "answerable"
                    or state.label is None
                ):
                    continue
                if state.label not in seen:
                    seen.add(state.label)
                    old.append(state)
                if state.label != last_label:
                    last_label = state.label
                    consecutive.append(state)
            group_events.extend((source, state) for state in old)
            capability = source.family.capability
            old_turns[capability] += len(old)
            consecutive_turns[capability] += len(consecutive)
            if len(consecutive) > len(old):
                affected[capability] += 1
                missed += len(consecutive) - len(old)
            if capability == source_capability and not any(
                state.prefix_length == view.frame_count for state in old
            ):
                by_source[source_capability]["missing_source_endpoint"] += 1
            if capability == "path_integration_magnitude" and old:
                first_event = old[0]
                if first_event.prefix_length == 2:
                    left, right = view.poses[:2]
                    if (
                        math.hypot(right.x - left.x, right.y - left.y) < 1e-6
                        and _angle_delta(left.yaw_deg, right.yaw_deg) < 1e-6
                    ):
                        early[source_capability] += 1
        if len(group_events) < 2:
            skips[source_capability] += 1
        if not group_events or max(state.prefix_length for _, state in group_events) < view.frame_count:
            by_source[source_capability]["tail_not_released"] += 1
        by_source[source_capability]["episodes"] += 1
    return {
        "source_episode_count": len(by_group),
        "by_source_capability": {
            key: dict(value) for key, value in sorted(by_source.items())
        },
        "transition_affected_family_count": sum(affected.values()),
        "transition_affected_families": dict(sorted(affected.items())),
        "transition_missed_turn_count": missed,
        "old_turns_by_capability": dict(sorted(old_turns.items())),
        "consecutive_dedup_turns_by_capability": dict(sorted(consecutive_turns.items())),
        "early_identical_pose_path_magnitude_by_source": dict(sorted(early.items())),
        "fewer_than_two_turn_groups": dict(sorted(skips.items())),
    }


def audit(source_root: Path, output_root: Path, *, prefix_audit: bool) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    manifest_path = source_root / "coverage.plan.json"
    status_path = source_root / "coverage.status.json"
    dataset_path = source_root / "dataset.json"
    ledger_path = source_root / "coverage.shard_merge.json"
    manifest = _read(manifest_path)
    status = _read(status_path)
    dataset = _read(dataset_path)
    ledger = _read(ledger_path)
    if dataset.get("episode_count") != len(status.get("episodes", {})):
        raise ValueError("dataset.json is stale relative to coverage.status.json")
    if _sha256(manifest_path) != ledger["current_manifest_sha256"]:
        raise ValueError("coverage plan differs from merge ledger")
    if _sha256(status_path) != ledger["current_status_sha256"]:
        raise ValueError("coverage status differs from merge ledger")
    partial_snapshot = len(ledger["applied_shards"]) < 2

    candidates = {
        candidate["candidate_id"]: candidate
        for cell in manifest["cells"]
        for candidate in cell.get("candidates", ())
    }
    source_counts: Counter[str] = Counter()
    source_capabilities: Counter[str] = Counter()
    source_labels: dict[str, Counter[str]] = defaultdict(Counter)
    family_counts: Counter[str] = Counter()
    raw_counts: Counter[tuple[str, str, str]] = Counter()
    p2_margins: list[float] = []
    dispositions: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    bad_npz: list[dict[str, Any]] = []
    rejected_candidates, rejected_summary = _audit_rejected_candidates(manifest, status)

    for episode in dataset["episodes"]:
        episode_id = episode["episode_id"]
        capability = episode["source_capability"]
        tier = _tier(capability)
        source_counts[tier] += 1
        source_capabilities[capability] += 1
        candidate = candidates[episode_id]
        plan = _read(Path(candidate["plan_record"]))
        poses = plan["poses"]
        path_length = sum(
            math.hypot(right["x"] - left["x"], right["y"] - left["y"])
            for left, right in zip(poses, poses[1:])
        )
        endpoint_displacement = math.hypot(
            poses[-1]["x"] - poses[0]["x"],
            poses[-1]["y"] - poses[0]["y"],
        )
        cumulative_turn = sum(
            _angle_delta(left["yaw_deg"], right["yaw_deg"])
            for left, right in zip(poses, poses[1:])
        )
        group_path = Path(episode["group"])
        group, family = _source_family(group_path, capability)
        canonical = _canonical(family)
        source_labels[capability][canonical["label"]] += 1
        visibility = canonical["certificate"].get("target_visibility") or []
        visible_frames = sum(row.get("tristate") == "visible" for row in visibility)

        for question in group["questions"]:
            if not question.get("family"):
                continue
            attached = _read(group_path.parent / question["family"])
            attached_tier = _tier(attached["capability"])
            family_counts[attached_tier] += 1
            for variant in attached["episodes"]:
                raw_counts[
                    (attached_tier, variant["kind"], variant["certificate"]["status"])
                ] += 1
            if attached["capability"].startswith("reference_frame_transform"):
                attached_canonical = _canonical(attached)
                outcome = next(
                    row
                    for row in attached_canonical["certificate"]["clause_outcomes"]
                    if row["predicate"] == "imagined_curve_sector_margins_ge"
                )
                p2_margins.append(float(outcome["witness"]["minimum_margin_deg"]))

        reasons: list[str] = []
        if capability == "self_motion_update" and path_length < 1.0:
            reasons.append("generic_path_lt_1m")
        if capability == "self_motion_update_multi_turn" and path_length < 2.0:
            reasons.append("multi_turn_path_lt_2m")
        if capability == "self_motion_update_occluded":
            reasons.append("occluded_terminal_definition")
        if capability in SELF_MOTION and visible_frames < 2:
            reasons.append("target_visible_fewer_than_2_frames")

        if reasons:
            disposition = "quarantine_replan_rerender"
        elif tier == "P2":
            disposition = "hold_qa_v2_policy"
        elif capability in {
            "self_motion_update_pure_rotation",
            "self_motion_update_pure_translation",
        }:
            disposition = "keep_hold_until_label_balanced"
        else:
            disposition = "keep_recompile_qa_v2"
        row = {
            "episode_id": episode_id,
            "tier": tier,
            "scene_key": episode["scene_key"],
            "source_capability": capability,
            "label": canonical["label"],
            "disposition": disposition,
            "reasons": reasons,
            "metrics": {
                "frame_count": len(poses),
                "path_length_m": round(path_length, 6),
                "endpoint_displacement_m": round(endpoint_displacement, 6),
                "cumulative_turn_deg": round(cumulative_turn, 6),
                "target_visible_frame_count": visible_frames,
            },
        }
        dispositions.append(row)
        if reasons:
            quarantine.append(
                {
                    **row,
                    "credited_cells": list(episode.get("credited_cells", ())),
                    "required_action": "revoke_all_derived_credit_then_targeted_backfill",
                }
            )

        for sensor_path in sorted(Path(episode["bundle"]).glob("views/view-*.sensors.npz")):
            try:
                with zipfile.ZipFile(sensor_path) as archive:
                    if not archive.namelist():
                        raise zipfile.BadZipFile("empty archive")
            except (OSError, zipfile.BadZipFile) as error:
                bad_npz.append(
                    {
                        "episode_id": episode_id,
                        "path": str(sensor_path),
                        "error": f"{type(error).__name__}:{error}",
                    }
                )

    p3_cells = [cell for cell in manifest["cells"] if _tier(cell["capability"]) == "P3"]
    reason_counts = Counter(reason for row in quarantine for reason in row["reasons"])
    quarantine_by_capability = Counter(row["source_capability"] for row in quarantine)
    quarantined_ids = {row["episode_id"] for row in quarantine}
    cell_by_id = {cell["cell_id"]: cell for cell in manifest["cells"]}
    marginal_deficits: list[dict[str, Any]] = []
    all_partial_deficits: list[dict[str, Any]] = []
    post_quarantine_counts: dict[str, int] = {}
    for cell_id, cell in cell_by_id.items():
        accepted_ids = list(status["cells"][cell_id]["accepted_episode_ids"])
        retained_ids = [item for item in accepted_ids if item not in quarantined_ids]
        target = int(cell["target_accepted"])
        before = max(0, target - len(accepted_ids))
        after = max(0, target - len(retained_ids))
        post_quarantine_counts[cell_id] = len(retained_ids)
        row = {
            "cell_id": cell_id,
            "scene_key": cell["scene_key"],
            "capability": cell["capability"],
            "target_accepted": target,
            "accepted_before": len(accepted_ids),
            "accepted_after_quarantine": len(retained_ids),
            "current_missing_slots": after,
        }
        if after:
            all_partial_deficits.append(row)
        if after > before:
            marginal_deficits.append({**row, "new_missing_slots": after - before})

    rejected_deficit_cells = sorted(
        {
            row["cell_id"]
            for row in rejected_candidates
            if post_quarantine_counts[row["cell_id"]]
            < int(cell_by_id[row["cell_id"]]["target_accepted"])
        }
    )
    local_repair_cell_ids = sorted(
        {row["cell_id"] for row in marginal_deficits} | set(rejected_deficit_cells)
    )
    summary: dict[str, Any] = {
        "schema_version": SCHEMA,
        "source": {
            "coverage_root": str(source_root),
            "collection_id": manifest["collection_id"],
            "manifest_sha256": _sha256(manifest_path),
            "status_sha256": _sha256(status_path),
            "merge_ledger": str(ledger_path),
            "applied_shards": ledger["applied_shards"],
            "partial": partial_snapshot,
        },
        "counts": {
            "source_episode_count": len(dataset["episodes"]),
            "source_episode_by_tier": dict(sorted(source_counts.items())),
            "source_episode_by_capability": dict(sorted(source_capabilities.items())),
            "family_by_tier": dict(sorted(family_counts.items())),
            "raw_variant_status": [
                {"tier": key[0], "variant": key[1], "status": key[2], "count": value}
                for key, value in sorted(raw_counts.items())
            ],
        },
        "labels": {
            key: dict(sorted(value.items())) for key, value in sorted(source_labels.items())
        },
        "quarantine": {
            "episode_count": len(quarantine),
            "by_capability": dict(sorted(quarantine_by_capability.items())),
            "by_reason": dict(sorted(reason_counts.items())),
            "credit_policy": "recompute_in_local_v2_overlay; never mutate v1",
            "marginal_affected_cell_count": len(marginal_deficits),
            "marginal_missing_slot_count": sum(
                row["new_missing_slots"] for row in marginal_deficits
            ),
        },
        "distribution_holds": {
            "self_motion_update_pure_rotation": dict(
                source_labels["self_motion_update_pure_rotation"]
            ),
            "self_motion_update_pure_translation": dict(
                source_labels["self_motion_update_pure_translation"]
            ),
            "self_motion_update_occluded": dict(source_labels["self_motion_update_occluded"]),
        },
        "P2": {
            "source_episode_count": source_counts["P2"],
            "family_count": family_counts["P2"],
            "minimum_curve_margin": {
                "count": len(p2_margins),
                "minimum_deg": min(p2_margins) if p2_margins else None,
                "lt_16_deg": sum(value < 16.0 for value in p2_margins),
                "lt_18_deg": sum(value < 18.0 for value in p2_margins),
            },
            "disposition": "hold_for_v2_streaming_policy_split_and_label_balance",
        },
        "P3": {
            "source_episode_count": source_counts["P3"],
            "cell_count": len(p3_cells),
            "candidate_count": sum(len(cell.get("candidates", ())) for cell in p3_cells),
            "cell_status": dict(
                sorted(Counter(status["cells"][cell["cell_id"]]["status"] for cell in p3_cells).items())
            ),
            "disposition": (
                "blocked_waiting_for_remaining_shard"
                if partial_snapshot
                else "full_snapshot_audited"
            ),
        },
        "integrity": {
            "accepted_npz_container_count": sum(
                len(list(Path(episode["bundle"]).glob("views/view-*.sensors.npz")))
                for episode in dataset["episodes"]
            ),
            "bad_npz_container_count": len(bad_npz),
        },
        "rejected": rejected_summary,
        "v2_policy": {
            "v1_mutation": "forbidden",
            "healthy_media_reuse": "reference_or_hardlink",
            "quarantine_credit_recompute": (
                "current_partial_snapshot_now; repeat after later merge"
                if partial_snapshot
                else "full_merged_snapshot"
            ),
            "qa_recompile": "v2_only",
        },
    }
    if prefix_audit:
        summary["self_motion_streaming"] = _streaming_audit(source_root, dataset)

    _write_json(output_root / "audit.summary.json", summary)
    _write_jsonl(output_root / "episode_dispositions.jsonl", dispositions)
    _write_jsonl(output_root / "quarantine.jsonl", quarantine)
    _write_jsonl(output_root / "bad_npz.jsonl", bad_npz)
    _write_jsonl(output_root / "rejected_candidates.jsonl", rejected_candidates)
    _write_json(
        output_root / "rejected_repair_queue.json",
        {
            "schema_version": "epispace.behavior51_v2_rejected_repair_queue.v1",
            "source_audit": "audit.summary.json",
            "candidate_count": len(rejected_candidates),
            "action_scope": rejected_summary["repair_action_scope"],
            "actions": {
                action: [
                    row["candidate_id"]
                    for row in rejected_candidates
                    if row["repair_action"] == action
                ]
                for action in sorted(
                    {row["repair_action"] for row in rejected_candidates}
                )
            },
        },
    )
    _write_json(
        output_root / "backfill_requirements.json",
        {
            "schema_version": "epispace.behavior51_v2_backfill_requirements.v1",
            "source_audit": "audit.summary.json",
            "replacement_source_episodes": dict(sorted(quarantine_by_capability.items())),
            "additional_distribution_work": {
                "self_motion_update_pure_rotation": "add left/right and downsample back per binding",
                "self_motion_update_pure_translation": "add left/right and downsample back per binding",
                "self_motion_update_occluded": "redesign occlusion at intermediate t_occ then move to t_q",
            },
            "current_partial_snapshot": {
                "snapshot_scope": "partial" if partial_snapshot else "full_merged",
                "quarantine_marginal_cell_count": len(marginal_deficits),
                "quarantine_marginal_missing_slots": sum(
                    row["new_missing_slots"] for row in marginal_deficits
                ),
                "rejected_deficit_cell_count": len(rejected_deficit_cells),
                "local_repair_cell_count": len(local_repair_cell_ids),
                "all_current_deficit_cell_count": len(all_partial_deficits),
                "all_current_missing_slots": sum(
                    row["current_missing_slots"] for row in all_partial_deficits
                ),
            },
            "merge_policy": (
                "start local repair now; if another canonical shard is merged later, "
                "recompute and deduplicate its episode credits before final v2 release"
                if partial_snapshot
                else "all canonical shards are merged; build the final v2 overlay from this audit"
            ),
        },
    )
    _write_json(
        output_root / "local_repair_cell_ids.json",
        {
            "schema_version": "epispace.behavior51_v2_local_repair_cells.v1",
            "source_manifest_sha256": _sha256(manifest_path),
            "source_status_sha256": _sha256(status_path),
            "cell_ids": local_repair_cell_ids,
            "quarantine_marginal_deficits": marginal_deficits,
            "rejected_deficit_cell_ids": rejected_deficit_cells,
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-prefix-audit", action="store_true")
    args = parser.parse_args()
    summary = audit(args.source, args.output, prefix_audit=not args.skip_prefix_audit)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
