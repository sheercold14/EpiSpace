"""Plan and refresh resumable same-scene intervention sweeps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omnigibson_episode.intervention_plan import plan_relation_flip_interventions
from omnigibson_episode.io import sha256_file, write_json_atomic
from omnigibson_episode.model_swap_plan import plan_model_swap_interventions
from omnigibson_episode.sweep import refresh_sweep_plan


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def intervention_bundle_status(
    bundle: Path, base_bundle: Path | None = None
) -> tuple[str, str | None]:
    pair_path = bundle / "minimal_pair.json"
    quality_path = bundle / "quality_report.json"
    if pair_path.is_file() and quality_path.is_file():
        pair = _read(pair_path)
        quality = _read(quality_path)
        if pair.get("status") != "certified":
            return "failed", "minimal-pair certificate is not certified"
        if quality.get("integrity_status") != "pass":
            return "failed", "variant sensor integrity is not a full pass"
        base_quality = None
        if base_bundle is not None:
            base_quality_path = base_bundle / "quality_report.json"
            if not base_quality_path.is_file():
                return "failed", "base quality report is unavailable"
            base_quality = _read(base_quality_path)
        if base_quality is not None and base_quality.get("integrity_status") != "pass":
            return "failed", "base sensor integrity is not a full pass"
        if quality.get("visual_status") == "pass" and (
            base_quality is None or base_quality.get("visual_status") == "pass"
        ):
            return "certified", None
        return "needs_review", "base or variant visual quality requires human review"
    failure = bundle / "failure_report.json"
    if failure.is_file():
        payload = _read(failure)
        return "failed", payload.get("error", {}).get("message", "intervention failed")
    certification_failure = bundle / "minimal_pair_failure.json"
    if certification_failure.is_file():
        payload = _read(certification_failure)
        return "failed", payload.get("error", {}).get(
            "message", "minimal-pair certification failed"
        )
    if (bundle / "render_report.json").is_file():
        return "acquired", "compile, inspect and pair certification remain"
    if bundle.exists():
        return "incomplete", "bundle exists without a terminal certificate"
    return "pending", None


def build_intervention_sweep_plan(
    *,
    static_sweep_plan_path: Path,
    object_inventory_path: Path,
    split_manifest_path: Path,
    output_root: Path,
    plan_path: Path,
    maximum_candidates_per_scene: int = 3,
    intervention_kind: str = "relation_flip",
    sweep_id: str | None = None,
    model_swap_maximum_baseline_overlap_count: int = 1,
    model_swap_minimum_nonoverlap_clearance_m: float = 0.05,
    model_swap_maximum_aspect_ratio_factor: float = 1.5,
    model_swap_execution_mode: str = "physics_settle",
    intervention_settle_steps: int = 12,
) -> dict[str, Any]:
    if maximum_candidates_per_scene < 1:
        raise ValueError("maximum_candidates_per_scene must be positive")
    static_path = static_sweep_plan_path.resolve()
    object_path = object_inventory_path.resolve()
    split_path = split_manifest_path.resolve()
    root = output_root.resolve()
    if intervention_kind not in {"relation_flip", "model_swap"}:
        raise ValueError(f"unsupported intervention kind: {intervention_kind}")
    if intervention_settle_steps < 0:
        raise ValueError("intervention_settle_steps must be non-negative")
    if intervention_kind == "relation_flip" and intervention_settle_steps < 1:
        raise ValueError("relation flips require at least one settle step")
    effective_sweep_id = sweep_id or (
        "relation-flip-m2-v1"
        if intervention_kind == "relation_flip"
        else "model-swap-m2-v1"
    )
    if not effective_sweep_id.strip():
        raise ValueError("intervention sweep_id must be non-empty")
    static = refresh_sweep_plan(static_path)
    splits = _read(split_path)
    split_by_scene = {
        str(item["scene_model"]): item for item in splits["assignments"]
    }
    jobs = []
    skipped = []
    for base_job in sorted(static["jobs"], key=lambda item: str(item["scene_model"])):
        scene_model = str(base_job["scene_model"])
        base_bundle = Path(str(base_job["bundle"])).resolve()
        if base_job["status"] not in {"passed", "needs_review"}:
            skipped.append(
                {
                    "scene_model": scene_model,
                    "reason": f"static status is {base_job['status']}",
                }
            )
            continue
        audit_path = base_bundle / "reasoning_audit.json"
        if not audit_path.is_file() or _read(audit_path).get("evidence_eligibility") != "pass":
            skipped.append(
                {"scene_model": scene_model, "reason": "evidence eligibility is not a pass"}
            )
            continue
        assignment = split_by_scene.get(scene_model)
        if assignment is None:
            raise ValueError(f"scene has no split assignment: {scene_model}")
        proposal_path = root / "proposals" / f"{base_job['job_id']}.json"
        planner = (
            plan_relation_flip_interventions
            if intervention_kind == "relation_flip"
            else plan_model_swap_interventions
        )
        planner_kwargs: dict[str, Any] = {}
        if intervention_kind == "model_swap":
            planner_kwargs = {
                "maximum_baseline_overlap_count": (
                    model_swap_maximum_baseline_overlap_count
                ),
                "minimum_nonoverlap_clearance_m": (
                    model_swap_minimum_nonoverlap_clearance_m
                ),
                "maximum_aspect_ratio_factor": (
                    model_swap_maximum_aspect_ratio_factor
                ),
                "execution_mode": model_swap_execution_mode,
                "settle_steps": intervention_settle_steps,
            }
        proposal_manifest = planner(
            bundle_directory=base_bundle,
            object_inventory_path=object_path,
            output_path=proposal_path,
            maximum_proposals=max(20, maximum_candidates_per_scene),
            **planner_kwargs,
        )
        selected = []
        used_targets = set()
        for proposal in proposal_manifest["proposals"]:
            target = proposal["target_source_entity_id"]
            if target in used_targets:
                continue
            used_targets.add(target)
            selected.append(proposal)
            if len(selected) >= maximum_candidates_per_scene:
                break
        if not selected:
            skipped.append(
                {"scene_model": scene_model, "reason": "no conservative intervention proposal"}
            )
            continue
        static_root = Path(str(static["output_root"])).resolve()
        recipe = static_root / "recipes" / f"{base_job['job_id']}.yaml"
        if not recipe.is_file():
            raise FileNotFoundError(recipe)
        for proposal in selected:
            proposal_id = str(proposal["proposal_id"])
            job_id = f"{base_job['job_id']}__{proposal_id}"
            bundle = root / "bundles" / str(base_job["job_id"]) / proposal_id
            status, detail = intervention_bundle_status(bundle, base_bundle)
            jobs.append(
                {
                    "job_id": job_id,
                    "scene_model": scene_model,
                    "split": assignment["split"],
                    "split_group": assignment["split_group"],
                    "base_job_id": base_job["job_id"],
                    "base_bundle": str(base_bundle),
                    "recipe": str(recipe),
                    "proposal_plan": str(proposal_path),
                    "proposal_id": proposal_id,
                    "intervention_type": proposal_manifest["intervention_type"],
                    "target_source_entity_id": proposal["target_source_entity_id"],
                    "anchor_source_entity_id": proposal["anchor_source_entity_id"],
                    "relation_before": proposal["relation_before"],
                    "relation_after": proposal["relation_after"],
                    "source_model": proposal.get("source_model"),
                    "replacement_model": proposal.get("replacement_model"),
                    "execution_mode": proposal.get(
                        "execution_mode", "physics_settle"
                    ),
                    "settle_steps": int(
                        proposal.get("settle_steps", intervention_settle_steps)
                    ),
                    "bundle": str(bundle),
                    "receipt": str(root / "job_status" / f"{job_id}.json"),
                    "status": status,
                    "status_detail": detail,
                }
            )
    plan = {
        "schema_version": "omnigibson_intervention_sweep.v1",
        "sweep_id": effective_sweep_id,
        "intervention_kind": intervention_kind,
        "static_sweep_plan": str(static_path),
        "static_sweep_plan_sha256": sha256_file(static_path),
        "object_inventory": str(object_path),
        "object_inventory_sha256": sha256_file(object_path),
        "split_manifest": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "output_root": str(root),
        "maximum_candidates_per_scene": maximum_candidates_per_scene,
        "planner_policy": (
            {
                "maximum_baseline_overlap_count": (
                    model_swap_maximum_baseline_overlap_count
                ),
                "minimum_nonoverlap_clearance_m": (
                    model_swap_minimum_nonoverlap_clearance_m
                ),
                "maximum_aspect_ratio_factor": (
                    model_swap_maximum_aspect_ratio_factor
                ),
                "execution_mode": model_swap_execution_mode,
                "settle_steps": intervention_settle_steps,
            }
            if intervention_kind == "model_swap"
            else {"policy": "room_safe_relation_flip"}
        ),
        "execution_policy": {
            "mode": (
                model_swap_execution_mode
                if intervention_kind == "model_swap"
                else "physics_settle"
            ),
            "settle_steps": intervention_settle_steps,
        },
        "job_count": len(jobs),
        "scene_count": len({job["scene_model"] for job in jobs}),
        "skipped_scenes": skipped,
        "jobs": jobs,
    }
    write_json_atomic(plan_path, plan)
    return plan


def refresh_intervention_sweep_plan(plan_path: Path) -> dict[str, Any]:
    plan = _read(plan_path)
    counts: dict[str, int] = {}
    for job in plan["jobs"]:
        status, detail = intervention_bundle_status(
            Path(job["bundle"]), Path(job["base_bundle"])
        )
        receipt_path = Path(job["receipt"])
        if status in {"pending", "incomplete"} and receipt_path.is_file():
            receipt = _read(receipt_path)
            status = str(receipt.get("status", status))
            detail = receipt.get("detail", detail)
        job["status"] = status
        job["status_detail"] = detail
        counts[status] = counts.get(status, 0) + 1
    plan["status_counts"] = dict(sorted(counts.items()))
    write_json_atomic(plan_path, plan)
    return plan
