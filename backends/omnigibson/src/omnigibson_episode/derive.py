"""Rebuild all deterministic derivatives for terminal acquisition bundles."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omnigibson_episode.compile_shared import compile_bundle
from omnigibson_episode.io import write_json_atomic
from omnigibson_episode.quality import inspect_bundle
from omnigibson_episode.reasoning_audit import audit_reasoning_bundle
from omnigibson_episode.reasoning_tasks import compile_reasoning_tasks
from omnigibson_episode.sweep import refresh_sweep_plan


def _increment(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _coverage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    refreshed = [record for record in records if record["status"] == "refreshed"]
    capability_counts: dict[str, dict[str, int]] = {}
    missing_capability_scenes: dict[str, list[str]] = {}
    sampled_without_evidence_scenes: dict[str, list[str]] = {}
    reasoning_status_counts: dict[str, int] = {}
    evidence_eligibility_counts: dict[str, int] = {}
    integrity_status_counts: dict[str, int] = {}
    visual_status_counts: dict[str, int] = {}
    task_counts: list[int] = []
    for record in refreshed:
        _increment(reasoning_status_counts, str(record["reasoning_status"]))
        _increment(evidence_eligibility_counts, str(record["evidence_eligibility"]))
        _increment(integrity_status_counts, str(record["quality"]["integrity"]))
        _increment(visual_status_counts, str(record["quality"]["visual"]))
        task_counts.append(int(record["task_count"]))
        for capability, state in record.get("capability_matrix", {}).items():
            counts = capability_counts.setdefault(
                str(capability), {"evidence_supported": 0, "sampled": 0}
            )
            if state.get("evidence_supported"):
                counts["evidence_supported"] += 1
            if state.get("sampled"):
                counts["sampled"] += 1
                if not state.get("evidence_supported"):
                    sampled_without_evidence_scenes.setdefault(
                        str(capability), []
                    ).append(str(record["scene_model"]))
            else:
                missing_capability_scenes.setdefault(str(capability), []).append(
                    str(record["scene_model"])
                )
    return {
        "refreshed_acquisition_count": len(refreshed),
        "reasoning_status_counts": dict(sorted(reasoning_status_counts.items())),
        "evidence_eligibility_counts": dict(
            sorted(evidence_eligibility_counts.items())
        ),
        "integrity_status_counts": dict(sorted(integrity_status_counts.items())),
        "visual_status_counts": dict(sorted(visual_status_counts.items())),
        "task_count": {
            "total": sum(task_counts),
            "minimum_per_acquisition": min(task_counts, default=0),
            "maximum_per_acquisition": max(task_counts, default=0),
        },
        "capability_counts": dict(sorted(capability_counts.items())),
        "missing_capability_scenes": {
            capability: sorted(scenes)
            for capability, scenes in sorted(missing_capability_scenes.items())
        },
        "sampled_without_evidence_scenes": {
            capability: sorted(scenes)
            for capability, scenes in sorted(
                sampled_without_evidence_scenes.items()
            )
        },
    }


def refresh_derived_bundles(
    plan_path: Path,
    *,
    scene_models: set[str] | None = None,
    generator_version: str = "omnigibson-spatial-episode/0.1.0",
) -> dict[str, Any]:
    """Recompile, inspect and audit every selected terminal render bundle.

    Acquisition files are immutable inputs to this operation. Failures are
    collected instead of aborting the batch so the report is a complete audit
    ledger rather than a first-error log.
    """

    path = plan_path.resolve()
    plan = refresh_sweep_plan(path)
    records = []
    for job in plan["jobs"]:
        scene_model = str(job["scene_model"])
        if scene_models is not None and scene_model not in scene_models:
            continue
        if job["status"] not in {"passed", "needs_review"}:
            records.append(
                {
                    "job_id": job["job_id"],
                    "scene_model": scene_model,
                    "status": "skipped",
                    "detail": f"static job status is {job['status']}",
                }
            )
            continue
        bundle = Path(str(job["bundle"])).resolve()
        try:
            episode = compile_bundle(
                bundle_directory=bundle,
                output_path=bundle / "spatial_episode.json",
                generator_version=generator_version,
            )
            quality = inspect_bundle(bundle)
            tasks = compile_reasoning_tasks(bundle)
            audit = audit_reasoning_bundle(bundle)
            records.append(
                {
                    "job_id": job["job_id"],
                    "scene_model": scene_model,
                    "status": "refreshed",
                    "episode_id": str(episode.episode_id),
                    "quality": {
                        "integrity": quality["integrity_status"],
                        "visual": quality["visual_status"],
                    },
                    "task_count": int(tasks["task_count"]),
                    "task_coverage_status": tasks.get("coverage_status"),
                    "reasoning_status": audit["reasoning_status"],
                    "evidence_eligibility": audit["evidence_eligibility"],
                    "evidence_summary": audit.get("evidence_summary", {}),
                    "capability_matrix": audit.get("capability_matrix", {}),
                    "gaps": audit.get("gaps", []),
                }
            )
        except Exception as error:
            records.append(
                {
                    "job_id": job["job_id"],
                    "scene_model": scene_model,
                    "status": "failed",
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            )
    refreshed_plan = refresh_sweep_plan(path)
    counts: dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    report = {
        "schema_version": "omnigibson_derived_refresh.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "sweep_id": refreshed_plan["sweep_id"],
        "generator_version": generator_version,
        "counts": dict(sorted(counts.items())),
        "static_status_counts": refreshed_plan["status_counts"],
        "coverage_summary": _coverage_summary(records),
        "records": records,
    }
    output = Path(str(refreshed_plan["output_root"])) / "derived_refresh_report.json"
    write_json_atomic(output, report)
    return report
