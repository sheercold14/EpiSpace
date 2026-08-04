"""Resolve typed-trajectory warnings into reproducible training dispositions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from omnigibson_episode.io import sha256_file, write_json_atomic

VISUAL_RULE_VERSION = "typed-visual-information-v1"


def _visual_blockers(quality: dict[str, Any]) -> list[dict[str, Any]]:
    """Return high-precision visual failures; flat texture alone is not enough."""

    blockers = []
    for view in quality.get("views", []):
        reasons = []
        if float(view["rgb_std"]) < 5.0:
            reasons.append("near_uniform_rgb")
        if float(view["rgb_p99"]) < 20.0:
            reasons.append("severe_underexposure")
        if float(view["rgb_p01"]) > 235.0:
            reasons.append("severe_overexposure")
        if (
            float(view["sharpness_laplacian_variance"]) < 1.5
            and int(view["visible_instance_count"]) <= 2
        ):
            reasons.append("low_information_or_camera_collision")
        if reasons:
            blockers.append({"view_id": view["view_id"], "reasons": reasons})
    return blockers


def adjudicate_quality(quality: dict[str, Any]) -> dict[str, Any]:
    """Separate sensor integrity, typed evidence and visual information gates."""

    trajectory_class = str(quality.get("trajectory_class", "unknown"))
    gate = quality.get("gates", {}).get(trajectory_class, {})
    failed_checks = sorted(
        name for name, passed in gate.get("checks", {}).items() if not passed
    )
    visual_blockers = _visual_blockers(quality)
    if quality.get("integrity_status") != "pass":
        decision = "reject_integrity"
    elif gate and gate.get("status") != "pass":
        decision = "reject_typed_evidence"
    elif visual_blockers:
        decision = "reject_visual_information"
    elif quality.get("visual_status") != "pass":
        decision = "review_visual_ambiguity"
    else:
        decision = "accept_typed_training"
    return {
        "decision": decision,
        "trajectory_class": trajectory_class,
        "failed_typed_checks": failed_checks,
        "visual_blockers": visual_blockers,
        "salvage_policy": (
            "typed_training"
            if decision == "accept_typed_training"
            else "audit_only; may be recompiled for another task only after new certificates"
        ),
    }


def adjudicate_sweeps(plan_paths: list[Path], output_path: Path) -> dict[str, Any]:
    records = []
    sources = []
    for raw_path in plan_paths:
        plan_path = raw_path.resolve()
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        sources.append(
            {
                "sweep_id": plan["sweep_id"],
                "plan": str(plan_path),
                "sha256": sha256_file(plan_path),
            }
        )
        for job in plan["jobs"]:
            quality_path = Path(job["bundle"]) / "quality_report.json"
            if not quality_path.is_file():
                records.append(
                    {
                        "job_id": job["job_id"],
                        "scene_model": job["scene_model"],
                        "decision": "unavailable_no_quality_report",
                    }
                )
                continue
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            records.append(
                {
                    "job_id": job["job_id"],
                    "scene_model": job["scene_model"],
                    "quality_report": str(quality_path),
                    "quality_sha256": sha256_file(quality_path),
                    **adjudicate_quality(quality),
                }
            )
    counts = Counter(record["decision"] for record in records)
    payload = {
        "schema_version": "omnigibson_typed_adjudication.v1",
        "visual_rule_version": VISUAL_RULE_VERSION,
        "visual_rule": {
            "near_uniform_rgb": "rgb_std < 5",
            "severe_underexposure": "rgb_p99 < 20",
            "severe_overexposure": "rgb_p01 > 235",
            "low_information_or_camera_collision": (
                "sharpness_laplacian_variance < 1.5 and visible_instance_count <= 2"
            ),
            "principle": (
                "typed evidence failure always excludes the bundle from that typed task; "
                "visual inspection cannot promote an invalid geometric certificate"
            ),
        },
        "sources": sources,
        "counts": dict(sorted(counts.items())),
        "records": records,
    }
    write_json_atomic(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = adjudicate_sweeps(args.plan, args.output)
    print(json.dumps(payload["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
