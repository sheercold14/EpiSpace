"""Build and verify a portable metadata index for Episode3D-OG releases."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omnigibson_episode.io import sha256_file, write_json_atomic
from omnigibson_episode.sweep import refresh_sweep_plan

_METADATA_MEMBERS = (
    "render_report.json",
    "trajectory_selection.json",
    "trajectory_plan.json",
    "scene_snapshot.json",
    "scene_ir.json",
    "relation_oracle.json",
    "spatial_episode.json",
    "quality_report.json",
    "reasoning_audit.json",
    "reasoning_tasks.json",
)
_HUMAN_DECISIONS = {"accept", "review", "reject", "unreviewed"}
_LOCAL_POSIX_PATH = re.compile(r"/(?:[^\s'\"\[\],]+/)*[^\s'\"\[\],]+")


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _portable_status_detail(value: Any) -> str | None:
    """Retain failure meaning without publishing producer-local paths."""

    if value is None:
        return None
    return _LOCAL_POSIX_PATH.sub("<local-path>", str(value))


def _record_tier(
    *, job_status: str, quality: dict[str, Any] | None, audit: dict[str, Any] | None
) -> tuple[str, list[str]]:
    reasons = []
    if job_status not in {"passed", "needs_review"}:
        return "excluded", [f"static job status is {job_status}"]
    if quality is None or quality.get("integrity_status") != "pass":
        reasons.append("sensor integrity is not a full pass")
    if audit is None or audit.get("evidence_eligibility") != "pass":
        reasons.append("reasoning evidence eligibility is insufficient")
    if reasons:
        return "audit_only", reasons
    if quality.get("visual_status") != "pass":
        return "candidate", ["visual warning requires explicit human review"]
    return "candidate", ["awaiting explicit human acceptance"]


def _human_review_decisions(
    review_path: Path | None,
    *,
    expected_sweep_id: str,
    known_job_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    if review_path is None:
        return {}, None
    path = review_path.resolve()
    payload = _read(path)
    if payload.get("schema_version") != "omnigibson_human_review.v1":
        raise ValueError("unsupported human review schema")
    if payload.get("sweep_id") != expected_sweep_id:
        raise ValueError("human review sweep_id does not match release sweep")
    reviewer = payload.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("human review requires a non-empty reviewer")
    raw_decisions = payload.get("decisions")
    if not isinstance(raw_decisions, dict):
        raise ValueError("human review decisions must be an object")
    unknown = sorted(set(raw_decisions) - known_job_ids)
    if unknown:
        raise ValueError(f"human review contains unknown acquisition IDs: {unknown}")

    decisions: dict[str, dict[str, Any]] = {}
    for job_id, item in raw_decisions.items():
        if not isinstance(item, dict):
            raise ValueError(f"human review decision must be an object: {job_id}")
        decision = item.get("decision", "unreviewed")
        if decision not in _HUMAN_DECISIONS:
            raise ValueError(f"invalid human review decision for {job_id}: {decision}")
        note = item.get("note", "")
        if not isinstance(note, str):
            raise ValueError(f"human review note must be text: {job_id}")
        updated_at_utc = item.get("updated_at_utc")
        if updated_at_utc is not None and not isinstance(updated_at_utc, str):
            raise ValueError(f"human review timestamp must be text: {job_id}")
        decisions[str(job_id)] = {
            "decision": decision,
            "note": note,
            "updated_at_utc": updated_at_utc,
        }
    provenance = {
        "reviewer": reviewer.strip(),
        "generated_at_utc": payload.get("generated_at_utc"),
        "source_sha256": sha256_file(path),
    }
    return decisions, provenance


def _apply_human_decision(
    automated_tier: str,
    automated_reasons: list[str],
    decision: dict[str, Any],
) -> tuple[str, list[str]]:
    value = decision["decision"]
    if value == "reject":
        return "excluded", [*automated_reasons, "rejected by human review"]
    if automated_tier != "candidate":
        return automated_tier, automated_reasons
    if value == "accept":
        return "accepted", ["all automated gates and explicit human review passed"]
    if value == "review":
        return "audit_only", ["human review requested further inspection"]
    return automated_tier, automated_reasons


def build_release_manifest(
    *,
    sweep_plan_path: Path,
    split_manifest_path: Path,
    output_path: Path,
    release_id: str,
    human_review_path: Path | None = None,
    freeze: bool = False,
) -> dict[str, Any]:
    """Index all 46 acquisitions without confusing baseline and training acceptance."""

    sweep_path = sweep_plan_path.resolve()
    split_path = split_manifest_path.resolve()
    sweep = refresh_sweep_plan(sweep_path)
    splits = _read(split_path)
    split_by_scene = {
        str(item["scene_model"]): item for item in splits["assignments"]
    }
    output_root = Path(str(sweep["output_root"])).resolve()
    aggregate_audit_path = output_root / "derived_refresh_report.json"
    aggregate_audit = (
        _read(aggregate_audit_path) if aggregate_audit_path.is_file() else None
    )
    aggregate_audit_artifact = (
        _artifact(aggregate_audit_path) if aggregate_audit_path.is_file() else None
    )
    known_job_ids = {str(item["job_id"]) for item in sweep["jobs"]}
    human_decisions, human_review_provenance = _human_review_decisions(
        human_review_path,
        expected_sweep_id=str(sweep["sweep_id"]),
        known_job_ids=known_job_ids,
    )
    records = []
    tier_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    human_decision_counts: dict[str, int] = {}
    for job in sorted(sweep["jobs"], key=lambda item: str(item["scene_model"])):
        scene_model = str(job["scene_model"])
        assignment = split_by_scene.get(scene_model)
        if assignment is None:
            raise ValueError(f"scene has no split assignment: {scene_model}")
        bundle = Path(str(job["bundle"])).resolve()
        try:
            bundle_relpath = str(bundle.relative_to(output_root))
        except ValueError as error:
            raise ValueError("bundle lies outside the sweep output root") from error
        quality_path = bundle / "quality_report.json"
        audit_path = bundle / "reasoning_audit.json"
        quality = _read(quality_path) if quality_path.is_file() else None
        audit = _read(audit_path) if audit_path.is_file() else None
        automated_tier, automated_reasons = _record_tier(
            job_status=str(job["status"]), quality=quality, audit=audit
        )
        human_decision = human_decisions.get(
            str(job["job_id"]),
            {"decision": "unreviewed", "note": "", "updated_at_utc": None},
        )
        tier, reasons = _apply_human_decision(
            automated_tier, automated_reasons, human_decision
        )
        metadata = {
            name: _artifact(bundle / name)
            for name in _METADATA_MEMBERS
            if (bundle / name).is_file()
        }
        episode = (
            _read(bundle / "spatial_episode.json")
            if (bundle / "spatial_episode.json").is_file()
            else None
        )
        report = (
            _read(bundle / "render_report.json")
            if (bundle / "render_report.json").is_file()
            else None
        )
        record = {
            "acquisition_id": str(job["job_id"]),
            "scene_model": scene_model,
            "domain": assignment["domain"],
            "classification": job["classification"],
            "split": assignment["split"],
            "split_group": assignment["split_group"],
            "seed": int(job["seed"]),
            "static_job_status": job["status"],
            "status_detail": _portable_status_detail(job.get("status_detail")),
            "automated_release_tier": automated_tier,
            "release_tier": tier,
            "release_tier_reasons": reasons,
            "human_review": human_decision,
            "bundle_relpath": bundle_relpath,
            "episode_id": episode.get("episode_id") if episode else None,
            "family_id": episode.get("family_id") if episode else None,
            "family_variant": episode.get("family_variant") if episode else None,
            "view_count": len(report.get("views", [])) if report else 0,
            "quality": {
                "integrity_status": quality.get("integrity_status"),
                "visual_status": quality.get("visual_status"),
                "warnings": quality.get("warnings", []),
            }
            if quality
            else None,
            "reasoning": {
                "evidence_eligibility": audit.get("evidence_eligibility"),
                "reasoning_status": audit.get("reasoning_status"),
                "observed_entity_count": audit.get("evidence_summary", {}).get(
                    "observed_entity_count"
                ),
                "observed_category_count": audit.get("evidence_summary", {}).get(
                    "observed_category_count"
                ),
            }
            if audit
            else None,
            "metadata_artifacts": metadata,
        }
        records.append(record)
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        human_value = str(human_decision["decision"])
        human_decision_counts[human_value] = human_decision_counts.get(human_value, 0) + 1
        split_counts[assignment["split"]] = split_counts.get(assignment["split"], 0) + 1
    manifest = {
        "schema_version": "episode3d_release.v1",
        "release_id": release_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_name": "Episode3D-OG",
        "unit": "independently rendered acquisition bundle",
        "source": {
            "simulator": "OmniGibson 3.9.0 / Isaac Sim 5.1.0",
            "assets": "BEHAVIOR-1K 3.9.0",
            "sweep_id": sweep["sweep_id"],
            "sweep_plan_sha256": sha256_file(sweep_path),
            "split_id": splits["split_id"],
            "split_manifest_sha256": sha256_file(split_path),
            "aggregate_audit": {
                "artifact": aggregate_audit_artifact,
                "counts": aggregate_audit.get("counts", {}),
                "coverage_summary": aggregate_audit.get("coverage_summary", {}),
            }
            if aggregate_audit is not None
            else None,
            "human_review": human_review_provenance,
        },
        "license_boundary": {
            "release_content": "metadata_and_certificates",
            "rendered_media_redistribution": "not_authorized_without_separate_review",
            "source_assets": "user_must_obtain_upstream",
        },
        "path_contract": {
            "bundle_relpath_base": "sweep output_root",
            "portable_data_root_argument_required": True,
        },
        "counts": {
            "acquisitions": len(records),
            "by_release_tier": dict(sorted(tier_counts.items())),
            "by_human_decision": dict(sorted(human_decision_counts.items())),
            "by_split": dict(sorted(split_counts.items())),
        },
        "records": records,
    }
    if freeze:
        _assert_freezable(manifest)
    write_json_atomic(output_path, manifest)
    return manifest


def _assert_freezable(manifest: dict[str, Any]) -> None:
    """Reject a versioned snapshot while production or review remains open."""

    records = manifest["records"]
    if len(records) != 46:
        raise ValueError("frozen static M2 release must contain exactly 46 acquisitions")
    unfinished = [
        record["acquisition_id"]
        for record in records
        if record["static_job_status"] not in {"passed", "needs_review"}
    ]
    if unfinished:
        raise ValueError(f"frozen release has unfinished acquisitions: {unfinished}")
    unresolved = [
        record["acquisition_id"]
        for record in records
        if record["release_tier"] == "candidate"
    ]
    if unresolved:
        raise ValueError(f"frozen release has unresolved review candidates: {unresolved}")
    if manifest.get("source", {}).get("human_review") is None:
        raise ValueError("frozen release requires a provenance-tracked human review")
    aggregate = manifest.get("source", {}).get("aggregate_audit")
    if not isinstance(aggregate, dict):
        raise ValueError("frozen release requires a provenance-tracked aggregate audit")
    counts = aggregate.get("counts", {})
    if counts.get("refreshed") != 46 or any(
        int(counts.get(status, 0)) != 0 for status in ("skipped", "failed")
    ):
        raise ValueError("frozen release aggregate audit must refresh all 46 acquisitions")
    coverage = aggregate.get("coverage_summary", {})
    if coverage.get("refreshed_acquisition_count") != 46:
        raise ValueError("frozen release aggregate coverage does not contain 46 acquisitions")
    if coverage.get("sampled_without_evidence_scenes"):
        raise ValueError("frozen release contains capabilities sampled without evidence")


def verify_release_manifest(
    *, manifest_path: Path, data_root: Path
) -> dict[str, Any]:
    manifest = _read(manifest_path)
    root = data_root.resolve()
    checks = []
    global_errors = []
    if manifest.get("schema_version") != "episode3d_release.v1":
        global_errors.append("unsupported release schema_version")
    if manifest.get("dataset_name") != "Episode3D-OG":
        global_errors.append("unexpected dataset_name")
    aggregate = manifest.get("source", {}).get("aggregate_audit")
    if isinstance(aggregate, dict):
        artifact = aggregate.get("artifact")
        if not isinstance(artifact, dict):
            global_errors.append("aggregate audit artifact metadata is unavailable")
        else:
            relative = Path(str(artifact.get("path", "")))
            if relative.is_absolute() or len(relative.parts) != 1:
                global_errors.append("aggregate audit path is unsafe")
            else:
                aggregate_path = root / relative
                if not aggregate_path.is_file():
                    global_errors.append("aggregate audit artifact is missing")
                elif aggregate_path.stat().st_size != int(artifact.get("byte_size", -1)):
                    global_errors.append("aggregate audit artifact size does not match")
                elif sha256_file(aggregate_path) != artifact.get("sha256"):
                    global_errors.append("aggregate audit artifact hash does not match")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("release records must be an array")
    acquisition_ids = [str(record.get("acquisition_id")) for record in records]
    if len(acquisition_ids) != len(set(acquisition_ids)):
        global_errors.append("duplicate acquisition_id")
    actual_tier_counts: dict[str, int] = {}
    for record in records:
        tier = str(record.get("release_tier"))
        actual_tier_counts[tier] = actual_tier_counts.get(tier, 0) + 1
    declared_counts = manifest.get("counts", {}).get("by_release_tier")
    if declared_counts is not None and declared_counts != dict(sorted(actual_tier_counts.items())):
        global_errors.append("counts.by_release_tier does not match records")
    for record in records:
        bundle = (root / str(record["bundle_relpath"])).resolve()
        try:
            bundle.relative_to(root)
        except ValueError:
            checks.append(
                {
                    "acquisition_id": record["acquisition_id"],
                    "passed": False,
                    "error": "bundle path escapes data root",
                }
            )
            continue
        errors = []
        tier = str(record.get("release_tier"))
        metadata = record.get("metadata_artifacts", {})
        if not isinstance(metadata, dict):
            metadata = {}
            errors.append("metadata_artifacts is not an object")
        if tier in {"accepted", "candidate"}:
            missing = sorted(set(_METADATA_MEMBERS) - set(metadata))
            errors.extend(f"required:{name}" for name in missing)
        if tier == "accepted" and record.get("human_review", {}).get("decision") != "accept":
            errors.append("accepted tier lacks human accept decision")
        for key, artifact in metadata.items():
            artifact_relpath = Path(str(artifact["path"]))
            if artifact_relpath.is_absolute() or len(artifact_relpath.parts) != 1:
                errors.append(f"unsafe-artifact-path:{artifact_relpath}")
                continue
            if artifact_relpath.name != key:
                errors.append(f"artifact-key-mismatch:{key}")
                continue
            path = (bundle / artifact_relpath).resolve()
            try:
                path.relative_to(bundle)
            except ValueError:
                errors.append(f"artifact-path-escapes-bundle:{key}")
                continue
            if not path.is_file():
                errors.append(f"missing:{path.name}")
                continue
            if path.stat().st_size != int(artifact["byte_size"]):
                errors.append(f"size:{path.name}")
            elif sha256_file(path) != artifact["sha256"]:
                errors.append(f"sha256:{path.name}")
        checks.append(
            {
                "acquisition_id": record["acquisition_id"],
                "passed": not errors,
                "errors": errors,
            }
        )
    failed = [item for item in checks if not item["passed"]]
    return {
        "schema_version": "episode3d_release_verification.v1",
        "release_id": manifest.get("release_id"),
        "passed": not failed and not global_errors,
        "record_count": len(checks),
        "failed_count": len(failed),
        "global_errors": global_errors,
        "checks": checks,
    }


class EpisodeDatasetIndex:
    """Dependency-free community loader for a metadata release index."""

    def __init__(self, manifest_path: Path, data_root: Path) -> None:
        self.manifest_path = manifest_path.resolve()
        self.data_root = data_root.resolve()
        self.manifest = _read(self.manifest_path)
        if self.manifest.get("schema_version") != "episode3d_release.v1":
            raise ValueError("unsupported release schema_version")
        if self.manifest.get("dataset_name") != "Episode3D-OG":
            raise ValueError("unexpected release dataset_name")
        if not isinstance(self.manifest.get("records"), list):
            raise ValueError("release records must be an array")

    def records(
        self,
        *,
        split: str | None = None,
        release_tier: str | None = "accepted",
    ) -> Iterator[dict[str, Any]]:
        """Iterate training-safe records by default.

        Pass ``release_tier=None`` only for an explicit full-registry audit.
        This prevents unreviewed, audit-only, or excluded acquisitions from
        silently entering a community training loop.
        """

        for record in self.manifest["records"]:
            if split is not None and record["split"] != split:
                continue
            if release_tier is not None and record["release_tier"] != release_tier:
                continue
            yield record

    def bundle_path(self, record: dict[str, Any]) -> Path:
        path = (self.data_root / str(record["bundle_relpath"])).resolve()
        path.relative_to(self.data_root)
        return path

    def load_episode(self, record: dict[str, Any]) -> dict[str, Any]:
        return _read(self.bundle_path(record) / "spatial_episode.json")
