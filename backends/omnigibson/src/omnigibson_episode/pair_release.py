"""Portable release index for certified same-scene minimal pairs."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omnigibson_episode.intervention_sweep import refresh_intervention_sweep_plan
from omnigibson_episode.io import sha256_file, write_json_atomic
from omnigibson_episode.release import _apply_human_decision, _portable_status_detail

_PAIR_VARIANT_MEMBERS = (
    "render_report.json",
    "trajectory_plan.json",
    "scene_snapshot.json",
    "scene_ir.json",
    "spatial_episode.json",
    "quality_report.json",
    "reasoning_tasks.json",
    "reasoning_audit.json",
    "intervention_execution.json",
    "minimal_pair.json",
)
_HUMAN_DECISIONS = {"accept", "review", "reject", "unreviewed"}


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)}


def _relative(path: Path, root: Path, *, label: str) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as error:
        raise ValueError(f"{label} bundle lies outside its declared data root") from error


def _review_decisions(
    review_path: Path | None,
    *,
    expected_sweep_id: str,
    known_job_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    if review_path is None:
        return {}, None
    path = review_path.resolve()
    payload = _read(path)
    if payload.get("schema_version") != "omnigibson_intervention_human_review.v1":
        raise ValueError("unsupported intervention human review schema")
    if payload.get("sweep_id") != expected_sweep_id:
        raise ValueError("intervention human review sweep_id does not match")
    reviewer = payload.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("intervention human review requires a reviewer")
    raw = payload.get("decisions")
    if not isinstance(raw, dict):
        raise ValueError("intervention human review decisions must be an object")
    unknown = sorted(set(raw) - known_job_ids)
    if unknown:
        raise ValueError(f"intervention review contains unknown job IDs: {unknown}")
    decisions = {}
    for job_id, item in raw.items():
        if not isinstance(item, dict):
            raise ValueError(f"intervention review decision must be an object: {job_id}")
        decision = item.get("decision", "unreviewed")
        note = item.get("note", "")
        timestamp = item.get("updated_at_utc")
        if decision not in _HUMAN_DECISIONS:
            raise ValueError(f"invalid intervention review decision: {decision}")
        if not isinstance(note, str):
            raise ValueError(f"intervention review note must be text: {job_id}")
        if timestamp is not None and not isinstance(timestamp, str):
            raise ValueError(f"intervention review timestamp must be text: {job_id}")
        decisions[str(job_id)] = {
            "decision": decision,
            "note": note,
            "updated_at_utc": timestamp,
        }
    provenance = {
        "reviewer": reviewer.strip(),
        "generated_at_utc": payload.get("generated_at_utc"),
        "source_sha256": sha256_file(path),
    }
    return decisions, provenance


def _automated_tier(
    *, job_status: str, base_quality: dict[str, Any] | None, variant_quality: dict[str, Any] | None
) -> tuple[str, list[str]]:
    if job_status not in {"certified", "needs_review"}:
        return "excluded", [f"intervention job status is {job_status}"]
    reasons = []
    if base_quality is None or base_quality.get("integrity_status") != "pass":
        reasons.append("base sensor integrity is not a full pass")
    if variant_quality is None or variant_quality.get("integrity_status") != "pass":
        reasons.append("variant sensor integrity is not a full pass")
    if reasons:
        return "audit_only", reasons
    if (
        base_quality.get("visual_status") != "pass"
        or variant_quality.get("visual_status") != "pass"
    ):
        return "candidate", ["base or variant visual warning requires human review"]
    return "candidate", ["awaiting explicit pair-level human acceptance"]


def build_pair_release_manifest(
    *,
    plan_path: Path,
    output_path: Path,
    release_id: str,
    human_review_path: Path | None = None,
    freeze: bool = False,
) -> dict[str, Any]:
    plan_file = plan_path.resolve()
    plan = refresh_intervention_sweep_plan(plan_file)
    variant_root = Path(str(plan["output_root"])).resolve()
    static_plan = _read(Path(str(plan["static_sweep_plan"])))
    static_root = Path(str(static_plan["output_root"])).resolve()
    known_job_ids = {str(item["job_id"]) for item in plan["jobs"]}
    decisions, review_provenance = _review_decisions(
        human_review_path,
        expected_sweep_id=str(plan["sweep_id"]),
        known_job_ids=known_job_ids,
    )
    records = []
    tier_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for job in sorted(plan["jobs"], key=lambda item: str(item["job_id"])):
        base = Path(str(job["base_bundle"])).resolve()
        variant = Path(str(job["bundle"])).resolve()
        base_quality_path = base / "quality_report.json"
        variant_quality_path = variant / "quality_report.json"
        base_quality = _read(base_quality_path) if base_quality_path.is_file() else None
        variant_quality = (
            _read(variant_quality_path) if variant_quality_path.is_file() else None
        )
        automated_tier, automated_reasons = _automated_tier(
            job_status=str(job["status"]),
            base_quality=base_quality,
            variant_quality=variant_quality,
        )
        decision = decisions.get(
            str(job["job_id"]),
            {"decision": "unreviewed", "note": "", "updated_at_utc": None},
        )
        tier, reasons = _apply_human_decision(
            automated_tier, automated_reasons, decision
        )
        pair_path = variant / "minimal_pair.json"
        pair = _read(pair_path) if pair_path.is_file() else None
        artifacts = {
            name: _artifact(variant / name)
            for name in _PAIR_VARIANT_MEMBERS
            if (variant / name).is_file()
        }
        record = {
            "intervention_job_id": str(job["job_id"]),
            "scene_model": str(job["scene_model"]),
            "split": job["split"],
            "split_group": job["split_group"],
            "intervention_type": job["intervention_type"],
            "proposal_id": job["proposal_id"],
            "target_source_entity_id": job["target_source_entity_id"],
            "anchor_source_entity_id": job["anchor_source_entity_id"],
            "source_model": job.get("source_model"),
            "replacement_model": job.get("replacement_model"),
            "job_status": job["status"],
            "status_detail": _portable_status_detail(job.get("status_detail")),
            "automated_release_tier": automated_tier,
            "release_tier": tier,
            "release_tier_reasons": reasons,
            "human_review": decision,
            "base_bundle_relpath": _relative(base, static_root, label="base"),
            "variant_bundle_relpath": _relative(variant, variant_root, label="variant"),
            "pair_id": pair.get("pair_id") if pair else None,
            "family_id": pair.get("family_id") if pair else None,
            "episode_split_group": pair.get("split_group") if pair else None,
            "learning_signal": pair.get("learning_signal") if pair else None,
            "answers": {
                "base": pair.get("episodes", {}).get("base", {}).get("answer"),
                "variant": pair.get("episodes", {}).get("variant", {}).get("answer"),
            }
            if pair
            else None,
            "variant_metadata_artifacts": artifacts,
        }
        records.append(record)
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        split = str(job["split"])
        split_counts[split] = split_counts.get(split, 0) + 1
        status = str(job["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    manifest = {
        "schema_version": "episode3d_pair_release.v1",
        "release_id": release_id,
        "dataset_name": "Episode3D-OG",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "intervention_kind": plan["intervention_kind"],
        "source": {
            "simulator": "OmniGibson 3.9.0 / Isaac Sim 5.1.0",
            "assets": "BEHAVIOR-1K 3.9.0",
            "sweep_id": plan["sweep_id"],
            "sweep_plan_sha256": sha256_file(plan_file),
            "static_sweep_plan_sha256": sha256_file(Path(plan["static_sweep_plan"])),
            "object_inventory_sha256": plan["object_inventory_sha256"],
            "split_manifest_sha256": plan["split_manifest_sha256"],
            "planner_policy": plan.get("planner_policy"),
            "execution_policy": plan.get("execution_policy"),
            "human_review": review_provenance,
        },
        "license_boundary": {
            "release_content": "metadata_and_certificates",
            "rendered_media_redistribution": "not_authorized_without_separate_review",
            "source_assets": "user_must_obtain_upstream",
        },
        "path_contract": {
            "base_bundle_relpath_base": "caller-supplied static_data_root",
            "variant_bundle_relpath_base": "caller-supplied intervention_data_root",
        },
        "counts": {
            "pairs": len(records),
            "by_release_tier": dict(sorted(tier_counts.items())),
            "by_job_status": dict(sorted(status_counts.items())),
            "by_split": dict(sorted(split_counts.items())),
        },
        "records": records,
    }
    if freeze:
        _assert_pair_freezable(manifest)
    write_json_atomic(output_path, manifest)
    return manifest


def _assert_pair_freezable(manifest: dict[str, Any]) -> None:
    """Reject a named pair snapshot while production or review remains open."""

    records = manifest.get("records", [])
    if not records:
        raise ValueError("frozen pair release must contain at least one planned pair")
    unfinished = [
        str(record.get("intervention_job_id"))
        for record in records
        if record.get("job_status")
        not in {"certified", "needs_review", "failed"}
    ]
    if unfinished:
        raise ValueError(f"frozen pair release has unfinished jobs: {unfinished}")
    unresolved = [
        str(record.get("intervention_job_id"))
        for record in records
        if record.get("release_tier") == "candidate"
    ]
    if unresolved:
        raise ValueError(f"frozen pair release has unresolved review candidates: {unresolved}")
    if manifest.get("source", {}).get("human_review") is None:
        raise ValueError("frozen pair release requires provenance-tracked human review")
    if not any(record.get("release_tier") == "accepted" for record in records):
        raise ValueError("frozen pair release requires at least one accepted pair")


def verify_pair_release_manifest(
    *, manifest_path: Path, static_data_root: Path, intervention_data_root: Path
) -> dict[str, Any]:
    manifest = _read(manifest_path)
    static_root = static_data_root.resolve()
    intervention_root = intervention_data_root.resolve()
    global_errors = []
    if manifest.get("schema_version") != "episode3d_pair_release.v1":
        global_errors.append("unsupported pair release schema_version")
    if manifest.get("dataset_name") != "Episode3D-OG":
        global_errors.append("unexpected dataset_name")
    intervention_kind = manifest.get("intervention_kind")
    execution_policy = manifest.get("source", {}).get("execution_policy")
    if not isinstance(execution_policy, dict):
        global_errors.append("source.execution_policy is unavailable")
    else:
        mode = execution_policy.get("mode")
        settle_steps = execution_policy.get("settle_steps")
        if mode not in {"physics_settle", "kinematic_counterfactual"}:
            global_errors.append("source.execution_policy mode is invalid")
        elif not isinstance(settle_steps, int) or settle_steps < 0:
            global_errors.append("source.execution_policy settle_steps is invalid")
        elif mode == "physics_settle" and settle_steps < 1:
            global_errors.append("physics_settle requires at least one settle step")
        elif mode == "kinematic_counterfactual" and settle_steps != 0:
            global_errors.append("kinematic_counterfactual requires zero settle steps")
        if intervention_kind == "relation_flip" and mode != "physics_settle":
            global_errors.append("relation_flip must use physics_settle")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("pair release records must be an array")
    identifiers = [str(item.get("intervention_job_id")) for item in records]
    if len(identifiers) != len(set(identifiers)):
        global_errors.append("duplicate intervention_job_id")
    checks = []
    for record in records:
        errors = []
        base = (static_root / str(record["base_bundle_relpath"])).resolve()
        variant = (intervention_root / str(record["variant_bundle_relpath"])).resolve()
        for label, path, root in (
            ("base", base, static_root),
            ("variant", variant, intervention_root),
        ):
            try:
                path.relative_to(root)
            except ValueError:
                errors.append(f"{label} bundle path escapes data root")
        tier = str(record.get("release_tier"))
        metadata = record.get("variant_metadata_artifacts", {})
        if tier in {"accepted", "candidate"}:
            missing = sorted(set(_PAIR_VARIANT_MEMBERS) - set(metadata))
            errors.extend(f"required:{name}" for name in missing)
        if tier == "accepted" and record.get("human_review", {}).get("decision") != "accept":
            errors.append("accepted pair lacks human accept decision")
        for name, artifact in metadata.items():
            relative = Path(str(artifact.get("path", "")))
            if relative.is_absolute() or len(relative.parts) != 1 or relative.name != name:
                errors.append(f"unsafe-artifact-path:{name}")
                continue
            path = variant / relative
            if not path.is_file():
                errors.append(f"missing:{name}")
            elif path.stat().st_size != int(artifact["byte_size"]):
                errors.append(f"size:{name}")
            elif sha256_file(path) != artifact["sha256"]:
                errors.append(f"sha256:{name}")
        certificate_path = variant / "minimal_pair.json"
        if certificate_path.is_file():
            certificate = _read(certificate_path)
            if certificate.get("pair_id") != record.get("pair_id"):
                errors.append("pair_id does not match certificate")
            if certificate.get("split_group") != record.get("episode_split_group"):
                errors.append("episode_split_group does not match certificate")
        if record.get("split_group") != f"scene:{record.get('scene_model')}":
            errors.append("registry split_group is not locked to scene_model")
        checks.append(
            {
                "intervention_job_id": record["intervention_job_id"],
                "passed": not errors,
                "errors": errors,
            }
        )
    actual_counts: dict[str, int] = {}
    for record in records:
        tier = str(record.get("release_tier"))
        actual_counts[tier] = actual_counts.get(tier, 0) + 1
    declared_counts = manifest.get("counts", {}).get("by_release_tier")
    if declared_counts != dict(sorted(actual_counts.items())):
        global_errors.append("counts.by_release_tier does not match records")
    failed = [item for item in checks if not item["passed"]]
    return {
        "schema_version": "episode3d_pair_release_verification.v1",
        "release_id": manifest.get("release_id"),
        "passed": not failed and not global_errors,
        "record_count": len(records),
        "failed_count": len(failed),
        "global_errors": global_errors,
        "checks": checks,
    }


class EpisodePairIndex:
    """Dependency-free, training-safe loader for a pair release."""

    def __init__(
        self, manifest_path: Path, static_data_root: Path, intervention_data_root: Path
    ) -> None:
        self.manifest = _read(manifest_path.resolve())
        if self.manifest.get("schema_version") != "episode3d_pair_release.v1":
            raise ValueError("unsupported pair release schema_version")
        if self.manifest.get("dataset_name") != "Episode3D-OG":
            raise ValueError("unexpected pair release dataset_name")
        if not isinstance(self.manifest.get("records"), list):
            raise ValueError("pair release records must be an array")
        self.static_data_root = static_data_root.resolve()
        self.intervention_data_root = intervention_data_root.resolve()

    def records(
        self, *, split: str | None = None, release_tier: str | None = "accepted"
    ) -> Iterator[dict[str, Any]]:
        for record in self.manifest["records"]:
            if split is not None and record["split"] != split:
                continue
            if release_tier is not None and record["release_tier"] != release_tier:
                continue
            yield record

    def bundle_paths(self, record: dict[str, Any]) -> tuple[Path, Path]:
        base = (self.static_data_root / record["base_bundle_relpath"]).resolve()
        variant = (
            self.intervention_data_root / record["variant_bundle_relpath"]
        ).resolve()
        base.relative_to(self.static_data_root)
        variant.relative_to(self.intervention_data_root)
        return base, variant

    def load_certificate(self, record: dict[str, Any]) -> dict[str, Any]:
        _, variant = self.bundle_paths(record)
        return _read(variant / "minimal_pair.json")

    def load_episodes(
        self, record: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load the immutable base and its intervention variant together."""

        base, variant = self.bundle_paths(record)
        return (
            _read(base / "spatial_episode.json"),
            _read(variant / "spatial_episode.json"),
        )

    def load_pair(self, record: dict[str, Any]) -> dict[str, Any]:
        """Return one training-ready pair without hiding its certificate."""

        base_episode, variant_episode = self.load_episodes(record)
        return {
            "record": record,
            "base_episode": base_episode,
            "variant_episode": variant_episode,
            "certificate": self.load_certificate(record),
        }
