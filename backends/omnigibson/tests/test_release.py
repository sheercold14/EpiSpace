from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnigibson_episode.io import sha256_file
from omnigibson_episode.release import (
    EpisodeDatasetIndex,
    _apply_human_decision,
    _assert_freezable,
    _human_review_decisions,
    _portable_status_detail,
    _record_tier,
    verify_release_manifest,
)


def test_public_status_detail_redacts_producer_paths() -> None:
    detail = "command exceeded: ['/data/private/scripts/run.sh', '--flag']"

    sanitized = _portable_status_detail(detail)

    assert sanitized == "command exceeded: ['<local-path>', '--flag']"
    assert "/data/private" not in sanitized


def test_release_loader_and_verifier_are_root_relative(tmp_path: Path) -> None:
    root = tmp_path / "data"
    bundle = root / "bundles" / "scene_seed0"
    bundle.mkdir(parents=True)
    episode = {"episode_id": "episode-a"}
    episode_path = bundle / "spatial_episode.json"
    episode_path.write_text(json.dumps(episode), encoding="utf-8")
    manifest = {
        "schema_version": "episode3d_release.v1",
        "release_id": "test-v1",
        "dataset_name": "Episode3D-OG",
        "counts": {"by_release_tier": {"audit_only": 1}},
        "records": [
            {
                "acquisition_id": "scene_seed0",
                "split": "train",
                "release_tier": "audit_only",
                "bundle_relpath": "bundles/scene_seed0",
                "metadata_artifacts": {
                    "spatial_episode.json": {
                        "path": "spatial_episode.json",
                        "byte_size": episode_path.stat().st_size,
                        "sha256": sha256_file(episode_path),
                    }
                },
            }
        ],
    }
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = verify_release_manifest(manifest_path=manifest_path, data_root=root)
    index = EpisodeDatasetIndex(manifest_path, root)
    record = next(index.records(split="train", release_tier="audit_only"))

    assert report["passed"]
    assert index.bundle_path(record) == bundle
    assert index.load_episode(record) == episode


def test_release_loader_defaults_to_human_accepted_records(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "episode3d_release.v1",
        "release_id": "test-v1",
        "dataset_name": "Episode3D-OG",
        "records": [
            {"acquisition_id": "accepted", "split": "train", "release_tier": "accepted"},
            {"acquisition_id": "candidate", "split": "train", "release_tier": "candidate"},
            {"acquisition_id": "audit", "split": "test", "release_tier": "audit_only"},
            {"acquisition_id": "excluded", "split": "test", "release_tier": "excluded"},
        ],
    }
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    index = EpisodeDatasetIndex(manifest_path, tmp_path)

    assert [record["acquisition_id"] for record in index.records()] == ["accepted"]
    assert len(list(index.records(release_tier=None))) == 4


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (
            {"schema_version": "unknown", "dataset_name": "Episode3D-OG", "records": []},
            "unsupported release schema_version",
        ),
        (
            {"schema_version": "episode3d_release.v1", "dataset_name": "other", "records": []},
            "unexpected release dataset_name",
        ),
        (
            {
                "schema_version": "episode3d_release.v1",
                "dataset_name": "Episode3D-OG",
                "records": {},
            },
            "release records must be an array",
        ),
    ],
)
def test_release_loader_rejects_incompatible_manifests(
    tmp_path: Path, manifest: dict[str, object], message: str
) -> None:
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        EpisodeDatasetIndex(manifest_path, tmp_path)


def test_human_review_promotes_only_automated_candidates(tmp_path: Path) -> None:
    review_path = tmp_path / "review.json"
    review_path.write_text(
        json.dumps(
            {
                "schema_version": "omnigibson_human_review.v1",
                "sweep_id": "sweep-a",
                "reviewer": "reviewer@example.org",
                "generated_at_utc": "2026-07-14T00:00:00Z",
                "decisions": {
                    "job-a": {
                        "decision": "accept",
                        "note": "clear",
                        "updated_at_utc": "2026-07-14T00:00:00Z",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    decisions, provenance = _human_review_decisions(
        review_path,
        expected_sweep_id="sweep-a",
        known_job_ids={"job-a"},
    )

    assert provenance is not None
    assert provenance["reviewer"] == "reviewer@example.org"
    assert _apply_human_decision("candidate", ["pending"], decisions["job-a"])[0] == "accepted"
    assert _apply_human_decision("audit_only", ["weak"], decisions["job-a"])[0] == "audit_only"


def test_visual_warning_is_reviewable_but_integrity_failure_is_not() -> None:
    warning_quality = {"integrity_status": "pass", "visual_status": "warning"}
    eligible_audit = {"evidence_eligibility": "pass"}
    tier, reasons = _record_tier(
        job_status="needs_review", quality=warning_quality, audit=eligible_audit
    )

    assert tier == "candidate"
    assert "visual warning" in reasons[0]
    assert _record_tier(
        job_status="needs_review",
        quality={"integrity_status": "fail", "visual_status": "warning"},
        audit=eligible_audit,
    )[0] == "audit_only"


def test_freeze_gate_rejects_unfinished_or_unreviewed_registry() -> None:
    records = [
        {
            "acquisition_id": f"job-{index}",
            "static_job_status": "passed",
            "release_tier": "accepted",
        }
        for index in range(46)
    ]
    manifest = {
        "source": {
            "human_review": {"reviewer": "reviewer"},
            "aggregate_audit": {
                "counts": {"refreshed": 46, "skipped": 0, "failed": 0},
                "coverage_summary": {
                    "refreshed_acquisition_count": 46,
                    "sampled_without_evidence_scenes": {},
                },
            },
        },
        "records": records,
    }
    _assert_freezable(manifest)

    records[0]["static_job_status"] = "running"
    with pytest.raises(ValueError, match="unfinished"):
        _assert_freezable(manifest)
    records[0]["static_job_status"] = "passed"
    records[0]["release_tier"] = "candidate"
    with pytest.raises(ValueError, match="unresolved"):
        _assert_freezable(manifest)

    records[0]["release_tier"] = "accepted"
    manifest["source"]["aggregate_audit"]["counts"]["refreshed"] = 45
    with pytest.raises(ValueError, match="all 46"):
        _assert_freezable(manifest)


def test_human_review_rejects_unknown_acquisition(tmp_path: Path) -> None:
    review_path = tmp_path / "review.json"
    review_path.write_text(
        json.dumps(
            {
                "schema_version": "omnigibson_human_review.v1",
                "sweep_id": "sweep-a",
                "reviewer": "reviewer",
                "decisions": {"unknown": {"decision": "accept", "note": ""}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown acquisition"):
        _human_review_decisions(
            review_path,
            expected_sweep_id="sweep-a",
            known_job_ids={"job-a"},
        )
