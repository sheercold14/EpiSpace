import json
from pathlib import Path

import pytest

from omnigibson_episode import pair_release
from omnigibson_episode.pair_release import (
    EpisodePairIndex,
    _assert_pair_freezable,
    build_pair_release_manifest,
    verify_pair_release_manifest,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_pair_release_is_root_relative_reviewed_and_verifiable(
    tmp_path: Path, monkeypatch
) -> None:
    static_root = tmp_path / "static"
    variant_root = tmp_path / "variants"
    base = static_root / "bundles" / "scene_seed0"
    variant = variant_root / "bundles" / "scene_seed0" / "flip"
    _write(
        base / "quality_report.json",
        {"integrity_status": "pass", "visual_status": "pass"},
    )
    _write(base / "spatial_episode.json", {"episode_id": "base-episode"})
    pair = {
        "status": "certified",
        "pair_id": "pair-a",
        "family_id": "family-a",
        "split_group": "scene:scene",
        "learning_signal": "answer_flip",
        "episodes": {
            "base": {"answer": "left_of"},
            "variant": {"answer": "right_of"},
        },
    }
    for name in pair_release._PAIR_VARIANT_MEMBERS:
        payload = pair if name == "minimal_pair.json" else {"name": name}
        if name == "quality_report.json":
            payload = {"integrity_status": "pass", "visual_status": "pass"}
        _write(variant / name, payload)
    static_plan_path = static_root / "sweep_plan.json"
    _write(static_plan_path, {"output_root": str(static_root)})
    plan_path = variant_root / "sweep_plan.json"
    plan = {
        "sweep_id": "flip-sweep",
        "intervention_kind": "model_swap",
        "output_root": str(variant_root),
        "static_sweep_plan": str(static_plan_path),
        "object_inventory_sha256": "a" * 64,
        "split_manifest_sha256": "b" * 64,
        "planner_policy": {"minimum_nonoverlap_clearance_m": 0.15},
        "execution_policy": {"mode": "kinematic_counterfactual", "settle_steps": 0},
        "jobs": [
            {
                "job_id": "scene_seed0__flip",
                "scene_model": "scene",
                "split": "train",
                "split_group": "scene:scene",
                "intervention_type": "single_object_relation_flip",
                "proposal_id": "flip",
                "target_source_entity_id": "chair_0",
                "anchor_source_entity_id": "table_0",
                "base_bundle": str(base),
                "bundle": str(variant),
                "status": "certified",
                "status_detail": None,
            }
        ],
    }
    _write(plan_path, plan)
    monkeypatch.setattr(pair_release, "refresh_intervention_sweep_plan", lambda _: plan)
    review_path = tmp_path / "review.json"
    _write(
        review_path,
        {
            "schema_version": "omnigibson_intervention_human_review.v1",
            "sweep_id": "flip-sweep",
            "reviewer": "reviewer",
            "decisions": {
                "scene_seed0__flip": {
                    "decision": "accept",
                    "note": "single visible change",
                    "updated_at_utc": "2026-07-14T00:00:00Z",
                }
            },
        },
    )
    manifest_path = tmp_path / "pair-release.json"

    manifest = build_pair_release_manifest(
        plan_path=plan_path,
        output_path=manifest_path,
        release_id="pair-release-v1",
        human_review_path=review_path,
    )
    verification = verify_pair_release_manifest(
        manifest_path=manifest_path,
        static_data_root=static_root,
        intervention_data_root=variant_root,
    )
    index = EpisodePairIndex(manifest_path, static_root, variant_root)
    record = next(index.records())
    loaded = index.load_pair(record)

    assert manifest["counts"]["by_release_tier"] == {"accepted": 1}
    assert manifest["source"]["planner_policy"] == {
        "minimum_nonoverlap_clearance_m": 0.15
    }
    assert manifest["source"]["execution_policy"] == {
        "mode": "kinematic_counterfactual",
        "settle_steps": 0,
    }
    assert manifest["source"]["split_manifest_sha256"] == "b" * 64
    assert verification["passed"]
    assert index.load_certificate(record)["pair_id"] == "pair-a"
    assert loaded["base_episode"]["episode_id"] == "base-episode"
    assert loaded["variant_episode"]["name"] == "spatial_episode.json"
    assert loaded["certificate"]["learning_signal"] == "answer_flip"


def test_pair_verifier_rejects_incoherent_execution_policy(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write(
        manifest_path,
        {
            "schema_version": "episode3d_pair_release.v1",
            "release_id": "bad-policy",
            "dataset_name": "Episode3D-OG",
            "intervention_kind": "relation_flip",
            "source": {
                "execution_policy": {
                    "mode": "kinematic_counterfactual",
                    "settle_steps": 0,
                }
            },
            "counts": {"by_release_tier": {}},
            "records": [],
        },
    )

    report = verify_pair_release_manifest(
        manifest_path=manifest_path,
        static_data_root=tmp_path / "static",
        intervention_data_root=tmp_path / "variants",
    )

    assert not report["passed"]
    assert "relation_flip must use physics_settle" in report["global_errors"]


def test_pair_freeze_gate_rejects_unfinished_or_unreviewed_registry() -> None:
    manifest = {
        "source": {"human_review": {"reviewer": "reviewer"}},
        "records": [
            {
                "intervention_job_id": "pair-a",
                "job_status": "certified",
                "release_tier": "accepted",
            },
            {
                "intervention_job_id": "pair-b",
                "job_status": "failed",
                "release_tier": "excluded",
            },
        ],
    }
    _assert_pair_freezable(manifest)

    manifest["records"][1]["job_status"] = "running"
    with pytest.raises(ValueError, match="unfinished"):
        _assert_pair_freezable(manifest)
    manifest["records"][1]["job_status"] = "failed"
    manifest["records"][0]["release_tier"] = "candidate"
    with pytest.raises(ValueError, match="unresolved"):
        _assert_pair_freezable(manifest)

    manifest["records"][0]["release_tier"] = "audit_only"
    with pytest.raises(ValueError, match="at least one accepted"):
        _assert_pair_freezable(manifest)


def test_pair_loader_rejects_non_array_records(tmp_path: Path) -> None:
    manifest_path = tmp_path / "pair-release.json"
    _write(
        manifest_path,
        {
            "schema_version": "episode3d_pair_release.v1",
            "dataset_name": "Episode3D-OG",
            "records": {},
        },
    )

    with pytest.raises(ValueError, match="pair release records must be an array"):
        EpisodePairIndex(manifest_path, tmp_path / "static", tmp_path / "variants")
