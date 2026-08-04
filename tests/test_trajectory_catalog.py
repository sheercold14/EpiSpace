from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "web" / "data" / "trajectory_catalog.v1.json"


def _catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def test_catalog_contains_all_rendered_trajectory_families() -> None:
    catalog = _catalog()
    assert catalog["schema_version"] == "episode3d.trajectory_catalog.v2"
    assert catalog["summary"]["real_trajectory_count"] == 6
    assert catalog["summary"]["newly_acquired_trajectory_count"] == 276
    trajectories = {item["trajectory_class"]: item for item in catalog["trajectories"]}
    assert trajectories.keys() == {"T1", "T3", "T4", "T7", "T8", "T10"}
    assert trajectories["T1"]["view_count"] == 11
    assert trajectories["T3"]["view_count"] == 6
    assert trajectories["T4"]["view_count"] == 8
    assert trajectories["T7"]["view_count"] == 3
    assert trajectories["T8"]["view_count"] == 6
    assert trajectories["T10"]["view_count"] == 3
    assert all(item["model_visible"] == ["rgb"] for item in trajectories.values())


def test_collection_accounting_matches_completed_sweeps() -> None:
    catalog = _catalog()
    summary = catalog["summary"]
    assert summary["acquisition_job_count"] == 322
    assert summary["passed_episode_count"] == 193
    assert summary["needs_review_episode_count"] == 60
    assert summary["failed_episode_count"] == 69
    assert summary["browsable_episode_count"] == 253
    assert len(catalog["collection_jobs"]) == 322
    collections = {
        item["trajectory_class"]: item for item in catalog["collections"]
    }
    assert (
        collections["T3"]["passed"],
        collections["T3"]["needs_review"],
        collections["T3"]["failed"],
    ) == (83, 7, 2)
    assert (
        collections["T4"]["passed"],
        collections["T4"]["needs_review"],
        collections["T4"]["failed"],
    ) == (10, 3, 33)
    assert (
        collections["T7"]["passed"],
        collections["T7"]["needs_review"],
        collections["T7"]["failed"],
    ) == (43, 1, 2)
    assert (
        collections["T8"]["passed"],
        collections["T8"]["needs_review"],
        collections["T8"]["failed"],
    ) == (8, 14, 24)
    assert (
        collections["T10"]["passed"],
        collections["T10"]["needs_review"],
        collections["T10"]["failed"],
    ) == (21, 17, 8)
    assert (
        collections["T9"]["passed"],
        collections["T9"]["needs_review"],
        collections["T9"]["failed"],
    ) == (14, 10, 37)


def test_every_complete_bundle_has_a_lazy_loaded_detail_and_thumbnail() -> None:
    catalog = _catalog()
    browsable = [job for job in catalog["collection_jobs"] if job["detail_url"]]
    rejected = [job for job in catalog["collection_jobs"] if job["status"] == "failed"]
    assert len(browsable) == 253
    assert len(rejected) == 69
    assert all(job["status"] in {"passed", "needs_review"} for job in browsable)
    assert all(job["detail_url"] is None for job in rejected)
    assert all(job["thumbnail_url"] is None for job in rejected)
    for job in browsable:
        detail_path = ROOT / "web" / job["detail_url"]
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        assert detail["trajectory_id"] == job["job_id"]
        assert detail["status"] == job["status"]
        assert detail["view_count"] == job["view_count"]
        assert job["thumbnail_url"] == detail["views"][0]["media"]["rgb"]
        for view in detail["views"]:
            for relative in view["media"].values():
                assert (ROOT / "web" / relative).is_file(), relative


def test_t3_is_exactly_zero_baseline_and_passes_evidence_gate() -> None:
    t3 = next(
        item for item in _catalog()["trajectories"] if item["trajectory_class"] == "T3"
    )
    gate = t3["quality"]["gates"]["T3"]
    assert gate["status"] == "pass"
    assert gate["maximum_translational_baseline_m"] == 0.0
    assert gate["panorama_core_entity_count"] >= 8
    assert gate["rear_only_core_entity_count"] >= 2
    assert {tuple(view["position_m"]) for view in t3["views"]} == {(-0.1, -1.3, 0.0)}
    assert [view["yaw_deg"] for view in t3["views"]] == [
        0.0,
        60.0,
        120.0,
        180.0,
        240.0,
        300.0,
    ]


def test_derived_t2_and_t5_records_remain_evidence_backed() -> None:
    derived = _catalog()["derived"]
    for subset in derived["T2"]["selected_subsets"]:
        assert set(subset["view_ids"]) == set(subset["legal_shuffle_view_ids"])
        if subset["overlap_class"] == "zero":
            assert set(subset["all_pair_common_core_entity_counts"]) == {0}
            assert subset["answerability"] == "unknown"
    for pair in derived["T5"]["selected_pairs"]:
        overlap = pair["overlap_class"]
        common = pair["common_core_entity_count"]
        assert (overlap == "high" and common >= 5) or (
            overlap == "low" and common in {1, 2}
        ) or (overlap == "zero" and common == 0)


def test_web_media_and_legacy_t6_boundary_are_explicit() -> None:
    catalog = _catalog()
    for trajectory in catalog["trajectories"]:
        for view in trajectory["views"]:
            for relative in view["media"].values():
                assert (ROOT / "web" / relative).is_file(), relative
    audit = catalog["static_sweep_t6_audit"]
    assert audit["scene_count"] == 46
    assert audit["ordered_room_sequence_count"] == 0
    assert audit["bridge_certified_count"] == 0
    assert catalog["derived"]["T6"]["audit_status"] == "partial"


def test_only_passed_jobs_supply_collection_representatives() -> None:
    catalog = _catalog()
    job_status = {item["job_id"]: item["status"] for item in catalog["collection_jobs"]}
    for trajectory in catalog["trajectories"]:
        source_job_id = trajectory.get("source_job_id")
        if source_job_id is not None:
            assert job_status[source_job_id] == "passed"
            gate = trajectory["quality"]["gates"][trajectory["trajectory_class"]]
            assert gate["status"] == "pass"
