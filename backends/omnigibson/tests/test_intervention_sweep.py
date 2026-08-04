from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from omnigibson_episode import intervention_runner
from omnigibson_episode.intervention_sweep import intervention_bundle_status
from omnigibson_episode.pair_review import build_intervention_review


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_intervention_status_separates_certification_and_visual_review(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "pair"
    _write(bundle / "minimal_pair.json", {"status": "certified"})
    _write(
        bundle / "quality_report.json",
        {"integrity_status": "pass", "visual_status": "warning"},
    )
    assert intervention_bundle_status(bundle)[0] == "needs_review"

    _write(
        bundle / "quality_report.json",
        {"integrity_status": "pass", "visual_status": "pass"},
    )
    assert intervention_bundle_status(bundle) == ("certified", None)

    base = tmp_path / "base"
    _write(
        base / "quality_report.json",
        {"integrity_status": "pass", "visual_status": "warning"},
    )
    assert intervention_bundle_status(bundle, base)[0] == "needs_review"


def test_intervention_status_preserves_certificate_failure(tmp_path: Path) -> None:
    bundle = tmp_path / "pair"
    _write(bundle / "render_report.json", {"status": "success"})
    _write(
        bundle / "minimal_pair_failure.json",
        {
            "status": "failed",
            "error": {
                "message": "minimal-pair certification failed: single_mutable_entity"
            },
            "checks": [
                {"name": "single_mutable_entity", "passed": False}
            ],
        },
    )

    assert intervention_bundle_status(bundle) == (
        "failed",
        "minimal-pair certification failed: single_mutable_entity",
    )


def test_intervention_workers_keep_stable_gpu_assignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = [
        {"job_id": f"job-{index}", "scene_model": f"scene-{index}", "status": "pending"}
        for index in range(7)
    ]
    plan = {"jobs": jobs}
    thread_gpu: dict[int, int] = {}
    lock = threading.Lock()

    def fake_refresh(_: Path) -> dict:
        return plan

    def fake_run(**kwargs: object) -> dict[str, object]:
        gpu_id = int(kwargs["gpu_id"])
        thread_id = threading.get_ident()
        with lock:
            previous = thread_gpu.setdefault(thread_id, gpu_id)
            assert previous == gpu_id
        job = kwargs["job"]
        assert isinstance(job, dict)
        return {"job_id": job["job_id"], "status": "certified"}

    monkeypatch.setattr(intervention_runner, "refresh_intervention_sweep_plan", fake_refresh)
    monkeypatch.setattr(intervention_runner, "run_intervention_job", fake_run)
    results = intervention_runner.run_intervention_sweep(
        plan_path=tmp_path / "plan.json",
        gpu_ids=[0, 1],
        workers=2,
    )

    assert len(results) == 7
    assert set(thread_gpu.values()).issubset({0, 1})


def test_intervention_sweep_filters_exact_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = {
        "jobs": [
            {
                "job_id": "job-a",
                "scene_model": "scene",
                "proposal_id": "proposal-a",
                "status": "pending",
            },
            {
                "job_id": "job-b",
                "scene_model": "scene",
                "proposal_id": "proposal-b",
                "status": "pending",
            },
        ]
    }

    monkeypatch.setattr(
        intervention_runner, "refresh_intervention_sweep_plan", lambda _: plan
    )
    monkeypatch.setattr(
        intervention_runner,
        "run_intervention_job",
        lambda **kwargs: {
            "job_id": kwargs["job"]["job_id"],
            "status": "certified",
        },
    )
    results = intervention_runner.run_intervention_sweep(
        plan_path=tmp_path / "plan.json",
        gpu_ids=[0],
        workers=1,
        proposal_ids={"proposal-b"},
    )

    assert [item["job_id"] for item in results] == ["job-b"]


def test_intervention_sweep_treats_needs_review_as_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = {
        "jobs": [
            {
                "job_id": "reviewed-by-machine",
                "scene_model": "scene",
                "proposal_id": "proposal-review",
                "status": "needs_review",
            },
            {
                "job_id": "pending",
                "scene_model": "scene",
                "proposal_id": "proposal-pending",
                "status": "pending",
            },
        ]
    }
    executed: list[str] = []

    monkeypatch.setattr(
        intervention_runner, "refresh_intervention_sweep_plan", lambda _: plan
    )

    def fake_run(**kwargs: object) -> dict[str, str]:
        job = kwargs["job"]
        assert isinstance(job, dict)
        executed.append(str(job["job_id"]))
        return {"job_id": str(job["job_id"]), "status": "certified"}

    monkeypatch.setattr(intervention_runner, "run_intervention_job", fake_run)
    results = intervention_runner.run_intervention_sweep(
        plan_path=tmp_path / "plan.json",
        gpu_ids=[0],
        workers=1,
    )

    assert executed == ["pending"]
    assert [item["job_id"] for item in results] == ["pending"]


def test_pair_review_renders_matched_base_variant_views(tmp_path: Path) -> None:
    root = tmp_path / "sweep"
    base = tmp_path / "static" / "scene"
    variant = root / "bundles" / "pair"
    for bundle in (base, variant):
        (bundle / "preview").mkdir(parents=True)
        (bundle / "preview" / "view-000.png").write_bytes(b"png")
        (bundle / "preview.html").write_text("preview", encoding="utf-8")
        _write(
            bundle / "quality_report.json",
            {"integrity_status": "pass", "visual_status": "pass"},
        )
    _write(
        variant / "minimal_pair.json",
        {
            "status": "certified",
            "learning_signal": "answer_invariance",
            "episodes": {
                "base": {"answer": "left_of"},
                "variant": {"answer": "left_of"},
            },
            "evidence": {
                "base": {"target_view_ids": ["view-000"]},
                "variant": {"target_view_ids": ["view-000"]},
            },
            "checks": [{"name": "answers_invariant", "passed": True}],
        },
    )
    proposal = root / "proposals" / "scene.json"
    _write(
        proposal,
        {
            "intervention_type": "single_object_model_swap",
            "proposals": [
                {
                    "proposal_id": "swap",
                    "target_source_entity_id": "chair_0",
                    "anchor_source_entity_id": "table_0",
                    "source_model": "aaaaaa",
                    "replacement_model": "bbbbbb",
                    "execution_mode": "kinematic_counterfactual",
                    "settle_steps": 0,
                    "baseline_overlap_count": 0,
                    "minimum_nonoverlap_clearance_m": 0.15,
                }
            ],
        },
    )
    plan = root / "sweep_plan.json"
    _write(
        plan,
        {
            "sweep_id": "model-swap-test",
            "intervention_kind": "model_swap",
            "output_root": str(root),
            "jobs": [
                {
                    "job_id": "pair",
                    "scene_model": "scene",
                    "proposal_id": "swap",
                    "proposal_plan": str(proposal),
                    "base_bundle": str(base),
                    "bundle": str(variant),
                    "receipt": str(root / "job_status" / "pair.json"),
                    "status": "certified",
                }
            ],
        },
    )

    page = build_intervention_review(plan)
    text = page.read_text(encoding="utf-8")

    assert "answer_invariance" in text
    assert "view-000" in text
    assert "aaaaaa" in text and "bbbbbb" in text
    assert "kinematic_counterfactual" in text
    assert "clearance" in text
    assert '"actionable_count": 1' in text
    assert '"review_actionable": true' in text
    assert 'value="actionable"' in text
    assert "自动排除原因" in text
    assert "该记录已被自动门禁排除" in text
