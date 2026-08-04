from pathlib import Path
from types import SimpleNamespace

from omnigibson_episode import derive


def test_refresh_derived_processes_only_terminal_bundles(
    tmp_path: Path, monkeypatch
) -> None:
    ready = tmp_path / "ready"
    ready.mkdir()
    plan = {
        "sweep_id": "sweep-a",
        "output_root": str(tmp_path),
        "status_counts": {"passed": 1, "failed": 1},
        "jobs": [
            {
                "job_id": "ready_seed0",
                "scene_model": "ready",
                "bundle": str(ready),
                "status": "passed",
            },
            {
                "job_id": "bad_seed0",
                "scene_model": "bad",
                "bundle": str(tmp_path / "bad"),
                "status": "failed",
            },
        ],
    }
    monkeypatch.setattr(derive, "refresh_sweep_plan", lambda _: plan)
    monkeypatch.setattr(
        derive,
        "compile_bundle",
        lambda **_: SimpleNamespace(episode_id="episode-a"),
    )
    monkeypatch.setattr(
        derive,
        "inspect_bundle",
        lambda _: {"integrity_status": "pass", "visual_status": "pass"},
    )
    monkeypatch.setattr(
        derive,
        "compile_reasoning_tasks",
        lambda _: {"task_count": 8, "coverage_status": "complete"},
    )
    monkeypatch.setattr(
        derive,
        "audit_reasoning_bundle",
        lambda _: {
            "reasoning_status": "complete",
            "evidence_eligibility": "pass",
            "evidence_summary": {"observed_entity_count": 12},
            "capability_matrix": {
                "G": {"evidence_supported": True, "sampled": True},
                "P": {"evidence_supported": True, "sampled": False},
            },
            "gaps": [{"capability": "P", "reason": "fixture"}],
        },
    )

    report = derive.refresh_derived_bundles(tmp_path / "plan.json")

    assert report["counts"] == {"refreshed": 1, "skipped": 1}
    assert report["records"][0]["episode_id"] == "episode-a"
    assert report["coverage_summary"]["task_count"] == {
        "total": 8,
        "minimum_per_acquisition": 8,
        "maximum_per_acquisition": 8,
    }
    assert report["coverage_summary"]["capability_counts"]["G"] == {
        "evidence_supported": 1,
        "sampled": 1,
    }
    assert report["coverage_summary"]["missing_capability_scenes"] == {
        "P": ["ready"]
    }
    assert report["coverage_summary"]["sampled_without_evidence_scenes"] == {}
    assert (tmp_path / "derived_refresh_report.json").is_file()
