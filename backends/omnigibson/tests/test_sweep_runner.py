import json
import threading
from pathlib import Path

import pytest

from omnigibson_episode import sweep_runner


def test_terminal_bundle_reconciles_stale_receipt_without_rerender(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "job_status" / "scene_seed0.json"
    receipt.parent.mkdir()
    receipt.write_text(
        json.dumps({"status": "failed", "detail": "old timeout", "attempt": 2}),
        encoding="utf-8",
    )
    job = {
        "job_id": "scene_seed0",
        "scene_model": "scene",
        "bundle": str(tmp_path / "bundle"),
        "receipt": str(receipt),
    }
    monkeypatch.setattr(
        sweep_runner,
        "_bundle_status",
        lambda _: ("needs_review", "visual warning"),
    )
    monkeypatch.setattr(
        sweep_runner,
        "_run_command",
        lambda *_, **__: pytest.fail("terminal bundle must not rerender"),
    )

    result = sweep_runner.run_job(
        job=job,
        base_recipe=tmp_path / "base.yaml",
        project_root=tmp_path,
        gpu_id=0,
        retry_failed=True,
        acquisition_timeout_seconds=60,
    )
    reconciled = json.loads(receipt.read_text(encoding="utf-8"))

    assert result["status"] == "needs_review"
    assert reconciled["status"] == "needs_review"
    assert reconciled["attempt"] == 2
    assert reconciled["reconciliation"]["previous_status"] == "failed"


def test_workers_keep_a_stable_gpu_assignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = [
        {
            "job_id": f"job-{index}",
            "scene_model": f"scene-{index}",
            "status": "pending",
        }
        for index in range(9)
    ]
    plan = {
        "base_recipe": str(tmp_path / "base.yaml"),
        "jobs": jobs,
    }
    thread_gpu: dict[int, int] = {}
    lock = threading.Lock()

    def fake_refresh(_: Path) -> dict:
        return plan

    def fake_run_job(**kwargs: object) -> dict[str, object]:
        gpu_id = int(kwargs["gpu_id"])
        thread_id = threading.get_ident()
        with lock:
            previous = thread_gpu.setdefault(thread_id, gpu_id)
            assert previous == gpu_id
        job = kwargs["job"]
        assert isinstance(job, dict)
        return {"job_id": job["job_id"], "status": "passed"}

    monkeypatch.setattr(sweep_runner, "refresh_sweep_plan", fake_refresh)
    monkeypatch.setattr(sweep_runner, "run_job", fake_run_job)
    results = sweep_runner.run_sweep(
        plan_path=tmp_path / "plan.json",
        gpu_ids=[0, 1, 2],
        workers=3,
    )
    assert len(results) == 9
    assert set(thread_gpu.values()).issubset({0, 1, 2})
