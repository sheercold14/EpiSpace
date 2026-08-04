"""Execute intervention jobs with stable one-worker-per-GPU scheduling."""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from omnigibson_episode.intervention_sweep import (
    intervention_bundle_status,
    refresh_intervention_sweep_plan,
)
from omnigibson_episode.io import write_json_atomic
from omnigibson_episode.sweep_runner import _run_command, _timestamp


def _receipt(job: dict[str, Any], **updates: Any) -> None:
    write_json_atomic(
        Path(job["receipt"]),
        {
            "schema_version": "omnigibson_intervention_job_receipt.v1",
            "job_id": job["job_id"],
            "scene_model": job["scene_model"],
            "proposal_id": job["proposal_id"],
            "intervention_type": job.get("intervention_type"),
            "execution_mode": job.get("execution_mode"),
            "settle_steps": job.get("settle_steps", 12),
            **updates,
        },
    )


def run_intervention_job(
    *,
    job: dict[str, Any],
    project_root: Path,
    gpu_id: int,
    retry_failed: bool,
    acquisition_timeout_seconds: int,
) -> dict[str, Any]:
    bundle = Path(job["bundle"])
    base_bundle = Path(job["base_bundle"])
    status, detail = intervention_bundle_status(bundle, base_bundle)
    if status in {"certified", "needs_review"}:
        return {
            "job_id": job["job_id"],
            "status": status,
            "detail": (
                "already certified"
                if status == "certified"
                else "already certified; human visual review pending"
            ),
        }
    if status in {"failed", "incomplete"} and not retry_failed:
        return {"job_id": job["job_id"], "status": status, "detail": detail}
    root = Path(job["receipt"]).parents[1]
    log = root / "logs" / f"{job['job_id']}.log"
    started = _timestamp()
    _receipt(
        job,
        status="running",
        detail=None,
        gpu_id=gpu_id,
        pid=os.getpid(),
        started_at=started,
        log=str(log),
    )
    env = os.environ.copy()
    env["EPISODE_PROJECT_ROOT"] = str(project_root)
    env["OMNIGIBSON_DATA_PATH"] = str(project_root / ".data" / "omnigibson")
    pythonpath = [str(project_root / "src"), str(project_root.parent / "habitat" / "src")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    try:
        status, _ = intervention_bundle_status(bundle, base_bundle)
        if status != "acquired":
            command = [
                "bash",
                str(project_root / "scripts" / "run_in_omnigibson.sh"),
                "--accept-eula",
                "python",
                "-m",
                "omnigibson_episode.cli",
                "acquire-intervention",
                "--recipe",
                str(job["recipe"]),
                "--base-bundle",
                str(job["base_bundle"]),
                "--plan",
                str(job["proposal_plan"]),
                "--proposal-id",
                str(job["proposal_id"]),
                "--output",
                str(bundle),
                "--gpu-id",
                str(gpu_id),
                "--headless",
                "--settle-steps",
                str(int(job.get("settle_steps", 12))),
            ]
            stale = any(bundle.parent.glob(f".{bundle.name}.staging.*"))
            if bundle.exists() or retry_failed or stale:
                command.append("--overwrite")
            _run_command(
                command,
                log_path=log,
                env=env,
                timeout_seconds=acquisition_timeout_seconds,
            )
            status, detail = intervention_bundle_status(bundle, base_bundle)
            if status != "acquired":
                raise RuntimeError(
                    f"intervention acquisition did not produce a render bundle: {status}: {detail}"
                )
        for command in (
            [
                sys.executable,
                "-m",
                "omnigibson_episode.cli",
                "compile",
                "--bundle",
                str(bundle),
                "--output",
                str(bundle / "spatial_episode.json"),
            ],
            [
                sys.executable,
                "-m",
                "omnigibson_episode.cli",
                "inspect",
                "--bundle",
                str(bundle),
            ],
            [
                sys.executable,
                "-m",
                "omnigibson_episode.cli",
                "compile-reasoning-tasks",
                "--bundle",
                str(bundle),
            ],
            [
                sys.executable,
                "-m",
                "omnigibson_episode.cli",
                "audit",
                "--bundle",
                str(bundle),
            ],
            [
                sys.executable,
                "-m",
                "omnigibson_episode.cli",
                "certify-intervention",
                "--base-bundle",
                str(job["base_bundle"]),
                "--variant-bundle",
                str(bundle),
            ],
        ):
            _run_command(command, log_path=log, env=env, timeout_seconds=600)
        final_status, final_detail = intervention_bundle_status(bundle, base_bundle)
        if final_status not in {"certified", "needs_review"}:
            raise RuntimeError(f"intervention did not certify: {final_status}: {final_detail}")
        _receipt(
            job,
            status=final_status,
            detail=final_detail,
            gpu_id=gpu_id,
            started_at=started,
            finished_at=_timestamp(),
            log=str(log),
        )
        return {"job_id": job["job_id"], "status": final_status, "detail": final_detail}
    except Exception as error:
        _receipt(
            job,
            status="failed",
            detail=str(error),
            gpu_id=gpu_id,
            started_at=started,
            finished_at=_timestamp(),
            log=str(log),
        )
        return {"job_id": job["job_id"], "status": "failed", "detail": str(error)}


def run_intervention_sweep(
    *,
    plan_path: Path,
    gpu_ids: list[int],
    workers: int,
    limit: int | None = None,
    scene_models: set[str] | None = None,
    proposal_ids: set[str] | None = None,
    retry_failed: bool = False,
    acquisition_timeout_seconds: int = 1200,
) -> list[dict[str, Any]]:
    if workers < 1 or workers > len(gpu_ids):
        raise ValueError("workers must be between one and the number of GPU IDs")
    plan = refresh_intervention_sweep_plan(plan_path)
    jobs = [
        job
        for job in plan["jobs"]
        if job["status"] not in {"certified", "needs_review"}
        and (scene_models is None or job["scene_model"] in scene_models)
        and (proposal_ids is None or job["proposal_id"] in proposal_ids)
        and (retry_failed or job["status"] not in {"failed", "incomplete"})
    ]
    if limit is not None:
        jobs = jobs[:limit]
    queue: Queue[dict[str, Any]] = Queue()
    for job in jobs:
        queue.put(job)
    project_root = Path(__file__).resolve().parents[2]

    def worker(gpu_id: int) -> list[dict[str, Any]]:
        results = []
        while True:
            try:
                job = queue.get_nowait()
            except Empty:
                return results
            try:
                results.append(
                    run_intervention_job(
                        job=job,
                        project_root=project_root,
                        gpu_id=gpu_id,
                        retry_failed=retry_failed,
                        acquisition_timeout_seconds=acquisition_timeout_seconds,
                    )
                )
            finally:
                queue.task_done()

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, gpu_id) for gpu_id in gpu_ids[:workers]]
        for future in as_completed(futures):
            results.extend(future.result())
    refresh_intervention_sweep_plan(plan_path)
    return sorted(results, key=lambda item: item["job_id"])
