"""Execute sweep jobs with per-job logs and atomic receipts."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from omnigibson_episode.io import sha256_file, write_json_atomic
from omnigibson_episode.sweep import (
    _bundle_status,
    materialize_job_recipe,
    refresh_sweep_plan,
)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _run_command(
    command: list[str], *, log_path: Path, env: dict[str, str], timeout_seconds: int
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_timestamp()}] command={json.dumps(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=env["EPISODE_PROJECT_ROOT"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            log.write(f"[{_timestamp()}] timeout_seconds={timeout_seconds}\n")
            raise RuntimeError(
                f"command exceeded {timeout_seconds} seconds: {command[0:4]}"
            ) from error
        log.write(f"[{_timestamp()}] returncode={returncode}\n")
    if returncode != 0:
        raise RuntimeError(f"command returned {returncode}: {command[0:4]}")


def _write_receipt(job: dict[str, Any], **updates: Any) -> None:
    path = Path(job["receipt"])
    payload = {
        "schema_version": "omnigibson_sweep_job_receipt.v1",
        "job_id": job["job_id"],
        "scene_model": job["scene_model"],
        **updates,
    }
    write_json_atomic(path, payload)


def _reconcile_terminal_receipt(
    job: dict[str, Any], *, status: str, detail: str | None
) -> None:
    """Make a terminal sensor bundle authoritative over a stale runner receipt."""

    path = Path(job["receipt"])
    previous: dict[str, Any] = {}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            previous = payload
    if previous.get("status") == status and previous.get("detail") == detail:
        return
    reconciled = {
        **previous,
        "schema_version": "omnigibson_sweep_job_receipt.v1",
        "job_id": job["job_id"],
        "scene_model": job["scene_model"],
        "status": status,
        "detail": detail,
        "reconciled_at": _timestamp(),
        "reconciliation": {
            "source": "terminal bundle validation",
            "previous_status": previous.get("status"),
            "previous_detail": previous.get("detail"),
        },
    }
    write_json_atomic(path, reconciled)


def run_job(
    *,
    job: dict[str, Any],
    base_recipe: Path,
    project_root: Path,
    gpu_id: int,
    retry_failed: bool,
    acquisition_timeout_seconds: int,
) -> dict[str, Any]:
    bundle = Path(job["bundle"])
    status, detail = _bundle_status(bundle)
    if status in {"passed", "needs_review"}:
        _reconcile_terminal_receipt(job, status=status, detail=detail)
        return {
            "job_id": job["job_id"],
            "status": status,
            "detail": f"already terminal: {detail}" if detail else "already terminal",
        }
    if status in {"failed", "incomplete"} and not retry_failed:
        return {"job_id": job["job_id"], "status": status, "detail": detail}

    output_root = Path(job["receipt"]).parents[1]
    recipe_path = output_root / "recipes" / f"{job['job_id']}.yaml"
    log_path = output_root / "logs" / f"{job['job_id']}.log"
    materialize_job_recipe(
        base_recipe_path=base_recipe,
        scene_model=job["scene_model"],
        seed=int(job["seed"]),
        output_path=recipe_path,
        overrides=job.get("recipe_overrides", {}),
    )
    effective_timeout_seconds = int(
        job.get("acquisition_timeout_seconds", acquisition_timeout_seconds)
    )
    if effective_timeout_seconds < 1:
        raise ValueError("effective acquisition timeout must be positive")
    recipe_sha256 = sha256_file(recipe_path)
    receipt_path = Path(job["receipt"])
    attempt = 1
    if receipt_path.is_file():
        previous_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        attempt = int(previous_receipt.get("attempt", 1)) + 1
    started = _timestamp()
    _write_receipt(
        job,
        status="running",
        detail=None,
        gpu_id=gpu_id,
        pid=os.getpid(),
        attempt=attempt,
        started_at=started,
        recipe=str(recipe_path),
        recipe_sha256=recipe_sha256,
        acquisition_timeout_seconds=effective_timeout_seconds,
        log=str(log_path),
    )
    env = os.environ.copy()
    env["EPISODE_PROJECT_ROOT"] = str(project_root)
    env["OMNIGIBSON_DATA_PATH"] = str(project_root / ".data" / "omnigibson")
    pythonpath = [str(project_root / "src"), str(project_root.parent / "habitat" / "src")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    try:
        status, _ = _bundle_status(bundle)
        if status != "acquired":
            command = [
                "bash",
                str(project_root / "scripts" / "run_in_omnigibson.sh"),
                "--accept-eula",
                "python",
                "-m",
                "omnigibson_episode.cli",
                "acquire",
                "--recipe",
                str(recipe_path),
                "--output",
                str(bundle),
                "--gpu-id",
                str(gpu_id),
                "--headless",
            ]
            stale_staging = any(bundle.parent.glob(f".{bundle.name}.staging.*"))
            if bundle.exists() or retry_failed or stale_staging:
                command.append("--overwrite")
            _run_command(
                command,
                log_path=log_path,
                env=env,
                timeout_seconds=effective_timeout_seconds,
            )
            acquired_status, acquired_detail = _bundle_status(bundle)
            if acquired_status != "acquired":
                raise RuntimeError(
                    "acquisition command did not produce a render bundle: "
                    f"{acquired_status}: {acquired_detail}"
                )
        _run_command(
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
            log_path=log_path,
            env=env,
            timeout_seconds=600,
        )
        _run_command(
            [
                sys.executable,
                "-m",
                "omnigibson_episode.cli",
                "inspect",
                "--bundle",
                str(bundle),
            ],
            log_path=log_path,
            env=env,
            timeout_seconds=600,
        )
        _run_command(
            [
                sys.executable,
                "-m",
                "omnigibson_episode.cli",
                "compile-reasoning-tasks",
                "--bundle",
                str(bundle),
            ],
            log_path=log_path,
            env=env,
            timeout_seconds=600,
        )
        _run_command(
            [
                sys.executable,
                "-m",
                "omnigibson_episode.cli",
                "audit",
                "--bundle",
                str(bundle),
            ],
            log_path=log_path,
            env=env,
            timeout_seconds=600,
        )
        final_status, final_detail = _bundle_status(bundle)
        if final_status not in {"passed", "needs_review"}:
            raise RuntimeError(f"job did not reach a terminal quality state: {final_status}")
        _write_receipt(
            job,
            status=final_status,
            detail=final_detail,
            gpu_id=gpu_id,
            attempt=attempt,
            started_at=started,
            finished_at=_timestamp(),
            recipe=str(recipe_path),
            recipe_sha256=recipe_sha256,
            acquisition_timeout_seconds=effective_timeout_seconds,
            log=str(log_path),
        )
        return {"job_id": job["job_id"], "status": final_status, "detail": final_detail}
    except Exception as error:
        _write_receipt(
            job,
            status="failed",
            detail=str(error),
            gpu_id=gpu_id,
            attempt=attempt,
            started_at=started,
            finished_at=_timestamp(),
            recipe=str(recipe_path),
            recipe_sha256=recipe_sha256,
            acquisition_timeout_seconds=effective_timeout_seconds,
            log=str(log_path),
        )
        return {"job_id": job["job_id"], "status": "failed", "detail": str(error)}


def run_sweep(
    *,
    plan_path: Path,
    gpu_ids: list[int],
    workers: int,
    limit: int | None = None,
    scene_models: set[str] | None = None,
    variant_ids: set[str] | None = None,
    retry_failed: bool = False,
    acquisition_timeout_seconds: int = 1200,
) -> list[dict[str, Any]]:
    if workers < 1 or workers > len(gpu_ids):
        raise ValueError("workers must be between one and the number of GPU IDs")
    if acquisition_timeout_seconds < 1:
        raise ValueError("acquisition timeout must be positive")
    plan = refresh_sweep_plan(plan_path)
    jobs = [
        job
        for job in plan["jobs"]
        if job["status"] != "passed"
        and (scene_models is None or job["scene_model"] in scene_models)
        and (variant_ids is None or job.get("variant_id") in variant_ids)
        and (retry_failed or job["status"] not in {"failed", "incomplete"})
    ]
    if limit is not None:
        jobs = jobs[:limit]
    project_root = Path(__file__).resolve().parents[2]
    base_recipe = Path(plan["base_recipe"])
    job_queue: Queue[dict[str, Any]] = Queue()
    for job in jobs:
        job_queue.put(job)

    def worker_loop(gpu_id: int) -> list[dict[str, Any]]:
        worker_results = []
        while True:
            try:
                job = job_queue.get_nowait()
            except Empty:
                return worker_results
            try:
                worker_results.append(
                    run_job(
                        job=job,
                        base_recipe=base_recipe,
                        project_root=project_root,
                        gpu_id=gpu_id,
                        retry_failed=retry_failed,
                        acquisition_timeout_seconds=acquisition_timeout_seconds,
                    )
                )
            finally:
                job_queue.task_done()

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker_loop, gpu_id) for gpu_id in gpu_ids[:workers]]
        for future in as_completed(futures):
            results.extend(future.result())
    refresh_sweep_plan(plan_path)
    return sorted(results, key=lambda item: item["job_id"])
