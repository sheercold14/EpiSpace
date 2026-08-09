"""Parallel OmniGibson renderer for a planned scriptgen collection."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .collection import CollectionJob, CollectionManifest

RENDER_STATUS_SCHEMA_VERSION = "scriptgen_render_status.v1"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def rendered_bundle_complete(job: CollectionJob) -> bool:
    """Check the render contract and expected sequence/auxiliary counts."""
    bundle = Path(job.bundle)
    report_path = bundle / "render_report.json"
    trajectory_path = bundle / "trajectory_plan.json"
    if not report_path.is_file() or not trajectory_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return bool(
        report.get("status") == "success"
        and len(report.get("views", ())) == job.frame_count
        and len(report.get("auxiliary_views", ())) == job.auxiliary_view_count
        and len(trajectory.get("views", ())) == job.frame_count
    )


def render_collection(
    manifest: CollectionManifest,
    *,
    og_root: Path,
    gpu_ids: tuple[int, ...],
    workers: int | None = None,
    timeout_minutes: int = 20,
    retry_failed: bool = False,
    limit: int | None = None,
) -> Path:
    """Render pending jobs with one persistent work queue per GPU."""
    if not gpu_ids:
        raise ValueError("at least one GPU id is required")
    if timeout_minutes <= 0:
        raise ValueError("timeout_minutes must be positive")
    worker_count = min(workers or len(gpu_ids), len(gpu_ids))
    if worker_count <= 0:
        raise ValueError("workers must be positive")
    output_root = Path(manifest.output_root)
    status_path = output_root / "render.status.json"
    log_root = output_root / "logs"
    pending: list[CollectionJob] = []
    records: dict[str, dict[str, Any]] = {}
    for job in manifest.jobs:
        if rendered_bundle_complete(job):
            records[job.job_id] = {"status": "rendered", "gpu_id": None, "returncode": 0}
            continue
        failure_exists = (Path(job.bundle) / "failure_report.json").is_file()
        if failure_exists and not retry_failed:
            records[job.job_id] = {
                "status": "failed",
                "gpu_id": None,
                "returncode": None,
                "reason": "existing_failure_report",
            }
            continue
        pending.append(job)
    if limit is not None:
        pending = pending[:limit]
    for job in pending:
        records[job.job_id] = {"status": "pending", "gpu_id": None, "returncode": None}

    lock = threading.Lock()
    work: queue.Queue[CollectionJob] = queue.Queue()
    for job in pending:
        work.put(job)

    def save() -> None:
        counts: dict[str, int] = {}
        for record in records.values():
            status = str(record["status"])
            counts[status] = counts.get(status, 0) + 1
        _write_json(
            status_path,
            {
                "schema_version": RENDER_STATUS_SCHEMA_VERSION,
                "collection_id": manifest.collection_id,
                "updated_unix_s": round(time.time(), 3),
                "counts": counts,
                "jobs": records,
            },
        )

    def worker(gpu_id: int) -> None:
        while True:
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            bundle = Path(job.bundle)
            overwrite = bundle.exists()
            command = [
                "bash",
                str(og_root / "scripts" / "run_in_omnigibson.sh"),
                "--accept-eula",
                "python",
                "-m",
                "omnigibson_episode.cli",
                "acquire",
                "--recipe",
                job.recipe,
                "--output",
                job.bundle,
                "--gpu-id",
                str(gpu_id),
                "--headless",
            ]
            if overwrite:
                command.append("--overwrite")
            log_path = log_root / f"{job.job_id}.log"
            with lock:
                records[job.job_id] = {
                    "status": "running",
                    "gpu_id": gpu_id,
                    "returncode": None,
                    "log": str(log_path),
                    "started_unix_s": round(time.time(), 3),
                }
                save()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    result = subprocess.run(
                        command,
                        cwd=og_root,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=timeout_minutes * 60,
                        check=False,
                    )
                succeeded = result.returncode == 0 and rendered_bundle_complete(job)
                record = {
                    "status": "rendered" if succeeded else "failed",
                    "gpu_id": gpu_id,
                    "returncode": result.returncode,
                    "log": str(log_path),
                    "finished_unix_s": round(time.time(), 3),
                }
                if not succeeded:
                    record["reason"] = "acquire_failed_or_bundle_incomplete"
            except subprocess.TimeoutExpired:
                record = {
                    "status": "failed",
                    "gpu_id": gpu_id,
                    "returncode": None,
                    "log": str(log_path),
                    "finished_unix_s": round(time.time(), 3),
                    "reason": f"timeout_after_{timeout_minutes}_minutes",
                }
            with lock:
                records[job.job_id] = record
                save()
                print(
                    f"render {record['status']} job={job.job_id} gpu={gpu_id}",
                    flush=True,
                )
            work.task_done()

    with lock:
        save()
    threads = [
        threading.Thread(target=worker, args=(gpu_id,), name=f"scriptgen-gpu-{gpu_id}")
        for gpu_id in gpu_ids[:worker_count]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with lock:
        save()
    return status_path
