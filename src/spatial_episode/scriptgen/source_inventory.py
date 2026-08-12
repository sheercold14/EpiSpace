"""Prepare immutable OmniGibson source bundles for coverage generation."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from .spec import SpecModel

SOURCE_INDEX_SCHEMA_VERSION = "scriptgen_source_index.v1"


class SourceSceneRecord(SpecModel):
    scene_key: str
    scene_model: str
    strategy: str
    status: Literal["ready", "failed"]
    bundle: str
    recipe: str
    scene_ir: str | None = None
    scene_snapshot: str | None = None
    source_digest: str | None = None
    scene_ir_sha256: str | None = None
    recipe_sha256: str | None = None
    attempts: int = Field(ge=0)
    gpu_id: int | None = None
    log: str | None = None
    failure_reason: str | None = None


class SourceIndex(SpecModel):
    schema_version: Literal["scriptgen_source_index.v1"] = SOURCE_INDEX_SCHEMA_VERSION
    source_id: str
    source_root: str
    og_root: str
    data_root: str
    source_version: str
    requested_scene_count: int = Field(ge=1)
    ready_scene_count: int = Field(ge=0)
    failed_scene_count: int = Field(ge=0)
    scenes: tuple[SourceSceneRecord, ...]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deep_update(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def discover_behavior_scenes(data_root: Path) -> tuple[str, ...]:
    scene_root = data_root / "behavior-1k-assets" / "scenes"
    if not scene_root.is_dir():
        raise FileNotFoundError(f"BEHAVIOR scene catalog is missing: {scene_root}")
    scenes = tuple(sorted(path.name for path in scene_root.iterdir() if path.is_dir()))
    if not scenes:
        raise ValueError(f"no BEHAVIOR scenes found under {scene_root}")
    return scenes


def _runtime_env(og_root: Path, data_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    epispace_src = Path(__file__).resolve().parents[2]
    python_path = os.pathsep.join((str(og_root / "src"), str(epispace_src)))
    if env.get("PYTHONPATH"):
        python_path += os.pathsep + env["PYTHONPATH"]
    env.update(
        {
            "PYTHONPATH": python_path,
            "OMNIGIBSON_DATA_PATH": str(data_root),
            "OMNIGIBSON_HEADLESS": "True",
            "OMNI_KIT_ACCEPT_EULA": "YES",
        }
    )
    env.pop("DISPLAY", None)
    return env


def _source_recipe(
    scene_model: str,
    *,
    og_root: Path,
) -> tuple[dict[str, Any], int]:
    garden = scene_model.endswith("_garden")
    template_name = "omnigibson_static_m1.yaml" if garden else "omnigibson_static_m2_room_aware.yaml"
    template_path = og_root / "configs" / template_name
    payload = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    payload["recipe_id"] = f"epispace_source_{scene_model}"
    payload["source"]["scene_model"] = scene_model
    payload["trajectory"]["sampling_strategy"] = (
        "traversable_random" if garden else "room_aware"
    )
    timeout_minutes = 20
    override_path = og_root / "configs" / "omnigibson_static_m2_scene_overrides.yaml"
    if override_path.is_file():
        overrides = yaml.safe_load(override_path.read_text(encoding="utf-8")) or {}
        scene_override = overrides.get("overrides", {}).get(scene_model, {})
        timeout_minutes = int(scene_override.get("acquisition_timeout_minutes", timeout_minutes))
        recipe_override = scene_override.get("recipe")
        if recipe_override:
            _deep_update(payload, recipe_override)
            payload["source"]["scene_model"] = scene_model
    return payload, timeout_minutes


def _acquisition_ready(bundle: Path) -> bool:
    required = (
        bundle / "scene_snapshot.json",
        bundle / "render_report.json",
        bundle / "trajectory_plan.json",
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        report = json.loads((bundle / "render_report.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return report.get("status") == "success"


def _bundle_ready(bundle: Path) -> bool:
    return _acquisition_ready(bundle) and (bundle / "scene_ir.json").is_file()


def _record_ready(
    scene_model: str,
    strategy: str,
    bundle: Path,
    recipe: Path,
    *,
    attempts: int,
    gpu_id: int | None,
    log: Path | None,
) -> SourceSceneRecord:
    snapshot_path = bundle / "scene_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    scene_ir = bundle / "scene_ir.json"
    return SourceSceneRecord(
        scene_key=scene_model,
        scene_model=scene_model,
        strategy=strategy,
        status="ready",
        bundle=str(bundle),
        recipe=str(recipe),
        scene_ir=str(scene_ir),
        scene_snapshot=str(snapshot_path),
        source_digest=str(snapshot["source_digest"]),
        scene_ir_sha256=_sha256(scene_ir),
        recipe_sha256=_sha256(recipe),
        attempts=attempts,
        gpu_id=gpu_id,
        log=str(log) if log is not None else None,
    )


def _record_still_ready(record: SourceSceneRecord) -> bool:
    if record.status != "ready" or record.scene_ir is None:
        return False
    recipe = Path(record.recipe)
    scene_ir = Path(record.scene_ir)
    snapshot = Path(record.scene_snapshot or "")
    if not (
        _bundle_ready(Path(record.bundle))
        and recipe.is_file()
        and snapshot.is_file()
        and record.recipe_sha256 == _sha256(recipe)
        and record.scene_ir_sha256 == _sha256(scene_ir)
    ):
        return False
    try:
        snapshot_payload = json.loads(snapshot.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return snapshot_payload.get("source_digest") == record.source_digest


def _acquire_source(
    scene_model: str,
    *,
    source_root: Path,
    og_root: Path,
    data_root: Path,
    conda_env: str,
    gpu_id: int,
) -> SourceSceneRecord:
    payload, timeout_minutes = _source_recipe(scene_model, og_root=og_root)
    strategy = str(payload["trajectory"]["sampling_strategy"])
    bundle = source_root / "scenes" / scene_model
    recipe = source_root / "recipes" / f"{scene_model}.yaml"
    log_path = source_root / "logs" / f"{scene_model}.log"
    recipe.parent.mkdir(parents=True, exist_ok=True)
    temporary = recipe.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    temporary.replace(recipe)
    if _bundle_ready(bundle):
        return _record_ready(
            scene_model,
            strategy,
            bundle,
            recipe,
            attempts=0,
            gpu_id=None,
            log=log_path if log_path.is_file() else None,
        )

    conda = shutil.which("conda")
    if conda is None:
        raise FileNotFoundError("conda executable not found")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    last_reason = "acquisition_not_started"
    for attempt in range(1, 3):
        acquire = [
            conda,
            "run",
            "--no-capture-output",
            "-n",
            conda_env,
            "python",
            "-m",
            "omnigibson_episode.cli",
            "acquire",
            "--recipe",
            str(recipe),
            "--output",
            str(bundle),
            "--gpu-id",
            str(gpu_id),
            "--headless",
        ]
        if bundle.exists():
            acquire.append("--overwrite")
        try:
            if not _acquisition_ready(bundle):
                with log_path.open("a", encoding="utf-8") as log:
                    result = subprocess.run(
                        acquire,
                        cwd=og_root,
                        env=_runtime_env(og_root, data_root),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=timeout_minutes * 60,
                        check=False,
                    )
                if result.returncode != 0:
                    last_reason = f"acquire_exit_{result.returncode}"
                    continue
            compile_command = [
                conda,
                "run",
                "--no-capture-output",
                "-n",
                conda_env,
                "python",
                "-m",
                "omnigibson_episode.cli",
                "compile",
                "--bundle",
                str(bundle),
                "--output",
                str(bundle / "spatial_episode.json"),
            ]
            with log_path.open("a", encoding="utf-8") as log:
                compiled = subprocess.run(
                    compile_command,
                    cwd=og_root,
                    env=_runtime_env(og_root, data_root),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=10 * 60,
                    check=False,
                )
            if compiled.returncode != 0:
                last_reason = f"compile_exit_{compiled.returncode}"
                continue
            if not _bundle_ready(bundle):
                last_reason = "compiled_bundle_incomplete"
                continue
            return _record_ready(
                scene_model,
                strategy,
                bundle,
                recipe,
                attempts=attempt,
                gpu_id=gpu_id,
                log=log_path,
            )
        except subprocess.TimeoutExpired:
            last_reason = f"timeout_after_{timeout_minutes}_minutes"
    return SourceSceneRecord(
        scene_key=scene_model,
        scene_model=scene_model,
        strategy=strategy,
        status="failed",
        bundle=str(bundle),
        recipe=str(recipe),
        attempts=2,
        gpu_id=gpu_id,
        log=str(log_path),
        failure_reason=last_reason,
    )


def prepare_sources(
    *,
    source_root: Path,
    og_root: Path,
    data_root: Path,
    conda_env: str,
    gpu_ids: tuple[int, ...],
    workers: int | None = None,
    scenes: tuple[str, ...] | None = None,
    source_id: str = "behavior51_sources_v1",
) -> Path:
    """Acquire and compile a resumable immutable source-scene inventory."""
    if not gpu_ids:
        raise ValueError("at least one GPU id is required")
    from .single import _preflight

    source_root = source_root.resolve()
    og_root = og_root.resolve()
    data_root = data_root.resolve()
    _preflight(og_root, data_root, conda_env)
    available = discover_behavior_scenes(data_root)
    selected = available if scenes is None else tuple(dict.fromkeys(scenes))
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"unknown BEHAVIOR scenes: {unknown}")
    if not selected:
        raise ValueError("no source scenes selected")
    worker_count = min(workers or len(gpu_ids), len(gpu_ids))
    records: dict[str, SourceSceneRecord] = {}
    lock = threading.Lock()
    work: queue.Queue[str] = queue.Queue()
    for scene in selected:
        work.put(scene)
    index_path = source_root / "source.index.json"
    previous: dict[str, SourceSceneRecord] = {}
    if index_path.is_file():
        prior_index = load_source_index(index_path)
        expected_metadata = (
            source_id,
            str(source_root),
            str(og_root),
            str(data_root),
            len(selected),
        )
        observed_metadata = (
            prior_index.source_id,
            prior_index.source_root,
            prior_index.og_root,
            prior_index.data_root,
            prior_index.requested_scene_count,
        )
        if observed_metadata != expected_metadata:
            raise ValueError(
                f"existing source inventory settings differ: {index_path}"
            )
        prior_scenes = {record.scene_key for record in prior_index.scenes}
        if not prior_scenes.issubset(selected):
            raise ValueError(
                f"existing source inventory scene set differs: {index_path}"
            )
        previous = {record.scene_key: record for record in prior_index.scenes}

    def save() -> None:
        ordered = tuple(records[name] for name in selected if name in records)
        ready = sum(record.status == "ready" for record in ordered)
        index = SourceIndex(
            source_id=source_id,
            source_root=str(source_root),
            og_root=str(og_root),
            data_root=str(data_root),
            source_version="behavior-1k-v3.9.0",
            requested_scene_count=len(selected),
            ready_scene_count=ready,
            failed_scene_count=len(ordered) - ready,
            scenes=ordered,
        )
        _write_json(index_path, index.model_dump(mode="json"))

    def worker(gpu_id: int) -> None:
        while True:
            try:
                scene_model = work.get_nowait()
            except queue.Empty:
                return
            try:
                prior = previous.get(scene_model)
                if prior is not None and _record_still_ready(prior):
                    record = prior
                else:
                    record = _acquire_source(
                        scene_model,
                        source_root=source_root,
                        og_root=og_root,
                        data_root=data_root,
                        conda_env=conda_env,
                        gpu_id=gpu_id,
                    )
            except Exception as error:  # worker must preserve the remaining inventory
                record = SourceSceneRecord(
                    scene_key=scene_model,
                    scene_model=scene_model,
                    strategy=("traversable_random" if scene_model.endswith("_garden") else "room_aware"),
                    status="failed",
                    bundle=str(source_root / "scenes" / scene_model),
                    recipe=str(source_root / "recipes" / f"{scene_model}.yaml"),
                    attempts=0,
                    gpu_id=gpu_id,
                    failure_reason=f"{type(error).__name__}:{error}",
                )
            with lock:
                records[scene_model] = record
                save()
                print(
                    f"source {record.status} scene={scene_model} gpu={gpu_id} "
                    f"progress={len(records)}/{len(selected)}",
                    flush=True,
                )
            work.task_done()

    source_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    threads = [
        threading.Thread(target=worker, args=(gpu_id,), name=f"source-gpu-{gpu_id}")
        for gpu_id in gpu_ids[:worker_count]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with lock:
        save()
    failed = [record.scene_key for record in records.values() if record.status == "failed"]
    print(f"source inventory elapsed_s={time.time() - started:.1f}", flush=True)
    if failed:
        raise RuntimeError(f"source inventory incomplete; failed scenes: {sorted(failed)}")
    return index_path


def load_source_index(path: Path) -> SourceIndex:
    return SourceIndex.model_validate_json(path.read_text(encoding="utf-8"))
