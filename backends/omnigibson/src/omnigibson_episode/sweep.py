"""Deterministic, resumable planning for multi-scene acquisition sweeps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from omnigibson_episode.io import sha256_file, write_json_atomic


def _merge_known_fields(
    base: dict[str, Any], overrides: dict[str, Any], *, path: str = "recipe"
) -> dict[str, Any]:
    """Recursively merge overrides while rejecting misspelled or invented fields."""

    merged = dict(base)
    for key, value in overrides.items():
        field_path = f"{path}.{key}"
        if key not in base:
            raise ValueError(f"unknown override field: {field_path}")
        if isinstance(value, dict):
            if not isinstance(base[key], dict):
                raise ValueError(f"cannot apply a mapping override to {field_path}")
            merged[key] = _merge_known_fields(base[key], value, path=field_path)
        else:
            merged[key] = value
    return merged


def _load_scene_overrides(path: Path | None) -> tuple[Path | None, dict[str, Any]]:
    if path is None:
        return None, {}
    resolved = path.resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "omnigibson_scene_overrides.v1"
    ):
        raise ValueError("scene overrides must use omnigibson_scene_overrides.v1")
    unknown = set(payload) - {"schema_version", "overrides"}
    if unknown:
        raise ValueError(f"unknown scene override top-level fields: {sorted(unknown)}")
    overrides = payload.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("scene overrides.overrides must be a mapping")
    return resolved, overrides


def _bundle_status(bundle: Path) -> tuple[str, str | None]:
    quality_path = bundle / "quality_report.json"
    episode_path = bundle / "spatial_episode.json"
    if quality_path.is_file() and episode_path.is_file():
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        if quality.get("integrity_status") == "pass" and quality.get("visual_status") == "pass":
            return "passed", None
        return "needs_review", "quality report is not a full pass"
    failure_path = bundle / "failure_report.json"
    if failure_path.is_file():
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        return "failed", failure.get("error", {}).get("message", "acquisition failed")
    if (bundle / "render_report.json").is_file():
        return "acquired", "compile and inspection remain"
    if bundle.exists():
        return "incomplete", "bundle exists without a terminal manifest"
    return "pending", None


def build_sweep_plan(
    *,
    inventory_path: Path,
    base_recipe_path: Path,
    output_root: Path,
    plan_path: Path | None = None,
    seed: int | None = None,
    scene_overrides_path: Path | None = None,
    split_manifest_path: Path | None = None,
) -> dict[str, Any]:
    inventory_path = inventory_path.resolve()
    base_recipe_path = base_recipe_path.resolve()
    output_root = output_root.resolve()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    recipe = yaml.safe_load(base_recipe_path.read_text(encoding="utf-8"))
    resolved_overrides_path, scene_overrides = _load_scene_overrides(scene_overrides_path)
    effective_seed = int(recipe["seed"] if seed is None else seed)
    trajectory_class = str(recipe.get("trajectory", {}).get("trajectory_class", "T1"))
    recipe_slug = str(recipe["recipe_id"]).removeprefix("omnigibson_").replace("_", "-")
    resolved_split_path = split_manifest_path.resolve() if split_manifest_path else None
    split_by_scene: dict[str, dict[str, str]] = {}
    if resolved_split_path is not None:
        split_payload = json.loads(resolved_split_path.read_text(encoding="utf-8"))
        split_by_scene = {
            str(item["scene_model"]): {
                "split": str(item["split"]),
                "split_group": str(item["split_group"]),
            }
            for item in split_payload["assignments"]
        }
    jobs = []
    eligible_scene_models = {
        scene["scene_model"]
        for scene in inventory["scenes"]
        if scene["eligibility"]["static_multiview"]
    }
    unknown_scenes = set(scene_overrides) - eligible_scene_models
    if unknown_scenes:
        raise ValueError(
            f"scene overrides reference ineligible or unknown scenes: {sorted(unknown_scenes)}"
        )
    for scene in inventory["scenes"]:
        if not scene["eligibility"]["static_multiview"]:
            continue
        scene_model = scene["scene_model"]
        class_suffix = "" if trajectory_class == "T1" else f"_{trajectory_class.lower()}"
        job_id = f"{scene_model}{class_suffix}_seed{effective_seed}"
        bundle = output_root / "bundles" / job_id
        status, detail = _bundle_status(bundle)
        override = scene_overrides.get(scene_model, {})
        if not isinstance(override, dict):
            raise ValueError(f"override for {scene_model} must be a mapping")
        unknown_override_fields = set(override) - {
            "recipe",
            "acquisition_timeout_minutes",
        }
        if unknown_override_fields:
            raise ValueError(
                f"unknown override fields for {scene_model}: {sorted(unknown_override_fields)}"
            )
        recipe_overrides = override.get("recipe", {})
        if not isinstance(recipe_overrides, dict):
            raise ValueError(f"recipe override for {scene_model} must be a mapping")
        _merge_known_fields(recipe, recipe_overrides)
        job = {
            "job_id": job_id,
            "scene_model": scene_model,
            "seed": effective_seed,
            "floor": 0,
            "classification": scene["classification"],
            "bundle": str(bundle),
            "receipt": str(output_root / "job_status" / f"{job_id}.json"),
            "recipe_overrides": recipe_overrides,
            "status": status,
            "status_detail": detail,
        }
        if resolved_split_path is not None:
            if scene_model not in split_by_scene:
                raise ValueError(f"split manifest is missing scene: {scene_model}")
            job.update(split_by_scene[scene_model])
        if "acquisition_timeout_minutes" in override:
            timeout_minutes = int(override["acquisition_timeout_minutes"])
            if timeout_minutes < 1:
                raise ValueError(f"acquisition timeout for {scene_model} must be positive")
            job["acquisition_timeout_seconds"] = timeout_minutes * 60
        jobs.append(job)
    plan = {
        "schema_version": "omnigibson_scene_sweep.v1",
        "sweep_id": f"{recipe_slug}-indoor-seed{effective_seed}-v1",
        "inventory": str(inventory_path),
        "inventory_sha256": sha256_file(inventory_path),
        "base_recipe": str(base_recipe_path),
        "base_recipe_sha256": sha256_file(base_recipe_path),
        "scene_overrides": (
            str(resolved_overrides_path) if resolved_overrides_path is not None else None
        ),
        "scene_overrides_sha256": (
            sha256_file(resolved_overrides_path) if resolved_overrides_path is not None else None
        ),
        "split_manifest": str(resolved_split_path) if resolved_split_path else None,
        "split_manifest_sha256": (
            sha256_file(resolved_split_path) if resolved_split_path else None
        ),
        "trajectory_class": trajectory_class,
        "output_root": str(output_root),
        "job_count": len(jobs),
        "jobs": jobs,
    }
    if plan_path is not None:
        write_json_atomic(plan_path, plan)
    return plan


def materialize_job_recipe(
    *,
    base_recipe_path: Path,
    scene_model: str,
    seed: int,
    output_path: Path,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """Write the exact recipe consumed by one acquisition job."""

    payload = yaml.safe_load(base_recipe_path.read_text(encoding="utf-8"))
    payload = _merge_known_fields(payload, overrides or {})
    payload["seed"] = int(seed)
    payload["source"]["scene_model"] = scene_model
    payload["source"]["scene_instance"] = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    temporary.replace(output_path)
    return output_path


def refresh_sweep_plan(plan_path: Path) -> dict[str, Any]:
    """Refresh only runtime status while preserving the immutable job specification."""

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for job in plan["jobs"]:
        if "receipt" not in job:
            bundle = Path(job["bundle"])
            job["receipt"] = str(bundle.parents[1] / "job_status" / f"{job['job_id']}.json")
        status, detail = _bundle_status(Path(job["bundle"]))
        receipt_path = Path(job.get("receipt", ""))
        if status in {"pending", "incomplete"} and receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            status = str(receipt.get("status", status))
            detail = receipt.get("detail", detail)
        job["status"] = status
        job["status_detail"] = detail
    counts: dict[str, int] = {}
    for job in plan["jobs"]:
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    plan["status_counts"] = dict(sorted(counts.items()))
    write_json_atomic(plan_path, plan)
    return plan
