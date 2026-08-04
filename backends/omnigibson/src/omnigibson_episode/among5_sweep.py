"""Plan paired procedural Among-5 acquisitions without importing OmniGibson."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from omnigibson_episode.config import load_recipe
from omnigibson_episode.io import sha256_file, write_json_atomic
from omnigibson_episode.sweep import _bundle_status


def _load_mapping(path: Path, *, schema_version: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != schema_version:
        raise ValueError(f"{path} must use {schema_version}")
    return payload


def _rotate(values: list[str], offset: int) -> list[str]:
    normalized = offset % len(values)
    return values[normalized:] + values[:normalized]


def build_among5_sweep_plan(
    *,
    family_config_path: Path,
    output_root: Path,
    plan_path: Path,
) -> dict[str, Any]:
    """Build a scene-disjoint counterfactual sweep compatible with ``run-sweep``."""

    family_config_path = family_config_path.resolve()
    output_root = output_root.resolve()
    plan_path = plan_path.resolve()
    config = _load_mapping(
        family_config_path, schema_version="omnigibson_among5_family_config.v1"
    )
    base_recipe_path = (family_config_path.parent / config["base_recipe"]).resolve()
    split_manifest_path = (
        family_config_path.parent / config["split_manifest"]
    ).resolve()
    base_recipe = load_recipe(base_recipe_path)
    if base_recipe.among5 is None:
        raise ValueError("Among-5 family base recipe has no among5 contract")
    split_payload = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    split_by_scene = {
        str(item["scene_model"]): {
            "split": str(item["split"]),
            "split_group": str(item["split_group"]),
        }
        for item in split_payload["assignments"]
    }
    scene_models = [str(value) for value in config["scene_models"]]
    if len(scene_models) != len(set(scene_models)):
        raise ValueError("Among-5 family scene_models must be unique")
    missing_splits = set(scene_models) - set(split_by_scene)
    if missing_splits:
        raise ValueError(f"split manifest is missing scenes: {sorted(missing_splits)}")
    variants = config["variants"]
    if not isinstance(variants, list) or not variants:
        raise ValueError("Among-5 family variants must be a non-empty list")
    variant_ids = [str(item["variant_id"]) for item in variants]
    if len(variant_ids) != len(set(variant_ids)) or "identity" not in variant_ids:
        raise ValueError("Among-5 variants must be unique and include identity")

    satellites = list(base_recipe.among5.satellite_order)
    jobs = []
    base_seed = int(config["base_seed"])
    for scene_index, scene_model in enumerate(scene_models):
        scene_permutation = scene_index % len(satellites)
        reference_order = _rotate(satellites, scene_permutation)
        identity_job_id = f"{scene_model}_t2_among5_identity_seed{base_seed}"
        for variant_index, raw_variant in enumerate(variants):
            if not isinstance(raw_variant, dict):
                raise ValueError("each Among-5 variant must be a mapping")
            variant_id = str(raw_variant["variant_id"])
            permutation_offset = int(raw_variant.get("permutation_offset", 0))
            satellite_order = _rotate(reference_order, permutation_offset)
            seed = base_seed + variant_index
            job_id = f"{scene_model}_t2_among5_{variant_id}_seed{seed}"
            bundle = output_root / "bundles" / job_id
            status, detail = _bundle_status(bundle)
            recipe_overrides = {
                "recipe_id": f"omnigibson_{job_id}",
                "among5": {
                    "layout_id": f"{scene_model}.{variant_id}.p{scene_permutation}",
                    "satellite_order": satellite_order,
                    "layout_yaw_deg": float(raw_variant.get("layout_yaw_deg", 0.0)),
                    "mirrored": bool(raw_variant.get("mirrored", False)),
                    "layout_radius_m": float(
                        raw_variant.get(
                            "layout_radius_m", base_recipe.among5.layout_radius_m
                        )
                    ),
                    "camera_azimuth_deg": [
                        float(value)
                        for value in raw_variant.get(
                            "camera_azimuth_deg",
                            base_recipe.among5.camera_azimuth_deg,
                        )
                    ],
                },
            }
            jobs.append(
                {
                    "job_id": job_id,
                    "scene_model": scene_model,
                    "seed": seed,
                    "floor": 0,
                    "classification": "controlled_procedural_among5",
                    "family_id": f"{scene_model}.among5.cf.v1",
                    "variant_id": variant_id,
                    "counterfactual_parent_job_id": (
                        None if variant_id == "identity" else identity_job_id
                    ),
                    "bundle": str(bundle),
                    "receipt": str(output_root / "job_status" / f"{job_id}.json"),
                    "recipe_overrides": recipe_overrides,
                    "status": status,
                    "status_detail": detail,
                    **split_by_scene[scene_model],
                }
            )
    plan = {
        "schema_version": "omnigibson_scene_sweep.v1",
        "sweep_id": str(config["family_id"]),
        "family_schema_version": "omnigibson_among5_counterfactual_family.v1",
        "family_config": str(family_config_path),
        "family_config_sha256": sha256_file(family_config_path),
        "base_recipe": str(base_recipe_path),
        "base_recipe_sha256": sha256_file(base_recipe_path),
        "split_manifest": str(split_manifest_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "trajectory_class": "T2",
        "output_root": str(output_root),
        "scene_count": len(scene_models),
        "variant_count": len(variants),
        "job_count": len(jobs),
        "jobs": jobs,
    }
    write_json_atomic(plan_path, plan)
    return plan
