from __future__ import annotations

import json
from pathlib import Path

import yaml

from omnigibson_episode.among5_sweep import build_among5_sweep_plan

ROOT = Path(__file__).resolve().parents[1]


def test_among5_sweep_is_scene_disjoint_and_counterfactually_paired(
    tmp_path: Path,
) -> None:
    source = yaml.safe_load(
        (ROOT / "configs" / "omnigibson_t2_among5_family_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    source["base_recipe"] = str(
        (ROOT / "configs" / "omnigibson_t2_among5_pilot.yaml").resolve()
    )
    source["split_manifest"] = str(
        (
            ROOT
            / "dataset"
            / "manifests"
            / "behavior-1k-v3.9.0-scene-splits.v1.json"
        ).resolve()
    )
    source["scene_models"] = source["scene_models"][:2]
    config = tmp_path / "family.yaml"
    config.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")

    plan = build_among5_sweep_plan(
        family_config_path=config,
        output_root=tmp_path / "output",
        plan_path=tmp_path / "sweep_plan.json",
    )

    assert plan["scene_count"] == 2
    assert plan["variant_count"] == 5
    assert plan["job_count"] == 10
    by_scene: dict[str, list[dict[str, object]]] = {}
    for job in plan["jobs"]:
        by_scene.setdefault(str(job["scene_model"]), []).append(job)
    assert all(len({str(job["split"]) for job in jobs}) == 1 for jobs in by_scene.values())
    for jobs in by_scene.values():
        identity = next(job for job in jobs if job["variant_id"] == "identity")
        siblings = [job for job in jobs if job["variant_id"] != "identity"]
        assert all(
            job["counterfactual_parent_job_id"] == identity["job_id"]
            for job in siblings
        )
        orders = {
            str(job["variant_id"]): job["recipe_overrides"]["among5"]["satellite_order"]
            for job in jobs
        }
        assert orders["identity"] == orders["rotate90"] == orders["mirror"]
        assert orders["identity"] != orders["permute1"]
    written = json.loads((tmp_path / "sweep_plan.json").read_text(encoding="utf-8"))
    assert written["job_count"] == 10
