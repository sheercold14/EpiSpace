import json
from pathlib import Path

import pytest
from PIL import Image

from omnigibson_episode.review import build_sweep_review
from omnigibson_episode.scene_inventory import build_scene_inventory
from omnigibson_episode.sweep import (
    build_sweep_plan,
    materialize_job_recipe,
    refresh_sweep_plan,
)

ROOT = Path(__file__).resolve().parents[1]


def _fake_scene(assets: Path, name: str, rooms: list[str], count: int = 24) -> None:
    scene = assets / "scenes" / name
    (scene / "json").mkdir(parents=True)
    (scene / "layout").mkdir()
    Image.new("L", (2, 2), 255).save(scene / "layout" / "floor_trav_0.png")
    init_info = {}
    registry = {}
    for index in range(count):
        object_name = f"chair_model_{index}"
        init_info[object_name] = {
            "args": {
                "category": f"category_{index % 12}",
                "model": "model",
                "fixed_base": False,
                "visual_only": False,
                "in_rooms": [rooms[index % len(rooms)]],
            }
        }
        registry[object_name] = {"root_link": {"pos": [0, 0, 0]}}
    payload = {
        "versions": {"behavior-1k-assets": {"version": "test"}},
        "objects_info": {"init_info": init_info},
        "state": {"registry": {"object_registry": registry}},
    }
    (scene / "json" / f"{name}_best.json").write_text(json.dumps(payload))


def test_inventory_separates_indoor_and_garden(tmp_path: Path) -> None:
    assets = tmp_path / "behavior-assets"
    _fake_scene(assets, "home_int", ["kitchen_0", "living_room_0"])
    _fake_scene(assets, "home_garden", ["garden_0"])
    output = tmp_path / "inventory.json"
    inventory = build_scene_inventory(assets, output)

    assert inventory["summary"]["scene_count"] == 2
    assert inventory["summary"]["indoor_scene_count"] == 1
    assert inventory["summary"]["static_multiview_eligible_count"] == 1
    indoor = next(scene for scene in inventory["scenes"] if scene["scene_kind"] == "indoor")
    assert indoor["classification"] == "multi_room_rich"
    assert indoor["rearrangeable_object_count"] == 24
    assert output.is_file()


def test_sweep_plan_is_deterministic_and_refreshable(tmp_path: Path) -> None:
    assets = tmp_path / "behavior-assets"
    _fake_scene(assets, "home_int", ["kitchen_0", "living_room_0"])
    inventory_path = tmp_path / "inventory.json"
    build_scene_inventory(assets, inventory_path)
    plan_path = tmp_path / "plan.json"
    output_root = tmp_path / "sweep"
    plan = build_sweep_plan(
        inventory_path=inventory_path,
        base_recipe_path=ROOT / "configs" / "omnigibson_static_m1.yaml",
        output_root=output_root,
        plan_path=plan_path,
    )
    assert plan["job_count"] == 1
    assert plan["jobs"][0]["status"] == "pending"

    legacy = json.loads(plan_path.read_text())
    del legacy["jobs"][0]["receipt"]
    plan_path.write_text(json.dumps(legacy))
    migrated = refresh_sweep_plan(plan_path)
    assert migrated["jobs"][0]["receipt"].endswith("home_int_seed17.json")

    bundle = Path(plan["jobs"][0]["bundle"])
    bundle.mkdir(parents=True)
    (bundle / "failure_report.json").write_text(
        json.dumps({"error": {"message": "synthetic failure"}})
    )
    refreshed = refresh_sweep_plan(plan_path)
    assert refreshed["status_counts"] == {"failed": 1}

    recipe_path = materialize_job_recipe(
        base_recipe_path=ROOT / "configs" / "omnigibson_static_m1.yaml",
        scene_model="home_int",
        seed=23,
        output_path=tmp_path / "job.yaml",
    )
    text = recipe_path.read_text()
    assert "scene_model: home_int" in text
    assert "seed: 23" in text

    review = build_sweep_review(plan_path)
    assert review.is_file()
    assert "Episode3D 场景审阅台" in review.read_text()


def test_typed_sweep_names_class_and_inherits_scene_split(tmp_path: Path) -> None:
    assets = tmp_path / "behavior-assets"
    _fake_scene(assets, "home_int", ["kitchen_0", "living_room_0"])
    inventory_path = tmp_path / "inventory.json"
    build_scene_inventory(assets, inventory_path)
    split_path = tmp_path / "splits.json"
    split_path.write_text(
        json.dumps(
            {
                "assignments": [
                    {
                        "scene_model": "home_int",
                        "split": "train",
                        "split_group": "scene:home_int",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    plan = build_sweep_plan(
        inventory_path=inventory_path,
        base_recipe_path=ROOT / "configs" / "omnigibson_t7_elevation.yaml",
        output_root=tmp_path / "sweep",
        split_manifest_path=split_path,
    )

    assert plan["trajectory_class"] == "T7"
    assert plan["jobs"][0]["job_id"] == "home_int_t7_seed17"
    assert plan["jobs"][0]["split"] == "train"
    assert plan["jobs"][0]["split_group"] == "scene:home_int"


def test_scene_overrides_are_validated_and_materialized(tmp_path: Path) -> None:
    assets = tmp_path / "behavior-assets"
    _fake_scene(assets, "home_int", ["kitchen_0", "living_room_0"])
    inventory_path = tmp_path / "inventory.json"
    build_scene_inventory(assets, inventory_path)
    overrides_path = tmp_path / "overrides.yaml"
    overrides_path.write_text(
        """schema_version: omnigibson_scene_overrides.v1
overrides:
  home_int:
    recipe:
      trajectory:
        candidate_count: 256
        fallback_minimum_distance_m: 0.5
    acquisition_timeout_minutes: 40
""",
        encoding="utf-8",
    )
    plan = build_sweep_plan(
        inventory_path=inventory_path,
        base_recipe_path=ROOT / "configs" / "omnigibson_static_m2_room_aware.yaml",
        output_root=tmp_path / "sweep",
        scene_overrides_path=overrides_path,
    )
    job = plan["jobs"][0]
    assert job["recipe_overrides"]["trajectory"]["candidate_count"] == 256
    assert job["acquisition_timeout_seconds"] == 2400
    assert plan["scene_overrides_sha256"]

    recipe_path = materialize_job_recipe(
        base_recipe_path=ROOT / "configs" / "omnigibson_static_m2_room_aware.yaml",
        scene_model="home_int",
        seed=17,
        output_path=tmp_path / "job.yaml",
        overrides=job["recipe_overrides"],
    )
    recipe_text = recipe_path.read_text(encoding="utf-8")
    assert "candidate_count: 256" in recipe_text
    assert "fallback_minimum_distance_m: 0.5" in recipe_text

    job["recipe_overrides"]["trajectory"]["misspelled_field"] = 1
    with pytest.raises(ValueError, match="unknown override field"):
        materialize_job_recipe(
            base_recipe_path=ROOT / "configs" / "omnigibson_static_m2_room_aware.yaml",
            scene_model="home_int",
            seed=17,
            output_path=tmp_path / "invalid.yaml",
            overrides=job["recipe_overrides"],
        )
