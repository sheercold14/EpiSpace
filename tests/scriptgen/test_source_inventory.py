"""Source inventory scene discovery and strategy contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path

from spatial_episode.scriptgen.dataset_cli import parser
from spatial_episode.scriptgen.source_inventory import (
    SourceSceneRecord,
    _acquisition_ready,
    _bundle_ready,
    _record_still_ready,
    _runtime_env,
    _source_recipe,
    discover_behavior_scenes,
)


def _template(root: Path, name: str, strategy: str) -> None:
    path = root / "configs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                "recipe_id: template",
                "source:",
                "  scene_model: old",
                "trajectory:",
                f"  sampling_strategy: {strategy}",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def test_source_recipe_uses_room_aware_inside_and_traversable_in_gardens(
    tmp_path: Path,
) -> None:
    _template(tmp_path, "omnigibson_static_m1.yaml", "traversable_random")
    _template(tmp_path, "omnigibson_static_m2_room_aware.yaml", "room_aware")

    indoor, indoor_timeout = _source_recipe("Rs_int", og_root=tmp_path)
    garden, garden_timeout = _source_recipe("Rs_garden", og_root=tmp_path)

    assert indoor["source"]["scene_model"] == "Rs_int"
    assert indoor["trajectory"]["sampling_strategy"] == "room_aware"
    assert garden["source"]["scene_model"] == "Rs_garden"
    assert garden["trajectory"]["sampling_strategy"] == "traversable_random"
    assert indoor_timeout == garden_timeout == 20


def test_scene_discovery_is_sorted(tmp_path: Path) -> None:
    root = tmp_path / "behavior-1k-assets" / "scenes"
    (root / "scene_b").mkdir(parents=True)
    (root / "scene_a").mkdir()

    assert discover_behavior_scenes(tmp_path) == ("scene_a", "scene_b")


def test_source_resume_compiles_an_acquired_bundle_without_rerendering(tmp_path: Path) -> None:
    (tmp_path / "scene_snapshot.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "trajectory_plan.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "render_report.json").write_text(
        '{"status":"success"}\n', encoding="utf-8"
    )

    assert _acquisition_ready(tmp_path)
    assert not _bundle_ready(tmp_path)

    (tmp_path / "scene_ir.json").write_text("{}\n", encoding="utf-8")
    assert _bundle_ready(tmp_path)


def test_source_resume_requires_the_same_snapshot_digest(tmp_path: Path) -> None:
    snapshot = tmp_path / "scene_snapshot.json"
    recipe = tmp_path / "recipe.yaml"
    scene_ir = tmp_path / "scene_ir.json"
    snapshot.write_text('{"source_digest":"digest-a"}\n', encoding="utf-8")
    recipe.write_text("source: {}\n", encoding="utf-8")
    scene_ir.write_text("{}\n", encoding="utf-8")
    (tmp_path / "trajectory_plan.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "render_report.json").write_text(
        '{"status":"success"}\n', encoding="utf-8"
    )
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    record = SourceSceneRecord(
        scene_key="scene",
        scene_model="scene",
        strategy="room_aware",
        status="ready",
        bundle=str(tmp_path),
        recipe=str(recipe),
        scene_ir=str(scene_ir),
        scene_snapshot=str(snapshot),
        source_digest="digest-a",
        scene_ir_sha256=sha(scene_ir),
        recipe_sha256=sha(recipe),
        attempts=1,
    )

    assert _record_still_ready(record)
    snapshot.write_text('{"source_digest":"digest-b"}\n', encoding="utf-8")
    assert not _record_still_ready(record)


def test_source_runtime_uses_an_absolute_epispace_import_path(tmp_path: Path) -> None:
    environment = _runtime_env(tmp_path / "og", tmp_path / "data")
    paths = environment["PYTHONPATH"].split(":")

    assert all(Path(path).is_absolute() for path in paths[:2])
    assert any((Path(path) / "spatial_episode").is_dir() for path in paths)


def test_coverage_cli_defaults_to_ten_of_one_hundred_fifty() -> None:
    args = parser().parse_args(
        [
            "coverage-plan",
            "--source-index",
            "source.index.json",
            "--output-root",
            "output",
        ]
    )

    assert args.accepted_per_binding == 10
    assert args.attempts_per_binding == 150
    assert args.initial_attempts_per_binding == 30
    assert args.maximum_multislot_bindings == 128


def test_pipeline_cli_exposes_the_gpu_runtime() -> None:
    args = parser().parse_args(
        [
            "coverage-pipeline",
            "--manifest",
            "coverage.plan.json",
            "--og-root",
            "og",
            "--data-root",
            "data",
            "--gpu-ids",
            "0",
            "1",
        ]
    )

    assert args.gpu_ids == [0, 1]
    assert args.workers is None
