from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "coverage_render_shards.py"
SPEC = importlib.util.spec_from_file_location("coverage_render_shards", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SHARDS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SHARDS)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_scene_assignment_balances_whole_scenes() -> None:
    assignments = SHARDS._assign_scenes(
        {"large": 100, "medium": 60, "small": 40},
        {"large": 10, "medium": 6, "small": 4},
        2,
    )

    assert assignments[0]["scene_keys"] == ["large"]
    assert assignments[1]["scene_keys"] == ["medium", "small"]
    assert sum(item["pending_candidates"] for item in assignments) == 20


def test_build_and_prepare_rebases_all_runtime_paths(
    tmp_path: Path, monkeypatch
) -> None:
    source_index = tmp_path / "source.index.json"
    scene_root = tmp_path / "source_scene"
    source_recipe = tmp_path / "source.yaml"
    plan_record = tmp_path / "candidate.record.json"
    render_plan = tmp_path / "candidate.views.json"
    render_recipe = tmp_path / "candidate.yaml"
    _write_json(source_index, {"schema_version": "scriptgen_source_index.v1"})
    _write_json(scene_root / "scene_ir.json", {"entities": []})
    _write_json(scene_root / "scene_snapshot.json", {"source_digest": "digest"})
    source_recipe.write_text("source:\n  scene_model: scene\n", encoding="utf-8")
    _write_json(plan_record, {"plan_id": "plan"})
    _write_json(render_plan, {"views": [{}, {}], "auxiliary_views": []})
    render_recipe.write_text(
        "trajectory:\n  sampling_strategy: scripted_plan\n"
        f"  plan_path: {render_plan}\n",
        encoding="utf-8",
    )
    candidate = {
        "candidate_id": "candidate",
        "plan_id": "plan",
        "attempt_index": 0,
        "plan_record": str(plan_record),
        "render_plan": str(render_plan),
        "recipe": str(render_recipe),
        "bundle": str(tmp_path / "master" / "bundles" / "candidate"),
        "group": str(tmp_path / "master" / "groups" / "candidate"),
        "log": str(tmp_path / "master" / "logs" / "candidate.log"),
    }
    manifest = {
        "schema_version": "scriptgen_binding_coverage.v1",
        "collection_id": "collection",
        "standard_version": "std.v10",
        "source_index": str(source_index),
        "source_index_sha256": SHARDS._sha256(source_index),
        "output_root": str(tmp_path / "master"),
        "accepted_per_binding": 10,
        "attempts_per_binding": 150,
        "initial_attempts_per_binding": 30,
        "maximum_multislot_bindings": 128,
        "limit_bindings_per_capability": None,
        "capabilities": ["self_motion_update"],
        "scenes": [
            {
                "scene_key": "scene",
                "scene_id": "scene-id",
                "source_scene_id": "source-id",
                "scene_model": "scene",
                "scene_instance": None,
                "source_digest": "digest",
                "scene_ir": str(scene_root / "scene_ir.json"),
                "source_recipe": str(source_recipe),
                "scene_ir_sha256": SHARDS._sha256(scene_root / "scene_ir.json"),
                "source_recipe_sha256": SHARDS._sha256(source_recipe),
            }
        ],
        "scene_skips": {},
        "cells": [
            {
                "cell_id": "cell",
                "scene_key": "scene",
                "scene_id": "scene-id",
                "capability": "self_motion_update",
                "binding": {"target": "target"},
                "seed": 17,
                "target_accepted": 10,
                "geometry_pool_size": 1,
                "diverse_pool_size": 1,
                "search_attempt_limit": 30,
                "raw_plan_limit": 20,
                "search_pool_exhausted": False,
                "rejection_counts": {},
                "candidates": [candidate],
            }
        ],
    }
    status = {
        "schema_version": "scriptgen_binding_coverage_status.v1",
        "collection_id": "collection",
        "cells": {
            "cell": {
                "status": "planned",
                "accepted_episode_ids": [],
                "candidate_statuses": {
                    "candidate": {"status": "pending", "reason": None}
                },
            }
        },
        "episodes": {},
    }
    manifest_path = tmp_path / "coverage.plan.json"
    status_path = tmp_path / "coverage.status.json"
    _write_json(manifest_path, manifest)
    _write_json(status_path, status)
    package_root = tmp_path / "package"
    SHARDS._build_one_package(
        package_root=package_root,
        manifest=manifest,
        status=status,
        source_manifest_path=manifest_path,
        source_status_path=status_path,
        assignment={
            "index": 0,
            "scene_keys": ["scene"],
            "estimated_frames": 2,
            "pending_candidates": 1,
        },
        shard_count=1,
        hardlink=False,
        code_revisions={
            "epispace": {"remote": "example/epispace", "commit": "a" * 40},
            "omnigibson_episode": {"remote": "example/backend", "commit": "b" * 40},
        },
    )
    assert not (package_root / "code").exists()
    assert not (package_root / "backend").exists()
    assert not (package_root / "tools").exists()
    work_root = tmp_path / "work"
    args = type("Args", (), {"package": package_root, "work_root": work_root})()
    SHARDS.command_prepare(args)

    runtime_manifest = json.loads(
        (work_root / "output" / "coverage.plan.json").read_text(encoding="utf-8")
    )
    runtime_candidate = runtime_manifest["cells"][0]["candidates"][0]
    assert runtime_manifest["source_index"] == str(
        package_root.resolve() / "sources" / "source.index.json"
    )
    assert runtime_candidate["bundle"] == str(
        work_root.resolve() / "output" / "bundles" / "candidate"
    )
    runtime_recipe = (work_root / "output" / "recipes" / "candidate.yaml").read_text(
        encoding="utf-8"
    )
    assert str(package_root.resolve() / "planning" / "plans" / "candidate.views.json") in runtime_recipe
    assert SHARDS.PACKAGE_TOKEN not in runtime_recipe


def test_remote_backfill_candidate_is_copied_and_rebased(tmp_path: Path) -> None:
    result_root = tmp_path / "remote" / "output"
    master_output = tmp_path / "master"
    candidate_id = "scene__capability__binding__a031"
    _write_json(result_root / "plans" / f"{candidate_id}.record.json", {"plan_id": "p"})
    _write_json(result_root / "plans" / f"{candidate_id}.views.json", {"views": []})
    (result_root / "recipes").mkdir(parents=True)
    (result_root / "recipes" / f"{candidate_id}.yaml").write_text(
        "trajectory:\n"
        f"  plan_path: {result_root}/plans/{candidate_id}.views.json\n",
        encoding="utf-8",
    )

    localized = SHARDS._localize_remote_candidate(
        {
            "candidate_id": candidate_id,
            "plan_id": "p",
            "attempt_index": 31,
            "plan_record": "remote-record",
            "render_plan": "remote-views",
            "recipe": "remote-recipe",
            "bundle": "remote-bundle",
            "group": "remote-group",
            "log": "remote-log",
        },
        result_root=result_root,
        master_output=master_output,
    )

    assert localized["plan_record"] == str(
        master_output / "plans" / f"{candidate_id}.record.json"
    )
    assert localized["bundle"] == str(master_output / "bundles" / candidate_id)
    recipe = Path(localized["recipe"]).read_text(encoding="utf-8")
    assert str(master_output / "plans" / f"{candidate_id}.views.json") in recipe
    assert str(result_root) not in recipe
