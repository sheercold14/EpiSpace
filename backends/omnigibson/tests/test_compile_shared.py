import hashlib
import json
from pathlib import Path

import pytest

from omnigibson_episode.compile_shared import compile_bundle
from omnigibson_episode.geometry import build_closed_trajectory, stable_scene_id
from omnigibson_episode.presentation import build_oral_presentation
from omnigibson_episode.quality import inspect_bundle
from omnigibson_episode.reasoning_audit import audit_reasoning_bundle


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_fake_acquisition_compiles_to_shared_spatial_episode(tmp_path: Path) -> None:
    import numpy as np
    from spatial_episode.contracts.episode_v1 import SpatialEpisodeV1

    source_version = "behavior-test"
    source_scene_id = "test_room:best"
    scene_id = stable_scene_id(source_version, source_scene_id)
    snapshot = {
        "protocol_version": "omnigibson_scene_snapshot.v1",
        "source_name": "omnigibson",
        "source_version": source_version,
        "source_scene_id": source_scene_id,
        "source_digest": "0" * 64,
        "license": {
            "identifier": "TEST",
            "academic_only": True,
            "redistribution": "metadata_only",
        },
        "entities": [
            {
                "source_entity_id": "chair_0",
                "category": "chair",
                "model": "chair_model",
                "prim_path": "/World/chair_0",
                "region": "",
                "aabb_center_m": [-1.0, 1.0, 0.5],
                "aabb_extent_m": [0.6, 0.6, 1.0],
                "articulated": False,
                "editable": True,
            },
            {
                "source_entity_id": "table_0",
                "category": "table",
                "model": "table_model",
                "prim_path": "/World/table_0",
                "region": "   ",
                "aabb_center_m": [1.0, 1.0, 0.5],
                "aabb_extent_m": [1.0, 1.0, 1.0],
                "articulated": False,
                "editable": True,
            },
        ],
        "runtime_instance_registry": {"2": "chair_0", "3": "table_0"},
    }
    trajectory = build_closed_trajectory(
        scene_id=scene_id,
        recipe_id="fake_og",
        seed=0,
        source_points=[(0, 0, 0), (0, 2, 0), (0, 4, 0)],
        outbound_view_count=3,
        floor_area_m2=25.0,
        backend_version="fake",
    )
    artifact = tmp_path / "sensors.npz"
    instance = np.full((8, 8), 2, dtype=np.uint32)
    instance[:, 4:] = 3
    np.savez_compressed(
        artifact,
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_m=np.ones((8, 8), dtype=np.float32),
        instance_id=instance,
        semantic_id=np.ones((8, 8), dtype=np.uint32),
    )
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    rendered_views = [
        {
            "step": view["step"],
            "view_id": view["view_id"],
            "artifact": {
                "path": str(artifact),
                "sha256": digest,
                "byte_size": artifact.stat().st_size,
                "media_type": "application/x.numpy-npz",
            },
            "visible_runtime_semantic_ids": [2, 3],
            "visible_instance_count": 2,
            "valid_depth_fraction": 1.0,
            "camera_height_m": 1.5,
        }
        for view in trajectory["views"]
    ]
    report = {
        "protocol_version": "omnigibson_render_bundle.v1",
        "status": "success",
        "scene_id": scene_id,
        "sensor_contract": {
            "width_px": 8,
            "height_px": 8,
            "horizontal_fov_deg": 90.0,
            "sensor_height_m": 1.5,
            "near_m": 0.05,
            "far_m": 30.0,
            "minimum_visible_instance_pixels": 8,
        },
        "views": rendered_views,
    }
    _write_json(tmp_path / "scene_snapshot.json", snapshot)
    _write_json(tmp_path / "trajectory_plan.json", trajectory)
    _write_json(tmp_path / "render_report.json", report)
    output = tmp_path / "spatial_episode.json"
    episode = compile_bundle(bundle_directory=tmp_path, output_path=output)
    validated = SpatialEpisodeV1.model_validate_json(output.read_text(encoding="utf-8"))
    assert validated.episode_id == episode.episode_id
    assert len(validated.observations) == 5
    assert validated.queries
    assert {query.answer for query in validated.queries} & {"left_of", "right_of"}
    assert (tmp_path / "scene_ir.json").is_file()
    assert (tmp_path / "relation_oracle.json").is_file()

    quality = inspect_bundle(tmp_path)
    audit = audit_reasoning_bundle(tmp_path)
    presentation = build_oral_presentation(tmp_path)
    page = presentation.read_text(encoding="utf-8")
    assert quality["integrity_status"] == "pass"
    assert audit["reasoning_status"] == "incomplete"
    assert audit["capability_matrix"]["R"]["sampled"] is True
    assert (tmp_path / "reasoning_audit.json").is_file()
    assert presentation == tmp_path / "oral_demo" / "index.html"
    assert "omnigibson_oral_demo.v1" in page
    assert "Isolated-QA SFT" in page
    assert "__EPISODE_DATA__" not in page
    assert len(list((tmp_path / "oral_demo" / "media").glob("*.webp"))) == 25


def test_compile_rejects_tampered_render_evidence(tmp_path: Path) -> None:
    report = {
        "status": "success",
        "views": [
            {
                "artifact": {
                    "path": str(tmp_path / "view.npz"),
                    "sha256": "0" * 64,
                    "byte_size": 3,
                }
            }
        ],
    }
    (tmp_path / "view.npz").write_bytes(b"bad")
    from omnigibson_episode.compile_shared import _verify_render_evidence

    with pytest.raises(ValueError, match="digest mismatch"):
        _verify_render_evidence(tmp_path, report)
