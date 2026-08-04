from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from omnigibson_episode.io import sha256_file
from omnigibson_episode.quality import _among5_gate, _object_orbit_gate, inspect_bundle


def test_among5_gate_replays_cardinal_layout_and_multiview_visibility() -> None:
    names = ["anchor", "front", "right", "back", "left"]
    positions = {
        "anchor": [0.0, 0.0, 0.3],
        "front": [0.0, 1.1, 0.3],
        "right": [1.1, 0.0, 0.3],
        "back": [0.0, -1.1, 0.3],
        "left": [-1.1, 0.0, 0.3],
    }
    placements = [
        {
            "name": name,
            "category": f"category_{name}",
            "role": "anchor" if name == "anchor" else "satellite",
            "world_bearing_deg": bearing,
        }
        for name, bearing in zip(names, (0.0, 0.0, 90.0, 180.0, 270.0), strict=True)
    ]
    snapshot = {
        "entities": [
            {"name": name, "aabb_center_m": position}
            for name, position in positions.items()
        ],
        "runtime_instance_registry": {
            str(index + 2): name for index, name in enumerate(names)
        },
    }
    render = {
        "views": [
            {"visible_runtime_instance_ids": [2, 3, 4, 5, 6]},
            {"visible_runtime_instance_ids": [2, 3, 4, 5, 6]},
            {"visible_runtime_instance_ids": [2, 3, 4, 5, 6]},
            {"visible_runtime_instance_ids": [2, 3, 4, 5, 6]},
        ]
    }
    selection = {
        "placements": placements,
        "layout_radius_m": 1.1,
        "minimum_declared_object_gap_m": 0.2,
        "required_object_gap_m": 0.1,
        "camera_checks": [
            {"same_room": True, "traversable": True} for _ in range(4)
        ],
    }

    gate = _among5_gate(
        trajectory={"views": [{}, {}, {}, {}]},
        render=render,
        snapshot=snapshot,
        selection=selection,
    )

    assert gate["status"] == "pass"
    assert gate["checks"]["cardinal_bearing_error_at_most_5_deg"] is True
    assert gate["visible_view_count_by_asset"]["anchor"] == 4


def test_mindcube_among_gate_rejects_single_view_shortcuts() -> None:
    names = ["anchor", "front", "right", "back", "left"]
    placements = [
        {
            "name": name,
            "category": f"category_{name}",
            "role": "anchor" if name == "anchor" else "satellite",
            "world_bearing_deg": bearing,
        }
        for name, bearing in zip(names, (0.0, 0.0, 90.0, 180.0, 270.0), strict=True)
    ]
    snapshot = {
        "entities": [
            {"name": name, "aabb_center_m": position}
            for name, position in zip(
                names,
                ([0, 0, 0.2], [0, 1.6, 0.4], [1.6, 0, 0.4], [0, -1.6, 0.4], [-1.6, 0, 0.4]),
                strict=True,
            )
        ],
        "runtime_instance_registry": {
            str(index + 2): name for index, name in enumerate(names)
        },
    }
    visible = [[2, 3], [2, 4], [2, 5], [2, 6]]
    selection = {
        "visibility_contract": "central_plus_one_partial",
        "placements": placements,
        "layout_radius_m": 1.6,
        "minimum_declared_object_gap_m": 0.4,
        "required_object_gap_m": 0.25,
        "camera_checks": [
            {
                "same_room": True,
                "traversable": True,
                "expected_target_name": target,
            }
            for target in names[1:]
        ],
    }
    gate = _among5_gate(
        trajectory={"views": [{}, {}, {}, {}]},
        render={
            "views": [
                {"visible_runtime_instance_ids": identifiers}
                for identifiers in visible
            ]
        },
        snapshot=snapshot,
        selection=selection,
    )

    assert gate["status"] == "pass"
    assert gate["checks"]["no_single_view_reveals_full_layout"] is True
    shortcut_render = {
        "views": [
            {"visible_runtime_instance_ids": [2, 3, 4, 5, 6]}
            for _ in range(4)
        ]
    }
    failed = _among5_gate(
        trajectory={"views": [{}, {}, {}, {}]},
        render=shortcut_render,
        snapshot=snapshot,
        selection=selection,
    )
    assert failed["status"] == "fail"
    assert failed["checks"]["no_single_view_reveals_full_layout"] is False


def test_partial_t4_arc_must_meet_its_declared_coverage() -> None:
    azimuths = [270.0 * index / 7 for index in range(8)]
    trajectory = {
        "focus_entity_id": "chair_0",
        "azimuth_deg_per_view": azimuths,
        "orientation_questions_allowed": False,
    }
    render = {
        "views": [
            {"visible_runtime_instance_ids": [2, 3, 4, 5]}
            for _ in azimuths
        ]
    }
    snapshot = {"runtime_instance_registry": {"2": "chair_0"}}
    selection = {
        "arc_coverage_deg": 270.0,
        "minimum_arc_required_deg": 270.0,
        "complete_orbit": False,
        "adaptive_radius_used": True,
    }

    gate = _object_orbit_gate(
        trajectory=trajectory,
        render=render,
        snapshot=snapshot,
        selection=selection,
    )

    assert gate["status"] == "pass"
    assert gate["checks"]["opposite_view_pair_exists"] is True
    assert gate["checks"]["arc_meets_declared_minimum"] is True
    assert gate["complete_orbit"] is False
    assert gate["adaptive_radius_used"] is True

    trajectory["azimuth_deg_per_view"] = [250.0 * index / 7 for index in range(8)]
    failed = _object_orbit_gate(
        trajectory=trajectory,
        render=render,
        snapshot=snapshot,
        selection=selection,
    )
    assert failed["status"] == "fail"


def test_inspect_bundle_verifies_channels_and_writes_preview(tmp_path: Path) -> None:
    views = tmp_path / "views"
    views.mkdir()
    artifact = views / "v000.sensors.npz"
    rgb = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    depth = np.full((8, 8), 2.0, dtype=np.float32)
    instance = np.full((8, 8), 2, dtype=np.uint32)
    semantic = np.full((8, 8), 7, dtype=np.uint32)
    np.savez_compressed(
        artifact,
        rgb=rgb,
        depth_m=depth,
        instance_id=instance,
        semantic_id=semantic,
    )
    render = {
        "status": "success",
        "sensor_contract": {
            "height_px": 8,
            "width_px": 8,
            "near_m": 0.05,
            "far_m": 30.0,
            "minimum_visible_instance_pixels": 8,
        },
        "views": [
            {
                "view_id": "v000",
                "artifact": {"path": str(artifact), "sha256": sha256_file(artifact)},
                "valid_depth_fraction": 1.0,
                "visible_runtime_semantic_ids": [2],
            },
            {
                "view_id": "v001",
                "artifact": {"path": str(artifact), "sha256": sha256_file(artifact)},
                "valid_depth_fraction": 1.0,
                "visible_runtime_semantic_ids": [2],
            },
        ],
    }
    (tmp_path / "render_report.json").write_text(json.dumps(render), encoding="utf-8")

    report = inspect_bundle(tmp_path)

    assert report["integrity_status"] == "pass"
    assert report["view_count"] == 2
    assert report["loop_closure"] == {
        "first_view_id": "v000",
        "last_view_id": "v001",
        "rgb_exact": True,
            "rgb_mae": 0.0,
            "rgb_psnr_db": None,
            "last_minus_first_channel_mean": [0.0, 0.0, 0.0],
            "bias_corrected_rgb_psnr_db": None,
            "rgb_structure_correlation": 1.0,
            "exposure_shift_likely": False,
            "depth_exact": True,
        "instance_exact": True,
        "semantic_exact": True,
    }
    assert (tmp_path / "quality_report.json").is_file()
    assert (tmp_path / "preview.html").is_file()
    assert (tmp_path / "preview" / "v000.png").is_file()


def test_rotation_station_uses_t3_gate_instead_of_loop_gate(tmp_path: Path) -> None:
    views_root = tmp_path / "views"
    views_root.mkdir()
    render_views = []
    yaws = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    visible_by_view = [
        list(range(2, 10)),
        list(range(2, 10)),
        list(range(10, 18)),
        list(range(10, 18)),
        list(range(10, 18)),
        list(range(2, 10)),
    ]
    for index, visible in enumerate(visible_by_view):
        view_id = f"view-{index:03d}"
        artifact = views_root / f"{view_id}.sensors.npz"
        rgb = np.full((8, 8, 3), index * 20, dtype=np.uint8)
        depth = np.full((8, 8), 2.0, dtype=np.float32)
        instance = np.full((8, 8), visible[0], dtype=np.uint32)
        semantic = instance.copy()
        np.savez_compressed(
            artifact,
            rgb=rgb,
            depth_m=depth,
            instance_id=instance,
            semantic_id=semantic,
        )
        render_views.append(
            {
                "view_id": view_id,
                "artifact": {"path": str(artifact), "sha256": sha256_file(artifact)},
                "valid_depth_fraction": 1.0,
                "visible_runtime_semantic_ids": [visible[0]],
                "visible_runtime_instance_ids": visible,
            }
        )
    render = {
        "status": "success",
        "sensor_contract": {
            "height_px": 8,
            "width_px": 8,
            "near_m": 0.05,
            "far_m": 30.0,
            "minimum_visible_instance_pixels": 8,
        },
        "views": render_views,
    }
    trajectory = {
        "closed_loop": False,
        "trajectory_class": "T3",
        "station_id": "living_room_0:station0",
        "yaw_sequence_deg": yaws,
        "depth_questions_forbidden": True,
        "views": [
            {"world_from_agent": {"translation_m": [1.0, 2.0, 0.0]}}
            for _ in yaws
        ],
    }
    entities = [
        {"name": f"object_{identifier}", "category": "chair"}
        for identifier in range(2, 18)
    ]
    snapshot = {
        "entities": entities,
        "runtime_instance_registry": {
            str(identifier): f"object_{identifier}" for identifier in range(2, 18)
        },
    }
    (tmp_path / "render_report.json").write_text(json.dumps(render), encoding="utf-8")
    (tmp_path / "trajectory_plan.json").write_text(
        json.dumps(trajectory), encoding="utf-8"
    )
    (tmp_path / "scene_snapshot.json").write_text(
        json.dumps(snapshot), encoding="utf-8"
    )

    report = inspect_bundle(tmp_path)

    assert report["loop_closure"] is None
    assert report["trajectory_status"] == "pass"
    assert report["gates"]["T3"]["maximum_translational_baseline_m"] == 0.0
    assert report["gates"]["T3"]["rear_only_core_entity_count"] == 8
