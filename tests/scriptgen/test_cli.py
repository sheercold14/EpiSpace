"""CLI routing between the demo layout and real scene_ir inputs."""

from __future__ import annotations

import json
import sys

from spatial_episode.scriptgen import cli


def _obb(center: list[float], half_extents: list[float]) -> dict[str, object]:
    return {
        "center_m": center,
        "half_extents_m": half_extents,
        "world_from_obb": {
            "parent_frame": "world",
            "child_frame": "obb",
            "convention": "active_child_to_parent",
            "translation_m": center,
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }


def test_cli_uses_real_scene_ir(tmp_path, monkeypatch) -> None:
    scene_ir = tmp_path / "scene_ir.json"
    scene_ir.write_text(
        json.dumps(
            {
                "scene_id": "real_scene_for_cli_test",
                "entities": [
                    {
                        "entity_id": "floor-1",
                        "raw_label": "floors",
                        "obb": _obb([5.0, 5.0, 0.0], [5.0, 5.0, 0.1]),
                    },
                    {
                        "entity_id": "sofa-1",
                        "raw_label": "sofa",
                        "obb": _obb([2.0, 6.5, 0.6], [0.9, 0.5, 0.6]),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "plans.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scriptgen",
            "--scene-ir",
            str(scene_ir),
            "--capability",
            "self_motion_update",
            "--attempts",
            "1",
            "--plans-per-binding",
            "1",
            "--out",
            str(output),
        ],
    )

    assert cli.main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["scene_id"] == "real_scene_for_cli_test"


def test_cli_defaults_to_demo_layout(tmp_path, monkeypatch) -> None:
    output = tmp_path / "plans.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scriptgen",
            "--capability",
            "self_motion_update",
            "--attempts",
            "1",
            "--plans-per-binding",
            "1",
            "--out",
            str(output),
        ],
    )

    assert cli.main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["scene_id"] == "demo_livingroom_v1"
