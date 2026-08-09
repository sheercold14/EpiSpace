"""Post-render collection audit tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from spatial_episode.scriptgen.behavior import RenderSceneView
from spatial_episode.scriptgen.dataset import _simulator_failure_reason, audit_auxiliary_views
from spatial_episode.scriptgen.sceneview import Pose2D, SceneLayout, SceneObject
from spatial_episode.scriptgen.standards import STD_V1


def test_auxiliary_audit_compares_geometry_to_render_pixels(tmp_path: Path) -> None:
    bundle = tmp_path / "bundles" / "job"
    auxiliary_dir = bundle / "auxiliary_views"
    auxiliary_dir.mkdir(parents=True)
    visible = np.full((32, 32), 91, dtype=np.uint32)
    invisible = np.zeros((32, 32), dtype=np.uint32)
    for view_id, instance in (("aux-visible", visible), ("aux-hidden", invisible)):
        np.savez_compressed(
            auxiliary_dir / f"{view_id}.sensors.npz",
            rgb=np.zeros((32, 32, 3), dtype=np.uint8),
            depth_m=np.ones((32, 32), dtype=np.float32),
            instance_id=instance,
        )
    render_plan = tmp_path / "plans" / "job.views.json"
    render_plan.parent.mkdir()
    render_plan.write_text(
        json.dumps(
            {
                "auxiliary_views": [
                    {
                        "view_id": "aux-visible",
                        "yaw_offset_deg": 0,
                        "target_entity_id": "target",
                        "geometry_label": "visible",
                    },
                    {
                        "view_id": "aux-hidden",
                        "yaw_offset_deg": 45,
                        "target_entity_id": "target",
                        "geometry_label": "not_visible",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    layout = SceneLayout(
        "scene",
        (SceneObject("target", "chair", (1.0, 0.0), 1.0, "target"),),
    )
    view = RenderSceneView(
        layout=layout,
        poses=(Pose2D(0.0, 0.0, 0.0),),
        std=STD_V1,
        bundle_root=bundle,
        entity_runtime_ids={"target": (91,)},
    )
    job = SimpleNamespace(
        render_plan=str(render_plan),
        auxiliary_view_count=2,
        question_group=str(tmp_path / "groups" / "job"),
        bundle=str(bundle),
        job_id="job",
    )

    rows = audit_auxiliary_views(job, view, tmp_path)  # type: ignore[arg-type]

    assert [(row.render_label, row.matches) for row in rows] == [
        ("visible", True),
        ("not_visible", True),
    ]
    assert all((tmp_path / row.rgb).is_file() for row in rows)


def test_simulator_failure_reason_reads_protocol_error_object(tmp_path: Path) -> None:
    failure = tmp_path / "failure_report.json"
    failure.write_text(
        json.dumps(
            {
                "protocol_version": "omnigibson_acquisition_failure.v1",
                "status": "failure",
                "error": {"type": "RuntimeError", "message": "camera hits wall"},
            }
        ),
        encoding="utf-8",
    )

    assert _simulator_failure_reason(failure) == "simulator:RuntimeError:camera hits wall"
