"""Regression: navfix must reject the three pre-std.v3 scripted plans."""

from __future__ import annotations

import pytest
from _batchdata import BATCH_ROOT, SCENE_IR_PATH, load_plan_record

from spatial_episode.scriptgen.behavior import layout_from_scene_ir
from spatial_episode.scriptgen.generate import generate_plans
from spatial_episode.scriptgen.library import SELF_MOTION
from spatial_episode.scriptgen.predicates import get_predicate
from spatial_episode.scriptgen.sceneview import GeometrySceneView, Pose2D
from spatial_episode.scriptgen.standards import STD_V1

needs_legacy_plans = pytest.mark.skipif(
    not (BATCH_ROOT / "plan_0.record.json").is_file() or not SCENE_IR_PATH.is_file(),
    reason="pre-navfix scripted plans not present on this host",
)

EXPECTED_WORST = {
    0: {"poses_clear": "sofa", "path_clear": "coffee_table"},
    1: {"poses_clear": "bed", "path_clear": "armchair"},
    2: {"poses_clear": "bed", "path_clear": "armchair"},
}


@needs_legacy_plans
@pytest.mark.parametrize("index", [0, 1, 2])
def test_legacy_plan_is_rejected_by_clearance_predicates(index: int) -> None:
    plan = load_plan_record(index)
    layout = layout_from_scene_ir(SCENE_IR_PATH)
    poses = tuple(Pose2D(row["x"], row["y"], row["yaw_deg"]) for row in plan["poses"])
    view = GeometrySceneView(layout=layout, poses=poses, std=STD_V1)
    frames = range(view.frame_count)

    for predicate_name, expected_obstacle in EXPECTED_WORST[index].items():
        verdict = get_predicate(predicate_name)(view, STD_V1, frames=frames)
        assert verdict.holds is False
        assert verdict.witness["worst_collision"]["obstacle"] == expected_obstacle


@needs_legacy_plans
def test_navfix_motif_generates_clear_real_scene_plans() -> None:
    layout = layout_from_scene_ir(SCENE_IR_PATH)
    report = generate_plans(layout, SELF_MOTION, STD_V1, seed=17, max_plans=3)
    assert report.plans, report.rejection_counts
    for plan in report.plans:
        poses = tuple(Pose2D(row.x, row.y, row.yaw_deg) for row in plan.poses)
        view = GeometrySceneView(layout=layout, poses=poses, std=STD_V1)
        frames = range(view.frame_count)
        assert get_predicate("poses_clear")(view, STD_V1, frames=frames).holds is True
        assert get_predicate("path_clear")(view, STD_V1, frames=frames).holds is True
