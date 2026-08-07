"""BEHAVIOR bundle adapter tests.

Pure-math tests always run; tests that need a real acquired bundle skip
cleanly when the sweep outputs are not present on this machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spatial_episode.scriptgen import STD_V1, generate_plans
from spatial_episode.scriptgen.behavior import (
    RenderSceneView,
    layout_from_scene_ir,
    plan_to_agent_views,
    poses_from_trajectory_plan,
    quaternion_xyzw_from_yaw_deg,
    yaw_deg_from_quaternion_xyzw,
)
from spatial_episode.scriptgen.library import SELF_MOTION
from spatial_episode.scriptgen.predicates import get_predicate
from spatial_episode.scriptgen.sceneview import GeometrySceneView, Pose2D

SWEEP_ROOT = Path(
    "/data/shichao/data/dataV100/code/OminiGibson/outputs/sweeps/"
    "static-m2-room-aware-indoor-seed17-v1/bundles"
)
HALL_BUNDLE = SWEEP_ROOT / "hall_arch_wood_seed17"
HOME_BUNDLE = SWEEP_ROOT / "Beechwood_0_int_seed17"

needs_bundles = pytest.mark.skipif(
    not HALL_BUNDLE.exists(), reason="acquired sweep bundles not present on this host"
)


def test_yaw_quaternion_roundtrip() -> None:
    for yaw in (-156.0, -90.0, 0.0, 45.0, 120.0, 180.0):
        q = quaternion_xyzw_from_yaw_deg(yaw)
        recovered = yaw_deg_from_quaternion_xyzw(q)
        assert abs(((recovered - yaw + 180.0) % 360.0) - 180.0) < 1e-6, yaw


def test_plan_to_agent_views_roundtrip() -> None:
    """Exported camera schedule replays to the exact planned poses."""
    from spatial_episode.scriptgen.demo import DEMO_LAYOUT

    report = generate_plans(DEMO_LAYOUT, SELF_MOTION, STD_V1, seed=17, max_plans=1)
    assert report.plans
    plan = report.plans[0]
    views = plan_to_agent_views(plan)
    recovered = poses_from_trajectory_plan({"views": views})
    assert len(recovered) == len(plan.poses)
    for planned, back in zip(plan.poses, recovered, strict=True):
        assert abs(planned.x - back.x) < 1e-9
        assert abs(planned.y - back.y) < 1e-9
        assert abs(((planned.yaw_deg - back.yaw_deg + 180.0) % 360.0) - 180.0) < 1e-6


def _scene_ir_entity(
    label: str,
    entity_id: str,
    center: tuple[float, float, float],
    half_extents: tuple[float, float, float],
) -> dict[str, object]:
    return {
        "entity_id": entity_id,
        "raw_label": label,
        "obb": {
            "center_m": list(center),
            "half_extents_m": list(half_extents),
            "world_from_obb": {"rotation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        },
    }


def test_walls_are_collision_obstacles_but_not_question_objects() -> None:
    scene_ir = {
        "scene_id": "wall-regression",
        "entities": [
            _scene_ir_entity("floors", "floor", (0.0, 0.0, 0.0), (3.0, 3.0, 0.05)),
            _scene_ir_entity("ceilings", "ceiling", (0.0, 0.0, 3.0), (3.0, 3.0, 0.05)),
            _scene_ir_entity("walls", "wall", (0.0, 0.0, 1.5), (0.1, 2.0, 1.5)),
            _scene_ir_entity("armchair", "chair", (2.0, 2.0, 0.5), (0.4, 0.4, 0.5)),
        ],
    }
    layout = layout_from_scene_ir(scene_ir)

    assert [obj.category for obj in layout.objects] == ["armchair"]
    assert [obstacle.label for obstacle in layout.obstacles] == ["walls", "armchair"]

    view = GeometrySceneView(
        layout=layout,
        poses=(Pose2D(-1.0, 0.0, 0.0), Pose2D(1.0, 0.0, 0.0)),
        std=STD_V1,
    )
    assert get_predicate("poses_clear")(view, STD_V1, frames=[0, 1]).holds is True
    crossing = get_predicate("path_clear")(view, STD_V1, frames=[0, 1])
    assert crossing.holds is False
    assert crossing.witness["checked_wall_count"] == 1
    assert crossing.witness["worst_collision"]["obstacle"] == "walls"


@needs_bundles
def test_layout_from_real_scene_ir() -> None:
    layout = layout_from_scene_ir(HALL_BUNDLE / "scene_ir.json")
    assert layout.objects, "expected question-able objects"
    assert len(layout.obstacles) > len(layout.objects)
    assert layout.occluders, "expected wall occluders"
    labels = {obj.category for obj in layout.objects}
    assert "walls" not in labels and "floors" not in labels
    obstacle_labels = {obstacle.label for obstacle in layout.obstacles}
    assert "walls" in obstacle_labels and "floors" not in obstacle_labels
    (min_x, min_y), (max_x, max_y) = layout.walkable_min, layout.walkable_max
    assert min_x < max_x and min_y < max_y


@needs_bundles
def test_poses_from_real_trajectory_plan() -> None:
    poses = poses_from_trajectory_plan(HALL_BUNDLE / "trajectory_plan.json")
    assert len(poses) >= 2
    # Golden value cross-checked by hand from the quaternion in the bundle.
    assert abs(poses[0].yaw_deg - (-156.0)) < 1.0


@needs_bundles
def test_render_backend_matches_render_report() -> None:
    """Mask-based visibility must agree with the acquisition render report."""
    view = RenderSceneView.from_bundle(HALL_BUNDLE, STD_V1)
    report = json.loads((HALL_BUNDLE / "render_report.json").read_text(encoding="utf-8"))
    for t in range(view.frame_count):
        reported = set(report["views"][t]["visible_runtime_instance_ids"])
        for obj in view.objects():
            observation = view.visibility(obj.name, t)
            state = observation.tristate(STD_V1)
            if state is None:
                continue  # ambiguous band: neither side is authoritative
            in_report = any(rid in reported for rid in view.entity_runtime_ids.get(obj.name, ()))
            if state is True:
                assert in_report, (obj.category, t)
            # state False with in_report True is allowed: the report counts
            # any-pixel visibility, our standard requires the threshold.


@needs_bundles
def test_planning_succeeds_on_residential_scene() -> None:
    layout = layout_from_scene_ir(HOME_BUNDLE / "scene_ir.json")
    result = generate_plans(
        layout, SELF_MOTION, STD_V1, seed=17, attempts_per_binding=80, max_plans=2
    )
    assert result.plans, f"no plans on real home scene; rejections={result.rejection_counts}"
    plan = result.plans[0]
    assert plan.provisional_answer.margin_deg >= STD_V1.sector_margin_deg
