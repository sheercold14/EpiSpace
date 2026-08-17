"""BEHAVIOR bundle adapter tests.

Pure-math tests always run; tests that need a real acquired bundle skip
cleanly when the sweep outputs are not present on this machine.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
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
from spatial_episode.scriptgen.geometry import bearing_deg, wrap_deg
from spatial_episode.scriptgen.motifs import walk_to_occlusion
from spatial_episode.scriptgen.predicates import get_predicate
from spatial_episode.scriptgen.sceneview import (
    GeometrySceneView,
    Obstacle,
    Pose2D,
    SceneLayout,
    SceneObject,
    blocking_occluders,
)

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


def test_potential_occluders_keep_vertical_spans_for_target_specific_rays() -> None:
    scene_ir = {
        "scene_id": "height-occluder",
        "entities": [
            _scene_ir_entity("floors", "floor", (0.0, 0.0, 0.0), (4.0, 4.0, 0.05)),
            _scene_ir_entity("bookcase", "tall", (0.0, 0.0, 1.0), (0.5, 0.2, 1.0)),
            _scene_ir_entity("table", "low", (2.0, 0.0, 0.4), (0.5, 0.5, 0.4)),
        ],
    }
    layout = layout_from_scene_ir(scene_ir)
    assert {obstacle.entity_id for obstacle in layout.occlusion_obstacles} == {"tall", "low"}


def test_downward_sightline_can_be_blocked_below_camera_height() -> None:
    target = SceneObject(
        "target", "desk", (4.0, 0.0), 1.0, "target", center_z=0.5, half_height=0.5
    )
    layout = SceneLayout(
        "height-aware",
        (target,),
        obstacles=(
            Obstacle("bookcase", (2.0, 0.0), (0.4, 0.5), 0.0, 0.0, 1.48, "bookcase"),
            Obstacle("coffee_table", (1.0, 0.0), (0.4, 0.5), 0.0, 0.0, 0.5, "table"),
        ),
        occlusion_obstacles=(
            Obstacle("bookcase", (2.0, 0.0), (0.4, 0.5), 0.0, 0.0, 1.48, "bookcase"),
            Obstacle("coffee_table", (1.0, 0.0), (0.4, 0.5), 0.0, 0.0, 0.5, "table"),
        ),
    )

    assert blocking_occluders(layout, (0.0, 0.0), target) == ("bookcase",)


def test_ray_cache_survives_concurrent_eviction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planner and QA threads may evict the shared cache concurrently."""
    import sys
    from concurrent.futures import ThreadPoolExecutor

    from spatial_episode.scriptgen import sceneview

    layouts = []
    for index in range(8):
        target = SceneObject(
            f"target-{index}",
            "desk",
            (4.0, 0.0),
            1.0,
            f"target-{index}",
            center_z=0.5,
            half_height=0.5,
        )
        blocker = Obstacle(
            f"blocker-{index}",
            (2.0, 0.0),
            (0.4, 0.5),
            0.0,
            0.0,
            1.5,
            f"blocker-{index}",
        )
        layouts.append(
            SceneLayout(
                f"scene-{index}",
                (target,),
                obstacles=(blocker,),
                occlusion_obstacles=(blocker,),
            )
        )

    monkeypatch.setattr(sceneview, "_MAX_SCENE_OBJECT_RAY_CACHE", 1)
    with sceneview._SCENE_OBJECT_RAY_CACHE_LOCK:
        sceneview._SCENE_OBJECT_RAY_CACHE.clear()
        sceneview._CACHED_RAY_LAYOUTS.clear()
        sceneview._CACHED_RAY_OBJECT_IDS.clear()

    def exercise(index: int) -> tuple[str, ...]:
        layout = layouts[index % len(layouts)]
        target = layout.objects[0]
        result: tuple[str, ...] = ()
        for step in range(100):
            result = blocking_occluders(
                layout,
                (step / 10_000.0, 0.0),
                target,
            )
        return result

    previous_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = tuple(pool.map(exercise, range(32)))
    finally:
        sys.setswitchinterval(previous_interval)
        with sceneview._SCENE_OBJECT_RAY_CACHE_LOCK:
            sceneview._SCENE_OBJECT_RAY_CACHE.clear()
            sceneview._CACHED_RAY_LAYOUTS.clear()
            sceneview._CACHED_RAY_OBJECT_IDS.clear()

    assert all(
        result == (f"blocker-{index % len(layouts)}",)
        for index, result in enumerate(results)
    )


def test_geometry_occlusion_rejects_a_ray_that_only_clips_an_inflated_obb_top() -> None:
    target = SceneObject(
        "target", "desk", (4.0, 0.0), 1.0, "target", center_z=0.5, half_height=0.5
    )
    shallow_clip = Obstacle(
        "bed", (1.5, 0.0), (0.2, 0.5), 0.0, 0.0, 1.3, "bed"
    )
    layout = SceneLayout(
        "inflated-top",
        (target,),
        obstacles=(shallow_clip,),
        occlusion_obstacles=(shallow_clip,),
    )

    # At the near face the downward ray is only 0.125 m below the fitted OBB
    # top.  That is weaker than std.v10's 0.15 m mesh-robustness margin.
    assert blocking_occluders(layout, (0.0, 0.0), target) == ()


def test_walk_to_occlusion_ends_at_a_verified_blocked_free_pose() -> None:
    target = SceneObject(
        "target", "desk", (3.0, 0.0), 1.0, "target", center_z=0.5, half_height=0.5
    )
    blocker = SceneObject(
        "blocker", "bed", (1.5, 0.0), 1.6, "blocker", center_z=0.5, half_height=0.5
    )
    layout = SceneLayout(
        "occluded-route",
        (target, blocker),
        obstacles=(
            Obstacle("desk", (3.0, 0.0), (0.5, 0.5), 0.0, 0.0, 1.0, "target"),
            Obstacle("bed", (1.5, 0.0), (0.8, 0.8), 0.0, 0.0, 1.0, "blocker"),
        ),
        occlusion_obstacles=(
            Obstacle("desk", (3.0, 0.0), (0.5, 0.5), 0.0, 0.0, 1.0, "target"),
            Obstacle("bed", (1.5, 0.0), (0.8, 0.8), 0.0, 0.0, 1.0, "blocker"),
        ),
        walkable_min=(-4.0, -4.0),
        walkable_max=(4.0, 4.0),
    )

    poses = walk_to_occlusion(layout, {"target": "target"}, 16, random.Random(17))

    assert poses is not None
    assert len(poses) == 16
    assert poses[0].xy != poses[-1].xy
    assert blocking_occluders(layout, poses[0].xy, target) == ()
    assert blocking_occluders(layout, poses[-1].xy, target) == ("blocker",)
    assert len({pose.xy for pose in poses[-5:]}) == 1
    for pose in poses[:-5]:
        assert abs(wrap_deg(pose.yaw_deg - bearing_deg(pose.xy, target.xy))) < 1e-9
    final_bearing = bearing_deg(poses[-1].xy, target.xy)
    assert {
        round(wrap_deg(pose.yaw_deg - final_bearing), 6) for pose in poses[-5:]
    } == {-21.0, 0.0, 21.0}


def _render_occlusion_view(
    tmp_path: Path,
    *,
    patch_ids: np.ndarray,
    patch_depth: np.ndarray,
    runtime_entity_ids: dict[int, str],
) -> RenderSceneView:
    bundle = tmp_path / "render-occlusion"
    views = bundle / "views"
    views.mkdir(parents=True)
    instance = np.zeros((64, 64), dtype=np.uint32)
    depth = np.full((64, 64), 30.0, dtype=np.float32)
    instance[28:37, 28:37] = patch_ids
    depth[28:37, 28:37] = patch_depth
    np.savez_compressed(views / "view-000.sensors.npz", instance_id=instance, depth_m=depth)
    target = SceneObject(
        "target", "desk", (4.0, 0.0), 1.0, "target", center_z=1.5, half_height=0.5
    )
    return RenderSceneView(
        layout=SceneLayout("render", (target,)),
        poses=(Pose2D(0.0, 0.0, 0.0),),
        std=STD_V1,
        bundle_root=bundle,
        entity_runtime_ids={"target": (91,)},
        runtime_entity_ids=runtime_entity_ids,
        entity_categories={entity: "bookcase" for entity in runtime_entity_ids.values()},
        sensor_contract={"horizontal_fov_deg": 90.0},
    )


def test_render_occlusion_attribution_uses_instance_and_depth(tmp_path: Path) -> None:
    view = _render_occlusion_view(
        tmp_path,
        patch_ids=np.full((9, 9), 42, dtype=np.uint32),
        patch_depth=np.full((9, 9), 2.0, dtype=np.float32),
        runtime_entity_ids={42: "bookcase"},
    )

    evidence = view.occlusion("target", 0)

    assert evidence.status == "occluded"
    assert evidence.occluder_ids == ("bookcase",)
    assert evidence.witness["backend"] == "render_instance_depth"
    assert evidence.witness["support_pixels"] == 81
    assert evidence.witness["depth_margin_m"] == 2.0


def test_render_visibility_and_occlusion_results_are_cached(tmp_path: Path) -> None:
    view = _render_occlusion_view(
        tmp_path,
        patch_ids=np.full((9, 9), 42, dtype=np.uint32),
        patch_depth=np.full((9, 9), 2.0, dtype=np.float32),
        runtime_entity_ids={42: "bookcase"},
    )

    visibility = view.visibility("target", 0)
    occlusion = view.occlusion("target", 0)
    view._mask_cache.clear()
    view._depth_cache.clear()
    (view.bundle_root / "views" / "view-000.sensors.npz").unlink()

    assert view.visibility("target", 0) is visibility
    assert view.occlusion("target", 0) is occlusion


def test_render_occlusion_attribution_rejects_competing_instances(tmp_path: Path) -> None:
    identifiers = np.full((9, 9), 42, dtype=np.uint32)
    identifiers[:, 5:] = 43
    view = _render_occlusion_view(
        tmp_path,
        patch_ids=identifiers,
        patch_depth=np.full((9, 9), 2.0, dtype=np.float32),
        runtime_entity_ids={42: "bookcase", 43: "chair"},
    )

    evidence = view.occlusion("target", 0)

    assert evidence.status == "ambiguous"
    assert evidence.witness["reason"] == "competing_foreground_instances"


def test_render_occlusion_attribution_rejects_foreground_behind_target(
    tmp_path: Path,
) -> None:
    view = _render_occlusion_view(
        tmp_path,
        patch_ids=np.full((9, 9), 42, dtype=np.uint32),
        patch_depth=np.full((9, 9), 4.1, dtype=np.float32),
        runtime_entity_ids={42: "bookcase"},
    )

    evidence = view.occlusion("target", 0)

    assert evidence.status == "ambiguous"
    assert evidence.witness["reason"] == "insufficient_foreground_support"


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


def test_render_backend_resolves_ids_from_replayed_snapshot(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    views = bundle / "views"
    views.mkdir(parents=True)
    (bundle / "trajectory_plan.json").write_text(
        json.dumps(
            {
                "views": [
                    {
                        "world_from_agent": {
                            "translation_m": [0.0, 0.0, 0.0],
                            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (bundle / "scene_snapshot.json").write_text(
        json.dumps({"runtime_instance_registry": {"91": "chair_source"}}),
        encoding="utf-8",
    )
    scene_ir = tmp_path / "scene_ir.json"
    scene_ir.write_text(
        json.dumps(
            {
                "scene_id": "scene",
                "runtime_semantic_id_map": {"17": "target"},
                "entities": [
                    {
                        "entity_id": "target",
                        "source_entity_id": "chair_source",
                        "raw_label": "chair",
                        "obb": {
                            "center_m": [1.0, 0.0, 0.5],
                            "half_extents_m": [0.5, 0.5, 0.5],
                            "world_from_obb": {"rotation_xyzw": [0.0, 0.0, 0.0, 1.0]},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        views / "view-000.sensors.npz",
        instance_id=np.array([[91, 91], [0, 91]], dtype=np.uint32),
    )

    view = RenderSceneView.from_bundle(bundle, STD_V1, scene_ir=scene_ir)

    assert view.entity_runtime_ids["target"] == (91,)
    assert view.visibility("target", 0).value == 3


@needs_bundles
def test_planning_succeeds_on_residential_scene() -> None:
    layout = layout_from_scene_ir(HOME_BUNDLE / "scene_ir.json")
    result = generate_plans(
        layout, SELF_MOTION, STD_V1, seed=17, attempts_per_binding=80, max_plans=2
    )
    assert result.plans, f"no plans on real home scene; rejections={result.rejection_counts}"
    plan = result.plans[0]
    assert plan.provisional_answer.witness["margin_deg"] >= STD_V1.sector_margin_deg
