import math

import numpy as np
import pytest

from omnigibson_episode.acquire import (
    _instance_id,
    _remap_renderer_instances,
    _semantic_from_instances,
    _semantic_id,
)
from omnigibson_episode.geometry import (
    build_among5_trajectory,
    build_closed_trajectory,
    build_elevation_trajectory,
    build_object_orbit_trajectory,
    build_occlusion_reveal_trajectory,
    build_rotation_station_trajectory,
    build_target_view_trajectory,
    clockwise_yaw_quaternion,
    dominant_planar_relation,
    omnigibson_camera_quaternion,
    planar_aabb_half_extents_in_yaw_frame,
    planar_delta_to_world,
    planar_deltas_in_yaw_frame,
    resample_polyline,
    yaw_pitch_quaternion,
    yaw_quaternion,
)
from omnigibson_episode.quality import _rgb_loop_diagnostics


def test_planar_frame_transform_round_trips_and_classifies() -> None:
    rotation = yaw_quaternion((0.0, 0.0), (1.0, 1.0))
    right, forward = planar_deltas_in_yaw_frame([2.0, 0.5], [0.0, 0.0], rotation)
    dx, dy = planar_delta_to_world(right, forward, rotation)
    relation, axis, margin, measured_right, measured_forward = dominant_planar_relation(
        [2.0, 0.5], [0.0, 0.0], rotation
    )

    assert (dx, dy) == pytest.approx((2.0, 0.5))
    assert measured_right == pytest.approx(right)
    assert measured_forward == pytest.approx(forward)
    assert axis in {"right", "forward"}
    assert relation in {"left_of", "right_of", "in_front_of", "behind"}
    assert margin >= 0.0
    projected = planar_aabb_half_extents_in_yaw_frame([0.5, 0.25], rotation)
    assert all(value > 0.0 for value in projected)


def test_resampling_stays_on_source_segments() -> None:
    samples, distance = resample_polyline([(0, 0, 0), (0, 2, 0), (2, 2, 0)], 5)
    assert distance == pytest.approx(4.0)
    assert samples == pytest.approx(
        [(0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 2, 0), (2, 2, 0)]
    )


def test_camera_quaternion_is_unit_and_applies_optical_axis_adapter() -> None:
    yaw = yaw_quaternion((0, 0, 0), (0, 1, 0))
    camera = omnigibson_camera_quaternion(yaw)
    assert math.sqrt(sum(value * value for value in camera)) == pytest.approx(1.0)
    assert camera == pytest.approx((math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)))


def test_clockwise_yaw_quaternion_uses_agent_forward_convention() -> None:
    rotation = clockwise_yaw_quaternion(90.0)

    assert rotation == pytest.approx((0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5)))


def test_closed_trajectory_satisfies_shared_contract() -> None:
    from spatial_episode.contracts.trajectory_v1 import TrajectoryPlanV1

    payload = build_closed_trajectory(
        scene_id="8b8cb560-e2eb-575f-a8ff-203661255e4c",
        recipe_id="test",
        seed=4,
        source_points=[(0, 0, 0), (0, 3, 0), (2, 3, 0)],
        outbound_view_count=4,
        floor_area_m2=20.0,
        backend_version="test",
    )
    plan = TrajectoryPlanV1.model_validate(payload)
    assert plan.closed_loop
    assert len(plan.views) == 7
    assert (
        plan.views[-1].world_from_agent.translation_m
        == plan.views[0].world_from_agent.translation_m
    )
    assert plan.views[-1].cumulative_distance_m == pytest.approx(
        2.0 * plan.outbound_geodesic_distance_m, abs=1e-3
    )


def test_rotation_station_has_one_position_and_uniform_yaws() -> None:
    from spatial_episode.contracts.trajectory_v1 import TrajectoryPlanV1

    payload = build_rotation_station_trajectory(
        scene_id="8b8cb560-e2eb-575f-a8ff-203661255e4c",
        recipe_id="test-t3",
        seed=17,
        station_position=(1.0, 2.0, 0.0),
        station_id="living_room_0:station0",
        yaw_step_deg=60.0,
        view_count=6,
        floor_area_m2=20.0,
        backend_version="test",
    )
    plan = TrajectoryPlanV1.model_validate(payload)

    assert plan.trajectory_class.value == "T3"
    assert plan.yaw_sequence_deg == (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
    assert {view.world_from_agent.translation_m for view in plan.views} == {
        (1.0, 2.0, 0.0)
    }
    assert {view.cumulative_distance_m for view in plan.views} == {0.0}


def test_among5_trajectory_has_four_certified_target_facing_views() -> None:
    from spatial_episode.contracts.trajectory_v1 import TrajectoryPlanV1

    payload = build_among5_trajectory(
        scene_id="8b8cb560-e2eb-575f-a8ff-203661255e4c",
        recipe_id="test-among5",
        seed=17,
        camera_positions=(
            (0.0, -2.8, 0.0),
            (2.8, 0.0, 0.0),
            (0.0, 2.8, 0.0),
            (-2.8, 0.0, 0.0),
        ),
        focus_position=(0.0, 0.0, 0.0),
        camera_pitch_down_deg=25.0,
        floor_area_m2=40.0,
        backend_version="test",
    )
    plan = TrajectoryPlanV1.model_validate(payload)

    assert plan.trajectory_class.value == "T2"
    assert len(plan.views) == 4
    assert plan.views[0].role.value == "initial"
    assert all(view.role.value in {"initial", "orbit"} for view in plan.views)
    assert all(
        abs(view.world_from_agent.rotation_xyzw[0])
        + abs(view.world_from_agent.rotation_xyzw[1])
        > 0.0
        for view in plan.views
    )


def test_typed_trajectory_builders_satisfy_shared_contract() -> None:
    from spatial_episode.contracts.trajectory_v1 import TrajectoryPlanV1

    common = {
        "scene_id": "8b8cb560-e2eb-575f-a8ff-203661255e4c",
        "seed": 17,
        "floor_area_m2": 20.0,
        "backend_version": "test",
    }
    orbit = build_object_orbit_trajectory(
        **common,
        recipe_id="test-t4",
        orbit_positions=[
            (math.cos(index * math.pi / 4), math.sin(index * math.pi / 4), 0.0)
            for index in range(8)
        ],
        focus_position=(0.0, 0.0, 0.5),
        focus_entity_id="chair_0",
        azimuth_deg_per_view=[index * 45.0 for index in range(8)],
    )
    elevation = build_elevation_trajectory(
        **common,
        recipe_id="test-t7",
        station_position=(1.0, 2.0, 0.0),
        station_id="living_room_0:station0",
        heights_m=(0.3, 1.5, 2.5),
        pitch_down_deg=(0.0, 0.0, 45.0),
    )
    occlusion = build_occlusion_reveal_trajectory(
        **common,
        recipe_id="test-t8",
        source_points=((0.0, 0.0, 0.0), (0.0, 3.0, 0.0)),
        view_count=6,
        target_position=(0.0, 4.0, 0.5),
        target_entity_id="chair_0",
        occluder_entity_id="sofa_0",
    )
    target = build_target_view_trajectory(
        **common,
        recipe_id="test-t10",
        camera_positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        anchor_pairs=(("a", "b"), ("c", "d"), ("e", "f")),
        facing_positions=((0.0, 2.0, 0.0), (1.0, 2.0, 0.0), (2.0, 2.0, 0.0)),
    )

    assert TrajectoryPlanV1.model_validate(orbit).trajectory_class.value == "T4"
    assert TrajectoryPlanV1.model_validate(elevation).trajectory_class.value == "T7"
    assert TrajectoryPlanV1.model_validate(occlusion).trajectory_class.value == "T8"
    assert TrajectoryPlanV1.model_validate(target).trajectory_class.value == "T10"


def test_yaw_pitch_quaternion_is_normalized() -> None:
    quaternion = yaw_pitch_quaternion(90.0, 45.0)

    assert math.sqrt(sum(value * value for value in quaternion)) == pytest.approx(1.0)


def test_semantic_mask_is_a_stable_projection_of_instances() -> None:
    instance = np.array([[0, 1, 2], [3, 2, 4]], dtype=np.uint32)
    semantic, registry = _semantic_from_instances(
        instance,
        {0: "background", 1: "unlabelled", 2: "chair_0", 3: "chair_1", 4: "mystery"},
        {"chair_0": "chair", "chair_1": "chair"},
    )

    chair_id = _semantic_id("chair")
    assert semantic.dtype == np.uint32
    assert semantic.tolist() == [[0, 1, chair_id], [chair_id, chair_id, 1]]
    assert registry == {0: "background", 1: "unlabelled", chair_id: "chair"}


def test_renderer_mesh_ids_are_merged_into_stable_object_ids() -> None:
    renderer = np.array([[0, 4, 5], [8, 9, 10]], dtype=np.uint32)
    instance, registry, audit = _remap_renderer_instances(
        renderer,
        {
            4: "/World/scene_0/chair_0/visuals/mesh_0",
            5: "/World/scene_0/chair_0/visuals/mesh_1",
            8: "/World/scene_0/table_0/visuals/mesh_0",
            9: "/World/debug_marker",
            10: "BACKGROUND",
        },
        {
            "/World/scene_0/chair_0": "chair_0",
            "/World/scene_0/table_0": "table_0",
        },
    )

    chair_id = _instance_id("chair_0")
    table_id = _instance_id("table_0")
    assert instance.tolist() == [[0, chair_id, chair_id], [table_id, 1, 0]]
    assert registry == {
        0: "background",
        1: "unlabelled",
        chair_id: "chair_0",
        table_id: "table_0",
    }
    assert audit == {
        "renderer_id_count": 6,
        "mapped_renderer_id_count": 3,
        "mapped_object_count": 2,
        "mapped_resolvable_pixel_fraction": 0.75,
        "background_pixel_fraction": 0.333333,
        "known_unresolvable_pixel_fraction": 0.0,
        "invalid_label_pixel_fraction": 0.0,
        "unlabelled_pixel_fraction": 0.166667,
        "largest_unmapped_renderer_ids": [
            {
                "renderer_id": 9,
                "prim_path": "/World/debug_marker",
                "pixel_count": 1,
                "reason": "no_scene_object_ancestor",
                "pixel_fraction": 0.166667,
            }
        ],
    }


def test_renderer_invalid_labels_are_not_claimed_as_resolvable_objects() -> None:
    instance, registry, audit = _remap_renderer_instances(
        np.array([[2, 3, 4]], dtype=np.uint32),
        {
            2: "/World/scene_0/chair_0/visuals/mesh_0",
            3: "INVALID",
            4: "UNLABELLED",
        },
        {"/World/scene_0/chair_0": "chair_0"},
    )

    chair_id = _instance_id("chair_0")
    assert instance.tolist() == [[chair_id, 1, 1]]
    assert registry[chair_id] == "chair_0"
    assert audit["mapped_resolvable_pixel_fraction"] == 1.0
    assert audit["known_unresolvable_pixel_fraction"] == pytest.approx(0.666667)
    assert audit["invalid_label_pixel_fraction"] == pytest.approx(0.333333)
    assert audit["largest_unmapped_renderer_ids"][0]["reason"] == (
        "renderer_label_invalid"
    )


def test_renderer_mapping_uses_longest_object_ancestor() -> None:
    instance, registry, _ = _remap_renderer_instances(
        np.array([[7]], dtype=np.uint32),
        {7: "/World/scene_0/cabinet_0/door_0/visuals/mesh_0"},
        {
            "/World/scene_0/cabinet_0": "cabinet_0",
            "/World/scene_0/cabinet_0/door_0": "door_0",
        },
    )
    assert instance.item() == _instance_id("door_0")
    assert registry[_instance_id("door_0")] == "door_0"


def test_rgb_loop_diagnostics_separates_exposure_bias_from_structure() -> None:
    first = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    last = np.clip(first.astype(np.int16) + np.array([5, 7, 9]), 0, 255).astype(
        np.uint8
    )

    diagnostics = _rgb_loop_diagnostics(first, last)

    assert diagnostics["rgb_exact"] is False
    assert diagnostics["last_minus_first_channel_mean"] == [5.0, 7.0, 9.0]
    assert diagnostics["bias_corrected_rgb_psnr_db"] is None
    assert diagnostics["rgb_structure_correlation"] == pytest.approx(1.0)
    assert diagnostics["exposure_shift_likely"] is True
