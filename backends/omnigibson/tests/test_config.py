import sys
import types
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from omnigibson_episode.acquire import (
    _environment_config,
    _sample_path,
    _sample_rotation_station,
)
from omnigibson_episode.config import load_recipe

ROOT = Path(__file__).resolve().parents[1]


def test_reference_recipe_is_valid() -> None:
    recipe = load_recipe(ROOT / "configs" / "omnigibson_static_m1.yaml")
    assert recipe.source.scene_model == "Rs_int"


def test_external_camera_uses_upstream_sensor_contract() -> None:
    recipe = load_recipe(Path("configs/omnigibson_static_m1.yaml"))
    environment = _environment_config(recipe)
    sensor = environment["env"]["external_sensors"][0]

    assert environment["env"]["action_frequency"] == 60
    assert environment["env"]["rendering_frequency"] == 60
    assert sensor["sensor_type"] == "VisionSensor"
    assert "type" not in sensor
    assert sensor["sensor_kwargs"]["image_width"] == 1024
    assert sensor["sensor_kwargs"]["clipping_range"] == [0.05, 30.0]
    assert sensor["modalities"] == ["rgb", "depth_linear"]
    assert recipe.trajectory.outbound_view_count == 6
    assert recipe.trajectory.sampling_strategy == "traversable_random"
    assert recipe.trajectory.fallback_minimum_distance_m == 4.0
    assert recipe.sensor.width_px == 1024
    assert recipe.sensor.focal_length_mm > 0.0


def test_invalid_distance_range_is_rejected(tmp_path: Path) -> None:
    source = (ROOT / "configs" / "omnigibson_static_m1.yaml").read_text(encoding="utf-8")
    path = tmp_path / "invalid.yaml"
    path.write_text(
        source.replace("minimum_distance_m: 4.0", "minimum_distance_m: 15.0"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="distance range"):
        load_recipe(path)


def test_room_aware_recipe_is_versioned() -> None:
    recipe = load_recipe(ROOT / "configs" / "omnigibson_static_m2_room_aware.yaml")
    assert recipe.recipe_id == "omnigibson_static_m2_room_aware"
    assert recipe.trajectory.sampling_strategy == "room_aware"
    assert recipe.trajectory.fallback_minimum_distance_m == 1.5


def test_rotation_station_recipe_encodes_zero_parallax_scan() -> None:
    recipe = load_recipe(ROOT / "configs" / "omnigibson_t3_rotation_station_pilot.yaml")

    assert recipe.trajectory.trajectory_class == "T3"
    assert recipe.trajectory.sampling_strategy == "rotation_station"
    assert recipe.trajectory.outbound_view_count == 6
    assert recipe.trajectory.yaw_step_deg == 60.0
    assert recipe.trajectory.minimum_wall_clearance_m == 1.0
    assert recipe.trajectory.rotation_station_rank == 0
    assert recipe.sensor.minimum_instance_pixels == 128


def test_second_rotation_station_is_explicit_and_geometrically_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = types.SimpleNamespace(
        float32=np.float32,
        tensor=lambda value, dtype: np.asarray(value, dtype=dtype),
    )
    def distance_transform(mask: np.ndarray, *_: object) -> np.ndarray:
        result = np.zeros(mask.shape, dtype=np.float32)
        zeros = np.argwhere(mask == 0)
        for pixel in np.argwhere(mask != 0):
            result[tuple(pixel)] = np.linalg.norm(zeros - pixel, axis=1).min()
        return result

    fake_cv2 = types.SimpleNamespace(
        DIST_L2=2,
        distanceTransform=distance_transform,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    room_map = np.zeros((24, 24), dtype=np.int64)
    room_map[1:-1, 1:-1] = 1

    class Segmentation:
        room_ins_name_to_ins_id: ClassVar[dict[str, int]] = {"living_room_0": 1}
        room_ins_map: ClassVar[np.ndarray] = room_map
        map_resolution = 0.2
        floor_heights: ClassVar[list[float]] = [0.0]

        @staticmethod
        def map_to_world(pixel: np.ndarray) -> np.ndarray:
            return pixel.astype(np.float32) * 0.2

    class Traversability:
        floor_map: ClassVar[list[np.ndarray]] = [
            np.full((24, 24), 255, dtype=np.uint8)
        ]

    class Scene:
        seg_map = Segmentation()
        trav_map = Traversability()

    first_recipe = load_recipe(
        ROOT / "configs" / "omnigibson_t3_rotation_station_pilot.yaml"
    )
    second_recipe = load_recipe(
        ROOT / "configs" / "omnigibson_t3_rotation_station_rank1.yaml"
    )
    first, first_selection = _sample_rotation_station(Scene(), first_recipe)
    second, second_selection = _sample_rotation_station(Scene(), second_recipe)

    assert first_selection["requested_station_rank"] == 0
    assert second_selection["requested_station_rank"] == 1
    assert first_selection["station_id"] != second_selection["station_id"]
    assert np.linalg.norm(np.asarray(first[:2]) - np.asarray(second[:2])) >= 1.0


def test_adaptive_t4_recipe_versions_the_relaxed_arc_contract() -> None:
    recipe = load_recipe(
        ROOT / "configs" / "omnigibson_t4_object_orbit_adaptive_arc_v3.yaml"
    )

    assert recipe.recipe_id == "omnigibson_t4_object_orbit_adaptive_arc_v3"
    assert recipe.trajectory.orbit_minimum_radius_m == 0.7
    assert recipe.trajectory.orbit_minimum_arc_deg == 270.0
    assert recipe.trajectory.orbit_object_clearance_m == 0.35
    assert recipe.trajectory.orbit_adaptive_radius is True


def test_procedural_among5_recipe_has_exact_assets_and_views() -> None:
    recipe = load_recipe(ROOT / "configs" / "omnigibson_t2_among5_pilot.yaml")

    assert recipe.trajectory.trajectory_class == "T2"
    assert recipe.trajectory.sampling_strategy == "procedural_among5"
    assert recipe.trajectory.outbound_view_count == 4
    assert recipe.source.load_mode == "structure_only"
    assert recipe.among5 is not None
    assert len(recipe.among5.assets) == 5
    assert sum(asset.role == "anchor" for asset in recipe.among5.assets) == 1
    assert len(set(recipe.among5.satellite_order)) == 4
    environment = _environment_config(recipe)
    assert len(environment["objects"]) == 5
    assert all(item["fixed_base"] and item["kinematic_only"] for item in environment["objects"])


def test_mindcube_among_recipe_enforces_partial_visibility_geometry() -> None:
    recipe = load_recipe(ROOT / "configs" / "omnigibson_t2_mindcube_among_v2.yaml")

    assert recipe.among5 is not None
    assert recipe.among5.protocol_version == "omnigibson_mindcube_among_layout.v2"
    assert recipe.among5.visibility_contract == "central_plus_one_partial"
    assert recipe.among5.camera_radius_m < recipe.among5.layout_radius_m
    assert recipe.sensor.horizontal_fov_deg == 65.0


@pytest.mark.parametrize(
    ("filename", "trajectory_class", "strategy", "view_count"),
    [
        ("omnigibson_t4_object_orbit.yaml", "T4", "object_orbit", 8),
        ("omnigibson_t7_elevation.yaml", "T7", "elevation_station", 3),
        ("omnigibson_t8_occlusion_reveal.yaml", "T8", "occlusion_reveal", 6),
        ("omnigibson_t10_target_view.yaml", "T10", "target_view", 3),
    ],
)
def test_typed_trajectory_recipes_are_valid(
    filename: str, trajectory_class: str, strategy: str, view_count: int
) -> None:
    recipe = load_recipe(ROOT / "configs" / filename)

    assert recipe.trajectory.trajectory_class == trajectory_class
    assert recipe.trajectory.sampling_strategy == strategy
    assert recipe.trajectory.outbound_view_count == view_count


def test_room_aware_sampler_stays_inside_semantic_rooms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())

    class Segmentation:
        def __init__(self) -> None:
            self.room_ins_name_to_ins_id = {"kitchen_0": 1, "living_room_0": 2}

        @staticmethod
        def get_random_point_by_room_instance(room: str) -> tuple[int, np.ndarray]:
            y = 0.0 if room == "kitchen_0" else 6.0
            return 0, np.array([0.0, y, 0.0])

        @staticmethod
        def get_room_instance_by_point(xy: np.ndarray) -> str:
            return "kitchen_0" if float(xy[1]) < 3.0 else "living_room_0"

    class Scene:
        seg_map = Segmentation()

        @staticmethod
        def get_shortest_path(**_: object) -> tuple[np.ndarray, float]:
            return (
                np.array([[0.0, 0.0], [0.0, 2.0], [0.0, 4.0], [0.0, 6.0]]),
                6.0,
            )

    recipe = load_recipe(ROOT / "configs" / "omnigibson_static_m2_room_aware.yaml")
    points, distance, selection = _sample_path(Scene(), recipe)
    assert len(points) == 4
    assert distance == 6.0
    assert selection["sampling_strategy"] == "room_aware"
    assert selection["path_indoor_fraction"] == 1.0
    assert selection["distance_range_relaxed"] is False


def test_traversable_indoor_sampler_requires_semantic_room_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    source = (ROOT / "configs" / "omnigibson_static_m2_room_aware.yaml").read_text(
        encoding="utf-8"
    )
    recipe_path = tmp_path / "traversable-indoor.yaml"
    recipe_path.write_text(
        source.replace("sampling_strategy: room_aware", "sampling_strategy: traversable_indoor")
        .replace("candidate_count: 64", "candidate_count: 2"),
        encoding="utf-8",
    )

    class Segmentation:
        room_ins_name_to_ins_id: ClassVar[dict[str, int]] = {"hall_0": 1}

        @staticmethod
        def get_room_instance_by_point(xy: np.ndarray) -> str | None:
            return "hall_0" if float(xy[0]) >= 0.0 else None

    class Scene:
        seg_map = Segmentation()
        calls = 0

        @classmethod
        def get_random_point(cls, **_: object) -> tuple[int, np.ndarray]:
            cls.calls += 1
            x = -1.0 if cls.calls <= 2 else float(cls.calls - 3) * 5.0
            return 0, np.array([x, 0.0, 0.0])

        @staticmethod
        def get_shortest_path(
            source_world: np.ndarray, target_world: np.ndarray, **_: object
        ) -> tuple[np.ndarray, float]:
            return np.array([source_world, target_world]), 5.0

    _, distance, selection = _sample_path(Scene(), load_recipe(recipe_path))

    assert distance == 5.0
    assert selection["sampling_strategy"] == "traversable_indoor"
    assert selection["requested_room_instances"] == ["hall_0", "hall_0"]
    assert selection["rejection_counts"] == {"endpoint_outside_semantic_room": 1}
