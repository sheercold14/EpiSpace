"""Strict recipe parsing without importing the heavy simulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _mapping(payload: Any, name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a mapping")
    return payload


def _positive_number(value: Any, name: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_integer(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class SourceConfig:
    scene_model: str
    scene_instance: str | None
    source_version: str
    dataset_name: str
    load_mode: str
    floor: int


@dataclass(frozen=True, slots=True)
class TrajectoryConfig:
    trajectory_class: str
    sampling_strategy: str
    outbound_view_count: int
    candidate_count: int
    minimum_distance_m: float
    fallback_minimum_distance_m: float
    maximum_distance_m: float
    camera_height_m: float
    yaw_step_deg: float
    minimum_wall_clearance_m: float
    rotation_station_rank: int
    orbit_minimum_radius_m: float
    orbit_maximum_radius_m: float
    orbit_minimum_arc_deg: float
    orbit_object_clearance_m: float
    orbit_adaptive_radius: bool
    focus_categories: tuple[str, ...]
    elevation_heights_m: tuple[float, ...]
    elevation_pitch_down_deg: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SensorConfig:
    width_px: int
    height_px: int
    horizontal_fov_deg: float
    near_m: float
    far_m: float
    minimum_instance_pixels: int
    render_warmup_frames: int
    high_quality_rendering: bool

    @property
    def horizontal_aperture_mm(self) -> float:
        return 20.955

    @property
    def focal_length_mm(self) -> float:
        import math

        radians = math.radians(self.horizontal_fov_deg)
        return self.horizontal_aperture_mm / (2.0 * math.tan(radians / 2.0))


@dataclass(frozen=True, slots=True)
class Among5AssetConfig:
    """One explicitly versioned asset in the controlled Among-5 layout."""

    name: str
    category: str
    model: str
    bbox_size_m: tuple[float, float, float]
    role: str


@dataclass(frozen=True, slots=True)
class Among5Config:
    """Simulator contract for an exact five-object relative-layout task."""

    protocol_version: str
    visibility_contract: str
    layout_id: str
    assets: tuple[Among5AssetConfig, ...]
    satellite_order: tuple[str, ...]
    layout_radius_m: float
    camera_radius_m: float
    layout_yaw_deg: float
    mirrored: bool
    camera_azimuth_deg: tuple[float, ...]
    camera_pitch_down_deg: float
    center_rank: int
    minimum_object_gap_m: float
    minimum_camera_wall_margin_m: float


@dataclass(frozen=True, slots=True)
class EpisodeRecipe:
    recipe_id: str
    seed: int
    source: SourceConfig
    trajectory: TrajectoryConfig
    sensor: SensorConfig
    among5: Among5Config | None = None


def load_recipe(path: Path) -> EpisodeRecipe:
    root = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "recipe")
    if root.get("schema_version") != "omnigibson_episode_recipe.v1":
        raise ValueError("unsupported recipe schema_version")
    recipe_id = str(root.get("recipe_id", "")).strip()
    if not recipe_id:
        raise ValueError("recipe_id cannot be empty")

    source_raw = _mapping(root.get("source"), "source")
    scene_model = str(source_raw.get("scene_model", "")).strip()
    source_version = str(source_raw.get("source_version", "")).strip()
    if not scene_model or not source_version:
        raise ValueError("source scene_model and source_version are required")
    load_mode = str(source_raw.get("load_mode", "full"))
    if load_mode not in {"full", "structure_only"}:
        raise ValueError("source.load_mode must be full or structure_only")
    source = SourceConfig(
        scene_model=scene_model,
        scene_instance=source_raw.get("scene_instance"),
        source_version=source_version,
        dataset_name=str(source_raw.get("dataset_name", "behavior-1k-assets")),
        load_mode=load_mode,
        floor=int(source_raw.get("floor", 0)),
    )
    if source.floor < 0:
        raise ValueError("source.floor cannot be negative")

    trajectory_raw = _mapping(root.get("trajectory"), "trajectory")
    trajectory_class = str(trajectory_raw.get("trajectory_class", "T1"))
    if trajectory_class not in {f"T{index}" for index in range(1, 11)}:
        raise ValueError("trajectory.trajectory_class must be T1 through T10")
    sampling_strategy = str(trajectory_raw.get("sampling_strategy", "traversable_random"))
    if sampling_strategy not in {
        "traversable_random",
        "traversable_indoor",
        "room_aware",
        "rotation_station",
        "object_orbit",
        "elevation_station",
        "occlusion_reveal",
        "target_view",
        "procedural_among5",
    }:
        raise ValueError("trajectory sampling_strategy is unsupported")
    trajectory = TrajectoryConfig(
        trajectory_class=trajectory_class,
        sampling_strategy=sampling_strategy,
        outbound_view_count=_positive_integer(
            trajectory_raw.get("outbound_view_count"), "outbound_view_count"
        ),
        candidate_count=_positive_integer(
            trajectory_raw.get("candidate_count"), "candidate_count"
        ),
        minimum_distance_m=_positive_number(
            trajectory_raw.get("minimum_distance_m"), "minimum_distance_m"
        ),
        fallback_minimum_distance_m=_positive_number(
            trajectory_raw.get(
                "fallback_minimum_distance_m", trajectory_raw.get("minimum_distance_m")
            ),
            "fallback_minimum_distance_m",
        ),
        maximum_distance_m=_positive_number(
            trajectory_raw.get("maximum_distance_m"), "maximum_distance_m"
        ),
        camera_height_m=_positive_number(
            trajectory_raw.get("camera_height_m"), "camera_height_m"
        ),
        yaw_step_deg=_positive_number(
            trajectory_raw.get("yaw_step_deg", 60.0), "yaw_step_deg"
        ),
        minimum_wall_clearance_m=_positive_number(
            trajectory_raw.get("minimum_wall_clearance_m", 1.0),
            "minimum_wall_clearance_m",
        ),
        rotation_station_rank=int(trajectory_raw.get("rotation_station_rank", 0)),
        orbit_minimum_radius_m=_positive_number(
            trajectory_raw.get("orbit_minimum_radius_m", 1.0),
            "orbit_minimum_radius_m",
        ),
        orbit_maximum_radius_m=_positive_number(
            trajectory_raw.get("orbit_maximum_radius_m", 3.0),
            "orbit_maximum_radius_m",
        ),
        orbit_minimum_arc_deg=_positive_number(
            trajectory_raw.get("orbit_minimum_arc_deg", 360.0),
            "orbit_minimum_arc_deg",
        ),
        orbit_object_clearance_m=_positive_number(
            trajectory_raw.get("orbit_object_clearance_m", 0.55),
            "orbit_object_clearance_m",
        ),
        orbit_adaptive_radius=_boolean(
            trajectory_raw.get("orbit_adaptive_radius", False),
            "orbit_adaptive_radius",
        ),
        focus_categories=tuple(
            str(value).strip()
            for value in trajectory_raw.get(
                "focus_categories",
                [
                    "armchair",
                    "chair",
                    "desk_chair",
                    "office_chair",
                    "refrigerator",
                    "sofa",
                    "television",
                ],
            )
            if str(value).strip()
        ),
        elevation_heights_m=tuple(
            float(value)
            for value in trajectory_raw.get(
                "elevation_heights_m", [0.3, 1.5, 2.5]
            )
        ),
        elevation_pitch_down_deg=tuple(
            float(value)
            for value in trajectory_raw.get(
                "elevation_pitch_down_deg", [0.0, 0.0, 45.0]
            )
        ),
    )
    if trajectory.outbound_view_count < 3:
        raise ValueError("outbound_view_count must be at least three")
    if trajectory.maximum_distance_m <= trajectory.minimum_distance_m:
        raise ValueError("trajectory distance range is invalid")
    if trajectory.fallback_minimum_distance_m > trajectory.minimum_distance_m:
        raise ValueError("fallback minimum distance cannot exceed the preferred minimum")
    if trajectory.sampling_strategy == "rotation_station":
        if trajectory.trajectory_class != "T3":
            raise ValueError("rotation_station sampling requires trajectory_class T3")
        if trajectory.yaw_step_deg not in {45.0, 60.0, 90.0}:
            raise ValueError("T3 yaw_step_deg must be one of 45, 60 or 90 degrees")
        expected_views = round(360.0 / trajectory.yaw_step_deg)
        if trajectory.outbound_view_count != expected_views:
            raise ValueError(
                "T3 outbound_view_count must equal a full 360-degree yaw scan"
            )
        if trajectory.rotation_station_rank < 0:
            raise ValueError("T3 rotation_station_rank cannot be negative")
    expected_class_by_strategy = {
        "procedural_among5": "T2",
        "object_orbit": "T4",
        "elevation_station": "T7",
        "occlusion_reveal": "T8",
        "target_view": "T10",
    }
    expected_class = expected_class_by_strategy.get(trajectory.sampling_strategy)
    if expected_class is not None and trajectory.trajectory_class != expected_class:
        raise ValueError(
            f"{trajectory.sampling_strategy} sampling requires trajectory_class "
            f"{expected_class}"
        )
    if trajectory.orbit_maximum_radius_m <= trajectory.orbit_minimum_radius_m:
        raise ValueError("orbit radius range is invalid")
    if trajectory.sampling_strategy == "object_orbit":
        if not 6 <= trajectory.outbound_view_count <= 12:
            raise ValueError("T4 requires 6 through 12 orbit views")
        if trajectory.outbound_view_count % 2:
            raise ValueError("T4 requires an even view count for opposite views")
        if not trajectory.focus_categories:
            raise ValueError("T4 focus_categories cannot be empty")
        if not 180.0 <= trajectory.orbit_minimum_arc_deg <= 360.0:
            raise ValueError("T4 orbit_minimum_arc_deg must be between 180 and 360")
    if trajectory.sampling_strategy == "elevation_station":
        if len(trajectory.elevation_heights_m) != trajectory.outbound_view_count:
            raise ValueError("T7 height count must equal outbound_view_count")
        if len(trajectory.elevation_pitch_down_deg) != trajectory.outbound_view_count:
            raise ValueError("T7 pitch count must equal outbound_view_count")
        if any(height <= 0.0 for height in trajectory.elevation_heights_m):
            raise ValueError("T7 heights must be positive")
        if any(not 0.0 <= pitch <= 60.0 for pitch in trajectory.elevation_pitch_down_deg):
            raise ValueError("T7 downward pitches must be between 0 and 60 degrees")
    if trajectory.sampling_strategy == "occlusion_reveal" and not (
        6 <= trajectory.outbound_view_count <= 12
    ):
        raise ValueError("T8 requires 6 through 12 views")
    if trajectory.sampling_strategy == "target_view" and not (
        3 <= trajectory.outbound_view_count <= 5
    ):
        raise ValueError("T10 requires 3 through 5 held-out views")

    among5_raw = root.get("among5")
    among5: Among5Config | None = None
    if trajectory.sampling_strategy == "procedural_among5":
        if source.load_mode != "structure_only":
            raise ValueError("procedural Among-5 requires source.load_mode structure_only")
        payload = _mapping(among5_raw, "among5")
        protocol_version = str(payload.get("protocol_version", ""))
        if protocol_version not in {
            "omnigibson_among5_layout.v1",
            "omnigibson_mindcube_among_layout.v2",
        }:
            raise ValueError("among5.protocol_version is unsupported")
        visibility_contract = str(
            payload.get("visibility_contract", "all_five_each_view")
        )
        expected_visibility = {
            "omnigibson_among5_layout.v1": "all_five_each_view",
            "omnigibson_mindcube_among_layout.v2": "central_plus_one_partial",
        }[protocol_version]
        if visibility_contract != expected_visibility:
            raise ValueError(
                "among5 visibility_contract disagrees with protocol_version"
            )
        layout_id = str(payload.get("layout_id", "")).strip()
        if not layout_id:
            raise ValueError("among5.layout_id cannot be empty")
        assets_raw = payload.get("assets")
        if not isinstance(assets_raw, list) or len(assets_raw) != 5:
            raise ValueError("among5.assets must contain exactly five assets")
        assets: list[Among5AssetConfig] = []
        for index, raw_asset in enumerate(assets_raw):
            asset = _mapping(raw_asset, f"among5.assets[{index}]")
            name = str(asset.get("name", "")).strip()
            category = str(asset.get("category", "")).strip()
            model = str(asset.get("model", "")).strip()
            role = str(asset.get("role", "")).strip()
            bbox_raw = asset.get("bbox_size_m")
            if not name or not category or not model:
                raise ValueError("among5 asset name, category and model are required")
            if role not in {"anchor", "satellite"}:
                raise ValueError("among5 asset role must be anchor or satellite")
            if not isinstance(bbox_raw, list) or len(bbox_raw) != 3:
                raise ValueError("among5 asset bbox_size_m must be a three-vector")
            bbox = tuple(_positive_number(value, "among5 asset bbox size") for value in bbox_raw)
            assets.append(
                Among5AssetConfig(
                    name=name,
                    category=category,
                    model=model,
                    bbox_size_m=bbox,  # type: ignore[arg-type]
                    role=role,
                )
            )
        names = [asset.name for asset in assets]
        categories = [asset.category for asset in assets]
        if len(set(names)) != 5 or len(set(categories)) != 5:
            raise ValueError("among5 asset names and categories must be unique")
        anchors = [asset for asset in assets if asset.role == "anchor"]
        satellites = [asset for asset in assets if asset.role == "satellite"]
        if len(anchors) != 1 or len(satellites) != 4:
            raise ValueError("among5 requires one anchor and four satellites")
        satellite_order = tuple(str(value) for value in payload.get("satellite_order", []))
        if len(satellite_order) != 4 or set(satellite_order) != {
            asset.name for asset in satellites
        }:
            raise ValueError("among5.satellite_order must permute the four satellite names")
        camera_azimuth = tuple(
            float(value) % 360.0 for value in payload.get("camera_azimuth_deg", [])
        )
        if len(camera_azimuth) != 4 or len({round(value, 6) for value in camera_azimuth}) != 4:
            raise ValueError("among5.camera_azimuth_deg must contain four unique views")
        if trajectory.outbound_view_count != 4:
            raise ValueError("procedural Among-5 requires exactly four views")
        camera_pitch = float(payload.get("camera_pitch_down_deg", 25.0))
        if not 0.0 <= camera_pitch <= 60.0:
            raise ValueError("among5.camera_pitch_down_deg must be between 0 and 60")
        center_rank = int(payload.get("center_rank", 0))
        if center_rank < 0:
            raise ValueError("among5.center_rank cannot be negative")
        among5 = Among5Config(
            protocol_version=protocol_version,
            visibility_contract=visibility_contract,
            layout_id=layout_id,
            assets=tuple(assets),
            satellite_order=satellite_order,
            layout_radius_m=_positive_number(
                payload.get("layout_radius_m", 1.1), "among5.layout_radius_m"
            ),
            camera_radius_m=_positive_number(
                payload.get("camera_radius_m", 2.8), "among5.camera_radius_m"
            ),
            layout_yaw_deg=float(payload.get("layout_yaw_deg", 0.0)) % 360.0,
            mirrored=_boolean(payload.get("mirrored", False), "among5.mirrored"),
            camera_azimuth_deg=camera_azimuth,
            camera_pitch_down_deg=camera_pitch,
            center_rank=center_rank,
            minimum_object_gap_m=_positive_number(
                payload.get("minimum_object_gap_m", 0.1), "among5.minimum_object_gap_m"
            ),
            minimum_camera_wall_margin_m=_positive_number(
                payload.get("minimum_camera_wall_margin_m", 0.25),
                "among5.minimum_camera_wall_margin_m",
            ),
        )
        if visibility_contract == "all_five_each_view":
            if among5.camera_radius_m <= among5.layout_radius_m:
                raise ValueError("full-layout camera radius must exceed layout radius")
        elif among5.camera_radius_m >= among5.layout_radius_m:
            raise ValueError("MindCube-Among camera must lie inside the satellite ring")
    elif among5_raw is not None:
        raise ValueError("among5 is only valid with procedural_among5 sampling")

    sensor_raw = _mapping(root.get("sensor"), "sensor")
    sensor = SensorConfig(
        width_px=_positive_integer(sensor_raw.get("width_px"), "width_px"),
        height_px=_positive_integer(sensor_raw.get("height_px"), "height_px"),
        horizontal_fov_deg=_positive_number(
            sensor_raw.get("horizontal_fov_deg"), "horizontal_fov_deg"
        ),
        near_m=_positive_number(sensor_raw.get("near_m"), "near_m"),
        far_m=_positive_number(sensor_raw.get("far_m"), "far_m"),
        minimum_instance_pixels=_positive_integer(
            sensor_raw.get("minimum_instance_pixels"), "minimum_instance_pixels"
        ),
        render_warmup_frames=_positive_integer(
            sensor_raw.get("render_warmup_frames", 4), "render_warmup_frames"
        ),
        high_quality_rendering=bool(sensor_raw.get("high_quality_rendering", False)),
    )
    if not 0.0 < sensor.horizontal_fov_deg < 180.0:
        raise ValueError("horizontal_fov_deg must be between 0 and 180")
    if sensor.far_m <= sensor.near_m:
        raise ValueError("sensor far_m must be greater than near_m")
    if (
        among5 is not None
        and among5.visibility_contract == "central_plus_one_partial"
        and sensor.horizontal_fov_deg > 65.0
    ):
        raise ValueError("MindCube-Among horizontal FOV must be at most 65 degrees")
    return EpisodeRecipe(
        recipe_id=recipe_id,
        seed=int(root.get("seed", 0)),
        source=source,
        trajectory=trajectory,
        sensor=sensor,
        among5=among5,
    )
