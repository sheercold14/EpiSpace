"""Heavy OmniGibson worker: acquire evidence without importing Forge contracts."""

from __future__ import annotations

import math
import os
import random
import shutil
import zlib
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

from omnigibson_episode.config import EpisodeRecipe
from omnigibson_episode.geometry import (
    build_among5_trajectory,
    build_closed_trajectory,
    build_elevation_trajectory,
    build_object_orbit_trajectory,
    build_occlusion_reveal_trajectory,
    build_rotation_station_trajectory,
    build_target_view_trajectory,
    omnigibson_camera_quaternion,
    stable_scene_id,
)
from omnigibson_episode.io import (
    sha256_file,
    sha256_json,
    write_json_atomic,
    write_npz_atomic,
)


def _as_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return value


def _as_float_list(value: Any) -> list[float]:
    array = _as_numpy(value)
    return [float(item) for item in array]


def _rgb_u8(value: Any) -> Any:
    import numpy as np

    array = np.asarray(_as_numpy(value))
    if array.ndim != 3 or array.shape[-1] not in {3, 4}:
        raise RuntimeError(f"unexpected RGB shape: {array.shape}")
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        maximum = float(np.nanmax(array)) if array.size else 0.0
        if maximum <= 1.0 + 1e-6:
            array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _mask_u32(value: Any, name: str, shape: tuple[int, int]) -> Any:
    import numpy as np

    array = np.asarray(_as_numpy(value))
    if array.shape != shape:
        raise RuntimeError(f"unexpected {name} shape: {array.shape}, expected {shape}")
    if np.any(array < 0):
        raise RuntimeError(f"{name} contains negative identifiers")
    return array.astype(np.uint32)


def _depth_f32(value: Any, shape: tuple[int, int]) -> Any:
    import numpy as np

    array = np.asarray(_as_numpy(value), dtype=np.float32)
    if array.shape != shape:
        raise RuntimeError(f"unexpected depth shape: {array.shape}, expected {shape}")
    return array


def _semantic_id(category: str) -> int:
    """Return a stable non-reserved uint32 identifier for a category label."""

    return 2 + zlib.crc32(category.encode("utf-8")) % (2**32 - 2)


def _instance_id(name: str) -> int:
    """Return a stable non-reserved uint32 identifier for a scene object."""

    return 2 + zlib.crc32(f"object:{name}".encode()) % (2**32 - 2)


def _remap_renderer_instances(
    renderer_instance: Any,
    id_to_prim_path: dict[Any, Any],
    object_name_by_prim_path: dict[str, str],
) -> tuple[Any, dict[int, str], dict[str, Any]]:
    """Merge renderer mesh IDs into stable OmniGibson object IDs.

    ``instance_id_segmentation_fast`` avoids Isaac Sim's semantic mapping graph,
    while still exposing the exact USD prim path behind every renderer ID.  An
    OmniGibson object may contain many visual mesh prims, so the longest scene
    object ancestor defines the canonical object identity.
    """

    import numpy as np

    raw = np.asarray(_as_numpy(renderer_instance))
    if raw.ndim != 2:
        raise RuntimeError(f"unexpected renderer instance shape: {raw.shape}")
    if np.any(raw < 0):
        raise RuntimeError("renderer instance mask contains negative identifiers")
    raw = raw.astype(np.uint32)
    labels = {int(identifier): str(path) for identifier, path in id_to_prim_path.items()}
    object_paths = sorted(
        (
            (str(path).rstrip("/"), str(name))
            for path, name in object_name_by_prim_path.items()
            if str(path).startswith("/") and str(name)
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    if not object_paths:
        raise RuntimeError("scene exposes no object prim paths for instance remapping")

    instance = np.ones(raw.shape, dtype=np.uint32)
    registry = {0: "background", 1: "unlabelled"}
    mapped_renderer_ids = 0
    mapped_pixels = 0
    background_pixels = 0
    reserved_unlabelled_pixels = 0
    invalid_label_pixels = 0
    unmapped: list[dict[str, Any]] = []
    for raw_identifier, count in zip(*np.unique(raw, return_counts=True), strict=True):
        identifier = int(raw_identifier)
        pixel_count = int(count)
        prim_path = labels.get(identifier)
        if prim_path is None and identifier == 0:
            instance[raw == identifier] = 0
            background_pixels += pixel_count
            continue
        # Replicator reserves BACKGROUND / UNLABELLED and may emit INVALID for
        # renderer-only emissive surfaces whose prim path is unavailable. Keep
        # those pixels explicitly unknown instead of inventing an object ID;
        # their full-frame share is audited and capped separately below.
        normalized_label = prim_path.casefold() if prim_path is not None else None
        if prim_path is None or normalized_label in {
            "background",
            "unlabelled",
            "invalid",
        }:
            if prim_path is not None and prim_path.casefold() == "background":
                instance[raw == identifier] = 0
                background_pixels += pixel_count
            elif normalized_label == "unlabelled":
                reserved_unlabelled_pixels += pixel_count
            elif normalized_label == "invalid":
                invalid_label_pixels += pixel_count
                unmapped.append(
                    {
                        "renderer_id": identifier,
                        "prim_path": prim_path,
                        "pixel_count": pixel_count,
                        "reason": "renderer_label_invalid",
                    }
                )
            else:
                unmapped.append(
                    {
                        "renderer_id": identifier,
                        "prim_path": prim_path,
                        "pixel_count": pixel_count,
                        "reason": "missing_renderer_label",
                    }
                )
            continue
        name = next(
            (
                candidate
                for object_path, candidate in object_paths
                if prim_path == object_path or prim_path.startswith(object_path + "/")
            ),
            None,
        )
        if name is None:
            unmapped.append(
                {
                    "renderer_id": identifier,
                    "prim_path": prim_path,
                    "pixel_count": pixel_count,
                    "reason": "no_scene_object_ancestor",
                }
            )
            continue
        stable_identifier = _instance_id(name)
        previous = registry.get(stable_identifier)
        if previous is not None and previous != name:
            raise RuntimeError(f"instance identifier hash collision: {previous!r} and {name!r}")
        registry[stable_identifier] = name
        instance[raw == identifier] = stable_identifier
        mapped_renderer_ids += 1
        mapped_pixels += pixel_count

    known_unresolvable_pixels = reserved_unlabelled_pixels + invalid_label_pixels
    resolvable_pixels = raw.size - background_pixels - known_unresolvable_pixels
    mapped_fraction = mapped_pixels / resolvable_pixels if resolvable_pixels else 1.0
    audit = {
        "renderer_id_count": len(np.unique(raw)),
        "mapped_renderer_id_count": mapped_renderer_ids,
        "mapped_object_count": len(registry) - 2,
        "mapped_resolvable_pixel_fraction": round(float(mapped_fraction), 6),
        "background_pixel_fraction": round(float(background_pixels / raw.size), 6),
        "known_unresolvable_pixel_fraction": round(float(known_unresolvable_pixels / raw.size), 6),
        "invalid_label_pixel_fraction": round(float(invalid_label_pixels / raw.size), 6),
        "unlabelled_pixel_fraction": round(float(np.count_nonzero(instance == 1) / raw.size), 6),
        "largest_unmapped_renderer_ids": [
            {
                **item,
                "pixel_fraction": round(float(item["pixel_count"] / raw.size), 6),
            }
            for item in sorted(
                unmapped,
                key=lambda value: (-int(value["pixel_count"]), int(value["renderer_id"])),
            )[:8]
        ],
    }
    return instance, registry, audit


def _semantic_from_instances(
    instance: Any,
    instance_registry: dict[int, str],
    category_by_name: dict[str, str],
) -> tuple[Any, dict[int, str]]:
    """Project exact instance masks into category masks without a second RTX annotator."""

    import numpy as np

    semantic = np.ones(instance.shape, dtype=np.uint32)
    semantic[instance == 0] = 0
    registry = {0: "background", 1: "unlabelled"}
    for raw_identifier in np.unique(instance):
        identifier = int(raw_identifier)
        if identifier <= 1:
            continue
        name = instance_registry.get(identifier)
        category = category_by_name.get(name) if name is not None else None
        if category is None:
            continue
        category_identifier = _semantic_id(category)
        previous = registry.get(category_identifier)
        if previous is not None and previous != category:
            raise RuntimeError(f"semantic category hash collision: {previous!r} and {category!r}")
        registry[category_identifier] = category
        semantic[instance == identifier] = category_identifier
    return semantic, registry


def _scene_source_id(recipe: EpisodeRecipe) -> str:
    instance = recipe.source.scene_instance or "best"
    return f"{recipe.source.scene_model}:{instance}"


def _environment_config(recipe: EpisodeRecipe) -> dict[str, Any]:
    source = recipe.source
    sensor = recipe.sensor
    scene: dict[str, Any] = {
        "type": "InteractiveTraversableScene",
        "scene_model": source.scene_model,
        "scene_instance": source.scene_instance,
        "dataset_name": source.dataset_name,
        "trav_map_resolution": 0.1,
        "default_erosion_radius": 0.0,
        "trav_map_with_objects": True,
        "num_waypoints": recipe.trajectory.outbound_view_count,
        "waypoint_resolution": 0.2,
        "include_robots": False,
    }
    if source.load_mode == "structure_only":
        # Resolved by the worker after importing OmniGibson constants.
        scene["load_object_categories"] = "__STRUCTURE_CATEGORIES__"
    objects: list[dict[str, Any]] = []
    if recipe.among5 is not None:
        for index, asset in enumerate(recipe.among5.assets):
            objects.append(
                {
                    "type": "DatasetObject",
                    "name": asset.name,
                    "category": asset.category,
                    "model": asset.model,
                    "dataset_name": source.dataset_name,
                    "fixed_base": True,
                    "kinematic_only": True,
                    # Objects are parked outside the scene until a certified
                    # room centre has been selected after environment load.
                    "position": [150.0 + 2.0 * index, 150.0, 150.0],
                }
            )
    return {
        "env": {
            # OmniGibson's HQ renderer enables isosurface support globally and
            # rejects lower render rates, even for static camera-only scenes.
            # Headless mode additionally requires action and render dt equality.
            "action_frequency": 60,
            "rendering_frequency": 60,
            "automatic_reset": False,
            "flatten_action_space": False,
            "flatten_obs_space": False,
            "external_sensors": [
                {
                    "sensor_type": "VisionSensor",
                    "name": "episode_camera",
                    "relative_prim_path": "/episode_camera",
                    "modalities": [
                        "rgb",
                        "depth_linear",
                    ],
                    "sensor_kwargs": {
                        "image_height": sensor.height_px,
                        "image_width": sensor.width_px,
                        "focal_length": sensor.focal_length_mm,
                        "horizontal_aperture": sensor.horizontal_aperture_mm,
                        "clipping_range": [sensor.near_m, sensor.far_m],
                    },
                    "position": [0.0, 0.0, recipe.trajectory.camera_height_m],
                    "orientation": [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)],
                    "pose_frame": "world",
                    "include_in_obs": True,
                }
            ],
        },
        "render": {
            "viewer_width": sensor.width_px,
            "viewer_height": sensor.height_px,
        },
        "scene": scene,
        "robots": [],
        "objects": objects,
        "task": {"type": "DummyTask"},
    }


def _floor_area(scene: Any, floor: int) -> float:
    import numpy as np

    trav_map = scene.trav_map
    image = np.asarray(_as_numpy(trav_map.floor_map[floor]))
    resolution = float(getattr(trav_map, "map_resolution", 0.1))
    area = float(np.count_nonzero(image == 255)) * resolution * resolution
    if area <= 0.0:
        raise RuntimeError(f"floor {floor} has no traversable area")
    return area


def _sample_path(
    scene: Any, recipe: EpisodeRecipe
) -> tuple[list[list[float]], float, dict[str, Any]]:
    import torch as th

    floor = recipe.source.floor
    strategy = recipe.trajectory.sampling_strategy
    segmentation = getattr(scene, "seg_map", None)
    room_instances = (
        sorted(segmentation.room_ins_name_to_ins_id)
        if segmentation is not None
        and isinstance(getattr(segmentation, "room_ins_name_to_ins_id", None), dict)
        else []
    )
    room_constrained = strategy in {"room_aware", "traversable_indoor"}
    if room_constrained and (floor != 0 or not room_instances):
        raise RuntimeError("room-constrained sampling requires floor-0 room-instance segmentation")
    best: tuple[Any, float, float, tuple[float, ...], dict[str, Any]] | None = None
    valid_candidate_count = 0
    rejection_counts: Counter[str] = Counter()
    print(
        "trajectory_sampling "
        f"strategy={strategy} candidates={recipe.trajectory.candidate_count} "
        f"room_instances={len(room_instances)}",
        flush=True,
    )

    def sample_room_point(room_instance: str) -> Any:
        # OmniGibson 3.9's helper calls ``torch.tensor(torch.where(...))``,
        # which is invalid on the installed PyTorch. Sample the same room mask
        # directly while preserving the upstream map-to-world convention.
        if all(
            hasattr(segmentation, name)
            for name in ("room_ins_map", "map_to_world", "floor_heights")
        ):
            room_id = segmentation.room_ins_name_to_ins_id[room_instance]
            valid = th.nonzero(segmentation.room_ins_map == room_id, as_tuple=False)
            if len(valid) == 0:
                raise RuntimeError(f"room instance has no pixels: {room_instance}")
            xy = segmentation.map_to_world(valid[random.randrange(len(valid))])
            height = float(segmentation.floor_heights[floor])
            return th.cat((xy, xy.new_tensor([height])))
        _, point = segmentation.get_random_point_by_room_instance(room_instance)
        return point

    for candidate_index in range(recipe.trajectory.candidate_count):
        if candidate_index and candidate_index % 8 == 0:
            print(
                "trajectory_sampling "
                f"progress={candidate_index}/{recipe.trajectory.candidate_count} "
                f"valid={valid_candidate_count} rejected={dict(rejection_counts)}",
                flush=True,
            )
        requested_rooms: list[str] = []
        if strategy == "room_aware":
            if len(room_instances) >= 2 and candidate_index % 4 != 0:
                requested_rooms = random.sample(room_instances, 2)
            else:
                room = random.choice(room_instances)
                requested_rooms = [room, room]
            start = sample_room_point(requested_rooms[0])
            goal = sample_room_point(requested_rooms[1])
        else:
            _, start = scene.get_random_point(floor=floor)
            _, goal = scene.get_random_point(floor=floor, reference_point=start)
            if strategy == "traversable_indoor":
                requested_rooms = [
                    segmentation.get_room_instance_by_point(point[:2]) for point in (start, goal)
                ]
                if any(room is None for room in requested_rooms):
                    rejection_counts["endpoint_outside_semantic_room"] += 1
                    continue
                requested_rooms = [str(room) for room in requested_rooms]
        path, geodesic = scene.get_shortest_path(
            floor=floor,
            source_world=start[:2],
            target_world=goal[:2],
            entire_path=True,
        )
        if path is None or geodesic is None:
            rejection_counts["no_path"] += 1
            continue
        distance_m = float(geodesic.item() if hasattr(geodesic, "item") else geodesic)
        if not (
            recipe.trajectory.fallback_minimum_distance_m
            <= distance_m
            <= recipe.trajectory.maximum_distance_m
        ):
            rejection_counts["distance_out_of_range"] += 1
            continue
        path_array = _as_numpy(path)
        if len(path_array) == 0:
            rejection_counts["empty_path"] += 1
            continue
        path_rooms: set[str] = set()
        path_room_sequence: list[str] = []
        indoor_samples = 0
        if segmentation is not None:
            for point in path_array:
                room = segmentation.get_room_instance_by_point(point[:2])
                if room is not None:
                    room_name = str(room)
                    path_rooms.add(room_name)
                    if not path_room_sequence or path_room_sequence[-1] != room_name:
                        path_room_sequence.append(room_name)
                    indoor_samples += 1
        indoor_fraction = indoor_samples / len(path_array)
        if room_constrained and indoor_fraction < 0.8:
            rejection_counts["indoor_fraction_below_0.8"] += 1
            continue
        valid_candidate_count += 1
        preferred_distance = distance_m >= recipe.trajectory.minimum_distance_m
        score = (float(preferred_distance), float(len(path_rooms)), indoor_fraction, distance_m)
        details = {
            "requested_room_instances": requested_rooms,
            "path_room_instances": sorted(path_rooms),
            "path_room_sequence": path_room_sequence,
            "path_indoor_fraction": round(indoor_fraction, 6),
        }
        if best is None or score > best[3]:
            best = (path, distance_m, float(start[2]), score, details)
    if best is None:
        raise RuntimeError("no traversable path satisfies the configured distance range")
    path, distance_m, floor_height, _, details = best
    points = []
    for point in _as_numpy(path):
        values = [float(value) for value in point]
        if len(values) == 2:
            values.append(floor_height)
        points.append(values)
    if len(points) < 2:
        raise RuntimeError("OmniGibson returned a degenerate shortest path")
    # Keep a reference to torch in this worker to make the dependency explicit.
    assert th is not None
    selection = {
        "schema_version": "omnigibson_trajectory_selection.v1",
        "sampling_strategy": strategy,
        "candidate_count": recipe.trajectory.candidate_count,
        "valid_candidate_count": valid_candidate_count,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "selected_geodesic_distance_m": round(distance_m, 6),
        "preferred_minimum_distance_m": recipe.trajectory.minimum_distance_m,
        "fallback_minimum_distance_m": recipe.trajectory.fallback_minimum_distance_m,
        "distance_range_relaxed": distance_m < recipe.trajectory.minimum_distance_m,
        **details,
    }
    print(
        "trajectory_sampling "
        f"complete={recipe.trajectory.candidate_count}/{recipe.trajectory.candidate_count} "
        f"valid={valid_candidate_count} rejected={dict(rejection_counts)} "
        f"selected_distance_m={distance_m:.3f}",
        flush=True,
    )
    return points, distance_m, selection


def _sample_rotation_station(
    scene: Any, recipe: EpisodeRecipe
) -> tuple[list[float], dict[str, Any]]:
    """Choose a ranked, geometrically distinct room-interior rotation station."""

    import cv2
    import numpy as np
    import torch as th

    floor = recipe.source.floor
    if floor != 0:
        raise RuntimeError("T3 rotation-station sampling currently requires floor 0")
    segmentation = getattr(scene, "seg_map", None)
    if segmentation is None or not isinstance(
        getattr(segmentation, "room_ins_name_to_ins_id", None), dict
    ):
        raise RuntimeError("T3 rotation-station sampling requires room-instance segmentation")
    room_map = np.asarray(_as_numpy(segmentation.room_ins_map))
    traversable = np.asarray(_as_numpy(scene.trav_map.floor_map[floor])) == 255
    if room_map.shape != traversable.shape:
        raise RuntimeError("room and traversability maps must share one pixel grid")
    resolution_m = float(segmentation.map_resolution)
    minimum_clearance_m = recipe.trajectory.minimum_wall_clearance_m
    candidates: list[dict[str, Any]] = []
    rejected_rooms: dict[str, str] = {}

    for room_name, raw_room_id in sorted(segmentation.room_ins_name_to_ins_id.items()):
        room_mask = room_map == int(raw_room_id)
        if not np.any(room_mask):
            rejected_rooms[str(room_name)] = "empty_room_mask"
            continue
        clearance_px = cv2.distanceTransform(room_mask.astype(np.uint8), cv2.DIST_L2, 5)
        valid = room_mask & traversable & (clearance_px * resolution_m >= minimum_clearance_m)
        indices = np.argwhere(valid)
        if len(indices) == 0:
            rejected_rooms[str(room_name)] = "no_traversable_point_with_wall_clearance"
            continue
        room_indices = np.argwhere(room_mask)
        centroid = room_indices.mean(axis=0)
        centroid_distances = np.linalg.norm(indices - centroid, axis=1)
        primary_index = int(np.argmin(centroid_distances))
        primary_pixel = indices[primary_index]
        nominated = [(primary_pixel, 0)]
        separation_m = np.linalg.norm(indices - primary_pixel, axis=1) * resolution_m
        secondary_indices = np.flatnonzero(
            separation_m >= recipe.trajectory.minimum_distance_m
        )
        if len(secondary_indices):
            clearance_m = clearance_px[indices[:, 0], indices[:, 1]] * resolution_m
            scores = (
                clearance_m[secondary_indices]
                + 0.25 * np.minimum(separation_m[secondary_indices], 3.0)
                - 0.05 * centroid_distances[secondary_indices] * resolution_m
            )
            secondary_index = int(secondary_indices[int(np.argmax(scores))])
            nominated.append((indices[secondary_index], 1))
        for pixel, station_rank_in_room in nominated:
            clearance_m = float(clearance_px[tuple(pixel)] * resolution_m)
            world_xy = _as_numpy(
                segmentation.map_to_world(th.tensor(pixel, dtype=th.float32))
            )
            candidates.append(
                {
                    "room_instance": str(room_name),
                    "station_rank_in_room": station_rank_in_room,
                    "pixel": [int(pixel[0]), int(pixel[1])],
                    "position_m": [
                        float(world_xy[0]),
                        float(world_xy[1]),
                        float(segmentation.floor_heights[floor]),
                    ],
                    "wall_clearance_m": clearance_m,
                    "centroid_distance_m": float(
                        np.linalg.norm(pixel - centroid) * resolution_m
                    ),
                    "room_area_m2": float(
                        np.count_nonzero(room_mask) * resolution_m**2
                    ),
                }
            )
    if not candidates:
        raise RuntimeError(
            "no room contains a traversable T3 station with the configured wall clearance"
        )
    # Prefer a room with more visual opportunity, then a station close to its
    # geometric centre. The post-render panorama gate remains authoritative.
    ranked = sorted(
        candidates,
        key=lambda item: (
            item["room_area_m2"],
            -item["station_rank_in_room"],
            -item["centroid_distance_m"],
            item["wall_clearance_m"],
        ),
        reverse=True,
    )
    station_rank = recipe.trajectory.rotation_station_rank
    if station_rank >= len(ranked):
        raise RuntimeError(
            "T3 rotation station rank is unavailable: "
            f"requested={station_rank} available={len(ranked)}"
        )
    selected = ranked[station_rank]
    selection = {
        "schema_version": "omnigibson_trajectory_selection.v1",
        "sampling_strategy": "rotation_station",
        "trajectory_class": "T3",
        "candidate_room_count": len(segmentation.room_ins_name_to_ins_id),
        "valid_room_count": len(candidates),
        "rejected_rooms": rejected_rooms,
        "station_id": (
            f"{selected['room_instance']}:station{selected['station_rank_in_room']}"
        ),
        "requested_station_rank": station_rank,
        "station_rank_in_room": selected["station_rank_in_room"],
        "room_instance": selected["room_instance"],
        "station_position_m": selected["position_m"],
        "minimum_wall_clearance_m": minimum_clearance_m,
        "measured_wall_clearance_m": round(selected["wall_clearance_m"], 6),
        "centroid_distance_m": round(selected["centroid_distance_m"], 6),
        "selection_policy": (
            "ranked_room_area_then_distinct_clearance_certified_station"
        ),
    }
    return list(selected["position_m"]), selection


def _sample_among5_layout(
    scene: Any, recipe: EpisodeRecipe
) -> tuple[list[list[float]], dict[str, Any]]:
    """Place one exact five-object cross and certify four camera stations."""

    import cv2
    import numpy as np
    import torch as th

    contract = recipe.among5
    if contract is None:
        raise ValueError("procedural Among-5 sampling requires an among5 contract")
    if recipe.source.floor != 0:
        raise RuntimeError("procedural Among-5 currently requires floor 0")
    segmentation = getattr(scene, "seg_map", None)
    if segmentation is None or not isinstance(
        getattr(segmentation, "room_ins_name_to_ins_id", None), dict
    ):
        raise RuntimeError("procedural Among-5 requires room-instance segmentation")
    room_map = np.asarray(_as_numpy(segmentation.room_ins_map))
    traversable = np.asarray(_as_numpy(scene.trav_map.floor_map[0])) == 255
    if room_map.shape != traversable.shape:
        raise RuntimeError("room and traversability maps must share one pixel grid")
    resolution_m = float(segmentation.map_resolution)
    maximum_asset_radius_m = max(
        math.hypot(asset.bbox_size_m[0], asset.bbox_size_m[1]) / 2.0
        for asset in contract.assets
    )
    required_clearance_m = (
        max(
            contract.camera_radius_m,
            contract.layout_radius_m + maximum_asset_radius_m,
        )
        + contract.minimum_camera_wall_margin_m
    )
    floor_height = float(segmentation.floor_heights[0])
    candidates: list[dict[str, Any]] = []
    rejected_rooms: dict[str, str] = {}

    for room_name, raw_room_id in sorted(segmentation.room_ins_name_to_ins_id.items()):
        room_mask = room_map == int(raw_room_id)
        if not np.any(room_mask):
            rejected_rooms[str(room_name)] = "empty_room_mask"
            continue
        clearance_px = cv2.distanceTransform(room_mask.astype(np.uint8), cv2.DIST_L2, 5)
        valid = room_mask & traversable & (
            clearance_px * resolution_m >= required_clearance_m
        )
        indices = np.argwhere(valid)
        if len(indices) == 0:
            rejected_rooms[str(room_name)] = "insufficient_camera_ring_clearance"
            continue
        clearances = clearance_px[indices[:, 0], indices[:, 1]] * resolution_m
        chosen: tuple[Any, Any, list[list[float]], list[dict[str, Any]], float] | None = None
        evaluated_center_count = 0
        for candidate_index in np.argsort(-clearances):
            evaluated_center_count += 1
            pixel_center = indices[int(candidate_index)]
            center_xy = np.asarray(
                _as_numpy(
                    segmentation.map_to_world(
                        th.tensor(pixel_center, dtype=th.float32)
                    )
                ),
                dtype=np.float64,
            )
            camera_positions = []
            camera_checks = []
            for azimuth_deg in contract.camera_azimuth_deg:
                angle = math.radians(azimuth_deg)
                ideal_xy = center_xy + contract.camera_radius_m * np.asarray(
                    [math.sin(angle), -math.cos(angle)], dtype=np.float64
                )
                camera_pixel = np.asarray(
                    _as_numpy(
                        segmentation.world_to_map(
                            th.tensor(ideal_xy, dtype=th.float32)
                        )
                    ),
                    dtype=np.int64,
                )
                in_bounds = bool(
                    0 <= camera_pixel[0] < room_map.shape[0]
                    and 0 <= camera_pixel[1] < room_map.shape[1]
                )
                same_room = bool(
                    in_bounds and room_map[tuple(camera_pixel)] == int(raw_room_id)
                )
                is_traversable = bool(in_bounds and traversable[tuple(camera_pixel)])
                camera_checks.append(
                    {
                        "azimuth_deg": round(float(azimuth_deg), 6),
                        "map_pixel": [int(camera_pixel[0]), int(camera_pixel[1])],
                        "in_bounds": in_bounds,
                        "same_room": same_room,
                        "traversable": is_traversable,
                    }
                )
                camera_positions.append(
                    [float(ideal_xy[0]), float(ideal_xy[1]), floor_height]
                )
            if all(
                item["same_room"] and item["traversable"] for item in camera_checks
            ):
                chosen = (
                    pixel_center,
                    center_xy,
                    camera_positions,
                    camera_checks,
                    float(clearances[int(candidate_index)]),
                )
                break
        if chosen is None:
            rejected_rooms[str(room_name)] = (
                "camera_ring_not_fully_traversable:"
                f"centers_tested={evaluated_center_count}"
            )
            continue
        best_pixel, center_xy, camera_positions, camera_checks, measured_clearance = chosen
        candidates.append(
            {
                "room_instance": str(room_name),
                "room_area_m2": float(np.count_nonzero(room_mask) * resolution_m**2),
                "center_pixel": [int(best_pixel[0]), int(best_pixel[1])],
                "center_xy": center_xy,
                "wall_clearance_m": measured_clearance,
                "camera_positions_m": camera_positions,
                "camera_checks": camera_checks,
                "evaluated_center_count": evaluated_center_count,
            }
        )
    if not candidates:
        raise RuntimeError(
            "no semantic room supports the configured procedural Among-5 camera ring; "
            f"rejections={rejected_rooms}"
        )
    ranked = sorted(
        candidates,
        key=lambda item: (item["room_area_m2"], item["wall_clearance_m"]),
        reverse=True,
    )
    if contract.center_rank >= len(ranked):
        raise RuntimeError(
            "Among-5 center rank is unavailable: "
            f"requested={contract.center_rank} available={len(ranked)}"
        )
    selected = ranked[contract.center_rank]
    center_xy = selected["center_xy"]

    assets_by_name = {asset.name: asset for asset in contract.assets}
    anchor = next(asset for asset in contract.assets if asset.role == "anchor")
    slot_bearings = {"front": 0.0, "right": 90.0, "back": 180.0, "left": 270.0}
    placements = []

    def place(asset_name: str, slot: str, radius_m: float) -> None:
        asset = assets_by_name[asset_name]
        base_bearing = slot_bearings.get(slot, 0.0)
        transformed = -base_bearing if contract.mirrored else base_bearing
        world_bearing = (transformed + contract.layout_yaw_deg) % 360.0
        angle = math.radians(world_bearing)
        xy = center_xy + radius_m * np.asarray(
            [math.sin(angle), math.cos(angle)], dtype=np.float64
        )
        position = [
            float(xy[0]),
            float(xy[1]),
            floor_height + float(asset.bbox_size_m[2]) / 2.0,
        ]
        obj = scene.object_registry("name", asset.name)
        if obj is None:
            raise RuntimeError(f"configured Among-5 asset was not loaded: {asset.name}")
        obj.set_position_orientation(
            position=th.tensor(position, dtype=th.float32),
            orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
            frame="world",
        )
        placements.append(
            {
                "name": asset.name,
                "category": asset.category,
                "model": asset.model,
                "role": asset.role,
                "layout_slot": slot,
                "base_bearing_deg": base_bearing,
                "world_bearing_deg": round(world_bearing, 6),
                "position_m": [round(value, 6) for value in position],
                "declared_bbox_size_m": [round(value, 6) for value in asset.bbox_size_m],
            }
        )

    place(anchor.name, "anchor", 0.0)
    for slot, asset_name in zip(
        ("front", "right", "back", "left"),
        contract.satellite_order,
        strict=True,
    ):
        place(asset_name, slot, contract.layout_radius_m)

    for camera_index, (camera_position, camera_check) in enumerate(
        zip(selected["camera_positions_m"], selected["camera_checks"], strict=True)
    ):
        camera_xy = np.asarray(camera_position[:2], dtype=np.float64)
        forward = center_xy - camera_xy
        forward /= np.linalg.norm(forward)
        targets = []
        for placement in placements:
            if placement["role"] != "satellite":
                continue
            delta = np.asarray(placement["position_m"][:2], dtype=np.float64) - camera_xy
            forward_m = float(np.dot(delta, forward))
            lateral_m = float(abs(forward[0] * delta[1] - forward[1] * delta[0]))
            if forward_m <= contract.camera_radius_m:
                continue
            offset_deg = math.degrees(math.atan2(lateral_m, forward_m))
            targets.append((offset_deg, placement["name"]))
        if not targets:
            raise RuntimeError(f"Among-5 view {camera_index} has no satellite behind anchor")
        offset_deg, target_name = min(targets)
        camera_check["expected_target_name"] = target_name
        camera_check["expected_target_offset_deg"] = round(offset_deg, 6)

    minimum_gap_m = math.inf
    for left_index, left in enumerate(placements):
        for right in placements[left_index + 1 :]:
            center_distance = math.dist(left["position_m"][:2], right["position_m"][:2])
            left_radius = math.hypot(*left["declared_bbox_size_m"][:2]) / 2.0
            right_radius = math.hypot(*right["declared_bbox_size_m"][:2]) / 2.0
            minimum_gap_m = min(minimum_gap_m, center_distance - left_radius - right_radius)
    if minimum_gap_m < contract.minimum_object_gap_m:
        raise RuntimeError(
            "procedural Among-5 objects violate the declared planar gap: "
            f"measured={minimum_gap_m:.3f} required={contract.minimum_object_gap_m:.3f}"
        )

    selection = {
        "schema_version": "omnigibson_trajectory_selection.v1",
        "sampling_strategy": "procedural_among5",
        "trajectory_class": "T2",
        "among5_protocol_version": contract.protocol_version,
        "visibility_contract": contract.visibility_contract,
        "layout_id": contract.layout_id,
        "room_instance": selected["room_instance"],
        "center_rank": contract.center_rank,
        "center_position_m": [
            round(float(center_xy[0]), 6),
            round(float(center_xy[1]), 6),
            round(floor_height, 6),
        ],
        "layout_radius_m": contract.layout_radius_m,
        "camera_radius_m": contract.camera_radius_m,
        "layout_yaw_deg": contract.layout_yaw_deg,
        "mirrored": contract.mirrored,
        "satellite_order": list(contract.satellite_order),
        "placements": placements,
        "camera_azimuth_deg": list(contract.camera_azimuth_deg),
        "camera_pitch_down_deg": contract.camera_pitch_down_deg,
        "camera_positions_m": [
            [round(float(value), 6) for value in position]
            for position in selected["camera_positions_m"]
        ],
        "camera_checks": selected["camera_checks"],
        "minimum_declared_object_gap_m": round(float(minimum_gap_m), 6),
        "required_object_gap_m": contract.minimum_object_gap_m,
        "required_camera_wall_clearance_m": round(required_clearance_m, 6),
        "measured_center_wall_clearance_m": round(selected["wall_clearance_m"], 6),
        "candidate_room_count": len(segmentation.room_ins_name_to_ins_id),
        "valid_room_count": len(ranked),
        "rejected_rooms": rejected_rooms,
        "selection_policy": "largest_room_with_exact_traversable_cardinal_camera_ring",
    }
    return selected["camera_positions_m"], selection


def _traversable_room_points(scene: Any, floor: int) -> dict[str, Any]:
    """Return vectorized room-aware traversable points in world coordinates."""

    import numpy as np
    import torch as th

    segmentation = getattr(scene, "seg_map", None)
    if (
        floor != 0
        or segmentation is None
        or not isinstance(getattr(segmentation, "room_ins_name_to_ins_id", None), dict)
    ):
        raise RuntimeError("typed trajectory sampling requires floor-0 room segmentation")
    room_map = np.asarray(_as_numpy(segmentation.room_ins_map))
    traversable = np.asarray(_as_numpy(scene.trav_map.floor_map[floor])) == 255
    if room_map.shape != traversable.shape:
        raise RuntimeError("room and traversability maps must share one pixel grid")
    floor_height = float(segmentation.floor_heights[floor])
    by_room: dict[str, Any] = {}
    for room_name, room_id in sorted(segmentation.room_ins_name_to_ins_id.items()):
        pixels = np.argwhere(traversable & (room_map == int(room_id)))
        if len(pixels) == 0:
            continue
        world = _as_numpy(segmentation.map_to_world(th.tensor(pixels, dtype=th.float32)))
        by_room[str(room_name)] = {
            "pixels": pixels,
            "world_xy": np.asarray(world, dtype=np.float64),
            "floor_height": floor_height,
        }
    if not by_room:
        raise RuntimeError("scene contains no room-labelled traversable points")
    return by_room


def _scene_object_records(scene: Any) -> list[dict[str, Any]]:
    """Extract the minimum reliable geometry needed before bundle compilation."""

    records = []
    structural = {
        "background",
        "ceilings",
        "driveway",
        "fence",
        "floors",
        "lawn",
        "roof",
        "walls",
    }
    for obj in scene.objects:
        snapshot = _object_snapshot(obj)
        if snapshot is None or snapshot["category"] in structural:
            continue
        records.append(snapshot)
    return records


def _nearest_world_point(cloud: Any, ideal_xy: Any) -> tuple[Any, float]:
    import numpy as np

    distances = np.linalg.norm(cloud - np.asarray(ideal_xy, dtype=np.float64), axis=1)
    index = int(np.argmin(distances))
    return cloud[index], float(distances[index])


def _sample_object_orbit(
    scene: Any, recipe: EpisodeRecipe
) -> tuple[list[list[float]], list[float], dict[str, Any]]:
    """Select a focus with a certified full or minimum-arc traversable orbit."""

    import numpy as np

    by_room = _traversable_room_points(scene, recipe.source.floor)
    focus_categories = set(recipe.trajectory.focus_categories)
    view_count = recipe.trajectory.outbound_view_count
    candidates = []
    rejected: Counter[str] = Counter()

    full_azimuths = [360.0 * index / view_count for index in range(view_count)]
    azimuth_sequences = [(full_azimuths, 360.0, True)]
    minimum_arc_deg = recipe.trajectory.orbit_minimum_arc_deg
    if minimum_arc_deg < 360.0:
        start_step = 360.0 / view_count
        for start_index in range(view_count):
            start = start_index * start_step
            azimuth_sequences.append(
                (
                    [
                        (start + minimum_arc_deg * index / (view_count - 1)) % 360.0
                        for index in range(view_count)
                    ],
                    minimum_arc_deg,
                    False,
                )
            )

    def locally_connected(
        positions: list[list[float]], *, complete_orbit: bool
    ) -> bool:
        edges = list(pairwise(positions))
        if complete_orbit:
            edges.append((positions[-1], positions[0]))
        for start, goal in edges:
            path, geodesic = scene.get_shortest_path(
                floor=recipe.source.floor,
                source_world=start[:2],
                target_world=goal[:2],
                entire_path=True,
            )
            if path is None or geodesic is None:
                return False
            chord = math.hypot(goal[0] - start[0], goal[1] - start[1])
            geodesic_m = float(
                geodesic.item() if hasattr(geodesic, "item") else geodesic
            )
            if geodesic_m > 2.5 * chord + 0.5:
                return False
        return True

    def measured_arc_coverage(
        azimuths: list[float], *, complete_orbit: bool
    ) -> float:
        if complete_orbit:
            return 360.0
        ordered = sorted(float(value) % 360.0 for value in azimuths)
        gaps = [
            (ordered[(index + 1) % len(ordered)] - value) % 360.0
            for index, value in enumerate(ordered)
        ]
        return 360.0 - max(gaps)

    for entity in _scene_object_records(scene):
        if entity["category"] not in focus_categories:
            rejected["category_not_focus_eligible"] += 1
            continue
        room = entity["region"]
        if room not in by_room:
            rejected["room_has_no_traversable_cloud"] += 1
            continue
        center = np.asarray(entity["aabb_center_m"], dtype=np.float64)
        extent = np.asarray(entity["aabb_extent_m"], dtype=np.float64)
        planar_radius = float(np.linalg.norm(extent[:2]) / 2.0)
        minimum_radius = max(
            recipe.trajectory.orbit_minimum_radius_m,
            planar_radius + recipe.trajectory.orbit_object_clearance_m,
        )
        if minimum_radius > recipe.trajectory.orbit_maximum_radius_m:
            rejected["focus_too_large_for_orbit_radius"] += 1
            continue
        cloud = by_room[room]["world_xy"]
        relative = cloud - center[:2]
        cloud_radii = np.linalg.norm(relative, axis=1)
        cloud_azimuths = np.degrees(np.arctan2(relative[:, 1], relative[:, 0])) % 360.0
        geometric_options = []
        connected_options = []
        desired_radii = np.linspace(
            minimum_radius, recipe.trajectory.orbit_maximum_radius_m, 7
        )
        for desired_azimuths, coverage_deg, complete_orbit in azimuth_sequences:
            for desired_radius in desired_radii:
                positions: list[list[float]] = []
                actual_azimuths: list[float] = []
                actual_radii: list[float] = []
                used_indices: set[int] = set()
                maximum_angular_error = 0.0
                maximum_radius_error = 0.0
                valid = True
                for azimuth in desired_azimuths:
                    angular_errors = np.abs(
                        (cloud_azimuths - azimuth + 180.0) % 360.0 - 180.0
                    )
                    radius_errors = np.abs(cloud_radii - desired_radius)
                    score = angular_errors / 30.0 + radius_errors / 0.75
                    allowed = (
                        (cloud_radii >= minimum_radius)
                        & (cloud_radii <= recipe.trajectory.orbit_maximum_radius_m)
                    )
                    score[~allowed] = np.inf
                    for used_index in used_indices:
                        score[used_index] = np.inf
                    index = int(np.argmin(score))
                    if (
                        not np.isfinite(score[index])
                        or angular_errors[index] > 35.0
                        or radius_errors[index] > 0.75
                    ):
                        valid = False
                        break
                    used_indices.add(index)
                    maximum_angular_error = max(
                        maximum_angular_error, float(angular_errors[index])
                    )
                    maximum_radius_error = max(
                        maximum_radius_error, float(radius_errors[index])
                    )
                    point = cloud[index]
                    positions.append(
                        [
                            float(point[0]),
                            float(point[1]),
                            float(by_room[room]["floor_height"]),
                        ]
                    )
                    actual_azimuths.append(float(cloud_azimuths[index]))
                    actual_radii.append(float(cloud_radii[index]))
                if valid:
                    if (
                        measured_arc_coverage(
                            actual_azimuths, complete_orbit=complete_orbit
                        )
                        + 10.0
                        < minimum_arc_deg
                    ):
                        continue
                    geometric_options.append(True)
                    option = (
                        (0 if complete_orbit else 1) * 10_000.0
                        + maximum_angular_error
                        + maximum_radius_error * 10.0,
                        positions,
                        actual_azimuths,
                        actual_radii,
                        float(desired_radius),
                        coverage_deg,
                        complete_orbit,
                        False,
                    )
                    if locally_connected(positions, complete_orbit=complete_orbit):
                        connected_options.append(option)

            if recipe.trajectory.orbit_adaptive_radius:
                positions = []
                actual_azimuths = []
                actual_radii = []
                used_indices = set()
                maximum_angular_error = 0.0
                valid = True
                previous_radius: float | None = None
                for azimuth in desired_azimuths:
                    angular_errors = np.abs(
                        (cloud_azimuths - azimuth + 180.0) % 360.0 - 180.0
                    )
                    allowed = (
                        (angular_errors <= 35.0)
                        & (cloud_radii >= minimum_radius)
                        & (cloud_radii <= recipe.trajectory.orbit_maximum_radius_m)
                    )
                    for used_index in used_indices:
                        allowed[used_index] = False
                    if not np.any(allowed):
                        valid = False
                        break
                    radial_span = max(
                        recipe.trajectory.orbit_maximum_radius_m - minimum_radius,
                        1e-6,
                    )
                    score = angular_errors / 30.0 + 0.15 * (
                        cloud_radii - minimum_radius
                    ) / radial_span
                    if previous_radius is not None:
                        score += np.abs(cloud_radii - previous_radius) / 0.75
                    score[~allowed] = np.inf
                    index = int(np.argmin(score))
                    used_indices.add(index)
                    maximum_angular_error = max(
                        maximum_angular_error, float(angular_errors[index])
                    )
                    previous_radius = float(cloud_radii[index])
                    point = cloud[index]
                    positions.append(
                        [
                            float(point[0]),
                            float(point[1]),
                            float(by_room[room]["floor_height"]),
                        ]
                    )
                    actual_azimuths.append(float(cloud_azimuths[index]))
                    actual_radii.append(previous_radius)
                if valid:
                    if (
                        measured_arc_coverage(
                            actual_azimuths, complete_orbit=complete_orbit
                        )
                        + 10.0
                        < minimum_arc_deg
                    ):
                        continue
                    geometric_options.append(True)
                    radial_range = max(actual_radii) - min(actual_radii)
                    option = (
                        (0 if complete_orbit else 1) * 10_000.0
                        + maximum_angular_error
                        + radial_range * 2.0,
                        positions,
                        actual_azimuths,
                        actual_radii,
                        float(np.mean(actual_radii)),
                        coverage_deg,
                        complete_orbit,
                        True,
                    )
                    if locally_connected(positions, complete_orbit=complete_orbit):
                        connected_options.append(option)

        if not geometric_options:
            rejected["incomplete_traversable_orbit"] += 1
            continue
        if not connected_options:
            rejected["orbit_not_locally_connected"] += 1
            continue
        (
            _,
            positions,
            actual_azimuths,
            actual_radii,
            desired_radius,
            coverage_deg,
            complete_orbit,
            used_adaptive_radius,
        ) = min(connected_options, key=lambda item: item[0])
        score = (
            int(complete_orbit),
            min(float(extent[0]), float(extent[1])),
            float(extent[2]),
            -abs(
                desired_radius
                - float(
                    np.mean(
                        [
                            math.hypot(point[0] - center[0], point[1] - center[1])
                            for point in positions
                        ]
                    )
                )
            ),
        )
        candidates.append(
            (
                score,
                entity,
                positions,
                actual_azimuths,
                actual_radii,
                desired_radius,
                coverage_deg,
                complete_orbit,
                used_adaptive_radius,
            )
        )
    if not candidates:
        raise RuntimeError(
            "no focus object supports the configured traversable T4 orbit; "
            f"rejections={dict(rejected)}"
        )
    (
        _,
        focus,
        positions,
        azimuths,
        actual_radii,
        desired_radius,
        coverage_deg,
        complete_orbit,
        used_adaptive_radius,
    ) = max(candidates, key=lambda item: item[0])
    measured_arc_coverage_deg = measured_arc_coverage(
        azimuths, complete_orbit=complete_orbit
    )
    selection = {
        "schema_version": "omnigibson_trajectory_selection.v1",
        "sampling_strategy": "object_orbit",
        "trajectory_class": "T4",
        "focus_entity_id": focus["source_entity_id"],
        "focus_category": focus["category"],
        "focus_position_m": focus["aabb_center_m"],
        "focus_extent_m": focus["aabb_extent_m"],
        "room_instance": focus["region"],
        "candidate_count": len(candidates),
        "rejection_counts": dict(sorted(rejected.items())),
        "desired_radius_m": round(float(desired_radius), 6),
        "actual_radius_range_m": [
            round(float(min(actual_radii)), 6),
            round(float(max(actual_radii)), 6),
        ],
        "minimum_orbit_radius_m": recipe.trajectory.orbit_minimum_radius_m,
        "object_clearance_m": recipe.trajectory.orbit_object_clearance_m,
        "planned_arc_coverage_deg": round(float(coverage_deg), 6),
        "arc_coverage_deg": round(float(measured_arc_coverage_deg), 6),
        "arc_coverage_tolerance_deg": 10.0,
        "minimum_arc_required_deg": round(float(minimum_arc_deg), 6),
        "complete_orbit": complete_orbit,
        "adaptive_radius_used": used_adaptive_radius,
        "orientation_questions_allowed": False,
        "orientation_evidence_reason": ("scene IR has no certified category-level canonical front"),
        "selection_policy": (
            "prefer_complete_then_minimum_arc; adaptive_radius; connected_edges; "
            "largest_distinctive_extent"
        ),
    }
    return positions, azimuths, selection


def _sample_elevation_station(
    scene: Any, recipe: EpisodeRecipe
) -> tuple[list[float], dict[str, Any]]:
    position, base = _sample_rotation_station(scene, recipe)
    floor_height = float(position[2])
    ceiling_bottoms = []
    for obj in scene.objects:
        if str(getattr(obj, "category", "")) != "ceilings":
            continue
        snapshot = _object_snapshot(obj)
        if snapshot is None:
            continue
        center = snapshot["aabb_center_m"]
        extent = snapshot["aabb_extent_m"]
        if (
            abs(position[0] - center[0]) <= extent[0] / 2.0
            and abs(position[1] - center[1]) <= extent[1] / 2.0
        ):
            bottom = float(center[2] - extent[2] / 2.0)
            if bottom > floor_height + 1.8:
                ceiling_bottoms.append(bottom)
    configured_heights = list(recipe.trajectory.elevation_heights_m)
    measured_ceiling_m = min(ceiling_bottoms) if ceiling_bottoms else None
    safe_high_m = configured_heights[-1]
    if measured_ceiling_m is not None:
        safe_high_m = min(safe_high_m, measured_ceiling_m - floor_height - 0.25)
    if safe_high_m < 2.0:
        raise RuntimeError(
            "T7 station has insufficient vertical clearance for a high view: "
            f"safe_height_m={safe_high_m:.3f}"
        )
    heights = [*configured_heights[:-1], round(safe_high_m, 6)]
    selection = {
        **base,
        "sampling_strategy": "elevation_station",
        "trajectory_class": "T7",
        "height_sequence_m": heights,
        "pitch_sequence_deg": list(recipe.trajectory.elevation_pitch_down_deg),
        "measured_ceiling_bottom_m": (
            round(measured_ceiling_m, 6) if measured_ceiling_m is not None else None
        ),
        "minimum_ceiling_clearance_m": 0.25,
        "selection_policy": "reuse_room_central_station_for_clean_height_intervention",
    }
    return position, selection


def _sample_occlusion_reveal(
    scene: Any, recipe: EpisodeRecipe
) -> tuple[list[list[float]], dict[str, Any]]:
    """Search geometry for a camera-occluder-target alignment and reveal path."""

    import numpy as np

    by_room = _traversable_room_points(scene, recipe.source.floor)
    objects = _scene_object_records(scene)
    candidates = []
    rejection: Counter[str] = Counter()
    for occluder in objects:
        occ_center = np.asarray(occluder["aabb_center_m"], dtype=np.float64)
        occ_extent = np.asarray(occluder["aabb_extent_m"], dtype=np.float64)
        if occluder["category"] not in {
            "armchair",
            "bar",
            "bed",
            "bench",
            "bookcase",
            "bottom_cabinet",
            "breakfast_table",
            "commercial_kitchen_table",
            "countertop",
            "cubicle",
            "desk",
            "display_case",
            "grocery_shelf",
            "locker",
            "room_divider",
            "shelf",
            "sofa",
            "wardrobe",
        }:
            continue
        if max(occ_extent[:2]) < 0.8 or occ_extent[2] < 0.8:
            continue
        room = occluder["region"]
        if room not in by_room:
            continue
        cloud = by_room[room]["world_xy"]
        for target in objects:
            if target["source_entity_id"] == occluder["source_entity_id"]:
                continue
            if target["region"] != room:
                continue
            target_center = np.asarray(target["aabb_center_m"], dtype=np.float64)
            target_extent = np.asarray(target["aabb_extent_m"], dtype=np.float64)
            if target["category"] in {
                "ceilings",
                "door",
                "downlight",
                "electric_switch",
                "fire_alarm",
                "fire_sprinkler",
                "fixed_window",
                "floors",
                "openable_window",
                "painting",
                "picture",
                "room_light",
                "square_light",
                "wall_socket",
                "walls",
            }:
                continue
            separation = float(np.linalg.norm(target_center[:2] - occ_center[:2]))
            if not 0.7 <= separation <= 4.0 or max(target_extent[:2]) < 0.2:
                continue
            away = (occ_center[:2] - target_center[:2]) / separation
            ideal_camera = occ_center[:2] + away * max(1.2, float(max(occ_extent[:2])))
            camera_xy, camera_snap = _nearest_world_point(cloud, ideal_camera)
            if camera_snap > 0.75:
                rejection["no_occluded_station_near_ray"] += 1
                continue
            camera_to_target = target_center[:2] - camera_xy
            ray_length = float(np.linalg.norm(camera_to_target))
            if ray_length <= 1e-6:
                continue
            ray = camera_to_target / ray_length
            occ_relative = occ_center[:2] - camera_xy
            occ_depth = float(np.dot(occ_relative, ray))
            occ_perpendicular = float(np.linalg.norm(occ_relative - occ_depth * ray))
            occ_radius = 0.5 * float(max(occ_extent[:2]))
            target_radius = 0.5 * float(max(target_extent[:2]))
            if not (
                0.5 < occ_depth < ray_length - 0.25
                # A center-line hit alone is not enough: a similarly sized
                # target often remains visible around the AABB silhouette.
                # Keep a conservative size margin and let the semantic-mask
                # gate below remain the final authority.
                and target_radius <= 0.65 * occ_radius
                and target_extent[2] <= 0.8 * occ_extent[2]
                and occ_perpendicular + target_radius <= 0.8 * occ_radius
            ):
                rejection["weak_centerline_occlusion"] += 1
                continue
            perpendicular = np.array([-ray[1], ray[0]])
            reveal_ideal = target_center[:2] + perpendicular * max(
                1.2, 0.5 * float(max(target_extent[:2])) + 0.8
            )
            reveal_xy, reveal_snap = _nearest_world_point(cloud, reveal_ideal)
            if reveal_snap > 0.9:
                reveal_ideal = target_center[:2] - perpendicular * max(
                    1.2, 0.5 * float(max(target_extent[:2])) + 0.8
                )
                reveal_xy, reveal_snap = _nearest_world_point(cloud, reveal_ideal)
            if reveal_snap > 0.9:
                rejection["no_reveal_station"] += 1
                continue
            reveal_ray = target_center[:2] - reveal_xy
            reveal_length = float(np.linalg.norm(reveal_ray))
            if reveal_length <= 1e-6:
                continue
            reveal_unit = reveal_ray / reveal_length
            occ_from_reveal = occ_center[:2] - reveal_xy
            reveal_occ_depth = float(np.dot(occ_from_reveal, reveal_unit))
            reveal_occ_perpendicular = float(
                np.linalg.norm(occ_from_reveal - reveal_occ_depth * reveal_unit)
            )
            if 0.0 < reveal_occ_depth < reveal_length and reveal_occ_perpendicular <= (
                occ_radius + 0.2
            ):
                rejection["reveal_still_geometrically_occluded"] += 1
                continue
            path, geodesic = scene.get_shortest_path(
                floor=recipe.source.floor,
                source_world=camera_xy,
                target_world=reveal_xy,
                entire_path=True,
            )
            if path is None or geodesic is None:
                rejection["no_reveal_path"] += 1
                continue
            geodesic_m = float(geodesic.item() if hasattr(geodesic, "item") else geodesic)
            if not 1.0 <= geodesic_m <= recipe.trajectory.maximum_distance_m:
                rejection["reveal_path_distance_out_of_range"] += 1
                continue
            path_points = []
            for point in _as_numpy(path):
                values = [float(value) for value in point]
                if len(values) == 2:
                    values.append(float(by_room[room]["floor_height"]))
                path_points.append(values)
            radial_cover_margin = occ_radius - occ_perpendicular - target_radius
            vertical_cover_margin = occ_extent[2] - target_extent[2]
            score = (
                radial_cover_margin,
                vertical_cover_margin,
                max(target_extent[:2]),
                -camera_snap - reveal_snap,
            )
            candidates.append((score, occluder, target, path_points, geodesic_m))
    if not candidates:
        raise RuntimeError(
            f"no geometric T8 occlusion/reveal candidate found; rejections={dict(rejection)}"
        )
    _, occluder, target, path_points, geodesic_m = max(candidates, key=lambda item: item[0])
    selection = {
        "schema_version": "omnigibson_trajectory_selection.v1",
        "sampling_strategy": "occlusion_reveal",
        "trajectory_class": "T8",
        "target_entity_id": target["source_entity_id"],
        "target_category": target["category"],
        "target_position_m": target["aabb_center_m"],
        "occluder_entity_id": occluder["source_entity_id"],
        "occluder_category": occluder["category"],
        "room_instance": target["region"],
        "candidate_count": len(candidates),
        "rejection_counts": dict(sorted(rejection.items())),
        "selected_geodesic_distance_m": round(geodesic_m, 6),
        "selection_policy": "aabb_centerline_occlusion_then_navmesh_reveal",
        "post_render_gate_authoritative": True,
    }
    return path_points, selection


def _sample_target_views(
    scene: Any, recipe: EpisodeRecipe
) -> tuple[list[list[float]], list[tuple[str, str]], list[list[float]], dict[str, Any]]:
    """Choose independently reachable origin/facing anchor pairs for T10."""

    import numpy as np

    by_room = _traversable_room_points(scene, recipe.source.floor)
    objects = _scene_object_records(scene)
    anchor_categories = {
        "armchair",
        "bar",
        "bed",
        "bench",
        "bookcase",
        "bottom_cabinet",
        "breakfast_table",
        "coffee_table",
        "commercial_kitchen_table",
        "console_table",
        "countertop",
        "desk",
        "display_case",
        "fridge",
        "grocery_shelf",
        "locker",
        "nightstand",
        "pedestal_table",
        "room_divider",
        "shelf",
        "sofa",
        "straight_chair",
        "swivel_chair",
        "top_cabinet",
        "wardrobe",
        "wall_mounted_tv",
    }
    candidates = []
    for origin in objects:
        if origin["category"] not in anchor_categories:
            continue
        room = origin["region"]
        if room not in by_room:
            continue
        origin_center = np.asarray(origin["aabb_center_m"], dtype=np.float64)
        origin_extent = np.asarray(origin["aabb_extent_m"], dtype=np.float64)
        if max(origin_extent[:2]) < 0.35 or origin_extent[2] < 0.35:
            continue
        cloud = by_room[room]["world_xy"]
        for facing in objects:
            if facing["category"] not in anchor_categories:
                continue
            if facing["source_entity_id"] == origin["source_entity_id"]:
                continue
            if facing["region"] != room:
                continue
            facing_center = np.asarray(facing["aabb_center_m"], dtype=np.float64)
            facing_extent = np.asarray(facing["aabb_extent_m"], dtype=np.float64)
            if max(facing_extent[:2]) < 0.45 or facing_extent[2] < 0.45:
                continue
            separation = float(np.linalg.norm(facing_center[:2] - origin_center[:2]))
            if not 1.5 <= separation <= 8.0:
                continue
            away = (origin_center[:2] - facing_center[:2]) / separation
            origin_radius = 0.5 * float(max(origin_extent[:2]))
            ideal = origin_center[:2] + away * (origin_radius + 0.6)
            station, snap_m = _nearest_world_point(cloud, ideal)
            distance_to_origin = float(np.linalg.norm(station - origin_center[:2]))
            if snap_m > 0.75 or not (
                origin_radius + 0.25 <= distance_to_origin <= origin_radius + 1.2
            ):
                continue
            score = (
                float(max(facing_extent[:2]) * facing_extent[2]),
                separation,
                -snap_m,
            )
            candidates.append((score, origin, facing, station))
    if len(candidates) < recipe.trajectory.outbound_view_count:
        raise RuntimeError(
            "insufficient reachable T10 anchor pairs: "
            f"needed={recipe.trajectory.outbound_view_count} found={len(candidates)}"
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = []
    used_origins: set[str] = set()
    used_rooms: Counter[str] = Counter()
    for item in candidates:
        _, origin, facing, _ = item
        if origin["source_entity_id"] in used_origins:
            continue
        # Encourage room diversity without making single-room scenes impossible.
        if used_rooms[origin["region"]] >= 2 and any(
            used_rooms[candidate[1]["region"]] == 0 for candidate in candidates
        ):
            continue
        selected.append(item)
        used_origins.add(origin["source_entity_id"])
        used_rooms[origin["region"]] += 1
        if len(selected) == recipe.trajectory.outbound_view_count:
            break
    if len(selected) < recipe.trajectory.outbound_view_count:
        for item in candidates:
            if item in selected or item[1]["source_entity_id"] in used_origins:
                continue
            selected.append(item)
            used_origins.add(item[1]["source_entity_id"])
            if len(selected) == recipe.trajectory.outbound_view_count:
                break
    if len(selected) < recipe.trajectory.outbound_view_count:
        raise RuntimeError("T10 could not select enough distinct origin anchors")
    positions = []
    pairs = []
    facing_positions = []
    for _, origin, facing, station in selected:
        positions.append(
            [
                float(station[0]),
                float(station[1]),
                float(by_room[origin["region"]]["floor_height"]),
            ]
        )
        pairs.append((origin["source_entity_id"], facing["source_entity_id"]))
        facing_positions.append(list(facing["aabb_center_m"]))
    selection = {
        "schema_version": "omnigibson_trajectory_selection.v1",
        "sampling_strategy": "target_view",
        "trajectory_class": "T10",
        "candidate_count": len(candidates),
        "anchor_pairs": [
            {
                "origin_entity_id": origin,
                "facing_entity_id": facing,
            }
            for origin, facing in pairs
        ],
        "held_out": True,
        "selection_policy": "reachable_origin_neighbor_facing_distinct_semantic_anchor",
    }
    return positions, pairs, facing_positions, selection


def _object_snapshot(obj: Any) -> dict[str, Any] | None:
    try:
        center = _as_float_list(obj.aabb_center)
        extent = _as_float_list(obj.aabb_extent)
    except Exception:
        return None
    if len(center) != 3 or len(extent) != 3 or any(value <= 1e-6 for value in extent):
        return None
    category = str(getattr(obj, "category", "object"))
    name = str(getattr(obj, "name", "")).strip()
    if not name:
        return None
    rooms_raw = getattr(obj, "in_rooms", None)
    if rooms_raw is None:
        rooms = ["unassigned"]
    elif isinstance(rooms_raw, str):
        rooms = [rooms_raw.strip()] if rooms_raw.strip() else ["unassigned"]
    else:
        rooms = sorted({str(room).strip() for room in rooms_raw if str(room).strip()}) or [
            "unassigned"
        ]
    model = str(getattr(obj, "model", getattr(obj, "model_name", "unknown")))
    return {
        "source_entity_id": name,
        "name": name,
        "category": category,
        "model": model,
        "prim_path": str(getattr(obj, "prim_path", "")),
        "region": rooms[0],
        "aabb_center_m": center,
        "aabb_extent_m": extent,
        "articulated": bool(getattr(obj, "articulated", False)),
        "editable": True,
    }


def _capture_scene_snapshot(
    env: Any,
    recipe: EpisodeRecipe,
    runtime_registry: dict[int, str],
    runtime_semantic_registry: dict[int, str],
) -> dict[str, Any]:
    entities = []
    for obj in env.scene.objects:
        snapshot = _object_snapshot(obj)
        if snapshot is not None:
            entities.append(snapshot)
    entities.sort(key=lambda item: item["source_entity_id"])
    raw_scene_file = getattr(env.scene, "scene_file", None)
    scene_file = (
        Path(raw_scene_file)
        if isinstance(raw_scene_file, (str, os.PathLike)) and str(raw_scene_file)
        else None
    )
    if scene_file is not None and scene_file.is_file():
        source_digest = sha256_file(scene_file)
        source_path = str(scene_file.resolve())
    else:
        source_path = None
        source_digest = sha256_json(
            {
                "source_version": recipe.source.source_version,
                "source_scene_id": _scene_source_id(recipe),
                "entities": [
                    (item["source_entity_id"], item["category"], item["model"]) for item in entities
                ],
            }
        )
    return {
        "protocol_version": "omnigibson_scene_snapshot.v1",
        "source_name": "omnigibson",
        "source_version": recipe.source.source_version,
        "source_scene_id": _scene_source_id(recipe),
        "scene_model": recipe.source.scene_model,
        "scene_instance": recipe.source.scene_instance,
        "source_path": source_path,
        "source_digest": source_digest,
        "license": {
            "identifier": "BEHAVIOR-1K-ASSET-LICENSE",
            "academic_only": True,
            "redistribution": "metadata_only",
            "terms_uri": "https://behavior.stanford.edu/",
        },
        "entities": entities,
        "runtime_instance_registry": {
            str(identifier): name for identifier, name in sorted(runtime_registry.items())
        },
        "runtime_semantic_registry": {
            str(identifier): category
            for identifier, category in sorted(runtime_semantic_registry.items())
        },
    }


def _render_views(
    *,
    og: Any,
    sensor: Any,
    trajectory: dict[str, Any],
    recipe: EpisodeRecipe,
    category_by_name: dict[str, str],
    object_name_by_prim_path: dict[str, str],
    staging_root: Path,
    public_root: Path,
) -> tuple[list[dict[str, Any]], dict[int, str], dict[int, str]]:
    import numpy as np
    import torch as th

    expected_shape = (recipe.sensor.height_px, recipe.sensor.width_px)
    high_quality_observations = {}

    # Phase 1: collect model-visible evidence with the HQ path tracer. Isaac Sim
    # 5.1's semantic annotator segfaults when attached to this render mode, so
    # no segmentation graph is present during this phase.
    for expected_step, view in enumerate(trajectory["views"]):
        if int(view["step"]) != expected_step:
            raise ValueError("trajectory steps must be contiguous")
        transform = view["world_from_agent"]
        position = list(transform["translation_m"])
        effective_height = float(view.get("camera_height_m", recipe.trajectory.camera_height_m))
        position[2] += effective_height
        camera_rotation = omnigibson_camera_quaternion(transform["rotation_xyzw"])
        sensor.set_position_orientation(
            position=th.tensor(position, dtype=th.float32),
            orientation=th.tensor(camera_rotation, dtype=th.float32),
            frame="world",
        )
        for _ in range(recipe.sensor.render_warmup_frames):
            og.sim.render()
        observations, _ = sensor.get_obs()
        rgb = _rgb_u8(observations["rgb"])
        depth = _depth_f32(observations["depth_linear"], expected_shape)
        high_quality_observations[view["view_id"]] = (rgb, depth)
        write_npz_atomic(
            staging_root / "intermediate" / f"{view['view_id']}.rgb_depth.npz",
            rgb=rgb,
            depth_m=depth,
        )
        print(
            f"render phase=hq_rgb_depth view={view['view_id']} "
            f"progress={expected_step + 1}/{len(trajectory['views'])}",
            flush=True,
        )

    # Phase 2: switch the same camera and scene to stable real-time ray tracing
    # before attaching segmentation. Pixel labels are geometric evidence and do
    # not depend on the RGB shading mode; poses and resolution remain identical.
    import carb

    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "RaytracedLighting")
    settings.set_int("/rtx/post/dlss/execMode", 0)
    for _ in range(4):
        og.sim.render()
    import omni.replicator.core as rep

    annotator = rep.AnnotatorRegistry.get_annotator(
        "instance_id_segmentation_fast",
        device="cpu",
    )
    with og.sim.editing_usd():
        annotator.attach([sensor._render_product.path])

    rendered = []
    runtime_instance_registry = {0: "background", 1: "unlabelled"}
    runtime_semantic_registry = {0: "background", 1: "unlabelled"}
    try:
        for expected_step, view in enumerate(trajectory["views"]):
            transform = view["world_from_agent"]
            position = list(transform["translation_m"])
            effective_height = float(view.get("camera_height_m", recipe.trajectory.camera_height_m))
            position[2] += effective_height
            camera_rotation = omnigibson_camera_quaternion(transform["rotation_xyzw"])
            sensor.set_position_orientation(
                position=th.tensor(position, dtype=th.float32),
                orientation=th.tensor(camera_rotation, dtype=th.float32),
                frame="world",
            )
            for _ in range(recipe.sensor.render_warmup_frames):
                og.sim.render()
            sensor.get_obs()
            raw = annotator.get_data()
            if not isinstance(raw, dict) or "data" not in raw or "info" not in raw:
                raise RuntimeError("instance-ID annotator returned no mapping metadata")
            renderer_instance = _mask_u32(
                raw["data"],
                "renderer instance",
                expected_shape,
            )
            id_to_prim_path = raw["info"].get("idToLabels", {})
            if not isinstance(id_to_prim_path, dict) or not id_to_prim_path:
                raise RuntimeError("instance-ID annotator returned an empty prim-path map")
            instance, registry, label_audit = _remap_renderer_instances(
                renderer_instance,
                id_to_prim_path,
                object_name_by_prim_path,
            )
            write_json_atomic(
                staging_root / "diagnostics" / f"{view['view_id']}.instance_label_audit.json",
                label_audit,
            )
            if label_audit["mapped_resolvable_pixel_fraction"] < 0.95:
                raise RuntimeError(
                    "less than 95% of prim-resolvable renderer pixels map to scene objects: "
                    f"{label_audit}"
                )
            if label_audit["known_unresolvable_pixel_fraction"] > 0.10:
                raise RuntimeError(
                    "more than 10% of rendered pixels have an explicit INVALID or "
                    f"UNLABELLED renderer label: {label_audit}"
                )
            runtime_instance_registry.update(registry)
            rgb, depth = high_quality_observations[view["view_id"]]
            semantic, semantic_registry = _semantic_from_instances(
                instance,
                registry,
                category_by_name,
            )
            runtime_semantic_registry.update(semantic_registry)
            identifiers, counts = np.unique(instance, return_counts=True)
            visible = [
                int(identifier)
                for identifier, count in zip(identifiers, counts, strict=True)
                if int(identifier) > 1 and int(count) >= recipe.sensor.minimum_instance_pixels
            ]
            relative = Path("views") / f"{view['view_id']}.sensors.npz"
            staging_path = staging_root / relative
            write_npz_atomic(
                staging_path,
                rgb=rgb,
                depth_m=depth,
                instance_id=instance,
                semantic_id=semantic,
            )
            valid_depth = (
                np.isfinite(depth)
                & (depth >= recipe.sensor.near_m)
                & (depth <= recipe.sensor.far_m)
            )
            rendered.append(
                {
                    "step": expected_step,
                    "view_id": view["view_id"],
                    "artifact": {
                        "path": str((public_root / relative).resolve()),
                        "sha256": sha256_file(staging_path),
                        "byte_size": staging_path.stat().st_size,
                        "media_type": "application/x.numpy-npz",
                    },
                    "visible_runtime_semantic_ids": visible,
                    "visible_runtime_instance_ids": visible,
                    "visible_instance_count": len(visible),
                    "instance_label_audit": label_audit,
                    "valid_depth_fraction": round(
                        float(np.count_nonzero(valid_depth)) / depth.size, 6
                    ),
                    "camera_height_m": effective_height,
                }
            )
            print(
                f"render phase=instance_labels view={view['view_id']} "
                f"progress={expected_step + 1}/{len(trajectory['views'])}",
                flush=True,
            )
    finally:
        with og.sim.editing_usd():
            annotator.detach()
    shutil.rmtree(staging_root / "intermediate")
    shutil.rmtree(staging_root / "diagnostics", ignore_errors=True)
    return rendered, runtime_instance_registry, runtime_semantic_registry


def acquire_episode(
    *,
    recipe: EpisodeRecipe,
    output_directory: Path,
    gpu_id: int,
    headless: bool,
    overwrite: bool = False,
) -> Path:
    if gpu_id < 0:
        raise ValueError("gpu_id cannot be negative")
    output_directory = output_directory.resolve()
    if output_directory.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output_directory}")
        shutil.rmtree(output_directory)
    if overwrite:
        for stale in output_directory.parent.glob(f".{output_directory.name}.staging.*"):
            shutil.rmtree(stale)
    staging = output_directory.with_name(f".{output_directory.name}.staging.{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    os.environ["OMNIGIBSON_GPU_ID"] = str(gpu_id)
    env = None
    result_path: Path | None = None
    pending_error: Exception | None = None
    try:
        import numpy as np
        import omnigibson as og
        import torch as th
        from omnigibson.macros import gm
        from omnigibson.utils.constants import STRUCTURE_CATEGORIES

        random.seed(recipe.seed)
        np.random.seed(recipe.seed)
        th.manual_seed(recipe.seed)
        gm.HEADLESS = headless
        gm.ENABLE_HQ_RENDERING = recipe.sensor.high_quality_rendering
        config = _environment_config(recipe)
        if config["scene"].get("load_object_categories") == "__STRUCTURE_CATEGORIES__":
            config["scene"]["load_object_categories"] = sorted(STRUCTURE_CATEGORIES)
        env = og.Environment(configs=config)
        floor_area = _floor_area(env.scene, recipe.source.floor)
        source_scene_id = _scene_source_id(recipe)
        scene_id = stable_scene_id(recipe.source.source_version, source_scene_id)
        backend_version = str(getattr(og, "__version__", recipe.source.source_version))
        strategy = recipe.trajectory.sampling_strategy
        if strategy == "procedural_among5":
            camera_positions, selection = _sample_among5_layout(env.scene, recipe)
            for _ in range(2):
                og.sim.step()
            trajectory = build_among5_trajectory(
                scene_id=scene_id,
                recipe_id=recipe.recipe_id,
                seed=recipe.seed,
                camera_positions=camera_positions,
                focus_position=selection["center_position_m"],
                camera_pitch_down_deg=selection["camera_pitch_down_deg"],
                floor_area_m2=floor_area,
                backend_version=backend_version,
            )
        elif strategy == "rotation_station":
            station_position, selection = _sample_rotation_station(env.scene, recipe)
            trajectory = build_rotation_station_trajectory(
                scene_id=scene_id,
                recipe_id=recipe.recipe_id,
                seed=recipe.seed,
                station_position=station_position,
                station_id=selection["station_id"],
                yaw_step_deg=recipe.trajectory.yaw_step_deg,
                view_count=recipe.trajectory.outbound_view_count,
                floor_area_m2=floor_area,
                backend_version=backend_version,
            )
        elif strategy == "object_orbit":
            orbit_positions, azimuths, selection = _sample_object_orbit(env.scene, recipe)
            trajectory = build_object_orbit_trajectory(
                scene_id=scene_id,
                recipe_id=recipe.recipe_id,
                seed=recipe.seed,
                orbit_positions=orbit_positions,
                focus_position=selection["focus_position_m"],
                focus_entity_id=selection["focus_entity_id"],
                azimuth_deg_per_view=azimuths,
                floor_area_m2=floor_area,
                backend_version=backend_version,
            )
        elif strategy == "elevation_station":
            station_position, selection = _sample_elevation_station(env.scene, recipe)
            trajectory = build_elevation_trajectory(
                scene_id=scene_id,
                recipe_id=recipe.recipe_id,
                seed=recipe.seed,
                station_position=station_position,
                station_id=selection["station_id"],
                heights_m=selection["height_sequence_m"],
                pitch_down_deg=recipe.trajectory.elevation_pitch_down_deg,
                floor_area_m2=floor_area,
                backend_version=backend_version,
            )
        elif strategy == "occlusion_reveal":
            path_points, selection = _sample_occlusion_reveal(env.scene, recipe)
            trajectory = build_occlusion_reveal_trajectory(
                scene_id=scene_id,
                recipe_id=recipe.recipe_id,
                seed=recipe.seed,
                source_points=path_points,
                view_count=recipe.trajectory.outbound_view_count,
                target_position=selection["target_position_m"],
                target_entity_id=selection["target_entity_id"],
                occluder_entity_id=selection["occluder_entity_id"],
                floor_area_m2=floor_area,
                backend_version=backend_version,
            )
        elif strategy == "target_view":
            positions, pairs, facing_positions, selection = _sample_target_views(env.scene, recipe)
            trajectory = build_target_view_trajectory(
                scene_id=scene_id,
                recipe_id=recipe.recipe_id,
                seed=recipe.seed,
                camera_positions=positions,
                anchor_pairs=pairs,
                facing_positions=facing_positions,
                floor_area_m2=floor_area,
                backend_version=backend_version,
            )
        else:
            points, _, selection = _sample_path(env.scene, recipe)
            trajectory = build_closed_trajectory(
                scene_id=scene_id,
                recipe_id=recipe.recipe_id,
                seed=recipe.seed,
                source_points=points,
                outbound_view_count=recipe.trajectory.outbound_view_count,
                floor_area_m2=floor_area,
                backend_version=backend_version,
            )
        write_json_atomic(staging / "trajectory_selection.json", selection)
        if strategy == "procedural_among5":
            write_json_atomic(
                staging / "among5_layout_truth.json",
                {
                    "protocol_version": selection["among5_protocol_version"],
                    "visibility_contract": selection["visibility_contract"],
                    "layout_id": selection["layout_id"],
                    "scene_id": scene_id,
                    "room_instance": selection["room_instance"],
                    "center_position_m": selection["center_position_m"],
                    "layout_yaw_deg": selection["layout_yaw_deg"],
                    "mirrored": selection["mirrored"],
                    "placements": selection["placements"],
                    "camera_azimuth_deg": selection["camera_azimuth_deg"],
                    "camera_positions_m": selection["camera_positions_m"],
                    "camera_checks": selection["camera_checks"],
                },
            )
        write_json_atomic(staging / "trajectory_plan.json", trajectory)
        sensor = env.external_sensors["episode_camera"]
        category_by_name = {
            str(obj.name): str(getattr(obj, "category", "object")) for obj in env.scene.objects
        }
        object_name_by_prim_path = {
            str(obj.prim_path).rstrip("/"): str(obj.name)
            for obj in env.scene.objects
            if str(getattr(obj, "prim_path", "")).startswith("/")
        }
        rendered, instance_registry, semantic_registry = _render_views(
            og=og,
            sensor=sensor,
            trajectory=trajectory,
            recipe=recipe,
            category_by_name=category_by_name,
            object_name_by_prim_path=object_name_by_prim_path,
            staging_root=staging,
            public_root=output_directory,
        )
        snapshot = _capture_scene_snapshot(
            env,
            recipe,
            instance_registry,
            semantic_registry,
        )
        write_json_atomic(staging / "scene_snapshot.json", snapshot)
        trajectory_path = staging / "trajectory_plan.json"
        report = {
            "protocol_version": "omnigibson_render_bundle.v1",
            "status": "success",
            "scene_id": scene_id,
            "trajectory_sha256": sha256_file(trajectory_path),
            "gpu_id": gpu_id,
            "headless": headless,
            "high_quality_rendering": recipe.sensor.high_quality_rendering,
            "sensor_contract": {
                "width_px": recipe.sensor.width_px,
                "height_px": recipe.sensor.height_px,
                "horizontal_fov_deg": recipe.sensor.horizontal_fov_deg,
                "sensor_height_m": recipe.trajectory.camera_height_m,
                "near_m": recipe.sensor.near_m,
                "far_m": recipe.sensor.far_m,
                "depth_unit": "meter",
                "instance_background_ids": [0, 1],
                "instance_source": "replicator_instance_id_fast_object_projection",
                "renderer_id_mapping": "longest_scene_object_prim_ancestor",
                "renderer_unresolvable_policy": (
                    "explicit_background_invalid_unlabelled_separate_v2"
                ),
                "semantic_source": "instance_category_projection",
                "rgb_render_mode": "RealTimePathTracing",
                "label_render_mode": "RaytracedLighting",
                "minimum_visible_instance_pixels": recipe.sensor.minimum_instance_pixels,
            },
            "views": rendered,
        }
        write_json_atomic(staging / "render_report.json", report)
        os.replace(staging, output_directory)
        result_path = output_directory / "render_report.json"
    except Exception as error:
        failure = {
            "protocol_version": "omnigibson_acquisition_failure.v1",
            "status": "failure",
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        write_json_atomic(staging / "failure_report.json", failure)
        if output_directory.exists():
            shutil.rmtree(output_directory)
        os.replace(staging, output_directory)
        pending_error = error
    finally:
        if env is not None:
            try:
                import omnigibson as og

                og.shutdown()
            except BaseException:
                if pending_error is None:
                    raise
    if pending_error is not None:
        raise pending_error
    if result_path is None:
        raise RuntimeError("acquisition ended without a success or failure result")
    return result_path
