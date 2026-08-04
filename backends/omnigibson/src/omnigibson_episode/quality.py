"""Validate a render bundle and build a dependency-free local HTML preview."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from omnigibson_episode.io import sha256_file, write_json_atomic


def _mask_rgb(mask: np.ndarray) -> np.ndarray:
    identifiers = mask.astype(np.uint64)
    colored = np.stack(
        (
            (identifiers * 73 + 19) % 251,
            (identifiers * 151 + 47) % 253,
            (identifiers * 199 + 89) % 255,
        ),
        axis=-1,
    ).astype(np.uint8)
    colored[identifiers == 0] = 0
    colored[identifiers == 1] = 32
    return colored


def _depth_rgb(depth: np.ndarray, near_m: float, far_m: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth >= near_m) & (depth <= far_m)
    normalized = np.zeros(depth.shape, dtype=np.float32)
    if np.any(valid):
        lower, upper = np.percentile(depth[valid], (2.0, 98.0))
        if upper <= lower:
            upper = lower + 1.0
        normalized[valid] = np.clip((depth[valid] - lower) / (upper - lower), 0.0, 1.0)
    # A compact blue-cyan-yellow ramp; invalid pixels remain black.
    red = np.clip(2.0 * normalized - 0.25, 0.0, 1.0)
    green = np.clip(2.0 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(1.25 - 2.0 * normalized, 0.0, 1.0)
    result = (np.stack((red, green, blue), axis=-1) * 255.0).astype(np.uint8)
    result[~valid] = 0
    return result


def _sharpness(rgb: np.ndarray) -> float:
    gray = np.asarray(rgb, dtype=np.float32).mean(axis=-1)
    laplacian = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(np.var(laplacian))


def _rgb_loop_diagnostics(first_rgb: np.ndarray, last_rgb: np.ndarray) -> dict[str, Any]:
    first = first_rgb.astype(np.float32)
    last = last_rgb.astype(np.float32)
    difference = first - last
    mse = float(np.mean(difference * difference))
    exact = mse == 0.0
    psnr = None if exact else float(10.0 * np.log10(255.0**2 / mse))
    channel_bias = np.mean(last - first, axis=(0, 1))
    residual = (last - first) - channel_bias.reshape(1, 1, -1)
    residual_mse = float(np.mean(residual * residual))
    residual_psnr = None if residual_mse == 0.0 else float(10.0 * np.log10(255.0**2 / residual_mse))
    first_flat = (first - np.mean(first, axis=(0, 1), keepdims=True)).reshape(-1)
    last_flat = (last - np.mean(last, axis=(0, 1), keepdims=True)).reshape(-1)
    correlation = (
        float(np.corrcoef(first_flat, last_flat)[0, 1])
        if float(np.std(first_flat)) > 1e-6 and float(np.std(last_flat)) > 1e-6
        else None
    )
    return {
        "rgb_exact": exact,
        "rgb_mae": round(float(np.mean(np.abs(difference))), 3),
        "rgb_psnr_db": round(psnr, 3) if psnr is not None else None,
        "last_minus_first_channel_mean": [round(float(value), 3) for value in channel_bias],
        "bias_corrected_rgb_psnr_db": (
            round(residual_psnr, 3) if residual_psnr is not None else None
        ),
        "rgb_structure_correlation": (round(correlation, 6) if correlation is not None else None),
        "exposure_shift_likely": bool(
            not exact
            and correlation is not None
            and correlation >= 0.95
            and (residual_psnr is None or residual_psnr >= 28.0)
        ),
    }


def _write_card(
    *,
    preview_root: Path,
    view_id: str,
    rgb: np.ndarray,
    depth: np.ndarray,
    instance: np.ndarray,
    near_m: float,
    far_m: float,
) -> Path:
    panels = [
        Image.fromarray(rgb),
        Image.fromarray(_depth_rgb(depth, near_m, far_m)),
        Image.fromarray(_mask_rgb(instance)),
    ]
    panel_width = 384
    panel_height = round(rgb.shape[0] * panel_width / rgb.shape[1])
    panels = [
        panel.resize((panel_width, panel_height), Image.Resampling.LANCZOS) for panel in panels
    ]
    title_height = 36
    card = Image.new("RGB", (panel_width * 3, panel_height + title_height), "#111827")
    draw = ImageDraw.Draw(card)
    draw.text((10, 10), f"{view_id} | RGB", fill="white")
    draw.text((panel_width + 10, 10), "metric depth", fill="white")
    draw.text((panel_width * 2 + 10, 10), "instance IDs", fill="white")
    for index, panel in enumerate(panels):
        card.paste(panel, (index * panel_width, title_height))
    path = preview_root / f"{view_id}.png"
    card.save(path, optimize=True)
    return path


def _rotation_station_gate(
    *,
    bundle_directory: Path,
    trajectory: dict[str, Any],
    render: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the T3 zero-parallax and panorama-evidence requirements."""

    snapshot = json.loads((bundle_directory / "scene_snapshot.json").read_text(encoding="utf-8"))
    structural = {"background", "ceilings", "driveway", "fence", "floors", "lawn", "roof", "walls"}
    category_by_name = {
        str(entity["name"]): str(entity.get("category", "object"))
        for entity in snapshot.get("entities", [])
    }
    name_by_runtime_id = {
        int(identifier): str(name)
        for identifier, name in snapshot.get("runtime_instance_registry", {}).items()
        if int(identifier) > 1
    }

    def core_ids(view: dict[str, Any]) -> set[int]:
        return {
            int(identifier)
            for identifier in view["visible_runtime_instance_ids"]
            if category_by_name.get(name_by_runtime_id.get(int(identifier), ""), "")
            not in structural
        }

    visible_sets = [core_ids(view) for view in render["views"]]
    yaws = [float(value) % 360.0 for value in trajectory["yaw_sequence_deg"]]
    initial_yaw = yaws[0]

    def signed_delta(yaw: float) -> float:
        return ((yaw - initial_yaw + 180.0) % 360.0) - 180.0

    aligned = list(zip(yaws, visible_sets, strict=True))
    front = set().union(*(visible for yaw, visible in aligned if abs(signed_delta(yaw)) < 90.0))
    rear = set().union(*(visible for yaw, visible in aligned if abs(signed_delta(yaw)) > 90.0))
    rear_only = rear - front
    positions = [
        tuple(float(value) for value in view["world_from_agent"]["translation_m"])
        for view in trajectory["views"]
    ]
    origin = np.asarray(positions[0], dtype=np.float64)
    maximum_baseline_m = max(
        float(np.linalg.norm(np.asarray(position, dtype=np.float64) - origin))
        for position in positions
    )
    panorama_union = set().union(*visible_sets)
    checks = {
        "zero_parallax": maximum_baseline_m <= 1e-6,
        "panorama_core_entities_at_least_8": len(panorama_union) >= 8,
        "rear_only_core_entities_at_least_2": len(rear_only) >= 2,
        "depth_questions_forbidden": bool(trajectory.get("depth_questions_forbidden")),
        "yaw_sequence_matches_views": len(yaws) == len(visible_sets),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "station_id": trajectory.get("station_id"),
        "yaw_sequence_deg": yaws,
        "maximum_translational_baseline_m": round(maximum_baseline_m, 9),
        "panorama_core_entity_count": len(panorama_union),
        "minimum_core_entities_per_view": min(map(len, visible_sets)),
        "rear_only_core_entity_count": len(rear_only),
        "rear_only_runtime_instance_ids": sorted(rear_only),
    }


def _among5_gate(
    *,
    trajectory: dict[str, Any],
    render: dict[str, Any],
    snapshot: dict[str, Any],
    selection: dict[str, Any],
    bundle_directory: Path | None = None,
) -> dict[str, Any]:
    """Verify exact layout cardinality, geometry, and multi-view evidence."""

    placements = selection.get("placements", [])
    declared_names = [str(item.get("name", "")) for item in placements]
    declared_categories = [str(item.get("category", "")) for item in placements]
    entities = {
        str(item.get("name", item.get("source_entity_id", ""))): item
        for item in snapshot.get("entities", [])
    }
    runtime_by_name = {
        str(name): int(identifier)
        for identifier, name in snapshot.get("runtime_instance_registry", {}).items()
        if int(identifier) > 1
    }
    visible_sets = [
        {int(identifier) for identifier in view.get("visible_runtime_instance_ids", [])}
        for view in render.get("views", [])
    ]
    visible_view_count = {
        name: sum(runtime_by_name.get(name) in visible for visible in visible_sets)
        for name in declared_names
    }
    among_visible_per_view = [
        sum(runtime_by_name.get(name) in visible for name in declared_names)
        for visible in visible_sets
    ]

    anchor_items = [item for item in placements if item.get("role") == "anchor"]
    relation_errors_deg: dict[str, float] = {}
    radial_errors_m: dict[str, float] = {}
    if len(anchor_items) == 1 and anchor_items[0].get("name") in entities:
        anchor_entity = entities[str(anchor_items[0]["name"])]
        anchor_xy = np.asarray(anchor_entity["aabb_center_m"][:2], dtype=np.float64)
        for item in placements:
            if item.get("role") != "satellite" or item.get("name") not in entities:
                continue
            entity = entities[str(item["name"])]
            delta = np.asarray(entity["aabb_center_m"][:2], dtype=np.float64) - anchor_xy
            measured_bearing = float(np.degrees(np.arctan2(delta[0], delta[1])) % 360.0)
            expected_bearing = float(item["world_bearing_deg"]) % 360.0
            error = abs((measured_bearing - expected_bearing + 180.0) % 360.0 - 180.0)
            relation_errors_deg[str(item["name"])] = round(error, 6)
            radial_errors_m[str(item["name"])] = round(
                abs(float(np.linalg.norm(delta)) - float(selection["layout_radius_m"])),
                6,
            )

    checks: dict[str, bool] = {
        "exactly_five_declared_assets": len(placements) == 5,
        "one_anchor_four_satellites": (
            len(anchor_items) == 1
            and sum(item.get("role") == "satellite" for item in placements) == 4
        ),
        "unique_names_and_categories": (
            len(set(declared_names)) == 5 and len(set(declared_categories)) == 5
        ),
        "all_assets_present_in_snapshot": all(name in entities for name in declared_names),
        "all_assets_have_runtime_ids": all(name in runtime_by_name for name in declared_names),
        "four_views": len(trajectory.get("views", [])) == len(visible_sets) == 4,
        "cardinal_bearing_error_at_most_5_deg": (
            len(relation_errors_deg) == 4 and max(relation_errors_deg.values()) <= 5.0
        ),
        "radial_error_at_most_5_cm": (
            len(radial_errors_m) == 4 and max(radial_errors_m.values()) <= 0.05
        ),
        "declared_object_gap_passes": (
            float(selection.get("minimum_declared_object_gap_m", -1.0))
            >= float(selection.get("required_object_gap_m", 0.0))
        ),
        "camera_ring_certified": all(
            bool(item.get("same_room")) and bool(item.get("traversable"))
            for item in selection.get("camera_checks", [])
        ),
    }
    visibility_contract = str(
        selection.get("visibility_contract", "all_five_each_view")
    )
    expected_targets = [
        str(item.get("expected_target_name", ""))
        for item in selection.get("camera_checks", [])
    ]
    model_asset_footprints: list[dict[str, Any]] = []
    if bundle_directory is not None and len(render.get("views", [])) == 4:
        for view, target_name in zip(
            render["views"], expected_targets, strict=True
        ):
            view_id = str(view["view_id"])
            artifact = bundle_directory / "views" / f"{view_id}.sensors.npz"
            if not artifact.is_file():
                continue
            with np.load(artifact, allow_pickle=False) as sensors:
                instance = sensors["instance_id"]
            names = [
                str(anchor_items[0]["name"]) if len(anchor_items) == 1 else "",
                target_name,
            ]
            for role, name in zip(("anchor", "target"), names, strict=True):
                runtime_id = runtime_by_name.get(name)
                ys, xs = (
                    np.where(instance == runtime_id)
                    if runtime_id is not None
                    else (np.asarray([], dtype=int), np.asarray([], dtype=int))
                )
                if len(xs):
                    width = int(np.ptp(xs)) + 1
                    height = int(np.ptp(ys)) + 1
                    touches_border = bool(
                        int(xs.min()) == 0
                        or int(ys.min()) == 0
                        or int(xs.max()) == instance.shape[1] - 1
                        or int(ys.max()) == instance.shape[0] - 1
                    )
                else:
                    width = height = 0
                    touches_border = True
                model_asset_footprints.append(
                    {
                        "view_id": view_id,
                        "role": role,
                        "name": name,
                        "visible_pixels": len(xs),
                        "bbox_width_px": width,
                        "bbox_height_px": height,
                        "touches_image_border": touches_border,
                    }
                )
    if visibility_contract == "all_five_each_view":
        checks.update(
            {
                "every_asset_visible_in_at_least_two_views": (
                    bool(visible_view_count) and min(visible_view_count.values()) >= 2
                ),
                "at_least_three_among_assets_per_view": (
                    len(among_visible_per_view) == 4
                    and min(among_visible_per_view) >= 3
                ),
            }
        )
    elif visibility_contract == "central_plus_one_partial":
        anchor_name = str(anchor_items[0]["name"]) if len(anchor_items) == 1 else ""
        anchor_id = runtime_by_name.get(anchor_name)
        checks.update(
            {
                "anchor_visible_in_all_views": (
                    anchor_id is not None
                    and len(visible_sets) == 4
                    and all(anchor_id in visible for visible in visible_sets)
                ),
                "one_unique_expected_satellite_per_view": (
                    len(expected_targets) == 4
                    and len(set(expected_targets)) == 4
                    and all(name in declared_names for name in expected_targets)
                ),
                "expected_satellite_visible_in_its_view": (
                    len(expected_targets) == len(visible_sets) == 4
                    and all(
                        runtime_by_name.get(target) in visible
                        for target, visible in zip(
                            expected_targets, visible_sets, strict=True
                        )
                    )
                ),
                "each_satellite_has_exactly_one_witness_view": all(
                    visible_view_count.get(name) == 1
                    for name in declared_names
                    if name != anchor_name
                ),
                "each_view_contains_exactly_anchor_and_one_satellite": (
                    among_visible_per_view == [2, 2, 2, 2]
                ),
                "no_single_view_reveals_full_layout": max(
                    among_visible_per_view, default=5
                )
                < 5,
            }
        )
        if bundle_directory is not None:
            anchors = [
                item for item in model_asset_footprints if item["role"] == "anchor"
            ]
            targets = [
                item for item in model_asset_footprints if item["role"] == "target"
            ]
            checks.update(
                {
                    "anchor_is_recognizable_and_uncropped": (
                        len(anchors) == 4
                        and all(
                            item["visible_pixels"] >= 10_000
                            and min(item["bbox_width_px"], item["bbox_height_px"])
                            >= 128
                            and not item["touches_image_border"]
                            for item in anchors
                        )
                    ),
                    "satellite_is_recognizable_and_uncropped": (
                        len(targets) == 4
                        and all(
                            item["visible_pixels"] >= 4_096
                            and min(item["bbox_width_px"], item["bbox_height_px"])
                            >= 96
                            and not item["touches_image_border"]
                            for item in targets
                        )
                    ),
                }
            )
    else:
        checks["known_visibility_contract"] = False
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "declared_asset_names": declared_names,
        "visibility_contract": visibility_contract,
        "expected_target_by_view": expected_targets,
        "visible_view_count_by_asset": visible_view_count,
        "among_asset_count_per_view": among_visible_per_view,
        "cardinal_bearing_error_deg": relation_errors_deg,
        "radial_error_m": radial_errors_m,
        "model_asset_footprints": model_asset_footprints,
    }


def _runtime_id_for_source(snapshot: dict[str, Any], source_id: str) -> int | None:
    registry = {
        str(name): int(identifier)
        for identifier, name in snapshot.get("runtime_instance_registry", {}).items()
        if int(identifier) > 1
    }
    return registry.get(source_id)


def _object_orbit_gate(
    *,
    trajectory: dict[str, Any],
    render: dict[str, Any],
    snapshot: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    focus_id = _runtime_id_for_source(snapshot, str(trajectory["focus_entity_id"]))
    visible_sets = [
        {int(identifier) for identifier in view["visible_runtime_instance_ids"]}
        for view in render["views"]
    ]
    azimuths = [float(value) % 360.0 for value in trajectory["azimuth_deg_per_view"]]
    opposite_pairs = []
    for left in range(len(azimuths)):
        for right in range(left + 1, len(azimuths)):
            delta = abs(((azimuths[right] - azimuths[left] + 180.0) % 360.0) - 180.0)
            if abs(delta - 180.0) <= 30.0:
                opposite_pairs.append([left, right])
    context_counts = [
        len(visible - ({focus_id} if focus_id is not None else set())) for visible in visible_sets
    ]
    complete_orbit = bool(selection.get("complete_orbit", True))
    ordered_azimuths = sorted(azimuths)
    circular_gaps = [
        (ordered_azimuths[(index + 1) % len(ordered_azimuths)] - value) % 360.0
        for index, value in enumerate(ordered_azimuths)
    ]
    measured_arc_coverage_deg = (
        360.0 if complete_orbit else 360.0 - max(circular_gaps)
    )
    minimum_arc_required_deg = float(
        selection.get("minimum_arc_required_deg", 360.0)
    )
    arc_coverage_tolerance_deg = float(
        selection.get("arc_coverage_tolerance_deg", 10.0)
    )
    checks = {
        "focus_resolves_to_runtime_instance": focus_id is not None,
        "focus_visible_in_every_view": focus_id is not None
        and all(focus_id in visible for visible in visible_sets),
        "opposite_view_pair_exists": bool(opposite_pairs),
        "arc_meets_declared_minimum": measured_arc_coverage_deg
        + arc_coverage_tolerance_deg
        >= minimum_arc_required_deg,
        "at_least_three_context_instances_every_view": min(context_counts) >= 3,
        "orientation_questions_disabled_without_front_evidence": (
            trajectory.get("orientation_questions_allowed") is False
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "focus_entity_id": trajectory["focus_entity_id"],
        "focus_runtime_instance_id": focus_id,
        "opposite_view_index_pairs": opposite_pairs,
        "arc_coverage_deg": round(measured_arc_coverage_deg, 6),
        "arc_coverage_tolerance_deg": arc_coverage_tolerance_deg,
        "minimum_arc_required_deg": minimum_arc_required_deg,
        "complete_orbit": complete_orbit,
        "adaptive_radius_used": bool(selection.get("adaptive_radius_used", False)),
        "minimum_context_instance_count": min(context_counts),
    }


def _elevation_gate(*, trajectory: dict[str, Any], render: dict[str, Any]) -> dict[str, Any]:
    heights = [float(value) for value in trajectory["height_sequence_m"]]
    pitches = [float(value) for value in trajectory["pitch_sequence_deg"]]
    positions = [
        tuple(float(value) for value in view["world_from_agent"]["translation_m"][:2])
        for view in trajectory["views"]
    ]
    visible_sets = [
        {int(identifier) for identifier in view["visible_runtime_instance_ids"]}
        for view in render["views"]
    ]
    common = set.intersection(*visible_sets) if visible_sets else set()
    checks = {
        "same_planar_station": len(set(positions)) == 1,
        "height_and_pitch_align_with_views": (len(heights) == len(pitches) == len(render["views"])),
        "low_human_high_height_range": min(heights) <= 0.35
        and any(abs(height - 1.5) <= 0.1 for height in heights)
        and max(heights) >= 2.0,
        "high_view_has_downward_pitch": max(pitches) >= 30.0,
        "at_least_two_cross_height_anchor_instances": len(common) >= 2,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "height_sequence_m": heights,
        "pitch_sequence_deg": pitches,
        "cross_height_anchor_instance_count": len(common),
        "cross_height_anchor_runtime_instance_ids": sorted(common),
    }


def _occlusion_reveal_gate(
    *, trajectory: dict[str, Any], render: dict[str, Any], snapshot: dict[str, Any]
) -> dict[str, Any]:
    annotation = trajectory["occlusion_annotation"]
    target_runtime_id = _runtime_id_for_source(snapshot, str(annotation["target_id"]))
    occluder_runtime_id = _runtime_id_for_source(snapshot, str(annotation["occluder_id"]))
    visible_by_view = {
        str(view["view_id"]): {
            int(identifier) for identifier in view["visible_runtime_instance_ids"]
        }
        for view in render["views"]
    }
    occluded_views = [str(value) for value in annotation["occluded_views"]]
    decisive_view = str(annotation["decisive_view"])
    checks = {
        "target_resolves_to_runtime_instance": target_runtime_id is not None,
        "occluder_resolves_to_runtime_instance": occluder_runtime_id is not None,
        "target_below_visibility_threshold_in_occluded_views": (
            target_runtime_id is not None
            and all(target_runtime_id not in visible_by_view[view_id] for view_id in occluded_views)
        ),
        "occluder_visible_in_occluded_views": (
            occluder_runtime_id is not None
            and all(occluder_runtime_id in visible_by_view[view_id] for view_id in occluded_views)
        ),
        "target_visible_in_decisive_view": (
            target_runtime_id is not None and target_runtime_id in visible_by_view[decisive_view]
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "target_id": annotation["target_id"],
        "target_runtime_instance_id": target_runtime_id,
        "occluder_id": annotation["occluder_id"],
        "occluder_runtime_instance_id": occluder_runtime_id,
        "occluded_views": occluded_views,
        "decisive_view": decisive_view,
    }


def _target_view_gate(
    *, trajectory: dict[str, Any], render: dict[str, Any], snapshot: dict[str, Any]
) -> dict[str, Any]:
    visible_by_view = {
        str(view["view_id"]): {
            int(identifier) for identifier in view["visible_runtime_instance_ids"]
        }
        for view in render["views"]
    }
    anchor_results = []
    for anchor in trajectory["target_anchors"]:
        facing_runtime_id = _runtime_id_for_source(snapshot, str(anchor["facing_entity_id"]))
        anchor_results.append(
            {
                **anchor,
                "facing_runtime_instance_id": facing_runtime_id,
                "facing_visible": facing_runtime_id is not None
                and facing_runtime_id in visible_by_view[str(anchor["view_id"])],
            }
        )
    checks = {
        "trajectory_marked_held_out": trajectory.get("held_out") is True,
        "all_views_use_target_holdout_role": all(
            view.get("role") == "target_holdout" for view in trajectory["views"]
        ),
        "all_facing_anchors_resolve_and_are_visible": all(
            result["facing_visible"] for result in anchor_results
        ),
        "at_least_three_target_views": len(anchor_results) >= 3,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "anchors": anchor_results,
    }


def inspect_bundle(bundle_directory: Path) -> dict[str, Any]:
    """Audit all rendered channels and create ``quality_report.json`` + HTML."""

    bundle_directory = bundle_directory.resolve()
    render_path = bundle_directory / "render_report.json"
    if not render_path.is_file():
        raise FileNotFoundError(f"render report is missing: {render_path}")
    render = json.loads(render_path.read_text(encoding="utf-8"))
    if render.get("status") != "success":
        raise ValueError("render report is not successful")
    contract = render["sensor_contract"]
    trajectory_path = bundle_directory / "trajectory_plan.json"
    trajectory = (
        json.loads(trajectory_path.read_text(encoding="utf-8"))
        if trajectory_path.is_file()
        else {"closed_loop": True, "trajectory_class": "T1"}
    )
    selection_path = bundle_directory / "trajectory_selection.json"
    selection = (
        json.loads(selection_path.read_text(encoding="utf-8"))
        if selection_path.is_file()
        else {}
    )
    expected_shape = (int(contract["height_px"]), int(contract["width_px"]))
    near_m, far_m = float(contract["near_m"]), float(contract["far_m"])
    minimum_pixels = int(contract["minimum_visible_instance_pixels"])
    preview_root = bundle_directory / "preview"
    preview_root.mkdir(exist_ok=True)
    metrics: list[dict[str, Any]] = []
    warnings: list[str] = []
    artifact_paths: list[Path] = []

    for view in render["views"]:
        view_id = str(view["view_id"])
        artifact = view["artifact"]
        path = Path(str(artifact["path"])).resolve()
        try:
            path.relative_to(bundle_directory)
        except ValueError as error:
            raise ValueError(f"view {view_id} is outside its bundle") from error
        if sha256_file(path) != str(artifact["sha256"]):
            raise ValueError(f"view {view_id} digest mismatch")
        artifact_paths.append(path)
        with np.load(path, allow_pickle=False) as arrays:
            required = {"rgb", "depth_m", "instance_id", "semantic_id"}
            if not required.issubset(arrays.files):
                raise ValueError(f"view {view_id} is missing sensor channels")
            rgb = arrays["rgb"]
            depth = arrays["depth_m"]
            instance = arrays["instance_id"]
            semantic = arrays["semantic_id"]
        if rgb.shape != (*expected_shape, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"view {view_id} has invalid RGB contract: {rgb.shape} {rgb.dtype}")
        for name, array, dtype in (
            ("depth", depth, np.float32),
            ("instance", instance, np.uint32),
            ("semantic", semantic, np.uint32),
        ):
            if array.shape != expected_shape or array.dtype != dtype:
                raise ValueError(
                    f"view {view_id} has invalid {name} contract: {array.shape} {array.dtype}"
                )
        valid = np.isfinite(depth) & (depth >= near_m) & (depth <= far_m)
        valid_fraction = float(np.count_nonzero(valid) / depth.size)
        if abs(valid_fraction - float(view["valid_depth_fraction"])) > 1e-6:
            raise ValueError(f"view {view_id} valid-depth fraction mismatch")
        identifiers, counts = np.unique(instance, return_counts=True)
        visible_ids = [
            int(identifier)
            for identifier, count in zip(identifiers, counts, strict=True)
            if int(identifier) > 1 and int(count) >= minimum_pixels
        ]
        if visible_ids != list(view["visible_runtime_semantic_ids"]):
            raise ValueError(f"view {view_id} visible-instance registry mismatch")
        percentiles = np.percentile(rgb, (1.0, 50.0, 99.0))
        item = {
            "view_id": view_id,
            "rgb_mean": round(float(rgb.mean()), 3),
            "rgb_std": round(float(rgb.std()), 3),
            "rgb_p01": round(float(percentiles[0]), 3),
            "rgb_p50": round(float(percentiles[1]), 3),
            "rgb_p99": round(float(percentiles[2]), 3),
            "sharpness_laplacian_variance": round(_sharpness(rgb), 3),
            "valid_depth_fraction": round(valid_fraction, 6),
            "median_valid_depth_m": (
                round(float(np.median(depth[valid])), 4) if np.any(valid) else None
            ),
            "visible_instance_count": len(visible_ids),
            "semantic_label_count": int(np.unique(semantic).size),
        }
        if item["rgb_std"] < 5.0:
            warnings.append(f"{view_id}: near-uniform RGB")
        if item["rgb_p99"] < 20.0:
            warnings.append(f"{view_id}: severe underexposure")
        if item["rgb_p01"] > 235.0:
            warnings.append(f"{view_id}: severe overexposure")
        if valid_fraction < 0.2:
            warnings.append(f"{view_id}: less than 20% valid metric depth")
        card = _write_card(
            preview_root=preview_root,
            view_id=view_id,
            rgb=rgb,
            depth=depth,
            instance=instance,
            near_m=near_m,
            far_m=far_m,
        )
        item["preview"] = str(card.relative_to(bundle_directory))
        metrics.append(item)

    closure = None
    if trajectory.get("closed_loop") and len(artifact_paths) >= 2:
        with (
            np.load(artifact_paths[0], allow_pickle=False) as first,
            np.load(artifact_paths[-1], allow_pickle=False) as last,
        ):
            closure = {
                "first_view_id": str(render["views"][0]["view_id"]),
                "last_view_id": str(render["views"][-1]["view_id"]),
                **_rgb_loop_diagnostics(first["rgb"], last["rgb"]),
                "depth_exact": bool(
                    np.array_equal(first["depth_m"], last["depth_m"], equal_nan=True)
                ),
                "instance_exact": bool(np.array_equal(first["instance_id"], last["instance_id"])),
                "semantic_exact": bool(np.array_equal(first["semantic_id"], last["semantic_id"])),
            }
        if not all(closure[name] for name in ("depth_exact", "instance_exact", "semantic_exact")):
            warnings.append("loop closure: geometric channels are not exact")
        # HQ path tracing is stochastic, so RGB is not expected to close bit for
        # bit. Geometry and labels must be exact; 28 dB is the blocking RGB
        # reproducibility floor. The deterministic Rs_int reference reaches
        # 31.6 dB, while visually sound larger scenes can sit near 29 dB.
        if closure["rgb_psnr_db"] is not None and closure["rgb_psnr_db"] < 28.0:
            warnings.append("loop closure: RGB PSNR is below 28 dB")

    gates: dict[str, Any] = {}
    trajectory_class = trajectory.get("trajectory_class", "T1")
    if trajectory_class == "T2" and selection.get("sampling_strategy") == "procedural_among5":
        snapshot = json.loads(
            (bundle_directory / "scene_snapshot.json").read_text(encoding="utf-8")
        )
        gates["T2"] = _among5_gate(
            trajectory=trajectory,
            render=render,
            snapshot=snapshot,
            selection=selection,
            bundle_directory=bundle_directory,
        )
        if gates["T2"]["status"] != "pass":
            warnings.append("T2 procedural Among-5 evidence gate failed")
    elif trajectory_class == "T3":
        gates["T3"] = _rotation_station_gate(
            bundle_directory=bundle_directory,
            trajectory=trajectory,
            render=render,
        )
        if gates["T3"]["status"] != "pass":
            warnings.append("T3 rotation-station evidence gate failed")
    elif trajectory_class in {"T4", "T7", "T8", "T10"}:
        snapshot = json.loads(
            (bundle_directory / "scene_snapshot.json").read_text(encoding="utf-8")
        )
        if trajectory_class == "T4":
            gates["T4"] = _object_orbit_gate(
                trajectory=trajectory,
                render=render,
                snapshot=snapshot,
                selection=selection,
            )
        elif trajectory_class == "T7":
            gates["T7"] = _elevation_gate(trajectory=trajectory, render=render)
        elif trajectory_class == "T8":
            gates["T8"] = _occlusion_reveal_gate(
                trajectory=trajectory, render=render, snapshot=snapshot
            )
        else:
            gates["T10"] = _target_view_gate(
                trajectory=trajectory, render=render, snapshot=snapshot
            )
        if gates[trajectory_class]["status"] != "pass":
            warnings.append(f"{trajectory_class} evidence gate failed")

    sharpness_values = [float(item["sharpness_laplacian_variance"]) for item in metrics]
    report = {
        "protocol_version": "omnigibson_episode_quality.v1",
        "integrity_status": "pass",
        "visual_status": "pass" if not warnings else "warning",
        "trajectory_status": (
            "pass" if all(gate["status"] == "pass" for gate in gates.values()) else "fail"
        ),
        "trajectory_class": trajectory_class,
        "view_count": len(metrics),
        "warnings": warnings,
        "loop_closure": closure,
        "gates": gates,
        "summary": {
            "median_rgb_std": round(float(np.median([item["rgb_std"] for item in metrics])), 3),
            "median_sharpness_laplacian_variance": round(float(np.median(sharpness_values)), 3),
            "minimum_valid_depth_fraction": round(
                min(float(item["valid_depth_fraction"]) for item in metrics), 6
            ),
            "minimum_visible_instance_count": min(
                int(item["visible_instance_count"]) for item in metrics
            ),
        },
        "views": metrics,
    }
    write_json_atomic(bundle_directory / "quality_report.json", report)
    card_items = []
    for item in metrics:
        source = html.escape(item["preview"])
        alternative = html.escape(item["view_id"])
        metadata = html.escape(json.dumps(item, indent=2))
        card_items.append(
            f'<article><img src="{source}" alt="{alternative}"><pre>{metadata}</pre></article>'
        )
    cards = "\n".join(card_items)
    summary_text = html.escape(
        json.dumps({"summary": report["summary"], "loop_closure": closure}, indent=2)
    )
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OmniGibson episode preview</title>
<style>
body{{margin:0;padding:24px;background:#0b1020;color:#e5e7eb;font:14px system-ui}}
h1{{margin-top:0}}
article{{background:#111827;border:1px solid #374151;border-radius:10px;
margin:18px 0;overflow:hidden}}
img{{display:block;width:100%;height:auto}}
pre{{white-space:pre-wrap;padding:12px;margin:0;color:#cbd5e1}}
</style>
<h1>OmniGibson episode preview</h1>
<p>Integrity: {report["integrity_status"]} · visual: {report["visual_status"]}
· views: {len(metrics)}</p>
<pre>{summary_text}</pre>
{cards}
</html>
"""
    (bundle_directory / "preview.html").write_text(page, encoding="utf-8")
    return report
