"""Build a deterministic, release-bound packet for independent semantic visual QA.

This module deliberately runs *after* compilation.  It does not admit, reject,
or relabel corpus examples.  Oracle instance masks are used only to prioritize
examples near the frozen recognizability boundary; every selected row remains
unreviewed until an independent reviewer fills transparent provenance fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from episode3d.bundles import Bundle
from episode3d.compilers import STRUCTURAL_LABELS

PACKET_SCHEMA = "epispace.semantic_visual_audit_packet.v1"
RELEASE_SCHEMA = "epispace.release_manifest.v1"
FINAL_INDEX_SCHEMA = "epispace.final_release_index.v1"
DEFAULT_SAMPLE_SIZE = 48
DEFAULT_SEED = 20270717
DEFAULT_RISK_MULTIPLIER = 1.5
NEAR_THRESHOLD_SCORE = 0.5


class SemanticVisualAuditError(ValueError):
    """Raised when an audit packet cannot be bound to trustworthy inputs."""


@dataclass(frozen=True)
class _Candidate:
    episode: Mapping[str, Any]
    question: Mapping[str, Any]
    program_id: str
    stratum: tuple[str, str, str, str]
    risk_score: float
    near_threshold: bool
    screening: Mapping[str, Any]

    @property
    def fact_id(self) -> str:
        return str(self.question["fact_id"])


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SemanticVisualAuditError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticVisualAuditError(f"cannot read valid JSON: {path}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SemanticVisualAuditError(f"cannot read JSONL: {path}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SemanticVisualAuditError(
                f"invalid JSONL at {path}:{line_number}"
            ) from exc
        _require(
            isinstance(value, dict),
            f"JSONL row must be an object at {path}:{line_number}",
        )
        rows.append(value)
    return rows


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _release_path(release_dir: Path, value: Any, label: str) -> Path:
    _require(isinstance(value, str) and value, f"{label} must be a relative path")
    raw = Path(value)
    _require(not raw.is_absolute(), f"{label} must be relative")
    result = (release_dir / raw).resolve()
    try:
        result.relative_to(release_dir.resolve())
    except ValueError as exc:
        raise SemanticVisualAuditError(f"{label} escapes the release directory") from exc
    return result


def _load_release(
    release_dir: Path,
) -> tuple[
    dict[str, Any],
    Path,
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    release_dir = release_dir.expanduser().resolve()
    manifest_path = release_dir / "release_manifest.json"
    _require(manifest_path.is_file(), f"release manifest is absent: {manifest_path}")
    manifest = _read_json(manifest_path)
    _require(manifest.get("schema_version") == RELEASE_SCHEMA, "unsupported release manifest")
    _require(manifest.get("status") == "pass", "candidate release manifest is not passing")
    corpus_gates = _mapping(manifest.get("corpus_gates"), "manifest.corpus_gates")
    _require(corpus_gates.get("status") == "pass", "candidate corpus gates are not passing")
    gate_checks = _mapping(corpus_gates.get("checks"), "manifest.corpus_gates.checks")
    _require(
        bool(gate_checks) and all(value is True for value in gate_checks.values()),
        "candidate release contains a failed or non-boolean corpus gate",
    )

    artifacts = _mapping(manifest.get("artifacts"), "manifest.artifacts")
    integrity = _mapping(manifest.get("artifact_integrity"), "manifest.artifact_integrity")
    _require("episode_ir" in artifacts, "release does not declare episode_ir")
    ir_record = _mapping(integrity.get("episode_ir"), "artifact_integrity.episode_ir")
    _require(ir_record.get("path") == artifacts["episode_ir"], "episode_ir paths disagree")
    ir_path = _release_path(release_dir, artifacts["episode_ir"], "episode_ir")
    _require(ir_path.is_file(), f"episode_ir is absent: {ir_path}")
    ir_sha = _sha256(ir_path)
    _require(ir_sha == ir_record.get("sha256"), "episode_ir SHA does not match manifest")
    if ir_record.get("bytes") is not None:
        _require(ir_path.stat().st_size == ir_record["bytes"], "episode_ir byte count mismatch")
    episodes = _read_jsonl(ir_path)
    expected_records = ir_record.get("records")
    if expected_records is not None:
        _require(expected_records == len(episodes), "episode_ir record count does not match manifest")

    _require("source_inventory" in artifacts, "release does not declare source_inventory")
    inventory_record = _mapping(
        integrity.get("source_inventory"), "artifact_integrity.source_inventory"
    )
    _require(
        inventory_record.get("path") == artifacts["source_inventory"],
        "source_inventory paths disagree",
    )
    inventory_path = _release_path(
        release_dir, artifacts["source_inventory"], "source_inventory"
    )
    _require(inventory_path.is_file(), f"source_inventory is absent: {inventory_path}")
    inventory_sha = _sha256(inventory_path)
    _require(
        inventory_sha == inventory_record.get("sha256"),
        "source_inventory SHA does not match manifest",
    )
    if inventory_record.get("bytes") is not None:
        _require(
            inventory_path.stat().st_size == inventory_record["bytes"],
            "source_inventory byte count mismatch",
        )
    source_inventory = _read_json(inventory_path)
    _require(
        source_inventory.get("schema_version") == "epispace.source_inventory.v1",
        "unsupported source inventory",
    )

    binding: dict[str, Any] = {
        "release_dir": str(release_dir),
        "dataset_id": manifest.get("dataset_id"),
        "release_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "episode_ir": {
            "path": str(ir_path),
            "sha256": ir_sha,
            "records": len(episodes),
        },
        "source_inventory": {
            "path": str(inventory_path),
            "sha256": inventory_sha,
        },
    }
    final_path = release_dir / "final_release_index.json"
    if final_path.is_file():
        final_index = _read_json(final_path)
        _require(
            final_index.get("schema_version") == FINAL_INDEX_SCHEMA,
            "unsupported final release index",
        )
        _require(final_index.get("status") == "pass", "final release index is not passing")
        _require(
            final_index.get("release_manifest_sha256")
            == binding["release_manifest"]["sha256"],
            "final release index is bound to a different manifest",
        )
        indexed_ir = _mapping(
            _mapping(final_index.get("base_artifacts"), "final_index.base_artifacts").get(
                "episode_ir"
            ),
            "final_index.base_artifacts.episode_ir",
        )
        _require(indexed_ir.get("sha256") == ir_sha, "final index episode_ir SHA mismatch")
        binding["final_release_index"] = {
            "path": str(final_path),
            "sha256": _sha256(final_path),
        }
    else:
        binding["final_release_index"] = None
    return manifest, ir_path, episodes, binding, source_inventory


def _inventory_view_map(
    source_inventory: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    raw_bundles = source_inventory.get("bundles")
    _require(isinstance(raw_bundles, list), "source inventory bundles must be a list")
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw_bundle in raw_bundles:
        bundle = _mapping(raw_bundle, "source inventory bundle")
        bundle_root = bundle.get("bundle_root")
        _require(isinstance(bundle_root, str) and bundle_root, "inventory bundle_root is absent")
        views = bundle.get("views")
        _require(isinstance(views, list), "source inventory views must be a list")
        for raw_view in views:
            view = _mapping(raw_view, "source inventory view")
            view_id = view.get("view_id")
            _require(isinstance(view_id, str) and view_id, "inventory view_id is absent")
            key = (str(Path(bundle_root).expanduser().resolve()), view_id)
            _require(key not in result, f"duplicate source inventory view: {key}")
            result[key] = view
    _require(result, "source inventory contains no views")
    return result


def _verify_inventory_file(
    record: Mapping[str, Any],
    *,
    path_key: str,
    sha_key: str,
    bytes_key: str,
    expected_path: Path,
    verified_hashes: dict[Path, str],
) -> str:
    declared_path = record.get(path_key)
    expected_sha = record.get(sha_key)
    expected_bytes = record.get(bytes_key)
    _require(
        isinstance(declared_path, str) and Path(declared_path).expanduser().resolve() == expected_path,
        f"source inventory path mismatch for {expected_path}",
    )
    _require(isinstance(expected_sha, str) and expected_sha, f"missing {sha_key}")
    _require(
        isinstance(expected_bytes, int)
        and not isinstance(expected_bytes, bool)
        and expected_bytes >= 0,
        f"invalid {bytes_key}",
    )
    _require(expected_path.is_file(), f"release evidence file is absent: {expected_path}")
    _require(expected_path.stat().st_size == expected_bytes, f"byte count changed: {expected_path}")
    if expected_path not in verified_hashes:
        verified_hashes[expected_path] = _sha256(expected_path)
    actual_sha = verified_hashes[expected_path]
    _require(actual_sha == expected_sha, f"SHA changed since source inventory: {expected_path}")
    return actual_sha


def _release_visual_quality_policy(manifest: Mapping[str, Any]) -> dict[str, Any]:
    research_contract = _mapping(manifest.get("research_contract"), "research_contract")
    policy = dict(
        _mapping(
            research_contract.get("visual_quality_policy"),
            "research_contract.visual_quality_policy",
        )
    )
    required_numeric = (
        "minimum_visible_pixels",
        "minimum_bbox_side_px",
        "minimum_mask_bbox_fill_ratio",
        "severe_multi_border_outer_band_pixel_fraction",
        "severe_single_border_outer_band_pixel_fraction",
        "maximum_severely_clipped_axis_fraction",
        "maximum_rgb_dominant_color_fraction",
        "minimum_rgb_quantized_entropy_bits",
        "maximum_near_nonstructural_pixel_fraction",
        "maximum_near_any_geometry_pixel_fraction",
        "maximum_close_geometry_pixel_fraction",
        "minimum_near_enclosure_pixel_fraction",
        "maximum_near_enclosure_median_depth_m",
        "maximum_near_enclosure_depth_spread_m",
        "minimum_dominant_foreground_pixel_fraction",
        "maximum_dominant_foreground_median_depth_m",
        "minimum_raw_negative_category_pixels",
        "minimum_raw_negative_category_bbox_side_px",
        "minimum_t10_anchor_pixels",
        "minimum_t10_anchor_bbox_side_px",
        "maximum_near_enclosure_median_depth_m",
        "maximum_near_enclosure_depth_spread_m",
        "maximum_dominant_foreground_median_depth_m",
        "maximum_t10_anchor_outer_band_pixel_fraction",
        "minimum_relation_dominance_ratio",
    )
    _require(
        isinstance(policy.get("policy_id"), str) and bool(policy["policy_id"]),
        "release visual quality policy has no policy_id",
    )
    for key in required_numeric:
        value = policy.get(key)
        _require(
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value)),
            f"release visual quality policy has invalid {key}",
        )
    positive_keys = (
        "minimum_visible_pixels",
        "minimum_bbox_side_px",
        "minimum_rgb_quantized_entropy_bits",
        "minimum_raw_negative_category_pixels",
        "minimum_raw_negative_category_bbox_side_px",
        "minimum_t10_anchor_pixels",
        "minimum_t10_anchor_bbox_side_px",
    )
    for key in positive_keys:
        _require(float(policy[key]) > 0, f"release visual quality policy {key} must be positive")
    fraction_keys = (
        "minimum_mask_bbox_fill_ratio",
        "severe_multi_border_outer_band_pixel_fraction",
        "severe_single_border_outer_band_pixel_fraction",
        "maximum_severely_clipped_axis_fraction",
        "maximum_rgb_dominant_color_fraction",
        "maximum_near_nonstructural_pixel_fraction",
        "maximum_near_any_geometry_pixel_fraction",
        "maximum_close_geometry_pixel_fraction",
        "minimum_near_enclosure_pixel_fraction",
        "minimum_dominant_foreground_pixel_fraction",
        "maximum_t10_anchor_outer_band_pixel_fraction",
    )
    for key in fraction_keys:
        _require(
            0.0 < float(policy[key]) <= 1.0,
            f"release visual quality policy {key} must be in (0, 1]",
        )
    _require(
        float(policy["minimum_relation_dominance_ratio"]) >= 1.0,
        "release visual quality policy minimum_relation_dominance_ratio must be at least 1",
    )
    category_limits = _mapping(
        policy.get("category_max_height_m"), "category_max_height_m"
    )
    _require(
        all(
            isinstance(label, str)
            and bool(label)
            and isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0
            for label, value in category_limits.items()
        ),
        "category_max_height_m contains an invalid constraint",
    )
    confusable_groups = policy.get("visually_confusable_category_groups")
    _require(
        isinstance(confusable_groups, list)
        and all(
            isinstance(group, list)
            and len(group) >= 2
            and len(set(group)) == len(group)
            and all(isinstance(label, str) and bool(label) for label in group)
            for group in confusable_groups
        ),
        "visually_confusable_category_groups is invalid",
    )
    required_borders = policy.get("dominant_foreground_required_border_sides")
    _require(
        isinstance(required_borders, list)
        and set(required_borders) == {"left", "right", "top"}
        and len(required_borders) == 3,
        "dominant_foreground_required_border_sides is invalid",
    )
    return policy


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _visual_signal(
    stats: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    rgb_stats: Mapping[str, Any],
    entity_label: str,
    entity_height_m: float,
) -> dict[str, Any]:
    pixels = int(stats["visible_pixels"])
    width = int(stats["bbox_width_px"])
    height = int(stats["bbox_height_px"])
    min_side = min(width, height)
    fill = float(stats["mask_bbox_fill_ratio"])
    outer = float(stats["outer_five_percent_pixel_fraction"])
    image_width = max(int(stats["image_width_px"]), 1)
    image_height = max(int(stats["image_height_px"]), 1)
    clipped_axis_fraction = min(width / image_width, height / image_height)

    components: list[dict[str, Any]] = []
    minimum_pixels = int(policy["minimum_visible_pixels"])
    pixel_score = _clamp01(1.0 - (pixels - minimum_pixels) / (2.0 * minimum_pixels))
    if pixel_score > 0:
        components.append(
            {
                "code": "visible_pixels_near_minimum",
                "score": round(pixel_score, 6),
                "measured": pixels,
                "admission_threshold": minimum_pixels,
            }
        )

    minimum_side = int(policy["minimum_bbox_side_px"])
    bbox_score = _clamp01(1.0 - (min_side - minimum_side) / (2.0 * minimum_side))
    if bbox_score > 0:
        components.append(
            {
                "code": "bbox_side_near_minimum",
                "score": round(bbox_score, 6),
                "measured": min_side,
                "admission_threshold": minimum_side,
            }
        )

    minimum_fill = float(policy["minimum_mask_bbox_fill_ratio"])
    fill_score = _clamp01(1.0 - (fill - minimum_fill) / (2.0 * minimum_fill))
    if fill_score > 0:
        components.append(
            {
                "code": "mask_fill_near_minimum",
                "score": round(fill_score, 6),
                "measured": round(fill, 6),
                "admission_threshold": minimum_fill,
            }
        )

    if bool(stats["touches_image_border"]):
        multi_border = len(stats["border_sides"]) >= 2
        severe_outer = float(
            policy[
                "severe_multi_border_outer_band_pixel_fraction"
                if multi_border
                else "severe_single_border_outer_band_pixel_fraction"
            ]
        )
        severe_axis = float(policy["maximum_severely_clipped_axis_fraction"])
        outer_score = _clamp01(
            (outer - severe_outer * 0.5) / max(severe_outer * 0.5, 1e-9)
        )
        axis_score = _clamp01((2.0 * severe_axis - clipped_axis_fraction) / severe_axis)
        vertical_full_span = (
            len(stats["border_sides"]) >= 3
            and {"top", "bottom"}.issubset(set(stats["border_sides"]))
        )
        crop_score = (
            1.0
            if vertical_full_span
            else (outer_score if multi_border else min(outer_score, axis_score))
        )
        if crop_score > 0:
            components.append(
                {
                    "code": "border_crop_near_severe_boundary",
                    "score": round(crop_score, 6),
                    "measured": {
                        "outer_five_percent_pixel_fraction": round(outer, 6),
                        "clipped_axis_fraction": round(clipped_axis_fraction, 6),
                        "border_sides": list(stats["border_sides"]),
                    },
                    "admission_threshold": {
                        "severe_outer_band_pixel_fraction_for_border_count": severe_outer,
                        "maximum_severely_clipped_axis_fraction": severe_axis,
                    },
                }
            )
    dominant = float(rgb_stats["dominant_quantized_color_fraction"])
    entropy = float(rgb_stats["quantized_color_entropy_bits"])
    maximum_dominant = float(policy["maximum_rgb_dominant_color_fraction"])
    minimum_entropy = float(policy["minimum_rgb_quantized_entropy_bits"])
    dominant_score = _clamp01(
        (dominant - maximum_dominant * 0.5) / (maximum_dominant * 0.5)
    )
    entropy_score = _clamp01(1.0 - (entropy - minimum_entropy) / 2.0)
    rgb_score = min(dominant_score, entropy_score)
    if rgb_score > 0:
        components.append(
            {
                "code": "rgb_degeneration_near_boundary",
                "score": round(rgb_score, 6),
                "measured": {
                    "dominant_quantized_color_fraction": round(dominant, 6),
                    "quantized_color_entropy_bits": round(entropy, 6),
                },
                "admission_threshold": {
                    "maximum_rgb_dominant_color_fraction": maximum_dominant,
                    "minimum_rgb_quantized_entropy_bits": minimum_entropy,
                },
            }
        )
    maximum_height = _mapping(
        policy["category_max_height_m"], "category_max_height_m"
    ).get(entity_label)
    if maximum_height is not None:
        height_score = _clamp01(
            1.0 - (float(maximum_height) - entity_height_m) / (0.25 * float(maximum_height))
        )
        if height_score > 0:
            components.append(
                {
                    "code": "category_height_near_maximum",
                    "score": round(height_score, 6),
                    "measured": round(entity_height_m, 6),
                    "admission_threshold": float(maximum_height),
                    "category": entity_label,
                }
            )
    score = max((float(item["score"]) for item in components), default=0.0)
    return {
        "score": round(score, 6),
        "measurements": {
            "visible_pixels": pixels,
            "bbox_width_px": width,
            "bbox_height_px": height,
            "mask_bbox_fill_ratio": round(fill, 6),
            "outer_five_percent_pixel_fraction": round(outer, 6),
            "touches_image_border": bool(stats["touches_image_border"]),
            "border_sides": list(stats["border_sides"]),
            "dominant_quantized_color_fraction": round(dominant, 6),
            "quantized_color_entropy_bits": round(entropy, 6),
            "entity_label": entity_label,
            "entity_height_m": round(entity_height_m, 6),
        },
        "components": components,
    }


def _relation_ratios(certificate: Any) -> list[float]:
    ratios: list[float] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            if value.get("name") == "relation_dominance_ratio":
                measured = value.get("measured_value", value.get("measured"))
                if isinstance(measured, int | float) and not isinstance(measured, bool):
                    ratios.append(float(measured))
            direct = value.get("dominance_ratio")
            if isinstance(direct, int | float) and not isinstance(direct, bool):
                ratios.append(float(direct))
            for right_key, front_key in (
                ("ego_right_delta_m", "ego_front_delta_m"),
                ("canonical_right_delta_m", "canonical_front_delta_m"),
                ("query_right_m", "query_front_m"),
            ):
                right = value.get(right_key)
                front = value.get(front_key)
                if (
                    isinstance(right, int | float)
                    and not isinstance(right, bool)
                    and isinstance(front, int | float)
                    and not isinstance(front, bool)
                ):
                    dominant = max(abs(float(right)), abs(float(front)))
                    minor = min(abs(float(right)), abs(float(front)))
                    if minor > 0:
                        ratios.append(dominant / minor)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(certificate)
    return [value for value in ratios if math.isfinite(value)]


def _frame_rgb_signal(
    rgb_stats: Mapping[str, Any],
    collision_stats: Mapping[str, Any],
    context_stats: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Score one actual model input frame near any frozen frame gate."""

    dominant = float(rgb_stats["dominant_quantized_color_fraction"])
    entropy = float(rgb_stats["quantized_color_entropy_bits"])
    maximum_dominant = float(policy["maximum_rgb_dominant_color_fraction"])
    minimum_entropy = float(policy["minimum_rgb_quantized_entropy_bits"])
    rgb_degenerate = dominant >= maximum_dominant and entropy < minimum_entropy
    near_fraction = float(collision_stats["near_nonstructural_pixel_fraction"])
    maximum_near = float(policy["maximum_near_nonstructural_pixel_fraction"])
    near_any_fraction = float(collision_stats["near_geometry_pixel_fraction"])
    maximum_near_any = float(policy["maximum_near_any_geometry_pixel_fraction"])
    legacy_camera_collision = (
        near_fraction >= maximum_near or near_any_fraction >= maximum_near_any
    )
    close_fraction = float(context_stats["close_geometry_pixel_fraction"])
    maximum_close = float(policy["maximum_close_geometry_pixel_fraction"])
    enclosure_fraction = float(context_stats["enclosure_pixel_fraction"])
    minimum_enclosure = float(policy["minimum_near_enclosure_pixel_fraction"])
    depth_median = float(context_stats["depth_median_m"])
    maximum_enclosure_depth = float(
        policy["maximum_near_enclosure_median_depth_m"]
    )
    depth_spread = float(context_stats["depth_p90_minus_p10_m"])
    maximum_enclosure_spread = float(
        policy["maximum_near_enclosure_depth_spread_m"]
    )
    foreground = _mapping(
        context_stats["dominant_nonstructural_instance"],
        "dominant_nonstructural_instance",
    )
    foreground_fraction = float(foreground["pixel_fraction"])
    minimum_foreground = float(
        policy["minimum_dominant_foreground_pixel_fraction"]
    )
    foreground_depth_raw = foreground.get("median_depth_m")
    foreground_depth = (
        float(foreground_depth_raw)
        if isinstance(foreground_depth_raw, int | float)
        and not isinstance(foreground_depth_raw, bool)
        else math.inf
    )
    maximum_foreground_depth = float(
        policy["maximum_dominant_foreground_median_depth_m"]
    )
    required_borders = set(policy["dominant_foreground_required_border_sides"])
    near_surface_saturation = close_fraction >= maximum_close
    near_enclosure = (
        enclosure_fraction >= minimum_enclosure
        and depth_median <= maximum_enclosure_depth
        and depth_spread <= maximum_enclosure_spread
    )
    foreground_occlusion = (
        foreground_fraction >= minimum_foreground
        and required_borders.issubset(set(foreground["border_sides"]))
        and foreground_depth <= maximum_foreground_depth
    )
    camera_collision = (
        legacy_camera_collision
        or near_surface_saturation
        or near_enclosure
        or foreground_occlusion
    )
    dominant_score = _clamp01(
        (dominant - maximum_dominant * 0.5) / (maximum_dominant * 0.5)
    )
    entropy_score = _clamp01(1.0 - (entropy - minimum_entropy) / 2.0)
    rgb_score = min(dominant_score, entropy_score)
    collision_score = _clamp01(
        (near_fraction - maximum_near * 0.5) / max(maximum_near * 0.5, 1e-9)
    )
    collision_score = max(
        collision_score,
        _clamp01(
            (near_any_fraction - maximum_near_any * 0.5)
            / max(maximum_near_any * 0.5, 1e-9)
        ),
    )
    close_score = _clamp01(
        (close_fraction - maximum_close * 0.5) / max(maximum_close * 0.5, 1e-9)
    )
    enclosure_score = min(
        _clamp01(
            (enclosure_fraction - minimum_enclosure * 0.5)
            / max(minimum_enclosure * 0.5, 1e-9)
        ),
        _clamp01(
            (2.0 * maximum_enclosure_depth - depth_median)
            / max(maximum_enclosure_depth, 1e-9)
        ),
        _clamp01(
            (2.0 * maximum_enclosure_spread - depth_spread)
            / max(maximum_enclosure_spread, 1e-9)
        ),
    )
    foreground_score = min(
        _clamp01(
            (foreground_fraction - minimum_foreground * 0.5)
            / max(minimum_foreground * 0.5, 1e-9)
        ),
        _clamp01(
            (2.0 * maximum_foreground_depth - foreground_depth)
            / max(maximum_foreground_depth, 1e-9)
        ),
        float(required_borders.issubset(set(foreground["border_sides"]))),
    )
    hard_degenerate = rgb_degenerate or camera_collision
    if foreground_occlusion:
        code = "foreground_occlusion"
    elif near_enclosure:
        code = "near_enclosure"
    elif near_surface_saturation:
        code = "near_surface_saturation"
    elif legacy_camera_collision:
        code = "camera_collision"
    elif rgb_degenerate:
        code = "rgb_frame_hard_degenerate"
    else:
        code = "frame_near_admission_boundary"
    return {
        "code": code,
        "score": round(
            max(
                rgb_score,
                collision_score,
                close_score,
                enclosure_score,
                foreground_score,
            ),
            6,
        ),
        "hard_degenerate": hard_degenerate,
        "measured": {
            "dominant_quantized_color_fraction": round(dominant, 6),
            "quantized_color_entropy_bits": round(entropy, 6),
            "near_nonstructural_pixel_fraction": round(near_fraction, 6),
            "near_geometry_pixel_fraction": round(near_any_fraction, 6),
            "close_geometry_pixel_fraction": round(close_fraction, 6),
            "enclosure_pixel_fraction": round(enclosure_fraction, 6),
            "depth_median_m": round(depth_median, 6),
            "depth_p90_minus_p10_m": round(depth_spread, 6),
            "dominant_nonstructural_pixel_fraction": round(
                foreground_fraction, 6
            ),
            "dominant_nonstructural_border_sides": list(
                foreground["border_sides"]
            ),
            "dominant_nonstructural_median_depth_m": (
                round(foreground_depth, 6) if math.isfinite(foreground_depth) else None
            ),
        },
        "admission_threshold": {
            "maximum_rgb_dominant_color_fraction": maximum_dominant,
            "minimum_rgb_quantized_entropy_bits": minimum_entropy,
            "maximum_near_nonstructural_pixel_fraction": maximum_near,
            "maximum_near_any_geometry_pixel_fraction": maximum_near_any,
            "maximum_close_geometry_pixel_fraction": maximum_close,
            "minimum_near_enclosure_pixel_fraction": minimum_enclosure,
            "maximum_near_enclosure_median_depth_m": maximum_enclosure_depth,
            "maximum_near_enclosure_depth_spread_m": maximum_enclosure_spread,
            "minimum_dominant_foreground_pixel_fraction": minimum_foreground,
            "maximum_dominant_foreground_median_depth_m": (
                maximum_foreground_depth
            ),
            "dominant_foreground_required_border_sides": sorted(required_borders),
        },
    }


def _mask_passes_release_policy(
    bundle: Bundle,
    entity_id: str,
    view_id: str,
    policy: Mapping[str, Any],
) -> bool:
    """Replay recognizability from the release-bound policy, not stored flags."""

    stats = bundle.instance_visual_stats(entity_id, view_id)
    width = int(stats["bbox_width_px"])
    height = int(stats["bbox_height_px"])
    clipped_axis_fraction = min(
        width / max(int(stats["image_width_px"]), 1),
        height / max(int(stats["image_height_px"]), 1),
    )
    border_sides = set(stats["border_sides"])
    border_count = len(border_sides)
    outer_fraction = float(stats["outer_five_percent_pixel_fraction"])
    severe_crop = (
        (border_count >= 3 and {"top", "bottom"}.issubset(border_sides))
        or (
            border_count >= 2
            and outer_fraction
            >= float(policy["severe_multi_border_outer_band_pixel_fraction"])
        )
        or (
            bool(stats["touches_image_border"])
            and outer_fraction
            >= float(policy["severe_single_border_outer_band_pixel_fraction"])
            and clipped_axis_fraction
            < float(policy["maximum_severely_clipped_axis_fraction"])
        )
    )
    entity = bundle.entity(entity_id)
    maximum_height = _mapping(
        policy["category_max_height_m"], "category_max_height_m"
    ).get(entity.label)
    return (
        int(stats["visible_pixels"]) >= int(policy["minimum_visible_pixels"])
        and min(width, height) >= int(policy["minimum_bbox_side_px"])
        and float(stats["mask_bbox_fill_ratio"])
        >= float(policy["minimum_mask_bbox_fill_ratio"])
        and not severe_crop
        and (
            maximum_height is None
            or float(entity.extent_m[2]) <= float(maximum_height)
        )
    )


def _recognizable_entities_for_release_policy(
    bundle: Bundle,
    view_id: str,
    policy: Mapping[str, Any],
) -> set[str]:
    return {
        entity_id
        for entity_id in bundle.view_by_id[view_id].visible_entity_ids
        if entity_id in bundle.entities
        and bundle.entities[entity_id].label not in STRUCTURAL_LABELS
        and _mask_passes_release_policy(bundle, entity_id, view_id, policy)
    }


def _screen_candidate(
    episode: Mapping[str, Any],
    question: Mapping[str, Any],
    *,
    bundle: Bundle,
    policy: Mapping[str, Any],
    inventory_views: Mapping[tuple[str, str], Mapping[str, Any]],
    verified_hashes: dict[Path, str],
) -> tuple[float, bool, dict[str, Any]]:
    observations = episode.get("observations")
    _require(isinstance(observations, list) and observations, "episode has no observations")
    all_view_ids = tuple(str(item["view_id"]) for item in observations)
    raw_model_views = question.get("model_view_ids")
    model_view_ids = (
        tuple(str(value) for value in raw_model_views)
        if isinstance(raw_model_views, list)
        else all_view_ids
    )
    screening_view_ids = model_view_ids
    raw_entities = question.get("evidence_entity_ids")
    entity_ids = (
        tuple(str(value) for value in raw_entities) if isinstance(raw_entities, list) else ()
    )
    visual_signals: list[dict[str, Any]] = []
    frame_signals: list[dict[str, Any]] = []
    screening_sources: list[dict[str, Any]] = []
    for view_id in screening_view_ids:
        _require(view_id in bundle.view_by_id, f"unknown model view {view_id}")
        inventory_key = (str(bundle.root), view_id)
        _require(
            inventory_key in inventory_views,
            f"source inventory has no screening view: {inventory_key}",
        )
        sensor_path = bundle.view_by_id[view_id].sensor_path.resolve()
        sensor_sha = _verify_inventory_file(
            inventory_views[inventory_key],
            path_key="source_sensor",
            sha_key="source_sensor_sha256",
            bytes_key="source_sensor_bytes",
            expected_path=sensor_path,
            verified_hashes=verified_hashes,
        )
        screening_sources.append(
            {"view_id": view_id, "path": str(sensor_path), "sha256": sensor_sha}
        )
        frame_signal = _frame_rgb_signal(
            bundle.rgb_visual_stats(view_id),
            bundle.frame_collision_stats(view_id),
            bundle.frame_context_stats(view_id),
            policy,
        )
        _require(
            not frame_signal["hard_degenerate"],
            (
                f"fact {question.get('fact_id')} contains an inadmissible "
                f"actual model input frame: {view_id}"
            ),
        )
        if frame_signal["score"] > 0:
            frame_signals.append({"view_id": view_id, **frame_signal})
    if question.get("task_type") == "rotation_change_detection":
        evidence_views = tuple(str(value) for value in question.get("evidence_view_ids", ()))
        _require(
            len(evidence_views) == 2,
            f"rotation fact {question.get('fact_id')} has no before/after pair",
        )
        before_id, after_id = evidence_views
        entered = _recognizable_entities_for_release_policy(
            bundle, after_id, policy
        ) - _recognizable_entities_for_release_policy(bundle, before_id, policy)
        _require(
            entered == set(entity_ids),
            (
                f"rotation fact {question.get('fact_id')} does not have one claimed "
                f"recognizable entrant: entered={sorted(entered)}"
            ),
        )
    for raw_entity_id in entity_ids:
        try:
            entity_id = bundle.entity(raw_entity_id).entity_id
        except ValueError as exc:
            raise SemanticVisualAuditError(
                f"fact {question.get('fact_id')} references unknown entity {raw_entity_id}"
            ) from exc
        for view_id in screening_view_ids:
            stats = bundle.instance_visual_stats(entity_id, view_id)
            if int(stats["visible_pixels"]) <= 0:
                continue
            entity = bundle.entity(entity_id)
            signal = _visual_signal(
                stats,
                policy,
                rgb_stats=bundle.rgb_visual_stats(view_id),
                entity_label=entity.label,
                entity_height_m=float(entity.extent_m[2]),
            )
            if signal["score"] > 0:
                visual_signals.append(
                    {
                        "entity_id": entity_id,
                        "view_id": view_id,
                        **signal,
                    }
                )

    relation_signals: list[dict[str, Any]] = []
    minimum_ratio = float(policy["minimum_relation_dominance_ratio"])
    for ratio in sorted(set(round(value, 12) for value in _relation_ratios(question.get("certificate")))):
        score = (
            1.0
            if ratio < minimum_ratio
            else _clamp01(1.0 - (ratio - minimum_ratio) / 0.5)
        )
        if score > 0:
            relation_signals.append(
                {
                    "code": (
                        "relation_ratio_below_policy"
                        if ratio < minimum_ratio
                        else "relation_ratio_near_minimum"
                    ),
                    "score": round(score, 6),
                    "measured": round(ratio, 6),
                    "admission_threshold": minimum_ratio,
                }
            )

    visual_signals.sort(
        key=lambda item: (-float(item["score"]), item["entity_id"], item["view_id"])
    )
    frame_signals.sort(key=lambda item: (-float(item["score"]), item["view_id"]))
    relation_signals.sort(key=lambda item: (-float(item["score"]), item["measured"]))
    risk_score = max(
        [float(item["score"]) for item in visual_signals]
        + [float(item["score"]) for item in frame_signals]
        + [float(item["score"]) for item in relation_signals]
        + [0.0]
    )
    return (
        round(risk_score, 6),
        risk_score >= NEAR_THRESHOLD_SCORE,
        {
            "purpose": "sampling_priority_only_not_an_independent_quality_judgment",
            "near_threshold_score_cutoff": NEAR_THRESHOLD_SCORE,
            "risk_score": round(risk_score, 6),
            "near_threshold": risk_score >= NEAR_THRESHOLD_SCORE,
            "screened_view_scope": "all_actual_model_input_views",
            "screened_view_ids": list(screening_view_ids),
            "oracle_screening_source_sensors": screening_sources,
            "positive_entity_view_pairs_with_signal": len(visual_signals),
            "model_input_frames_with_signal": len(frame_signals),
            "critical_frame_signal": frame_signals[0] if frame_signals else None,
            "critical_visual_signal": visual_signals[0] if visual_signals else None,
            "critical_relation_signal": relation_signals[0] if relation_signals else None,
        },
    )


def _stable_key(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\x1f{namespace}\x1f{value}".encode()).hexdigest()


def _risk_quota(
    *, population_size: int, risk_count: int, sample_size: int, multiplier: float
) -> tuple[int, float, bool]:
    if not population_size or not risk_count or not sample_size:
        return 0, 0.0, False
    population_share = risk_count / population_size
    requested_share = max(population_share, min(0.6, population_share * multiplier))
    quota = min(risk_count, sample_size, math.ceil(sample_size * requested_share))
    feasible = sample_size < population_size and quota / sample_size > population_share
    return quota, requested_share, feasible


def _round_robin_risk(
    candidates: Sequence[_Candidate], *, quota: int, seed: int
) -> list[_Candidate]:
    groups: dict[tuple[str, str, str, str], list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.near_threshold:
            groups[candidate.stratum].append(candidate)
    for _stratum, items in groups.items():
        items.sort(
            key=lambda item: (
                -item.risk_score,
                _stable_key(seed, "risk-item", item.fact_id),
            )
        )
    strata = sorted(
        groups,
        key=lambda value: _stable_key(seed, "risk-stratum", "\x1f".join(value)),
    )
    selected: list[_Candidate] = []
    while len(selected) < quota:
        progressed = False
        for stratum in strata:
            if groups[stratum] and len(selected) < quota:
                selected.append(groups[stratum].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def _balanced_remainder(
    candidates: Sequence[_Candidate],
    *,
    selected: list[_Candidate],
    sample_size: int,
    seed: int,
    near_threshold_target: int,
) -> list[_Candidate]:
    selected_ids = {item.fact_id for item in selected}
    dimension_counts: list[Counter[str]] = [Counter() for _ in range(4)]
    stratum_counts: Counter[tuple[str, str, str, str]] = Counter()
    for item in selected:
        stratum_counts[item.stratum] += 1
        for index, value in enumerate(item.stratum):
            dimension_counts[index][value] += 1
    remaining = [item for item in candidates if item.fact_id not in selected_ids]
    while len(selected) < sample_size and remaining:
        selected_risk = sum(item.near_threshold for item in selected)
        if selected_risk >= near_threshold_target:
            eligible = [item for item in remaining if not item.near_threshold]
        else:
            eligible = [item for item in remaining if item.near_threshold]
        if not eligible:
            eligible = remaining

        def priority(item: _Candidate) -> tuple[float, str]:
            gain = 2.0 / (1.0 + stratum_counts[item.stratum])
            gain += sum(
                1.0 / (1.0 + dimension_counts[index][value])
                for index, value in enumerate(item.stratum)
            )
            return (-gain, _stable_key(seed, "balanced-item", item.fact_id))

        chosen = min(eligible, key=priority)
        remaining.remove(chosen)
        selected.append(chosen)
        stratum_counts[chosen.stratum] += 1
        for index, value in enumerate(chosen.stratum):
            dimension_counts[index][value] += 1
    return selected


def _distribution(candidates: Sequence[_Candidate], index: int) -> dict[str, int]:
    return dict(sorted(Counter(item.stratum[index] for item in candidates).items()))


def _summary(candidates: Sequence[_Candidate]) -> dict[str, Any]:
    return {
        "questions": len(candidates),
        "near_threshold_questions": sum(item.near_threshold for item in candidates),
        "near_threshold_fraction": round(
            sum(item.near_threshold for item in candidates) / len(candidates), 6
        )
        if candidates
        else 0.0,
        "split": _distribution(candidates, 0),
        "trajectory_class": _distribution(candidates, 1),
        "program_id": _distribution(candidates, 2),
        "family_variant": _distribution(candidates, 3),
        "joint_strata": [
            {
                "split": key[0],
                "trajectory_class": key[1],
                "program_id": key[2],
                "family_variant": key[3],
                "count": count,
            }
            for key, count in sorted(Counter(item.stratum for item in candidates).items())
        ],
    }


def _rgb_inputs(
    episode: Mapping[str, Any],
    question: Mapping[str, Any],
    inventory_views: Mapping[tuple[str, str], Mapping[str, Any]],
    verified_hashes: dict[Path, str],
) -> list[dict[str, Any]]:
    observations = episode.get("observations")
    _require(isinstance(observations, list), "episode observations must be a list")
    by_view = {str(item["view_id"]): item for item in observations}
    raw_model_views = question.get("model_view_ids")
    view_ids = (
        [str(value) for value in raw_model_views]
        if isinstance(raw_model_views, list)
        else list(by_view)
    )
    result: list[dict[str, Any]] = []
    bundle_root = str(Path(str(episode["source_bundle"])).expanduser().resolve())
    for order, view_id in enumerate(view_ids, 1):
        _require(view_id in by_view, f"model view is absent from observations: {view_id}")
        path = Path(str(by_view[view_id]["rgb"])).expanduser().resolve()
        camera_height = by_view[view_id].get("camera_height_m")
        horizontal_fov = by_view[view_id].get("horizontal_fov_deg")
        _require(
            isinstance(camera_height, int | float)
            and not isinstance(camera_height, bool)
            and math.isfinite(float(camera_height))
            and float(camera_height) > 0,
            f"model-visible camera height is invalid: {view_id}",
        )
        _require(
            isinstance(horizontal_fov, int | float)
            and not isinstance(horizontal_fov, bool)
            and math.isfinite(float(horizontal_fov))
            and 0 < float(horizontal_fov) < 180,
            f"model-visible horizontal FOV is invalid: {view_id}",
        )
        inventory_key = (bundle_root, view_id)
        _require(
            inventory_key in inventory_views,
            f"source inventory has no model RGB view: {inventory_key}",
        )
        rgb_sha = _verify_inventory_file(
            inventory_views[inventory_key],
            path_key="rgb",
            sha_key="rgb_file_sha256",
            bytes_key="rgb_bytes",
            expected_path=path,
            verified_hashes=verified_hashes,
        )
        result.append(
            {
                "order": order,
                "view_id": view_id,
                "path": str(path),
                "sha256": rgb_sha,
                "camera_height_m": round(float(camera_height), 6),
                "horizontal_fov_deg": round(float(horizontal_fov), 6),
            }
        )
    return result


def _reviewer_fields() -> dict[str, Any]:
    return {
        "reviewer_type": None,
        "reviewer_id": "",
        "reviewer_system": "",
        "reviewer_model": "",
        "review_protocol_id": "",
        "review_prompt_sha256": "",
        "reviewed_at": "",
        "overall_status": None,
        "referents_recognizable": None,
        "answer_supported_by_model_rgb": None,
        "family_intervention_valid": None,
        "severity": None,
        "reason_codes": [],
        "notes_zh": "",
    }


def _reviewer_only_oracle_evidence(
    question: Mapping[str, Any],
    inventory_views: Mapping[tuple[str, str], Mapping[str, Any]],
    verified_hashes: dict[Path, str],
) -> dict[str, Any] | None:
    """Expose a held-out T10 render to the reviewer, never to the model.

    Target-view answers are certified by an oracle render.  Without that image
    an independent reviewer can inspect the prompt but cannot judge the target.
    The path is accepted only when the certificate, source inventory, file size
    and SHA-256 all agree.
    """

    if question.get("task_type") != "target_view_prediction":
        return None
    certificate = _mapping(question.get("certificate"), "question.certificate")
    raw_bundle = certificate.get("oracle_target_bundle")
    raw_view_id = certificate.get("oracle_target_view_id")
    raw_rgb = certificate.get("oracle_target_rgb")
    _require(
        isinstance(raw_bundle, str) and raw_bundle,
        "target-view certificate has no oracle_target_bundle",
    )
    _require(
        isinstance(raw_view_id, str) and raw_view_id,
        "target-view certificate has no oracle_target_view_id",
    )
    _require(
        isinstance(raw_rgb, str) and raw_rgb,
        "target-view certificate has no oracle_target_rgb",
    )
    bundle_root = str(Path(raw_bundle).expanduser().resolve())
    rgb_path = Path(raw_rgb).expanduser().resolve()
    inventory_key = (bundle_root, raw_view_id)
    _require(
        inventory_key in inventory_views,
        f"source inventory has no held-out target RGB view: {inventory_key}",
    )
    rgb_sha = _verify_inventory_file(
        inventory_views[inventory_key],
        path_key="rgb",
        sha_key="rgb_file_sha256",
        bytes_key="rgb_bytes",
        expected_path=rgb_path,
        verified_hashes=verified_hashes,
    )
    return {
        "purpose": "independent_answer_review_only",
        "never_model_input": True,
        "oracle_target_bundle": bundle_root,
        "oracle_target_view_id": raw_view_id,
        "target_rgb": {"path": str(rgb_path), "sha256": rgb_sha},
    }


def _item(
    candidate: _Candidate,
    *,
    rank: int,
    inventory_views: Mapping[tuple[str, str], Mapping[str, Any]],
    verified_hashes: dict[Path, str],
    consistency_members: Mapping[str, Sequence[_Candidate]],
) -> dict[str, Any]:
    episode = candidate.episode
    question = candidate.question
    consistency_group = question.get("consistency_group")
    sibling_items = []
    if isinstance(consistency_group, str) and consistency_group:
        for sibling in consistency_members.get(consistency_group, ()):
            if sibling.fact_id == candidate.fact_id:
                continue
            sibling_items.append(
                {
                    "fact_id": sibling.fact_id,
                    "episode_id": str(sibling.episode["episode_id"]),
                    "family_variant": sibling.stratum[3],
                    "question_zh": sibling.question.get("question_zh"),
                    "answer_zh": sibling.question.get("answer_zh"),
                    "answer_status": sibling.question.get("answer_status"),
                    "actual_model_rgb": _rgb_inputs(
                        sibling.episode,
                        sibling.question,
                        inventory_views,
                        verified_hashes,
                    ),
                }
            )
    return {
        "audit_index": rank,
        "fact_id": candidate.fact_id,
        "episode_id": str(episode["episode_id"]),
        "scene_id": str(episode["scene_id"]),
        "split": candidate.stratum[0],
        "trajectory_class": candidate.stratum[1],
        "source_sweep": episode.get("source_sweep"),
        "source_bundle": episode.get("source_bundle"),
        "program_id": candidate.program_id,
        "semantic_signature": _mapping(question.get("program"), "question.program").get(
            "semantic_signature"
        ),
        "task_type": question.get("task_type"),
        "family_variant": candidate.stratum[3],
        "consistency_group": consistency_group,
        "family_siblings_for_intervention_review": sibling_items,
        "question_zh": question.get("question_zh"),
        "answer_zh": question.get("answer_zh"),
        "answer_status": question.get("answer_status"),
        "answer_value": question.get("answer_value"),
        "evidence_view_ids": list(question.get("evidence_view_ids", [])),
        "evidence_entity_ids": list(question.get("evidence_entity_ids", [])),
        "actual_model_rgb": _rgb_inputs(
            episode, question, inventory_views, verified_hashes
        ),
        "reviewer_only_oracle_evidence": _reviewer_only_oracle_evidence(
            question, inventory_views, verified_hashes
        ),
        "automated_sampling_signal": dict(candidate.screening),
        "reviewer_fields": _reviewer_fields(),
    }


def _packet_id(
    binding: Mapping[str, Any],
    sampling: Mapping[str, Any],
    review_evidence_sha256: str,
) -> str:
    digest = hashlib.sha256(
        _canonical_json(
            {
                "release_manifest_sha256": binding["release_manifest"]["sha256"],
                "episode_ir_sha256": binding["episode_ir"]["sha256"],
                "source_inventory_sha256": binding["source_inventory"]["sha256"],
                "sampling": sampling,
                "review_evidence_sha256": review_evidence_sha256,
            }
        ).encode()
    ).hexdigest()[:20]
    return f"semantic-audit-{digest}"


def build_semantic_visual_audit_packet(
    release_dir: Path,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    risk_oversample_multiplier: float = DEFAULT_RISK_MULTIPLIER,
) -> dict[str, Any]:
    """Return an unfilled, deterministic independent-review packet."""

    _require(sample_size > 0, "sample_size must be positive")
    _require(
        math.isfinite(risk_oversample_multiplier)
        and risk_oversample_multiplier >= 1.0,
        "risk_oversample_multiplier must be finite and at least 1",
    )
    manifest, _ir_path, episodes, binding, source_inventory = _load_release(release_dir)
    policy = _release_visual_quality_policy(manifest)
    inventory_views = _inventory_view_map(source_inventory)
    verified_hashes: dict[Path, str] = {}
    bundle_cache: dict[tuple[str, str, str], Bundle] = {}
    candidates: list[_Candidate] = []
    fact_ids: set[str] = set()
    for episode in episodes:
        split = str(episode.get("split"))
        trajectory = str(episode.get("trajectory_class"))
        source_sweep = str(episode.get("source_sweep"))
        source_bundle = str(episode.get("source_bundle"))
        _require(source_bundle, "episode source_bundle is absent")
        bundle_key = (str(Path(source_bundle).resolve()), trajectory, source_sweep)
        if bundle_key not in bundle_cache:
            try:
                bundle_cache[bundle_key] = Bundle(
                    Path(source_bundle),
                    trajectory_class=trajectory,
                    source_sweep=source_sweep,
                    job_status="passed",
                )
            except (OSError, KeyError, TypeError, ValueError) as exc:
                raise SemanticVisualAuditError(
                    f"cannot load source bundle for semantic screening: {source_bundle}"
                ) from exc
        questions = episode.get("questions")
        _require(isinstance(questions, list), "episode questions must be a list")
        for question in questions:
            _require(isinstance(question, Mapping), "question must be an object")
            fact_id = str(question.get("fact_id"))
            _require(fact_id and fact_id != "None", "question fact_id is absent")
            _require(fact_id not in fact_ids, f"duplicate fact_id: {fact_id}")
            fact_ids.add(fact_id)
            program = _mapping(question.get("program"), f"program for {fact_id}")
            program_id = str(program.get("program_id"))
            variant = str(question.get("family_variant", "canonical"))
            risk_score, near_threshold, screening = _screen_candidate(
                episode,
                question,
                bundle=bundle_cache[bundle_key],
                policy=policy,
                inventory_views=inventory_views,
                verified_hashes=verified_hashes,
            )
            candidates.append(
                _Candidate(
                    episode=episode,
                    question=question,
                    program_id=program_id,
                    stratum=(split, trajectory, program_id, variant),
                    risk_score=risk_score,
                    near_threshold=near_threshold,
                    screening=screening,
                )
            )
    _require(candidates, "episode_ir contains no questions")
    candidates.sort(key=lambda item: item.fact_id)
    effective_size = min(sample_size, len(candidates))
    risk_count = sum(item.near_threshold for item in candidates)
    quota, requested_share, feasible = _risk_quota(
        population_size=len(candidates),
        risk_count=risk_count,
        sample_size=effective_size,
        multiplier=risk_oversample_multiplier,
    )
    selected = _round_robin_risk(candidates, quota=quota, seed=seed)
    selected = _balanced_remainder(
        candidates,
        selected=selected,
        sample_size=effective_size,
        seed=seed,
        near_threshold_target=quota,
    )
    selected.sort(
        key=lambda item: (
            item.stratum,
            _stable_key(seed, "display-order", item.fact_id),
        )
    )
    sampling_contract = {
        "seed": seed,
        "requested_sample_size": sample_size,
        "effective_sample_size": effective_size,
        "stratum_key": ["split", "trajectory_class", "program_id", "family_variant"],
        "algorithm": "risk_target_round_robin_then_risk_capped_greedy_marginal_balance.v1",
        "near_threshold_score_cutoff": NEAR_THRESHOLD_SCORE,
        "risk_oversample_multiplier": risk_oversample_multiplier,
        "population_near_threshold_fraction": round(risk_count / len(candidates), 6),
        "requested_near_threshold_fraction": round(requested_share, 6),
        "near_threshold_target_count": quota,
        "oversampling_feasible": feasible,
        "automatic_signal_role": "sampling_priority_only_not_an_independent_quality_judgment",
    }
    consistency_members: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        consistency_group = candidate.question.get("consistency_group")
        if isinstance(consistency_group, str) and consistency_group:
            consistency_members[consistency_group].append(candidate)
    for members in consistency_members.values():
        members.sort(key=lambda item: (item.stratum[3], item.fact_id))
    items = [
        _item(
            candidate,
            rank=index,
            inventory_views=inventory_views,
            verified_hashes=verified_hashes,
            consistency_members=consistency_members,
        )
        for index, candidate in enumerate(selected, 1)
    ]
    sample_summary = _summary(selected)
    sampling_contract["achieved_near_threshold_fraction"] = sample_summary[
        "near_threshold_fraction"
    ]
    review_evidence_payload = [
        {key: value for key, value in item.items() if key != "reviewer_fields"}
        for item in items
    ]
    review_evidence_sha256 = hashlib.sha256(
        _canonical_json(review_evidence_payload).encode()
    ).hexdigest()
    packet = {
        "schema_version": PACKET_SCHEMA,
        "packet_id": _packet_id(binding, sampling_contract, review_evidence_sha256),
        "status": "awaiting_independent_review",
        "independent_review_completed": False,
        "reviewer_provenance_policy": {
            "allowed_reviewer_types": [
                "model_assisted_independent",
                "human_independent",
            ],
            "required_when_reviewed": [
                "reviewer_type",
                "reviewer_id",
                "reviewed_at",
                "review_protocol_id",
            ],
            "required_for_model_assisted": [
                "reviewer_system",
                "reviewer_model",
                "review_prompt_sha256",
            ],
            "claim_boundary": (
                "model_assisted_independent review must never be reported as human review"
            ),
        },
        "disclaimer_zh": (
            "本文件只生成确定性分层样本和自动抽样优先级；自动信号不是独立语义质量结论，"
            "所有 reviewer_fields 均须由看过实际模型 RGB 及同一模型可见相机标定提示的独立"
            "复核者填写。复核者可以是"
            "透明标注的模型辅助独立复核或真人独立复核；前者不得表述为人工复核。"
        ),
        "release_binding": binding,
        "review_evidence_binding": {
            "sha256": review_evidence_sha256,
            "includes": [
                "selected fact/question/answer metadata",
                "actual model RGB paths, model-visible calibration and verified SHA-256",
                "oracle screening sensor paths and verified SHA-256",
                "automatic sampling signals",
                "sibling family review context",
                "reviewer-only held-out T10 RGB paths and verified SHA-256",
            ],
        },
        "visual_quality_policy": policy,
        "sampling_contract": sampling_contract,
        "population_summary": _summary(candidates),
        "sample_summary": sample_summary,
        "items": items,
        "source_release_status": manifest.get("status"),
    }
    return packet


def render_markdown(packet: Mapping[str, Any]) -> str:
    """Render a Chinese independent-review sheet without inventing outcomes."""

    binding = _mapping(packet["release_binding"], "release_binding")
    sampling = _mapping(packet["sampling_contract"], "sampling_contract")
    population = _mapping(packet["population_summary"], "population_summary")
    sample = _mapping(packet["sample_summary"], "sample_summary")
    lines = [
        "# EpiSpace 发布后独立语义视觉复核包（待填写）",
        "",
        "> **当前状态：尚未进行独立复核。** 自动 mask/几何信号只用于抽样排序，不构成语义质量结论。复核可由透明标注的模型辅助独立 reviewer 或真人独立 reviewer 完成；模型复核不得写成人工复核。",
        "",
        "## 发布绑定与抽样合同",
        "",
        f"- Packet ID：`{packet['packet_id']}`",
        f"- Dataset：`{binding.get('dataset_id')}`",
        f"- Release manifest SHA-256：`{binding['release_manifest']['sha256']}`",
        f"- Episode IR SHA-256：`{binding['episode_ir']['sha256']}`",
        f"- Source inventory SHA-256：`{binding['source_inventory']['sha256']}`",
        f"- Review evidence SHA-256：`{packet['review_evidence_binding']['sha256']}`",
        f"- 随机种子：`{sampling['seed']}`",
        f"- 分层键：`{' × '.join(sampling['stratum_key'])}`",
        f"- 总体 / 抽样：{population['questions']} / {sample['questions']} 个问题",
        (
            "- 接近阈值样本占比（总体 → 抽样）："
            f"{100 * population['near_threshold_fraction']:.1f}% → "
            f"{100 * sample['near_threshold_fraction']:.1f}%"
        ),
        "",
        "## 独立复核口径",
        "",
        "逐条打开下面列出的**实际模型输入 RGB**，并使用旁边与训练输入完全一致的相机高度和水平视场角提示。检查：物体是否可由自然语言指称；问题是否仅凭这些图与公开标定提示可回答；标准答案是否受证据支持；若属于 sibling family，干预是否只改变声明的证据或参照系。不要依据自动风险分数直接判定通过或失败。",
        "",
        "建议 `overall_status` 使用 `pass / minor_issue / major_issue / unreviewable`；`severity` 使用 `none / minor / major`。",
        "",
        "## 分层覆盖",
        "",
        "| 维度 | 抽样分布 |",
        "|---|---|",
    ]
    for label, key in (
        ("Split", "split"),
        ("轨迹类", "trajectory_class"),
        ("Program", "program_id"),
        ("Family variant", "family_variant"),
    ):
        distribution = ", ".join(f"{name}: {count}" for name, count in sample[key].items())
        lines.append(f"| {label} | {distribution} |")
    lines.extend(("", "## 审核条目", ""))
    for item in packet["items"]:
        lines.extend(
            (
                f"### {int(item['audit_index']):03d} · `{item['fact_id']}`",
                "",
                f"- Episode / Scene：`{item['episode_id']}` / `{item['scene_id']}`",
                (
                    "- 分层："
                    f"`{item['split']} × {item['trajectory_class']} × "
                    f"{item['program_id']} × {item['family_variant']}`"
                ),
                f"- Consistency group：`{item.get('consistency_group') or 'none'}`",
                f"- 问题：{item['question_zh']}",
                f"- 标准答案：{item['answer_zh']}（status=`{item['answer_status']}`）",
                (
                    "- 自动抽样信号："
                    f"risk={item['automated_sampling_signal']['risk_score']:.3f}，"
                    f"near-threshold={str(item['automated_sampling_signal']['near_threshold']).lower()}；"
                    "**仅用于抽样优先级**"
                ),
                "- 实际模型输入 RGB（按送入顺序）：",
                "",
            )
        )
        for rgb in item["actual_model_rgb"]:
            lines.append(
                f"  {rgb['order']}. `{rgb['view_id']}` — "
                f"camera_height={rgb['camera_height_m']:.2f}m, "
                f"horizontal_fov={rgb['horizontal_fov_deg']:.1f}° — "
                f"`{rgb['path']}` — SHA-256 `{rgb['sha256']}`"
            )
        oracle_evidence = item.get("reviewer_only_oracle_evidence")
        if isinstance(oracle_evidence, Mapping):
            target_rgb = _mapping(
                oracle_evidence["target_rgb"], "reviewer_only_oracle_evidence.target_rgb"
            )
            lines.extend(
                (
                    "",
                    "- **仅供独立 reviewer 核验答案的 held-out T10 RGB；绝不是模型输入：**",
                    (
                        f"  - `{oracle_evidence['oracle_target_view_id']}` — "
                        f"`{target_rgb['path']}` — SHA-256 `{target_rgb['sha256']}`"
                    ),
                )
            )
        siblings = item["family_siblings_for_intervention_review"]
        if siblings:
            lines.extend(("", "- 同一 consistency family 的 sibling（用于干预有效性检查）：", ""))
            for sibling in siblings:
                lines.append(
                    f"  - `{sibling['fact_id']}` / `{sibling['family_variant']}`："
                    f"{sibling['question_zh']} → {sibling['answer_zh']}"
                )
                for rgb in sibling["actual_model_rgb"]:
                    lines.append(
                        f"    - `{rgb['view_id']}` — "
                        f"camera_height={rgb['camera_height_m']:.2f}m, "
                        f"horizontal_fov={rgb['horizontal_fov_deg']:.1f}° — "
                        f"`{rgb['path']}` — SHA-256 `{rgb['sha256']}`"
                    )
        lines.extend(
            (
                "",
        "独立 reviewer 填写：",
        "",
        "- Reviewer type： [ ] model_assisted_independent  [ ] human_independent",
        "- Reviewer ID：",
        "- Reviewer system：",
        "- Reviewer model：",
        "- Review protocol ID：",
        "- Review prompt SHA-256：",
        "- Reviewed at：",
                "- Overall status： [ ] pass  [ ] minor_issue  [ ] major_issue  [ ] unreviewable",
                "- Referents recognizable： [ ] yes  [ ] no",
                "- Answer supported by model RGB： [ ] yes  [ ] no",
                "- Family intervention valid： [ ] yes  [ ] no  [ ] not_applicable",
                "- Severity： [ ] none  [ ] minor  [ ] major",
                "- Reason codes：",
                "- Notes：",
                "",
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def write_semantic_visual_audit_packet(
    release_dir: Path,
    *,
    json_output: Path,
    markdown_output: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    risk_oversample_multiplier: float = DEFAULT_RISK_MULTIPLIER,
) -> dict[str, Any]:
    packet = build_semantic_visual_audit_packet(
        release_dir,
        sample_size=sample_size,
        seed=seed,
        risk_oversample_multiplier=risk_oversample_multiplier,
    )
    json_text = json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    markdown_text = render_markdown(packet)
    _write_text_atomic(json_output, json_text)
    _write_text_atomic(markdown_output, markdown_text)
    return packet


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic, unfilled independent semantic visual audit packet."
    )
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--risk-oversample-multiplier",
        type=float,
        default=DEFAULT_RISK_MULTIPLIER,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    release_dir = args.release_dir.expanduser().resolve()
    json_output = args.json_output or release_dir / "semantic_visual_audit_packet.json"
    markdown_output = args.markdown_output or release_dir / "semantic_visual_audit_packet.md"
    packet = write_semantic_visual_audit_packet(
        release_dir,
        json_output=json_output,
        markdown_output=markdown_output,
        sample_size=args.sample_size,
        seed=args.seed,
        risk_oversample_multiplier=args.risk_oversample_multiplier,
    )
    print(
        f"{packet['packet_id']}: {len(packet['items'])} unreviewed items -> "
        f"{json_output}, {markdown_output}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
