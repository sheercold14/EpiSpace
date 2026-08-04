#!/usr/bin/env python3
"""Compile real OmniGibson trajectory bundles into a web/audit catalog.

The script never invents views or quality values. It exports browser media from
the NPZ sensor artifacts, derives T2/T5 records from measured visibility, and
marks T6 as partial whenever the saved T1 bundle lacks an ordered room timeline.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

STRUCTURAL_CATEGORIES = {
    "background",
    "ceilings",
    "driveway",
    "fence",
    "floors",
    "lawn",
    "roof",
    "walls",
}

CLASS_METADATA = {
    "T1": {
        "name_zh": "覆盖漫游",
        "axes": "平移 · 稠密链 · 全覆盖/revisit · 场景中心",
        "capability_hypothesis": "稠密 ego-motion 支持增量建图、记忆与闭环持久性。",
    },
    "T3": {
        "name_zh": "原地旋转站",
        "axes": "纯旋转 · 相邻高重叠 · 局部全景 · 观察者锚定",
        "capability_hypothesis": "零视差条件下，模型必须依赖朝向变换完成全景整合与背后推理。",
    },
    "T4": {
        "name_zh": "物体环绕",
        "axes": "目标中心 · 八方位 · 连通闭环 · 物体身份绑定",
        "capability_hypothesis": "从多个方位持续绑定同一物体，并组合局部观察形成物体中心表征。",
    },
    "T7": {
        "name_zh": "高度与俯仰",
        "axes": "同一 XY · 低/人眼/高机位 · 俯仰干预",
        "capability_hypothesis": "在相机高度和俯仰变化下保持实体绑定，并学习垂直 frame 变换。",
    },
    "T8": {
        "name_zh": "遮挡揭示",
        "axes": "遮挡 → 位移 → 显露 · target/occluder certificate",
        "capability_hypothesis": "目标暂时不可见时保持 belief，并用后续显露证据更新而非重新猜测。",
    },
    "T10": {
        "name_zh": "目标视图",
        "axes": "对象锚定 · target holdout · origin→facing 朝向",
        "capability_hypothesis": "把共享 canonical state 变换到未见目标视角，执行 perspective read。",
    },
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _mask_rgb(mask: np.ndarray) -> np.ndarray:
    identifiers = mask.astype(np.uint64)
    result = np.stack(
        (
            (identifiers * 73 + 19) % 251,
            (identifiers * 151 + 47) % 253,
            (identifiers * 199 + 89) % 255,
        ),
        axis=-1,
    ).astype(np.uint8)
    result[identifiers == 0] = 0
    result[identifiers == 1] = 32
    return result


def _depth_rgb(depth: np.ndarray, near_m: float, far_m: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth >= near_m) & (depth <= far_m)
    normalized = np.zeros(depth.shape, dtype=np.float32)
    if np.any(valid):
        lower, upper = np.percentile(depth[valid], (2.0, 98.0))
        upper = max(float(upper), float(lower) + 1e-6)
        normalized[valid] = np.clip((depth[valid] - lower) / (upper - lower), 0.0, 1.0)
    red = np.clip(2.0 * normalized - 0.25, 0.0, 1.0)
    green = np.clip(2.0 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(1.25 - 2.0 * normalized, 0.0, 1.0)
    result = (np.stack((red, green, blue), axis=-1) * 255.0).astype(np.uint8)
    result[~valid] = 0
    return result


def _save_webp(path: Path, array: np.ndarray, *, lossless: bool = False) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(
        path,
        "WEBP",
        quality=90,
        method=4,
        lossless=lossless,
    )


def _runtime_core_sets(
    snapshot: dict[str, Any], render: dict[str, Any]
) -> list[set[int]]:
    category_by_name = {
        str(entity["name"]): str(entity.get("category", "object"))
        for entity in snapshot["entities"]
    }
    name_by_id = {
        int(identifier): str(name)
        for identifier, name in snapshot["runtime_instance_registry"].items()
        if int(identifier) > 1
    }
    return [
        {
            int(identifier)
            for identifier in view["visible_runtime_instance_ids"]
            if category_by_name.get(name_by_id.get(int(identifier), ""), "")
            not in STRUCTURAL_CATEGORIES
        }
        for view in render["views"]
    ]


def _clockwise_yaw_deg(view: dict[str, Any]) -> float:
    quaternion = view["world_from_agent"]["rotation_xyzw"]
    x, y, z, w = (float(value) for value in quaternion)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return round((-math.degrees(yaw)) % 360.0, 6)


def _yaw_difference(left: float, right: float) -> float:
    return round(abs((right - left + 180.0) % 360.0 - 180.0), 6)


def _yaw_bin(value: float) -> str:
    if value < 45.0:
        return "lt_45"
    if value <= 120.0:
        return "45_to_120"
    return "gt_120"


def _connected(view_indices: tuple[int, ...], core_sets: list[set[int]]) -> bool:
    reached = {view_indices[0]}
    changed = True
    while changed:
        changed = False
        for left in tuple(reached):
            for right in view_indices:
                if right not in reached and core_sets[left] & core_sets[right]:
                    reached.add(right)
                    changed = True
    return len(reached) == len(view_indices)


def _derive_t2(
    trajectory: dict[str, Any], core_sets: list[set[int]], *, seed: int
) -> dict[str, Any]:
    views = trajectory["views"]
    rng = random.Random(seed)
    counts = {str(k): {"high": 0, "low": 0, "zero": 0} for k in range(2, 6)}
    selected: dict[tuple[int, str], dict[str, Any]] = {}
    for k in range(2, min(5, len(views)) + 1):
        for indices in itertools.combinations(range(len(views)), k):
            adjacent = [
                len(core_sets[left] & core_sets[right])
                for left, right in itertools.pairwise(indices)
            ]
            all_pairs = [
                len(core_sets[left] & core_sets[right])
                for left, right in itertools.combinations(indices, 2)
            ]
            overlap_class = None
            if all(value == 0 for value in all_pairs):
                overlap_class = "zero"
            elif adjacent and min(adjacent) >= 5:
                overlap_class = "high"
            elif adjacent and all(1 <= value <= 2 for value in adjacent):
                overlap_class = "low"
            if overlap_class is None:
                continue
            counts[str(k)][overlap_class] += 1
            key = (k, overlap_class)
            span = indices[-1] - indices[0]
            previous = selected.get(key)
            if previous is not None and span <= previous["index_span"]:
                continue
            view_ids = [views[index]["view_id"] for index in indices]
            shuffled = list(view_ids)
            rng.shuffle(shuffled)
            if shuffled == view_ids:
                shuffled.reverse()
            selected[key] = {
                "trajectory_id": f"T2-k{k}-{overlap_class}",
                "k": k,
                "overlap_class": overlap_class,
                "view_ids": view_ids,
                "legal_shuffle_view_ids": shuffled,
                "adjacent_common_core_entity_counts": adjacent,
                "all_pair_common_core_entity_counts": all_pairs,
                "co_visibility_connected": _connected(indices, core_sets),
                "answerability": "unknown" if overlap_class == "zero" else "answerable",
                "index_span": span,
            }
    return {
        "schema_version": "episode3d.t2_derivation.v1",
        "candidate_counts": counts,
        "selected_subsets": [
            selected[key] for key in sorted(selected, key=lambda item: (item[0], item[1]))
        ],
        "coverage_gaps": [
            {"k": k, "overlap_class": overlap}
            for k in range(2, min(5, len(views)) + 1)
            for overlap in ("high", "low", "zero")
            if (k, overlap) not in selected
        ],
    }


def _derive_t5(
    trajectory_id: str,
    trajectory: dict[str, Any],
    core_sets: list[set[int]],
) -> dict[str, Any]:
    views = trajectory["views"]
    yaws = [
        float(value)
        for value in trajectory.get(
            "yaw_sequence_deg", [_clockwise_yaw_deg(view) for view in views]
        )
    ]
    bins: dict[str, list[dict[str, Any]]] = {"high": [], "low": [], "zero": []}
    for left, right in itertools.combinations(range(len(views)), 2):
        common = len(core_sets[left] & core_sets[right])
        overlap_class = "high" if common >= 5 else "low" if common in {1, 2} else "zero" if common == 0 else None
        if overlap_class is None:
            continue
        left_position = np.asarray(
            views[left]["world_from_agent"]["translation_m"], dtype=np.float64
        )
        right_position = np.asarray(
            views[right]["world_from_agent"]["translation_m"], dtype=np.float64
        )
        yaw_difference = _yaw_difference(yaws[left], yaws[right])
        bins[overlap_class].append(
            {
                "pair_id": f"{trajectory_id}:T5:{left:03d}-{right:03d}",
                "source_trajectory_id": trajectory_id,
                "view_ids": [views[left]["view_id"], views[right]["view_id"]],
                "common_core_entity_count": common,
                "overlap_class": overlap_class,
                "translation_distance_m": round(
                    float(np.linalg.norm(right_position - left_position)), 6
                ),
                "yaw_difference_deg": yaw_difference,
                "yaw_bin": _yaw_bin(yaw_difference),
                "answerability": "unknown" if overlap_class == "zero" else "answerable",
            }
        )
    selected = []
    for overlap_class in ("high", "low", "zero"):
        candidates = bins[overlap_class]
        if candidates:
            selected.append(
                max(
                    candidates,
                    key=lambda item: (
                        item["yaw_difference_deg"], item["translation_distance_m"]
                    ),
                )
            )
    return {
        "candidate_counts": {key: len(value) for key, value in bins.items()},
        "selected_pairs": selected,
    }


def _audit_t6(bundle: Path) -> dict[str, Any]:
    selection_path = bundle / "trajectory_selection.json"
    selection = _read(selection_path) if selection_path.is_file() else {}
    room_sequence = selection.get("path_room_sequence")
    rooms = selection.get("path_room_instances", [])
    missing = []
    if not room_sequence:
        missing.append("ordered path room timeline")
    if not selection.get("doorway_crossings"):
        missing.append("explicit doorway crossings")
    return {
        "audit_status": "complete" if not missing else "partial",
        "bridge_eligible": bool(
            room_sequence
            and len(set(room_sequence)) >= 3
            and len(selection.get("doorway_crossings", [])) >= 2
        ),
        "covered_room_instances": rooms,
        "ordered_room_sequence": room_sequence,
        "missing_evidence": missing,
    }


def _audit_static_sweep(plan_path: Path | None) -> dict[str, Any] | None:
    if plan_path is None:
        return None
    plan = _read(plan_path)
    audited = []
    for job in plan["jobs"]:
        bundle = Path(job["bundle"])
        selection_path = bundle / "trajectory_selection.json"
        if not selection_path.is_file():
            audited.append({"job_id": job["job_id"], "status": "missing_selection"})
            continue
        selection = _read(selection_path)
        rooms = selection.get("path_room_instances", [])
        sequence = selection.get("path_room_sequence")
        audited.append(
            {
                "job_id": job["job_id"],
                "status": "complete" if sequence else "legacy_metadata_partial",
                "room_count": len(rooms),
                "has_ordered_room_sequence": bool(sequence),
            }
        )
    return {
        "sweep_id": plan["sweep_id"],
        "scene_count": len(audited),
        "three_room_coverage_count": sum(
            int(item.get("room_count", 0) >= 3) for item in audited
        ),
        "ordered_room_sequence_count": sum(
            int(item.get("has_ordered_room_sequence", False)) for item in audited
        ),
        "bridge_certified_count": 0,
        "conclusion": (
            "Legacy T1 bundles can screen room count, but cannot certify T6 until "
            "ordered room transitions and doorway crossings are reacquired."
        ),
        "jobs": audited,
    }


def _code_url(path: Path) -> str:
    """Return a URL for servers rooted at the shared ``code`` directory."""

    resolved = path.resolve()
    parts = resolved.parts
    try:
        code_index = len(parts) - 1 - list(reversed(parts)).index("code")
    except ValueError:
        return ""
    return "/" + "/".join(parts[code_index + 1 :])


def _reason_code(status: str, detail: str | None) -> str:
    if status == "passed":
        return "machine_verified"
    if status == "needs_review":
        return "evidence_gate_review"
    message = detail or ""
    if "no focus object supports" in message:
        return "no_complete_navigable_orbit"
    if "exceeded 2400 seconds" in message:
        return "scene_timeout"
    if "insufficient vertical clearance" in message:
        return "insufficient_vertical_clearance"
    if "no geometric T8" in message:
        return "no_occlusion_reveal_candidate"
    if "insufficient reachable T10 anchor pairs" in message:
        return "insufficient_target_anchor_pairs"
    return "acquisition_failure"


def _summarize_sweep(
    plan_path: Path, trajectory_class: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan = _read(plan_path)
    statuses = Counter(str(job["status"]) for job in plan["jobs"])
    jobs = []
    reasons: Counter[str] = Counter()
    for job in plan["jobs"]:
        bundle = Path(job["bundle"])
        status = str(job["status"])
        detail = job.get("status_detail")
        reason_code = _reason_code(status, detail)
        if status != "passed":
            reasons[reason_code] += 1
        quality_path = bundle / "quality_report.json"
        trajectory_path = bundle / "trajectory_plan.json"
        quality = _read(quality_path) if quality_path.is_file() else {}
        trajectory = _read(trajectory_path) if trajectory_path.is_file() else {}
        preview_path = bundle / "preview.html"
        jobs.append(
            {
                "trajectory_class": trajectory_class,
                "job_id": job["job_id"],
                "scene": job["scene_model"],
                "split": job.get("split", "unassigned"),
                "classification": job.get("classification", "unknown"),
                "status": status,
                "reason_code": reason_code,
                "status_detail": detail,
                "view_count": len(trajectory.get("views", [])),
                "visual_status": quality.get("visual_status"),
                "trajectory_status": quality.get("trajectory_status"),
                "preview_url": _code_url(preview_path) if preview_path.is_file() else None,
                "quality_url": _code_url(quality_path) if quality_path.is_file() else None,
                "bundle_path": str(bundle),
            }
        )
    total = len(jobs)
    passed = statuses.get("passed", 0)
    return (
        {
            "trajectory_class": trajectory_class,
            "name_zh": CLASS_METADATA[trajectory_class]["name_zh"],
            "sweep_id": plan["sweep_id"],
            "total": total,
            "passed": passed,
            "needs_review": statuses.get("needs_review", 0),
            "failed": statuses.get("failed", 0),
            "strict_pass_rate": round(passed / total, 4) if total else 0.0,
            "reason_counts": dict(sorted(reasons.items())),
            "plan_path": str(plan_path),
            "plan_sha256": _sha256(plan_path),
        },
        jobs,
    )


def _summarize_sweeps(
    plan_paths: list[Path], trajectory_class: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Aggregate disjoint acquisition sweeps without hiding their provenance."""

    if not plan_paths:
        raise ValueError(f"no {trajectory_class} sweep plans supplied")
    summaries: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for plan_path in plan_paths:
        summary, sweep_jobs = _summarize_sweep(plan_path, trajectory_class)
        summaries.append(summary)
        jobs.extend(sweep_jobs)
        reasons.update(summary["reason_counts"])
    total = sum(item["total"] for item in summaries)
    passed = sum(item["passed"] for item in summaries)
    return (
        {
            "trajectory_class": trajectory_class,
            "name_zh": CLASS_METADATA[trajectory_class]["name_zh"],
            "sweep_id": "+".join(item["sweep_id"] for item in summaries),
            "sweep_ids": [item["sweep_id"] for item in summaries],
            "total": total,
            "passed": passed,
            "needs_review": sum(item["needs_review"] for item in summaries),
            "failed": sum(item["failed"] for item in summaries),
            "strict_pass_rate": round(passed / total, 4) if total else 0.0,
            "reason_counts": dict(sorted(reasons.items())),
            "plan_paths": [item["plan_path"] for item in summaries],
            "plan_sha256": {
                item["sweep_id"]: item["plan_sha256"] for item in summaries
            },
        },
        jobs,
    )


def _select_representative(plan_path: Path) -> tuple[Path, dict[str, Any]]:
    plan = _read(plan_path)
    candidates = []
    for job in plan["jobs"]:
        if job["status"] != "passed":
            continue
        bundle = Path(job["bundle"])
        quality_path = bundle / "quality_report.json"
        if not quality_path.is_file():
            continue
        quality = _read(quality_path)
        summary = quality["summary"]
        score = (
            int(summary["minimum_visible_instance_count"]),
            float(summary["median_sharpness_laplacian_variance"]),
            str(job["job_id"]),
        )
        candidates.append((score, bundle, job))
    if not candidates:
        raise ValueError(f"no passed representative in {plan_path}")
    _, bundle, job = max(candidates, key=lambda item: item[0])
    return bundle, job


def _summarize_t9(release_path: Path) -> dict[str, Any]:
    release = _read(release_path)
    status = release["counts"]["by_job_status"]
    total = int(release["counts"]["pairs"])
    certified = int(status.get("certified", 0))
    return {
        "trajectory_class": "T9",
        "name_zh": "反事实干预对",
        "collection_id": release["dataset_name"],
        "total": total,
        "passed": certified,
        "needs_review": int(status.get("needs_review", 0)),
        "failed": int(status.get("failed", 0)),
        "strict_pass_rate": round(certified / total, 4) if total else 0.0,
        "release_path": str(release_path),
        "release_sha256": _sha256(release_path),
    }


def _export_bundle(
    *,
    bundle: Path,
    trajectory_id: str,
    media_root: Path,
    media_url_root: str,
    status: str = "passed",
    enforce_gate_pass: bool = True,
) -> tuple[dict[str, Any], list[set[int]]]:
    trajectory = _read(bundle / "trajectory_plan.json")
    render = _read(bundle / "render_report.json")
    quality = _read(bundle / "quality_report.json")
    snapshot = _read(bundle / "scene_snapshot.json")
    episode = _read(bundle / "spatial_episode.json")
    if episode["channel_policy"]["model_visible"] != ["rgb"]:
        raise ValueError(f"{trajectory_id} violates the RGB-only model-visible contract")
    trajectory_class = trajectory.get("trajectory_class", "T1")
    if enforce_gate_pass and trajectory_class in {"T3", "T4", "T7", "T8", "T10"}:
        gate_status = quality.get("gates", {}).get(trajectory_class, {}).get("status")
        if gate_status != "pass":
            raise ValueError(
                f"{trajectory_id} failed its {trajectory_class} evidence gate"
            )
    core_sets = _runtime_core_sets(snapshot, render)
    metric_by_view = {view["view_id"]: view for view in quality["views"]}
    render_by_view = {view["view_id"]: view for view in render["views"]}
    sensor = render["sensor_contract"]
    yaws = trajectory.get("yaw_sequence_deg") or [
        _clockwise_yaw_deg(view) for view in trajectory["views"]
    ]
    views = []
    for index, view in enumerate(trajectory["views"]):
        view_id = view["view_id"]
        artifact = Path(render_by_view[view_id]["artifact"]["path"])
        with np.load(artifact, allow_pickle=False) as arrays:
            channels = {
                "rgb": arrays["rgb"],
                "depth": _depth_rgb(
                    arrays["depth_m"], float(sensor["near_m"]), float(sensor["far_m"])
                ),
                "instance": _mask_rgb(arrays["instance_id"]),
                "semantic": _mask_rgb(arrays["semantic_id"]),
            }
        media = {}
        for channel, array in channels.items():
            relative = Path(trajectory_id) / f"{view_id}-{channel}.webp"
            _save_webp(media_root / relative, array, lossless=channel != "rgb")
            media[channel] = f"{media_url_root.rstrip('/')}/{relative.as_posix()}"
        metric = metric_by_view[view_id]
        views.append(
            {
                "view_id": view_id,
                "step": int(view["step"]),
                "role": view["role"],
                "position_m": view["world_from_agent"]["translation_m"],
                "yaw_deg": float(yaws[index]),
                "visible_instance_count": int(
                    render_by_view[view_id]["visible_instance_count"]
                ),
                "visible_core_entity_count": len(core_sets[index]),
                "valid_depth_fraction": metric["valid_depth_fraction"],
                "sharpness": metric["sharpness_laplacian_variance"],
                "media": media,
            }
        )
    class_copy = CLASS_METADATA[trajectory_class]
    selection_path = bundle / "trajectory_selection.json"
    selection = _read(selection_path) if selection_path.is_file() else {}
    return (
        {
            "trajectory_id": trajectory_id,
            "trajectory_class": trajectory_class,
            **class_copy,
            "scene": snapshot["source_scene_id"],
            "seed": int(trajectory["seed"]),
            "status": status,
            "view_count": len(views),
            "model_visible": ["rgb"],
            "supervision": episode["channel_policy"]["supervision"],
            "selection": selection,
            "quality": {
                "integrity_status": quality["integrity_status"],
                "visual_status": quality["visual_status"],
                "trajectory_status": quality.get("trajectory_status", "pass"),
                "summary": quality["summary"],
                "loop_closure": quality.get("loop_closure"),
                "gates": quality.get("gates", {}),
            },
            "views": views,
            "provenance": {
                name: {"path": str(bundle / name), "sha256": _sha256(bundle / name)}
                for name in (
                    "trajectory_plan.json",
                    "trajectory_selection.json",
                    "render_report.json",
                    "scene_snapshot.json",
                    "quality_report.json",
                    "spatial_episode.json",
                )
                if (bundle / name).is_file()
            },
        },
        core_sets,
    )


def _export_job_details(
    *,
    jobs: list[dict[str, Any]],
    media_root: Path,
    media_url_root: str,
    details_root: Path,
    details_url_root: str,
) -> int:
    """Export one lazy-loaded detail record for every complete render bundle."""

    eligible = [job for job in jobs if job["status"] != "failed"]
    for index, job in enumerate(eligible, start=1):
        detail, _ = _export_bundle(
            bundle=Path(job["bundle_path"]),
            trajectory_id=str(job["job_id"]),
            media_root=media_root,
            media_url_root=media_url_root,
            status=str(job["status"]),
            enforce_gate_pass=job["status"] == "passed",
        )
        detail.update(
            {
                "source_job_id": job["job_id"],
                "source_split": job["split"],
                "classification": job["classification"],
                "status_detail": job["status_detail"],
                "reason_code": job["reason_code"],
            }
        )
        detail_path = details_root / f"{job['job_id']}.json"
        _write_json(detail_path, detail)
        job["detail_url"] = (
            f"{details_url_root.rstrip('/')}/{detail_path.name}"
        )
        job["thumbnail_url"] = detail["views"][0]["media"]["rgb"]
        if index % 10 == 0 or index == len(eligible):
            print(f"exported trajectory detail {index}/{len(eligible)}")

    for job in jobs:
        job.pop("bundle_path", None)
        if job["status"] == "failed":
            job["detail_url"] = None
            job["thumbnail_url"] = None
    return len(eligible)


def build_catalog(
    *,
    t1_bundle: Path,
    t3_bundle: Path,
    media_root: Path,
    media_url_root: str,
    details_root: Path,
    details_url_root: str,
    static_sweep_plan: Path | None,
    t3_sweep_plans: list[Path],
    typed_sweep_plans: dict[str, Path],
    t9_release_path: Path,
) -> dict[str, Any]:
    t1, t1_sets = _export_bundle(
        bundle=t1_bundle,
        trajectory_id="Rs_int_t1_seed17",
        media_root=media_root,
        media_url_root=media_url_root,
    )
    t3, t3_sets = _export_bundle(
        bundle=t3_bundle,
        trajectory_id="Rs_int_t3_seed17",
        media_root=media_root,
        media_url_root=media_url_root,
    )
    t1_plan = _read(t1_bundle / "trajectory_plan.json")
    t3_plan = _read(t3_bundle / "trajectory_plan.json")
    t2 = _derive_t2(t1_plan, t1_sets, seed=17)
    t5_t1 = _derive_t5(t1["trajectory_id"], t1_plan, t1_sets)
    t5_t3 = _derive_t5(t3["trajectory_id"], t3_plan, t3_sets)
    t5 = {
        "schema_version": "episode3d.t5_derivation.v1",
        "sources": [t5_t1, t5_t3],
        "selected_pairs": t5_t1["selected_pairs"] + t5_t3["selected_pairs"],
    }
    static_audit = _audit_static_sweep(static_sweep_plan)
    if static_sweep_plan is None:
        raise ValueError("static T1 sweep plan is required for collection accounting")

    t1_collection, collection_jobs = _summarize_sweep(static_sweep_plan, "T1")
    t1_collection["representative_trajectory_id"] = t1["trajectory_id"]
    t3_collection, t3_jobs = _summarize_sweeps(t3_sweep_plans, "T3")
    t3_collection["representative_trajectory_id"] = t3["trajectory_id"]
    t3_collection["representative_scope"] = "legacy_mechanism_pilot"
    collection_jobs.extend(t3_jobs)

    representatives = [t1, t3]
    collections = [t1_collection, t3_collection]
    for trajectory_class in ("T4", "T7", "T8", "T10"):
        plan_path = typed_sweep_plans[trajectory_class]
        collection, jobs = _summarize_sweep(plan_path, trajectory_class)
        bundle, representative_job = _select_representative(plan_path)
        representative, _ = _export_bundle(
            bundle=bundle,
            trajectory_id=representative_job["job_id"],
            media_root=media_root,
            media_url_root=media_url_root,
        )
        representative["source_job_id"] = representative_job["job_id"]
        representative["source_split"] = representative_job.get("split")
        collection["representative_trajectory_id"] = representative["trajectory_id"]
        representatives.append(representative)
        collections.append(collection)
        collection_jobs.extend(jobs)

    browsable_episode_count = _export_job_details(
        jobs=collection_jobs,
        media_root=media_root,
        media_url_root=media_url_root,
        details_root=details_root,
        details_url_root=details_url_root,
    )

    t9_collection = _summarize_t9(t9_release_path)
    collections.append(t9_collection)
    collections.sort(key=lambda item: int(item["trajectory_class"][1:]))
    acquired_collections = [
        item for item in collections if item["trajectory_class"] != "T9"
    ]
    acquisition_job_count = sum(item["total"] for item in acquired_collections)
    passed_episode_count = sum(item["passed"] for item in acquired_collections)
    review_episode_count = sum(
        item["needs_review"] for item in acquired_collections
    )
    failed_episode_count = sum(item["failed"] for item in acquired_collections)

    collection_by_class = {item["trajectory_class"]: item for item in collections}

    def coverage_item(
        trajectory_class: str,
        name: str,
        status: str,
        count: int,
        *,
        passed: int | None = None,
        needs_review: int = 0,
        failed: int = 0,
    ) -> dict[str, Any]:
        return {
            "class": trajectory_class,
            "name": name,
            "status": status,
            "count": count,
            "passed": passed if passed is not None else count,
            "needs_review": needs_review,
            "failed": failed,
        }

    return {
        "schema_version": "episode3d.trajectory_catalog.v2",
        "collection_id": "episode3d-og-trajectories-collection-v1",
        "research_rule": (
            "Trajectory is a query-agnostic SE(3) x time information structure; "
            "questions are compiled only after acquisition."
        ),
        "channel_contract": {
            "model_visible": ["rgb"],
            "supervision": ["depth_m", "instance_id", "semantic_id", "camera_pose"],
            "oracle_only": ["scene_snapshot", "scene_ir", "relation_oracle"],
        },
        "summary": {
            "real_trajectory_count": len(representatives),
            "representative_trajectory_count": len(representatives),
            "real_view_count": sum(len(item["views"]) for item in representatives),
            "acquisition_job_count": acquisition_job_count,
            "passed_episode_count": passed_episode_count,
            "needs_review_episode_count": review_episode_count,
            "failed_episode_count": failed_episode_count,
            "browsable_episode_count": browsable_episode_count,
            "strict_pass_rate": round(
                passed_episode_count / acquisition_job_count, 4
            ),
            "newly_acquired_trajectory_count": sum(
                collection_by_class[key]["total"]
                for key in ("T3", "T4", "T7", "T8", "T10")
            ),
            "intervention_pair_count": t9_collection["total"],
            "certified_intervention_pair_count": t9_collection["passed"],
            "t3_panorama_core_entity_count": t3["quality"]["gates"]["T3"][
                "panorama_core_entity_count"
            ],
            "t3_rear_only_core_entity_count": t3["quality"]["gates"]["T3"][
                "rear_only_core_entity_count"
            ],
            "t2_selected_subset_count": len(t2["selected_subsets"]),
            "t5_selected_pair_count": len(t5["selected_pairs"]),
        },
        "trajectories": representatives,
        "collections": collections,
        "collection_jobs": sorted(
            collection_jobs,
            key=lambda item: (item["trajectory_class"], item["job_id"]),
        ),
        "derived": {"T2": t2, "T5": t5, "T6": _audit_t6(t1_bundle)},
        "static_sweep_t6_audit": static_audit,
        "coverage": [
            coverage_item(
                "T1",
                "覆盖漫游",
                "acquired_collection",
                t1_collection["total"],
                passed=t1_collection["passed"],
                needs_review=t1_collection["needs_review"],
                failed=t1_collection["failed"],
            ),
            coverage_item(
                "T2", "稀疏子采样", "derived_pilot", len(t2["selected_subsets"])
            ),
            coverage_item(
                "T3",
                "原地旋转站",
                "acquired_collection",
                t3_collection["total"],
                passed=t3_collection["passed"],
                needs_review=t3_collection["needs_review"],
                failed=t3_collection["failed"],
            ),
            coverage_item(
                "T4",
                "物体环绕",
                "acquired_collection",
                collection_by_class["T4"]["total"],
                passed=collection_by_class["T4"]["passed"],
                needs_review=collection_by_class["T4"]["needs_review"],
                failed=collection_by_class["T4"]["failed"],
            ),
            coverage_item(
                "T5", "双视图对", "derived_pilot", len(t5["selected_pairs"])
            ),
            coverage_item("T6", "跨房间桥接", "audit_partial", 0, passed=0),
            coverage_item(
                "T7",
                "高度/俯仰",
                "acquired_collection",
                collection_by_class["T7"]["total"],
                passed=collection_by_class["T7"]["passed"],
                needs_review=collection_by_class["T7"]["needs_review"],
                failed=collection_by_class["T7"]["failed"],
            ),
            coverage_item(
                "T8",
                "遮挡揭示",
                "acquired_collection",
                collection_by_class["T8"]["total"],
                passed=collection_by_class["T8"]["passed"],
                needs_review=collection_by_class["T8"]["needs_review"],
                failed=collection_by_class["T8"]["failed"],
            ),
            coverage_item(
                "T9",
                "反事实干预对",
                "intervention_collection",
                t9_collection["total"],
                passed=t9_collection["passed"],
                needs_review=t9_collection["needs_review"],
                failed=t9_collection["failed"],
            ),
            coverage_item(
                "T10",
                "目标视图",
                "acquired_collection",
                collection_by_class["T10"]["total"],
                passed=collection_by_class["T10"]["passed"],
                needs_review=collection_by_class["T10"]["needs_review"],
                failed=collection_by_class["T10"]["failed"],
            ),
        ],
        "evidence_boundary": {
            "measured": [
                "RGB/depth/instance/semantic channels and camera poses",
                "T3 zero translational baseline, 60-degree yaw schedule and wall clearance",
                "T4/T7/T8/T10 trajectory-specific executable quality gates",
                "Visibility-derived panorama, occlusion and cross-height entity counts",
            ],
            "derived_without_rendering": ["T2 sparse subsets", "T5 two-view pairs"],
            "not_claimed": [
                "Needs-review episodes are not counted as trainable passes",
                "Rejected scene attempts are provenance, not training trajectories",
                "T4 has no semantic facing label without a certified canonical front",
                "Legacy T1 metadata cannot fully certify T6 doorway transitions",
                "No model improvement is claimed before controlled training experiments",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1-bundle", type=Path, required=True)
    parser.add_argument("--t3-bundle", type=Path, required=True)
    parser.add_argument(
        "--t3-sweep-plan",
        type=Path,
        action="append",
        required=True,
        help="Repeat for each disjoint T3 acquisition sweep.",
    )
    parser.add_argument("--static-sweep-plan", type=Path)
    parser.add_argument("--t4-sweep-plan", type=Path, required=True)
    parser.add_argument("--t7-sweep-plan", type=Path, required=True)
    parser.add_argument("--t8-sweep-plan", type=Path, required=True)
    parser.add_argument("--t10-sweep-plan", type=Path, required=True)
    parser.add_argument("--t9-release", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--media-url-root", default="data/trajectory_media")
    parser.add_argument("--details-root", type=Path, required=True)
    parser.add_argument("--details-url-root", default="data/trajectory_details")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()
    catalog = build_catalog(
        t1_bundle=args.t1_bundle.resolve(),
        t3_bundle=args.t3_bundle.resolve(),
        media_root=args.media_root.resolve(),
        media_url_root=args.media_url_root,
        details_root=args.details_root.resolve(),
        details_url_root=args.details_url_root,
        static_sweep_plan=(
            args.static_sweep_plan.resolve() if args.static_sweep_plan else None
        ),
        t3_sweep_plans=[path.resolve() for path in args.t3_sweep_plan],
        typed_sweep_plans={
            "T4": args.t4_sweep_plan.resolve(),
            "T7": args.t7_sweep_plan.resolve(),
            "T8": args.t8_sweep_plan.resolve(),
            "T10": args.t10_sweep_plan.resolve(),
        },
        t9_release_path=args.t9_release.resolve(),
    )
    _write_json(args.output, catalog)
    if args.manifest_output:
        _write_json(args.manifest_output, catalog)


if __name__ == "__main__":
    main()
