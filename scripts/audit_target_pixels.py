"""Measure how large question targets actually appear in rendered frames.

Reads a rendered collection root, and for every compiled family reports the
pixel area its target occupies in each frame of the compiled sequence, together
with the camera distance at that frame.  Used to calibrate the compile-time
groundability gate (a target nobody can see is not a groundable referent).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def _scene_entity_names(scene_ir_path: Path) -> dict[str, str]:
    payload = json.loads(scene_ir_path.read_text(encoding="utf-8"))
    return {
        entity["entity_id"]: entity["source_entity_id"] for entity in payload["entities"]
    }


def _scene_entity_extents(scene_ir_path: Path) -> dict[str, float]:
    payload = json.loads(scene_ir_path.read_text(encoding="utf-8"))
    extents: dict[str, float] = {}
    for entity in payload["entities"]:
        half = entity["obb"]["half_extents_m"]
        extents[entity["entity_id"]] = 2.0 * max(float(value) for value in half)
    return extents


def _scene_entity_centers(scene_ir_path: Path) -> dict[str, tuple[float, float]]:
    payload = json.loads(scene_ir_path.read_text(encoding="utf-8"))
    centers: dict[str, tuple[float, float]] = {}
    for entity in payload["entities"]:
        center = entity["obb"]["center_m"]
        centers[entity["entity_id"]] = (float(center[0]), float(center[1]))
    return centers


def _scene_ir_paths(plan_path: Path) -> dict[str, Path]:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    return {
        scene["scene_key"]: Path(scene["scene_ir"]) for scene in payload["scenes"]
    }


def _runtime_ids_by_name(scene_snapshot: Path) -> dict[str, int]:
    payload = json.loads(scene_snapshot.read_text(encoding="utf-8"))
    registry = payload["runtime_instance_registry"]
    return {name: int(runtime_id) for runtime_id, name in registry.items()}


def _frame_pixels(view_npz: Path, runtime_id: int) -> int:
    with np.load(view_npz) as sensors:
        instance = sensors["instance_id"]
        return int(np.count_nonzero(instance == runtime_id))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    root: Path = args.output_root
    scene_irs = _scene_ir_paths(root / "coverage.plan.json")
    names_cache: dict[str, dict[str, str]] = {}
    extent_cache: dict[str, dict[str, float]] = {}
    center_cache: dict[str, dict[str, tuple[float, float]]] = {}

    records: list[dict[str, object]] = []
    for group_dir in sorted((root / "groups").iterdir()):
        if not group_dir.is_dir():
            continue
        scene_key = group_dir.name.split("__", 1)[0]
        if scene_key not in names_cache:
            names_cache[scene_key] = _scene_entity_names(scene_irs[scene_key])
            extent_cache[scene_key] = _scene_entity_extents(scene_irs[scene_key])
            center_cache[scene_key] = _scene_entity_centers(scene_irs[scene_key])
        bundle_dir = root / "bundles" / group_dir.name
        snapshot = bundle_dir / "scene_snapshot.json"
        if not snapshot.exists():
            continue
        runtime_ids = _runtime_ids_by_name(snapshot)

        for family_path in sorted(group_dir.glob("*/family.json")):
            family = json.loads(family_path.read_text(encoding="utf-8"))
            labels = {
                episode["label"]
                for episode in family["episodes"]
                if episode.get("kind") == "canonical"
            }
            for role, referent in family["referents"].items():
                entity = referent["entity_id"]
                source_name = names_cache[scene_key].get(entity)
                runtime_id = runtime_ids.get(source_name) if source_name else None
                if runtime_id is None:
                    records.append(
                        {
                            "group": group_dir.name,
                            "capability": family["capability"],
                            "role": role,
                            "category": referent["category"],
                            "status": "unmapped",
                        }
                    )
                    continue
                size_m = extent_cache[scene_key].get(entity, float("nan"))
                center = center_cache[scene_key].get(entity)
                per_frame: list[dict[str, float]] = []
                for frame in family["frames"]:
                    view_npz = (
                        bundle_dir / "views" / f"view-{int(frame['frame']):03d}.sensors.npz"
                    )
                    if not view_npz.exists():
                        continue
                    pixels = _frame_pixels(view_npz, runtime_id)
                    distance = float("nan")
                    if center is not None:
                        distance = math.hypot(
                            center[0] - float(frame["x"]), center[1] - float(frame["y"])
                        )
                    per_frame.append(
                        {
                            "frame": int(frame["frame"]),
                            "pixels": pixels,
                            "distance_m": distance,
                        }
                    )
                records.append(
                    {
                        "group": group_dir.name,
                        "capability": family["capability"],
                        "role": role,
                        "category": referent["category"],
                        "entity_id": entity,
                        "source_entity_id": source_name,
                        "size_m": size_m,
                        "labels": sorted(labels),
                        "frames": per_frame,
                        "max_pixels": max((item["pixels"] for item in per_frame), default=0),
                        "status": "measured",
                    }
                )

    measured = [item for item in records if item["status"] == "measured"]
    print(f"families measured: {len(measured)} (unmapped: {len(records) - len(measured)})")

    line = "reference_frame_transform"
    families = {(item["group"], item["capability"]) for item in measured}
    print(f"distinct families: {len(families)}")

    by_role: dict[str, list[int]] = defaultdict(list)
    for item in measured:
        key = f"{'P2' if str(item['capability']).startswith(line[:15]) else 'P3'}:{item['role']}"
        by_role[key].append(int(item["max_pixels"]))
    print("\nbest-frame referent pixel area (1024x1024 = 1048576 px)")
    for role in sorted(by_role):
        values = sorted(by_role[role])
        percentiles = np.percentile(values, [0, 10, 50, 90, 100])
        formatted = "  ".join(f"{value:>8.0f}" for value in percentiles)
        print(f"  {role:<28} n={len(values):>4}  min/p10/p50/p90/max {formatted}")

    print("\nweakest 20 referents by best-frame pixel area")
    seen: set[tuple[str, str, str]] = set()
    shown = 0
    for item in sorted(measured, key=lambda entry: entry["max_pixels"]):
        key = (str(item["group"]), str(item["role"]), str(item["entity_id"]))
        if key in seen:
            continue
        seen.add(key)
        best = max(item["frames"], key=lambda frame: frame["pixels"], default=None)
        distance = best["distance_m"] if best else float("nan")
        share = 100.0 * float(item["max_pixels"]) / (1024 * 1024)
        visible_frames = sum(1 for frame in item["frames"] if frame["pixels"] >= 256)
        print(
            f"  {item['max_pixels']:>7} px ({share:5.2f}%)  d={distance:5.1f}m  "
            f"size={item['size_m']:4.2f}m  seen_in={visible_frames}/{len(item['frames'])}  "
            f"{str(item['role']):<10} {item['category']:<22} {item['group']}"
        )
        shown += 1
        if shown >= 20:
            break

    if args.json_out:
        args.json_out.write_text(
            json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
