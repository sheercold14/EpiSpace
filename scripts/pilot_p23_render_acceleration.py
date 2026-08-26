#!/usr/bin/env python3
"""Prepare and audit a lossless same-scene render batching pilot.

The pilot does not alter production manifests or bundles.  It compares ordinary
one-process-per-candidate renders with a single scene load that renders the same
camera poses in one combined bundle.  Combined artifacts can then be split back
into ordinary per-candidate bundles for the existing EpiSpace authority checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_group(
    index: dict[str, Any], tier: str, count: int, cell_id: str | None
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in index["trajectories"]:
        if row["tier"] == tier:
            grouped[(row["scene_key"], row["cell_id"])].append(row)
    eligible = [rows for rows in grouped.values() if len(rows) >= count]
    if cell_id is not None:
        eligible = [rows for rows in eligible if rows[0]["cell_id"] == cell_id]
    if not eligible:
        suffix = "" if cell_id is None else f" for cell {cell_id}"
        raise ValueError(
            f"no {tier} scene/cell contains {count} selected trajectories{suffix}"
        )
    # Prefer a compact early group so the pilot is deterministic and cheap.
    rows = min(eligible, key=lambda items: (sum(_view_count(row) for row in items[:count]), items[0]["ordinal"]))
    return rows[:count]


def _view_count(row: dict[str, Any]) -> int:
    plan = _read_json(Path(row["render_plan"]))
    return len(plan["views"]) + len(plan.get("auxiliary_views", ()))


def prepare(
    index_path: Path,
    tier: str,
    count: int,
    output_root: Path,
    cell_id: str | None = None,
) -> Path:
    index = _read_json(index_path.resolve())
    rows = _selected_group(index, tier, count, cell_id)
    tier_root = output_root.resolve() / tier
    tier_root.mkdir(parents=True, exist_ok=False)

    plans = [_read_json(Path(row["render_plan"])) for row in rows]
    first_auxiliary = plans[0].get("auxiliary_views", [])
    if tier == "p2":
        canonical = json.dumps(first_auxiliary, sort_keys=True, separators=(",", ":"))
        for plan in plans[1:]:
            other = json.dumps(plan.get("auxiliary_views", []), sort_keys=True, separators=(",", ":"))
            if other != canonical:
                raise ValueError("P2 pilot group does not have identical auxiliary views")

    combined_views = json.loads(json.dumps(plans[0]["views"]))
    combined_auxiliary = json.loads(json.dumps(first_auxiliary))
    mappings: list[dict[str, Any]] = []
    mappings.append(
        {
            "candidate_id": rows[0]["candidate_id"],
            "main": {view["view_id"]: ["views", view["view_id"]] for view in plans[0]["views"]},
            "auxiliary": {
                view["view_id"]: ["auxiliary_views", view["view_id"]]
                for view in first_auxiliary
            },
        }
    )
    for candidate_index, (row, plan) in enumerate(zip(rows[1:], plans[1:], strict=True), start=1):
        main_mapping: dict[str, list[str]] = {}
        for view in plan["views"]:
            original_id = str(view["view_id"])
            combined_id = f"c{candidate_index:02d}--main--{original_id}"
            combined = json.loads(json.dumps(view))
            combined["view_id"] = combined_id
            combined["purpose"] = "acceleration_pilot_main_view"
            transform = combined["world_from_agent"]
            transform["translation_m"] = [
                round(float(value), 6) for value in transform["translation_m"]
            ]
            transform["rotation_xyzw"] = [
                round(float(value), 9) for value in transform["rotation_xyzw"]
            ]
            combined_auxiliary.append(combined)
            main_mapping[original_id] = ["auxiliary_views", combined_id]
        auxiliary_mapping = {
            view["view_id"]: ["auxiliary_views", view["view_id"]]
            for view in first_auxiliary
        }
        mappings.append(
            {
                "candidate_id": row["candidate_id"],
                "main": main_mapping,
                "auxiliary": auxiliary_mapping,
            }
        )

    combined_plan = {
        **plans[0],
        "plan_id": f"acceleration-pilot.{tier}.{rows[0]['scene_key']}",
        "views": combined_views,
        "auxiliary_views": combined_auxiliary,
    }
    combined_plan_path = tier_root / "combined.views.json"
    _write_json(combined_plan_path, combined_plan)

    recipe_path = Path(rows[0]["recipe"])
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    recipe["recipe_id"] = f"acceleration_pilot_{tier}_{rows[0]['scene_key']}"
    recipe["trajectory"]["plan_path"] = str(combined_plan_path)
    combined_recipe_path = tier_root / "combined.recipe.yaml"
    combined_recipe_path.write_text(
        yaml.safe_dump(recipe, sort_keys=False),
        encoding="utf-8",
    )

    jobs = []
    for row in rows:
        jobs.append(
            {
                **row,
                "baseline_bundle": str(tier_root / "baseline" / row["candidate_id"]),
                "baseline_log": str(tier_root / "logs" / f"baseline.{row['candidate_id']}.log"),
                "split_bundle": str(tier_root / "split" / row["candidate_id"]),
            }
        )
    payload = {
        "schema_version": "p23_render_acceleration_pilot.v1",
        "tier": tier,
        "scene_key": rows[0]["scene_key"],
        "cell_id": rows[0]["cell_id"],
        "candidate_count": len(rows),
        "ordinary_total_views": sum(_view_count(row) for row in rows),
        "combined_total_views": len(combined_views) + len(combined_auxiliary),
        "combined_plan": str(combined_plan_path),
        "combined_recipe": str(combined_recipe_path),
        "combined_bundle": str(tier_root / "combined.bundle"),
        "combined_log": str(tier_root / "logs" / "combined.log"),
        "jobs": jobs,
        "mappings": mappings,
    }
    pilot_path = tier_root / "pilot.json"
    _write_json(pilot_path, payload)
    print(pilot_path)
    return pilot_path


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _entry_index(report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (subdir, str(entry["view_id"])): entry
        for subdir in ("views", "auxiliary_views")
        for entry in report.get(subdir, [])
    }


def split(pilot_path: Path) -> None:
    pilot = _read_json(pilot_path.resolve())
    combined_root = Path(pilot["combined_bundle"])
    combined_report = _read_json(combined_root / "render_report.json")
    combined_entries = _entry_index(combined_report)
    mapping_by_candidate = {row["candidate_id"]: row for row in pilot["mappings"]}

    for job in pilot["jobs"]:
        candidate_id = job["candidate_id"]
        baseline_root = Path(job["baseline_bundle"])
        split_root = Path(job["split_bundle"])
        if split_root.exists():
            raise FileExistsError(split_root)
        split_root.mkdir(parents=True)
        for name in ("trajectory_plan.json", "trajectory_selection.json"):
            shutil.copy2(baseline_root / name, split_root / name)
        shutil.copy2(combined_root / "scene_snapshot.json", split_root / "scene_snapshot.json")

        baseline_report = _read_json(baseline_root / "render_report.json")
        mapping = mapping_by_candidate[candidate_id]
        rewritten: dict[str, list[dict[str, Any]]] = {"views": [], "auxiliary_views": []}
        for target_subdir, mapping_key in (("views", "main"), ("auxiliary_views", "auxiliary")):
            baseline_by_id = {
                str(entry["view_id"]): entry for entry in baseline_report.get(target_subdir, [])
            }
            for step, (original_id, source_key) in enumerate(mapping[mapping_key].items()):
                source_subdir, source_id = source_key
                source_entry = combined_entries[(source_subdir, source_id)]
                source_artifact = Path(source_entry["artifact"]["path"])
                destination = split_root / target_subdir / f"{original_id}.sensors.npz"
                _link(source_artifact, destination)
                entry = json.loads(json.dumps(source_entry))
                entry["view_id"] = original_id
                entry["step"] = step if target_subdir == "views" else len(mapping["main"]) + step
                entry["artifact"] = {
                    **entry["artifact"],
                    "path": str(destination.resolve()),
                    "sha256": _sha256(destination),
                    "byte_size": destination.stat().st_size,
                }
                if target_subdir == "views":
                    for key in ("purpose", "target_entity_id", "yaw_offset_deg", "geometry_label"):
                        entry.pop(key, None)
                else:
                    template = baseline_by_id[original_id]
                    for key in ("purpose", "target_entity_id", "yaw_offset_deg", "geometry_label"):
                        entry[key] = template.get(key)
                rewritten[target_subdir].append(entry)
        report = {
            **baseline_report,
            "views": rewritten["views"],
            "auxiliary_views": rewritten["auxiliary_views"],
        }
        _write_json(split_root / "render_report.json", report)


def _array_metrics(left: Path, right: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    with np.load(left) as ordinary, np.load(right) as batched:
        if set(ordinary.files) != set(batched.files):
            raise ValueError(f"array keys differ: {left} {right}")
        for name in ordinary.files:
            a = ordinary[name]
            b = batched[name]
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError(f"array contract differs for {name}: {left} {right}")
            if np.issubdtype(a.dtype, np.integer) and name != "rgb":
                result[name] = {"exact_fraction": float(np.mean(a == b))}
            else:
                finite = np.ones(a.shape, dtype=bool) if name == "rgb" else np.isfinite(a) & np.isfinite(b)
                delta = np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64))
                row = {
                    "finite_fraction": float(np.mean(finite)),
                    "exact_fraction": float(np.mean(a == b)),
                    "mean_absolute_error": float(delta.mean()) if delta.size else math.nan,
                    "maximum_absolute_error": float(delta.max()) if delta.size else math.nan,
                }
                if name == "rgb" and delta.size:
                    mse = float(np.mean(delta * delta))
                    row["psnr_db"] = math.inf if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
                result[name] = row
    return result


def compare(pilot_path: Path, output: Path | None) -> dict[str, Any]:
    pilot = _read_json(pilot_path.resolve())
    comparisons = []
    for job in pilot["jobs"]:
        baseline_root = Path(job["baseline_bundle"])
        split_root = Path(job["split_bundle"])
        files = sorted(
            path.relative_to(baseline_root)
            for subdir in ("views", "auxiliary_views")
            for path in (baseline_root / subdir).glob("*.sensors.npz")
        )
        comparisons.append(
            {
                "candidate_id": job["candidate_id"],
                "views": {
                    str(relative): _array_metrics(baseline_root / relative, split_root / relative)
                    for relative in files
                },
            }
        )
    result = {
        "schema_version": "p23_render_acceleration_comparison.v1",
        "tier": pilot["tier"],
        "candidate_count": pilot["candidate_count"],
        "ordinary_total_views": pilot["ordinary_total_views"],
        "combined_total_views": pilot["combined_total_views"],
        "comparisons": comparisons,
    }
    if output is not None:
        _write_json(output.resolve(), result)
    print(json.dumps(result, indent=2))
    return result


def run(
    pilot_path: Path,
    mode: str,
    gpu_id: int,
    og_root: Path,
    data_root: Path,
    conda_env: str,
) -> dict[str, Any]:
    from spatial_episode.scriptgen.single import _render

    pilot = _read_json(pilot_path.resolve())
    timing_path = pilot_path.resolve().parent / f"timing.{mode}.json"
    if timing_path.exists():
        raise FileExistsError(timing_path)
    if mode == "baseline":
        jobs = [
            {
                "candidate_id": job["candidate_id"],
                "recipe": Path(job["recipe"]),
                "bundle": Path(job["baseline_bundle"]),
                "log": Path(job["baseline_log"]),
            }
            for job in pilot["jobs"]
        ]
    else:
        jobs = [
            {
                "candidate_id": "combined",
                "recipe": Path(pilot["combined_recipe"]),
                "bundle": Path(pilot["combined_bundle"]),
                "log": Path(pilot["combined_log"]),
            }
        ]
    rows = []
    started = time.perf_counter()
    for job in jobs:
        job_started = time.perf_counter()
        rendered, reason = _render(
            recipe=job["recipe"],
            bundle=job["bundle"],
            log_path=job["log"],
            og_root=og_root.resolve(),
            data_root=data_root.resolve(),
            conda_env=conda_env,
            gpu_id=gpu_id,
            timeout_minutes=30,
        )
        rows.append(
            {
                "candidate_id": job["candidate_id"],
                "rendered": rendered,
                "reason": reason,
                "wall_seconds": round(time.perf_counter() - job_started, 3),
            }
        )
        if not rendered:
            break
    result = {
        "schema_version": "p23_render_acceleration_timing.v1",
        "tier": pilot["tier"],
        "mode": mode,
        "gpu_id": gpu_id,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "jobs": rows,
    }
    _write_json(timing_path, result)
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--index", type=Path, required=True)
    prepare_parser.add_argument("--tier", choices=("p2", "p3"), required=True)
    prepare_parser.add_argument("--count", type=int, default=4)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--cell-id")
    split_parser = commands.add_parser("split")
    split_parser.add_argument("--pilot", type=Path, required=True)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--pilot", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--pilot", type=Path, required=True)
    run_parser.add_argument("--mode", choices=("baseline", "combined"), required=True)
    run_parser.add_argument("--gpu-id", type=int, required=True)
    run_parser.add_argument("--og-root", type=Path, required=True)
    run_parser.add_argument("--data-root", type=Path, required=True)
    run_parser.add_argument("--conda-env", default="behavior-spatialep")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.index, args.tier, args.count, args.output_root, args.cell_id)
    elif args.command == "split":
        split(args.pilot)
    elif args.command == "compare":
        compare(args.pilot, args.output)
    else:
        run(args.pilot, args.mode, args.gpu_id, args.og_root, args.data_root, args.conda_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
