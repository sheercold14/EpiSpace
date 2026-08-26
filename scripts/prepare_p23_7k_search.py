#!/usr/bin/env python
"""Freeze the measured quality contract and P3 binding set for the 7k search."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

P2_TARGET = 3500
P3_TARGETS = {1: 1750, 2: 1050, 3: 700}
P2_MAX_PER_BINDING = 10
P3_MAX_PER_BINDING = 12


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p3-pilot-manifest", type=Path, required=True)
    parser.add_argument("--capacity-report", type=Path, required=True)
    parser.add_argument("--p3-wide-capacity-report", type=Path, required=True)
    parser.add_argument("--allowlist-output", type=Path, required=True)
    parser.add_argument("--contract-output", type=Path, required=True)
    args = parser.parse_args()

    manifest = _read(args.p3_pilot_manifest)
    capacity = _read(args.capacity_report)
    p3_wide_capacity = _read(args.p3_wide_capacity_report)
    selected: dict[str, list[dict]] = {f"k{k}": [] for k in (1, 2, 3)}
    by_scene: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for cell in manifest["cells"]:
        capability = cell["capability"]
        if not capability.startswith("cross_view_ego_k"):
            continue
        k = int(capability.rsplit("_k", 1)[1])
        row = {"scene_key": cell["scene_key"], "binding": cell["binding"]}
        selected[f"k{k}"].append(row)
        by_scene[cell["scene_key"]].append(cell["binding"])

    expected = {1: 182, 2: 109, 3: 73}
    observed = {k: len(selected[f"k{k}"]) for k in expected}
    if observed != expected:
        raise RuntimeError(f"P3 pilot binding set changed: {observed} != {expected}")

    allowlist = {
        "schema_version": "p23_7k_binding_allowlist.v1",
        "source_manifest": str(args.p3_pilot_manifest.resolve()),
        "source_manifest_sha256": _sha256(args.p3_pilot_manifest),
        "bindings_by_scene": dict(sorted(by_scene.items())),
        "selected": selected,
    }
    contract = {
        "schema_version": "p23_7k_search_contract.v1",
        "physical_trajectory_target": P2_TARGET + sum(P3_TARGETS.values()),
        "allocation": {
            "p2": P2_TARGET,
            "p3": {f"k{k}": target for k, target in P3_TARGETS.items()},
        },
        "allocation_policy": {
            "p2_preferred": P2_TARGET,
            "p2_minimum": 3000,
            "p3_fills_p2_shortfall": True,
            "p3_k_ratio": {"k1": 50, "k2": 30, "k3": 20},
        },
        "quality": {
            "global_exact_pose_duplicates_allowed_per_scene": 0,
            "pairwise_structural_diversity_required_within_scene_and_tier": True,
            "p2_max_trajectories_per_binding": P2_MAX_PER_BINDING,
            "p2_minimum_distinct_bindings": (P2_TARGET + P2_MAX_PER_BINDING - 1)
            // P2_MAX_PER_BINDING,
            "p3_max_trajectories_per_binding": P3_MAX_PER_BINDING,
            "p3_minimum_distinct_bindings": {
                f"k{k}": (target + P3_MAX_PER_BINDING - 1) // P3_MAX_PER_BINDING
                for k, target in P3_TARGETS.items()
            },
            "p3_walking_only": True,
            "marker_min_badged_frames": 2,
        },
        "search": {
            "p2": {
                "seed_namespace": "p23_p2_badged_v2",
                "reference_pair_frontier": 256,
                "binding_limit_per_scene": 128,
                "accepted_diverse_cap": P2_MAX_PER_BINDING,
                "attempts_per_binding": 80,
                "raw_plan_oversample": 8,
            },
            "p3": {
                "seed_namespace": "p23_p3stream_badged_v1",
                "binding_counts": {f"k{k}": count for k, count in observed.items()},
                "accepted_diverse_cap": P3_MAX_PER_BINDING,
                "attempts_per_binding": 80,
                "raw_plan_oversample": 5,
            },
        },
        "capacity_report": str(args.capacity_report.resolve()),
        "capacity_report_sha256": _sha256(args.capacity_report),
        "p3_wide_capacity_report": str(args.p3_wide_capacity_report.resolve()),
        "p3_wide_capacity_report_sha256": _sha256(args.p3_wide_capacity_report),
        "measured_diverse_yield": {
            "p2": capacity["groups"]["p2"]["diverse_yield"],
            **{
                group: {
                    **p3_wide_capacity["groups"][group]["diverse_yield"],
                    "sampled_bindings": len(p3_wide_capacity["groups"][group]["rows"]),
                    "mean_after_per_binding_cap": round(
                        sum(
                            min(P3_MAX_PER_BINDING, row["diverse_plans"])
                            for row in p3_wide_capacity["groups"][group]["rows"]
                        )
                        / len(p3_wide_capacity["groups"][group]["rows"]),
                        3,
                    ),
                }
                for group in ("p3_k1", "p3_k2", "p3_k3")
            },
        },
    }
    _write(args.allowlist_output, allowlist)
    _write(args.contract_output, contract)
    print(json.dumps(contract, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
