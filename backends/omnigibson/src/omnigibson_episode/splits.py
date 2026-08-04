"""Create a deterministic, domain-stratified scene-disjoint split manifest."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from omnigibson_episode.io import sha256_file, write_json_atomic


def _allocate(counts: dict[str, int], target: int) -> dict[str, int]:
    total = sum(counts.values())
    if total == 0 or target == 0:
        return {key: 0 for key in counts}
    exact = {key: value * target / total for key, value in counts.items()}
    result = {key: min(value, math.floor(exact[key])) for key, value in counts.items()}
    remaining = target - sum(result.values())
    order = sorted(
        counts,
        key=lambda key: (exact[key] - math.floor(exact[key]), counts[key], key),
        reverse=True,
    )
    while remaining:
        changed = False
        for key in order:
            if result[key] < counts[key]:
                result[key] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            raise ValueError("cannot allocate requested split size")
    return result


def _rank(scene_model: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{scene_model}".encode()).hexdigest()


def build_split_manifest(
    inventory_path: Path,
    output_path: Path | None = None,
    *,
    validation_count: int = 7,
    test_count: int = 7,
    salt: str = "episode3d-scene-split-v1",
) -> dict[str, Any]:
    inventory_path = inventory_path.resolve()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    scenes = [
        scene for scene in inventory["scenes"] if scene["eligibility"]["static_multiview"]
    ]
    if validation_count + test_count >= len(scenes):
        raise ValueError("validation and test splits leave no training scenes")
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scene in scenes:
        by_domain[scene["domain"]].append(scene)
    for domain in by_domain:
        by_domain[domain].sort(key=lambda scene: _rank(scene["scene_model"], salt))

    domain_counts = {domain: len(items) for domain, items in by_domain.items()}
    test_quota = _allocate(domain_counts, test_count)
    remaining_counts = {
        domain: domain_counts[domain] - test_quota[domain] for domain in domain_counts
    }
    validation_quota = _allocate(remaining_counts, validation_count)
    assignments = []
    for domain, items in sorted(by_domain.items()):
        for index, scene in enumerate(items):
            if index < test_quota[domain]:
                split = "test"
            elif index < test_quota[domain] + validation_quota[domain]:
                split = "validation"
            else:
                split = "train"
            assignments.append(
                {
                    "scene_model": scene["scene_model"],
                    "domain": domain,
                    "classification": scene["classification"],
                    "split": split,
                    "split_group": f"scene:{scene['scene_model']}",
                }
            )
    assignments.sort(key=lambda item: item["scene_model"])
    split_counts = Counter(item["split"] for item in assignments)
    manifest = {
        "schema_version": "episode3d_scene_split.v1",
        "split_id": "behavior-1k-indoor-scene-disjoint-v1",
        # A public split manifest carries a content-addressed reference, not the
        # producer's machine-specific absolute path.
        "inventory": inventory_path.name,
        "inventory_sha256": sha256_file(inventory_path),
        "salt": salt,
        "policy": {
            "unit": "scene_model",
            "domain_stratified": True,
            "family_lock": (
                "all trajectories, interventions and language variants from a scene "
                "inherit its split"
            ),
        },
        "counts": dict(sorted(split_counts.items())),
        "assignments": assignments,
    }
    if output_path is not None:
        write_json_atomic(output_path, manifest)
    return manifest
