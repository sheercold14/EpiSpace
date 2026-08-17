#!/usr/bin/env python3
"""Extract unique rendered-occlusion bindings from an immutable coverage dataset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    by_scene: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    source_ids = []
    for episode in dataset["episodes"]:
        if episode["source_capability"] != "self_motion_update_occluded":
            continue
        binding = {str(key): str(value) for key, value in episode["binding"].items()}
        canonical = json.dumps(binding, sort_keys=True, separators=(",", ":"))
        by_scene[episode["scene_key"]][canonical] = binding
        source_ids.append(episode["episode_id"])
    payload = {
        "schema_version": "epispace.occlusion_binding_allowlist.v1",
        "source_collection_id": dataset.get("collection_id"),
        "source_episode_count": len(source_ids),
        "unique_binding_count": sum(len(rows) for rows in by_scene.values()),
        "bindings_by_scene": {
            scene: [rows[key] for key in sorted(rows)]
            for scene, rows in sorted(by_scene.items())
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
