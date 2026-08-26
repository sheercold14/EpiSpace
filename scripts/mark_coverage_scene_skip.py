#!/usr/bin/env python
"""Atomically record an explicit scene skip in a coverage search manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from spatial_episode.scriptgen.binding_coverage import CoverageManifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()

    manifest = CoverageManifest.model_validate_json(args.manifest.read_text(encoding="utf-8"))
    known_scenes = {scene.scene_key for scene in manifest.scenes}
    if args.scene not in known_scenes:
        raise ValueError(f"unknown scene: {args.scene}")
    skips = dict(manifest.scene_skips)
    existing = skips.get(args.scene)
    if existing is not None and existing != args.reason:
        raise ValueError(f"scene already has a different skip reason: {existing}")
    skips[args.scene] = args.reason
    updated = manifest.model_copy(update={"scene_skips": skips})
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(updated.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.manifest)
    print(f"coverage scene_skip_recorded={args.scene} reason={args.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
