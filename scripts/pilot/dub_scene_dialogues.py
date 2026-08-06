#!/usr/bin/env python3
"""Dub compiled scene dialogues into human-cognition natural language."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from episode3d.scene_llm_pipeline.dubbing import dub_scene


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    for name in args.scene:
        artifact = dub_scene(
            args.scenes_dir / f"{name}.dialogue.json",
            args.output_dir / f"{name}.dubbed.json",
            args.cache_dir,
            model=args.model,
        )
        fallbacks = sum(1 for r in artifact["rounds"] if r["dubbing"]["fallback"])
        print(f"{name}: rounds={len(artifact['rounds'])} fallbacks={fallbacks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
