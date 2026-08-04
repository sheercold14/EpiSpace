#!/usr/bin/env python3
"""Build or replay the curated claim-grounded QA pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main() -> None:
    from episode3d.qa_generation.backends import ReplayBackend
    from episode3d.qa_generation.pipeline import build_qa_generation_pilot

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/qa_generation_pilot_v1.json")
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Require existing content-addressed Codex responses; never call a model.",
    )
    arguments = parser.parse_args()
    backend = None
    if arguments.replay:
        config = json.loads(arguments.config.read_text(encoding="utf-8"))
        cache = (arguments.config.parent / config["output"]["cache_directory"]).resolve()
        provider = str(config["backend"].get("provider", "codex-cli"))
        model = str(config["backend"].get("model", "default"))
        backend = ReplayBackend(cache, provider=provider, model=model)
    manifest = build_qa_generation_pilot(arguments.config, backend=backend)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
