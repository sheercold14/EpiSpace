#!/usr/bin/env python3
"""CLI wrapper for :mod:`episode3d.semantic_visual_audit`."""

from __future__ import annotations

import sys
from pathlib import Path


def _main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from episode3d.semantic_visual_audit import main

    return main()

if __name__ == "__main__":
    raise SystemExit(_main())
