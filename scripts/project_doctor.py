#!/usr/bin/env python3
"""Fast repository and environment diagnostics for collaborators."""

from __future__ import annotations

import importlib
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(label: str, passed: bool, detail: str) -> bool:
    marker = "PASS" if passed else "FAIL"
    print(f"[{marker}] {label}: {detail}")
    return passed


def main() -> int:
    results: list[bool] = []
    results.append(
        check(
            "python",
            sys.version_info >= (3, 10),
            f"{platform.python_version()} ({sys.executable})",
        )
    )

    for module_name in ("numpy", "PIL", "yaml", "episode3d"):
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            results.append(check(f"import {module_name}", True, str(version)))
        except Exception as error:  # pragma: no cover - diagnostic boundary
            results.append(check(f"import {module_name}", False, str(error)))

    required_paths = (
        ROOT / "src" / "episode3d",
        ROOT / "backends" / "omnigibson" / "src" / "omnigibson_episode",
        ROOT / "manifests" / "datasets",
        ROOT / "examples" / "transform_pilot" / "showcase.json",
    )
    for path in required_paths:
        results.append(check("repository path", path.exists(), str(path.relative_to(ROOT))))

    for manifest in sorted((ROOT / "manifests").rglob("*.json")):
        try:
            json.loads(manifest.read_text(encoding="utf-8"))
            results.append(check("manifest JSON", True, str(manifest.relative_to(ROOT))))
        except (OSError, json.JSONDecodeError) as error:
            results.append(check("manifest JSON", False, f"{manifest}: {error}"))

    workspace_keys = (
        "EPISPACE_DATA_ROOT",
        "OMNIGIBSON_ROOT",
        "OMNIGIBSON_OUTPUT_ROOT",
        "OMNIGIBSON_DATA_PATH",
        "MODEL_ROOT",
    )
    configured = [key for key in workspace_keys if os.environ.get(key)]
    print(
        f"[INFO] external workspace: {len(configured)}/{len(workspace_keys)} roots configured; "
        "not required for quickstart"
    )

    passed = all(results)
    print(f"doctor={'PASS' if passed else 'FAIL'} checks={len(results)}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

