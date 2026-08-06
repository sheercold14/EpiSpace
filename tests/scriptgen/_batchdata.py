"""Test data access: the three rendered scripted-demo trajectories.

Integration tests run against real rendered bundles (no Isaac Sim needed —
only their exported npz masks and JSON truths) and skip cleanly on hosts
without the acquisition outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from spatial_episode.scriptgen.behavior import RenderSceneView
from spatial_episode.scriptgen.standards import STD_V1

BATCH_ROOT = Path(
    "/data/shichao/data/dataV100/code/OminiGibson/outputs/scripted_demo/batch"
)
SCENE_IR_PATH = Path(
    "/data/shichao/data/dataV100/code/OminiGibson/outputs/sweeps/"
    "t10-target-view-seed17-v1/bundles/gates_bedroom_t10_seed17/scene_ir.json"
)

needs_batch = pytest.mark.skipif(
    not (BATCH_ROOT / "render_0").exists() or not SCENE_IR_PATH.exists(),
    reason="scripted-demo rendered batch not present on this host",
)


def load_plan_record(index: int) -> dict[str, Any]:
    return json.loads((BATCH_ROOT / f"plan_{index}.record.json").read_text(encoding="utf-8"))


def load_render_view(index: int) -> RenderSceneView:
    return RenderSceneView.from_bundle(
        BATCH_ROOT / f"render_{index}", STD_V1, scene_ir=SCENE_IR_PATH
    )
