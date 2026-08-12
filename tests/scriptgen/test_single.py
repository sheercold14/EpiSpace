"""Safety and CLI surface checks for the one-episode closure command."""

from __future__ import annotations

from pathlib import Path

import pytest

from spatial_episode.scriptgen.dataset_cli import parser
from spatial_episode.scriptgen.single import _validate_output_root


def test_run_one_cli_parses_the_real_scene_inputs() -> None:
    args = parser().parse_args(
        [
            "run-one",
            "--scene-ir",
            "scene_ir.json",
            "--source-recipe",
            "source.yaml",
            "--output-root",
            "output",
            "--capability",
            "self_motion_update_occluded",
            "--og-root",
            "og",
            "--data-root",
            "data",
        ]
    )

    assert args.command == "run-one"
    assert args.attempts_per_binding == 150
    assert args.candidate_plans == args.max_render_candidates == 10


def test_single_run_output_cannot_contain_its_inputs(tmp_path: Path) -> None:
    output = tmp_path / "output"
    scene_ir = output / "scene_ir.json"

    with pytest.raises(ValueError, match="contains required input"):
        _validate_output_root(output.resolve(), scene_ir.resolve())


def test_single_run_output_rejects_workspace_root() -> None:
    with pytest.raises(ValueError, match="broad protected output root"):
        _validate_output_root(Path.cwd().resolve())
