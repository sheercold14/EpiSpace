"""Safety and CLI surface checks for the one-episode closure command."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from spatial_episode.scriptgen.dataset_cli import parser
from spatial_episode.scriptgen.single import (
    _render,
    _render_failure_is_retryable,
    _validate_output_root,
)


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


def _render_args(tmp_path: Path) -> dict[str, object]:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("recipe_id: test\n", encoding="utf-8")
    return {
        "recipe": recipe,
        "bundle": tmp_path / "bundle",
        "log_path": tmp_path / "render.log",
        "og_root": tmp_path,
        "data_root": tmp_path / "data",
        "conda_env": "behavior",
        "gpu_id": 0,
        "timeout_minutes": 1,
    }


def test_render_accepts_atomically_published_bundle_after_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _render_args(tmp_path)
    bundle = args["bundle"]
    assert isinstance(bundle, Path)
    bundle.mkdir()
    (bundle / "render_report.json").write_text(
        json.dumps({"status": "success"}), encoding="utf-8"
    )

    class Process:
        pid = 123

        def wait(self, timeout: int | None = None) -> int:
            return 1

    monkeypatch.setattr("spatial_episode.scriptgen.single.shutil.which", lambda _: "conda")
    monkeypatch.setattr(
        "spatial_episode.scriptgen.single.subprocess.Popen",
        lambda *a, **k: Process(),
    )

    assert _render(**args) == (True, None)  # type: ignore[arg-type]


def test_render_timeout_kills_process_group_but_keeps_complete_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _render_args(tmp_path)
    bundle = args["bundle"]
    assert isinstance(bundle, Path)
    bundle.mkdir()
    (bundle / "render_report.json").write_text(
        json.dumps({"status": "success"}), encoding="utf-8"
    )
    calls = 0
    killed: list[tuple[int, int]] = []

    class Process:
        pid = 456

        def wait(self, timeout: int | None = None) -> int:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise subprocess.TimeoutExpired("renderer", timeout)
            return -15

    monkeypatch.setattr("spatial_episode.scriptgen.single.shutil.which", lambda _: "conda")
    monkeypatch.setattr(
        "spatial_episode.scriptgen.single.subprocess.Popen",
        lambda *a, **k: Process(),
    )
    monkeypatch.setattr(
        "spatial_episode.scriptgen.single.os.killpg",
        lambda pid, sig: killed.append((pid, sig)),
    )

    assert _render(**args) == (True, None)  # type: ignore[arg-type]
    assert killed and killed[0][0] == 456


def test_render_preserves_backend_failure_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _render_args(tmp_path)
    bundle = args["bundle"]
    assert isinstance(bundle, Path)
    bundle.mkdir()
    (bundle / "failure_report.json").write_text(
        json.dumps(
            {
                "status": "failure",
                "error": {
                    "type": "RuntimeError",
                    "message": "auxiliary camera failed physics clearance: test",
                },
            }
        ),
        encoding="utf-8",
    )

    class Process:
        pid = 789

        def wait(self, timeout: int | None = None) -> int:
            return 1

    monkeypatch.setattr("spatial_episode.scriptgen.single.shutil.which", lambda _: "conda")
    monkeypatch.setattr(
        "spatial_episode.scriptgen.single.subprocess.Popen",
        lambda *a, **k: Process(),
    )

    ok, reason = _render(**args)  # type: ignore[arg-type]
    assert ok is False
    assert reason == (
        "renderer_failure:RuntimeError:"
        "auxiliary camera failed physics clearance: test"
    )
    assert _render_failure_is_retryable(reason) is False
    assert _render_failure_is_retryable("render_timeout_after_20_minutes") is True
