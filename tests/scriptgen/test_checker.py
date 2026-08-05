"""Checker expression-language and failure-path tests."""

from __future__ import annotations

import pytest

from spatial_episode.scriptgen.checker import (
    ScriptError,
    check_clauses,
    eval_frame_range,
    eval_int_expr,
)
from spatial_episode.scriptgen.library import SELF_MOTION
from spatial_episode.scriptgen.sceneview import (
    GeometrySceneView,
    Pose2D,
    SceneLayout,
    SceneObject,
)
from spatial_episode.scriptgen.standards import STD_V1


def test_eval_int_expr() -> None:
    assert eval_int_expr("$t_seen+1", {"t_seen": 4}) == 5
    assert eval_int_expr("$t_q-$t_seen", {"t_q": 9, "t_seen": 4}) == 5
    with pytest.raises(ScriptError):
        eval_int_expr("$missing+1", {})
    with pytest.raises(ScriptError):
        eval_int_expr("__import__('os')", {})


def test_eval_frame_range_inclusive_and_clamped() -> None:
    assert eval_frame_range("0:$t", {"t": 3}, frame_count=10) == [0, 1, 2, 3]
    assert eval_frame_range("$t+1:99", {"t": 3}, frame_count=6) == [4, 5]


def test_unresolvable_frame_var_is_soft_reject() -> None:
    """A trajectory that never sees the target fails cleanly, not with a crash."""
    layout = SceneLayout(
        scene_id="s",
        objects=(SceneObject(name="sofa", category="sofa", xy=(0.0, 3.0), size_m=1.5, uid="u"),),
    )
    # All poses face away from the sofa: last_visible($target) has no answer.
    poses = tuple(Pose2D(0.0, 0.0, -90.0) for _ in range(8))
    view = GeometrySceneView(layout=layout, poses=poses, std=STD_V1)
    report = check_clauses(view, SELF_MOTION, {"target": "sofa"}, STD_V1)
    assert report.passed is False
    assert report.failed_clause == "t_seen"
