"""Predicate tests on a hand-built layout with known visibility."""

from __future__ import annotations

from spatial_episode.scriptgen.predicates import get_predicate
from spatial_episode.scriptgen.sceneview import (
    GeometrySceneView,
    Pose2D,
    SceneLayout,
    SceneObject,
    VisibilityObservation,
)
from spatial_episode.scriptgen.standards import STD_V1

LAYOUT = SceneLayout(
    scene_id="test_scene",
    objects=(SceneObject(name="sofa", category="sofa", xy=(0.0, 3.0), size_m=1.5, uid="u1"),),
)


def _view(poses: list[Pose2D]) -> GeometrySceneView:
    return GeometrySceneView(layout=LAYOUT, poses=tuple(poses), std=STD_V1)


def test_double_threshold_tristate() -> None:
    visible = VisibilityObservation("geom_ratio", 0.5, 1.0)
    invisible = VisibilityObservation("geom_ratio", 0.01, 1.0)
    ambiguous = VisibilityObservation("geom_ratio", 0.08, 1.0)
    assert visible.tristate(STD_V1) is True
    assert invisible.tristate(STD_V1) is False
    assert ambiguous.tristate(STD_V1) is None


def test_visible_then_invisible_after_turning_away() -> None:
    # Frame 0 faces the sofa from 3 m; frame 1 faces the opposite direction.
    facing = Pose2D(0.0, 0.0, 90.0)
    away = Pose2D(0.0, 0.0, -90.0)
    view = _view([facing, away])
    assert view.visibility("sofa", 0).tristate(STD_V1) is True
    assert view.visibility("sofa", 1).tristate(STD_V1) is False


def test_invisible_in_range_predicate_with_witness() -> None:
    away = Pose2D(0.0, 0.0, -90.0)
    view = _view([away, away, away])
    verdict = get_predicate("invisible_in_range")(view, STD_V1, obj="sofa", frames=[0, 1, 2])
    assert verdict.holds is True
    assert len(verdict.witness["frames"]) == 3


def test_sector_margin_ge_tighten_raises_bar() -> None:
    # Sofa azimuth 30° from front-facing pose at (0,0) yaw=60 → margin 15°.
    pose = Pose2D(0.0, 0.0, 60.0)
    view = _view([pose])
    loose = get_predicate("sector_margin_ge")(view, STD_V1, obj="sofa", frame=0)
    tight = get_predicate("sector_margin_ge")(view, STD_V1, obj="sofa", frame=0, tighten=True)
    assert loose.holds is True  # margin 15 >= required 15
    assert tight.holds is False  # tightened requirement 18 > 15
    assert loose.witness["sector"] == "front"


def test_cum_turn_ge() -> None:
    poses = [Pose2D(0, 0, 0.0), Pose2D(1, 0, 50.0), Pose2D(2, 0, 100.0)]
    view = _view(poses)
    verdict = get_predicate("cum_turn_ge")(view, STD_V1, frames=[0, 1, 2], deg=90)
    assert verdict.holds is True
    assert verdict.witness["cum_turn_deg"] == 100.0
