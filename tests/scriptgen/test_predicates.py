"""Predicate tests on a hand-built layout with known visibility."""

from __future__ import annotations

from spatial_episode.scriptgen.predicates import get_predicate
from spatial_episode.scriptgen.sceneview import (
    GeometrySceneView,
    Obstacle,
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


def test_partially_in_fov_is_ambiguous() -> None:
    """An object straddling the FOV edge must be ambiguous, never invisible.

    Regression for the first closed-loop render: the geometry backend claimed
    invisible while the render showed an 8k-pixel edge sliver.
    """
    # Sofa (1.5 m wide) at 3 m, center azimuth ~50 deg: outside the 45 deg
    # half-FOV, but its ~14 deg angular half-width keeps an edge inside.
    pose = Pose2D(0.0, 0.0, 40.0)  # sofa bearing 90 -> azimuth 50
    view = _view([pose])
    assert view.visibility("sofa", 0).tristate(STD_V1) is None
    # Fully outside (azimuth 90) stays definitely invisible.
    far_out = Pose2D(0.0, 0.0, 0.0)
    assert _view([far_out]).visibility("sofa", 0).tristate(STD_V1) is False


def _clearance_view(poses: list[Pose2D], *, z_low: float = 0.0, z_high: float = 1.0):
    obstacle = Obstacle(
        label="table",
        center_xy=(0.0, 0.0),
        half_extents_xy=(0.5, 0.25),
        yaw_deg=45.0,
        z_low=z_low,
        z_high=z_high,
    )
    layout = SceneLayout(scene_id="clearance", objects=(), obstacles=(obstacle,))
    return GeometrySceneView(layout=layout, poses=tuple(poses), std=STD_V1)


def test_poses_clear_rejects_pose_inside_rotated_obstacle() -> None:
    view = _clearance_view([Pose2D(0.0, 0.0, 0.0)])
    verdict = get_predicate("poses_clear")(view, STD_V1, frames=[0])
    assert verdict.holds is False
    assert verdict.witness["worst_collision"]["obstacle"] == "table"
    assert verdict.witness["worst_collision"]["penetration_depth_m"] > 0.0


def test_path_clear_rejects_segment_crossing_rotated_obstacle() -> None:
    view = _clearance_view([Pose2D(-2.0, 0.0, 0.0), Pose2D(2.0, 0.0, 0.0)])
    verdict = get_predicate("path_clear")(view, STD_V1, frames=[0, 1])
    assert verdict.holds is False
    assert verdict.witness["worst_collision"] == {"frames": [0, 1], "obstacle": "table"}


def test_clearance_ignores_obstacle_outside_body_height_band() -> None:
    view = _clearance_view(
        [Pose2D(-2.0, 0.0, 0.0), Pose2D(0.0, 0.0, 0.0)], z_low=2.0, z_high=2.5
    )
    assert get_predicate("poses_clear")(view, STD_V1, frames=[0, 1]).holds is True
    assert get_predicate("path_clear")(view, STD_V1, frames=[0, 1]).holds is True


def test_clear_trajectory_passes_both_clearance_predicates() -> None:
    view = _clearance_view([Pose2D(-2.0, 2.0, 0.0), Pose2D(2.0, 2.0, 0.0)])
    assert get_predicate("poses_clear")(view, STD_V1, frames=[0, 1]).holds is True
    assert get_predicate("path_clear")(view, STD_V1, frames=[0, 1]).holds is True
