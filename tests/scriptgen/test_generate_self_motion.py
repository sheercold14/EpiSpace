"""End-to-end: the SELF_MOTION script yields qualifying plans on the demo scene."""

from __future__ import annotations

from itertools import pairwise

from spatial_episode.scriptgen import STD_V1, generate_plans
from spatial_episode.scriptgen.demo import DEMO_LAYOUT
from spatial_episode.scriptgen.library import SELF_MOTION
from spatial_episode.scriptgen.sceneview import GeometrySceneView, Pose2D


def test_generates_at_least_one_plan() -> None:
    report = generate_plans(DEMO_LAYOUT, SELF_MOTION, STD_V1, seed=17)
    assert report.plans, f"no plans; rejections={report.rejection_counts}"


def test_plan_invariants_replay() -> None:
    """Every emitted plan must satisfy its own claims when replayed."""
    from spatial_episode.scriptgen.geometry import wrap_deg

    report = generate_plans(DEMO_LAYOUT, SELF_MOTION, STD_V1, seed=17)
    for plan in report.plans:
        poses = tuple(Pose2D(p.x, p.y, p.yaw_deg) for p in plan.poses)
        view = GeometrySceneView(layout=DEMO_LAYOUT, poses=poses, std=STD_V1)
        target = plan.binding["target"]
        t_seen = plan.frame_vars["t_seen"]
        t_gone = plan.frame_vars["t_gone"]
        t_q = plan.frame_vars["t_q"]
        assert t_seen < t_gone <= t_q

        # Clearly visible at t_seen; definitely invisible from t_gone on.
        # Frames in between form the transition zone and may be anything.
        assert view.visibility(target, t_seen).tristate(STD_V1) is True
        for t in range(t_gone, t_q + 1):
            assert view.visibility(target, t).tristate(STD_V1) is False, (t, plan.plan_id)

        # Ego-motion trackability: bounded per-frame rotation (std.v2).
        for a, b in pairwise(poses):
            assert abs(wrap_deg(b.yaw_deg - a.yaw_deg)) <= STD_V1.max_step_turn_deg

        # Total turn stays in the mental-rotation range declared by the script.
        turned = plan.clause_witnesses["turned"]
        assert 80 <= turned["cum_turn_deg"] <= 200

        # Gold answer margin honours the standard.
        assert plan.provisional_answer.margin_deg >= STD_V1.sector_margin_deg
        assert plan.provisional_answer.sector in {"front", "left", "back", "right"}

        # Knob level matches the recorded frame variables.
        assert plan.knob_levels["delay"] == float(t_q - t_gone)

        # Witnesses cite the script's clauses.
        assert set(plan.clause_witnesses) == {
            "poses_clear",
            "path_clear",
            "seen_early",
            "trackable",
            "gone",
            "gap",
            "turned",
            "margin_ok",
        }


def test_generation_is_deterministic() -> None:
    a = generate_plans(DEMO_LAYOUT, SELF_MOTION, STD_V1, seed=23)
    b = generate_plans(DEMO_LAYOUT, SELF_MOTION, STD_V1, seed=23)
    assert [p.plan_id for p in a.plans] == [p.plan_id for p in b.plans]
    assert a.model_dump() == b.model_dump()


def test_slot_rejections_are_reported() -> None:
    report = generate_plans(DEMO_LAYOUT, SELF_MOTION, STD_V1, seed=17)
    # The small plant fails the 0.5 m size requirement and must be recorded.
    assert report.slot_rejections.get("too_small", 0) >= 1
