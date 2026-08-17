"""P1: balanced C/combination-one trajectory subtypes and diagnostic tier."""

from __future__ import annotations

from collections import Counter
from functools import cache
from pathlib import Path

import pytest

from spatial_episode.scriptgen import generate_plans
from spatial_episode.scriptgen.answers import get_answer_mode, registered_answer_modes
from spatial_episode.scriptgen.compiler import CapabilityCompiler
from spatial_episode.scriptgen.demo import DEMO_LAYOUT
from spatial_episode.scriptgen.library import (
    HOMING,
    MULTI_TURN,
    NET_TURN,
    NET_TURN_MAGNITUDE,
    OCCLUDED_MOTION,
    PURE_ROTATION,
    PURE_TRANSLATION,
    SCRIPT_LIBRARY,
    SELF_MOTION,
)
from spatial_episode.scriptgen.predicates import get_predicate, registered_predicates
from spatial_episode.scriptgen.sceneview import (
    GeometrySceneView,
    Obstacle,
    Pose2D,
    SceneLayout,
    SceneObject,
)
from spatial_episode.scriptgen.standards import STD_V1
from spatial_episode.scriptgen.variants import VariantBuilder

P1_SUBTYPES = (PURE_ROTATION, PURE_TRANSLATION, MULTI_TURN, OCCLUDED_MOTION)
HOME_SCENE_IR = Path(
    "/data/shichao/data/dataV100/code/OminiGibson/outputs/sweeps/"
    "static-m2-room-aware-indoor-seed17-v1/bundles/Beechwood_0_int_seed17/scene_ir.json"
)


@cache
def _generated(capability: str):
    script = SCRIPT_LIBRARY[capability]
    report = generate_plans(
        DEMO_LAYOUT,
        script,
        STD_V1,
        seed=17,
        attempts_per_binding=300,
    )
    assert report.plans, (capability, report.rejection_counts)
    return report.plans[0]


def _view(plan) -> GeometrySceneView:
    poses = tuple(Pose2D(pose.x, pose.y, pose.yaw_deg) for pose in plan.poses)
    return GeometrySceneView(layout=DEMO_LAYOUT, poses=poses, std=STD_V1)


def test_p1_registry_surface_is_complete() -> None:
    assert STD_V1.standard_version == "std.v10"
    assert "net_turn_magnitude" in registered_answer_modes()
    assert {
        "displacement_below",
        "turn_below",
        "turn_segments_between",
        "occluded_in_view",
        "net_turn_magnitude_margin_ge",
    } <= set(registered_predicates())
    assert {script.capability for script in (*P1_SUBTYPES, NET_TURN_MAGNITUDE)} <= set(
        SCRIPT_LIBRARY
    )


@pytest.mark.parametrize("script", P1_SUBTYPES, ids=lambda script: script.capability)
def test_each_subtype_generates_and_recompiles_from_truth(script) -> None:
    plan = _generated(script.capability)
    cert = CapabilityCompiler(script=script, std=STD_V1).compile(
        _view(plan),
        plan.binding,
        geometry_plan=plan.model_dump(),
    )
    assert cert.status == "answerable"
    assert cert.mismatch is None
    assert cert.answer is not None
    assert cert.answer.label == plan.provisional_answer.label
    assert cert.answer.witness == plan.provisional_answer.witness
    assert cert.frame_vars["t_seen"] < cert.frame_vars["t_gone"] <= cert.frame_vars["t_q"]


def test_subtype_isolation_witnesses() -> None:
    rotation = _generated(PURE_ROTATION.capability)
    assert rotation.clause_witnesses["stationary"] == {
        "max_displacement_m": 0.0,
        "allowed_m": STD_V1.pure_rotation_max_displacement_m,
    }
    turned = next(clause for clause in PURE_ROTATION.clauses if clause.name == "turned")
    assert turned.args["frames"] == "0:$t_q"
    assert 80 <= rotation.clause_witnesses["turned"]["cum_turn_deg"] <= 200

    translation = _generated(PURE_TRANSLATION.capability)
    assert translation.clause_witnesses["heading_constant"] == {
        "cum_turn_deg": 0.0,
        "allowed_deg": STD_V1.pure_translation_max_turn_deg,
    }
    net_turn = CapabilityCompiler(NET_TURN, STD_V1).compile(_view(translation), translation.binding)
    assert net_turn.status == "invalid"
    assert net_turn.reason == "clause:net_turn_margin_ok"

    multi = _generated(MULTI_TURN.capability)
    assert 2 <= multi.clause_witnesses["turn_segments"]["turn_segment_count"] <= 3

    occluded = _generated(OCCLUDED_MOTION.capability)
    assert occluded.provisional_answer.label == "front"
    assert occluded.clause_witnesses["occluded_at_question"]["blocked_by"]

    homing = CapabilityCompiler(HOMING, STD_V1).compile(_view(rotation), rotation.binding)
    assert homing.status == "invalid"
    assert homing.reason == "clause:start_far_enough"


@pytest.mark.parametrize("script", P1_SUBTYPES, ids=lambda script: script.capability)
def test_each_subtype_variant_matrix_is_machine_verified(script) -> None:
    plan = _generated(script.capability)
    view = _view(plan)
    compiler = CapabilityCompiler(script, STD_V1)
    canonical = compiler.compile(
        view,
        plan.binding,
        geometry_plan=plan.model_dump(),
        with_essential=True,
    )
    variants = VariantBuilder(compiler, view, plan.binding, canonical).build_all(seed=17)
    labels = {variant.kind: variant.gold for variant in variants}
    assert labels["permute"] == labels["drop_key"] == script.templates[0].abstain_option
    assert labels["drop_filler"] == labels["delay"] == canonical.answer.label


def test_net_turn_magnitude_answer_and_margin_are_declared() -> None:
    plan = _generated(MULTI_TURN.capability)
    view = _view(plan)
    frame_range = list(range(view.frame_count))
    answer = get_answer_mode("net_turn_magnitude")(view, STD_V1, frames=frame_range)
    margin = get_predicate("net_turn_magnitude_margin_ge")(view, STD_V1, frames=frame_range)
    assert answer.label in NET_TURN_MAGNITUDE.templates[0].options
    assert answer.witness["threshold_deg"] == STD_V1.net_turn_magnitude_deg
    assert margin.holds is True


def test_walk_and_turn_meets_preregistered_balance_gate() -> None:
    counts: Counter[str] = Counter()
    for seed in range(40):
        report = generate_plans(
            DEMO_LAYOUT,
            SELF_MOTION,
            STD_V1,
            seed=seed,
            plans_per_binding=1,
        )
        counts.update(plan.provisional_answer.label for plan in report.plans)
    total = sum(counts.values())
    assert total == 120
    assert counts["left"] / total >= 0.25
    assert counts["right"] / total >= 0.25
    assert counts["back"] / total >= 0.15


@pytest.mark.parametrize("script", (SELF_MOTION, PURE_ROTATION))
def test_empty_occupancy_grid_is_a_normal_motif_rejection(script) -> None:
    layout = SceneLayout(
        scene_id="fully-blocked",
        objects=(
            SceneObject(
                name="target",
                category="chair",
                xy=(1.0, 1.0),
                size_m=0.5,
                uid="target",
            ),
        ),
        obstacles=(
            Obstacle(
                label="enclosing_fence_proxy",
                center_xy=(1.0, 1.0),
                half_extents_xy=(2.0, 2.0),
                yaw_deg=0.0,
                z_low=0.0,
                z_high=2.0,
            ),
        ),
        walkable_min=(0.0, 0.0),
        walkable_max=(2.0, 2.0),
    )

    report = generate_plans(
        layout,
        script,
        STD_V1,
        seed=17,
        attempts_per_binding=3,
        plans_per_binding=1,
        candidate_bindings=({"target": "target"},),
    )

    assert report.plans == ()
    assert report.rejection_counts == {
        f"motif:{script.motifs[0]}:no_candidate": 3,
        "binding_exhausted": 1,
    }


def test_multi_turn_activates_both_declared_segment_levels() -> None:
    levels: set[int] = set()
    for seed in range(20):
        report = generate_plans(DEMO_LAYOUT, MULTI_TURN, STD_V1, seed=seed)
        levels.update(
            plan.clause_witnesses["turn_segments"]["turn_segment_count"] for plan in report.plans
        )
    assert levels == {2, 3}


@pytest.mark.skipif(not HOME_SCENE_IR.exists(), reason="Beechwood scene_ir is unavailable")
def test_all_subtypes_build_clear_plans_on_real_multiroom_geometry() -> None:
    from spatial_episode.scriptgen.behavior import layout_from_scene_ir

    layout = layout_from_scene_ir(HOME_SCENE_IR, std=STD_V1)
    for script in P1_SUBTYPES:
        report = generate_plans(
            layout,
            script,
            STD_V1,
            seed=17,
            attempts_per_binding=80,
            max_plans=1,
        )
        assert report.plans, (script.capability, report.rejection_counts)
        plan = report.plans[0]
        assert plan.clause_witnesses["poses_clear"]["collision_count"] == 0
        assert plan.clause_witnesses["path_clear"]["collision_count"] == 0
