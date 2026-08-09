"""P2: survey evidence and compiler-derived imagined-viewpoint covariance."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import pytest

from spatial_episode.scriptgen import generate_plans
from spatial_episode.scriptgen.answers import registered_answer_modes
from spatial_episode.scriptgen.behavior import layout_from_scene_ir
from spatial_episode.scriptgen.compiler import CapabilityCompiler
from spatial_episode.scriptgen.family import QUESTION_SCRIPT_SETS, build_family_doc
from spatial_episode.scriptgen.geometry import wrap_deg
from spatial_episode.scriptgen.library import (
    EXISTENCE_SUFFICIENCY,
    IMAGINED_VIEWPOINT_OFFSETS,
    REFERENCE_FRAME_DEEP,
    REFERENCE_FRAME_SCRIPTS,
    REFERENCE_FRAME_SHALLOW,
)
from spatial_episode.scriptgen.predicates import registered_predicates
from spatial_episode.scriptgen.sceneview import (
    GeometrySceneView,
    Pose2D,
    SceneLayout,
    SceneObject,
)
from spatial_episode.scriptgen.standards import STD_V1
from spatial_episode.scriptgen.variants import VariantBuilder

P2_LAYOUT = SceneLayout(
    scene_id="reference_frame_fixture",
    objects=(
        SceneObject("P", "door", (4.0, 0.0), 1.0, "p"),
        SceneObject("Q", "bed", (-1.196, 3.0), 1.0, "q"),
        SceneObject("X", "cabinet", (4.783, 5.948), 1.0, "x"),
    ),
    walkable_min=(-1.0, -1.0),
    walkable_max=(1.0, 1.0),
)
HOME_SCENE_IR = Path(
    "/data/shichao/data/dataV100/code/OminiGibson/outputs/sweeps/"
    "static-m2-room-aware-indoor-seed17-v1/bundles/Beechwood_0_int_seed17/scene_ir.json"
)


@cache
def _plan_and_view():
    report = generate_plans(
        P2_LAYOUT,
        REFERENCE_FRAME_DEEP[0],
        STD_V1,
        seed=17,
        max_plans=1,
    )
    assert report.plans, report.rejection_counts
    plan = report.plans[0]
    poses = tuple(Pose2D(pose.x, pose.y, pose.yaw_deg) for pose in plan.poses)
    return plan, GeometrySceneView(P2_LAYOUT, poses, STD_V1)


def test_p2_registry_and_standard_surface_is_complete() -> None:
    assert STD_V1.standard_version == "std.v9"
    assert STD_V1.imagined_viewpoint_offsets_deg == IMAGINED_VIEWPOINT_OFFSETS
    assert {"imagined_sector", "imagined_visibility"} <= set(registered_answer_modes())
    assert {
        "all_landmarks_visible",
        "never_all_covisible",
        "imagined_pose_valid",
        "imagined_curve_sector_margins_ge",
        "imagined_curve_visibility_decisive",
    } <= set(registered_predicates())
    assert len(REFERENCE_FRAME_DEEP) == len(REFERENCE_FRAME_SHALLOW) == 5
    assert all(script.motifs == ("survey",) for script in REFERENCE_FRAME_SCRIPTS)


def test_survey_forces_cross_frame_evidence_without_motion_memory() -> None:
    plan, view = _plan_and_view()
    assert len({pose.xy for pose in view.poses}) == 1
    assert plan.clause_witnesses["stationary_survey"]["max_displacement_m"] == 0.0
    assert plan.clause_witnesses["no_single_frame_shortcut"] == {
        "all_visible_frames": [],
        "ambiguous_frames": [],
    }
    counts = plan.clause_witnesses["landmark_evidence"]["visible_counts"]
    assert all(count >= 2 for count in counts.values())


def test_deep_and_shallow_curves_recompile_on_one_trajectory() -> None:
    plan, view = _plan_and_view()
    deep = [
        CapabilityCompiler(script, STD_V1).compile(view, plan.binding)
        for script in REFERENCE_FRAME_DEEP
    ]
    shallow = [
        CapabilityCompiler(script, STD_V1).compile(view, plan.binding)
        for script in REFERENCE_FRAME_SHALLOW
    ]
    assert all(certificate.status == "answerable" for certificate in (*deep, *shallow))
    assert [certificate.answer.label for certificate in deep] == [
        "right",
        "right",
        "back",
        "back",
        "left",
    ]
    assert [certificate.answer.label for certificate in shallow] == ["not_visible"] * 5

    base_azimuth = deep[0].answer.witness["azimuth_deg"]
    for offset, certificate in zip(IMAGINED_VIEWPOINT_OFFSETS, deep, strict=True):
        actual = certificate.answer.witness["azimuth_deg"]
        assert wrap_deg(actual - base_azimuth) == pytest.approx(wrap_deg(-offset))
        assert certificate.answer.witness["yaw_offset_deg"] == float(offset)


@pytest.mark.parametrize("script", [REFERENCE_FRAME_DEEP[0], REFERENCE_FRAME_SHALLOW[0]])
def test_reference_frame_variant_signature_is_machine_verified(script) -> None:
    plan, view = _plan_and_view()
    compiler = CapabilityCompiler(script, STD_V1)
    canonical = compiler.compile(view, plan.binding, with_essential=True)
    variants = VariantBuilder(compiler, view, plan.binding, canonical).build_all(seed=17)
    labels = {variant.kind: variant.gold for variant in variants}
    assert labels["permute"] == labels["drop_filler"] == labels["delay"]
    assert labels["drop_key"] == script.templates[0].abstain_option


def test_family_packages_every_bound_referent_and_fills_question_slots() -> None:
    plan, view = _plan_and_view()
    doc = build_family_doc(
        view,
        plan.model_dump(),
        REFERENCE_FRAME_DEEP[0],
        STD_V1,
        seed=17,
    )
    assert doc.schema_version == "scriptgen_family.v4"
    assert set(doc.referents) == {"viewpoint", "facing", "target"}
    assert len({referent.entity_id for referent in doc.referents.values()}) == 3
    assert all(token not in doc.question.text for token in ("{viewpoint}", "{facing}", "{target}"))
    assert QUESTION_SCRIPT_SETS[REFERENCE_FRAME_DEEP[0].capability] == (
        *REFERENCE_FRAME_SCRIPTS,
        EXISTENCE_SUFFICIENCY,
    )


@pytest.mark.skipif(not HOME_SCENE_IR.exists(), reason="Beechwood scene_ir is unavailable")
def test_reference_frame_curve_builds_on_real_multiroom_geometry() -> None:
    layout = layout_from_scene_ir(HOME_SCENE_IR, std=STD_V1)
    report = generate_plans(
        layout,
        REFERENCE_FRAME_DEEP[0],
        STD_V1,
        seed=17,
        attempts_per_binding=20,
        max_plans=1,
    )
    assert report.plans, report.rejection_counts
    plan = report.plans[0]
    view = GeometrySceneView(
        layout,
        tuple(Pose2D(pose.x, pose.y, pose.yaw_deg) for pose in plan.poses),
        STD_V1,
    )
    deep_labels = [
        CapabilityCompiler(script, STD_V1).compile(view, plan.binding).answer.label
        for script in REFERENCE_FRAME_DEEP
    ]
    shallow_labels = [
        CapabilityCompiler(script, STD_V1).compile(view, plan.binding).answer.label
        for script in REFERENCE_FRAME_SHALLOW
    ]
    assert deep_labels == ["front", "right", "right", "back", "back"]
    assert shallow_labels == [
        "visible",
        "not_visible",
        "not_visible",
        "not_visible",
        "not_visible",
    ]
