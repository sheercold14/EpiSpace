"""P4: cross-frame free-space coverage calibrates category absence claims."""

from __future__ import annotations

from functools import cache

import pytest

from spatial_episode.scriptgen import generate_plans
from spatial_episode.scriptgen.answers import get_answer_mode, registered_answer_modes
from spatial_episode.scriptgen.compiler import CapabilityCompiler
from spatial_episode.scriptgen.coverage import measure_coverage
from spatial_episode.scriptgen.family import (
    QUESTION_GROUP_SCRIPTS,
    QUESTION_SCRIPT_SETS,
    build_family_doc,
)
from spatial_episode.scriptgen.library import (
    EXISTENCE_SUFFICIENCY,
    REFERENCE_FRAME_DEEP,
    SCRIPT_LIBRARY,
)
from spatial_episode.scriptgen.predicates import registered_predicates
from spatial_episode.scriptgen.sceneview import (
    GeometrySceneView,
    Obstacle,
    Pose2D,
    SceneLayout,
    SceneObject,
)
from spatial_episode.scriptgen.standards import STD_V1
from spatial_episode.scriptgen.variants import VariantBuilder

SMALL_LAYOUT = SceneLayout(
    scene_id="coverage_small",
    objects=(SceneObject("T", "chair", (0.5, 0.5), 0.5, "t"),),
    walkable_min=(-1.0, -1.0),
    walkable_max=(1.0, 1.0),
)
CARDINAL_POSES = tuple(Pose2D(0.0, 0.0, yaw) for yaw in (0.0, 90.0, 180.0, -90.0))


def _view(layout: SceneLayout, poses: tuple[Pose2D, ...]) -> GeometrySceneView:
    return GeometrySceneView(layout, poses, STD_V1)


@cache
def _canonical():
    view = _view(SMALL_LAYOUT, CARDINAL_POSES)
    compiler = CapabilityCompiler(EXISTENCE_SUFFICIENCY, STD_V1)
    certificate = compiler.compile(view, {"target": "T"}, with_essential=True)
    return view, compiler, certificate


def test_p4_registry_standard_and_library_surface_is_complete() -> None:
    assert STD_V1.standard_version == "std.v8"
    assert STD_V1.coverage_ratio_levels == (0.50, 0.70, 0.85)
    assert STD_V1.coverage_hidden_object_size_m == 1.0
    assert "existence_sufficiency" in registered_answer_modes()
    assert {
        "category_absent",
        "coverage_ratio_ge",
        "no_single_frame_coverage_sufficient",
        "drop_target_breaks_coverage",
    } <= set(registered_predicates())
    assert SCRIPT_LIBRARY[EXISTENCE_SUFFICIENCY.capability] is EXISTENCE_SUFFICIENCY
    assert EXISTENCE_SUFFICIENCY.motifs == ("survey",)
    assert QUESTION_GROUP_SCRIPTS[-1] is EXISTENCE_SUFFICIENCY
    assert QUESTION_SCRIPT_SETS[REFERENCE_FRAME_DEEP[0].capability][-1] is (
        EXISTENCE_SUFFICIENCY
    )


def test_frustum_union_reaches_full_coverage_only_across_frames() -> None:
    view = _view(SMALL_LAYOUT, CARDINAL_POSES)
    report = measure_coverage(view, STD_V1, frames=list(range(4)))
    assert report.covered_cell_count == report.total_free_cell_count == 676
    assert report.coverage_ratio == 1.0
    assert report.uncovered_cell_count == report.uncovered_component_count == 0
    assert report.max_hidden_diameter_m == 0.0

    individual = [measure_coverage(view, STD_V1, frames=[frame]) for frame in range(4)]
    assert [round(row.coverage_ratio, 4) for row in individual] == [
        0.25,
        0.25,
        0.2825,
        0.2751,
    ]
    assert all(row.coverage_ratio < STD_V1.coverage_ratio_levels[-1] for row in individual)


def test_eye_height_blocker_truncates_the_coverage_frustum() -> None:
    wall = Obstacle(
        label="wall",
        center_xy=(1.0, 0.0),
        half_extents_xy=(0.05, 1.5),
        yaw_deg=0.0,
        z_low=0.0,
        z_high=2.0,
        entity_id="wall",
    )
    fields = {
        "scene_id": "coverage_wall",
        "objects": (SceneObject("T", "chair", (-0.5, -0.5), 0.5, "t"),),
        "obstacles": (wall,),
        "walkable_min": (-1.0, -2.0),
        "walkable_max": (3.0, 2.0),
    }
    clear_layout = SceneLayout(**fields)
    blocked_layout = SceneLayout(**fields, occlusion_obstacles=(wall,))
    poses = (Pose2D(0.0, 0.0, 0.0),)
    clear = measure_coverage(_view(clear_layout, poses), STD_V1, frames=[0])
    blocked = measure_coverage(_view(blocked_layout, poses), STD_V1, frames=[0])
    assert clear.total_free_cell_count == blocked.total_free_cell_count == 3300
    assert clear.covered_cell_count == 1765
    assert blocked.covered_cell_count == 140


def test_answer_mode_covers_present_absent_and_objectively_insufficient_evidence() -> None:
    answer = get_answer_mode("existence_sufficiency")
    absent = answer(
        _view(SMALL_LAYOUT, CARDINAL_POSES),
        STD_V1,
        category="bed",
        frames=list(range(4)),
    )
    assert absent.label == "absent"
    assert absent.witness["coverage_tier"] == "high"

    large_absent = SceneLayout(
        scene_id="coverage_large_absent",
        objects=(SceneObject("T", "chair", (-3.0, -3.0), 0.5, "t"),),
        walkable_min=(-5.0, -5.0),
        walkable_max=(5.0, 5.0),
    )
    insufficient = answer(
        _view(large_absent, (Pose2D(0.0, 0.0, 0.0),)),
        STD_V1,
        category="bed",
        frames=[0],
    )
    assert insufficient.label == "无法判断"
    assert insufficient.witness["coverage_ratio"] == 0.2527
    assert insufficient.witness["max_hidden_diameter_m"] == 5.424

    present_layout = SceneLayout(
        scene_id="coverage_present",
        objects=(
            SceneObject("T", "chair", (-3.0, -3.0), 0.5, "t"),
            SceneObject("B", "bed", (2.0, 0.0), 1.0, "b"),
        ),
        walkable_min=(-5.0, -5.0),
        walkable_max=(5.0, 5.0),
    )
    present = answer(
        _view(present_layout, (Pose2D(0.0, 0.0, 0.0),)),
        STD_V1,
        category="bed",
        frames=[0],
    )
    hidden = answer(
        _view(present_layout, (Pose2D(0.0, 0.0, 180.0),)),
        STD_V1,
        category="bed",
        frames=[0],
    )
    assert present.label == "present"
    assert present.witness["first_visible_frame"] == 0
    assert hidden.label == "无法判断"


def test_production_spec_blocks_static_positive_and_every_single_frame_shortcut() -> None:
    _, _, canonical = _canonical()
    assert canonical.status == "answerable"
    assert canonical.answer is not None and canonical.answer.label == "absent"
    outcomes = {outcome.name: outcome for outcome in canonical.clause_outcomes}
    assert outcomes["no_single_frame_shortcut"].witness["sufficient_frames"] == []
    assert outcomes["coverage_sufficient"].witness["coverage_ratio"] == 1.0

    view, compiler, _ = _canonical()
    for frame in range(view.frame_count):
        one_frame = compiler.compile(view, {"target": "T"}, frame_sequence=(frame,))
        assert one_frame.status == "abstain"
        assert one_frame.reason == "clause:coverage_sufficient"

    with_bed = SceneLayout(
        scene_id="coverage_static_positive",
        objects=(*SMALL_LAYOUT.objects, SceneObject("B", "bed", (0.0, 0.5), 1.0, "b")),
        walkable_min=SMALL_LAYOUT.walkable_min,
        walkable_max=SMALL_LAYOUT.walkable_max,
    )
    blocked = CapabilityCompiler(EXISTENCE_SUFFICIENCY, STD_V1).compile(
        _view(with_bed, CARDINAL_POSES),
        {"target": "T"},
    )
    assert blocked.status == "invalid"
    assert blocked.reason == "clause:category_absent"


def test_coverage_meta_variant_signature_is_recompiled_not_assigned() -> None:
    view, compiler, canonical = _canonical()
    variants = VariantBuilder(compiler, view, {"target": "T"}, canonical).build_all(seed=17)
    by_kind = {variant.kind: variant for variant in variants}
    assert set(by_kind) == {"permute", "drop_key", "delay"}
    assert by_kind["permute"].gold == by_kind["delay"].gold == "absent"
    assert by_kind["drop_key"].gold == "无法判断"
    assert by_kind["drop_key"].certificate.status == "abstain"
    assert by_kind["drop_key"].certificate.reason == "clause:coverage_sufficient"
    outcome = next(
        row
        for row in canonical.clause_outcomes
        if row.name == "drop_key_destroys_evidence"
    )
    assert outcome.witness["residual_coverage_ratio"] == 0.537

    family = build_family_doc(
        view,
        {
            "plan_id": "coverage-source-plan",
            "scene_id": SMALL_LAYOUT.scene_id,
            "capability": "different_source",
            "binding": {"target": "T"},
        },
        EXISTENCE_SUFFICIENCY,
        STD_V1,
        seed=17,
    )
    assert family.role == "primary"
    assert [episode.label for episode in family.episodes] == [
        "absent",
        "absent",
        "无法判断",
        "absent",
    ]
    assert family.checks.answer_token_leak is False


def test_existing_survey_motif_generates_the_meta_spec_by_accept_reject() -> None:
    report = generate_plans(
        SMALL_LAYOUT,
        EXISTENCE_SUFFICIENCY,
        STD_V1,
        seed=17,
        attempts_per_binding=20,
        max_plans=1,
    )
    assert report.plans, report.rejection_counts
    plan = report.plans[0]
    assert len(plan.poses) == 18
    assert plan.provisional_answer.label == "absent"
    assert plan.clause_witnesses["coverage_sufficient"]["coverage_ratio"] == 1.0
    assert plan.clause_witnesses["no_single_frame_shortcut"]["sufficient_frames"] == []
    assert plan.clause_witnesses["drop_key_destroys_evidence"][
        "residual_coverage_ratio"
    ] == pytest.approx(0.7308)
