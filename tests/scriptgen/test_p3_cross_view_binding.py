"""P3: landmark-chain evidence forces cross-view spatial binding."""

from __future__ import annotations

import math
from functools import cache
from pathlib import Path

import pytest

from spatial_episode.scriptgen import generate_plans
from spatial_episode.scriptgen.answers import registered_answer_modes
from spatial_episode.scriptgen.behavior import layout_from_scene_ir
from spatial_episode.scriptgen.compiler import CapabilityCompiler
from spatial_episode.scriptgen.family import QUESTION_SCRIPT_SETS, build_family_doc
from spatial_episode.scriptgen.library import (
    CROSS_VIEW_CLOSER,
    CROSS_VIEW_RELATION,
    CROSS_VIEW_SCRIPTS,
    EXISTENCE_SUFFICIENCY,
    SCRIPT_LIBRARY,
)
from spatial_episode.scriptgen.predicates import registered_predicates
from spatial_episode.scriptgen.sceneview import (
    GeometrySceneView,
    Pose2D,
    SceneLayout,
    SceneObject,
)
from spatial_episode.scriptgen.spec import ScriptSpec
from spatial_episode.scriptgen.standards import STD_V1
from spatial_episode.scriptgen.variants import VariantBuilder

HOME_SCENE_IR = Path(
    "/data/shichao/data/dataV100/code/OminiGibson/outputs/sweeps/"
    "static-m2-room-aware-indoor-seed17-v1/bundles/"
    "Beechwood_0_int_seed17/scene_ir.json"
)


def _fixture_layout(chain_length: int, *, reference_yaw_deg: float = 0.0) -> SceneLayout:
    """Separated X/Y with a geometric bridge of uniquely named landmarks."""
    other_xy = (4.0, 0.0)
    angle = math.radians(155.0)
    target_xy = (
        other_xy[0] + 8.0 * math.cos(angle),
        other_xy[1] + 8.0 * math.sin(angle),
    )
    fractions = {
        1: (0.25,),
        2: (0.20, 0.60),
        3: (0.15, 0.45, 0.75),
    }[chain_length]
    objects = [
        SceneObject("X", "xcat", target_xy, 0.6, "x"),
        SceneObject(
            "Y",
            "ycat",
            other_xy,
            0.6,
            "y",
            yaw_deg=reference_yaw_deg,
        ),
    ]
    for index, fraction in enumerate(fractions, start=1):
        objects.append(
            SceneObject(
                f"A{index}",
                f"acat{index}",
                (
                    target_xy[0] + fraction * (other_xy[0] - target_xy[0]),
                    target_xy[1] + fraction * (other_xy[1] - target_xy[1]),
                ),
                0.6,
                f"a{index}",
            )
        )
    return SceneLayout(
        scene_id=f"cross_view_k{chain_length}",
        objects=tuple(objects),
        walkable_min=(-6.0, -6.0),
        walkable_max=(6.0, 6.0),
    )


def _fixed_binding(script: ScriptSpec, chain_length: int) -> ScriptSpec:
    """Constrain fixture slots by category without changing production specs."""
    categories = {
        "target": ("xcat",),
        "other": ("ycat",),
        **{
            f"anchor{index}": (f"acat{index}",)
            for index in range(1, chain_length + 1)
        },
    }
    return script.model_copy(
        update={
            "slots": {
                name: slot.model_copy(update={"categories": categories[name]})
                for name, slot in script.slots.items()
            }
        }
    )


@cache
def _case(chain_length: int):
    layout = _fixture_layout(chain_length)
    script = _fixed_binding(CROSS_VIEW_RELATION[chain_length - 1], chain_length)
    report = generate_plans(
        layout,
        script,
        STD_V1,
        seed=17,
        attempts_per_binding=150,
        max_plans=1,
    )
    assert report.plans, report.rejection_counts
    plan = report.plans[0]
    view = GeometrySceneView(
        layout,
        tuple(Pose2D(pose.x, pose.y, pose.yaw_deg) for pose in plan.poses),
        STD_V1,
    )
    return script, plan, view


def test_p3_registry_standard_and_library_surface_is_complete() -> None:
    assert STD_V1.standard_version == "std.v10"
    assert STD_V1.landmark_chain_lengths == (1, 2, 3)
    assert {"pair_relation", "closer_of"} <= set(registered_answer_modes())
    assert {
        "never_covisible",
        "chain_connected",
        "pair_relation_margin_ge",
        "closer_ratio_ge",
    } <= set(registered_predicates())
    assert len(CROSS_VIEW_RELATION) == len(CROSS_VIEW_CLOSER) == 3
    assert len(CROSS_VIEW_SCRIPTS) == 6
    assert all(script.motifs == ("visit_landmarks",) for script in CROSS_VIEW_SCRIPTS)
    assert all(SCRIPT_LIBRARY[script.capability] is script for script in CROSS_VIEW_SCRIPTS)


@pytest.mark.parametrize("chain_length", (1, 2, 3))
def test_landmark_visit_forces_cross_frame_chain_without_single_frame_shortcut(
    chain_length: int,
) -> None:
    _, plan, _ = _case(chain_length)
    assert plan.binding == {
        "target": "X",
        **{f"anchor{index}": f"A{index}" for index in range(1, chain_length + 1)},
        "other": "Y",
    }
    assert plan.clause_witnesses["queried_pair_never_covisible"] == {
        "covisible_frames": [],
        "ambiguous_frames": [],
    }
    chain = plan.clause_witnesses["anchor_chain_evidence"]
    assert len(chain["covisible_counts"]) == chain_length + 1
    assert all(
        count >= STD_V1.chain_min_covisible_frames
        for count in chain["covisible_counts"].values()
    )
    assert plan.clause_witnesses["poses_clear"]["collision_count"] == 0
    assert plan.clause_witnesses["path_clear"]["collision_count"] == 0


@pytest.mark.parametrize(
    ("chain_length", "expected_ratio"),
    ((1, 3.0), (2, 4.0), (3, 5.667)),
)
def test_relation_and_distance_questions_recompile_on_the_same_chain(
    chain_length: int,
    expected_ratio: float,
) -> None:
    relation_script, plan, view = _case(chain_length)
    relation = CapabilityCompiler(relation_script, STD_V1).compile(view, plan.binding)
    closer_script = _fixed_binding(CROSS_VIEW_CLOSER[chain_length - 1], chain_length)
    closer = CapabilityCompiler(closer_script, STD_V1).compile(view, plan.binding)

    assert relation.status == closer.status == "answerable"
    assert relation.answer is not None and closer.answer is not None
    # Values are copied from the compiler's deterministic output, never hand-adjusted gold.
    assert relation.answer.label == "back"
    assert relation.answer.witness == {
        "reference_yaw_deg": 0.0,
        "azimuth_deg": 155.0,
        "margin_deg": 20.0,
    }
    assert closer.answer.label == "first"
    assert closer.answer.witness["distance_ratio"] == expected_ratio
    assert QUESTION_SCRIPT_SETS[relation_script.capability] == (
        CROSS_VIEW_RELATION[chain_length - 1],
        CROSS_VIEW_CLOSER[chain_length - 1],
        EXISTENCE_SUFFICIENCY,
    )


def test_pair_relation_covaries_with_reference_objects_intrinsic_yaw() -> None:
    script, plan, view = _case(1)
    base = CapabilityCompiler(script, STD_V1).compile(view, plan.binding)
    rotated_view = GeometrySceneView(
        _fixture_layout(1, reference_yaw_deg=90.0),
        view.poses,
        STD_V1,
    )
    rotated = CapabilityCompiler(script, STD_V1).compile(rotated_view, plan.binding)

    assert base.answer is not None and rotated.answer is not None
    assert base.answer.label == "back"
    assert rotated.answer.label == "left"
    assert rotated.answer.witness["reference_yaw_deg"] == 90.0
    assert rotated.answer.witness["azimuth_deg"] == 65.0


@pytest.mark.parametrize("chain_length", (1, 3))
def test_cross_view_variant_signature_is_machine_verified(chain_length: int) -> None:
    script, plan, view = _case(chain_length)
    compiler = CapabilityCompiler(script, STD_V1)
    canonical = compiler.compile(view, plan.binding, with_essential=True)
    variants = VariantBuilder(compiler, view, plan.binding, canonical).build_all(seed=17)
    by_kind = {variant.kind: variant for variant in variants}

    assert by_kind["permute"].gold == canonical.answer.label  # type: ignore[union-attr]
    assert by_kind["drop_key"].certificate.status == "abstain"
    assert by_kind["drop_key"].certificate.reason == "clause:anchor_chain_evidence"
    assert by_kind["drop_filler"].gold == canonical.answer.label  # type: ignore[union-attr]
    assert by_kind["delay"].gold == canonical.answer.label  # type: ignore[union-attr]


def test_cross_view_family_packages_every_chain_referent() -> None:
    script, plan, view = _case(1)
    family = build_family_doc(view, plan.model_dump(), script, STD_V1, seed=17)
    assert family.role == "primary"
    assert set(family.referents) == {"target", "anchor1", "other"}
    assert family.episodes[0].label == "back"
    assert all(token not in family.question.text for token in ("{target}", "{other}"))


@pytest.mark.skipif(not HOME_SCENE_IR.exists(), reason="Beechwood scene_ir is unavailable")
def test_k1_chain_builds_on_real_multiroom_geometry() -> None:
    layout = layout_from_scene_ir(HOME_SCENE_IR, std=STD_V1)
    report = generate_plans(
        layout,
        CROSS_VIEW_RELATION[0],
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
    relation = CapabilityCompiler(CROSS_VIEW_RELATION[0], STD_V1).compile(view, plan.binding)
    closer = CapabilityCompiler(CROSS_VIEW_CLOSER[0], STD_V1).compile(view, plan.binding)
    assert relation.answer is not None and closer.answer is not None
    assert relation.answer.label == "left"
    assert closer.answer.label == "second"
    assert plan.clause_witnesses["queried_pair_never_covisible"]["covisible_frames"] == []
    assert all(
        count >= STD_V1.chain_min_covisible_frames
        for count in plan.clause_witnesses["anchor_chain_evidence"][
            "covisible_counts"
        ].values()
    )
