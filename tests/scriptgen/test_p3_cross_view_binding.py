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
    CROSS_VIEW_ANCHOR,
    CROSS_VIEW_CLOSER,
    CROSS_VIEW_EGO,
    CROSS_VIEW_SCRIPT_SETS,
    CROSS_VIEW_SCRIPTS,
    CROSS_VIEW_SNAPSHOT_ANCHOR,
    CROSS_VIEW_SNAPSHOT_CLOSER,
    CROSS_VIEW_SNAPSHOT_EGO,
    CROSS_VIEW_SNAPSHOT_SCRIPTS,
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
    """Separated X/Y with a geometric bridge of uniquely named landmarks.

    The last anchor sits close to Y so the final-edge station sees X far off
    the viewing axis: that is what lets the ego question clear its sector
    margin in an occluder-free room.
    """
    other_xy = (4.0, 0.0)
    angle = math.radians(155.0)
    target_xy = (
        other_xy[0] + 8.0 * math.cos(angle),
        other_xy[1] + 8.0 * math.sin(angle),
    )
    fractions = {
        1: (0.6,),
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
def _case(chain_length: int, snapshot: bool = False):
    ego_spec = (CROSS_VIEW_SNAPSHOT_EGO if snapshot else CROSS_VIEW_EGO)[chain_length - 1]
    layout = _fixture_layout(chain_length)
    script = _fixed_binding(ego_spec, chain_length)
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


def _trio_certificates(chain_length: int, snapshot: bool):
    ego_script, plan, view = _case(chain_length, snapshot)
    certificates = {}
    for sibling in CROSS_VIEW_SCRIPT_SETS[ego_script.capability]:
        bound = _fixed_binding(sibling, chain_length)
        certificates[sibling.capability] = CapabilityCompiler(bound, STD_V1).compile(
            view, plan.binding
        )
    return certificates


def test_p3_registry_standard_and_library_surface_is_complete() -> None:
    assert STD_V1.standard_version == "std.v11"
    assert STD_V1.landmark_chain_lengths == (1, 2, 3)
    assert STD_V1.snapshot_frames_per_station == 2
    assert {"target_sector", "imagined_sector", "closer_of"} <= set(
        registered_answer_modes()
    )
    assert {
        "never_covisible",
        "chain_connected",
        "invisible_in_range",
        "sector_margin_ge",
        "imagined_pose_valid",
        "imagined_sector_margin_ge",
        "closer_ratio_ge",
    } <= set(registered_predicates())
    assert (
        len(CROSS_VIEW_EGO)
        == len(CROSS_VIEW_ANCHOR)
        == len(CROSS_VIEW_CLOSER)
        == len(CROSS_VIEW_SNAPSHOT_EGO)
        == len(CROSS_VIEW_SNAPSHOT_ANCHOR)
        == len(CROSS_VIEW_SNAPSHOT_CLOSER)
        == 3
    )
    assert len(CROSS_VIEW_SCRIPTS) == 18
    walking = CROSS_VIEW_EGO + CROSS_VIEW_ANCHOR + CROSS_VIEW_CLOSER
    assert all(script.motifs == ("visit_landmarks",) for script in walking)
    assert all(
        script.motifs == ("snapshot_landmarks",) for script in CROSS_VIEW_SNAPSHOT_SCRIPTS
    )
    assert all(SCRIPT_LIBRARY[script.capability] is script for script in CROSS_VIEW_SCRIPTS)
    for index in range(3):
        chain_length = index + 1
        walk_trio = (
            CROSS_VIEW_EGO[index],
            CROSS_VIEW_ANCHOR[index],
            CROSS_VIEW_CLOSER[index],
        )
        snap_trio = (
            CROSS_VIEW_SNAPSHOT_EGO[index],
            CROSS_VIEW_SNAPSHOT_ANCHOR[index],
            CROSS_VIEW_SNAPSHOT_CLOSER[index],
        )
        for trio in (walk_trio, snap_trio):
            for script in trio:
                assert CROSS_VIEW_SCRIPT_SETS[script.capability] == trio
                assert QUESTION_SCRIPT_SETS[script.capability] == (
                    *trio,
                    EXISTENCE_SUFFICIENCY,
                )
        assert {script.capability for script in snap_trio} == {
            f"cross_view_snapshot_{question}_k{chain_length}"
            for question in ("ego", "anchor", "closer")
        }


@pytest.mark.parametrize("snapshot", (False, True), ids=("walk", "snapshot"))
@pytest.mark.parametrize("chain_length", (1, 2, 3))
def test_landmark_chain_forces_cross_frame_evidence(
    chain_length: int, snapshot: bool
) -> None:
    _, plan, _ = _case(chain_length, snapshot)
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
    if snapshot:
        assert "path_clear" not in plan.clause_witnesses
        assert len(plan.poses) == STD_V1.snapshot_frames_per_station * (chain_length + 1)
    else:
        assert plan.clause_witnesses["path_clear"]["collision_count"] == 0


# Labels and witnesses below are copied from the compiler's deterministic
# output, never hand-adjusted gold.
TRIO_EXPECTATIONS = {
    (1, False): ("left", "second", 1.5),
    (2, False): ("left", "first", 4.0),
    (3, False): ("right", "first", 5.667),
    (1, True): ("left", "second", 1.5),
    (2, True): ("left", "first", 4.0),
    (3, True): ("right", "first", 5.667),
}


@pytest.mark.parametrize("snapshot", (False, True), ids=("walk", "snapshot"))
@pytest.mark.parametrize("chain_length", (1, 2, 3))
def test_ego_anchor_and_closer_recompile_on_the_same_chain(
    chain_length: int, snapshot: bool
) -> None:
    certificates = _trio_certificates(chain_length, snapshot)
    ego_label, closer_label, distance_ratio = TRIO_EXPECTATIONS[(chain_length, snapshot)]
    line = "cross_view_snapshot" if snapshot else "cross_view"
    ego = certificates[f"{line}_ego_k{chain_length}"]
    anchor = certificates[f"{line}_anchor_k{chain_length}"]
    closer = certificates[f"{line}_closer_k{chain_length}"]

    assert ego.status == anchor.status == closer.status == "answerable"
    assert ego.answer.label == ego_label
    assert ego.answer.witness["sector"] == ego_label
    assert ego.answer.witness["question_frame"] == len(_case(chain_length, snapshot)[1].poses) - 1
    assert ego.answer.witness["margin_deg"] >= STD_V1.sector_margin_deg
    assert anchor.answer.label == "front"
    assert anchor.answer.witness["imagined_x"] == 4.0
    assert anchor.answer.witness["imagined_y"] == 0.0
    assert anchor.answer.witness["imagined_yaw_deg"] == 155.0
    assert anchor.answer.witness["azimuth_deg"] == 0.0
    assert closer.answer.label == closer_label
    assert closer.answer.witness["distance_ratio"] == distance_ratio


def test_anchor_frame_ignores_reference_objects_intrinsic_yaw() -> None:
    ego_script, plan, view = _case(1)
    anchor_script = _fixed_binding(CROSS_VIEW_ANCHOR[0], 1)
    base = CapabilityCompiler(anchor_script, STD_V1).compile(view, plan.binding)
    rotated_view = GeometrySceneView(
        _fixture_layout(1, reference_yaw_deg=90.0),
        view.poses,
        STD_V1,
    )
    rotated = CapabilityCompiler(anchor_script, STD_V1).compile(rotated_view, plan.binding)

    assert base.answer is not None and rotated.answer is not None
    # The anchor frame is defined by Y's position and the direction to the
    # adjacent anchor, so rotating Y in place must not move the answer.
    assert base.answer.label == rotated.answer.label == "front"
    assert base.answer.witness["imagined_yaw_deg"] == 155.0
    assert rotated.answer.witness["imagined_yaw_deg"] == 155.0


@pytest.mark.parametrize("snapshot", (False, True), ids=("walk", "snapshot"))
def test_cross_view_variant_signature_is_machine_verified(snapshot: bool) -> None:
    script, plan, view = _case(1, snapshot)
    compiler = CapabilityCompiler(script, STD_V1)
    canonical = compiler.compile(view, plan.binding, with_essential=True)
    variants = VariantBuilder(compiler, view, plan.binding, canonical).build_all(seed=17)
    by_kind = {variant.kind: variant for variant in variants}

    expected_kinds = {"permute", "drop_key", "delay"} | (
        set() if snapshot else {"drop_filler"}
    )
    assert set(by_kind) == expected_kinds
    assert by_kind["permute"].gold == canonical.answer.label  # type: ignore[union-attr]
    assert by_kind["drop_key"].certificate.status == "abstain"
    assert by_kind["drop_key"].certificate.reason == "clause:anchor_chain_evidence"
    assert by_kind["delay"].gold == canonical.answer.label  # type: ignore[union-attr]
    if not snapshot:
        assert by_kind["drop_filler"].gold == canonical.answer.label  # type: ignore[union-attr]


@pytest.mark.parametrize("snapshot", (False, True), ids=("walk", "snapshot"))
def test_cross_view_family_packages_every_chain_referent(snapshot: bool) -> None:
    script, plan, view = _case(1, snapshot)
    family = build_family_doc(view, plan.model_dump(), script, STD_V1, seed=17)
    assert family.role == "primary"
    assert set(family.referents) == {"target", "anchor1", "other"}
    assert family.episodes[0].label == "left"
    assert all(token not in family.question.text for token in ("{target}", "{other}"))


@pytest.mark.skipif(not HOME_SCENE_IR.exists(), reason="Beechwood scene_ir is unavailable")
def test_k1_chain_builds_on_real_multiroom_geometry() -> None:
    layout = layout_from_scene_ir(HOME_SCENE_IR, std=STD_V1)
    report = generate_plans(
        layout,
        CROSS_VIEW_EGO[0],
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
    labels = {
        script.capability: CapabilityCompiler(script, STD_V1)
        .compile(view, plan.binding)
        .answer.label
        for script in CROSS_VIEW_SCRIPT_SETS[CROSS_VIEW_EGO[0].capability]
    }
    # Values are copied from the compiler's deterministic output, never
    # hand-adjusted gold.
    assert labels == {
        "cross_view_ego_k1": "front",
        "cross_view_anchor_k1": "front",
        "cross_view_closer_k1": "first",
    }
    assert plan.clause_witnesses["queried_pair_never_covisible"]["covisible_frames"] == []
    assert all(
        count >= STD_V1.chain_min_covisible_frames
        for count in plan.clause_witnesses["anchor_chain_evidence"][
            "covisible_counts"
        ].values()
    )
