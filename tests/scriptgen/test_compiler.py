"""Authoritative compiler tests.

Unit tests drive a hand-built fake render backend through every status path
(answerable / abstain / invalid) without any bundle on disk; integration
tests replay the three rendered gates_bedroom trajectories and pin the
values the closed loop already verified by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from _batchdata import load_plan_record, load_render_view, needs_batch

from spatial_episode.scriptgen.compiler import CapabilityCompiler, Certificate
from spatial_episode.scriptgen.library import (
    HOMING,
    NET_TURN,
    SCRIPT_LIBRARY,
    SELF_MOTION,
    VIEW_SIDE,
)
from spatial_episode.scriptgen.sceneview import (
    Pose2D,
    ReindexedSceneView,
    SceneObject,
    VisibilityObservation,
)
from spatial_episode.scriptgen.standards import STD_V1

TARGET = SceneObject(name="tgt", category="armchair", xy=(0.0, 2.0), size_m=0.8, uid="tgt")


@dataclass(frozen=True)
class FakeRenderView:
    """Scripted render backend: poses plus a per-frame target pixel table."""

    yaws: tuple[float, ...]
    pixels: tuple[int, ...]
    positions: tuple[tuple[float, float], ...] = ()
    _objects: tuple[SceneObject, ...] = field(default=(TARGET,))

    @property
    def frame_count(self) -> int:
        return len(self.yaws)

    def camera_pose(self, t: int) -> Pose2D:
        xy = self.positions[t] if self.positions else (0.0, 0.0)
        return Pose2D(xy[0], xy[1], self.yaws[t])

    def object(self, name: str) -> SceneObject:
        return next(o for o in self._objects if o.name == name)

    def objects(self) -> list[SceneObject]:
        return list(self._objects)

    def visibility(self, name: str, t: int) -> VisibilityObservation:
        value = float(self.pixels[t]) if name == TARGET.name else 0.0
        return VisibilityObservation("render_pixels", value, 1.0)


def qualifying_fake() -> FakeRenderView:
    """Target seen at frames 0-1 dead ahead, then a tracked 105-degree turn.

    Final yaw -15 puts the target (bearing 90) at azimuth 105: sector left,
    30 degrees of margin. All clauses of SELF_MOTION pass.
    """
    yaws = (90.0, 90.0, 60.0, 30.0, 0.0, -15.0, -15.0, -15.0, -15.0, -15.0, -15.0, -15.0, -15.0)
    pixels = (5000, 5000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    return FakeRenderView(yaws=yaws, pixels=pixels)


def qualifying_all_modes_fake() -> FakeRenderView:
    """Extend the original fake with smooth motion for homing/view-side tests."""
    base = qualifying_fake()
    positions = (
        (2.0, -2.0),
        (2.0, -2.0),
        (1.5, -1.5),
        (1.0, -1.0),
        (0.5, -0.5),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    )
    return FakeRenderView(yaws=base.yaws, pixels=base.pixels, positions=positions)


BINDING = {"target": "tgt"}


@pytest.fixture(scope="module")
def compiler() -> CapabilityCompiler:
    return CapabilityCompiler(script=SELF_MOTION, std=STD_V1)


def test_answerable_certificate(compiler: CapabilityCompiler) -> None:
    cert = compiler.compile(qualifying_fake(), BINDING)
    assert SELF_MOTION.schema_version == "scriptgen_spec.v5"
    assert cert.schema_version == "scriptgen_certificate.v3"
    assert cert.template_index == 0
    assert cert.status == "answerable" and cert.reason is None
    assert cert.frame_vars == {"t_seen": 1, "t_gone": 2, "t_q": 12}
    assert cert.answer is not None
    assert cert.answer.mode == "target_sector"
    assert cert.answer.label == "left"
    assert cert.answer.witness["azimuth_deg"] == pytest.approx(105.0)
    assert cert.answer.witness["margin_deg"] == pytest.approx(30.0)
    assert cert.backend == "render_pixels"
    assert cert.knob_levels == {"delay": 10.0}
    assert [v.tristate for v in cert.target_visibility[:3]] == ["visible", "visible", "invisible"]
    # Search-only path validity belongs to acquisition, not reindexed evidence.
    assert {o.name for o in cert.clause_outcomes} == {
        c.name for c in SELF_MOTION.clauses if c.phase != "search_only"
    }
    assert all(o.holds is True for o in cert.clause_outcomes)


@pytest.mark.parametrize(
    ("script", "label", "witness"),
    [
        (
            SELF_MOTION,
            "left",
            {
                "azimuth_deg": 105.0,
                "sector": "left",
                "margin_deg": 30.0,
                "question_frame": 12,
            },
        ),
        (NET_TURN, "right", {"net_turn_deg": -105.0}),
        (
            HOMING,
            "front",
            {
                "azimuth_deg": -30.0,
                "sector": "front",
                "margin_deg": 15.0,
                "start_distance_m": 2.83,
                "question_frame": 12,
            },
        ),
        (
            VIEW_SIDE,
            "left_half",
            {"azimuth_deg": 26.6, "margin_deg": 26.6, "frame": 1},
        ),
    ],
)
def test_all_declared_answer_modes_compile_on_fake(
    script, label: str, witness: dict[str, float | int | str]
) -> None:
    cert = CapabilityCompiler(script=script, std=STD_V1).compile(
        qualifying_all_modes_fake(), BINDING
    )
    assert cert.status == "answerable"
    assert cert.answer is not None
    assert cert.answer.label == label
    assert cert.answer.witness == witness


def test_ambiguous_frame_never_counts_as_seen(compiler: CapabilityCompiler) -> None:
    """A frame inside the double-threshold gap must not resolve t_seen."""
    fake = qualifying_fake()
    pixels = list(fake.pixels)
    pixels[2] = 500  # between render_max_invisible (300) and min_visible (900)
    cert = compiler.compile(FakeRenderView(yaws=fake.yaws, pixels=tuple(pixels)), BINDING)
    # t_seen stays 1; the ambiguous frame becomes the transition zone, and
    # t_gone moves past it.
    assert cert.frame_vars["t_seen"] == 1
    assert cert.frame_vars["t_gone"] == 3
    assert cert.status == "answerable"


def test_unseen_target_is_abstain(compiler: CapabilityCompiler) -> None:
    fake = qualifying_fake()
    cert = compiler.compile(FakeRenderView(yaws=fake.yaws, pixels=(0,) * len(fake.yaws)), BINDING)
    assert cert.status == "abstain"
    assert cert.reason == "frame_var_unresolvable:t_seen"
    assert cert.answer is None


def test_untrackable_motion_is_abstain(compiler: CapabilityCompiler) -> None:
    """A snap turn beyond the per-frame bound is an evidence violation."""
    fake = qualifying_fake()
    yaws = list(fake.yaws)
    yaws[6] = yaws[5] - 90.0  # 90-degree snap: > max_step_turn_deg
    yaws[7:] = [yaws[6]] * len(yaws[7:])
    cert = compiler.compile(FakeRenderView(yaws=tuple(yaws), pixels=fake.pixels), BINDING)
    assert cert.status == "abstain"
    assert cert.reason == "clause:trackable"


def test_insufficient_turn_is_invalid(compiler: CapabilityCompiler) -> None:
    """Too small a cumulative turn voids the question (validity clause)."""
    yaws = (90.0, 90.0) + (60.0,) * 11  # only 30 degrees of total turn
    fake = qualifying_fake()
    cert = compiler.compile(FakeRenderView(yaws=yaws, pixels=fake.pixels), BINDING)
    assert cert.status == "invalid"
    assert cert.reason == "clause:turned"


def test_reindexed_view_maps_frames() -> None:
    base = qualifying_fake()
    view = ReindexedSceneView(base=base, frames=(2, 0, 0, 5))
    assert view.frame_count == 4
    assert view.camera_pose(0).yaw_deg == 60.0
    assert view.visibility("tgt", 1).value == 5000.0
    assert view.visibility("tgt", 2).value == 5000.0
    with pytest.raises(ValueError):
        ReindexedSceneView(base=base, frames=(99,))
    with pytest.raises(ValueError):
        ReindexedSceneView(base=base, frames=())


def test_essential_frames_leave_one_out(compiler: CapabilityCompiler) -> None:
    cert = compiler.compile(qualifying_fake(), BINDING, with_essential=True)
    assert cert.essential_frames is not None
    # The turn frames cannot be dropped: merging two 30-degree steps breaks
    # the 40-degree trackability bound.
    assert {2, 3} <= set(cert.essential_frames)
    # A redundant sighting frame and a constant-yaw filler frame can.
    assert 0 not in cert.essential_frames
    assert 8 not in cert.essential_frames


def test_geometry_disagreement_sets_mismatch(compiler: CapabilityCompiler) -> None:
    plan = {
        "standard_version": STD_V1.standard_version,
        "frame_vars": {"t_seen": 0, "t_gone": 2, "t_q": 12},
        "provisional_answer": {"sector": "right", "azimuth_deg": -100.0},
    }
    cert = compiler.compile(qualifying_fake(), BINDING, geometry_plan=plan)
    assert cert.status == "answerable"
    assert cert.mismatch == "label_disagreement:render=left,geometry=right"


def test_every_library_spec_has_mandatory_traversal_clauses() -> None:
    for script in SCRIPT_LIBRARY.values():
        # Snapshot sequences teleport between stations, so only the stations
        # themselves must be collision-free; there is no walked path to check.
        expected = [("poses_clear", "poses_clear", "search_only")]
        if script.motifs != ("snapshot_landmarks",):
            expected.append(("path_clear", "path_clear", "search_only"))
        traversal = script.clauses[: len(expected)]
        assert [
            (clause.name, clause.predicate, clause.phase) for clause in traversal
        ] == expected
        assert all(clause.args == {"frames": "0:$t_q"} for clause in traversal)


def test_standard_drift_sets_mismatch(compiler: CapabilityCompiler) -> None:
    plan = {"standard_version": "std.v0", "provisional_answer": {"sector": "left"}}
    cert = compiler.compile(qualifying_fake(), BINDING, geometry_plan=plan)
    assert cert.mismatch is not None and cert.mismatch.startswith("standard_version_drift")


def test_v11_requires_regeneration_of_prior_plan_standards(
    compiler: CapabilityCompiler,
) -> None:
    for version in ("std.v10", "std.v9", "std.v3"):
        plan = {"standard_version": version, "provisional_answer": {"sector": "left"}}
        blocked = compiler.compile(qualifying_fake(), BINDING, geometry_plan=plan)
        assert blocked.mismatch == f"standard_version_drift:plan={version},compile=std.v11"


# --- integration: the three rendered gates_bedroom trajectories ---

EXPECTED = {
    0: {
        "label": "left",
        "azimuth_deg": 97.1,
        "margin_deg": 37.9,
        "t_seen": 1,
        "t_gone": 2,
        "t_q": 11,
    },
    1: {
        "label": "left",
        "azimuth_deg": 77.4,
        "margin_deg": 32.4,
        "t_seen": 1,
        "t_gone": 2,
        "t_q": 13,
    },
    2: {
        "label": "right",
        "azimuth_deg": -84.5,
        "margin_deg": 39.5,
        "t_seen": 1,
        "t_gone": 2,
        "t_q": 10,
    },
}


@needs_batch
@pytest.mark.parametrize("index", [0, 1, 2])
def test_rendered_bundles_compile_answerable(
    index: int, self_motion_compiler: CapabilityCompiler
) -> None:
    plan = load_plan_record(index)
    view = load_render_view(index)
    cert = self_motion_compiler.compile(view, plan["binding"], geometry_plan=plan)
    expected = EXPECTED[index]
    assert cert.status == "answerable"
    assert cert.mismatch is None
    assert cert.answer is not None and cert.answer.label == expected["label"]
    assert cert.answer.witness == {
        "azimuth_deg": expected["azimuth_deg"],
        "sector": expected["label"],
        "margin_deg": expected["margin_deg"],
        "question_frame": expected["t_q"],
    }
    assert cert.frame_vars == {key: expected[key] for key in ("t_seen", "t_gone", "t_q")}
    assert cert.answer.witness["margin_deg"] >= STD_V1.sector_margin_deg
    # Round-trips through JSON as a frozen contract (witness dicts hold Any,
    # so equality is on the serialised form, not tuple-vs-list identity).
    serialized = cert.model_dump_json()
    assert Certificate.model_validate_json(serialized).model_dump_json() == serialized


@needs_batch
def test_render_overrides_geometry_frame_vars(
    self_motion_compiler: CapabilityCompiler,
) -> None:
    """render_0: geometry picks frame 0, masks keep the target clear through frame 1."""
    plan = load_plan_record(0)
    cert = self_motion_compiler.compile(load_render_view(0), plan["binding"], geometry_plan=plan)
    assert plan["frame_vars"]["t_seen"] == 0
    assert cert.frame_vars["t_seen"] == 1
    assert cert.geometry is not None and cert.geometry.frame_vars["t_seen"] == 0
