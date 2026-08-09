"""Family packing tests.

Unit tests audit the blocking rules on the fake backend; the integration
test runs the real one-command path on a rendered bundle into a temp dir
and checks the shipped artifacts.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pytest
from _batchdata import BATCH_ROOT, SCENE_IR_PATH, load_render_view, needs_batch
from test_compiler import TARGET, FakeRenderView, qualifying_fake

from spatial_episode.contracts.schema import CONTRACTS
from spatial_episode.scriptgen.family import (
    FamilyBlocked,
    ScriptgenFamilyV4,
    ScriptgenQuestionGroupV1,
    build_family_doc,
    build_family_site,
    build_question_group,
)
from spatial_episode.scriptgen.library import SELF_MOTION
from spatial_episode.scriptgen.sceneview import SceneLayout, SceneObject
from spatial_episode.scriptgen.spec import Template
from spatial_episode.scriptgen.standards import STD_V1

PLAN = {
    "plan_id": "fake-plan",
    "scene_id": "fake-scene",
    "capability": "self_motion_update",
    "standard_version": STD_V1.standard_version,
    "binding": {"target": "tgt"},
    "frame_vars": {"t_seen": 1, "t_gone": 2, "t_q": 12},
    "provisional_answer": {"sector": "left", "azimuth_deg": 105.0},
}


@dataclass(frozen=True)
class FakeBundleView(FakeRenderView):
    """Fake render backend that also exposes a scene layout (for audits)."""

    layout: SceneLayout = SceneLayout(scene_id="fake-scene", objects=(TARGET,))


def fake_bundle_view(extra_objects: tuple[SceneObject, ...] = ()) -> FakeBundleView:
    base = qualifying_fake()
    return FakeBundleView(
        yaws=base.yaws,
        pixels=base.pixels,
        layout=SceneLayout(scene_id="fake-scene", objects=(TARGET, *extra_objects)),
        _objects=(TARGET, *extra_objects),
    )


def test_family_doc_shape() -> None:
    doc = build_family_doc(fake_bundle_view(), PLAN, SELF_MOTION, STD_V1, seed=3)
    assert doc.schema_version == "scriptgen_family.v4"
    assert doc.role == "primary"
    assert doc.referents == {"target": doc.target}
    assert doc.question_group_id == "fake-plan.question_group"
    assert [e.kind for e in doc.episodes] == [
        "canonical",
        "permute",
        "drop_key",
        "drop_filler",
        "delay",
    ]
    labels = {e.kind: e.label for e in doc.episodes}
    assert labels["canonical"] == "left"
    assert labels["permute"] == labels["drop_key"] == "无法判断"
    assert labels["drop_filler"] == labels["delay"] == "left"
    # One shared verbatim question, target filled, no leftover braces.
    assert doc.target.category in doc.question.text
    assert "{" not in doc.question.text
    # Media paths are relative (decision #2).
    assert all(not f.rgb.startswith("/") for f in doc.frames)
    serialized = doc.model_dump_json()
    assert ScriptgenFamilyV4.model_validate_json(serialized).model_dump_json() == serialized


def test_duplicate_referent_blocks() -> None:
    twin = SceneObject(name="tgt2", category="armchair", xy=(3.0, 3.0), size_m=0.8, uid="tgt2")
    with pytest.raises(FamilyBlocked, match="referent"):
        build_family_doc(fake_bundle_view((twin,)), PLAN, SELF_MOTION, STD_V1, seed=3)


def test_frame_number_leak_blocks() -> None:
    leaky = SELF_MOTION.model_copy(
        update={
            "templates": (
                Template(
                    text="你在第 3 帧看到过{target}。它在你的哪个方向?",
                    options=SELF_MOTION.templates[0].options,
                ),
            )
        }
    )
    with pytest.raises(FamilyBlocked, match="leak"):
        build_family_doc(fake_bundle_view(), PLAN, leaky, STD_V1, seed=3)


def test_geometry_disagreement_blocks() -> None:
    plan = {**PLAN, "provisional_answer": {"sector": "right", "azimuth_deg": -100.0}}
    with pytest.raises(FamilyBlocked, match="mismatch"):
        build_family_doc(fake_bundle_view(), plan, SELF_MOTION, STD_V1, seed=3)


def test_family_schema_registered_in_contracts() -> None:
    assert CONTRACTS["scriptgen_family.v4.schema.json"] is ScriptgenFamilyV4
    assert "scriptgen_family.v3.schema.json" not in CONTRACTS
    schema = ScriptgenFamilyV4.model_json_schema()
    assert schema["properties"]["schema_version"]["const"] == "scriptgen_family.v4"


def test_selected_template_drives_validation_and_packaging() -> None:
    alternate = Template(
        text="请根据完整序列判断:{target}最终相对你的方向是什么?",
        options=SELF_MOTION.templates[0].options,
    )
    script = SELF_MOTION.model_copy(
        update={
            "templates": (
                Template(text="无效候选:{target}?", options=("right", "无法判断")),
                alternate,
            )
        }
    )
    with pytest.raises(ValueError, match="outside question options"):
        build_family_doc(fake_bundle_view(), PLAN, script, STD_V1, seed=3)

    doc = build_family_doc(fake_bundle_view(), PLAN, script, STD_V1, seed=3, template_index=1)
    assert doc.question.text == alternate.text.format(target=TARGET.category)
    assert all(episode.certificate.template_index == 1 for episode in doc.episodes)


@needs_batch
def test_one_command_site_from_render_0(tmp_path: Path) -> None:
    family_path = build_family_site(
        BATCH_ROOT / "render_0",
        BATCH_ROOT / "plan_0.record.json",
        SCENE_IR_PATH,
        tmp_path / "f0",
        STD_V1,
        seed=17,
    )
    doc = ScriptgenFamilyV4.model_validate_json(family_path.read_text(encoding="utf-8"))
    assert {e.kind: e.label for e in doc.episodes} == {
        "canonical": "left",
        "permute": "无法判断",
        "drop_key": "无法判断",
        "drop_filler": "left",
        "delay": "left",
    }
    assert doc.checks.referent_unique
    # Certificates all carry the render backend and the frozen standard.
    assert all(e.certificate.backend == "render_pixels" for e in doc.episodes)
    assert all(e.certificate.standard_version == STD_V1.standard_version for e in doc.episodes)
    # The review page and every referenced frame image were shipped.
    assert (family_path.parent / "index.html").exists()
    for frame in doc.frames:
        for path in (frame.rgb, frame.depth, frame.instance):
            assert (family_path.parent / path).exists(), path
    # The document is the single source: it parses as plain JSON too.
    payload = json.loads(family_path.read_text(encoding="utf-8"))
    assert payload["family_id"].endswith(".family.s17")


def _wrap_deg(angle: float) -> float:
    wrapped = math.fmod(angle, 360.0)
    if wrapped <= -180.0:
        wrapped += 360.0
    elif wrapped > 180.0:
        wrapped -= 360.0
    return wrapped


def _azimuth(from_xy: tuple[float, float], yaw: float, to_xy: tuple[float, float]) -> float:
    bearing = math.degrees(math.atan2(to_xy[1] - from_xy[1], to_xy[0] - from_xy[0]))
    return _wrap_deg(bearing - yaw)


def _sector(azimuth: float) -> str:
    if -45.0 < azimuth <= 45.0:
        return "front"
    if 45.0 < azimuth <= 135.0:
        return "left"
    if -135.0 < azimuth <= -45.0:
        return "right"
    return "back"


def _sector_margin(azimuth: float) -> float:
    boundaries = (-135.0, -45.0, 45.0, 135.0, 180.0)
    return min(abs(_wrap_deg(azimuth - boundary)) for boundary in boundaries)


@needs_batch
@pytest.mark.parametrize("index", [0, 1, 2])
def test_wallfix_question_group_geometry_and_skips(index: int, tmp_path: Path) -> None:
    group_path = build_question_group(
        BATCH_ROOT / f"render_{index}",
        BATCH_ROOT / f"plan_{index}.record.json",
        SCENE_IR_PATH,
        tmp_path / f"group_{index}",
        STD_V1,
        seed=17,
    )
    group = ScriptgenQuestionGroupV1.model_validate_json(group_path.read_text(encoding="utf-8"))
    questions = {question.capability: question for question in group.questions}
    assert set(questions) == {
        "self_motion_update",
        "path_integration",
        "path_integration_magnitude",
        "homing_probe",
        "view_side_check",
        "existence_sufficiency_bed",
    }

    view = load_render_view(index)
    poses = [view.camera_pose(t) for t in range(view.frame_count)]
    expected_direction = ("left", "left", "right")[index]
    assert questions["self_motion_update"].label == expected_direction

    net_turn = sum(
        _wrap_deg(later.yaw_deg - earlier.yaw_deg) for earlier, later in zip(poses, poses[1:])
    )
    expected_net = "left" if net_turn > 0.0 else "right"
    assert questions["path_integration"].label == expected_net
    assert questions["path_integration_magnitude"].label == "at_most_90"

    start, end = poses[0], poses[-1]
    start_distance = math.dist(start.xy, end.xy)
    start_azimuth = _azimuth(end.xy, end.yaw_deg, start.xy)
    homing_qualifies = (
        start_distance >= STD_V1.homing_min_distance_m
        and _sector_margin(start_azimuth) >= STD_V1.sector_margin_deg
    )
    homing = questions["homing_probe"]
    if homing_qualifies:
        assert homing.label == _sector(start_azimuth)
        assert homing.skip_reason is None
    else:
        assert homing.label is None and homing.family_id is None
        assert homing.skip_reason == "invalid:clause:start_sector_margin_ok"

    target = view.object("72faab69-77d3-53b7-9c1e-2feeb010c0f2")
    visible = [
        t
        for t in range(view.frame_count)
        if view.visibility(target.name, t).tristate(STD_V1) is True
    ]
    t_seen = visible[-1]
    seen_pose = poses[t_seen]
    side_azimuth = _azimuth(seen_pose.xy, seen_pose.yaw_deg, target.xy)
    view_side = questions["view_side_check"]
    if abs(side_azimuth) >= STD_V1.view_side_margin_deg:
        assert view_side.label == ("left_half" if side_azimuth > 0.0 else "right_half")
        assert view_side.skip_reason is None
    else:
        assert view_side.label is None
        assert view_side.skip_reason == "invalid:clause:view_side_margin_ok"

    existence = questions["existence_sufficiency_bed"]
    assert existence.label is None and existence.family_id is None
    assert existence.skip_reason == "invalid:clause:category_absent"

    for capability, question in questions.items():
        if question.family is None:
            continue
        family_path = group_path.parent / question.family
        family = ScriptgenFamilyV4.model_validate_json(family_path.read_text(encoding="utf-8"))
        canonical = next(ep for ep in family.episodes if ep.kind == "canonical")
        assert canonical.label == question.label
        assert family.question_group_id == group.question_group_id
        assert family.frames[0].rgb.startswith("../media/")
        if capability == "self_motion_update":
            assert canonical.certificate.geometry is not None
            assert canonical.certificate.geometry.standard_version == "std.v3"
        else:
            assert canonical.certificate.geometry is None
        assert (family_path.parent / "index.html").exists()
        assert (family_path.parent / family.frames[0].rgb).exists()

    view_family_path = group_path.parent / (questions["view_side_check"].family or "")
    if questions["view_side_check"].family is not None:
        view_family = ScriptgenFamilyV4.model_validate_json(
            view_family_path.read_text(encoding="utf-8")
        )
        episodes = {episode.kind: episode for episode in view_family.episodes}
        assert episodes["permute"].label == episodes["canonical"].label
