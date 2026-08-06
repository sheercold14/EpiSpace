"""Family packing tests.

Unit tests audit the blocking rules on the fake backend; the integration
test runs the real one-command path on a rendered bundle into a temp dir
and checks the shipped artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from spatial_episode.contracts.schema import CONTRACTS
from spatial_episode.scriptgen.family import (
    FamilyBlocked,
    ScriptgenFamilyV1,
    build_family_doc,
    build_family_site,
)
from spatial_episode.scriptgen.library import SELF_MOTION
from spatial_episode.scriptgen.sceneview import SceneLayout, SceneObject
from spatial_episode.scriptgen.spec import Template
from spatial_episode.scriptgen.standards import STD_V1

from _batchdata import BATCH_ROOT, SCENE_IR_PATH, needs_batch
from test_compiler import TARGET, FakeRenderView, qualifying_fake

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
    assert doc.schema_version == "scriptgen_family.v1"
    assert [e.kind for e in doc.episodes] == [
        "canonical",
        "permute",
        "drop_key",
        "drop_filler",
        "delay",
    ]
    golds = {e.kind: e.gold for e in doc.episodes}
    assert golds["canonical"] == "left"
    assert golds["permute"] == golds["drop_key"] == "无法判断"
    assert golds["drop_filler"] == golds["delay"] == "left"
    # One shared verbatim question, target filled, no leftover braces.
    assert doc.target.category in doc.question.text
    assert "{" not in doc.question.text
    # Media paths are relative (decision #2).
    assert all(not f.rgb.startswith("/") for f in doc.frames)
    serialized = doc.model_dump_json()
    assert ScriptgenFamilyV1.model_validate_json(serialized).model_dump_json() == serialized


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
    assert CONTRACTS["scriptgen_family.v1.schema.json"] is ScriptgenFamilyV1
    schema = ScriptgenFamilyV1.model_json_schema()
    assert schema["properties"]["schema_version"]["const"] == "scriptgen_family.v1"


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
    doc = ScriptgenFamilyV1.model_validate_json(family_path.read_text(encoding="utf-8"))
    assert {e.kind: e.gold for e in doc.episodes} == {
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
