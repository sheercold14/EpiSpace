"""Old walking cross-view bundles recompile into eval-pool question groups."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spatial_episode.scriptgen.cross_view_recompile import _OLD_CAPABILITY, _walking_trio
from spatial_episode.scriptgen.family import FamilyBlocked, build_question_group
from spatial_episode.scriptgen.library import (
    CROSS_VIEW_ANCHOR,
    CROSS_VIEW_CLOSER,
    CROSS_VIEW_EGO,
)
from spatial_episode.scriptgen.standards import STD_V1

OLD_COLLECTION = Path(
    "/data/shichao/data/dataV100/code/OminiGibson/outputs/scripted_demo/"
    "scriptgen_current_2x_v1"
)
OLD_CLOSER_JOB = "cross_view_closer_k1__r0__Benevolence_1_int_seed17"


def test_old_capability_pattern_and_trio_mapping() -> None:
    assert _OLD_CAPABILITY.match("cross_view_pair_relation_k2").group(1) == "2"
    assert _OLD_CAPABILITY.match("cross_view_closer_k3").group(1) == "3"
    assert _OLD_CAPABILITY.match("cross_view_ego_k1") is None
    assert _OLD_CAPABILITY.match("cross_view_snapshot_closer_k1") is None
    assert _walking_trio(2) == (CROSS_VIEW_EGO[1], CROSS_VIEW_ANCHOR[1], CROSS_VIEW_CLOSER[1])


def test_non_canonical_recompile_requires_explicit_scripts(tmp_path: Path) -> None:
    plan_record = tmp_path / "plan.record.json"
    plan_record.write_text(json.dumps({"plan_id": "p", "binding": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="explicit scripts"):
        build_question_group(
            tmp_path / "bundle",
            plan_record,
            tmp_path / "scene_ir.json",
            tmp_path / "out",
            STD_V1,
            canonical_plan=False,
        )


@pytest.mark.skipif(not OLD_COLLECTION.exists(), reason="old collection is unavailable")
def test_old_closer_bundle_recompiles_only_without_canonical_authority(
    tmp_path: Path,
) -> None:
    bundle = OLD_COLLECTION / "bundles" / OLD_CLOSER_JOB
    plan_record = OLD_COLLECTION / "plans" / f"{OLD_CLOSER_JOB}.record.json"
    jobs = json.loads((OLD_COLLECTION / "collection.plan.json").read_text(encoding="utf-8"))[
        "jobs"
    ]
    scene_ir = Path(
        next(job for job in jobs if job["job_id"] == OLD_CLOSER_JOB)["scene"]["scene_ir"]
    )
    trio = _walking_trio(1)

    # The plan still names a live capability but predates the std.v11 lineage,
    # so treating it as canonical must block on standard drift.
    with pytest.raises(FamilyBlocked, match="standard_version_drift"):
        build_question_group(
            bundle, plan_record, scene_ir, tmp_path / "canonical", STD_V1, trio
        )

    group_path = build_question_group(
        bundle,
        plan_record,
        scene_ir,
        tmp_path / "recompiled",
        STD_V1,
        trio,
        canonical_plan=False,
    )
    group = json.loads(group_path.read_text(encoding="utf-8"))
    entries = {question["capability"]: question for question in group["questions"]}
    assert set(entries) == {
        "cross_view_ego_k1",
        "cross_view_anchor_k1",
        "cross_view_closer_k1",
    }
    # Values are copied from the tool's deterministic output on this bundle.
    assert entries["cross_view_closer_k1"]["label"] == "second"
    assert entries["cross_view_closer_k1"]["skip_reason"] is None
