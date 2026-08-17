"""Binding-coverage planning, diversity, and shared-credit contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from spatial_episode.scriptgen.binding_coverage import (
    CandidateSearchResult,
    CoverageCandidate,
    CoverageCell,
    CoverageManifest,
    DEFERRED_INITIAL_CAPABILITIES,
    _binding_cache_key,
    _bindings_for_scene,
    _confirmed_render_unresolvable_targets,
    _completed_scene_prefix,
    _derived_seed,
    _geometry_pool,
    _new_candidates,
    _scene_occupancy_proxy_rejection,
    _validate_rendered_candidate,
    compatible_credit_cell_ids,
    initialize_coverage_status,
    load_coverage,
    plan_coverage,
    run_coverage,
    run_coverage_pipeline,
    trajectories_are_diverse,
    verify_coverage_sources,
)
from spatial_episode.scriptgen.collection import CollectionScene
from spatial_episode.scriptgen.family import (
    FamilyBlocked,
    QuestionGroupEntry,
    QuestionGroupTrajectory,
    ScriptgenQuestionGroupV1,
)
from spatial_episode.scriptgen.plan import (
    PlannedPose,
    ProvisionalAnswer,
    TrajectoryPlan,
)
from spatial_episode.scriptgen.source_inventory import SourceIndex, SourceSceneRecord
from spatial_episode.scriptgen.standards import STD_V1


def _plan(
    plan_id: str,
    *,
    start_x: float = 0.0,
    end_x: float = 1.0,
    end_yaw: float = 90.0,
    t_q: int = 2,
    capability: str = "self_motion_update",
) -> TrajectoryPlan:
    return TrajectoryPlan(
        plan_id=plan_id,
        scene_id="scene",
        capability=capability,
        standard_version=STD_V1.standard_version,
        seed=17,
        binding={"target": "target"},
        frame_vars={"t_seen": 0, "t_gone": 1, "t_q": t_q},
        poses=(
            PlannedPose(frame=0, x=start_x, y=0.0, yaw_deg=0.0),
            PlannedPose(frame=1, x=(start_x + end_x) / 2.0, y=0.0, yaw_deg=end_yaw / 2),
            PlannedPose(frame=2, x=end_x, y=0.0, yaw_deg=end_yaw),
        ),
        knob_levels={},
        clause_witnesses={},
        provisional_answer=ProvisionalAnswer(mode="target_sector", label="front", witness={}),
    )


def _scene(tmp_path: Path) -> CollectionScene:
    scene_ir = tmp_path / "scene_ir.json"
    recipe = tmp_path / "recipe.yaml"
    scene_ir.write_text('{"scene_id":"scene"}\n', encoding="utf-8")
    recipe.write_text("source: {}\n", encoding="utf-8")
    digest = hashlib.sha256(scene_ir.read_bytes()).hexdigest()
    return CollectionScene(
        scene_key="scene",
        scene_id="scene",
        source_scene_id="scene:best",
        scene_model="scene",
        scene_instance=None,
        source_digest="source",
        scene_ir=str(scene_ir),
        source_recipe=str(recipe),
        scene_ir_sha256=digest,
        source_recipe_sha256=hashlib.sha256(recipe.read_bytes()).hexdigest(),
    )


def test_structural_diversity_rejects_jitter_and_accepts_real_changes() -> None:
    base = _plan("base.a0")
    jitter = _plan("jitter.a1", start_x=0.05, end_x=1.05, end_yaw=94.0)
    different_start = _plan("start.a2", start_x=0.6, end_x=1.0)
    different_turn = _plan("turn.a3", end_yaw=130.0)
    different_timing = _plan("timing.a4", t_q=4)

    assert not trajectories_are_diverse(jitter, base)
    assert trajectories_are_diverse(different_start, base)
    assert trajectories_are_diverse(different_turn, base)
    assert trajectories_are_diverse(different_timing, base)


def test_binding_seed_is_stable_and_binding_specific() -> None:
    first = _derived_seed("collection", "scene", "self_motion_update", {"target": "a"})
    second = _derived_seed("collection", "scene", "self_motion_update", {"target": "a"})
    other = _derived_seed("collection", "scene", "self_motion_update", {"target": "b"})

    assert first == second
    assert first != other


@pytest.mark.parametrize(
    "capability",
    ("self_motion_update", "existence_sufficiency_bed"),
)
def test_every_single_slot_capability_enumerates_all_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capability: str,
) -> None:
    scene = _scene(tmp_path)
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage.layout_from_scene_ir",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage._scene_can_bind",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage.iter_bindings",
        lambda *args, **kwargs: (
            iter(({"target": "object-a"}, {"target": "object-b"})),
            [],
        ),
    )

    bindings = _bindings_for_scene(
        scene,
        capability,
        maximum_multislot_bindings=1,
        std=STD_V1,
    )

    assert bindings == ({"target": "object-a"}, {"target": "object-b"})


def test_source_registry_absence_does_not_drop_otherwise_valid_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = _scene(tmp_path)
    Path(scene.scene_ir).write_text(
        json.dumps(
            {
                "scene_id": "scene",
                "entities": [
                    {"entity_id": "object-a", "source_entity_id": "source-a"},
                    {"entity_id": "object-b", "source_entity_id": "source-b"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (Path(scene.scene_ir).parent / "scene_snapshot.json").write_text(
        json.dumps({"runtime_instance_registry": {"91": "source-a"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage.layout_from_scene_ir",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage._scene_can_bind",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage.iter_bindings",
        lambda *args, **kwargs: (
            iter(({"target": "object-a"}, {"target": "object-b"})),
            [],
        ),
    )

    bindings = _bindings_for_scene(
        scene,
        "self_motion_update",
        maximum_multislot_bindings=1,
        std=STD_V1,
    )

    # A static source sweep may not observe every valid object. Runtime-id
    # absence only becomes authoritative after repeated scripted replays.
    assert bindings == ({"target": "object-a"}, {"target": "object-b"})


def test_binding_enumeration_cache_groups_shared_capability_shapes() -> None:
    assert _binding_cache_key("self_motion_update") == _binding_cache_key(
        "path_integration"
    )
    assert _binding_cache_key("reference_frame_transform") == _binding_cache_key(
        "reference_frame_visibility_yaw180"
    )
    assert _binding_cache_key("cross_view_pair_relation_k1") == _binding_cache_key(
        "cross_view_closer_k1"
    )
    assert _binding_cache_key("cross_view_pair_relation_k1") != _binding_cache_key(
        "cross_view_closer_k2"
    )


def test_shared_question_consumers_defer_initial_geometry_search() -> None:
    assert {
        "path_integration",
        "reference_frame_visibility_yaw90",
        "cross_view_closer_k1",
        "existence_sufficiency_bed",
    }.issubset(DEFERRED_INITIAL_CAPABILITIES)
    assert "self_motion_update" not in DEFERRED_INITIAL_CAPABILITIES
    assert "reference_frame_transform" not in DEFERRED_INITIAL_CAPABILITIES
    assert "cross_view_pair_relation_k1" not in DEFERRED_INITIAL_CAPABILITIES


def test_initial_search_uses_a_small_frontier_before_the_150_attempt_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = _scene(tmp_path)
    cell = CoverageCell(
        cell_id="cell",
        scene_key="scene",
        scene_id="scene",
        capability="self_motion_update",
        binding={"target": "target"},
        seed=17,
        target_accepted=10,
        geometry_pool_size=0,
        diverse_pool_size=0,
        search_attempt_limit=0,
        raw_plan_limit=0,
    )
    manifest = CoverageManifest(
        collection_id="coverage",
        standard_version=STD_V1.standard_version,
        source_index=str(tmp_path / "source.index.json"),
        source_index_sha256="0" * 64,
        output_root=str(tmp_path),
        accepted_per_binding=10,
        attempts_per_binding=150,
        initial_attempts_per_binding=30,
        maximum_multislot_bindings=128,
        capabilities=("self_motion_update",),
        scenes=(scene,),
        cells=(cell,),
    )
    observed: dict[str, int] = {}

    def geometry_pool(*args, **kwargs):
        observed["attempts"] = kwargs["attempts_per_binding"]
        observed["plans"] = kwargs["plans_per_binding"]
        return (), {"binding_exhausted": 1}

    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage._geometry_pool", geometry_pool
    )
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage.layout_from_scene_ir",
        lambda *args, **kwargs: object(),
    )

    search = _new_candidates(
        cell,
        scene,
        manifest,
        requested=10,
        attempt_limit=30,
    )

    assert observed == {"attempts": 30, "plans": 20}
    assert search.search_attempt_limit == 30
    assert search.raw_plan_limit == 20
    assert not search.search_pool_exhausted


def test_label_aware_geometry_pool_scans_frontier_and_keeps_only_desired_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = _scene(tmp_path)
    cell = CoverageCell(
        cell_id="cell",
        scene_key="scene",
        scene_id="scene",
        capability="self_motion_update_pure_rotation",
        binding={"target": "target"},
        seed=17,
        desired_answer_label="left",
        target_accepted=3,
        geometry_pool_size=0,
        diverse_pool_size=0,
    )
    left = _plan(
        "left.a0", capability="self_motion_update_pure_rotation"
    ).model_copy(
        update={
            "provisional_answer": ProvisionalAnswer(
                mode="target_sector", label="left", witness={}
            )
        }
    )
    back = _plan(
        "back.a1", capability="self_motion_update_pure_rotation"
    ).model_copy(
        update={
            "provisional_answer": ProvisionalAnswer(
                mode="target_sector", label="back", witness={}
            )
        }
    )
    observed = {}

    def fake_generate(*args, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(plans=(left, back), rejection_counts={})

    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage.layout_from_scene_ir",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage.generate_plans", fake_generate
    )

    plans, rejection_counts = _geometry_pool(
        cell,
        scene,
        attempts_per_binding=30,
        plans_per_binding=6,
        std=STD_V1,
    )

    assert plans == (left,)
    assert rejection_counts == {"answer_label:not_left": 1}
    assert observed["plans_per_binding"] == 30


def _write_coverage_manifest(
    tmp_path: Path,
    *,
    cells: tuple[CoverageCell, ...],
    scenes: tuple[CollectionScene, ...],
) -> Path:
    manifest = CoverageManifest(
        collection_id="coverage",
        standard_version=STD_V1.standard_version,
        source_index=str(tmp_path / "source.index.json"),
        source_index_sha256="0" * 64,
        output_root=str(tmp_path),
        accepted_per_binding=10,
        attempts_per_binding=150,
        initial_attempts_per_binding=30,
        maximum_multislot_bindings=128,
        capabilities=("self_motion_update",),
        scenes=scenes,
        cells=cells,
    )
    path = tmp_path / "coverage.plan.json"
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def test_render_pass_does_not_backfill_or_rewrite_manifest_while_planner_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = _scene(tmp_path)
    cell = CoverageCell(
        cell_id="cell",
        scene_key="scene",
        scene_id="scene",
        capability="self_motion_update",
        binding={"target": "target"},
        seed=17,
        target_accepted=10,
        geometry_pool_size=0,
        diverse_pool_size=0,
        search_attempt_limit=30,
        raw_plan_limit=20,
    )
    manifest_path = _write_coverage_manifest(tmp_path, cells=(cell,), scenes=(scene,))
    before = manifest_path.read_bytes()
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage.verify_coverage_sources",
        lambda manifest: None,
    )

    status_path = run_coverage(
        manifest_path,
        og_root=tmp_path,
        conda_env="behavior",
        data_root=tmp_path,
        gpu_ids=(0,),
        scene_keys=("scene",),
        allow_backfill=False,
        skip_preflight=True,
    )

    assert manifest_path.read_bytes() == before
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["cells"]["cell"]["status"] == "awaiting_candidates"


def test_initialize_status_without_rendering_and_validate_existing_index(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    cell = CoverageCell(
        cell_id="cell",
        scene_key="scene",
        scene_id="scene",
        capability="self_motion_update",
        binding={"target": "target"},
        seed=17,
        target_accepted=10,
        geometry_pool_size=0,
        diverse_pool_size=0,
    )
    manifest_path = _write_coverage_manifest(tmp_path, cells=(cell,), scenes=(scene,))

    status_path = initialize_coverage_status(manifest_path)
    first = status_path.read_bytes()
    assert initialize_coverage_status(manifest_path) == status_path
    assert status_path.read_bytes() == first
    status = json.loads(first)
    assert status["cells"]["cell"] == {
        "status": "planned",
        "accepted_episode_ids": [],
        "candidate_statuses": {},
    }


def test_question_variant_assertion_is_a_candidate_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = _scene(tmp_path)
    plan = _plan("plan.a0")
    candidate = CoverageCandidate(
        candidate_id="candidate",
        plan_id=plan.plan_id,
        attempt_index=0,
        plan_record=str(tmp_path / "candidate.record.json"),
        render_plan=str(tmp_path / "candidate.views.json"),
        recipe=str(tmp_path / "candidate.yaml"),
        bundle=str(tmp_path / "bundle"),
        group=str(tmp_path / "group"),
        log=str(tmp_path / "candidate.log"),
    )
    Path(candidate.plan_record).write_text(
        plan.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage.RenderSceneView.from_bundle",
        lambda *args, **kwargs: object(),
    )
    certificate = type(
        "Certificate",
        (),
        {"clause_outcomes": (), "status": "answerable", "mismatch": None, "reason": None},
    )()
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage.CapabilityCompiler.compile",
        lambda *args, **kwargs: certificate,
    )
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage.build_question_group",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("variant permute: no candidate sequences")
        ),
    )
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage._write_json", lambda *args, **kwargs: None
    )

    with pytest.raises(
        FamilyBlocked,
        match="question_variant_unavailable:variant permute: no candidate sequences",
    ):
        _validate_rendered_candidate(
            candidate,
            CoverageCell(
                cell_id="cell",
                scene_key="scene",
                scene_id="scene",
                capability="self_motion_update",
                binding={"target": "target"},
                seed=17,
                target_accepted=10,
                geometry_pool_size=1,
                diverse_pool_size=1,
                candidates=(candidate,),
            ),
            scene,
            std=STD_V1,
        )


def test_render_only_unresolvable_verdict_never_rewrites_planner_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = _scene(tmp_path)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "render_report.json").write_text(
        json.dumps({"status": "success"}),
        encoding="utf-8",
    )
    candidate = CoverageCandidate(
        candidate_id="candidate",
        plan_id="plan.a0",
        attempt_index=0,
        plan_record=str(tmp_path / "candidate.record.json"),
        render_plan=str(tmp_path / "candidate.views.json"),
        recipe=str(tmp_path / "candidate.yaml"),
        bundle=str(bundle),
        group=str(tmp_path / "group"),
        log=str(tmp_path / "candidate.log"),
    )
    cell = CoverageCell(
        cell_id="cell",
        scene_key="scene",
        scene_id="scene",
        capability="self_motion_update",
        binding={"target": "target"},
        seed=17,
        target_accepted=10,
        geometry_pool_size=1,
        diverse_pool_size=1,
        search_attempt_limit=30,
        raw_plan_limit=20,
        candidates=(candidate,),
    )
    manifest_path = _write_coverage_manifest(tmp_path, cells=(cell,), scenes=(scene,))
    before = manifest_path.read_bytes()
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage.verify_coverage_sources",
        lambda manifest: None,
    )
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage._validate_rendered_candidate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            FamilyBlocked("frame_var_unresolvable:t_seen")
        ),
    )
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage._confirmed_render_unresolvable_targets",
        lambda *args, **kwargs: ("target",),
    )

    status_path = run_coverage(
        manifest_path,
        og_root=tmp_path,
        conda_env="behavior",
        data_root=tmp_path,
        gpu_ids=(0,),
        scene_keys=("scene",),
        allow_backfill=False,
        skip_preflight=True,
    )

    assert manifest_path.read_bytes() == before
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["cells"]["cell"]["status"] == "exhausted"
    assert status["cells"]["cell"]["candidate_statuses"]["candidate"] == {
        "status": "rejected",
        "reason": "authority:FamilyBlocked:frame_var_unresolvable:t_seen",
        "gpu_id": 0,
    }


def test_authority_keyerror_rejection_is_revalidated_after_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = _scene(tmp_path)
    plan = _plan("plan.a0")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "render_report.json").write_text(
        json.dumps({"status": "success"}),
        encoding="utf-8",
    )
    plan_record = tmp_path / "candidate.record.json"
    plan_record.write_text(plan.model_dump_json(indent=2) + "\n", encoding="utf-8")
    candidate = CoverageCandidate(
        candidate_id="candidate",
        plan_id=plan.plan_id,
        attempt_index=0,
        plan_record=str(plan_record),
        render_plan=str(tmp_path / "candidate.views.json"),
        recipe=str(tmp_path / "candidate.yaml"),
        bundle=str(bundle),
        group=str(tmp_path / "group"),
        log=str(tmp_path / "candidate.log"),
    )
    cell = CoverageCell(
        cell_id="cell",
        scene_key="scene",
        scene_id="scene",
        capability="self_motion_update",
        binding={"target": "target"},
        seed=17,
        target_accepted=1,
        geometry_pool_size=1,
        diverse_pool_size=1,
        candidates=(candidate,),
    )
    manifest_path = _write_coverage_manifest(tmp_path, cells=(cell,), scenes=(scene,))
    (tmp_path / "coverage.status.json").write_text(
        json.dumps(
            {
                "schema_version": "scriptgen_binding_coverage_status.v1",
                "collection_id": "coverage",
                "cells": {
                    "cell": {
                        "status": "planned",
                        "accepted_episode_ids": [],
                        "candidate_statuses": {
                            "candidate": {
                                "status": "rejected",
                                "reason": "authority:KeyError:125013205818896",
                            }
                        },
                    }
                },
                "episodes": {},
            }
        ),
        encoding="utf-8",
    )
    group = ScriptgenQuestionGroupV1(
        question_group_id="group",
        standard_version=STD_V1.standard_version,
        trajectory=QuestionGroupTrajectory(
            bundle=str(bundle),
            plan_record=str(plan_record),
            scene_ir=scene.scene_ir,
            plan_id=plan.plan_id,
            scene_id="scene",
        ),
        questions=(
            QuestionGroupEntry(
                capability="self_motion_update",
                role="primary",
                family_id="self",
                label="front",
                family="self/family.json",
                skip_reason=None,
            ),
        ),
    )
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage.verify_coverage_sources",
        lambda manifest: None,
    )
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage._validate_rendered_candidate",
        lambda *args, **kwargs: (group, plan),
    )

    status_path = run_coverage(
        manifest_path,
        og_root=tmp_path,
        conda_env="behavior",
        data_root=tmp_path,
        gpu_ids=(0,),
        skip_preflight=True,
    )

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["cells"]["cell"]["status"] == "complete"
    assert status["cells"]["cell"]["candidate_statuses"]["candidate"]["status"] == (
        "accepted"
    )


def test_legacy_zero_pool_cell_is_exhausted_without_replaying_150_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = _scene(tmp_path)
    cell = CoverageCell(
        cell_id="cell",
        scene_key="scene",
        scene_id="scene",
        capability="self_motion_update",
        binding={"target": "target"},
        seed=17,
        target_accepted=10,
        geometry_pool_size=0,
        diverse_pool_size=0,
    )
    manifest_path = _write_coverage_manifest(tmp_path, cells=(cell,), scenes=(scene,))
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage.verify_coverage_sources",
        lambda manifest: None,
    )
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage._new_candidates",
        lambda *args, **kwargs: pytest.fail("legacy zero pool must not be replayed"),
    )

    status_path = run_coverage(
        manifest_path,
        og_root=tmp_path,
        conda_env="behavior",
        data_root=tmp_path,
        gpu_ids=(0,),
        skip_preflight=True,
    )

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["cells"]["cell"]["status"] == "exhausted"


def test_existing_unresolvable_cell_is_exhausted_without_backfill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = _scene(tmp_path)
    Path(scene.scene_ir).write_text(
        json.dumps(
            {
                "scene_id": "scene",
                "entities": [
                    {"entity_id": "target", "source_entity_id": "missing-source"},
                ],
            }
        ),
        encoding="utf-8",
    )
    candidates = []
    for index in range(3):
        bundle = tmp_path / f"bundle-{index}"
        bundle.mkdir()
        (bundle / "render_report.json").write_text(
            json.dumps({"status": "success"}),
            encoding="utf-8",
        )
        (bundle / "scene_snapshot.json").write_text(
            json.dumps({"runtime_instance_registry": {"91": "other-source"}}),
            encoding="utf-8",
        )
        candidates.append(
            CoverageCandidate(
                candidate_id=f"candidate-{index}",
                plan_id=f"plan.a{index}",
                attempt_index=index,
                plan_record=str(tmp_path / f"candidate-{index}.record.json"),
                render_plan=str(tmp_path / f"candidate-{index}.views.json"),
                recipe=str(tmp_path / f"candidate-{index}.yaml"),
                bundle=str(bundle),
                group=str(tmp_path / f"group-{index}"),
                log=str(tmp_path / f"candidate-{index}.log"),
            )
        )
    cell = CoverageCell(
        cell_id="cell",
        scene_key="scene",
        scene_id="scene",
        capability="self_motion_update",
        binding={"target": "target"},
        seed=17,
        target_accepted=10,
        geometry_pool_size=1,
        diverse_pool_size=1,
        search_attempt_limit=30,
        raw_plan_limit=20,
        candidates=tuple(candidates),
    )
    manifest_path = _write_coverage_manifest(tmp_path, cells=(cell,), scenes=(scene,))
    (tmp_path / "coverage.status.json").write_text(
        json.dumps(
            {
                "schema_version": "scriptgen_binding_coverage_status.v1",
                "collection_id": "coverage",
                "cells": {
                    "cell": {
                        "status": "planned",
                        "accepted_episode_ids": [],
                        "candidate_statuses": {
                            candidate.candidate_id: {
                                "status": "rejected",
                                "reason": "authority:frame_var_unresolvable:t_seen",
                            }
                            for candidate in candidates
                        },
                    }
                },
                "episodes": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage.verify_coverage_sources",
        lambda manifest: None,
    )
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage._new_candidates",
        lambda *args, **kwargs: pytest.fail("unresolvable binding must not backfill"),
    )

    status_path = run_coverage(
        manifest_path,
        og_root=tmp_path,
        conda_env="behavior",
        data_root=tmp_path,
        gpu_ids=(0,),
        skip_preflight=True,
    )

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["cells"]["cell"]["status"] == "exhausted"
    updated = load_coverage(manifest_path).cells[0]
    assert updated.search_pool_exhausted
    assert updated.rejection_counts["binding:render_unresolvable"] == 1


def test_one_positive_runtime_id_replay_prevents_binding_quarantine(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    Path(scene.scene_ir).write_text(
        json.dumps(
            {
                "scene_id": "scene",
                "entities": [
                    {"entity_id": "target", "source_entity_id": "target-source"},
                ],
            }
        ),
        encoding="utf-8",
    )
    candidates = []
    candidate_statuses = {}
    for index in range(4):
        bundle = tmp_path / f"bundle-{index}"
        bundle.mkdir()
        (bundle / "render_report.json").write_text(
            json.dumps({"status": "success"}),
            encoding="utf-8",
        )
        registry = {"91": "target-source"} if index == 3 else {"92": "other-source"}
        (bundle / "scene_snapshot.json").write_text(
            json.dumps({"runtime_instance_registry": registry}),
            encoding="utf-8",
        )
        candidate = CoverageCandidate(
            candidate_id=f"candidate-{index}",
            plan_id=f"plan.a{index}",
            attempt_index=index,
            plan_record=str(tmp_path / f"candidate-{index}.record.json"),
            render_plan=str(tmp_path / f"candidate-{index}.views.json"),
            recipe=str(tmp_path / f"candidate-{index}.yaml"),
            bundle=str(bundle),
            group=str(tmp_path / f"group-{index}"),
            log=str(tmp_path / f"candidate-{index}.log"),
        )
        candidates.append(candidate)
        candidate_statuses[candidate.candidate_id] = {
            "status": "accepted" if index == 3 else "rejected",
            "reason": None,
        }
    cell = CoverageCell(
        cell_id="cell",
        scene_key="scene",
        scene_id="scene",
        capability="self_motion_update",
        binding={"target": "target"},
        seed=17,
        target_accepted=10,
        geometry_pool_size=4,
        diverse_pool_size=4,
        search_attempt_limit=30,
        raw_plan_limit=20,
        candidates=tuple(candidates),
    )
    row = {
        "status": "planned",
        "accepted_episode_ids": ["candidate-3"],
        "candidate_statuses": candidate_statuses,
    }

    assert _confirmed_render_unresolvable_targets(cell, scene, row) == ()


def test_pipeline_renders_completed_scenes_without_manifest_backfill_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _scene(tmp_path).model_copy(update={"scene_key": "scene-a", "scene_id": "a"})
    second = first.model_copy(update={"scene_key": "scene-b", "scene_id": "b"})
    cell = CoverageCell(
        cell_id="cell-a",
        scene_key="scene-a",
        scene_id="a",
        capability="self_motion_update",
        binding={"target": "target"},
        seed=17,
        target_accepted=10,
        geometry_pool_size=0,
        diverse_pool_size=0,
    )
    manifest_path = _write_coverage_manifest(
        tmp_path,
        cells=(cell,),
        scenes=(first, second),
    )
    manifest = load_coverage(manifest_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage.verify_coverage_sources",
        lambda value: None,
    )
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage._preflight",
        lambda *args, **kwargs: None,
    )

    def fake_plan_coverage(*args, **kwargs):
        callback = kwargs["on_scene_completed"]
        callback("scene-a")
        callback("scene-b")
        return manifest_path

    def fake_run_coverage(*args, **kwargs):
        calls.append(kwargs)
        return tmp_path / "coverage.status.json"

    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage.plan_coverage", fake_plan_coverage
    )
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage.run_coverage", fake_run_coverage
    )

    result = run_coverage_pipeline(
        manifest_path,
        og_root=tmp_path,
        conda_env="behavior",
        data_root=tmp_path,
        gpu_ids=(0,),
    )

    assert result == tmp_path / "coverage.status.json"
    assert any(call["allow_backfill"] is False for call in calls[:-1])
    assert calls[-1]["allow_backfill"] is True
    rendered_scenes = {
        scene
        for call in calls[:-1]
        for scene in (call.get("scene_keys") or ())
    }
    assert rendered_scenes == {"scene-a", "scene-b"}
    assert _completed_scene_prefix(manifest) == ()


def test_answerable_companions_credit_their_projected_binding_cells(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    cells = tuple(
        CoverageCell(
            cell_id=capability,
            scene_key="scene",
            scene_id="scene",
            capability=capability,
            binding={"target": "target"},
            seed=17,
            target_accepted=10,
            geometry_pool_size=1,
            diverse_pool_size=1,
        )
        for capability in ("self_motion_update", "path_integration")
    )
    manifest = CoverageManifest(
        collection_id="coverage",
        standard_version=STD_V1.standard_version,
        source_index=str(tmp_path / "source.index.json"),
        source_index_sha256="0" * 64,
        output_root=str(tmp_path),
        accepted_per_binding=10,
        attempts_per_binding=150,
        maximum_multislot_bindings=128,
        capabilities=("self_motion_update", "path_integration"),
        scenes=(scene,),
        cells=cells,
    )
    group = ScriptgenQuestionGroupV1(
        question_group_id="group",
        standard_version=STD_V1.standard_version,
        trajectory=QuestionGroupTrajectory(
            bundle="bundle",
            plan_record="plan",
            scene_ir=scene.scene_ir,
            plan_id="plan.a0",
            scene_id="scene",
        ),
        questions=(
            QuestionGroupEntry(
                capability="self_motion_update",
                role="primary",
                family_id="self",
                label="front",
                family="self/family.json",
                skip_reason=None,
            ),
            QuestionGroupEntry(
                capability="path_integration",
                role="primary",
                family_id="turn",
                label="left",
                family="turn/family.json",
                skip_reason=None,
            ),
        ),
    )

    credited = compatible_credit_cell_ids(
        manifest,
        scene_key="scene",
        plan=_plan("plan.a0"),
        group=group,
    )

    assert credited == ("self_motion_update", "path_integration")


def test_full_producer_continues_for_explicit_derived_credit_deficit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = _scene(tmp_path)
    plan = _plan("plan.a0")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "render_report.json").write_text(
        json.dumps({"status": "success"}),
        encoding="utf-8",
    )
    plan_record = tmp_path / "candidate.record.json"
    plan_record.write_text(plan.model_dump_json(indent=2) + "\n", encoding="utf-8")
    candidate = CoverageCandidate(
        candidate_id="candidate",
        plan_id=plan.plan_id,
        attempt_index=0,
        plan_record=str(plan_record),
        render_plan=str(tmp_path / "candidate.views.json"),
        recipe=str(tmp_path / "candidate.yaml"),
        bundle=str(bundle),
        group=str(tmp_path / "group"),
        log=str(tmp_path / "candidate.log"),
    )
    producer = CoverageCell(
        cell_id="producer",
        scene_key="scene",
        scene_id="scene",
        capability="self_motion_update",
        binding={"target": "target"},
        seed=17,
        target_accepted=1,
        geometry_pool_size=1,
        diverse_pool_size=1,
        candidates=(candidate,),
    )
    derived = CoverageCell(
        cell_id="derived",
        scene_key="scene",
        scene_id="scene",
        capability="path_integration",
        binding={"target": "target"},
        seed=29,
        target_accepted=1,
        geometry_pool_size=0,
        diverse_pool_size=0,
    )
    manifest_path = _write_coverage_manifest(
        tmp_path,
        cells=(producer, derived),
        scenes=(scene,),
    )
    (tmp_path / "coverage.status.json").write_text(
        json.dumps(
            {
                "schema_version": "scriptgen_binding_coverage_status.v1",
                "collection_id": "coverage",
                "cells": {
                    "producer": {
                        "status": "complete",
                        "accepted_episode_ids": ["existing"],
                        "candidate_statuses": {
                            "candidate": {"status": "pending", "reason": None}
                        },
                    },
                    "derived": {
                        "status": "planned",
                        "accepted_episode_ids": [],
                        "candidate_statuses": {},
                    },
                },
                "episodes": {},
            }
        ),
        encoding="utf-8",
    )
    group = ScriptgenQuestionGroupV1(
        question_group_id="group",
        standard_version=STD_V1.standard_version,
        trajectory=QuestionGroupTrajectory(
            bundle=str(bundle),
            plan_record=str(plan_record),
            scene_ir=scene.scene_ir,
            plan_id=plan.plan_id,
            scene_id="scene",
        ),
        questions=(
            QuestionGroupEntry(
                capability="self_motion_update",
                role="primary",
                family_id="self",
                label="back",
                family="self/family.json",
                skip_reason=None,
            ),
            QuestionGroupEntry(
                capability="path_integration",
                role="primary",
                family_id="path",
                label="right",
                family="path/family.json",
                skip_reason=None,
            ),
        ),
    )
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage.verify_coverage_sources",
        lambda manifest: None,
    )
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage._validate_rendered_candidate",
        lambda *args, **kwargs: (group, plan),
    )

    status_path = run_coverage(
        manifest_path,
        og_root=tmp_path,
        conda_env="behavior",
        data_root=tmp_path,
        gpu_ids=(0,),
        cell_ids=("producer",),
        credit_cell_ids=("derived",),
        allow_backfill=False,
        skip_preflight=True,
    )

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["cells"]["producer"]["status"] == "complete"
    assert status["cells"]["derived"]["accepted_episode_ids"] == ["candidate"]
    assert status["cells"]["producer"]["candidate_statuses"]["candidate"][
        "credited_cells"
    ] == ["derived"]


def test_coverage_plan_resumes_without_duplicating_cells(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    scene_ir = source_root / "scene_ir.json"
    snapshot = source_root / "scene_snapshot.json"
    recipe = source_root / "scene.yaml"
    scene_ir.write_text('{"scene_id":"scene"}\n', encoding="utf-8")
    snapshot.write_text(
        '{"source_scene_id":"scene:best","scene_model":"scene",'
        '"scene_instance":null,"source_digest":"digest"}\n',
        encoding="utf-8",
    )
    recipe.write_text("source: {}\n", encoding="utf-8")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    index = SourceIndex(
        source_id="source",
        source_root=str(source_root),
        og_root="og",
        data_root="data",
        source_version="behavior-1k-v3.9.0",
        requested_scene_count=1,
        ready_scene_count=1,
        failed_scene_count=0,
        scenes=(
            SourceSceneRecord(
                scene_key="scene",
                scene_model="scene",
                strategy="room_aware",
                status="ready",
                bundle=str(source_root),
                recipe=str(recipe),
                scene_ir=str(scene_ir),
                scene_snapshot=str(snapshot),
                source_digest="digest",
                scene_ir_sha256=sha(scene_ir),
                recipe_sha256=sha(recipe),
                attempts=1,
            ),
        ),
    )
    index_path = source_root / "source.index.json"
    index_path.write_text(index.model_dump_json(indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage._bindings_for_scene",
        lambda *args, **kwargs: (
            {"target": "target"},
            {"target": "not-allowed"},
        ),
    )
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage._new_candidates",
        lambda *args, **kwargs: CandidateSearchResult(
            candidates=(),
            geometry_pool_size=0,
            diverse_pool_size=0,
            rejection_counts={},
            search_attempt_limit=2,
            raw_plan_limit=2,
            search_pool_exhausted=True,
        ),
    )
    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage._scene_occupancy_proxy_rejection",
        lambda *args, **kwargs: None,
    )
    output = tmp_path / "coverage"

    first = plan_coverage(
        source_index_path=index_path,
        output_root=output,
        collection_id="coverage",
        accepted_per_binding=1,
        attempts_per_binding=2,
        maximum_multislot_bindings=1,
        capabilities=("self_motion_update",),
        binding_allowlist={"scene": [{"target": "target"}]},
    )
    second = plan_coverage(
        source_index_path=index_path,
        output_root=output,
        collection_id="coverage",
        accepted_per_binding=1,
        attempts_per_binding=2,
        maximum_multislot_bindings=1,
        capabilities=("self_motion_update",),
        binding_allowlist={"scene": [{"target": "target"}]},
    )

    assert first == second
    manifest = load_coverage(first)
    assert len(manifest.cells) == 1
    assert manifest.cells[0].binding == {"target": "target"}

    recipe.write_text("source: {changed: true}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source recipe changed"):
        verify_coverage_sources(manifest)


def test_scene_with_empty_occupancy_proxy_is_explicitly_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = _scene(tmp_path)

    class FakeLayout:
        obstacles = ()

    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage.layout_from_scene_ir",
        lambda *args, **kwargs: FakeLayout(),
    )

    class EmptyGrid:
        width = 623
        height = 867
        free_cells = ()

    monkeypatch.setattr(
        "spatial_episode.scriptgen.binding_coverage.proposal_occupancy_grid",
        lambda layout: EmptyGrid(),
    )
    reason = _scene_occupancy_proxy_rejection(scene, std=STD_V1)

    assert reason == (
        "invalid_occupancy_proxy:no_free_cells:grid=623x867:obstacles=0"
    )
