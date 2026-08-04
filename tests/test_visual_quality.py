from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from episode3d.bundles import Bundle, stable_id
from episode3d.compilers import (
    _bindings_unambiguous,
    _model_input_frame_quality,
    _recognizable_category_candidates,
    _referent_visual_quality,
    _relation_is_unambiguous,
    _source_relation_components,
    _surface_anchor_views,
    compile_bundle_questions,
    compile_t10_predictions,
    recognizable_witnesses,
)
from episode3d.exporters import export_state_aux
from episode3d.language import localize_categories
from episode3d.models import CompiledBundle, QuestionSpec
from episode3d.programs import program_for
from episode3d.verifiers import replay_question

ROOT = Path(__file__).resolve().parents[1]
SWEEPS = ROOT.parent / "OminiGibson" / "outputs" / "sweeps"
MEDIA = ROOT / "data" / "epispace_rgb_v1"


class _StatsBundle:
    def __init__(
        self,
        stats: dict,
        *,
        rgb_stats: dict | None = None,
        label: str = "chair",
        extent_m: tuple[float, float, float] = (0.5, 0.5, 1.0),
    ) -> None:
        self.stats = stats
        self.rgb_stats = rgb_stats or {
            "dominant_quantized_color_fraction": 0.1,
            "quantized_color_entropy_bits": 5.0,
        }
        self.entity_value = SimpleNamespace(label=label, extent_m=extent_m)

    def instance_visual_stats(self, entity_id: str, view_id: str) -> dict:
        assert entity_id == "entity"
        assert view_id == "view"
        return self.stats

    def rgb_visual_stats(self, view_id: str) -> dict:
        assert view_id == "view"
        return self.rgb_stats

    def frame_collision_stats(self, view_id: str) -> dict:
        assert view_id == "view"
        return {
            "near_nonstructural_pixel_count": 0,
            "valid_top_half_depth_pixel_count": 100,
            "near_nonstructural_pixel_fraction": 0.0,
            "near_geometry_pixel_count": 0,
            "near_geometry_pixel_fraction": 0.0,
            "top_half_pixel_count": 100,
            "depth_threshold_m": 0.4,
        }

    def frame_context_stats(self, view_id: str) -> dict:
        assert view_id == "view"
        return {
            "valid_depth_pixel_count": 100,
            "frame_pixel_count": 100,
            "close_geometry_pixel_count": 0,
            "close_geometry_pixel_fraction": 0.0,
            "close_depth_threshold_m": 0.75,
            "depth_p10_m": 1.0,
            "depth_median_m": 2.0,
            "depth_p90_m": 3.0,
            "depth_p90_minus_p10_m": 2.0,
            "enclosure_pixel_count": 0,
            "enclosure_pixel_fraction": 0.0,
            "dominant_nonstructural_instance": {
                "entity_id": None,
                "raw_label": None,
                "pixel_count": 0,
                "pixel_fraction": 0.0,
                "border_sides": [],
                "median_depth_m": None,
            },
        }

    def entity(self, entity_id: str):
        assert entity_id == "entity"
        return self.entity_value


def _stats(**updates: object) -> dict:
    result = {
        "visible_pixels": 2_000,
        "image_width_px": 1_024,
        "image_height_px": 1_024,
        "bbox_xyxy_px": [100, 100, 149, 149],
        "bbox_width_px": 50,
        "bbox_height_px": 50,
        "mask_bbox_fill_ratio": 0.8,
        "outer_five_percent_pixel_fraction": 0.0,
        "touches_image_border": False,
        "border_sides": [],
    }
    result.update(updates)
    return result


def _oracle_stats_bundle(tmp_path: Path) -> Bundle:
    root = tmp_path / "oracle_stats_bundle"
    views = root / "views"
    views.mkdir(parents=True)

    def entity(entity_id: str, raw_label: str) -> dict:
        return {
            "entity_id": entity_id,
            "source_entity_id": f"source-{entity_id}",
            "raw_label": raw_label,
            "region_id": "room-0",
            "world_from_entity": {"translation_m": [0.0, 0.0, 0.0]},
            "obb": {"half_extents_m": [0.25, 0.25, 0.25]},
        }

    payloads = {
        "scene_ir.json": {
            "entities": [
                entity("chair", "chair"),
                entity("wall", "wall"),
                entity("alarm-a", "fire_alarm"),
                entity("alarm-b", "fire_alarm"),
            ],
            "runtime_semantic_id_map": {
                "10": "chair",
                "20": "wall",
                "30": "alarm-a",
                "31": "alarm-b",
            },
        },
        "spatial_episode.json": {
            "scene_id": "synthetic-scene",
            "episode_id": "synthetic-episode",
            "split_group": "synthetic-group",
            "observations": [
                {
                    "view_id": "view-000",
                    "step": 0,
                    # The alarm instances are intentionally absent.  Raw
                    # oracle statistics must still recover their pixels.
                    "visible_entity_ids": ["chair"],
                    "world_from_camera": {
                        "translation_m": [0.0, 0.0, 1.5],
                        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                }
            ],
        },
        "trajectory_plan.json": {
            "trajectory_class": "T1",
            "views": [{"view_id": "view-000", "role": "anchor"}],
        },
        "quality_report.json": {
            "integrity_status": "pass",
            "trajectory_status": "pass",
        },
    }
    for filename, payload in payloads.items():
        (root / filename).write_text(json.dumps(payload), encoding="utf-8")

    height, width = 8, 10
    depth = np.ones((height, width), dtype=np.float32)
    instance = np.zeros((height, width), dtype=np.uint32)
    instance[0:2, 0:3] = 10  # six near chair pixels in the top half
    depth[0:2, 0:3] = 0.2
    instance[0:2, 3:6] = 20  # near structural wall: must be excluded
    depth[0:2, 3:6] = 0.2
    instance[2:4, 0:2] = 30  # four undeclared near alarm pixels
    depth[2:4, 0:2] = 0.2
    instance[4:7, 5:9] = 31  # 3x4 undeclared alarm instance
    depth[0, 9] = 0.0
    depth[1, 9] = np.nan
    np.savez(
        views / "view-000.sensors.npz",
        rgb=np.zeros((height, width, 3), dtype=np.uint8),
        depth_m=depth,
        instance_id=instance,
    )
    return Bundle(
        root,
        trajectory_class="T1",
        source_sweep="synthetic-oracle-stats",
        job_status="passed",
    )


def test_oracle_stats_use_raw_channels_ignore_visibility_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _oracle_stats_bundle(tmp_path)

    collision = bundle.frame_collision_stats("view-000")
    assert collision == {
        "near_nonstructural_pixel_count": 10,
        "valid_top_half_depth_pixel_count": 38,
        "near_nonstructural_pixel_fraction": pytest.approx(10 / 38),
        "near_geometry_pixel_count": 16,
        "near_geometry_pixel_fraction": pytest.approx(16 / 38),
        "top_half_pixel_count": 40,
        "depth_threshold_m": 0.4,
    }

    context = bundle.frame_context_stats("view-000")
    assert context["valid_depth_pixel_count"] == 78
    assert context["close_geometry_pixel_count"] == 16
    assert context["close_geometry_pixel_fraction"] == pytest.approx(16 / 78)
    assert context["enclosure_pixel_fraction"] == pytest.approx(6 / 80)
    assert context["dominant_nonstructural_instance"] == {
        "entity_id": "alarm-b",
        "raw_label": "fire_alarm",
        "pixel_count": 12,
        "pixel_fraction": pytest.approx(12 / 80),
        "border_sides": [],
        "median_depth_m": pytest.approx(1.0),
    }

    category = bundle.raw_category_visual_stats("view-000", ["fire alarm"])
    assert category == {
        "labels": ("fire_alarm",),
        "matching_pixel_count": 16,
        "matching_instance_count": 2,
        "max_instance_bbox_min_side_px": 3,
        "image_width_px": 10,
        "image_height_px": 8,
    }
    assert "alarm-a" not in bundle.view_by_id["view-000"].visible_entity_ids
    assert "alarm-b" not in bundle.view_by_id["view-000"].visible_entity_ids

    # Returned dictionaries cannot mutate the cached source of truth, and a
    # repeated normalized request must not reopen the sensor archive.
    collision["near_nonstructural_pixel_count"] = 999
    category["matching_pixel_count"] = 999

    def fail_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("cached oracle statistics unexpectedly reloaded the sensor")

    monkeypatch.setattr("episode3d.bundles.np.load", fail_load)
    assert bundle.frame_collision_stats("view-000")["near_nonstructural_pixel_count"] == 10
    assert (
        bundle.raw_category_visual_stats("view-000", ["fire_alarm"])["matching_pixel_count"] == 16
    )


@pytest.mark.parametrize(
    ("updates", "passed", "flag"),
    [
        ({"visible_pixels": 499}, False, "below_minimum_pixels"),
        (
            {"bbox_width_px": 19},
            False,
            "undersized_bbox",
        ),
        (
            {"mask_bbox_fill_ratio": 0.099},
            False,
            "fragmented_mask",
        ),
        (
            {
                "visible_pixels": 9_000,
                "bbox_width_px": 78,
                "bbox_height_px": 171,
                "outer_five_percent_pixel_fraction": 0.8,
                "touches_image_border": True,
                "border_sides": ["bottom"],
            },
            False,
            "severe_border_crop",
        ),
        (
            {
                "visible_pixels": 40_000,
                "bbox_width_px": 220,
                "bbox_height_px": 220,
                "outer_five_percent_pixel_fraction": 0.81,
                "touches_image_border": True,
                "border_sides": ["left", "bottom"],
            },
            False,
            "severe_border_crop",
        ),
        (
            {
                "visible_pixels": 22_000,
                "bbox_width_px": 180,
                "bbox_height_px": 220,
                "outer_five_percent_pixel_fraction": 0.399,
                "touches_image_border": True,
                "border_sides": ["right", "bottom"],
            },
            False,
            "severe_border_crop",
        ),
        ({}, True, None),
    ],
)
def test_referent_quality_boundary_cases(updates: dict, passed: bool, flag: str | None) -> None:
    quality = _referent_visual_quality(_StatsBundle(_stats(**updates)), "entity", "view")
    assert quality["passed"] is passed
    if flag:
        assert quality["failure_flags"][flag] is True


def test_referent_quality_rejects_degenerate_rgb_and_bad_category_geometry() -> None:
    degenerate = _referent_visual_quality(
        _StatsBundle(
            _stats(),
            rgb_stats={
                "dominant_quantized_color_fraction": 0.7,
                "quantized_color_entropy_bits": 2.0,
            },
        ),
        "entity",
        "view",
    )
    assert not degenerate["passed"]
    assert degenerate["failure_flags"]["degenerate_rgb"]

    mislabeled_table = _referent_visual_quality(
        _StatsBundle(
            _stats(),
            label="coffee_table",
            extent_m=(1.0, 2.0, 0.75),
        ),
        "entity",
        "view",
    )
    assert not mislabeled_table["passed"]
    assert mislabeled_table["failure_flags"]["implausible_category_geometry"]


def test_every_actual_model_input_frame_is_hard_gated_and_replayed() -> None:
    bundle = _bundle(
        "static-m2-room-aware-indoor-seed17-v1/bundles/hotel_suite_small_seed17",
        "T1",
    )
    bad_views = {
        view.view_id
        for view in bundle.views
        if not _model_input_frame_quality(bundle, view.view_id)["passed"]
    }
    assert bad_views == {"view-005"}

    questions, rejections = compile_bundle_questions(bundle)
    # Clean, genuinely view-local atoms remain usable even when another frame
    # in the source trajectory is corrupt.  The compiler must neither attach
    # that frame to the model input nor leave a surface reference to it.
    assert {question.fact_id for question in questions} == {
        "fact-51a0df288ca4f3969ad8",
        "fact-30c76d7916bf8f707461",
    }
    assert all(not (set(question.model_view_ids) & bad_views) for question in questions)
    assert all(question.question_zh.startswith("在这张图里") for question in questions)
    assert all(replay_question(bundle, question)[0] for question in questions)
    rejected_ids = {
        item.item_id
        for item in rejections
        if item.reason_code == "inadmissible_model_input_frame"
    }
    assert rejected_ids

    contaminated = QuestionSpec(
        fact_id="fact-regression-degenerate-context",
        task_type="unknown_abstention",
        question_zh="根据当前观察能否确定场景里存在某物？",
        answer_zh="无法确定。",
        answer_value=None,
        answer_status="unknown",
        program=program_for("unknown_abstention"),
        evidence_view_ids=tuple(view.view_id for view in bundle.views),
        evidence_entity_ids=(),
        certificate={"result": "unknown"},
        rationale_zh="只依据当前观察。",
        source="quality-regression-test",
        model_view_ids=tuple(view.view_id for view in bundle.views),
    )
    passed, detail = replay_question(bundle, contaminated)
    assert not passed
    assert "inadmissible actual model input" in detail

    assert (
        export_state_aux(
            CompiledBundle(bundle=bundle, split="train"),
            observable_belief={"schema_version": "test", "entities": []},
        )
        is None
    )


def test_relation_dominance_boundary() -> None:
    assert _relation_is_unambiguous(1.25, 1.0)
    assert not _relation_is_unambiguous(1.249, 1.0)
    assert _relation_is_unambiguous(1.0, 0.0)
    assert _source_relation_components(
        {"oracle": {"target_ego_right_m": 1.0, "target_ego_front_m": 0.5}}
    ) == (1.0, 0.5)


def _bundle(relative: str, trajectory_class: str) -> Bundle:
    bundle = Bundle(
        SWEEPS / relative,
        trajectory_class=trajectory_class,
        source_sweep="quality-regression-test",
        job_status="passed",
    )
    bundle.materialize_model_rgb(MEDIA)
    return bundle


def _stale_source_spec(bundle: Bundle, fact_id: str) -> QuestionSpec:
    """Recreate a pre-gate source spec to prove replay rejects stale releases."""

    task = next(
        task
        for task in bundle.tasks
        if stable_id(
            "fact",
            bundle.scene_id,
            bundle.episode_id,
            str(task["task_id"]),
        )
        == fact_id
    )
    return QuestionSpec(
        fact_id=fact_id,
        task_type=str(task["task_type"]),
        question_zh=localize_categories(str(task["question_zh"])),
        answer_zh=localize_categories(str(task["surface_answer_zh"])),
        answer_value=task["answer"],
        answer_status=str(task["status"]),
        program=program_for(str(task["task_type"])),
        evidence_view_ids=tuple(map(str, task["evidence_view_ids"])),
        evidence_entity_ids=tuple(map(str, task["evidence_entity_ids"])),
        certificate=dict(task["certificate"]),
        rationale_zh="stale pre-gate regression fixture",
        source="quality-regression-test",
        model_view_ids=tuple(view.view_id for view in bundle.views),
    )


def test_surface_bound_anchor_cannot_borrow_a_clean_later_view() -> None:
    bundle = _bundle(
        "t7-elevation-seed17-v1/bundles/restaurant_brunch_t7_seed17",
        "T7",
    )
    questions, rejections = compile_bundle_questions(bundle)
    rejected = {item.item_id: item.reason_code for item in rejections}
    assert rejected["ogtask-metric-a5e08d62d8c587"] == (
        "unresolvable_or_unrecognizable_entity_reference"
    )
    assert rejected["ogtask-frame-0750524e41925a"] == (
        "unresolvable_or_unrecognizable_entity_reference"
    )
    assert not any(
        spec.task_type in {"metric_distance", "egocentric_relation"} and "门" in spec.question_zh
        for spec in questions
    )


def test_repeated_countertop_track_is_rejected_without_killing_unique_anchor() -> None:
    bundle = _bundle(
        "t4-object-orbit-adaptive-arc-seed17-v3/bundles/Pomaria_1_int_t4_seed17",
        "T4",
    )
    fact_id = "fact-c7d0336fc008cdf9fa87"
    stale = _stale_source_spec(bundle, fact_id)
    target_id = stale.evidence_entity_ids[0]
    all_views = tuple(view.view_id for view in bundle.views)

    # The explicit fifth-view phrase is a valid local anchor: the second raw
    # countertop fragment in that frame is too small to be a usable referent.
    assert _bindings_unambiguous(
        bundle,
        (target_id,),
        all_views,
        surface_question=stale.question_zh,
    )
    # It is not a valid identity track: later frames contain other clean
    # countertops without a language-visible qualifier.
    assert not _bindings_unambiguous(
        bundle,
        (target_id,),
        all_views,
        surface_question=stale.question_zh,
        require_unique_track=True,
    )

    questions, rejections = compile_bundle_questions(bundle)
    assert fact_id not in {spec.fact_id for spec in questions}
    assert any(
        item.item_id == "ogtask-memory-36fb3067a90265"
        and item.reason_code == "unresolvable_or_unrecognizable_entity_reference"
        for item in rejections
    )
    passed, detail = replay_question(bundle, stale)
    assert not passed
    assert "surface binding has no unique recognizable witness" in detail


def test_orbit_selector_replaces_ambiguous_two_armchair_pair() -> None:
    bundle = _bundle(
        "t4-object-orbit-adaptive-arc-seed17-v3/bundles/Pomaria_2_int_t4_seed17",
        "T4",
    )
    fact_id = "fact-b2311d6fd771d031724d"
    questions, _ = compile_bundle_questions(bundle)
    clean = next(spec for spec in questions if spec.fact_id == fact_id)

    assert clean.evidence_view_ids != ("view-001", "view-005")
    assert all(
        _recognizable_category_candidates(bundle, "armchair", view_id)
        == clean.evidence_entity_ids
        for view_id in clean.evidence_view_ids
    )
    assert replay_question(bundle, clean)[0]

    # Replay must distrust the old certificate and reject the previously
    # selected pair, whose first frame visibly contains two armchairs.
    stale = replace(
        clean,
        evidence_view_ids=("view-001", "view-005"),
        model_view_ids=("view-001", "view-005"),
    )
    passed, detail = replay_question(bundle, stale)
    assert not passed
    assert "surface binding has no unique recognizable witness" in detail


def test_confusable_ceiling_lights_invalidate_downlight_last_seen() -> None:
    bundle = _bundle(
        "t3-rotation-station-r0-seed17-v1/bundles/office_large_t3_seed17",
        "T3",
    )
    fact_id = "fact-73a2844008af87887c0a"
    stale = _stale_source_spec(bundle, fact_id)
    candidates = _recognizable_category_candidates(bundle, "downlight", "view-003")
    assert stale.evidence_entity_ids[0] in candidates
    assert len(candidates) > 1

    questions, rejections = compile_bundle_questions(bundle)
    assert fact_id not in {spec.fact_id for spec in questions}
    assert any(
        item.item_id == "ogtask-memory-147285906780f8"
        and item.reason_code == "unresolvable_or_unrecognizable_entity_reference"
        for item in rejections
    )
    passed, detail = replay_question(bundle, stale)
    assert not passed
    assert "surface binding has no unique recognizable witness" in detail


def test_t8_uses_clean_post_occlusion_witness_and_changes_fact_ids() -> None:
    bundle = _bundle(
        "t8-occlusion-reveal-seed17-v1/bundles/grocery_store_cafe_t8_seed17",
        "T8",
    )
    questions, _ = compile_bundle_questions(bundle)
    family = [spec for spec in questions if spec.source.endswith("t8.v3")]
    assert {spec.family_variant for spec in family} == {
        "prefix_unknown",
        "revealed",
        "decisive_deleted",
    }
    revealed = next(spec for spec in family if spec.family_variant == "revealed")
    assert revealed.model_view_ids == ("view-000", "view-004")
    assert revealed.fact_id != "fact-b28a53b15cd7d588dc61"
    decisive = revealed.certificate["checks"][1]
    assert decisive["quality_selected_fallback"] is True
    assert decisive["annotated_decisive_view"] == "view-005"


def test_bundle_wide_frame_search_recovers_a_clean_alternative_pair() -> None:
    bundle = _bundle(
        "t3-rotation-station-r1-seed23-v1/bundles/gates_bedroom_t3_seed23",
        "T3",
    )
    questions, _ = compile_bundle_questions(bundle)
    family = [spec for spec in questions if spec.family_variant in {"frame_a", "frame_b"}]
    assert {spec.family_variant for spec in family} == {"frame_a", "frame_b"}
    assert len({spec.model_view_ids for spec in family}) == 1
    assert len({spec.answer_value for spec in family}) == 2
    assert all(spec.source == "epispace.frame_family_compiler.v2" for spec in family)


def test_t10_pose_instruction_restores_test_signature_without_weak_anchors() -> None:
    source = _bundle(
        "t3-rotation-station-r0-seed17-v1/bundles/Beechwood_0_int_t3_seed17",
        "T3",
    )
    target = _bundle(
        "t10-target-view-seed17-v1/bundles/Beechwood_0_int_t10_seed17",
        "T10",
    )
    questions = compile_t10_predictions(target, source)
    assert questions
    assert {spec.answer_value for spec in questions} == {False, True}
    assert all("relative_pose_instruction" in spec.tags for spec in questions)
    assert all(spec.oracle_held_out_view_ids for spec in questions)


def test_context_and_recognizable_change_gates_reject_fresh_audit_majors() -> None:
    office = _bundle(
        "static-m2-room-aware-indoor-seed17-v1/bundles/office_cubicles_right_seed17",
        "T1",
    )
    wall = _model_input_frame_quality(office, "view-003")
    enclosure = _model_input_frame_quality(office, "view-006")
    assert wall["failure_flags"]["near_surface_saturation"]
    assert enclosure["failure_flags"]["near_enclosure"]
    office_questions, _ = compile_bundle_questions(office)
    assert "fact-200add140c7d20da127e" not in {
        question.fact_id for question in office_questions
    }

    gym = _bundle(
        "t3-rotation-station-r0-seed17-v1/bundles/school_gym_t3_seed17",
        "T3",
    )
    obstruction = _model_input_frame_quality(gym, "view-003")
    assert obstruction["failure_flags"]["foreground_occlusion"]
    assert (
        obstruction["frame_context_stats"]["dominant_nonstructural_instance"][
            "raw_label"
        ]
        == "volleyball_net"
    )
    gym_questions, _ = compile_bundle_questions(gym)
    assert "fact-3fec112820f532cc85c4" not in {
        question.fact_id for question in gym_questions
    }

    rotation = _bundle(
        "t3-rotation-station-r0-seed17-v1/bundles/Rs_int_t3_seed17",
        "T3",
    )
    rotation_questions, _ = compile_bundle_questions(rotation)
    assert "fact-71783c3b59c47c7183c9" not in {
        question.fact_id for question in rotation_questions
    }

    # The enclosure rule is deliberately stricter than "mostly wall/door".
    # These two coherent hallway/door views sit just outside its joint depth
    # boundary and must remain usable.
    merom = _bundle(
        "static-m2-room-aware-indoor-seed17-v1/bundles/Merom_1_int_seed17",
        "T1",
    )
    ihlen = _bundle(
        "t3-rotation-station-r0-seed17-v1/bundles/Ihlen_0_int_t3_seed17",
        "T3",
    )
    assert _model_input_frame_quality(merom, "view-004")["passed"]
    assert _model_input_frame_quality(ihlen, "view-003")["passed"]


def test_t10_chinese_prefix_labels_replay_to_exact_instances() -> None:
    source = _bundle(
        "t3-rotation-station-r1-seed23-v1/bundles/hotel_suite_large_t3_seed23",
        "T3",
    )
    bed_id = next(
        entity.entity_id
        for entity in source.entities.values()
        if entity.label == "bed" and "view-005" in source.visible_views(entity.entity_id)
    )
    question = (
        "假设站在第4个视角里看到的床头柜的位置，并面向第6个视角里看到的床，"
        "此时能看到第1个视角里看到的固定窗吗？"
    )
    anchors = _surface_anchor_views(
        source,
        bed_id,
        source.visible_views(bed_id),
        question,
    )
    assert anchors == ("view-005",)


def test_semantic_rgb_calibration_majors_are_rejected_by_general_policy() -> None:
    """Every major from the 48-item calibration audit is now fail-closed."""

    audited_major_fact_ids = {
        "fact-25a8c9ef9058a69569eb",
        "fact-065db0d45e1ca6426338",
        "fact-0f66519bbdf903e48859",
        "fact-05471e9dfcdbebb4d810",
        "fact-eb0f09549ec26d4d0c6c",
        "fact-9b7c9439ec4cd9f653f2",
        "fact-7ce632c206687b423917",
        "fact-e57a5a87d740cfd63b32",
        "fact-9b9ab2df2e6b56378a01",
        "fact-aca5fe338674619680c9",
        "fact-274bd60a2842fd306411",
        "fact-186fb5d7753b1a844894",
        "fact-7ec58fc5c985312744fd",
        "fact-6d87d685ce6e89d26f1e",
        "fact-290098c0c744f8db001f",
        "fact-63c6df00901248027234",
    }
    sources = (
        ("static-m2-room-aware-indoor-seed17-v1", "office_cubicles_right_seed17", "T1"),
        ("static-m2-room-aware-indoor-seed17-v1", "restaurant_asian_seed17", "T1"),
        ("t3-rotation-station-r0-seed17-v1", "Beechwood_0_int_t3_seed17", "T3"),
        ("t3-rotation-station-r0-seed17-v1", "Ihlen_0_int_t3_seed17", "T3"),
        ("t3-rotation-station-r0-seed17-v1", "gates_bedroom_t3_seed17", "T3"),
        ("t3-rotation-station-r0-seed17-v1", "restaurant_brunch_t3_seed17", "T3"),
        ("t3-rotation-station-r1-seed23-v1", "gates_bedroom_t3_seed23", "T3"),
        (
            "t4-object-orbit-adaptive-arc-seed17-v3",
            "Wainscott_0_int_t4_seed17",
            "T4",
        ),
        ("t7-elevation-seed17-v1", "Ihlen_0_int_t7_seed17", "T7"),
        ("t7-elevation-seed17-v1", "hall_conference_large_t7_seed17", "T7"),
        ("t7-elevation-seed17-v1", "office_vendor_machine_t7_seed17", "T7"),
        ("t7-elevation-seed17-v1", "restaurant_urban_t7_seed17", "T7"),
        (
            "t7-elevation-seed17-v1",
            "school_computer_lab_and_infirmary_t7_seed17",
            "T7",
        ),
        (
            "t8-occlusion-reveal-seed17-v1",
            "office_cubicles_right_t8_seed17",
            "T8",
        ),
    )
    accepted: set[str] = set()
    source_bundles: dict[str, Bundle] = {}
    for sweep, bundle_name, trajectory_class in sources:
        bundle = _bundle(f"{sweep}/bundles/{bundle_name}", trajectory_class)
        source_bundles[bundle_name] = bundle
        questions, _ = compile_bundle_questions(bundle)
        accepted.update(spec.fact_id for spec in questions)

    target = _bundle(
        "t10-target-view-seed17-v1/bundles/restaurant_brunch_t10_seed17",
        "T10",
    )
    accepted.update(
        spec.fact_id
        for spec in compile_t10_predictions(
            target,
            source_bundles["restaurant_brunch_t3_seed17"],
        )
    )
    assert not (accepted & audited_major_fact_ids)


def test_state_witnesses_exclude_the_audited_unrecognizable_table() -> None:
    bundle = _bundle(
        "static-m2-room-aware-indoor-seed17-v1/bundles/Pomaria_0_int_seed17",
        "T1",
    )
    tasks = json.loads((bundle.root / "reasoning_tasks.json").read_text())["tasks"]
    cross_view = next(task for task in tasks if task["task_type"] == "cross_view_relation")
    breakfast_table_id = str(cross_view["evidence_entity_ids"][0])
    witnesses = recognizable_witnesses(bundle)
    assert breakfast_table_id not in witnesses
    state = bundle.canonical_belief(evidence_view_ids_by_entity=witnesses)
    assert state["entity_count"] == len(state["entities"])
    assert all(entity["evidence_views"] for entity in state["entities"])


@pytest.mark.parametrize(
    ("relative", "trajectory_class", "rejected_fact_ids"),
    [
        (
            "t3-rotation-station-r1-seed23-v1/bundles/office_vendor_machine_t3_seed23",
            "T3",
            {"fact-c0af4f5fbd83340bc3c5"},
        ),
        (
            "t7-elevation-seed17-v1/bundles/house_double_floor_upper_t7_seed17",
            "T7",
            {"fact-6dd9c767ed30dbba9fb6"},
        ),
        (
            "t7-elevation-seed17-v1/bundles/restaurant_brunch_t7_seed17",
            "T7",
            {"fact-3078b3b7c337430017d0"},
        ),
        (
            "t7-elevation-seed17-v1/bundles/Merom_1_int_t7_seed17",
            "T7",
            {"fact-7d0f4c4220e0f9dde881", "fact-96c596bd2bb1e4e71db5"},
        ),
        (
            "static-m2-room-aware-indoor-seed17-v1/bundles/Pomaria_0_int_seed17",
            "T1",
            {"fact-1ac2a9ce719194ad4dca"},
        ),
    ],
)
def test_calibration_major_examples_are_rejected_by_general_gates(
    relative: str, trajectory_class: str, rejected_fact_ids: set[str]
) -> None:
    bundle = _bundle(relative, trajectory_class)
    questions, _ = compile_bundle_questions(bundle)
    accepted = {spec.fact_id for spec in questions}
    assert rejected_fact_ids.isdisjoint(accepted)


def test_implausibly_tall_coffee_table_is_not_used_as_a_referent() -> None:
    bundle = _bundle(
        "t4-object-orbit-adaptive-arc-seed17-v3/bundles/Beechwood_0_int_t4_seed17",
        "T4",
    )
    questions, _ = compile_bundle_questions(bundle)
    bad_ids = {
        entity.entity_id
        for entity in bundle.entities.values()
        if entity.label == "coffee_table" and entity.extent_m[2] > 0.65
    }
    assert bad_ids
    assert all(bad_ids.isdisjoint(spec.evidence_entity_ids) for spec in questions)


def test_presence_family_selects_the_cleanest_staircase_witness() -> None:
    bundle = _bundle(
        "static-m2-room-aware-indoor-seed17-v1/bundles/house_double_floor_upper_seed17",
        "T1",
    )
    questions, _ = compile_bundle_questions(bundle)
    revealed = next(
        spec
        for spec in questions
        if spec.task_type == "evidence_presence_reveal"
        and bundle.entity(spec.evidence_entity_ids[0]).label == "stairs"
    )
    decisive = next(
        check
        for check in revealed.certificate["checks"]
        if check["name"] == "target_category_visible_in_actual_input"
    )["decisive_view_id"]
    quality = _referent_visual_quality(bundle, revealed.evidence_entity_ids[0], decisive)
    assert quality["outer_five_percent_pixel_fraction"] < 0.1
