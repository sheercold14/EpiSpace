from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from episode3d.compilers import visual_quality_policy
from episode3d.semantic_visual_audit import (
    PACKET_SCHEMA,
    SemanticVisualAuditError,
    _frame_rgb_signal,
    _reviewer_only_oracle_evidence,
    build_semantic_visual_audit_packet,
    write_semantic_visual_audit_packet,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _frame_policy() -> dict[str, Any]:
    return {
        "maximum_rgb_dominant_color_fraction": 0.6,
        "minimum_rgb_quantized_entropy_bits": 2.75,
        "maximum_near_nonstructural_pixel_fraction": 0.35,
        "maximum_near_any_geometry_pixel_fraction": 0.9,
        "maximum_close_geometry_pixel_fraction": 0.9,
        "minimum_near_enclosure_pixel_fraction": 0.95,
        "maximum_near_enclosure_median_depth_m": 1.0,
        "maximum_near_enclosure_depth_spread_m": 0.75,
        "minimum_dominant_foreground_pixel_fraction": 0.6,
        "maximum_dominant_foreground_median_depth_m": 0.75,
        "dominant_foreground_required_border_sides": ["left", "right", "top"],
    }


def _safe_context_stats() -> dict[str, Any]:
    return {
        "close_geometry_pixel_fraction": 0.0,
        "enclosure_pixel_fraction": 0.0,
        "depth_median_m": 2.0,
        "depth_p90_minus_p10_m": 2.0,
        "dominant_nonstructural_instance": {
            "pixel_fraction": 0.0,
            "border_sides": [],
            "median_depth_m": None,
        },
    }


def test_frame_rgb_signal_catches_context_only_degeneration() -> None:
    signal = _frame_rgb_signal(
        {
            "dominant_quantized_color_fraction": 0.74,
            "quantized_color_entropy_bits": 1.59,
        },
        {
            "near_nonstructural_pixel_fraction": 0.0,
            "near_geometry_pixel_fraction": 0.0,
        },
        _safe_context_stats(),
        _frame_policy(),
    )
    assert signal["hard_degenerate"] is True
    assert signal["code"] == "rgb_frame_hard_degenerate"


def test_frame_rgb_signal_catches_camera_collision() -> None:
    signal = _frame_rgb_signal(
        {
            "dominant_quantized_color_fraction": 0.2,
            "quantized_color_entropy_bits": 5.0,
        },
        {
            "near_nonstructural_pixel_fraction": 0.36,
            "near_geometry_pixel_fraction": 0.4,
        },
        _safe_context_stats(),
        _frame_policy(),
    )
    assert signal["hard_degenerate"] is True
    assert signal["code"] == "camera_collision"


@pytest.mark.parametrize(
    ("context_updates", "code"),
    [
        ({"close_geometry_pixel_fraction": 0.91}, "near_surface_saturation"),
        (
            {
                "enclosure_pixel_fraction": 0.98,
                "depth_median_m": 0.84,
                "depth_p90_minus_p10_m": 0.44,
            },
            "near_enclosure",
        ),
        (
            {
                "dominant_nonstructural_instance": {
                    "pixel_fraction": 0.638,
                    "border_sides": ["left", "right", "top"],
                    "median_depth_m": 0.624,
                }
            },
            "foreground_occlusion",
        ),
    ],
)
def test_frame_rgb_signal_catches_whole_frame_context_failures(
    context_updates: dict[str, Any], code: str
) -> None:
    context = _safe_context_stats()
    context.update(context_updates)
    signal = _frame_rgb_signal(
        {
            "dominant_quantized_color_fraction": 0.2,
            "quantized_color_entropy_bits": 5.0,
        },
        {
            "near_nonstructural_pixel_fraction": 0.0,
            "near_geometry_pixel_fraction": 0.0,
        },
        context,
        _frame_policy(),
    )
    assert signal["hard_degenerate"] is True
    assert signal["code"] == code


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_bundle(root: Path) -> tuple[Path, Path]:
    bundle = root / "source" / "bundle"
    views = bundle / "views"
    views.mkdir(parents=True)
    entities = [
        {
            "entity_id": "entity-risk",
            "source_entity_id": "source-risk",
            "raw_label": "chair",
            "region_id": "room-1",
            "world_from_entity": {"translation_m": [0.0, 1.0, 0.5]},
            "obb": {"half_extents_m": [0.2, 0.2, 0.5]},
        },
        {
            "entity_id": "entity-safe",
            "source_entity_id": "source-safe",
            "raw_label": "table",
            "region_id": "room-1",
            "world_from_entity": {"translation_m": [1.0, 1.0, 0.5]},
            "obb": {"half_extents_m": [0.5, 0.5, 0.5]},
        },
    ]
    _write_json(
        bundle / "scene_ir.json",
        {
            "entities": entities,
            "runtime_semantic_id_map": {"1": "entity-risk", "2": "entity-safe"},
        },
    )
    frame = {"translation_m": [0.0, 0.0, 1.5], "rotation_xyzw": [0, 0, 0, 1]}
    _write_json(
        bundle / "spatial_episode.json",
        {
            "scene_id": "scene-source",
            "episode_id": "episode-source",
            "split_group": "scene-source",
            "observations": [
                {
                    "view_id": "view-000",
                    "step": 0,
                    "visible_entity_ids": ["entity-risk", "entity-safe"],
                    "world_from_camera": frame,
                }
            ],
        },
    )
    _write_json(
        bundle / "trajectory_plan.json",
        {
            "views": [
                {"view_id": "view-000", "role": "initial", "camera_height_m": 1.5}
            ],
        },
    )
    _write_json(
        bundle / "quality_report.json",
        {"integrity_status": "pass", "trajectory_status": "pass"},
    )
    instance = np.zeros((100, 100), dtype=np.int32)
    instance[10:30, 10:40] = 1  # 600 px, exactly 20 px on its short axis: admitted but risky.
    instance[40:100, 40:100] = 2  # 3600 px and visually far from the admission boundary.
    y, x = np.indices((100, 100), dtype=np.uint16)
    rgb = np.stack(
        ((x * 2 + y) % 256, (x + y * 2) % 256, (x * 3 + y * 5) % 256),
        axis=2,
    ).astype(np.uint8)
    rgb[instance == 1] = [220, 30, 30]
    rgb[instance == 2] = [30, 220, 30]
    np.savez_compressed(
        views / "view-000.sensors.npz",
        rgb=rgb,
        instance_id=instance,
        depth_m=np.ones((100, 100), dtype=np.float32),
    )
    model_rgb = root / "model-rgb.png"
    Image.fromarray(rgb, mode="RGB").save(model_rgb)
    return bundle, model_rgb


def test_t10_reviewer_oracle_rgb_is_hash_bound_and_never_model_input(
    tmp_path: Path,
) -> None:
    bundle, model_rgb = _source_bundle(tmp_path)
    question = {
        "task_type": "target_view_prediction",
        "certificate": {
            "oracle_target_bundle": str(bundle),
            "oracle_target_view_id": "view-000",
            "oracle_target_rgb": str(model_rgb),
        },
    }
    inventory = {
        (str(bundle.resolve()), "view-000"): {
            "rgb": str(model_rgb),
            "rgb_file_sha256": _sha256(model_rgb),
            "rgb_bytes": model_rgb.stat().st_size,
        }
    }
    evidence = _reviewer_only_oracle_evidence(question, inventory, {})
    assert evidence == {
        "purpose": "independent_answer_review_only",
        "never_model_input": True,
        "oracle_target_bundle": str(bundle.resolve()),
        "oracle_target_view_id": "view-000",
        "target_rgb": {
            "path": str(model_rgb.resolve()),
            "sha256": _sha256(model_rgb),
        },
    }
    assert _reviewer_only_oracle_evidence(
        {"task_type": "grounding_presence"}, inventory, {}
    ) is None


def _question(index: int, entity_id: str) -> dict[str, Any]:
    program_id = "grounding_presence.v1" if index % 2 else "metric_distance.v1"
    variant = "canonical" if index % 3 else "revealed"
    return {
        "fact_id": f"fact-{index:02d}",
        "task_type": "grounding_presence",
        "question_zh": f"第{index}个问题中的物体可见吗？",
        "answer_zh": "能。",
        "answer_value": True,
        "answer_status": "accepted",
        "program": {
            "program_id": program_id,
            "semantic_signature": "G(view,entity)->V",
        },
        "evidence_view_ids": ["view-000"],
        "evidence_entity_ids": [entity_id],
        "certificate": {"result": "pass", "checks": []},
        "family_variant": variant,
        "consistency_group": f"consistency-{index // 2}",
        "model_view_ids": ["view-000"],
    }


def _release(root: Path, *, risk_count: int = 2) -> Path:
    source_bundle, model_rgb = _source_bundle(root)
    release = root / "release"
    release.mkdir()
    splits = ("train", "val", "test", "train", "val", "test", "train", "val")
    trajectories = ("T1", "T3", "T4", "T7", "T8", "T1", "T3", "T7")
    episodes = []
    for index in range(8):
        episodes.append(
            {
                "schema_version": "epispace.episode_ir.v1",
                "episode_id": f"episode-audit-{index}",
                "scene_id": f"scene-audit-{index}",
                "split": splits[index],
                "trajectory_class": trajectories[index],
                "source_sweep": "synthetic-sweep",
                "source_bundle": str(source_bundle),
                "observations": [
                    {
                        "view_id": "view-000",
                        "step": 0,
                        "role": "initial",
                        "rgb": str(model_rgb),
                        "camera_height_m": 1.5,
                        "horizontal_fov_deg": 90.0,
                    }
                ],
                "questions": [
                    _question(index, "entity-risk" if index < risk_count else "entity-safe")
                ],
            }
        )
    ir_path = release / "episodes.ir.jsonl"
    ir_path.write_text(
        "".join(json.dumps(episode) + "\n" for episode in episodes),
        encoding="utf-8",
    )
    sensor_path = source_bundle / "views" / "view-000.sensors.npz"
    inventory_path = release / "source_inventory.json"
    _write_json(
        inventory_path,
        {
            "schema_version": "epispace.source_inventory.v1",
            "bundles": [
                {
                    "bundle_root": str(source_bundle),
                    "views": [
                        {
                            "view_id": "view-000",
                            "rgb": str(model_rgb),
                            "rgb_file_sha256": _sha256(model_rgb),
                            "rgb_bytes": model_rgb.stat().st_size,
                            "source_sensor": str(sensor_path),
                            "source_sensor_sha256": _sha256(sensor_path),
                            "source_sensor_bytes": sensor_path.stat().st_size,
                        }
                    ],
                }
            ],
        },
    )
    _write_json(
        release / "release_manifest.json",
        {
            "schema_version": "epispace.release_manifest.v1",
            "dataset_id": "synthetic-audit-release",
            "status": "pass",
            "research_contract": {
                "visual_quality_policy": visual_quality_policy()
            },
            "corpus_gates": {
                "status": "pass",
                "checks": {"synthetic_fixture_gate": True},
            },
            "artifacts": {
                "episode_ir": "episodes.ir.jsonl",
                "source_inventory": "source_inventory.json",
            },
            "artifact_integrity": {
                "episode_ir": {
                    "path": "episodes.ir.jsonl",
                    "sha256": _sha256(ir_path),
                    "bytes": ir_path.stat().st_size,
                    "records": len(episodes),
                },
                "source_inventory": {
                    "path": "source_inventory.json",
                    "sha256": _sha256(inventory_path),
                    "bytes": inventory_path.stat().st_size,
                },
            },
        },
    )
    return release


def test_packet_is_reproducible_unreviewed_and_risk_oversampled(tmp_path: Path) -> None:
    release = _release(tmp_path)
    first_json = tmp_path / "first.json"
    first_md = tmp_path / "first.md"
    second_json = tmp_path / "second.json"
    second_md = tmp_path / "second.md"
    first = write_semantic_visual_audit_packet(
        release,
        json_output=first_json,
        markdown_output=first_md,
        sample_size=4,
        seed=17,
        risk_oversample_multiplier=2.0,
    )
    second = write_semantic_visual_audit_packet(
        release,
        json_output=second_json,
        markdown_output=second_md,
        sample_size=4,
        seed=17,
        risk_oversample_multiplier=2.0,
    )

    assert first == second
    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_md.read_bytes() == second_md.read_bytes()
    assert first["schema_version"] == PACKET_SCHEMA
    assert first["status"] == "awaiting_independent_review"
    assert first["independent_review_completed"] is False
    assert first["reviewer_provenance_policy"]["allowed_reviewer_types"] == [
        "model_assisted_independent",
        "human_independent",
    ]
    assert first["release_binding"]["episode_ir"]["sha256"] == _sha256(
        release / "episodes.ir.jsonl"
    )
    assert first["release_binding"]["source_inventory"]["sha256"] == _sha256(
        release / "source_inventory.json"
    )
    assert len(first["review_evidence_binding"]["sha256"]) == 64
    assert first["sampling_contract"]["stratum_key"] == [
        "split",
        "trajectory_class",
        "program_id",
        "family_variant",
    ]
    assert first["population_summary"]["near_threshold_fraction"] == 0.25
    assert first["sample_summary"]["near_threshold_fraction"] == 0.5
    assert len(first["sample_summary"]["split"]) >= 2
    assert len(first["sample_summary"]["trajectory_class"]) >= 3
    assert len(first["sample_summary"]["program_id"]) == 2
    assert len(first["sample_summary"]["family_variant"]) == 2
    assert {item["fact_id"] for item in first["items"]} >= {"fact-00", "fact-01"}
    for item in first["items"]:
        assert item["reviewer_fields"] == {
            "reviewer_type": None,
            "reviewer_id": "",
            "reviewer_system": "",
            "reviewer_model": "",
            "review_protocol_id": "",
            "review_prompt_sha256": "",
            "reviewed_at": "",
            "overall_status": None,
            "referents_recognizable": None,
            "answer_supported_by_model_rgb": None,
            "family_intervention_valid": None,
            "severity": None,
            "reason_codes": [],
            "notes_zh": "",
        }
        assert len(item["actual_model_rgb"]) == 1
        rgb = item["actual_model_rgb"][0]
        assert Path(rgb["path"]).is_file()
        assert rgb["sha256"] == _sha256(Path(rgb["path"]))
        assert rgb["camera_height_m"] == 1.5
        assert rgb["horizontal_fov_deg"] == 90.0
    risk_items = {item["fact_id"]: item for item in first["items"] if item["fact_id"] in {"fact-00", "fact-01"}}
    assert risk_items["fact-00"]["family_siblings_for_intervention_review"][0][
        "fact_id"
    ] == "fact-01"
    assert risk_items["fact-01"]["family_siblings_for_intervention_review"][0][
        "fact_id"
    ] == "fact-00"
    markdown = first_md.read_text(encoding="utf-8")
    assert "当前状态：尚未进行独立复核" in markdown
    assert "模型复核不得写成人工复核" in markdown
    assert "相机高度和水平视场角" in markdown
    assert "Reviewer type： [ ] model_assisted_independent" in markdown
    assert "实际模型输入 RGB" in markdown
    assert "Overall status： [ ] pass" in markdown


def test_manifest_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    release = _release(tmp_path)
    with (release / "episodes.ir.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(SemanticVisualAuditError, match="SHA"):
        build_semantic_visual_audit_packet(release, sample_size=2)


def test_risk_target_is_a_cap_when_nonrisk_candidates_exist(tmp_path: Path) -> None:
    release = _release(tmp_path, risk_count=4)
    packet = build_semantic_visual_audit_packet(
        release,
        sample_size=4,
        seed=23,
        risk_oversample_multiplier=1.5,
    )
    assert packet["sampling_contract"]["near_threshold_target_count"] == 3
    assert packet["sample_summary"]["near_threshold_questions"] == 3


def test_nonpassing_candidate_release_is_not_presented_for_independent_review(
    tmp_path: Path,
) -> None:
    release = _release(tmp_path)
    manifest_path = release / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "fail"
    _write_json(manifest_path, manifest)
    with pytest.raises(SemanticVisualAuditError, match="not passing"):
        build_semantic_visual_audit_packet(release, sample_size=2)


def test_failed_corpus_gate_cannot_be_hidden_behind_manifest_pass(tmp_path: Path) -> None:
    release = _release(tmp_path)
    manifest_path = release / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["corpus_gates"]["checks"]["synthetic_fixture_gate"] = False
    _write_json(manifest_path, manifest)
    with pytest.raises(SemanticVisualAuditError, match="failed or non-boolean"):
        build_semantic_visual_audit_packet(release, sample_size=2)


def test_model_rgb_drift_from_source_inventory_fails_closed(tmp_path: Path) -> None:
    release = _release(tmp_path)
    inventory = json.loads((release / "source_inventory.json").read_text(encoding="utf-8"))
    rgb_path = Path(inventory["bundles"][0]["views"][0]["rgb"])
    image = np.asarray(Image.open(rgb_path).convert("RGB")).copy()
    image[0, 0] = [255, 255, 255]
    Image.fromarray(image, mode="RGB").save(rgb_path)
    with pytest.raises(SemanticVisualAuditError, match="changed"):
        build_semantic_visual_audit_packet(release, sample_size=2)


def test_source_sensor_drift_from_source_inventory_fails_closed(tmp_path: Path) -> None:
    release = _release(tmp_path)
    inventory = json.loads((release / "source_inventory.json").read_text(encoding="utf-8"))
    sensor_path = Path(inventory["bundles"][0]["views"][0]["source_sensor"])
    with sensor_path.open("ab") as handle:
        handle.write(b"drift")
    with pytest.raises(SemanticVisualAuditError, match="changed"):
        build_semantic_visual_audit_packet(release, sample_size=2)


def test_invalid_visual_policy_range_fails_closed(tmp_path: Path) -> None:
    release = _release(tmp_path)
    manifest_path = release / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["research_contract"]["visual_quality_policy"]["minimum_visible_pixels"] = 0
    _write_json(manifest_path, manifest)
    with pytest.raises(SemanticVisualAuditError, match="must be positive"):
        build_semantic_visual_audit_packet(release, sample_size=2)


def test_standalone_cli_wrapper_runs_outside_project(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_semantic_visual_audit_packet.py"),
            "--help",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "release_dir" in result.stdout
