from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from episode3d.benchmark_evaluator import evaluate_files, write_oracle_predictions
from episode3d.bundles import _read_valid_rgb_metadata
from episode3d.compute_matching import build_compute_matched_schedules
from episode3d.implementation_binding import (
    RELEASE_CRITICAL_PATHS,
    build_implementation_binding,
)
from episode3d.pipeline import _invalidate_derived_outputs
from episode3d.release_verifier import (
    ReleaseVerificationError,
    _verify_audit,
    _verify_base,
    _verify_evaluator,
    _verify_implementation_binding,
    _verify_release_impl,
    _verify_schedules,
    _verify_semantic_visual_audit,
    _verify_source_inventory,
    main,
)
from episode3d.semantic_visual_audit import (
    PACKET_SCHEMA,
)
from episode3d.semantic_visual_audit import (
    render_markdown as render_audit_packet_markdown,
)
from episode3d.semantic_visual_audit_results import (
    OPTIONAL_REVIEW_SCHEMA,
    write_semantic_visual_audit_result,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _semantic_audit_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict, Path, Path]:
    release = tmp_path / "release"
    audit = release / "semantic_visual_audit"
    review_path = audit / "reviews" / "review-a.json"
    episode_ir = release / "episodes.ir.jsonl"
    source_inventory = release / "source_inventory.json"
    _write_jsonl(episode_ir, [{"episode_id": "episode-a"}])
    _write_json(source_inventory, {"schema_version": "epispace.source_inventory.v1"})
    manifest = {
        "schema_version": "epispace.release_manifest.v1",
        "dataset_id": "epispace-test",
        "status": "pass",
        "artifacts": {
            "episode_ir": episode_ir.name,
            "source_inventory": source_inventory.name,
        },
        "artifact_integrity": {
            "episode_ir": {
                "path": episode_ir.name,
                "sha256": _sha(episode_ir),
                "bytes": episode_ir.stat().st_size,
                "records": 1,
            },
            "source_inventory": {
                "path": source_inventory.name,
                "sha256": _sha(source_inventory),
                "bytes": source_inventory.stat().st_size,
            },
        },
    }
    manifest_path = release / "release_manifest.json"
    _write_json(manifest_path, manifest)
    reviewer_fields = {
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
    item = {
        "audit_index": 1,
        "fact_id": "fact-a",
        "episode_id": "episode-a",
        "scene_id": "scene-a",
        "split": "test",
        "trajectory_class": "T1",
        "source_sweep": "sweep-a",
        "source_bundle": "/source/bundle-a",
        "program_id": "grounding.presence.v1",
        "semantic_signature": "G->V",
        "task_type": "grounding",
        "family_variant": "canonical",
        "consistency_group": None,
        "family_siblings_for_intervention_review": [],
        "question_zh": "画面中有椅子吗？",
        "answer_zh": "有。",
        "answer_status": "answerable",
        "answer_value": True,
        "evidence_view_ids": ["view-000"],
        "evidence_entity_ids": ["chair-1"],
        "actual_model_rgb": [
            {
                "order": 1,
                "view_id": "view-000",
                "path": "/rgb/view-000.png",
                "sha256": "a" * 64,
                "camera_height_m": 1.5,
                "horizontal_fov_deg": 90.0,
            }
        ],
        "automated_sampling_signal": {
            "risk_score": 0.0,
            "near_threshold": False,
        },
        "reviewer_fields": reviewer_fields,
    }
    evidence = [{key: value for key, value in item.items() if key != "reviewer_fields"}]
    evidence_sha = hashlib.sha256(
        json.dumps(
            evidence,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    summary = {
        "questions": 1,
        "near_threshold_fraction": 0.0,
        "split": {"test": 1},
        "trajectory_class": {"T1": 1},
        "program_id": {"grounding.presence.v1": 1},
        "family_variant": {"canonical": 1},
    }
    packet = {
        "schema_version": PACKET_SCHEMA,
        "packet_id": "semantic-audit-fixture",
        "status": "awaiting_independent_review",
        "independent_review_completed": False,
        "reviewer_provenance_policy": {
            "allowed_reviewer_types": [
                "model_assisted_independent",
                "human_independent",
            ]
        },
        "release_binding": {
            "release_dir": str(release.resolve()),
            "dataset_id": manifest["dataset_id"],
            "release_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": _sha(manifest_path),
            },
            "episode_ir": {
                "path": str(episode_ir.resolve()),
                "sha256": _sha(episode_ir),
                "records": 1,
            },
            "source_inventory": {
                "path": str(source_inventory.resolve()),
                "sha256": _sha(source_inventory),
            },
            "final_release_index": None,
        },
        "review_evidence_binding": {"sha256": evidence_sha},
        "sampling_contract": {
            "seed": 17,
            "stratum_key": [
                "split",
                "trajectory_class",
                "program_id",
                "family_variant",
            ],
        },
        "population_summary": summary,
        "sample_summary": summary,
        "items": [item],
        "source_release_status": "pass",
    }
    packet_path = audit / "packet.json"
    _write_json(packet_path, packet)
    (audit / "packet.md").write_text(
        render_audit_packet_markdown(packet), encoding="utf-8"
    )
    prompt = "Independently review actual model RGB evidence."
    _write_json(
        review_path,
        {
            "schema_version": OPTIONAL_REVIEW_SCHEMA,
            "packet_id": packet["packet_id"],
            "reviewer_provenance": {
                "reviewer_type": "model_assisted_independent",
                "reviewer_id": "agent-a",
                "reviewer_system": "OpenAI Codex",
                "reviewer_model": "GPT-5",
                "review_protocol_id": "epispace.semantic_rgb_review.v1",
                "review_prompt": prompt,
                "review_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "reviewed_at": "2026-07-17T01:02:03Z",
            },
            "reviews": [
                {
                    "index": 0,
                    "fact_id": "fact-a",
                    "overall_status": "pass",
                    "referents_recognizable": True,
                    "answer_supported": True,
                    "family_intervention_valid": None,
                    "severity": "none",
                    "reason_codes": [],
                    "notes_zh": "独立查看实际模型 RGB 后通过。",
                }
            ],
        },
    )
    result_path = audit / "result.json"
    write_semantic_visual_audit_result(
        packet_path,
        [review_path],
        json_output=result_path,
        markdown_output=audit / "result.md",
    )
    return release, manifest_path, manifest, review_path, result_path


def _training_row(
    record_id: str,
    comparison_id: str,
    arm: str,
    facts: list[str],
    image: Path,
) -> dict:
    return {
        "record_id": record_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image)},
                    {"type": "text", "text": "问题"},
                ],
            },
            {"role": "assistant", "content": "答案"},
        ],
        "comparison_contract": {
            "comparison_id": comparison_id,
            "arm": arm,
            "fact_ids": facts,
        },
    }


def _schedule_fixture(tmp_path: Path) -> tuple[Path, dict]:
    release = tmp_path / "release"
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 6), (10, 20, 30)).save(image)
    episode = release / "train.episode_sft.jsonl"
    isolated = release / "train.isolated_sft.jsonl"
    _write_jsonl(
        episode,
        [_training_row("episode", "comparison", "episode", ["fact-a", "fact-b"], image)],
    )
    _write_jsonl(
        isolated,
        [
            _training_row("isolated-a", "comparison", "isolated", ["fact-a"], image),
            _training_row("isolated-b", "comparison", "isolated", ["fact-b"], image),
        ],
    )
    build_compute_matched_schedules(episode, isolated, release / "compute_matching", seed=17)
    manifest = {
        "artifacts": {
            "episode_sft": episode.name,
            "isolated_sft": isolated.name,
        },
        "artifact_integrity": {
            "episode_sft": {"path": episode.name, "sha256": _sha(episode)},
            "isolated_sft": {"path": isolated.name, "sha256": _sha(isolated)},
        },
    }
    return release, manifest


def _source_inventory_fixture(tmp_path: Path) -> tuple[Path, dict, Path, Path]:
    release = tmp_path / "release"
    release.mkdir()
    config = tmp_path / "configs" / "pipeline.json"
    sweep = tmp_path / "sources" / "sweep.json"
    bundle = tmp_path / "sources" / "bundle"
    media = tmp_path / "media"
    rgb = media / "bundle" / "view-000.png"
    sidecar = rgb.with_suffix(".rgb.json")
    sensor = bundle / "views" / "view-000.sensors.npz"
    _write_json(
        config,
        {
            "schema_version": "epispace.pipeline_config.v1",
            "sources": [
                {
                    "name": "sweep",
                    "trajectory_class": "T1",
                    "role": "write_source",
                    "sweep_plan": str(sweep),
                }
            ],
        },
    )
    _write_json(sweep, {"jobs": [{"status": "passed"}]})
    for name in (
        "scene_ir.json",
        "spatial_episode.json",
        "trajectory_plan.json",
        "quality_report.json",
    ):
        _write_json(bundle / name, {"name": name})
    sensor.parent.mkdir(parents=True)
    sensor.write_bytes(b"fixed-size-sensor")
    rgb.parent.mkdir(parents=True)
    Image.new("RGB", (4, 3), (40, 50, 60)).save(rgb)
    with Image.open(rgb) as image:
        pixel_sha = hashlib.sha256(image.tobytes()).hexdigest()
    sensor_stat = sensor.stat()
    metadata = {
        "schema_version": "epispace.raw_rgb.v1",
        "source_sensor": str(sensor),
        "source_size": sensor_stat.st_size,
        "source_mtime_ns": sensor_stat.st_mtime_ns,
        "width": 4,
        "height": 3,
        "mode": "RGB",
        "rgb_sha256": pixel_sha,
        "instance_pixel_counts": {},
    }
    _write_json(sidecar, metadata)
    source_files = {
        name: {
            "sha256": _sha(bundle / name),
            "bytes": (bundle / name).stat().st_size,
        }
        for name in (
            "scene_ir.json",
            "spatial_episode.json",
            "trajectory_plan.json",
            "quality_report.json",
        )
    }
    inventory = {
        "schema_version": "epispace.source_inventory.v1",
        "config": {"path": str(config), "sha256": _sha(config)},
        "sweeps": [{"name": "sweep", "path": str(sweep), "sha256": _sha(sweep)}],
        "bundle_count": 1,
        "view_count": 1,
        "bundles": [
            {
                "episode_id": "episode",
                "source_sweep": "sweep",
                "trajectory_class": "T1",
                "bundle_root": str(bundle),
                "source_files": source_files,
                "absent_optional_files": [
                    "render_report.json",
                    "reasoning_tasks.json",
                ],
                "views": [
                    {
                        "view_id": "view-000",
                        "rgb": str(rgb),
                        "rgb_file_sha256": _sha(rgb),
                        "rgb_bytes": rgb.stat().st_size,
                        "rgb_pixel_sha256": pixel_sha,
                        "rgb_sidecar": str(sidecar),
                        "rgb_sidecar_sha256": _sha(sidecar),
                        "source_sensor": str(sensor),
                        "source_sensor_sha256": _sha(sensor),
                        "source_sensor_bytes": sensor_stat.st_size,
                        "source_sensor_mtime_ns": sensor_stat.st_mtime_ns,
                    }
                ],
            }
        ],
    }
    inventory_path = release / "source_inventory.json"
    _write_json(inventory_path, inventory)
    manifest = {
        "artifacts": {"source_inventory": inventory_path.name},
        "source_integrity": {
            "source_inventory_sha256": _sha(inventory_path),
            "config_sha256": _sha(config),
            "sweep_plan_sha256": {"sweep": _sha(sweep)},
        },
        "statistics": {
            "acquisition": {
                "model_rgb_root": str(media),
                "planned_jobs": 1,
                "strict_bundles_loaded": 1,
                "sweeps": [
                    {
                        "name": "sweep",
                        "trajectory_class": "T1",
                        "role": "write_source",
                        "sweep_plan": str(sweep),
                        "sweep_plan_sha256": _sha(sweep),
                        "planned_jobs": 1,
                        "status_counts": {"passed": 1},
                        "strict_bundles_loaded": 1,
                        "missing_optional_reasoning_task_manifests": 1,
                    }
                ],
            }
        },
    }
    return release, manifest, sensor, bundle / "quality_report.json"


def test_source_inventory_replays_every_source_and_sensor_content(tmp_path: Path) -> None:
    release, manifest, sensor, _quality = _source_inventory_fixture(tmp_path)
    summary, referenced, _media = _verify_source_inventory(release, manifest)
    assert summary["view_count"] == 1
    assert len(referenced) == 1

    original_stat = sensor.stat()
    sensor.write_bytes(b"changed-sensor!!!")
    os.utime(sensor, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert sensor.stat().st_size == original_stat.st_size
    with pytest.raises(ReleaseVerificationError, match="sensor content changed"):
        _verify_source_inventory(release, manifest)


def test_source_inventory_detects_bundle_json_change(tmp_path: Path) -> None:
    release, manifest, _sensor, quality = _source_inventory_fixture(tmp_path)
    _write_json(quality, {"name": "mutated"})
    with pytest.raises(ReleaseVerificationError, match=r"bundle source.*SHA mismatch"):
        _verify_source_inventory(release, manifest)


def test_source_inventory_detects_newly_appearing_optional_input(tmp_path: Path) -> None:
    release, manifest, _sensor, quality = _source_inventory_fixture(tmp_path)
    _write_json(quality.parent / "reasoning_tasks.json", {"tasks": []})
    with pytest.raises(ReleaseVerificationError, match="optional source appeared"):
        _verify_source_inventory(release, manifest)


def test_rgb_cache_rejects_same_shape_pixel_corruption(tmp_path: Path) -> None:
    sensor = tmp_path / "view.sensors.npz"
    sensor.write_bytes(b"sensor")
    image_path = tmp_path / "view.png"
    Image.new("RGB", (4, 3), (1, 2, 3)).save(image_path)
    with Image.open(image_path) as image:
        pixel_sha = hashlib.sha256(image.tobytes()).hexdigest()
    stat = sensor.stat()
    metadata_path = tmp_path / "view.rgb.json"
    _write_json(
        metadata_path,
        {
            "schema_version": "epispace.raw_rgb.v1",
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "width": 4,
            "height": 3,
            "rgb_sha256": pixel_sha,
            "instance_pixel_counts": {},
        },
    )
    assert _read_valid_rgb_metadata(metadata_path, image_path, sensor) is not None
    Image.new("RGB", (4, 3), (9, 8, 7)).save(image_path)
    assert _read_valid_rgb_metadata(metadata_path, image_path, sensor) is None


def test_schedule_replay_rejects_tamper_even_if_derived_hash_is_updated(
    tmp_path: Path,
) -> None:
    release, manifest = _schedule_fixture(tmp_path)
    schedule = release / "compute_matching" / "image_occurrence_matched.episode.schedule.jsonl"
    rows = [json.loads(line) for line in schedule.read_text(encoding="utf-8").splitlines()]
    rows[0]["fact_ids"] = ["invented-fact"]
    _write_jsonl(schedule, rows)
    matching_manifest_path = release / "compute_matching" / "compute_matching_manifest.json"
    matching_manifest = json.loads(matching_manifest_path.read_text(encoding="utf-8"))
    matching_manifest["regimes"]["image_occurrence_matched"]["schedule_files"]["episode"][
        "sha256"
    ] = _sha(schedule)
    _write_json(matching_manifest_path, matching_manifest)

    with pytest.raises(ReleaseVerificationError, match="fact_ids mismatch"):
        _verify_schedules(release, manifest)


def test_schedule_replay_checks_each_draw_source_digest_explicitly(
    tmp_path: Path,
) -> None:
    release, manifest = _schedule_fixture(tmp_path)
    schedule = release / "compute_matching" / "image_occurrence_matched.episode.schedule.jsonl"
    rows = [json.loads(line) for line in schedule.read_text(encoding="utf-8").splitlines()]
    rows[0]["source_jsonl_sha256"] = "0" * 64
    _write_jsonl(schedule, rows)
    matching_manifest_path = release / "compute_matching" / "compute_matching_manifest.json"
    matching_manifest = json.loads(matching_manifest_path.read_text(encoding="utf-8"))
    matching_manifest["regimes"]["image_occurrence_matched"]["schedule_files"]["episode"][
        "sha256"
    ] = _sha(schedule)
    _write_json(matching_manifest_path, matching_manifest)

    with pytest.raises(ReleaseVerificationError, match="source_jsonl_sha256 mismatch"):
        _verify_schedules(release, manifest)


def test_schedule_replay_rejects_fabricated_manifest_summary(tmp_path: Path) -> None:
    release, manifest = _schedule_fixture(tmp_path)
    matching_manifest_path = release / "compute_matching" / "compute_matching_manifest.json"
    matching_manifest = json.loads(matching_manifest_path.read_text(encoding="utf-8"))
    matching_manifest["pairing"]["comparison_count"] = 999
    _write_json(matching_manifest_path, matching_manifest)
    with pytest.raises(ReleaseVerificationError, match="deterministic replay"):
        _verify_schedules(release, manifest)


def test_base_rejects_artifact_path_escape(tmp_path: Path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    manifest = {
        "schema_version": "epispace.release_manifest.v1",
        "status": "pass",
        "corpus_gates": {"checks": {"gate": True}},
        "artifacts": {"escaped": "../outside.jsonl"},
        "artifact_integrity": {
            "escaped": {
                "path": "../outside.jsonl",
                "sha256": _sha(outside),
                "bytes": outside.stat().st_size,
                "records": 1,
            }
        },
    }
    with pytest.raises(ReleaseVerificationError, match="escapes its release boundary"):
        _verify_base(release, manifest)


def test_evaluator_report_is_replayed_instead_of_trusted(tmp_path: Path) -> None:
    release = tmp_path / "release"
    benchmark = release / "benchmark.core.jsonl"
    predictions = release / "evaluation_smoke" / "oracle.predictions.jsonl"
    evaluation = release / "evaluation_smoke" / "evaluation.json"
    _write_jsonl(
        benchmark,
        [
            {
                "record_id": "record",
                "split": "test",
                "family_id": "family",
                "family_variant": "canonical",
                "program": {"program_id": "relation.v1"},
                "target": {"task_type": "relation", "answer_value": "left_of"},
            }
        ],
    )
    write_oracle_predictions(benchmark, predictions)
    report = evaluate_files(benchmark, predictions)
    _write_json(evaluation, report)
    manifest = {
        "artifacts": {"benchmark_core": benchmark.name},
        "artifact_integrity": {"benchmark_core": {"sha256": _sha(benchmark)}},
    }
    assert _verify_evaluator(release, manifest)["replayed"] is True

    report["per_task"]["relation"]["correct"] = 0
    _write_json(evaluation, report)
    with pytest.raises(ReleaseVerificationError, match="does not match evaluator replay"):
        _verify_evaluator(release, manifest)


def test_corpus_audit_report_is_replayed_instead_of_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    manifest_path = release / "release_manifest.json"
    _write_json(manifest_path, {"schema_version": "epispace.release_manifest.v1"})
    stored = {
        "schema_version": "epispace.corpus_audit.v1",
        "source_manifest_sha256": _sha(manifest_path),
        "sections": {"images": {"status": "pass"}},
        "summary": {"status": "pass", "warnings": []},
    }
    _write_json(release / "corpus_audit.json", stored)

    monkeypatch.setattr(
        "episode3d.corpus_audit.audit_release",
        lambda _release: {
            "sections": {"images": {"status": "fail"}},
            "summary": {"status": "fail", "warnings": []},
        },
    )
    with pytest.raises(ReleaseVerificationError, match=r"audit sections.*replay"):
        _verify_audit(release, manifest_path)


def test_semantic_visual_audit_is_replayed_as_formal_release_authority(
    tmp_path: Path,
) -> None:
    release, manifest_path, manifest, _review, _result = _semantic_audit_fixture(
        tmp_path
    )

    verified = _verify_semantic_visual_audit(release, manifest_path, manifest)

    assert verified["status"] == "pass"
    assert verified["gate"] is True
    assert verified["packet"]["path"] == "semantic_visual_audit/packet.json"
    assert verified["packet"]["item_count"] == 1
    assert verified["reviews"][0]["reviewer_id"] == "agent-a"
    assert verified["reviews"][0]["reviewer_type"] == (
        "model_assisted_independent"
    )
    assert verified["result"]["semantic_visual_audit_gate"] is True
    assert verified["summary"]["reviewed_items"] == 1
    assert verified["release_binding"]["release_manifest_sha256"] == _sha(
        manifest_path
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("packet_binding", "release_manifest SHA is stale"),
        ("prior_index", "must not bind a prior final release index"),
        ("packet_markdown", "packet Markdown"),
        ("review", "does not exactly match adjudication replay"),
        ("result", "does not exactly match adjudication replay"),
        ("result_markdown", "result Markdown"),
    ],
)
def test_semantic_visual_audit_fails_closed_on_stale_or_tampered_inputs(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    release, manifest_path, manifest, review_path, result_path = (
        _semantic_audit_fixture(tmp_path)
    )
    audit = release / "semantic_visual_audit"
    if mutation in {"packet_binding", "prior_index"}:
        packet_path = audit / "packet.json"
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        if mutation == "packet_binding":
            packet["release_binding"]["release_manifest"]["sha256"] = "0" * 64
        else:
            packet["release_binding"]["final_release_index"] = {
                "path": str(release / "final_release_index.json"),
                "sha256": "0" * 64,
            }
        _write_json(packet_path, packet)
        (audit / "packet.md").write_text(
            render_audit_packet_markdown(packet), encoding="utf-8"
        )
    elif mutation == "packet_markdown":
        (audit / "packet.md").write_text("tampered\n", encoding="utf-8")
    elif mutation == "review":
        review = json.loads(review_path.read_text(encoding="utf-8"))
        review["reviews"][0]["notes_zh"] = "内容被修改。"
        _write_json(review_path, review)
    elif mutation == "result":
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["summary"]["reviewed_items"] = 99
        _write_json(result_path, result)
    else:
        (audit / "result.md").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ReleaseVerificationError, match=match):
        _verify_semantic_visual_audit(release, manifest_path, manifest)


@pytest.mark.parametrize("extra_name", ["notes.txt", "nested", "linked.json"])
def test_semantic_visual_audit_review_directory_is_an_exact_json_closure(
    tmp_path: Path, extra_name: str
) -> None:
    release, manifest_path, manifest, _review, _result = _semantic_audit_fixture(
        tmp_path
    )
    extra = release / "semantic_visual_audit" / "reviews" / extra_name
    if extra_name == "nested":
        extra.mkdir()
    elif extra_name == "linked.json":
        extra.symlink_to(release / "semantic_visual_audit" / "packet.json")
    else:
        extra.write_text("not bound", encoding="utf-8")
    with pytest.raises(ReleaseVerificationError, match=r"only regular \.json"):
        _verify_semantic_visual_audit(release, manifest_path, manifest)


def test_rebuild_and_failed_cli_remove_stale_final_index(tmp_path: Path) -> None:
    release = tmp_path / "release"
    (release / "compute_matching").mkdir(parents=True)
    (release / "evaluation_smoke").mkdir()
    (release / "semantic_visual_audit" / "reviews").mkdir(parents=True)
    (release / "semantic_visual_audit" / "packet.json").write_text(
        "stale", encoding="utf-8"
    )
    (release / "semantic_visual_audit_packet.json").write_text(
        "legacy stale", encoding="utf-8"
    )
    for name in (
        "corpus_audit.json",
        "corpus_audit.md",
        "final_release_index.json",
    ):
        (release / name).write_text("stale", encoding="utf-8")
    _invalidate_derived_outputs(release)
    assert not (release / "compute_matching").exists()
    assert not (release / "evaluation_smoke").exists()
    assert not (release / "semantic_visual_audit").exists()
    assert not (release / "semantic_visual_audit_packet.json").exists()
    assert not (release / "final_release_index.json").exists()

    _write_json(release / "release_manifest.json", {})
    (release / "final_release_index.json").write_text("old pass", encoding="utf-8")
    assert main([str(release)]) == 1
    assert not (release / "final_release_index.json").exists()

    manifest = release / "release_manifest.json"
    before = manifest.read_bytes()
    assert main([str(release), "--output", str(manifest)]) == 1
    assert manifest.read_bytes() == before


def _implementation_tree(root: Path) -> None:
    for index, relative in enumerate(RELEASE_CRITICAL_PATHS):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"implementation file {index}: {relative}\n", encoding="utf-8")


def test_implementation_binding_rejects_changed_compiler(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _implementation_tree(project)
    manifest = {"implementation_integrity": build_implementation_binding(project)}
    verified = _verify_implementation_binding(manifest, implementation_root=project)
    assert verified["verified_files"] == len(RELEASE_CRITICAL_PATHS)
    assert "episode3d/compilers.py" in verified["files"]
    assert verified["files"]["episode3d/compilers.py"]["path"] == (
        "episode3d/compilers.py"
    )
    assert "episode3d/semantic_visual_audit.py" in verified["files"]
    assert "episode3d/semantic_visual_audit_results.py" in verified["files"]
    assert "scripts/build_semantic_visual_audit_packet.py" in verified["files"]
    assert "scripts/adjudicate_semantic_visual_audit.py" in verified["files"]

    compiler = project / "episode3d/compilers.py"
    compiler.write_text("changed compiler semantics\n", encoding="utf-8")
    with pytest.raises(
        ReleaseVerificationError,
        match=r"implementation SHA mismatch: episode3d/compilers.py",
    ):
        _verify_implementation_binding(manifest, implementation_root=project)


def test_implementation_binding_is_required_and_closed(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _implementation_tree(project)
    with pytest.raises(ReleaseVerificationError, match="implementation_integrity"):
        _verify_implementation_binding({}, implementation_root=project)

    binding = build_implementation_binding(project)
    del binding["files"]["episode3d/compilers.py"]
    with pytest.raises(ReleaseVerificationError, match="exact release-critical closure"):
        _verify_implementation_binding(
            {"implementation_integrity": binding}, implementation_root=project
        )


def test_final_index_contains_replayed_implementation_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = tmp_path / "release"
    media = tmp_path / "media"
    release.mkdir()
    media.mkdir()
    binding = build_implementation_binding()
    _write_json(release / "release_manifest.json", {"implementation_integrity": binding})

    monkeypatch.setattr("episode3d.release_verifier._verify_base", lambda *_: {})
    monkeypatch.setattr(
        "episode3d.release_verifier._verify_source_inventory",
        lambda *_: ({}, set(), media),
    )
    monkeypatch.setattr(
        "episode3d.release_verifier._collect_exported_pngs", lambda *_: set()
    )
    monkeypatch.setattr("episode3d.release_verifier._verify_schedules", lambda *_: {})
    monkeypatch.setattr("episode3d.release_verifier._verify_statistics", lambda *_: {})
    monkeypatch.setattr("episode3d.release_verifier._verify_audit", lambda *_: {})
    monkeypatch.setattr("episode3d.release_verifier._verify_evaluator", lambda *_: {})
    monkeypatch.setattr(
        "episode3d.release_verifier._verify_semantic_visual_audit", lambda *_: {}
    )

    index = _verify_release_impl(release)
    assert index["status"] == "pass"
    assert index["implementation_integrity"]["aggregate_sha256"] == binding[
        "aggregate_sha256"
    ]
    assert index["implementation_integrity"]["verified_files"] == len(
        RELEASE_CRITICAL_PATHS
    )
