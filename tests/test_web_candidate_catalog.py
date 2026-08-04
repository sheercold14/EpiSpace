from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from scripts.build_web_candidate_catalog import (
    CATALOG_SCHEMA,
    DETAIL_SCHEMA,
    WebCandidateBuildError,
    build_web_candidate_catalog,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def _episode(
    project: Path,
    *,
    episode_id: str,
    trajectory_class: str,
    split: str,
    cross_view: bool,
) -> dict[str, Any]:
    observations = []
    for index in range(3 if cross_view else 2):
        image = project / "media" / episode_id / f"view-{index:03d}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"synthetic-rgb")
        observations.append(
            {
                "view_id": f"view-{index:03d}",
                "step": index,
                "role": "initial" if index == 0 else "explore",
                "rgb": str(image),
                "camera_height_m": 1.2,
                "horizontal_fov_deg": 90.0,
            }
        )
    evidence = ["view-000", "view-001"] if cross_view else ["view-000"]
    return {
        "schema_version": "epispace.episode_ir.v1",
        "episode_id": episode_id,
        "scene_id": f"scene-{episode_id}",
        "scene_family_id": f"scene-family-{episode_id}",
        "trajectory_family_id": f"trajectory-family-{trajectory_class}",
        "split": split,
        "trajectory_class": trajectory_class,
        "observations": observations,
        "observable_belief": {
            "frame_id": "anchor_camera@view-000",
            "quantization_m": 0.5,
            "entity_count": 3,
        },
        "questions": [
            {
                "fact_id": f"fact-{episode_id}",
                "task_type": "cross_view_relation" if cross_view else "grounding_presence",
                "question_zh": "目标物体在两个视角中的关系是什么？",
                "answer_zh": "它位于参照物左侧。",
                "answer_status": "accepted",
                "family_variant": "claim_true" if cross_view else "canonical",
                "model_view_ids": evidence,
                "evidence_view_ids": evidence,
                "program": {
                    "program_id": "cross_view_relation.v1" if cross_view else "grounding.v1",
                    "semantic_signature": "G->F->R->V" if cross_view else "G->V",
                    "atoms": ["G", "F", "R", "V"] if cross_view else ["G", "V"],
                    "nodes": [
                        {
                            "node_id": "entity",
                            "operation": "G",
                            "input_types": ["view"],
                            "output_type": "entity",
                        },
                        {
                            "node_id": "answer",
                            "operation": "V",
                            "input_types": ["entity"],
                            "output_type": "answer",
                        },
                    ],
                    "answer_node": "answer",
                },
                "certificate": {
                    "result": "pass",
                    "verifier_version": "test/1",
                    "checks": [{"name": "geometry_replay", "passed": True}],
                },
            }
        ],
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    project = tmp_path / "project"
    release = project / "data" / "candidate"
    config = project / "configs" / "candidate.json"
    output = project / "web" / "data" / "candidate_catalog.v1.json"
    details = project / "web" / "data" / "candidate_details"
    _write_json(
        config,
        {
            "schema_version": "epispace.pipeline_config.v1",
            "dataset_id": "candidate-test-v1",
            "composition_holdout": {"program_ids": ["target_view_prediction.v1"]},
        },
    )
    _write_jsonl(
        release / "episodes.ir.jsonl",
        [
            _episode(
                project,
                episode_id="ep-t1",
                trajectory_class="T1",
                split="train",
                cross_view=True,
            ),
            _episode(
                project,
                episode_id="ep-t4",
                trajectory_class="T4",
                split="val",
                cross_view=False,
            ),
            _episode(
                project,
                episode_id="ep-t8",
                trajectory_class="T8",
                split="test",
                cross_view=True,
            ),
        ],
    )
    return project, release, config, output, details


def _add_failed_semantic_audit(release: Path, config: Path) -> None:
    audit = release / "semantic_visual_audit"
    reviews_dir = audit / "reviews"
    reviews_dir.mkdir(parents=True)
    manifest = release / "release_manifest.json"
    inventory = release / "source_inventory.json"
    _write_json(manifest, {"schema_version": "test.manifest.v1", "status": "pass"})
    _write_json(inventory, {"schema_version": "test.inventory.v1", "records": 3})
    episode_ir = release / "episodes.ir.jsonl"
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    packet_id = "semantic-audit-test-packet"
    evidence_sha = "b" * 64
    packet = {
        "schema_version": "epispace.semantic_visual_audit_packet.v1",
        "packet_id": packet_id,
        "items": [
            {"episode_id": "ep-t1", "fact_id": "fact-ep-t1"},
            {"episode_id": "ep-t8", "fact_id": "fact-ep-t8"},
        ],
        "release_binding": {
            "dataset_id": config_payload["dataset_id"],
            "episode_ir": {
                "path": str(episode_ir),
                "sha256": hashlib.sha256(episode_ir.read_bytes()).hexdigest(),
                "records": 3,
            },
            "release_manifest": {
                "path": str(manifest),
                "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            },
            "source_inventory": {
                "path": str(inventory),
                "sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
            },
        },
        "review_evidence_binding": {"sha256": evidence_sha},
        "sampling_contract": {"seed": 20270813},
    }
    packet_path = audit / "packet.json"
    _write_json(packet_path, packet)

    prompt = "Inspect the actual RGB independently and reject unsupported facts."
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    provenance = {
        "reviewer_type": "model_assisted_independent",
        "reviewer_id": "candidate-test-reviewer",
        "reviewer_system": "test-system",
        "reviewer_model": "test-model",
        "review_protocol_id": "epispace.semantic_rgb_review.v1",
        "review_prompt": prompt,
        "review_prompt_sha256": prompt_sha,
        "reviewed_at": "2026-07-17T00:00:00Z",
    }
    major_decision = {
        "index": 0,
        "fact_id": "fact-ep-t1",
        "overall_status": "major_issue",
        "referents_recognizable": False,
        "answer_supported": False,
        "family_intervention_valid": None,
        "severity": "major",
        "reason_codes": ["answer_not_supported_by_rgb"],
        "notes_zh": "测试中的语义证据不足。",
    }
    pass_decision = {
        "index": 1,
        "fact_id": "fact-ep-t8",
        "overall_status": "pass",
        "referents_recognizable": True,
        "answer_supported": True,
        "family_intervention_valid": None,
        "severity": "none",
        "reason_codes": [],
        "notes_zh": "该 Episode 的全部事实均通过本轮抽样复核。",
    }
    review_path = reviews_dir / "reviewer.json"
    _write_json(
        review_path,
        {
            "schema_version": "epispace.semantic_visual_audit_review.v1",
            "packet_id": packet_id,
            "reviewer_provenance": provenance,
            "reviews": [major_decision, pass_decision],
        },
    )
    result_major = {
        **major_decision,
        "audit_index": 1,
        "reviewer_id": provenance["reviewer_id"],
    }
    result_pass = {
        **pass_decision,
        "audit_index": 2,
        "reviewer_id": provenance["reviewer_id"],
    }
    result = {
        "schema_version": "epispace.semantic_visual_audit_result.v1",
        "status": "fail",
        "result_id": "semantic-audit-test-result",
        "packet_binding": {
            "path": str(packet_path),
            "sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
            "bytes": packet_path.stat().st_size,
            "packet_id": packet_id,
            "item_count": 2,
            "review_evidence_sha256": evidence_sha,
        },
        "reviewer_bindings": [
            {
                "path": str(review_path),
                "sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
                "bytes": review_path.stat().st_size,
                "review_count": 2,
                "covered_index_min": 0,
                "covered_index_max": 1,
                "provenance": {
                    key: value for key, value in provenance.items() if key != "review_prompt"
                },
            }
        ],
        "item_decisions": [result_major, result_pass],
        "summary": {
            "overall_status_counts": {
                "pass": 1,
                "minor_issue": 0,
                "major_issue": 1,
                "unreviewable": 0,
            },
            "reviewed_items": 2,
            "reviewer_count": 1,
        },
        "decision": {
            "semantic_visual_audit_gate": False,
            "failed_item_count": 1,
            "minor_item_count": 0,
            "failure_reasons": ["major_unreviewable_or_failed_semantic_check"],
        },
    }
    _write_json(audit / "result.json", result)


def _walk(value: Any):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def test_candidate_projection_is_exhaustive_hash_bound_and_deterministic(
    tmp_path: Path,
) -> None:
    project, release, config, output, details = _fixture(tmp_path)
    first = build_web_candidate_catalog(
        release, config, output, details, project_root=project
    )
    first_bytes = output.read_bytes()
    second = build_web_candidate_catalog(
        release, config, output, details, project_root=project
    )

    assert first == second
    assert output.read_bytes() == first_bytes
    projected_text = output.read_text(encoding="utf-8")
    assert '"status": "verified"' not in projected_text
    assert "VERIFIED RELEASE" not in projected_text
    assert "正式验收" not in projected_text
    assert first["schema_version"] == CATALOG_SCHEMA
    assert first["status"] == "machine_verified_candidate"
    assert first["authority"] == {
        "level": "machine_verified_candidate",
        "semantic_visual_audit": "pending",
        "disposition": "audit_required",
        "formal_release": False,
        "claim_boundary_zh": "仅表示自动编译、证书重放与媒体闭合通过；独立 RGB 语义复核尚未完成。",
    }
    assert first["provenance"]["pipeline_config_sha256"] == hashlib.sha256(
        config.read_bytes()
    ).hexdigest()
    assert first["provenance"]["episode_ir_sha256"] == hashlib.sha256(
        (release / "episodes.ir.jsonl").read_bytes()
    ).hexdigest()
    assert len(first["episodes"]) == first["funnel"]["compiled_episodes"] == 3
    assert {entry["episode_id"] for entry in first["episodes"]} == {
        "ep-t1",
        "ep-t4",
        "ep-t8",
    }
    assert set(first["featured_episode_ids"]) == {"ep-t1", "ep-t4", "ep-t8"}
    assert first["featured_episode_ids"][0] == "ep-t1"

    detail_paths = sorted(details.glob("*.json"))
    assert len(detail_paths) == 3
    for path in detail_paths:
        detail = json.loads(path.read_text(encoding="utf-8"))
        assert detail["schema_version"] == DETAIL_SCHEMA
        assert detail["status"] == "machine_verified_candidate"
        assert detail["authority"]["semantic_visual_audit"] == "pending"
        assert detail["authority"]["disposition"] == "audit_required"
        assert detail["authority"]["formal_release"] is False
        assert detail["candidate_binding"]["episode_ir_sha256"] == first[
            "provenance"
        ]["episode_ir_sha256"]
    for value in _walk(first):
        if isinstance(value, str):
            assert not Path(value).is_absolute()


def test_candidate_generator_removes_stale_output_on_failed_certificate(
    tmp_path: Path,
) -> None:
    project, release, config, output, details = _fixture(tmp_path)
    rows = [json.loads(line) for line in (release / "episodes.ir.jsonl").read_text().splitlines()]
    rows[0]["questions"][0]["certificate"]["checks"][0]["passed"] = False
    _write_jsonl(release / "episodes.ir.jsonl", rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("stale", encoding="utf-8")
    details.mkdir(parents=True)
    (details / "stale.json").write_text("stale", encoding="utf-8")

    with pytest.raises(WebCandidateBuildError, match="failed check"):
        build_web_candidate_catalog(
            release, config, output, details, project_root=project
        )
    assert not output.exists()
    assert not details.exists()


def test_completed_failed_audit_is_hash_bound_and_forces_rework(tmp_path: Path) -> None:
    project, release, config, output, details = _fixture(tmp_path)
    _add_failed_semantic_audit(release, config)
    catalog = build_web_candidate_catalog(
        release, config, output, details, project_root=project
    )

    assert catalog["status"] == "machine_verified_candidate"
    assert catalog["authority"]["semantic_visual_audit"] == "failed"
    assert catalog["authority"]["disposition"] == "rework_required"
    assert catalog["authority"]["formal_release"] is False
    quality = catalog["quality"]["semantic_visual_audit"]
    assert quality["status"] == "failed"
    assert quality["source_result_status"] == "fail"
    assert quality["gate"] is False
    assert quality["failed_item_count"] == 1
    assert quality["counts"] == {
        "pass": 1,
        "minor_issue": 0,
        "major_issue": 1,
        "unreviewable": 0,
    }
    audit = release / "semantic_visual_audit"
    assert quality["hash_binding"]["packet_sha256"] == hashlib.sha256(
        (audit / "packet.json").read_bytes()
    ).hexdigest()
    assert quality["hash_binding"]["result_sha256"] == hashlib.sha256(
        (audit / "result.json").read_bytes()
    ).hexdigest()
    assert quality["hash_binding"]["review_sha256"] == [
        hashlib.sha256((audit / "reviews" / "reviewer.json").read_bytes()).hexdigest()
    ]
    assert {
        item["name"] for item in catalog["artifacts"]
    } >= {"semantic_audit_packet", "semantic_review_1", "semantic_audit_result"}
    reviewed_entry = next(item for item in catalog["episodes"] if item["episode_id"] == "ep-t8")
    assert reviewed_entry["fully_sample_reviewed_pass"] is True
    assert reviewed_entry["sample_reviewed_fact_count"] == 1
    assert reviewed_entry["sample_reviewed_fact_total"] == 1
    assert reviewed_entry["sample_review_status_counts"] == {"pass": 1}
    assert catalog["featured_episode_ids"][0] == "ep-t8"
    assert quality["fully_sample_reviewed_pass_episode_ids"] == ["ep-t8"]
    assert quality["fully_sample_reviewed_pass_episode_count"] == 1
    assert all(
        item["fully_sample_reviewed_pass"] is False
        for item in catalog["episodes"]
        if item["episode_id"] != "ep-t8"
    )
    detail = json.loads(next(details.glob("*.json")).read_text(encoding="utf-8"))
    assert detail["authority"]["semantic_visual_audit"] == "failed"
    assert detail["authority"]["disposition"] == "rework_required"
    assert detail["authority"]["formal_release"] is False
    projected_text = output.read_text(encoding="utf-8")
    assert '"status": "verified"' not in projected_text
    assert "VERIFIED RELEASE" not in projected_text

    with (audit / "reviews" / "reviewer.json").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(WebCandidateBuildError, match=r"review\[0\] SHA mismatch"):
        build_web_candidate_catalog(
            release, config, output, details, project_root=project
        )
    assert not output.exists()
    assert not details.exists()


def test_page_uses_verified_first_candidate_on_404_and_keeps_core_optional() -> None:
    javascript = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "web" / "styles.css").read_text(encoding="utf-8")

    assert 'fetch("data/release_catalog.v1.json")' in javascript
    assert "if (verified.status !== 404)" in javascript
    assert 'fetch("data/candidate_catalog.v1.json")' in javascript
    assert "MACHINE-VERIFIED CANDIDATE" in javascript
    assert "SEMANTIC RGB AUDIT PENDING" in javascript
    assert "REWORK REQUIRED · AUDIT FAILED" in javascript
    assert "FAILED · REWORK REQUIRED" in javascript
    assert "本轮抽样逐事实复核" in javascript
    assert "仍非整库正式 release" in javascript
    assert "renderReleaseUnavailable(releaseError)" in javascript
    assert "await loadReleaseCatalog()" in javascript
    assert "#release\\/(.+)" in javascript
    assert 'id="release-browser-title"' in html
    assert 'id="release-authority-note"' in html
    assert ".release-authority.candidate" in css
    assert ".candidate-mode .semantic-audit-card" in css
