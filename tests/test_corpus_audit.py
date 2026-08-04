from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from episode3d.corpus_audit import audit_release, main, render_markdown


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _message(image_paths: list[Path], question: str, answer: str) -> list[dict]:
    content: list[dict] = []
    for path in image_paths:
        content.append({"type": "image", "image": str(path)})
    content.append({"type": "text", "text": question})
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": answer},
    ]


def _training_record(
    *,
    record_id: str,
    arm: str,
    comparison_id: str,
    facts: list[str],
    image_paths: list[Path],
) -> dict:
    return {
        "record_id": record_id,
        "family_id": "family-train",
        "scene_id": "scene-train",
        "split": "train",
        "messages": _message(image_paths, "物体在哪里？", "在左侧。"),
        "comparison_contract": {
            "comparison_id": comparison_id,
            "arm": arm,
            "fact_ids": facts,
            "unique_image_count": len(set(image_paths)),
            "question_count": len(facts),
            "supervision_matched": True,
            "compute_matching_requires_sampler": True,
        },
    }


def _question(
    fact_id: str,
    *,
    task: str,
    value: object,
    status: str = "accepted",
    group: str | None = None,
    variant: str = "canonical",
    views: list[str] | None = None,
    question: str = "仅根据这些视角，能否确定场景里存在包装箱？",
) -> dict:
    return {
        "fact_id": fact_id,
        "task_type": task,
        "question_zh": question,
        "answer_zh": "能确定。" if status == "accepted" else "无法确定。",
        "rationale_zh": "只依据画面作答。",
        "answer_value": value,
        "answer_status": status,
        "family_variant": variant,
        "consistency_group": group,
        "model_view_ids": views or ["view-000", "view-001"],
    }


def _make_release(tmp_path: Path) -> tuple[Path, list[Path]]:
    release = tmp_path / "release"
    release.mkdir()
    images = [tmp_path / f"rgb-{index}.png" for index in range(4)]
    for index, path in enumerate(images):
        yy, xx = np.indices((12, 16), dtype=np.uint16)
        textured = np.stack(
            (
                (xx * 17 + index * 7) % 256,
                (yy * 23 + index * 11) % 256,
                ((xx + yy) * 19 + index * 13) % 256,
            ),
            axis=2,
        ).astype(np.uint8)
        Image.fromarray(textured, mode="RGB").save(path)

    train_question = _question(
        "fact-train", task="grounding_presence", value=True, question="能看到椅子吗？"
    )
    evidence_questions = [
        _question(
            "fact-prefix",
            task="cross_view_unknown",
            value=None,
            status="unknown",
            group="evidence-family",
            variant="prefix_unknown",
            views=["view-000", "view-001"],
        ),
        _question(
            "fact-reveal",
            task="cross_view_relation",
            value="left_of",
            group="evidence-family",
            variant="revealed",
            views=["view-000", "view-002"],
        ),
        _question(
            "fact-deleted",
            task="cross_view_unknown",
            value=None,
            status="unknown",
            group="evidence-family",
            variant="decisive_deleted",
            views=["view-000", "view-003"],
        ),
    ]
    frame_questions = [
        _question(
            "fact-frame-a",
            task="egocentric_relation",
            value="left_of",
            group="frame-family",
            variant="frame_a",
            views=["view-000", "view-001"],
            question="以第1张图的相机朝向为正前方，椅子位于桌子的哪个方向？",
        ),
        _question(
            "fact-frame-b",
            task="egocentric_relation",
            value="behind",
            group="frame-family",
            variant="frame_b",
            views=["view-000", "view-001"],
            question="以第2张图的相机朝向为正前方，椅子位于桌子的哪个方向？",
        ),
    ]
    for index, question in enumerate(frame_questions):
        variant = question["family_variant"]
        question["program"] = {
            "program_id": "egocentric_relation.v1",
            "semantic_signature": "G(view,a)+G(view,b)->F_ego(camera)->R->V",
        }
        question["certificate"] = {
            "checks": [
                {
                    "name": "same_two_images_across_frame_siblings",
                    "passed": True,
                    "model_view_ids": ["view-000", "view-001"],
                },
                {
                    "name": "camera_frame_relation_recomputed",
                    "passed": True,
                    "anchor_view_id": f"view-{index:03d}",
                    "relation": question["answer_value"],
                    "yaw_separation_deg": 60.0,
                },
            ],
            "variant": variant,
        }
    episodes = [
        {
            "episode_id": "episode-train",
            "scene_id": "scene-train",
            "scene_family_id": "scene-family-train",
            "trajectory_family_id": "trajectory-family-train",
            "split": "train",
            "observations": [
                {"view_id": f"view-{index:03d}", "rgb": str(path)}
                for index, path in enumerate(images)
            ],
            "questions": [train_question],
        },
        {
            "episode_id": "episode-val",
            "scene_id": "scene-val",
            "scene_family_id": "scene-family-val",
            "trajectory_family_id": "trajectory-family-val",
            "split": "val",
            "observations": [
                {"view_id": f"view-{index:03d}", "rgb": str(path)}
                for index, path in enumerate(images)
            ],
            "questions": evidence_questions + frame_questions,
        },
    ]
    _write_jsonl(release / "episodes.ir.jsonl", episodes)

    episode_record = _training_record(
        record_id="episode-arm",
        arm="episode",
        comparison_id="comparison-1",
        facts=["fact-a", "fact-b"],
        image_paths=images[:2],
    )
    isolated_records = [
        _training_record(
            record_id=f"isolated-{suffix}",
            arm="isolated",
            comparison_id="comparison-1",
            facts=[fact],
            image_paths=images[:2],
        )
        for suffix, fact in (("a", "fact-a"), ("b", "fact-b"))
    ]
    _write_jsonl(release / "train.episode_sft.jsonl", [episode_record])
    _write_jsonl(release / "train.isolated_sft.jsonl", isolated_records)
    _write_jsonl(release / "train.state_aux_sft.jsonl", [])
    _write_jsonl(release / "train.rlvr.jsonl", [])
    _write_jsonl(release / "benchmark.jsonl", [])
    _write_jsonl(release / "benchmark.family.jsonl", [])
    _write_jsonl(release / "benchmark.composition.jsonl", [])
    _write_jsonl(release / "benchmark.core.jsonl", [])
    return release, images


def test_audit_measures_targets_families_images_and_training_arms(tmp_path: Path) -> None:
    release, _images = _make_release(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            {"fact_id": "fact-prefix", "correct": True},
            {"fact_id": "fact-reveal", "correct": True},
            {"fact_id": "fact-deleted", "correct": True},
        ],
    )

    report = audit_release(release, predictions_path=predictions)

    target = report["sections"]["targets_and_baselines"]
    grounding = next(
        group
        for group in target["groups"]
        if group["split"] == "train" and group["task_type"] == "grounding_presence"
    )
    assert grounding["within_split_majority"]["accuracy"] == 1.0
    assert grounding["field_distributions"]["value"] == [
        {"target": "true", "count": 1, "fraction": 1.0}
    ]

    families = report["sections"]["families"]
    assert families["complete_contract_counts"] == {
        "evidence_intervention_triple": 1,
        "frame_reference_pair": 1,
    }
    replacement = families["shared_view_single_replacement"]
    assert replacement["pass_count"] == 1
    assert replacement["fail_count"] == 0
    assert replacement["all_passed"] is True
    assert replacement["ordered_slot_stability"]["all_passed"] is True
    frame_transform = families["frame_same_input_answer_transform"]
    assert frame_transform["evaluated_family_count"] == 1
    assert frame_transform["pass_count"] == 1
    assert frame_transform["fail_count"] == 0
    assert frame_transform["all_passed"] is True
    sensitivity = families["accuracy_conditioned_evidence_sensitivity"]
    assert sensitivity["structurally_evaluable_family_count"] == 1
    assert sensitivity["metric_status"] == "evaluated"
    assert sensitivity["score"] == 1.0
    assert sensitivity["family_joint_accuracy"] == 1.0

    image_report = report["sections"]["images"]
    assert image_report["status"] == "pass"
    assert image_report["dimensions"] == {"16x12": 4}
    assert image_report["modes"] == {"RGB": 4}
    rgb_gate = report["sections"]["model_input_rgb_quality"]
    assert rgb_gate["status"] == "pass"
    assert rgb_gate["policy"] == {
        "maximum_rgb_dominant_color_fraction": 0.6,
        "minimum_rgb_quantized_entropy_bits": 2.75,
        "source": "frozen_default",
        "policy_id": "epispace.corpus_audit.frozen_rgb_gate.v1",
        "fallback_reason": "release_manifest.json is missing",
    }
    assert rgb_gate["model_input_reference_count"] == 6
    assert rgb_gate["model_input_unique_path_count"] == 2
    assert rgb_gate["hard_degenerate_occurrence_count"] == 0

    comparison = report["sections"]["training_arm_comparison"]
    assert comparison["status"] == "pass"
    assert comparison["global_fact_multiset_equal"] is True
    assert comparison["image_exposure"]["episode_reference_count"] == 2
    assert comparison["image_exposure"]["isolated_reference_count"] == 4
    assert comparison["image_exposure"]["isolated_to_episode_pixel_ratio"] == 2.0
    assert comparison["image_exposure"]["compute_match_status"] == "not_matched_in_raw_exports"

    assert report["sections"]["ontology_leakage"]["status"] == "pass"
    assert "Structured targets" in render_markdown(report)


def test_state_aux_uniform_model_input_fails_rgb_degeneration_gate(
    tmp_path: Path,
) -> None:
    release, _images = _make_release(tmp_path)
    bad_image = tmp_path / "state-aux-uniform.png"
    Image.new("RGB", (16, 12), (32, 32, 32)).save(bad_image)
    state_aux = {
        "record_id": "state-aux-bad-rgb",
        "family_id": "family-state-aux",
        "scene_id": "scene-state-aux",
        "split": "train",
        "messages": _message([bad_image], "记住这个空间。", "已记录。"),
    }
    _write_jsonl(release / "train.state_aux_sft.jsonl", [state_aux])

    report = audit_release(release)

    rgb_gate = report["sections"]["model_input_rgb_quality"]
    assert rgb_gate["status"] == "fail"
    assert rgb_gate["hard_degenerate_occurrence_count"] == 1
    assert rgb_gate["hard_degenerate_unique_path_count"] == 1
    assert rgb_gate["by_artifact"]["train.state_aux_sft.jsonl"] == {
        "reference_count": 1,
        "unique_path_count": 1,
        "hard_degenerate_occurrence_count": 1,
        "hard_degenerate_unique_path_count": 1,
    }
    sample = rgb_gate["hard_degenerate_samples"][0]
    assert sample["path"] == str(bad_image)
    assert sample["dominant_quantized_color_fraction"] == 1.0
    assert sample["quantized_color_entropy_bits"] == 0.0
    assert report["sections"]["images"]["status"] == "pass"
    assert report["summary"]["status"] == "fail"
    assert "Actual model-input RGB gate: **fail**" in render_markdown(report)


def test_ir_only_uniform_rgb_is_excluded_from_model_input_gate(tmp_path: Path) -> None:
    release, _images = _make_release(tmp_path)
    ir_only_bad = tmp_path / "ir-only-uniform.png"
    Image.new("RGB", (16, 12), (48, 48, 48)).save(ir_only_bad)
    episodes = [
        json.loads(line)
        for line in (release / "episodes.ir.jsonl").read_text().splitlines()
    ]
    episodes[0]["observations"].append(
        {"view_id": "view-ir-only-bad", "rgb": str(ir_only_bad)}
    )
    _write_jsonl(release / "episodes.ir.jsonl", episodes)

    report = audit_release(release)

    rgb_gate = report["sections"]["model_input_rgb_quality"]
    assert rgb_gate["status"] == "pass"
    assert rgb_gate["hard_degenerate_occurrence_count"] == 0
    assert rgb_gate["hard_degenerate_unique_path_count"] == 0
    assert str(ir_only_bad) not in {
        sample["path"] for sample in rgb_gate["hard_degenerate_samples"]
    }
    assert report["sections"]["images"]["status"] == "pass"
    assert not any(
        "hard-degeneration gate" in failure
        for failure in report["summary"]["failures"]
    )


def test_audit_explicitly_reports_missing_fields_and_leaks(tmp_path: Path) -> None:
    release, images = _make_release(tmp_path)
    episodes = [json.loads(line) for line in (release / "episodes.ir.jsonl").read_text().splitlines()]
    del episodes[0]["questions"][0]["answer_value"]
    episodes[0]["questions"][0]["question_zh"] = "Can you see the raw bed category?"
    episodes[0]["observations"][0]["rgb"] = str(tmp_path / "missing.png")
    frame_b = next(
        question
        for question in episodes[1]["questions"]
        if question.get("family_variant") == "frame_b"
    )
    frame_b["model_view_ids"] = ["view-001", "view-000"]
    _write_jsonl(release / "episodes.ir.jsonl", episodes)

    report = audit_release(release)

    target = report["sections"]["targets_and_baselines"]
    assert target["status"] == "partial"
    assert target["missing_fields"] == {"question.answer_value": 1}
    ontology = report["sections"]["ontology_leakage"]
    assert ontology["status"] == "fail"
    assert ontology["model_visible_hit_counts_by_token"]["bed"] >= 1
    image_report = report["sections"]["images"]
    assert image_report["status"] == "fail"
    assert image_report["missing_path_count"] == 1
    assert str(images[0]) not in image_report["missing_path_samples"]
    frame_transform = report["sections"]["families"]["frame_same_input_answer_transform"]
    assert frame_transform["fail_count"] == 1
    assert frame_transform["all_passed"] is False


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    release, _images = _make_release(tmp_path)
    json_output = tmp_path / "audit.json"
    markdown_output = tmp_path / "audit.md"

    assert (
        main(
            [
                str(release),
                "--json",
                str(json_output),
                "--markdown",
                str(markdown_output),
            ]
        )
        == 0
    )
    assert json.loads(json_output.read_text())["schema_version"] == "epispace.corpus_audit.v1"
    assert markdown_output.read_text().startswith("# EpiSpace corpus audit")


def test_cli_returns_nonzero_for_failed_audit(tmp_path: Path) -> None:
    release, _images = _make_release(tmp_path)
    episodes = [
        json.loads(line)
        for line in (release / "episodes.ir.jsonl").read_text().splitlines()
    ]
    episodes[0]["questions"][0]["question_zh"] = "Can you see the raw bed category?"
    _write_jsonl(release / "episodes.ir.jsonl", episodes)

    assert main([str(release)]) == 1
