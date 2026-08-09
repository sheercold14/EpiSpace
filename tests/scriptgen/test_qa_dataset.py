from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from test_family import PLAN, fake_bundle_view

from spatial_episode.contracts.schema import CONTRACTS
from spatial_episode.scriptgen.family import (
    QuestionGroupEntry,
    QuestionGroupTrajectory,
    ScriptgenQuestionGroupV1,
    build_family_doc,
)
from spatial_episode.scriptgen.library import SELF_MOTION
from spatial_episode.scriptgen.qa_dataset import (
    FamilySource,
    QADatasetError,
    SourceHasher,
    _input_hash,
    _raw_record,
    capability_tier,
    format_question,
    verify_source_snapshot,
)
from spatial_episode.scriptgen.qa_evaluation import answer_format_compliant, parse_answer_label
from spatial_episode.scriptgen.qa_reporting import _raw_case, _stream_case
from spatial_episode.scriptgen.standards import STD_V1


def _source(tmp_path: Path) -> FamilySource:
    family = build_family_doc(fake_bundle_view(), PLAN, SELF_MOTION, STD_V1, seed=3)
    family_dir = tmp_path / "groups" / "fake" / SELF_MOTION.capability
    family_dir.mkdir(parents=True)
    family_path = family_dir / "family.json"
    family_path.write_text(family.model_dump_json(indent=1), encoding="utf-8")
    media = family_dir / "media"
    media.mkdir()
    for frame in family.frames:
        Image.new("RGB", (8, 8), (frame.frame, 0, 0)).save(
            media / f"view-{frame.frame:03d}.rgb.png"
        )
    group_path = family_dir.parent / "group.json"
    group = ScriptgenQuestionGroupV1(
        question_group_id=family.question_group_id,
        standard_version=STD_V1.standard_version,
        trajectory=QuestionGroupTrajectory(
            bundle=str(tmp_path / "bundle"),
            plan_record=str(tmp_path / "plan.json"),
            scene_ir=str(tmp_path / "scene_ir.json"),
            plan_id=PLAN["plan_id"],
            scene_id=PLAN["scene_id"],
        ),
        questions=(
            QuestionGroupEntry(
                capability=SELF_MOTION.capability,
                role="primary",
                family_id=family.family_id,
                label=family.episodes[0].label,
                family=f"{SELF_MOTION.capability}/family.json",
                skip_reason=None,
            ),
        ),
    )
    group_path.write_text(group.model_dump_json(indent=1), encoding="utf-8")
    return FamilySource(group_path, group, family_path, family, PLAN)


def test_question_surface_is_explicit_without_changing_family_text() -> None:
    text = format_question("目标在哪个方向?", ("front", "left", "无法判断"), streaming=True)
    assert text.startswith("截至当前已收到的图像，目标在哪个方向?")
    assert "front（前方）" in text
    assert "<answer>标签</answer>" in text


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("<answer>right</answer>", "right"),
        ("<answer>右侧</answer>", "right"),
        ("<answer>不可见</answer>", None),
        ("right", "right"),
        ("答案是 right。", "right"),
        ("可能是 left，也可能是 right", None),
        ("<answer>无法判断</answer>", "无法判断"),
    ],
)
def test_answer_parser_requires_one_declared_label(response: str, expected: str | None) -> None:
    assert parse_answer_label(response, ("left", "right", "无法判断")) == expected


def test_answer_parser_accepts_semantic_entity_for_first_second() -> None:
    question = "dishwasher和fridge中,哪一个离microwave更近?"
    assert (
        parse_answer_label(
            "<answer>fridge</answer>",
            ("first", "second", "无法判断"),
            question=question,
        )
        == "second"
    )
    assert (
        parse_answer_label("<answer>不可见</answer>", ("visible", "not_visible", "无法判断"))
        == "not_visible"
    )
    assert not answer_format_compliant(
        "<answer>不可见</answer>", ("visible", "not_visible", "无法判断")
    )
    assert answer_format_compliant(
        "<answer>not_visible</answer>", ("visible", "not_visible", "无法判断")
    )


def test_raw_record_keeps_oracle_out_of_eval_input(tmp_path: Path) -> None:
    source = _source(tmp_path)
    canonical = next(episode for episode in source.family.episodes if episode.kind == "canonical")
    record, eval_input, oracle = _raw_record(
        tmp_path,
        source,
        canonical,
        SourceHasher(tmp_path),
    )
    assert record["answer"]["label"] == canonical.label
    assert record["messages"][-1]["content"] == f"<answer>{canonical.label}</answer>"
    assert len(record["images"]) == len(canonical.frame_sequence)
    assert all(image["path"].endswith(".rgb.png") for image in record["images"])
    assert "answer" not in eval_input
    assert "certificate" not in eval_input
    assert oracle["answer"]["label"] == canonical.label
    assert oracle["certificate"]["answer"]["label"] == canonical.label


def test_streaming_input_hash_is_independent_of_oracle_fields() -> None:
    payload = {
        "record_id": "stream-1",
        "input_sha256": "",
        "batch_id": "streaming-0001",
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "<answer>left</answer>"},
        ],
        "turns": [
            {
                "turn_id": "t1",
                "question": "question",
                "answer": "left",
                "status": "answerable",
                "certificate_sha256": "certificate-a",
            }
        ],
        "assistant_span_contract": [{"answer_sha256": "answer-a"}],
        "reward_spec": {"target": "left"},
    }
    changed_gold = {
        **payload,
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "<answer>right</answer>"},
        ],
        "turns": [
            {
                **payload["turns"][0],
                "answer": "right",
                "status": "abstain",
                "certificate_sha256": "certificate-b",
            }
        ],
        "assistant_span_contract": [{"answer_sha256": "answer-b"}],
        "reward_spec": {"target": "right"},
    }
    assert _input_hash(payload) == _input_hash(changed_gold)


def test_capability_tiers_and_bed_exclusion() -> None:
    assert capability_tier("path_integration") == "P1"
    assert capability_tier("reference_frame_transform") == "P2"
    assert capability_tier("cross_view_pair_relation_k1") == "P3"
    with pytest.raises(QADatasetError, match="out of scope"):
        capability_tier("existence_sufficiency_bed")


def test_qa_training_schemas_are_public_contracts() -> None:
    assert (
        CONTRACTS["scriptgen_raw_qa.v1.schema.json"].model_json_schema()["properties"][
            "schema_version"
        ]["const"]
        == "scriptgen.raw_qa.v1"
    )
    assert (
        CONTRACTS["scriptgen_streaming_qa.v1.schema.json"].model_json_schema()["properties"][
            "schema_version"
        ]["const"]
        == "scriptgen.streaming_qa.v1"
    )


def test_source_snapshot_detects_drift(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    item = source / "value.json"
    item.write_text("{}", encoding="utf-8")
    hasher = SourceHasher(source)
    hasher.hash(item)
    snapshot = tmp_path / "snapshot.jsonl"
    row = hasher.rows()[0]
    snapshot.write_text(__import__("json").dumps(row) + "\n", encoding="utf-8")
    assert verify_source_snapshot(source_root=source, snapshot_path=snapshot)["status"] == "pass"
    item.write_text('{"changed":true}', encoding="utf-8")
    assert verify_source_snapshot(source_root=source, snapshot_path=snapshot)["status"] == "fail"


def test_model_errors_do_not_relabel_compiler_gold() -> None:
    assert _raw_case(
        {
            "multimodal": {"correct": False, "parseable": True},
            "vision_free": {"correct": False, "parseable": True},
        }
    ) == ("valid_hard", "compiler_valid_but_frozen_model_wrong")
    assert (
        _stream_case(
            {
                condition: {
                    "turns": [
                        {"correct": condition != "multimodal_free_running", "parseable": True}
                    ]
                }
                for condition in (
                    "multimodal_teacher_forced",
                    "multimodal_free_running",
                    "vision_free_teacher_forced",
                    "vision_free_free_running",
                )
            }
        )[0]
        == "shortcut_risk"
    )


def test_evaluator_runs_all_raw_and_streaming_conditions_resumably(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from spatial_episode.scriptgen import qa_evaluation

    dataset = tmp_path / "dataset"
    source = tmp_path / "source"
    dataset.mkdir()
    source.mkdir()
    (dataset / "manifest.json").write_text(
        __import__("json").dumps({"source_root": str(source)}), encoding="utf-8"
    )
    raw_input = {
        "record_id": "raw-1",
        "input_sha256": "a",
        "batch_id": "raw-0001",
        "tier": "P1",
        "capability": "path_integration",
        "role": "primary",
        "variant": "canonical",
        "cluster_ids": {"scene": "s", "trajectory": "p", "family": "f"},
        "images": [],
        "system": "system",
        "question": "question",
        "choices": ["left", "right", "无法判断"],
    }
    raw_oracle = {
        "record_id": "raw-1",
        "answer": {"label": "right", "choices": raw_input["choices"]},
    }
    stream_input = {
        "record_id": "stream-1",
        "input_sha256": "b",
        "batch_id": "streaming-0001",
        "tier": "P2",
        "stream_kind": "evidence_reveal",
        "capabilities": ["reference_frame_transform"],
        "cluster_ids": {"scene": "s", "trajectory": "p", "stream": "g"},
        "system": "system",
        "turns": [
            {
                "turn_id": "t1",
                "capability": "reference_frame_transform",
                "prefix_length": 1,
                "new_images": [],
                "question": "q1",
                "choices": ["front", "无法判断"],
            },
            {
                "turn_id": "t2",
                "capability": "reference_frame_transform",
                "prefix_length": 2,
                "new_images": [],
                "question": "q2",
                "choices": ["front", "无法判断"],
            },
        ],
    }
    stream_oracle = {
        "record_id": "stream-1",
        "turns": [
            {"turn_id": "t1", "answer": "无法判断"},
            {"turn_id": "t2", "answer": "front"},
        ],
    }

    def write_jsonl(name: str, rows: list[dict]) -> None:
        (dataset / name).write_text(
            "".join(__import__("json").dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    write_jsonl("raw_eval_inputs.jsonl", [raw_input])
    write_jsonl("raw_eval_oracle.jsonl", [raw_oracle])
    write_jsonl("streaming_eval_inputs.jsonl", [stream_input])
    write_jsonl("streaming_eval_oracle.jsonl", [stream_oracle])
    write_jsonl(
        "sensenova_predictions.jsonl",
        [{"record_id": "stale-record", "condition": "multimodal"}],
    )

    class FakeEvaluator:
        load_seconds = 0.1

        def __init__(self, model_path: Path) -> None:
            self.model_path = model_path

        def raw(self, item: dict, source_root: Path, *, vision: bool) -> str:
            return "<answer>right</answer>" if vision else "<answer>left</answer>"

        def streaming(
            self,
            item: dict,
            oracle: dict,
            source_root: Path,
            *,
            vision: bool,
            teacher_forced: bool,
        ) -> list[dict]:
            return [
                {
                    "turn_id": turn["turn_id"],
                    "capability": turn["capability"],
                    "prefix_length": turn["prefix_length"],
                    "ground_truth": gold["answer"],
                    "prediction": gold["answer"] if vision else None,
                    "correct": vision,
                    "parseable": vision,
                    "response": "fake",
                    "elapsed_seconds": 0.0,
                    "cumulative_image_count": 0,
                }
                for turn, gold in zip(item["turns"], oracle["turns"], strict=True)
            ]

    monkeypatch.setattr(qa_evaluation, "SenseNovaEvaluator", FakeEvaluator)
    summary = qa_evaluation.evaluate_qa_dataset(dataset_root=dataset, model_path=tmp_path / "model")
    assert summary["prediction_record_count"] == 6
    assert summary["raw"]["conditions"]["multimodal"]["accuracy"] == 1.0
    assert summary["raw"]["conditions"]["vision_free"]["accuracy"] == 0.0
    assert (
        summary["streaming"]["conditions"]["multimodal_teacher_forced"]["all_turns_correct_rate"]
        == 1.0
    )
    assert len(qa_evaluation.read_jsonl(dataset / "sensenova_predictions.jsonl")) == 6
