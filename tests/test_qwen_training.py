from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scripts.train_qwen3vl_lora import (
    load_scheduled_records_snapshot,
    record_group_reduction,
    require_snapshot_unchanged,
)

from episode3d.qwen_training import (
    ScheduledRecord,
    TrainingContractError,
    assistant_token_groups,
    build_comparison_groups,
    load_scheduled_records,
    normalize_qwen_messages,
    ordered_comparison_ids,
    supervised_answer_segments,
    supervised_assistant_turns,
    validate_paired_comparison_groups,
)

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "data" / "epispace_pilot_v1"


class CharacterTokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, object]:
        assert not add_special_tokens
        assert return_offsets_mapping
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def test_incremental_dialogue_supervises_every_assistant_turn_and_eos() -> None:
    answers = ["先看到冰箱。", "最后一次是第3个视角。"]
    messages = [
        {"role": "system", "content": "按序观察。"},
        {"role": "user", "content": "图1？"},
        {"role": "assistant", "content": answers[0]},
        {"role": "user", "content": "图2？"},
        {"role": "assistant", "content": answers[1]},
    ]
    spans = [
        {
            "message_index": 2,
            "turn_id": "r1",
            "answer_sha256": hashlib.sha256(answers[0].encode()).hexdigest(),
            "supervise_eos": True,
        },
        {
            "message_index": 4,
            "turn_id": "r2",
            "answer_sha256": hashlib.sha256(answers[1].encode()).hexdigest(),
            "supervise_eos": True,
        },
    ]
    record = {
        "messages": messages,
        "assistant_span_contract": spans,
        "loss_policy": {
            "train_on": "all_assistant_turns",
            "normalization": "mean_tokens_per_turn_then_mean_turns",
        },
    }
    assert [value.turn_id for value in supervised_assistant_turns(record)] == ["r1", "r2"]
    eos = 1
    full_ids = [99, *map(ord, answers[0]), eos, 98, *map(ord, answers[1]), eos, 97]
    groups = assistant_token_groups(CharacterTokenizer(), full_ids, record)
    assert [group[0] for group in groups] == ["r1", "r2"]
    assert full_ids[groups[0][1][-1]] == eos
    assert full_ids[groups[1][1][-1]] == eos
    assert record_group_reduction(record) == "mean"
    assert record_group_reduction({"loss_policy": {}}) == "sum"


def test_training_schedule_snapshots_drive_rows_and_reject_own_and_paired_drift(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    image = tmp_path / "view.png"
    image.write_bytes(b"model-input-placeholder")
    source_row = {
        "record_id": "record-1",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.name},
                    {"type": "text", "text": "问题"},
                ],
            },
            {"role": "assistant", "content": "答案"},
        ],
        "comparison_contract": {
            "comparison_id": "comparison-1",
            "arm": "episode",
            "fact_ids": ["fact-1"],
        },
    }
    source.write_text(json.dumps(source_row) + "\n", encoding="utf-8")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    schedule_row = {
        "record_id": "record-1",
        "source_jsonl": source.name,
        "source_jsonl_sha256": source_sha256,
        "source_line": 1,
        "arm": "episode",
        "comparison_id": "comparison-1",
        "repeat_index": 0,
        "sample_weight": 1.0,
        "fact_ids": ["fact-1"],
        "image_paths": [str(image.resolve())],
    }
    own = tmp_path / "image_matched.episode.schedule.jsonl"
    paired = tmp_path / "image_matched.isolated.schedule.jsonl"
    for path in (own, paired):
        path.write_text(json.dumps(schedule_row) + "\n", encoding="utf-8")

    own_rows, own_sha256 = load_scheduled_records_snapshot(own)
    paired_rows, paired_sha256 = load_scheduled_records_snapshot(paired)
    assert own_rows[0].record == source_row
    assert paired_rows[0].record == source_row

    replacement = {**schedule_row, "record_id": "record-replaced"}
    for path in (own, paired):
        path.write_text(json.dumps(replacement) + "\n", encoding="utf-8")
    with pytest.raises(TrainingContractError, match="training schedule changed"):
        require_snapshot_unchanged(own, own_sha256, label="training schedule")
    with pytest.raises(TrainingContractError, match="paired training schedule changed"):
        require_snapshot_unchanged(
            paired,
            paired_sha256,
            label="paired training schedule",
        )


def _first_row(name: str) -> dict[str, object]:
    return json.loads((RELEASE / name).read_text(encoding="utf-8").splitlines()[0])


def test_normalize_qwen_messages_wraps_string_content() -> None:
    row = _first_row("train.episode_sft.jsonl")
    normalized = normalize_qwen_messages(row["messages"])
    assert normalized[0]["content"] == [{"type": "text", "text": row["messages"][0]["content"]}]
    assert isinstance(normalized[1]["content"], list)


def test_every_release_record_has_one_segment_per_fact() -> None:
    for name in ("train.episode_sft.jsonl", "train.isolated_sft.jsonl"):
        for line in (RELEASE / name).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            segments = supervised_answer_segments(row)
            assert [segment.fact_id for segment in segments] == row["comparison_contract"][
                "fact_ids"
            ]
            assert all(segment.text for segment in segments)


def test_token_groups_exclude_numbering_and_preserve_fact_order() -> None:
    record = {
        "messages": [
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "1. 左侧。\n2. 无法确定。"},
        ],
        "comparison_contract": {"fact_ids": ["fact-a", "fact-b"]},
    }
    assistant = record["messages"][-1]["content"]
    full_ids = [1, 2, 3, *[ord(character) for character in assistant], 4]
    groups = assistant_token_groups(CharacterTokenizer(), full_ids, record)
    assert [fact_id for fact_id, _ in groups] == ["fact-a", "fact-b"]
    recovered = [
        "".join(chr(full_ids[position]) for position in positions) for _, positions in groups
    ]
    assert recovered == ["左侧。", "无法确定。"]


def test_single_fact_numbered_surface_excludes_numbering() -> None:
    record = {
        "messages": [{"role": "assistant", "content": "1. 左侧。"}],
        "comparison_contract": {"fact_ids": ["fact-a"]},
    }
    assistant = record["messages"][-1]["content"]
    full_ids = [7, *[ord(character) for character in assistant], 8]
    groups = assistant_token_groups(CharacterTokenizer(), full_ids, record)
    recovered = "".join(chr(full_ids[position]) for position in groups[0][1])
    assert recovered == "左侧。"


def test_malformed_numbering_fails_closed() -> None:
    record = {
        "messages": [{"role": "assistant", "content": "1. 是。\n3. 否。"}],
        "comparison_contract": {"fact_ids": ["a", "b"]},
    }
    with pytest.raises(TrainingContractError, match="canonically numbered"):
        supervised_answer_segments(record)


def test_image_matched_schedules_resolve_all_source_rows() -> None:
    schedule_root = RELEASE / "compute_matching"
    for arm in ("episode", "isolated"):
        rows = load_scheduled_records(
            schedule_root / f"image_occurrence_matched.{arm}.schedule.jsonl"
        )
        assert len(rows) == 586
        assert sum(len(item.schedule["image_paths"]) for item in rows) == 2753


def test_comparison_groups_are_exact_optimizer_units() -> None:
    schedule_root = RELEASE / "compute_matching"
    episode = build_comparison_groups(
        load_scheduled_records(schedule_root / "image_occurrence_matched.episode.schedule.jsonl")
    )
    isolated = build_comparison_groups(
        load_scheduled_records(schedule_root / "image_occurrence_matched.isolated.schedule.jsonl")
    )
    assert validate_paired_comparison_groups(episode, isolated) == {
        "comparison_groups": 335,
        "draws_per_arm": 586,
        "facts": 586,
    }
    assert all(group.draw_count == len(group.fact_ids) for group in episode.values())
    assert all(group.draw_count == len(group.fact_ids) for group in isolated.values())


def test_seeded_group_order_is_identical_across_arms() -> None:
    schedule_root = RELEASE / "compute_matching"
    episode = build_comparison_groups(
        load_scheduled_records(schedule_root / "image_occurrence_matched.episode.schedule.jsonl")
    )
    isolated = build_comparison_groups(
        load_scheduled_records(schedule_root / "image_occurrence_matched.isolated.schedule.jsonl")
    )
    first = ordered_comparison_ids(episode, seed=17, epoch=0)
    assert first == ordered_comparison_ids(isolated, seed=17, epoch=0)
    assert first != ordered_comparison_ids(episode, seed=17, epoch=1)


def test_episode_group_with_per_draw_unit_weight_fails_closed() -> None:
    schedule_path = RELEASE / "compute_matching" / "image_occurrence_matched.episode.schedule.jsonl"
    rows = load_scheduled_records(schedule_path)
    target = next(item for item in rows if len(item.schedule["fact_ids"]) > 1)
    broken_schedule = dict(target.schedule)
    broken_schedule["sample_weight"] = 1.0
    broken = [
        ScheduledRecord(schedule=broken_schedule, record=target.record) if item is target else item
        for item in rows
    ]
    with pytest.raises(TrainingContractError, match="weight is not 1/k"):
        build_comparison_groups(broken)


def test_schedule_image_accounting_is_replayed_from_source_messages(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-decoded-by-loader")
    source = tmp_path / "source.jsonl"
    record = {
        "record_id": "record-a",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.name},
                    {"type": "text", "text": "问题"},
                ],
            },
            {"role": "assistant", "content": "1. 左侧。"},
        ],
        "comparison_contract": {"fact_ids": ["fact-a"]},
    }
    source.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    schedule = tmp_path / "schedule.jsonl"
    row = {
        "record_id": "record-a",
        "fact_ids": ["fact-a"],
        "source_jsonl": source.name,
        "source_jsonl_sha256": source_sha256,
        "source_line": 1,
        "sample_weight": 1.0,
        "image_paths": [str(image.resolve())],
    }
    schedule.write_text(json.dumps(row) + "\n", encoding="utf-8")
    loaded = load_scheduled_records(schedule)
    assert loaded[0].model_image_paths == (str(image.resolve()),)
    training_loaded, _ = load_scheduled_records_snapshot(schedule)
    assert training_loaded[0].model_image_paths == (str(image.resolve()),)

    row["image_paths"] = [str((tmp_path / "different.png").resolve())]
    schedule.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(TrainingContractError, match="image_paths disagree with source messages"):
        load_scheduled_records(schedule)
    with pytest.raises(TrainingContractError, match="image_paths disagree with source messages"):
        load_scheduled_records_snapshot(schedule)
