from __future__ import annotations

from episode3d.transform_pilot.dataset import (
    _assistant_target,
    _bearing_phrase,
    _evaluation_input,
    _options,
    _prune_unreferenced_media,
)


def test_direction_options_are_deterministic_and_cover_four_labels() -> None:
    first = _options("sample-a", "left")
    second = _options("sample-a", "left")
    assert first == second
    assert {item["label"] for item in first["choices"]} == {
        "front",
        "right",
        "back",
        "left",
    }
    assert first["answer_text"] == "左侧"


def test_bearing_language_keeps_qualitative_visual_cue() -> None:
    assert _bearing_phrase(3.0) == "大致正前方"
    assert _bearing_phrase(-48.0) == "左前方"
    assert _bearing_phrase(177.0) == "大致后方"


def test_paired_targets_differ_only_by_grounded_trace() -> None:
    record = {
        "answer": {"surface": "B. 左侧"},
        "grounded_trace": {
            "cue": {"surface": "第1图看到目标。"},
            "transform": {"surface": "原地右转，反向更新目标方位。"},
        },
    }
    assert _assistant_target(record, "answer_only") == "<answer>B. 左侧</answer>"
    cot = _assistant_target(record, "grounded_cot")
    assert cot.startswith("<cue>")
    assert "<transform>" in cot
    assert cot.endswith("<answer>B. 左侧</answer>")


def test_media_pruner_keeps_only_model_referenced_files(tmp_path) -> None:
    keep = tmp_path / "media" / "family" / "keep.png"
    stale = tmp_path / "media" / "stale" / "stale.png"
    keep.parent.mkdir(parents=True)
    stale.parent.mkdir(parents=True)
    keep.write_bytes(b"keep")
    stale.write_bytes(b"stale")
    records = [{"model_input": {"images": ["media/family/keep.png"]}}]

    assert _prune_unreferenced_media(tmp_path, records) == 1
    assert keep.is_file()
    assert not stale.exists()
    assert not stale.parent.exists()


def test_evaluation_input_preserves_split_for_all_scope_reporting() -> None:
    row = {
        "record_id": "record-a",
        "task_id": "among5_cross_view_relation.v1",
        "scene_id": "scene-a",
        "split": "validation",
        "task_tags": ["cross_view"],
        "variant_id": "identity",
        "counterfactual_group_id": "group-a",
        "query_signature": {"subject": "chair", "reference": "plant"},
        "model_input": {
            "images": ["media/view-000.png"],
            "system": "system",
            "question": "question",
        },
    }

    item = _evaluation_input(row)

    assert item["split"] == "validation"
