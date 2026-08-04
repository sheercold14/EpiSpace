from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from episode3d.pipeline import build_dataset

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "epispace_pilot_v1.json"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_real_pilot_pipeline_is_reproducible_and_passes_corpus_gates(
    tmp_path: Path,
) -> None:
    output = tmp_path / "release"
    latex = tmp_path / "latex"
    manifest = build_dataset(CONFIG, output_dir=output, latex_dir=latex)

    assert manifest["status"] == "pass"
    assert manifest["statistics"]["acquisition"]["planned_jobs"] == 322
    assert manifest["statistics"]["acquisition"]["strict_bundles_loaded"] == 193
    assert manifest["statistics"]["compiled"]["unique_scenes"] == 46
    assert all(manifest["corpus_gates"]["checks"].values())
    assert (latex / "dataset_stats.tex").is_file()
    assert (latex / "dataset_table.tex").is_file()


def test_trajectory_semantics_and_training_control_contract() -> None:
    release = ROOT / "data" / "epispace_pilot_v1"
    ir = _rows(release / "episodes.ir.jsonl")
    episode = _rows(release / "train.episode_sft.jsonl")
    isolated = _rows(release / "train.isolated_sft.jsonl")
    benchmark = _rows(release / "benchmark.jsonl")
    benchmark_family = _rows(release / "benchmark.family.jsonl")
    benchmark_core = _rows(release / "benchmark.core.jsonl")

    assert not any(
        question["task_type"] == "metric_distance"
        for row in ir
        if row["trajectory_class"] == "T3"
        for question in row["questions"]
    )
    assert (
        sum(
            question["task_type"] == "orbit_identity"
            for row in ir
            for question in row["questions"]
        )
        == 8
    )

    consistency_groups: dict[str, set[str]] = defaultdict(set)
    for row in ir:
        for question in row["questions"]:
            if question.get("consistency_group"):
                consistency_groups[question["consistency_group"]].add(
                    question["family_variant"]
                )
    evidence_shape = {"prefix_unknown", "revealed", "decisive_deleted"}
    claim_shape = {"claim_false", "claim_true"}
    frame_shape = {"frame_a", "frame_b"}
    assert sum(variants == evidence_shape for variants in consistency_groups.values()) >= 80
    assert sum(variants == claim_shape for variants in consistency_groups.values()) >= 30
    assert sum(variants == frame_shape for variants in consistency_groups.values()) >= 1
    assert all(
        variants
        in {frozenset(evidence_shape), frozenset(claim_shape), frozenset(frame_shape)}
        for variants in map(frozenset, consistency_groups.values())
    )

    episode_facts = Counter(
        fact
        for row in episode
        for fact in row["comparison_contract"]["fact_ids"]
    )
    isolated_facts = Counter(
        fact
        for row in isolated
        for fact in row["comparison_contract"]["fact_ids"]
    )
    assert episode_facts == isolated_facts
    for row in episode + isolated:
        assert row["comparison_contract"]["surface_format"] == (
            "numbered_answer_list.v1"
        )
        final_text = row["messages"][-2]["content"][-1]["text"]
        assert final_text.startswith("观察阶段到此结束。请依次回答下列问题：\n1. ")
        assert row["messages"][-1]["content"].startswith("1. ")
    assert all(
        row["presentation_contract"] == "numbered_answer_list.v1"
        and row["model_input"][-1]["content"][-1]["text"].startswith(
            "观察阶段到此结束。请依次回答下列问题：\n1. "
        )
        for row in benchmark
    )

    heldout = {
        "counterfactual_cross_view.v1",
        "target_view_prediction.v1",
    }
    assert not any(
        program in heldout
        for row in episode
        for program in row["hidden_meta"]["program_ids"]
    )
    assert heldout <= {row["program"]["program_id"] for row in benchmark}
    assert any(
        "cross_view_register_relation.v1" in row["hidden_meta"]["program_ids"]
        for row in episode
    )

    assert {row["record_id"] for row in benchmark_family} <= {
        row["record_id"] for row in benchmark_core
    }
    assert len({row["consistency_group"] for row in benchmark_family}) >= 35
    grouped_benchmark: dict[str, list[dict]] = defaultdict(list)
    for row in benchmark_family:
        grouped_benchmark[row["consistency_group"]].append(row)
    for rows in grouped_benchmark.values():
        variants = {row["family_variant"] for row in rows}
        assert variants in (evidence_shape, claim_shape, frame_shape)
        assert len({row["family_id"] for row in rows}) == 1
        if variants == evidence_shape:
            questions = {
                row["model_input"][1]["content"][-1]["text"] for row in rows
            }
            image_counts = {
                sum(
                    item.get("type") == "image"
                    for item in row["model_input"][1]["content"]
                )
                for row in rows
            }
            assert len(questions) == 1
            assert len(image_counts) == 1

    train_input = json.dumps(
        [row["messages"] for row in episode + isolated], ensure_ascii=False
    )
    for row in benchmark:
        target_path = row["certificate"].get("oracle_target_rgb")
        if target_path:
            assert target_path not in train_input

    for row in episode + isolated + benchmark:
        messages = row.get("messages", row.get("model_input", []))
        for message in messages:
            if not isinstance(message.get("content"), list):
                continue
            for item in message["content"]:
                if item.get("type") == "image":
                    assert "/preview/" not in item["image"]
                    assert item["image"].endswith(".png")
