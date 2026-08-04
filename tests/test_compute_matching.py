from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from episode3d.compute_matching import (
    ComputeMatchingError,
    build_compute_matched_schedules,
    estimate_text_tokens,
)
from episode3d.qwen_training import TrainingContractError, load_scheduled_records


def _write_image(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), (20, 40, 60)).save(path)


def _row(
    record_id: str,
    comparison_id: str,
    arm: str,
    facts: list[str],
    images: list[Path],
    question: str,
) -> dict:
    content: list[dict] = []
    for index, image in enumerate(images, 1):
        content.extend(
            [
                {"type": "text", "text": f"<image-{index}>"},
                {"type": "image", "image": str(image)},
            ]
        )
    content.append({"type": "text", "text": question})
    return {
        "record_id": record_id,
        "messages": [
            {"role": "system", "content": "只根据图像回答。"},
            {"role": "user", "content": content},
            {"role": "assistant", "content": "答案。"},
        ],
        "comparison_contract": {
            "comparison_id": comparison_id,
            "arm": arm,
            "fact_ids": facts,
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    image_a = tmp_path / "images" / "a.png"
    image_b = tmp_path / "images" / "b.png"
    image_c = tmp_path / "images" / "c.png"
    _write_image(image_a, 4, 3)
    _write_image(image_b, 2, 5)
    _write_image(image_c, 3, 3)

    episode_path = tmp_path / "episode.jsonl"
    isolated_path = tmp_path / "isolated.jsonl"
    _write_jsonl(
        episode_path,
        [
            _row("ep-ab", "cmp-ab", "episode", ["fact-a", "fact-b"], [image_a, image_b], "甲乙？"),
            _row("ep-c", "cmp-c", "episode", ["fact-c"], [image_c], "丙？"),
        ],
    )
    _write_jsonl(
        isolated_path,
        [
            _row("iso-a", "cmp-ab", "isolated", ["fact-a"], [image_a, image_b], "甲？"),
            _row("iso-b", "cmp-ab", "isolated", ["fact-b"], [image_a, image_b], "乙？"),
            _row("iso-c", "cmp-c", "isolated", ["fact-c"], [image_c], "丙？"),
        ],
    )
    return episode_path, isolated_path


def test_builds_exact_fact_and_image_matched_regimes_reproducibly(tmp_path: Path) -> None:
    episode_path, isolated_path = _fixture(tmp_path)
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    manifest = build_compute_matched_schedules(
        episode_path, isolated_path, output_a, seed=23
    )
    manifest_again = build_compute_matched_schedules(
        episode_path, isolated_path, output_b, seed=23
    )

    assert manifest == manifest_again
    assert manifest["status"] == "pass"
    assert manifest["pairing"]["comparison_count"] == 2
    assert manifest["pairing"]["episode_repetition_by_comparison"] == {
        "cmp-ab": 2,
        "cmp-c": 1,
    }

    fact = manifest["regimes"]["fact_matched"]
    assert fact["status"] == "pass"
    assert fact["observed_invariants"]["actual_fact_multiset_equal"]
    assert not fact["observed_invariants"]["actual_image_occurrence_multiset_equal"]
    assert fact["arms"]["episode"]["schedule_draw_count"] == 2
    assert fact["arms"]["isolated"]["schedule_draw_count"] == 3

    image = manifest["regimes"]["image_occurrence_matched"]
    assert image["status"] == "pass"
    assert image["observed_invariants"]["actual_image_occurrence_multiset_equal"]
    assert image["observed_invariants"]["actual_pixel_exposure_equal"]
    assert image["observed_invariants"]["effective_fact_weight_multiset_equal"]
    assert image["arms"]["episode"]["schedule_draw_count"] == 3
    assert image["arms"]["isolated"]["schedule_draw_count"] == 3
    assert image["arms"]["episode"]["pixel_exposure"] == 53
    assert image["arms"]["isolated"]["pixel_exposure"] == 53
    assert image["arms"]["episode"]["effective_fact_weight_multiset"] == {
        "fact-a": "1",
        "fact-b": "1",
        "fact-c": "1",
    }

    episode_schedule = [
        json.loads(line)
        for line in (output_a / "image_occurrence_matched.episode.schedule.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    repeated = [row for row in episode_schedule if row["comparison_id"] == "cmp-ab"]
    assert len(repeated) == 2
    assert {row["sample_weight_ratio"] for row in repeated} == {"1/2"}
    assert {row["source_jsonl_sha256"] for row in repeated} == {
        hashlib.sha256(episode_path.read_bytes()).hexdigest()
    }

    filenames = [
        "fact_matched.episode.schedule.jsonl",
        "fact_matched.isolated.schedule.jsonl",
        "image_occurrence_matched.episode.schedule.jsonl",
        "image_occurrence_matched.isolated.schedule.jsonl",
        "compute_matching_manifest.json",
    ]
    for filename in filenames:
        assert (output_a / filename).read_bytes() == (output_b / filename).read_bytes()


def test_rejects_per_comparison_fact_mismatch(tmp_path: Path) -> None:
    episode_path, isolated_path = _fixture(tmp_path)
    rows = [json.loads(line) for line in isolated_path.read_text().splitlines()]
    rows[0]["comparison_contract"]["fact_ids"] = ["wrong-fact"]
    _write_jsonl(isolated_path, rows)

    with pytest.raises(ComputeMatchingError, match="different fact multisets"):
        build_compute_matched_schedules(
            episode_path, isolated_path, tmp_path / "output", seed=17
        )


def test_rejects_nonidentical_isolated_visual_context(tmp_path: Path) -> None:
    episode_path, isolated_path = _fixture(tmp_path)
    rows = [json.loads(line) for line in isolated_path.read_text().splitlines()]
    user_content = rows[0]["messages"][1]["content"]
    rows[0]["messages"][1]["content"] = [
        block
        for block in user_content
        if not (block.get("type") == "image" and block["image"].endswith("b.png"))
    ]
    _write_jsonl(isolated_path, rows)

    with pytest.raises(ComputeMatchingError, match="identical visual context"):
        build_compute_matched_schedules(
            episode_path, isolated_path, tmp_path / "output", seed=17
        )


def test_schedule_loader_rejects_source_jsonl_content_mutation(tmp_path: Path) -> None:
    episode_path, isolated_path = _fixture(tmp_path)
    output = tmp_path / "output"
    build_compute_matched_schedules(
        episode_path,
        isolated_path,
        output,
        seed=17,
        regimes=("image_occurrence_matched",),
    )
    schedule = output / "image_occurrence_matched.episode.schedule.jsonl"
    assert len(load_scheduled_records(schedule)) == 3

    # A semantically inert trailing blank line still changes the authoritative
    # bytes and must invalidate every previously frozen schedule row.
    episode_path.write_bytes(episode_path.read_bytes() + b"\n")
    with pytest.raises(TrainingContractError, match="source JSONL hash mismatch"):
        load_scheduled_records(schedule)

    rebuilt = tmp_path / "rebuilt"
    build_compute_matched_schedules(
        episode_path,
        isolated_path,
        rebuilt,
        seed=17,
        regimes=("image_occurrence_matched",),
    )
    rebuilt_schedule = rebuilt / "image_occurrence_matched.episode.schedule.jsonl"
    rebuilt_rows = [
        json.loads(line) for line in rebuilt_schedule.read_text(encoding="utf-8").splitlines()
    ]
    assert {row["source_jsonl_sha256"] for row in rebuilt_rows} == {
        hashlib.sha256(episode_path.read_bytes()).hexdigest()
    }
    assert len(load_scheduled_records(rebuilt_schedule)) == 3


def test_token_proxy_is_transparent_and_deterministic() -> None:
    assert estimate_text_tokens("左边 chair-2，约 1.5m。") == 8
    assert estimate_text_tokens("左边 chair-2，约 1.5m。") == estimate_text_tokens(
        "左边 chair-2，约 1.5m。"
    )
