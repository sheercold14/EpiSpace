#!/usr/bin/env python3
"""Measure exact Qwen tokenizer and visual-token exposure for paired schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from episode3d.qwen_training import (  # noqa: E402
    ScheduledRecord,
    assistant_token_groups,
    load_scheduled_records,
    normalize_qwen_messages,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_schedule(path: Path) -> tuple[bytes, str]:
    """Return the exact schedule bytes consumed by the profiler and their hash."""

    payload = path.resolve().read_bytes()
    return payload, hashlib.sha256(payload).hexdigest()


def require_schedule_unchanged(path: Path, expected_sha256: str, *, arm: str) -> None:
    """Reject a profile whose schedule drifted after its input snapshot."""

    if sha256(path.resolve()) != expected_sha256:
        raise RuntimeError(f"{arm} schedule changed after its token-profile snapshot")


def profile_record(processor: Any, item: ScheduledRecord) -> dict[str, Any]:
    batch = processor.apply_chat_template(
        normalize_qwen_messages(item.record["messages"]),
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = batch["input_ids"][0].tolist()
    groups = assistant_token_groups(processor.tokenizer, input_ids, item.record)
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    image_tokens = sum(token == image_token_id for token in input_ids)
    grid_tokens = int(
        sum(
            int(grid.prod().item())
            // int(processor.image_processor.merge_size) ** 2
            for grid in batch["image_grid_thw"]
        )
    )
    if image_tokens != grid_tokens:
        raise RuntimeError(
            f"image token/grid mismatch for {item.record['record_id']}: "
            f"{image_tokens} != {grid_tokens}"
        )
    answer_positions = {position for _, positions in groups for position in positions}
    return {
        "input_tokens": len(input_ids),
        "image_tokens": image_tokens,
        "text_tokens": len(input_ids) - image_tokens,
        "supervised_answer_tokens": len(answer_positions),
        "fact_count": len(groups),
    }


def aggregate(
    processor: Any, scheduled: list[ScheduledRecord], cache: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    effective_fact_weight: Counter[str] = Counter()
    for item in scheduled:
        record_id = item.record["record_id"]
        if record_id not in cache:
            cache[record_id] = profile_record(processor, item)
        profile = cache[record_id]
        totals.update(profile)
        totals["draws"] += 1
        for fact_id in item.schedule["fact_ids"]:
            effective_fact_weight[fact_id] += item.schedule["sample_weight"]
    return {
        **dict(totals),
        "unique_records": len({item.record["record_id"] for item in scheduled}),
        "effective_facts": len(effective_fact_weight),
        "effective_fact_weight_sum": sum(effective_fact_weight.values()),
        "effective_fact_weight_all_one": all(
            abs(weight - 1.0) < 1e-9 for weight in effective_fact_weight.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episode-schedule", type=Path, required=True)
    parser.add_argument("--isolated-schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-min-pixels", type=int, default=65_536)
    parser.add_argument("--image-max-pixels", type=int, default=262_144)
    args = parser.parse_args()

    from transformers import AutoProcessor
    from transformers import __version__ as transformers_version

    processor = AutoProcessor.from_pretrained(args.model.resolve(), local_files_only=True)
    processor.image_processor.size = {
        "shortest_edge": args.image_min_pixels,
        "longest_edge": args.image_max_pixels,
    }
    episode_schedule = args.episode_schedule.resolve()
    isolated_schedule = args.isolated_schedule.resolve()
    episode_payload, episode_schedule_sha256 = snapshot_schedule(episode_schedule)
    isolated_payload, isolated_schedule_sha256 = snapshot_schedule(isolated_schedule)
    episode = load_scheduled_records(
        episode_schedule, schedule_payload=episode_payload
    )
    isolated = load_scheduled_records(
        isolated_schedule, schedule_payload=isolated_payload
    )
    cache: dict[str, dict[str, Any]] = {}
    episode_profile = aggregate(processor, episode, cache)
    isolated_profile = aggregate(processor, isolated, cache)
    checks = {
        "draws_equal": episode_profile["draws"] == isolated_profile["draws"],
        "image_tokens_equal": (
            episode_profile["image_tokens"] == isolated_profile["image_tokens"]
        ),
        "effective_facts_equal": (
            episode_profile["effective_facts"] == isolated_profile["effective_facts"]
        ),
        "effective_fact_weights_one": (
            episode_profile["effective_fact_weight_all_one"]
            and isolated_profile["effective_fact_weight_all_one"]
        ),
    }
    report = {
        "schema_version": "epispace.qwen_token_profile.v1",
        "status": "pass" if all(checks.values()) else "fail",
        "model": str(args.model.resolve()),
        "transformers_version": transformers_version,
        "image_min_pixels": args.image_min_pixels,
        "image_max_pixels": args.image_max_pixels,
        "sources": {
            "episode_schedule_sha256": episode_schedule_sha256,
            "isolated_schedule_sha256": isolated_schedule_sha256,
        },
        "episode": episode_profile,
        "isolated": isolated_profile,
        "checks": checks,
    }
    if report["status"] != "pass":
        raise RuntimeError(f"token-profile invariant failed: {checks}")
    require_schedule_unchanged(
        episode_schedule, episode_schedule_sha256, arm="episode"
    )
    require_schedule_unchanged(
        isolated_schedule, isolated_schedule_sha256, arm="isolated"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
