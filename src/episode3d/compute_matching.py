"""Reproducible compute-matched schedules for episode-vs-isolated SFT.

The two source arms intentionally differ in packing: one episode record supervises
several facts after observing a scene once, whereas an isolated record supervises
one fact and repeats the visual context.  This module turns those records into two
auditable experimental regimes:

``fact_matched``
    Every source record is drawn once.  The two arms therefore expose the exact
    same fact multiset, while retaining their natural visual-compute difference.

``image_occurrence_matched``
    Each episode record is repeated by the integer visual-exposure ratio of its
    paired isolated records.  Its per-draw loss weight is the reciprocal of that
    ratio, so actual image occurrences match and the gradient summed *within a
    comparison group* has matched effective fact weight.  The weights alone do
    not make per-draw Adam updates equivalent: the causal trainer must accumulate
    every group draw before one optimizer step.

No model-specific tokenizer is required.  The manifest reports a transparent
Unicode-lexeme token proxy and exact decoded pixel exposure separately.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from pathlib import Path
from typing import Any

from PIL import Image

FACT_MATCHED = "fact_matched"
IMAGE_OCCURRENCE_MATCHED = "image_occurrence_matched"
SUPPORTED_REGIMES = (FACT_MATCHED, IMAGE_OCCURRENCE_MATCHED)

_TOKEN_PROXY = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
    r"|[A-Za-z]+(?:[-_][A-Za-z0-9]+)*"
    r"|\d+(?:\.\d+)?"
    r"|[^\s]"
)


class ComputeMatchingError(ValueError):
    """Raised when the paired training arms cannot satisfy the strict contract."""


@dataclass(frozen=True)
class RecordRef:
    """A validated source record and the measurements needed by the scheduler."""

    row: dict[str, Any]
    source_path: Path
    source_sha256: str
    source_line: int
    arm: str
    comparison_id: str
    record_id: str
    fact_ids: tuple[str, ...]
    image_paths: tuple[Path, ...]
    pixel_exposure: int
    prompt_token_proxy: int
    target_token_proxy: int


@dataclass(frozen=True)
class ScheduledDraw:
    """One actual dataloader draw plus its exact rational loss weight."""

    record: RecordRef
    repeat_index: int
    loss_weight: Fraction


def estimate_text_tokens(text: str) -> int:
    """Return a deterministic, model-independent Unicode lexeme token proxy.

    Each CJK character, Latin/alphanumeric run, numeric literal, and remaining
    non-whitespace symbol counts as one proxy token.  This is deliberately not
    presented as an exact Qwen tokenizer count.
    """

    return len(_TOKEN_PROXY.findall(text))


def _stable_json(value: Any, *, indent: int | None = None) -> str:
    separators = None if indent is not None else (",", ":")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
        separators=separators,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ComputeMatchingError("message content must be a string or a list")
    texts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            raise ComputeMatchingError("multimodal message blocks must be objects")
        if block.get("type") == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ComputeMatchingError("text blocks must contain a string 'text'")
            texts.append(text)
    return "\n".join(texts)


def _message_images(message: dict[str, Any], source_path: Path) -> list[Path]:
    content = message.get("content", "")
    if isinstance(content, str):
        return []
    if not isinstance(content, list):
        raise ComputeMatchingError("message content must be a string or a list")
    images: list[Path] = []
    for block in content:
        if not isinstance(block, dict):
            raise ComputeMatchingError("multimodal message blocks must be objects")
        if block.get("type") != "image":
            continue
        raw_path = block.get("image")
        if not isinstance(raw_path, str) or not raw_path:
            raise ComputeMatchingError("image blocks must contain a non-empty path")
        image_path = Path(raw_path).expanduser()
        if not image_path.is_absolute():
            image_path = source_path.parent / image_path
        images.append(image_path.resolve())
    return images


@cache
def _image_size(image_path: Path) -> tuple[int, int]:
    if not image_path.is_file():
        raise ComputeMatchingError(f"image does not exist: {image_path}")
    with Image.open(image_path) as image:
        return image.width, image.height


def _parse_record(
    row: dict[str, Any],
    *,
    source_path: Path,
    source_sha256: str,
    source_line: int,
    expected_arm: str,
) -> RecordRef:
    record_id = row.get("record_id")
    if not isinstance(record_id, str) or not record_id:
        raise ComputeMatchingError(f"{source_path}:{source_line}: missing record_id")
    contract = row.get("comparison_contract")
    if not isinstance(contract, dict):
        raise ComputeMatchingError(
            f"{source_path}:{source_line}: missing comparison_contract"
        )
    arm = contract.get("arm")
    if arm != expected_arm:
        raise ComputeMatchingError(
            f"{source_path}:{source_line}: expected arm {expected_arm!r}, got {arm!r}"
        )
    comparison_id = contract.get("comparison_id")
    if not isinstance(comparison_id, str) or not comparison_id:
        raise ComputeMatchingError(
            f"{source_path}:{source_line}: missing comparison_id"
        )
    raw_facts = contract.get("fact_ids")
    if not isinstance(raw_facts, list) or not raw_facts:
        raise ComputeMatchingError(f"{source_path}:{source_line}: fact_ids must be non-empty")
    if not all(isinstance(fact, str) and fact for fact in raw_facts):
        raise ComputeMatchingError(f"{source_path}:{source_line}: invalid fact_ids")
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ComputeMatchingError(f"{source_path}:{source_line}: messages must be non-empty")

    images: list[Path] = []
    prompt_tokens = 0
    target_tokens = 0
    for message in messages:
        if not isinstance(message, dict):
            raise ComputeMatchingError(f"{source_path}:{source_line}: message must be an object")
        text_tokens = estimate_text_tokens(_message_text(message))
        if message.get("role") == "assistant":
            target_tokens += text_tokens
        else:
            prompt_tokens += text_tokens
        images.extend(_message_images(message, source_path))
    if not images:
        raise ComputeMatchingError(f"{source_path}:{source_line}: no model-input images")

    pixels = sum(width * height for width, height in map(_image_size, images))
    return RecordRef(
        row=row,
        source_path=source_path,
        source_sha256=source_sha256,
        source_line=source_line,
        arm=arm,
        comparison_id=comparison_id,
        record_id=record_id,
        fact_ids=tuple(raw_facts),
        image_paths=tuple(images),
        pixel_exposure=pixels,
        prompt_token_proxy=prompt_tokens,
        target_token_proxy=target_tokens,
    )


def _load_records(path: Path, expected_arm: str) -> list[RecordRef]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ComputeMatchingError(f"source JSONL does not exist: {path}")
    source_bytes = path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ComputeMatchingError(f"source JSONL is not UTF-8: {path}") from error
    records: list[RecordRef] = []
    record_ids: set[str] = set()
    for line_number, line in enumerate(source_text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ComputeMatchingError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(row, dict):
            raise ComputeMatchingError(f"{path}:{line_number}: record must be an object")
        record = _parse_record(
            row,
            source_path=path,
            source_sha256=source_sha256,
            source_line=line_number,
            expected_arm=expected_arm,
        )
        if record.record_id in record_ids:
            raise ComputeMatchingError(f"{path}: duplicate record_id {record.record_id}")
        record_ids.add(record.record_id)
        records.append(record)
    if not records:
        raise ComputeMatchingError(f"source JSONL has no records: {path}")
    return records


def _sum_counters(counters: Iterable[Counter[str]]) -> Counter[str]:
    result: Counter[str] = Counter()
    for counter in counters:
        result.update(counter)
    return result


def _validate_pairs(
    episode_records: Sequence[RecordRef], isolated_records: Sequence[RecordRef]
) -> tuple[dict[str, RecordRef], dict[str, list[RecordRef]], dict[str, int]]:
    episode_by_comparison: dict[str, RecordRef] = {}
    for record in episode_records:
        if record.comparison_id in episode_by_comparison:
            raise ComputeMatchingError(
                f"comparison {record.comparison_id} has multiple episode records"
            )
        episode_by_comparison[record.comparison_id] = record

    isolated_by_comparison: dict[str, list[RecordRef]] = defaultdict(list)
    for record in isolated_records:
        isolated_by_comparison[record.comparison_id].append(record)

    episode_keys = set(episode_by_comparison)
    isolated_keys = set(isolated_by_comparison)
    if episode_keys != isolated_keys:
        missing_isolated = sorted(episode_keys - isolated_keys)
        missing_episode = sorted(isolated_keys - episode_keys)
        raise ComputeMatchingError(
            "comparison IDs differ between arms: "
            f"missing isolated={missing_isolated[:5]}, missing episode={missing_episode[:5]}"
        )

    repetition_by_comparison: dict[str, int] = {}
    for comparison_id in sorted(episode_keys):
        episode = episode_by_comparison[comparison_id]
        isolated = isolated_by_comparison[comparison_id]
        if any(len(record.fact_ids) != 1 for record in isolated):
            raise ComputeMatchingError(
                f"comparison {comparison_id} has an isolated record with != 1 fact"
            )
        episode_facts = Counter(episode.fact_ids)
        isolated_facts = _sum_counters(Counter(record.fact_ids) for record in isolated)
        if episode_facts != isolated_facts:
            raise ComputeMatchingError(
                f"comparison {comparison_id} has different fact multisets between arms"
            )

        episode_images = Counter(map(str, episode.image_paths))
        if any(Counter(map(str, record.image_paths)) != episode_images for record in isolated):
            raise ComputeMatchingError(
                f"comparison {comparison_id} does not reuse identical visual context "
                "in every isolated record"
            )
        isolated_images = _sum_counters(
            Counter(map(str, record.image_paths)) for record in isolated
        )
        if set(episode_images) != set(isolated_images):
            raise ComputeMatchingError(
                f"comparison {comparison_id} has different image identities between arms"
            )
        ratios = {
            Fraction(isolated_images[path], episode_count)
            for path, episode_count in episode_images.items()
        }
        if len(ratios) != 1:
            raise ComputeMatchingError(
                f"comparison {comparison_id} has no scalar image-occurrence ratio"
            )
        ratio = ratios.pop()
        if ratio.denominator != 1 or ratio.numerator < 1:
            raise ComputeMatchingError(
                f"comparison {comparison_id} requires non-integer episode repetition {ratio}"
            )
        repetition_by_comparison[comparison_id] = ratio.numerator

    return episode_by_comparison, isolated_by_comparison, repetition_by_comparison


def _hash_rank(draw: ScheduledDraw, *, regime: str, seed: int) -> tuple[str, str, int]:
    payload = (
        f"epispace.compute_schedule.v1\0{seed}\0{regime}\0{draw.record.arm}\0"
        f"{draw.record.record_id}\0{draw.repeat_index}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), draw.record.record_id, draw.repeat_index


def _make_schedules(
    episode_records: Sequence[RecordRef],
    isolated_records: Sequence[RecordRef],
    repetition_by_comparison: dict[str, int],
    *,
    regime: str,
    seed: int,
) -> dict[str, list[ScheduledDraw]]:
    if regime == FACT_MATCHED:
        episode_draws = [ScheduledDraw(record, 0, Fraction(1)) for record in episode_records]
    elif regime == IMAGE_OCCURRENCE_MATCHED:
        episode_draws = []
        for record in episode_records:
            repetitions = repetition_by_comparison[record.comparison_id]
            episode_draws.extend(
                ScheduledDraw(record, repeat_index, Fraction(1, repetitions))
                for repeat_index in range(repetitions)
            )
    else:
        raise ComputeMatchingError(f"unsupported regime: {regime}")
    isolated_draws = [ScheduledDraw(record, 0, Fraction(1)) for record in isolated_records]
    return {
        "episode": sorted(
            episode_draws,
            key=lambda draw: _hash_rank(draw, regime=regime, seed=seed),
        ),
        "isolated": sorted(
            isolated_draws,
            key=lambda draw: _hash_rank(draw, regime=regime, seed=seed),
        ),
    }


def _draw_json(
    draw: ScheduledDraw, schedule_index: int, *, schedule_dir: Path
) -> dict[str, Any]:
    record = draw.record
    return {
        "schedule_index": schedule_index,
        "record_id": record.record_id,
        "source_jsonl": os.path.relpath(record.source_path, start=schedule_dir),
        "source_jsonl_sha256": record.source_sha256,
        "source_line": record.source_line,
        "arm": record.arm,
        "comparison_id": record.comparison_id,
        "repeat_index": draw.repeat_index,
        "sample_weight": float(draw.loss_weight),
        "sample_weight_ratio": str(draw.loss_weight),
        "fact_ids": list(record.fact_ids),
        "image_paths": list(map(str, record.image_paths)),
        "exposure_per_draw": {
            "image_occurrences": len(record.image_paths),
            "pixels": record.pixel_exposure,
            "estimated_prompt_tokens": record.prompt_token_proxy,
            "estimated_target_tokens": record.target_token_proxy,
            "estimated_text_tokens": record.prompt_token_proxy + record.target_token_proxy,
        },
    }


def _fraction_json(value: Fraction) -> dict[str, int | float | str]:
    return {
        "exact": str(value),
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": float(value),
    }


def _arm_summary(draws: Sequence[ScheduledDraw]) -> tuple[dict[str, Any], dict[str, Any]]:
    actual_facts: Counter[str] = Counter()
    effective_facts: defaultdict[str, Fraction] = defaultdict(Fraction)
    actual_images: Counter[str] = Counter()
    effective_images: defaultdict[str, Fraction] = defaultdict(Fraction)
    actual_pixels = 0
    effective_pixels = Fraction(0)
    actual_prompt_tokens = 0
    effective_prompt_tokens = Fraction(0)
    actual_target_tokens = 0
    effective_target_tokens = Fraction(0)

    for draw in draws:
        record = draw.record
        actual_facts.update(record.fact_ids)
        for fact_id in record.fact_ids:
            effective_facts[fact_id] += draw.loss_weight
        actual_images.update(map(str, record.image_paths))
        for image_path in record.image_paths:
            effective_images[str(image_path)] += draw.loss_weight
        actual_pixels += record.pixel_exposure
        effective_pixels += record.pixel_exposure * draw.loss_weight
        actual_prompt_tokens += record.prompt_token_proxy
        effective_prompt_tokens += record.prompt_token_proxy * draw.loss_weight
        actual_target_tokens += record.target_token_proxy
        effective_target_tokens += record.target_token_proxy * draw.loss_weight

    unique_records = {draw.record.record_id for draw in draws}
    summary = {
        "source_record_count": len(unique_records),
        "schedule_draw_count": len(draws),
        "fact_occurrence_count": sum(actual_facts.values()),
        "fact_multiset": dict(sorted(actual_facts.items())),
        "effective_fact_weight_multiset": {
            key: str(value) for key, value in sorted(effective_facts.items())
        },
        "image_occurrence_count": sum(actual_images.values()),
        "unique_image_count": len(actual_images),
        "image_occurrence_multiset": dict(sorted(actual_images.items())),
        "effective_image_weight_multiset": {
            key: str(value) for key, value in sorted(effective_images.items())
        },
        "pixel_exposure": actual_pixels,
        "effective_weighted_pixel_exposure": _fraction_json(effective_pixels),
        "estimated_token_exposure": {
            "prompt": actual_prompt_tokens,
            "target": actual_target_tokens,
            "total": actual_prompt_tokens + actual_target_tokens,
        },
        "effective_weighted_estimated_token_exposure": {
            "prompt": _fraction_json(effective_prompt_tokens),
            "target": _fraction_json(effective_target_tokens),
            "total": _fraction_json(effective_prompt_tokens + effective_target_tokens),
        },
    }
    internal = {
        "actual_facts": actual_facts,
        "effective_facts": Counter(effective_facts),
        "actual_images": actual_images,
        "effective_images": Counter(effective_images),
        "actual_pixels": actual_pixels,
    }
    return summary, internal


def _regime_manifest(
    regime: str,
    schedules: dict[str, list[ScheduledDraw]],
) -> tuple[dict[str, Any], bool]:
    summaries: dict[str, Any] = {}
    internals: dict[str, Any] = {}
    for arm in ("episode", "isolated"):
        summaries[arm], internals[arm] = _arm_summary(schedules[arm])

    observed = {
        "actual_fact_multiset_equal": (
            internals["episode"]["actual_facts"] == internals["isolated"]["actual_facts"]
        ),
        "effective_fact_weight_multiset_equal": (
            internals["episode"]["effective_facts"]
            == internals["isolated"]["effective_facts"]
        ),
        "actual_image_occurrence_multiset_equal": (
            internals["episode"]["actual_images"]
            == internals["isolated"]["actual_images"]
        ),
        "actual_pixel_exposure_equal": (
            internals["episode"]["actual_pixels"]
            == internals["isolated"]["actual_pixels"]
        ),
    }
    if regime == FACT_MATCHED:
        required = {
            "actual_fact_multiset_equal": True,
            "effective_fact_weight_multiset_equal": True,
        }
    else:
        required = {
            "effective_fact_weight_multiset_equal": True,
            "actual_image_occurrence_multiset_equal": True,
            "actual_pixel_exposure_equal": True,
        }
    passed = all(observed[key] == value for key, value in required.items())
    return {
        "status": "pass" if passed else "fail",
        "required_invariants": required,
        "observed_invariants": observed,
        "arms": summaries,
    }, passed


def build_compute_matched_schedules(
    episode_path: Path | str,
    isolated_path: Path | str,
    output_dir: Path | str,
    *,
    seed: int = 17,
    regimes: Sequence[str] = SUPPORTED_REGIMES,
) -> dict[str, Any]:
    """Build deterministic schedules and return their audit manifest.

    The function is strict by design: it rejects unpaired comparison groups,
    different per-group fact multisets, different image identities, or a visual
    exposure ratio that cannot be represented by integer episode repetitions.
    """

    selected_regimes = tuple(dict.fromkeys(regimes))
    if not selected_regimes:
        raise ComputeMatchingError("at least one regime must be selected")
    unsupported = sorted(set(selected_regimes) - set(SUPPORTED_REGIMES))
    if unsupported:
        raise ComputeMatchingError(f"unsupported regimes: {unsupported}")

    episode_source = Path(episode_path).expanduser().resolve()
    isolated_source = Path(isolated_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    episode_records = _load_records(episode_source, "episode")
    isolated_records = _load_records(isolated_source, "isolated")
    episode_by_comparison, isolated_by_comparison, repetitions = _validate_pairs(
        episode_records, isolated_records
    )

    manifest: dict[str, Any] = {
        "schema_version": "epispace.compute_matching.v1",
        "status": "pass",
        "seed": seed,
        "schedule_order": "sha256(seed, regime, arm, record_id, repeat_index)",
        "token_estimator": {
            "name": "unicode_lexeme_proxy_v1",
            "definition": (
                "One token per CJK character, Latin/alphanumeric run, numeric literal, "
                "or remaining non-whitespace symbol; excludes image and chat-template tokens."
            ),
            "is_exact_model_tokenizer": False,
        },
        "pixel_estimator": {
            "name": "image_header_width_times_height",
            "definition": "Sum of input-image width*height for every scheduled image occurrence.",
            "is_exact_for_input_pixels": True,
        },
        "sources": {
            "episode": {
                "path": os.path.relpath(episode_source, start=output),
                "sha256": episode_records[0].source_sha256,
                "record_count": len(episode_records),
            },
            "isolated": {
                "path": os.path.relpath(isolated_source, start=output),
                "sha256": isolated_records[0].source_sha256,
                "record_count": len(isolated_records),
            },
        },
        "pairing": {
            "comparison_count": len(episode_by_comparison),
            "fact_multiset_checked_per_comparison": True,
            "identical_visual_context_checked_per_isolated_record": True,
            "episode_repetition_by_comparison": dict(sorted(repetitions.items())),
            "isolated_record_count_by_comparison": {
                key: len(value) for key, value in sorted(isolated_by_comparison.items())
            },
        },
        "regimes": {},
    }

    output.mkdir(parents=True, exist_ok=True)
    all_passed = True
    for regime in selected_regimes:
        schedules = _make_schedules(
            episode_records,
            isolated_records,
            repetitions,
            regime=regime,
            seed=seed,
        )
        regime_manifest, passed = _regime_manifest(regime, schedules)
        all_passed &= passed
        schedule_files: dict[str, Any] = {}
        for arm in ("episode", "isolated"):
            filename = f"{regime}.{arm}.schedule.jsonl"
            destination = output / filename
            lines = [
                _stable_json(_draw_json(draw, index, schedule_dir=output))
                for index, draw in enumerate(schedules[arm])
            ]
            _atomic_write(destination, "\n".join(lines) + "\n")
            schedule_files[arm] = {
                "filename": filename,
                "sha256": _sha256(destination),
            }
        regime_manifest["schedule_files"] = schedule_files
        manifest["regimes"][regime] = regime_manifest

    for source_path, expected_sha256 in (
        (episode_source, episode_records[0].source_sha256),
        (isolated_source, isolated_records[0].source_sha256),
    ):
        if _sha256(source_path) != expected_sha256:
            raise ComputeMatchingError(
                f"source JSONL changed while schedules were being built: {source_path}"
            )
    manifest["status"] = "pass" if all_passed else "fail"
    _atomic_write(
        output / "compute_matching_manifest.json",
        _stable_json(manifest, indent=2) + "\n",
    )
    if not all_passed:
        raise ComputeMatchingError("one or more requested matching regimes failed")
    return manifest


__all__ = [
    "FACT_MATCHED",
    "IMAGE_OCCURRENCE_MATCHED",
    "SUPPORTED_REGIMES",
    "ComputeMatchingError",
    "build_compute_matched_schedules",
    "estimate_text_tokens",
]
