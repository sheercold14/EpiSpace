"""Model-facing helpers for controlled Qwen-VL SFT experiments.

The release JSONL deliberately stays framework agnostic.  This module bridges
that contract to a Qwen-style chat processor without importing PyTorch or
Transformers, so the data invariants remain testable in the lightweight core
environment.

There are two independent units in the causal training contract:

* the *loss* unit is one fact.  ``assistant_token_groups`` returns one token
  group per ``fact_id``; a trainer averages CE within each fact, sums facts,
  and applies the draw's ``sample_weight``;
* the *optimizer* unit is one ``comparison_id``.  Image-matched episode rows
  repeat the same multi-fact record ``k`` times with weight ``1/k``, while the
  isolated arm has ``k`` one-fact rows with weight one.  All ``k`` draws must
  accumulate gradients before a single optimizer step.  Scaling each episode
  draw and stepping Adam ``k`` times is not equivalent because Adam is nearly
  scale invariant and changes its state on every step.

``build_comparison_groups`` and ``validate_paired_comparison_groups`` keep the
optimizer contract testable without importing PyTorch or Transformers.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class TrainingContractError(ValueError):
    """Raised when a training row or schedule violates the paired contract."""


class OffsetTokenizer(Protocol):
    """Minimal fast-tokenizer interface used by ``assistant_token_groups``."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class AnswerSegment:
    """One fact's supervised character span inside the assistant message."""

    fact_id: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class AssistantTurnSegment:
    """One supervised assistant message in an incremental dialogue."""

    turn_id: str
    message_index: int
    text: str
    supervise_eos: bool


@dataclass(frozen=True)
class ScheduledRecord:
    """A schedule draw bound to its authoritative source SFT record."""

    schedule: dict[str, Any]
    record: dict[str, Any]
    model_image_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComparisonGroup:
    """Validated optimizer unit for one arm and one ``comparison_id``."""

    comparison_id: str
    arm: str
    fact_ids: tuple[str, ...]
    items: tuple[ScheduledRecord, ...]

    @property
    def draw_count(self) -> int:
        return len(self.items)


_NUMBERED_ANSWER = re.compile(r"^(?P<number>[1-9][0-9]*)\.\s+(?P<answer>.+)$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: Path) -> str:
    """Return the content digest used to bind a schedule to its source SFT."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_qwen_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize string chat content to Qwen's list-of-content-items schema."""

    normalized: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role:
            raise TrainingContractError(f"message {index} has no valid role")
        if isinstance(content, str):
            normalized_content: Any = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            normalized_content = content
        else:
            raise TrainingContractError(f"message {index} has unsupported content")
        normalized.append({"role": role, "content": normalized_content})
    return normalized


def supervised_answer_segments(record: Mapping[str, Any]) -> list[AnswerSegment]:
    """Recover the exact per-fact answer spans from an exported SFT record."""

    messages = record.get("messages")
    contract = record.get("comparison_contract")
    if not isinstance(messages, list) or not messages:
        raise TrainingContractError("training record has no messages")
    if not isinstance(contract, Mapping):
        raise TrainingContractError("training record has no comparison_contract")
    fact_ids = contract.get("fact_ids")
    if (
        not isinstance(fact_ids, list)
        or not fact_ids
        or not all(isinstance(item, str) and item for item in fact_ids)
    ):
        raise TrainingContractError("comparison_contract.fact_ids is invalid")
    assistant = messages[-1]
    if not isinstance(assistant, Mapping) or assistant.get("role") != "assistant":
        raise TrainingContractError("last message must be the assistant target")
    content = assistant.get("content")
    if not isinstance(content, str) or not content:
        raise TrainingContractError("assistant target must be non-empty text")

    if len(fact_ids) == 1:
        # Accept legacy bare answers and the format-matched canonical ``1.``
        # surface.  When numbered, supervise only the answer text just as for
        # multi-fact episode turns; formatting tokens are never a treatment.
        visible_content = content.rstrip("\r\n")
        match = _NUMBERED_ANSWER.fullmatch(visible_content)
        if match is not None:
            if int(match.group("number")) != 1:
                raise TrainingContractError(
                    "single-fact assistant target must use canonical number 1"
                )
            return [
                AnswerSegment(
                    fact_ids[0],
                    match.group("answer"),
                    match.start("answer"),
                    match.end("answer"),
                )
            ]
        return [AnswerSegment(fact_ids[0], content, 0, len(content))]

    lines = content.splitlines(keepends=True)
    if len(lines) != len(fact_ids):
        raise TrainingContractError(f"assistant has {len(lines)} lines for {len(fact_ids)} facts")
    segments: list[AnswerSegment] = []
    cursor = 0
    for expected_number, (fact_id, line) in enumerate(zip(fact_ids, lines, strict=True), 1):
        visible_line = line.rstrip("\r\n")
        match = _NUMBERED_ANSWER.fullmatch(visible_line)
        if match is None or int(match.group("number")) != expected_number:
            raise TrainingContractError(
                f"assistant line {expected_number} is not canonically numbered"
            )
        answer = match.group("answer")
        start = cursor + match.start("answer")
        end = cursor + match.end("answer")
        segments.append(AnswerSegment(fact_id, answer, start, end))
        cursor += len(line)
    return segments


def supervised_assistant_turns(
    record: Mapping[str, Any],
) -> list[AssistantTurnSegment]:
    """Validate and recover every supervised assistant turn in message order."""

    messages = record.get("messages")
    raw_contract = record.get("assistant_span_contract")
    loss_policy = record.get("loss_policy")
    if not isinstance(messages, list) or not messages:
        raise TrainingContractError("incremental record has no messages")
    if not isinstance(raw_contract, list) or not raw_contract:
        raise TrainingContractError("incremental record has no assistant_span_contract")
    if not isinstance(loss_policy, Mapping) or loss_policy.get("train_on") != "all_assistant_turns":
        raise TrainingContractError("incremental record must train on all assistant turns")

    expected_indices = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, Mapping) and message.get("role") == "assistant"
    ]
    contract_indices: list[int] = []
    result: list[AssistantTurnSegment] = []
    seen_turns: set[str] = set()
    for index, raw in enumerate(raw_contract):
        if not isinstance(raw, Mapping):
            raise TrainingContractError(f"assistant span {index} is not an object")
        message_index = raw.get("message_index")
        turn_id = raw.get("turn_id")
        expected_sha256 = raw.get("answer_sha256")
        supervise_eos = raw.get("supervise_eos")
        if not isinstance(message_index, int) or not isinstance(turn_id, str) or not turn_id:
            raise TrainingContractError(f"assistant span {index} has invalid identity")
        if turn_id in seen_turns:
            raise TrainingContractError(f"duplicate assistant turn_id {turn_id}")
        if message_index < 0 or message_index >= len(messages):
            raise TrainingContractError(f"assistant span {index} message_index is out of range")
        message = messages[message_index]
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            raise TrainingContractError(f"message {message_index} is not an assistant turn")
        if not isinstance(content, str) or not content:
            raise TrainingContractError(f"assistant message {message_index} is not text")
        observed_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if expected_sha256 != observed_sha256:
            raise TrainingContractError(f"assistant span {turn_id} answer_sha256 mismatch")
        if supervise_eos is not True:
            raise TrainingContractError(f"assistant span {turn_id} must supervise EOS")
        contract_indices.append(message_index)
        seen_turns.add(turn_id)
        result.append(AssistantTurnSegment(turn_id, message_index, content, True))
    if contract_indices != expected_indices:
        raise TrainingContractError(
            "assistant_span_contract must cover every assistant message exactly once in order"
        )
    return result


def _last_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    if not needle:
        raise TrainingContractError("assistant token sequence is empty")
    for start in range(len(haystack) - len(needle), -1, -1):
        if list(haystack[start : start + len(needle)]) == list(needle):
            return start
    raise TrainingContractError("assistant token sequence is absent from processor input")


def _next_subsequence(haystack: Sequence[int], needle: Sequence[int], *, start: int) -> int:
    if not needle:
        raise TrainingContractError("assistant token sequence is empty")
    for index in range(start, len(haystack) - len(needle) + 1):
        if list(haystack[index : index + len(needle)]) == list(needle):
            return index
    raise TrainingContractError("ordered assistant token sequence is absent from processor input")


def incremental_assistant_token_groups(
    tokenizer: OffsetTokenizer,
    input_ids: Sequence[int],
    record: Mapping[str, Any],
) -> list[tuple[str, tuple[int, ...]]]:
    """Map every assistant turn to content-token positions plus its EOS token."""

    groups: list[tuple[str, tuple[int, ...]]] = []
    cursor = 0
    for segment in supervised_assistant_turns(record):
        encoded = tokenizer(
            segment.text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        assistant_ids = encoded.get("input_ids")
        if not isinstance(assistant_ids, list) or not assistant_ids:
            raise TrainingContractError("assistant content tokenization is empty")
        start = _next_subsequence(input_ids, assistant_ids, start=cursor)
        positions = list(range(start, start + len(assistant_ids)))
        end_position = start + len(assistant_ids)
        if end_position >= len(input_ids):
            raise TrainingContractError(
                f"assistant turn {segment.turn_id} has no following EOS token"
            )
        positions.append(end_position)
        groups.append((segment.turn_id, tuple(positions)))
        cursor = end_position + 1
    return groups


def assistant_token_groups(
    tokenizer: OffsetTokenizer,
    input_ids: Sequence[int],
    record: Mapping[str, Any],
) -> list[tuple[str, tuple[int, ...]]]:
    """Map each supervised fact to absolute input-token positions.

    ``input_ids`` must be the processor output for the complete conversation.
    The assistant text is tokenized as one block so newline-merging BPE tokens
    exactly match the suffix embedded in the multimodal sequence.
    """

    if record.get("assistant_span_contract") is not None:
        return incremental_assistant_token_groups(tokenizer, input_ids, record)

    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise TrainingContractError("training record has no messages")
    assistant_text = messages[-1].get("content")
    if not isinstance(assistant_text, str):
        raise TrainingContractError("assistant target must be text")
    encoded = tokenizer(
        assistant_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    assistant_ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")
    if not isinstance(assistant_ids, list) or not isinstance(offsets, list):
        raise TrainingContractError("a fast tokenizer with offsets is required")
    if len(assistant_ids) != len(offsets):
        raise TrainingContractError("token IDs and offsets have different lengths")
    assistant_start = _last_subsequence(input_ids, assistant_ids)

    result: list[tuple[str, tuple[int, ...]]] = []
    for segment in supervised_answer_segments(record):
        relative_positions = [
            index
            for index, offset in enumerate(offsets)
            if isinstance(offset, Sequence)
            and len(offset) == 2
            and int(offset[1]) > segment.start
            and int(offset[0]) < segment.end
        ]
        if not relative_positions:
            raise TrainingContractError(f"fact {segment.fact_id} maps to zero tokens")
        result.append(
            (
                segment.fact_id,
                tuple(assistant_start + position for position in relative_positions),
            )
        )
    return result


def _read_jsonl_payload(path: Path, payload: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TrainingContractError(f"{path}: input is not UTF-8") from exc
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainingContractError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise TrainingContractError(f"{path}:{line_number}: row is not an object")
        rows.append(row)
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse JSONL from one byte snapshot."""

    return _read_jsonl_payload(path, path.read_bytes())


def record_model_image_paths(record: Mapping[str, Any], *, source_dir: Path) -> tuple[str, ...]:
    """Recover the ordered RGB inputs from the authoritative chat record.

    Schedule ``image_paths`` are accounting metadata, not an independent source
    of truth.  Training therefore derives the exposure sequence from the exact
    message blocks that the processor will consume and normalizes every path
    against the source JSONL directory.
    """

    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise TrainingContractError("training record has no messages")
    images: list[str] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TrainingContractError(f"message {message_index} is not an object")
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block_index, block in enumerate(content):
            if not isinstance(block, Mapping):
                raise TrainingContractError(
                    f"message {message_index} block {block_index} is not an object"
                )
            if block.get("type") != "image":
                continue
            image = block.get("image")
            if not isinstance(image, str) or not image:
                raise TrainingContractError(
                    f"message {message_index} block {block_index} has no image path"
                )
            path = Path(image).expanduser()
            if not path.is_absolute():
                path = source_dir / path
            images.append(str(path.resolve()))
    if not images:
        raise TrainingContractError("training record contains no model image inputs")
    return tuple(images)


def normalized_schedule_image_paths(
    schedule: Mapping[str, Any], *, schedule_dir: Path
) -> tuple[str, ...]:
    """Normalize the schedule's claimed exposure sequence for exact replay."""

    raw_paths = schedule.get("image_paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise TrainingContractError("schedule row has no image_paths")
    result: list[str] = []
    for index, raw_path in enumerate(raw_paths):
        if not isinstance(raw_path, str) or not raw_path:
            raise TrainingContractError(f"schedule image_paths[{index}] is not a non-empty string")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = schedule_dir / path
        result.append(str(path.resolve()))
    return tuple(result)


def load_scheduled_records(
    schedule_path: Path, *, schedule_payload: bytes | None = None
) -> list[ScheduledRecord]:
    """Resolve and validate every draw against a content-bound source JSONL."""

    schedule_path = schedule_path.resolve()
    if schedule_payload is None:
        schedule_payload = schedule_path.read_bytes()
    schedules = _read_jsonl_payload(schedule_path, schedule_payload)
    source_cache: dict[Path, list[dict[str, Any]]] = {}
    source_sha256_cache: dict[Path, str] = {}
    result: list[ScheduledRecord] = []
    for index, schedule in enumerate(schedules):
        source_name = schedule.get("source_jsonl")
        expected_source_sha256 = schedule.get("source_jsonl_sha256")
        source_line = schedule.get("source_line")
        if not isinstance(source_name, str) or not isinstance(source_line, int):
            raise TrainingContractError(f"schedule row {index} has no source pointer")
        if not isinstance(expected_source_sha256, str) or not _SHA256.fullmatch(
            expected_source_sha256
        ):
            raise TrainingContractError(f"schedule row {index} has no valid source_jsonl_sha256")
        source = Path(source_name)
        if not source.is_absolute():
            source = (schedule_path.parent / source).resolve()
        if source not in source_cache:
            source_payload = source.read_bytes()
            source_cache[source] = _read_jsonl_payload(source, source_payload)
            source_sha256_cache[source] = hashlib.sha256(source_payload).hexdigest()
        if source_sha256_cache[source] != expected_source_sha256:
            raise TrainingContractError(
                f"schedule row {index} source JSONL hash mismatch for {source}"
            )
        if source_line < 1 or source_line > len(source_cache[source]):
            raise TrainingContractError(f"schedule row {index} source_line is out of range")
        record = source_cache[source][source_line - 1]
        if record.get("record_id") != schedule.get("record_id"):
            raise TrainingContractError(f"schedule row {index} record_id mismatch")
        contract = record.get("comparison_contract")
        fact_ids = contract.get("fact_ids") if isinstance(contract, Mapping) else None
        if fact_ids != schedule.get("fact_ids"):
            raise TrainingContractError(f"schedule row {index} fact_ids mismatch")
        model_image_paths = record_model_image_paths(record, source_dir=source.parent)
        claimed_image_paths = normalized_schedule_image_paths(
            schedule, schedule_dir=schedule_path.parent
        )
        if model_image_paths != claimed_image_paths:
            raise TrainingContractError(
                f"schedule row {index} image_paths disagree with source messages"
            )
        weight = schedule.get("sample_weight")
        if not isinstance(weight, int | float) or isinstance(weight, bool) or weight <= 0:
            raise TrainingContractError(f"schedule row {index} has invalid sample_weight")
        result.append(
            ScheduledRecord(
                schedule=schedule,
                record=record,
                model_image_paths=model_image_paths,
            )
        )
    if not result:
        raise TrainingContractError("schedule is empty")
    return result


def _item_sort_key(item: ScheduledRecord) -> tuple[int, str, str]:
    repeat_index = item.schedule.get("repeat_index")
    if not isinstance(repeat_index, int) or repeat_index < 0:
        raise TrainingContractError("schedule row has invalid repeat_index")
    record_id = item.record.get("record_id")
    if not isinstance(record_id, str) or not record_id:
        raise TrainingContractError("scheduled record has invalid record_id")
    fact_ids = item.schedule.get("fact_ids")
    first_fact = fact_ids[0] if isinstance(fact_ids, list) and fact_ids else ""
    return repeat_index, first_fact, record_id


def build_comparison_groups(
    scheduled: Sequence[ScheduledRecord],
) -> dict[str, ComparisonGroup]:
    """Group one schedule arm and enforce its optimizer-unit invariants.

    Episode groups must consist of ``k`` byte-equivalent references to one
    ``k``-fact record, with repeat indices ``0..k-1`` and weights ``1/k``.
    Isolated groups must consist of ``k`` distinct one-fact records, each with
    weight one.  In either arm every fact has effective group weight exactly
    one (within a tight floating-point tolerance).
    """

    if not scheduled:
        raise TrainingContractError("cannot group an empty schedule")
    buckets: dict[str, list[ScheduledRecord]] = defaultdict(list)
    for index, item in enumerate(scheduled):
        comparison_id = item.schedule.get("comparison_id")
        if not isinstance(comparison_id, str) or not comparison_id:
            raise TrainingContractError(f"schedule row {index} has invalid comparison_id")
        contract = item.record.get("comparison_contract")
        record_comparison = contract.get("comparison_id") if isinstance(contract, Mapping) else None
        if record_comparison != comparison_id:
            raise TrainingContractError(
                f"schedule row {index} comparison_id does not match source record"
            )
        buckets[comparison_id].append(item)

    result: dict[str, ComparisonGroup] = {}
    for comparison_id, raw_items in buckets.items():
        items = tuple(sorted(raw_items, key=_item_sort_key))
        arms = {item.schedule.get("arm") for item in items}
        if len(arms) != 1 or next(iter(arms)) not in {"episode", "isolated"}:
            raise TrainingContractError(f"comparison {comparison_id} mixes or omits valid arms")
        arm = str(next(iter(arms)))
        for item in items:
            contract = item.record.get("comparison_contract")
            if not isinstance(contract, Mapping) or contract.get("arm") != arm:
                raise TrainingContractError(
                    f"comparison {comparison_id} arm differs from source record"
                )

        if arm == "episode":
            canonical = items[0]
            fact_ids = tuple(canonical.schedule["fact_ids"])
            expected_draws = len(fact_ids)
            if len(items) != expected_draws:
                raise TrainingContractError(
                    f"episode comparison {comparison_id} has {len(items)} draws "
                    f"for {expected_draws} facts"
                )
            if [item.schedule["repeat_index"] for item in items] != list(range(expected_draws)):
                raise TrainingContractError(
                    f"episode comparison {comparison_id} repeat indices are not 0..k-1"
                )
            for item in items:
                if item.record != canonical.record:
                    raise TrainingContractError(
                        f"episode comparison {comparison_id} repeats different records"
                    )
                if tuple(item.schedule["fact_ids"]) != fact_ids:
                    raise TrainingContractError(
                        f"episode comparison {comparison_id} changes facts across repeats"
                    )
                expected_weight = 1.0 / expected_draws
                if abs(float(item.schedule["sample_weight"]) - expected_weight) > 1e-12:
                    raise TrainingContractError(
                        f"episode comparison {comparison_id} weight is not 1/k"
                    )
        else:
            record_ids = [str(item.record.get("record_id")) for item in items]
            if len(set(record_ids)) != len(record_ids):
                raise TrainingContractError(f"isolated comparison {comparison_id} repeats a record")
            facts: list[str] = []
            for item in items:
                row_facts = tuple(item.schedule["fact_ids"])
                if len(row_facts) != 1:
                    raise TrainingContractError(
                        f"isolated comparison {comparison_id} contains a multi-fact draw"
                    )
                if item.schedule["repeat_index"] != 0:
                    raise TrainingContractError(
                        f"isolated comparison {comparison_id} has a repeated draw"
                    )
                if abs(float(item.schedule["sample_weight"]) - 1.0) > 1e-12:
                    raise TrainingContractError(
                        f"isolated comparison {comparison_id} weight is not one"
                    )
                facts.extend(row_facts)
            if len(set(facts)) != len(facts):
                raise TrainingContractError(f"isolated comparison {comparison_id} repeats a fact")
            fact_ids = tuple(facts)

        effective_weights: Counter[str] = Counter()
        for item in items:
            weight = float(item.schedule["sample_weight"])
            for fact_id in item.schedule["fact_ids"]:
                effective_weights[fact_id] += weight
        if any(abs(weight - 1.0) > 1e-12 for weight in effective_weights.values()):
            raise TrainingContractError(
                f"comparison {comparison_id} facts do not have unit effective weight"
            )
        result[comparison_id] = ComparisonGroup(
            comparison_id=comparison_id,
            arm=arm,
            fact_ids=fact_ids,
            items=items,
        )
    schedule_arms = {group.arm for group in result.values()}
    if len(schedule_arms) != 1:
        raise TrainingContractError("one schedule cannot mix episode and isolated groups")
    return result


def validate_paired_comparison_groups(
    first: Mapping[str, ComparisonGroup],
    second: Mapping[str, ComparisonGroup],
) -> dict[str, int]:
    """Require two arms to contain identical groups, draw counts, and facts."""

    arms = {group.arm for group in first.values()} | {group.arm for group in second.values()}
    if arms != {"episode", "isolated"}:
        raise TrainingContractError(
            "paired comparison schedules must contain one episode and one isolated arm"
        )
    first_ids = set(first)
    second_ids = set(second)
    if first_ids != second_ids:
        missing = sorted(first_ids ^ second_ids)[:3]
        raise TrainingContractError(f"paired schedules have different comparison IDs: {missing}")
    draw_count = 0
    fact_count = 0
    for comparison_id in sorted(first_ids):
        left = first[comparison_id]
        right = second[comparison_id]
        if left.draw_count != right.draw_count:
            raise TrainingContractError(f"comparison {comparison_id} has unequal arm draw counts")
        if Counter(left.fact_ids) != Counter(right.fact_ids):
            raise TrainingContractError(f"comparison {comparison_id} has unequal arm fact coverage")
        if any(not item.model_image_paths for item in (*left.items, *right.items)):
            raise TrainingContractError(
                f"comparison {comparison_id} lacks source-derived model image paths"
            )
        left_images = Counter(
            image_path for item in left.items for image_path in item.model_image_paths
        )
        right_images = Counter(
            image_path for item in right.items for image_path in item.model_image_paths
        )
        if left_images != right_images:
            raise TrainingContractError(f"comparison {comparison_id} has unequal image occurrences")
        draw_count += left.draw_count
        fact_count += len(left.fact_ids)
    return {
        "comparison_groups": len(first_ids),
        "draws_per_arm": draw_count,
        "facts": fact_count,
    }


def ordered_comparison_ids(
    groups: Mapping[str, ComparisonGroup], *, seed: int, epoch: int
) -> list[str]:
    """Return a cross-arm-identical seeded order without RNG-version coupling."""

    def key(comparison_id: str) -> tuple[str, str]:
        payload = f"epispace-comparison-order-v1:{seed}:{epoch}:{comparison_id}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest(), comparison_id

    return sorted(groups, key=key)
