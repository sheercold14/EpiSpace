"""Compile Scriptgen question families into raw and streaming VLM QA records.

The rendered collection is immutable input.  This module only writes derived
JSONL artifacts and never copies or edits simulator output.  Gold answers are
always taken from, or recompiled by, :class:`CapabilityCompiler`.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .behavior import RenderSceneView
from .compiler import CapabilityCompiler, Certificate
from .family import ScriptgenFamilyV4, ScriptgenQuestionGroupV1
from .library import SCRIPT_LIBRARY
from .occlusion import OCCLUDER_DISPLAY_NAMES_ZH
from .standards import STD_V1

RAW_SCHEMA = "scriptgen.raw_qa.v1"
STREAMING_SCHEMA = "scriptgen.streaming_qa.v1"
EVAL_INPUT_SCHEMA = "scriptgen.eval_input.v1"
EVAL_ORACLE_SCHEMA = "scriptgen.eval_oracle.v1"
MANIFEST_SCHEMA = "scriptgen.qa_dataset_manifest.v1"
POLICY_SCHEMA = "scriptgen.qa_generation_policy.v1"

DEFAULT_EXCLUDED_CAPABILITIES = frozenset({"existence_sufficiency_bed"})

LABEL_GLOSS_ZH = {
    "front": "前方",
    "left": "左侧",
    "back": "后方",
    "right": "右侧",
    "left_half": "画面左半边",
    "right_half": "画面右半边",
    "over_90": "超过90度",
    "at_most_90": "不超过90度",
    "visible": "可见",
    "not_visible": "不可见",
    "present": "存在",
    "absent": "不存在",
    "first": "第一个对象",
    "second": "第二个对象",
    "occluded": "被物体遮挡",
    "out_of_view": "移出视野",
    **OCCLUDER_DISPLAY_NAMES_ZH,
    "无法判断": "证据不足",
}


class QADatasetError(RuntimeError):
    """The immutable source cannot produce a valid QA release."""


class MediaRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ordinal: int
    source_frame: int
    path: str
    sha256: str


class SourceLineage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    plan_id: str
    question_group_id: str
    family_id: str | None
    episode_id: str | None
    family_path: str | None
    group_path: str
    standard_version: str


class RawQARecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["scriptgen.raw_qa.v1"] = RAW_SCHEMA
    record_id: str
    input_sha256: str
    pool: Literal["development_pool"] = "development_pool"
    sample_type: Literal["raw_qa"] = "raw_qa"
    temporal_mode: Literal["full", "immediate", "delayed"] = "full"
    tier: Literal["P1", "P2", "P3"]
    capability: str
    role: Literal["primary", "probe", "check"]
    variant: str
    cluster_ids: dict[str, str]
    source: SourceLineage
    images: tuple[MediaRef, ...]
    messages: tuple[dict[str, Any], ...]
    answer: dict[str, Any]
    loss_policy: dict[str, Any]
    reward_spec: dict[str, Any]
    batch_id: str


class StreamingTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_id: str
    capability: str
    prefix_length: int
    new_images: tuple[MediaRef, ...]
    question: str
    choices: tuple[str, ...]
    answer: str
    status: Literal["answerable", "abstain"]
    certificate_sha256: str


class StreamingQARecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["scriptgen.streaming_qa.v1"] = STREAMING_SCHEMA
    record_id: str
    input_sha256: str
    pool: Literal["development_pool"] = "development_pool"
    sample_type: Literal["streaming_qa"] = "streaming_qa"
    temporal_mode: Literal["full", "immediate", "delayed"] = "full"
    stream_kind: Literal["trajectory_multi_question", "evidence_reveal"]
    tier: Literal["P1", "P2", "P3"]
    capabilities: tuple[str, ...]
    cluster_ids: dict[str, str]
    source: SourceLineage
    images: tuple[MediaRef, ...]
    turns: tuple[StreamingTurn, ...]
    messages: tuple[dict[str, Any], ...]
    assistant_span_contract: tuple[dict[str, Any], ...]
    loss_policy: dict[str, Any]
    reward_spec: dict[str, Any]
    batch_id: str


@dataclass(frozen=True)
class FamilySource:
    group_path: Path
    group: ScriptgenQuestionGroupV1
    family_path: Path
    family: ScriptgenFamilyV4
    plan: dict[str, Any]


@dataclass(frozen=True)
class PrefixState:
    prefix_length: int
    certificate: Certificate

    @property
    def label(self) -> str | None:
        return self.certificate.answer.label if self.certificate.answer else None


@dataclass
class SourceHasher:
    root: Path

    def __post_init__(self) -> None:
        self._cache: dict[Path, tuple[str, int]] = {}

    def hash(self, path: Path) -> str:
        path = path.resolve()
        cached = self._cache.get(path)
        if cached is not None:
            return cached[0]
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        value = digest.hexdigest()
        self._cache[path] = (value, size)
        return value

    def rows(self) -> list[dict[str, Any]]:
        rows = []
        for path, (digest, size) in sorted(self._cache.items(), key=lambda item: str(item[0])):
            try:
                stored_path = path.relative_to(self.root).as_posix()
                path_kind = "source_relative"
            except ValueError:
                stored_path = str(path)
                path_kind = "absolute_dependency"
            rows.append(
                {
                    "path": stored_path,
                    "path_kind": path_kind,
                    "size_bytes": size,
                    "sha256": digest,
                }
            )
        return rows


def _json_sha(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise QADatasetError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def capability_tier(capability: str) -> Literal["P1", "P2", "P3"]:
    if capability.startswith("reference_frame_"):
        return "P2"
    if capability.startswith("cross_view_"):
        return "P3"
    if capability == "existence_sufficiency_bed":
        raise QADatasetError("bed existence is out of scope for this release")
    return "P1"


def _option_surface(options: Sequence[str]) -> str:
    rendered = [
        f"{label}（{LABEL_GLOSS_ZH[label]}）" if label in LABEL_GLOSS_ZH else label
        for label in options
    ]
    return "、".join(rendered)


def format_question(text: str, options: Sequence[str], *, streaming: bool) -> str:
    prefix = "截至当前已收到的图像，" if streaming else ""
    return (
        f"{prefix}{text}\n可选答案标签：{_option_surface(options)}。"
        "请只输出一个标签，格式为 <answer>标签</answer>。"
    )


def _answer_text(label: str) -> str:
    return f"<answer>{label}</answer>"


def _raw_system() -> str:
    return (
        "这些RGB图像按给定顺序来自同一段连续观察。只依据图像及其顺序回答；"
        "证据不足时必须选择“无法判断”，不要使用场景常识补全未观察事实。"
    )


def _streaming_system() -> str:
    return (
        "你会按时间顺序逐轮收到同一条轨迹的新RGB图像。回答时只能使用当前已经释放的"
        "图像和此前对话；不得假设未来画面，证据不足时选择“无法判断”。"
    )


def _family_sources(
    source_root: Path,
    excluded: frozenset[str],
    *,
    group_paths: Sequence[Path] | None = None,
    excluded_episode_capabilities: dict[str, frozenset[str]] | None = None,
) -> Iterator[FamilySource]:
    if group_paths is None:
        group_paths = sorted((source_root / "groups").glob("*/group.json"))
    for group_path in group_paths:
        group = ScriptgenQuestionGroupV1.model_validate_json(group_path.read_text(encoding="utf-8"))
        plan = _read_json(Path(group.trajectory.plan_record))
        episode_id = group_path.parent.name
        episode_excluded = (excluded_episode_capabilities or {}).get(
            episode_id, frozenset()
        )
        for entry in group.questions:
            if (
                entry.capability in excluded
                or entry.capability in episode_excluded
                or entry.family is None
            ):
                continue
            family_path = group_path.parent / entry.family
            family = ScriptgenFamilyV4.model_validate_json(family_path.read_text(encoding="utf-8"))
            if family.capability != entry.capability:
                raise QADatasetError(
                    f"group/family capability mismatch: {group_path} {entry.capability}"
                )
            yield FamilySource(group_path, group, family_path, family, plan)


def _coverage_group_paths(
    source_root: Path,
    dataset: dict[str, Any],
    *,
    excluded_episode_ids: frozenset[str] = frozenset(),
) -> tuple[Path, ...]:
    """Resolve only the accepted groups named by a packaged coverage snapshot."""

    episodes = dataset.get("episodes")
    if not isinstance(episodes, list):
        raise QADatasetError("coverage dataset episodes must be a list")
    if dataset.get("episode_count") != len(episodes):
        raise QADatasetError("coverage dataset episode_count does not match episodes")
    result: list[Path] = []
    seen: set[str] = set()
    for episode in episodes:
        if not isinstance(episode, dict) or not isinstance(episode.get("episode_id"), str):
            raise QADatasetError("coverage dataset contains an invalid episode row")
        episode_id = episode["episode_id"]
        if episode_id in seen:
            raise QADatasetError(f"duplicate coverage episode: {episode_id}")
        seen.add(episode_id)
        if episode_id in excluded_episode_ids:
            continue
        group_path = (source_root / "groups" / episode_id / "group.json").resolve()
        declared_group = episode.get("group")
        if not isinstance(declared_group, str):
            raise QADatasetError(f"coverage episode has no group path: {episode_id}")
        if Path(declared_group).resolve() != group_path:
            raise QADatasetError(f"coverage episode group path mismatch: {episode_id}")
        if not group_path.is_file():
            raise QADatasetError(f"missing accepted coverage group: {group_path}")
        result.append(group_path)
    return tuple(sorted(result))


def _relative_source_path(source_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(source_root).as_posix()
    except ValueError as error:
        raise QADatasetError(f"path escapes source root: {path}") from error


def _media_ref(
    *,
    source_root: Path,
    family_path: Path,
    family: ScriptgenFamilyV4,
    source_frame: int,
    ordinal: int,
    hasher: SourceHasher,
) -> MediaRef:
    try:
        frame = next(row for row in family.frames if row.frame == source_frame)
    except StopIteration as error:
        raise QADatasetError(
            f"{family.family_id}: missing media row for frame {source_frame}"
        ) from error
    image_path = (family_path.parent / frame.rgb).resolve()
    if not image_path.is_file():
        raise QADatasetError(f"missing RGB image: {image_path}")
    return MediaRef(
        ordinal=ordinal,
        source_frame=source_frame,
        path=_relative_source_path(source_root, image_path),
        sha256=hasher.hash(image_path),
    )


def _lineage(source_root: Path, source: FamilySource, episode_id: str | None) -> SourceLineage:
    return SourceLineage(
        scene_id=source.family.scene_id,
        plan_id=source.family.plan_id,
        question_group_id=source.family.question_group_id,
        family_id=source.family.family_id,
        episode_id=episode_id,
        family_path=_relative_source_path(source_root, source.family_path),
        group_path=_relative_source_path(source_root, source.group_path),
        standard_version=source.family.standard_version,
    )


def _input_hash(payload: dict[str, Any]) -> str:
    excluded = {
        "input_sha256",
        "batch_id",
        "answer",
        "assistant_span_contract",
        "reward_spec",
    }
    clean = {key: value for key, value in payload.items() if key not in excluded}
    if "messages" in clean:
        clean["messages"] = [
            message for message in clean["messages"] if message.get("role") != "assistant"
        ]
    if "turns" in clean:
        clean["turns"] = [
            {
                key: value
                for key, value in turn.items()
                if key not in {"answer", "certificate_sha256", "status"}
            }
            for turn in clean["turns"]
        ]
    return _json_sha(clean)


def _raw_record(
    source_root: Path,
    source: FamilySource,
    episode: Any,
    hasher: SourceHasher,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    images = tuple(
        _media_ref(
            source_root=source_root,
            family_path=source.family_path,
            family=source.family,
            source_frame=source_frame,
            ordinal=ordinal,
            hasher=hasher,
        )
        for ordinal, source_frame in enumerate(episode.frame_sequence)
    )
    question = format_question(
        source.family.question.text,
        source.family.question.options,
        streaming=False,
    )
    record_id = "raw-" + hashlib.sha256(episode.episode_id.encode()).hexdigest()[:20]
    messages = (
        {"role": "system", "content": _raw_system()},
        {
            "role": "user",
            "content": [
                *({"type": "image", "image": image.path} for image in images),
                {"type": "text", "text": question},
            ],
        },
        {"role": "assistant", "content": _answer_text(episode.label)},
    )
    payload: dict[str, Any] = {
        "schema_version": RAW_SCHEMA,
        "record_id": record_id,
        "input_sha256": "",
        "pool": "development_pool",
        "sample_type": "raw_qa",
        "temporal_mode": "full",
        "tier": capability_tier(source.family.capability),
        "capability": source.family.capability,
        "role": source.family.role,
        "variant": episode.kind,
        "cluster_ids": {
            "scene": source.family.scene_id,
            "trajectory": source.family.plan_id,
            "family": source.family.family_id,
        },
        "source": _lineage(source_root, source, episode.episode_id).model_dump(),
        "images": [image.model_dump() for image in images],
        "messages": list(messages),
        "answer": {
            "label": episode.label,
            "choices": list(source.family.question.options),
        },
        "loss_policy": {
            "train_on": "assistant_only",
            "mask_system_user_visual": True,
            "supervise_eos": True,
        },
        "reward_spec": {
            "type": "exact_label",
            "target": episode.label,
            "abstain_label": source.family.question.abstain_option,
        },
        "batch_id": "",
    }
    payload["input_sha256"] = _input_hash(payload)
    validated = RawQARecord.model_validate(payload).model_dump(mode="json")
    certificate = episode.certificate.model_dump(mode="json")
    oracle = {
        "schema_version": EVAL_ORACLE_SCHEMA,
        "record_id": record_id,
        "input_sha256": payload["input_sha256"],
        "answer": payload["answer"],
        "certificate_sha256": _json_sha(certificate),
        "certificate": certificate,
        "source": payload["source"],
    }
    eval_input = {
        "schema_version": EVAL_INPUT_SCHEMA,
        "record_id": record_id,
        "input_sha256": payload["input_sha256"],
        "sample_type": "raw_qa",
        "tier": payload["tier"],
        "capability": payload["capability"],
        "role": payload["role"],
        "variant": payload["variant"],
        "cluster_ids": payload["cluster_ids"],
        "images": payload["images"],
        "system": _raw_system(),
        "question": question,
        "choices": list(source.family.question.options),
    }
    return validated, eval_input, oracle


def _compile_prefixes(
    view: RenderSceneView,
    family: ScriptgenFamilyV4,
    binding: dict[str, str],
    *,
    stop_after_first_answerable: bool,
) -> list[PrefixState]:
    canonical = next(episode for episode in family.episodes if episode.kind == "canonical")
    return _compile_sequence_prefixes(
        view,
        family,
        binding,
        frame_sequence=canonical.frame_sequence,
        expected_episode=canonical,
        stop_after_first_answerable=stop_after_first_answerable,
    )


def _compile_sequence_prefixes(
    view: RenderSceneView,
    family: ScriptgenFamilyV4,
    binding: dict[str, str],
    *,
    frame_sequence: Sequence[int],
    expected_episode: Any,
    stop_after_first_answerable: bool,
) -> list[PrefixState]:
    """Compile prefixes of canonical or intervened frame sequences."""

    compiler = CapabilityCompiler(script=SCRIPT_LIBRARY[family.capability], std=STD_V1)
    states: list[PrefixState] = []
    sequence = tuple(frame_sequence)
    for prefix_length in range(1, len(sequence) + 1):
        certificate = compiler.compile(
            view,
            binding,
            frame_sequence=sequence[:prefix_length],
        )
        states.append(PrefixState(prefix_length, certificate))
        if stop_after_first_answerable and certificate.status == "answerable":
            break
    full = compiler.compile(view, binding, frame_sequence=sequence)
    if full.status != expected_episode.certificate.status:
        raise QADatasetError(
            f"{family.family_id}: recompiled {expected_episode.kind} status "
            f"{full.status} != packaged {expected_episode.certificate.status}"
        )
    full_label = full.answer.label if full.answer is not None else family.question.abstain_option
    if full_label != expected_episode.label:
        raise QADatasetError(
            f"{family.family_id}: recompiled {full_label} != packaged {expected_episode.label}"
        )
    return states


def _event_question(family: ScriptgenFamilyV4) -> str:
    return format_question(family.question.text, family.question.options, streaming=True)


def _prefix_has_pose_transition(view: RenderSceneView, prefix_length: int) -> bool:
    """Whether a prefix contains observable camera motion in the pose record.

    Simulator warm-up can change RGB pixels while leaving the camera pose
    exactly unchanged.  A self-motion question must therefore be gated by the
    authoritative pose sequence, never by pixel differences alone.
    """

    for frame in range(1, prefix_length):
        left = view.camera_pose(frame - 1)
        right = view.camera_pose(frame)
        translation = math.hypot(right.x - left.x, right.y - left.y)
        turn = abs((right.yaw_deg - left.yaw_deg + 180.0) % 360.0 - 180.0)
        if translation > 1e-6 or turn > 1e-6:
            return True
    return False


def _p1_event_states(
    view: RenderSceneView,
    states: Sequence[PrefixState],
    *,
    force_endpoint: bool,
) -> tuple[PrefixState, ...]:
    """Select P1 label transitions without dropping ``A -> B -> A``.

    Consecutive equal labels are redundant, but a label returning after an
    intervening value is a real state transition.  The source capability also
    receives a canonical full-episode event so the stream releases its tail
    even when the answer stabilized earlier.
    """

    selected: list[PrefixState] = []
    last_label: str | None = None
    for state in states:
        if state.prefix_length < 2 or not _prefix_has_pose_transition(view, state.prefix_length):
            continue
        if state.certificate.status != "answerable" or state.label is None:
            continue
        if state.label == last_label:
            continue
        selected.append(state)
        last_label = state.label

    if force_endpoint and states:
        endpoint = states[-1]
        if (
            endpoint.certificate.status == "answerable"
            and endpoint.label is not None
            and _prefix_has_pose_transition(view, endpoint.prefix_length)
            and not any(state.prefix_length == endpoint.prefix_length for state in selected)
        ):
            selected.append(endpoint)
    return tuple(selected)


def _stream_record(
    *,
    source_root: Path,
    record_id: str,
    stream_kind: str,
    tier: str,
    sources: Sequence[FamilySource],
    event_rows: Sequence[tuple[FamilySource, PrefixState]],
    hasher: SourceHasher,
    temporal_mode: Literal["full", "immediate", "delayed"] = "full",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not event_rows:
        raise QADatasetError(f"{record_id}: streaming record has no turns")
    first = sources[0]
    messages: list[dict[str, Any]] = [{"role": "system", "content": _streaming_system()}]
    turns: list[dict[str, Any]] = []
    spans: list[dict[str, Any]] = []
    oracle_turns: list[dict[str, Any]] = []
    flat_images: list[MediaRef] = []
    released = 0
    for turn_index, (source, state) in enumerate(event_rows, start=1):
        if state.prefix_length < released:
            raise QADatasetError(f"{record_id}: non-monotonic streaming events")
        frame_sequence = state.certificate.frame_sequence
        new_images = tuple(
            _media_ref(
                source_root=source_root,
                family_path=source.family_path,
                family=source.family,
                source_frame=frame,
                ordinal=len(flat_images) + offset,
                hasher=hasher,
            )
            for offset, frame in enumerate(frame_sequence[released : state.prefix_length])
        )
        flat_images.extend(new_images)
        released = state.prefix_length
        certificate = state.certificate.model_dump(mode="json")
        label = state.label
        if state.certificate.status not in {"answerable", "abstain"}:
            raise QADatasetError(
                f"{record_id}: attempted to serialize invalid state {state.certificate.status}"
            )
        if state.certificate.status == "abstain":
            label = source.family.question.abstain_option
        if label is None:
            raise QADatasetError(f"{record_id}: valid streaming state has no label")
        turn_id = f"{record_id}-t{turn_index:02d}"
        question = _event_question(source.family)
        user_content = [
            *({"type": "image", "image": image.path} for image in new_images),
            {"type": "text", "text": question},
        ]
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": _answer_text(label)})
        certificate_sha = _json_sha(certificate)
        turn = StreamingTurn(
            turn_id=turn_id,
            capability=source.family.capability,
            prefix_length=state.prefix_length,
            new_images=new_images,
            question=question,
            choices=source.family.question.options,
            answer=label,
            status=state.certificate.status,
            certificate_sha256=certificate_sha,
        ).model_dump(mode="json")
        turns.append(turn)
        spans.append(
            {
                "message_index": len(messages) - 1,
                "turn_id": turn_id,
                "answer_sha256": hashlib.sha256(_answer_text(label).encode()).hexdigest(),
                "supervise_eos": True,
            }
        )
        oracle_turns.append(
            {
                "turn_id": turn_id,
                "prefix_length": state.prefix_length,
                "answer": label,
                "choices": list(source.family.question.options),
                "status": state.certificate.status,
                "certificate_sha256": certificate_sha,
                "certificate": certificate,
                "family_id": source.family.family_id,
            }
        )

    capabilities = tuple(dict.fromkeys(source.family.capability for source, _ in event_rows))
    lineage = _lineage(source_root, first, None).model_dump()
    lineage["family_id"] = None if len(sources) > 1 else sources[0].family.family_id
    lineage["family_path"] = None if len(sources) > 1 else lineage["family_path"]
    payload: dict[str, Any] = {
        "schema_version": STREAMING_SCHEMA,
        "record_id": record_id,
        "input_sha256": "",
        "pool": "development_pool",
        "sample_type": "streaming_qa",
        "temporal_mode": temporal_mode,
        "stream_kind": stream_kind,
        "tier": tier,
        "capabilities": list(capabilities),
        "cluster_ids": {
            "scene": first.family.scene_id,
            "trajectory": first.family.plan_id,
            "stream": first.family.question_group_id,
        },
        "source": lineage,
        "images": [image.model_dump() for image in flat_images],
        "turns": turns,
        "messages": messages,
        "assistant_span_contract": spans,
        "loss_policy": {
            "train_on": "all_assistant_turns",
            "mask_system_user_visual": True,
            "supervise_eos": True,
            "normalization": "mean_tokens_per_turn_then_mean_turns",
        },
        "reward_spec": {
            "type": "per_turn_exact_label",
            "history_modes": ["teacher_forced", "free_running"],
            "targets": [turn["answer"] for turn in turns],
        },
        "batch_id": "",
    }
    payload["input_sha256"] = _input_hash(payload)
    validated = StreamingQARecord.model_validate(payload).model_dump(mode="json")
    eval_input = {
        "schema_version": EVAL_INPUT_SCHEMA,
        "record_id": record_id,
        "input_sha256": payload["input_sha256"],
        "sample_type": "streaming_qa",
        "temporal_mode": temporal_mode,
        "stream_kind": stream_kind,
        "tier": tier,
        "capabilities": list(capabilities),
        "cluster_ids": payload["cluster_ids"],
        "system": _streaming_system(),
        "turns": [
            {
                "turn_id": turn["turn_id"],
                "capability": turn["capability"],
                "prefix_length": turn["prefix_length"],
                "new_images": turn["new_images"],
                "question": turn["question"],
                "choices": turn["choices"],
            }
            for turn in turns
        ],
    }
    oracle = {
        "schema_version": EVAL_ORACLE_SCHEMA,
        "record_id": record_id,
        "input_sha256": payload["input_sha256"],
        "turns": oracle_turns,
        "source": payload["source"],
        "temporal_mode": temporal_mode,
    }
    return validated, eval_input, oracle


_OCCLUSION_ISOLATED_CAPABILITIES = frozenset(
    {"occluder_identification", "disappearance_cause"}
)


def _stable_source_order(sources: Sequence[FamilySource], namespace: str) -> list[FamilySource]:
    return sorted(
        sources,
        key=lambda source: hashlib.sha256(
            f"{namespace}:{source.family.family_id}".encode()
        ).hexdigest(),
    )


def _family_episode(source: FamilySource, kind: str) -> Any:
    try:
        return next(episode for episode in source.family.episodes if episode.kind == kind)
    except StopIteration as error:
        raise QADatasetError(f"{source.family.family_id}: missing {kind} episode") from error


def _build_occlusion_isolated_streaming(
    source_root: Path,
    sources: Sequence[FamilySource],
    hasher: SourceHasher,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Immediate/delayed one-question records with calibrated abstention.

    Cause records are sampled exactly 1:1:1 across occluded, out_of_view, and
    unable.  Identity records keep every answerable family and add enough
    deterministic drop-key records for an approximately 25% unable rate.
    Every record has one turn, so neither an earlier answer nor another
    capability can leak into its history.
    """

    special = [
        source
        for source in sources
        if source.family.capability in _OCCLUSION_ISOLATED_CAPABILITIES
    ]
    if not special:
        return [], [], [], []

    cause_by_label: dict[str, list[FamilySource]] = defaultdict(list)
    identity_sources: list[FamilySource] = []
    for source in special:
        canonical = _family_episode(source, "canonical")
        if source.family.capability == "disappearance_cause":
            cause_by_label[canonical.label].append(source)
        else:
            identity_sources.append(source)

    cause_sources: list[FamilySource] = []
    if cause_by_label:
        expected = {"occluded", "out_of_view"}
        if set(cause_by_label) != expected:
            raise QADatasetError(
                "disappearance_cause overlay must contain occluded and out_of_view labels"
            )
        balanced_count = min(len(cause_by_label[label]) for label in expected)
        for label in sorted(expected):
            cause_sources.extend(
                _stable_source_order(cause_by_label[label], f"cause:{label}")[:balanced_count]
            )
        cause_unable_sources = _stable_source_order(
            cause_sources, "cause:unable"
        )[:balanced_count]
    else:
        cause_unable_sources = []

    identity_sources = _stable_source_order(identity_sources, "identity:answerable")
    # k / (N + k) ~= 25%, where both answerable and unable families produce
    # one immediate plus one delayed isolated record.
    identity_unable_count = math.ceil(len(identity_sources) / 3)
    identity_unable_sources = _stable_source_order(
        identity_sources, "identity:unable"
    )[:identity_unable_count]
    unable_ids = {
        (source.family.capability, source.family.family_id)
        for source in (*cause_unable_sources, *identity_unable_sources)
    }
    answerable_ids = {
        (source.family.capability, source.family.family_id)
        for source in (*cause_sources, *identity_sources)
    }

    records: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    oracles: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    current_group: Path | None = None
    view: RenderSceneView | None = None
    for source in sorted(
        special,
        key=lambda value: (str(value.group_path), value.family.capability),
    ):
        key = (source.family.capability, source.family.family_id)
        emit_answerable = key in answerable_ids
        emit_unable = key in unable_ids
        if not emit_answerable and not emit_unable:
            continue
        if source.group_path != current_group:
            view = RenderSceneView.from_bundle(
                Path(source.group.trajectory.bundle),
                STD_V1,
                scene_ir=Path(source.group.trajectory.scene_ir),
            )
            current_group = source.group_path
        assert view is not None
        binding = dict(source.plan["binding"])

        def emit(state: PrefixState, temporal_mode: Literal["immediate", "delayed"], label: str) -> None:
            record_id = "stream-occlusion-" + hashlib.sha256(
                f"{source.family.family_id}:{label}:{temporal_mode}".encode()
            ).hexdigest()[:20]
            triple = _stream_record(
                source_root=source_root,
                record_id=record_id,
                stream_kind="evidence_reveal",
                tier=capability_tier(source.family.capability),
                sources=(source,),
                event_rows=((source, state),),
                hasher=hasher,
                temporal_mode=temporal_mode,
            )
            records.append(triple[0])
            inputs.append(triple[1])
            oracles.append(triple[2])

        if emit_answerable:
            canonical = _family_episode(source, "canonical")
            states = _compile_sequence_prefixes(
                view,
                source.family,
                binding,
                frame_sequence=canonical.frame_sequence,
                expected_episode=canonical,
                stop_after_first_answerable=False,
            )
            immediate = next(
                (state for state in states if state.certificate.status == "answerable"),
                None,
            )
            delayed = states[-1]
            if immediate is None:
                raise QADatasetError(f"{source.family.family_id}: no answerable event prefix")
            emit(immediate, "immediate", "answerable")
            if delayed.prefix_length > immediate.prefix_length:
                emit(delayed, "delayed", "answerable")
            else:
                skips.append(
                    {
                        "family_id": source.family.family_id,
                        "tier": "P1",
                        "reason": "no_post_event_delay",
                    }
                )

        if emit_unable:
            drop_key = _family_episode(source, "drop_key")
            states = _compile_sequence_prefixes(
                view,
                source.family,
                binding,
                frame_sequence=drop_key.frame_sequence,
                expected_episode=drop_key,
                stop_after_first_answerable=False,
            )
            abstains = [state for state in states if state.certificate.status == "abstain"]
            if not abstains:
                raise QADatasetError(f"{source.family.family_id}: drop_key has no abstain prefix")
            immediate = next(
                (state for state in abstains if state.prefix_length >= 2), abstains[0]
            )
            emit(immediate, "immediate", "unable")
            if abstains[-1].prefix_length > immediate.prefix_length:
                emit(abstains[-1], "delayed", "unable")
            else:
                skips.append(
                    {
                        "family_id": source.family.family_id,
                        "tier": "P1",
                        "reason": "drop_key_too_short_for_delay",
                    }
                )
    return records, inputs, oracles, skips


def _build_streaming(
    source_root: Path,
    family_sources: Sequence[FamilySource],
    hasher: SourceHasher,
    *,
    source_capabilities: dict[Path, str] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    records, inputs, oracles, skips = _build_occlusion_isolated_streaming(
        source_root, family_sources, hasher
    )
    by_group: dict[Path, list[FamilySource]] = defaultdict(list)
    for source in family_sources:
        if source.family.capability in _OCCLUSION_ISOLATED_CAPABILITIES:
            continue
        by_group[source.group_path].append(source)
    for group_path, sources in sorted(by_group.items(), key=lambda item: str(item[0])):
        first = sources[0]
        tier = capability_tier(first.family.capability)
        source_capability = (
            source_capabilities.get(group_path)
            if source_capabilities is not None
            else None
        ) or str(first.plan["capability"])
        view = RenderSceneView.from_bundle(
            Path(first.group.trajectory.bundle),
            STD_V1,
            scene_ir=Path(first.group.trajectory.scene_ir),
        )
        binding = dict(first.plan["binding"])
        if tier == "P1":
            events: list[tuple[FamilySource, PrefixState]] = []
            for source in sorted(sources, key=lambda value: value.family.capability):
                states = _compile_prefixes(
                    view,
                    source.family,
                    binding,
                    stop_after_first_answerable=False,
                )
                for state in _p1_event_states(
                    view,
                    states,
                    force_endpoint=source.family.capability == source_capability,
                ):
                    events.append((source, state))
            events.sort(key=lambda item: (item[1].prefix_length, item[0].family.capability))
            if len(events) < 2:
                skips.append(
                    {
                        "group": _relative_source_path(source_root, group_path),
                        "tier": tier,
                        "reason": "fewer_than_two_valid_turns",
                    }
                )
                continue
            record_id = (
                "stream-p1-"
                + hashlib.sha256(first.family.question_group_id.encode()).hexdigest()[:20]
            )
            triple = _stream_record(
                source_root=source_root,
                record_id=record_id,
                stream_kind="trajectory_multi_question",
                tier=tier,
                sources=sources,
                event_rows=events,
                hasher=hasher,
            )
            records.append(triple[0])
            inputs.append(triple[1])
            oracles.append(triple[2])
            continue

        for source in sorted(sources, key=lambda value: value.family.capability):
            states = _compile_prefixes(
                view,
                source.family,
                binding,
                stop_after_first_answerable=True,
            )
            first_answerable_index = next(
                (
                    index
                    for index, state in enumerate(states)
                    if state.certificate.status == "answerable"
                ),
                None,
            )
            if first_answerable_index is None:
                reason = "no_answerable_prefix"
            else:
                prior_abstains = [
                    state
                    for state in states[:first_answerable_index]
                    if state.certificate.status == "abstain"
                ]
                reason = "no_abstain_before_answer" if not prior_abstains else None
            if reason is not None:
                skips.append(
                    {
                        "family_id": source.family.family_id,
                        "tier": tier,
                        "reason": reason,
                    }
                )
                continue
            assert first_answerable_index is not None
            event_rows = (prior_abstains[-1], states[first_answerable_index])
            record_id = (
                "stream-" + hashlib.sha256(source.family.family_id.encode()).hexdigest()[:20]
            )
            triple = _stream_record(
                source_root=source_root,
                record_id=record_id,
                stream_kind="evidence_reveal",
                tier=tier,
                sources=(source,),
                event_rows=tuple((source, state) for state in event_rows),
                hasher=hasher,
            )
            records.append(triple[0])
            inputs.append(triple[1])
            oracles.append(triple[2])
    return records, inputs, oracles, skips


def _stratified_order(rows: Sequence[dict[str, Any]], *, streaming: bool) -> list[int]:
    buckets: dict[tuple[str, ...], deque[int]] = defaultdict(deque)
    for index, row in enumerate(rows):
        if streaming:
            key = (
                str(row["tier"]),
                str(row["stream_kind"]),
                str(row["capabilities"][0]),
            )
        else:
            key = (
                str(row["tier"]),
                str(row["capability"]),
                str(row["variant"]),
                str(row["answer"]["label"]),
            )
        buckets[key].append(index)
    ordered: list[int] = []
    keys = sorted(buckets)
    while keys:
        next_keys = []
        for key in keys:
            ordered.append(buckets[key].popleft())
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys
    return ordered


def _assign_batches(
    records: list[dict[str, Any]],
    inputs: list[dict[str, Any]],
    oracles: list[dict[str, Any]],
    *,
    prefix: str,
    batch_size: int,
    streaming: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    input_by_id = {row["record_id"]: row for row in inputs}
    oracle_by_id = {row["record_id"]: row for row in oracles}
    result_records = []
    result_inputs = []
    result_oracles = []
    for order, index in enumerate(_stratified_order(records, streaming=streaming)):
        record = records[index]
        batch_id = f"{prefix}-{order // batch_size + 1:04d}"
        record["batch_id"] = batch_id
        input_row = input_by_id[record["record_id"]]
        input_row["batch_id"] = batch_id
        oracle_row = oracle_by_id[record["record_id"]]
        oracle_row["batch_id"] = batch_id
        result_records.append(record)
        result_inputs.append(input_row)
        result_oracles.append(oracle_row)
    return result_records, result_inputs, result_oracles


def _source_digest(rows: Sequence[dict[str, Any]]) -> str:
    return _json_sha(
        [
            {
                "path": row["path"],
                "path_kind": row["path_kind"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
            }
            for row in rows
        ]
    )


def build_qa_dataset(
    *,
    source_root: Path,
    output_root: Path,
    batch_size: int = 100,
    excluded_capabilities: frozenset[str] = DEFAULT_EXCLUDED_CAPABILITIES,
    excluded_episode_ids: frozenset[str] = frozenset(),
    excluded_episode_capabilities: dict[str, frozenset[str]] | None = None,
) -> dict[str, Any]:
    """Build training-ready records plus answer-hidden evaluation projections."""

    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    dataset_path = source_root / "dataset.json"
    if not dataset_path.is_file():
        raise QADatasetError(f"missing source dataset index: {dataset_path}")
    dataset = _read_json(dataset_path)
    excluded_episode_capabilities = excluded_episode_capabilities or {}
    if dataset.get("standard_version") != STD_V1.standard_version:
        raise QADatasetError(
            f"source standard {dataset.get('standard_version')} != compiler {STD_V1.standard_version}"
        )

    if dataset.get("schema_version") == "scriptgen_binding_coverage_dataset.v1":
        known_episode_ids = {episode["episode_id"] for episode in dataset["episodes"]}
        unknown_exclusions = excluded_episode_ids - known_episode_ids
        if unknown_exclusions:
            raise QADatasetError(
                "excluded coverage episodes are unknown: "
                + ", ".join(sorted(unknown_exclusions)[:10])
            )
        unknown_scoped = set(excluded_episode_capabilities) - known_episode_ids
        if unknown_scoped:
            raise QADatasetError(
                "capability exclusions name unknown coverage episodes: "
                + ", ".join(sorted(unknown_scoped)[:10])
            )
        group_paths = _coverage_group_paths(
            source_root,
            dataset,
            excluded_episode_ids=excluded_episode_ids,
        )
        source_trajectory_count = len(group_paths)
        source_capabilities = {
            (source_root / "groups" / episode["episode_id"] / "group.json").resolve(): episode[
                "source_capability"
            ]
            for episode in dataset["episodes"]
            if episode["episode_id"] not in excluded_episode_ids
        }
    else:
        if excluded_episode_ids or excluded_episode_capabilities:
            raise QADatasetError(
                "episode and capability exclusions require a coverage dataset source"
            )
        if dataset.get("failed_trajectory_count") != 0:
            raise QADatasetError("source collection contains failed trajectories")
        group_paths = None
        source_trajectory_count = dataset.get("accepted_trajectory_count")
        source_capabilities = None

    hasher = SourceHasher(source_root)
    hasher.hash(dataset_path)
    sources = list(
        _family_sources(
            source_root,
            excluded_capabilities,
            group_paths=group_paths,
            excluded_episode_capabilities=excluded_episode_capabilities,
        )
    )
    seen_groups: set[Path] = set()
    for source in sources:
        if source.group_path in seen_groups:
            continue
        seen_groups.add(source.group_path)
        hasher.hash(Path(source.group.trajectory.plan_record))
        hasher.hash(Path(source.group.trajectory.scene_ir))
        bundle = Path(source.group.trajectory.bundle)
        for name in ("trajectory_plan.json", "scene_snapshot.json"):
            path = bundle / name
            if path.is_file():
                hasher.hash(path)
        for path in sorted((bundle / "views").glob("view-*.sensors.npz")):
            hasher.hash(path)
    raw_records: list[dict[str, Any]] = []
    raw_inputs: list[dict[str, Any]] = []
    raw_oracles: list[dict[str, Any]] = []
    for source in sources:
        hasher.hash(source.group_path)
        hasher.hash(source.family_path)
        for episode in source.family.episodes:
            triple = _raw_record(source_root, source, episode, hasher)
            raw_records.append(triple[0])
            raw_inputs.append(triple[1])
            raw_oracles.append(triple[2])

    streaming_records, streaming_inputs, streaming_oracles, streaming_skips = _build_streaming(
        source_root,
        sources,
        hasher,
        source_capabilities=source_capabilities,
    )
    raw_records, raw_inputs, raw_oracles = _assign_batches(
        raw_records,
        raw_inputs,
        raw_oracles,
        prefix="raw",
        batch_size=batch_size,
        streaming=False,
    )
    streaming_records, streaming_inputs, streaming_oracles = _assign_batches(
        streaming_records,
        streaming_inputs,
        streaming_oracles,
        prefix="streaming",
        batch_size=batch_size,
        streaming=True,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    source_link = output_root / "source"
    if source_link.is_symlink():
        if source_link.resolve() != source_root:
            raise QADatasetError(
                f"existing source link points elsewhere: {source_link} -> {source_link.resolve()}"
            )
    elif source_link.exists():
        raise QADatasetError(f"output source mount is not a symlink: {source_link}")
    else:
        source_link.symlink_to(source_root, target_is_directory=True)
    _write_jsonl(output_root / "raw_qa.jsonl", raw_records)
    _write_jsonl(output_root / "raw_eval_inputs.jsonl", raw_inputs)
    _write_jsonl(output_root / "raw_eval_oracle.jsonl", raw_oracles)
    _write_jsonl(output_root / "streaming_qa.jsonl", streaming_records)
    _write_jsonl(output_root / "streaming_eval_inputs.jsonl", streaming_inputs)
    _write_jsonl(output_root / "streaming_eval_oracle.jsonl", streaming_oracles)
    _write_jsonl(output_root / "streaming_skips.jsonl", streaming_skips)

    source_rows = hasher.rows()
    _write_jsonl(output_root / "source_snapshot.jsonl", source_rows)
    policy = {
        "schema_version": POLICY_SCHEMA,
        "policy_version": "generation_policy.v2",
        "source_is_read_only": True,
        "model_visible_modalities": ["rgb"],
        "excluded_capabilities": sorted(excluded_capabilities),
        "excluded_episode_ids": sorted(excluded_episode_ids),
        "excluded_episode_capabilities": {
            episode_id: sorted(capabilities)
            for episode_id, capabilities in sorted(excluded_episode_capabilities.items())
        },
        "raw_variants": ["canonical", "permute", "drop_key", "drop_filler", "delay"],
        "streaming": {
            "P1": (
                "group questions at consecutive valid label transitions; preserve A-B-A; "
                "require an authoritative pose transition and force the source capability "
                "at the canonical full-episode endpoint"
            ),
            "P2_P3": "last abstain prefix then first answerable prefix per family",
            "occlusion_isolated": (
                "one question per record at immediate and delayed prefixes; cause labels "
                "are 1:1:1 occluded/out_of_view/unable; identity adds deterministic "
                "drop-key records for approximately 25% unable"
            ),
            "variants": ["canonical"],
            "invalid_prefix_policy": "never serialize as abstain",
        },
        "batch_size": batch_size,
        "language_policy": "preserve family question; append label glossary and exact output contract",
        "gold_policy": "compiler_only; model predictions never modify gold",
    }
    _write_json(output_root / "generation_policy.v2.json", policy)
    _write_jsonl(
        output_root / "generation_policy_history.jsonl",
        [
            {
                "iteration": 0,
                "status": "candidate",
                "rule": "serialize every compiler-answerable P1 prefix label transition",
                "finding": (
                    "path_integration_magnitude can be geometrically answerable at prefix 1, "
                    "before any visual motion transition exists"
                ),
            },
            {
                "iteration": 1,
                "status": "adopted",
                "rule": "P1 streaming questions require prefix_length >= 2",
                "reason": (
                    "remove convention-only single-frame net-turn cases without changing gold"
                ),
            },
            {
                "iteration": 2,
                "status": "superseded",
                "rule": "generation_policy.v1",
                "reason": "audit found global label dedup, missing endpoint, and pose evidence bugs",
            },
            {
                "iteration": 3,
                "status": "frozen",
                "rule": "generation_policy.v2",
                "reason": (
                    "use consecutive label dedup, canonical source endpoint, and pose-transition "
                    "evidence gate; compiler gold remains authoritative"
                ),
            },
        ],
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "dataset_id": "scriptgen_qa_v2",
        "pool": "development_pool",
        "source_root": str(source_root),
        "source_collection_id": dataset.get("collection_id"),
        "source_standard_version": dataset.get("standard_version"),
        "source_trajectory_count": source_trajectory_count,
        "source_snapshot_sha256": _source_digest(source_rows),
        "source_snapshot_file_count": len(source_rows),
        "excluded_capabilities": sorted(excluded_capabilities),
        "excluded_episode_ids": sorted(excluded_episode_ids),
        "excluded_episode_capabilities": {
            episode_id: sorted(capabilities)
            for episode_id, capabilities in sorted(excluded_episode_capabilities.items())
        },
        "family_count": len(sources),
        "raw_record_count": len(raw_records),
        "streaming_record_count": len(streaming_records),
        "streaming_turn_count": sum(len(record["turns"]) for record in streaming_records),
        "streaming_skip_count": len(streaming_skips),
        "batch_size": batch_size,
        "batch_counts": {
            "raw": len({record["batch_id"] for record in raw_records}),
            "streaming": len({record["batch_id"] for record in streaming_records}),
        },
        "artifacts": {
            "raw_training": "raw_qa.jsonl",
            "raw_eval_inputs": "raw_eval_inputs.jsonl",
            "raw_eval_oracle": "raw_eval_oracle.jsonl",
            "streaming_training": "streaming_qa.jsonl",
            "streaming_eval_inputs": "streaming_eval_inputs.jsonl",
            "streaming_eval_oracle": "streaming_eval_oracle.jsonl",
            "source_snapshot": "source_snapshot.jsonl",
            "source_mount": "source/",
            "policy": "generation_policy.v2.json",
            "policy_history": "generation_policy_history.jsonl",
        },
        "training_status": "not_run",
        "split_status": "not_assigned; preserve cluster_ids for later scaling",
    }
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def verify_source_snapshot(*, source_root: Path, snapshot_path: Path) -> dict[str, Any]:
    """Rehash every source file used by the release and refuse silent drift."""

    source_root = source_root.resolve()
    before = read_jsonl(snapshot_path)
    after = []
    hasher = SourceHasher(source_root)
    for row in before:
        path = (
            source_root / row["path"]
            if row.get("path_kind", "source_relative") == "source_relative"
            else Path(row["path"])
        )
        digest = hasher.hash(path)
        after.append(
            {
                "path": row["path"],
                "path_kind": row.get("path_kind", "source_relative"),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    changed = [
        {"before": old, "after": new} for old, new in zip(before, after, strict=True) if old != new
    ]
    return {
        "status": "pass" if not changed else "fail",
        "file_count": len(before),
        "snapshot_sha256": _source_digest(after),
        "changed": changed,
    }
