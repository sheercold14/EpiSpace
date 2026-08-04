#!/usr/bin/env python3
"""Exact-fact-loss LoRA SFT for the paired EpiSpace Qwen3-VL experiment.

The causal protocol uses one optimizer update per ``comparison_id``.  Its
``k`` schedule draws accumulate gradients before Adam is called once.  A
legacy per-draw mode remains available only for diagnostic compatibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from episode3d.pilot_summary import (  # noqa: E402
    verify_inventory_artifact,
    write_inventory_artifact,
)
from episode3d.qwen_training import (  # noqa: E402
    ComparisonGroup,
    ScheduledRecord,
    TrainingContractError,
    assistant_token_groups,
    build_comparison_groups,
    normalize_qwen_messages,
    normalized_schedule_image_paths,
    ordered_comparison_ids,
    record_model_image_paths,
    validate_paired_comparison_groups,
)


@dataclass(frozen=True)
class TrainConfig:
    model: str
    model_inventory: str
    schedule: str
    paired_schedule: str | None
    output_dir: str
    optimizer_unit: str
    seed: int
    epochs: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    gradient_accumulation_steps: int
    max_steps: int | None
    image_min_pixels: int
    image_max_pixels: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    gradient_checkpointing: bool
    max_grad_norm: float
    log_every: int
    overwrite: bool


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train one Qwen3-VL LoRA arm with exact per-fact loss weighting."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--model-inventory",
        type=Path,
        required=True,
        help="Reusable inventory artifact created by build_qwen_weight_inventory.py.",
    )
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument(
        "--paired-schedule",
        type=Path,
        help=(
            "Counterpart arm schedule. In comparison_group mode this is required, "
            "but is inferred from a .episode./.isolated. filename when possible."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--optimizer-unit",
        choices=("comparison_group", "draw"),
        default="comparison_group",
        help=(
            "comparison_group is the strict causal protocol; draw is a legacy "
            "diagnostic whose Adam updates are not episode/isolated matched."
        ),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--image-min-pixels", type=int, default=65_536)
    parser.add_argument("--image-max-pixels", type=int, default=262_144)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.image_min_pixels > args.image_max_pixels:
        parser.error("image_min_pixels cannot exceed image_max_pixels")
    return TrainConfig(
        model=str(args.model.resolve()),
        model_inventory=str(args.model_inventory.resolve()),
        schedule=str(args.schedule.resolve()),
        paired_schedule=(
            str(args.paired_schedule.resolve()) if args.paired_schedule else None
        ),
        output_dir=str(args.output_dir.resolve()),
        optimizer_unit=args.optimizer_unit,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        log_every=args.log_every,
        overwrite=args.overwrite,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_rows_from_bytes(path: Path, payload: bytes) -> list[dict[str, Any]]:
    """Decode one immutable JSONL byte snapshot."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TrainingContractError(f"{path}: JSONL is not UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainingContractError(
                f"{path}:{line_number}: invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise TrainingContractError(f"{path}:{line_number}: row is not an object")
        rows.append(row)
    return rows


def read_jsonl_snapshot(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Read a JSONL once and return rows plus the digest of those exact bytes."""

    resolved = path.resolve()
    try:
        payload = resolved.read_bytes()
    except OSError as exc:
        raise TrainingContractError(f"cannot snapshot {resolved}: {exc}") from exc
    return _jsonl_rows_from_bytes(resolved, payload), hashlib.sha256(payload).hexdigest()


def require_snapshot_unchanged(path: Path, expected_sha256: str, *, label: str) -> None:
    """Fail before provenance is written if an input changed after snapshotting."""

    observed = sha256(path.resolve())
    if observed != expected_sha256:
        raise TrainingContractError(
            f"{label} changed after its training snapshot: {path.resolve()}"
        )


def load_scheduled_records_snapshot(
    schedule_path: Path,
) -> tuple[list[ScheduledRecord], str]:
    """Resolve a schedule exclusively from one content-hashed byte snapshot.

    Source JSONLs are likewise decoded from the same bytes whose SHA is checked
    against every schedule row.  This prevents a later hash from accidentally
    binding training performed on a different in-memory schedule.
    """

    schedule_path = schedule_path.resolve()
    schedules, schedule_sha256 = read_jsonl_snapshot(schedule_path)
    source_cache: dict[Path, list[dict[str, Any]]] = {}
    source_sha256_cache: dict[Path, str] = {}
    result: list[ScheduledRecord] = []
    for index, schedule in enumerate(schedules):
        source_name = schedule.get("source_jsonl")
        expected_source_sha256 = schedule.get("source_jsonl_sha256")
        source_line = schedule.get("source_line")
        if not isinstance(source_name, str) or not isinstance(source_line, int):
            raise TrainingContractError(f"schedule row {index} has no source pointer")
        if not (
            isinstance(expected_source_sha256, str)
            and len(expected_source_sha256) == 64
            and all(character in "0123456789abcdef" for character in expected_source_sha256)
        ):
            raise TrainingContractError(
                f"schedule row {index} has no valid source_jsonl_sha256"
            )
        source = Path(source_name)
        if not source.is_absolute():
            source = (schedule_path.parent / source).resolve()
        if source not in source_cache:
            source_cache[source], source_sha256_cache[source] = read_jsonl_snapshot(
                source
            )
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
        model_image_paths = record_model_image_paths(
            record, source_dir=source.parent
        )
        claimed_image_paths = normalized_schedule_image_paths(
            schedule, schedule_dir=schedule_path.parent
        )
        if model_image_paths != claimed_image_paths:
            raise TrainingContractError(
                f"schedule row {index} image_paths disagree with source messages"
            )
        weight = schedule.get("sample_weight")
        if (
            not isinstance(weight, int | float)
            or isinstance(weight, bool)
            or not math.isfinite(float(weight))
            or weight <= 0
        ):
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
    return result, schedule_sha256


def git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def distributed_context(torch: Any) -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def set_seed(torch: Any, seed: int, rank: int) -> None:
    local_seed = seed + rank
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed_all(local_seed)


def build_model(config: TrainConfig, local_rank: int) -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(config.model, local_files_only=True)
    processor.image_processor.size = {
        "shortest_edge": config.image_min_pixels,
        "longest_edge": config.image_max_pixels,
    }
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    )
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    if hasattr(model, "visual"):
        model.visual.requires_grad_(False)
    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    model.to(torch.device("cuda", local_rank))
    return model, processor


def prepare_example(
    processor: Any,
    item: ScheduledRecord,
    device: Any,
) -> tuple[dict[str, Any], list[tuple[str, tuple[int, ...]]]]:
    messages = normalize_qwen_messages(item.record["messages"])
    batch = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    token_ids = batch["input_ids"][0].tolist()
    groups = assistant_token_groups(processor.tokenizer, token_ids, item.record)
    return ({key: value.to(device) for key, value in batch.items()}, groups)


def exact_fact_loss(
    torch: Any,
    logits: Any,
    input_ids: Any,
    groups: list[tuple[str, tuple[int, ...]]],
    sample_weight: float,
    group_reduction: str = "sum",
) -> tuple[Any, list[float]]:
    """Mean within each group, then sum or mean groups and apply weight."""

    import torch.nn.functional as functional

    fact_losses = []
    for _, positions in groups:
        target_positions = torch.tensor(positions, device=input_ids.device, dtype=torch.long)
        if bool((target_positions <= 0).any()):
            raise RuntimeError("assistant target unexpectedly starts at token zero")
        selected_logits = logits[0].index_select(0, target_positions - 1)
        selected_targets = input_ids[0].index_select(0, target_positions)
        fact_losses.append(
            functional.cross_entropy(
                selected_logits.float(), selected_targets, reduction="mean"
            )
        )
    if group_reduction not in {"sum", "mean"}:
        raise RuntimeError(f"unsupported group_reduction {group_reduction!r}")
    stacked = torch.stack(fact_losses)
    total = (stacked.mean() if group_reduction == "mean" else stacked.sum()) * float(
        sample_weight
    )
    return total, [float(value.detach()) for value in fact_losses]


def record_group_reduction(record: Mapping[str, Any]) -> str:
    """Use per-turn mean only when the incremental export explicitly requests it."""

    policy = record.get("loss_policy")
    if (
        isinstance(policy, Mapping)
        and policy.get("normalization") == "mean_tokens_per_turn_then_mean_turns"
    ):
        return "mean"
    return "sum"


def infer_paired_schedule(schedule_path: Path, arm: str) -> Path:
    """Infer the other arm while failing closed on ambiguous filenames."""

    other_arm = "isolated" if arm == "episode" else "episode"
    marker = f".{arm}."
    if marker not in schedule_path.name:
        raise RuntimeError(
            "cannot infer paired schedule: filename must contain "
            f"{marker!r}; pass --paired-schedule explicitly"
        )
    candidate = schedule_path.with_name(
        schedule_path.name.replace(marker, f".{other_arm}.", 1)
    )
    if not candidate.is_file():
        raise FileNotFoundError(
            f"inferred paired schedule does not exist: {candidate}; "
            "pass --paired-schedule explicitly"
        )
    return candidate


def train_comparison_groups(
    *,
    torch: Any,
    model: Any,
    processor: Any,
    groups: dict[str, ComparisonGroup],
    optimizer: Any,
    scheduler: Any,
    trainable: list[Any],
    config: TrainConfig,
    local_rank: int,
    output_dir: Path,
) -> tuple[int, int, int, float]:
    """Train with exactly one optimizer update per comparison group."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    update_step = 0
    micro_step = 0
    completed_groups = 0
    start_time = time.time()
    log_path = output_dir / "train_log.jsonl"
    device = torch.device("cuda", local_rank)
    for epoch in range(config.epochs):
        for comparison_id in ordered_comparison_ids(
            groups, seed=config.seed, epoch=epoch
        ):
            group = groups[comparison_id]
            group_weighted_loss = 0.0
            group_fact_losses: list[float] = []
            for item in group.items:
                batch, fact_token_groups = prepare_example(processor, item, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(**batch, use_cache=False)
                    weighted_loss, fact_losses = exact_fact_loss(
                        torch,
                        output.logits,
                        batch["input_ids"],
                        fact_token_groups,
                        float(item.schedule["sample_weight"]),
                        record_group_reduction(item.record),
                    )
                weighted_loss.backward()
                micro_step += 1
                group_weighted_loss += float(weighted_loss.detach())
                group_fact_losses.extend(fact_losses)
                # Release logits before the next variable-size multimodal forward.
                del batch, output, weighted_loss

            torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_step += 1
            completed_groups += 1

            if update_step == 1 or update_step % config.log_every == 0:
                event = {
                    "update_step": update_step,
                    "micro_step": micro_step,
                    "epoch": epoch,
                    "comparison_id": comparison_id,
                    "group_draws": group.draw_count,
                    "group_facts": len(group.fact_ids),
                    "group_weighted_loss": group_weighted_loss,
                    "mean_observed_fact_loss": (
                        sum(group_fact_losses) / len(group_fact_losses)
                    ),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "elapsed_seconds": time.time() - start_time,
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                print(json.dumps(event, ensure_ascii=False), flush=True)

            if config.max_steps is not None and update_step >= config.max_steps:
                return update_step, micro_step, completed_groups, time.time() - start_time
    return update_step, micro_step, completed_groups, time.time() - start_time


def train_legacy_draws(
    *,
    torch: Any,
    model: Any,
    processor: Any,
    scheduled: list[ScheduledRecord],
    optimizer: Any,
    scheduler: Any,
    trainable: list[Any],
    config: TrainConfig,
    rank: int,
    local_rank: int,
    world_size: int,
    output_dir: Path,
) -> tuple[int, int, float]:
    """Legacy per-draw updates; retained only as a non-causal diagnostic."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    update_step = 0
    micro_step = 0
    start_time = time.time()
    log_path = output_dir / "train_log.jsonl"
    for epoch in range(config.epochs):
        order = list(range(len(scheduled)))
        random.Random(config.seed + epoch).shuffle(order)
        local_order = order[rank::world_size]
        for local_index, schedule_index in enumerate(local_order):
            item = scheduled[schedule_index]
            batch, groups = prepare_example(
                processor,
                item,
                torch.device("cuda", local_rank),
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**batch, use_cache=False)
                weighted_loss, fact_losses = exact_fact_loss(
                    torch,
                    output.logits,
                    batch["input_ids"],
                    groups,
                    float(item.schedule["sample_weight"]),
                    record_group_reduction(item.record),
                )
                backward_loss = weighted_loss / config.gradient_accumulation_steps
            backward_loss.backward()
            micro_step += 1
            is_epoch_end = local_index + 1 == len(local_order)
            should_update = (
                micro_step % config.gradient_accumulation_steps == 0 or is_epoch_end
            )
            if should_update:
                torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1

                if rank == 0 and (
                    update_step == 1 or update_step % config.log_every == 0
                ):
                    event = {
                        "update_step": update_step,
                        "micro_step": micro_step,
                        "epoch": epoch,
                        "record_id": item.record["record_id"],
                        "fact_count": len(groups),
                        "sample_weight": item.schedule["sample_weight"],
                        "weighted_loss": float(weighted_loss.detach()),
                        "mean_fact_loss": sum(fact_losses) / len(fact_losses),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "elapsed_seconds": time.time() - start_time,
                    }
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                    print(json.dumps(event, ensure_ascii=False), flush=True)
            stop = config.max_steps is not None and update_step >= config.max_steps
            del batch, backward_loss, output, weighted_loss
            if stop:
                return update_step, micro_step, time.time() - start_time
    return update_step, micro_step, time.time() - start_time


def main() -> None:
    config = parse_args()
    base_model_provenance = verify_inventory_artifact(
        Path(config.model_inventory), root=Path(config.model), kind="model"
    )
    import torch
    from torch.nn.parallel import DistributedDataParallel
    from transformers import get_cosine_schedule_with_warmup

    requested_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if config.optimizer_unit == "comparison_group":
        if config.gradient_accumulation_steps != 1:
            raise RuntimeError(
                "comparison_group mode defines one optimizer update per group; "
                "--gradient-accumulation-steps must be 1"
            )
        if requested_world_size != 1:
            raise RuntimeError(
                "comparison_group mode currently requires one process/GPU. "
                "DDP would require group-wise partitioning plus no_sync and is "
                "refused rather than silently changing the optimizer contract."
            )

    rank, local_rank, world_size = distributed_context(torch)
    set_seed(torch, config.seed, rank)
    schedule_path = Path(config.schedule)
    scheduled, schedule_snapshot_sha256 = load_scheduled_records_snapshot(
        schedule_path
    )
    if config.optimizer_unit == "draw" and len(scheduled) % world_size:
        raise RuntimeError(
            f"{len(scheduled)} draws are not divisible by world_size={world_size}; "
            "refusing silent sampler padding"
        )
    comparison_groups: dict[str, ComparisonGroup] = {}
    paired_schedule_path: Path | None = None
    paired_validation: dict[str, int] | None = None
    if config.optimizer_unit == "comparison_group":
        comparison_groups = build_comparison_groups(scheduled)
        arm = next(iter(comparison_groups.values())).arm
        paired_schedule_path = (
            Path(config.paired_schedule)
            if config.paired_schedule is not None
            else infer_paired_schedule(schedule_path, arm)
        )
        paired_scheduled, paired_schedule_snapshot_sha256 = (
            load_scheduled_records_snapshot(paired_schedule_path)
        )
        paired_groups = build_comparison_groups(paired_scheduled)
        paired_validation = validate_paired_comparison_groups(
            comparison_groups, paired_groups
        )
    else:
        paired_schedule_snapshot_sha256 = None
    output_dir = Path(config.output_dir)
    if rank == 0:
        if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
            raise FileExistsError(f"non-empty output exists: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        torch.distributed.barrier()

    model, processor = build_model(config, local_rank)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    local_draws_per_epoch = len(scheduled) // world_size
    if config.optimizer_unit == "comparison_group":
        planned_updates = len(comparison_groups) * config.epochs
    else:
        if local_draws_per_epoch % config.gradient_accumulation_steps:
            raise RuntimeError(
                "local draws per epoch must be divisible by gradient accumulation; "
                "refusing a partially scaled final update"
            )
        micro_steps = local_draws_per_epoch * config.epochs
        planned_updates = math.ceil(micro_steps / config.gradient_accumulation_steps)
    if config.max_steps is not None:
        planned_updates = min(planned_updates, config.max_steps)
    warmup_steps = round(planned_updates * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=planned_updates,
    )
    if config.optimizer_unit == "draw" and world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    if config.optimizer_unit == "comparison_group":
        update_step, micro_step, completed_groups, elapsed_seconds = (
            train_comparison_groups(
                torch=torch,
                model=model,
                processor=processor,
                groups=comparison_groups,
                optimizer=optimizer,
                scheduler=scheduler,
                trainable=trainable,
                config=config,
                local_rank=local_rank,
                output_dir=output_dir,
            )
        )
    else:
        update_step, micro_step, elapsed_seconds = train_legacy_draws(
            torch=torch,
            model=model,
            processor=processor,
            scheduled=scheduled,
            optimizer=optimizer,
            scheduler=scheduler,
            trainable=trainable,
            config=config,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            output_dir=output_dir,
        )
        completed_groups = 0

    if world_size > 1:
        torch.distributed.barrier()
    if rank == 0:
        unwrapped = model.module if hasattr(model, "module") else model
        adapter_dir = output_dir / "adapter"
        unwrapped.save_pretrained(adapter_dir)
        processor.save_pretrained(output_dir / "processor")
        adapter_provenance = write_inventory_artifact(
            adapter_dir,
            output_dir / "adapter_inventory.json",
            kind="directory",
            overwrite=config.overwrite,
        )
        require_snapshot_unchanged(
            schedule_path,
            schedule_snapshot_sha256,
            label="training schedule",
        )
        if (
            paired_schedule_path is not None
            and paired_schedule_snapshot_sha256 is not None
        ):
            require_snapshot_unchanged(
                paired_schedule_path,
                paired_schedule_snapshot_sha256,
                label="paired training schedule",
            )
        manifest = {
            "schema_version": "epispace.qwen3vl_lora_run.v2",
            "status": "smoke" if config.max_steps is not None else "complete",
            "config": asdict(config),
            "base_model_inventory_sha256": base_model_provenance[
                "inventory_sha256"
            ],
            "adapter_inventory_sha256": adapter_provenance["inventory_sha256"],
            "base_model_provenance": base_model_provenance,
            "adapter_provenance": adapter_provenance,
            "schedule_sha256": schedule_snapshot_sha256,
            "paired_schedule_sha256": paired_schedule_snapshot_sha256,
            "draws": len(scheduled),
            "optimizer_unit": config.optimizer_unit,
            "comparison_groups": len(comparison_groups),
            "groups_per_epoch": (
                len(comparison_groups)
                if config.optimizer_unit == "comparison_group"
                else None
            ),
            "paired_group_validation": paired_validation,
            "world_size": world_size,
            "local_draws_per_epoch": local_draws_per_epoch,
            "planned_optimizer_updates": planned_updates,
            "optimizer_updates": update_step,
            "completed_comparison_groups": completed_groups,
            "completed_micro_steps_per_rank": micro_step,
            "loss_contract": (
                "mean CE within each fact; sum facts; multiply schedule sample_weight; "
                "in comparison_group mode accumulate all draws for one comparison_id "
                "before exactly one optimizer step"
            ),
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
            "total_parameters": sum(parameter.numel() for parameter in unwrapped.parameters()),
            "elapsed_seconds": elapsed_seconds,
            "git": git_state(Path(__file__).resolve().parents[1]),
            "torch_version": torch.__version__,
        }
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, ensure_ascii=False), flush=True)
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
