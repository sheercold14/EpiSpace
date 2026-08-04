#!/usr/bin/env python3
"""Generate auditable Qwen3-VL predictions for an EpiSpace benchmark shard."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from episode3d.model_output_parser import parse_model_output  # noqa: E402
from episode3d.pilot_summary import verify_inventory_artifact  # noqa: E402
from episode3d.qwen_training import TrainingContractError, normalize_qwen_messages  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--model-inventory",
        type=Path,
        required=True,
        help="Reusable content inventory for the base model weights.",
    )
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--adapter-inventory",
        type=Path,
        help="Required content inventory when --adapter is supplied.",
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--image-min-pixels", type=int, default=65_536)
    parser.add_argument("--image-max-pixels", type=int, default=262_144)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--image-mode", choices=("full", "none"), default="full")
    parser.add_argument(
        "--prompt-mode",
        choices=("native", "episode_single"),
        default="native",
        help="Optional format diagnostic; episode_single wraps the same question as item 1.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must satisfy 0 <= index < num_shards")
    if args.image_min_pixels > args.image_max_pixels:
        parser.error("image_min_pixels cannot exceed image_max_pixels")
    if (args.adapter is None) != (args.adapter_inventory is None):
        parser.error("--adapter and --adapter-inventory must be supplied together")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_benchmark_snapshot(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Parse a benchmark from the exact bytes represented by its returned SHA."""

    resolved = path.resolve()
    try:
        payload = resolved.read_bytes()
    except OSError as exc:
        raise TrainingContractError(f"cannot snapshot {resolved}: {exc}") from exc
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TrainingContractError(f"{resolved}: benchmark is not UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainingContractError(
                f"{resolved}:{line_number}: invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise TrainingContractError(
                f"{resolved}:{line_number}: row is not an object"
            )
        rows.append(row)
    return rows, hashlib.sha256(payload).hexdigest()


def require_benchmark_snapshot_unchanged(
    path: Path, expected_sha256: str
) -> None:
    """Reject provenance if the benchmark drifted during inference."""

    observed = sha256(path.resolve())
    if observed != expected_sha256:
        raise TrainingContractError(
            f"benchmark changed after its inference snapshot: {path.resolve()}"
        )


def output_paths(output_dir: Path, shard_index: int, num_shards: int) -> tuple[Path, Path]:
    if num_shards == 1:
        return output_dir / "predictions.jsonl", output_dir / "raw_generations.jsonl"
    suffix = f"shard-{shard_index:05d}-of-{num_shards:05d}.jsonl"
    return output_dir / f"predictions.{suffix}", output_dir / f"raw_generations.{suffix}"


def prediction_artifact_bindings(
    predictions_path: Path, raw_generations_path: Path
) -> dict[str, str]:
    """Bind a prediction manifest to the exact two output artifacts."""

    return {
        "predictions": str(predictions_path.resolve()),
        "predictions_sha256": sha256(predictions_path),
        "raw_generations": str(raw_generations_path.resolve()),
        "raw_generations_sha256": sha256(raw_generations_path),
    }


def apply_prompt_mode(model_input: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    """Change only question presentation for an explicit format-shift diagnostic."""

    messages = copy.deepcopy(model_input)
    if mode == "native":
        return messages
    if mode != "episode_single":
        raise ValueError(f"unsupported prompt mode: {mode}")
    user = messages[-1]
    if user.get("role") != "user" or not isinstance(user.get("content"), list):
        raise ValueError("benchmark does not end in a multimodal user turn")
    text_items = [
        item
        for item in user["content"]
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    if not text_items:
        raise ValueError("benchmark user turn has no text question")
    question_item = text_items[-1]
    question = str(question_item.get("text", "")).strip()
    if not question:
        raise ValueError("benchmark question is empty")
    question_item["text"] = (
        "观察阶段到此结束。请依次回答下列问题：\n"
        f"1. {question}"
    )
    return messages


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_path = args.model.resolve()
    benchmark_path = args.benchmark.resolve()
    benchmark, benchmark_snapshot_sha256 = read_benchmark_snapshot(benchmark_path)
    base_model_provenance = verify_inventory_artifact(
        args.model_inventory, root=model_path, kind="model"
    )
    adapter_path = args.adapter.resolve() if args.adapter else None
    adapter_provenance = (
        verify_inventory_artifact(
            args.adapter_inventory,
            root=adapter_path,
            kind="directory",
            verify_content=True,
        )
        if adapter_path is not None and args.adapter_inventory is not None
        else None
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path, raw_path = output_paths(
        output_dir, args.shard_index, args.num_shards
    )
    for path in (predictions_path, raw_path):
        if path.exists() and path.stat().st_size and not args.overwrite:
            raise FileExistsError(f"output exists: {path}")

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.image_processor.size = {
        "shortest_edge": args.image_min_pixels,
        "longest_edge": args.image_max_pixels,
    }
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.cuda().eval()

    selected = [
        row
        for index, row in enumerate(benchmark)
        if index % args.num_shards == args.shard_index
    ]
    predictions: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    start = time.time()
    for local_index, row in enumerate(selected, 1):
        model_input = apply_prompt_mode(row["model_input"], args.prompt_mode)
        if args.image_mode == "none":
            model_input = [
                {
                    **message,
                    "content": [
                        item
                        for item in message["content"]
                        if not isinstance(item, dict) or item.get("type") != "image"
                    ]
                    if isinstance(message.get("content"), list)
                    else message.get("content"),
                }
                for message in model_input
            ]
        messages = normalize_qwen_messages(model_input)
        batch = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        continuation = generated[:, batch["input_ids"].shape[1] :]
        text = processor.batch_decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        prediction = parse_model_output(row, text)
        predictions.append(prediction)
        raw_rows.append(
            {
                "record_id": row["record_id"],
                "model_output": text,
                "parse_status": prediction["parse_status"],
                "parse_error": prediction.get("parse_error"),
                "input_tokens": int(batch["input_ids"].shape[1]),
                "generated_tokens": int(continuation.shape[1]),
            }
        )
        if local_index == 1 or local_index % 10 == 0:
            print(
                json.dumps(
                    {
                        "shard": args.shard_index,
                        "completed": local_index,
                        "total": len(selected),
                        "parsed": sum(
                            item["parse_status"] == "parsed" for item in predictions
                        ),
                        "elapsed_seconds": time.time() - start,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    predictions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    raw_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in raw_rows),
        encoding="utf-8",
    )
    require_benchmark_snapshot_unchanged(
        benchmark_path, benchmark_snapshot_sha256
    )
    manifest = {
        "schema_version": "epispace.qwen3vl_predictions.v1",
        "model": str(model_path),
        "adapter": str(adapter_path) if adapter_path else None,
        "base_model_inventory_sha256": base_model_provenance[
            "inventory_sha256"
        ],
        "adapter_inventory_sha256": (
            adapter_provenance["inventory_sha256"]
            if adapter_provenance is not None
            else None
        ),
        "base_model_provenance": base_model_provenance,
        "adapter_provenance": adapter_provenance,
        "benchmark": str(benchmark_path),
        "benchmark_sha256": benchmark_snapshot_sha256,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "records": len(selected),
        "parsed": sum(item["parse_status"] == "parsed" for item in predictions),
        "image_min_pixels": args.image_min_pixels,
        "image_max_pixels": args.image_max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "image_mode": args.image_mode,
        "prompt_mode": args.prompt_mode,
        "elapsed_seconds": time.time() - start,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        **prediction_artifact_bindings(predictions_path, raw_path),
    }
    manifest_name = (
        "prediction_manifest.json"
        if args.num_shards == 1
        else f"prediction_manifest.shard-{args.shard_index:05d}-of-{args.num_shards:05d}.json"
    )
    (output_dir / manifest_name).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
