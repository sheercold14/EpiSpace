#!/usr/bin/env python3
"""Merge deterministic inference shards and prove exact benchmark coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from episode3d.pilot_summary import (  # noqa: E402
    PilotSummaryError,
    validate_inventory_binding,
)
from episode3d.qwen_training import TrainingContractError, read_jsonl  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _declared_shard_path(
    manifest: dict,
    *,
    key: str,
    expected: Path,
    shard_index: int,
) -> Path:
    value = manifest.get(key)
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise TrainingContractError(
            f"shard {shard_index} {key} path must be an explicit absolute path"
        )
    declared = Path(value).resolve()
    if declared != expected.resolve():
        raise TrainingContractError(f"shard {shard_index} declared {key} path mismatch")
    if not declared.is_file():
        raise TrainingContractError(f"shard {shard_index} declared {key} file is absent")
    expected_sha256 = manifest.get(f"{key}_sha256")
    actual_sha256 = sha256(declared)
    if expected_sha256 != actual_sha256:
        raise TrainingContractError(f"shard {shard_index} {key} SHA-256 mismatch")
    return declared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()

    benchmark_path = args.benchmark.resolve()
    benchmark = read_jsonl(benchmark_path)
    expected_ids = [row.get("record_id") for row in benchmark]
    if not all(isinstance(record_id, str) and record_id for record_id in expected_ids):
        raise TrainingContractError("benchmark contains an invalid record_id")
    if len(set(expected_ids)) != len(expected_ids):
        raise TrainingContractError("benchmark contains duplicate record IDs")
    benchmark_sha256 = sha256(benchmark_path)
    shard_dir = args.shard_dir.resolve()
    manifest_files = sorted(shard_dir.glob("prediction_manifest.shard-*.json"))
    if not manifest_files:
        raise TrainingContractError("prediction shard manifests are absent")
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in manifest_files]
    shard_count = len(manifests)
    shard_indices = [item.get("shard_index") for item in manifests]
    if not all(
        isinstance(index, int) and not isinstance(index, bool) for index in shard_indices
    ) or set(shard_indices) != set(range(shard_count)):
        raise TrainingContractError("shard indices are incomplete or duplicated")
    if any(item.get("num_shards") != shard_count for item in manifests):
        raise TrainingContractError("shard manifests disagree on num_shards")
    manifest_by_index = {
        int(manifest["shard_index"]): (path.resolve(), manifest)
        for path, manifest in zip(manifest_files, manifests, strict=True)
    }
    expected_manifest_paths = {
        (
            shard_dir
            / f"prediction_manifest.shard-{index:05d}-of-{shard_count:05d}.json"
        ).resolve()
        for index in range(shard_count)
    }
    if {path.resolve() for path in manifest_files} != expected_manifest_paths:
        raise TrainingContractError("shard manifest filenames do not match shard_index")

    expected_prediction_paths = {
        (
            shard_dir / f"predictions.shard-{index:05d}-of-{shard_count:05d}.jsonl"
        ).resolve()
        for index in range(shard_count)
    }
    expected_raw_paths = {
        (
            shard_dir
            / f"raw_generations.shard-{index:05d}-of-{shard_count:05d}.jsonl"
        ).resolve()
        for index in range(shard_count)
    }
    if {
        path.resolve() for path in shard_dir.glob("predictions.shard-*.jsonl")
    } != expected_prediction_paths:
        raise TrainingContractError("prediction shard files do not exactly match manifests")
    if {
        path.resolve() for path in shard_dir.glob("raw_generations.shard-*.jsonl")
    } != expected_raw_paths:
        raise TrainingContractError("raw-generation shard files do not exactly match manifests")

    invariant_keys = (
        "schema_version",
        "model",
        "adapter",
        "base_model_inventory_sha256",
        "adapter_inventory_sha256",
        "base_model_provenance",
        "adapter_provenance",
        "benchmark",
        "benchmark_sha256",
        "image_min_pixels",
        "image_max_pixels",
        "max_new_tokens",
        "image_mode",
        "prompt_mode",
    )
    for key in invariant_keys:
        values = {json.dumps(item.get(key), sort_keys=True) for item in manifests}
        if len(values) != 1:
            raise TrainingContractError(f"shard manifest mismatch for {key}")
    first = manifest_by_index[0][1]
    predictions: list[dict] = []
    raw_rows: list[dict] = []
    ordered_manifest_files: list[Path] = []
    for shard_index in range(shard_count):
        manifest_path, manifest = manifest_by_index[shard_index]
        ordered_manifest_files.append(manifest_path)
        declared_benchmark = manifest.get("benchmark")
        if (
            not isinstance(declared_benchmark, str)
            or not declared_benchmark
            or not Path(declared_benchmark).is_absolute()
        ):
            raise TrainingContractError(
                f"shard {shard_index} benchmark path must be an explicit absolute path"
            )
        if Path(declared_benchmark).resolve() != benchmark_path:
            raise TrainingContractError(f"shard {shard_index} benchmark path mismatch")
        if manifest.get("benchmark_sha256") != benchmark_sha256:
            raise TrainingContractError(f"shard {shard_index} benchmark SHA-256 mismatch")

        suffix = f"shard-{shard_index:05d}-of-{shard_count:05d}.jsonl"
        prediction_path = _declared_shard_path(
            manifest,
            key="predictions",
            expected=shard_dir / f"predictions.{suffix}",
            shard_index=shard_index,
        )
        raw_path = _declared_shard_path(
            manifest,
            key="raw_generations",
            expected=shard_dir / f"raw_generations.{suffix}",
            shard_index=shard_index,
        )
        shard_predictions = read_jsonl(prediction_path)
        shard_raw = read_jsonl(raw_path)
        expected_shard_ids = [
            record_id
            for index, record_id in enumerate(expected_ids)
            if index % shard_count == shard_index
        ]
        prediction_ids = [row.get("record_id") for row in shard_predictions]
        raw_ids = [row.get("record_id") for row in shard_raw]
        if prediction_ids != expected_shard_ids:
            raise TrainingContractError(
                f"shard {shard_index} prediction IDs violate benchmark shard assignment"
            )
        if raw_ids != expected_shard_ids:
            raise TrainingContractError(
                f"shard {shard_index} raw IDs violate benchmark shard assignment"
            )
        if manifest.get("records") != len(shard_predictions) or len(shard_raw) != len(
            shard_predictions
        ):
            raise TrainingContractError(f"shard {shard_index} record count mismatch")
        parsed = sum(row.get("parse_status") == "parsed" for row in shard_predictions)
        if manifest.get("parsed") != parsed:
            raise TrainingContractError(f"shard {shard_index} parsed count mismatch")
        predictions.extend(shard_predictions)
        raw_rows.extend(shard_raw)

    try:
        model_binding = validate_inventory_binding(
            first.get("base_model_provenance"),
            root=Path(str(first.get("model", ""))),
            kind="model",
            expected_inventory_sha256=first.get("base_model_inventory_sha256"),
        )
        adapter_value = first.get("adapter")
        if adapter_value is None:
            if first.get("adapter_inventory_sha256") is not None or first.get(
                "adapter_provenance"
            ) is not None:
                raise TrainingContractError(
                    "adapter-free shards unexpectedly declare adapter provenance"
                )
            adapter_binding = None
        else:
            adapter_binding = validate_inventory_binding(
                first.get("adapter_provenance"),
                root=Path(str(adapter_value)),
                kind="directory",
                expected_inventory_sha256=first.get("adapter_inventory_sha256"),
                verify_content=True,
            )
    except PilotSummaryError as exc:
        raise TrainingContractError(str(exc)) from exc
    predicted_ids = [row.get("record_id") for row in predictions]
    raw_ids = [row.get("record_id") for row in raw_rows]
    duplicate_ids = sorted(
        record_id
        for record_id, count in Counter(predicted_ids).items()
        if count != 1
    )
    if duplicate_ids:
        raise TrainingContractError(f"duplicate prediction IDs: {duplicate_ids[:5]}")
    if set(predicted_ids) != set(expected_ids) or set(raw_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(predicted_ids))
        extra = sorted(set(predicted_ids) - set(expected_ids))
        raise TrainingContractError(
            f"benchmark coverage mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    prediction_by_id = {row["record_id"]: row for row in predictions}
    raw_by_id = {row["record_id"]: row for row in raw_rows}
    ordered_predictions = [prediction_by_id[record_id] for record_id in expected_ids]
    ordered_raw = [raw_by_id[record_id] for record_id in expected_ids]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in ordered_predictions
        ),
        encoding="utf-8",
    )
    args.raw_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered_raw),
        encoding="utf-8",
    )
    manifest_output = args.manifest_output or args.output.parent / "prediction_manifest.json"
    merged_manifest = {
        "schema_version": "epispace.qwen3vl_predictions.v1",
        "model": first["model"],
        "adapter": first.get("adapter"),
        "base_model_inventory_sha256": model_binding["inventory_sha256"],
        "adapter_inventory_sha256": (
            adapter_binding["inventory_sha256"] if adapter_binding else None
        ),
        "base_model_provenance": first["base_model_provenance"],
        "adapter_provenance": first.get("adapter_provenance"),
        "benchmark": first["benchmark"],
        "benchmark_sha256": first["benchmark_sha256"],
        "shard_index": 0,
        "num_shards": 1,
        "merged_from_num_shards": shard_count,
        "records": len(ordered_predictions),
        "parsed": sum(
            row.get("parse_status") == "parsed" for row in ordered_predictions
        ),
        "image_min_pixels": first["image_min_pixels"],
        "image_max_pixels": first["image_max_pixels"],
        "max_new_tokens": first["max_new_tokens"],
        "image_mode": first.get("image_mode", "full"),
        "prompt_mode": first.get("prompt_mode", "native"),
        "elapsed_seconds_parallel_wall_proxy": max(
            float(item.get("elapsed_seconds", 0.0)) for item in manifests
        ),
        "elapsed_seconds_sum_gpu_processes": sum(
            float(item.get("elapsed_seconds", 0.0)) for item in manifests
        ),
        "peak_cuda_bytes_max": max(
            int(item.get("peak_cuda_bytes", 0)) for item in manifests
        ),
        "predictions": str(args.output.resolve()),
        "predictions_sha256": sha256(args.output),
        "raw_generations": str(args.raw_output.resolve()),
        "raw_generations_sha256": sha256(args.raw_output),
        "shard_manifests": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in ordered_manifest_files
        ],
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(merged_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "records": len(ordered_predictions),
                "parsed": sum(
                    row.get("parse_status") == "parsed" for row in ordered_predictions
                ),
                "output": str(args.output),
                "raw_output": str(args.raw_output),
                "manifest_output": str(manifest_output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
