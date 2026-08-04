from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from scripts.merge_qwen_predictions import main as merge_predictions
from scripts.predict_qwen3vl import (
    apply_prompt_mode,
    prediction_artifact_bindings,
    read_benchmark_snapshot,
    require_benchmark_snapshot_unchanged,
)

from episode3d.pilot_summary import write_inventory_artifact
from episode3d.qwen_training import TrainingContractError


def _messages() -> list[dict]:
    return [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<image-1>"},
                {"type": "image", "image": "/tmp/view.png"},
                {"type": "text", "text": "钢琴存在吗？"},
            ],
        },
    ]


def test_benchmark_snapshot_binds_parsed_rows_and_rejects_later_drift(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "benchmark.jsonl"
    original = {"record_id": "record-original", "model_input": []}
    benchmark.write_text(json.dumps(original) + "\n", encoding="utf-8")

    rows, snapshot_sha256 = read_benchmark_snapshot(benchmark)
    assert rows == [original]
    assert snapshot_sha256 == hashlib.sha256(benchmark.read_bytes()).hexdigest()

    benchmark.write_text(
        json.dumps({"record_id": "record-replaced", "model_input": []}) + "\n",
        encoding="utf-8",
    )
    assert rows == [original]
    with pytest.raises(TrainingContractError, match="changed after its inference snapshot"):
        require_benchmark_snapshot_unchanged(benchmark, snapshot_sha256)


def test_episode_single_wraps_only_the_question_and_does_not_mutate_source() -> None:
    source = _messages()
    wrapped = apply_prompt_mode(source, "episode_single")
    assert source[-1]["content"][-1]["text"] == "钢琴存在吗？"
    assert wrapped[-1]["content"][:-1] == source[-1]["content"][:-1]
    assert wrapped[-1]["content"][-1]["text"] == (
        "观察阶段到此结束。请依次回答下列问题：\n1. 钢琴存在吗？"
    )


def test_native_is_content_equal_but_independent() -> None:
    source = _messages()
    native = apply_prompt_mode(source, "native")
    assert native == source
    assert native is not source


def test_prediction_artifact_bindings_include_paths_and_content_hashes(
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "predictions.jsonl"
    raw_generations = tmp_path / "raw_generations.jsonl"
    predictions.write_bytes(b'{"record_id":"record-1"}\n')
    raw_generations.write_bytes(b'{"record_id":"record-1","model_output":"yes"}\n')

    bindings = prediction_artifact_bindings(predictions, raw_generations)

    assert bindings == {
        "predictions": str(predictions.resolve()),
        "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
        "raw_generations": str(raw_generations.resolve()),
        "raw_generations_sha256": hashlib.sha256(
            raw_generations.read_bytes()
        ).hexdigest(),
    }


def test_prediction_artifact_hash_changes_when_output_changes(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    raw_generations = tmp_path / "raw_generations.jsonl"
    predictions.write_text("before\n", encoding="utf-8")
    raw_generations.write_text("raw\n", encoding="utf-8")
    before = prediction_artifact_bindings(predictions, raw_generations)

    predictions.write_text("after\n", encoding="utf-8")
    after = prediction_artifact_bindings(predictions, raw_generations)

    assert before["predictions_sha256"] != after["predictions_sha256"]
    assert before["raw_generations_sha256"] == after["raw_generations_sha256"]


def _merge_fixture(tmp_path: Path) -> dict[str, object]:
    benchmark = tmp_path / "benchmark.jsonl"
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3_vl"}\n')
    (model / "model.safetensors").write_bytes(b"weights")
    model_inventory_path = tmp_path / "model_inventory.json"
    model_provenance = write_inventory_artifact(
        model, model_inventory_path, kind="model"
    )
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    benchmark_rows = [{"record_id": "record-0"}, {"record_id": "record-1"}]
    benchmark.write_text(
        "".join(json.dumps(row) + "\n" for row in benchmark_rows), encoding="utf-8"
    )
    manifests: list[Path] = []
    predictions_by_shard: list[Path] = []
    raw_by_shard: list[Path] = []
    for shard_index, row in enumerate(benchmark_rows):
        suffix = f"shard-{shard_index:05d}-of-00002"
        predictions = shard_dir / f"predictions.{suffix}.jsonl"
        raw_generations = shard_dir / f"raw_generations.{suffix}.jsonl"
        predictions.write_text(
            json.dumps({**row, "prediction": "yes", "parse_status": "parsed"})
            + "\n",
            encoding="utf-8",
        )
        raw_generations.write_text(
            json.dumps({**row, "model_output": "yes"}) + "\n", encoding="utf-8"
        )
        manifest = {
            "schema_version": "epispace.qwen3vl_predictions.v1",
            "model": str(model.resolve()),
            "adapter": None,
            "base_model_inventory_sha256": model_provenance[
                "inventory_sha256"
            ],
            "adapter_inventory_sha256": None,
            "base_model_provenance": model_provenance,
            "adapter_provenance": None,
            "benchmark": str(benchmark.resolve()),
            "benchmark_sha256": hashlib.sha256(benchmark.read_bytes()).hexdigest(),
            "shard_index": shard_index,
            "num_shards": 2,
            "records": 1,
            "parsed": 1,
            "image_min_pixels": 65_536,
            "image_max_pixels": 262_144,
            "max_new_tokens": 96,
            "image_mode": "full",
            "prompt_mode": "native",
            **prediction_artifact_bindings(predictions, raw_generations),
        }
        manifest_path = shard_dir / f"prediction_manifest.{suffix}.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        manifests.append(manifest_path)
        predictions_by_shard.append(predictions)
        raw_by_shard.append(raw_generations)
    return {
        "benchmark": benchmark,
        "model_provenance": model_provenance,
        "shard_dir": shard_dir,
        "manifests": manifests,
        "predictions": predictions_by_shard,
        "raw": raw_by_shard,
    }


def _set_merge_argv(
    monkeypatch,
    *,
    benchmark: Path,
    shard_dir: Path,
    output: Path,
    raw_output: Path,
    manifest_output: Path | None = None,
) -> None:
    argv = [
        "merge_qwen_predictions.py",
        "--benchmark",
        str(benchmark),
        "--shard-dir",
        str(shard_dir),
        "--output",
        str(output),
        "--raw-output",
        str(raw_output),
    ]
    if manifest_output is not None:
        argv.extend(["--manifest-output", str(manifest_output)])
    monkeypatch.setattr(sys, "argv", argv)


def test_hashed_shard_manifests_remain_merge_compatible(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _merge_fixture(tmp_path)
    benchmark = fixture["benchmark"]
    shard_dir = fixture["shard_dir"]
    model_provenance = fixture["model_provenance"]
    assert isinstance(benchmark, Path)
    assert isinstance(shard_dir, Path)
    assert isinstance(model_provenance, dict)

    output = tmp_path / "predictions.jsonl"
    raw_output = tmp_path / "raw_generations.jsonl"
    manifest_output = tmp_path / "prediction_manifest.json"
    _set_merge_argv(
        monkeypatch,
        benchmark=benchmark,
        shard_dir=shard_dir,
        output=output,
        raw_output=raw_output,
        manifest_output=manifest_output,
    )

    merge_predictions()

    merged = json.loads(manifest_output.read_text(encoding="utf-8"))
    assert merged["records"] == 2
    assert merged["predictions_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert merged["raw_generations_sha256"] == hashlib.sha256(
        raw_output.read_bytes()
    ).hexdigest()
    assert (
        merged["base_model_inventory_sha256"]
        == model_provenance["inventory_sha256"]
    )
    assert merged["adapter_inventory_sha256"] is None


@pytest.mark.parametrize(
    ("tamper", "error"),
    [
        ("predictions_path", "declared predictions path mismatch"),
        ("raw_path", "declared raw_generations path mismatch"),
        ("predictions_sha256", "predictions SHA-256 mismatch"),
        ("raw_sha256", "raw_generations SHA-256 mismatch"),
        ("records", "record count mismatch"),
        ("parsed", "parsed count mismatch"),
        ("prediction_assignment", "prediction IDs violate benchmark shard assignment"),
        ("raw_assignment", "raw IDs violate benchmark shard assignment"),
        ("benchmark_path", "benchmark path must be an explicit absolute path"),
        ("benchmark_sha256", "benchmark SHA-256 mismatch"),
    ],
)
def test_merge_rejects_shard_manifest_or_assignment_tampering(
    tmp_path: Path,
    monkeypatch,
    tamper: str,
    error: str,
) -> None:
    fixture = _merge_fixture(tmp_path)
    benchmark = fixture["benchmark"]
    shard_dir = fixture["shard_dir"]
    manifests = fixture["manifests"]
    predictions = fixture["predictions"]
    raw = fixture["raw"]
    assert isinstance(benchmark, Path)
    assert isinstance(shard_dir, Path)
    assert isinstance(manifests, list)
    assert isinstance(predictions, list)
    assert isinstance(raw, list)

    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if tamper == "predictions_path":
        manifest["predictions"] = str(predictions[1].resolve())
    elif tamper == "raw_path":
        manifest["raw_generations"] = str(raw[1].resolve())
    elif tamper == "predictions_sha256":
        manifest["predictions_sha256"] = "0" * 64
    elif tamper == "raw_sha256":
        manifest["raw_generations_sha256"] = "0" * 64
    elif tamper == "records":
        manifest["records"] = 2
    elif tamper == "parsed":
        manifest["parsed"] = 0
    elif tamper == "prediction_assignment":
        row = json.loads(predictions[0].read_text(encoding="utf-8"))
        row["record_id"] = "record-1"
        predictions[0].write_text(json.dumps(row) + "\n", encoding="utf-8")
        manifest["predictions_sha256"] = hashlib.sha256(
            predictions[0].read_bytes()
        ).hexdigest()
    elif tamper == "raw_assignment":
        row = json.loads(raw[0].read_text(encoding="utf-8"))
        row["record_id"] = "record-1"
        raw[0].write_text(json.dumps(row) + "\n", encoding="utf-8")
        manifest["raw_generations_sha256"] = hashlib.sha256(
            raw[0].read_bytes()
        ).hexdigest()
    elif tamper in {"benchmark_path", "benchmark_sha256"}:
        for manifest_path in manifests:
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
            if tamper == "benchmark_path":
                current["benchmark"] = benchmark.name
            else:
                current["benchmark_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(current), encoding="utf-8")
        manifest = None
    else:  # pragma: no cover - the parametrization is the exhaustive contract.
        raise AssertionError(tamper)
    if manifest is not None:
        manifests[0].write_text(json.dumps(manifest), encoding="utf-8")

    _set_merge_argv(
        monkeypatch,
        benchmark=benchmark,
        shard_dir=shard_dir,
        output=tmp_path / "merged.jsonl",
        raw_output=tmp_path / "merged.raw.jsonl",
    )
    with pytest.raises(TrainingContractError, match=error):
        merge_predictions()


def test_merge_rejects_stale_model_inventory(
    tmp_path: Path, monkeypatch
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3_vl"}\n')
    weight = model / "model.safetensors"
    weight.write_bytes(b"before")
    inventory_path = tmp_path / "model_inventory.json"
    provenance = write_inventory_artifact(model, inventory_path, kind="model")
    weight.write_bytes(b"after-with-a-different-size")

    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text('{"record_id":"record-0"}\n', encoding="utf-8")
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    predictions = shard_dir / "predictions.shard-00000-of-00001.jsonl"
    raw = shard_dir / "raw_generations.shard-00000-of-00001.jsonl"
    predictions.write_text(
        '{"record_id":"record-0","parse_status":"parsed"}\n', encoding="utf-8"
    )
    raw.write_text('{"record_id":"record-0"}\n', encoding="utf-8")
    manifest = {
        "schema_version": "epispace.qwen3vl_predictions.v1",
        "model": str(model.resolve()),
        "adapter": None,
        "base_model_inventory_sha256": provenance["inventory_sha256"],
        "adapter_inventory_sha256": None,
        "base_model_provenance": provenance,
        "adapter_provenance": None,
        "benchmark": str(benchmark.resolve()),
        "benchmark_sha256": hashlib.sha256(benchmark.read_bytes()).hexdigest(),
        "shard_index": 0,
        "num_shards": 1,
        "records": 1,
        "parsed": 1,
        "image_min_pixels": 65_536,
        "image_max_pixels": 262_144,
        "max_new_tokens": 96,
        "image_mode": "full",
        "prompt_mode": "native",
        **prediction_artifact_bindings(predictions, raw),
    }
    (shard_dir / "prediction_manifest.shard-00000-of-00001.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_qwen_predictions.py",
            "--benchmark",
            str(benchmark),
            "--shard-dir",
            str(shard_dir),
            "--output",
            str(tmp_path / "merged.jsonl"),
            "--raw-output",
            str(tmp_path / "merged.raw.jsonl"),
        ],
    )
    with pytest.raises(TrainingContractError, match="identity changed"):
        merge_predictions()
