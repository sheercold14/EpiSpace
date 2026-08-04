from __future__ import annotations

import json
from pathlib import Path

import pytest

from episode3d.pilot_summary import (
    PilotSummaryError,
    model_inventory,
    verify_inventory_artifact,
    write_inventory_artifact,
)


def _model(root: Path) -> Path:
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3_vl"}\n')
    (model / "model.safetensors").write_bytes(b"weights")
    runtime_files = {
        "generation_config.json": b"{}",
        "preprocessor_config.json": b"{}",
        "video_preprocessor_config.json": b"{}",
        "processor_config.json": b"{}",
        "tokenizer_config.json": b"{}",
        "tokenizer.json": b"before",
        "tokenizer.extra.json": b"{}",
        "vocab.json": b"{}",
        "merges.txt": b"a b\n",
        "special_tokens_map.json": b"{}",
        "added_tokens.json": b"{}",
        "chat_template.json": b"{}",
        "chat_templates/tool.jinja": b"{{ messages }}",
    }
    for relative, content in runtime_files.items():
        path = model / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (model / "README.md").write_text("not runtime state\n")
    metadata = model / ".cache" / "huggingface" / "download" / "vocab.json.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("not runtime state\n")
    return model


def test_model_inventory_covers_processor_tokenizer_and_chat_runtime(
    tmp_path: Path,
) -> None:
    model = _model(tmp_path)
    inventory = model_inventory(model)
    roles = {entry["path"]: entry["role"] for entry in inventory["files"]}

    assert roles == {
        "added_tokens.json": "tokenizer_added_tokens",
        "chat_template.json": "chat_template",
        "chat_templates/tool.jinja": "chat_template",
        "config.json": "config",
        "generation_config.json": "generation_config",
        "merges.txt": "tokenizer_merges",
        "model.safetensors": "weight",
        "preprocessor_config.json": "image_preprocessor_config",
        "processor_config.json": "processor_config",
        "special_tokens_map.json": "tokenizer_special_tokens",
        "tokenizer.extra.json": "tokenizer",
        "tokenizer.json": "tokenizer",
        "tokenizer_config.json": "tokenizer_config",
        "video_preprocessor_config.json": "video_preprocessor_config",
        "vocab.json": "tokenizer_vocab",
    }
    assert "README.md" not in roles
    assert ".cache/huggingface/download/vocab.json.metadata" not in roles


def test_tokenizer_change_invalidates_fast_inventory_verification(tmp_path: Path) -> None:
    model = _model(tmp_path)
    artifact = tmp_path / "model_inventory.json"
    write_inventory_artifact(model, artifact, kind="model")

    (model / "tokenizer.json").write_bytes(b"changed-size")

    with pytest.raises(PilotSummaryError, match="identity changed"):
        verify_inventory_artifact(artifact, root=model, kind="model")


def test_final_full_rehash_rejects_content_change_even_if_stats_are_rebound(
    tmp_path: Path,
) -> None:
    model = _model(tmp_path)
    artifact = tmp_path / "model_inventory.json"
    write_inventory_artifact(model, artifact, kind="model")
    tokenizer = model / "tokenizer.json"
    tokenizer.write_bytes(b"after!")  # same byte length as ``before``

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    stat = tokenizer.stat()
    rebound = {
        "bytes": stat.st_size,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }
    for item in payload["inventory"]["file_stats"]:
        if item["path"] == "tokenizer.json":
            item.update(rebound)
            break
    artifact.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    verify_inventory_artifact(artifact, root=model, kind="model")
    with pytest.raises(PilotSummaryError, match="content digest is stale"):
        verify_inventory_artifact(
            artifact, root=model, kind="model", verify_content=True
        )
