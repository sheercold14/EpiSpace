from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scripts.build_qwen_pilot_subset import (
    release_lineage,
    relocate_source_pointers,
    source_jsonl_inventory,
)


def _schedule_fixture(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    source = tmp_path / "release" / "train.episode_sft.jsonl"
    source.parent.mkdir()
    source.write_text(json.dumps({"record_id": "record-1"}) + "\n", encoding="utf-8")
    schedule = tmp_path / "release" / "compute_matching" / "schedule.jsonl"
    schedule.parent.mkdir()
    row = {
        "source_jsonl": "../train.episode_sft.jsonl",
        "source_jsonl_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_line": 1,
    }
    return source, schedule, [row]


def test_relocated_pilot_rows_keep_a_verified_source_closure(tmp_path: Path) -> None:
    source, schedule, rows = _schedule_fixture(tmp_path)
    output = tmp_path / "experiment" / "pilot128"
    output.mkdir(parents=True)

    relocated = relocate_source_pointers(
        rows,
        source_schedule=schedule,
        output_dir=output,
    )
    inventory = source_jsonl_inventory(relocated, schedule_dir=output)

    assert (output / relocated[0]["source_jsonl"]).resolve() == source.resolve()
    assert inventory == [
        {
            "path": relocated[0]["source_jsonl"],
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "bytes": source.stat().st_size,
        }
    ]


def test_pilot_relocation_rejects_a_stale_source_digest(tmp_path: Path) -> None:
    source, schedule, rows = _schedule_fixture(tmp_path)
    source.write_text("mutated\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="source JSONL hash mismatch"):
        relocate_source_pointers(
            rows,
            source_schedule=schedule,
            output_dir=tmp_path / "pilot128",
        )


def _formal_lineage_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    release = tmp_path / "release"
    compute = release / "compute_matching"
    compute.mkdir(parents=True)
    episode_sft = release / "train.episode_sft.jsonl"
    episode_sft.write_text("{}\n", encoding="utf-8")
    episode_schedule = compute / "image_occurrence_matched.episode.schedule.jsonl"
    isolated_schedule = compute / "image_occurrence_matched.isolated.schedule.jsonl"
    episode_schedule.write_text("{}\n", encoding="utf-8")
    isolated_schedule.write_text("{}\n", encoding="utf-8")
    episode_sha = hashlib.sha256(episode_schedule.read_bytes()).hexdigest()
    isolated_sha = hashlib.sha256(isolated_schedule.read_bytes()).hexdigest()
    compute_manifest = compute / "compute_matching_manifest.json"
    compute_manifest.write_text(
        json.dumps(
            {
                "status": "pass",
                "sources": {
                    "episode": {
                        "path": "../train.episode_sft.jsonl",
                        "sha256": hashlib.sha256(episode_sft.read_bytes()).hexdigest(),
                    }
                },
                "regimes": {
                    "image_occurrence_matched": {
                        "schedule_files": {
                            "episode": {
                                "filename": episode_schedule.name,
                                "sha256": episode_sha,
                            },
                            "isolated": {
                                "filename": isolated_schedule.name,
                                "sha256": isolated_sha,
                            },
                        }
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    release_manifest = release / "release_manifest.json"
    release_manifest.write_text('{"status":"pass"}\n', encoding="utf-8")
    final_index = release / "final_release_index.json"
    final_index.write_text(
        json.dumps(
            {
                "status": "pass",
                "release_manifest_sha256": hashlib.sha256(
                    release_manifest.read_bytes()
                ).hexdigest(),
                "compute_matching": {
                    "manifest_sha256": hashlib.sha256(
                        compute_manifest.read_bytes()
                    ).hexdigest(),
                    "schedules": {
                        "image_occurrence_matched.episode": episode_sha,
                        "image_occurrence_matched.isolated": isolated_sha,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return episode_sft, episode_schedule, isolated_schedule, final_index


def test_pilot_subset_binds_final_release_authority(tmp_path: Path) -> None:
    episode_sft, episode_schedule, isolated_schedule, final_index = (
        _formal_lineage_fixture(tmp_path)
    )
    bindings, checks = release_lineage(
        episode_sft=episode_sft,
        episode_schedule=episode_schedule,
        isolated_schedule=isolated_schedule,
        output_dir=tmp_path / "experiment" / "pilot128",
    )
    assert all(checks.values())
    assert set(bindings) == {
        "episode_selection_sft",
        "episode_source_schedule",
        "isolated_source_schedule",
        "compute_matching_manifest",
        "release_manifest",
        "final_release_index",
    }

    final_payload = json.loads(final_index.read_text(encoding="utf-8"))
    final_payload["release_manifest_sha256"] = "0" * 64
    final_index.write_text(json.dumps(final_payload) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="formal release lineage verification failed"):
        release_lineage(
            episode_sft=episode_sft,
            episode_schedule=episode_schedule,
            isolated_schedule=isolated_schedule,
            output_dir=tmp_path / "experiment" / "pilot128",
        )
