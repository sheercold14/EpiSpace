#!/usr/bin/env python3
"""Build an episode-first static viewer for a generated streaming QA dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_root = (args.out or dataset_root / "viewer").resolve()
    template = Path(__file__).resolve().parent / "templates" / "streaming_qa_viewer.html"
    records = _read_jsonl(dataset_root / "streaming_qa.jsonl")
    oracles = {
        row["record_id"]: row
        for row in _read_jsonl(dataset_root / "streaming_eval_oracle.jsonl")
    }
    if {row["record_id"] for row in records} != set(oracles):
        raise RuntimeError("streaming records and oracles do not have identical record IDs")

    index_rows = []
    for record in records:
        record_id = record["record_id"]
        oracle = oracles[record_id]
        oracle_turns = {turn["turn_id"]: turn for turn in oracle["turns"]}
        turns = []
        for turn in record["turns"]:
            gold = oracle_turns.get(turn["turn_id"])
            if gold is None or gold["answer"] != turn["answer"]:
                raise RuntimeError(f"training/oracle turn mismatch: {turn['turn_id']}")
            turns.append({**turn, "certificate": gold["certificate"]})
        group_path = Path(record["source"]["group_path"])
        episode_id = group_path.parent.name
        payload = {
            "record_id": record_id,
            "episode_id": episode_id,
            "tier": record["tier"],
            "stream_kind": record["stream_kind"],
            "temporal_mode": record.get("temporal_mode", "full"),
            "capabilities": record["capabilities"],
            "cluster_ids": record["cluster_ids"],
            "source": record["source"],
            "images": record["images"],
            "turns": turns,
        }
        relative_record = f"data/records/{record_id}.json"
        _write_json(output_root / relative_record, payload)
        index_rows.append(
            {
                "record_id": record_id,
                "episode_id": episode_id,
                "scene_id": record["source"]["scene_id"],
                "plan_id": record["source"]["plan_id"],
                "tier": record["tier"],
                "stream_kind": record["stream_kind"],
                "temporal_mode": record.get("temporal_mode", "full"),
                "capabilities": record["capabilities"],
                "frame_count": len(record["images"]),
                "turn_count": len(turns),
                "path": relative_record,
            }
        )

    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    _write_json(
        output_root / "data" / "index.json",
        {
            "dataset_id": manifest["dataset_id"],
            "source_collection_id": manifest["source_collection_id"],
            "record_count": len(index_rows),
            "turn_count": sum(row["turn_count"] for row in index_rows),
            "records": index_rows,
        },
    )
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template, output_root / "index.html")
    print(output_root / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
