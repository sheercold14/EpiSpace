#!/usr/bin/env python3
"""Select a deterministic, program-covering paired subset for fast model pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from episode3d.qwen_training import read_jsonl  # noqa: E402

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_binding(path: Path, *, relative_to: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"lineage artifact does not exist: {resolved}")
    return {
        "path": os.path.relpath(resolved, relative_to),
        "sha256": file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def release_lineage(
    *,
    episode_sft: Path,
    episode_schedule: Path,
    isolated_schedule: Path,
    output_dir: Path,
) -> tuple[dict[str, dict[str, object]], dict[str, bool]]:
    """Bind a pilot subset to the independently verified formal release."""

    episode_sft = episode_sft.resolve()
    episode_schedule = episode_schedule.resolve()
    isolated_schedule = isolated_schedule.resolve()
    compute_dir = episode_schedule.parent
    release_dir = compute_dir.parent
    if isolated_schedule.parent != compute_dir:
        raise RuntimeError("formal source schedules do not share a compute directory")
    if episode_sft != release_dir / "train.episode_sft.jsonl":
        raise RuntimeError("episode SFT is not the formal release training artifact")

    compute_manifest_path = compute_dir / "compute_matching_manifest.json"
    release_manifest_path = release_dir / "release_manifest.json"
    final_index_path = release_dir / "final_release_index.json"
    for path in (compute_manifest_path, release_manifest_path, final_index_path):
        if not path.is_file():
            raise FileNotFoundError(f"formal release authority is missing: {path}")

    compute_manifest = json.loads(compute_manifest_path.read_text(encoding="utf-8"))
    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    final_index = json.loads(final_index_path.read_text(encoding="utf-8"))
    regime = compute_manifest.get("regimes", {}).get("image_occurrence_matched", {})
    schedule_files = regime.get("schedule_files", {})
    declared_episode = schedule_files.get("episode", {})
    declared_isolated = schedule_files.get("isolated", {})
    declared_episode_path = (
        compute_dir / str(declared_episode.get("filename", ""))
    ).resolve()
    declared_isolated_path = (
        compute_dir / str(declared_isolated.get("filename", ""))
    ).resolve()
    episode_schedule_sha256 = file_sha256(episode_schedule)
    isolated_schedule_sha256 = file_sha256(isolated_schedule)
    compute_manifest_sha256 = file_sha256(compute_manifest_path)
    release_manifest_sha256 = file_sha256(release_manifest_path)
    compute_sources = compute_manifest.get("sources", {})
    declared_episode_source = compute_sources.get("episode", {})
    declared_episode_source_path = (
        compute_dir / str(declared_episode_source.get("path", ""))
    ).resolve()
    final_compute = final_index.get("compute_matching", {})
    final_schedule_hashes = final_compute.get("schedules", {})
    checks = {
        "release_manifest_status_pass": release_manifest.get("status") == "pass",
        "compute_manifest_status_pass": compute_manifest.get("status") == "pass",
        "final_release_index_status_pass": final_index.get("status") == "pass",
        "formal_schedule_paths_match": (
            declared_episode_path == episode_schedule
            and declared_isolated_path == isolated_schedule
        ),
        "compute_manifest_schedule_hashes_match": (
            declared_episode.get("sha256") == episode_schedule_sha256
            and declared_isolated.get("sha256") == isolated_schedule_sha256
        ),
        "compute_manifest_episode_sft_bound": (
            declared_episode_source_path == episode_sft
            and declared_episode_source.get("sha256") == file_sha256(episode_sft)
        ),
        "final_index_release_manifest_bound": (
            final_index.get("release_manifest_sha256") == release_manifest_sha256
        ),
        "final_index_compute_manifest_bound": (
            final_compute.get("manifest_sha256") == compute_manifest_sha256
        ),
        "final_index_source_schedules_bound": (
            final_schedule_hashes.get("image_occurrence_matched.episode")
            == episode_schedule_sha256
            and final_schedule_hashes.get("image_occurrence_matched.isolated")
            == isolated_schedule_sha256
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"formal release lineage verification failed: {checks}")
    paths = {
        "episode_selection_sft": episode_sft,
        "episode_source_schedule": episode_schedule,
        "isolated_source_schedule": isolated_schedule,
        "compute_matching_manifest": compute_manifest_path,
        "release_manifest": release_manifest_path,
        "final_release_index": final_index_path,
    }
    return (
        {
            name: _artifact_binding(path, relative_to=output_dir)
            for name, path in paths.items()
        },
        checks,
    )


def rank(seed: int, record_id: str) -> str:
    return hashlib.sha256(f"{seed}\x1f{record_id}".encode()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def relocate_source_pointers(
    rows: list[dict], *, source_schedule: Path, output_dir: Path
) -> list[dict]:
    """Keep copied schedule rows resolvable from their new directory.

    Official schedules use paths relative to their own directory.  A pilot
    schedule lives elsewhere, so copying those strings verbatim would make
    them point at a non-existent experiment-local JSONL.  Store a portable
    relative path from the pilot directory to the authoritative source row.
    """

    relocated: list[dict] = []
    digest_cache: dict[Path, str] = {}
    for index, row in enumerate(rows):
        source_name = row.get("source_jsonl")
        expected_digest = row.get("source_jsonl_sha256")
        if not isinstance(source_name, str) or not source_name:
            raise RuntimeError(f"schedule row {index} has no source_jsonl")
        if not isinstance(expected_digest, str) or not _SHA256.fullmatch(
            expected_digest
        ):
            raise RuntimeError(
                f"schedule row {index} has no valid source_jsonl_sha256"
            )
        source_path = Path(source_name)
        if not source_path.is_absolute():
            source_path = (source_schedule.parent / source_path).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"schedule source does not exist: {source_path}")
        if source_path not in digest_cache:
            digest_cache[source_path] = file_sha256(source_path)
        if digest_cache[source_path] != expected_digest:
            raise RuntimeError(
                f"schedule row {index} source JSONL hash mismatch: {source_path}"
            )
        copied = dict(row)
        copied["source_jsonl"] = os.path.relpath(source_path, output_dir)
        relocated.append(copied)
    return relocated


def source_jsonl_inventory(rows: list[dict], *, schedule_dir: Path) -> list[dict[str, object]]:
    """Return the verified source closure encoded by copied schedule rows."""

    inventory: dict[Path, dict[str, object]] = {}
    for index, row in enumerate(rows):
        source_name = row.get("source_jsonl")
        expected_digest = row.get("source_jsonl_sha256")
        if not isinstance(source_name, str) or not source_name:
            raise RuntimeError(f"schedule row {index} has no source_jsonl")
        if not isinstance(expected_digest, str) or not _SHA256.fullmatch(
            expected_digest
        ):
            raise RuntimeError(
                f"schedule row {index} has no valid source_jsonl_sha256"
            )
        source_path = Path(source_name)
        if not source_path.is_absolute():
            source_path = (schedule_dir / source_path).resolve()
        observed_digest = file_sha256(source_path)
        if observed_digest != expected_digest:
            raise RuntimeError(
                f"schedule row {index} source JSONL hash mismatch: {source_path}"
            )
        previous = inventory.get(source_path)
        entry = {
            "path": os.path.relpath(source_path, schedule_dir),
            "sha256": observed_digest,
            "bytes": source_path.stat().st_size,
        }
        if previous is not None and previous != entry:
            raise RuntimeError(f"inconsistent source binding for {source_path}")
        inventory[source_path] = entry
    return [inventory[path] for path in sorted(inventory, key=str)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-sft", type=Path, required=True)
    parser.add_argument("--episode-schedule", type=Path, required=True)
    parser.add_argument("--isolated-schedule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-facts", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    if args.max_facts < 1:
        parser.error("--max-facts must be positive")

    episode_records = read_jsonl(args.episode_sft.resolve())
    by_program: dict[str, list[dict]] = defaultdict(list)
    for row in episode_records:
        for program in set(row["hidden_meta"]["program_ids"]):
            by_program[program].append(row)
    candidates = sorted(
        episode_records, key=lambda row: rank(args.seed, row["record_id"])
    )
    selected: dict[str, dict] = {}
    selected_facts = 0

    def try_add(row: dict) -> bool:
        nonlocal selected_facts
        if row["record_id"] in selected:
            return True
        fact_count = len(row["comparison_contract"]["fact_ids"])
        if selected_facts + fact_count > args.max_facts:
            return False
        selected[row["record_id"]] = row
        selected_facts += fact_count
        return True

    # First cover every train program, then use a deterministic hash ranking.
    for program in sorted(by_program):
        for row in sorted(by_program[program], key=lambda item: rank(args.seed, item["record_id"])):
            if try_add(row):
                break
    for row in candidates:
        try_add(row)

    fact_ids = {
        fact_id
        for row in selected.values()
        for fact_id in row["comparison_contract"]["fact_ids"]
    }
    episode_schedule_all = read_jsonl(args.episode_schedule.resolve())
    isolated_schedule_all = read_jsonl(args.isolated_schedule.resolve())
    episode_schedule = [
        row for row in episode_schedule_all if row["record_id"] in selected
    ]
    isolated_schedule = [
        row
        for row in isolated_schedule_all
        if len(row["fact_ids"]) == 1 and row["fact_ids"][0] in fact_ids
    ]
    episode_images = Counter(
        image for row in episode_schedule for image in row["image_paths"]
    )
    isolated_images = Counter(
        image for row in isolated_schedule for image in row["image_paths"]
    )
    effective_facts: Counter[str] = Counter()
    for row in episode_schedule:
        for fact_id in row["fact_ids"]:
            effective_facts[fact_id] += row["sample_weight"]
    checks = {
        "selected_fact_count_matches": len(fact_ids) == selected_facts,
        "draw_count_equal": len(episode_schedule) == len(isolated_schedule) == len(fact_ids),
        "image_occurrence_multiset_equal": episode_images == isolated_images,
        "episode_effective_fact_weight_one": all(
            abs(effective_facts[fact_id] - 1.0) < 1e-9 for fact_id in fact_ids
        ),
        "isolated_fact_set_equal": {
            row["fact_ids"][0] for row in isolated_schedule
        }
        == fact_ids,
        "all_train_programs_covered": set(by_program)
        == {
            program
            for row in selected.values()
            for program in row["hidden_meta"]["program_ids"]
        },
    }
    if not all(checks.values()):
        raise RuntimeError(f"pilot subset invariant failed: {checks}")

    output_dir = args.output_dir.resolve()
    lineage, lineage_checks = release_lineage(
        episode_sft=args.episode_sft,
        episode_schedule=args.episode_schedule,
        isolated_schedule=args.isolated_schedule,
        output_dir=output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_schedule = relocate_source_pointers(
        episode_schedule,
        source_schedule=args.episode_schedule.resolve(),
        output_dir=output_dir,
    )
    isolated_schedule = relocate_source_pointers(
        isolated_schedule,
        source_schedule=args.isolated_schedule.resolve(),
        output_dir=output_dir,
    )
    episode_source_inventory = source_jsonl_inventory(
        episode_schedule, schedule_dir=output_dir
    )
    isolated_source_inventory = source_jsonl_inventory(
        isolated_schedule, schedule_dir=output_dir
    )
    episode_selection_source = args.episode_sft.resolve()
    episode_selection_source_bound = any(
        (output_dir / str(entry["path"])).resolve() == episode_selection_source
        and entry["sha256"] == file_sha256(episode_selection_source)
        for entry in episode_source_inventory
    )
    if not episode_selection_source_bound:
        raise RuntimeError(
            "episode selection SFT is not content-identical to the episode schedule source"
        )
    checks["source_jsonl_hashes_verified"] = True
    checks["episode_selection_source_bound"] = True
    episode_output = output_dir / "image_matched.episode.schedule.jsonl"
    isolated_output = output_dir / "image_matched.isolated.schedule.jsonl"
    write_jsonl(episode_output, episode_schedule)
    write_jsonl(isolated_output, isolated_schedule)
    program_counts = Counter(
        program
        for row in selected.values()
        for program in row["hidden_meta"]["program_ids"]
    )
    manifest = {
        "schema_version": "epispace.qwen_pilot_subset.v1",
        "status": "pass",
        "seed": args.seed,
        "max_facts": args.max_facts,
        "selected_facts": len(fact_ids),
        "selected_episode_records": len(selected),
        "selected_scenes": len({row["scene_id"] for row in selected.values()}),
        "draws_per_arm": len(episode_schedule),
        "image_occurrences_per_arm": sum(episode_images.values()),
        "program_counts": dict(sorted(program_counts.items())),
        "checks": checks,
        "sources": {
            "episode_sft": file_sha256(args.episode_sft.resolve()),
            "episode_schedule": file_sha256(args.episode_schedule.resolve()),
            "isolated_schedule": file_sha256(args.isolated_schedule.resolve()),
            "source_jsonl_inventory": {
                "episode": episode_source_inventory,
                "isolated": isolated_source_inventory,
            },
        },
        "lineage": lineage,
        "lineage_checks": lineage_checks,
        "artifacts": {
            "episode": {
                "path": episode_output.name,
                "sha256": file_sha256(episode_output),
            },
            "isolated": {
                "path": isolated_output.name,
                "sha256": file_sha256(isolated_output),
            },
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
