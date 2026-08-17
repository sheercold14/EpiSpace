#!/usr/bin/env python3
"""Derive causal occlusion QA from immutable behavior51 v1 pixels.

The source coverage tree is read-only.  This script authority-replays the
first visible -> invisible event, retains 173 eligible legacy occlusion
episodes, selects the same number of out-of-view negatives, and writes a new
question-group overlay whose media are hard-linked to the original RGB files.
It then compiles raw and isolated immediate/delayed streaming QA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from spatial_episode.scriptgen.behavior import RenderSceneView
from spatial_episode.scriptgen.compiler import CapabilityCompiler
from spatial_episode.scriptgen.family import (
    QuestionGroupEntry,
    QuestionGroupTrajectory,
    ScriptgenQuestionGroupV1,
    build_family_doc,
)
from spatial_episode.scriptgen.library import (
    DISAPPEARANCE_CAUSE,
    OCCLUDER_IDENTIFICATION,
)
from spatial_episode.scriptgen.occlusion import (
    DISAPPEARANCE_POLICY_VERSION,
    first_decisive_disappearance,
)
from spatial_episode.scriptgen.qa_dataset import build_qa_dataset
from spatial_episode.scriptgen.standards import STD_V1

OVERLAY_SCHEMA = "epispace.occlusion_qa_overlay.v1"
SELF_MOTION_SOURCES = frozenset(
    {
        "self_motion_update",
        "self_motion_update_pure_rotation",
        "self_motion_update_pure_translation",
        "self_motion_update_multi_turn",
        "self_motion_update_occluded",
    }
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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


def _quarantine_ids(path: Path | None) -> frozenset[str]:
    if path is None:
        return frozenset()
    return frozenset(
        json.loads(line)["episode_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _stable_hash(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}:{value}".encode()).hexdigest()


def _stratified_select(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for row in sorted(rows, key=lambda value: _stable_hash("negative", value["episode_id"])):
        buckets[(row["scene_key"], row["source_capability"])].append(row)
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    while keys and len(selected) < count:
        remaining = []
        for key in keys:
            if len(selected) >= count:
                break
            selected.append(buckets[key].popleft())
            if buckets[key]:
                remaining.append(key)
        keys = remaining
    return selected


def _longest_visible_run(view: RenderSceneView, target: str) -> int:
    longest = current = 0
    for frame in range(view.frame_count):
        if view.visibility(target, frame).tristate(STD_V1) is True:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _qualifying_row(root: Path, episode: dict[str, Any]) -> dict[str, Any] | None:
    group_path = root / "groups" / episode["episode_id"] / "group.json"
    if not group_path.is_file():
        return None
    group = _read(group_path)
    plan_path = Path(group["trajectory"]["plan_record"])
    plan = _read(plan_path)
    target = str(plan["binding"]["target"])
    view = RenderSceneView.from_bundle(
        Path(group["trajectory"]["bundle"]),
        STD_V1,
        scene_ir=Path(group["trajectory"]["scene_ir"]),
    )
    event = first_decisive_disappearance(view, STD_V1, target)
    if event is None or _longest_visible_run(view, target) < 2:
        return None
    script = (
        OCCLUDER_IDENTIFICATION if event.cause == "occluded" else DISAPPEARANCE_CAUSE
    )
    certificate = CapabilityCompiler(script, STD_V1).compile(view, dict(plan["binding"]))
    if certificate.status != "answerable" or certificate.answer is None:
        return None
    if event.cause == "occluded" and not event.eligible_occluder:
        return None
    return {
        **episode,
        "source_group": str(group_path.parent.resolve()),
        "plan_record": str(plan_path.resolve()),
        "scene_ir": str(Path(group["trajectory"]["scene_ir"]).resolve()),
        "bundle": str(Path(group["trajectory"]["bundle"]).resolve()),
        "plan": plan,
        "event": {
            "frame": event.frame,
            "first_invisible_frame": event.first_invisible_frame,
            "last_visible_frame": event.last_visible_frame,
            "cause": event.cause,
            "occluder_entity_id": event.occluder_entity_id,
            "occluder_category": event.occluder_category,
            "eligible_occluder": event.eligible_occluder,
            "witness": event.witness,
        },
    }


def _link_media(source_group: Path, output_group: Path) -> None:
    source_media = source_group / "media"
    output_media = output_group / "media"
    output_media.mkdir(parents=True, exist_ok=False)
    for source in sorted(source_media.iterdir()):
        if not source.is_file():
            continue
        destination = output_media / source.name
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def _build_group(root: Path, out: Path, row: dict[str, Any]) -> dict[str, Any]:
    episode_id = row["episode_id"]
    source_group = Path(row["source_group"])
    output_group = out / "groups" / episode_id
    output_group.mkdir(parents=True, exist_ok=False)
    _link_media(source_group, output_group)
    plan = row["plan"]
    view = RenderSceneView.from_bundle(
        Path(row["bundle"]), STD_V1, scene_ir=Path(row["scene_ir"])
    )
    scripts = (
        (OCCLUDER_IDENTIFICATION, DISAPPEARANCE_CAUSE)
        if row["event"]["cause"] == "occluded"
        else (DISAPPEARANCE_CAUSE,)
    )
    question_group_id = f"{plan['plan_id']}.occlusion_qa.{DISAPPEARANCE_POLICY_VERSION}"
    entries: list[QuestionGroupEntry] = []
    for script in scripts:
        family_id = f"{question_group_id}.{script.capability}.family"
        family = build_family_doc(
            view,
            plan,
            script,
            STD_V1,
            seed=int(plan["seed"]),
            role="primary",
            question_group_id=question_group_id,
            family_id=family_id,
            media_prefix="../media",
        )
        family_dir = output_group / script.capability
        family_dir.mkdir()
        family_path = family_dir / "family.json"
        family_path.write_text(family.model_dump_json(indent=1), encoding="utf-8")
        canonical = next(item for item in family.episodes if item.kind == "canonical")
        entries.append(
            QuestionGroupEntry(
                capability=script.capability,
                role="primary",
                family_id=family.family_id,
                label=canonical.label,
                family=str(family_path.relative_to(output_group)),
                skip_reason=None,
            )
        )
    group = ScriptgenQuestionGroupV1(
        question_group_id=question_group_id,
        standard_version=STD_V1.standard_version,
        trajectory=QuestionGroupTrajectory(
            bundle=row["bundle"],
            plan_record=row["plan_record"],
            scene_ir=row["scene_ir"],
            plan_id=plan["plan_id"],
            scene_id=plan["scene_id"],
        ),
        questions=tuple(entries),
    )
    group_path = output_group / "group.json"
    group_path.write_text(group.model_dump_json(indent=1), encoding="utf-8")
    return {
        "episode_id": episode_id,
        "scene_key": row["scene_key"],
        "source_capability": row["source_capability"],
        "binding": row["binding"],
        "plan_id": row["plan_id"],
        "bundle": row["bundle"],
        "group": str(group_path.resolve()),
        "credited_cells": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/behavior51_coverage_v1"))
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/behavior51_occlusion_qa_overlay_v1")
    )
    parser.add_argument(
        "--quarantine",
        type=Path,
        default=Path("outputs/behavior51_coverage_v2_partial_s1/audit/quarantine.jsonl"),
    )
    parser.add_argument("--negative-count", type=int)
    parser.add_argument("--skip-qa", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing overlay: {out}")
    dataset = _read(source / "dataset.json")
    quarantine = _quarantine_ids(args.quarantine if args.quarantine.is_file() else None)

    occluded: list[dict[str, Any]] = []
    out_of_view_pool: list[dict[str, Any]] = []
    for index, episode in enumerate(dataset["episodes"], start=1):
        capability = episode["source_capability"]
        if capability not in SELF_MOTION_SOURCES:
            continue
        if capability != "self_motion_update_occluded" and episode["episode_id"] in quarantine:
            continue
        row = _qualifying_row(source, episode)
        if row is None:
            continue
        if capability == "self_motion_update_occluded" and row["event"]["cause"] == "occluded":
            occluded.append(row)
        elif capability != "self_motion_update_occluded" and row["event"]["cause"] == "out_of_view":
            out_of_view_pool.append(row)
        if index % 250 == 0:
            print(
                f"authority replay {index}/{dataset['episode_count']} "
                f"occluded={len(occluded)} out_of_view_pool={len(out_of_view_pool)}",
                flush=True,
            )

    occluded.sort(key=lambda row: row["episode_id"])
    negative_count = args.negative_count or len(occluded)
    if len(occluded) != 173:
        raise RuntimeError(f"expected 173 eligible legacy occlusions, found {len(occluded)}")
    if negative_count != len(occluded):
        raise ValueError("v1 release requires equal occluded and out_of_view counts")
    negatives = _stratified_select(out_of_view_pool, negative_count)
    if len(negatives) != negative_count:
        raise RuntimeError(
            f"not enough out_of_view negatives: requested {negative_count}, got {len(negatives)}"
        )

    out.mkdir(parents=True)
    selected = sorted((*occluded, *negatives), key=lambda row: row["episode_id"])
    episode_rows = []
    for index, row in enumerate(selected, start=1):
        episode_rows.append(_build_group(source, out, row))
        if index % 25 == 0:
            print(f"built groups {index}/{len(selected)}", flush=True)

    overlay_dataset = {
        "schema_version": "scriptgen_binding_coverage_dataset.v1",
        "collection_id": "behavior51_occlusion_qa_overlay_v1",
        "standard_version": STD_V1.standard_version,
        "episode_count": len(episode_rows),
        "episodes": episode_rows,
        "coverage_report": {
            "overlay_schema": OVERLAY_SCHEMA,
            "event_policy_version": DISAPPEARANCE_POLICY_VERSION,
            "occluded_episode_count": len(occluded),
            "out_of_view_episode_count": len(negatives),
            "source_collection_id": dataset.get("collection_id"),
        },
    }
    _write(out / "dataset.json", overlay_dataset)
    _write_jsonl(
        out / "selection.audit.jsonl",
        [
            {
                "episode_id": row["episode_id"],
                "scene_key": row["scene_key"],
                "source_capability": row["source_capability"],
                "event": row["event"],
            }
            for row in selected
        ],
    )
    _write_jsonl(
        out / "legacy_direction.exclusions.jsonl",
        (
            {
                "episode_id": episode["episode_id"],
                "capability": "self_motion_update_occluded",
                "reason": "legacy_terminal_occluder_definition_not_first_event_direction",
                "scope": "capability_only",
            }
            for episode in dataset["episodes"]
            if episode["source_capability"] == "self_motion_update_occluded"
        ),
    )
    _write(
        out / "overlay.policy.json",
        {
            "schema_version": OVERLAY_SCHEMA,
            "source_root": str(source),
            "source_is_immutable": True,
            "event_policy_version": DISAPPEARANCE_POLICY_VERSION,
            "attribution_grace_frames": 2,
            "minimum_consecutive_visible_frames": 2,
            "excluded_occluder_categories": ["floors", "ceilings"],
            "skeletal_frame_policy": "allowed_as_框架结构",
            "cause_balance": "1:1:1 in streaming QA",
            "streaming_history": "isolated single-question immediate and delayed records",
        },
    )
    if not args.skip_qa:
        manifest = build_qa_dataset(source_root=out, output_root=out / "qa")
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
