"""Corpus driver and Qwen multimodal SFT export."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .catalog import read_json
from .planners import FRAME_SURFACE_ZH, PlanError, compile_bundle

SUPPORTED_CLASSES = ("T3", "T4", "T7", "T8", "T10")
REQUIRED_INPUTS = (
    "scene_ir.json",
    "spatial_episode.json",
    "trajectory_plan.json",
    "trajectory_selection.json",
    "quality_report.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_bundle(code_root: Path, quality_url: str) -> Path:
    quality = code_root / quality_url.lstrip("/")
    if not quality.exists():
        raise PlanError(f"catalog quality path does not exist: {quality}")
    return quality.parent


def export_rgb(bundle_root: Path, output_dir: Path, view_ids: list[str]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, str] = {}
    for view_id in view_ids:
        target = output_dir / f"{view_id}.png"
        if not target.exists():
            with np.load(bundle_root / "views" / f"{view_id}.sensors.npz") as arrays:
                Image.fromarray(arrays["rgb"]).save(target)
        result[view_id] = str(target.resolve())
    return result


def to_sft_records(
    artifact: dict[str, Any],
    rgb_paths: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    system = (
        "你正在按真实观察顺序查看同一场景。请只使用当前及此前已经给出的图像维护空间信息；"
        "证据不足时明确回答无法确定。" + FRAME_SURFACE_ZH
    )
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for round_ in artifact["rounds"]:
        content: list[dict[str, Any]] = [
            {"type": "image", "image": rgb_paths[view_id]} for view_id in round_["new_view_ids"]
        ]
        content.append({"type": "text", "text": round_["question_zh"]})
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": round_["answer_zh"]})
    record_id = f"trajectory-{artifact['trajectory_class'].lower()}-{artifact['acquisition_id']}"
    episode = {
        "record_id": record_id,
        "format": "qwen_multimodal_chat",
        "sample_type": "incremental_dialogue",
        "scene_id": artifact["scene_id"],
        "family_id": artifact["family_id"],
        "split_lock": artifact["split_lock"],
        "trajectory_class": artifact["trajectory_class"],
        "task_scope": artifact["task_scope"],
        "messages": messages,
        "loss_policy": {"train_on": "assistant_only"},
        "hidden_meta": {
            "turn_ids": [r["turn_id"] for r in artifact["rounds"]],
            "semantic_signatures": [r["program"]["semantic_signature"] for r in artifact["rounds"]],
            "source_dialogue": artifact["dialogue_path"],
        },
    }
    isolated: list[dict[str, Any]] = []
    for round_ in artifact["rounds"]:
        evidence = round_["evidence_view_ids"] or round_["new_view_ids"]
        if not evidence:
            continue
        content = [{"type": "image", "image": rgb_paths[v]} for v in evidence]
        content.append({"type": "text", "text": "以下图片来自同一场景。" + round_["question_zh"]})
        isolated.append(
            {
                "record_id": f"isolated-{artifact['trajectory_class'].lower()}-{artifact['acquisition_id']}-{round_['turn_id']}",
                "format": "qwen_multimodal_chat",
                "sample_type": "isolated_qa",
                "scene_id": artifact["scene_id"],
                "family_id": artifact["family_id"],
                "split_lock": artifact["split_lock"],
                "trajectory_class": artifact["trajectory_class"],
                "task_scope": artifact["task_scope"],
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": round_["answer_zh"]},
                ],
                "loss_policy": {"train_on": "assistant_only"},
                "hidden_meta": {"turn_id": round_["turn_id"], "comparison_of": record_id},
            }
        )
    return episode, isolated


def compile_catalog(
    catalog_path: Path,
    code_root: Path,
    output_dir: Path,
    limit_per_class: int | None = None,
    classes: tuple[str, ...] = SUPPORTED_CLASSES,
) -> dict[str, Any]:
    catalog_path = Path(catalog_path).resolve()
    code_root = Path(code_root).resolve()
    output_dir = Path(output_dir).resolve()
    catalog = read_json(catalog_path)
    jobs = [
        job
        for job in catalog["collection_jobs"]
        if job.get("status") == "passed" and job.get("trajectory_class") in classes
    ]
    jobs.sort(key=lambda row: (row["trajectory_class"], row["job_id"]))
    if limit_per_class is not None:
        kept: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        for job in jobs:
            cls = job["trajectory_class"]
            if counts[cls] >= limit_per_class:
                continue
            counts[cls] += 1
            kept.append(job)
        jobs = kept

    if output_dir.exists():
        # Rebuilding a generated corpus should not leave stale scene files.
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    direction_balance: dict[str, int] = {}
    report: dict[str, Any] = {
        "schema_version": "epispace.trajectory_dialogue_corpus_report.v2",
        "input_catalog": str(catalog_path),
        "input_catalog_sha256": sha256(catalog_path),
        "selection_rule": "collection_jobs.status == passed",
        "requested": Counter(job["trajectory_class"] for job in jobs),
        "compiled": [],
        "rejected": [],
        "direction_balance": direction_balance,
        "task_scope_counts": {},
        "round_counts": {},
    }
    episode_rows: list[dict[str, Any]] = []
    isolated_rows: list[dict[str, Any]] = []
    round_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()

    for job in jobs:
        acquisition_id = job["job_id"]
        trajectory_class = job["trajectory_class"]
        try:
            bundle = resolve_bundle(code_root, job["quality_url"])
            for filename in REQUIRED_INPUTS:
                if not (bundle / filename).exists():
                    raise PlanError(f"missing required input: {filename}")
            artifact = compile_bundle(bundle, direction_balance=direction_balance)
            artifact["acquisition_id"] = acquisition_id
            artifact["catalog_status"] = job["status"]
            artifact["catalog_scene"] = job["scene"]
            artifact["input_manifest"] = {
                filename: sha256(bundle / filename) for filename in REQUIRED_INPUTS
            }
            scene_dir = output_dir / "scenes" / trajectory_class
            scene_dir.mkdir(parents=True, exist_ok=True)
            dialogue_path = scene_dir / f"{acquisition_id}.dialogue.json"
            artifact["dialogue_path"] = str(dialogue_path.resolve())
            rgb_paths = export_rgb(
                bundle,
                output_dir / "rgb" / trajectory_class / acquisition_id,
                artifact["view_ids"],
            )
            artifact["rgb_paths"] = rgb_paths
            dialogue_path.write_text(
                json.dumps(artifact, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            episode, isolated = to_sft_records(artifact, rgb_paths)
            episode_rows.append(episode)
            isolated_rows.extend(isolated)
            scope_counts[artifact["task_scope"]] += 1
            for round_ in artifact["rounds"]:
                round_counts[round_["turn_id"]] += 1
            report["compiled"].append(
                {
                    "acquisition_id": acquisition_id,
                    "trajectory_class": trajectory_class,
                    "scene": job["scene"],
                    "task_scope": artifact["task_scope"],
                    "round_count": len(artifact["rounds"]),
                    "dialogue_path": str(dialogue_path.resolve()),
                }
            )
        except (PlanError, KeyError, ValueError) as error:
            report["rejected"].append(
                {
                    "acquisition_id": acquisition_id,
                    "trajectory_class": trajectory_class,
                    "scene": job["scene"],
                    "reason": str(error),
                }
            )

    for filename, rows in (
        ("all.dialogue_episode_sft.jsonl", episode_rows),
        ("all.dialogue_isolated_sft.jsonl", isolated_rows),
    ):
        with (output_dir / filename).open("w", encoding="utf-8") as sink:
            for row in rows:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    report["requested"] = dict(report["requested"])
    report["compiled_counts"] = dict(Counter(row["trajectory_class"] for row in report["compiled"]))
    report["rejected_counts"] = dict(Counter(row["trajectory_class"] for row in report["rejected"]))
    report["task_scope_counts"] = dict(scope_counts)
    report["round_counts"] = dict(round_counts)
    (output_dir / "corpus_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return report
