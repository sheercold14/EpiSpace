"""Strict T1 source -> T10 held-out perspective composition episodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from episode3d.scene_llm_pipeline.lexicon import QUADRANT_ZH

from .catalog import MIN_PIXELS, CatalogError, Entity, TrajectoryCatalog, read_json
from .compiler import resolve_bundle, sha256, to_sft_records
from .contracts import SCHEMA_VERSION, ContractError, RoundBuilder, validate_artifact
from .planners import FRAME_SURFACE_ZH, PlanError


def _logical(prefix: str, view_id: str) -> str:
    return f"{prefix}-{view_id}"


def _copy_rgb(bundle: Path, view_id: str, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with np.load(bundle / "views" / f"{view_id}.sensors.npz") as arrays:
            Image.fromarray(arrays["rgb"]).save(target)
    return str(target.resolve())


def _local_ref(catalog: TrajectoryCatalog, entity: Entity, view_id: str) -> str:
    return catalog.reference(entity, view_id).replace(
        f"第{catalog.view_ids.index(view_id) + 1}个视角", "该视角"
    )


def compile_t1_to_t10(t1_root: Path, t10_root: Path, acquisition_id: str) -> dict[str, Any]:
    try:
        source = TrajectoryCatalog(t1_root)
        target_catalog = TrajectoryCatalog(t10_root)
    except CatalogError as error:
        raise PlanError(str(error)) from error
    if source.trajectory_class != "T1" or target_catalog.trajectory_class != "T10":
        raise PlanError("composition requires T1 source and T10 target")
    if source.scene_id != target_catalog.scene_id:
        raise PlanError("T1 and T10 scene ids differ")

    best: tuple[tuple[int, float, int], dict[str, Any]] | None = None
    for pair in target_catalog.plan["target_anchors"]:
        view = pair["view_id"]
        try:
            origin_s = source.entity(pair["origin_entity_id"])
            facing_s = source.entity(pair["facing_entity_id"])
            origin_t = target_catalog.entity(pair["origin_entity_id"])
            facing_t = target_catalog.entity(pair["facing_entity_id"])
        except CatalogError:
            continue
        origin_view = source.well_visible(origin_s, source.view_ids)
        facing_view = source.well_visible(facing_s, source.view_ids)
        if origin_view is None or facing_view is None:
            continue
        for candidate_t in target_catalog.named_visible(view):
            if candidate_t.source_id in {origin_t.source_id, facing_t.source_id}:
                continue
            try:
                candidate_s = source.entity(candidate_t.source_id)
            except CatalogError:
                continue
            candidate_view = source.well_visible(candidate_s, source.view_ids)
            if candidate_view is None:
                continue
            fact = target_catalog.object_frame_relation(origin_t, facing_t, candidate_t)
            if not fact["hard_quadrant_ok"]:
                continue
            all_three_covis = any(
                all(
                    source.pixels(entity, source_view) >= MIN_PIXELS
                    for entity in (origin_s, facing_s, candidate_s)
                )
                for source_view in source.view_ids
            )
            score = (
                0 if all_three_covis else 1,
                float(fact["angle_margin_deg"]),
                target_catalog.pixels(candidate_t, view),
            )
            payload = {
                "target_view": view,
                "origin_s": origin_s,
                "facing_s": facing_s,
                "candidate_s": candidate_s,
                "candidate_t": candidate_t,
                "origin_view": origin_view,
                "facing_view": facing_view,
                "candidate_view": candidate_view,
                "fact": fact,
                "cross_view_required": not all_three_covis,
            }
            if best is None or score > best[0]:
                best = (score, payload)
    if best is None:
        raise PlanError("no T1-grounded hard-margin T10 perspective query")
    x = best[1]
    origin: Entity = x["origin_s"]
    facing: Entity = x["facing_s"]
    target: Entity = x["candidate_s"]
    fact = x["fact"]
    source_ids = [_logical("source", view) for view in source.view_ids]
    heldout_id = _logical("heldout", x["target_view"])

    # Round 1: ingest the first half without asking for the final perspective answer.
    first = source.view_ids[:6]
    names: list[str] = []
    for view in first:
        for entity in source.named_visible(view):
            if entity.name not in names:
                names.append(entity.name)
            if len(names) == 6:
                break
        if len(names) == 6:
            break
    if len(names) < 3:
        raise PlanError("T1 source has insufficient named grounding in first half")
    r1 = RoundBuilder(
        turn_id="t10c-r01-source-ingest-a",
        new_view_ids=[_logical("source", view) for view in first],
        evidence_view_ids=[_logical("source", view) for view in first],
        capability={
            "primary": "CR",
            "supporting": ["SR"],
            "sense_nova_subtask": "incremental source-state construction",
        },
        semantic_signature="G_sequence->F_pose_chain->B_partial->V",
        question_zh="先按顺序观察这六个漫游视角。暂时不用做假想转向，只说你已经能稳定辨认哪些主要地标。",
        answer_key={"grounded_categories_zh": names},
    )
    r1.node("g1", "G", ["view_sequence", "referent_set"], "entity_set", names)
    r1.node("f1", "F", ["camera_pose_sequence"], "transform_set", "source_pose_chain_a", ["g1"])
    r1.node("b1", "B", ["entity_set", "transform_set"], "belief_partial", names, ["g1", "f1"])
    r1.node("v1", "V", ["belief", "instance_masks"], "boolean", True, ["b1"])
    c11 = r1.claim(
        "c1",
        "g1",
        "grounding",
        "前半段漫游建立了第一批稳定地标。",
        names,
        r1.evidence_view_ids,
        "conclusion",
    )
    r1.sentence(
        "conclusion",
        f"目前我能稳定辨认出{'、'.join(names)}。我会先保留这些地标之间的布局，不提前猜新的观察方向。",
        [c11],
    )

    second = source.view_ids[6:]
    ov, fv, tv = x["origin_view"], x["facing_view"], x["candidate_view"]
    origin_ref = source.reference(origin, ov)
    facing_ref = source.reference(facing, fv)
    target_ref = source.reference(target, tv)
    r2 = RoundBuilder(
        turn_id="t10c-r02-source-ingest-b",
        new_view_ids=[_logical("source", view) for view in second],
        evidence_view_ids=sorted(
            {_logical("source", view) for view in [ov, fv, tv, *second]},
            key=source_ids.index,
        ),
        capability={
            "primary": "CR",
            "supporting": ["SR", "PT"],
            "sense_nova_subtask": "global state commit across source views",
        },
        semantic_signature="G_anchors->F_pose_chain->B_global->V",
        question_zh="继续看完余下五个漫游视角。为了后面做视角采择，请确认现在是否已经能把三个地标放进同一空间：作为站位参照的物体、作为面向参照的物体，以及稍后要定位的目标。",
        answer_key={"global_state_ready": True, "cross_view_required": x["cross_view_required"]},
    )
    r2.node(
        "g1",
        "G",
        ["view_sequence", "entity_triplet"],
        "entity_set",
        [origin.source_id, facing.source_id, target.source_id],
    )
    r2.node(
        "f1",
        "F",
        ["camera_pose_sequence", "first_view_frame"],
        "transform_set",
        "source_pose_chain_full",
        ["g1"],
    )
    r2.node(
        "b1",
        "B",
        ["entity_set", "transform_set"],
        "belief_global",
        [origin.source_id, facing.source_id, target.source_id],
        ["g1", "f1"],
    )
    r2.node("v1", "V", ["belief", "grounding_certificates"], "boolean", True, ["b1"])
    c21 = r2.claim(
        "c1",
        "g1",
        "anchor_grounding",
        f"站位、面向与目标地标分别由{origin_ref}、{facing_ref}和{target_ref}提供。",
        [origin.source_id, facing.source_id, target.source_id],
        [_logical("source", view) for view in [ov, fv, tv]],
        "cue",
    )
    c22 = r2.claim(
        "c2",
        "b1",
        "state_commit",
        "三个地标已经注册到同一全局状态。",
        True,
        r2.evidence_view_ids,
        "conclusion",
    )
    r2.sentence(
        "cue",
        f"可以：站位参照是{origin_ref}，面向参照是{facing_ref}，待定位目标是{target_ref}。",
        [c21],
    )
    r2.sentence(
        "conclusion",
        "它们不必在同一张图里同时出现；沿漫游顺序对齐后，已经能放进同一空间状态。",
        [c22],
    )

    quadrant = QUADRANT_ZH[fact["quadrant"]]
    r3 = RoundBuilder(
        turn_id="t10c-r03-heldout-prediction",
        new_view_ids=[],
        evidence_view_ids=[
            _logical("source", view) for view in sorted({ov, fv, tv}, key=source.view_ids.index)
        ],
        capability={
            "primary": "PT",
            "supporting": ["CR", "SR"],
            "sense_nova_subtask": "held-out object-anchored perspective prediction",
        },
        semantic_signature="B_global->F_object_anchor->P->R->V",
        question_zh=f"现在不看新图。假如你走到{origin_ref}旁边，面向{facing_ref}站定，{target_ref}会落在你的哪个方位？",
        answer_key={"quadrant": fact["quadrant"], "cross_view_required": x["cross_view_required"]},
        diagnostics={
            "angle_margin_deg": fact["angle_margin_deg"],
            "query_xy_m": fact["query_xy_m"],
        },
    )
    r3.node(
        "b1",
        "B",
        ["committed_scene_state"],
        "belief_global",
        [origin.source_id, facing.source_id, target.source_id],
    )
    r3.node(
        "f1",
        "F",
        ["origin_entity", "facing_entity"],
        "frame_object_anchored",
        {"origin": origin.source_id, "facing": facing.source_id},
        ["b1"],
    )
    r3.node(
        "p1",
        "P",
        ["belief_global", "object_anchored_frame"],
        "prediction_query_relation",
        fact,
        ["b1", "f1"],
    )
    r3.node("r1", "R", ["entity_pair", "object_anchored_frame"], "relation_quadrant", fact, ["p1"])
    r3.node("v1", "V", ["relation", "angle_extent_margin"], "boolean", True, ["r1"])
    c31 = r3.claim(
        "c1",
        "b1",
        "memory_cue",
        f"三个查询地标来自已经看过的{origin_ref}、{facing_ref}和{target_ref}。",
        [origin.source_id, facing.source_id, target.source_id],
        r3.evidence_view_ids,
        "cue",
    )
    c32 = r3.claim(
        "c2",
        "f1",
        "frame_transform",
        f"以{origin.name}为站位、朝向{facing.name}建立新的前后左右。",
        {"origin": origin.source_id, "facing": facing.source_id},
        r3.evidence_view_ids,
        "transform",
    )
    c33 = r3.claim(
        "c3",
        "r1",
        "prediction",
        f"{target.name}位于{quadrant}。",
        fact,
        r3.evidence_view_ids,
        "conclusion",
    )
    r3.sentence(
        "cue", f"我先从已有漫游中取回{origin_ref}、{facing_ref}和{target_ref}的位置。", [c31]
    )
    r3.sentence(
        "transform",
        f"把自己放到{origin.name}旁，并把朝向{facing.name}的方向设为正前方后，重新换算目标的位置。",
        [c32],
    )
    r3.sentence("conclusion", f"{target.name}会在你的{quadrant}。", [c33])

    target_local = _local_ref(target_catalog, x["candidate_t"], x["target_view"])
    r4 = RoundBuilder(
        turn_id="t10c-r04-render-verification",
        new_view_ids=[heldout_id],
        evidence_view_ids=[heldout_id],
        capability={
            "primary": "PT",
            "supporting": ["V"],
            "sense_nova_subtask": "rendered counterfactual verification",
        },
        semantic_signature="G_rendered_target->R_observed->V(prediction,observation)",
        question_zh="现在给出模拟器从该假想站位渲染的真实目标视角。它是否支持刚才的方位预测？",
        answer_key={"prediction_verified": True, "quadrant": fact["quadrant"]},
    )
    r4.node("g1", "G", ["heldout_view", "target"], "entity", x["candidate_t"].source_id)
    r4.node("r1", "R", ["entity_pair", "heldout_view_frame"], "relation_quadrant", fact, ["g1"])
    r4.node("v1", "V", ["prediction", "observed_relation"], "boolean", True, ["r1"])
    c41 = r4.claim(
        "c1",
        "g1",
        "render_cue",
        f"真实目标视角中可看到{target_local}。",
        x["candidate_t"].source_id,
        [heldout_id],
        "cue",
    )
    c42 = r4.claim(
        "c2",
        "v1",
        "verification",
        f"渲染视角与先前的{quadrant}预测一致。",
        True,
        [heldout_id],
        "conclusion",
    )
    r4.sentence("cue", f"真实目标视角里，{target_local}落在按面向锚点定义的{quadrant}区域。", [c41])
    r4.sentence("conclusion", "因此它支持刚才的预测；这个结论是在答案之后才用新视角核验的。", [c42])

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "scene_id": source.scene_id,
        "family_id": source.family_id,
        "split_lock": source.split_group,
        "trajectory_class": "T10C",
        "task_scope": "source_to_heldout_perspective_prediction",
        "frame_contract": {"surface_zh": FRAME_SURFACE_ZH},
        "source_bundles": {"source_t1": str(source.root), "heldout_t10": str(target_catalog.root)},
        "view_ids": [*source_ids, heldout_id],
        "acquisition_id": acquisition_id,
        "rounds": [r.payload(i + 1) for i, r in enumerate([r1, r2, r3, r4])],
        "composition_certificate": {
            "source_view_count": len(source.view_ids),
            "target_view_id": x["target_view"],
            "cross_view_required": x["cross_view_required"],
            "prediction_before_target_release": True,
        },
    }
    try:
        validate_artifact(artifact)
    except ContractError as error:
        raise PlanError(str(error)) from error
    return artifact


def compile_composition_catalog(
    catalog_path: Path,
    code_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    catalog_path, code_root, output_dir = map(Path, (catalog_path, code_root, output_dir))
    catalog = read_json(catalog_path)
    t1 = {
        job["scene"]: job
        for job in catalog["collection_jobs"]
        if job["trajectory_class"] == "T1" and job["status"] == "passed"
    }
    jobs = [
        job
        for job in catalog["collection_jobs"]
        if job["trajectory_class"] == "T10" and job["status"] == "passed" and job["scene"] in t1
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"requested": len(jobs), "compiled": [], "rejected": []}
    episode_rows, isolated_rows = [], []
    for job in sorted(jobs, key=lambda row: row["job_id"]):
        acquisition_id = f"{job['scene']}_t1_to_t10_seed17"
        try:
            t1_root = resolve_bundle(code_root, t1[job["scene"]]["quality_url"])
            t10_root = resolve_bundle(code_root, job["quality_url"])
            artifact = compile_t1_to_t10(t1_root, t10_root, acquisition_id)
            rgb_paths = {}
            for view in TrajectoryCatalog(t1_root).view_ids:
                logical = _logical("source", view)
                rgb_paths[logical] = _copy_rgb(
                    t1_root, view, output_dir / "rgb" / acquisition_id / f"{logical}.png"
                )
            target_view = artifact["composition_certificate"]["target_view_id"]
            heldout = _logical("heldout", target_view)
            rgb_paths[heldout] = _copy_rgb(
                t10_root, target_view, output_dir / "rgb" / acquisition_id / f"{heldout}.png"
            )
            artifact["rgb_paths"] = rgb_paths
            scene_dir = output_dir / "scenes"
            scene_dir.mkdir(exist_ok=True)
            path = scene_dir / f"{acquisition_id}.dialogue.json"
            artifact["dialogue_path"] = str(path.resolve())
            artifact["input_manifest"] = {
                "t1_quality_sha256": sha256(t1_root / "quality_report.json"),
                "t10_quality_sha256": sha256(t10_root / "quality_report.json"),
            }
            path.write_text(json.dumps(artifact, ensure_ascii=False, indent=1), encoding="utf-8")
            episode, isolated = to_sft_records(artifact, rgb_paths)
            episode_rows.append(episode)
            isolated_rows.extend(isolated)
            report["compiled"].append(
                {
                    "acquisition_id": acquisition_id,
                    "dialogue_path": str(path.resolve()),
                    "cross_view_required": artifact["composition_certificate"][
                        "cross_view_required"
                    ],
                }
            )
        except (PlanError, CatalogError, KeyError, ValueError) as error:
            report["rejected"].append({"acquisition_id": acquisition_id, "reason": str(error)})
    for filename, rows in (
        ("all.dialogue_episode_sft.jsonl", episode_rows),
        ("all.dialogue_isolated_sft.jsonl", isolated_rows),
    ):
        with (output_dir / filename).open("w", encoding="utf-8") as sink:
            for row in rows:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "corpus_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return report
