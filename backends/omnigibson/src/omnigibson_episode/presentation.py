"""Build a self-contained, evidence-backed oral-demo website for one episode."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from omnigibson_episode.quality import _depth_rgb, _mask_rgb

RELATION_LABELS = {
    "left_of": "左侧",
    "right_of": "右侧",
    "in_front_of": "前方",
    "behind": "后方",
    "above": "上方",
    "below": "下方",
}

OPERATION_LABELS = {
    "G": "Ground",
    "F": "Frame",
    "B": "Belief",
    "M": "Metric",
    "R": "Relation",
    "P": "Perspective",
    "V": "Verify",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"presentation input is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"presentation input must be an object: {path.name}")
    return payload


def _round_vector(values: list[Any], digits: int = 3) -> list[float]:
    return [round(float(value), digits) for value in values]


def _display_name(entity: dict[str, Any]) -> str:
    category = str(entity.get("raw_label") or "object").replace("_", " ")
    source = str(entity.get("source_entity_id") or "unknown")
    suffix = source.rsplit("_", 1)[-1]
    return f"{category} · {suffix}"


def _write_rgb(rgb: np.ndarray, path: Path, *, width: int, quality: int) -> None:
    image = Image.fromarray(rgb)
    height = round(image.height * width / image.width)
    image.resize((width, height), Image.Resampling.LANCZOS).save(
        path,
        format="WEBP",
        quality=quality,
        method=6,
    )


def _write_mask(mask: np.ndarray, path: Path, *, width: int) -> None:
    image = Image.fromarray(_mask_rgb(mask))
    height = round(image.height * width / image.width)
    image.resize((width, height), Image.Resampling.NEAREST).save(
        path,
        format="WEBP",
        lossless=True,
        method=6,
    )


def _write_depth(depth: np.ndarray, path: Path, *, width: int, near_m: float, far_m: float) -> None:
    image = Image.fromarray(_depth_rgb(depth, near_m, far_m))
    height = round(image.height * width / image.width)
    image.resize((width, height), Image.Resampling.LANCZOS).save(
        path,
        format="WEBP",
        quality=92,
        method=6,
    )


def _entity_payload(
    scene: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    regions = {
        str(region["region_id"]): str(region.get("raw_label") or "unassigned")
        for region in scene["regions"]
    }
    entities = []
    by_id = {}
    for raw in scene["entities"]:
        entity = {
            "id": str(raw["entity_id"]),
            "short_id": str(raw["entity_id"])[:8],
            "name": _display_name(raw),
            "source_name": str(raw["source_entity_id"]),
            "category": str(raw.get("raw_label") or "object"),
            "region": regions.get(str(raw.get("region_id")), "unassigned"),
            "position_m": _round_vector(raw["world_from_entity"]["translation_m"]),
        }
        entities.append(entity)
        by_id[entity["id"]] = entity
    return entities, by_id


def _query_entities(query: dict[str, Any]) -> tuple[str, str]:
    parameters = {
        str(node["node_id"]): node.get("parameters", {})
        for node in query["operation_graph"]["nodes"]
    }
    subject = str(parameters.get("ground_subject", {}).get("entity_id", ""))
    reference = str(parameters.get("ground_reference", {}).get("entity_id", ""))
    if not subject or not reference:
        evidence = [str(identifier) for identifier in query["evidence_entity_ids"]]
        if len(evidence) != 2:
            raise ValueError("query must identify exactly two evidence entities")
        return evidence[0], evidence[1]
    return subject, reference


def _query_payload(
    queries: list[dict[str, Any]],
    entity_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for index, query in enumerate(queries):
        subject_id, reference_id = _query_entities(query)
        subject = entity_by_id[subject_id]
        reference = entity_by_id[reference_id]
        relation = str(query["answer"])
        graph = query["operation_graph"]
        nodes = []
        for node in graph["nodes"]:
            operation = str(node["operation"])
            nodes.append(
                {
                    "id": str(node["node_id"]),
                    "operation": operation,
                    "operation_name": OPERATION_LABELS.get(operation, operation),
                    "inputs": [str(value) for value in node["inputs"]],
                    "output_type": str(node["output_type"]),
                    "parameters": node.get("parameters", {}),
                }
            )
        result.append(
            {
                "index": index,
                "id": str(query["query_id"]),
                "short_id": str(query["query_id"])[:8],
                "subject": subject,
                "reference": reference,
                "relation": relation,
                "relation_label": RELATION_LABELS.get(relation, relation),
                "formal_question": (
                    f"在 canonical frame 中, {subject['name']} 相对 "
                    f"{reference['name']} 的空间关系是什么?"
                ),
                "answer": relation,
                "answer_label": RELATION_LABELS.get(relation, relation),
                "margin_m": round(float(query["decision_margin"]), 3),
                "evidence_view_ids": [str(value) for value in query["evidence_view_ids"]],
                "status": str(query["status"]),
                "certificate": query["certificate"],
                "nodes": nodes,
                "answer_node": str(graph["answer_node"]),
                "external_inputs": graph["external_inputs"],
            }
        )
    return result


def _view_payload(
    *,
    bundle: Path,
    output: Path,
    episode: dict[str, Any],
    trajectory: dict[str, Any],
    quality: dict[str, Any],
    render: dict[str, Any],
) -> list[dict[str, Any]]:
    observation_by_id = {str(item["view_id"]): item for item in episode["observations"]}
    delta_by_step = {int(item["step"]): item for item in episode["state_deltas"]}
    quality_by_id = {str(item["view_id"]): item for item in quality["views"]}
    render_by_id = {str(item["view_id"]): item for item in render["views"]}
    media = output / "media"
    media.mkdir(parents=True)
    accumulated: set[str] = set()
    views = []
    near_m = float(render["sensor_contract"]["near_m"])
    far_m = float(render["sensor_contract"]["far_m"])
    for item in trajectory["views"]:
        view_id = str(item["view_id"])
        step = int(item["step"])
        observation = observation_by_id[view_id]
        delta = delta_by_step[step]
        accumulated.update(str(value) for value in delta["added"])
        artifact = Path(str(render_by_id[view_id]["artifact"]["path"])).resolve()
        try:
            artifact.relative_to(bundle)
        except ValueError as error:
            raise ValueError(f"view artifact is outside bundle: {view_id}") from error
        with np.load(artifact, allow_pickle=False) as arrays:
            rgb = arrays["rgb"]
            depth = arrays["depth_m"]
            instance = arrays["instance_id"]
            semantic = arrays["semantic_id"]
        names = {
            "rgb": f"media/{view_id}-rgb.webp",
            "depth": f"media/{view_id}-depth.webp",
            "instance": f"media/{view_id}-instance.webp",
            "semantic": f"media/{view_id}-semantic.webp",
            "thumbnail": f"media/{view_id}-thumb.webp",
        }
        _write_rgb(rgb, output / names["rgb"], width=1024, quality=92)
        _write_rgb(rgb, output / names["thumbnail"], width=240, quality=82)
        _write_depth(depth, output / names["depth"], width=1024, near_m=near_m, far_m=far_m)
        _write_mask(instance, output / names["instance"], width=1024)
        _write_mask(semantic, output / names["semantic"], width=1024)
        views.append(
            {
                "id": view_id,
                "step": step,
                "role": str(item["role"]),
                "position_m": _round_vector(item["world_from_agent"]["translation_m"]),
                "rotation_xyzw": _round_vector(item["world_from_agent"]["rotation_xyzw"]),
                "cumulative_distance_m": round(float(item["cumulative_distance_m"]), 3),
                "visible_count": len(observation["visible_entity_ids"]),
                "visible_entity_ids": [
                    str(identifier) for identifier in observation["visible_entity_ids"]
                ],
                "added_count": len(delta["added"]),
                "reobserved_count": len(delta["reobserved"]),
                "belief_entity_count": len(accumulated),
                "valid_depth_fraction": quality_by_id[view_id]["valid_depth_fraction"],
                "sharpness": quality_by_id[view_id]["sharpness_laplacian_variance"],
                "media": names,
            }
        )
    return views


def _presentation_payload(bundle: Path, output: Path) -> dict[str, Any]:
    episode = _read_json(bundle / "spatial_episode.json")
    scene = _read_json(bundle / "scene_ir.json")
    trajectory = _read_json(bundle / "trajectory_plan.json")
    render = _read_json(bundle / "render_report.json")
    quality = _read_json(bundle / "quality_report.json")
    if quality.get("integrity_status") != "pass":
        raise ValueError("oral presentation requires a passing quality report")
    entities, entity_by_id = _entity_payload(scene)
    views = _view_payload(
        bundle=bundle,
        output=output,
        episode=episode,
        trajectory=trajectory,
        quality=quality,
        render=render,
    )
    queries = _query_payload(episode["queries"], entity_by_id)
    category_counts = Counter(item["category"] for item in entities)
    relation_counts = Counter(item["answer"] for item in queries)
    operation_counts = Counter(
        node["operation"] for query in queries for node in query["nodes"]
    )
    capabilities = scene["capabilities"]
    return {
        "schema_version": "omnigibson_oral_demo.v1",
        "episode": {
            "id": str(episode["episode_id"]),
            "short_id": str(episode["episode_id"])[:8],
            "family_id": str(episode["family_id"]),
            "recipe_id": str(episode["recipe_id"]),
            "scene_name": str(episode["provenance"]["source_scene_id"]).split(":")[0],
            "source": str(episode["provenance"]["source_name"]),
            "source_version": str(episode["provenance"]["source_version"]),
            "resolution": (
                f"{render['sensor_contract']['width_px']}x"
                f"{render['sensor_contract']['height_px']}"
            ),
            "model_visible": episode["channel_policy"]["model_visible"],
            "supervision": episode["channel_policy"]["supervision"],
            "oracle_only": episode["channel_policy"]["oracle_only"],
        },
        "metrics": {
            "view_count": len(views),
            "entity_count": len(entities),
            "rendered_entity_count": len(scene["runtime_semantic_id_map"]),
            "query_count": len(queries),
            "relation_candidate_count": len(
                _read_json(bundle / "relation_oracle.json")["candidates"]
            ),
            "path_length_m": round(float(trajectory["outbound_geodesic_distance_m"]), 2),
            "loop_rgb_psnr_db": quality["loop_closure"]["rgb_psnr_db"],
            "minimum_depth_fraction": quality["summary"]["minimum_valid_depth_fraction"],
        },
        "quality": quality,
        "trajectory": {
            "closed_loop": bool(trajectory["closed_loop"]),
            "canonical_frame": "+X right · +Y forward · +Z up",
            "seed": int(trajectory["seed"]),
            "area_m2": round(float(trajectory["primary_island_area_m2"]), 2),
        },
        "entities": entities,
        "top_categories": [
            {"name": name, "count": count}
            for name, count in category_counts.most_common(10)
        ],
        "relation_counts": dict(sorted(relation_counts.items())),
        "operation_counts": dict(sorted(operation_counts.items())),
        "views": views,
        "queries": queries,
        "capabilities": capabilities["capabilities"],
        "capability_gaps": capabilities["gaps"],
        "provenance": episode["provenance"],
    }


def build_oral_presentation(
    bundle_directory: Path,
    output_directory: Path | None = None,
) -> Path:
    """Generate the interactive oral demo atomically and return its index path."""

    bundle = bundle_directory.resolve()
    output = (output_directory or bundle / "oral_demo").resolve()
    staging = output.with_name(f".{output.name}.staging.{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        payload = _presentation_payload(bundle, staging)
        template_root = Path(__file__).with_name("templates")
        html_template = (template_root / "oral_demo.html").read_text(encoding="utf-8")
        safe_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
            "</", "<\\/"
        )
        page = html_template.replace("__EPISODE_DATA__", safe_payload)
        (staging / "index.html").write_text(page, encoding="utf-8")
        for filename in ("oral_demo.css", "oral_demo.js"):
            shutil.copy2(template_root / filename, staging / filename)
        if output.exists():
            shutil.rmtree(output)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "index.html"
