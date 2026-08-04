"""End-to-end geometry-to-episode dataset production pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from episode3d.bundles import Bundle, file_sha256, read_json, stable_id
from episode3d.compilers import (
    compile_bundle_questions,
    compile_t10_predictions,
    model_input_frame_is_admissible,
    recognizable_witnesses,
    visual_quality_policy,
)
from episode3d.exporters import (
    export_benchmark,
    export_rlvr,
    export_state_aux,
    export_training_arms,
)
from episode3d.implementation_binding import build_implementation_binding
from episode3d.models import CompiledBundle, QuestionSpec, Rejection
from episode3d.verifiers import replay_question

SCHEMA_VERSION = "epispace.release_manifest.v1"
UTC = timezone.utc


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )
    _write_text_atomic(path, text)


def _source_inventory(
    *, config_path: Path, bundles: list[Bundle], acquisition: dict[str, Any]
) -> dict[str, Any]:
    """Freeze every consumed JSON and materialized RGB without copying sources.

    Sensor archives are large and live under the simulator asset boundary.  We
    record their size/mtime identity plus the decoded raw-pixel SHA stored by
    the RGB extractor.  Every small geometry/task JSON and every release-owned
    PNG/sidecar receives a cryptographic file hash.
    """

    inventory_bundles: list[dict[str, Any]] = []
    for bundle in sorted(bundles, key=lambda item: (item.source_sweep, item.episode_id)):
        source_files: dict[str, dict[str, Any]] = {}
        optional_filenames = ("render_report.json", "reasoning_tasks.json")
        for filename in (
            *Bundle.REQUIRED_FILES,
            *optional_filenames,
        ):
            path = bundle.root / filename
            if path.is_file():
                source_files[filename] = {
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
        views: list[dict[str, Any]] = []
        for view in bundle.views:
            metadata_path = view.rgb_path.with_suffix(".rgb.json")
            metadata = read_json(metadata_path)
            sensor_stat = view.sensor_path.stat()
            with Image.open(view.rgb_path) as image:
                image.load()
                if (
                    image.mode != "RGB"
                    or metadata.get("mode") != "RGB"
                    or image.width != metadata.get("width")
                    or image.height != metadata.get("height")
                ):
                    raise ValueError(
                        f"source inventory RGB contract mismatch: {view.rgb_path}"
                    )
                pixel_sha256 = hashlib.sha256(image.tobytes()).hexdigest()
                if pixel_sha256 != metadata.get("rgb_sha256"):
                    raise ValueError(
                        f"source inventory RGB pixel digest mismatch: {view.rgb_path}"
                    )
            views.append(
                {
                    "view_id": view.view_id,
                    "rgb": str(view.rgb_path),
                    "rgb_file_sha256": file_sha256(view.rgb_path),
                    "rgb_bytes": view.rgb_path.stat().st_size,
                    "rgb_pixel_sha256": pixel_sha256,
                    "rgb_sidecar": str(metadata_path),
                    "rgb_sidecar_sha256": file_sha256(metadata_path),
                    "source_sensor": str(view.sensor_path),
                    "source_sensor_sha256": file_sha256(view.sensor_path),
                    "source_sensor_bytes": sensor_stat.st_size,
                    "source_sensor_mtime_ns": sensor_stat.st_mtime_ns,
                }
            )
        inventory_bundles.append(
            {
                "source_sweep": bundle.source_sweep,
                "trajectory_class": bundle.trajectory_class,
                "scene_id": bundle.scene_id,
                "episode_id": bundle.episode_id,
                "bundle_root": str(bundle.root),
                "source_files": source_files,
                "absent_optional_files": [
                    filename for filename in optional_filenames if filename not in source_files
                ],
                "views": views,
            }
        )
    return {
        "schema_version": "epispace.source_inventory.v1",
        "config": {
            "path": str(config_path),
            "sha256": file_sha256(config_path),
        },
        "sweeps": [
            {
                "name": item["name"],
                "path": item["sweep_plan"],
                "sha256": item["sweep_plan_sha256"],
            }
            for item in acquisition["sweeps"]
        ],
        "bundle_count": len(inventory_bundles),
        "view_count": sum(len(item["views"]) for item in inventory_bundles),
        "bundles": inventory_bundles,
    }


def _artifact_record(path: Path, *, records: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.name,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }
    if records is not None:
        result["records"] = records
    return result


_DERIVED_OUTPUTS = (
    "compute_matching",
    "evaluation_smoke",
    "corpus_audit.json",
    "corpus_audit.md",
    "semantic_visual_audit",
    # Pre-closure packet names are removed as well.  They are not valid
    # formal-release locations and must not survive a corpus rebuild where
    # their release/source hashes would become stale.
    "semantic_visual_audit_packet.json",
    "semantic_visual_audit_packet.md",
    "semantic_visual_audit_packet.result.json",
    "semantic_visual_audit_packet.result.md",
    "final_release_index.json",
)


def _invalidate_derived_outputs(output_dir: Path) -> None:
    """Remove reports whose source hashes became stale after a dataset rebuild."""

    for name in _DERIVED_OUTPUTS:
        path = output_dir / name
        if path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _scene_splits(
    scene_ids: set[str],
    *,
    seed: str,
    requested_counts: dict[str, int],
) -> dict[str, str]:
    expected = sum(requested_counts.values())
    if expected != len(scene_ids):
        raise ValueError(f"split counts sum to {expected}, but discovered {len(scene_ids)} scenes")
    ordered = sorted(
        scene_ids,
        key=lambda scene_id: stable_id("splitrank", seed, scene_id),
    )
    result: dict[str, str] = {}
    offset = 0
    for split in ("train", "val", "test"):
        count = int(requested_counts[split])
        for scene_id in ordered[offset : offset + count]:
            result[scene_id] = split
        offset += count
    return result


def _load_bundles(
    config: dict[str, Any],
    config_path: Path,
    media_root: Path,
) -> tuple[list[Bundle], list[Rejection], dict[str, Any]]:
    allowed_statuses = set(config["quality_policy"]["accepted_plan_statuses"])
    bundles: list[Bundle] = []
    rejections: list[Rejection] = []
    acquisition: dict[str, Any] = {"sweeps": []}
    seen_episode_ids: set[str] = set()
    for source in config["sources"]:
        plan_path = _resolve(config_path, source["sweep_plan"])
        plan = read_json(plan_path)
        status_counts: Counter[str] = Counter()
        loaded = 0
        missing_reasoning_manifests = 0
        for job in plan["jobs"]:
            status = str(job.get("status", "missing"))
            status_counts[status] += 1
            job_id = str(job["job_id"])
            if status not in allowed_statuses:
                rejections.append(
                    Rejection(
                        "acquisition_job",
                        job_id,
                        f"plan_status_{status}",
                        "job excluded by strict quality policy",
                        str(plan_path),
                    )
                )
                continue
            root = Path(job["bundle"])
            try:
                bundle = Bundle(
                    root,
                    trajectory_class=str(source["trajectory_class"]),
                    source_sweep=str(source["name"]),
                    job_status=status,
                )
                bundle.materialize_model_rgb(media_root)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                rejections.append(
                    Rejection(
                        "bundle",
                        job_id,
                        "bundle_validation_failed",
                        str(error),
                        str(plan_path),
                    )
                )
                continue
            if bundle.episode_id in seen_episode_ids:
                raise ValueError(f"duplicate episode_id: {bundle.episode_id}")
            seen_episode_ids.add(bundle.episode_id)
            bundles.append(bundle)
            loaded += 1
            if not bundle.reasoning_tasks_present:
                missing_reasoning_manifests += 1
        acquisition["sweeps"].append(
            {
                "name": source["name"],
                "trajectory_class": source["trajectory_class"],
                "role": source["role"],
                "sweep_plan": str(plan_path),
                "sweep_plan_sha256": file_sha256(plan_path),
                "planned_jobs": len(plan["jobs"]),
                "status_source": "sweep_plan.jobs[].status",
                "status_counts": dict(sorted(status_counts.items())),
                "strict_bundles_loaded": loaded,
                "missing_optional_reasoning_task_manifests": (missing_reasoning_manifests),
            }
        )
    acquisition["planned_jobs"] = sum(item["planned_jobs"] for item in acquisition["sweeps"])
    acquisition["strict_bundles_loaded"] = len(bundles)
    acquisition["model_rgb_root"] = str(media_root)
    acquisition["model_rgb_policy"] = "raw rgb channel extracted from sensors.npz"
    return bundles, rejections, acquisition


def _compile(
    bundles: list[Bundle],
    splits: dict[str, str],
    rejections: list[Rejection],
) -> list[CompiledBundle]:
    compiled: list[CompiledBundle] = []
    t10_by_scene: dict[str, list[Bundle]] = defaultdict(list)
    source_by_scene: dict[str, list[CompiledBundle]] = defaultdict(list)
    source_rank = {"T1": 0, "T3": 1, "T4": 2, "T7": 3, "T8": 4}
    for bundle in bundles:
        if bundle.trajectory_class == "T10":
            t10_by_scene[bundle.scene_id].append(bundle)
            rejections.append(
                Rejection(
                    "bundle",
                    bundle.episode_id,
                    "held_out_verifier_only",
                    "T10 target RGB is excluded from write prefixes",
                    str(bundle.root),
                )
            )
            continue
        questions, task_rejections = compile_bundle_questions(bundle)
        rejections.extend(task_rejections)
        item = CompiledBundle(
            bundle=bundle,
            split=splits[bundle.scene_id],
            questions=questions,
        )
        compiled.append(item)
        source_by_scene[bundle.scene_id].append(item)
    for scene_id, targets in t10_by_scene.items():
        candidates = source_by_scene.get(scene_id, [])
        if not candidates:
            for target in targets:
                rejections.append(
                    Rejection(
                        "bundle",
                        target.episode_id,
                        "t10_source_prefix_missing",
                        "no strict non-T10 trajectory exists for this scene",
                        str(target.root),
                    )
                )
            continue
        for target in targets:
            compiled_sources = [
                (compile_t10_predictions(target, candidate.bundle), candidate)
                for candidate in candidates
            ]
            predictions, source = max(
                compiled_sources,
                key=lambda item: (
                    len(item[0]),
                    len({spec.answer_value for spec in item[0]}),
                    -source_rank.get(item[1].bundle.trajectory_class, 100),
                    item[1].bundle.episode_id,
                ),
            )
            if predictions:
                source.questions.extend(predictions)
            else:
                rejections.append(
                    Rejection(
                        "bundle",
                        target.episode_id,
                        "t10_no_quality_admissible_source_prefix",
                        "no same-scene trajectory yields resolvable target-view anchors",
                        str(target.root),
                    )
                )
    retained: list[CompiledBundle] = []
    for item in compiled:
        if item.questions:
            retained.append(item)
            continue
        rejections.append(
            Rejection(
                "bundle",
                item.bundle.episode_id,
                "no_compiled_questions_after_quality_gating",
                "trajectory retained by acquisition but produced no admissible QA",
                str(item.bundle.root),
            )
        )
    return retained


def _episode_ir(item: CompiledBundle, observable_belief: dict[str, Any]) -> dict[str, Any]:
    bundle = item.bundle
    return {
        "schema_version": "epispace.episode_ir.v1",
        "episode_id": bundle.episode_id,
        "scene_id": bundle.scene_id,
        "scene_family_id": stable_id("scene-family", bundle.scene_id),
        "trajectory_family_id": stable_id("trajectory-family", bundle.scene_id, bundle.episode_id),
        "split": item.split,
        "split_lock": f"scene:{bundle.scene_id}",
        "trajectory_class": bundle.trajectory_class,
        "source_sweep": bundle.source_sweep,
        "source_bundle": str(bundle.root),
        "channel_policy": {
            "model_visible": [
                "rgb",
                "view_order",
                "camera_height_m",
                "horizontal_fov_deg",
                "natural_language_question",
            ],
            "supervision_only": ["observable_belief", "typed_program"],
            "oracle_only": ["depth", "instance", "pose", "certificate", "world_truth"],
        },
        "observations": [
            {
                "view_id": view.view_id,
                "step": view.step,
                "role": view.role,
                "rgb": str(view.rgb_path),
                "camera_height_m": view.camera_height_m,
                "horizontal_fov_deg": view.horizontal_fov_deg,
            }
            for view in bundle.views
        ],
        "observable_belief": observable_belief,
        "questions": [question.as_ir_dict() for question in item.questions],
    }


def _corpus_gates(
    *,
    compiled: list[CompiledBundle],
    selected_train: list[QuestionSpec],
    benchmark: list[dict[str, Any]],
    episode_sft: list[dict[str, Any]],
    isolated_sft: list[dict[str, Any]],
    state_aux: list[dict[str, Any]],
    rlvr: list[dict[str, Any]],
    heldout_program_ids: set[str],
    media_root: Path,
    observable_beliefs: dict[str, dict[str, Any]],
    requirements: dict[str, Any],
) -> dict[str, Any]:
    train_programs = {spec.program.program_id for spec in selected_train}
    train_atoms = {atom for spec in selected_train for atom in spec.program.atoms}
    heldout_specs = [
        row for row in benchmark if row["program"]["program_id"] in heldout_program_ids
    ]
    heldout_atoms = {atom for row in heldout_specs for atom in row["program"].get("atoms", ())}
    episode_fact_ids = [
        fact_id for row in episode_sft for fact_id in row["comparison_contract"]["fact_ids"]
    ]
    isolated_fact_ids = [
        fact_id for row in isolated_sft for fact_id in row["comparison_contract"]["fact_ids"]
    ]
    scene_to_splits: dict[str, set[str]] = defaultdict(set)
    consistency_to_splits: dict[str, set[str]] = defaultdict(set)
    consistency_to_variants: dict[str, set[str]] = defaultdict(set)
    consistency_to_specs: dict[str, list[QuestionSpec]] = defaultdict(list)
    all_fact_ids: list[str] = []
    for item in compiled:
        scene_to_splits[item.bundle.scene_id].add(item.split)
        for question in item.questions:
            all_fact_ids.append(question.fact_id)
            if question.consistency_group:
                consistency_to_splits[question.consistency_group].add(item.split)
                consistency_to_variants[question.consistency_group].add(question.family_variant)
                consistency_to_specs[question.consistency_group].append(question)
    target_paths = {
        str(row["certificate"].get("oracle_target_rgb"))
        for row in benchmark
        if row["certificate"].get("oracle_target_rgb")
    }
    sft_messages = [row["messages"] for row in episode_sft + isolated_sft + state_aux]
    serialized_inputs = json.dumps(sft_messages, ensure_ascii=False)
    all_message_groups = [
        *sft_messages,
        *(row["prompt"] for row in rlvr),
        *(row["model_input"] for row in benchmark),
    ]
    all_image_paths = {
        path for messages in all_message_groups for path in _message_image_paths(messages)
    }
    model_frame_admission: dict[str, bool] = {}
    for item in compiled:
        for view in item.bundle.views:
            path = str(view.rgb_path)
            passed = model_input_frame_is_admissible(item.bundle, view.view_id)
            if path in model_frame_admission and model_frame_admission[path] != passed:
                raise ValueError(f"inconsistent RGB frame admission for {path}")
            model_frame_admission[path] = passed
    inadmissible_exported_model_frames = sorted(
        path
        for path in all_image_paths
        if not model_frame_admission.get(path, False)
    )
    sft_image_paths = {path for messages in sft_messages for path in _message_image_paths(messages)}
    rgb_digests = {
        path: _verified_raw_rgb_digest(Path(path), media_root)
        for path in all_image_paths | target_paths
        if path and path != "None"
    }
    target_hashes = {rgb_digests[path] for path in target_paths if path in rgb_digests}
    sft_hashes = {rgb_digests[path] for path in sft_image_paths if path in rgb_digests}
    input_targets: dict[str, set[str]] = defaultdict(set)
    family_ids_by_consistency: dict[str, set[str]] = defaultdict(set)
    for row in benchmark:
        input_key = json.dumps(row["model_input"], ensure_ascii=False, sort_keys=True)
        target_key = json.dumps(
            {
                "status": row["target"]["answer_status"],
                "value": row["target"]["answer_value"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        input_targets[input_key].add(target_key)
        if row.get("consistency_group"):
            family_ids_by_consistency[row["consistency_group"]].add(row["family_id"])

    expected_family_variants = {
        "claim": {"claim_false", "claim_true"},
        "evidence": {"prefix_unknown", "revealed", "decisive_deleted"},
        "frame": {"frame_a", "frame_b"},
    }
    families_complete = True
    for variants in consistency_to_variants.values():
        if variants & expected_family_variants["claim"]:
            families_complete &= variants == expected_family_variants["claim"]
        elif variants & expected_family_variants["evidence"]:
            families_complete &= variants == expected_family_variants["evidence"]
        elif variants & expected_family_variants["frame"]:
            families_complete &= variants == expected_family_variants["frame"]
        else:
            families_complete = False
    family_interventions_matched = True
    for specs in consistency_to_specs.values():
        variants = {spec.family_variant for spec in specs}
        if variants == expected_family_variants["evidence"]:
            view_lists = [spec.model_view_ids or spec.evidence_view_ids for spec in specs]
            image_counts = {len(view_ids) for view_ids in view_lists}
            shared = set.intersection(*(set(view_ids) for view_ids in view_lists))
            shared_slots_fixed = all(
                len({view_ids.index(view_id) for view_ids in view_lists}) == 1
                for view_id in shared
            )
            statuses = {spec.family_variant: spec.answer_status for spec in specs}
            family_interventions_matched &= (
                len(image_counts) == 1
                and len(shared) == next(iter(image_counts)) - 1
                and shared_slots_fixed
                and len({spec.question_zh for spec in specs}) == 1
                and statuses.get("revealed") == "accepted"
                and statuses.get("prefix_unknown") == "unknown"
                and statuses.get("decisive_deleted") == "unknown"
            )
        elif variants == expected_family_variants["frame"]:
            family_interventions_matched &= (
                len({spec.model_view_ids for spec in specs}) == 1
                and len({spec.answer_value for spec in specs}) == 2
            )
        elif variants == expected_family_variants["claim"]:
            structured = [spec.answer_value for spec in specs]
            family_interventions_matched &= (
                len({spec.model_view_ids for spec in specs}) == 1
                and {value.get("claim_correct") for value in structured} == {False, True}
                and len({value.get("relation") for value in structured}) == 1
            )
    replay_results = [
        replay_question(item.bundle, question) for item in compiled for question in item.questions
    ]
    shape_counts = Counter(
        "+".join(sorted(variants)) for variants in consistency_to_variants.values()
    )
    required_shape_counts = {
        str(shape): int(count)
        for shape, count in requirements.get("minimum_family_groups_by_shape", {}).items()
    }
    benchmark_program_counts_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    benchmark_variants_by_split_group: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in benchmark:
        row_split = str(row["split"])
        benchmark_program_counts_by_split[row_split][
            str(row["program"]["program_id"])
        ] += 1
        if row.get("consistency_group"):
            benchmark_variants_by_split_group[
                (row_split, str(row["consistency_group"]))
            ].add(str(row["family_variant"]))
    required_program_counts_by_split = {
        str(split): {str(program): int(count) for program, count in programs.items()}
        for split, programs in requirements.get(
            "minimum_benchmark_program_records_by_split", {}
        ).items()
    }
    benchmark_shape_counts_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for (split, _), variants in benchmark_variants_by_split_group.items():
        benchmark_shape_counts_by_split[split]["+".join(sorted(variants))] += 1
    required_family_counts_by_split = {
        str(split): {str(shape): int(count) for shape, count in shapes.items()}
        for split, shapes in requirements.get(
            "minimum_benchmark_family_groups_by_split", {}
        ).items()
    }
    compiled_task_counts = Counter(
        question.task_type for item in compiled for question in item.questions
    )
    required_task_counts = {
        str(task_type): int(count)
        for task_type, count in requirements.get(
            "minimum_compiled_task_type_records", {}
        ).items()
    }
    state_witness_contract = all(
        belief["entity_count"] == len(belief["entities"])
        and all(entity["evidence_views"] for entity in belief["entities"])
        for belief in observable_beliefs.values()
    )
    checks = {
        "typed_programs_valid": all(_programs_validate(item.questions) for item in compiled),
        "heldout_program_absent_from_train": not (train_programs & heldout_program_ids),
        "heldout_atoms_seen_in_train": heldout_atoms.issubset(train_atoms),
        "scene_split_lock": all(len(values) == 1 for values in scene_to_splits.values()),
        "family_split_lock": all(len(values) == 1 for values in consistency_to_splits.values()),
        "family_members_complete": families_complete,
        "family_intervention_contract": family_interventions_matched,
        "family_id_shared_within_consistency_group": all(
            len(values) == 1 for values in family_ids_by_consistency.values()
        ),
        "paired_fact_multiset_equal": Counter(episode_fact_ids) == Counter(isolated_fact_ids),
        "fact_ids_globally_unique": len(all_fact_ids) == len(set(all_fact_ids)),
        "record_ids_unique": len(
            {row["record_id"] for row in episode_sft + isolated_sft + benchmark}
        )
        == len(episode_sft) + len(isolated_sft) + len(benchmark),
        "heldout_target_rgb_absent_from_sft_input": not any(
            path and path != "None" and path in serialized_inputs for path in target_paths
        ),
        "heldout_target_rgb_hash_absent_from_sft_input": not (target_hashes & sft_hashes),
        "rgb_channel_purity": all(rgb_digests.values()),
        "all_exported_model_frames_admissible": not (
            inadmissible_exported_model_frames
        ),
        "normalized_input_functional_dependency": all(
            len(targets) == 1 for targets in input_targets.values()
        ),
        "all_certificates_terminal": all(
            question.certificate.get("result") in {"pass", "unknown"}
            for item in compiled
            for question in item.questions
        ),
        "all_certificate_checks_pass": all(
            check.get("passed") is True
            for item in compiled
            for question in item.questions
            for check in question.certificate.get("checks", ())
        ),
        "certificate_full_reexecution": all(passed for passed, _ in replay_results),
        "all_compiled_episodes_nonempty": all(item.questions for item in compiled),
        "required_family_shapes_present": all(
            shape_counts[shape] >= minimum
            for shape, minimum in required_shape_counts.items()
        ),
        "required_benchmark_programs_present_per_split": all(
            benchmark_program_counts_by_split[split][program] >= minimum
            for split, programs in required_program_counts_by_split.items()
            for program, minimum in programs.items()
        ),
        "required_benchmark_family_shapes_present_per_split": all(
            benchmark_shape_counts_by_split[split][shape] >= minimum
            for split, shapes in required_family_counts_by_split.items()
            for shape, minimum in shapes.items()
        ),
        "required_compiled_task_types_present": all(
            compiled_task_counts[task_type] >= minimum
            for task_type, minimum in required_task_counts.items()
        ),
        "state_entities_have_quality_gated_witnesses": state_witness_contract,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "train_program_ids": sorted(train_programs),
        "train_atoms": sorted(train_atoms),
        "heldout_program_ids": sorted(heldout_program_ids),
        "heldout_atoms": sorted(heldout_atoms),
        "heldout_benchmark_records": len(heldout_specs),
        "certificate_replay": {
            "checked": len(replay_results),
            "passed": sum(passed for passed, _ in replay_results),
            "failures": [detail for passed, detail in replay_results if not passed][:20],
        },
        "exported_model_frame_admission": {
            "checked_unique_paths": len(all_image_paths),
            "inadmissible_count": len(inadmissible_exported_model_frames),
            "inadmissible_path_samples": inadmissible_exported_model_frames[:20],
            "policy_id": visual_quality_policy()["policy_id"],
        },
        "coverage_contract": {
            "requirements": requirements,
            "observed_family_groups_by_shape": dict(sorted(shape_counts.items())),
            "observed_benchmark_program_records_by_split": {
                split: dict(sorted(counts.items()))
                for split, counts in sorted(benchmark_program_counts_by_split.items())
            },
            "observed_benchmark_family_groups_by_split": {
                split: dict(sorted(counts.items()))
                for split, counts in sorted(benchmark_shape_counts_by_split.items())
            },
            "observed_compiled_task_type_records": dict(sorted(compiled_task_counts.items())),
        },
    }


def _message_image_paths(messages: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        paths.extend(
            str(item["image"])
            for item in content
            if item.get("type") == "image" and item.get("image")
        )
    return paths


def _verified_raw_rgb_digest(path: Path, media_root: Path) -> str:
    try:
        resolved = path.resolve()
        resolved.relative_to(media_root.resolve())
        if "preview" in resolved.parts or resolved.suffix.lower() != ".png":
            return ""
        metadata = read_json(resolved.with_suffix(".rgb.json"))
        if metadata.get("schema_version") != "epispace.raw_rgb.v1":
            return ""
        with Image.open(resolved) as image:
            if (
                image.mode != "RGB"
                or image.width != metadata.get("width")
                or image.height != metadata.get("height")
            ):
                return ""
            digest = hashlib.sha256(image.tobytes()).hexdigest()
        return digest if digest == metadata.get("rgb_sha256") else ""
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return ""


def _programs_validate(questions: list[QuestionSpec]) -> bool:
    for question in questions:
        question.program.validate()
    return True


def _statistics(
    *,
    acquisition: dict[str, Any],
    compiled: list[CompiledBundle],
    rejections: list[Rejection],
    episode_sft: list[dict[str, Any]],
    isolated_sft: list[dict[str, Any]],
    state_aux: list[dict[str, Any]],
    benchmark: list[dict[str, Any]],
    benchmark_family: list[dict[str, Any]],
    benchmark_composition: list[dict[str, Any]],
    benchmark_core: list[dict[str, Any]],
    rlvr: list[dict[str, Any]],
) -> dict[str, Any]:
    trajectory_counts = Counter(item.bundle.trajectory_class for item in compiled)
    split_scenes: dict[str, set[str]] = defaultdict(set)
    split_bundles = Counter()
    task_types = Counter()
    programs = Counter()
    family_variants = Counter()
    consistency_variants: dict[str, set[str]] = defaultdict(set)
    views = 0
    for item in compiled:
        split_scenes[item.split].add(item.bundle.scene_id)
        split_bundles[item.split] += 1
        views += len(item.bundle.views)
        for question in item.questions:
            task_types[question.task_type] += 1
            programs[question.program.program_id] += 1
            family_variants[question.family_variant] += 1
            if question.consistency_group:
                consistency_variants[question.consistency_group].add(question.family_variant)
    family_shapes = Counter(tuple(sorted(variants)) for variants in consistency_variants.values())
    return {
        "acquisition": acquisition,
        "compiled": {
            "unique_scenes": len({item.bundle.scene_id for item in compiled}),
            "source_trajectory_bundles": len(compiled),
            "source_views": views,
            "trajectory_counts": dict(sorted(trajectory_counts.items())),
            "questions": sum(task_types.values()),
            "task_types": dict(sorted(task_types.items())),
            "programs": dict(sorted(programs.items())),
            "family_variants": dict(sorted(family_variants.items())),
            "consistency_families": len(consistency_variants),
            "family_shapes": {
                "+".join(shape): count for shape, count in sorted(family_shapes.items())
            },
            "split_scenes": {key: len(value) for key, value in sorted(split_scenes.items())},
            "split_bundles": dict(sorted(split_bundles.items())),
        },
        "exports": {
            "episode_sft_records": len(episode_sft),
            "isolated_sft_records": len(isolated_sft),
            "state_aux_records": len(state_aux),
            "benchmark_records": len(benchmark),
            "benchmark_family_records": len(benchmark_family),
            "benchmark_family_groups": len(
                {row["consistency_group"] for row in benchmark_family}
            ),
            "benchmark_composition_records": len(benchmark_composition),
            "benchmark_core_records": len(benchmark_core),
            "rlvr_records": len(rlvr),
            "episode_supervised_facts": sum(
                len(row["comparison_contract"]["fact_ids"]) for row in episode_sft
            ),
        },
        "rejections": {
            "count": len(rejections),
            "reason_counts": dict(sorted(Counter(item.reason_code for item in rejections).items())),
        },
    }


def _latex_stats(stats: dict[str, Any], gates: dict[str, Any]) -> str:
    acquisition = stats["acquisition"]
    compiled = stats["compiled"]
    exports = stats["exports"]
    family_shapes = compiled["family_shapes"]
    lines = [
        "% Generated by episode3d.pipeline; do not edit by hand.",
        rf"\newcommand{{\EpiPlannedJobs}}{{{acquisition['planned_jobs']}}}",
        rf"\newcommand{{\EpiStrictBundles}}{{{acquisition['strict_bundles_loaded']}}}",
        rf"\newcommand{{\EpiScenes}}{{{compiled['unique_scenes']}}}",
        rf"\newcommand{{\EpiSourceTrajectories}}{{{compiled['source_trajectory_bundles']}}}",
        rf"\newcommand{{\EpiSourceViews}}{{{compiled['source_views']}}}",
        rf"\newcommand{{\EpiCompiledQuestions}}{{{compiled['questions']}}}",
        rf"\newcommand{{\EpiEpisodeRecords}}{{{exports['episode_sft_records']}}}",
        rf"\newcommand{{\EpiIsolatedRecords}}{{{exports['isolated_sft_records']}}}",
        rf"\newcommand{{\EpiBenchmarkRecords}}{{{exports['benchmark_records']}}}",
        rf"\newcommand{{\EpiCoreBenchmarkRecords}}{{{exports['benchmark_core_records']}}}",
        rf"\newcommand{{\EpiFamilyBenchmarkRecords}}{{{exports['benchmark_family_records']}}}",
        rf"\newcommand{{\EpiFamilyBenchmarkGroups}}{{{exports['benchmark_family_groups']}}}",
        rf"\newcommand{{\EpiConsistencyFamilies}}{{{compiled['consistency_families']}}}",
        rf"\newcommand{{\EpiEvidenceFamilies}}{{{family_shapes.get('decisive_deleted+prefix_unknown+revealed', 0)}}}",
        rf"\newcommand{{\EpiClaimFamilies}}{{{family_shapes.get('claim_false+claim_true', 0)}}}",
        rf"\newcommand{{\EpiFrameFamilies}}{{{family_shapes.get('frame_a+frame_b', 0)}}}",
        rf"\newcommand{{\EpiSupervisedFacts}}{{{exports['episode_supervised_facts']}}}",
        rf"\newcommand{{\EpiStateRecords}}{{{exports['state_aux_records']}}}",
        rf"\newcommand{{\EpiRLVRRecords}}{{{exports['rlvr_records']}}}",
        rf"\newcommand{{\EpiCompositionRecords}}{{{exports['benchmark_composition_records']}}}",
        rf"\newcommand{{\EpiRejectedItems}}{{{stats['rejections']['count']}}}",
        rf"\newcommand{{\EpiReplayedCertificates}}{{{gates['certificate_replay']['checked']}}}",
        rf"\newcommand{{\EpiHeldoutRecords}}{{{gates['heldout_benchmark_records']}}}",
        rf"\newcommand{{\EpiCorpusGate}}{{{gates['status']}}}",
        "",
    ]
    return "\n".join(lines)


def _latex_table(stats: dict[str, Any]) -> str:
    rows = []
    acquisition_by_class: dict[str, dict[str, int]] = defaultdict(
        lambda: {"planned": 0, "strict": 0}
    )
    for sweep in stats["acquisition"]["sweeps"]:
        item = acquisition_by_class[sweep["trajectory_class"]]
        item["planned"] += int(sweep["planned_jobs"])
        item["strict"] += int(sweep["strict_bundles_loaded"])
    compiled_counts = stats["compiled"]["trajectory_counts"]
    for trajectory_class in ("T1", "T3", "T4", "T7", "T8", "T10"):
        item = acquisition_by_class[trajectory_class]
        role = "verifier only" if trajectory_class == "T10" else "write source"
        rows.append(
            f"{trajectory_class} & {item['planned']} & {item['strict']} & "
            f"{compiled_counts.get(trajectory_class, 0)} & {role} \\\\"
        )
    return "\n".join(
        [
            "% Generated by episode3d.pipeline; do not edit by hand.",
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{Pilot trajectory assets after strict typed quality gating. "
            r"T10 RGB is held out and used only as a perspective verifier.}",
            r"\label{tab:pilot-assets}",
            r"\begin{tabular}{lrrrl}",
            r"\toprule",
            r"Class & Planned & Strict & Write bundles & Role \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def build_dataset(
    config_path: Path,
    *,
    output_dir: Path | None = None,
    latex_dir: Path | None = None,
) -> dict[str, Any]:
    """Build and verify a complete pilot release from sweep plans."""

    config_path = config_path.resolve()
    config = read_json(config_path)
    if config.get("schema_version") != "epispace.pipeline_config.v1":
        raise ValueError("unsupported pipeline config schema")
    if output_dir is None:
        output_dir = _resolve(config_path, config["output_dir"])
    else:
        output_dir = output_dir.resolve()
    if latex_dir is None:
        latex_dir = _resolve(config_path, config["latex_output_dir"])
    else:
        latex_dir = latex_dir.resolve()

    # A failed rebuild must never leave a stale pass manifest/final index next
    # to partially replaced artifacts.  Invalidate release markers before any
    # source access or cache materialization can raise.
    output_dir.mkdir(parents=True, exist_ok=True)
    _invalidate_derived_outputs(output_dir)
    (output_dir / "release_manifest.json").unlink(missing_ok=True)

    media_root = _resolve(config_path, config["model_rgb_dir"])
    bundles, rejections, acquisition = _load_bundles(config, config_path, media_root)
    scene_ids = {bundle.scene_id for bundle in bundles}
    split_config = config["splits"]
    splits = _scene_splits(
        scene_ids,
        seed=str(split_config["seed"]),
        requested_counts={key: int(split_config[key]) for key in ("train", "val", "test")},
    )
    compiled = _compile(bundles, splits, rejections)
    heldout_program_ids = set(config["composition_holdout"]["program_ids"])
    max_questions = int(config["export"]["max_questions_per_episode"])

    observable_beliefs = {
        item.bundle.episode_id: item.bundle.canonical_belief(
            evidence_view_ids_by_entity=recognizable_witnesses(item.bundle)
        )
        for item in compiled
    }
    ir_rows = [
        _episode_ir(item, observable_beliefs[item.bundle.episode_id]) for item in compiled
    ]
    episode_sft: list[dict[str, Any]] = []
    isolated_sft: list[dict[str, Any]] = []
    state_aux: list[dict[str, Any]] = []
    benchmark: list[dict[str, Any]] = []
    rlvr: list[dict[str, Any]] = []
    selected_train: list[QuestionSpec] = []
    for item in compiled:
        episode_rows, isolated_rows, selected = export_training_arms(
            item,
            heldout_program_ids=heldout_program_ids,
            max_questions=max_questions,
        )
        episode_sft.extend(episode_rows)
        isolated_sft.extend(isolated_rows)
        selected_train.extend(selected)
        auxiliary = export_state_aux(
            item,
            observable_belief=observable_beliefs[item.bundle.episode_id],
        )
        if auxiliary:
            state_aux.append(auxiliary)
        benchmark.extend(export_benchmark(item))
        rlvr.extend(export_rlvr(item, selected))

    benchmark_family = [row for row in benchmark if row.get("consistency_group")]
    benchmark_composition = [
        row for row in benchmark if row["program"]["program_id"] in heldout_program_ids
    ]
    core_by_id = {
        row["record_id"]: row for row in (*benchmark_family, *benchmark_composition)
    }
    benchmark_core = [core_by_id[record_id] for record_id in sorted(core_by_id)]

    gates = _corpus_gates(
        compiled=compiled,
        selected_train=selected_train,
        benchmark=benchmark,
        episode_sft=episode_sft,
        isolated_sft=isolated_sft,
        state_aux=state_aux,
        rlvr=rlvr,
        heldout_program_ids=heldout_program_ids,
        media_root=media_root,
        observable_beliefs=observable_beliefs,
        requirements=dict(config.get("corpus_requirements", {})),
    )
    stats = _statistics(
        acquisition=acquisition,
        compiled=compiled,
        rejections=rejections,
        episode_sft=episode_sft,
        isolated_sft=isolated_sft,
        state_aux=state_aux,
        benchmark=benchmark,
        benchmark_family=benchmark_family,
        benchmark_composition=benchmark_composition,
        benchmark_core=benchmark_core,
        rlvr=rlvr,
    )
    source_inventory = _source_inventory(
        config_path=config_path,
        bundles=bundles,
        acquisition=acquisition,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": config["dataset_id"],
        "created_at": datetime.now(UTC).isoformat(),
        "config": str(config_path),
        "status": gates["status"],
        "implementation_integrity": build_implementation_binding(),
        "research_contract": {
            "main_interface": "RGB + natural-language questions and answers",
            "program_visibility": "hidden compiler IR",
            "state_visibility": "auxiliary only when explicitly requested",
            "quality_policy": "strict plan status + bundle typed gate",
            "visual_quality_policy": visual_quality_policy(),
            "split_unit": "scene",
            "language_policy": "geometry first; deterministic answers; answer-blind question realization",
        },
        "statistics": stats,
        "corpus_gates": gates,
        "artifacts": {
            "episode_ir": "episodes.ir.jsonl",
            "episode_sft": "train.episode_sft.jsonl",
            "isolated_sft": "train.isolated_sft.jsonl",
            "state_aux_sft": "train.state_aux_sft.jsonl",
            "rlvr": "train.rlvr.jsonl",
            "benchmark": "benchmark.jsonl",
            "benchmark_family": "benchmark.family.jsonl",
            "benchmark_composition": "benchmark.composition.jsonl",
            "benchmark_core": "benchmark.core.jsonl",
            "rejections": "rejections.jsonl",
            "source_inventory": "source_inventory.json",
        },
        "license_boundary": (
            "BEHAVIOR-1K-derived render media remain subject to the source academic-use "
            "terms; this internal manifest does not grant redistribution rights."
        ),
    }

    _write_jsonl(output_dir / "episodes.ir.jsonl", ir_rows)
    _write_jsonl(output_dir / "train.episode_sft.jsonl", episode_sft)
    _write_jsonl(output_dir / "train.isolated_sft.jsonl", isolated_sft)
    _write_jsonl(output_dir / "train.state_aux_sft.jsonl", state_aux)
    _write_jsonl(output_dir / "train.rlvr.jsonl", rlvr)
    _write_jsonl(output_dir / "benchmark.jsonl", benchmark)
    _write_jsonl(output_dir / "benchmark.family.jsonl", benchmark_family)
    _write_jsonl(output_dir / "benchmark.composition.jsonl", benchmark_composition)
    _write_jsonl(output_dir / "benchmark.core.jsonl", benchmark_core)
    _write_jsonl(output_dir / "rejections.jsonl", [item.as_dict() for item in rejections])
    _write_json(output_dir / "source_inventory.json", source_inventory)
    artifact_rows = {
        "episode_ir": len(ir_rows),
        "episode_sft": len(episode_sft),
        "isolated_sft": len(isolated_sft),
        "state_aux_sft": len(state_aux),
        "rlvr": len(rlvr),
        "benchmark": len(benchmark),
        "benchmark_family": len(benchmark_family),
        "benchmark_composition": len(benchmark_composition),
        "benchmark_core": len(benchmark_core),
        "rejections": len(rejections),
        "source_inventory": None,
    }
    manifest["artifact_integrity"] = {
        name: _artifact_record(output_dir / relative_path, records=artifact_rows[name])
        for name, relative_path in manifest["artifacts"].items()
    }
    manifest["source_integrity"] = {
        "config_sha256": source_inventory["config"]["sha256"],
        "source_inventory_sha256": manifest["artifact_integrity"]["source_inventory"][
            "sha256"
        ],
        "sweep_plan_sha256": {
            item["name"]: item["sweep_plan_sha256"] for item in acquisition["sweeps"]
        },
    }
    manifest["derived_artifacts"] = {
        "status": "invalidated_on_dataset_build",
        "invalidation_policy": list(_DERIVED_OUTPUTS),
        "required_rebuild_order": [
            "semantic_visual_audit",
            "compute_matching",
            "corpus_audit",
            "evaluation_smoke",
            "final_release_index",
        ],
    }
    _write_json(output_dir / "release_manifest.json", manifest)
    _write_text_atomic(latex_dir / "dataset_stats.tex", _latex_stats(stats, gates))
    _write_text_atomic(latex_dir / "dataset_table.tex", _latex_table(stats))
    if gates["status"] != "pass":
        raise RuntimeError(f"corpus gates failed; inspect {output_dir / 'release_manifest.json'}")
    return manifest
