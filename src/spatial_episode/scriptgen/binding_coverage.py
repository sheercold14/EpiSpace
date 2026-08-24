"""Binding-coverage planning, acquisition, shared credit, and packaging."""

from __future__ import annotations

import hashlib
import json
import math
import queue
import re
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field

from .behavior import RenderSceneView, layout_from_scene_ir
from .collection import (
    CHAIN_MOTIFS,
    CollectionScene,
    REFERENCE_CAPABILITIES,
    _binding_chain_source_covisible,
    _reference_binding_eligible,
    _scene_can_bind,
    _source_render_evidence,
    _visit_render_robust_filter,
    ranked_bindings,
    render_plan_payload,
)
from .compiler import CapabilityCompiler
from .family import FamilyBlocked, ScriptgenQuestionGroupV1, build_question_group
from .generate import generate_plans
from .library import SCRIPT_LIBRARY
from .motifs import proposal_occupancy_grid
from .plan import TrajectoryPlan
from .single import _preflight, _render, _render_failure_is_retryable, _write_recipe
from .slotting import iter_bindings
from .source_inventory import SourceIndex, SourceSceneRecord, load_source_index
from .spec import SpecModel
from .standards import STD_V1, CompileStandard
from .variants import FamilyMismatch

COVERAGE_SCHEMA_VERSION = "scriptgen_binding_coverage.v1"
COVERAGE_STATUS_SCHEMA_VERSION = "scriptgen_binding_coverage_status.v1"
COVERAGE_DATASET_SCHEMA_VERSION = "scriptgen_binding_coverage_dataset.v1"
RENDER_UNRESOLVABLE_CONFIRMATIONS = 3

# These cells can be credited by an earlier producer in the same question
# group.  Their own geometry search is deferred until authoritative rendering
# shows that shared episodes did not fill the quota.
DEFERRED_INITIAL_CAPABILITIES = frozenset(
    {
        "path_integration",
        "path_integration_magnitude",
        "homing_probe",
        "view_side_check",
        "existence_sufficiency_bed",
        "occluder_identification",
        "disappearance_cause",
        *(
            capability
            for capability in REFERENCE_CAPABILITIES
            if capability != "reference_frame_transform"
        ),
        # Each cross-view trio (ego/anchor/closer, walking and snapshot) is
        # compiled on one trajectory; the ego spec is the producer, the other
        # two are credited through the shared question group.
        *(
            capability
            for capability in SCRIPT_LIBRARY
            if capability.startswith("cross_view_") and "_ego_" not in capability
        ),
    }
)


class CoverageCandidate(SpecModel):
    candidate_id: str
    plan_id: str
    attempt_index: int = Field(ge=0)
    plan_record: str
    render_plan: str
    recipe: str
    bundle: str
    group: str
    log: str


class CoverageCell(SpecModel):
    cell_id: str
    scene_key: str
    scene_id: str
    capability: str
    binding: dict[str, str]
    seed: int
    # Optional geometry / render answer constraint used by label-aware repair
    # overlays.  Separate manifests are used for separate labels so the
    # ordinary capability+binding coverage key remains unambiguous.
    desired_answer_label: str | None = None
    # Optional repair-round suffix for candidate IDs.  A new deterministic
    # seed can then reuse attempt indices 0..N without colliding with the
    # immutable candidates retained from an earlier collection version.
    candidate_namespace: str = ""
    target_accepted: int = Field(ge=1)
    geometry_pool_size: int = Field(ge=0)
    diverse_pool_size: int = Field(ge=0)
    # Legacy manifests searched the complete 150-attempt / 150-plan pool.
    # New cells set these fields explicitly to their smaller initial frontier.
    search_attempt_limit: int = Field(default=150, ge=0)
    raw_plan_limit: int = Field(default=150, ge=0)
    search_pool_exhausted: bool = False
    rejection_counts: dict[str, int] = Field(default_factory=dict)
    candidates: tuple[CoverageCandidate, ...] = ()


class CoverageManifest(SpecModel):
    schema_version: Literal["scriptgen_binding_coverage.v1"] = COVERAGE_SCHEMA_VERSION
    collection_id: str
    standard_version: str
    source_index: str
    source_index_sha256: str
    output_root: str
    accepted_per_binding: int = Field(ge=1)
    attempts_per_binding: int = Field(ge=1)
    initial_attempts_per_binding: int = Field(default=30, ge=1)
    maximum_multislot_bindings: int = Field(ge=1)
    limit_bindings_per_capability: int | None = Field(default=None, ge=1)
    capabilities: tuple[str, ...]
    desired_answer_label: str | None = None
    scenes: tuple[CollectionScene, ...]
    scene_skips: dict[str, str] = Field(default_factory=dict)
    cells: tuple[CoverageCell, ...]


@dataclass(frozen=True)
class CandidateSearchResult:
    candidates: tuple[CoverageCandidate, ...]
    geometry_pool_size: int
    diverse_pool_size: int
    rejection_counts: dict[str, int]
    search_attempt_limit: int
    raw_plan_limit: int
    search_pool_exhausted: bool


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hex(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def _canonical_binding(binding: dict[str, str]) -> str:
    return json.dumps(binding, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _derived_seed(collection_id: str, scene_id: str, capability: str, binding: dict[str, str]) -> int:
    digest = _stable_hex(collection_id, scene_id, capability, _canonical_binding(binding))
    return int(digest[:8], 16) & 0x7FFFFFFF


def _source_scene(record: SourceSceneRecord) -> CollectionScene:
    assert record.scene_ir is not None
    assert record.scene_snapshot is not None
    assert record.scene_ir_sha256 is not None
    assert record.recipe_sha256 is not None
    snapshot = json.loads(Path(record.scene_snapshot).read_text(encoding="utf-8"))
    scene_ir = json.loads(Path(record.scene_ir).read_text(encoding="utf-8"))
    if _sha256(Path(record.scene_ir)) != record.scene_ir_sha256:
        raise ValueError(f"source scene_ir changed: {record.scene_key}")
    if _sha256(Path(record.recipe)) != record.recipe_sha256:
        raise ValueError(f"source recipe changed: {record.scene_key}")
    if snapshot.get("source_digest") != record.source_digest:
        raise ValueError(f"source snapshot changed: {record.scene_key}")
    return CollectionScene(
        scene_key=record.scene_key,
        scene_id=str(scene_ir["scene_id"]),
        source_scene_id=str(snapshot["source_scene_id"]),
        scene_model=str(snapshot["scene_model"]),
        scene_instance=snapshot.get("scene_instance"),
        source_digest=str(snapshot["source_digest"]),
        scene_ir=record.scene_ir,
        source_recipe=record.recipe,
        scene_ir_sha256=record.scene_ir_sha256,
        source_recipe_sha256=record.recipe_sha256,
    )


def _bindings_for_scene(
    scene: CollectionScene,
    capability: str,
    *,
    maximum_multislot_bindings: int,
    std: CompileStandard,
) -> tuple[dict[str, str], ...]:
    script = SCRIPT_LIBRARY[capability]
    layout = layout_from_scene_ir(scene.scene_ir, std=std)
    if not _scene_can_bind(layout, script):
        return ()
    if len(script.slots) == 1:
        bindings, _ = iter_bindings(layout, script)
        return tuple(bindings)
    if capability in REFERENCE_CAPABILITIES:
        return tuple(
            binding
            for binding in ranked_bindings(
                layout,
                script,
                maximum=maximum_multislot_bindings,
            )
            if _reference_binding_eligible(layout, binding, std)
        )
    if script.motifs in CHAIN_MOTIFS:
        evidence = _source_render_evidence(scene, std)
        return ranked_bindings(
            layout,
            script,
            maximum=maximum_multislot_bindings,
            allowed_entity_ids=evidence.visible_entities,
            binding_filter=lambda binding: _binding_chain_source_covisible(binding, evidence),
        )
    return ranked_bindings(layout, script, maximum=maximum_multislot_bindings)


def _binding_cache_key(capability: str) -> tuple[Any, ...]:
    """Group capabilities whose binding eligibility and ranking are identical."""
    script = SCRIPT_LIBRARY[capability]
    slots = tuple(
        (
            name,
            slot.min_size_m,
            tuple(slot.categories),
            slot.unique_referent,
        )
        for name, slot in script.slots.items()
    )
    if capability.startswith("existence_sufficiency_"):
        eligibility = capability
    elif capability in REFERENCE_CAPABILITIES:
        eligibility = "reference"
    elif script.motifs in CHAIN_MOTIFS:
        eligibility = "landmark_chain"
    else:
        eligibility = "generic"
    return eligibility, slots


def _resample_plan(plan: TrajectoryPlan, count: int = 20) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray([(pose.x, pose.y) for pose in plan.poses], dtype=np.float64)
    yaw = np.unwrap(np.radians([pose.yaw_deg for pose in plan.poses]))
    source = np.linspace(0.0, 1.0, len(plan.poses))
    target = np.linspace(0.0, 1.0, count)
    xy = np.column_stack(
        (
            np.interp(target, source, points[:, 0]),
            np.interp(target, source, points[:, 1]),
        )
    )
    return xy, np.interp(target, source, yaw)


def _blocked_by(plan: TrajectoryPlan) -> frozenset[str]:
    witness = plan.clause_witnesses.get("occluded_at_question", {})
    return frozenset(str(item) for item in witness.get("blocked_by", ()))


def trajectories_are_diverse(candidate: TrajectoryPlan, prior: TrajectoryPlan) -> bool:
    """Require at least one structural difference, not a tiny pose perturbation."""
    start_distance = math.dist(
        (candidate.poses[0].x, candidate.poses[0].y),
        (prior.poses[0].x, prior.poses[0].y),
    )
    end_distance = math.dist(
        (candidate.poses[-1].x, candidate.poses[-1].y),
        (prior.poses[-1].x, prior.poses[-1].y),
    )
    candidate_xy, candidate_yaw = _resample_plan(candidate)
    prior_xy, prior_yaw = _resample_plan(prior)
    mean_path_distance = float(np.linalg.norm(candidate_xy - prior_xy, axis=1).mean())
    yaw_delta = np.angle(np.exp(1j * (candidate_yaw - prior_yaw)))
    mean_yaw_difference = math.degrees(float(np.abs(yaw_delta).mean()))
    candidate_net_turn = math.degrees(float(candidate_yaw[-1] - candidate_yaw[0]))
    prior_net_turn = math.degrees(float(prior_yaw[-1] - prior_yaw[0]))
    key_frames = set(candidate.frame_vars) & set(prior.frame_vars) & {"t_seen", "t_gone", "t_q"}
    key_frame_difference = max(
        (abs(candidate.frame_vars[key] - prior.frame_vars[key]) for key in key_frames),
        default=0,
    )
    blocker_difference = bool(_blocked_by(candidate) or _blocked_by(prior)) and (
        _blocked_by(candidate) != _blocked_by(prior)
    )
    return bool(
        start_distance >= 0.5
        or end_distance >= 0.5
        or mean_path_distance >= 0.30
        or mean_yaw_difference >= 15.0
        or abs(candidate_net_turn - prior_net_turn) >= 30.0
        or key_frame_difference >= 2
        or blocker_difference
    )


def _diverse_sequence(plans: tuple[TrajectoryPlan, ...]) -> tuple[TrajectoryPlan, ...]:
    retained: list[TrajectoryPlan] = []
    for plan in plans:
        if all(trajectories_are_diverse(plan, prior) for prior in retained):
            retained.append(plan)
    return tuple(retained)


def _attempt_index(plan_id: str) -> int:
    match = re.search(r"\.a(\d+)$", plan_id)
    if match is None:
        raise ValueError(f"plan id has no attempt suffix: {plan_id}")
    return int(match.group(1))


def _candidate_from_plan(
    plan: TrajectoryPlan,
    cell: CoverageCell,
    scene: CollectionScene,
    layout: Any,
    manifest: CoverageManifest,
    std: CompileStandard,
) -> CoverageCandidate:
    output_root = Path(manifest.output_root)
    candidate_id = (
        f"{cell.cell_id}{cell.candidate_namespace}__a{_attempt_index(plan.plan_id):03d}"
    )
    plan_record = output_root / "plans" / f"{candidate_id}.record.json"
    render_plan = output_root / "plans" / f"{candidate_id}.views.json"
    recipe = output_root / "recipes" / f"{candidate_id}.yaml"
    bundle = output_root / "bundles" / candidate_id
    group = output_root / "groups" / candidate_id
    log = output_root / "logs" / f"{candidate_id}.log"
    _write_json(plan_record, plan)
    _write_json(render_plan, render_plan_payload(plan, layout, std))
    _write_recipe(Path(scene.source_recipe), render_plan, recipe, plan.seed)
    return CoverageCandidate(
        candidate_id=candidate_id,
        plan_id=plan.plan_id,
        attempt_index=_attempt_index(plan.plan_id),
        plan_record=str(plan_record),
        render_plan=str(render_plan),
        recipe=str(recipe),
        bundle=str(bundle),
        group=str(group),
        log=str(log),
    )


def _geometry_pool(
    cell: CoverageCell,
    scene: CollectionScene,
    *,
    attempts_per_binding: int,
    plans_per_binding: int,
    std: CompileStandard,
) -> tuple[tuple[TrajectoryPlan, ...], dict[str, int]]:
    script = SCRIPT_LIBRARY[cell.capability]
    layout = layout_from_scene_ir(scene.scene_ir, std=std)
    report = generate_plans(
        layout,
        script,
        std,
        seed=cell.seed,
        attempts_per_binding=attempts_per_binding,
        # A label-aware pass must inspect every geometry-valid proposal in the
        # attempt frontier.  Stopping after N plans of any label can otherwise
        # miss a desired side label even though later attempts contain it.
        plans_per_binding=(
            attempts_per_binding
            if cell.desired_answer_label is not None
            else plans_per_binding
        ),
        candidate_bindings=(cell.binding,),
        candidate_filter=(
            _visit_render_robust_filter if script.motifs in CHAIN_MOTIFS else None
        ),
    )
    if cell.desired_answer_label is None:
        return report.plans, report.rejection_counts
    plans = tuple(
        plan
        for plan in report.plans
        if plan.provisional_answer.label == cell.desired_answer_label
    )
    counts = Counter(report.rejection_counts)
    counts[f"answer_label:not_{cell.desired_answer_label}"] += len(report.plans) - len(plans)
    return plans, dict(counts)


def _new_candidates(
    cell: CoverageCell,
    scene: CollectionScene,
    manifest: CoverageManifest,
    *,
    requested: int,
    attempt_limit: int | None = None,
    additional_prior_plans: tuple[TrajectoryPlan, ...] = (),
    std: CompileStandard = STD_V1,
) -> CandidateSearchResult:
    attempt_limit = min(
        attempt_limit or manifest.attempts_per_binding,
        manifest.attempts_per_binding,
    )
    minimum_raw_plans = max(
        manifest.accepted_per_binding * 2,
        len(cell.candidates) + requested * 2,
    )
    raw_plan_limit = min(
        attempt_limit,
        max(minimum_raw_plans, cell.raw_plan_limit * 2),
    )
    pool, rejection_counts = _geometry_pool(
        cell,
        scene,
        attempts_per_binding=attempt_limit,
        plans_per_binding=max(1, raw_plan_limit),
        std=std,
    )
    diverse = _diverse_sequence(pool)
    known_ids = {candidate.plan_id for candidate in cell.candidates}
    prior_plans = [
        TrajectoryPlan.model_validate_json(Path(candidate.plan_record).read_text(encoding="utf-8"))
        for candidate in cell.candidates
    ]
    prior_plans.extend(additional_prior_plans)
    selected: list[TrajectoryPlan] = []
    duplicate_count = 0
    has_more_eligible = False
    for plan in diverse:
        if plan.plan_id in known_ids:
            continue
        if all(trajectories_are_diverse(plan, prior) for prior in (*prior_plans, *selected)):
            if len(selected) < requested:
                selected.append(plan)
            else:
                has_more_eligible = True
        else:
            duplicate_count += 1
    counts = Counter(rejection_counts)
    counts["diversity:near_duplicate"] += len(pool) - len(diverse) + duplicate_count
    layout = layout_from_scene_ir(scene.scene_ir, std=std)
    candidates = tuple(
        _candidate_from_plan(plan, cell, scene, layout, manifest, std) for plan in selected
    )
    generation_exhausted = bool(
        attempt_limit >= manifest.attempts_per_binding
        and (len(pool) < raw_plan_limit or raw_plan_limit >= attempt_limit)
    )
    return CandidateSearchResult(
        candidates=candidates,
        geometry_pool_size=len(pool),
        diverse_pool_size=len(diverse),
        rejection_counts=dict(counts),
        search_attempt_limit=attempt_limit,
        raw_plan_limit=raw_plan_limit,
        search_pool_exhausted=generation_exhausted and not has_more_eligible,
    )


def _manifest_path(output_root: Path) -> Path:
    return output_root / "coverage.plan.json"


def _scene_occupancy_proxy_rejection(
    scene: CollectionScene,
    *,
    std: CompileStandard,
) -> str | None:
    """Reject a scene whose conservative 2-D proxy has no navigable cell."""
    layout = layout_from_scene_ir(scene.scene_ir, std=std)
    grid = proposal_occupancy_grid(layout)
    if grid.free_cells:
        return None
    return (
        "invalid_occupancy_proxy:no_free_cells:"
        f"grid={grid.width}x{grid.height}:obstacles={len(layout.obstacles)}"
    )


def plan_coverage(
    *,
    source_index_path: Path,
    output_root: Path,
    collection_id: str,
    accepted_per_binding: int = 10,
    attempts_per_binding: int = 150,
    initial_attempts_per_binding: int | None = None,
    maximum_multislot_bindings: int = 128,
    limit_bindings_per_capability: int | None = None,
    capabilities: tuple[str, ...] | None = None,
    scene_keys: tuple[str, ...] | None = None,
    binding_allowlist: dict[str, list[dict[str, str]]] | None = None,
    desired_answer_label: str | None = None,
    std: CompileStandard = STD_V1,
    on_scene_completed: Callable[[str], None] | None = None,
    initialize_only: bool = False,
) -> Path:
    """Enumerate coverage cells and persist their first geometry candidate wave."""
    if accepted_per_binding <= 0 or attempts_per_binding <= 0:
        raise ValueError("coverage quotas must be positive")
    if initial_attempts_per_binding is None:
        initial_attempts_per_binding = min(30, attempts_per_binding)
    if not 1 <= initial_attempts_per_binding <= attempts_per_binding:
        raise ValueError(
            "initial_attempts_per_binding must be between one and the maximum attempt budget"
        )
    source_index_path = source_index_path.resolve()
    output_root = output_root.resolve()
    source_index = load_source_index(source_index_path)
    if source_index.ready_scene_count != source_index.requested_scene_count:
        raise ValueError("source inventory is incomplete")
    requested_capabilities = tuple(capabilities or SCRIPT_LIBRARY)
    unknown = sorted(set(requested_capabilities) - set(SCRIPT_LIBRARY))
    if unknown:
        raise ValueError(f"unknown capabilities: {unknown}")
    requested_scenes = set(scene_keys or (record.scene_key for record in source_index.scenes))
    scenes = tuple(
        _source_scene(record)
        for record in source_index.scenes
        if record.scene_key in requested_scenes and record.status == "ready"
    )
    missing_scenes = sorted(requested_scenes - {scene.scene_key for scene in scenes})
    if missing_scenes:
        raise ValueError(f"source scenes are unavailable: {missing_scenes}")
    manifest = CoverageManifest(
        collection_id=collection_id,
        standard_version=std.standard_version,
        source_index=str(source_index_path),
        source_index_sha256=_sha256(source_index_path),
        output_root=str(output_root),
        accepted_per_binding=accepted_per_binding,
        attempts_per_binding=attempts_per_binding,
        initial_attempts_per_binding=initial_attempts_per_binding,
        maximum_multislot_bindings=maximum_multislot_bindings,
        limit_bindings_per_capability=limit_bindings_per_capability,
        capabilities=requested_capabilities,
        desired_answer_label=desired_answer_label,
        scenes=scenes,
        cells=(),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    existing_path = _manifest_path(output_root)
    if existing_path.is_file():
        existing = load_coverage(existing_path)
        expected = (
            collection_id,
            std.standard_version,
            str(source_index_path),
            _sha256(source_index_path),
            accepted_per_binding,
            attempts_per_binding,
            initial_attempts_per_binding,
            maximum_multislot_bindings,
            limit_bindings_per_capability,
            requested_capabilities,
            desired_answer_label,
            tuple(scene.scene_key for scene in scenes),
        )
        observed = (
            existing.collection_id,
            existing.standard_version,
            existing.source_index,
            existing.source_index_sha256,
            existing.accepted_per_binding,
            existing.attempts_per_binding,
            existing.initial_attempts_per_binding,
            existing.maximum_multislot_bindings,
            existing.limit_bindings_per_capability,
            existing.capabilities,
            existing.desired_answer_label,
            tuple(scene.scene_key for scene in existing.scenes),
        )
        if observed != expected:
            raise ValueError(f"existing coverage settings differ: {existing_path}")
        manifest = existing
    if initialize_only:
        _write_json(_manifest_path(output_root), manifest)
        return _manifest_path(output_root)
    cells: list[CoverageCell] = list(manifest.cells)
    existing_cell_ids = {cell.cell_id for cell in cells}
    for scene in scenes:
        scene_rejection = _scene_occupancy_proxy_rejection(scene, std=std)
        if scene_rejection is not None:
            scene_skips = dict(manifest.scene_skips)
            scene_skips[scene.scene_key] = scene_rejection
            manifest = manifest.model_copy(update={"scene_skips": scene_skips})
            _write_json(_manifest_path(output_root), manifest)
            print(
                f"coverage scene_skipped={scene.scene_key} reason={scene_rejection}",
                flush=True,
            )
            if on_scene_completed is not None:
                on_scene_completed(scene.scene_key)
            continue
        binding_cache: dict[tuple[Any, ...], tuple[dict[str, str], ...]] = {}
        for capability in requested_capabilities:
            cache_key = _binding_cache_key(capability)
            bindings = binding_cache.get(cache_key)
            if bindings is None:
                bindings = _bindings_for_scene(
                    scene,
                    capability,
                    maximum_multislot_bindings=maximum_multislot_bindings,
                    std=std,
                )
                binding_cache[cache_key] = bindings
            if limit_bindings_per_capability is not None:
                bindings = bindings[:limit_bindings_per_capability]
            if binding_allowlist is not None:
                allowed = {
                    _canonical_binding(binding)
                    for binding in binding_allowlist.get(scene.scene_key, ())
                }
                bindings = tuple(
                    binding
                    for binding in bindings
                    if _canonical_binding(binding) in allowed
                )
            for binding in bindings:
                cell_id = (
                    f"{scene.scene_key}__{capability}__"
                    f"{_stable_hex(_canonical_binding(binding))[:12]}"
                )
                if cell_id in existing_cell_ids:
                    continue
                cell = CoverageCell(
                    cell_id=cell_id,
                    scene_key=scene.scene_key,
                    scene_id=scene.scene_id,
                    capability=capability,
                    binding=binding,
                    seed=_derived_seed(collection_id, scene.scene_id, capability, binding),
                    desired_answer_label=desired_answer_label,
                    target_accepted=accepted_per_binding,
                    geometry_pool_size=0,
                    diverse_pool_size=0,
                    search_attempt_limit=0,
                    raw_plan_limit=0,
                )
                provisional = manifest.model_copy(update={"cells": tuple((*cells, cell))})
                if capability in DEFERRED_INITIAL_CAPABILITIES:
                    search = CandidateSearchResult(
                        candidates=(),
                        geometry_pool_size=0,
                        diverse_pool_size=0,
                        rejection_counts={"search:deferred_for_shared_credit": 1},
                        search_attempt_limit=0,
                        raw_plan_limit=0,
                        search_pool_exhausted=False,
                    )
                else:
                    search = _new_candidates(
                        cell,
                        scene,
                        provisional,
                        requested=accepted_per_binding,
                        attempt_limit=initial_attempts_per_binding,
                        std=std,
                    )
                cell = cell.model_copy(
                    update={
                        "candidates": search.candidates,
                        "geometry_pool_size": search.geometry_pool_size,
                        "diverse_pool_size": search.diverse_pool_size,
                        "search_attempt_limit": search.search_attempt_limit,
                        "raw_plan_limit": search.raw_plan_limit,
                        "search_pool_exhausted": search.search_pool_exhausted,
                        "rejection_counts": search.rejection_counts,
                    }
                )
                cells.append(cell)
                existing_cell_ids.add(cell_id)
                manifest = manifest.model_copy(update={"cells": tuple(cells)})
                _write_json(_manifest_path(output_root), manifest)
                print(
                    f"coverage planned cells={len(cells)} scene={scene.scene_key} "
                    f"capability={capability} candidates={len(search.candidates)} "
                    f"attempt_limit={search.search_attempt_limit}"
                    + (
                        " search=deferred_shared_credit"
                        if capability in DEFERRED_INITIAL_CAPABILITIES
                        else ""
                    ),
                    flush=True,
                )
        _write_json(_manifest_path(output_root), manifest)
        if on_scene_completed is not None:
            on_scene_completed(scene.scene_key)
    _write_json(_manifest_path(output_root), manifest)
    return _manifest_path(output_root)


def load_coverage(path: Path) -> CoverageManifest:
    return CoverageManifest.model_validate_json(path.read_text(encoding="utf-8"))


def verify_coverage_sources(manifest: CoverageManifest) -> None:
    source_index = Path(manifest.source_index)
    if _sha256(source_index) != manifest.source_index_sha256:
        raise ValueError(f"source index changed: {source_index}")
    for scene in manifest.scenes:
        if _sha256(Path(scene.scene_ir)) != scene.scene_ir_sha256:
            raise ValueError(f"source scene_ir changed: {scene.scene_key}")
        if _sha256(Path(scene.source_recipe)) != scene.source_recipe_sha256:
            raise ValueError(f"source recipe changed: {scene.scene_key}")
        snapshot = json.loads(
            (Path(scene.scene_ir).parent / "scene_snapshot.json").read_text(encoding="utf-8")
        )
        if snapshot.get("source_digest") != scene.source_digest:
            raise ValueError(f"source snapshot changed: {scene.scene_key}")


def _bundle_complete(candidate: CoverageCandidate) -> bool:
    report_path = Path(candidate.bundle) / "render_report.json"
    if not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return report.get("status") == "success"


def _candidate_target_lacks_runtime_id(
    candidate: CoverageCandidate,
    scene: CollectionScene,
    target_entity_id: str,
) -> bool | None:
    """Whether one successful replay omitted the planned target from its id registry.

    A source bundle cannot prove this: its static camera sweep may simply never
    observe the object. A scripted candidate is stronger evidence because its
    geometry plan deliberately includes a target-visible phase. Multiple
    independent candidate replays are required before quarantining a binding.
    """
    if not _bundle_complete(candidate):
        return None
    snapshot_path = Path(candidate.bundle) / "scene_snapshot.json"
    if not snapshot_path.is_file():
        return None
    scene_ir = json.loads(Path(scene.scene_ir).read_text(encoding="utf-8"))
    source_entity_id = next(
        (
            str(entity.get("source_entity_id", entity["entity_id"]))
            for entity in scene_ir.get("entities", ())
            if str(entity["entity_id"]) == target_entity_id
        ),
        None,
    )
    if source_entity_id is None:
        return None
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    registry = snapshot.get("runtime_instance_registry")
    if not isinstance(registry, dict):
        return None
    return source_entity_id not in {str(value) for value in registry.values()}


def _confirmed_render_unresolvable_targets(
    cell: CoverageCell,
    scene: CollectionScene,
    row: dict[str, Any],
) -> tuple[str, ...]:
    """Return a target only after repeated authoritative replay omissions."""
    target = cell.binding.get("target")
    if target is None:
        return ()
    missing_count = 0
    for candidate in cell.candidates:
        candidate_row = row["candidate_statuses"].get(candidate.candidate_id, {})
        candidate_status = candidate_row.get("status")
        if candidate_status not in {"accepted", "redundant", "rejected"}:
            continue
        # Accepted and redundant candidates both passed the complete
        # authoritative compile.  They are therefore conclusive positive
        # evidence that the target produced a usable runtime instance mask.
        # Recording this fact in status also makes render-only shards
        # independent of large historical RGB / instance bundles.
        if candidate_status in {"accepted", "redundant"}:
            return ()
        lacks_runtime_id = _candidate_target_lacks_runtime_id(candidate, scene, target)
        # One positive replay is conclusive counter-evidence: the entity can
        # produce a runtime mask, even if other trajectories never saw it.
        if lacks_runtime_id is False:
            return ()
        if (
            candidate_status == "rejected"
            and lacks_runtime_id is True
            and "frame_var_unresolvable:t_seen" in str(candidate_row.get("reason"))
        ):
            missing_count += 1
    return (target,) if missing_count >= RENDER_UNRESOLVABLE_CONFIRMATIONS else ()


def _validate_rendered_candidate(
    candidate: CoverageCandidate,
    cell: CoverageCell,
    scene: CollectionScene,
    *,
    std: CompileStandard,
) -> tuple[ScriptgenQuestionGroupV1, TrajectoryPlan]:
    plan_path = Path(candidate.plan_record)
    plan = TrajectoryPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    view = RenderSceneView.from_bundle(candidate.bundle, std, scene_ir=scene.scene_ir)
    certificate = CapabilityCompiler(SCRIPT_LIBRARY[cell.capability], std).compile(
        view,
        dict(plan.binding),
        geometry_plan=plan.model_dump(mode="json"),
    )
    group_root = Path(candidate.group)
    _write_json(group_root / "primary.certificate.json", certificate)
    outcome = next(
        (item for item in certificate.clause_outcomes if item.predicate == "occluded_in_view"),
        None,
    )
    if outcome is not None:
        _write_json(
            group_root / "occlusion_attribution.json",
            {
                "schema_version": "scriptgen_occlusion_attribution.v1",
                "plan_id": plan.plan_id,
                "scene_id": plan.scene_id,
                "standard_version": std.standard_version,
                "status": "pass" if outcome.holds is True else "fail",
                "evidence": outcome.witness,
            },
        )
    if certificate.status != "answerable" or certificate.mismatch is not None:
        reason = certificate.reason or certificate.mismatch or "primary_not_answerable"
        raise FamilyBlocked(str(reason))
    if (
        cell.desired_answer_label is not None
        and certificate.answer is not None
        and certificate.answer.label != cell.desired_answer_label
    ):
        raise FamilyBlocked(
            "desired_answer_label_mismatch:"
            f"expected={cell.desired_answer_label}:actual={certificate.answer.label}"
        )
    try:
        group_path = build_question_group(
            Path(candidate.bundle),
            plan_path,
            Path(scene.scene_ir),
            group_root,
            std,
            seed=plan.seed,
        )
    except AssertionError as error:
        # Variant generation can legitimately have no valid permutation for
        # a short / degenerate clause sequence.  That makes this rendered
        # candidate unusable as a complete question group, but must not kill
        # the long-lived GPU worker thread.
        raise FamilyBlocked(f"question_variant_unavailable:{error}") from error
    group = ScriptgenQuestionGroupV1.model_validate_json(group_path.read_text(encoding="utf-8"))
    primary = next(item for item in group.questions if item.capability == cell.capability)
    if primary.label is None or primary.family is None:
        raise FamilyBlocked(f"primary question skipped: {primary.skip_reason}")
    if (
        cell.desired_answer_label is not None
        and primary.label != cell.desired_answer_label
    ):
        raise FamilyBlocked(
            "desired_group_label_mismatch:"
            f"expected={cell.desired_answer_label}:actual={primary.label}"
        )
    return group, plan


def _status_template(manifest: CoverageManifest) -> dict[str, Any]:
    return {
        "schema_version": COVERAGE_STATUS_SCHEMA_VERSION,
        "collection_id": manifest.collection_id,
        "cells": {
            cell.cell_id: {
                "status": "planned",
                "accepted_episode_ids": [],
                "candidate_statuses": {
                    candidate.candidate_id: {"status": "pending", "reason": None}
                    for candidate in cell.candidates
                },
            }
            for cell in manifest.cells
        },
        "episodes": {},
    }


def initialize_coverage_status(manifest_path: Path) -> Path:
    """Create an idle status snapshot without rendering or runtime preflight.

    Distributed render packages need a hash-bound base status even when a new
    geometry manifest has never been run locally.  Existing status is accepted
    only when it belongs to the same collection and indexes the exact cells and
    initial candidates declared by the manifest.
    """
    manifest_path = manifest_path.resolve()
    manifest = load_coverage(manifest_path)
    status_path = Path(manifest.output_root) / "coverage.status.json"
    expected = _status_template(manifest)
    if status_path.is_file():
        observed = json.loads(status_path.read_text(encoding="utf-8"))
        if observed.get("collection_id") != manifest.collection_id:
            raise ValueError(f"coverage status collection differs: {status_path}")
        if set(observed.get("cells", {})) != set(expected["cells"]):
            raise ValueError(f"coverage status cell index differs: {status_path}")
        for cell_id, expected_row in expected["cells"].items():
            observed_candidates = set(
                observed["cells"][cell_id].get("candidate_statuses", {})
            )
            expected_candidates = set(expected_row["candidate_statuses"])
            if observed_candidates != expected_candidates:
                raise ValueError(
                    f"coverage status candidate index differs: {status_path}/{cell_id}"
                )
        return status_path
    _write_json(status_path, expected)
    return status_path


def _refresh_status_for_manifest(
    status: dict[str, Any],
    manifest: CoverageManifest,
    *,
    recover_interrupted: bool,
) -> None:
    """Add new manifest candidates without disturbing active workers.

    ``running`` rows are stale only when a new render pass starts. Manifest
    backfill and render-unresolvable updates also call this helper while the
    current pass is live, so they must not reset candidates owned by another
    GPU worker.
    """
    for cell in manifest.cells:
        row = status["cells"].setdefault(
            cell.cell_id,
            {"status": "planned", "accepted_episode_ids": [], "candidate_statuses": {}},
        )
        if recover_interrupted and row["status"] == "running":
            row["status"] = "planned"
        for candidate in cell.candidates:
            candidate_row = row["candidate_statuses"].setdefault(
                candidate.candidate_id,
                {"status": "pending", "reason": None},
            )
            if not recover_interrupted:
                continue
            if candidate_row["status"] == "running":
                candidate_row.update(
                    status="pending",
                    reason="recovered_after_interruption",
                )
                candidate_row.pop("gpu_id", None)
            elif candidate_row["status"] == "rejected" and str(
                candidate_row.get("reason") or ""
            ).startswith("authority:KeyError:"):
                candidate_row.update(
                    status="pending",
                    reason="recovered_after_transient_authority_keyerror",
                )
                candidate_row.pop("gpu_id", None)


def _cell_credit_key(scene_key: str, capability: str, binding: dict[str, str]) -> str:
    return f"{scene_key}\0{capability}\0{_canonical_binding(binding)}"


def _candidate_lookup(manifest: CoverageManifest) -> dict[str, CoverageCandidate]:
    return {
        candidate.candidate_id: candidate
        for cell in manifest.cells
        for candidate in cell.candidates
    }


def compatible_credit_cell_ids(
    manifest: CoverageManifest,
    *,
    scene_key: str,
    plan: TrajectoryPlan,
    group: ScriptgenQuestionGroupV1,
) -> tuple[str, ...]:
    """Map every answerable question to its normalized capability binding cell."""
    cells = {
        _cell_credit_key(cell.scene_key, cell.capability, cell.binding): cell.cell_id
        for cell in manifest.cells
    }
    result: list[str] = []
    for question in group.questions:
        if question.label is None or question.family is None:
            continue
        script = SCRIPT_LIBRARY[question.capability]
        if any(slot not in plan.binding for slot in script.slots):
            continue
        binding = {slot: plan.binding[slot] for slot in script.slots}
        cell_id = cells.get(_cell_credit_key(scene_key, question.capability, binding))
        if cell_id is not None:
            result.append(cell_id)
    return tuple(dict.fromkeys(result))


def run_coverage(
    manifest_path: Path,
    *,
    og_root: Path,
    conda_env: str,
    data_root: Path,
    gpu_ids: tuple[int, ...],
    workers: int | None = None,
    timeout_minutes: int = 20,
    limit_cells: int | None = None,
    scene_keys: tuple[str, ...] | None = None,
    cell_ids: tuple[str, ...] | None = None,
    credit_cell_ids: tuple[str, ...] | None = None,
    allow_backfill: bool = True,
    skip_preflight: bool = False,
    std: CompileStandard = STD_V1,
) -> Path:
    """Render cells in parallel while keeping candidates within a cell serial."""
    if not gpu_ids:
        raise ValueError("at least one GPU id is required")
    manifest_path = manifest_path.resolve()
    manifest = load_coverage(manifest_path)
    verify_coverage_sources(manifest)
    if not skip_preflight:
        _preflight(og_root.resolve(), data_root.resolve(), conda_env)
    status_path = Path(manifest.output_root) / "coverage.status.json"
    status = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.is_file()
        else _status_template(manifest)
    )
    scenes = {scene.scene_key: scene for scene in manifest.scenes}
    cells_by_id = {cell.cell_id: cell for cell in manifest.cells}
    requested_credit_ids = set(credit_cell_ids or ())
    unknown_credit_ids = requested_credit_ids - set(cells_by_id)
    if unknown_credit_ids:
        raise ValueError(
            "unknown credit cell ids: " + ", ".join(sorted(unknown_credit_ids))
        )
    lock = threading.RLock()

    def missing_slots(cell_id: str) -> int:
        cell = cells_by_id[cell_id]
        return max(
            0,
            cell.target_accepted - len(status["cells"][cell_id]["accepted_episode_ids"]),
        )

    def potential_credit_ids(cell: CoverageCell) -> tuple[str, ...]:
        """Deficient repair targets that this producer binding may serve."""

        if not requested_credit_ids:
            return ()
        result = []
        for target_id in requested_credit_ids:
            target = cells_by_id[target_id]
            if target.scene_key != cell.scene_key or missing_slots(target_id) == 0:
                continue
            slots = SCRIPT_LIBRARY[target.capability].slots
            if all(
                name in cell.binding and cell.binding[name] == target.binding.get(name)
                for name in slots
            ):
                result.append(target_id)
        return tuple(result)

    def source_needs_work(cell: CoverageCell) -> bool:
        return missing_slots(cell.cell_id) > 0 or bool(potential_credit_ids(cell))

    _refresh_status_for_manifest(status, manifest, recover_interrupted=True)

    def save() -> None:
        _write_json(status_path, status)

    def credit_episode(
        source_candidate: CoverageCandidate,
        source_plan: TrajectoryPlan,
        group: ScriptgenQuestionGroupV1,
        source_cell: CoverageCell,
    ) -> tuple[str, ...]:
        credited: list[str] = []
        candidates = _candidate_lookup(manifest)
        for cell_id in compatible_credit_cell_ids(
            manifest,
            scene_key=source_cell.scene_key,
            plan=source_plan,
            group=group,
        ):
            row = status["cells"][cell_id]
            target_cell = next(cell for cell in manifest.cells if cell.cell_id == cell_id)
            if len(row["accepted_episode_ids"]) >= target_cell.target_accepted:
                continue
            prior_plans = []
            for episode_id in row["accepted_episode_ids"]:
                prior_candidate = candidates.get(episode_id)
                if prior_candidate is not None:
                    prior_plans.append(
                        TrajectoryPlan.model_validate_json(
                            Path(prior_candidate.plan_record).read_text(encoding="utf-8")
                        )
                    )
            if all(trajectories_are_diverse(source_plan, prior) for prior in prior_plans):
                row["accepted_episode_ids"].append(source_candidate.candidate_id)
                credited.append(cell_id)
                if len(row["accepted_episode_ids"]) >= target_cell.target_accepted:
                    row["status"] = "complete"
        return tuple(credited)

    work: queue.Queue[str] = queue.Queue()
    selected_cells = list(manifest.cells)
    if scene_keys is not None:
        selected_scene_keys = set(scene_keys)
        selected_cells = [
            cell for cell in selected_cells if cell.scene_key in selected_scene_keys
        ]
    if cell_ids is not None:
        selected_cell_ids = set(cell_ids)
        known_cell_ids = {cell.cell_id for cell in manifest.cells}
        unknown_cell_ids = selected_cell_ids - known_cell_ids
        if unknown_cell_ids:
            raise ValueError(
                "unknown coverage cell ids: " + ", ".join(sorted(unknown_cell_ids))
            )
        selected_cells = [
            cell for cell in selected_cells if cell.cell_id in selected_cell_ids
        ]
    if limit_cells is not None:
        selected_cells = selected_cells[:limit_cells]
    for cell in selected_cells:
        work.put(cell.cell_id)

    def mark_render_unresolvable(cell_id: str) -> None:
        nonlocal manifest
        if not allow_backfill:
            raise RuntimeError(
                "render-only coverage passes must not write coverage.plan.json"
            )
        cell = next(item for item in manifest.cells if item.cell_id == cell_id)
        if cell.search_pool_exhausted:
            return
        rejection_counts = Counter(cell.rejection_counts)
        rejection_counts["binding:render_unresolvable"] += 1
        updated_cell = cell.model_copy(
            update={
                "search_pool_exhausted": True,
                "rejection_counts": dict(rejection_counts),
            }
        )
        manifest = manifest.model_copy(
            update={
                "cells": tuple(
                    updated_cell if item.cell_id == cell_id else item
                    for item in manifest.cells
                )
            }
        )
        _write_json(manifest_path, manifest)
        _refresh_status_for_manifest(status, manifest, recover_interrupted=False)

    def append_backfill(cell_id: str) -> int:
        nonlocal manifest
        cell = next(item for item in manifest.cells if item.cell_id == cell_id)
        # Deferred capabilities are question-only coverage targets.  Their
        # episodes must come from compatible producer trajectories, never from
        # an independent geometry search using the deferred question spec.
        if cell.capability in DEFERRED_INITIAL_CAPABILITIES:
            return 0
        if cell.search_pool_exhausted:
            return 0
        if (
            cell.search_attempt_limit >= manifest.attempts_per_binding
            and cell.geometry_pool_size == 0
        ):
            return 0
        scene = scenes[cell.scene_key]
        row = status["cells"][cell_id]
        missing_runtime_ids = _confirmed_render_unresolvable_targets(cell, scene, row)
        if missing_runtime_ids:
            mark_render_unresolvable(cell_id)
            save()
            print(
                f"coverage cell={cell_id} backfill=skipped "
                f"render_unresolvable={','.join(missing_runtime_ids)}",
                flush=True,
            )
            return 0
        candidates = _candidate_lookup(manifest)
        credited_plans = tuple(
            TrajectoryPlan.model_validate_json(
                Path(candidates[episode_id].plan_record).read_text(encoding="utf-8")
            )
            for episode_id in row["accepted_episode_ids"]
            if episode_id in candidates
        )
        downstream_missing = [missing_slots(item) for item in potential_credit_ids(cell)]
        needed = max(
            1,
            cell.target_accepted - len(row["accepted_episode_ids"]),
            *downstream_missing,
        )
        search = _new_candidates(
            cell,
            scene,
            manifest,
            requested=needed,
            attempt_limit=manifest.attempts_per_binding,
            additional_prior_plans=credited_plans,
            std=std,
        )
        updated_cell = cell.model_copy(
            update={
                "candidates": tuple((*cell.candidates, *search.candidates)),
                "geometry_pool_size": search.geometry_pool_size,
                "diverse_pool_size": search.diverse_pool_size,
                "search_attempt_limit": search.search_attempt_limit,
                "raw_plan_limit": search.raw_plan_limit,
                "search_pool_exhausted": search.search_pool_exhausted,
                "rejection_counts": search.rejection_counts,
            }
        )
        manifest = manifest.model_copy(
            update={
                "cells": tuple(
                    updated_cell if item.cell_id == cell_id else item for item in manifest.cells
                )
            }
        )
        _write_json(manifest_path, manifest)
        _refresh_status_for_manifest(status, manifest, recover_interrupted=False)
        save()
        return len(search.candidates)

    def process_cell(cell_id: str, gpu_id: int) -> None:
        while True:
            with lock:
                cell = next(item for item in manifest.cells if item.cell_id == cell_id)
                row = status["cells"][cell_id]
                if not source_needs_work(cell):
                    row["status"] = "complete"
                    save()
                    return
                pending = next(
                    (
                        candidate
                        for candidate in cell.candidates
                        if row["candidate_statuses"][candidate.candidate_id]["status"]
                        == "pending"
                    ),
                    None,
                )
                if pending is None:
                    if not allow_backfill:
                        row["status"] = "awaiting_candidates"
                        save()
                        return
                    if append_backfill(cell_id) == 0:
                        row["status"] = (
                            "complete"
                            if len(row["accepted_episode_ids"]) >= cell.target_accepted
                            else "exhausted"
                        )
                        save()
                        return
                    continue
                row["status"] = "running"
                row["candidate_statuses"][pending.candidate_id] = {
                    "status": "running",
                    "reason": None,
                    "gpu_id": gpu_id,
                }
                save()
            scene = scenes[cell.scene_key]
            rendered = _bundle_complete(pending)
            reason = None
            if not rendered:
                rendered, reason = _render(
                    recipe=Path(pending.recipe),
                    bundle=Path(pending.bundle),
                    log_path=Path(pending.log),
                    og_root=og_root.resolve(),
                    data_root=data_root.resolve(),
                    conda_env=conda_env,
                    gpu_id=gpu_id,
                    timeout_minutes=timeout_minutes,
                )
                if not rendered and _render_failure_is_retryable(reason):
                    rendered, reason = _render(
                        recipe=Path(pending.recipe),
                        bundle=Path(pending.bundle),
                        log_path=Path(pending.log),
                        og_root=og_root.resolve(),
                        data_root=data_root.resolve(),
                        conda_env=conda_env,
                        gpu_id=gpu_id,
                        timeout_minutes=timeout_minutes,
                    )
            group = None
            plan = None
            if rendered:
                try:
                    group, plan = _validate_rendered_candidate(
                        pending,
                        cell,
                        scene,
                        std=std,
                    )
                except (
                    FamilyBlocked,
                    FamilyMismatch,
                    OSError,
                    KeyError,
                    ValueError,
                ) as error:
                    reason = f"authority:{type(error).__name__}:{error}"
            terminal_missing_runtime_ids: tuple[str, ...] = ()
            with lock:
                row = status["cells"][cell_id]
                if group is None or plan is None:
                    row["candidate_statuses"][pending.candidate_id] = {
                        "status": "rejected",
                        "reason": reason or "render_or_authority_failure",
                        "gpu_id": gpu_id,
                    }
                    terminal_missing_runtime_ids = _confirmed_render_unresolvable_targets(
                        cell,
                        scene,
                        row,
                    )
                    if terminal_missing_runtime_ids:
                        # While planning is active this render-only pass may
                        # hold an older manifest snapshot. Persist the verdict
                        # only in status; the final backfill pass reloads the
                        # completed manifest and merges it there. Otherwise an
                        # old snapshot can replace all newly planned cells.
                        if allow_backfill:
                            mark_render_unresolvable(cell_id)
                            row = status["cells"][cell_id]
                        row["status"] = "exhausted"
                else:
                    credited = credit_episode(pending, plan, group, cell)
                    candidate_status = "accepted" if credited else "redundant"
                    row["candidate_statuses"][pending.candidate_id] = {
                        "status": candidate_status,
                        "reason": (
                            None
                            if credited
                            else "quota_already_filled_or_near_duplicate"
                        ),
                        "gpu_id": gpu_id,
                        "credited_cells": list(credited),
                    }
                    if credited:
                        status["episodes"][pending.candidate_id] = {
                            "scene_key": cell.scene_key,
                            "source_capability": cell.capability,
                            "binding": plan.binding,
                            "plan_id": plan.plan_id,
                            "bundle": pending.bundle,
                            "group": str(Path(pending.group) / "group.json"),
                            "credited_cells": list(credited),
                        }
                save()
                print(
                    f"coverage candidate={pending.candidate_id} "
                    f"status={row['candidate_statuses'][pending.candidate_id]['status']} "
                    f"accepted={len(row['accepted_episode_ids'])}/{cell.target_accepted}",
                    flush=True,
                )
            if terminal_missing_runtime_ids:
                print(
                    f"coverage cell={cell_id} candidates=stopped "
                    f"render_unresolvable={','.join(terminal_missing_runtime_ids)}",
                    flush=True,
                )
                return
    worker_count = min(workers or len(gpu_ids), len(gpu_ids))
    worker_failed = threading.Event()
    worker_errors: list[tuple[str, BaseException]] = []

    def worker(gpu_id: int) -> None:
        while not worker_failed.is_set():
            try:
                cell_id = work.get_nowait()
            except queue.Empty:
                return
            try:
                process_cell(cell_id, gpu_id)
            except BaseException as error:
                with lock:
                    worker_errors.append((cell_id, error))
                    save()
                worker_failed.set()
                return
            finally:
                work.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu_id,), name=f"coverage-gpu-{gpu_id}")
        for gpu_id in gpu_ids[:worker_count]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with lock:
        save()
    if worker_errors:
        cell_id, error = worker_errors[0]
        raise RuntimeError(f"coverage worker failed: cell={cell_id}") from error
    return status_path


def _completed_scene_prefix(manifest: CoverageManifest) -> tuple[str, ...]:
    """Infer the definitely complete scene prefix of an interrupted planner."""
    if not manifest.cells:
        return ()
    scene_order = [scene.scene_key for scene in manifest.scenes]
    last_scene = manifest.cells[-1].scene_key
    try:
        last_index = scene_order.index(last_scene)
    except ValueError:
        return ()
    return tuple(scene_order[:last_index])


def run_coverage_pipeline(
    manifest_path: Path,
    *,
    og_root: Path,
    conda_env: str,
    data_root: Path,
    gpu_ids: tuple[int, ...],
    workers: int | None = None,
    timeout_minutes: int = 20,
    std: CompileStandard = STD_V1,
) -> Path:
    """Resume planning while rendering only scene-complete manifest snapshots.

    The planner is the sole writer of ``coverage.plan.json`` until it finishes.
    Render passes write only ``coverage.status.json`` during that interval, so
    atomic manifest replacement and backfill can never race with each other.
    """
    if not gpu_ids:
        raise ValueError("at least one GPU id is required")
    manifest_path = manifest_path.resolve()
    initial = load_coverage(manifest_path)
    if initial.standard_version != std.standard_version:
        raise ValueError(
            f"pipeline standard differs: {initial.standard_version} != {std.standard_version}"
        )
    verify_coverage_sources(initial)
    _preflight(og_root.resolve(), data_root.resolve(), conda_env)

    ready_scenes: queue.Queue[str] = queue.Queue()
    queued_scenes = set(_completed_scene_prefix(initial))
    for scene_key in queued_scenes:
        ready_scenes.put(scene_key)
    queue_lock = threading.Lock()
    planner_errors: list[BaseException] = []

    def scene_completed(scene_key: str) -> None:
        with queue_lock:
            if scene_key in queued_scenes:
                return
            queued_scenes.add(scene_key)
            ready_scenes.put(scene_key)
        print(f"coverage pipeline scene_ready={scene_key}", flush=True)

    def planner() -> None:
        try:
            plan_coverage(
                source_index_path=Path(initial.source_index),
                output_root=Path(initial.output_root),
                collection_id=initial.collection_id,
                accepted_per_binding=initial.accepted_per_binding,
                attempts_per_binding=initial.attempts_per_binding,
                initial_attempts_per_binding=initial.initial_attempts_per_binding,
                maximum_multislot_bindings=initial.maximum_multislot_bindings,
                limit_bindings_per_capability=initial.limit_bindings_per_capability,
                capabilities=initial.capabilities,
                desired_answer_label=initial.desired_answer_label,
                scene_keys=tuple(scene.scene_key for scene in initial.scenes),
                std=std,
                on_scene_completed=scene_completed,
            )
        except BaseException as error:  # surfaced by the coordinating thread
            planner_errors.append(error)

    planner_thread = threading.Thread(
        target=planner,
        name="coverage-planner",
        daemon=True,
    )
    planner_thread.start()

    while planner_thread.is_alive() or not ready_scenes.empty():
        try:
            first_scene = ready_scenes.get(timeout=1.0)
        except queue.Empty:
            continue
        batch = [first_scene]
        while True:
            try:
                batch.append(ready_scenes.get_nowait())
            except queue.Empty:
                break
        print(
            f"coverage pipeline render_scenes={','.join(batch)} planner_alive="
            f"{planner_thread.is_alive()}",
            flush=True,
        )
        run_coverage(
            manifest_path,
            og_root=og_root,
            conda_env=conda_env,
            data_root=data_root,
            gpu_ids=gpu_ids,
            workers=workers,
            timeout_minutes=timeout_minutes,
            scene_keys=tuple(batch),
            allow_backfill=False,
            skip_preflight=True,
            std=std,
        )
        for _ in batch:
            ready_scenes.task_done()

    planner_thread.join()
    if planner_errors:
        raise RuntimeError("coverage planner failed") from planner_errors[0]

    print("coverage pipeline planning_complete; starting authoritative backfill", flush=True)
    return run_coverage(
        manifest_path,
        og_root=og_root,
        conda_env=conda_env,
        data_root=data_root,
        gpu_ids=gpu_ids,
        workers=workers,
        timeout_minutes=timeout_minutes,
        allow_backfill=True,
        skip_preflight=True,
        std=std,
    )


def package_coverage(manifest_path: Path) -> tuple[Path, Path]:
    manifest = load_coverage(manifest_path)
    verify_coverage_sources(manifest)
    status_path = Path(manifest.output_root) / "coverage.status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"coverage status is missing: {status_path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    cells = {cell.cell_id: cell for cell in manifest.cells}
    cell_rows = []
    for cell_id, row in status["cells"].items():
        cell = cells[cell_id]
        cell_rows.append(
            {
                "cell_id": cell_id,
                "scene_key": cell.scene_key,
                "capability": cell.capability,
                "binding": cell.binding,
                "status": row["status"],
                "target_accepted": cell.target_accepted,
                "accepted_count": len(row["accepted_episode_ids"]),
                "accepted_episode_ids": row["accepted_episode_ids"],
                "candidate_counts": dict(
                    Counter(item["status"] for item in row["candidate_statuses"].values())
                ),
                "rejection_counts": cell.rejection_counts,
            }
        )
    report = {
        "schema_version": "scriptgen_binding_coverage_report.v1",
        "collection_id": manifest.collection_id,
        "standard_version": manifest.standard_version,
        "complete_cell_count": sum(row["status"] == "complete" for row in cell_rows),
        "exhausted_cell_count": sum(row["status"] == "exhausted" for row in cell_rows),
        "cells": cell_rows,
    }
    dataset = {
        "schema_version": COVERAGE_DATASET_SCHEMA_VERSION,
        "collection_id": manifest.collection_id,
        "standard_version": manifest.standard_version,
        "episode_count": len(status["episodes"]),
        "episodes": [
            {"episode_id": episode_id, **payload}
            for episode_id, payload in sorted(status["episodes"].items())
        ],
        "coverage_report": "coverage.report.json",
    }
    output_root = Path(manifest.output_root)
    report_path = output_root / "coverage.report.json"
    dataset_path = output_root / "dataset.json"
    _write_json(report_path, report)
    _write_json(dataset_path, dataset)
    index = output_root / "index.html"
    index.write_text(
        """<!doctype html><meta charset=\"utf-8\"><title>EpiSpace coverage</title>
<style>body{font:14px system-ui;margin:2rem}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.35rem}code{font-size:12px}</style>
<h1>EpiSpace binding coverage</h1><p id=\"summary\"></p><table><thead><tr><th>scene</th><th>capability</th><th>status</th><th>accepted</th><th>binding</th></tr></thead><tbody id=\"rows\"></tbody></table>
<script>fetch('coverage.report.json').then(r=>r.json()).then(d=>{summary.textContent=`${d.complete_cell_count} complete, ${d.exhausted_cell_count} exhausted`;for(const x of d.cells){const tr=document.createElement('tr');tr.innerHTML=`<td>${x.scene_key}</td><td>${x.capability}</td><td>${x.status}</td><td>${x.accepted_count}/${x.target_accepted}</td><td><code>${JSON.stringify(x.binding)}</code></td>`;rows.append(tr)}})</script>""",
        encoding="utf-8",
    )
    return dataset_path, report_path
