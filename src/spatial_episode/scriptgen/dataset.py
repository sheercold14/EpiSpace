"""Post-render audit, packaging and read-only review dataset contracts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Literal

from pydantic import Field

from .behavior import RenderSceneView
from .collection import CollectionJob, CollectionManifest
from .family import (
    FamilyBlocked,
    ScriptgenFamilyV4,
    ScriptgenQuestionGroupV1,
    build_question_group,
)
from .media import export_named_view_channels
from .rendering import rendered_bundle_complete
from .sceneview import VisibilityObservation
from .spec import SpecModel
from .standards import STD_V1, CompileStandard

DATASET_SCHEMA_VERSION = "scriptgen_dataset.v1"
DATASET_REVIEW_TEMPLATE = (
    Path(__file__).resolve().parents[3] / "web" / "scriptgen_dataset_review.html"
)


class AuxiliaryViewAudit(SpecModel):
    view_id: str
    yaw_offset_deg: int
    target_entity_id: str
    target_category: str
    geometry_label: Literal["visible", "not_visible"]
    render_label: Literal["visible", "not_visible", "ambiguous"]
    target_pixels: int = Field(ge=0)
    matches: bool
    rgb: str
    depth: str
    instance: str


class DatasetQuestion(SpecModel):
    capability: str
    role: str
    label: str | None
    skip_reason: str | None
    review: str | None


class DatasetFrame(SpecModel):
    frame: int
    rgb: str
    depth: str
    instance: str
    x: float
    y: float
    yaw_deg: float


class DatasetTrajectory(SpecModel):
    job_id: str
    capability: str
    motif: str
    replicate: int
    seed: int
    scene_key: str
    scene_model: str
    scene_id: str
    plan_id: str
    status: Literal["accepted", "failed"]
    failure_reason: str | None = None
    primary_label: str | None = None
    frame_count: int
    auxiliary_view_count: int
    frames: tuple[DatasetFrame, ...] = ()
    questions: tuple[DatasetQuestion, ...] = ()
    auxiliary_views: tuple[AuxiliaryViewAudit, ...] = ()
    group: str | None = None


class ScriptgenDatasetV1(SpecModel):
    schema_version: Literal["scriptgen_dataset.v1"] = DATASET_SCHEMA_VERSION
    collection_id: str
    standard_version: str
    requested_capability_count: int
    requested_per_capability: int
    planned_trajectory_count: int
    accepted_trajectory_count: int
    failed_trajectory_count: int
    trajectories: tuple[DatasetTrajectory, ...]


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def audit_auxiliary_views(
    job: CollectionJob,
    view: RenderSceneView,
    dataset_root: Path,
    *,
    std: CompileStandard = STD_V1,
) -> tuple[AuxiliaryViewAudit, ...]:
    """Compare each imagined-pose geometry label with rendered target pixels."""
    render_plan = json.loads(Path(job.render_plan).read_text(encoding="utf-8"))
    auxiliary = render_plan.get("auxiliary_views", [])
    if len(auxiliary) != job.auxiliary_view_count:
        raise ValueError(f"auxiliary view count mismatch for {job.job_id}")
    if not auxiliary:
        return ()
    target_ids: list[int] = []
    for item in auxiliary:
        target_ids.extend(view.entity_runtime_ids.get(item["target_entity_id"], ()))
    target_ids = sorted(set(target_ids))
    media_dir = Path(job.question_group) / "auxiliary_media"
    view_ids = [str(item["view_id"]) for item in auxiliary]
    pixel_counts = export_named_view_channels(
        Path(job.bundle) / "auxiliary_views",
        view_ids,
        target_ids,
        media_dir,
    )
    rows: list[AuxiliaryViewAudit] = []
    for item, pixels in zip(auxiliary, pixel_counts, strict=True):
        state = VisibilityObservation("render_pixels", float(pixels), 1.0).tristate(std)
        render_label: Literal["visible", "not_visible", "ambiguous"] = (
            "visible" if state is True else "not_visible" if state is False else "ambiguous"
        )
        entity_id = str(item["target_entity_id"])
        view_id = str(item["view_id"])
        prefix = _relative(media_dir, dataset_root)
        geometry_label = str(item["geometry_label"])
        if geometry_label not in {"visible", "not_visible"}:
            raise ValueError(f"unknown auxiliary geometry label: {geometry_label}")
        rows.append(
            AuxiliaryViewAudit(
                view_id=view_id,
                yaw_offset_deg=int(item["yaw_offset_deg"]),
                target_entity_id=entity_id,
                target_category=view.object(entity_id).category,
                geometry_label=geometry_label,  # type: ignore[arg-type]
                render_label=render_label,
                target_pixels=pixels,
                matches=render_label == geometry_label,
                rgb=f"{prefix}/{view_id}.rgb.png",
                depth=f"{prefix}/{view_id}.depth.png",
                instance=f"{prefix}/{view_id}.inst.png",
            )
        )
    return tuple(rows)


def _failed(job: CollectionJob, reason: str) -> DatasetTrajectory:
    return DatasetTrajectory(
        job_id=job.job_id,
        capability=job.capability,
        motif=job.motif,
        replicate=job.replicate,
        seed=job.seed,
        scene_key=job.scene.scene_key,
        scene_model=job.scene.scene_model,
        scene_id=job.scene.scene_id,
        plan_id=job.plan_id,
        status="failed",
        failure_reason=reason,
        frame_count=job.frame_count,
        auxiliary_view_count=job.auxiliary_view_count,
    )


def _simulator_failure_reason(path: Path) -> str:
    failure = json.loads(path.read_text(encoding="utf-8"))
    error = failure.get("error", failure)
    return f"simulator:{error.get('type', 'failure')}:{error.get('message', '')}"


def _package_job(
    job: CollectionJob,
    dataset_root: Path,
    std: CompileStandard,
) -> DatasetTrajectory:
    if not rendered_bundle_complete(job):
        failure_path = Path(job.bundle) / "failure_report.json"
        reason = "render_bundle_incomplete"
        if failure_path.is_file():
            reason = _simulator_failure_reason(failure_path)
        return _failed(job, reason)
    view = RenderSceneView.from_bundle(job.bundle, std, scene_ir=job.scene.scene_ir)
    try:
        auxiliary = audit_auxiliary_views(job, view, dataset_root, std=std)
    except (OSError, KeyError, ValueError) as error:
        return _failed(job, f"auxiliary_audit:{type(error).__name__}:{error}")
    mismatches = [row.view_id for row in auxiliary if not row.matches]
    if mismatches:
        return _failed(job, f"auxiliary_geometry_render_mismatch:{','.join(mismatches)}")
    try:
        group_path = build_question_group(
            Path(job.bundle),
            Path(job.plan_record),
            Path(job.scene.scene_ir),
            Path(job.question_group),
            std,
            seed=job.seed,
        )
    except (FamilyBlocked, OSError, KeyError, ValueError) as error:
        return _failed(job, f"question_group:{type(error).__name__}:{error}")
    group = ScriptgenQuestionGroupV1.model_validate_json(group_path.read_text(encoding="utf-8"))
    primary = next(
        question for question in group.questions if question.capability == job.capability
    )
    if primary.family is None or primary.label is None:
        return _failed(job, f"primary_question_skipped:{primary.skip_reason}")
    family_path = Path(job.question_group) / primary.family
    family = ScriptgenFamilyV4.model_validate_json(family_path.read_text(encoding="utf-8"))
    group_prefix = _relative(Path(job.question_group), dataset_root)
    frames = tuple(
        DatasetFrame(
            frame=frame.frame,
            rgb=_relative(family_path.parent / frame.rgb, dataset_root),
            depth=_relative(family_path.parent / frame.depth, dataset_root),
            instance=_relative(family_path.parent / frame.instance, dataset_root),
            x=frame.x,
            y=frame.y,
            yaw_deg=frame.yaw_deg,
        )
        for frame in family.frames
    )
    questions = tuple(
        DatasetQuestion(
            capability=question.capability,
            role=question.role,
            label=question.label,
            skip_reason=question.skip_reason,
            review=(
                f"{group_prefix}/{Path(question.family).parent}/index.html"
                if question.family is not None
                else None
            ),
        )
        for question in group.questions
    )
    return DatasetTrajectory(
        job_id=job.job_id,
        capability=job.capability,
        motif=job.motif,
        replicate=job.replicate,
        seed=job.seed,
        scene_key=job.scene.scene_key,
        scene_model=job.scene.scene_model,
        scene_id=job.scene.scene_id,
        plan_id=job.plan_id,
        status="accepted",
        primary_label=primary.label,
        frame_count=job.frame_count,
        auxiliary_view_count=job.auxiliary_view_count,
        frames=frames,
        questions=questions,
        auxiliary_views=auxiliary,
        group=_relative(group_path, dataset_root),
    )


def package_collection(
    manifest: CollectionManifest,
    *,
    std: CompileStandard = STD_V1,
) -> Path:
    """Audit every render, build question groups, and write the review portal."""
    dataset_root = Path(manifest.output_root)
    trajectories: list[DatasetTrajectory] = []
    for index, job in enumerate(manifest.jobs, start=1):
        trajectory = _package_job(job, dataset_root, std)
        trajectories.append(trajectory)
        print(
            f"package {index}/{len(manifest.jobs)} status={trajectory.status} job={job.job_id}",
            flush=True,
        )
    accepted = sum(item.status == "accepted" for item in trajectories)
    dataset = ScriptgenDatasetV1(
        collection_id=manifest.collection_id,
        standard_version=manifest.standard_version,
        requested_capability_count=len({job.capability for job in manifest.jobs}),
        requested_per_capability=manifest.requested_per_capability,
        planned_trajectory_count=len(manifest.jobs),
        accepted_trajectory_count=accepted,
        failed_trajectory_count=len(trajectories) - accepted,
        trajectories=tuple(trajectories),
    )
    path = dataset_root / "dataset.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(dataset.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    shutil.copyfile(DATASET_REVIEW_TEMPLATE, dataset_root / "index.html")
    return path
