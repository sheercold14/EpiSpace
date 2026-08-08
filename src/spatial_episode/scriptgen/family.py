"""Family and question-group packing from one rendered bundle.

A family is the evaluation unit of the whole programme: the canonical episode
plus its four intervention variants, all sharing one scene, one target and one
verbatim question, with every gold produced by the same authoritative
compiler. Decision #2: the family is ONE JSON document; frame imagery is
referenced by relative path so the document stays self-describing and
portable next to its media directory.

Packing refuses to ship anything questionable:

* the canonical certificate must be answerable with no render/geometry
  mismatch;
* every variant must match its declared expectation (else FamilyMismatch);
* the target referent must be unique in its category (else the question text
  "the armchair" is ambiguous);
* the question text must contain no leak: no unfilled placeholders, no frame
  numbers, and no occurrence of the gold answer token.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Literal

from .behavior import RenderSceneView
from .compiler import CapabilityCompiler, Certificate
from .library import (
    HOMING,
    MULTI_TURN,
    NET_TURN,
    NET_TURN_MAGNITUDE,
    OCCLUDED_MOTION,
    PURE_ROTATION,
    PURE_TRANSLATION,
    REFERENCE_FRAME_SCRIPTS,
    SCRIPT_LIBRARY,
    SELF_MOTION,
    VIEW_SIDE,
)
from .sceneview import SceneLayout
from .spec import ScriptSpec, SpecModel, Template
from .standards import CompileStandard
from .variants import Variant, VariantBuilder

WEB_TEMPLATE = Path(__file__).resolve().parents[3] / "web" / "scriptgen_family_review.html"

FamilyRole = Literal["primary", "probe", "check"]
QUESTION_ROLES: dict[str, FamilyRole] = {
    SELF_MOTION.capability: "primary",
    NET_TURN.capability: "primary",
    NET_TURN_MAGNITUDE.capability: "primary",
    HOMING.capability: "probe",
    VIEW_SIDE.capability: "check",
    PURE_ROTATION.capability: "primary",
    PURE_TRANSLATION.capability: "primary",
    MULTI_TURN.capability: "primary",
    OCCLUDED_MOTION.capability: "primary",
    **{script.capability: "primary" for script in REFERENCE_FRAME_SCRIPTS},
}
QUESTION_GROUP_SCRIPTS = (SELF_MOTION, NET_TURN, NET_TURN_MAGNITUDE, HOMING, VIEW_SIDE)
QUESTION_SCRIPT_SETS = {
    script.capability: REFERENCE_FRAME_SCRIPTS for script in REFERENCE_FRAME_SCRIPTS
}


class FamilyQuestion(SpecModel):
    """The one question every family member asks, verbatim."""

    text: str
    options: tuple[str, ...]
    abstain_option: str


class FamilyTarget(SpecModel):
    entity_id: str
    category: str


class FrameMedia(SpecModel):
    """One originally rendered frame: imagery paths plus pose ground truth."""

    frame: int
    rgb: str  # relative paths, decision #2
    depth: str
    instance: str
    x: float
    y: float
    yaw_deg: float


class FamilyChecks(SpecModel):
    """Packaging-time audit results; every field must be clean to ship."""

    referent_unique: bool
    unfilled_placeholders: bool  # True would mean a leftover {slot}
    frame_number_leak: bool  # True would mean the text cites frame indices
    answer_token_leak: bool  # True would mean the gold appears in the text


class FamilyEpisode(SpecModel):
    """Canonical or variant: an index sequence plus certificate and label."""

    episode_id: str
    kind: Literal["canonical", "permute", "drop_key", "drop_filler", "delay"]
    expectation: Literal["canonical", "same", "abstain"]
    frame_sequence: tuple[int, ...]
    label: str
    certificate: Certificate


class ScriptgenFamilyV4(SpecModel):
    """The versioned family contract (registered in contracts/schema.py)."""

    schema_version: Literal["scriptgen_family.v4"] = "scriptgen_family.v4"
    family_id: str
    question_group_id: str
    role: FamilyRole
    scene_id: str
    capability: str
    standard_version: str
    seed: int
    plan_id: str
    target: FamilyTarget
    referents: dict[str, FamilyTarget]
    question: FamilyQuestion
    frames: tuple[FrameMedia, ...]
    episodes: tuple[FamilyEpisode, ...]
    checks: FamilyChecks


class QuestionGroupTrajectory(SpecModel):
    """Files and plan identity from which a question group was compiled."""

    bundle: str
    plan_record: str
    scene_ir: str
    plan_id: str
    scene_id: str


class QuestionGroupEntry(SpecModel):
    """One requested question type, whether built or opportunity-skipped."""

    capability: str
    role: FamilyRole
    family_id: str | None
    label: str | None
    family: str | None
    skip_reason: str | None


class ScriptgenQuestionGroupV1(SpecModel):
    schema_version: Literal["scriptgen_question_group.v1"] = "scriptgen_question_group.v1"
    question_group_id: str
    standard_version: str
    trajectory: QuestionGroupTrajectory
    questions: tuple[QuestionGroupEntry, ...]


class FamilyBlocked(RuntimeError):
    """The family failed a packaging audit and must not ship."""


def build_family_doc(
    view: RenderSceneView,
    plan: dict[str, Any],
    script: ScriptSpec,
    std: CompileStandard,
    *,
    seed: int,
    drop_count: int = 2,
    delay_extra: int = 4,
    role: FamilyRole | None = None,
    question_group_id: str | None = None,
    family_id: str | None = None,
    media_prefix: str = "media",
    template_index: int = 0,
) -> ScriptgenFamilyV4:
    """Compile canonical + variants and assemble the audited family document."""
    binding = dict(plan["binding"])
    compiler = CapabilityCompiler(
        script=script,
        std=std,
        template_index=template_index,
    )
    geometry_plan = plan if plan.get("capability") == script.capability else None
    canonical = compiler.compile(
        view,
        binding,
        geometry_plan=geometry_plan,
        with_essential=True,
    )
    if canonical.status != "answerable" or canonical.answer is None:
        raise FamilyBlocked(f"canonical not answerable: {canonical.reason}")
    if canonical.mismatch is not None:
        raise FamilyBlocked(f"canonical mismatch: {canonical.mismatch}")

    builder = VariantBuilder(compiler=compiler, view=view, binding=binding, canonical=canonical)
    variants = builder.build_all(seed=seed, drop_count=drop_count, delay_extra=delay_extra)

    target_id = binding.get("target", next(iter(binding.values())))
    target = view.object(target_id)
    referents = {
        slot: FamilyTarget(entity_id=entity_id, category=view.object(entity_id).category)
        for slot, entity_id in binding.items()
    }
    template = compiler.template
    question_text = template.text.format(
        **{slot: referent.category for slot, referent in referents.items()}
    )
    checks = _audit(question_text, template, tuple(referents.values()), view.layout, canonical)
    if not checks.referent_unique:
        raise FamilyBlocked(f"referent not unique in scene: {target.category}")
    if checks.unfilled_placeholders or checks.frame_number_leak or checks.answer_token_leak:
        raise FamilyBlocked(f"question text failed leak audit: {checks}")

    role = role or _role_for(script)
    question_group_id = question_group_id or f"{plan['plan_id']}.question_group"
    family_id = family_id or f"{plan['plan_id']}.family.s{seed}"
    episodes = (
        FamilyEpisode(
            episode_id=f"{family_id}.canonical",
            kind="canonical",
            expectation="canonical",
            frame_sequence=canonical.frame_sequence,
            label=canonical.answer.label,
            certificate=canonical,
        ),
    ) + tuple(_episode(family_id, variant) for variant in variants)

    frames = tuple(
        FrameMedia(
            frame=t,
            rgb=f"{media_prefix}/view-{t:03d}.rgb.png",
            depth=f"{media_prefix}/view-{t:03d}.depth.png",
            instance=f"{media_prefix}/view-{t:03d}.inst.png",
            x=view.camera_pose(t).x,
            y=view.camera_pose(t).y,
            yaw_deg=view.camera_pose(t).yaw_deg,
        )
        for t in range(view.frame_count)
    )

    return ScriptgenFamilyV4(
        family_id=family_id,
        question_group_id=question_group_id,
        role=role,
        scene_id=plan["scene_id"],
        capability=script.capability,
        standard_version=std.standard_version,
        seed=seed,
        plan_id=plan["plan_id"],
        target=FamilyTarget(entity_id=target_id, category=target.category),
        referents=referents,
        question=FamilyQuestion(
            text=question_text,
            options=template.options,
            abstain_option=template.abstain_option,
        ),
        frames=frames,
        episodes=episodes,
        checks=checks,
    )


def build_family_site(
    bundle: Path,
    plan_record: Path,
    scene_ir: Path,
    out: Path,
    std: CompileStandard,
    *,
    seed: int = 17,
    drop_count: int = 2,
    delay_extra: int = 4,
    template_index: int = 0,
) -> Path:
    """One command: rendered bundle -> family.json + media + review page."""
    from .media import export_bundle_channels

    plan = json.loads(plan_record.read_text(encoding="utf-8"))
    script = SCRIPT_LIBRARY[plan["capability"]]
    view = RenderSceneView.from_bundle(bundle, std, scene_ir=scene_ir)
    doc = build_family_doc(
        view,
        plan,
        script,
        std,
        seed=seed,
        drop_count=drop_count,
        delay_extra=delay_extra,
        template_index=template_index,
    )

    out.mkdir(parents=True, exist_ok=True)
    target_ids = [
        runtime_id
        for referent in doc.referents.values()
        for runtime_id in view.entity_runtime_ids.get(referent.entity_id, ())
    ]
    export_bundle_channels(bundle, target_ids, view.frame_count, out / "media")
    (out / "family.json").write_text(doc.model_dump_json(indent=1), encoding="utf-8")
    shutil.copyfile(WEB_TEMPLATE, out / "index.html")
    return out / "family.json"


def build_question_group(
    bundle: Path,
    plan_record: Path,
    scene_ir: Path,
    out: Path,
    std: CompileStandard,
    scripts: tuple[ScriptSpec, ...] | None = None,
    *,
    seed: int = 17,
    drop_count: int = 2,
    delay_extra: int = 4,
) -> Path:
    """Build every qualifying declared question family over one trajectory."""
    from .media import export_bundle_channels

    plan = json.loads(plan_record.read_text(encoding="utf-8"))
    view = RenderSceneView.from_bundle(bundle, std, scene_ir=scene_ir)
    binding = dict(plan["binding"])
    question_group_id = f"{plan['plan_id']}.question_group.s{seed}"
    if scripts is None:
        source = SCRIPT_LIBRARY[plan["capability"]]
        scripts = QUESTION_SCRIPT_SETS.get(
            source.capability,
            tuple(
                {
                    script.capability: script for script in (source, *QUESTION_GROUP_SCRIPTS[1:])
                }.values()
            ),
        )

    out.mkdir(parents=True, exist_ok=True)
    target_ids = [
        runtime_id
        for entity_id in binding.values()
        for runtime_id in view.entity_runtime_ids.get(entity_id, ())
    ]
    export_bundle_channels(bundle, target_ids, view.frame_count, out / "media")

    questions: list[QuestionGroupEntry] = []
    for script in scripts:
        role = _role_for(script)
        compiler = CapabilityCompiler(script=script, std=std)
        geometry_plan = plan if plan.get("capability") == script.capability else None
        qualification = compiler.compile(
            view,
            binding,
            geometry_plan=geometry_plan,
        )
        if qualification.mismatch is not None:
            raise FamilyBlocked(
                f"canonical mismatch for {script.capability}: {qualification.mismatch}"
            )
        if qualification.status != "answerable" or qualification.answer is None:
            if script.capability == plan.get("capability"):
                raise FamilyBlocked(f"source capability not answerable: {qualification.reason}")
            questions.append(
                QuestionGroupEntry(
                    capability=script.capability,
                    role=role,
                    family_id=None,
                    label=None,
                    family=None,
                    skip_reason=f"{qualification.status}:{qualification.reason}",
                )
            )
            continue

        family_id = f"{question_group_id}.{script.capability}.family"
        doc = build_family_doc(
            view,
            plan,
            script,
            std,
            seed=seed,
            drop_count=drop_count,
            delay_extra=delay_extra,
            role=role,
            question_group_id=question_group_id,
            family_id=family_id,
            media_prefix="../media",
        )
        family_dir = out / script.capability
        family_dir.mkdir(parents=True, exist_ok=True)
        family_path = family_dir / "family.json"
        family_path.write_text(doc.model_dump_json(indent=1), encoding="utf-8")
        shutil.copyfile(WEB_TEMPLATE, family_dir / "index.html")
        questions.append(
            QuestionGroupEntry(
                capability=script.capability,
                role=role,
                family_id=doc.family_id,
                label=qualification.answer.label,
                family=str(family_path.relative_to(out)),
                skip_reason=None,
            )
        )

    group = ScriptgenQuestionGroupV1(
        question_group_id=question_group_id,
        standard_version=std.standard_version,
        trajectory=QuestionGroupTrajectory(
            bundle=str(bundle),
            plan_record=str(plan_record),
            scene_ir=str(scene_ir),
            plan_id=plan["plan_id"],
            scene_id=plan["scene_id"],
        ),
        questions=tuple(questions),
    )
    group_path = out / "group.json"
    group_path.write_text(group.model_dump_json(indent=1), encoding="utf-8")
    return group_path


def _episode(family_id: str, variant: Variant) -> FamilyEpisode:
    return FamilyEpisode(
        episode_id=f"{family_id}.{variant.kind}",
        kind=variant.kind,  # type: ignore[arg-type]
        expectation=variant.expectation,
        frame_sequence=variant.frame_sequence,
        label=variant.gold,
        certificate=variant.certificate,
    )


def _role_for(script: ScriptSpec) -> FamilyRole:
    try:
        return QUESTION_ROLES[script.capability]
    except KeyError as error:
        raise ValueError(f"no question-group role for {script.capability!r}") from error


def _audit(
    question_text: str,
    template: Template,
    referents: tuple[FamilyTarget, ...],
    layout: SceneLayout,
    canonical: Certificate,
) -> FamilyChecks:
    categories_unique = all(
        sum(obj.category == referent.category for obj in layout.objects) == 1
        for referent in referents
    )
    gold = canonical.answer.label if canonical.answer else ""
    return FamilyChecks(
        referent_unique=categories_unique,
        unfilled_placeholders=bool(re.search(r"[{}]", question_text)),
        frame_number_leak=bool(re.search(r"第\s*\d+\s*帧", question_text)),
        answer_token_leak=bool(gold) and gold in question_text,
    )
