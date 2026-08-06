"""Family packing: one rendered bundle -> one versioned family document.

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
from .library import SCRIPT_LIBRARY
from .sceneview import SceneLayout
from .spec import ScriptSpec, SpecModel, Template
from .standards import CompileStandard
from .variants import Variant, VariantBuilder

WEB_TEMPLATE = Path(__file__).resolve().parents[3] / "web" / "scriptgen_family_review.html"


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
    """Canonical or variant: an index sequence plus its certificate and gold."""

    episode_id: str
    kind: Literal["canonical", "permute", "drop_key", "drop_filler", "delay"]
    expectation: Literal["canonical", "same", "abstain"]
    frame_sequence: tuple[int, ...]
    gold: str
    certificate: Certificate


class ScriptgenFamilyV1(SpecModel):
    """The versioned family contract (registered in contracts/schema.py)."""

    schema_version: Literal["scriptgen_family.v1"] = "scriptgen_family.v1"
    family_id: str
    scene_id: str
    capability: str
    standard_version: str
    seed: int
    plan_id: str
    target: FamilyTarget
    question: FamilyQuestion
    frames: tuple[FrameMedia, ...]
    episodes: tuple[FamilyEpisode, ...]
    checks: FamilyChecks


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
) -> ScriptgenFamilyV1:
    """Compile canonical + variants and assemble the audited family document."""
    binding = dict(plan["binding"])
    compiler = CapabilityCompiler(script=script, std=std)
    canonical = compiler.compile(view, binding, geometry_plan=plan, with_essential=True)
    if canonical.status != "answerable" or canonical.answer is None:
        raise FamilyBlocked(f"canonical not answerable: {canonical.reason}")
    if canonical.mismatch is not None:
        raise FamilyBlocked(f"canonical mismatch: {canonical.mismatch}")

    builder = VariantBuilder(compiler=compiler, view=view, binding=binding, canonical=canonical)
    variants = builder.build_all(seed=seed, drop_count=drop_count, delay_extra=delay_extra)

    target_id = binding.get("target", next(iter(binding.values())))
    target = view.object(target_id)
    template = script.templates[0]
    question_text = template.text.format(target=target.category)
    checks = _audit(question_text, template, target.category, view.layout, canonical)
    if not checks.referent_unique:
        raise FamilyBlocked(f"referent not unique in scene: {target.category}")
    if checks.unfilled_placeholders or checks.frame_number_leak or checks.answer_token_leak:
        raise FamilyBlocked(f"question text failed leak audit: {checks}")

    family_id = f"{plan['plan_id']}.family.s{seed}"
    episodes = (
        FamilyEpisode(
            episode_id=f"{family_id}.canonical",
            kind="canonical",
            expectation="canonical",
            frame_sequence=canonical.frame_sequence,
            gold=canonical.answer.sector,
            certificate=canonical,
        ),
    ) + tuple(_episode(family_id, variant) for variant in variants)

    frames = tuple(
        FrameMedia(
            frame=t,
            rgb=f"media/view-{t:03d}.rgb.png",
            depth=f"media/view-{t:03d}.depth.png",
            instance=f"media/view-{t:03d}.inst.png",
            x=view.camera_pose(t).x,
            y=view.camera_pose(t).y,
            yaw_deg=view.camera_pose(t).yaw_deg,
        )
        for t in range(view.frame_count)
    )

    return ScriptgenFamilyV1(
        family_id=family_id,
        scene_id=plan["scene_id"],
        capability=script.capability,
        standard_version=std.standard_version,
        seed=seed,
        plan_id=plan["plan_id"],
        target=FamilyTarget(entity_id=target_id, category=target.category),
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
) -> Path:
    """One command: rendered bundle -> family.json + media + review page."""
    from .media import export_bundle_channels

    plan = json.loads(plan_record.read_text(encoding="utf-8"))
    script = SCRIPT_LIBRARY[plan["capability"]]
    view = RenderSceneView.from_bundle(bundle, std, scene_ir=scene_ir)
    doc = build_family_doc(
        view, plan, script, std, seed=seed, drop_count=drop_count, delay_extra=delay_extra
    )

    out.mkdir(parents=True, exist_ok=True)
    target_ids = list(view.entity_runtime_ids.get(doc.target.entity_id, ()))
    export_bundle_channels(bundle, target_ids, view.frame_count, out / "media")
    (out / "family.json").write_text(
        doc.model_dump_json(indent=1), encoding="utf-8"
    )
    shutil.copyfile(WEB_TEMPLATE, out / "index.html")
    return out / "family.json"


def _episode(family_id: str, variant: Variant) -> FamilyEpisode:
    return FamilyEpisode(
        episode_id=f"{family_id}.{variant.kind}",
        kind=variant.kind,  # type: ignore[arg-type]
        expectation=variant.expectation,
        frame_sequence=variant.frame_sequence,
        gold=variant.gold,
        certificate=variant.certificate,
    )


def _audit(
    question_text: str,
    template: Template,
    target_category: str,
    layout: SceneLayout,
    canonical: Certificate,
) -> FamilyChecks:
    same_category = [o for o in layout.objects if o.category == target_category]
    gold = canonical.answer.sector if canonical.answer else ""
    return FamilyChecks(
        referent_unique=len(same_category) == 1,
        unfilled_placeholders=bool(re.search(r"[{}]", question_text)),
        frame_number_leak=bool(re.search(r"第\s*\d+\s*帧", question_text)),
        answer_token_leak=bool(gold) and gold in question_text,
    )
