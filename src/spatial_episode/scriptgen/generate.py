"""Main generation loop: scene + script -> qualifying trajectory plans.

Flow per slot binding:

1. a motif proposes a candidate pose sequence (cheap, render-free),
2. the checker resolves frame variables and judges every clause on the
   geometry backend (compile-phase clauses are estimated here and re-checked
   authoritatively after rendering),
3. the first qualifying candidate becomes a :class:`TrajectoryPlan`; failures
   are tallied by failing clause so scene or script problems surface as a
   rejection histogram instead of silence.
"""

from __future__ import annotations

import random
from collections import Counter

from .checker import check_clauses
from .geometry import azimuth_deg, sector_margin_deg, sector_of
from .motifs import get_motif
from .plan import GenerationReport, PlannedPose, ProvisionalAnswer, TrajectoryPlan
from .sceneview import GeometrySceneView, SceneLayout
from .slotting import enumerate_bindings
from .spec import ScriptSpec
from .standards import CompileStandard


def generate_plans(
    layout: SceneLayout,
    script: ScriptSpec,
    std: CompileStandard,
    *,
    seed: int,
    attempts_per_binding: int = 150,
    max_plans: int | None = None,
) -> GenerationReport:
    rng = random.Random(seed)
    bindings, slot_rejections = enumerate_bindings(layout, script)
    rejection_counts: Counter[str] = Counter()
    plans: list[TrajectoryPlan] = []

    for binding_index, binding in enumerate(bindings):
        if max_plans is not None and len(plans) >= max_plans:
            break
        plan = _search_binding(
            layout,
            script,
            std,
            binding,
            binding_index,
            seed,
            rng,
            attempts_per_binding,
            rejection_counts,
        )
        if plan is not None:
            plans.append(plan)

    if not bindings:
        rejection_counts["no_slot_binding"] += 1
    return GenerationReport(
        scene_id=layout.scene_id,
        capability=script.capability,
        standard_version=std.standard_version,
        plans=tuple(plans),
        rejection_counts=dict(rejection_counts),
        slot_rejections=dict(Counter(r.reason for r in slot_rejections)),
    )


def _search_binding(
    layout: SceneLayout,
    script: ScriptSpec,
    std: CompileStandard,
    binding: dict[str, str],
    binding_index: int,
    seed: int,
    rng: random.Random,
    attempts: int,
    rejection_counts: Counter[str],
) -> TrajectoryPlan | None:
    for attempt in range(attempts):
        motif_name = script.motifs[attempt % len(script.motifs)]
        frame_count = rng.randint(*script.length)
        poses = get_motif(motif_name)(layout, binding, frame_count, rng)
        view = GeometrySceneView(layout=layout, poses=poses, std=std)

        report = check_clauses(view, script, binding, std)
        if not report.passed:
            rejection_counts[f"clause:{report.failed_clause}"] += 1
            continue

        env = {**binding, **report.frame_vars}
        return TrajectoryPlan(
            plan_id=f"{layout.scene_id}.{script.capability}.b{binding_index}.s{seed}.a{attempt}",
            scene_id=layout.scene_id,
            capability=script.capability,
            standard_version=std.standard_version,
            seed=seed,
            binding=binding,
            frame_vars=report.frame_vars,
            poses=tuple(PlannedPose.from_pose(frame, pose) for frame, pose in enumerate(poses)),
            knob_levels={knob.name: float(_eval_knob(knob.expr, env)) for knob in script.knobs},
            clause_witnesses=report.witnesses(),
            # v1 convention: the questioned slot is named "target".
            provisional_answer=_provisional_answer(
                view, binding.get("target", next(iter(binding.values()))), report.frame_vars
            ),
        )
    rejection_counts["binding_exhausted"] += 1
    return None


def _eval_knob(expr: str, env: dict[str, object]) -> int:
    from .checker import eval_int_expr

    return eval_int_expr(expr, env)


def _provisional_answer(
    view: GeometrySceneView, target: str, frame_vars: dict[str, int]
) -> ProvisionalAnswer:
    t_q = frame_vars.get("t_q", view.frame_count - 1)
    pose = view.camera_pose(t_q)
    azimuth = azimuth_deg(pose.xy, pose.yaw_deg, view.object(target).xy)
    return ProvisionalAnswer(
        question_frame=t_q,
        target=target,
        azimuth_deg=round(azimuth, 1),
        sector=sector_of(azimuth),
        margin_deg=round(sector_margin_deg(azimuth), 1),
    )
