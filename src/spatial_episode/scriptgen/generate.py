"""Main generation loop: scene + script -> qualifying trajectory plans.

Flow per slot binding:

1. a motif proposes a candidate pose sequence (cheap, render-free),
2. the checker resolves frame variables and judges every clause on the
   geometry backend (compile-phase clauses are estimated here and re-checked
   authoritatively after rendering),
3. qualifying candidates are collected until ``plans_per_binding`` plans have
   been retained or the per-binding attempt budget is exhausted; failures are
   tallied by failing clause so scene or script problems surface as a rejection
   histogram instead of silence.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Iterable

from .checker import check_clauses
from .compiler import derive_answer
from .motifs import get_motif
from .plan import GenerationReport, PlannedPose, ProvisionalAnswer, TrajectoryPlan
from .sceneview import GeometrySceneView, SceneLayout
from .slotting import iter_bindings
from .spec import ScriptSpec
from .standards import CompileStandard

CandidateFilter = Callable[[GeometrySceneView, dict[str, str]], str | None]


def generate_plans(
    layout: SceneLayout,
    script: ScriptSpec,
    std: CompileStandard,
    *,
    seed: int,
    attempts_per_binding: int = 150,
    plans_per_binding: int = 10,
    max_plans: int | None = None,
    maximum_bindings: int | None = None,
    candidate_bindings: Iterable[dict[str, str]] | None = None,
    candidate_filter: CandidateFilter | None = None,
) -> GenerationReport:
    """Generate up to ``plans_per_binding`` accepted plans for every binding.

    ``attempts_per_binding`` is the total candidate budget for one binding,
    not the number of attempts for each requested plan.  One seeded RNG drives
    the whole run, so callers do not need to manufacture a seed per trajectory.
    ``max_plans`` remains an optional global cap across all bindings.
    """
    if attempts_per_binding <= 0:
        raise ValueError("attempts_per_binding must be positive")
    if plans_per_binding <= 0:
        raise ValueError("plans_per_binding must be positive")
    if max_plans is not None and max_plans < 0:
        raise ValueError("max_plans must be non-negative")

    rng = random.Random(seed)
    if candidate_bindings is None:
        bindings, slot_rejections = iter_bindings(layout, script, maximum=maximum_bindings)
    else:
        bindings = iter(candidate_bindings)
        slot_rejections = []
    rejection_counts: Counter[str] = Counter()
    plans: list[TrajectoryPlan] = []
    binding_count = 0

    for binding_index, binding in enumerate(bindings):
        binding_count += 1
        if max_plans is not None and len(plans) >= max_plans:
            break
        remaining = None if max_plans is None else max_plans - len(plans)
        requested = plans_per_binding if remaining is None else min(plans_per_binding, remaining)
        binding_plans = _search_binding(
            layout,
            script,
            std,
            binding,
            binding_index,
            seed,
            rng,
            attempts_per_binding,
            requested,
            rejection_counts,
            candidate_filter,
        )
        plans.extend(binding_plans)

    if binding_count == 0:
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
    requested: int,
    rejection_counts: Counter[str],
    candidate_filter: CandidateFilter | None,
) -> list[TrajectoryPlan]:
    plans: list[TrajectoryPlan] = []
    for attempt in range(attempts):
        if len(plans) >= requested:
            break
        motif_name = script.motifs[attempt % len(script.motifs)]
        frame_count = rng.randint(*script.length)
        poses = get_motif(motif_name)(layout, binding, frame_count, rng)
        view = GeometrySceneView(layout=layout, poses=poses, std=std)

        report = check_clauses(
            view,
            script,
            binding,
            std,
            phases=("search", "search_only", "compile"),
        )
        if not report.passed:
            rejection_counts[f"clause:{report.failed_clause}"] += 1
            continue
        if candidate_filter is not None:
            rejection = candidate_filter(view, binding)
            if rejection is not None:
                rejection_counts[f"candidate_filter:{rejection}"] += 1
                continue

        env = {**binding, **report.frame_vars}
        plans.append(
            TrajectoryPlan(
                plan_id=(
                    f"{layout.scene_id}.{script.capability}."
                    f"b{binding_index}.s{seed}.a{attempt}"
                ),
                scene_id=layout.scene_id,
                capability=script.capability,
                standard_version=std.standard_version,
                seed=seed,
                binding=binding,
                frame_vars=report.frame_vars,
                poses=tuple(
                    PlannedPose.from_pose(frame, pose) for frame, pose in enumerate(poses)
                ),
                knob_levels={
                    knob.name: float(_eval_knob(knob.expr, env)) for knob in script.knobs
                },
                clause_witnesses=report.witnesses(),
                provisional_answer=_provisional_answer(
                    view,
                    std,
                    script,
                    binding,
                    report.frame_vars,
                ),
            )
        )

    if not plans:
        rejection_counts["binding_exhausted"] += 1
    elif len(plans) < requested:
        rejection_counts["binding_quota_unfilled"] += requested - len(plans)
    return plans


def _eval_knob(expr: str, env: dict[str, object]) -> int:
    from .checker import eval_int_expr

    return eval_int_expr(expr, env)


def _provisional_answer(
    view: GeometrySceneView,
    std: CompileStandard,
    script: ScriptSpec,
    binding: dict[str, str],
    frame_vars: dict[str, int],
) -> ProvisionalAnswer:
    answer = derive_answer(
        view,
        std,
        script,
        binding,
        frame_vars,
        template=script.templates[0],
    )
    return ProvisionalAnswer(
        mode=answer.mode,
        label=answer.label,
        witness=answer.witness,
    )
