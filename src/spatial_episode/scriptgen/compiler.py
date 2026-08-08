"""Authoritative capability compiler: rendered bundle -> certificate.

The compiler is the single authority for gold answers. Given a SceneView
(normally the render backend, whose visibility comes from instance-mask
pixels) it

1. re-resolves every frame variable on that backend — geometry-phase values
   (e.g. ``t_seen``) are NEVER trusted, only carried as comparison fields;
2. re-judges every clause of the script, search and compile phase alike, with
   search tightening OFF (full margins apply at compile time);
3. dispatches the spec's declared answer mode with arguments resolved by the
   same expression machinery as clause arguments;
4. classifies failures via the spec's declared semantics: an ``abstain``
   clause/frame-var failure makes the gold the abstain option (the evidence is
   gone), an ``invalid`` failure voids the question entirely;
5. optionally computes the leave-one-out essential frame set: the frames whose
   single removal breaks a clause or changes the answer.

The same ``compile`` entry point serves canonical bundles and intervened
variants — a variant is just a frame index sequence over the same rendered
frames (:class:`ReindexedSceneView`), so its gold is produced by exactly the
same code path. Answers are never hand-assigned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .answers import get_answer_mode
from .checker import (
    FrameVarUnresolvable,
    check_clauses,
    resolve_args,
    resolve_frame_vars,
)
from .sceneview import ReindexedSceneView, SceneView
from .spec import ScriptSpec, SpecModel
from .standards import COMPATIBLE_PLAN_STANDARDS, CompileStandard

CompileStatus = Literal["answerable", "abstain", "invalid"]

Tristate = Literal["visible", "invisible", "ambiguous"]
_TRISTATE_NAMES: dict[bool | None, Tristate] = {
    True: "visible",
    False: "invisible",
    None: "ambiguous",
}


class ClauseOutcome(SpecModel):
    """One clause's authoritative verdict with its witness."""

    name: str
    predicate: str
    phase: str
    on_violation: str
    holds: bool | None
    witness: dict[str, Any]


class AuthoritativeAnswer(SpecModel):
    """Answer produced by one registered mode from rendered evidence."""

    mode: str
    label: str
    witness: dict[str, float | int | str]


class FrameVisibility(SpecModel):
    """Per-frame target visibility as measured by the compile backend."""

    frame: int  # index in the (possibly reindexed) sequence
    source_frame: int  # index into the originally rendered frames
    value: float  # render backend: mask pixel count
    tristate: Tristate


class GeometryComparison(SpecModel):
    """Search-phase estimates carried along strictly for auditing.

    Nothing here feeds the answer. ``frame_vars`` are the geometry backend's
    resolutions; ``sector``/``azimuth_deg`` are the provisional answer.
    """

    standard_version: str | None = None
    frame_vars: dict[str, int] = {}
    sector: str | None = None
    azimuth_deg: float | None = None


class Certificate(SpecModel):
    """Complete, replayable record of one authoritative compilation."""

    schema_version: Literal["scriptgen_certificate.v2"] = "scriptgen_certificate.v2"
    capability: str
    standard_version: str
    backend: str  # visibility kind of the compile backend, e.g. render_pixels
    binding: dict[str, str]
    frame_sequence: tuple[int, ...]  # source frame shown at each index
    status: CompileStatus
    reason: str | None  # failed clause / unresolvable frame var, if any
    frame_vars: dict[str, int]  # authoritative (compile-backend) resolutions
    knob_levels: dict[str, float]
    clause_outcomes: tuple[ClauseOutcome, ...]
    answer: AuthoritativeAnswer | None
    target_visibility: tuple[FrameVisibility, ...]
    essential_frames: tuple[int, ...] | None  # leave-one-out, canonical only
    geometry: GeometryComparison | None  # comparison fields, never authority
    mismatch: str | None  # set when render and geometry authorities disagree


@dataclass(frozen=True)
class CapabilityCompiler:
    """Recompiles one capability's script against any SceneView."""

    script: ScriptSpec
    std: CompileStandard

    def compile(
        self,
        view: SceneView,
        binding: dict[str, str],
        *,
        frame_sequence: tuple[int, ...] | None = None,
        geometry_plan: dict[str, Any] | None = None,
        with_essential: bool = False,
    ) -> Certificate:
        """Authoritatively judge one frame sequence and derive its gold.

        ``frame_sequence`` selects/reorders/repeats source frames (None means
        the identity sequence — the canonical rendering). ``geometry_plan`` is
        the search-phase plan record dict; its estimates are recorded for
        comparison and cross-checked into ``mismatch``.
        """
        if frame_sequence is None:
            frame_sequence = tuple(range(view.frame_count))
            judged: SceneView = view
        else:
            judged = ReindexedSceneView(base=view, frames=frame_sequence)

        target = binding.get("target", next(iter(binding.values())))
        geometry = _geometry_comparison(geometry_plan)
        base = dict(
            capability=self.script.capability,
            standard_version=self.std.standard_version,
            backend=judged.visibility(target, 0).kind,
            binding=dict(binding),
            frame_sequence=frame_sequence,
            target_visibility=self._target_visibility(judged, target, frame_sequence),
            geometry=geometry,
        )

        try:
            frame_vars = resolve_frame_vars(judged, self.script, dict(binding), self.std)
        except FrameVarUnresolvable as unresolvable:
            status: CompileStatus = (
                "abstain"
                if unresolvable.var_name in self.script.abstain_on_unresolvable
                else "invalid"
            )
            return Certificate(
                **base,
                status=status,
                reason=f"frame_var_unresolvable:{unresolvable.var_name}",
                frame_vars={},
                knob_levels={},
                clause_outcomes=(),
                answer=None,
                essential_frames=None,
                mismatch=_mismatch(self.std, geometry, status, None),
            )

        report = check_clauses(
            judged,
            self.script,
            binding,
            self.std,
            phases=("search", "compile"),
            tighten_search=False,
        )
        outcomes = tuple(
            ClauseOutcome(
                name=result.clause.name,
                predicate=result.clause.predicate,
                phase=result.clause.phase,
                on_violation=result.clause.on_violation,
                holds=result.verdict.holds,
                witness=result.verdict.witness,
            )
            for result in report.results
        )

        if not report.passed:
            failed = next(c for c in self.script.clauses if c.name == report.failed_clause)
            status = failed.on_violation
            return Certificate(
                **base,
                status=status,
                reason=f"clause:{failed.name}",
                frame_vars=frame_vars,
                knob_levels=_knob_levels(self.script, binding, frame_vars),
                clause_outcomes=outcomes,
                answer=None,
                essential_frames=None,
                mismatch=_mismatch(self.std, geometry, status, None),
            )

        answer = derive_answer(
            judged,
            self.std,
            self.script,
            binding,
            frame_vars,
        )
        essential = (
            self._essential_frames(view, binding, frame_sequence, answer)
            if with_essential
            else None
        )
        return Certificate(
            **base,
            status="answerable",
            reason=None,
            frame_vars=frame_vars,
            knob_levels=_knob_levels(self.script, binding, frame_vars),
            clause_outcomes=outcomes,
            answer=answer,
            essential_frames=essential,
            mismatch=_mismatch(self.std, geometry, "answerable", answer),
        )

    def _target_visibility(
        self, judged: SceneView, target: str, frame_sequence: tuple[int, ...]
    ) -> tuple[FrameVisibility, ...]:
        rows = []
        for t in range(judged.frame_count):
            observation = judged.visibility(target, t)
            rows.append(
                FrameVisibility(
                    frame=t,
                    source_frame=frame_sequence[t],
                    value=round(observation.value, 4),
                    tristate=_TRISTATE_NAMES[observation.tristate(self.std)],
                )
            )
        return tuple(rows)

    def _essential_frames(
        self,
        view: SceneView,
        binding: dict[str, str],
        frame_sequence: tuple[int, ...],
        answer: AuthoritativeAnswer,
    ) -> tuple[int, ...]:
        """Leave-one-out essentiality over the compiled sequence.

        A frame is essential iff removing it (alone) makes the sequence
        uncompilable or changes the answer sector. Individually redundant
        evidence (e.g. several sighting frames) is correctly NOT essential;
        interventions that must destroy evidence therefore remove whole
        classes of frames, not single ones.
        """
        essential: list[int] = []
        for drop_at in range(len(frame_sequence)):
            reduced = frame_sequence[:drop_at] + frame_sequence[drop_at + 1 :]
            reduced_cert = self.compile(view, binding, frame_sequence=reduced)
            if reduced_cert.status != "answerable" or (
                reduced_cert.answer is not None
                and reduced_cert.answer.label != answer.label
            ):
                essential.append(frame_sequence[drop_at])
        return tuple(essential)


def derive_answer(
    view: SceneView,
    std: CompileStandard,
    script: ScriptSpec,
    binding: dict[str, str],
    frame_vars: dict[str, int],
) -> AuthoritativeAnswer:
    """Resolve and dispatch the spec's declared answer algorithm."""
    env: dict[str, Any] = {**binding, **frame_vars}
    kwargs = resolve_args(script.answer.args, env, view.frame_count)
    result = get_answer_mode(script.answer.mode)(view, std, **kwargs)
    if result.label not in script.templates[0].options:
        raise ValueError(
            f"answer mode {script.answer.mode!r} produced label {result.label!r} "
            f"outside question options {script.templates[0].options!r}"
        )
    return AuthoritativeAnswer(
        mode=script.answer.mode,
        label=result.label,
        witness=result.witness,
    )


def _knob_levels(
    script: ScriptSpec, binding: dict[str, str], frame_vars: dict[str, int]
) -> dict[str, float]:
    from .checker import eval_int_expr

    env: dict[str, Any] = {**binding, **frame_vars}
    return {knob.name: float(eval_int_expr(knob.expr, env)) for knob in script.knobs}


def _geometry_comparison(plan: dict[str, Any] | None) -> GeometryComparison | None:
    if plan is None:
        return None
    provisional = plan.get("provisional_answer", {})
    return GeometryComparison(
        standard_version=plan.get("standard_version"),
        frame_vars=dict(plan.get("frame_vars", {})),
        sector=provisional.get("sector"),
        azimuth_deg=provisional.get("azimuth_deg"),
    )


def _mismatch(
    std: CompileStandard,
    geometry: GeometryComparison | None,
    status: CompileStatus,
    answer: AuthoritativeAnswer | None,
) -> str | None:
    """Cross-check the render authority against the geometry estimate.

    Only run when a geometry plan was supplied, i.e. for canonical bundles:
    the plan promised an answerable question with a specific sector, so a
    disagreement means one backend is wrong and the bundle must be blocked,
    never silently trusted either way.
    """
    if geometry is None:
        return None
    compatible_plan_versions = COMPATIBLE_PLAN_STANDARDS.get(std.standard_version, ())
    if (
        geometry.standard_version is not None
        and geometry.standard_version != std.standard_version
        and geometry.standard_version not in compatible_plan_versions
    ):
        return (
            "standard_version_drift:"
            f"plan={geometry.standard_version},compile={std.standard_version}"
        )
    if status != "answerable":
        return f"canonical_not_answerable:{status}"
    if geometry.sector is not None and answer is not None and answer.label != geometry.sector:
        return f"sector_disagreement:render={answer.label},geometry={geometry.sector}"
    return None
