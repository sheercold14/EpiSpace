"""Intervention operators: rendered frames -> controlled variant certificates.

Every operator produces nothing but a frame index sequence over the already
rendered frames; the variant's gold answer is then produced by re-running the
SAME :class:`CapabilityCompiler` on that sequence. Answers are never assigned
by hand, and the spec's declared expectation (``variant_expectations``) is
only a cross-check: if the recompiled outcome disagrees with it, the family
is blocked with a :class:`FamilyMismatch` instead of shipping a wrong gold.

The four v1 operators:

* ``permute``   — shuffle the frames between t_gone and t_q. For self-motion
  the order IS the evidence (ego-motion must be integrable step by step), so
  the compiler is expected to find the motion untrackable -> gold = abstain.
* ``drop_key``  — remove every frame in which the target is not definitely
  invisible (all sightings and all ambiguous slivers). t_seen becomes
  unresolvable -> gold = abstain.
* ``drop_filler`` — remove frames the leave-one-out analysis marked
  non-essential, re-verifying after each removal -> gold unchanged.
* ``delay``     — repeat one mid-gone frame (a standstill pause: zero motion
  per repeated step, trivially trackable) -> the delay knob grows, the gold
  is unchanged.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from .checker import eval_frame_range
from .compiler import CapabilityCompiler, Certificate
from .sceneview import SceneView
from .spec import SpecModel

VariantKind = Literal["canonical", "permute", "drop_key", "drop_filler", "delay"]
INTERVENTION_KINDS: tuple[str, ...] = ("permute", "drop_key", "drop_filler", "delay")


class Variant(SpecModel):
    """One family member: an index sequence plus its recompiled certificate."""

    variant_id: str
    kind: VariantKind
    expectation: Literal["canonical", "same", "abstain"]
    frame_sequence: tuple[int, ...]
    gold: str  # answer option string, e.g. "left" or the abstain option
    certificate: Certificate


class FamilyMismatch(RuntimeError):
    """Recompiled outcome disagrees with the declared expectation: block."""

    def __init__(self, kind: str, expected: str, certificate: Certificate) -> None:
        actual = certificate.answer.label if certificate.answer else certificate.status
        super().__init__(
            f"variant {kind}: expected {expected}, compiler produced "
            f"status={certificate.status} answer={actual} reason={certificate.reason}"
        )
        self.kind = kind
        self.expected = expected
        self.certificate = certificate


@dataclass(frozen=True)
class VariantBuilder:
    """Builds the intervention variants declared by one script."""

    compiler: CapabilityCompiler
    view: SceneView
    binding: dict[str, str]
    canonical: Certificate

    def __post_init__(self) -> None:
        if self.canonical.status != "answerable" or self.canonical.answer is None:
            raise ValueError("variants require an answerable canonical certificate")
        if self.canonical.essential_frames is None:
            raise ValueError("canonical certificate must carry essential_frames")

    def build_all(
        self,
        *,
        seed: int,
        drop_count: int = 2,
        delay_extra: int = 4,
        max_tries: int = 8,
    ) -> tuple[Variant, ...]:
        rng = random.Random(seed)
        builders = {
            "permute": lambda: self._permute_sequences(rng, max_tries),
            "drop_key": lambda: [self._drop_key_sequence()],
            "drop_filler": lambda: [self._drop_filler_sequence(drop_count)],
            "delay": lambda: [self._delay_sequence(delay_extra)],
        }
        return tuple(
            self._verified(kind, builders[kind]())
            for kind in INTERVENTION_KINDS
            if kind in self.compiler.script.variant_expectations
        )

    # --- operators (index sequences only; no judgment happens here) ---

    def _permute_sequences(self, rng: random.Random, max_tries: int) -> list[tuple[int, ...]]:
        """Candidate shuffles of the spec-declared intervention window."""
        positions = self._intervention_positions()
        original = list(self.canonical.frame_sequence)
        candidates: list[tuple[int, ...]] = []
        for _ in range(max_tries):
            shuffled = [original[position] for position in positions]
            rng.shuffle(shuffled)
            if shuffled == [original[position] for position in positions]:
                continue
            candidate = original[:]
            for position, source_frame in zip(positions, shuffled, strict=True):
                candidate[position] = source_frame
            candidates.append(tuple(candidate))
        return candidates

    def _drop_key_sequence(self) -> tuple[int, ...]:
        """Keep only frames where the target is DEFINITELY invisible."""
        kept = [
            row.source_frame
            for row in self.canonical.target_visibility
            if row.tristate == "invisible"
        ]
        return tuple(kept)

    def _drop_filler_sequence(self, drop_count: int) -> tuple[int, ...]:
        """Greedily remove non-essential frames, re-verifying every removal.

        Every resolved frame variable is an anchor and stays materialised even
        if it is individually redundant.
        """
        assert self.canonical.essential_frames is not None
        anchors = set(self.canonical.frame_vars.values())
        protected = set(self.canonical.essential_frames) | anchors
        sequence = list(self.canonical.frame_sequence)
        dropped = 0
        for frame in list(sequence):
            if dropped >= drop_count or frame in protected:
                continue
            reduced = tuple(t for t in sequence if t != frame)
            cert = self.compiler.compile(self.view, self.binding, frame_sequence=reduced)
            if cert.status == "answerable" and (
                cert.answer is not None and cert.answer.label == self.canonical.answer.label  # type: ignore[union-attr]
            ):
                sequence = list(reduced)
                dropped += 1
        if dropped == 0:
            raise FamilyMismatch("drop_filler", "same", self.canonical)
        return tuple(sequence)

    def _delay_sequence(self, delay_extra: int) -> tuple[int, ...]:
        """Pause at the midpoint of the spec-declared intervention window."""
        positions = self._intervention_positions()
        pause_position = positions[len(positions) // 2]
        sequence = list(self.canonical.frame_sequence)
        pause_at = sequence[pause_position]
        return tuple(
            sequence[: pause_position + 1]
            + [pause_at] * delay_extra
            + sequence[pause_position + 1 :]
        )

    def _intervention_positions(self) -> list[int]:
        positions = eval_frame_range(
            self.compiler.script.intervention_window,
            self.canonical.frame_vars,
            len(self.canonical.frame_sequence),
        )
        if not positions:
            raise ValueError(
                f"empty intervention_window for {self.compiler.script.capability}: "
                f"{self.compiler.script.intervention_window!r}"
            )
        return positions

    # --- verification: recompile and cross-check the declared expectation ---

    def _verified(self, kind: str, candidates: list[tuple[int, ...]]) -> Variant:
        expectation = self.compiler.script.variant_expectations.get(kind)
        if expectation is None:
            raise KeyError(
                f"script {self.compiler.script.capability} declares no expectation "
                f"for variant kind {kind!r}"
            )
        last_cert: Certificate | None = None
        for sequence in candidates:
            cert = self.compiler.compile(self.view, self.binding, frame_sequence=sequence)
            last_cert = cert
            gold = self._gold_if_expected(expectation, cert)
            if gold is not None:
                return Variant(
                    variant_id=kind,
                    kind=kind,  # type: ignore[arg-type]
                    expectation=expectation,
                    frame_sequence=sequence,
                    gold=gold,
                    certificate=cert,
                )
        assert last_cert is not None, f"variant {kind}: no candidate sequences"
        raise FamilyMismatch(kind, expectation, last_cert)

    def _gold_if_expected(self, expectation: str, cert: Certificate) -> str | None:
        """The variant's gold answer, or None if it defies the expectation."""
        canonical_label = self.canonical.answer.label  # type: ignore[union-attr]
        if expectation == "same":
            if cert.status == "answerable" and cert.answer is not None:
                if cert.answer.label == canonical_label:
                    return cert.answer.label
            return None
        if expectation == "abstain":
            if cert.status == "abstain":
                return self.compiler.template.abstain_option
            return None
        raise ValueError(f"unknown expectation: {expectation!r}")
