"""Declarative script contracts.

A :class:`ScriptSpec` is the ONLY thing an author writes to add a capability:
slots (what objects the scene must offer), clauses (what a qualifying
trajectory must satisfy, as predicate references), frame variables (how the
key frame indices are derived), knobs (generalisation-axis difficulty levels)
and question templates. All fields are plain data — no callables — so specs
serialise, diff and version like any other contract.

Expression language (kept deliberately tiny):

* slot references: ``$target`` — resolved from the slot binding;
* frame variables: ``last_visible($target)``, ``last_frame()`` — resolved by
  the checker's resolver registry;
* integer arithmetic with ``+``/``-`` over resolved variables, e.g.
  ``$t_seen+1``;
* frame ranges: ``"<expr>:<expr>"`` — INCLUSIVE on both ends.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class SlotSpec(SpecModel):
    """Requirements a scene object must meet to fill a slot."""

    min_size_m: float = Field(gt=0.0, default=0.3)
    categories: tuple[str, ...] = ()  # empty means any category
    unique_referent: bool = True  # question text must identify it unambiguously


class Clause(SpecModel):
    """One requirement on a qualifying trajectory.

    ``phase`` declares which backend must confirm it: ``search`` clauses are
    decided render-free; ``compile`` clauses are re-checked authoritatively on
    the rendered bundle (they are still *estimated* during search so hopeless
    candidates are dropped early).

    ``on_violation`` declares what a violation MEANS when the compiler
    re-judges an intervened frame sequence: ``abstain`` marks an evidence
    clause (violated -> even an ideal agent cannot know the answer, so the
    gold becomes the abstain option); ``invalid`` marks a question-validity
    clause (violated -> the question itself leaves its design envelope and no
    gold of any kind may be asserted).
    """

    name: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    args: dict[str, str | int | float | bool]
    phase: Literal["search", "compile", "search_only"] = "compile"
    on_violation: Literal["abstain", "invalid"] = "invalid"


class Knob(SpecModel):
    """A generalisation axis: an expression and its difficulty levels."""

    name: str = Field(min_length=1)
    expr: str = Field(min_length=1)
    levels: tuple[float, ...] = Field(min_length=1)


class Template(SpecModel):
    """Question surface form with closed answer options.

    ``{slot}`` and ``{frame_var}`` placeholders are filled at compile time.
    The abstain option is mandatory: abstention must always be expressible.
    """

    text: str = Field(min_length=1)
    options: tuple[str, ...] = Field(min_length=2)
    abstain_option: str = "无法判断"

    @model_validator(mode="after")
    def abstain_present(self) -> Template:
        if self.abstain_option not in self.options:
            raise ValueError("options must include the abstain option")
        return self


VariantExpectation = Literal["same", "abstain"]


class ScriptSpec(SpecModel):
    """Complete declarative definition of one capability's trajectory needs.

    v2 adds the family contract: ``abstain_on_unresolvable`` names the frame
    variables whose failure to resolve means the evidence is gone (gold =
    abstain) rather than the question being malformed, and
    ``variant_expectations`` declares, per intervention kind, what the
    recompiled gold MUST come out as — the compiler decides the actual gold;
    the expectation only cross-checks it, and disagreement blocks packaging.
    """

    schema_version: Literal["scriptgen_spec.v3"] = "scriptgen_spec.v3"
    capability: str = Field(min_length=1)
    slots: dict[str, SlotSpec]
    frame_vars: dict[str, str]  # name -> resolver expression
    clauses: tuple[Clause, ...] = Field(min_length=1)
    knobs: tuple[Knob, ...] = ()
    length: tuple[int, int]  # inclusive frame-count range
    motifs: tuple[str, ...] = Field(min_length=1)
    templates: tuple[Template, ...] = Field(min_length=1)
    abstain_on_unresolvable: tuple[str, ...] = ()
    variant_expectations: dict[str, VariantExpectation] = {}

    @model_validator(mode="after")
    def length_ordered(self) -> ScriptSpec:
        low, high = self.length
        if low < 2 or high < low:
            raise ValueError("length must satisfy 2 <= low <= high")
        return self

    @model_validator(mode="after")
    def clause_names_unique(self) -> ScriptSpec:
        names = [clause.name for clause in self.clauses]
        if len(names) != len(set(names)):
            raise ValueError("clause names must be unique")
        return self
