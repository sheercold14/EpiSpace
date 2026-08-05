"""Clause checker: evaluates a ScriptSpec against one candidate trajectory.

The checker owns the tiny expression language declared in ``spec``:

* ``$name`` substitution from the environment (slot binding + frame vars),
* integer ``+``/``-`` arithmetic,
* inclusive ``start:end`` frame ranges,
* frame-variable resolvers such as ``last_visible($target)``.

It is deliberately dumb: no backtracking, no scoring. A candidate either
resolves and satisfies every clause of the requested phase, or it is rejected
with the first failing clause and its witness attached.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

from .predicates import Verdict, get_predicate
from .sceneview import SceneView
from .spec import Clause, ScriptSpec
from .standards import CompileStandard

_RESOLVER_PATTERN = re.compile(r"^(?P<fn>[a-z_]+)\((?P<arg>[^)]*)\)$")


class ScriptError(ValueError):
    """A spec references something the engine cannot resolve."""


def _substitute(expr: str, env: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in env:
            raise ScriptError(f"unresolved variable ${name}; environment: {sorted(env)}")
        return str(env[name])

    return re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", replace, expr)


def eval_int_expr(expr: str, env: dict[str, Any]) -> int:
    """Evaluate an integer expression with ``+``/``-`` only."""
    substituted = _substitute(expr, env)
    try:
        node = ast.parse(substituted, mode="eval").body
    except SyntaxError as error:
        raise ScriptError(f"bad expression: {expr!r}") from error
    return _eval_node(node, expr)


def _eval_node(node: ast.expr, source: str) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Sub):
        left = _eval_node(node.left, source)
        right = _eval_node(node.right, source)
        return left + right if isinstance(node.op, ast.Add) else left - right
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand, source)
    raise ScriptError(f"expression {source!r} uses unsupported syntax")


def eval_frame_range(expr: str, env: dict[str, Any], frame_count: int) -> list[int]:
    """Parse an inclusive ``start:end`` range, clamped to valid frames."""
    if ":" not in expr:
        raise ScriptError(f"frame range {expr!r} must contain ':'")
    start_expr, end_expr = expr.split(":", maxsplit=1)
    start = max(0, eval_int_expr(start_expr, env))
    end = min(frame_count - 1, eval_int_expr(end_expr, env))
    return list(range(start, end + 1))


def resolve_frame_vars(
    view: SceneView, script: ScriptSpec, env: dict[str, Any], std: CompileStandard
) -> dict[str, int]:
    """Evaluate the spec's frame-variable resolver expressions in order."""
    resolved: dict[str, int] = {}
    scope = dict(env)
    for name, expr in script.frame_vars.items():
        match = _RESOLVER_PATTERN.match(expr.strip())
        if match is None:
            raise ScriptError(f"frame var {name}: bad resolver expression {expr!r}")
        fn, raw_arg = match.group("fn"), match.group("arg").strip()
        arg = _substitute(raw_arg, scope) if raw_arg else ""
        value = _run_resolver(fn, arg, view, std, expr)
        if value is None:
            raise FrameVarUnresolvable(name, expr)
        resolved[name] = value
        scope[name] = value
    return resolved


class FrameVarUnresolvable(ScriptError):
    """The trajectory offers no frame satisfying the resolver (a soft reject)."""

    def __init__(self, name: str, expr: str) -> None:
        super().__init__(f"frame var {name} unresolvable via {expr!r}")
        self.var_name = name


def _run_resolver(
    fn: str, arg: str, view: SceneView, std: CompileStandard, expr: str
) -> int | None:
    if fn == "last_frame":
        return view.frame_count - 1
    if fn in {"last_visible", "first_visible"}:
        frames = range(view.frame_count)
        ordered = reversed(frames) if fn == "last_visible" else frames
        for t in ordered:
            if view.visibility(arg, t).tristate(std) is True:
                return t
        return None
    raise ScriptError(f"unknown frame-var resolver in {expr!r}")


@dataclass(frozen=True)
class ClauseResult:
    clause: Clause
    verdict: Verdict


@dataclass(frozen=True)
class ScriptReport:
    """Outcome of checking one candidate against one script phase."""

    passed: bool
    frame_vars: dict[str, int]
    results: tuple[ClauseResult, ...]
    failed_clause: str | None = None

    def witnesses(self) -> dict[str, dict[str, Any]]:
        return {result.clause.name: result.verdict.witness for result in self.results}


def check_clauses(
    view: SceneView,
    script: ScriptSpec,
    binding: dict[str, str],
    std: CompileStandard,
    *,
    phases: tuple[str, ...] = ("search", "compile"),
    tighten_search: bool = True,
) -> ScriptReport:
    """Check every clause of the requested phases against a candidate.

    During search, ``compile`` clauses are still evaluated (with the geometry
    backend) so hopeless candidates die early; the authoritative compile-phase
    run re-checks them on the render backend with ``tighten_search=False``.
    """
    env: dict[str, Any] = dict(binding)
    try:
        frame_vars = resolve_frame_vars(view, script, env, std)
    except FrameVarUnresolvable as unresolvable:
        return ScriptReport(
            passed=False, frame_vars={}, results=(), failed_clause=unresolvable.var_name
        )
    env.update(frame_vars)

    results: list[ClauseResult] = []
    for clause in script.clauses:
        if clause.phase not in phases:
            continue
        kwargs = _resolve_clause_args(clause, env, view.frame_count)
        if tighten_search and clause.predicate == "sector_margin_ge":
            kwargs.setdefault("tighten", clause.phase == "search")
        verdict = get_predicate(clause.predicate)(view, std, **kwargs)
        results.append(ClauseResult(clause, verdict))
        if verdict.holds is not True:
            return ScriptReport(
                passed=False,
                frame_vars=frame_vars,
                results=tuple(results),
                failed_clause=clause.name,
            )
    return ScriptReport(passed=True, frame_vars=frame_vars, results=tuple(results))


def _resolve_clause_args(clause: Clause, env: dict[str, Any], frame_count: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for key, raw in clause.args.items():
        if not isinstance(raw, str):
            kwargs[key] = raw
        elif ":" in raw:
            kwargs[key] = eval_frame_range(raw, env, frame_count)
        elif raw.startswith("$") or raw.lstrip("+-").isdigit() or "+" in raw or "-" in raw:
            resolved = _substitute(raw, env)
            kwargs[key] = int(resolved) if _is_int(resolved) else resolved
        else:
            kwargs[key] = raw
    return kwargs


def _is_int(text: str) -> bool:
    try:
        int(text)
    except ValueError:
        return False
    return True
