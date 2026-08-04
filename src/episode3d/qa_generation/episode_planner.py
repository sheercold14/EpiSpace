"""Compose geometry-backed facts into observe-once, read-many batches.

The production compiler intentionally keeps facts and counterfactual siblings
separate.  This module is the *curriculum* layer: it chooses a diverse set of
safe reads that can share one ordered observation prefix.  It never changes a
fact's answer, program, or certificate.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from episode3d.qa_generation.capabilities import capability_for
from episode3d.qa_generation.schemas import EpisodeBatchPlan, PlannedQuestion, content_id


class EpisodePlanningError(ValueError):
    """Raised when a requested batch would violate the episode contract."""


# These queries name their reference view(s) explicitly after naturalization,
# so additional earlier/later observations cannot change the executable fact.
# Evidence-intervention siblings are deliberately absent: their treatment is
# the exact image subset and must never be widened to the complete trajectory.
SAFE_CONTEXT_LIFT_TASKS = frozenset(
    {
        "metric_distance",
        "egocentric_relation",
        "cross_view_relation",
        "counterfactual_verification",
        "last_seen_memory",
        "orbit_identity",
        "rotation_change_detection",
        "elevation_relation_transfer",
        "object_centric_perspective",
        "target_view_prediction",
    }
)

EXACT_EVIDENCE_TASKS = frozenset(
    {
        "grounding_presence",
        "unknown_abstention",
        "evidence_presence_unknown",
        "evidence_presence_reveal",
        "occlusion_unknown",
        "occlusion_reveal",
    }
)

# High-information reads are considered before diagnostic presence questions.
TASK_ORDER = {
    "object_centric_perspective": 0,
    "cross_view_relation": 1,
    "elevation_relation_transfer": 2,
    "orbit_identity": 3,
    "rotation_change_detection": 4,
    "last_seen_memory": 5,
    "metric_distance": 6,
    "egocentric_relation": 7,
    "counterfactual_verification": 8,
    "target_view_prediction": 9,
    "unknown_abstention": 20,
    "grounding_presence": 21,
}


def read_episode_ir(path: Path) -> list[dict[str, Any]]:
    """Read and minimally validate the immutable Episode IR."""

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != "epispace.episode_ir.v1":
                raise EpisodePlanningError(f"line {line_number}: unsupported Episode IR schema")
            if not row.get("episode_id") or not isinstance(row.get("questions"), list):
                raise EpisodePlanningError(f"line {line_number}: malformed Episode IR row")
            rows.append(row)
    if not rows:
        raise EpisodePlanningError(f"no Episode IR rows in {path}")
    return rows


def semantic_audit_exclusions(path: Path | None) -> tuple[set[str], dict[str, str]]:
    """Return fail-closed fact exclusions and all sampled verdicts.

    Major and unreviewable facts are excluded.  If one belongs to a consistency
    family, the caller expands the exclusion to the complete family so a broken
    sibling can never survive alone.
    """

    if path is None:
        return set(), {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "epispace.semantic_visual_audit_result.v1":
        raise EpisodePlanningError("unsupported semantic visual audit schema")
    verdicts: dict[str, str] = {}
    excluded: set[str] = set()
    for item in payload.get("item_decisions", []):
        fact_id = str(item["fact_id"])
        status = str(item["overall_status"])
        verdicts[fact_id] = status
        if status in {"major_issue", "unreviewable"}:
            excluded.add(fact_id)
    return excluded, verdicts


def _expand_family_exclusions(
    episodes: Sequence[Mapping[str, Any]], excluded_fact_ids: set[str]
) -> set[str]:
    bad_groups = {
        str(question["consistency_group"])
        for episode in episodes
        for question in episode["questions"]
        if question.get("fact_id") in excluded_fact_ids and question.get("consistency_group")
    }
    return excluded_fact_ids | {
        str(question["fact_id"])
        for episode in episodes
        for question in episode["questions"]
        if question.get("consistency_group") in bad_groups
    }


def _capability_set(questions: Iterable[Mapping[str, Any]]) -> set[str]:
    result: set[str] = set()
    for question in questions:
        capability = capability_for(str(question["task_type"]))
        result.update(value for value in (capability.primary, *capability.supporting) if value != "AUX")
    return result


def _question_score(question: Mapping[str, Any]) -> tuple[int, str]:
    return (TASK_ORDER.get(str(question["task_type"]), 100), str(question["fact_id"]))


@dataclass(frozen=True)
class PlannerContract:
    min_questions: int = 4
    max_questions: int = 6
    min_capabilities: int = 3
    max_auxiliary_questions: int = 1
    require_pt_backbone: bool = True


class EpisodeBatchPlanner:
    """Build diverse shared-prefix batches without modifying executable truth."""

    def __init__(
        self,
        *,
        heldout_program_ids: Iterable[str] = (),
        contract: PlannerContract | None = None,
    ) -> None:
        self.heldout_program_ids = frozenset(heldout_program_ids)
        self.contract = contract or PlannerContract()

    def _eligible(
        self,
        episode: Mapping[str, Any],
        *,
        excluded_fact_ids: set[str],
        include_eval_only: bool,
        deduplicate: bool,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen_task: set[str] = set()
        seen_consistency_group: set[str] = set()
        for raw in sorted(episode["questions"], key=_question_score):
            question = dict(raw)
            fact_id = str(question["fact_id"])
            task_type = str(question["task_type"])
            program_id = str(question["program"]["program_id"])
            variant = str(question.get("family_variant", "canonical"))
            group = str(question.get("consistency_group", ""))
            if fact_id in excluded_fact_ids:
                continue
            if program_id in self.heldout_program_ids and not include_eval_only:
                continue
            full_views = tuple(str(item["view_id"]) for item in episode["observations"])
            source_views = tuple(str(value) for value in question.get("model_view_ids", ()))
            exact_full_context = (
                task_type == "unknown_abstention"
                and source_views == full_views
                and variant == "canonical"
            )
            if task_type not in SAFE_CONTEXT_LIFT_TASKS and not exact_full_context:
                continue
            # A batch needs complementary reads, not several paraphrases of the
            # same operation.  Counterfactual siblings remain separate eval
            # families; the canonical multi-read batch keeps at most one.
            if deduplicate and task_type in seen_task:
                continue
            if deduplicate and group and group in seen_consistency_group:
                continue
            if deduplicate and variant in {"claim_true", "frame_b"}:
                continue
            seen_task.add(task_type)
            if group:
                seen_consistency_group.add(group)
            result.append(question)
        return result

    def _select_questions(self, candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        aux_count = 0
        for question in candidates:
            capability = capability_for(str(question["task_type"]))
            if capability.primary == "AUX":
                if aux_count >= self.contract.max_auxiliary_questions:
                    continue
                aux_count += 1
            selected.append(question)
            if len(selected) >= self.contract.max_questions:
                break
        return selected

    def _validate_batch(self, questions: Sequence[Mapping[str, Any]]) -> None:
        if not self.contract.min_questions <= len(questions) <= self.contract.max_questions:
            raise EpisodePlanningError(
                f"batch has {len(questions)} questions; expected "
                f"{self.contract.min_questions}..{self.contract.max_questions}"
            )
        capabilities = _capability_set(questions)
        if len(capabilities) < self.contract.min_capabilities:
            raise EpisodePlanningError(
                f"batch covers only {sorted(capabilities)}; need "
                f"{self.contract.min_capabilities} SenseNova capabilities"
            )
        if self.contract.require_pt_backbone and "PT" not in capabilities:
            raise EpisodePlanningError("batch lacks a PT write/transform backbone")
        aux_count = sum(
            capability_for(str(question["task_type"])).primary == "AUX"
            for question in questions
        )
        if aux_count > self.contract.max_auxiliary_questions:
            raise EpisodePlanningError("batch contains too many auxiliary diagnostics")

    def _to_plan(
        self,
        episode: Mapping[str, Any],
        questions: Sequence[Mapping[str, Any]],
        *,
        audit_verdicts: Mapping[str, str],
    ) -> EpisodeBatchPlan:
        self._validate_batch(questions)
        exposure_view_ids = tuple(str(item["view_id"]) for item in episode["observations"])
        planned: list[PlannedQuestion] = []
        for question in questions:
            source_views = tuple(
                str(value) for value in question.get("model_view_ids", exposure_view_ids)
            )
            planned.append(
                PlannedQuestion(
                    fact_id=str(question["fact_id"]),
                    task_type=str(question["task_type"]),
                    program_id=str(question["program"]["program_id"]),
                    capability=capability_for(str(question["task_type"])),
                    exposure_view_ids=exposure_view_ids,
                    source_model_view_ids=source_views,
                    expanded_to_episode_context=source_views != exposure_view_ids,
                )
            )
        warnings = [
            "development_only: source semantic RGB audit is not a formal release",
            "MR unsupported: current assets lack certified canonical object fronts",
        ]
        sampled = {
            str(question["fact_id"]): audit_verdicts[str(question["fact_id"])]
            for question in questions
            if str(question["fact_id"]) in audit_verdicts
        }
        if sampled:
            warnings.append(
                "sampled_semantic_verdicts="
                + ",".join(f"{key}:{value}" for key, value in sorted(sampled.items()))
            )
        identity = {
            "episode_id": episode["episode_id"],
            "facts": [question["fact_id"] for question in questions],
            "exposure": exposure_view_ids,
        }
        return EpisodeBatchPlan(
            batch_id=content_id("qa-batch", identity),
            episode_id=str(episode["episode_id"]),
            scene_id=str(episode["scene_id"]),
            split=str(episode["split"]),
            trajectory_class=str(episode["trajectory_class"]),
            source_bundle=str(episode["source_bundle"]),
            exposure_view_ids=exposure_view_ids,
            questions=tuple(planned),
            development_only=True,
            warnings=tuple(warnings),
        )

    def plan(
        self,
        episodes: Sequence[Mapping[str, Any]],
        *,
        target_question_count: int,
        semantic_audit_path: Path | None = None,
        split: str = "train",
        include_eval_only: bool = False,
        curated: Mapping[str, Sequence[str]] | None = None,
    ) -> list[EpisodeBatchPlan]:
        """Plan batches greedily or validate an explicit episode→fact selection."""

        excluded, verdicts = semantic_audit_exclusions(semantic_audit_path)
        excluded = _expand_family_exclusions(episodes, excluded)
        plans: list[EpisodeBatchPlan] = []
        total = 0
        if curated is None:
            ordered_episodes = sorted(episodes, key=lambda value: str(value["episode_id"]))
        else:
            by_episode_id = {str(value["episode_id"]): value for value in episodes}
            missing_episodes = [episode_id for episode_id in curated if episode_id not in by_episode_id]
            if missing_episodes:
                raise EpisodePlanningError(
                    f"curated selection references unknown episodes: {missing_episodes}"
                )
            ordered_episodes = [by_episode_id[episode_id] for episode_id in curated]
        for episode in ordered_episodes:
            episode_id = str(episode["episode_id"])
            if str(episode["split"]) != split:
                continue
            candidates = self._eligible(
                episode,
                excluded_fact_ids=excluded,
                include_eval_only=include_eval_only,
                deduplicate=curated is None,
            )
            if curated is not None:
                if episode_id not in curated:
                    continue
                by_fact = {str(question["fact_id"]): question for question in candidates}
                missing = [fact_id for fact_id in curated[episode_id] if fact_id not in by_fact]
                if missing:
                    raise EpisodePlanningError(
                        f"curated episode {episode_id} contains ineligible facts: {missing}"
                    )
                selected = [by_fact[fact_id] for fact_id in curated[episode_id]]
            else:
                selected = self._select_questions(candidates)
                try:
                    self._validate_batch(selected)
                except EpisodePlanningError:
                    continue
            plan = self._to_plan(episode, selected, audit_verdicts=verdicts)
            plans.append(plan)
            total += len(plan.questions)
            if curated is None and total >= target_question_count:
                break
        if total < target_question_count:
            raise EpisodePlanningError(
                f"planned only {total} questions; target is {target_question_count}"
            )
        return plans


def question_by_fact(episode: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(question["fact_id"]): dict(question) for question in episode["questions"]}


def episode_by_id(episodes: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(episode["episode_id"]): dict(episode) for episode in episodes}
