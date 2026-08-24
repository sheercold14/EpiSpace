"""Label balancing over the questions a rendered collection compiled.

Three cross-view questions are compiled on one trajectory and sixteen
reference-frame questions on one survey, so which questions ship is a choice
made after rendering and costs nothing to change.  That choice matters:
several strata are structurally skewed, and a model can score well above
chance on them by reading the question text alone.

The unit of balance is the capability.  A capability name already carries the
two things that decide a label - the relay depth and question type for the
cross-view line (``cross_view_snapshot_anchor_k2``), the answer mode and
imagined yaw for the reference line (``reference_frame_visibility_yaw90``) -
so balancing inside a capability is the ``(k, question type)`` stratification
this collection settled on, and balancing every capability to the same shape
is what removes the text-only shortcut across capabilities.

A question is identified by the episode that rendered it and the capability it
compiles, which is the same pair the QA builder's ``--exclusions`` file takes,
so the report can be handed to the training-set build directly instead of
being a summary somebody has to remember to act on.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

QUESTION_BALANCE_SCHEMA_VERSION = "scriptgen_question_balance.v1"

# A stratum whose questions all share one answer teaches the prior and nothing
# else, so it ships no questions at all rather than a downsampled remnant.
MINIMUM_DISTINCT_LABELS = 2


@dataclass(frozen=True)
class QuestionKey:
    """Identifies one compiled question inside one rendered episode.

    An episode compiles at most one question per capability - three for a
    cross-view trajectory, sixteen for a reference survey - so the pair is
    unique, and it is the pair the QA builder excludes on.
    """

    episode_id: str
    capability: str

    def as_json(self) -> dict[str, str]:
        return {"episode_id": self.episode_id, "capability": self.capability}


def _labelled_questions(
    groups: dict[str, dict[str, Any]],
) -> dict[str, dict[str, list[QuestionKey]]]:
    """Group answerable questions by capability and then by answer label."""
    by_capability: dict[str, dict[str, list[QuestionKey]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for episode_id, group in groups.items():
        for question in group["questions"]:
            # A skipped question has no authoritative label to balance: the
            # compiler either abstained or ruled the instance invalid.
            if question.get("skip_reason") is not None:
                continue
            label = question.get("label")
            if label is None:
                continue
            key = QuestionKey(episode_id, question["capability"])
            by_capability[question["capability"]][label].append(key)
    return by_capability


def balance_question_labels(groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Select an equal number of questions per label within each capability.

    Returns the retained keys, the dropped keys, and a per-capability account,
    so a collapsed stratum is visible in the report rather than silently absent
    from the dataset.
    """
    by_capability = _labelled_questions(groups)
    retained: list[QuestionKey] = []
    dropped: list[QuestionKey] = []
    strata: list[dict[str, Any]] = []
    for capability in sorted(by_capability):
        by_label = by_capability[capability]
        counts = {label: len(keys) for label, keys in by_label.items()}
        total = sum(counts.values())
        if len(counts) < MINIMUM_DISTINCT_LABELS:
            dropped.extend(key for keys in by_label.values() for key in keys)
            strata.append(
                {
                    "capability": capability,
                    "collapsed": True,
                    "reason": f"only {len(counts)} distinct label(s)",
                    "label_counts": counts,
                    "retained_per_label": 0,
                    "retained": 0,
                    "dropped": total,
                }
            )
            continue
        per_label = min(counts.values())
        for label in sorted(by_label):
            keys = sorted(by_label[label], key=_ordering)
            retained.extend(keys[:per_label])
            dropped.extend(keys[per_label:])
        kept = per_label * len(counts)
        strata.append(
            {
                "capability": capability,
                "collapsed": False,
                "reason": None,
                "label_counts": counts,
                "retained_per_label": per_label,
                "retained": kept,
                "dropped": total - kept,
                # After balancing every present label is equally likely, so
                # this is the accuracy a constant answer would score.
                "chance_accuracy": round(1.0 / len(counts), 4),
            }
        )
    return {
        "schema_version": QUESTION_BALANCE_SCHEMA_VERSION,
        "stratum": "capability",
        "question_count": sum(
            len(keys) for by_label in by_capability.values() for keys in by_label.values()
        ),
        "retained_count": len(retained),
        "dropped_count": len(dropped),
        "collapsed_capabilities": tuple(
            row["capability"] for row in strata if row["collapsed"]
        ),
        "strata": tuple(strata),
        "retained": tuple(key.as_json() for key in sorted(retained, key=_ordering)),
        "dropped": tuple(key.as_json() for key in sorted(dropped, key=_ordering)),
    }


def exclusion_lines(report: dict[str, Any]) -> str:
    """Render the dropped questions as the QA builder's ``--exclusions`` file.

    The builder excludes by ``(episode_id, capability)``, which is what the
    report already stores, so balancing takes effect by being passed to the
    build rather than by being read by a person.
    """
    return "".join(
        json.dumps(entry, sort_keys=True) + "\n" for entry in report["dropped"]
    )


def _ordering(key: QuestionKey) -> tuple[str, str]:
    return (key.episode_id, key.capability)


def load_groups(group_paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """Read each episode's question group, keyed by the episode that rendered it."""
    return {
        episode_id: json.loads(path.read_text(encoding="utf-8"))
        for episode_id, path in group_paths.items()
    }
