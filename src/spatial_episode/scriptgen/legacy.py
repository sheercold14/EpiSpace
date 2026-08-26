"""Frozen compiler inputs for explicitly replaying immutable legacy data.

Legacy scripts live outside :mod:`library` so new search and P2/P3 rendering
cannot select them accidentally.  They use today's validated ``ScriptSpec``
container, but preserve the clauses, thresholds, answer and intervention
semantics of the original release.
"""

from __future__ import annotations

from types import MappingProxyType

from .spec import (
    AnswerSpec,
    Clause,
    Knob,
    ScriptSpec,
    SlotSpec,
    Template,
)

# P1 was released under std.v3 with only the self-motion memory question.  In
# particular it predates question sharing and the later P2/P3 script families.
LEGACY_P1_SELF_MOTION = ScriptSpec(
    capability="self_motion_update",
    slots={"target": SlotSpec(min_size_m=0.5, unique_referent=True)},
    frame_vars={
        "t_seen": "last_visible($target)",
        "t_gone": "first_invisible_after($target, $t_seen)",
        "t_q": "last_frame()",
    },
    clauses=(
        Clause(
            name="seen_early",
            predicate="visible_somewhere",
            args={"obj": "$target", "frames": "0:$t_seen"},
            phase="compile",
            on_violation="abstain",
        ),
        Clause(
            name="trackable",
            predicate="step_motion_bounded",
            args={"frames": "0:$t_q"},
            phase="search",
            on_violation="abstain",
        ),
        Clause(
            name="gone",
            predicate="invisible_in_range",
            args={"obj": "$target", "frames": "$t_gone:$t_q"},
            phase="compile",
        ),
        Clause(
            name="gap",
            predicate="frame_gap_ge",
            args={"later": "$t_q", "earlier": "$t_gone", "gap": 3},
            phase="search",
        ),
        Clause(
            name="turned",
            predicate="cum_turn_between",
            args={"frames": "$t_seen:$t_q", "deg_min": 80, "deg_max": 200},
            phase="search",
        ),
        Clause(
            name="margin_ok",
            predicate="sector_margin_ge",
            args={"obj": "$target", "frame": "$t_q"},
            phase="search",
        ),
    ),
    answer=AnswerSpec(
        mode="target_sector",
        args={"obj": "$target", "frame": "$t_q"},
    ),
    knobs=(Knob(name="delay", expr="$t_q-$t_gone", levels=(3, 6, 10)),),
    length=(10, 16),
    motifs=("walk_and_turn",),
    templates=(
        Template(
            text=(
                "这段第一人称序列记录了你在房间中的一次行走。"
                "以最后一帧你的位置和朝向为准:{target}现在在你的哪个方向?"
                '如果序列提供的证据不足以判断,选"无法判断"。'
            ),
            options=("front", "left", "back", "right", "无法判断"),
        ),
    ),
    intervention_window="$t_gone:$t_q-1",
    abstain_on_unresolvable=("t_seen",),
    variant_expectations={
        "permute": "abstain",
        "drop_key": "abstain",
        "drop_filler": "same",
        "delay": "same",
    },
)


LEGACY_SCRIPT_LIBRARIES = MappingProxyType(
    {
        "std.v3": MappingProxyType(
            {LEGACY_P1_SELF_MOTION.capability: LEGACY_P1_SELF_MOTION}
        )
    }
)


def legacy_script_for_version(version: str, capability: str) -> ScriptSpec:
    """Resolve an explicitly supported legacy script, failing closed."""

    try:
        return LEGACY_SCRIPT_LIBRARIES[version][capability]
    except KeyError as error:
        raise ValueError(
            f"no frozen legacy script for standard={version!r}, "
            f"capability={capability!r}"
        ) from error
