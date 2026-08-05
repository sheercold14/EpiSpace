"""Script library: one declarative ScriptSpec per capability row.

Adding a capability means adding one spec here (and, rarely, one predicate to
``predicates``). Engine code must not change.
"""

from __future__ import annotations

from .spec import Clause, Knob, ScriptSpec, SlotSpec, Template

SELF_MOTION = ScriptSpec(
    capability="self_motion_update",
    slots={
        "target": SlotSpec(min_size_m=0.5, unique_referent=True),
    },
    frame_vars={
        "t_seen": "last_visible($target)",  # last frame the target is visible
        "t_q": "last_frame()",  # the question frame
    },
    clauses=(
        # Target is observed at least once before it leaves the field of view.
        Clause(
            name="seen_early",
            predicate="visible_somewhere",
            args={"obj": "$target", "frames": "0:$t_seen"},
            phase="compile",
        ),
        # After t_seen the target stays out of sight up to the question frame,
        # forcing the answer to come from memory plus self-motion updating.
        Clause(
            name="out_of_view",
            predicate="invisible_in_range",
            args={"obj": "$target", "frames": "$t_seen+1:$t_q"},
            phase="compile",
        ),
        # Enough self-motion happened that the remembered viewpoint is stale.
        Clause(
            name="turned",
            predicate="cum_turn_ge",
            args={"frames": "$t_seen:$t_q", "deg": 90},
            phase="search",
        ),
        # The gold answer clears the sector boundary margin at the question frame.
        Clause(
            name="margin_ok",
            predicate="sector_margin_ge",
            args={"obj": "$target", "frame": "$t_q"},
            phase="search",
        ),
    ),
    knobs=(Knob(name="delay", expr="$t_q-$t_seen", levels=(3, 6, 10)),),
    length=(8, 14),
    motifs=("walk_and_turn",),
    templates=(
        Template(
            text=(
                "你在行走途中(第 {t_seen} 帧)看到过{target}。"
                "现在你位于第 {t_q} 帧的位置和朝向。{target}在你的哪个方向?"
            ),
            options=("front", "left", "back", "right", "无法判断"),
        ),
    ),
)

SCRIPT_LIBRARY: dict[str, ScriptSpec] = {
    SELF_MOTION.capability: SELF_MOTION,
}
