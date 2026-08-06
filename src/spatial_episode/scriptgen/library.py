"""Script library: one declarative ScriptSpec per capability row.

Adding a capability means adding one spec here (and, rarely, one predicate to
``predicates``). Engine code must not change.
"""

from __future__ import annotations

from .spec import Clause, Knob, ScriptSpec, SlotSpec, Template

# Three-phase structure: SEEN (target clearly visible) -> TRANSITION (target
# may slide out of frame gradually while the camera turns at a trackable rate)
# -> GONE (definitely invisible until the question frame). The transition zone
# is what makes trackable ego-motion and decisive disappearance compatible:
# partial-visibility frames are permitted there and only there.
SELF_MOTION = ScriptSpec(
    capability="self_motion_update",
    slots={
        "target": SlotSpec(min_size_m=0.5, unique_referent=True),
    },
    frame_vars={
        "t_seen": "last_visible($target)",  # last clearly-visible frame
        "t_gone": "first_invisible_after($target, $t_seen)",  # transition ends
        "t_q": "last_frame()",  # the question frame
    },
    clauses=(
        # Target is clearly observed before it leaves the field of view.
        Clause(
            name="seen_early",
            predicate="visible_somewhere",
            args={"obj": "$target", "frames": "0:$t_seen"},
            phase="compile",
        ),
        # Ego-motion is visually trackable across the WHOLE trajectory:
        # bounded per-frame rotation and translation (std.v2 contract).
        Clause(
            name="trackable",
            predicate="step_motion_bounded",
            args={"frames": "0:$t_q"},
            phase="search",
        ),
        # After the transition the target stays definitely out of sight.
        Clause(
            name="gone",
            predicate="invisible_in_range",
            args={"obj": "$target", "frames": "$t_gone:$t_q"},
            phase="compile",
        ),
        # The invisible stretch is long enough to force memory, not glimpse.
        Clause(
            name="gap",
            predicate="frame_gap_ge",
            args={"later": "$t_q", "earlier": "$t_gone", "gap": 3},
            phase="search",
        ),
        # Total turn is large enough to make the remembered viewpoint stale,
        # but bounded: far beyond a composable rotation the answer stops being
        # mental rotation and becomes a guess.
        Clause(
            name="turned",
            predicate="cum_turn_between",
            args={"frames": "$t_seen:$t_q", "deg_min": 80, "deg_max": 200},
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
    # Turn magnitude is not a knob expression: it is recorded in the "turned"
    # clause witness (cum_turn_deg) and bucketed at analysis time.
    knobs=(Knob(name="delay", expr="$t_q-$t_gone", levels=(3, 6, 10)),),
    length=(10, 16),
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
