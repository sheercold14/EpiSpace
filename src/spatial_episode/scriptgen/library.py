"""Script library: one declarative ScriptSpec per capability row.

Adding a capability means adding one spec here (and, rarely, one predicate to
``predicates``). Engine code must not change.
"""

from __future__ import annotations

from .spec import AnswerSpec, Clause, Knob, ScriptSpec, SlotSpec, Template

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
        # Physical validity belongs to the acquired path only. Interventions
        # reindex already-rendered frames, so the compiler must not re-judge
        # these two search-only clauses on a shuffled or shortened sequence.
        Clause(
            name="poses_clear",
            predicate="poses_clear",
            args={"frames": "0:$t_q"},
            phase="search_only",
        ),
        Clause(
            name="path_clear",
            predicate="path_clear",
            args={"frames": "0:$t_q"},
            phase="search_only",
        ),
        # Target is clearly observed before it leaves the field of view.
        # Evidence clause: without the sighting, even an ideal agent must abstain.
        Clause(
            name="seen_early",
            predicate="visible_somewhere",
            args={"obj": "$target", "frames": "0:$t_seen"},
            phase="compile",
            on_violation="abstain",
        ),
        # Ego-motion is visually trackable across the WHOLE trajectory:
        # bounded per-frame rotation and translation (std.v2 contract).
        # Evidence clause: untrackable self-motion means the current pose is
        # unknowable from the stream, so the only calibrated answer is abstain.
        Clause(
            name="trackable",
            predicate="step_motion_bounded",
            args={"frames": "0:$t_q"},
            phase="search",
            on_violation="abstain",
        ),
        # After the transition the target stays definitely out of sight.
        # Validity clause: a still-visible target makes this a perception
        # question, not a memory question — no gold may be asserted.
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
    answer=AnswerSpec(
        mode="target_sector",
        args={"obj": "$target", "frame": "$t_q"},
    ),
    # Turn magnitude is not a knob expression: it is recorded in the "turned"
    # clause witness (cum_turn_deg) and bucketed at analysis time.
    knobs=(Knob(name="delay", expr="$t_q-$t_gone", levels=(3, 6, 10)),),
    length=(10, 16),
    motifs=("walk_and_turn",),
    # The question text is shared verbatim by every variant of a family, so it
    # must not assert the sighting (false under drop_key) nor cite frame
    # numbers (they renumber under interventions and leak which frames are key).
    templates=(
        Template(
            text=(
                "这段第一人称序列记录了你在房间中的一次行走。"
                "以最后一帧你的位置和朝向为准:{target}现在在你的哪个方向?"
                "如果序列提供的证据不足以判断,选\"无法判断\"。"
            ),
            options=("front", "left", "back", "right", "无法判断"),
        ),
    ),
    # Family contract: how the recompiled gold must respond to each
    # intervention. t_seen unresolvable = the sighting frames are gone.
    abstain_on_unresolvable=("t_seen",),
    variant_expectations={
        "permute": "abstain",  # shuffled frames destroy ego-motion tracking
        "drop_key": "abstain",  # sighting removed: target direction unknowable
        "drop_filler": "same",  # redundant frames removed: answer invariant
        "delay": "same",  # a pause lengthens the delay knob, not the answer
    },
)

SCRIPT_LIBRARY: dict[str, ScriptSpec] = {
    SELF_MOTION.capability: SELF_MOTION,
}
