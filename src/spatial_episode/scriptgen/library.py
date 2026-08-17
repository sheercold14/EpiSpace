"""Script library: one declarative ScriptSpec per capability row.

Adding a capability means adding one spec here (and, rarely, one predicate to
``predicates``). Engine code must not change.
"""

from __future__ import annotations

from typing import Any

from .occlusion import OCCLUDER_DISPLAY_NAMES_ZH
from .spec import AnswerSpec, Clause, Knob, ScriptSpec, SlotSpec, Template

_TRAVERSAL_CLAUSES = (
    # Physical validity belongs to the acquired path only. Interventions
    # reindex already-rendered frames, so the compiler must not re-judge
    # these search-only clauses on a shuffled or shortened sequence.
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
)


def _script(*, clauses: tuple[Clause, ...], **fields: Any) -> ScriptSpec:
    """Construct a library spec with the mandatory traversal safety gates."""
    return ScriptSpec(clauses=_TRAVERSAL_CLAUSES + clauses, **fields)


# Three-phase structure: SEEN (target clearly visible) -> TRANSITION (target
# may slide out of frame gradually while the camera turns at a trackable rate)
# -> GONE (definitely invisible until the question frame). The transition zone
# is what makes trackable ego-motion and decisive disappearance compatible:
# partial-visibility frames are permitted there and only there.
SELF_MOTION = _script(
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
                '如果序列提供的证据不足以判断,选"无法判断"。'
            ),
            options=("front", "left", "back", "right", "无法判断"),
        ),
    ),
    intervention_window="$t_gone:$t_q-1",
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


NET_TURN = _script(
    capability="path_integration",
    slots={"target": SlotSpec(min_size_m=0.5, unique_referent=True)},
    frame_vars={"t_q": "last_frame()"},
    clauses=(
        Clause(
            name="trackable",
            predicate="step_motion_bounded",
            args={"frames": "0:$t_q"},
            phase="search",
            on_violation="abstain",
        ),
        Clause(
            name="net_turn_margin_ok",
            predicate="net_turn_margin_ge",
            args={"frames": "0:$t_q"},
            phase="search",
            on_violation="invalid",
        ),
    ),
    answer=AnswerSpec(mode="net_turn", args={"frames": "0:$t_q"}),
    length=(10, 16),
    motifs=("walk_and_turn",),
    templates=(
        Template(
            text=(
                "这段第一人称序列记录了你在房间中的一次行走。"
                "这段路你的净转向是向左还是向右?"
                '如果序列提供的证据不足以判断,选"无法判断"。'
            ),
            options=("left", "right", "无法判断"),
        ),
    ),
    intervention_window="1:$t_q-1",
    variant_expectations={
        "permute": "abstain",
        "drop_filler": "same",
        "delay": "same",
    },
)


NET_TURN_MAGNITUDE = _script(
    capability="path_integration_magnitude",
    slots={"target": SlotSpec(min_size_m=0.5, unique_referent=True)},
    frame_vars={"t_q": "last_frame()"},
    clauses=(
        Clause(
            name="trackable",
            predicate="step_motion_bounded",
            args={"frames": "0:$t_q"},
            phase="search",
            on_violation="abstain",
        ),
        Clause(
            name="magnitude_margin_ok",
            predicate="net_turn_magnitude_margin_ge",
            args={"frames": "0:$t_q"},
            phase="search",
        ),
    ),
    answer=AnswerSpec(mode="net_turn_magnitude", args={"frames": "0:$t_q"}),
    length=(10, 18),
    motifs=("walk_and_turn",),
    templates=(
        Template(
            text=(
                "这段第一人称序列记录了你在房间中的一次移动。"
                "你的净转角是否超过90度?"
                '如果序列提供的证据不足以判断,选"无法判断"。'
            ),
            options=("over_90", "at_most_90", "无法判断"),
        ),
    ),
    intervention_window="1:$t_q-1",
    variant_expectations={
        "permute": "abstain",
        "drop_filler": "same",
        "delay": "same",
    },
)


def _self_motion_subtype(
    *,
    capability: str,
    motif: str,
    motion_clauses: tuple[Clause, ...],
    length: tuple[int, int],
    min_target_size_m: float = 0.5,
    intervention_window: str = "$t_gone:$t_q-1",
) -> ScriptSpec:
    """Declare one C/combination-one trajectory subtype over the shared memory task."""
    return _script(
        capability=capability,
        slots={"target": SlotSpec(min_size_m=min_target_size_m, unique_referent=True)},
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
            *motion_clauses,
            Clause(
                name="margin_ok",
                predicate="sector_margin_ge",
                args={"obj": "$target", "frame": "$t_q"},
                phase="search",
            ),
        ),
        answer=AnswerSpec(mode="target_sector", args={"obj": "$target", "frame": "$t_q"}),
        knobs=(Knob(name="delay", expr="$t_q-$t_gone", levels=(3, 6, 10)),),
        length=length,
        motifs=(motif,),
        templates=SELF_MOTION.templates,
        intervention_window=intervention_window,
        abstain_on_unresolvable=("t_seen",),
        variant_expectations=SELF_MOTION.variant_expectations,
    )


PURE_ROTATION = _self_motion_subtype(
    capability="self_motion_update_pure_rotation",
    motif="stand_and_turn",
    length=(10, 16),
    motion_clauses=(
        Clause(
            name="stationary",
            predicate="displacement_below",
            args={"frames": "0:$t_q"},
            phase="search",
        ),
        Clause(
            name="turned",
            predicate="cum_turn_between",
            # For a side-sector answer the target normally leaves the camera
            # after roughly half of an 80--120 degree in-place turn.  Requiring
            # another 80 degrees *after* the last sighting makes left / right
            # geometrically impossible and collapses accepted data to back.
            # Trackability still applies per step, while this clause measures
            # the complete observable in-place rotation.
            args={"frames": "0:$t_q", "deg_min": 80, "deg_max": 200},
            phase="search",
        ),
    ),
)


PURE_TRANSLATION = _self_motion_subtype(
    capability="self_motion_update_pure_translation",
    motif="walk_straight_past",
    length=(10, 16),
    intervention_window="1:$t_q-1",
    motion_clauses=(
        Clause(
            name="heading_constant",
            predicate="turn_below",
            args={"frames": "0:$t_q"},
            phase="search",
        ),
    ),
)


MULTI_TURN = _self_motion_subtype(
    capability="self_motion_update_multi_turn",
    motif="walk_multi_turn",
    length=(14, 18),
    motion_clauses=(
        Clause(
            name="turned",
            predicate="cum_turn_between",
            args={"frames": "$t_seen:$t_q", "deg_min": 80, "deg_max": 200},
            phase="search",
        ),
        Clause(
            name="turn_segments",
            predicate="turn_segments_between",
            args={"frames": "0:$t_q"},
            phase="search",
        ),
    ),
)


OCCLUDED_MOTION = _self_motion_subtype(
    capability="self_motion_update_occluded",
    motif="walk_to_occlusion",
    length=(14, 18),
    # Occlusion episodes need a blocker that is appreciably larger than the
    # referent.  Reusing the 0.5 m generic self-motion floor excludes useful
    # uniquely named targets such as nightstands and bins and leaves only
    # large furniture that cannot be fully hidden in real pixels.
    min_target_size_m=0.25,
    motion_clauses=(
        Clause(
            name="occluded_at_question",
            predicate="occluded_in_view",
            args={"obj": "$target", "frame": "$t_q"},
            phase="compile",
        ),
    ),
)


_DISAPPEARANCE_FRAME_VARS = {
    "t_event": "first_decisive_disappearance($target)",
    "t_seen": "last_visible_before($target, $t_event)",
    "t_q": "last_frame()",
}


def _disappearance_evidence_clauses(*, cause: str) -> tuple[Clause, ...]:
    """Shared authority gates for questions about the first disappearance."""

    return (
        Clause(
            name="identity_evidence",
            predicate="consecutive_visible_frames_ge",
            args={"obj": "$target", "frames": "0:$t_seen", "required": 2},
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
            name="decisive_disappearance",
            predicate="disappearance_event",
            args={"obj": "$target", "frame": "$t_event", "cause": cause},
            phase="compile",
        ),
        Clause(
            name="stays_gone",
            predicate="invisible_in_range",
            args={"obj": "$target", "frames": "$t_event:$t_q"},
            phase="compile",
        ),
    )


OCCLUDER_IDENTIFICATION = _script(
    capability="occluder_identification",
    slots={"target": SlotSpec(min_size_m=0.25, unique_referent=True)},
    frame_vars=_DISAPPEARANCE_FRAME_VARS,
    clauses=_disappearance_evidence_clauses(cause="occluded"),
    answer=AnswerSpec(
        mode="occluder_category_at",
        args={"obj": "$target", "frame": "$t_event"},
    ),
    length=(10, 22),
    motifs=("walk_through_occlusion",),
    templates=(
        Template(
            text=(
                "这段第一人称序列中，{target}第一次明确消失时，"
                "最先挡住它的是什么？如果证据不足，选‘无法判断’。"
            ),
            # The family compiler materialises the correct label plus two
            # deterministic distractors from this policy-level superset.
            options=(*OCCLUDER_DISPLAY_NAMES_ZH.keys(), "无法判断"),
        ),
    ),
    intervention_window="$t_event:$t_q",
    abstain_on_unresolvable=("t_event", "t_seen"),
    variant_expectations={"drop_key": "abstain", "delay": "same"},
)


DISAPPEARANCE_CAUSE = _script(
    capability="disappearance_cause",
    slots={"target": SlotSpec(min_size_m=0.25, unique_referent=True)},
    frame_vars=_DISAPPEARANCE_FRAME_VARS,
    clauses=_disappearance_evidence_clauses(cause="either"),
    answer=AnswerSpec(
        mode="disappearance_cause_at",
        args={"obj": "$target", "frame": "$t_event"},
    ),
    length=(10, 22),
    motifs=("walk_through_occlusion",),
    templates=(
        Template(
            text=(
                "这段第一人称序列中，{target}第一次明确消失的原因是什么？"
                "如果证据不足，选‘无法判断’。"
            ),
            options=("occluded", "out_of_view", "无法判断"),
        ),
    ),
    intervention_window="$t_event:$t_q",
    abstain_on_unresolvable=("t_event", "t_seen"),
    variant_expectations={"drop_key": "abstain", "delay": "same"},
)


AFTER_OCCLUSION_MOTION = _script(
    capability="self_motion_update_after_occlusion",
    slots={"target": SlotSpec(min_size_m=0.25, unique_referent=True)},
    frame_vars=_DISAPPEARANCE_FRAME_VARS,
    clauses=(
        *_disappearance_evidence_clauses(cause="occluded"),
        Clause(
            name="post_occlusion_gap",
            predicate="frame_gap_ge",
            args={"later": "$t_q", "earlier": "$t_event", "gap": 3},
            phase="search",
        ),
        Clause(
            name="post_occlusion_turn",
            predicate="cum_turn_between",
            args={"frames": "$t_event:$t_q", "deg_min": 80, "deg_max": 200},
            phase="search",
        ),
        Clause(
            name="margin_ok",
            predicate="sector_margin_ge",
            args={"obj": "$target", "frame": "$t_q"},
            phase="search",
        ),
    ),
    answer=AnswerSpec(mode="target_sector", args={"obj": "$target", "frame": "$t_q"}),
    knobs=(Knob(name="delay", expr="$t_q-$t_event", levels=(3, 6, 10)),),
    length=(16, 22),
    motifs=("walk_through_occlusion",),
    templates=SELF_MOTION.templates,
    intervention_window="$t_event:$t_q-1",
    abstain_on_unresolvable=("t_event", "t_seen"),
    variant_expectations=SELF_MOTION.variant_expectations,
)


HOMING = _script(
    capability="homing_probe",
    slots={"target": SlotSpec(min_size_m=0.5, unique_referent=True)},
    frame_vars={"t_q": "last_frame()"},
    clauses=(
        Clause(
            name="trackable",
            predicate="step_motion_bounded",
            args={"frames": "0:$t_q"},
            phase="search",
            on_violation="abstain",
        ),
        Clause(
            name="start_far_enough",
            predicate="start_far_enough",
            args={"frame": "$t_q"},
            phase="search",
            on_violation="invalid",
        ),
        Clause(
            name="start_sector_margin_ok",
            predicate="start_sector_margin_ge",
            args={"frame": "$t_q"},
            phase="search",
            on_violation="invalid",
        ),
    ),
    answer=AnswerSpec(mode="start_sector", args={"frame": "$t_q"}),
    length=(10, 16),
    motifs=("walk_and_turn",),
    templates=(
        Template(
            text=(
                "这段第一人称序列记录了你从起点开始的一次行走。"
                "以最后的位置和朝向为准,出发点现在在你的哪个方向?"
                '如果序列提供的证据不足以判断,选"无法判断"。'
            ),
            options=("front", "left", "back", "right", "无法判断"),
        ),
    ),
    intervention_window="1:$t_q-1",
    variant_expectations={
        "permute": "abstain",
        "drop_filler": "same",
        "delay": "same",
    },
)


VIEW_SIDE = _script(
    capability="view_side_check",
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
            name="view_side_margin_ok",
            predicate="view_side_margin_ge",
            args={"obj": "$target", "frame": "$t_seen"},
            phase="search",
            on_violation="invalid",
        ),
    ),
    answer=AnswerSpec(
        mode="view_side",
        args={"obj": "$target", "frame": "$t_seen"},
    ),
    length=(10, 16),
    motifs=("walk_and_turn",),
    templates=(
        Template(
            text=(
                "如果这段第一人称序列提供了足够证据,"
                "{target}最后出现时位于画面的左半边还是右半边?"
                '如果序列提供的证据不足以判断,选"无法判断"。'
            ),
            options=("left_half", "right_half", "无法判断"),
        ),
    ),
    intervention_window="$t_gone:$t_q-1",
    abstain_on_unresolvable=("t_seen",),
    variant_expectations={
        "permute": "same",
        "drop_key": "abstain",
        "drop_filler": "same",
        "delay": "same",
    },
)


IMAGINED_VIEWPOINT_OFFSETS = (0, 45, 90, 135, 180)


def _reference_frame_spec(
    *,
    answer_mode: str,
    yaw_offset_deg: int,
) -> ScriptSpec:
    """One point on the deep/shallow imagined-viewpoint covariance curve."""
    deep = answer_mode == "imagined_sector"
    suffix = "" if yaw_offset_deg == 0 else f"_yaw{yaw_offset_deg}"
    stem = "reference_frame_transform" if deep else "reference_frame_visibility"
    qualifier = Clause(
        name="imagined_answer_decisive",
        predicate=(
            "imagined_curve_sector_margins_ge" if deep else "imagined_curve_visibility_decisive"
        ),
        args={
            "viewpoint": "$viewpoint",
            "facing": "$facing",
            "obj": "$target",
        },
        phase="search",
    )
    if yaw_offset_deg:
        facing_text = f"朝向从{{facing}}方向向左旋转{yaw_offset_deg}度"
    else:
        facing_text = "面向{facing}"
    if deep:
        question = f"若你站在{{viewpoint}}处、{facing_text},{{target}}在你的哪个方向?"
        options = ("front", "left", "back", "right", "无法判断")
    else:
        question = f"若你站在{{viewpoint}}处、{facing_text},你能看见{{target}}吗?"
        options = ("visible", "not_visible", "无法判断")
    return _script(
        capability=f"{stem}{suffix}",
        slots={
            "viewpoint": SlotSpec(min_size_m=0.3, unique_referent=True),
            "facing": SlotSpec(min_size_m=0.3, unique_referent=True),
            "target": SlotSpec(min_size_m=0.3, unique_referent=True),
        },
        frame_vars={"t_q": "last_frame()"},
        clauses=(
            Clause(
                name="stationary_survey",
                predicate="displacement_below",
                args={"frames": "0:$t_q"},
                phase="search_only",
            ),
            Clause(
                name="trackable_survey",
                predicate="step_motion_bounded",
                args={"frames": "0:$t_q"},
                phase="search_only",
            ),
            Clause(
                name="landmark_evidence",
                predicate="all_landmarks_visible",
                args={
                    "viewpoint": "$viewpoint",
                    "facing": "$facing",
                    "obj": "$target",
                    "frames": "0:$t_q",
                },
                phase="compile",
                on_violation="abstain",
            ),
            Clause(
                name="no_single_frame_shortcut",
                predicate="never_all_covisible",
                args={
                    "viewpoint": "$viewpoint",
                    "facing": "$facing",
                    "obj": "$target",
                    "frames": "0:$t_q",
                },
                phase="compile",
            ),
            Clause(
                name="imagined_pose_stable",
                predicate="imagined_pose_valid",
                args={"viewpoint": "$viewpoint", "facing": "$facing"},
                phase="search",
            ),
            qualifier,
        ),
        answer=AnswerSpec(
            mode=answer_mode,
            args={
                "viewpoint": "$viewpoint",
                "facing": "$facing",
                "obj": "$target",
                "yaw_offset_deg": yaw_offset_deg,
            },
        ),
        knobs=(
            Knob(
                name="imagined_yaw_offset_deg",
                expr=str(yaw_offset_deg),
                levels=tuple(float(value) for value in IMAGINED_VIEWPOINT_OFFSETS),
            ),
        ),
        length=(14, 18),
        motifs=("survey",),
        templates=(
            Template(
                text=(
                    f'这段第一人称序列是从同一站位拍摄的环视。{question}如果证据不足,选"无法判断"。'
                ),
                options=options,
            ),
        ),
        intervention_window="0:$t_q",
        variant_expectations={
            "permute": "same",
            "drop_key": "abstain",
            "drop_filler": "same",
            "delay": "same",
        },
    )


REFERENCE_FRAME_DEEP = tuple(
    _reference_frame_spec(answer_mode="imagined_sector", yaw_offset_deg=offset)
    for offset in IMAGINED_VIEWPOINT_OFFSETS
)
REFERENCE_FRAME_SHALLOW = tuple(
    _reference_frame_spec(answer_mode="imagined_visibility", yaw_offset_deg=offset)
    for offset in IMAGINED_VIEWPOINT_OFFSETS
)
REFERENCE_FRAME_SCRIPTS = REFERENCE_FRAME_DEEP + REFERENCE_FRAME_SHALLOW


def _cross_view_spec(*, chain_length: int, answer_mode: str) -> ScriptSpec:
    """Declare one pair-relation/distance question over an anchor-chain length."""
    relation = answer_mode == "pair_relation"
    stem = "cross_view_pair_relation" if relation else "cross_view_closer"
    slots = {
        "target": SlotSpec(min_size_m=0.3, unique_referent=True),
        **{
            f"anchor{index}": SlotSpec(min_size_m=0.3, unique_referent=True)
            for index in range(1, chain_length + 1)
        },
        "other": SlotSpec(min_size_m=0.3, unique_referent=True),
    }
    chain_args: dict[str, str | int | float | bool] = {
        "first": "$target",
        "second": "$other",
        "anchor1": "$anchor1",
        "frames": "0:$t_q",
    }
    for index in range(2, chain_length + 1):
        chain_args[f"anchor{index}"] = f"$anchor{index}"
    if relation:
        qualifier = Clause(
            name="relation_margin",
            predicate="pair_relation_margin_ge",
            args={"obj": "$target", "reference": "$other"},
            phase="search",
        )
        answer = AnswerSpec(
            mode="pair_relation",
            args={"obj": "$target", "reference": "$other"},
        )
        question = (
            "以{other}自身的朝向为准,{target}在{other}的哪个方向?"
            '如果证据不足,选"无法判断"。'
        )
        options = ("front", "left", "back", "right", "无法判断")
    else:
        qualifier = Clause(
            name="distance_ratio",
            predicate="closer_ratio_ge",
            args={"first": "$target", "second": "$other", "anchor": "$anchor1"},
            phase="search",
        )
        answer = AnswerSpec(
            mode="closer_of",
            args={"first": "$target", "second": "$other", "anchor": "$anchor1"},
        )
        question = (
            "{target}和{other}中,哪一个离{anchor1}更近?"
            '第一个选项指{target},第二个选项指{other};证据不足时选"无法判断"。'
        )
        options = ("first", "second", "无法判断")
    length_ranges = {1: (24, 30), 2: (32, 40), 3: (40, 52)}
    return _script(
        capability=f"{stem}_k{chain_length}",
        slots=slots,
        frame_vars={"t_q": "last_frame()"},
        clauses=(
            Clause(
                name="trackable_visit",
                predicate="step_motion_bounded",
                args={"frames": "0:$t_q"},
                phase="search_only",
            ),
            Clause(
                name="queried_pair_never_covisible",
                predicate="never_covisible",
                args={"first": "$target", "second": "$other", "frames": "0:$t_q"},
                phase="compile",
            ),
            Clause(
                name="anchor_chain_evidence",
                predicate="chain_connected",
                args=chain_args,
                phase="compile",
                on_violation="abstain",
            ),
            qualifier,
        ),
        answer=answer,
        knobs=(
            Knob(
                name="anchor_chain_length",
                expr=str(chain_length),
                levels=(1.0, 2.0, 3.0),
            ),
        ),
        length=length_ranges[chain_length],
        motifs=("visit_landmarks",),
        templates=(Template(text=question, options=options),),
        intervention_window="0:$t_q",
        variant_expectations={
            "permute": "same",
            "drop_key": "abstain",
            "drop_filler": "same",
            "delay": "same",
        },
    )


CROSS_VIEW_RELATION = tuple(
    _cross_view_spec(chain_length=length, answer_mode="pair_relation")
    for length in (1, 2, 3)
)
CROSS_VIEW_CLOSER = tuple(
    _cross_view_spec(chain_length=length, answer_mode="closer_of")
    for length in (1, 2, 3)
)
CROSS_VIEW_SCRIPTS = CROSS_VIEW_RELATION + CROSS_VIEW_CLOSER
CROSS_VIEW_SCRIPT_SETS = {
    script.capability: (CROSS_VIEW_RELATION[index], CROSS_VIEW_CLOSER[index])
    for index in range(3)
    for script in (CROSS_VIEW_RELATION[index], CROSS_VIEW_CLOSER[index])
}


def _existence_sufficiency_spec(*, category: str) -> ScriptSpec:
    """Declare the cross-frame absence/insufficient-evidence meta question."""
    return _script(
        capability=f"existence_sufficiency_{category}",
        # Existing plans all carry a target binding. It remains a compatibility
        # anchor for compiler/family contracts; category truth is a spec constant.
        slots={"target": SlotSpec(min_size_m=0.3, unique_referent=True)},
        frame_vars={"t_q": "last_frame()"},
        clauses=(
            Clause(
                name="trackable_observation",
                predicate="step_motion_bounded",
                args={"frames": "0:$t_q"},
                phase="search_only",
            ),
            Clause(
                name="category_absent",
                predicate="category_absent",
                args={"category": category},
                phase="search",
            ),
            Clause(
                name="coverage_sufficient",
                predicate="coverage_ratio_ge",
                args={"frames": "0:$t_q"},
                phase="compile",
                on_violation="abstain",
            ),
            Clause(
                name="no_single_frame_shortcut",
                predicate="no_single_frame_coverage_sufficient",
                args={"frames": "0:$t_q"},
                phase="compile",
            ),
            Clause(
                name="drop_key_destroys_evidence",
                predicate="drop_target_breaks_coverage",
                args={"target": "$target", "frames": "0:$t_q"},
                phase="search",
            ),
        ),
        answer=AnswerSpec(
            mode="existence_sufficiency",
            args={"category": category, "frames": "0:$t_q"},
        ),
        # Actual ratio and analysis tier are recorded by the coverage witness;
        # unlike frame expressions, the knob evaluator intentionally has no floats.
        knobs=(),
        length=(10, 18),
        # No dedicated camera choreography: generation uses accept/reject over
        # existing target-compatible motifs, and question groups attach to any source.
        motifs=("survey",),
        templates=(
            Template(
                text=(
                    f"仅根据这段连续观察,能否确定这套房里有没有{category}?"
                    '请在给定选项中作答;如果证据不足,选"无法判断"。'
                ),
                options=("present", "absent", "无法判断"),
            ),
        ),
        intervention_window="0:$t_q",
        variant_expectations={
            "permute": "same",
            "drop_key": "abstain",
            "delay": "same",
        },
    )


EXISTENCE_SUFFICIENCY = _existence_sufficiency_spec(category="bed")

SCRIPT_LIBRARY: dict[str, ScriptSpec] = {
    script.capability: script
    for script in (
        SELF_MOTION,
        PURE_ROTATION,
        PURE_TRANSLATION,
        MULTI_TURN,
        OCCLUDED_MOTION,
        AFTER_OCCLUSION_MOTION,
        OCCLUDER_IDENTIFICATION,
        DISAPPEARANCE_CAUSE,
        NET_TURN,
        NET_TURN_MAGNITUDE,
        HOMING,
        VIEW_SIDE,
        *REFERENCE_FRAME_SCRIPTS,
        *CROSS_VIEW_SCRIPTS,
        EXISTENCE_SUFFICIENCY,
    )
}
