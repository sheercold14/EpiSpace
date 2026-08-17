"""Authoritative disappearance events and semantic occluder policy.

Rendered visibility answers *whether* a target is visible.  This module adds
the causal layer needed by occlusion questions: the first decisive event after
the target has been seen, with a short grace window for a partially covered
transition.  Every consumer (compiler, QA builder, audit, and renderer gate)
uses this implementation so terminal blockers can never be mistaken for the
object that initially caused the disappearance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .sceneview import SceneView
from .standards import CompileStandard

DISAPPEARANCE_POLICY_VERSION = "occlusion_semantic_policy.v1"
DEFAULT_ATTRIBUTION_GRACE_FRAMES = 2

OCCLUDER_DISPLAY_NAMES_ZH: dict[str, str] = {
    "bathtub": "浴缸",
    "fridge": "冰箱",
    "hanging_plant": "悬挂植物",
    "openable_window": "可开启窗户",
    "pillar": "柱子",
    "skeletal_frame": "框架结构",
    "walls": "墙",
}

DISALLOWED_OCCLUDER_CATEGORIES = frozenset(
    {
        "background",
        "ceilings",
        "driveway",
        "floors",
        "lawn",
        "roof",
        "unknown",
    }
)

DisappearanceCause = Literal["occluded", "out_of_view"]


def occluder_category_allowed(category: str | None) -> bool:
    """Whether a rendered category is both semantic and release-displayable."""

    return bool(
        category
        and category not in DISALLOWED_OCCLUDER_CATEGORIES
        and category in OCCLUDER_DISPLAY_NAMES_ZH
    )


@dataclass(frozen=True)
class DisappearanceEvent:
    frame: int
    first_invisible_frame: int
    last_visible_frame: int
    cause: DisappearanceCause
    occluder_entity_id: str | None
    occluder_category: str | None
    eligible_occluder: bool
    witness: dict[str, Any]


def _decisive_at(
    view: SceneView,
    obj: str,
    frame: int,
    *,
    first_invisible_frame: int,
    last_visible_frame: int,
) -> DisappearanceEvent | None:
    observation = view.occlusion(obj, frame)
    witness = {
        **dict(observation.witness),
        "event_policy_version": DISAPPEARANCE_POLICY_VERSION,
        "first_invisible_frame": first_invisible_frame,
        "last_visible_frame": last_visible_frame,
        "event_frame": frame,
    }
    if observation.status == "occluded":
        category = witness.get("occluder_category")
        entity_id = witness.get("occluder_entity_id")
        allowed = occluder_category_allowed(
            category if isinstance(category, str) else None
        )
        witness["occluder_category_allowed"] = allowed
        if not allowed:
            witness["semantic_reason"] = "disallowed_occluder_category"
        return DisappearanceEvent(
            frame=frame,
            first_invisible_frame=first_invisible_frame,
            last_visible_frame=last_visible_frame,
            cause="occluded",
            occluder_entity_id=entity_id if isinstance(entity_id, str) else None,
            occluder_category=category if isinstance(category, str) else None,
            eligible_occluder=allowed,
            witness=witness,
        )
    reason = witness.get("reason")
    if observation.status == "clear" and reason in {
        "target_behind_camera",
        "target_projection_outside_image",
    }:
        return DisappearanceEvent(
            frame=frame,
            first_invisible_frame=first_invisible_frame,
            last_visible_frame=last_visible_frame,
            cause="out_of_view",
            occluder_entity_id=None,
            occluder_category=None,
            eligible_occluder=False,
            witness=witness,
        )
    return None


def first_decisive_disappearance(
    view: SceneView,
    std: CompileStandard,
    obj: str,
    *,
    grace_frames: int = DEFAULT_ATTRIBUTION_GRACE_FRAMES,
) -> DisappearanceEvent | None:
    """Return the first resolvable visible-to-invisible causal event.

    A target can cross the render visibility threshold one frame before the
    centre patch contains enough foreground pixels for attribution.  Ambiguous
    attribution is therefore allowed for at most ``grace_frames`` while the
    target remains definitely invisible.  A decisive out-of-view observation
    is returned immediately; it is never overwritten by a later blocker.
    """

    if grace_frames < 0:
        raise ValueError("grace_frames must be non-negative")
    last_visible: int | None = None
    frame = 0
    while frame < view.frame_count:
        state = view.visibility(obj, frame).tristate(std)
        if state is True:
            last_visible = frame
            frame += 1
            continue
        if state is not False or last_visible is None:
            frame += 1
            continue

        first_invisible = frame
        stop = min(view.frame_count - 1, first_invisible + grace_frames)
        for event_frame in range(first_invisible, stop + 1):
            if view.visibility(obj, event_frame).tristate(std) is not False:
                break
            event = _decisive_at(
                view,
                obj,
                event_frame,
                first_invisible_frame=first_invisible,
                last_visible_frame=last_visible,
            )
            if event is not None:
                return event
        # This transition was not decisively attributable.  Do not divide one
        # long invisible run into artificial grace windows: attribution is
        # licensed only immediately after the actual visible -> invisible
        # boundary.  Resume only after a real reappearance, from which a new
        # transition may begin.
        frame = stop + 1
        while frame < view.frame_count:
            state = view.visibility(obj, frame).tristate(std)
            if state is True:
                last_visible = frame
                break
            frame += 1
    return None
