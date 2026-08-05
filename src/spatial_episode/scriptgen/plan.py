"""Trajectory plan: the engine's output artifact.

A plan is everything the renderer needs to realise one qualifying trajectory,
plus everything the compiler needs to audit it: poses, slot binding, resolved
frame variables, clause witnesses, knob levels and a PROVISIONAL gold answer.

The provisional answer comes from the geometry backend. It is a search-phase
estimate only — the authoritative answer is recompiled from rendered instance
masks after acquisition, and any disagreement rejects the bundle rather than
silently trusting either side.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .sceneview import Pose2D
from .spec import SpecModel


class PlannedPose(SpecModel):
    frame: int = Field(ge=0)
    x: float
    y: float
    yaw_deg: float

    @classmethod
    def from_pose(cls, frame: int, pose: Pose2D) -> PlannedPose:
        return cls(frame=frame, x=pose.x, y=pose.y, yaw_deg=pose.yaw_deg)


class ProvisionalAnswer(SpecModel):
    """Geometry-backend answer estimate recorded for post-render cross-check."""

    question_frame: int
    target: str
    azimuth_deg: float
    sector: str
    margin_deg: float


class TrajectoryPlan(SpecModel):
    schema_version: Literal["scriptgen_plan.v1"] = "scriptgen_plan.v1"
    plan_id: str
    scene_id: str
    capability: str
    standard_version: str
    seed: int
    binding: dict[str, str]
    frame_vars: dict[str, int]
    poses: tuple[PlannedPose, ...]
    knob_levels: dict[str, float]
    clause_witnesses: dict[str, dict[str, Any]]
    provisional_answer: ProvisionalAnswer


class GenerationReport(SpecModel):
    """Summary of one generation run, including every rejection reason."""

    scene_id: str
    capability: str
    standard_version: str
    plans: tuple[TrajectoryPlan, ...]
    rejection_counts: dict[str, int]
    slot_rejections: dict[str, int]
