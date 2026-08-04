"""Simulator-independent deterministic trajectory plan contract."""

from __future__ import annotations

from enum import StrEnum
from math import isclose, sqrt
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from spatial_episode.contracts.base import ContractModel, TransformV1, Vec3Value


class ViewRole(StrEnum):
    INITIAL = "initial"
    EXPLORE = "explore"
    TURNAROUND = "turnaround"
    REVISIT = "revisit"
    LOOP_CLOSURE = "loop_closure"
    ROTATION_STATION = "rotation_station"
    ORBIT = "orbit"
    BRIDGE = "bridge"
    OCCLUSION_PASS = "occlusion_pass"
    REVEAL = "reveal"
    ELEVATION = "elevation"
    TARGET_HOLDOUT = "target_holdout"


class TrajectoryClass(StrEnum):
    COVERAGE_WALK = "T1"
    SPARSE_SUBSAMPLE = "T2"
    ROTATION_STATION = "T3"
    OBJECT_ORBIT = "T4"
    TWO_VIEW_PAIR = "T5"
    CROSS_ROOM_BRIDGE = "T6"
    ELEVATION_CHANGE = "T7"
    OCCLUSION_REVEAL = "T8"
    INTERVENTION_PAIR = "T9"
    TARGET_VIEW = "T10"


class PlannedViewV1(ContractModel):
    step: int = Field(ge=0)
    view_id: str = Field(min_length=1)
    role: ViewRole
    world_from_agent: TransformV1
    distance_from_previous_m: float = Field(ge=0.0)
    cumulative_distance_m: float = Field(ge=0.0)
    backend_position_m: Vec3Value
    camera_height_m: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def transform_names_view_frame(self) -> PlannedViewV1:
        if self.world_from_agent.parent_frame != "world":
            raise ValueError("planned view parent frame must be canonical world")
        if self.world_from_agent.child_frame != f"agent:{self.view_id}":
            raise ValueError("planned view child frame must include its view ID")
        quaternion = self.world_from_agent.rotation_xyzw
        norm = sqrt(sum(value * value for value in quaternion))
        if not isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("planned view quaternion must be normalized")
        return self


class OcclusionAnnotationV1(ContractModel):
    target_id: str = Field(min_length=1)
    occluder_id: str = Field(min_length=1)
    occluded_views: tuple[str, ...] = Field(min_length=1)
    decisive_view: str = Field(min_length=1)


class TargetViewAnchorV1(ContractModel):
    view_id: str = Field(min_length=1)
    origin_entity_id: str = Field(min_length=1)
    facing_entity_id: str = Field(min_length=1)


class TrajectoryPlanV1(ContractModel):
    schema_version: Literal["trajectory_plan.v1"] = "trajectory_plan.v1"
    scene_id: UUID
    recipe_id: str = Field(min_length=1)
    seed: int
    backend_name: str = Field(min_length=1)
    backend_version: str = Field(min_length=1)
    backend_frame: str = Field(min_length=1)
    canonical_from_backend: TransformV1
    primary_island: int = Field(ge=0)
    primary_island_area_m2: float = Field(gt=0.0)
    outbound_geodesic_distance_m: float = Field(ge=0.0)
    closed_loop: bool
    trajectory_class: TrajectoryClass = TrajectoryClass.COVERAGE_WALK
    station_id: str | None = None
    yaw_sequence_deg: tuple[float, ...] | None = None
    depth_questions_forbidden: bool = False
    focus_entity_id: str | None = None
    azimuth_deg_per_view: tuple[float, ...] | None = None
    radius_m: float | None = Field(default=None, gt=0.0)
    orientation_questions_allowed: bool | None = None
    height_sequence_m: tuple[float, ...] | None = None
    pitch_sequence_deg: tuple[float, ...] | None = None
    occlusion_annotation: OcclusionAnnotationV1 | None = None
    target_anchors: tuple[TargetViewAnchorV1, ...] | None = None
    held_out: bool = False
    views: tuple[PlannedViewV1, ...] = Field(min_length=3)

    @model_validator(mode="after")
    def trajectory_is_contiguous_and_closed(self) -> TrajectoryPlanV1:
        if [view.step for view in self.views] != list(range(len(self.views))):
            raise ValueError("planned view steps must be contiguous from zero")
        view_ids = [view.view_id for view in self.views]
        if len(view_ids) != len(set(view_ids)):
            raise ValueError("planned view IDs must be unique")
        if (
            self.views[0].role is not ViewRole.INITIAL
            and self.trajectory_class is not TrajectoryClass.TARGET_VIEW
        ):
            raise ValueError("trajectory must begin with an initial view")
        if self.views[0].distance_from_previous_m != 0.0:
            raise ValueError("initial view cannot have a previous-view distance")
        if self.views[0].cumulative_distance_m != 0.0:
            raise ValueError("initial view cumulative distance must be zero")
        for previous, current in zip(self.views, self.views[1:], strict=False):
            expected = previous.cumulative_distance_m + current.distance_from_previous_m
            if not isclose(
                current.cumulative_distance_m,
                expected,
                rel_tol=0.0,
                abs_tol=1e-5,
            ):
                raise ValueError("trajectory cumulative distances are inconsistent")
        if self.closed_loop:
            if self.views[-1].role is not ViewRole.LOOP_CLOSURE:
                raise ValueError("closed trajectory must end with loop_closure")
            first = self.views[0].world_from_agent
            last = self.views[-1].world_from_agent
            if any(
                not isclose(left, right, rel_tol=0.0, abs_tol=1e-5)
                for left, right in zip(first.translation_m, last.translation_m, strict=True)
            ):
                raise ValueError("closed trajectory must return to its initial position")
            quaternion_dot = abs(
                sum(
                    left * right
                    for left, right in zip(first.rotation_xyzw, last.rotation_xyzw, strict=True)
                )
            )
            if not isclose(quaternion_dot, 1.0, rel_tol=0.0, abs_tol=1e-5):
                raise ValueError("closed trajectory must return to its initial orientation")
            if not isclose(
                self.views[-1].cumulative_distance_m,
                2.0 * self.outbound_geodesic_distance_m,
                rel_tol=0.0,
                abs_tol=1e-3,
            ):
                raise ValueError("closed trajectory distance must be twice the outbound path")
        if self.trajectory_class is TrajectoryClass.ROTATION_STATION:
            if self.closed_loop:
                raise ValueError("T3 rotation station is a panorama, not a path loop")
            if not self.station_id:
                raise ValueError("T3 rotation station must declare station_id")
            if self.yaw_sequence_deg is None or len(self.yaw_sequence_deg) != len(self.views):
                raise ValueError("T3 yaw_sequence_deg must align one-to-one with views")
            if not self.depth_questions_forbidden:
                raise ValueError("T3 must forbid parallax-dependent depth questions")
            if self.outbound_geodesic_distance_m != 0.0:
                raise ValueError("T3 must have zero translational path length")
            origin = self.views[0].world_from_agent.translation_m
            if any(
                not isclose(left, right, rel_tol=0.0, abs_tol=1e-6)
                for view in self.views[1:]
                for left, right in zip(
                    origin, view.world_from_agent.translation_m, strict=True
                )
            ):
                raise ValueError("T3 views must share one camera position")
        if self.trajectory_class is TrajectoryClass.OBJECT_ORBIT:
            if self.closed_loop:
                raise ValueError("T4 orbit samples do not duplicate the first view")
            if not self.focus_entity_id:
                raise ValueError("T4 must declare focus_entity_id")
            if self.azimuth_deg_per_view is None or len(
                self.azimuth_deg_per_view
            ) != len(self.views):
                raise ValueError("T4 azimuths must align one-to-one with views")
            if self.radius_m is None:
                raise ValueError("T4 must declare orbit radius")
            if any(view.role is not ViewRole.ORBIT for view in self.views[1:]):
                raise ValueError("T4 non-initial views must use the orbit role")
        if self.trajectory_class is TrajectoryClass.ELEVATION_CHANGE:
            if self.closed_loop:
                raise ValueError("T7 elevation trajectory is not a path loop")
            if not self.station_id:
                raise ValueError("T7 must declare station_id")
            if self.height_sequence_m is None or len(
                self.height_sequence_m
            ) != len(self.views):
                raise ValueError("T7 heights must align one-to-one with views")
            if self.pitch_sequence_deg is None or len(
                self.pitch_sequence_deg
            ) != len(self.views):
                raise ValueError("T7 pitches must align one-to-one with views")
            if any(view.camera_height_m is None for view in self.views):
                raise ValueError("T7 views must carry their effective camera height")
            origin = self.views[0].world_from_agent.translation_m[:2]
            if any(
                not isclose(left, right, rel_tol=0.0, abs_tol=1e-6)
                for view in self.views[1:]
                for left, right in zip(
                    origin, view.world_from_agent.translation_m[:2], strict=True
                )
            ):
                raise ValueError("T7 views must share one planar station")
        if self.trajectory_class is TrajectoryClass.OCCLUSION_REVEAL:
            if self.closed_loop:
                raise ValueError("T8 reveal trajectory is not a path loop")
            annotation = self.occlusion_annotation
            if annotation is None:
                raise ValueError("T8 must declare occlusion_annotation")
            known_view_ids = set(view_ids)
            if not set(annotation.occluded_views).issubset(known_view_ids):
                raise ValueError("T8 occluded views must belong to the trajectory")
            if annotation.decisive_view not in known_view_ids:
                raise ValueError("T8 decisive view must belong to the trajectory")
            decisive = self.views[view_ids.index(annotation.decisive_view)]
            if decisive.role is not ViewRole.REVEAL:
                raise ValueError("T8 decisive view must use the reveal role")
        if self.trajectory_class is TrajectoryClass.TARGET_VIEW:
            if self.closed_loop:
                raise ValueError("T10 held-out targets are not a path loop")
            if not self.held_out:
                raise ValueError("T10 views must be marked held_out")
            if self.target_anchors is None or len(self.target_anchors) != len(
                self.views
            ):
                raise ValueError("T10 anchors must align one-to-one with views")
            if any(view.role is not ViewRole.TARGET_HOLDOUT for view in self.views):
                raise ValueError("all T10 views must use target_holdout role")
            if [anchor.view_id for anchor in self.target_anchors] != view_ids:
                raise ValueError("T10 anchor view IDs must follow trajectory order")
        return self
