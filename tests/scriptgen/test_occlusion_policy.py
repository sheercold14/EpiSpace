"""First-event occlusion authority and the three causal QA capabilities."""

from __future__ import annotations

from dataclasses import dataclass

from spatial_episode.scriptgen.compiler import CapabilityCompiler
from spatial_episode.scriptgen.family import _materialized_question_options
from spatial_episode.scriptgen.library import (
    AFTER_OCCLUSION_MOTION,
    DISAPPEARANCE_CAUSE,
    OCCLUDER_IDENTIFICATION,
)
from spatial_episode.scriptgen.occlusion import first_decisive_disappearance
from spatial_episode.scriptgen.sceneview import (
    OcclusionObservation,
    Pose2D,
    SceneLayout,
    SceneObject,
    VisibilityObservation,
)
from spatial_episode.scriptgen.standards import STD_V1

TARGET = SceneObject("target", "chair", (0.0, 2.0), 0.8, "target")


@dataclass(frozen=True)
class EventView:
    pixels: tuple[int, ...]
    observations: tuple[tuple[str, str | None, str | None], ...]
    yaws: tuple[float, ...] = ()
    layout: SceneLayout = SceneLayout(
        scene_id="event-test",
        walkable_min=(-3.0, -3.0),
        walkable_max=(3.0, 3.0),
        objects=(TARGET,),
    )

    @property
    def frame_count(self) -> int:
        return len(self.pixels)

    def camera_pose(self, t: int) -> Pose2D:
        yaw = self.yaws[t] if self.yaws else 90.0
        return Pose2D(0.0, 0.0, yaw)

    def object(self, name: str) -> SceneObject:
        if name != TARGET.name:
            raise KeyError(name)
        return TARGET

    def objects(self) -> list[SceneObject]:
        return [TARGET]

    def visibility(self, name: str, t: int) -> VisibilityObservation:
        return VisibilityObservation("render_pixels", float(self.pixels[t]), 1.0)

    def occlusion(self, name: str, t: int) -> OcclusionObservation:
        status, reason, category = self.observations[t]
        witness = {"backend": "render_instance_depth", "frame": t, "reason": reason}
        ids: tuple[str, ...] = ()
        if category is not None:
            witness.update(
                {
                    "occluder_entity_id": f"{category}-1",
                    "occluder_category": category,
                }
            )
            ids = (f"{category}-1",)
        return OcclusionObservation(
            "render_instance_depth", status, ids, witness  # type: ignore[arg-type]
        )

    def occluders_between(self, name: str, t: int) -> tuple[str, ...]:
        return self.occlusion(name, t).occluder_ids


def _view(
    observations: tuple[tuple[str, str | None, str | None], ...],
    *,
    yaws: tuple[float, ...] = (),
) -> EventView:
    pixels = tuple(5000 if index < 2 else 0 for index in range(len(observations)))
    return EventView(pixels, observations, yaws)


def test_first_decisive_event_uses_short_attribution_grace_only() -> None:
    view = _view(
        (
            ("clear", "target_visible", None),
            ("clear", "target_visible", None),
            ("ambiguous", "insufficient_foreground_support", None),
            ("occluded", None, "walls"),
            ("occluded", None, "fridge"),
        )
    )
    event = first_decisive_disappearance(view, STD_V1, "target")
    assert event is not None
    assert event.frame == 3
    assert event.first_invisible_frame == 2
    assert event.occluder_category == "walls"

    too_late = _view(
        (
            ("clear", "target_visible", None),
            ("clear", "target_visible", None),
            ("ambiguous", "insufficient_foreground_support", None),
            ("ambiguous", "insufficient_foreground_support", None),
            ("ambiguous", "insufficient_foreground_support", None),
            ("occluded", None, "walls"),
        )
    )
    assert first_decisive_disappearance(too_late, STD_V1, "target") is None


def test_out_of_view_is_not_overwritten_by_a_later_blocker() -> None:
    view = _view(
        (
            ("clear", "target_visible", None),
            ("clear", "target_visible", None),
            ("clear", "target_projection_outside_image", None),
            ("occluded", None, "walls"),
        )
    )
    event = first_decisive_disappearance(view, STD_V1, "target")
    assert event is not None
    assert event.frame == 2
    assert event.cause == "out_of_view"


def test_semantic_floor_blocker_is_not_release_eligible() -> None:
    view = _view(
        (
            ("clear", "target_visible", None),
            ("clear", "target_visible", None),
            ("occluded", None, "floors"),
            ("occluded", None, "floors"),
        )
    )
    event = first_decisive_disappearance(view, STD_V1, "target")
    assert event is not None and not event.eligible_occluder
    certificate = CapabilityCompiler(OCCLUDER_IDENTIFICATION, STD_V1).compile(
        view, {"target": "target"}
    )
    assert certificate.status == "invalid"
    assert certificate.reason == "clause:decisive_disappearance"


def test_new_capabilities_compile_from_the_same_authoritative_event() -> None:
    view = _view(
        (
            ("clear", "target_visible", None),
            ("clear", "target_visible", None),
            ("occluded", None, "walls"),
            ("occluded", None, "walls"),
            ("occluded", None, "walls"),
            ("occluded", None, "walls"),
            ("occluded", None, "walls"),
            ("occluded", None, "walls"),
        ),
        yaws=(90.0, 90.0, 90.0, 60.0, 30.0, 0.0, -15.0, -15.0),
    )
    binding = {"target": "target"}
    identity = CapabilityCompiler(OCCLUDER_IDENTIFICATION, STD_V1).compile(view, binding)
    cause = CapabilityCompiler(DISAPPEARANCE_CAUSE, STD_V1).compile(view, binding)
    motion = CapabilityCompiler(AFTER_OCCLUSION_MOTION, STD_V1).compile(view, binding)
    assert identity.answer is not None and identity.answer.label == "walls"
    assert cause.answer is not None and cause.answer.label == "occluded"
    assert motion.answer is not None and motion.answer.label == "left"
    assert identity.frame_vars["t_event"] == cause.frame_vars["t_event"] == 2


def test_occluder_choices_are_deterministic_four_choice_materializations() -> None:
    options = OCCLUDER_IDENTIFICATION.templates[0].options
    first = _materialized_question_options(
        OCCLUDER_IDENTIFICATION, "walls", "family-1", options
    )
    second = _materialized_question_options(
        OCCLUDER_IDENTIFICATION, "walls", "family-1", options
    )
    assert first == second
    assert len(first) == 4
    assert "walls" in first
    assert first[-1] == "无法判断"
