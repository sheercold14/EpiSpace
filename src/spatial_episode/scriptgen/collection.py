"""Deterministic multi-scene planning for a reviewable scriptgen collection."""

from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from .behavior import (
    RenderSceneView,
    layout_from_scene_ir,
    plan_to_agent_views,
    quaternion_xyzw_from_yaw_deg,
)
from .compiler import CapabilityCompiler
from .generate import generate_plans
from .geometry import (
    azimuth_deg,
    bearing_deg,
    distance_m,
    sector_margin_deg,
    sector_of,
    wrap_deg,
)
from .library import REFERENCE_FRAME_SCRIPTS, SCRIPT_LIBRARY
from .motifs import (
    _landmark_chain,
    chain_edge_station_exists,
    imagined_station,
    imagined_station_placement,
)
from .plan import TrajectoryPlan
from .sceneview import GeometrySceneView, Pose2D, SceneLayout, SceneObject
from .spec import ScriptSpec, SpecModel
from .standards import STD_V1, CompileStandard

COLLECTION_SCHEMA_VERSION = "scriptgen_collection.v1"
RENDER_PLAN_SCHEMA_VERSION = "scriptgen_render_plan.v1"
DEFAULT_COLLECTION_ID = "scriptgen_current_2x_v1"
DEFAULT_SEEDS = (17, 29)
MAX_RANKED_OBJECTS = 18
MAX_RANKED_BINDINGS = 128
MAX_VISIT_BINDINGS = 128
REFERENCE_CAPABILITIES = frozenset(script.capability for script in REFERENCE_FRAME_SCRIPTS)

# Both cross-view motifs bind a target–anchor…–other landmark chain and share
# the same binding ranking, source-covisibility prefilter and render-robust
# candidate filter; only the camera choreography differs.
CHAIN_MOTIFS = (("visit_landmarks",), ("snapshot_landmarks",))

# The anchor-frame sector and the closer side are functions of object positions
# alone, so the trajectory search can never rebalance them: whichever labels the
# binding shortlist carries are the labels the collection ships. Ranking chains
# by geometry picks the straightest ones, which pins the anchor question to
# "front" and the closer question to "first". The shortlist is therefore built
# per label stratum and emitted round robin.
#
# The sector is the primary axis because it is always realisable. The closer
# side is secondary because the two are coupled at k=1: with a single anchor the
# sector is the angle at ``other`` in the triangle ``target``-``other``-anchor,
# so a "back" sector forces that angle to be the largest and the target to be
# the farther object. Sectors therefore cycle on the outside and sides
# interleave within a sector, with the starting side rotating so the secondary
# axis still balances wherever the geometry admits both.
CHAIN_SECTORS = ("front", "left", "back", "right")
CHAIN_CLOSER_SIDES = ("first", "second")

# Clearance two chain landmarks must keep beyond their own footprints before
# they count as separate places rather than parts of one fixture. Sized to a
# person's shoulder width so the gap is one a viewer could stand in.
LANDMARK_SEPARATION_MARGIN_M = 0.5

# The P2 curve asks the same binding about eight imagined headings spaced 45
# degrees apart, and the sensor spans 90 degrees, so a heading sees the target
# exactly when the target lies within 45 degrees of it. The shared sector-margin
# qualifier already keeps the target 15 degrees clear of every heading, so the
# target always falls strictly inside one 45-degree wedge and exactly the two
# headings bracketing that wedge report "visible" - never one, never three.
#
# Which two is therefore a property of the binding alone, fixed before any
# trajectory is searched. Ranking by compactness clusters bindings into the
# wedge adjacent to the facing landmark, where the visible pair is always
# {0, +-45}; the rendered curve then answers "visible" at 0 degrees and
# "not visible" at 180, and the rotation named in the question text predicts
# the label without looking at the images. Stratifying the shortlist by wedge
# spreads the visible pair uniformly over the curve, which is what makes the
# offset uninformative.
REFERENCE_VISIBLE_WEDGE_DEG = 45.0


@dataclass(frozen=True)
class SourceRenderEvidence:
    visible_entities: frozenset[str]
    covisible_pairs: frozenset[frozenset[str]]


class CollectionScene(SpecModel):
    scene_key: str
    scene_id: str
    source_scene_id: str
    scene_model: str
    scene_instance: str | None
    source_digest: str
    scene_ir: str
    source_recipe: str
    scene_ir_sha256: str
    source_recipe_sha256: str


class PlanningFailure(SpecModel):
    capability: str
    replicate: int
    seed: int
    scene_key: str
    reason: str
    rejection_counts: dict[str, int] = Field(default_factory=dict)


class CollectionJob(SpecModel):
    job_id: str
    capability: str
    motif: str
    replicate: int
    seed: int
    scene: CollectionScene
    plan_id: str
    frame_count: int
    auxiliary_view_count: int
    plan_record: str
    render_plan: str
    recipe: str
    bundle: str
    question_group: str
    status: Literal["planned", "rendered", "packaged", "failed"] = "planned"


class CollectionManifest(SpecModel):
    schema_version: Literal["scriptgen_collection.v1"] = COLLECTION_SCHEMA_VERSION
    collection_id: str
    standard_version: str
    requested_per_capability: int = Field(ge=1)
    scene_root: str
    output_root: str
    planning_parameters: dict[str, Any]
    jobs: tuple[CollectionJob, ...]
    failures: tuple[PlanningFailure, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload + ("" if payload.endswith("\n") else "\n"), encoding="utf-8")
    temporary.replace(path)


def discover_scenes(scene_root: Path) -> tuple[CollectionScene, ...]:
    """Discover the clean static-m2 scene inventory and its replay recipes."""
    scene_root = scene_root.resolve()
    recipe_root = scene_root.parent / "recipes"
    scenes: list[CollectionScene] = []
    for bundle in sorted(path for path in scene_root.iterdir() if path.is_dir()):
        scene_ir = bundle / "scene_ir.json"
        snapshot = bundle / "scene_snapshot.json"
        source_recipe = recipe_root / f"{bundle.name}.yaml"
        missing = [path for path in (scene_ir, snapshot, source_recipe) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete source scene {bundle.name}: {missing}")
        ir = json.loads(scene_ir.read_text(encoding="utf-8"))
        source = json.loads(snapshot.read_text(encoding="utf-8"))
        scenes.append(
            CollectionScene(
                scene_key=bundle.name,
                scene_id=str(ir["scene_id"]),
                source_scene_id=str(source["source_scene_id"]),
                scene_model=str(source["scene_model"]),
                scene_instance=source.get("scene_instance"),
                source_digest=str(source["source_digest"]),
                scene_ir=str(scene_ir),
                source_recipe=str(source_recipe),
                scene_ir_sha256=_sha256(scene_ir),
                source_recipe_sha256=_sha256(source_recipe),
            )
        )
    if not scenes:
        raise ValueError(f"no source scenes found under {scene_root}")
    return tuple(scenes)


def _stable_rank(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def _eligible_objects(
    layout: SceneLayout,
    script: ScriptSpec,
    *,
    allowed_entity_ids: frozenset[str] | None = None,
) -> dict[str, list[SceneObject]]:
    category_counts = Counter(obj.category for obj in layout.objects)
    candidates: dict[str, list[SceneObject]] = {}
    for slot_name, slot in script.slots.items():
        accepted = [
            obj
            for obj in layout.objects
            if (allowed_entity_ids is None or obj.name in allowed_entity_ids)
            and obj.size_m >= slot.min_size_m
            and (not slot.categories or obj.category in slot.categories)
            and (not slot.unique_referent or category_counts[obj.category] == 1)
        ]
        accepted.sort(key=lambda obj: (-min(obj.size_m, 2.0), obj.category, obj.name))
        candidates[slot_name] = accepted[:MAX_RANKED_OBJECTS]
    return candidates


def _scene_can_bind(layout: SceneLayout, script: ScriptSpec) -> bool:
    if script.capability.startswith("existence_sufficiency_"):
        category = script.capability.removeprefix("existence_sufficiency_")
        if any(obj.category == category for obj in layout.objects):
            return False
    candidates = _eligible_objects(layout, script)
    if any(not values for values in candidates.values()):
        return False
    return len({obj.name for values in candidates.values() for obj in values}) >= len(script.slots)


def _binding_score(script: ScriptSpec, objects: tuple[SceneObject, ...]) -> tuple[Any, ...]:
    if script.capability.startswith("existence_sufficiency_"):
        return (-round(objects[0].size_m, 6), objects[0].name)
    if script.motifs in (("survey",), ("survey_arc",)) and len(objects) >= 3:
        pair_sum = sum(
            distance_m(left.xy, right.xy) for left, right in itertools.combinations(objects, 2)
        )
        return (round(pair_sum, 6), tuple(obj.name for obj in objects))
    if script.motifs in CHAIN_MOTIFS:
        adjacent = [distance_m(left.xy, right.xy) for left, right in itertools.pairwise(objects)]
        return (
            round(sum(adjacent), 6),
            round(max(adjacent), 6),
            -round(distance_m(objects[0].xy, objects[-1].xy), 6),
            tuple(obj.name for obj in objects),
        )
    return tuple(obj.name for obj in objects)


def chain_label_stratum(
    script: ScriptSpec,
    objects: tuple[SceneObject, ...],
    std: CompileStandard,
    *,
    layout: SceneLayout | None = None,
) -> tuple[str, str] | None:
    """Anchor-frame sector and closer side a chain binding is bound to compile to.

    Mirrors ``imagined_sector`` / ``closer_of`` and their search-phase gates.
    Returns ``None`` when either question would be rejected as invalid, so those
    bindings sink below the labelled strata instead of consuming a slot.
    """
    by_slot = dict(zip(script.slots, objects, strict=True))
    anchors = sorted(
        (name for name in script.slots if name.startswith("anchor")),
        key=lambda name: int(name.removeprefix("anchor")),
    )
    if not anchors:
        return None
    target, other = by_slot["target"], by_slot["other"]
    origin = other.xy
    if layout is not None:
        origin = imagined_station(
            layout, other.name, by_slot[anchors[-1]].name, std.camera_height_m
        )
        if origin is None:
            return None
    yaw = bearing_deg(origin, by_slot[anchors[-1]].xy)
    azimuth = azimuth_deg(origin, yaw, target.xy)
    if sector_margin_deg(azimuth) < std.sector_margin_deg:
        return None
    first_anchor_xy = by_slot[anchors[0]].xy
    first_distance = distance_m(target.xy, first_anchor_xy)
    second_distance = distance_m(other.xy, first_anchor_xy)
    smaller, larger = sorted((first_distance, second_distance))
    if larger / max(smaller, 1e-9) < std.closer_min_distance_ratio:
        return None
    return sector_of(azimuth), ("first" if first_distance < second_distance else "second")


def reference_visibility_stratum(
    script: ScriptSpec,
    objects: tuple[SceneObject, ...],
    std: CompileStandard,
    *,
    layout: SceneLayout | None = None,
) -> int | None:
    """Index of the wedge the target occupies around the facing direction.

    Two bindings share an index exactly when the same pair of imagined headings
    can see their targets, so the index is the label the whole shallow curve is
    bound to produce. ``None`` marks a target too close to a wedge boundary for
    the shared sector-margin qualifier, which would reject the binding anyway.
    """
    by_slot = dict(zip(script.slots, objects, strict=True))
    viewpoint, facing, target = by_slot["viewpoint"], by_slot["facing"], by_slot["target"]
    origin = viewpoint.xy
    if layout is not None:
        origin = imagined_station(layout, viewpoint.name, facing.name, std.camera_height_m)
        if origin is None:
            return None
    azimuth = azimuth_deg(origin, bearing_deg(origin, facing.xy), target.xy)
    wedge, remainder = divmod(azimuth % 360.0, REFERENCE_VISIBLE_WEDGE_DEG)
    boundary = min(remainder, REFERENCE_VISIBLE_WEDGE_DEG - remainder)
    if boundary < std.sector_margin_deg:
        return None
    return int(wedge)


def _round_robin_by_stratum(
    strata: dict[Any, list[tuple[Any, tuple[SceneObject, ...]]]],
    maximum: int,
) -> list[tuple[SceneObject, ...]]:
    """Draw one binding per stratum in turn, unlabelled bindings last."""
    queues = [
        heapq.nsmallest(maximum, strata[key], key=lambda item: item[0])
        for key in sorted(key for key in strata if key is not None)
    ]
    selected: list[tuple[SceneObject, ...]] = []
    index = 0
    while len(selected) < maximum and any(index < len(queue) for queue in queues):
        for queue in queues:
            if index < len(queue):
                selected.append(queue[index][1])
                if len(selected) == maximum:
                    return selected
        index += 1
    for _, objects in heapq.nsmallest(
        maximum - len(selected), strata.get(None, ()), key=lambda item: item[0]
    ):
        selected.append(objects)
    return selected


def _interleave(left: list[Any], right: list[Any]) -> list[Any]:
    merged: list[Any] = []
    for first, second in itertools.zip_longest(left, right):
        merged.extend(item for item in (first, second) if item is not None)
    return merged


def _round_robin_strata(
    strata: dict[tuple[str, str] | None, list[tuple[Any, tuple[SceneObject, ...]]]],
    maximum: int,
) -> list[tuple[SceneObject, ...]]:
    """Interleave per-stratum shortlists, unlabelled bindings last."""

    def shortlist(key: tuple[str, str]) -> list[Any]:
        return heapq.nsmallest(maximum, strata.get(key, ()), key=lambda item: item[0])

    queues: list[list[Any]] = []
    for index, sector in enumerate(CHAIN_SECTORS):
        leading, trailing = CHAIN_CLOSER_SIDES if index % 2 == 0 else CHAIN_CLOSER_SIDES[::-1]
        queue = _interleave(shortlist((sector, leading)), shortlist((sector, trailing)))
        if queue:
            queues.append(queue)

    selected: list[tuple[SceneObject, ...]] = []
    index = 0
    while len(selected) < maximum and any(index < len(queue) for queue in queues):
        for queue in queues:
            if index < len(queue):
                selected.append(queue[index][1])
                if len(selected) == maximum:
                    return selected
        index += 1
    for _, objects in heapq.nsmallest(
        maximum - len(selected), strata.get(None, ()), key=lambda item: item[0]
    ):
        selected.append(objects)
    return selected


def ranked_bindings(
    layout: SceneLayout,
    script: ScriptSpec,
    *,
    maximum: int = MAX_RANKED_BINDINGS,
    allowed_entity_ids: frozenset[str] | None = None,
    binding_filter: Callable[[dict[str, str]], bool] | None = None,
    std: CompileStandard = STD_V1,
) -> tuple[dict[str, str], ...]:
    """Return a bounded deterministic binding shortlist for expensive motifs."""
    candidates = _eligible_objects(
        layout,
        script,
        allowed_entity_ids=allowed_entity_ids,
    )
    slot_names = tuple(script.slots)

    def combinations() -> Any:
        for objects in itertools.product(*(candidates[name] for name in slot_names)):
            if len({obj.name for obj in objects}) != len(objects):
                continue
            binding = dict(zip(slot_names, (obj.name for obj in objects), strict=True))
            if binding_filter is None or binding_filter(binding):
                yield objects

    if script.motifs in CHAIN_MOTIFS:
        strata: dict[tuple[str, str] | None, list[tuple[Any, tuple[SceneObject, ...]]]] = {}
        for objects in combinations():
            key = chain_label_stratum(script, objects, std, layout=layout)
            bucket = strata.setdefault(key, [])
            bucket.append((_binding_score(script, objects), objects))
            # Bounded memory over pools that reach seven figures at k=3; the
            # shortlist never needs more than ``maximum`` entries per stratum.
            if len(bucket) >= 4 * maximum:
                strata[key] = heapq.nsmallest(maximum, bucket, key=lambda item: item[0])
        chosen = _round_robin_strata(strata, maximum)
    elif script.capability in REFERENCE_CAPABILITIES:
        wedges: dict[Any, list[tuple[Any, tuple[SceneObject, ...]]]] = {}
        for objects in combinations():
            key = reference_visibility_stratum(script, objects, std, layout=layout)
            bucket = wedges.setdefault(key, [])
            bucket.append((_binding_score(script, objects), objects))
            if len(bucket) >= 4 * maximum:
                wedges[key] = heapq.nsmallest(maximum, bucket, key=lambda item: item[0])
        chosen = _round_robin_by_stratum(wedges, maximum)
    else:
        chosen = [
            objects
            for _, objects in heapq.nsmallest(
                maximum,
                ((_binding_score(script, objects), objects) for objects in combinations()),
                key=lambda item: item[0],
            )
        ]
    return tuple(
        dict(zip(slot_names, (obj.name for obj in objects), strict=True)) for objects in chosen
    )


def _generate_one(
    layout: SceneLayout,
    script: ScriptSpec,
    std: CompileStandard,
    *,
    seed: int,
    source_render_evidence: SourceRenderEvidence | None = None,
) -> tuple[TrajectoryPlan | None, dict[str, int]]:
    if len(script.slots) == 1:
        candidate_bindings = None
        attempts_per_binding = 80
        if script.capability.startswith("existence_sufficiency_"):
            candidate_bindings = ranked_bindings(
                layout,
                script,
                maximum=1,
            )
            attempts_per_binding = 1
        report = generate_plans(
            layout,
            script,
            std,
            seed=seed,
            attempts_per_binding=attempts_per_binding,
            max_plans=1,
            candidate_bindings=candidate_bindings,
        )
        return (report.plans[0] if report.plans else None), report.rejection_counts

    aggregate: Counter[str] = Counter()
    if script.capability in REFERENCE_CAPABILITIES:
        candidates = []
        for binding in ranked_bindings(layout, script, std=std):
            if _reference_binding_eligible(layout, binding, std):
                candidates.append(binding)
            else:
                aggregate["reference_binding_ineligible"] += 1
        attempts = 12
    else:
        candidates = list(
            ranked_bindings(
                layout,
                script,
                maximum=MAX_VISIT_BINDINGS,
                allowed_entity_ids=(
                    source_render_evidence.visible_entities
                    if source_render_evidence is not None
                    else None
                ),
                binding_filter=(
                    _chain_binding_filter(layout, script, source_render_evidence)
                    if source_render_evidence is not None
                    else None
                ),
            )
        )
        attempts = 20
    report = generate_plans(
        layout,
        script,
        std,
        seed=seed,
        attempts_per_binding=attempts,
        max_plans=1,
        candidate_bindings=candidates,
        candidate_filter=(_visit_render_robust_filter if script.motifs in CHAIN_MOTIFS else None),
    )
    aggregate.update(report.rejection_counts)
    if not report.plans:
        return None, dict(aggregate)
    plan = report.plans[0]
    if plan.capability in REFERENCE_CAPABILITIES:
        poses = tuple(Pose2D(pose.x, pose.y, pose.yaw_deg) for pose in plan.poses)
        view = GeometrySceneView(layout, poses, std)
        for companion in REFERENCE_FRAME_SCRIPTS:
            certificate = CapabilityCompiler(companion, std).compile(view, plan.binding)
            if certificate.status != "answerable":
                key = f"{companion.capability}:{certificate.status}:{certificate.reason}"
                aggregate[f"reference_group:{key}"] += 1
                return None, dict(aggregate)
    return plan, dict(aggregate)


def _visit_render_robust_filter(
    view: GeometrySceneView,
    binding: dict[str, str],
) -> str | None:
    """Reject query pairs whose image extents can overlap in any frame.

    The semantic clause still uses normal visibility, including occlusion.
    Collection planning is stricter because conservative scene boxes can claim
    an occlusion that the renderer does not reproduce. This angular check uses
    the same object-extent and field-of-view model as ``GeometrySceneView`` but
    deliberately does not use synthetic occlusion.
    """
    target = view.object(binding["target"])
    other = view.object(binding["other"])
    for pose in view.poses:
        potential = []
        for obj in (target, other):
            distance = distance_m(pose.xy, obj.xy)
            half_width = (
                180.0 if distance < 1e-6 else math.degrees(math.atan2(obj.size_m / 2.0, distance))
            )
            potential.append(
                abs(azimuth_deg(pose.xy, pose.yaw_deg, obj.xy))
                <= (view.std.fov_half_angle_deg * view.std.search_tighten_factor + half_width)
            )
        if all(potential):
            return "queried_pair_angular_overlap"
    return None


def _source_render_evidence(
    scene: CollectionScene,
    std: CompileStandard,
) -> SourceRenderEvidence:
    """Entity and pair visibility proven by the immutable static source render."""
    source_bundle = Path(scene.scene_ir).parent
    view = RenderSceneView.from_bundle(source_bundle, std, scene_ir=scene.scene_ir)
    visible_frames = {
        obj.name: frozenset(
            frame
            for frame in range(view.frame_count)
            if view.visibility(obj.name, frame).tristate(std) is True
        )
        for obj in view.layout.objects
    }
    visible_entities = frozenset(name for name, frames in visible_frames.items() if frames)
    covisible_pairs = frozenset(
        frozenset((left, right))
        for left, right in itertools.combinations(visible_entities, 2)
        if len(visible_frames[left] & visible_frames[right]) >= std.chain_min_covisible_frames
    )
    return SourceRenderEvidence(
        visible_entities=visible_entities,
        covisible_pairs=covisible_pairs,
    )


def _source_visible_entities(
    scene: CollectionScene,
    std: CompileStandard,
) -> frozenset[str]:
    """Compatibility helper for source render visibility diagnostics."""
    return _source_render_evidence(scene, std).visible_entities


def _binding_chain_source_covisible(
    binding: dict[str, str],
    evidence: SourceRenderEvidence,
) -> bool:
    chain = _landmark_chain(binding)
    return all(
        frozenset((left, right)) in evidence.covisible_pairs
        for left, right in itertools.pairwise(chain)
    )


def _binding_chain_stations_framable(layout: SceneLayout, binding: dict[str, str]) -> bool:
    """Every chain edge admits some snapshot station under motif constraints.

    Snapshot stations are searched over the whole free grid, so the static
    source survey's covisible pairs are far too strict a proxy for them.  Pair
    framability alone is too loose the other way: short chains keep the
    queried pair close together, and the never-covisible exclusion then
    empties station pools that pure pair geometry admits.  Judging each edge
    with the motif's own station admission keeps ranked bindings realisable.
    """
    chain = _landmark_chain(binding)
    edges = tuple(itertools.pairwise(chain))
    return all(
        chain_edge_station_exists(
            layout,
            left,
            right,
            binding["target"],
            binding["other"],
            final_edge=index == len(edges) - 1,
        )
        for index, (left, right) in enumerate(edges)
    )


def _chain_slots_are_distinct_places(layout: SceneLayout, binding: dict[str, str]) -> bool:
    """Every pair of bound landmarks names a separate place in the room.

    Scene inventories list stacked and built-in fixtures as independent
    entities: an oven, the microwave above it and the fridge beside it are
    three names for one kitchen block, within centimetres of each other in
    plan view. A chain over them still passes every angular gate, but the
    questions it produces are degenerate - the closer question compares a
    distance to itself, and the anchor question asks for a direction that is
    only defined up to the extent of a single appliance. Requiring the two
    footprints to clear each other by a walkable margin drops exactly those
    bindings and leaves genuinely separated landmarks untouched.
    """
    objects = tuple(layout.object(name) for name in binding.values())
    return all(
        distance_m(left.xy, right.xy)
        >= 0.5 * (left.size_m + right.size_m) + LANDMARK_SEPARATION_MARGIN_M
        for left, right in itertools.combinations(objects, 2)
    )


def _chain_binding_filter(
    layout: SceneLayout,
    script: ScriptSpec,
    evidence: SourceRenderEvidence,
) -> Callable[[dict[str, str]], bool]:
    """Prefilter chain bindings by the evidence a chain edge actually needs."""
    if script.motifs == ("snapshot_landmarks",):
        return lambda binding: (
            _chain_slots_are_distinct_places(layout, binding)
            and _binding_chain_stations_framable(layout, binding)
        )
    return lambda binding: (
        _chain_slots_are_distinct_places(layout, binding)
        and _binding_chain_source_covisible(binding, evidence)
    )


def _reference_binding_eligible(
    layout: SceneLayout,
    binding: dict[str, str],
    std: CompileStandard,
) -> bool:
    """Cheap binding-only prefilter for the shared P2 curve contract.

    All reference-frame specs share their trajectory clauses. Their two
    binding-only qualifiers each cover the complete registered yaw curve, so
    a binding that fails either can never become valid by resampling a survey.
    Final acceptance still recompiles every spec through the normal checker.
    """
    from .predicates import (
        imagined_curve_sector_margins_ge,
        imagined_curve_visibility_decisive,
        imagined_pose_valid,
    )

    station = imagined_station(layout, binding["viewpoint"], binding["facing"], std.camera_height_m)
    if station is None:
        return False
    probe = GeometrySceneView(layout, (Pose2D(station[0], station[1], 0.0),), std)
    args = {
        "viewpoint": binding["viewpoint"],
        "facing": binding["facing"],
        "obj": binding["target"],
    }
    return bool(
        imagined_pose_valid(
            probe,
            std,
            viewpoint=binding["viewpoint"],
            facing=binding["facing"],
        ).holds
        and imagined_curve_sector_margins_ge(probe, std, **args).holds
        and imagined_curve_visibility_decisive(probe, std, **args).holds
        and _reference_render_robust_filter(layout, binding, std)
    )


def _reference_render_robust_filter(
    layout: SceneLayout,
    binding: dict[str, str],
    std: CompileStandard,
) -> bool:
    """Do not license invisibility solely from conservative proxy occlusion."""
    station = imagined_station(layout, binding["viewpoint"], binding["facing"], std.camera_height_m)
    if station is None:
        return False
    facing = layout.object(binding["facing"])
    target = layout.object(binding["target"])
    base_yaw = bearing_deg(station, facing.xy)
    for offset in std.imagined_viewpoint_offsets_deg:
        pose = Pose2D(station[0], station[1], wrap_deg(base_yaw + offset))
        observation = GeometrySceneView(layout, (pose,), std).visibility(target.name, 0)
        if observation.tristate(std) is not False:
            continue
        distance = distance_m(pose.xy, target.xy)
        half_width = (
            180.0 if distance < 1e-6 else math.degrees(math.atan2(target.size_m / 2.0, distance))
        )
        if abs(azimuth_deg(pose.xy, pose.yaw_deg, target.xy)) <= (
            std.fov_half_angle_deg * std.search_tighten_factor + half_width
        ):
            return False
    return True


def _aux_view_id(offset: int) -> str:
    """Identifier-safe auxiliary view name: negative offsets use an "m" marker."""
    if offset < 0:
        return f"aux-imagined-yawm{-offset:03d}"
    return f"aux-imagined-yaw{offset:03d}"


def _auxiliary_views(
    plan: TrajectoryPlan,
    layout: SceneLayout,
    std: CompileStandard,
) -> list[dict[str, Any]]:
    if plan.capability not in REFERENCE_CAPABILITIES:
        return []
    placement = imagined_station_placement(
        layout, plan.binding["viewpoint"], plan.binding["facing"], std.camera_height_m
    )
    if placement is None:
        raise ValueError(f"no imagined station for {plan.plan_id}")
    station = placement.xy
    facing = layout.object(plan.binding["facing"])
    target = layout.object(plan.binding["target"])
    base_yaw = bearing_deg(station, facing.xy)
    views: list[dict[str, Any]] = []
    for offset in std.imagined_viewpoint_offsets_deg:
        yaw = wrap_deg(base_yaw + offset)
        pose = Pose2D(station[0], station[1], yaw)
        observation = GeometrySceneView(layout, (pose,), std).visibility(target.name, 0)
        state = observation.tristate(std)
        if state is None:
            raise ValueError(f"ambiguous auxiliary view for {plan.plan_id} yaw={offset}")
        views.append(
            {
                "view_id": _aux_view_id(offset),
                "purpose": "imagined_visibility_render_check",
                "target_entity_id": target.name,
                "yaw_offset_deg": offset,
                "geometry_label": "visible" if state else "not_visible",
                "camera_height_m": std.camera_height_m,
                "station_method": placement.method,
                "station_offset_m": placement.offset_m,
                "station_heading_shift_deg": placement.heading_shift_deg,
                "station_surface_standoff_m": placement.surface_standoff_m,
                "world_from_agent": {
                    "parent_frame": "world",
                    "child_frame": f"agent:{_aux_view_id(offset)}",
                    "convention": "active_child_to_parent",
                    "translation_m": [pose.x, pose.y, 0.0],
                    "rotation_xyzw": list(quaternion_xyzw_from_yaw_deg(pose.yaw_deg)),
                },
            }
        )
    return views


def render_plan_payload(
    plan: TrajectoryPlan,
    layout: SceneLayout,
    std: CompileStandard,
) -> dict[str, Any]:
    # Snapshot motifs teleport between stations: only the station poses exist
    # physically, so the renderer must not clearance-check interpolated
    # segments between them. Walked plans keep the default full-path contract.
    snapshot = SCRIPT_LIBRARY[plan.capability].motifs == ("snapshot_landmarks",)
    return {
        "schema_version": RENDER_PLAN_SCHEMA_VERSION,
        "plan_id": plan.plan_id,
        "capability": plan.capability,
        "standard_version": plan.standard_version,
        "path_contract": "teleport_cuts" if snapshot else "walked",
        "views": plan_to_agent_views(plan),
        "auxiliary_views": _auxiliary_views(plan, layout, std),
        "clearance": {
            "body_radius_m": std.body_radius_m,
            "z_low_m": std.clearance_z_low_m,
            "z_high_m": std.clearance_z_high_m,
        },
    }


def _write_recipe(
    source: CollectionScene,
    collection_id: str,
    job_id: str,
    motif: str,
    seed: int,
    render_plan: Path,
    out: Path,
) -> None:
    payload = yaml.safe_load(Path(source.source_recipe).read_text(encoding="utf-8"))
    payload["recipe_id"] = f"{collection_id}_{job_id}"
    payload["seed"] = seed
    trajectory = payload["trajectory"]
    trajectory["sampling_strategy"] = "scripted_plan"
    # The acquisition recipe schema retains the legacy T1--T10 enum. Scripted
    # plans carry the real motif/capability in their own versioned payload.
    trajectory["trajectory_class"] = "T1"
    trajectory["plan_path"] = str(render_plan.resolve())
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    temporary.replace(out)


def _manifest_path(output_root: Path) -> Path:
    return output_root / "collection.plan.json"


def plan_collection(
    scene_root: Path,
    output_root: Path,
    *,
    collection_id: str = DEFAULT_COLLECTION_ID,
    per_capability: int = 2,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    std: CompileStandard = STD_V1,
) -> Path:
    """Plan exactly ``per_capability`` trajectories for every registered capability."""
    if len(seeds) < per_capability:
        raise ValueError("one deterministic seed is required per capability replicate")
    output_root = output_root.resolve()
    scenes = discover_scenes(scene_root)
    layouts: dict[str, SceneLayout] = {}
    source_evidence: dict[str, SourceRenderEvidence] = {}
    usage: Counter[str] = Counter()
    jobs: list[CollectionJob] = []
    failures: list[PlanningFailure] = []
    existing_path = _manifest_path(output_root)
    if existing_path.is_file():
        existing = load_collection(existing_path)
        if (
            existing.collection_id != collection_id
            or existing.standard_version != std.standard_version
            or existing.requested_per_capability != per_capability
            or tuple(existing.planning_parameters.get("seeds", ())) != tuple(seeds[:per_capability])
        ):
            raise ValueError(f"existing collection settings differ: {existing_path}")
        jobs.extend(existing.jobs)
        failures.extend(existing.failures)
        usage.update(job.scene.scene_key for job in jobs)

    def save() -> Path:
        manifest = CollectionManifest(
            collection_id=collection_id,
            standard_version=std.standard_version,
            requested_per_capability=per_capability,
            scene_root=str(Path(scene_root).resolve()),
            output_root=str(output_root),
            planning_parameters={
                "seeds": list(seeds[:per_capability]),
                "maximum_ranked_objects": MAX_RANKED_OBJECTS,
                "maximum_ranked_bindings": MAX_RANKED_BINDINGS,
                "maximum_visit_bindings": MAX_VISIT_BINDINGS,
                "single_slot_attempts": 80,
                "existence_maximum_ranked_bindings": 1,
                "existence_attempts_per_binding": 1,
                "survey_attempts_per_binding": 12,
                "visit_attempts_per_binding": 20,
                "visit_requires_source_render_visibility": True,
                "visit_requires_source_render_chain_covisibility": True,
                "visit_rejects_query_angular_overlap": True,
            },
            jobs=tuple(jobs),
            failures=tuple(failures),
        )
        path = _manifest_path(output_root)
        _write_json(path, manifest.model_dump_json(indent=2))
        return path

    for capability, script in SCRIPT_LIBRARY.items():
        used_for_capability = {job.scene.scene_key for job in jobs if job.capability == capability}
        for replicate in range(per_capability):
            if any(job.capability == capability and job.replicate == replicate for job in jobs):
                continue
            seed = seeds[replicate]
            attempted = {
                failure.scene_key
                for failure in failures
                if failure.capability == capability and failure.replicate == replicate
            }
            if capability in REFERENCE_CAPABILITIES:
                # All reference-frame specs are required to qualify on the
                # same binding. With the same seed, a fully exhausted scene is
                # therefore a deterministic failure for every sibling source.
                attempted.update(
                    failure.scene_key
                    for failure in failures
                    if failure.capability in REFERENCE_CAPABILITIES
                    and failure.seed == seed
                    and failure.reason in {"scene_ineligible", "generation_exhausted"}
                )

            def scene_rank(
                scene: CollectionScene,
                capability: str = capability,
                replicate: int = replicate,
                script: ScriptSpec = script,
            ) -> tuple[int | str, int | str]:
                stable = _stable_rank(collection_id, capability, replicate, scene.scene_key)
                if script.motifs in CHAIN_MOTIFS:
                    return stable, usage[scene.scene_key]
                return usage[scene.scene_key], stable

            candidates = sorted(
                (
                    scene
                    for scene in scenes
                    if scene.scene_key not in used_for_capability
                    and scene.scene_key not in attempted
                ),
                key=scene_rank,
            )
            accepted = False
            for scene in candidates:
                layout = layouts.get(scene.scene_key)
                if layout is None:
                    layout = layout_from_scene_ir(scene.scene_ir, std=std)
                    layouts[scene.scene_key] = layout
                if not _scene_can_bind(layout, script):
                    failures.append(
                        PlanningFailure(
                            capability=capability,
                            replicate=replicate,
                            seed=seed,
                            scene_key=scene.scene_key,
                            reason="scene_ineligible",
                        )
                    )
                    continue
                render_evidence = None
                if script.motifs in CHAIN_MOTIFS:
                    render_evidence = source_evidence.get(scene.scene_key)
                    if render_evidence is None:
                        render_evidence = _source_render_evidence(scene, std)
                        source_evidence[scene.scene_key] = render_evidence
                plan, rejection_counts = _generate_one(
                    layout,
                    script,
                    std,
                    seed=seed,
                    source_render_evidence=render_evidence,
                )
                if plan is None:
                    failures.append(
                        PlanningFailure(
                            capability=capability,
                            replicate=replicate,
                            seed=seed,
                            scene_key=scene.scene_key,
                            reason="generation_exhausted",
                            rejection_counts=rejection_counts,
                        )
                    )
                    save()
                    continue

                job_id = f"{capability}__r{replicate}__{scene.scene_key}"
                plan_record = output_root / "plans" / f"{job_id}.record.json"
                render_plan = output_root / "plans" / f"{job_id}.views.json"
                recipe = output_root / "recipes" / f"{job_id}.yaml"
                bundle = output_root / "bundles" / job_id
                question_group = output_root / "groups" / job_id
                render_payload = render_plan_payload(plan, layout, std)
                _write_json(plan_record, plan.model_dump_json(indent=2))
                _write_json(render_plan, json.dumps(render_payload, ensure_ascii=False, indent=2))
                _write_recipe(
                    scene,
                    collection_id,
                    job_id,
                    script.motifs[0],
                    seed,
                    render_plan,
                    recipe,
                )
                jobs.append(
                    CollectionJob(
                        job_id=job_id,
                        capability=capability,
                        motif=script.motifs[0],
                        replicate=replicate,
                        seed=seed,
                        scene=scene,
                        plan_id=plan.plan_id,
                        frame_count=len(plan.poses),
                        auxiliary_view_count=len(render_payload["auxiliary_views"]),
                        plan_record=str(plan_record),
                        render_plan=str(render_plan),
                        recipe=str(recipe),
                        bundle=str(bundle),
                        question_group=str(question_group),
                    )
                )
                usage[scene.scene_key] += 1
                used_for_capability.add(scene.scene_key)
                save()
                print(
                    f"planned {len(jobs)}/{len(SCRIPT_LIBRARY) * per_capability} "
                    f"capability={capability} replicate={replicate} scene={scene.scene_key} "
                    f"frames={len(plan.poses)} auxiliary={len(render_payload['auxiliary_views'])}",
                    flush=True,
                )
                accepted = True
                break
            if not accepted:
                save()
                raise RuntimeError(
                    f"unable to plan {capability} replicate {replicate} in a distinct scene"
                )
    return save()


def load_collection(path: Path) -> CollectionManifest:
    return CollectionManifest.model_validate_json(path.read_text(encoding="utf-8"))


def refresh_recipes(manifest: CollectionManifest) -> int:
    """Regenerate replay recipes from their immutable source recipes."""
    for job in manifest.jobs:
        _write_recipe(
            job.scene,
            manifest.collection_id,
            job.job_id,
            job.motif,
            job.seed,
            Path(job.render_plan),
            Path(job.recipe),
        )
    return len(manifest.jobs)


def verify_source_hashes(manifest: CollectionManifest) -> None:
    """Fail if any read-only source artifact changed since planning."""
    scenes = {job.scene.scene_key: job.scene for job in manifest.jobs}
    for scene in scenes.values():
        if _sha256(Path(scene.scene_ir)) != scene.scene_ir_sha256:
            raise ValueError(f"source scene_ir changed: {scene.scene_key}")
        if _sha256(Path(scene.source_recipe)) != scene.source_recipe_sha256:
            raise ValueError(f"source recipe changed: {scene.scene_key}")


def reject_jobs(
    manifest_path: Path,
    rejected: dict[str, str],
) -> int:
    """Return post-render failures to the planner without deleting evidence."""
    manifest = load_collection(manifest_path)
    known = {job.job_id: job for job in manifest.jobs}
    unknown = sorted(set(rejected) - set(known))
    if unknown:
        raise KeyError(f"unknown collection jobs: {unknown}")
    failures = list(manifest.failures)
    for job_id, reason in rejected.items():
        job = known[job_id]
        failures.append(
            PlanningFailure(
                capability=job.capability,
                replicate=job.replicate,
                seed=job.seed,
                scene_key=job.scene.scene_key,
                reason=f"post_render_rejected:{reason}",
            )
        )
    updated = manifest.model_copy(
        update={
            "jobs": tuple(job for job in manifest.jobs if job.job_id not in rejected),
            "failures": tuple(failures),
        }
    )
    _write_json(manifest_path, updated.model_dump_json(indent=2))
    return len(rejected)
