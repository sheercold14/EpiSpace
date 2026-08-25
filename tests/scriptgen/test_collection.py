"""Batch collection planning contracts and deterministic selection."""

from __future__ import annotations

import json
import math
from collections import Counter
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

from spatial_episode.scriptgen.collection import (
    CollectionJob,
    CollectionManifest,
    SourceRenderEvidence,
    _binding_chain_source_covisible,
    _binding_score,
    _chain_slots_are_distinct_places,
    _reference_binding_eligible,
    _reference_render_robust_filter,
    _visit_render_robust_filter,
    chain_label_stratum,
    discover_scenes,
    load_collection,
    ranked_bindings,
    reference_visibility_stratum,
    reject_jobs,
    render_plan_payload,
)
from spatial_episode.scriptgen.library import (
    CROSS_VIEW_RELATION,
    CROSS_VIEW_SNAPSHOT_CLOSER,
    CROSS_VIEW_SNAPSHOT_EGO,
    EXISTENCE_SUFFICIENCY,
    REFERENCE_FRAME_DEEP,
)
from spatial_episode.scriptgen.motifs import (
    IMAGINED_STATION_MAX_HEADING_SHIFT_DEG,
    imagined_station_placement,
)
from spatial_episode.scriptgen.plan import PlannedPose, ProvisionalAnswer, TrajectoryPlan
from spatial_episode.scriptgen.rendering import rendered_bundle_complete
from spatial_episode.scriptgen.sceneview import (
    GeometrySceneView,
    Obstacle,
    Pose2D,
    SceneLayout,
    SceneObject,
)
from spatial_episode.scriptgen.slotting import iter_bindings
from spatial_episode.scriptgen.standards import STD_V1


def _object(name: str, category: str, x: float, y: float) -> SceneObject:
    return SceneObject(name, category, (x, y), 0.8, name)


def test_binding_iteration_can_be_bounded_without_materialising_product() -> None:
    objects = tuple(_object(f"o{i}", f"c{i}", float(i), 0.0) for i in range(12))
    layout = SceneLayout("many", objects)
    bindings, rejections = iter_bindings(layout, CROSS_VIEW_RELATION[2], maximum=7)
    assert len(list(bindings)) == 7
    assert not rejections


def test_ranked_landmark_bindings_are_bounded_and_deterministic() -> None:
    layout = SceneLayout(
        "ranking",
        tuple(_object(f"o{i}", f"c{i}", float(i % 3), float(i // 3)) for i in range(8)),
    )
    first = ranked_bindings(layout, CROSS_VIEW_RELATION[2], maximum=5)
    second = ranked_bindings(layout, CROSS_VIEW_RELATION[2], maximum=5)
    assert first == second
    assert len(first) == 5
    assert all(len(set(binding.values())) == 5 for binding in first)


def test_ranked_bindings_filter_objects_before_shortlisting() -> None:
    layout = SceneLayout(
        "visible-ranking",
        tuple(_object(f"o{i}", f"c{i}", float(i), 0.0) for i in range(8)),
    )
    allowed = frozenset({"o3", "o4", "o5", "o6", "o7"})

    bindings = ranked_bindings(
        layout,
        CROSS_VIEW_RELATION[2],
        maximum=4,
        allowed_entity_ids=allowed,
    )

    assert len(bindings) == 4
    assert all(set(binding.values()) <= allowed for binding in bindings)


def test_ranked_bindings_apply_pair_filter_before_shortlisting() -> None:
    layout = SceneLayout(
        "pair-ranking",
        tuple(_object(f"o{i}", f"c{i}", float(i), 0.0) for i in range(8)),
    )

    bindings = ranked_bindings(
        layout,
        CROSS_VIEW_RELATION[2],
        maximum=4,
        binding_filter=lambda binding: binding["target"] in {"o6", "o7"},
    )

    assert len(bindings) == 4
    assert all(binding["target"] in {"o6", "o7"} for binding in bindings)


def test_ranked_walking_bindings_grow_only_along_evidence_graph_edges() -> None:
    layout = SceneLayout(
        "evidence-graph",
        tuple(_object(f"o{i}", "repeated", float(i), 0.0) for i in range(8)),
    )
    adjacency = frozenset(frozenset((f"o{i}", f"o{i + 1}")) for i in range(7))

    bindings = ranked_bindings(
        layout,
        CROSS_VIEW_RELATION[2],
        maximum=32,
        chain_adjacency=adjacency,
    )

    assert bindings
    for binding in bindings:
        chain = [binding[slot] for slot in CROSS_VIEW_RELATION[2].slots]
        assert all(frozenset(pair) in adjacency for pair in pairwise(chain))


def test_marker_pool_accepts_repeated_categories_and_drops_giant_structures() -> None:
    layout = SceneLayout(
        "marker-pool",
        (
            _object("chair1", "chair", 0.0, 0.0),
            _object("chair2", "chair", 2.0, 0.0),
            _object("table", "table", 4.0, 0.0),
            SceneObject("fence", "fence", (6.0, 0.0), 12.0, "fence"),
        ),
    )

    bindings = ranked_bindings(layout, REFERENCE_FRAME_DEEP[0], maximum=20)

    assert bindings
    assert any({"chair1", "chair2"} <= set(binding.values()) for binding in bindings)
    assert all("fence" not in binding.values() for binding in bindings)


def test_existence_binding_ranking_prefers_large_targets() -> None:
    layout = SceneLayout(
        "existence-ranking",
        (
            SceneObject("small", "chair", (0.0, 0.0), 0.4, "small"),
            SceneObject("large", "table", (1.0, 0.0), 1.2, "large"),
        ),
    )

    bindings = ranked_bindings(layout, EXISTENCE_SUFFICIENCY, maximum=1)

    assert bindings == ({"target": "large"},)


def test_visit_render_filter_rejects_angular_overlap_without_using_occlusion() -> None:
    layout = SceneLayout(
        "overlap",
        (
            _object("target", "target_cat", 4.0, 0.5),
            _object("other", "other_cat", 4.0, -0.5),
        ),
    )
    overlapping = GeometrySceneView(layout, (Pose2D(0.0, 0.0, 0.0),), STD_V1)
    separated = GeometrySceneView(layout, (Pose2D(0.0, 0.0, 90.0),), STD_V1)
    binding = {"target": "target", "other": "other"}

    assert _visit_render_robust_filter(overlapping, binding) == ("queried_pair_angular_overlap")
    assert _visit_render_robust_filter(separated, binding) is None


def test_source_render_evidence_must_cover_every_chain_edge() -> None:
    binding = {
        "target": "target",
        "anchor1": "anchor1",
        "anchor2": "anchor2",
        "other": "other",
    }
    complete = SourceRenderEvidence(
        visible_entities=frozenset(binding.values()),
        covisible_pairs=frozenset(
            {
                frozenset(("target", "anchor1")),
                frozenset(("anchor1", "anchor2")),
                frozenset(("anchor2", "other")),
            }
        ),
    )
    missing_middle = SourceRenderEvidence(
        visible_entities=complete.visible_entities,
        covisible_pairs=complete.covisible_pairs - {frozenset(("anchor1", "anchor2"))},
    )

    assert _binding_chain_source_covisible(binding, complete)
    assert not _binding_chain_source_covisible(binding, missing_middle)


def test_reference_render_filter_rejects_proxy_only_occlusion() -> None:
    objects = (
        _object("viewpoint", "viewpoint_cat", 0.0, 0.0),
        _object("facing", "facing_cat", 2.0, 0.0),
        _object("target", "target_cat", 4.0, 0.0),
    )
    binding = {"viewpoint": "viewpoint", "facing": "facing", "target": "target"}
    clear = SceneLayout("clear", objects)
    blocked = SceneLayout(
        "blocked",
        objects,
        occlusion_obstacles=(Obstacle("wall", (2.0, 0.0), (0.1, 1.0), 0.0, 0.0, 2.0),),
    )

    assert _reference_render_robust_filter(clear, binding, STD_V1)
    assert not _reference_render_robust_filter(blocked, binding, STD_V1)


def test_reference_binding_reserves_auxiliary_camera_probe_radius() -> None:
    angle = math.radians(20.0)
    objects = (
        _object("viewpoint", "viewpoint_cat", 0.0, 0.0),
        _object("facing", "facing_cat", 2.0, 0.0),
        _object("target", "target_cat", 4.0 * math.cos(angle), 4.0 * math.sin(angle)),
    )
    binding = {"viewpoint": "viewpoint", "facing": "facing", "target": "target"}
    clear = SceneLayout("clear", objects)
    probe_collision = SceneLayout(
        "probe_collision",
        objects,
        obstacles=(Obstacle("cabinet", (0.14, 0.0), (0.1, 0.1), 0.0, 1.4, 1.6),),
    )

    assert _reference_binding_eligible(clear, binding, STD_V1)
    assert not _reference_binding_eligible(probe_collision, binding, STD_V1)


def _tall_reference_layout(*, block_facing_ray: bool) -> SceneLayout:
    objects = (
        SceneObject(
            "viewpoint",
            "locker",
            (0.0, 0.0),
            1.0,
            "viewpoint",
            center_z=1.0,
            half_height=1.0,
        ),
        _object("facing", "chair", 6.0, 0.0),
        _object("target", "table", -0.7680421975, 1.4850641810),
    )
    obstacles = [
        Obstacle(
            "viewpoint",
            (0.0, 0.0),
            (0.5, 0.5),
            0.0,
            0.0,
            2.0,
            "viewpoint",
        )
    ]
    if block_facing_ray:
        obstacles.append(Obstacle("blocker", (1.1, 0.0), (0.2, 0.25), 0.0, 0.0, 2.0, "blocker"))
    return SceneLayout(
        "tall-reference",
        objects,
        obstacles=tuple(obstacles),
        walkable_min=(-3.0, -3.0),
        walkable_max=(8.0, 3.0),
    )


def test_imagined_station_keeps_ray_then_uses_constrained_perimeter() -> None:
    ray = imagined_station_placement(
        _tall_reference_layout(block_facing_ray=False),
        "viewpoint",
        "facing",
        STD_V1.camera_height_m,
    )
    perimeter = imagined_station_placement(
        _tall_reference_layout(block_facing_ray=True),
        "viewpoint",
        "facing",
        STD_V1.camera_height_m,
    )

    assert ray is not None and ray.method == "facing_ray"
    assert ray.xy[1] == 0.0
    assert perimeter is not None and perimeter.method == "footprint_perimeter"
    assert perimeter.xy[0] >= 0.0  # facing-side half of the footprint only
    assert abs(perimeter.xy[1]) > 0.0
    assert perimeter.surface_standoff_m <= 0.6
    assert perimeter.heading_shift_deg <= IMAGINED_STATION_MAX_HEADING_SHIFT_DEG
    assert perimeter == imagined_station_placement(
        _tall_reference_layout(block_facing_ray=True),
        "viewpoint",
        "facing",
        STD_V1.camera_height_m,
    )


def test_reference_visibility_stratum_uses_the_final_station() -> None:
    layout = _tall_reference_layout(block_facing_ray=True)
    objects = tuple(layout.object(name) for name in ("viewpoint", "facing", "target"))
    centre_stratum = reference_visibility_stratum(REFERENCE_FRAME_DEEP[0], objects, STD_V1)
    station_stratum = reference_visibility_stratum(
        REFERENCE_FRAME_DEEP[0], objects, STD_V1, layout=layout
    )
    placement = imagined_station_placement(layout, "viewpoint", "facing", STD_V1.camera_height_m)

    assert placement is not None and placement.method == "footprint_perimeter"
    assert centre_stratum == 2
    assert station_stratum == 3


def test_reference_render_plan_keeps_auxiliary_views_outside_sequence() -> None:
    angle = math.radians(20.0)
    layout = SceneLayout(
        "reference",
        (
            _object("viewpoint", "viewpoint_cat", 0.0, 0.0),
            _object("facing", "facing_cat", 2.0, 0.0),
            _object("target", "target_cat", 4.0 * math.cos(angle), 4.0 * math.sin(angle)),
        ),
        walkable_min=(-5.0, -5.0),
        walkable_max=(5.0, 5.0),
    )
    poses = tuple(PlannedPose(frame=i, x=1.0, y=1.0, yaw_deg=float(i)) for i in range(14))
    plan = TrajectoryPlan(
        plan_id="reference.plan",
        scene_id=layout.scene_id,
        capability=REFERENCE_FRAME_DEEP[0].capability,
        standard_version=STD_V1.standard_version,
        seed=17,
        binding={"viewpoint": "viewpoint", "facing": "facing", "target": "target"},
        frame_vars={"t_q": 13},
        poses=poses,
        knob_levels={"imagined_yaw_offset_deg": 0.0},
        clause_witnesses={},
        provisional_answer=ProvisionalAnswer(mode="imagined_sector", label="front", witness={}),
    )

    payload = render_plan_payload(plan, layout, STD_V1)

    assert len(payload["views"]) == 14
    assert [view["step"] for view in payload["views"]] == list(range(14))
    assert len(payload["auxiliary_views"]) == 8
    assert [view["yaw_offset_deg"] for view in payload["auxiliary_views"]] == [
        0,
        45,
        90,
        135,
        180,
        -45,
        -90,
        -135,
    ]
    assert [view["view_id"] for view in payload["auxiliary_views"][4:]] == [
        "aux-imagined-yaw180",
        "aux-imagined-yawm045",
        "aux-imagined-yawm090",
        "aux-imagined-yawm135",
    ]
    assert payload["clearance"] == {
        "body_radius_m": 0.3,
        "z_low_m": 0.1,
        "z_high_m": 1.7,
    }
    assert payload["path_contract"] == "walked"


def test_snapshot_render_plan_declares_teleport_cut_path_contract() -> None:
    layout = SceneLayout(
        "snapshot",
        (
            _object("target", "target_cat", 0.0, 0.0),
            _object("other", "other_cat", 2.0, 0.0),
        ),
        walkable_min=(-5.0, -5.0),
        walkable_max=(5.0, 5.0),
    )
    plan = TrajectoryPlan(
        plan_id="snapshot.plan",
        scene_id=layout.scene_id,
        capability=CROSS_VIEW_SNAPSHOT_EGO[0].capability,
        standard_version=STD_V1.standard_version,
        seed=17,
        binding={"target": "target", "other": "other"},
        frame_vars={},
        poses=tuple(PlannedPose(frame=i, x=1.0, y=float(i), yaw_deg=90.0) for i in range(4)),
        knob_levels={},
        clause_witnesses={},
        provisional_answer=ProvisionalAnswer(mode="ego_frame", label="left", witness={}),
    )

    payload = render_plan_payload(plan, layout, STD_V1)

    assert payload["path_contract"] == "teleport_cuts"
    assert payload["auxiliary_views"] == []


def test_clean_scene_inventory_binds_bundle_to_replay_recipe(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    bundle = sweep / "bundles" / "room_seed17"
    recipes = sweep / "recipes"
    bundle.mkdir(parents=True)
    recipes.mkdir()
    (bundle / "scene_ir.json").write_text(
        json.dumps({"scene_id": "scene-uuid", "entities": []}), encoding="utf-8"
    )
    (bundle / "scene_snapshot.json").write_text(
        json.dumps(
            {
                "source_scene_id": "room:best",
                "scene_model": "room",
                "scene_instance": None,
                "source_digest": "digest",
            }
        ),
        encoding="utf-8",
    )
    (recipes / "room_seed17.yaml").write_text("recipe_id: source\n", encoding="utf-8")

    scenes = discover_scenes(sweep / "bundles")

    assert len(scenes) == 1
    assert scenes[0].scene_key == "room_seed17"
    assert scenes[0].scene_model == "room"
    assert scenes[0].scene_ir_sha256
    assert scenes[0].source_recipe_sha256


def test_rendered_bundle_contract_counts_sequence_and_auxiliary_views(tmp_path: Path) -> None:
    (tmp_path / "render_report.json").write_text(
        json.dumps(
            {
                "status": "success",
                "views": [{}, {}],
                "auxiliary_views": [{}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "trajectory_plan.json").write_text(
        json.dumps({"views": [{}, {}]}), encoding="utf-8"
    )
    job = SimpleNamespace(bundle=str(tmp_path), frame_count=2, auxiliary_view_count=1)

    assert rendered_bundle_complete(job)  # type: ignore[arg-type]


def test_post_render_rejection_returns_replicate_to_planner(tmp_path: Path) -> None:
    scene_root = tmp_path / "sweep" / "bundles" / "room_seed17"
    recipe_root = tmp_path / "sweep" / "recipes"
    scene_root.mkdir(parents=True)
    recipe_root.mkdir()
    (scene_root / "scene_ir.json").write_text(
        json.dumps({"scene_id": "scene", "entities": []}), encoding="utf-8"
    )
    (scene_root / "scene_snapshot.json").write_text(
        json.dumps(
            {
                "source_scene_id": "room:best",
                "scene_model": "room",
                "scene_instance": None,
                "source_digest": "digest",
            }
        ),
        encoding="utf-8",
    )
    (recipe_root / "room_seed17.yaml").write_text("recipe_id: source\n", encoding="utf-8")
    scene = discover_scenes(tmp_path / "sweep" / "bundles")[0]
    job = CollectionJob(
        job_id="capability__r0__room_seed17",
        capability="capability",
        motif="motif",
        replicate=0,
        seed=17,
        scene=scene,
        plan_id="plan",
        frame_count=10,
        auxiliary_view_count=0,
        plan_record="plan.json",
        render_plan="views.json",
        recipe="recipe.yaml",
        bundle="bundle",
        question_group="group",
    )
    manifest = CollectionManifest(
        collection_id="collection",
        standard_version="std.v8",
        requested_per_capability=2,
        scene_root=str(tmp_path / "sweep" / "bundles"),
        output_root=str(tmp_path),
        planning_parameters={"seeds": [17, 29]},
        jobs=(job,),
        failures=(),
    )
    path = tmp_path / "collection.plan.json"
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    assert reject_jobs(path, {job.job_id: "canonical_mismatch"}) == 1
    updated = load_collection(path)
    assert not updated.jobs
    assert updated.failures[0].scene_key == scene.scene_key
    assert updated.failures[0].reason == "post_render_rejected:canonical_mismatch"


def _sector_fixture_layout() -> SceneLayout:
    """One target candidate per anchor-frame sector, front ones ranked best.

    ``other`` sits at (4, 0) facing ``anchor1`` at the origin, so a target's
    anchor-frame sector is fixed by where it lies relative to that ray. The
    three front candidates sit on the ray just past the anchor, which makes
    them the shortest chains and therefore the ones a purely geometric ranking
    would pick to the exclusion of every other sector.
    """
    objects = [
        SceneObject("Y", "ycat", (4.0, 0.0), 0.6, "y"),
        SceneObject("A1", "acat1", (0.0, 0.0), 0.6, "a1"),
        SceneObject("Xf1", "xf1", (-2.0, 0.0), 0.6, "xf1"),
        SceneObject("Xf2", "xf2", (-2.5, 0.0), 0.6, "xf2"),
        SceneObject("Xf3", "xf3", (-3.0, 0.0), 0.6, "xf3"),
        SceneObject("Xl", "xl", (4.0, -5.0), 0.6, "xl"),
        SceneObject("Xr", "xr", (4.0, 5.0), 0.6, "xr"),
        SceneObject("Xb", "xb", (9.0, 0.0), 0.6, "xb"),
    ]
    return SceneLayout(
        scene_id="sector_fixture",
        objects=tuple(objects),
        walkable_min=(-8.0, -8.0),
        walkable_max=(12.0, 8.0),
    )


def _sector_fixture_script():
    script = CROSS_VIEW_SNAPSHOT_EGO[0]
    categories = {
        "target": ("xf1", "xf2", "xf3", "xl", "xr", "xb"),
        "anchor1": ("acat1",),
        "other": ("ycat",),
    }
    return script.model_copy(
        update={
            "slots": {
                name: slot.model_copy(update={"categories": categories[name]})
                for name, slot in script.slots.items()
            }
        }
    )


def test_chain_label_stratum_reports_anchor_sector_and_closer_side() -> None:
    layout = _sector_fixture_layout()
    script = _sector_fixture_script()
    by_name = {obj.name: obj for obj in layout.objects}

    def stratum(target: str) -> tuple[str, str] | None:
        objects = (by_name[target], by_name["A1"], by_name["Y"])
        return chain_label_stratum(script, objects, STD_V1)

    assert stratum("Xf1") == ("front", "first")
    assert stratum("Xl") == ("left", "second")
    assert stratum("Xr") == ("right", "second")
    assert stratum("Xb") == ("back", "second")


def test_chain_label_stratum_rejects_bindings_whose_questions_are_invalid() -> None:
    script = _sector_fixture_script()
    anchor = SceneObject("A1", "acat1", (0.0, 0.0), 0.6, "a1")
    other = SceneObject("Y", "ycat", (4.0, 0.0), 0.6, "y")
    # Equidistant from the anchor: the closer question cannot clear its ratio.
    equidistant = SceneObject("Xf1", "xf1", (-4.0, 0.0), 0.6, "xf1")
    assert chain_label_stratum(script, (equidistant, anchor, other), STD_V1) is None
    # On a sector boundary: the anchor question cannot clear its margin.
    on_boundary = SceneObject("Xf1", "xf1", (4.0 - 2.0, -2.0), 0.6, "xf1")
    assert chain_label_stratum(script, (on_boundary, anchor, other), STD_V1) is None


def test_chain_binding_shortlist_is_balanced_across_anchor_sectors() -> None:
    layout = _sector_fixture_layout()
    script = _sector_fixture_script()
    by_name = {obj.name: obj for obj in layout.objects}

    bindings = ranked_bindings(layout, script, maximum=4, std=STD_V1)
    sectors = [
        chain_label_stratum(
            script,
            tuple(by_name[binding[slot]] for slot in script.slots),
            STD_V1,
        )[0]
        for binding in bindings
    ]
    assert sorted(sectors) == ["back", "front", "left", "right"]

    # Without stratification the same pool concentrates on the straightest
    # chains, which crowd one sector -- that is the skew being fixed.
    geometric_order = sorted(
        (
            (
                _binding_score(script, tuple(by_name[binding[slot]] for slot in script.slots)),
                binding,
            )
            for binding in ranked_bindings(layout, script, maximum=64, std=STD_V1)
        ),
        key=lambda item: item[0],
    )
    unstratified = [
        chain_label_stratum(
            script,
            tuple(by_name[binding[slot]] for slot in script.slots),
            STD_V1,
        )[0]
        for _, binding in geometric_order[:4]
    ]
    assert Counter(unstratified).most_common(1) == [("front", 3)]


def test_chain_slots_stacked_in_one_fixture_are_not_distinct_places() -> None:
    """An oven and the microwave above it are one place, not two landmarks."""
    layout = SceneLayout(
        scene_id="stacked_fixture",
        objects=(
            SceneObject("oven", "oven", (-1.949, 15.880), 0.84, "oven"),
            SceneObject("microwave", "microwave", (-1.949, 15.877), 0.83, "microwave"),
            SceneObject("chest", "cedar_chest", (-1.239, 11.194), 0.57, "chest"),
            SceneObject("burner", "burner", (-1.759, 13.707), 0.89, "burner"),
        ),
        walkable_min=(-6.0, 9.0),
        walkable_max=(3.0, 18.0),
    )
    stacked = {"target": "oven", "anchor1": "microwave", "other": "chest"}
    separated = {"target": "oven", "anchor1": "burner", "other": "chest"}
    assert not _chain_slots_are_distinct_places(layout, stacked)
    assert _chain_slots_are_distinct_places(layout, separated)


def test_closer_question_is_anchored_at_the_middle_of_the_chain() -> None:
    """The anchor adjacent to the target would make the answer near-constant."""
    anchors = {
        script.capability: script.answer.args["anchor"] for script in CROSS_VIEW_SNAPSHOT_CLOSER
    }
    assert anchors == {
        "cross_view_snapshot_closer_k1": "$anchor1",
        "cross_view_snapshot_closer_k2": "$anchor1",
        # Only a three-anchor chain has a middle that is equidistant in hops
        # from both queried ends, so only k3 moves off the target's neighbour.
        "cross_view_snapshot_closer_k3": "$anchor2",
    }
    for script in CROSS_VIEW_SNAPSHOT_CLOSER:
        slot = script.answer.args["anchor"].removeprefix("$")
        assert f"{{{slot}}}" in script.templates[0].text
        assert script.clauses[-1].args["anchor"] == script.answer.args["anchor"]
