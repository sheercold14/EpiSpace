"""Batch collection planning contracts and deterministic selection."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

from spatial_episode.scriptgen.collection import (
    CollectionJob,
    CollectionManifest,
    SourceRenderEvidence,
    _binding_chain_source_covisible,
    _reference_binding_eligible,
    _reference_render_robust_filter,
    _visit_render_robust_filter,
    discover_scenes,
    load_collection,
    ranked_bindings,
    reject_jobs,
    render_plan_payload,
)
from spatial_episode.scriptgen.library import (
    CROSS_VIEW_RELATION,
    CROSS_VIEW_SNAPSHOT_EGO,
    EXISTENCE_SUFFICIENCY,
    REFERENCE_FRAME_DEEP,
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

    assert _visit_render_robust_filter(overlapping, binding) == (
        "queried_pair_angular_overlap"
    )
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
        covisible_pairs=complete.covisible_pairs
        - {frozenset(("anchor1", "anchor2"))},
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
        occlusion_obstacles=(
            Obstacle("wall", (2.0, 0.0), (0.1, 1.0), 0.0, 0.0, 2.0),
        ),
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
        poses=tuple(
            PlannedPose(frame=i, x=1.0, y=float(i), yaw_deg=90.0) for i in range(4)
        ),
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
