# Scriptgen Engine

Script-driven trajectory generation for capability-basis episodes.
Package: `src/spatial_episode/scriptgen/`. Status: self-motion capability
implemented end-to-end through free-space planning, rendering, authoritative
compilation, interventions and family packaging.

## Why

The legacy acquisition path hard-codes one sampler function per trajectory
class (T1..T10 in `backends/omnigibson/.../acquire.py`), with constraint
checks inlined in each sampler and thresholds scattered as literals. Adding a
capability meant a new sampler, a new recipe and a new build script, and the
same concept ("visible", "disappeared") existed in several implementations.

The scriptgen engine inverts this: capabilities are DECLARED, the engine is
generic.

```text
authoring surface           generic engine                     output
─────────────────           ──────────────────────────────     ──────────────
ScriptSpec (30 lines)  ──►  slotting → motifs → checker   ──►  TrajectoryPlan
  slots/clauses/knobs       (predicates over a SceneView)      (poses + witnesses
  frame_vars/templates      standards = all thresholds          + provisional answer)
```

## Layers

| Module | Role | Rule |
|---|---|---|
| `standards.py` | Every judgement threshold, frozen + versioned (current: `std.v3`) | predicates never hard-code thresholds; changing one bumps the version |
| `geometry.py` | Yaw/azimuth conventions, sector margins, rotated-rectangle point/segment tests | property-tested (rotation equivariance, left/right antisymmetry) |
| `sceneview.py` | `SceneView` protocol + geometry backend; layout objects, occluders and rotated obstacles | render backend implements the same protocol over bundle masks |
| `predicates.py` | Named pure checks returning `Verdict(holds, witness)` | the ONLY place constraint logic lives; shared by search and compile |
| `spec.py` / `library.py` | Declarative `ScriptSpec` per capability | adding a capability touches only `library.py` (plus, rarely, one new predicate) |
| `slotting.py` | Scene objects → slot bindings, with per-object rejection reasons | |
| `motifs.py` | Candidate pose generators; 5 cm occupancy grid + connected-component-aware free-space A* | propose only, never judge; exact predicates remain authoritative |
| `checker.py` | Frame-var resolvers + tiny `$var`/`+`/`-`/`a:b` expression language; clause evaluation | inclusive frame ranges; ambiguous evidence frames are rejects, not coercions |
| `generate.py` | Search loop; emits plans and a rejection histogram | silence is forbidden: every failure is counted by failing clause |
| `plan.py` | `TrajectoryPlan` contract consumed by acquisition backends | records standard version, witnesses and a PROVISIONAL geometry answer |
| `demo.py` / `cli.py` | Render-free demo layout and CLI | `python -m spatial_episode.scriptgen.cli --capability self_motion_update` |

## Two-phase visibility

Search phase uses the geometry backend (frustum + occluder ray sampling,
`geom_ratio = size_m / distance_m`) with tightened margins
(`search_tighten_factor`). After rendering, the same predicates re-run on the
render backend (instance-mask pixels). Certificates trust ONLY the render
backend; disagreement rejects the bundle instead of trusting either side.

## Adding a capability (the 30-line contract)

1. Write one `ScriptSpec` in `library.py`: slots, frame_vars, clauses
   (predicate references), knobs, length, motifs, templates.
2. If a genuinely new concept is needed, add ONE predicate to
   `predicates.py` with a witness, plus its golden/property tests.
3. Nothing else changes. Verification: `pytest tests/scriptgen`.

## BEHAVIOR adapters (`behavior.py`)

Implemented; all render-free (they read artifacts acquisition already wrote):

- `layout_from_scene_ir(scene_ir.json)` — real BEHAVIOR scenes as planning
  layouts: non-structural entities as objects, wall/pillar footprints spanning
  camera height as occluders, every non-structural rotated OBB + z span as an
  obstacle, structural wall OBBs as collision-only obstacles, and floor AABB
  union as walkable bounds. Walls never enter question-target objects.
  Obstacle extraction has no thresholds; std.v3 predicates decide whether
  its height blocks a body.
- `poses_from_trajectory_plan(trajectory_plan.json)` — replay acquired
  trajectories through the checker (yaw from the agent quaternion; +Y forward).
- `RenderSceneView.from_bundle(bundle_root)` — the authoritative render
  backend: pixel counts from `views/*.sensors.npz` ``instance_id`` masks
  (note: scene_ir's ``runtime_semantic_id_map`` keys on INSTANCE ids despite
  its name). Verified: 814/814 agreement with render_report on a real bundle.
- `plan_to_agent_views(plan)` — export a plan as the backend's camera-schedule
  view records (roundtrip-tested against `poses_from_trajectory_plan`).

## Traversable closed loop (verified 2026-08-07)

`walk_and_turn` rasterises furniture and wall OBBs at 5 cm with a 0.35 m
proposal clearance, partitions free space into connected components, samples
start/end cells in one component, and uses eight-connected A* (without
diagonal corner cutting) whenever a straight segment is blocked. Generation
then runs two std.v3 `search_only` hard clauses before any semantic clause:

- `poses_clear`: every camera/body centre is outside every obstacle expanded
  by the 0.30 m body radius in the 0.10–1.70 m body-height band;
- `path_clear`: every consecutive movement segment avoids those footprints.

On gates_bedroom, the three pre-navfix plans are permanently retained as
negative regression cases (coffee-table/sofa, bed/armchair collisions). The
current `batch_wallfix` trajectories (seeds 17/23/4; 12/14/11 frames) check
22 object OBBs plus 4 wall OBBs, pass both clauses with zero collisions, were
rendered at 1024² with RGB/depth/instance labels, and compile to
left/left/right.

## Authoritative compilation and families (Gaps A–C, 2026-08-06)

Downstream of rendering, three modules close the pipeline through packaging:

- `compiler.py` — `CapabilityCompiler` re-resolves frame variables on the
  render backend (masks override geometry: navfix render_0's t_seen moved 0→1),
  re-judges every clause with search tightening off, derives the answer from
  the rendered pose at t_q, and emits `scriptgen_certificate.v1` with
  geometry estimates demoted to comparison fields and a `mismatch` blocker.
  Optional leave-one-out analysis marks the essential frame set.
- `variants.py` — interventions are frame index sequences over the rendered
  frames (`ReindexedSceneView`): permute (gone-segment shuffle), drop_key
  (keep only definitely-invisible frames), drop_filler (verified removal of
  non-essential frames), delay (repeat one mid-gone frame = standstill).
  Every gold is produced by re-running the same compiler; the spec-declared
  expectation (`variant_expectations`) only cross-checks it, and
  disagreement raises `FamilyMismatch`.
- `family.py`/`family_cli.py` — one command packs canonical + 4 variants +
  certificates into a single `scriptgen_family.v1` JSON (registered in
  `contracts/schema.py`), audits referent uniqueness and question-text leaks
  (no placeholders, no frame numbers, no gold token), exports per-frame
  rgb/depth/instance PNGs and ships `web/scriptgen_family_review.html`.

Clause semantics under intervention live in the spec (`scriptgen_spec.v3`):
`on_violation="abstain"` marks evidence clauses (violated → gold becomes the
abstain option), `"invalid"` marks validity clauses (violated → no gold may
be asserted). `abstain_on_unresolvable` does the same for frame variables.
The third phase, `search_only`, validates the path that was physically acquired
but is intentionally excluded when compiling reindexed presentation variants:
otherwise a shuffled filmstrip would be mistaken for a physically traversed
teleport path and fail for the wrong reason.

## Remaining integration (next steps)

- Family-level scoring (Gap D): conditional consistency / covariance /
  abstention calibration from prediction records + family labels only.
- Referent disambiguation for duplicate-category scenes (halls reject most
  slots via `ambiguous_referent`; region-qualified referents lift this).
- Answer-balance control: the walk-away motif biases gold answers toward
  "back"; add motifs/turn patterns that distribute final relative bearings.
- Simulator-side capsule collision / navmesh validation before large-scale
  acquisition; current hard validity is based on scene_ir static rotated OBBs.
  Wall OBBs are deliberately conservative and can fill openings or irregular
  wall interiors, so their cross-scene false-positive rate must be calibrated
  against simulator collision geometry rather than assumed correct.
