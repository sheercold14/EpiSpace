# Scriptgen Engine

Script-driven trajectory generation for capability-basis episodes.
Package: `src/spatial_episode/scriptgen/`. Status: v1, self-motion capability
implemented end-to-end on the render-free path.

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
| `standards.py` | Every numeric threshold, frozen + versioned (`std.v1`) | predicates never hard-code numbers; changing a value bumps the version |
| `geometry.py` | Frozen conventions: yaw/azimuth signs, 4-sector map, margins | property-tested (rotation equivariance, left/right antisymmetry) |
| `sceneview.py` | `SceneView` protocol + geometry backend; double-threshold tri-state visibility | render backend implements the same protocol over bundle masks (next step) |
| `predicates.py` | Named pure checks returning `Verdict(holds, witness)` | the ONLY place constraint logic lives; shared by search and compile |
| `spec.py` / `library.py` | Declarative `ScriptSpec` per capability | adding a capability touches only `library.py` (plus, rarely, one new predicate) |
| `slotting.py` | Scene objects → slot bindings, with per-object rejection reasons | |
| `motifs.py` | Candidate pose generators; propose only, never judge | legacy T-samplers become motifs after stripping their inline checks |
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
  camera height as occluders, floor AABB union as walkable bounds. Verified on
  hall and residential scenes (0.1-0.3 s for 2-5 plans per scene).
- `poses_from_trajectory_plan(trajectory_plan.json)` — replay acquired
  trajectories through the checker (yaw from the agent quaternion; +Y forward).
- `RenderSceneView.from_bundle(bundle_root)` — the authoritative render
  backend: pixel counts from `views/*.sensors.npz` ``instance_id`` masks
  (note: scene_ir's ``runtime_semantic_id_map`` keys on INSTANCE ids despite
  its name). Verified: 814/814 agreement with render_report on a real bundle.
- `plan_to_agent_views(plan)` — export a plan as the backend's camera-schedule
  view records (roundtrip-tested against `poses_from_trajectory_plan`).

## Closed loop (verified 2026-08-06)

The OminiGibson production repo gained a `scripted_plan` sampling strategy
(new module `omnigibson_episode/scripted.py` + additive edits to `config.py`
and `acquire.py`): the acquisition worker renders a supplied plan file instead
of sampling its own trajectory. First end-to-end run (gates_bedroom, 12-frame
self-motion plan, GPU render ~63 s):

- post-render mask verification: 12/12 frames match the plan's visibility
  claims (target visible at t_seen, zero pixels afterwards);
- authoritative answer recomputed from the rendered bundle's poses equals the
  plan's provisional answer (back, -153.9 deg).

The first render attempt FAILED verification (an 8k-pixel edge sliver at one
"invisible" frame) — caught by this exact check, fixed by extent-aware
visibility, re-rendered clean. The two-phase design paid for itself on run one.

## Authoritative compilation and families (Gaps A–C, 2026-08-06)

Downstream of rendering, three modules close the pipeline through packaging:

- `compiler.py` — `CapabilityCompiler` re-resolves frame variables on the
  render backend (masks override geometry: render_0's t_seen moved 0→3),
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

Clause semantics under intervention live in the spec (`scriptgen_spec.v2`):
`on_violation="abstain"` marks evidence clauses (violated → gold becomes the
abstain option), `"invalid"` marks validity clauses (violated → no gold may
be asserted). `abstain_on_unresolvable` does the same for frame variables.

## Remaining integration (next steps)

- Family-level scoring (Gap D): conditional consistency / covariance /
  abstention calibration from prediction records + family labels only.
- Referent disambiguation for duplicate-category scenes (halls reject most
  slots via `ambiguous_referent`; region-qualified referents lift this).
- Answer-balance control: the walk-away motif biases gold answers toward
  "back"; add motifs/turn patterns that distribute final relative bearings.
