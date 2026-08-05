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

## Integration points (next steps)

- Render backend adapter: implement `SceneView` over acquisition bundles
  (instance masks) in `backends/omnigibson`; reuses `verifiers.py` logic.
- Plan-to-recipe adapter: convert `TrajectoryPlan.poses` into the acquisition
  worker's camera schedule so Isaac Sim renders planned trajectories.
- Layout export: build `SceneLayout` from BEHAVIOR scene snapshots instead of
  the demo layout (objects, OBBs as occluders, traversable bounds).
- Sibling expansion and family packaging sit downstream of plans and reuse the
  same predicates for recompilation.
