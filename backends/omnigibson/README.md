# OmniGibson Spatial Episode Backend

This project turns a BEHAVIOR-1K / OmniGibson scene into the same verifiable
`SpatialEpisodeV1` used by the adjacent Habitat/HM3D pipeline. It is an
acquisition backend, not a fork of OmniGibson and not a second episode format.

```text
BEHAVIOR scene -> traversable closed trajectory -> RGB/depth/instance evidence
               -> scene snapshot -> shared SceneIR/relation oracle
               -> SpatialEpisodeV1
```

The heavy Isaac Sim process only writes a backend-neutral acquisition bundle.
The final compilation runs in the lightweight Spatial Episode Forge environment.
This boundary keeps CUDA/Isaac dependencies out of training and lets HM3D and
OmniGibson share IDs, coordinates, channels, operation graphs and verification.

## Layout

- `configs/omnigibson_static_m1.yaml`: first static multi-view recipe.
- `src/omnigibson_episode/acquire.py`: OmniGibson scene, path and sensor worker.
- `src/omnigibson_episode/compile_shared.py`: snapshot to canonical episode.
- `src/omnigibson_episode/presentation.py`: evidence-backed oral demo generator.
- `tests/`: simulator-free geometry and end-to-end compiler tests.
- `scripts/bootstrap_behavior.sh`: pinned BEHAVIOR-1K installation helper.
- `scripts/install_isaacsim_cached.sh`: resumable Isaac Sim wheel installer.
- `scripts/install_headless_runtime.sh`: rootless GLU/OpenGL runtime installer.
- `scripts/run_in_omnigibson.sh`: reproducible simulator command wrapper.

The directory is intentionally named `OminiGibson` to match the requested
workspace path; Python package and upstream dependency names use the correct
`omnigibson` spelling.

## Environment

OmniGibson v3.9.0 requires an isolated Python 3.11 environment and Isaac Sim.
This host satisfies the GPU/RAM requirements. Enable the user's proxy for the
many small Git/Python packages, then install the pinned core environment:

```bash
proxy_on
bash scripts/bootstrap_behavior.sh
```

The script clones the pinned release under `.deps/` and delegates Isaac Sim EULA
handling to the official setup script. It does not accept terms automatically.
If NVIDIA wheel transport is interrupted, resume through the persistent project
cache after reading and accepting the NVIDIA Isaac Sim EULA:

```bash
proxy_on
bash scripts/install_isaacsim_cached.sh --accept-eula
```

Install the two Ubuntu rendering libraries required by Isaac Sim without root:

```bash
bash scripts/install_headless_runtime.sh
```

After personally reading and accepting the BEHAVIOR Data Bundle agreement,
download the scene assets. Large files use `hf-mirror`; downloads are resumable
and extraction is staged atomically:

```bash
# Dynamic direct-CDN selection; preferred when its probe is fast.
bash scripts/download_behavior_assets.sh --accept-license --transport fast

# Alternative: hf-mirror through proxy_on/aria2 when current proxy is faster.
proxy_on
bash scripts/download_behavior_assets.sh --accept-license --transport proxy
```

OmniGibson also requires its 641,004,353-byte runtime asset bundle even when the
recipe has `robots=[]`. Install it separately (or pass `--archive PATH` if it was
transferred from another server):

```bash
proxy_on
bash scripts/download_robot_assets.sh --transport proxy
```

The main scene archive is 31,457,673,073 bytes. Challenge task files remain
unnecessary. Assets live under ignored `.data/`; they are never vendored or
redistributed.

## Run one episode

Use the Python executable created by the official BEHAVIOR setup for acquisition:

```bash
OMNIGIBSON_DATA_PATH="$PWD/.data/omnigibson" \
bash scripts/run_in_omnigibson.sh --accept-eula \
  python -m omnigibson_episode.cli acquire \
  --recipe configs/omnigibson_static_m1.yaml \
  --output outputs/Rs_int_seed17 --gpu-id 0 --headless
```

Then compile in the lightweight Forge environment:

```bash
PYTHONPATH="$PWD/src:../habitat/src" ../habitat/.conda/core/bin/python \
  -m omnigibson_episode.cli compile \
  --bundle outputs/Rs_int_seed17 \
  --output outputs/Rs_int_seed17/spatial_episode.json

PYTHONPATH="$PWD/src" ../habitat/.conda/core/bin/python \
  -m omnigibson_episode.cli inspect \
  --bundle outputs/Rs_int_seed17
```

The bundle contains:

```text
scene_snapshot.json       # object poses/AABBs and runtime instance registry
trajectory_plan.json      # shared canonical +X right, +Y forward, +Z up poses
render_report.json        # evidence manifest and visibility measurements
views/*.sensors.npz       # RGB, metric depth, instance and semantic masks
scene_ir.json
relation_oracle.json
spatial_episode.json
quality_report.json       # integrity, exposure, depth and sharpness metrics
preview.html              # local RGB/depth/instance browser
oral_demo/                # generated oral narrative and interactive explorer
```

`acquire` fails atomically: a failed run writes `failure_report.json`, never a
partial success manifest. `compile` rejects view/ID mismatches and relations
without rendered evidence.

RGB and metric depth are captured with the path-traced HQ renderer. Labels use
Replicator's native instance-ID channel in a separate real-time pass: every
renderer mesh ID is resolved through its USD prim path and merged into a stable
OmniGibson object ID, then projected to the object's category. Each view records
the mapping coverage and acquisition rejects coverage below 95% among renderer
pixels that expose a resolvable prim path. Explicit `INVALID` / `UNLABELLED`
renderer surfaces are retained as unknown and separately capped at 10% of a
frame. `inspect` also checks the closed-loop endpoints: depth and both masks must
repeat exactly; stochastic RGB rendering must retain at least 30 dB PSNR.

### T3 rotation-station pilot

The trajectory specification also supports a true zero-baseline panorama. The
sampler chooses a traversable point near a room-instance centroid, enforces at
least one metre of wall clearance, and renders a uniform yaw scan without
moving the camera:

```bash
bash scripts/run_in_omnigibson.sh --accept-eula \
  python -m omnigibson_episode.cli acquire \
  --recipe configs/omnigibson_t3_rotation_station_pilot.yaml \
  --output outputs/sweeps/trajectory-pilot-v1/bundles/Rs_int_t3_seed17 \
  --gpu-id 0 --headless
```

For T3, `inspect` replaces the loop gate with class-specific checks: exact zero
translation, at least eight non-structural entities in the panorama, at least
two entities visible only in the rear half-scan, yaw/view alignment, and
`depth_questions_forbidden=true`. The model-visible channel remains RGB only.

The production expansion uses two explicitly ranked stations per scene rather
than pretending that a different random seed changes the deterministic room
centroid. Rank 1 must be at least one metre from rank 0 and retains the same
wall-clearance and panorama gates:

```bash
make plan-t3-r0 plan-t3-r1
make run-t3-r0 GPU_IDS=2 WORKERS=1
make run-t3-r1 GPU_IDS=3 WORKERS=1
```

### T4 adaptive object orbit

The frozen v3 repair first attempts a complete connected orbit, then permits an
eight-view 270-degree arc. It lowers the minimum focus radius to 0.7 m, permits
smooth per-view radius adaptation, and checks only adjacent path edges for a
partial arc. Relaxation never bypasses evidence: the measured arc must reach
270±10 degrees, at least one opposing-view pair must exist, the focus must be
visible in every view, and each view must retain at least three context
instances.

```bash
make plan-t4-v3
make run-t4-v3 GPU_IDS=0,1 WORKERS=2
```

The earlier v2 directory is a development pilot and must not be merged with the
v3 sweep. T8/T10 warning adjudication is reproducible with:

```bash
PYTHONPATH=src:../habitat/src ../habitat/.conda/core/bin/python \
  -m omnigibson_episode.typed_adjudication \
  --plan outputs/sweeps/t8-occlusion-reveal-seed17-v1/sweep_plan.json \
  --plan outputs/sweeps/t10-target-view-seed17-v1/sweep_plan.json \
  --output outputs/audits/t8-t10-typed-adjudication-v1.json
```

## Build the oral demo

Generate the presentation from the compiled bundle. The command validates the
quality report, reads only real episode artifacts, exports browser-friendly
sensor channels and replaces the output directory atomically:

```bash
PYTHONPATH="$PWD/src:../habitat/src" ../habitat/.conda/core/bin/python \
  -m omnigibson_episode.cli present \
  --bundle outputs/Rs_int_seed17

python -m http.server 8765 --directory outputs/Rs_int_seed17
```

Open `http://SERVER:8765/oral_demo/`. The page separates measured evidence from
research hypotheses: current episode counts, channels, relations, certificates
and quality metrics come from the bundle; training gains are presented only as
falsifiable evaluation targets. Selecting a query links its evidence view,
canonical map, typed operation graph and verification margin.

## Build and review the dataset

The production unit is an independently rendered scene trajectory, not a QA row.
The M2 registry always keeps all 46 indoor targets visible, including failed,
sparse and pending scenes:

```bash
make inventory object-inventory split-manifest plan-sweep-m2
PYTHONPATH=src:../habitat/src ../habitat/.conda/core/bin/python \
  -m omnigibson_episode.cli run-sweep \
  --plan outputs/sweeps/static-m2-room-aware-indoor-seed17-v1/sweep_plan.json \
  --gpu-ids 0,1,2,3 --workers 4
make derive-m2 release-m2 verify-release-m2 review-portal
```

`derive-m2` re-executes every deterministic compiler and certificate. Its
`derived_refresh_report.json` aggregates G/F/B/M/R/P/V, memory, cross-view and
unknown-abstention coverage across acquisitions. The working release records the
report hash and summary; a frozen release is rejected unless all 46 acquisitions
were refreshed and no capability was sampled without supporting evidence.

Build the two production minimal-pair protocols separately. Relation v2 is a
room-safe physical rearrangement whose answer must flip; model-swap v5 is a
strict-clearance, zero-step appearance counterfactual whose answer must remain
invariant:

```bash
make plan-interventions-m2-v2 plan-model-swaps-m2-v5
make run-interventions-m2-v2 GPU_IDS=0,1 WORKERS=2
make run-model-swaps-m2-v5 GPU_IDS=2,3 WORKERS=2
make verify-interventions-release-m2-v2 verify-model-swaps-release-m2-v5
make review-portal
```

The intervention runner treats both `certified` and `needs_review` as terminal:
the latter already has a complete machine certificate and waits only for human
visual judgment, so rerunning a sweep never destroys or re-renders it. Only
`pending` / `acquired` work is resumed by default; strict failures require an
explicit `--retry-failed` selection. Production targets use a 60-minute scene
acquisition timeout because the largest Isaac Sim scenes take more than 20
minutes to import.

The working manifests retain pending and failed jobs as an honest production
registry. Named pair snapshots can be frozen only after all jobs are terminal
and a provenance-bearing human review resolves every candidate; see the
`freeze-interventions-m2-v2` and `freeze-model-swaps-m2-v5` targets.

Serve the repository root and open the unified review hub:

```bash
python -m http.server 8770 --directory ..
# http://SERVER:8770/OminiGibson/outputs/review/
```

The browser stores decisions locally and exports a provenance-bearing JSON with
reviewer identity, timestamps and notes. Automated integrity gates cannot be
overridden by human acceptance. See `dataset/DATASET_CARD.md`,
`dataset/VERSIONING.md` and `dataset/RELEASE_CHECKLIST.md` before freezing or
publishing a release.

## Development

```bash
make check
OMNI_KIT_ACCEPT_EULA=YES make doctor
```

`doctor` distinguishes an installed simulator environment from full acquisition
readiness, which additionally requires the licensed BEHAVIOR assets.

Do not put BEHAVIOR assets, Isaac Sim files or generated episodes in git.
