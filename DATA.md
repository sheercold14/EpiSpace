# Data

## Data boundary

Git stores code, schemas, frozen configurations, manifests and compact showcase records. Large
trajectory bundles, sensor channels, training corpora, predictions and checkpoints stay outside
the repository and are located through `workspace.env`.

## Three data layers

1. **Acquisition bundle**: scene snapshot, trajectory plan, RGB/depth/instance/semantic channels,
   camera poses and quality report.
2. **Episode IR**: observable belief, typed operation graph, evidence pointers and certificates.
3. **Model records**: episode/isolated SFT, evaluation inputs and physically separated oracle.

The model receives only the channels declared in `model_input`; simulator truth is never copied
into ordinary prompts.

## Current datasets

| Dataset | Manifest | Role |
|---|---|---|
| EpiSpace Pilot v1 | `manifests/datasets/epispace_pilot_v1.release_manifest.json` | Main episode/isolated controlled study |
| Scene Dialogues v1 | `manifests/datasets/scene_dialogues_v1.corpus_report.json` | Natural multi-round dialogue surface |
| Transform Pilot v1 | `manifests/datasets/transform_pilot_v1.corpus_report.json` | Self-Rotation and Among-5 mechanism pilot |

## Split and counting rules

- Split unit is the scene; counterfactual siblings inherit the family split.
- Report scenes, geometric facts, program siblings and language surfaces separately.
- A larger number of QA surfaces is not a larger number of independent spatial worlds.
- Evaluation input and oracle must remain separate files joined only by stable `record_id`.

## Local data layout

The recommended external layout is:

```text
$EPISPACE_DATA_ROOT/
├── epispace_pilot_v1/
├── scene_dialogues_v1/
└── transform_pilot_v1/

$OMNIGIBSON_OUTPUT_ROOT/
└── sweeps/<trajectory-version>/bundles/<scene>/
```

