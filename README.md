# EpiSpace

EpiSpace studies whether multimodal language models can learn reusable spatial updating from
RGB episodes, rather than fitting isolated spatial question templates.

The model-facing interface is ordered RGB plus natural language. Camera poses, depth, instance
masks, typed `G/F/B/M/R/P/V` programs, canonical state and geometry certificates remain hidden
compiler or evaluator channels.

## Research question

Can `observe-first, read-many` episode training and counterfactual scene families improve
generalization to unseen spatial transformations and operation compositions under matched facts
and visual exposure?

```text
3D scene → controlled trajectory → RGB observations → observable belief
         → typed spatial program → verified QA/episode → SFT/evaluation
```

## Current snapshot

- Simulator backend: OmniGibson/BEHAVIOR scenes with T1/T2/T3/T4/T7/T8/T10 trajectories.
- Main EpiSpace pilot: 322 planned collection jobs and 193 strict source/verifier bundles.
- Scene dialogues: 29 compiled multi-round scenes and 146 isolated QA controls.
- Transform Pilot: 2,961 records, 1,803 geometric facts and 16 complete Among-5 families.
- Frozen zero-shot baselines: SenseNova-SI 1.1 and 1.5, Answer-only and Grounded-CoT.

Run `make status` for the manifest-derived snapshot. These numbers describe data and completed
zero-shot diagnostics; they do not claim an episode-training gain before the paired SFT study.

## Five-minute start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
make quickstart
make status
```

The quickstart is simulator-free. Full trajectory rendering lives in the separate
OminiGibson acquisition repository (single source of truth for simulator code);
EpiSpace only reads the bundles it produces.

For resumable 51-scene source acquisition and binding-level episode coverage,
see [docs/binding_coverage_batch.md](docs/binding_coverage_batch.md).

## Repository map

| Path | Purpose |
|---|---|
| `src/episode3d/` | Episode compiler, typed programs, audits, exporters and evaluation |
| `src/spatial_episode/` | Simulator-independent schemas and typed operation contracts |
| `scripts/pilot/` | Frozen generators of pilot datasets (provenance; see scripts/INDEX.md) |
| `configs/` | Frozen dataset and experiment configurations |
| `manifests/` | Dataset and experiment provenance snapshots |
| `examples/transform_pilot/` | Compact showcase records with real model responses |
| `tests/` | Core unit tests and external-artifact replay tests |
| `docs/research/` | INTERNAL working notes (plans, competitor analysis); excluded from any public release. `docs/` outside this folder is the public documentation surface |
| `paper/` | Paper source and generated-table interfaces |
| `web/` | Review UI source; generated media are not committed |

## Main commands

```bash
make doctor       # environment and repository diagnostics
make status       # current data and experiment snapshot
make quickstart   # fast simulator-free smoke test
make check        # static checks
make test         # portable CPU tests; no scene assets required
make test-artifact # full replay suite after connecting external data
```

See [GETTING_STARTED.md](GETTING_STARTED.md), [DATA.md](DATA.md) and
[REPRODUCE.md](REPRODUCE.md) before running full data generation or training.

## Project status

The source code in this repository is an independent copy of the working `episode3D` and
`OminiGibson` projects. The original directories remain untouched. `STATUS.md` records what has
already been produced and what remains to be validated.

## Citation

The paper is in preparation. A provisional citation entry is provided in `CITATION.cff` and will
be updated when the public preprint is available.
