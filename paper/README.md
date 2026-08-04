# EpiSpace paper artifact

This directory contains the ICLR-style data-asset paper for **EpiSpace:
Counterfactual Episode Families for Systematic Spatial Composition**. The
paper reports a frozen simulator-to-episode corpus, its independent integrity
audit, deterministic training schedules, and a structured evaluator. It does
not report trained-model improvements, external transfer, human evaluation, or
RL/GRPO results.

## Frozen evidence snapshot

The authoritative source is
`../data/epispace_pilot_v1/release_manifest.json`.

| Asset | Verified count |
|---|---:|
| Planned acquisition jobs | 322 |
| Strict bundles | 193 |
| Write trajectories / scenes / RGB views | 172 / 46 / 1,063 |
| Geometry-replayed questions | 1,093 / 1,093 |
| Complete sibling families | 133 |
| Evidence / claim / frame families | 91 / 37 / 5 |
| Episode / isolated train records | 340 / 651 |
| Shared effective training facts | 651 |
| Full / family / composition / core benchmark rows | 370 / 112 / 27 / 115 |

The fact-matched exports use 1,168 episode versus 3,227 isolated image
occurrences. The visual-exposure-matched schedules use 3,227 occurrences,
3,383,754,752 input pixels, and 206,528 exact Qwen3-VL image tokens in each
arm; inverse-repeat weights keep the same 651 effective facts. Total input
tokens remain 415,424 versus 362,185, so no equal-FLOP claim is made.

## Paper layout

- `main.tex`: title, abstract, and section assembly.
- `sections/principles.tex`: first-principles learning contract.
- `sections/engine.tex`: trajectory adapter, compiler, family construction,
  and release gates.
- `sections/training.tex`: SFT interfaces, matching regimes, evaluator, and
  falsifiable future comparison.
- `sections/experiments.tex`: completed data/corpus/schedule validation only.
- `sections/appendix.tex`: schemas, exact gates, metrics, and commands.
- `generated/`: tables and macros emitted by the dataset builder; do not edit
  them manually.

## Rebuild the data evidence

Run from the repository root:

```bash
PY=../habitat/.conda/core/bin/python

$PY -m episode3d.cli --config configs/epispace_pilot_v1.json

$PY scripts/build_compute_matched_schedules.py \
  --episode data/epispace_pilot_v1/train.episode_sft.jsonl \
  --isolated data/epispace_pilot_v1/train.isolated_sft.jsonl \
  --output-dir data/epispace_pilot_v1/compute_matching \
  --seed 17

$PY -m episode3d.corpus_audit data/epispace_pilot_v1

$PY -m pytest -q
$PY -m ruff check episode3d scripts tests
```

The independent corpus audit reads only serialized artifacts; it does not
trust or call the builder/compiler/verifier. Without model predictions,
accuracy-conditioned evidence sensitivity is correctly reported as not
evaluated.

## Smoke-test the evaluator

The following copies ground-truth structured values to test evaluator wiring.
Its 100% output is an oracle smoke test, never a model result.

```bash
$PY -m episode3d.benchmark_evaluator oracle \
  --benchmark data/epispace_pilot_v1/benchmark.core.jsonl \
  --output /tmp/epispace-oracle.jsonl

$PY -m episode3d.benchmark_evaluator evaluate \
  --benchmark data/epispace_pilot_v1/benchmark.core.jsonl \
  --predictions /tmp/epispace-oracle.jsonl \
  --output-dir /tmp/epispace-evaluation
```

A real prediction file has one JSON object per line:

```json
{"record_id":"benchmark-...","answer_value":"left_of"}
```

Missing, duplicate, extra, malformed, or non-finite predictions are scored as
incorrect.

## Build the manuscript

ICLR 2027 style files were not available at the July 2026 freeze, so the paper
uses the local ICLR 2026 style as a placeholder. With TeX Live installed:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Replace the style and bibliography files when the official ICLR 2027 kit is
released. Rebuild `main.pdf` after any source or generated-statistics change;
do not treat an older PDF timestamp as authoritative.

## Boundaries and portability

- Primary model inputs are raw RGB plus natural language. Pose, depth,
  instance masks, programs, certificates, and oracle target views are hidden.
- JSONL media references are currently absolute machine paths. Remap the
  `data/epispace_rgb_v1` prefix after transfer while preserving image hashes.
- BEHAVIOR-1K-derived renders remain under the source academic-use terms. This
  internal artifact does not grant redistribution rights.
- All 91 evidence families pass both set-matched single replacement and the
  stronger ordered-slot stability check.
