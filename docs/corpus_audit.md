# Independent corpus audit

`episode3d.corpus_audit` audits the files a downstream user receives. It does
not import or invoke the data builder, compiler, program executor, pipeline, or
release gates, so a builder regression cannot certify itself.

```bash
../habitat/.conda/core/bin/python -m episode3d.corpus_audit \
  data/epispace_pilot_v1
```

The default outputs are `corpus_audit.json` (complete machine-readable
evidence) and `corpus_audit.md` (compact review report) inside the release
directory. `--json` and `--markdown` override these paths.

The audit independently measures:

- exact and field-marginal structured-target distributions for every
  `split × task_type`;
- same-split oracle and train-fitted task-conditioned majority priors;
- counterfactual pair/triple completeness and split/scene locking;
- shared-view/single-replacement set matching for every evidence triple, plus
  the stronger ordered-slot stability diagnostic;
- same-input, frame-only question change, transformed answer, and certificate
  agreement for every ego-frame pair;
- structural eligibility for accuracy-conditioned evidence sensitivity;
- scene, scene-family, trajectory-family, consistency-family, and export-family
  cluster sizes and cross-split leakage;
- raw English category/relation tokens in model-visible inputs and supervised
  outputs, separately from allowed hidden structured targets;
- existence, readability, dimensions, modes, and portability of every unique
  image path;
- schema and target-prior summaries for the full, family, composition, and core
  benchmark views;
- per-comparison and global episode/isolated fact multisets, image sets,
  serialized image references, and pixel exposure.

The task-conditioned majority value is a label-prior diagnostic, not a VLM
result. Likewise, a structurally complete evidence family makes the metric
evaluable but does not constitute a model score. To compute the latter, pass a
prediction JSONL:

```json
{"fact_id": "fact-...", "correct": true}
```

```bash
python -m episode3d.corpus_audit RELEASE \
  --predictions model_correctness.jsonl
```

For each complete `{prefix_unknown, revealed, decisive_deleted}` family, the
accuracy-conditioned evidence-sensitivity score conditions on the revealed
member being correct and then asks whether both evidence-deleted siblings are
also correct. Missing artifacts, malformed JSON, absent schema fields, absent
predictions, or unavailable image dimensions are reported as explicit
`not_evaluable`/`partial` states rather than silently imputed.

Raw training exports are expected to be fact matched but not necessarily input
compute matched. The audit reports that distinction from serialized exposure;
use the separate compute-matching schedules for the controlled training run.
