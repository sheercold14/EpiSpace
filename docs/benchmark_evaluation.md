# EpiSpace benchmark evaluation

`episode3d.benchmark_evaluator` is the frozen-benchmark scorer. It is independent
of OmniGibson and the data compiler: evaluation needs only
`benchmark.core.jsonl` and model predictions.

## Prediction contract

Write one JSON object per line with exactly the public record identifier and a
structured JSON answer:

```json
{"record_id":"benchmark-...","answer_value":{"claim_correct":true,"relation":"left"}}
```

`answer_value` must have the same JSON structure as the benchmark target. Object
key order is irrelevant; object keys, enum strings, booleans, null values and
list order are otherwise exact. Numeric targets use
`certificate.answer_tolerance_m` when the record declares it, and absolute
tolerance `1e-6` otherwise. A missing, duplicate, or malformed prediction is
scored incorrect rather than silently skipped. Unknown answers are represented
by JSON `null`, not by a free-form sentence.

## Run evaluation

```bash
../habitat/.conda/core/bin/python -m episode3d.benchmark_evaluator evaluate \
  --benchmark data/epispace_pilot_v1/benchmark.core.jsonl \
  --predictions runs/qwen/predictions.jsonl \
  --output-dir runs/qwen/evaluation
```

The command always writes both:

- `evaluation.json`: the lossless machine-readable report, including every
  record/family result, every missing or extra prediction ID, and SHA-256 hashes
  binding the report to the exact benchmark and prediction files;
- `evaluation.md`: a compact table for experiment logs and paper drafting.

The report includes record accuracy, breakdowns by task/program/split, family
exact match, claim-pair exact match, evidence-triple exact match and revealed
accuracy, plus frame-pair exact match and explicit ego-frame equivariance
obligations. The evidence-sensitivity metric is deliberately
accuracy-conditioned:

```text
number of triples with revealed + both deletion controls correct
----------------------------------------------------------------
number of triples whose revealed member is correct
```

This prevents a model from receiving evidence-sensitivity credit when it never
answers the decisive revealed case correctly. When the denominator is zero,
JSON records `value: null` and `status: "NA_ZERO_DENOMINATOR"`; Markdown prints
`NA` explicitly.

## Oracle smoke test

Generate predictions by copying the hidden structured targets, then exercise
the exact production evaluator:

```bash
../habitat/.conda/core/bin/python -m episode3d.benchmark_evaluator oracle \
  --benchmark data/epispace_pilot_v1/benchmark.core.jsonl \
  --output /tmp/epispace-oracle.jsonl

../habitat/.conda/core/bin/python -m episode3d.benchmark_evaluator evaluate \
  --benchmark data/epispace_pilot_v1/benchmark.core.jsonl \
  --predictions /tmp/epispace-oracle.jsonl \
  --output-dir /tmp/epispace-oracle-report
```

All accuracy and exact-match metrics must be `1.0`. This is a scorer wiring
test, not a model result and must never be reported as one.

## Metric units

- **Record accuracy**: correct records divided by all benchmark records;
  missing predictions remain in the denominator.
- **Family exact match**: all records sharing `family_id` must be correct.
- **Claim-pair exact match**: both `claim_false` and `claim_true` in a complete
  `consistency_group` must be correct.
- **Frame-pair exact match / equivariance obligation**: both `frame_a` and
  `frame_b` must match their own frame-conditioned relation target. Each detail
  row records the two expected/predicted labels, whether the visual input is
  identical, and whether changing the query frame requires the relation to
  change. `prediction_change_rate_when_required` is diagnostic only: changing
  from one wrong label to another receives no equivariance credit.
- **Evidence revealed accuracy**: accuracy of the `revealed` member of each
  complete evidence triple.
- **Evidence-triple exact match / conditioned sensitivity**: all of
  `prefix_unknown`, `revealed`, and `decisive_deleted` must be correct; the
  latter reports this joint count over the revealed-correct count.

Incomplete or malformed consistency groups are listed in structural
diagnostics and excluded from the pair/triple denominator. They remain ordinary
records in record accuracy and family exact match, so release validation should
still require the benchmark-family completeness gate before publication.

The checked-in smoke artifact for the frozen pilot is under
`data/epispace_pilot_v1/evaluation_smoke/`. Its oracle predictions copy labels
and are only a scorer regression fixture. Rebuild it after every benchmark
freeze; a stale record count or benchmark SHA-256 is a failed smoke test.
