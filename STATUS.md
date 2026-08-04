# Project Status

> Generated from the copied manifests by `python scripts/project_status.py --write`.
> It summarizes completed artifacts; it does not imply an episode-training gain.

## Data construction

| Asset | Current snapshot |
|---|---:|
| Planned simulator jobs | 322 |
| Strict simulator bundles | 193 |
| Main episode SFT records | 340 |
| Main isolated SFT records | 466 |
| Main benchmark records | 279 |
| Scene dialogue episodes | 29 |
| Transform teacher records | 2961 |
| Transform geometric facts | 1803 |
| Complete Among-5 families | 16 |

## Frozen zero-shot evaluation

| Model | Prompt | Accuracy | Complete trace |
|---|---|---:|---:|
| SenseNova 1.5 | Answer-only | 36.24% | 0.00% |
| SenseNova 1.5 | Grounded-CoT | 38.33% | 0.00% |
| SenseNova 1.1 | Answer-only | 24.21% | 0.00% |
| SenseNova 1.1 | Grounded-CoT | 22.05% | 44.85% |

## Interpretation

- SenseNova 1.5 is substantially more accurate than 1.1 on the Transform Pilot.
- SenseNova 1.1 Grounded-CoT can produce the requested format but often changes a correct answer.
- SenseNova 1.5 ignores the cue/transform contract, so its prompt gain is not evidence of explicit trace reasoning.
- The paired episode-vs-isolated SFT experiment is still required before claiming episode-learning gains.

## Immediate engineering work

1. Replace copied legacy sibling paths with a workspace resolver.
2. Add a self-contained golden bundle for bundle-to-episode replay.
3. Freeze one paired training protocol and run three random seeds.
4. Generate paper tables directly from versioned summary JSON.
