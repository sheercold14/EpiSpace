# Architecture

EpiSpace separates heavy simulator acquisition from lightweight episode compilation.

```text
OmniGibson backend
  recipe → trajectory → sensor bundle → quality report
                                  │
                                  ▼
EpiSpace core
  bundle → observable belief → typed program → certificate
         → episode/isolated SFT → benchmark → evaluation/web
```

## Stable boundary

The acquisition backend writes backend-neutral bundles. The core compiler is read-only with
respect to those bundles. Shared contracts are stable IDs, canonical frame conventions, sensor
channel declarations and versioned schemas.

## Code ownership domains

- `src/episode3d/schemas and models`: serialized interfaces.
- `src/episode3d/programs`: typed spatial operations.
- `src/episode3d/pipeline and exporters`: dataset construction.
- `src/episode3d/*audit* and *verifier*`: independent validation.
- `src/episode3d/*training*`: model-specific adapters and loss accounting.
- `backends/omnigibson`: acquisition only; no training logic.

## Design invariants

1. Geometry first, language last.
2. Model-visible and oracle channels are physically distinguishable.
3. Query generation cannot alter the previously collected trajectory.
4. Same scene/family never crosses a split.
5. Failed and rejected records retain a reason code.
6. Dataset counts are generated from manifests.

