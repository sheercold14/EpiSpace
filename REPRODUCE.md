# Reproducing EpiSpace

Reproduction is reported at three levels rather than assuming path-traced RGB is bitwise stable
across GPU and driver versions.

| Level | Requirement | Main check |
|---|---|---|
| Source replay | Recompile an existing bundle | IR, labels and certificates match |
| Semantic rerun | Re-render the same recipe and assets | scene/pose/object semantics and quality gates match |
| Visual rerun | Compare RGB renders | PSNR/SSIM, exposure, entropy and sharpness thresholds |

## Simulator-free verification

```bash
make install
make quickstart
make test
```

The complete historical replay suite additionally reads trajectory bundles and generated
artifacts kept outside Git. After configuring `workspace.env`, run `make test-artifact`.

## Rebuild Transform Pilot from existing bundles

```bash
cp workspace.example.env workspace.env
# Edit paths, then:
make reproduce-transform-pilot
```

The copied legacy config currently references the original sibling workspace layout. The first
portability task is to route these sources through environment variables without changing the
source projects.

## Paper-result mapping

| Result | Frozen source |
|---|---|
| Data counts | `manifests/datasets/*.json` |
| SenseNova 1.1/1.5 zero-shot | `manifests/experiments/*.summary.json` |
| Representative QA and raw responses | `examples/transform_pilot/showcase.json` |
| Research claims and limitations | `docs/research/EpiSpace研究脉络与当前数据汇总_v1.md` |

Every future paper table should be generated from a versioned summary JSON, not manually copied
from terminal output.
