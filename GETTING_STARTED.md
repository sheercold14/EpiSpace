# Getting Started

## 1. Clone and install the lightweight environment

```bash
git clone <YOUR_GITHUB_URL> EpiSpace
cd EpiSpace
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verify the source snapshot without simulator assets:

```bash
make doctor
make quickstart
make status
```

## 2. Connect an existing workspace

Full replay uses trajectory bundles and models stored outside Git:

```bash
cp workspace.example.env workspace.env
```

Edit the paths in `workspace.env`. The repository must remain portable: do not commit this file
or add machine-specific absolute paths to versioned configs.

## 3. Install the acquisition backend

The lightweight package can be installed independently:

```bash
python -m pip install -e backends/omnigibson
```

Actual rendering uses the official OmniGibson/Isaac Sim environment. Follow
`backends/omnigibson/README.md`; do not install Isaac Sim into the core compiler environment.

## 4. Read the project in the intended order

1. `README.md`: research question and public interface.
2. `STATUS.md`: completed assets, baselines and open work.
3. `docs/architecture.md`: code and data boundaries.
4. `DATA.md`: schemas, datasets and generated artifacts.
5. `REPRODUCE.md`: exact replay levels and commands.
6. `docs/research/EpiSpace研究脉络与当前数据汇总_v1.md`: complete research context.

