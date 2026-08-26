# EpiSpace coverage render shards

This workflow distributes a data-only coverage snapshot to independent
OmniGibson workers. EpiSpace and the custom `omnigibson_episode` backend are
synchronized separately with Git. Every package records both exact commits,
and a worker refuses to run a different checkout.

Each worker renders existing candidates, runs authoritative instance-mask
validation, constructs question groups, and by default performs on-demand
geometry backfill up to the manifest's 150-attempt ceiling. A worker only
rewrites its private, scene-disjoint shard manifest.

Scenes, rather than individual cells, are the shard unit. A successful episode
may credit several capability cells in the same scene, so scene-level sharding
keeps all status writes disjoint.

## Frozen P2/P3 7k workflow

P2/P3 uses a stricter render-only wrapper than the general coverage workflow
below. It binds both phase manifests to the passing global 7k quality audit,
forces `EPISPACE_REMOTE_BACKFILL=0`, checks code commits, dependencies, GPUs
and a frame-count-derived disk budget, and verifies marker evidence before it
uploads a result.

Build both two-way distributions from a clean, committed EpiSpace and backend
checkout:

```bash
PYTHONPATH=src EPISPACE_PYTHON=python \
EPISPACE_OG_ROOT=/path/to/OminiGibson \
  bash scripts/build_p23_render_shards.sh
```

Upload each distribution under one task root:

```bash
OSS_TASK_ROOT=oss://YOUR_BUCKET/epispace/p23_7k_v1
bash scripts/upload_coverage_render_shards_to_oss.sh \
  outputs/p23_p2_7k_render_shards_v1 "${OSS_TASK_ROOT}/p2/input"
bash scripts/upload_coverage_render_shards_to_oss.sh \
  outputs/p23_p3stream_7k_render_shards_v1 "${OSS_TASK_ROOT}/p3/input"
```

Each four-GPU machine runs its assigned index for both phases. The persistent
work root also owns the sentinel-marked per-run scratch directories:

```bash
EPISPACE_OG_ROOT=/path/to/OminiGibson \
EPISPACE_DATA_ROOT=/path/to/omnigibson-data \
EPISPACE_CONDA_ENV=behavior-spatialep \
  bash scripts/run_p23_shard_from_oss.sh \
  p2 SHARD_INDEX "${OSS_TASK_ROOT}" /data/epispace-p23-worker

EPISPACE_OG_ROOT=/path/to/OminiGibson \
EPISPACE_DATA_ROOT=/path/to/omnigibson-data \
EPISPACE_CONDA_ENV=behavior-spatialep \
  bash scripts/run_p23_shard_from_oss.sh \
  p3 SHARD_INDEX "${OSS_TASK_ROOT}" /data/epispace-p23-worker
```

The worker records the environment preflight and shard verification in its
uploaded output. It requires every planned candidate to reach a terminal
status, but deliberately does not claim the merged release reached 7,000
accepted trajectories; that decision remains a post-merge step.

## 1. Preview while the local pipeline is still running

```bash
cd /home/wmq/project/EpiSpace
python scripts/coverage_render_shards.py build \
  --manifest outputs/behavior51_coverage_v1/coverage.plan.json \
  --output outputs/behavior51_coverage_shards_v1 \
  --num-shards 2 \
  --preview
```

Preview is read-only. A formal build deliberately fails while a process or a
status row is still running against the master manifest.

## 2. Freeze at a candidate boundary and build two packages

Stop the local coverage pipeline only after its current candidate exits. Then:

```bash
python scripts/coverage_render_shards.py build \
  --manifest outputs/behavior51_coverage_v1/coverage.plan.json \
  --output outputs/behavior51_coverage_shards_v1 \
  --num-shards 2
```

The command creates two data-only `.tar.zst` archives and SHA-256 sidecars. The
archives contain plans, recipes, scene IR and status, but no Python source. Do
not restart the local master pipeline after this snapshot; doing so would
overlap the remote scene assignments and invalidate the merge base.

## 3. Upload packages to OSS

```bash
OSS_PREFIX=oss://YOUR_BUCKET/epispace/behavior51_coverage_v1
export OSS_REGION=cn-shanghai  # replace when the bucket is in another region
OSSUTIL_BIN=/home/wmq/.local/bin/ossutil \
  bash scripts/upload_coverage_render_shards_to_oss.sh \
  outputs/behavior51_coverage_shards_v1 "${OSS_PREFIX}/input"
```

No credentials or source code are embedded in any package. Each machine uses
its own ossutil configuration or `OSS_...` environment variables.

## 4. Run one archive on each four-GPU worker

First clone or fast-forward both code repositories to the commits recorded by
the package. The EpiSpace worker script then selects the archive from
`distribution.json`, checks its SHA-256, verifies both Git HEADs, runs all four
GPUs with remote backfill enabled, and uploads the completed result. Worker 0
uses shard index `0`; worker 1 uses index `1`.

For the current Behavior-51 deployment, the convenience launcher supplies the
OSS prefix, four-GPU defaults, persistent work root, log file, and detached tmux
session. Run index `0` on the first machine and index `1` on the second:

```bash
cd /home/wmq/project/EpiSpace
bash scripts/launch_behavior51_remote_shard.sh 0
# On the other machine:
bash scripts/launch_behavior51_remote_shard.sh 1
```

Use `tmux attach -t epispace-s0` / `epispace-s1` or follow
`/home/wmq/epispace-distributed/logs/shard-INDEX.log`. The lower-level command
below remains available when paths or the OSS task root need customization.

```bash
OSS_PREFIX=oss://YOUR_BUCKET/epispace/behavior51_coverage_v1
export OSS_REGION=cn-shanghai  # replace when the bucket is in another region
cd /home/wmq/project/EpiSpace
EPISPACE_OG_ROOT=/home/wmq/project/bench/OminiGibson \
EPISPACE_DATA_ROOT=/home/wmq/project/bench/BEHAVIOR-1K/datasets \
EPISPACE_CONDA_ENV=behavior \
EPISPACE_GPU_IDS="0 1 2 3" \
EPISPACE_WORKERS=4 \
EPISPACE_REMOTE_BACKFILL=1 \
  bash scripts/run_behavior51_shard_from_oss.sh \
  "${OSS_PREFIX}" SHARD_INDEX /home/wmq/epispace-distributed
```

Rerunning the worker command resumes safely: its status and any newly generated
plans are retained, while an interrupted candidate is reset from `running` to
`pending`. Set `EPISPACE_REMOTE_BACKFILL=0` only for a strictly render-only
diagnostic pass.

## 5. Download and merge on the master machine

Download each `SHARD_NAME` reported by `shard.complete.json`:

```bash
bash scripts/sync_coverage_shard_results_from_oss.sh \
  "${OSS_PREFIX}/results" SHARD_NAME \
  outputs/behavior51_coverage_shard_results_v1
```

After both are present:

```bash
python scripts/coverage_render_shards.py merge \
  --manifest outputs/behavior51_coverage_v1/coverage.plan.json \
  --results \
    outputs/behavior51_coverage_shard_results_v1/SHARD_NAME_0 \
    outputs/behavior51_coverage_shard_results_v1/SHARD_NAME_1
```

Merge verifies the base manifest/status hashes, disjoint scene assignments,
final manifest/status hashes, and newly accepted bundle/group evidence. It
copies remotely generated plans and recipes back with local paths, writes
pre-merge backups, and maintains an idempotent merge ledger.
