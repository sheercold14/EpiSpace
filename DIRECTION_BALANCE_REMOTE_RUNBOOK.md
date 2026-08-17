# Direction Balance 四卡远端渲染与合并

本流程只迁移 `outputs/behavior51_direction_balance_v1`。本机正在运行的
`epispace-v2-local-repair` 和等待中的 `epispace-occlusion-repair` 不受影响。

四个 label manifest 各自生成一个完整 scene shard，远端依次使用 GPU
`0 1 2 3` 和 4 个 worker。初始为 0 candidate 的 cell 也包含在 shard 中，
远端开启 geometry backfill 后继续搜索至 manifest 声明的 150 次上限。

## 1. Git 前置条件

分片只包含数据和 plan，不包含代码。先提交并推送当前 EpiSpace 修改；
OminiGibson backend 也必须是干净、已推送的 commit。远端分别 checkout 到
shard 记录的精确 commit。若任一仓库的 commit 不一致，远端 runner 会拒绝启动。

## 2. 本机构建并上传

使用新的、不可变 OSS 前缀；建议把 EpiSpace commit 短哈希写入前缀：

```bash
cd /home/wmq/project/EpiSpace

export OSS_REGION=cn-shanghai
export OSSUTIL_BIN=/home/wmq/.local/bin/ossutil
export OSS_PREFIX=oss://brain-imagegen-sh/wmq/epispace/behavior51_direction_balance_v1_git_<commit>

bash scripts/upload_behavior51_direction_balance_to_oss.sh "$OSS_PREFIX"
```

上传结构为：

```text
<OSS_PREFIX>/direction_balance.distribution.json
<OSS_PREFIX>/pure_rotation_left/input/...
<OSS_PREFIX>/pure_rotation_right/input/...
<OSS_PREFIX>/pure_translation_left/input/...
<OSS_PREFIX>/pure_translation_right/input/...
```

上传脚本会初始化四份 idle `coverage.status.json`、检查本地没有 direction
renderer、检查 tracked code 已提交、建立 SHA-256 绑定的单 shard archive，再上传。

## 3. 四卡机启动

远端先 pull/checkout shard 对应的 EpiSpace 和 OminiGibson commit，然后设置实际路径：

```bash
cd /path/to/EpiSpace

export OSS_REGION=cn-shanghai
export OSSUTIL_BIN=/path/to/ossutil
export EPISPACE_REPO_ROOT=/path/to/EpiSpace
export EPISPACE_OG_ROOT=/path/to/OminiGibson
export EPISPACE_DATA_ROOT=/path/to/BEHAVIOR-1K/datasets
export EPISPACE_CONDA_ENV=behavior
export EPISPACE_GPU_IDS="0 1 2 3"
export EPISPACE_WORKERS=4

bash scripts/launch_behavior51_direction_balance_remote.sh \
  "$OSS_PREFIX" \
  /path/to/epispace-direction-balance
```

脚本创建 tmux `epispace-direction-balance-remote`。查看方式：

```bash
tmux attach -t epispace-direction-balance-remote
tail -f /path/to/epispace-direction-balance/logs/direction-balance-remote.log
```

每个 task 完成后立即 finalize 并上传到：

```text
<OSS_PREFIX>/<task>/results/<shard_name>/
```

中断后使用相同命令和相同 `LOCAL_ROOT` 重启，会复用已下载 archive 和 worker status。

## 4. 本机同步并合并

确认远端 tmux 已完成退出，再在本机运行：

```bash
cd /home/wmq/project/EpiSpace

export OSS_REGION=cn-shanghai
export OSSUTIL_BIN=/home/wmq/.local/bin/ossutil

bash scripts/merge_behavior51_direction_balance_from_oss.sh \
  "$OSS_PREFIX" \
  outputs/behavior51_direction_balance_remote_results_v1
```

合并器逐 task 执行：下载完成标记、校验 shard/base/status SHA-256、复制 bundle/group、
重定位远端 backfill plan/recipe 路径、合并 status、重建 dataset。重复执行是幂等的，
已经写入 `coverage.shard_merge.json` 的 shard 会跳过。

最终索引为：

```text
outputs/behavior51_direction_balance_v1/direction_balance.merge.json
```

四份 direction dataset 仍保持独立，后续 final release selector 再把它们与本机
v2 repair、occlusion repair 和最多四条旧 `back`/binding 组合；不直接改写 canonical v1。
