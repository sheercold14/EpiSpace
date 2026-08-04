# Qwen3-VL controlled pilot

这个实验入口用于验证 EpiSpace 的核心因果问题，不属于数据生成 gate。数据 release
即使没有模型实验仍可独立审计；任何 learning-gain 主张则必须来自这里产生的真实预测。

## 环境

本机已验证的训练环境：

```bash
PY=/home/shichao/miniconda3/envs/easyrl/bin/python
MODEL=/data/shichao/data/dataV100/models/viewfusion/base_Qwen3-VL-4B-Instruct
```

需要 `torch>=2.6`、`transformers>=4.57`、`peft`、`flash-attn` 和
`qwen-vl-utils`。当前 pilot 使用 4B，是训练链验证与效应量预估；论文若声称 8B
结论，必须用同一协议在冻结的 8B 初始化上重跑。

先把本仓库以 editable 方式装入训练环境（已有深度学习依赖时可避免重复解析）：

```bash
$PY -m pip install -e . --no-deps
```

## 为什么不能直接使用默认 loss 和逐 draw Adam

Episode 一条 assistant turn 可能包含 `k` 个答案。默认 causal-LM loss 会对整段
answer token 取一次平均；这样大 episode 中每个事实会被隐式缩小，破坏与 isolated
臂的事实配平。训练器执行严格顺序：

1. 从编号答案恢复每个 `fact_id` 的字符 span；
2. 通过 fast-tokenizer offsets 映射到模型 token；
3. 每个事实内部取 mean CE；
4. episode 内对事实 loss 求和；
5. 乘 schedule 的 `sample_weight=1/k`。

Image-occurrence schedule 中同一 episode 重复 `k` 次，因此每个事实跨 epoch 的总
有效权重恰好为 1，与 isolated 中单独出现一次相同。编号本身不进入 loss。单事实
样本既兼容旧的裸答案，也兼容格式配平后的 `1. answer`；后者同样只监督 answer span。

但这还不够：如果每个 draw 都调用一次 `Adam.step()`，episode 中每个事实会随同一
multi-fact record 参与 `k` 次 Adam 状态更新，而 isolated 中每个事实只参与自己的 1 次；
`1/k` 缩放不能把前者变成一次组更新，因为 Adam 对整体梯度缩放近似不敏感，而且每次
都会改变动量状态。因果主协议因此把 **一个 `comparison_id` 定义为一个 optimizer unit**：

1. 用 seed 对 comparison ID 排序，两臂得到完全相同的 group 顺序；
2. group 内逐 draw forward/backward，episode 每个 draw 乘 `1/k`，isolated 乘 1；
3. 组内梯度全部累积后，只调用一次 clipping、`optimizer.step()` 和 scheduler step；
4. `--max-steps` 按 optimizer update（也就是完整 group）计数，绝不在组中途停止。

训练器会同时加载另一臂 schedule，逐组验证 draw 数相等、episode 重复同一 record、
isolated 覆盖同一组 facts、权重和 repeat index 合法。主协议目前明确限制单进程/单 GPU；
DDP 需要按 group 分片并在组内使用 `no_sync`，未实现前会直接拒绝，避免悄悄改变实验。
`--optimizer-unit draw` 只保留为 legacy diagnostic，不能用于 episode-vs-isolated 因果结论。

## Smoke

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/train_qwen3vl_lora.py \
  --model "$MODEL" \
  --schedule data/epispace_pilot_v1/compute_matching/image_occurrence_matched.episode.schedule.jsonl \
  --output-dir experiments/epispace_qwen3vl4b/smoke_episode \
  --optimizer-unit comparison_group \
  --max-steps 2 --overwrite
```

## 冻结比较

严格协议使用单卡；两臂各有 340 个 comparison groups，因此每个 epoch 都是 340 次
optimizer updates。不同 comparison 的 `k` 可以不同，但同一 ID 在两臂严格相同：

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/train_qwen3vl_lora.py \
  --model "$MODEL" \
  --schedule data/epispace_pilot_v1/compute_matching/image_occurrence_matched.episode.schedule.jsonl \
  --output-dir experiments/epispace_qwen3vl4b/seed17_episode \
  --optimizer-unit comparison_group --seed 17 --epochs 3

CUDA_VISIBLE_DEVICES=1 $PY scripts/train_qwen3vl_lora.py \
  --model "$MODEL" \
  --schedule data/epispace_pilot_v1/compute_matching/image_occurrence_matched.isolated.schedule.jsonl \
  --output-dir experiments/epispace_qwen3vl4b/seed17_isolated \
  --optimizer-unit comparison_group --seed 17 --epochs 3
```

两臂必须保持模型、seed、分辨率、LoRA、optimizer、epoch 和 world size 相同。正式
报告至少使用三个 seed，并保留每个 `run_manifest.json` 和原始预测。两臂还必须使用
相同的提问前缀和编号 answer surface，使处理变量只剩“一问 vs 多问”，而不是格式。
manifest 会记录 `optimizer_unit`、group 数、计划/实际 updates 和配对 schedule 哈希。
4B pilot 不能被改写成 Qwen3.5-8B 结果。
