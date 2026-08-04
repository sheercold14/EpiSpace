# Episode 与 isolated 的事实/视觉暴露对照

输入是同一批 `comparison_id` 下事实一一对应的 `train.episode_sft.jsonl` 和
`train.isolated_sft.jsonl`。运行：

```bash
../habitat/.conda/core/bin/python scripts/build_compute_matched_schedules.py \
  --episode data/epispace_pilot_v1/train.episode_sft.jsonl \
  --isolated data/epispace_pilot_v1/train.isolated_sft.jsonl \
  --output-dir data/epispace_pilot_v1/compute_matching \
  --seed 17
```

输出四个 JSONL schedule 和一个 `compute_matching_manifest.json`：

- `fact_matched`：两臂所有源记录各出现一次，事实多重集严格相同；它测量 episode
  打包本身带来的视觉计算节省。
- `image_occurrence_matched`：逐 `comparison_id` 重复 episode 记录，使两臂每张图的
  实际出现次数严格相同；重复 `k` 次时每次 loss weight 为 `1/k`，因此有效事实权重
  在一个 group 内严格相同。它隔离“看图次数不同”这一混杂因素。

schedule 的 `source_jsonl + source_line` 指向原训练样本，`sample_weight` 直接乘到该条
样本的 assistant-only loss。固定 seed 后顺序由 SHA-256 排定，不依赖 Python 随机数
实现。manifest 记录源文件哈希、逐事实/逐图多重集、图像出现次数、输入像素暴露和
透明的 Unicode lexeme 文本 token 估计；token 估计不冒充 Qwen tokenizer 精确计数。
训练加载器应先比对 manifest 内的源 JSONL SHA-256；否则源文件重建后，旧的行号索引
不应继续使用。

loss 不能做整段 assistant token mean。先把每个编号答案映射到精确 tokenizer span，
对每个事实内部取 mean CE，再在 record 内求和并乘 `sample_weight`：

`loss_i = w_i * sum_f mean_{t in answer_span(i,f)} CE_it`。

编号、system/user、image placeholder、program 和 certificate 都不计 loss。这样 episode
含 `k` 个事实、重复 `k` 次且每次权重 `1/k` 时，每个事实跨 epoch 的有效权重
恰好为 1；整段 token mean 会让答案长度和一条记录中的事实数改变监督质量。两臂的
prompt 与 answer 编号格式也必须一致，避免“一问/多问”之外的 surface cue 成为处理变量。

## Optimizer matching：主协议必须按 comparison group 更新

schedule 配平本身不是完整的优化器契约。若把 651 个 draw 逐条执行
`Adam.step()`，即使 episode draw 乘了 `1/k`，也不能获得与组级梯度等价的更新：Adam
会在每个 draw 后更新一、二阶动量，而且对整体 scale 近似不敏感。高 `k` group 因而会
获得额外 optimizer dynamics，形成一个足以推翻因果比较的混杂。

正式训练必须使用：

```bash
scripts/train_qwen3vl_lora.py ... --optimizer-unit comparison_group
```

训练器忽略 JSONL 的全局交错次序，先按 `comparison_id` 重新分组并执行硬验证：

- episode 组必须是同一 `k`-fact record 的 `k` 次重复，repeat index 为 `0..k-1`，
  每次权重恰为 `1/k`；
- isolated 组必须是 `k` 条互异的单事实 record，每次权重为 1，facts 不重复；
- 配对两臂必须拥有相同 comparison IDs、每组相同 `k`、相同 fact multiset；
- group 顺序由 `SHA-256(seed, epoch, comparison_id)` 决定，两臂一致；
- group 内逐 draw 累积梯度，组末才调用一次 clipping、Adam 和 scheduler；
  `max_steps` 指完整 group updates。

该模式当前只允许单进程。`--optimizer-unit draw` 是向后兼容的诊断模式，不能支撑
episode-vs-isolated learning gain。由此主比较每个 epoch 两臂同为 340 个 optimizer
updates；651 是每臂的 multimodal forwards 数，不再被误当成独立 Adam steps。

## 当前 pilot 快照（seed 17）

最终 release 共编译 1,093 个问题；其中进入训练对照的是 340 条 episode 记录与 651 条
isolated 记录，组成 340 个 comparison group，并共享完全相同的 651 个训练事实。

| regime / arm | schedule draws | facts（实际） | image occurrences | pixels | text-token proxy |
|---|---:|---:|---:|---:|---:|
| fact-matched / episode | 340 | 651 | 1,168 | 1,224,736,768 | 82,031 |
| fact-matched / isolated | 651 | 651 | 3,227 | 3,383,754,752 | 122,437 |
| image-matched / episode | 651 | 2,083 | 3,227 | 3,383,754,752 | 207,510 |
| image-matched / isolated | 651 | 651 | 3,227 | 3,383,754,752 | 122,437 |

image-matched episode 行的 2,083 是重复 draw 产生的“实际事实出现次数”；乘以每条的
`sample_weight=1/k` 后，其 651 个事实的有效权重都恰好为 1，与 isolated 臂严格一致。
fact-matched 下 episode 只需 isolated 的 36.19% 图像输入（节省 63.81%）；image-matched
则主动消除这项视觉计算优势，并如实暴露两臂并不相等的文本 token 计算。

Qwen3-VL 在 `256×256` 输入上的精确 processor 统计如下（不是 Unicode proxy）：

| arm | input tokens | image tokens | text tokens | supervised answer tokens |
|---|---:|---:|---:|---:|
| image-matched episode | 415,424 | 206,528 | 208,896 | 30,034 |
| image-matched isolated | 362,185 | 206,528 | 155,657 | 10,468 |

每张图均对应 64 个 image tokens，因此视觉 token 完全相等；episode 总输入 token 多
14.70%。这是一项严格的 **visual-exposure control**，不是总 token/FLOP 相等。论文和
实验报告必须同时给出这张表；若主张 compute-matched，需另加 no-loss padding、按实测
FLOPs 截止或等价控制。

最终输入及产物 SHA-256：

- episode JSONL：`34135ef2658a9c6e706a4fcadc2319aa22b4c7fa8d9bbd50f975e0b4dbcbd0fe`
- isolated JSONL：`a325146f3a0d13b959c45a901c89584588ffab4b1c6ebe116ea8dd5e572c1a1c`
- manifest：`d3928445112c69b7716e1d33e4ef684d9e9e75e1f62a6ace34e1fb21b7f57889`
- fact-matched episode / isolated schedule：
  `59dd9d768ad2d08ffa4cb87a5050d02ff985fc56b619c59c1c817cc1fd7c7640` /
  `f3b571f86cc6540e765fd98d6c275be8576323831004ab18b81a2a4893f06069`
- image-matched episode / isolated schedule：
  `99307dd9e71c8cd318653f46a68830d5d09e74d969c45dcdc78852ee696e2770` /
  `a65811c67b0d5041fda98132504836902205294bac9c1af35a546aee3b577e42`
