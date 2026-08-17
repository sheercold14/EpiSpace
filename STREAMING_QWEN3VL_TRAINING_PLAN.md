# EpiSpace Streaming QA 训练方案：Qwen3-VL 与 SenseNova

日期：2026-08-14  
状态：训练设计稿；当前本机数据仅用于 smoke test，不作为正式训练集或 benchmark。

## 1. 结论

EpiSpace streaming QA 可以直接用于训练 `Qwen3-VL-8B-Instruct`。推荐的主训练形式是：

1. 将一整条 episode 序列化为一个交错的多轮多模态 conversation；
2. 一次 forward 处理整条 conversation；
3. system、user、图片及格式外的 token 不计算 loss；
4. 对每个 assistant turn 的 `<answer>标签</answer>` 和 EOS 计算 causal LM loss；
5. 每个 turn 内先对答案 token 取均值，再按 capability、label 和 episode 权重聚合；
6. 同时构造不包含历史 gold answer 的 `prefix-isolated` 数据，作为辅助训练和科学对照；
7. 使用独立的 `drop_key` / `untrackable` history 提供“无法判断”监督，不能把 `invalid` 样本改造成“无法判断”。

正式研究建议以 `Qwen3-VL-8B-Instruct` 为主模型，以冻结的 `SenseNova-SI-1.5-InternVL3-8B` 为空间专项零样本基线，并在资源允许时增加 SenseNova LoRA 作为第二底座实验。

## 2. 当前本机 streaming 数据状态

当前数据目录：

```text
outputs/behavior51_coverage_v1_local_qa/
```

截至本文档生成时：

| 指标 | 数量 |
|---|---:|
| streaming conversation | 518 |
| streaming turn | 2,673 |
| 引用 RGB | 5,311 |
| 平均 turn / conversation | 5.16 |
| turn 数范围 | 2–8 |
| 平均图片 / conversation | 10.25 |
| 图片数范围 | 2–18 |
| capability | 8 |
| 场景 | 2 |

场景分布为：

- `Beechwood_0_garden`：493 条 conversation；
- `school_gym`：25 条 conversation。

turn 的 capability 分布为：

| Capability | Turn 数 |
|---|---:|
| `path_integration_magnitude` | 899 |
| `path_integration` | 499 |
| `view_side_check` | 426 |
| `homing_probe` | 381 |
| `self_motion_update_multi_turn` | 162 |
| `self_motion_update_pure_rotation` | 153 |
| `self_motion_update` | 127 |
| `self_motion_update_pure_translation` | 26 |

这版数据还存在以下已确认问题：

- 所有 2,673 个 streaming turn 都是 `answerable`，没有“无法判断”监督；
- `front` 只有 2 个，答案标签严重不平衡；
- 部分 multi-turn 轨迹实际没有平移；
- 部分 episode 没有释放到完整尾帧，长 delay 难度丢失；
- streaming 的全局 label 去重会删除 `A -> B -> A` 中最后一次 A；
- 343 个 turn 没有新图片，它们依赖完整对话历史；
- 当前只有两个场景，不能支持可信的 scene-disjoint 泛化结论。

因此，当前数据可用于模型加载、loss mask、显存、吞吐量和小样本 overfit 测试，但不能用于正式训练结论。正式训练应等待 shard 合并，并在 QA v2 中完成已记录的清洗与补齐。

详细数据问题见 `SELF_MOTION_STREAMING_AUDIT.md`。

## 3. 一条完整 streaming conversation 如何训练

推荐保留当前的交错结构：

```text
system
user: [frame 0, frame 1] + question 1
assistant: <answer>back</answer>
user: [frame 2] + question 2
assistant: <answer>left_half</answer>
user: [frame 3, ..., frame 6] + question 3
assistant: <answer>left</answer>
...
```

训练时，整条 conversation 一次送入模型。设一条 episode 有 $T$ 个 assistant turn，第 $t$ 个答案 token 集合为 $A_t$，推荐目标为：

$$
L_{episode}
=
\frac{\sum_{t=1}^{T} w_t
\left(
\frac{1}{|A_t|}
\sum_{j \in A_t}
-\log p_\theta(x_j \mid x_{<j})
\right)}
{\sum_{t=1}^{T} w_t}.
$$

其中：

- $A_t$ 包含 `<answer>标签</answer>` 的 token 和该 assistant turn 的 EOS；
- system、user、图片 token、问题、选项和隐藏证书的 loss 均为 0；
- $w_t$ 用于平衡 capability、答案标签、scene 和 abstain；
- 不建议直接对整条 assistant token 流做普通 token mean，否则长答案、长 conversation 和高频 capability 会获得不同且不透明的权重。

当前代码已经具备增量对话的 assistant span 验证和 token 定位能力：

- `src/episode3d/qwen_training.py::supervised_assistant_turns()`；
- `src/episode3d/qwen_training.py::incremental_assistant_token_groups()`；
- `scripts/train_qwen3vl_lora.py::exact_fact_loss()`。

### 3.1 为什么一次 forward 不会看到未来图片

未来图片虽然出现在完整输入序列的后半部分，但 Qwen 使用自回归 causal attention。在计算前面 assistant answer 的 logits 时，attention mask 不允许其访问后续 user turn 和未来图片 token。因此，整条 conversation 一次 forward 仍然满足多模态因果训练要求。

需要注意：

- 不应使用破坏样本边界或 causal mask 的错误 packing；
- 第一版训练建议关闭 data packing；
- 任何上下文截断都必须在完整 image block 和 turn 边界上发生；
- 不得为了长度限制静默删除 certificate 要求的关键帧。

## 4. 图片是 history 还是当前输入

“history”和“input”分别描述语义与计算形式，并不冲突。

| 运行方式 | 旧图片 | 新图片 | 实际模型输入 |
|---|---|---|---|
| Stateful agent / KV cache | 保存在对话或 KV 中 | 当前 turn 追加 | 旧上下文状态 + 新 token |
| Stateless model API | 每轮重新发送 | 当前 turn 追加 | 截至当前的全部消息和图片 |
| 外部记忆 agent | 可被压缩成 belief/state | 当前 turn 追加 | 新图片 + 外部记忆摘要 |

模型本身不会在两个独立请求之间永久保存状态。因此：

- 语义上，早期帧是 history；
- 在普通无状态训练/推理中，早期帧仍要作为当前 forward 的 input；
- 只有 runner 显式保存 KV cache 时，才能不重新编码旧图片；
- 若使用外部结构化记忆，应作为“agent + memory”单独实验，不能和纯模型视觉记忆混为一谈。

当前一轮没有 `new_images` 的 343 个 turn 只有在 runner 保留完整多模态 history 时才成立。若使用 stateless 调用，必须重新提供截至当前的完整图片前缀。

## 5. 两种训练视图

### 5.1 Stateful streaming conversation

直接使用当前完整 `messages`，监督其中所有 assistant turn。

优点：

- 最接近真实“边走边问”；
- 历史图片只在一条训练样本中编码一次；
- 能学习跨轮记忆、标签变化和证据状态更新；
- 对同一 episode 的多个问题共享视觉计算。

风险：

- 后一轮训练会看到此前的 gold assistant answer；
- 训练是 teacher forcing，部署通常是 free-running；
- 模型可能利用先前答案而不是重新读取视觉历史；
- 某一轮回答错误可能在部署时污染后续历史。

### 5.2 Prefix-isolated QA

对每个 streaming turn 单独构造一条记录：

```text
system
user: [截至当前的全部已释放图片] + 当前问题
assistant: 当前答案
```

它不包含此前的 gold assistant answer，因此能检验当前问题是否真的能由图片前缀解决。

优点：

- 不存在历史 gold answer 泄漏；
- capability、label、delay 和 abstain 更容易精确采样；
- 对 stateless API 友好；
- 可作为 streaming conversation 的科学对照。

代价：

- 历史图片会在不同 turn 中反复编码；
- 总视觉 token 和训练 FLOPs 明显增加；
- 若直接与 streaming arm 比较，必须匹配 fact、权重、optimizer update 和视觉暴露量。

### 5.3 推荐实验组织

正式实验至少保留以下 arm：

| Arm | 输入组织 | 用途 |
|---|---|---|
| Base | 不训练 | 零样本基线 |
| Isolated-prefix LoRA | 每个 turn 独立 | 无历史答案对照 |
| Streaming LoRA | 完整多轮 conversation | 主 streaming 实验 |
| Mixed LoRA，可选 | 两种视图共同训练 | 面向最终产品性能 |

如果构造 Mixed 训练，不应让同一事实因为出现两次而权重翻倍。可将每个事实的总权重拆为：

```text
0.5 × streaming loss + 0.5 × isolated-prefix loss
```

如果目的是论文中的因果比较，应分别训练 Streaming 与 Isolated 两个 arm，不应先混合后再声称差异来自序列组织。

## 6. Teacher-forced、free-running 与答案泄漏

### 6.1 Teacher-forced

第 $t$ 轮推理历史中使用前面各轮 oracle answer。

它测量：在历史状态正确时，当前轮是否能完成空间判断。适合作为诊断指标，但会高估真实连续运行效果。

### 6.2 Free-running

第 $t$ 轮历史中保留模型前面各轮自己的输出。

它同时测量：

- 当前轮空间能力；
- 输出格式稳定性；
- 历史错误累积；
- 模型能否在新证据到达后修正旧答案。

正式 streaming 主指标应使用 free-running。Teacher-forced 只作为条件化诊断。

### 6.3 Prefix-isolated

完全不提供前面的 assistant answer，仅提供截至当前的视觉证据。它用于测量纯视觉前缀能力，并判断完整 conversation 的提升是否来自先前答案捷径。

## 7. “无法判断”监督

当前 P1 streaming 仅导出 `answerable` prefix，所以没有“无法判断”。正确方案不是把所有早期 prefix 都改为 abstain。

编译器语义必须保持：

| Certificate 状态 | 含义 | 是否进入训练 |
|---|---|---|
| `answerable` | 问题有效且证据充分 | 是，使用权威标签 |
| `abstain` | 问题有效但证据不足 | 是，gold 为“无法判断” |
| `invalid` | 不满足问题设计或几何条件 | 否 |

推荐主来源是独立 `drop_key` streaming：

```text
canonical history
  -> 保留关键 sighting frame
  -> answerable

drop_key history
  -> 删除关键 sighting frame
  -> compiler.status = abstain
  -> gold = 无法判断
```

约束：

- canonical 与 drop-key 必须是独立 conversation；
- drop-key history 中绝不能出现被删除的关键帧；
- 不能在同一 conversation 中先给 canonical 再给 drop-key，否则 gold history 已经泄漏；
- 仅使用 compiler 授权的 abstain；
- `invalid` 绝不能转换成“无法判断”。

推荐总体比例：

```text
answerable : abstain = 3:1 到 4:1
```

还需要按 capability、图片数量、prefix 长度和问题模板匹配两类样本，避免模型学到“图少”“问题短”或某个模板就回答“无法判断”。

## 8. 数据切分

不能随机按 JSONL 行切分。相同视觉内容和几何事实会以多个 capability、family variant 和 streaming turn 重复出现，随机切分会造成严重泄漏。

推荐切分规则：

1. 优先按 house/building cluster 切分；若没有 house 层，再按 scene；
2. 同一 trajectory 的所有 capability 放在同一 split；
3. 同一 binding、question group 和 family 的所有记录放在同一 split；
4. `canonical`、`permute`、`drop_key`、`drop_filler`、`delay` siblings 放在同一 split；
5. QA 模板改写不能跨 split 复制同一事实；
6. 先冻结 split，再导出训练与评测视图。

若最终确有 51 个独立场景，可从约 `35/8/8` 的 train/validation/test 场景划分开始；实际应根据 house cluster、capability 和目标类别覆盖进行约束优化，不能机械随机抽取。

## 9. 采样与 loss 权重

当前 capability 和 label 分布不平衡。若直接均匀采样 conversation，`path_integration_magnitude`、`back`、Beechwood 场景会主导梯度。

推荐分层采样顺序：

1. 选择 scene/house；
2. 选择 capability；
3. 选择答案标签；
4. 选择 delay/prefix bin；
5. 选择 episode/turn。

在不能完全平衡时，可使用平滑的 inverse-sqrt 权重：

$$
w_{cap}(c) \propto \frac{1}{\sqrt{N_c}},
\qquad
w_{label}(y\mid c) \propto \frac{1}{\sqrt{N_{c,y}}}.
$$

不建议直接使用完全 inverse frequency，因为少量异常样本会获得过大权重。

每条 episode 的总权重还应归一化，防止 8-turn conversation 天然比 2-turn conversation 获得四倍梯度。推荐每个 turn 内 token mean，然后按 $w_t$ 做归一化加权平均。

## 10. Qwen3-VL-8B 推荐配置

首选：

```text
Qwen/Qwen3-VL-8B-Instruct
```

不建议第一轮使用 Thinking 版。当前任务要求稳定输出有限标签，隐藏长推理过程更容易控制格式、计算量和评测解析。

建议的第一轮 LoRA 配置：

| 参数 | 建议值 |
|---|---|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| learning rate | 从 `2e-5` 开始 |
| LR sweep | `1e-5 / 3e-5 / 1e-4` |
| epoch | 2–3 |
| warmup ratio | 0.03–0.05 |
| precision | BF16 |
| gradient checkpointing | 开启 |
| Flash Attention 2 | 开启 |
| batch size / GPU | 1 条 conversation 起步 |
| effective global batch | 8–16 条 conversation |
| max grad norm | 1.0 |
| decoding during eval | greedy，temperature 0 |

第一阶段建议：

- 冻结 vision tower；
- 对语言模型 attention/MLP 使用 LoRA；
- 先验证仅语言适配能否学会 streaming 状态更新；
- 若 held-out scene 上仍表现为视觉域不匹配，再单独评估 projector/merger 或最后若干视觉层的小学习率适配。

不要默认同时训练全部视觉层。当前数据量较小，模拟器风格和标签偏置可能导致过拟合或破坏基础视觉能力。

### 10.1 图像分辨率

建议 profile：

```text
max_pixels = 65,536 / 131,072 / 262,144
```

选择不会显著降低小目标、遮挡和最后可见位置判断准确率的最低分辨率。当前源图为高分辨率，多图 conversation 的主要显存成本来自视觉 token 和中间激活，而不是文本上下文。

不能只因为模型支持长上下文就认为显存一定足够。最长 18 图的训练 forward 仍需要单独测试。

### 10.2 4×4090 训练

8B BF16 权重加上多图激活会使 24GB 单卡非常紧张。推荐优先考虑：

- 4-bit QLoRA；或
- FSDP / DeepSpeed 参数分片；
- Flash Attention 2；
- gradient checkpointing；
- 按图片数和估算 token 数做 length bucket；
- batch size 1，再通过 gradient accumulation 获得有效 batch。

普通 DDP 会在每张卡完整复制基础模型，不能解决权重与激活的单卡显存压力。

## 11. 数据增强边界

允许：

- 经过模板审计的同义问题改写；
- 选择顺序随机化；
- compiler 授权的 drop-key、drop-filler、delay 和安全 permute；
- 轻微、不会改变几何语义的颜色扰动，且需要单独做视觉质量检查。

不建议：

- 未同步修改标签的水平翻转；
- 随意打乱时间顺序；
- 随机删除关键帧后仍保留原答案；
- 任意 crop 导致目标消失；
- 将 pose、instance mask、certificate witness 或 oracle geometry 输入模型；
- 将失败轨迹当作普通数据增强。

如果未来部署的 agent 会收到动作或里程计，应新增 `RGB + action/odometry` 的独立 track。它与当前纯 RGB 空间记忆任务不同，不能在同一结果中混报。

## 12. 训练前验证

### 12.1 零样本基线

至少运行：

- Qwen3-VL-8B-Instruct zero-shot；
- SenseNova-SI-1.5-InternVL3-8B zero-shot；
- 多模态与 vision-free；
- teacher-forced、free-running 和 prefix-isolated。

### 12.2 小样本 overfit

在正式训练前，取 16–32 条 conversation：

1. 只训练这些记录；
2. 检查 loss 是否稳定下降；
3. 检查能否接近完全记忆；
4. 验证重复答案的 token span 是否定位正确；
5. 验证较早 assistant turn 的 loss 不依赖后续图片；
6. 验证一轮没有新图片时仍能读取历史图像；
7. 验证模型输出严格为 `<answer>标签</answer>`。

若小样本不能 overfit，不应继续扩量。应先检查 chat template、图片顺序、answer span、EOS、loss mask、梯度是否进入 LoRA 参数以及上下文是否被截断。

## 13. 评测指标

不能只报告全部 turn 的 micro accuracy。正式结果至少包括：

- turn exact-label accuracy；
- capability macro accuracy；
- label macro accuracy；
- scene/house macro accuracy；
- final-turn accuracy；
- episode all-turn correctness；
- 标签变化 / evidence-update accuracy；
- abstain precision、recall 和 F1；
- canonical 与 drop-key 的配对一致性；
- teacher-forced 与 free-running gap；
- prefix-isolated 与 full-history gap；
- 按 scene/trajectory 聚类 bootstrap 的置信区间。

必须加入 shortcut baseline：

1. vision-free：去掉所有图片；
2. shuffled/reversed frames：打乱或倒序；
3. drop-key：删除权威关键观察；
4. question-only label prior；
5. 可选 current-frame-only：只给当前新图片，不给历史。

如果去掉图片、打乱帧或删除关键帧后性能基本不下降，说明模型主要学习了题目模板、答案频率或历史 assistant answer，而不是空间因果序列。

## 14. SenseNova 1.5 的定位

`SenseNova-SI-1.5-InternVL3-8B` 基于 InternVL3-8B，而不是 Qwen3-VL。它已在大量空间智能数据上训练，因此适合作为空间专项冻结诊断模型。

推荐模型矩阵：

| 模型 | EpiSpace 训练 | 作用 |
|---|---|---|
| Qwen3-VL-8B | 否 | 通用底座零样本 |
| SenseNova 1.5 | 否 | 空间专项零样本 |
| Qwen3-VL-8B | LoRA | 验证 EpiSpace 对通用模型的教学增益 |
| SenseNova 1.5 | LoRA | 检验专项模型能否进一步学习 streaming/self-motion |

若只能选择一个正式研究主模型，优先 Qwen3-VL-8B，因为 EpiSpace 的增量贡献更容易解释。若目标是追求最高应用性能，可优先尝试 SenseNova LoRA。

SenseNova 可能已有较强的静态空间与立体几何能力，但不一定已经解决：

- 多轮增量图片释放；
- 第一人称长时目标记忆；
- 自运动状态更新；
- drop-key 后的可靠弃答；
- free-running 历史误差恢复。

因此 EpiSpace 对 SenseNova 仍可能具有互补价值。但 SenseNova 使用 InternVL3 架构，不能直接套用 Qwen trainer，需要单独的 InternVL/SenseNova LoRA 训练入口。

## 15. 当前代码与缺口

已有能力：

- `outputs/behavior51_coverage_v1_local_qa/streaming_qa.jsonl` 已是多轮多图训练记录；
- `assistant_span_contract` 标出全部 assistant turn；
- `loss_policy` 声明只训练 assistant turn；
- `src/episode3d/qwen_training.py` 支持 incremental assistant token group；
- `src/spatial_episode/scriptgen/qa_evaluation.py` 支持 SenseNova 的四条件 streaming 评测；
- 评测 input 与 oracle 分离，并使用 hash 绑定。

尚需实现：

1. `build_streaming_sft_schedules.py`
   - scene/house split；
   - full conversation 与 prefix-isolated 导出；
   - capability/label/abstain 权重；
   - 数据 hash、模型可见图片清单和 split manifest。
2. `train_qwen3vl_streaming_lora.py`
   - 直接读取 streaming schedule；
   - per-turn weighted loss；
   - QLoRA/FSDP 或多卡训练；
   - 可复现的 checkpoint 与 provenance。
3. `predict_qwen3vl_streaming.py`
   - teacher-forced；
   - free-running；
   - prefix-isolated；
   - vision-free 与干预条件。
4. streaming release verifier
   - 无未来帧；
   - image hash；
   - source frame 顺序；
   - assistant span；
   - oracle 隔离；
   - split cluster 无泄漏；
   - 每个 turn 的 compiler-authorized status。

现有 `scripts/train_qwen3vl_lora.py` 主要服务 paired episode/isolated comparison schedule。其 processor、span 和 loss 实现可以复用，但当前 loader 要求 `comparison_contract` 和 paired schedule，不能直接把 `streaming_qa.jsonl` 当作训练命令输入。

## 16. 推荐执行顺序

### 阶段 A：现在即可进行

1. 用 SenseNova 1.5 对 5 条 streaming conversation 做 smoke test；
2. 扩到 50 条，运行四条件零样本诊断；
3. 用 Qwen3-VL 做相同零样本测试；
4. 实现 streaming Qwen exporter/trainer；
5. 用 16–32 条记录做 Qwen LoRA overfit；
6. profile 三档图像分辨率和最长 18 图样本。

当前结果只用于检查协议与数据，不作为正式 benchmark。

### 阶段 B：等待全部 shard 合并

1. 保持正在运行的 v1 master manifest/status hash 不变；
2. 合并所有远端渲染结果；
3. 冻结原始 merged v1 和 ledger；
4. 对全部 accepted/rejected/pending 运行只读审计；
5. 生成 quarantine 和 coverage credit 撤销清单；
6. 修复 motif、checker、streaming event 和 abstain policy；
7. 生成新的 QA v2，不覆盖 v1。

### 阶段 C：正式训练

1. 冻结 scene/house split；
2. 生成 Streaming 与 Isolated 两套 fact-matched schedule；
3. 跑 Qwen/SenseNova 零样本；
4. 训练 Qwen Streaming LoRA；
5. 训练 Qwen Isolated-prefix LoRA；
6. 资源允许时训练 SenseNova LoRA；
7. 在 held-out scene 上运行四种历史条件和 shortcut baseline；
8. 汇总绝对性能、相对零样本增益、teacher/free gap 与证据敏感性。

## 17. 正式开训门槛

只有全部满足以下条件才进入正式训练：

- merged 数据已冻结并具有完整 hash/provenance；
- 训练、验证、测试按 scene/house cluster 无泄漏切分；
- degenerate multi-turn 已隔离或修复；
- streaming 不再丢失 `A -> B -> A` 标签返回事件；
- 需要完整 episode 的记录已释放尾部帧；
- source capability 的最终查询和 delay 分布已恢复；
- 每个运动问题前已有真实可观察运动；
- abstain 仅来自 compiler-authorized intervention；
- answerable/abstain、capability 和 label 分布达到发布阈值；
- 所有模型可见 RGB 存在且 hash 正确；
- oracle、certificate、pose、instance mask 和 hidden geometry 不进入模型输入；
- Qwen 16–32 条 overfit 测试通过；
- 最长 conversation 的显存与 token profile 通过；
- zero-shot、vision-free 和 frame intervention 基线已保存。

满足这些条件后，完整 streaming conversation SFT 才能既服务于模型训练，也支持对“多模态因果序列是否优于孤立 QA”的可信研究结论。
