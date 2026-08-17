# Self-Motion 与 Streaming QA 审计及修复方案

更新时间：2026-08-14  
适用项目：EpiSpace `feature/scriptgen-engine-v1`  
当前阶段：远端 render shard 尚未全部合并

## 1. 结论

当前 self-motion 数据的权威答案、媒体引用和 rendered pose 总体一致，但存在会阻止正式训练发布的问题：

1. 部分 multi-turn 轨迹退化为全程原地不动；
2. pure rotation / pure translation 的最终答案塌缩为 `back`；
3. 当前 occluded 定义即使生成成功，也会使答案几乎恒为 `front`；
4. streaming 事件去重会漏掉 `A -> B -> A` 的最后一次答案变化；
5. streaming 没有保证在完整 episode 末帧再次提问，丢失尾帧和 delay 难度；
6. 当前 2,673 个 streaming turn 全部是 `answerable`，没有任何“无法判断”监督；
7. 一部分运动问题在两张相同 pose 后就被提问；
8. 仍有文案、可见性、渲染稳定性和数据分布方面的质量风险。

应当等所有 shard 完成并按原始基准合并后，再统一审计、撤销 credit 和定向 backfill。现在不要修改 master 的 `coverage.plan.json` 或 `coverage.status.json`，否则可能破坏 shard merge 的基准哈希。

不需要全部重新渲染。健康的 scene IR、plan、RGB、depth、instance 和 bundle 都可以保留；只重新渲染被撤销后形成的 coverage 缺口，并从保留 bundle 重新构建 QA。

## 2. Self-motion capability 定义

五类 self-motion 共享以下问题语义：

> 目标物体已经离开当前直接视觉证据；以当前或最后的位置和朝向为准，目标现在位于 front、left、back、right 中的哪个方向？

共同关键帧：

- `t_seen`：目标最后一个确定可见的帧；
- `t_gone`：`t_seen` 之后第一个确定不可见的帧；
- `t_q`：当前 prefix 或完整 episode 的最后一帧；
- `delay = t_q - t_gone`；
- 当前要求 `delay >= 3`。

| Capability | Motif | 预期运动语义 | 额外 checker |
|---|---|---|---|
| `self_motion_update` | `walk_and_turn` | 一般行走并转向 | `t_seen:t_q` 累计转角 80–200 度 |
| `self_motion_update_pure_rotation` | `stand_and_turn` | 原地旋转 | 最大位移不超过 0.1m，累计转角 80–200 度 |
| `self_motion_update_pure_translation` | `walk_straight_past` | 朝向不变地直线经过目标 | 全程累计转角不超过 2 度 |
| `self_motion_update_multi_turn` | `walk_multi_turn` | 直行和转向交替 | 2–3 个分离转向段，累计转角 80–200 度 |
| `self_motion_update_occluded` | `walk_to_occlusion` | 目标被真实物体遮挡 | instance/depth 权威遮挡归因 |

`canonical / permute / drop_key / drop_filler / delay` 是与上述 capability subtype 正交的 family intervention 维度，不是另外五种运动类型。

## 3. 当前合并前快照

当前本机 accepted episode 共 520 条，其中 self-motion source episode 为：

| Source capability | Accepted episode |
|---|---:|
| `self_motion_update` | 110 |
| `self_motion_update_pure_rotation` | 153 |
| `self_motion_update_pure_translation` | 20 |
| `self_motion_update_multi_turn` | 121 |
| `self_motion_update_occluded` | 0 |
| 合计 | 404 |

当前派生 QA：

- family：2,034；
- raw QA：8,966；
- streaming record：518；
- streaming turn：2,673；
- streaming 引用 RGB：5,311 张，约 1.136GB。

这些数量是 shard 合并前快照，不能当作最终数据量。

## 4. 已确认正常的部分

以下检查已经通过：

- 404 条 self-motion plan pose 与 OmniGibson 实际渲染 pose 一致，误差小于 `1e-6`；
- 当前 canonical certificate、family label 和 group label 一致；
- 当前 2,673 个 streaming gold 都与对应 certificate 一致；
- 没有未来图片泄漏；
- prefix 单调，媒体 ordinal 连续；
- 5,311 张 streaming RGB 全部存在且 SHA256 匹配；
- family referent uniqueness、未填模板、帧号泄漏、答案 token 泄漏检查全部通过；
- pure translation 当前 20 条都是真正的直线运动，位移 4.50–5.81m，累计转角为 0；
- multi-turn 除已知 18 条退化轨迹外，其余 103 条都至少移动了 2m。

因此当前主要问题不是大范围 gold 算错，而是轨迹语义、采样分布和 streaming 调度不符合最终训练目标。

## 5. 问题总表

| 优先级 | 问题 | 合并前影响 | 修复方法 | 是否需要重新渲染 |
|---|---|---:|---|---|
| P0 | multi-turn fallback 返回相同起终点 | 121 accepted 中 18 条全程原地；另有 348 条 pending 退化 plan | fallback 返回 `None` / `MotifUnavailable`；增加最小位移 clause；撤销全部衍生 credit | 只补缺口 |
| P0 | streaming 全局 `seen_labels` 去重 | 50 个 family，漏 53 个 turn | 只去除连续相同答案，保留 `A -> B -> A` | 否 |
| P0 | source capability 未在完整 `t_q` 再问 | 404 条中 320 条缺少 full-episode source 问题 | 强制添加 canonical endpoint event | 否 |
| P0 | streaming 截断尾部帧 | 404 条中 318 条未释放完整轨迹 | endpoint event 释放所有剩余帧，或增加 final image-only event | 否 |
| P0 | pure rotation label collapse | 153/153 为 `back` | 采样约 ±100° 和 ±150°，按 left/right/back 配额接受 | 定向补采 |
| P0 | pure translation label collapse | 20/20 为 `back` | 使用 side/back 两种不对称终点，按标签配额接受 | 定向补采 |
| P0 | occluded 定义导致常量答案 | 当前 0 accepted；按现定义未来几乎全为 `front` | 遮挡发生在中间 `t_occ`，之后继续运动，到更晚的 `t_q` 提问 | 是 |
| P0 | streaming 没有 abstain gold | 2,673/2,673 都是 answerable | 增加独立 drop-key / untrackable intervention stream | 否，可复用现有帧 |
| P1 | 通用 self-motion 无最小行走距离 | 4 条路径不足 1m，9 条起终点不足 2m | 增加 `path_length_ge` / `endpoint_displacement_ge` | 只补淘汰缺口 |
| P1 | 两张相同 pose 就问运动问题 | pure rotation 153 + multi-turn 118，共 271 个开场事件 | 使用“已发生可观测 pose transition”门禁，而不是只检查 prefix 长度 | 否 |
| P1 | pure rotation 文案写成“行走” | 153 条 | 使用 subtype 独立模板“原地转动/改变朝向” | 否 |
| P1 | 目标只短暂、勉强可见仍通过 | 至少 1 条 scoreboard 仅一帧 1,178 像素 | 至少连续两帧清晰可见，优先要求 frame 0/1 可见 | 个别补采 |
| P1 | 相同 pose 的 RGB 仍变化 | pure rotation/multi-turn 开场最明显 | 渲染 warm-up、冻结非任务动态、增加 stationary consistency 审计 | 新数据需要 |
| P1 | same-prefix 问题没有新图片 | 全数据 343 turn | 明确 runner 必须保留 multimodal history；stateless 模式重复提供当前 prefix 图片 | 否 |
| P1 | 后续问题可见此前 assistant gold | 多问题对话普遍存在 | 增加不含历史 gold 的独立评测模式，或分开 conversation | 否 |
| P2 | 普通 non-occluded subtype 没有 `front` | 当前 self-motion / multi-turn 都没有 | 不强行补 front；合理支持集为 left/right/back。front 需要独立遮挡或远距离机制 | 设计决定 |
| P2 | 当前场景和类别偏斜 | 主要来自 Beechwood；pure translation 仅两种目标 | shard 合并后按 scene/category/binding 统计并补齐 | 视合并结果 |
| P2 | 转向步长几乎固定为 32 度 | 所有转向 motif | 在 trackable 范围内采样不同速度和节奏 | 新增数据 |
| P2 | 中文问题混入 ontology token | 如 `multi_station_furniture_sink` | 增加中文显示名映射，内部保留稳定 category/entity ID | 否 |

## 6. 轨迹问题的具体原因和修复

### 6.1 Multi-turn 退化

`_direct_multi_turn_route()` 在 96 次搜索失败后返回：

```python
fallback = ...
return fallback, fallback
```

此时所谓直行帧都位于同一个 `(x, y)`。如果中间 head sweep 和最终 turn 恰好构成两个分离转向段，当前 checker 仍会接受。

修复必须有两层：

1. proposal 层：找不到路径时返回 `None` 或抛出 `MotifUnavailable`；
2. checker 层：multi-turn 增加起终点至少 2m 或 path length 至少 2m 的声明式 clause。

只修 motif 不够，因为 checker 应当独立表达 capability 语义；只修 checker 也不够，因为会持续制造无用候选。

### 6.2 通用 self-motion 的短路径和 fallback

`_propose_route()` 同样存在 `[start_xy, start_xy]` fallback。当前 110 条 accepted 没有完全静止轨迹，但已有：

- 4 条总路径不足 1m；
- 9 条起终点不足 2m；
- 最短路径约 0.304m。

建议通用 self-motion 使用较温和的最小路径阈值，例如 1m；multi-turn 使用其原始设计范围的 2m。阈值属于 capability contract，修改后应更新 standard/spec lineage。

### 6.3 Pure rotation 全为 back

当前 `stand_and_turn` 固定使用约 ±150° 的目标最终方位。±150° 都属于 back sector。

建议采样：

```text
left  : +90° 到 +110°
right : -90° 到 -110°
back  : ±150°
```

普通 pure rotation 要求目标已经离开视野，因此不应强行制造 front。最终按每 binding 10 条配置：

```text
left=3, right=3, back=4
```

### 6.4 Pure translation 全为 back

当前路径在目标侧面经过，并在越过目标 2.2–3.0m 后结束，最终目标自然位于约 ±150°。

建议增加不对称 endpoint：

```text
side endpoint:
  起点在目标前方约 3m
  横向偏移约 1.2–1.6m
  越过目标约 0.5–1.0m 后结束
  -> left 或 right

back endpoint:
  越过目标 2.2–3.0m 后结束
  -> back
```

仍保持直线和平移朝向恒定，但不再让所有终点落入 back sector。

### 6.5 Occluded 需要重新定义事件时序

当前 occluded 在最终 `t_q` 要求目标仍被遮挡，同时相机最终重新正对目标。一个能在图像内完成权威遮挡归因的目标必然接近 front sector，因而答案会塌缩为 front。

正确事件顺序应为：

```text
看见目标
  -> t_occ: 目标在图像内被真实物体遮挡，并完成 instance/depth 归因
  -> 遮挡后继续发生可追踪的平移或转向
  -> t_q: 在更晚的位置和朝向询问目标方向
```

对应 clause 应改为类似：

```text
occluded_in_view(target, t_occ)
invisible_in_range(target, t_occ:t_q)
post_occlusion_motion(t_occ:t_q)
answer(target_sector at t_q)
```

不能继续使用“`occluded_at_question(t_q)` + 最终重新面向目标”的组合。

## 7. Streaming 调度问题

### 7.1 `seen_labels` 会漏掉答案返回

当前逻辑记录 capability 历史上出现过的所有 label，导致：

```text
right -> left -> right
```

最后一个 `right` 被去掉。正确规则是只和上一个已发出的 label 比较：

```text
A -> A       只保留第一个
A -> B -> A  三个都保留
```

逐 prefix 权威重编译显示：

- 50 个 family 受影响；
- 共漏 53 个 turn；
- 通用 self-motion source group 的附着问题漏 16 个；
- multi-turn source group 的附着问题漏 21 个。

当前 self-motion target-direction source family 自身尚未出现 label 回归遗漏，主要受影响的是附着的 path integration、magnitude 和 homing 问题。

### 7.2 必须增加 canonical endpoint event

当前策略只在首次出现某个 label 或 label 改变时提问，因此答案一旦提前稳定，完整 episode 的尾帧不会再被释放。

当前 source capability 缺少完整末帧问题的数量：

| Capability | 缺少 full `t_q` 问题 |
|---|---:|
| `self_motion_update` | 73 / 110 |
| `pure_rotation` | 153 / 153 |
| `pure_translation` | 13 / 20 |
| `multi_turn` | 81 / 121 |
| 合计 | 320 / 404 |

pure rotation 的完整 delay 为 4–12 帧，但当前 streaming 153 条全部在最早允许的 delay=3 提问，设计的长记忆难度完全丢失。

建议 P1 event 由三类组成：

```text
合法 abstain event
  + 连续 label transition event
  + source capability canonical endpoint event
```

即使 endpoint label 和此前相同，也必须保留 source endpoint event。

### 7.3 运动问题不能只要求两张图片

当前仅排除 `prefix_length < 2`，但两张图片的 camera pose 可能完全相同。

建议为运动类 capability 增加 streaming-only evidence gate：

```text
observed_translation > epsilon
or observed_turn > epsilon
```

更严格时可以要求至少一个超过视觉可辨阈值的 transition，例如：

```text
translation >= 0.05m or yaw change >= 5°
```

这个门禁只决定“何时发问”，不改变 compiler gold。

## 8. Streaming “无法判断”监督方案

### 8.1 为什么当前为零

当前 P1 streaming 只序列化：

```python
certificate.status == "answerable"
```

所有 `abstain` 和 `invalid` prefix 都被跳过，因此 2,673 个 turn 全部是 answerable。

### 8.2 不能把所有早期 prefix 改成“无法判断”

compiler 已经区分三种状态：

| 状态 | 含义 | 是否可写入 QA |
|---|---|---|
| `answerable` | 当前证据足够且问题处于能力设计范围内 | 是，使用权威 label |
| `abstain` | 问题成立，但证据不足以确定答案 | 是，gold 为“无法判断” |
| `invalid` | 当前问题不在定义的有效范围内 | 否，必须跳过 |

self-motion 的典型 early prefix 往往是 invalid，而不是 abstain：

- 目标仍然可见：这是直接 perception，不是 memory update，属于 invalid；
- 目标刚消失但 `delay < 3`：难度不足，属于 invalid；
- 目标可见性处在 ambiguous 阈值：采集证据不干净，属于 invalid；
- `t_gone` 尚不可解析：问题的 memory phase 尚未形成，属于 invalid。

以下才是合法 abstain：

- 目标从未被观察到，`t_seen` 无法解析；
- 关键 sighting 帧被删除；
- 帧顺序或运动跳变使 ego-motion 不可追踪，并且 clause 声明 `on_violation="abstain"`。

因此绝对不能把所有 skipped/invalid prefix 统一标成“无法判断”，否则会制造错误 gold。

### 8.3 推荐主方案：独立 drop-key streaming

当前 family 已经有 `drop_key` intervention。它删除目标 sighting 关键帧，使 `t_seen` 无法解析，而 self-motion ScriptSpec 已声明：

```text
abstain_on_unresolvable = ("t_seen",)
```

这正是机器可验证的“证据不足”。不需要重新渲染，只需要按 drop-key 的 `frame_sequence` 重新编译和组织 streaming conversation。

建议每个 drop-key family 生成一个独立 stream：

```text
新 system message
  -> 只按 drop_key.frame_sequence 释放图片
  -> 在完整 drop-key 序列末尾提问
  -> compiler.status 必须为 abstain
  -> gold 必须为“无法判断”
```

不能把 canonical 和 drop-key 放进同一 conversation，因为 canonical 历史已经泄露了被删除的目标 sighting。两种 stream 必须历史隔离。

当前 raw family 中有 830 个 drop-key variant。逐个核验其已有 certificate 后，830/830 都满足：

```text
status = abstain
label = 无法判断
reason = frame_var_unresolvable:t_seen
```

精确分布为：

| Capability | 可直接转换的 drop-key abstain stream |
|---|---:|
| `self_motion_update` | 110 |
| `self_motion_update_pure_rotation` | 153 |
| `self_motion_update_pure_translation` | 20 |
| `self_motion_update_multi_turn` | 121 |
| `view_side_check` | 426 |
| 合计 | 830 |

其中 self-motion 四个已有 subtype 合计 404 个；`self_motion_update_occluded` 当前没有 accepted family，因此也是 0。

如果每个生成一个 abstain streaming turn，相对当前 2,673 个 answerable turn，abstain 比例约为：

```text
830 / (2673 + 830) ~= 23.7%
```

这是一个可用的初始比例。最终应按 capability、scene、target category 和序列长度分层，而不是直接全部堆入训练集。

### 8.4 辅助方案：untrackable / permute streaming

可以从 permute intervention 中选择由 `trackable` clause 明确判为 abstain 的序列：

```text
帧存在，但顺序或运动跳变破坏 ego-motion tracking
  -> certificate.status == abstain
  -> gold = 无法判断
```

它训练的是“看见过目标，但无法可靠累计自身运动”，与 drop-key 的“根本没有看见目标”形成互补。

只允许 compiler 确认的 abstain，不能根据 intervention 名称手工指定答案。

### 8.5 实现要求

当前 `_compile_prefixes()` 默认编译原始 `range(prefix_length)`，`_stream_record()` 也假设 source frame 连续递增。variant stream 不能沿用这个假设。

需要增加类似接口：

```python
compile_sequence_prefixes(
    view,
    family,
    binding,
    frame_sequence=drop_key_episode.frame_sequence,
)
```

每个 prefix 应编译：

```python
frame_sequence[:prefix_length]
```

媒体释放必须使用 certificate/variant 中的真实 `source_frame`，不能使用：

```python
range(released, prefix_length)
```

建议将新记录升级为 streaming schema v2，并显式记录：

- `variant=canonical/drop_key/permute`；
- `stream_kind=trajectory_multi_question/evidence_removed/untrackable_motion`；
- 原始 source frame sequence；
- abstain reason；
- canonical 配对 family ID。

### 8.6 Abstain 采样和平衡

建议初始训练比例：

```text
answerable turn : abstain turn ~= 3:1 到 4:1
```

同时满足：

- 每个主要 capability 都有 abstain；
- 每个 scene/category 的 abstain 比例接近；
- abstain 和 answerable 的图片数量分布尽量匹配；
- 不让模型通过“序列很短”“总在第一轮提问”等表面特征猜 abstain；
- drop-key 和 canonical 使用相同问题文本和答案选项；
- 训练集既包含 teacher-forced 多轮流，也包含不暴露历史 gold 的独立流。

### 8.7 Abstain 发布门禁

每个 abstain turn 必须通过：

1. `certificate.status == "abstain"`；
2. 输出 label 等于模板声明的 `abstain_option`；
3. `certificate.reason` 属于允许的 evidence failure，例如：
   - `frame_var_unresolvable:t_seen`；
   - `clause:seen_early`；
   - `clause:trackable`；
4. 不允许任何 `invalid` certificate 进入 streaming；
5. drop-key history 中不得出现被删除的 sighting frame；
6. canonical/drop-key conversation 历史必须隔离；
7. 图片顺序、SHA256、source frame 和 prefix 必须一致；
8. reviewer 中必须能同时查看 canonical 配对和 abstain intervention。

## 9. History 输入约束

当前同一 prefix 上可能连续问多个 capability，后一个 turn 没有 `new_images`。这不是文件缺失，而是假设模型能看到此前 conversation 中的图片。

需要明确支持两种导出：

### Stateful conversation

- 每轮只发送新增图片；
- 模型保留完整 multimodal history；
- 适合 agent 和完整多轮 VLM training。

### Stateless model call

- 每个问题重新提供当前全部 prefix 图片，或提供显式视觉缓存；
- 不依赖此前 turn 的图片；
- 适合单样本 batch inference 和独立 benchmark。

评测还应提供不包含此前 assistant gold 的模式，防止后续 capability 借用先前标准答案形成捷径。

## 10. 等 shard 合并后的处理流程

必须按以下顺序执行：

```text
远端 shard 全部完成
  -> finalize 并上传
  -> 本机同步全部结果
  -> 使用原始 master manifest/status 完成所有 merge
  -> 冻结合并后的 v1 和 merge ledger
  -> 运行全量只读 audit
  -> 生成 quarantine 和 credit 撤销清单
  -> 应用新 standard / ScriptSpec / motif / QA policy
  -> 仅对 coverage 缺口 backfill
  -> 重新权威编译 group/certificate
  -> 构建 coverage v2 和 QA v2
  -> 运行最终发布门禁
```

在所有 shard 合并前不要：

- 修改 master `coverage.plan.json`；
- 修改 master `coverage.status.json`；
- 删除或移动 master bundle/group；
- 更新正在运行的远端工作副本；
- 用当前 QA 作为正式训练 release。

可以提前开发和测试修复代码，但不能将新旧规则混入正在运行的同一批 shard。

## 11. 合并后的 credit 修复

撤销 episode 时不能只删除 source cell，因为一个 accepted episode 可能同时给 path integration、magnitude、homing、view-side 等 capability 提供 credit。

必须同步重算：

- `candidate_statuses`；
- `accepted_episode_ids`；
- 顶层 `episodes`；
- `credited_cells`；
- cell 的 complete/planned/exhausted 状态；
- per-label quota；
- per-scene/category/binding coverage。

无效 bundle 建议移入 quarantine 并保留审计证据，不直接删除。

## 12. 数据版本建议

不要覆盖合并后的原始结果。建议保留：

```text
behavior51_coverage_v1
  原始 shard 合并结果和审计基线

behavior51_coverage_v2
  清理、重新计 credit、label-aware backfill 后的数据

behavior51_coverage_v2_qa
  新 raw/streaming QA、oracle、viewer 和 policy
```

健康 v1 bundle 可以被 v2 引用或使用硬链接复用；不需要复制或重新渲染全部 22GB 数据。

## 13. 最终 release gate

正式训练前至少要求：

- 所有 accepted source trajectory 满足 subtype 的显式运动约束；
- multi-turn 不存在零位移和退化直线段；
- pure rotation / translation 的 left/right/back 达到声明配额；
- occluded 在中间帧完成真实遮挡归因，并在遮挡后继续运动；
- target 至少连续两帧有足够像素证据；
- canonical endpoint source 问题覆盖率为 100%；
- streaming 完整释放 episode 或显式标注为 event-only stream；
- `A -> B -> A` 不丢 turn；
- 运动问题前至少发生一个真实运动 transition；
- streaming 同时包含 answerable 和 compiler-authorized abstain；
- invalid turn 数量为 0；
- abstain 不依靠长度、轮次或模板表面特征即可被猜中；
- stateful/stateless 输入契约明确；
- 所有媒体存在并通过 SHA256；
- certificate、oracle、QA answer 完全一致；
- scene、category、binding、label 和 difficulty 分布报告随 release 一起生成。

## 14. 2026-08-16 shard-1 training preview 审计

本节审计以下不可变预览：

```text
behavior51_s1_preview_v0.1.0__n2388__20260815T145222Z__c2abd4d5a65b
```

它来自 `shard-001-of-002` 的运行中快照，不是 canonical shard result。以下“合并”数字仅用于只读分析，不表示已经写入 master。

### 14.1 同步、完整性与合并边界

同步和完整性结果：

- OSS `READY.json` SHA256 为 `adf23f94d2e449141ee5a53cdc92d13d5213f100c49fbe06bfbc7a4963e048df`；
- 37/37 个 archive 的本地尺寸和 SHA256 正确，总计 79,764,337,549 bytes；
- 18/18 个 full-reproducibility bundle archive 通过 zstd 和 tar 路径安全检查；
- 2,388 个 bundle 已解出，`source_snapshot` 54,006/54,006 文件通过；
- 48,373 条 raw、4,410 条 streaming 的 oracle、assistant 文本、reward target 和 certificate hash 引用无不一致；
- streaming 没有未来帧、prefix 非单调、媒体 ordinal 断裂或 invalid turn；
- 1,805 条 self-motion 的 25,453 个计划 pose 与 `plan.views.json` 完全一致：平移最大误差为 0，考虑相机轴约定后的 yaw 最大误差为 `1.14e-13` 度。

训练档单独下载时有一个 packaging 问题：`source_snapshot.jsonl` 声明了 4,776 个 `source/bundles/*/{scene_snapshot,trajectory_plan}.json`，但 `train_runtime` profile 不包含它们，导致官方 verifier 报缺文件。下载 full-reproducibility profile 后校验通过。后续发布应把这 4,776 个小 JSON 放入 training/source-metadata 档，或给 training profile 生成独立 snapshot。

当前预览不能传给 `coverage_render_shards.py merge`：

- `READY.json` 明确记录 `merge_eligible=false`；
- 没有 `shard.complete.json`；
- 原 shard 分配了 25 个 scene、4,290 个 cell，预览只选择了 17 个 scene 的 accepted episode；
- `hall_conference_large`、`office_cubicles_right`、6 个 restaurant/school scene 尚未进入该快照；
- 预览没有保存全部 candidate/reject/pending/cell 状态，写回 master 会破坏 coverage 和 credit 账本。

兼容性方面是好的：预览 provenance 中的 base manifest/status SHA256 与本机 master 逐字节一致：

```text
coverage.plan.json   8994b8e8723cd90cee08a1a8003f7658518be85e92a85d3645a14c004d45fa15
coverage.status.json 55f4e39d4935a998e46eef3663e2315f1374fc8ddc83c835e1e88709d5d016c0
```

本机 520 条与该预览 2,388 条 episode ID、scene 均无交集，只读并集为 2,908 条、19 个 scene。真正合并必须等待远端完整结束，执行 `finalize` 并上传带 `shard.complete.json` 的 canonical result 后，再使用 shard merge 工具。

### 14.2 Self-motion 数据量与只读并集

| Source capability | 本机合并前 | shard-1 preview | 只读并集 |
|---|---:|---:|---:|
| `self_motion_update` | 110 | 386 | 496 |
| `self_motion_update_pure_rotation` | 153 | 696 | 849 |
| `self_motion_update_pure_translation` | 20 | 62 | 82 |
| `self_motion_update_multi_turn` | 121 | 527 | 648 |
| `self_motion_update_occluded` | 0 | 134 | 134 |
| 合计 | 404 | 1,805 | 2,209 |

预览覆盖 17 个 scene、63 个 target category、121 个 scene-target binding；单 binding 有 1–40 条轨迹。scene 仍明显偏斜，`Beechwood_0_int` 和 `Merom_0_garden` 合计占 711/1,805。pure translation 只有 7 个 target category，occluded 只有 18 个。

### 14.3 轨迹与标签审计

| Capability | Preview 结果 | 判定 |
|---|---|---|
| 通用 self-motion | 386 条中 14 条 path `<1m`，37 条 endpoint displacement `<2m`；标签 `back=190,left=95,right=101` | 短路径问题复现；标签支持集基本健康 |
| multi-turn | 78/527 全程零位移，且这 78 条 path 和 endpoint 都 `<1m/<2m`；标签 `back=193,left=166,right=168` | fallback 退化复现，P0 |
| pure rotation | 位移全为 0，累计旋转约 153–158 度；`back=696/696` | 运动本身符合 pure rotation，但标签完全塌缩，P0 |
| pure translation | 路径 4.45–6.02m，累计转角为 0；`back=62/62` | 运动本身健康，但标签完全塌缩，P0 |
| occluded | 134 条路径 1.00–11.01m；`front=134/134` | 真实遮挡归因成功，但现有 `occluded_at_question(t_q)` 定义造成标签完全塌缩，P0 |

multi-turn 的 78 条零位移不是渲染或导出误差：record pose、view pose 和实际媒体属于同一轨迹。与本机基线合计后，multi-turn 退化为 96/648；通用短路径 `<1m` 为 18/496，endpoint `<2m` 为 46/496。

occluded 的权威检查本身通过：134/134 使用 `render_instance_depth`，都有具体 `occluder_entity_id`、runtime instance ID 和正 depth margin，且 clause 均在 `t_q` 检查。遮挡物为 walls 63、pillar 44、fridge 12、bathtub 10、openable_window 4、floors 1。问题在事件定义，不在遮挡物归因实现。

目标可见性仍需加强：18/1,805 条 canonical 只有一帧达到 visible 阈值，其中通用 10、multi-turn 4、pure rotation 3、occluded 1；最低 peak 仅 903 像素，刚过 `render_min_visible_pixels=900`。正式门禁应要求至少连续两帧清晰可见。

### 14.4 Streaming 审计

预览整体 streaming 为：

```text
4,410 record
15,288 turn
13,028 answerable
2,260 abstain
```

因此“整个 streaming 完全没有无法判断”在这个新预览上已不再成立，但 2,260 个 abstain 全部来自 P2 reference-frame 的 `evidence_reveal`：每个 family 一轮 abstain、一轮 answerable。P1 的 10,768 个 turn 仍全部 answerable。

对 1,805 条 self-motion source：

- 1,805 个 P1 record、9,362 个 turn，全部 answerable；
- self-motion capability 自身仍没有任何 streaming abstain；
- raw 中已经有 1,805 个 `drop_key -> frame_var_unresolvable:t_seen` 和 1,805 个 `permute -> clause:trackable`，全部为 compiler-authorized `无法判断`，但 streaming policy 只导出 canonical，未使用它们；
- 726/9,362 个 turn 没有 `new_images`，只能依赖 stateful multimodal history；全 P1 为 1,068/10,768；
- 所有 subtype 仍共享“行走”文案，pure rotation 也写成行走；self-motion 问题中的对象名仍直接暴露英文 ontology token。

逐 prefix 使用 full bundle 权威重编译 7,424 个 P1 family，当前 9,362 个 packaged turn 与现有 `seen_labels` 实现完全一致，mismatch 为 0。这说明漏 turn 是确定性的调度器行为，不是打包漂移：

| 受影响问题 | 受影响 family | 被漏 turn |
|---|---:|---:|
| `homing_probe` | 13 | 14 |
| `path_integration` | 16 | 16 |
| `path_integration_magnitude` | 120 | 124 |
| 合计 | 149 | 154 |

按 source capability 分布：通用 self-motion 61 个 family 受影响；multi-turn 88 个 family 受影响。把全局历史去重改为只去连续重复后，self-motion-source P1 turn 会从 9,362 增至 9,516，并保留 `A -> B -> A`。

endpoint 和尾帧问题更严重：

| Source capability | 缺 source full-endpoint 问题 | 未释放完整轨迹 |
|---|---:|---:|
| `self_motion_update` | 263/386 | 240/386 |
| multi-turn | 357/527 | 350/527 |
| pure rotation | 696/696 | 696/696 |
| pure translation | 41/62 | 41/62 |
| occluded | 134/134 | 134/134 |
| 合计 | 1,491/1,805 | 1,461/1,805 |

`seen_labels` 修复不能替代 endpoint event；source capability 必须无条件在完整 `t_q` 再问一次，并释放剩余图片。

“两张图即可问运动”也在更大规模上复现：

- pure rotation source 上 696 个 `path_integration_magnitude` 首问发生在 prefix=2，而前两帧 pose 完全相同；
- multi-turn source 上另有 515 个同类首问发生在两张相同 pose 后；
- 合计 1,211 个开场问题在任何可观测平移或转向发生前被发出。

相同 pose 的 RGB 也不稳定：上述 pure rotation 696 对和 multi-turn 527 对，共 1,223 对图像没有一对逐像素相同。RGB mean absolute difference 的中位数分别为 2.04 和 1.93（0–255），改变像素比例中位数分别约 79.9% 和 78.4%。因此 streaming gate 必须基于 pose transition，同时新渲染需处理 simulator warm-up/动态冻结，不能把 RGB 噪声当成 motion evidence。

### 14.5 本次 release gate 结论

该预览可用于开发期可视化、错误分析和 QA pipeline 调试，但不能作为正式 self-motion 训练 release，也不能并入 canonical coverage。需要等待 shard 完整合并后统一执行第 10 节流程，并至少完成：

1. 撤销 multi-turn 零位移、通用短路径和弱可见性 episode 的全部衍生 credit；
2. 修复 proposal fallback 和声明式最小位移 clause，仅定向 backfill 缺口；
3. 对 pure rotation/translation 做 label-aware 补采，重构 occluded 的中间遮挡事件；
4. 修复连续 label transition、强制 source endpoint、加入 pose-motion streaming gate；
5. 从现有 drop-key/permute family 构造历史隔离的 self-motion abstain stream；
6. 重新生成 QA v2，并重新运行本节所有门禁。

## 15. 2026-08-16 本机与 shard-1 preview 的全 Tier 联合审计

本节把审计范围从 self-motion 专项扩展到当前可用的全部 P1、P2 和 P3 数据。联合审计使用两个只读 source root，未把远端 preview 写入本机 master，也未把两份 QA JSONL 直接拼成训练 release。

### 15.1 当前到底有哪些 P1、P2、P3 episode

| 数据源 | P1 source episode | P2 source episode | P3 source episode | 合计 |
|---|---:|---:|---:|---:|
| 本机合并前数据 | 520 | 0 | 0 | 520 |
| shard-1 preview | 2,162 | 226 | 0 | 2,388 |
| 只读并集 | 2,682 | 226 | 0 | 2,908 |

远端 preview 确实有 P2，但没有 P3。P2 的 226 条 source trajectory 会各自绑定 10 个 reference-frame capability，因此不要把 2,260 个 P2 family/streaming record 误认为 2,260 条独立轨迹。

| Tier | Family | Raw QA | Streaming record | Streaming turn | Streaming status |
|---|---:|---:|---:|---:|---|
| P1 | 10,425 | 46,039 | 2,668 | 13,441 | 13,441 answerable |
| P2 | 2,260 | 11,300 | 2,260 | 4,520 | 2,260 abstain + 2,260 answerable |
| P3 | 0 | 0 | 0 | 0 | 无数据，不能判为通过 |
| 合计 | 12,685 | 57,339 | 4,928 | 17,961 | 15,701 answerable + 2,260 abstain |

联合读取是安全的：两批 source episode、scene、raw record ID 和 streaming record ID 均无交集。它们的 source snapshot 分别为 18,317 和 54,006 个文件，重新哈希均为 `pass`，无 changed file；两批合计 72,323 个 source snapshot 条目。远端 preview 仍有 `merge_eligible=false`、缺少 `shard.complete.json` 的边界，所以这里只能做双根只读审计，不能冒充 canonical coverage merge。

### 15.2 P1 联合结论

P1 的 oracle、certificate hash、assistant answer、reward target、媒体顺序、prefix 单调性和未来帧隔离没有发现新不一致。14 个 source episode 因 `fewer_than_two_valid_turns` 没有 streaming record；其余 2,668 条形成 13,441 个 turn。

P1 raw 中已经有可合法训练的 abstain intervention：

- `permute -> clause:trackable`：8,295 条 abstain；
- `drop_key -> frame_var_unresolvable:t_seen`：4,339 条 abstain；
- canonical、delay 和 drop-filler 各 10,425 条全部 answerable；
- 另有 2,130 条 `view_side_check` permute 经重编译后仍为 answerable 且 label 不变，不是 certificate 错误。

但当前 P1 streaming policy 仍只选择 answerable prefix，所以 13,441/13,441 个 P1 turn 都可回答。raw 有 abstain 并不能替代 streaming abstain 监督。

其中 2,209 条 self-motion source episode 的联合结果为：

| 问题 | 联合影响 |
|---|---:|
| 通用 self-motion path `<1m` | 18/496 |
| 通用 self-motion endpoint displacement `<2m` | 46/496 |
| multi-turn 全程零位移 | 96/648 |
| pure rotation 标签为 `back` | 849/849 |
| pure translation 标签为 `back` | 82/82 |
| occluded 标签为 `front` | 134/134 |
| 只有一帧达到 visible 阈值 | 19/2,209 |
| 缺 source full-endpoint 问题 | 1,811/2,209 |
| streaming 未释放完整轨迹 | 1,779/2,209 |

逐 prefix 重编译 self-motion-source 上附着的 P1 family 后，联合共有 199 个 family 因全局 `seen_labels` 去重漏掉 207 个合法的 `A -> B -> A` turn。仅改为连续去重后，这部分 turn 会从 11,531 增至 11,738；它不包含另外需要新增的 endpoint event。

相同 pose 后过早提问的问题也被放大：pure rotation 和 multi-turn 合计有 1,482 个 `path_integration_magnitude` 开场问题发生在任何可观测 pose transition 之前。对这两类 episode 的 1,497 对首帧 RGB 做联合检查，没有一对逐像素相同；RGB mean absolute difference 中位数为 2.153/255，发生变化的颜色通道比例中位数为 80.619%。因此必须使用 pose transition gate，不能把 simulator 的像素抖动当成自运动证据。

P1 结论仍为：gold 完整性基本通过，但运动语义、标签覆盖、endpoint 调度和 abstain 训练构造未达到 release gate。

### 15.3 P2 source trajectory 与证据门禁

远端 226 条 P2 source episode 的 source capability 分布为：

```text
reference_frame_transform         204
reference_frame_transform_yaw45    15
reference_frame_transform_yaw90     5
reference_frame_transform_yaw135    2
```

这只是产生轨迹时的 source capability。每条成功轨迹都通过 checker 同时附着以下 10 个 family：5 个 `reference_frame_transform{,_yaw45,_yaw90,_yaw135,_yaw180}` 和对应的 5 个 `reference_frame_visibility*`，所以最终每种 capability 都有 226 个 family。

轨迹几何检查全部通过：

- 每条 14–18 帧；
- `(x, y)` 全程不变，path length 和 endpoint displacement 均为 0；
- 累计旋转为 360°，最终 yaw 回到初始 yaw；
- 单步旋转为 21.176–27.692°；
- 这是同一站位的完整环视，不是 P1 中的退化零位移轨迹。

2,260 个 canonical P2 certificate 也都满足：

- `viewpoint / facing / target` 三个 entity ID 互不相同；
- 三个 referent 均至少有两帧权威可见证据；
- `never_all_covisible` 为真，没有单帧同时看见三个 referent 的捷径；
- imagined anchor distance 为 1.233–4.770m，大于 1m 门槛；
- transform 的最小 sector margin 为 15.4–22.3°，均通过 15° 门槛；
- 所有 answerable certificate clause 均 holds，oracle/hash/assistant/reward 无不一致。

P2 的五种 raw family intervention 在当前 compiler 下机械一致，但 `permute` 的视觉充分性还不能判为通过：

| Variant | 数量 | Certificate status | 解释 |
|---|---:|---|---|
| canonical | 2,260 | answerable | 完整证据 |
| delay | 2,260 | answerable | 延后提问不改变静态关系 |
| drop_filler | 2,260 | answerable | 删除非关键帧仍可回答 |
| drop_key | 2,260 | abstain | `clause:landmark_evidence` |
| permute | 2,260 | answerable | 当前 spec 声明 `same`；视觉充分性有疑问 |

这里的 `permute` 不是轻微换序，而是把完整 14–18 帧环视随机打乱。`stationary_survey` 和 `trackable_survey` 都是 `search_only` clause，variant 重编译时不再检查环视顺序；compile 阶段的 `landmark_evidence` 只要求三个 referent 各自可见两次。因此 compiler 能证明世界坐标中的答案不变，却没有证明一个只收到 RGB、没有相机 yaw 的模型仍能从打乱后的图像恢复各视角的相对方位。

### 15.4 P2 streaming 的正确部分和风险

2,260 个 P2 streaming record 均为严格的 evidence-reveal 对：

```text
turn 1: abstain, reason=clause:landmark_evidence
  + 恰好 1 张新 RGB
turn 2: answerable
```

两轮 prefix 长度差全部为 1；第一轮 prefix 为 3–17 帧，第二轮为 4–18 帧。226 条 source trajectory 每条恰好产生 10 个 record，并且同一 trajectory 的 10 个 record 使用相同的 abstain/answerable 切换点。因而物理轨迹和独立证据序列的有效数量是 226，不是 2,260。

2250/2260 个 P2 stream 在首次证据充分时停止，之后仍有 1–13 帧未释放，中位数为 5 帧。这对 `evidence_reveal` 语义是有意设计，不能照搬 P1 的“必须问到末帧”规则；如果还需要完整环视后的 endpoint 问题，应作为另一个 record 构造。

P2 当前存在四个发布风险：

1. **轮次捷径**：每个 record 永远是第一轮“无法判断”、第二轮可回答，而且第二轮永远只增加一张图。模型可能只根据轮次作答。需要加入未揭示关键证据的 continuation、不同长度的 abstain 序列、单轮独立评测，以及不暴露上一轮 assistant gold 的模式。
2. **十倍相关样本**：同一 trajectory 的 10 个 yaw/visibility family 高度相关。训练/验证/测试必须按 `cluster_ids.trajectory`，最好再按 scene/binding，整体分组，不能逐 record 随机切分。
3. **标签偏斜**：五种 transform 合计为 `front=306, back=307, left=111, right=406`；五种 visibility 合计为 `visible=306, not_visible=824`。尤其 left/right 和 visible/not-visible 不平衡。应按 capability × yaw × label 做 quota 或补充正负 yaw/base-sector 采样。
4. **Permute 的 modality/compiler 缺口**：随机打乱后，几何 oracle 不变不代表 RGB 证据仍充分。若任务要求按时间顺序恢复环视，应把 P2 permute 改为 abstain，并增加 compile-phase survey-trackability clause；若任务有意把图片当无序集合，则必须在问题中明示、提供或可恢复视角关系，并增加 pairwise overlap/anchor connectivity 的权威检查。修复前不要把这 2,260 条 permute raw 当普通 answerable 训练样本。

另外，1,130 个 transform family 中有 645 个最小 margin `<18°`，125 个 `<16°`。它们都合法通过当前 15° 标准，但对视觉噪声较敏感；正式 release 可以保留为 hard split，或把主训练集门槛提高并单独标注 boundary difficulty。

P2 结论为：canonical evidence-reveal 的证据链和 gold 完整性通过；raw permute 的视觉充分性、数据独立性、轮次去捷径和标签覆盖尚未通过训练 release gate。

### 15.5 P3 为什么是 0，以及当前能否审计

当前两批已接受 episode 都没有 `cross_view_*` source capability，因此 P3 family、raw QA、streaming QA 全部为 0。这不是 QA exporter 把已有 P3 丢掉，而是当前可用 source episode 中根本没有 P3 accepted bundle。

完整 master plan 中确实规划了 4,972 个 P3 coverage cell，但初始只保存了 196 个 P3 geometry candidate：

| Shard | P3 cell | 初始 P3 candidate | 当前已同步 preview 的 P3 accepted episode |
|---|---:|---:|---:|
| shard-0 | 3,892 | 196 | 尚未同步完整结果 |
| shard-1 | 1,080 | 0 | 0 |

本节远端 preview 来自 shard-1，它的 candidate assignment 中本来就没有 P3 candidate。因此现在不能对 P3 的跨视图锚链、非共视约束、k1/k2/k3 组合、abstain reveal 或 label 分布给出“通过”结论；状态应写为 **blocked by absent accepted data**。

后续拿到 shard-0 canonical result 或 shard-1 remote backfill 形成的 P3 accepted bundle 后，必须单独检查：

- 六个 `cross_view_pair_relation/closer × k1/k2/k3` capability 是否都有 source 和 family 覆盖；
- anchor chain 是否逐段有证据、目标对是否保持非共视、是否存在单帧捷径；
- drop-key 是否形成 compiler-authorized abstain；
- streaming 是否在关键锚出现前后真正改变可回答性；
- k1/k2/k3、scene、binding 和 label 是否平衡；
- 同一 scene/trajectory/anchor-chain 派生数据是否保持在同一 split。

### 15.6 联合审计的最终判定

| Gate | P1 | P2 | P3 |
|---|---|---|---|
| source snapshot / 文件完整性 | 通过 | 通过 | 无数据 |
| certificate / oracle / answer 一致性 | 通过 | 通过 | 无数据 |
| streaming 时序结构 | 结构合法，但调度需修 | 结构合法，但有轮次捷径 | 无法审计 |
| capability 语义与标签覆盖 | 未通过 | 部分通过，分布需补 | 无法审计 |
| 可直接作为正式训练 release | 否 | 否 | 否 |

所以，本机和远端 episode 可以而且应该一起做 audit；当前已经完成的是“双 source root 的只读联合审计”。正式训练数据仍应等待 canonical shard merge 后统一重建 QA v2，而不是直接拼接现有 JSONL。重建时应复用健康媒体和 certificate，应用 P1 修复、P2 去捷径/平衡策略，并在 P3 accepted 数据到齐后补做 P3 审计。

## 16. 2026-08-17 canonical shard-001 合并与 v2 partial audit

本节不再使用 training preview，而是使用带 `shard.complete.json` 的 canonical result：

```text
behavior51_coverage_v1__8994b8e8723c__shard-001-of-002
```

### 16.1 合并完整性与 v1 边界

shard-001 已正式合入 `outputs/behavior51_coverage_v1`：

- shard 元数据声明 4,290 个 cell、7,324 个 manifest candidate、3,215 个 accepted episode；
- 远端动态 backfill 另在 status 中留下 37 个 manifest 漏登记 candidate，其中 28 accepted、9 rejected；它们的 plan/views/recipe 均完整；
- merger 现在会严格校验 cell、candidate ID、plan record、binding、seed 后恢复这些索引，合并后的 manifest/status 各有 14,544 个 candidate，集合完全一致；
- 合并后 canonical v1 为 3,710 个 accepted episode、26 个 scene；3,710/3,710 的 group 与成功 render report 均存在；
- merge ledger 仅登记 shard-001 的 25 个 scene；待合并 shard-000 的 25 个 scene 与其零重叠，base plan/status SHA256 一致；
- 当前 plan SHA256 为 `11452c6b137be399c74113f1d3ede4639dcb48b269db3c17551ab27b5c3fc81a`；
- 当前 status SHA256 为 `f46545976c31c2b4403830962c09f9831a67efe2cfb19806f4dd12279835aa8d`；
- base plan/status 已分别保存在 `coverage.plan.shard-base-8994b8e8723c.json` 与 `coverage.status.shard-base-55f4e39d4935.json`。

合并器还修复了 3,190 个 copied group 的 worker-local `bundle/plan_record/scene_ir` 路径。该修复只是路径本地化，不改变 pose、媒体、family、certificate 或 gold；原始 canonical shard 仍保存在独立 result root。`coverage.plan.json`、`coverage.status.json` 和 merge ledger 的哈希没有因此变化。

同步完整性审计发现并修复了一个本地截断文件：

```text
office_bike__self_motion_update_occluded__873828eb5c00__a022/
  views/view-006.sensors.npz
```

本地旧文件比 render report 少 30,220 bytes，这正好等于 OSS sync 总字节差。单对象从 OSS 强制重下后，大小为 3,015,722 bytes，SHA256 为 `7360c53a47b0135957b7746c4ea4127d37b2cd64988257ecae6918c10d0f206c`，与 render report 一致。随后 3,710 个 accepted bundle 的 52,369 个 NPZ 全部通过 ZIP 容器检查。

### 16.2 当前 canonical partial 的 Tier 规模

| Tier | Source episode | Family | 状态 |
|---|---:|---:|---|
| P1 | 3,406 | 13,419 | 已完成静态和 streaming prefix audit |
| P2 | 304 | 3,040 | 已完成静态、family 和 margin audit |
| P3 | 0 | 0 | 等待 shard-000，不能判为通过 |
| 合计 | 3,710 | 16,459 | partial v1，不是最终训练 release |

P3 仍为 4,972 个 cell、196 个初始 candidate、0 accepted episode。其中 shard-001 的 1,080 个 P3 cell 全部为 `awaiting_candidates`；shard-000 持有 3,892 个 P3 cell 和全部 196 个初始 candidate。

### 16.3 Self-motion 轨迹与标签

当前 self-motion source episode 共 2,933 条：

| Capability | Episode | 轨迹问题 | Label |
|---|---:|---|---|
| `self_motion_update` | 661 | path `<1m`：18；仅一帧可见：13 | back=294, left=174, right=193 |
| `self_motion_update_multi_turn` | 851 | path `<2m`：119，且 119 条全程零位移；仅一帧可见：4 | back=318, left=260, right=273 |
| `self_motion_update_pure_rotation` | 1,131 | 原地旋转符合旧定义；仅一帧可见：6 | back=1,131 |
| `self_motion_update_pure_translation` | 116 | 路径和直线运动健康 | back=116 |
| `self_motion_update_occluded` | 174 | 旧 terminal-occlusion 定义不适合目标方向训练；仅一帧可见：1 | front=174 |

按 v2 拟议门禁，需 quarantine 333 条 source episode：

- 通用短路径 18；
- multi-turn 路径不足 2m 119；
- terminal-occlusion 定义 174；
- self-motion 目标少于两帧清晰可见 24；
- 原因有重叠，因此去重后是 333，不是四项简单相加。

按 source capability 去重后为：通用 31、multi-turn 122、occluded 174、pure rotation 6。这里只生成 v2 quarantine 与全部衍生 credit 撤销提案；在 shard-000 合并前不改 v1 status，不立即重算最终 coverage 缺口。

### 16.4 Self-motion streaming 精确重编译

2,933 条 self-motion source 及其附着 P1 family 已用 render instance mask 逐 prefix 权威重编译，未发现 canonical full certificate/gold 漂移。旧 streaming policy 的精确影响为：

| Source capability | 缺 source full endpoint | 未释放完整尾帧 |
|---|---:|---:|
| 通用 self-motion | 464 / 661 | 433 / 661 |
| multi-turn | 584 / 851 | 576 / 851 |
| pure rotation | 1,131 / 1,131 | 1,131 / 1,131 |
| pure translation | 75 / 116 | 75 / 116 |
| occluded | 174 / 174 | 174 / 174 |
| 合计 | 2,428 / 2,933 | 2,389 / 2,933 |

全局 `seen_labels` 去重使 246 个 family 漏掉 254 个合法的 `A -> B -> A` turn：

```text
path_integration_magnitude  202 family / 209 turn
path_integration             23 family /  23 turn
homing_probe                  19 family /  20 turn
self_motion_update             2 family /   2 turn
```

另外，1,960 个 `path_integration_magnitude` 首问发生在两张完全相同 pose 后，其中 pure rotation 1,131、multi-turn 829。v2 必须同时应用：只去连续重复 label、强制 source endpoint、基于 pose transition 的运动证据门禁；三者不能互相替代。

P1 raw 已有可用于 v2 独立 abstain stream 的 compiler-authorized intervention：

- `drop_key -> frame_var_unresolvable:t_seen`：5,738；
- `permute -> clause:trackable`：10,614；
- 另有 2,805 个 P1 permute 仍 answerable，不能按 variant 名称强制改成“无法判断”。

### 16.5 P2 与 P3 判定

304 条 P2 source 各产生 10 个 family，共 3,040。canonical/delay/drop-filler/permute 各 3,040 个 answerable，drop-key 为 3,040 个 compiler-authorized abstain。五种 transform 共 1,520 个 family，其整条 imagined-yaw curve 的最小 margin 为 15.4°；645 个 `<18°`，125 个 `<16°`。

P2 不需要因本轮审计重渲染，但在 v2 中仍必须：

- 按 trajectory cluster 分 split，避免同轨迹十倍相关泄漏；
- 修复固定“第一轮 abstain、第二轮 answerable”的轮次捷径；
- 明确 permute 是时序任务还是无序集合任务，并增加对应视觉证据门禁；
- 平衡 transform 与 visibility label。

P3 状态保持 **blocked by absent accepted data**。收到 shard-000 后再执行第 15.5 节列出的跨视图专项审计。

### 16.6 v2 partial 产物与后续合并

本轮没有复制 161GB 媒体，也没有改写 v1 credit。可复现产物位于：

```text
outputs/behavior51_coverage_v2_partial_s1/audit/
  audit.summary.json
  episode_dispositions.jsonl   # 3,710
  quarantine.jsonl             # 333
  backfill_requirements.json
  bad_npz.jsonl                # 0
```

审计脚本为 `scripts/audit_behavior51_coverage_v2.py`。shard-000 到达后仍先增量合入 canonical v1；因为 shard scene 集合互斥，已审计的 shard-001 episode 不会被 shard-000 覆盖。随后对新增 P1/P2/P3 episode 补审，统一重算 v2 credit 和 backfill，最后才生成 QA v2。不要把当前 v1 QA JSONL 或 partial overlay 当作最终训练 release。

### 16.7 rejected candidate 根因审计

`rejected` 不能只按 status 中的外层字符串统计。当前 1,854 条 rejected 中，685 条原本只写成 `render_report_missing`，但 bundle 内的 `failure_report.json` 保存了更精确的失败原因；另有 13 条已经原子发布了完整成功 bundle，却在 OmniGibson shutdown 或外层 timeout 后被错误记为 rejected。本轮对全部 rejected 做了 artifact、failure report、primary certificate 和只读 primary replay 审计：

| 失败阶段 | 数量 | 含义 |
|---|---:|---|
| acquisition preflight | 684 | 649 条 P2 辅助相机碰撞；35 条真实 traversability clearance 不通过 |
| primary authority | 1,140 | 完整渲染后，render mask / motion / P2 evidence checker 正确拒绝 |
| family packaging | 14 | canonical 可回答，但 delay 或 permute family 构造失败 |
| post-render interrupted | 13 | 成功 bundle 已存在，外层状态没有继续完成 authority/group |
| render runtime | 3 | 1 条真 timeout、1 条 exit 139、1 条人工停止 |

authoritative 数据拒绝及只读 replay 的根因是：

| 根因 | 数量 | v2 动作 |
|---|---:|---|
| `t_seen` 无法解析 | 627 | 不重渲染原 plan；重新选择 target/视点并保证真实 sighting |
| `turned` 不足 | 315 | 在最后一次真实 sighting 之后预留额外转角，再 plan/render |
| `t_gone` 无法解析 | 102 | 重新设计真实 disappearance，occluded 尤其不能把几何遮挡当成像素消失 |
| landmark evidence 不足 | 64 | 重采 survey，使三个 landmark 各有至少两帧清晰证据 |
| 单帧共视捷径 | 21 | 重采 survey，禁止三个 landmark 在任意单帧同时可见 |
| disappearance gap 不足 | 17 | 在真实 `t_gone` 后保留至少三帧证据间隔并增加安全余量 |
| gone 区间含 ambiguous pixel | 3 | 重采到权威 invisible，而不是放宽阈值 |

最严重的系统性浪费是 649 条 P2 auxiliary-camera collision：640 条来自 `Ihlen_1_int` 的两个固定 binding，9 条来自 `grocery_store_asian` 的一个 binding。同一个不可站立的 imagined viewpoint 被反复渲染，说明这是 binding-level deterministic failure，不应该对同一 binding 重试。几何 OBB 原来只差约几毫米没有覆盖相机球体；v2 prefilter 现已按 backend 的 5 cm camera probe radius 扩张障碍 footprint，直接在 plan 前排除这种 binding。

35 条 traversability failure 说明 OBB/free-cell 近似仍不能完全替代 OmniGibson traversability map。它们不得原样重试；正式修复是把 source acquisition 的 authoritative traversability mask/clearance field 导入 planner，再重新规划路径。

309 条 status `clause:turned` 加上 6 条 post-render replay `clause:turned` 全部是实际累计转角 `<80°`，没有一条是 `>200°`。原因是 render truth 中 `last_visible(t_seen)` 通常比 geometry plan 更晚，前面已经发生的转动不再属于 `$t_seen:$t_q`。修复应增加 post-sighting turn reserve，而不是降低 80° 门槛。

13 条 completed-but-interrupted bundle 经只读 primary replay 后为：4 answerable、6 `clause:turned`、3 `t_seen` unresolvable。4 条 pure-rotation/back candidate 的完整 source family 也已只读重编验证通过，可以直接在 v2 重编 group，无需重渲染；其余 9 条虽然像素完整，仍是语义失败，必须重 plan。

14 条 family packaging failure 中，11 条是 multi-turn 的 delay 在连续转弯中间复制一帧，零转角 pause 把一个 turn segment 人为切成两个；delay 现在优先选择最近的直行/转弯边界 pose，同时 `turn_segments_between` 将 intervention 引入的连续相同 source frame 视为 neutral pause。11 条旧 bundle 已全部只读重编验证通过，不需要重渲染。其余 3 条是附着 family 的 intervention window 只有一个位置，无法构造非平凡 permute；需要在 v2 schema 中显式记录 variant unavailable，不能因此丢掉 canonical source episode。

本轮还修复了 renderer timeout 生命周期：外层现在终止整个 `conda run` process group，避免孤儿 OmniGibson 进程与 retry 同时发布相同 bundle；若 backend 已经原子发布成功 `render_report.json`，即使 shutdown 超时或非零退出也复用完整 bundle；`failure_report.json` 的真实错误会写回 status，确定性的辅助相机/路径 clearance 失败不再盲目重试。

可复现 rejected 产物为：

```text
outputs/behavior51_coverage_v2_partial_s1/audit/
  rejected_candidates.jsonl      # 1,854，逐 candidate 根因、证据、动作
  rejected_repair_queue.json     # 按修复动作分组的 candidate ID
```

这些清单仍是 v2 overlay，不改 canonical v1。收到 shard-000 后，对新增 rejected 执行同一审计，再统一生成最终 repair queue。

## 17. 2026-08-17 本机 v2 repair 启动

用户选择不再以 shard-000 到达作为 v2 修复的启动条件。等待 shard 的理由原本只是避免重复补采和提前计算不完整的 credit 缺口，并非 plan/render 的技术依赖。当前策略改为：冻结已合并的 canonical v1，在独立本机 overlay 中立即修复；以后若 shard-000 到达，只把它作为增量候选去重并重新计算 credit，不覆盖已经通过 v2 审计的结果。

当前 partial snapshot 的精确修复范围为：

- quarantine 导致 215 个 cell 新增 607 个缺额 slot；
- rejected 中有 301 个 cell 当前仍缺额；
- 两者去重后为 477 个 local repair cell；
- 11,071 个“当前未满 cell”包含尚未到达的 shard 和 deferred coverage，不能作为本轮补采规模。

不需要重渲染的 QA v2 已生成到：

```text
outputs/behavior51_coverage_v2_partial_s1/qa/
```

它排除了 333 个 quarantine source episode，保留 3,377 条健康 source trajectory，生成 69,263 条 raw QA 和 6,113 条 streaming record / 24,818 个 turn。数据级复检结果为：

- P1 record 3,073；
- source canonical endpoint 缺失 0；
- 无真实 pose transition 就提问的 turn 为 0；
- 新数据中实际保留了 278 个 capability stream 的 `A -> B -> A` 局部回归模式。

本机 plan/render overlay 位于：

```text
outputs/behavior51_coverage_v2_local_repair/
```

第一批选择 77 个 episode producer cell，为 202 个缺额 credit target 服务。它包括 15 个直接复用旧像素的 candidate、2 个同 plan runtime retry，以及使用独立 `__v2r1` candidate namespace 和新 repair seed 的新 plan。coverage runner 允许 producer 自己配额已满时继续为兼容的 derived capability 补 credit，从而保持 episode-first，而不是为每个 path-integration/homing cell 单独渲染一条轨迹。

另外 277 个 cell 保持 held，不会盲目消费旧 pending plan；它们需要先完成 intermediate-occlusion 时序、权威 traversability、binding replacement、landmark survey 或 post-sighting/disappearance reserve planner。held 只表示推迟到对应 planner 通过测试，不表示回到 shard 流程。

本机任务已经在 tmux `epispace-v2-local-repair` 中使用 GPU0 / 1 worker 启动。首条新 candidate：

```text
Beechwood_0_garden__self_motion_update__a07eee780273__v2r1__a067
```

已完成 OmniGibson 权威渲染并 accepted，一次同时补充 `self_motion_update`、`path_integration`、`path_integration_magnitude` 和 `view_side_check` 四个 cell，验证了本地 episode-first repair 闭环。

### 17.1 旧 terminal-occlusion 像素的派生 QA 与遮挡物筛选

旧 `self_motion_update_occluded` 的 174 条 episode 不再用于“目标现在在哪个方向”：它们在 terminal `t_q` 完成遮挡归因，方向标签全部塌缩为 `front`。但其 render instance/depth 遮挡证据仍可复用于两个独立的派生任务，无需重新 plan/render：

1. `occluder_identification`：目标被什么物体遮挡；gold 为权威 `occluder_category`；
2. `disappearance_cause`：目标是被遮挡、移出视野，还是证据不足；这批通过筛选的样本 gold 为“被遮挡”。

遮挡归因的几何真实性和 QA 的语义可回答性必须分开门禁。`floors`、`ceilings` 以及 `background`、`roof`、`lawn`、`driveway` 等非对象背景/承载面不得成为上述 QA 的有效 occluder；`unknown` 或无法映射到稳定显示名的类别也不得进入训练。命中排除类别时应记录 `disallowed_occluder_category`，跳过派生 QA；不能错误改标为“移出视野”，也不删除原始 episode 和 authority witness。`walls`、`pillar`、家具、设施和可辨认的结构件仍可作为遮挡物，但还须通过 instance ID、中心 patch support、depth margin、dominance 和视觉可辨认性门禁。

首次决定性消失 replay 进一步说明，不能用 terminal occluder 代替 first-event occluder。174 条旧 episode 都存在可归因的首次消失：161 条首次事件正好位于旧 `t_gone`，13 条在其后 1--2 帧内才获得足够 instance/depth support；其中 10 条的首次遮挡物和终点遮挡物不同。旧 terminal 统计里唯一的 `floors` 样本：

```text
office_bike__self_motion_update_occluded__92588223b8d6__a013
target=coffee_table_wzyqgx_0
occluder=floors_fazobp_0
```

其 first-event 实际由 `walls` 造成，因此不会因 terminal `floors` 被错误删除。真正不能发布的是只有一帧清晰 target sighting 的弱证据样本：

```text
Merom_0_garden__self_motion_update_occluded__0b6c4374aa2a__a006
```

所以最终仍是 173 条，但原因是至少两帧连续身份观察门禁，不是 terminal floor。173 条 first-event 遮挡物分布为：`walls=68`、`pillar=44`、`fridge=42`、`bathtub=9`、`hanging_plant=5`、`skeletal_frame=5`。用户确认 `skeletal_frame` 是有效遮挡物，稳定中文显示名使用“框架结构”；重复 episode 仍按同一 target-occluder binding 计算 diversity。

### 17.2 旧像素 causal-occlusion QA 已落地

不可变 v1 像素的派生 overlay 位于：

```text
outputs/behavior51_occlusion_qa_overlay_v1/
```

它包含 346 条轨迹、519 个 family：173 条 eligible first-event occlusion，以及从健康 self-motion 中按 scene/source 分层确定性选择的 173 条 `out_of_view`。媒体使用同文件系统 hard link，不复制或覆盖 v1；`selection.audit.jsonl` 保存每条 first-event authority witness，`legacy_direction.exclusions.jsonl` 对全部 174 条旧 `self_motion_update_occluded` 只撤销旧方向 capability，不撤销同一像素上新构造的 causal QA。

训练 QA 位于：

```text
outputs/behavior51_occlusion_qa_overlay_v1/qa/
```

结果为 1,557 条 raw QA 和 1,500 条 streaming QA。全部 streaming record 都是单问题、单 turn、history-isolated，并分别标记 `immediate` / `delayed`：

- `disappearance_cause`：`occluded=346`、`out_of_view=346`、`无法判断=346`，严格 1:1:1；
- `occluder_identification`：346 条可回答记录加 116 条 drop-key `无法判断`，abstain 比例 25.11%；
- 遮挡物问题始终是正确项 + 2 个由 family ID 决定的 distractor + `无法判断`，共四选一；
- source snapshot 共 12,252 个文件，完整 SHA-256 复检通过，变化文件为 0。

新的正确方向任务命名为 `self_motion_update_after_occlusion`。它以 first-event `t_event` 为边界，要求遮挡前至少两帧连续可见、`t_event:t_q` 始终没有可用 target sighting、遮挡后间隔至少三帧并累计 80--200 度可跟踪转动。新 `walk_through_occlusion` motif 保证目标不会先因距离消失再遇到 blocker，并在真实场景几何 binding 上验证可生成 front/left/right/back 四类 plan。

新 plan/render staging 为：

```text
outputs/behavior51_occlusion_repair_v1/
```

tmux `epispace-occlusion-repair` 先做 CPU geometry planning；规划结束后等待 `epispace-v2-local-repair` 完全退出，再使用 GPU0 渲染，因此不会影响当前审计修复任务。渲染通过后，同一 episode 会 episode-first 附着 `self_motion_update_after_occlusion`、`occluder_identification`、`disappearance_cause` 及其他通过 checker 的 probe。

首轮规划只枚举旧 rendered authority 已证明可发生遮挡的 23 个唯一 scene-binding（来自 174 条旧 episode，覆盖 8 个场景），避免重新扫描数百个理论上可绑定但实际从未产生遮挡的 target。manifest 共 69 个 cell：23 个新方向 producer、23 个 deferred 遮挡物问题、23 个 deferred 消失原因问题；后两类不自行生成或渲染轨迹。初始 1--30 次搜索保存 140 条 candidate，来自 14 个 producer binding；另 9 个 binding 当前为 0 candidate，GPU runner 后续会按同一 manifest 明确补搜至 150 次再标记 exhausted。

140 条 geometry plan 的质量复核为：frame ordering/gap 违规 0，post-event 转角 90--191.9°，最大单步平移 0.969m；方向分布 `front=39`、`left=34`、`back=38`、`right=29`，每类占 20.7%--27.9%，不再存在全 back 或 back 双倍采样偏置。当前 plan SHA-256 为：

```text
e2a88f4fe0728ec1d66e570acd83c87c43d7c2ed08a36c756a190057036fd9cc
```

### 17.3 Pure rotation / translation 标签平衡重采已排队

旧 partial canonical 中 `self_motion_update_pure_rotation` 的 1,131 条 source episode 全为 `back`，覆盖 176 个已有 accepted binding；`self_motion_update_pure_translation` 的 116 条 source episode 全为 `back`，覆盖 16 个 binding。普通 coverage runner 不能直接修复这一分布，因为这些 capability-binding cell 已经有 accepted credit，会在生成新的 side label 前被判为满额。

当前新增独立 label-aware overlay：

```text
outputs/behavior51_direction_balance_v1/
```

它把四个配额拆成互相独立的 manifest：

```text
pure_rotation_left
pure_rotation_right
pure_translation_left
pure_translation_right
```

每个 manifest 在 geometry 阶段只保留指定的 provisional label，OmniGibson 渲染后再要求 primary certificate 和 question group label 都与 `desired_answer_label` 一致；错误标签只能 rejected，不能填充配额。旧 `back` 像素不重渲染、不改标，最终 release 每 binding 最多选择约 4 条旧 `back`，并与新 `left=3/right=3` 组合。

首轮 1--30 次 geometry 规划结果为：

| 子任务 | binding/cell | candidate | 初始 0-candidate cell |
|---|---:|---:|---:|
| pure rotation / left | 176 | 435 | 12 |
| pure rotation / right | 176 | 418 | 20 |
| pure translation / left | 16 | 17 | 9 |
| pure translation / right | 16 | 24 | 7 |

全部 894 条初始 candidate 的 provisional label 与子任务严格一致。Rotation side plan 的总原地转角为 92.0--107.9 度、位移为 0；translation plan 的累计转角为 0。初始为 0 或不足 3 条的 cell 不会直接判空，runner 会按同一 deterministic seed 明确补搜至 150 次，再由真实 RGB/depth/instance 权威复检。

本次规划同时修正了 pure rotation 的系统性不可生成约束：旧 clause 在 `$t_seen:$t_q` 要求额外 80--200 度转动，而 side sector 在 18 度 margin 下最大方位角约为 117 度；目标通常到 45--60 度仍在相机中，因此旧定义会把可生成数据重新压回 `back`。新定义在 `0:$t_q` 检查完整实际原地旋转为 80--200 度，同时继续要求逐步运动可跟踪、目标随后持续不可见和 sector margin，通过这些独立门禁保持任务有效性。

tmux `epispace-direction-balance` 已启动。GPU 顺序固定为：

```text
epispace-v2-local-repair
  -> epispace-occlusion-repair
  -> epispace-direction-balance
```

所以当前只完成 CPU plan 并等待，不会与前两个任务争用 GPU0。四份 manifest SHA-256 分别为：

```text
pure_rotation_left        d50f6e2fca400fcbe12442ed92e4ac3a3d6a06b1fde82274db3c61cf7f9a91aa
pure_rotation_right       b2b246fe0ab96375a0848a18284883d71523381b906130b30b24f40ff3b73e03
pure_translation_left     65057ad42761c353866e43858cb2e30fb254f1cd629a44b81721065ec7523038
pure_translation_right    c209f828462f2a1b73e514fa968a85af732eb4ac00eef64d7a7d1555d64709e2
```

canonical v1 plan/status 哈希保持 `11452c6b...` / `f4654597...` 不变。`tests/scriptgen` 当前为 201 passed、27 skipped，`git diff --check` 通过。

### 17.4 Direction balance 四卡远端分片

Direction balance 的四份 geometry plan 已冻结为独立单 shard 任务，适合在一台四卡机器上让每个 manifest 内的 scene/cell 由 GPU `0 1 2 3`、4 worker 并行处理。不能把四个 manifest 简单地“一张卡一个”：rotation 各有 176 个 cell，而 translation 各只有 16 个 cell，会造成严重负载不均；远端 runner 因此依次用四卡执行 rotation-left、rotation-right、translation-left、translation-right。

本机等待中的 tmux `epispace-direction-balance` 已在 0 bundle、0 group、0 running status 时安全停止。`epispace-v2-local-repair` 和 `epispace-occlusion-repair` 未修改；前者继续本机渲染，后者仍在其后自动接管 GPU0。四份 direction manifest 已生成 idle `coverage.status.json` 作为不可变 shard merge base。

分布式闭环脚本为：

```text
scripts/upload_behavior51_direction_balance_to_oss.sh
scripts/launch_behavior51_direction_balance_remote.sh
scripts/merge_behavior51_direction_balance_from_oss.sh
scripts/direction_balance_shards.py
```

现有 scene-level sharder 已修复一个对本任务关键的边界：只要 cell 尚未满额，即使初始 candidate 为 0，其 scene 也必须进入 shard，远端才能在 `EPISPACE_REMOTE_BACKFILL=1` 下继续执行 31--150 次 geometry 搜索。四份 dry-run 分别完整包含 26、26、9、9 个 scene，初始 pending candidate 为 435、418、17、24；没有 live manifest process 或 running status。

上传内容不含代码，只记录 EpiSpace 和 OminiGibson 的精确 Git commit。上传 preflight 要求所有 tracked 修改已提交，并要求 direction runtime 新文件已经进入 Git；远端 checkout 不一致时 runner fail closed。每个远端 task 完成后独立 finalize 并上传，合并器校验 base/manifest/status/completion SHA-256，复制 pixels/question groups，重定位远端 backfill plan/recipe 路径，并为每个 task 写幂等的 `coverage.shard_merge.json`。四份合并完成后总索引写入：

```text
outputs/behavior51_direction_balance_v1/direction_balance.merge.json
```

完整操作命令见 `DIRECTION_BALANCE_REMOTE_RUNBOOK.md`。当前上传尚未执行：dry-run 正确停在 EpiSpace 工作树仍未 commit 的前置条件，没有写入 OSS。
