# 空间组合泛化 Episode 数据设计 v1

> 状态：研究契约与数据规范；适用于 OmniGibson/Habitat 采集、SenseNova-SI 混合训练与受控 A/B。
> 核心对象：`episode family`，不是孤立 QA，也不是不可执行的文本 CoT。
> 第一性原理：泛化来自**共享状态复用、typed I/O 约束、组合留出和可执行验证**，不是把多种题型放进同一数据池。

## 0. 执行摘要

研究假设：在相同基础模型、场景、视图、问题事实、图像曝光和训练预算下，`query-agnostic shared state + typed read + certificate` 相比 isolated-QA SFT，能提高训练未见 operation-graph signature 的准确率和 episode-family 一致性，同时不损伤普通空间 QA。

形式化数据单元：

\[
E=\{O_{1:M},\Delta S_{1:M},S_M,(q_k,\pi_k,y_k,c_k)_{k=1}^{K}\}
\]

- `O_1:M`：同场景有序 RGB 观察；主设置中这是唯一 model-visible 视觉通道。
- `ΔS_1:M`：每帧支持的状态增量，不含未观察 world truth。
- `S_M`：在问题出现前提交的 canonical belief。
- `q_k`：多能力问题；同一 state 被 K 个 read 复用。
- `π_k`：typed `G/F/B/M/R/P/V` DAG。
- `y_k`：类别、数值、路线或 `unknown/rejected`。
- `c_k`：独立几何执行产生的 certificate、margin、证据视图与失败原因。

组合泛化的必要判据：

\[
Atoms(test)\subseteq Atoms(train),\quad Signature(test)\notin Signature(train)
\]

如果测试只换语言模板、物体名或随机场景，而 graph signature 训练见过，则主要测分布迁移，不足以证明能力组合。

### 预注册胜利条件

- 主指标：scene-disjoint held-out graph-signature accuracy，至少 `+5pp`。
- 共同主指标：all-member family consistency，至少 `+10pp`。
- 守门指标：普通 IID 空间 QA 下降不超过 `1pp`。
- 强状态证据：answerable query 的 `state-only` 与 `full-context` 差距不超过 `3pp`，且 shuffled-state 明显下降。
- 统计单位：scene/episode family，而不是单条 QA；三训练种子，scene-cluster bootstrap 95% CI。
- 若计划样本后三种子组合提升不超过 `2pp`，停止扩大数据，先修改机制假设。

## 1. 目标能力不是数据桶，而是程序族

| SenseNova 能力表面 | 可执行定义 | 典型 signature | 模拟器真值 | 关键泛化轴 |
|---|---|---|---|---|
| Grounding / Localization | 像素区域绑定稳定实体和局部几何 | `G→V`、`G→F→B` | instance mask、entity UUID、depth | 新类别/遮挡/相似实例 |
| Perspective Taking | 构造相机、物体或假想观察者 frame | `B→F_query→R/P→V` | SE(3)、target view render | 训练只见 F 与 canonical R，测试其组合 |
| Spatial Relation | 在声明 frame 内判方向、包含、遮挡 | `G+G→F→B→R→V` | OBB、visible mask、margin | 不共同可见、ego/allo 改写 |
| Metric Measurement | 估中心/表面/高度/可达距离 | `G+G→B→M→V` | depth、mesh、navmesh | 新尺度、跨房间、视角变化 |
| Mental Reconstruction | 维护 visible/seen-not-current/unknown | `G→B→P/R→V` | observation prefix、visibility | 删除决定帧、延迟揭示、反事实 twin |
| Comprehensive Reasoning | 多分支、多跳、闭合或路线执行 | `(G/F/B/M/R/P)^+→V` | 可执行图、路径、直接几何 | graph 深度、分支数、未见拓扑 |

能力映射原则：

1. PT 是每个多视图 episode 的写入骨干，不再只是一个题型。
2. SR/MM/CR 是对共享 state 的 typed read，必须声明 query frame。
3. P 只在 target pose 可达且静态场景可重新渲染时成立；移动物体反事实要求 editable source。
4. V 不是语言自评，而是独立规则执行、证据完整性检查和 `pass/reject/unknown`。
5. 自然语言只做 program 的表面实现；几何正确性先于语言生成。

## 2. 三层状态与三类通道

### 2.1 严格状态分层

- `world_state`：完整场景真值；仅 oracle，包含隐藏物体和未观察关系。
- `observable_state_t`：截至 t 的图像/动作/标定证据能直接支持的事实。
- `belief_target_t`：模型应安全保留的最小状态；允许不确定、seen-not-current 和 unknown。

推荐实体 slot：

```json
{
  "entity_id": "stable-uuid",
  "category": "fridge",
  "position_world_bin": [61, 83, 15],
  "extent_bin": [12, 10, 18],
  "status": "visible|seen_not_current|unknown",
  "evidence_views": ["view-000", "view-002"],
  "last_seen_view": "view-002",
  "confidence_bin": 9
}
```

`ΔS_t` 至少支持：`add / update / reobserve / leave_current_view / reappear / mark_unknown`。`leave_current_view` 不能无证据地等价为真实遮挡；M1 没有 amodal mask，应使用保守状态名。

### 2.2 通道隔离

| 通道 | 内容 | 用途 |
|---|---|---|
| `model_visible` | RGB、问题、声明动作/视图顺序 | 模型实际输入 |
| `supervision` | depth、instance、pose、observable delta、node value | loss target，不作为主设置输入 |
| `oracle_only` | SceneIR、完整关系 oracle、隐藏状态、期望答案 | 生成与验证，永不进入模型上下文 |

同一字段不得跨通道。必须有自动 leakage test；训练 exporter 拒绝包含 `expected_relation`、完整 SceneIR 或 certificate oracle answer 的输入。

## 3. Query-agnostic write：防止“为题建图”

标准序列：

```text
RGB_1 + S_0 → ΔS_1 → S_1
RGB_2 + S_1 → ΔS_2 → S_2
...
RGB_M + S_{M-1} → ΔS_M → COMMIT(S_M)
COMMIT(S_M) + q_1 → π_1 → node values → y_1
...
COMMIT(S_M) + q_K → π_K → node values → y_K
```

- write phase 隐藏所有问题、候选项和答案。
- state schema 与 query 无关；不能只保留被问物体。
- 一个 state 至少服务 8–20 个跨能力 read，形成复用压力。
- state 提交后禁止回写图像证据；需要再次观察时必须显式新 step。
- state-only 测试直接使用提交态，检验它是否被真正执行。

## 4. Episode family：构造五条“模型性”性质

family siblings 共享 scene truth 和 semantic query set，并用同一 `family_id` 锁定 split。

| Variant | 构造 | 应保持/改变 | 对应性质 |
|---|---|---|---|
| `canonical` | 正常顺序、完整证据 | 参考状态与答案 | 基线 |
| `legal_set_shuffle` | 静态 set-view 合法重排 | 最终 state/答案保持；逐步轨迹任务不参与 | 路径无关 |
| `reverse_revisit` | 逆向观察并回访 anchor | identity/state 保持 | 持久性 |
| `loop_revisit` | 首尾同位姿 | depth/mask 闭合，RGB 达阈值 | 闭环 |
| `decisive_view_deleted` | 删除唯一决定证据 | accepted→unknown | 知所不知 |
| `delayed_reveal` | 决定帧延后 | prefix unknown，揭示后恢复 | 增量状态 |
| `novel_query_frame` | 图像/state 不变，替换 origin/facing | 答案按 frame 变换 | 视角等变 |
| `cross_view_binding` | 两实体从不共同可见 | 必须先注册到 shared state | 组合闭合 |
| `target_view_holdout` | 新可达相机位姿 | P 与真实 render 一致 | perspective |
| `counterfactual_twin` | 单物体移动/开关/遮挡 | 只改变因果相关答案 | 反捷径 |

严格规则：family siblings 不能跨 train/val/test；相同 RGB hash、scene、trajectory prefix 和 mutation parent 也不能跨 split。

## 5. 轨迹与问题的联合规划

### 5.1 轨迹角色

- `coverage`：覆盖主要区域，不追求问题特定最优角度。
- `anchor`：清楚观察关键实体，支撑 G/尺度标定。
- `long_baseline`：训练跨视角注册。
- `occlusion_reveal`：产生 unknown/known 状态转换。
- `bridge`：同时连接两个局部区域的实体，校验全局注册。
- `revisit/loop`：测身份持久与闭环。
- `held_out_target`：只供 P 验证，不进入 write prefix。

不能只采随机游走：随机轨迹常生成大量冗余帧和单图可解问题。也不能让问题直接控制所有视点：这会泄漏“被拍得最清楚的物体就是答案”。采用 coverage-first，再按能力缺口补少量 challenge views。

### 5.2 查询采样硬约束

- 至少 30% 关系/metric 查询的关键实体不共同可见。
- 至少 15% 为真实 `unknown/rejected`，不是随机塞入“不知道”。
- 六方向、四象限、metric bins、accepted/unknown 在 scene-cluster 层平衡。
- 关系 margin 分 easy/medium/boundary；低于合法阈值必须 unknown/rejected。
- 每个 query 存 evidence view、entity binding、frame、连续测量、margin 和 verifier version。
- 同一底层事实用 canonical、camera-ego、object-hypothetical 三种 frame 改写；正确答案按 frame 改变。
- 问题文本在几何编译后生成；语言模板不能决定答案分布。

## 6. `Rs_int` 泛化示范的精确设计

真实 substrate：OmniGibson `Rs_int:best`，11 个 1024² 视图，80 个 canonical entities，loop RGB PSNR 31.572 dB，depth/instance/semantic 闭环通过。网页数据由原始 NPZ、SceneIR、trajectory 和 observations 重新编译。

示范选择实体：coffee table、sofa、fridge、oven、door、standing TV。

### 6.1 已见原子/简单组合

- `G-fridge-view000`：从 instance mask 生成真实 bbox、pixel count 和稳定 UUID。
- `F-view000-view002`：监督显式 frame IDs、translation/yaw 和 composition closure。
- `R-oven-fridge-canonical`：`G+G→B→R(world)→V`，答案 `left_of`，margin 0.806 m。
- `M-coffee-fridge`：3D center distance 4.350 m，容差 0.1 m。
- `P-fridge-view005`：从 target view instance evidence 验证 fridge 不可见。

### 6.2 留出的 graph signatures

- PT：`B→F_query(coffee_table,sofa)→R(fridge)→V`；答案 back-right。训练只见独立 F probe 和 canonical R。
- Cross-view CR：fridge 仅在 kitchen views，TV 仅在 living views，两者共同可见集合为空；执行 `G+G→F*→B→R→V`。
- Closure CR：`R(oven,fridge)+R(fridge,door)→R(oven,door)→V`；三条均由直接几何复核。
- Epistemic：只保留 view-005…009 时 fridge 从未被观察，必须返回 unknown，而不是利用完整 SceneIR 猜答案。

这些 query 在网页中标注“seen atoms / held-out signature”。示范 family 是 development exemplar；正式 release 中整个 family 只能属于一个 split。

## 7. 训练导出与 loss

### 7.1 四类 SFT 记录

1. `operator_sft`：局部 `G/F/M/visibility`，建立可执行原子。
2. `state_sft`：`RGB_t,S_{t-1}→ΔS_t,S_t`，问题不可见。
3. `planner_sft`：`S,q→typed DAG`；目标包含 op、edge、类型和实体参数，不含答案参数。
4. `executor/verifier_sft`：`S,DAG→node values→answer`；以及注入错误状态/图/答案后的 reject/unknown。

### 7.2 纯 SFT 目标

\[
\mathcal L=
\lambda_\Delta L_{delta}+
\lambda_S L_{state}+
\lambda_\pi L_{graph}+
\lambda_n L_{node}+
\lambda_a L_{answer}+
\lambda_v L_{verify}+
\lambda_c L_{consistency}
\]

- 初版不要求修改 VLM：所有结构输出 token 化，使用 masked autoregressive CE。
- 坐标/角度/距离先量化为 bins；certificate 按连续 GT 容差验算，避免伪精度。
- `L_graph` 对非法类型/edge 单独加权；`L_answer` 保留 SenseNova raw answer 能力。
- `L_consistency` 可由 family sibling paired batch 实现；框架不支持成对 batch 时，用一致性过滤的 rejection sampling。
- 每条记录显式列 `input_tokens / target_spans / masked_out`，便于审计 loss 泄漏。

### 7.3 负 trace 生成

从每个正确 trace 产生 1–3 个单因素错误：

- entity swap、同类实例错绑；
- frame 方向/单位/变换顺序错误；
- state 删除、复制或提前暴露未见实体；
- left/right、front/back、距离 bin 错误；
- evidence view 缺失但仍自信作答；
- graph 类型合法但 node output 与最终答案不一致。

负 trace 必须记录唯一 corruption type；避免一次改多处导致 verifier 学不到责任节点。

## 8. SenseNova 与模拟器的三层混合

### 8.1 数据源角色

- `SenseNova raw QA`：真实图像、语言表面和任务广度；保留 answer loss。
- `OmniGibson gold episode`：严格几何、state、typed program、unknown/counterfactual 和 certificate。
- `SenseNova bridge episode`：按完全相同 image tuple 聚合 records，以 provenance 分级的 silver state/program 连接真实图像域。

本地 SenseNova 公开记录只有 `id/conversations/image`，没有可靠能力 taxonomy 字段；不能把图像前缀当作论文级能力标签。需要 question parser + program compiler，并保留 `source_label / cross_record_silver / source_cot_silver / programmatic_silver / unsupported`。

### 8.2 Pilot token mix

- 45% SenseNova raw QA。
- 35% simulator gold episode。
- 10% SenseNova bridge episode。
- 10% corrupted trace / unknown / verifier。

比例按视觉 token/图像曝光而非 JSON 行数计；通过消融调整，不作为先验结论。

## 9. 泛化 split 矩阵

| Split 轴 | Train | Test | 解释 |
|---|---|---|---|
| Graph signature | 全部 G/F/B/M/R/P/V 原子，canonical read | 未见 `F_query→R`、cross-view `F*→B→R`、branch closure | 主因果轴 |
| Scene | train scenes | scene-disjoint | 防场景记忆 |
| Family | 全部 siblings 同 split | 全部 siblings 同 split | 防 perturbation 泄漏 |
| Category pair | 常见配对 | 未见类别配对 | 支持性组合轴 |
| Camera path | coverage/短基线 | long baseline/novel target | 视角泛化 |
| Language | 多模板 | template/paraphrase holdout | 区分语言迁移 |
| Backend/domain | OmniGibson/HM3D train | opposite backend/真实图像 bridge | 域外证据 |

主结论只由“scene-disjoint + family-disjoint + graph-signature holdout”产生；其他轴作为诊断，不混成一个平均分。

## 10. 评测矩阵

- Answer：exact/categorical、numeric tolerance、route validity。
- State：entity binding F1、position bin error、status accuracy、state sufficiency gap。
- Program：op selection、type validity、edge F1、node execution accuracy。
- Consistency：all-member、pairwise、shuffle delta、frame equivariance、closure residual。
- Epistemic：selective risk、coverage-accuracy、ECE、abstention AUROC。
- Efficiency：建态 token、每 query 增量 token、K-query amortized latency。
- Data health：episode yield、rejection reason、operation/difficulty coverage、critical defect rate。

必须报告 accuracy-consistency 双轴；单一 accuracy 不能证明内部状态存在。

## 11. Pilot 与扩展门槛

### Pilot

- 至少 30 个 scene-disjoint 场景。
- 5,000 accepted families，每个 4–6 个有效 siblings。
- 每 canonical episode 8–16 views、12–24 queries。
- 每个 claimed operation/difficulty stratum 至少 100 families。
- queried entity mapping 100%；人工分层审计 critical defect <0.5%。
- 同一图像/事实生成 isolated-QA 与 episode 两种 exporter，严格等预算 A/B。

### Scale

仅在 pilot 通过预注册门槛后扩到至少 150 scenes、30,000 families。问题数量不是瓶颈；优先增加 scene geometry、遮挡机制、可编辑变化和 graph topology，而不是无限复制语言模板。

## 12. 实现优先级

### P0：研究正确性

1. 清除 executor 输入中的 `expected_relation` 等 oracle leakage。
2. 引入 `EpisodeFamilyPlan` 和 split-lock validator。
3. 扩展 belief delta：`leave_current_view/reappear/unknown`。
4. Query compiler 支持 M、P、F_query、cross-view 和 closure。
5. 按 query 输出 input/target/mask 与 corruption provenance。

### P1：训练闭环

1. JSONL/WebDataset exporter：operator/state/planner/executor/verifier。
2. family-aware sampler 与 paired batch。
3. isolated-QA 等预算 exporter。
4. state-only/full-context/shuffled-state evaluation。
5. scene-cluster bootstrap 与三种子分析脚本。

### P2：数据扩展

1. OmniGibson articulation、容器和单因素 object mutation。
2. HM3D real-scan geometry 作为外观/域对照。
3. SenseNova exact-image-tuple bridge compiler。
4. Blender/HSSD/ReplicaCAD 只通过统一 SceneIR/episode contract 接入。

## 13. 风险与拒绝条件

| 风险 | 自动信号 | 处理 |
|---|---|---|
| 单图捷径 | non-co-visible 子集不提升 | 提高 cross-view 比例，移除共同决定帧 |
| oracle leakage | no-vision/state-input 异常高分 | exporter fail closed，字段级审计 |
| 文本模板捷径 | template holdout 大幅下降 | 答案平衡、语义等价多表面、程序先生成 |
| 假 unknown | 删除证据后仍有替代证据 | 最小证据集/反事实 verifier |
| 坐标 common-mode bug | oracle 与 compiler 同错 | analytic micro-world + 独立投影/深度差分 |
| family 泄漏 | sibling hash 跨 split | 整个 release 作废并重分 |
| state 被忽略 | state-only 差、shuffle 无影响 | 提高 state复用、遮蔽原图 read、加入一致性 loss |
| 数据量掩盖机制 | episode 比 baseline 多看图/token | 严格配平事实、图像/像素/视觉 token 和 comparison-group 更新；文本 token/FLOP 差异单独报告 |

## 14. Definition of Done

该数据设计只有同时满足以下条件才可宣称“训练组合泛化”：

1. 每个 accepted answer 可从声明 evidence 和 state 独立重执行。
2. test graph signature 未在训练 exporter 中出现，但其全部原子出现过。
3. scene/family/hash 无跨 split 泄漏。
4. 主模型只见 RGB，depth/pose/mask/oracle 通道隔离有测试。
5. unknown 来源于可证明的证据不足或几何歧义。
6. isolated-QA 与 episode A/B 的问题事实、图像/像素/视觉 token 和 comparison-group 优化更新配平；总文本 token/FLOP 若不等则明示披露。
7. 按预注册阈值报告三种子、置信区间、失败模块与 guardrail。
8. 未达到门槛时明确否证，不以扩大数据规模替代机制修正。
