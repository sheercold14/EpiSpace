# EpiSpace 研究脉络与当前数据汇总 v1

> 更新时间：2026-07-24  
> 项目根目录：`/data/shichao/data/dataV100/code/episode3D`  
> 文档性质：研究上下文压缩、当前数据快照、实验解释与下一阶段契约。本文优先采用磁盘上的
> `release_manifest.json`、`corpus_report.json` 和冻结预测文件；规划文档中的旧数量不覆盖当前真源。  
> 当前结论边界：已经完成模拟器采集、几何编译、Transform Pilot 和 SenseNova 零训练诊断；
> **尚未完成能够证明 episode learning 有效的配对 SFT 主实验**。

---

## 0. 一页执行摘要

本项目要解决的不是“再造一个空间问答数据集”，而是一个监督学习中的结构性问题：
孤立的图文 QA 只要求模型从输入得到最终答案，无法区分模型是在构建可复用空间状态，还是按题型学习
局部视觉—语言捷径。SenseNova-SI 一类大规模空间数据证明了扩量可以提升单项准确率，但文本 CoT
收益不稳定；我们的本机结果进一步显示，答案准确率、推理格式和可执行空间推理可以彼此分离。

用户提出的关键方向是：**利用模拟器生成相似场景下可控的空间变换和物体干预，把训练单位从孤立
QA 改造成 episode experience；模型只看 RGB 序列和自然语言，但每个问题、答案和中间线索都由隐藏
空间程序与几何 certificate 生成和复核。** 模型不是程序提取器，也不在推理时读取坐标、点云或
simulator truth。隐藏程序负责把正确性做硬，语言负责把监督做得像人类可以理解和学习的经验。

当前研究假设可以写成：

> 在底层事实、RGB、优化预算和通用回放严格配平时，`observe-first/read-many` 的 episode 组织、
> 反事实 sibling family 和可执行空间变换监督，是否比 isolated QA 更容易学到跨问题复用的空间更新，
> 从而提升未见场景、未见变换和未见操作组合上的准确率与证据敏感性？

这一假设是可证伪的。若配对训练后组合留出、family exact match、顺序稳定性和 state sufficiency
均无改善，就不能把“episode”本身写成有效机制。

---

## 1. 对话中形成的核心问题意识

### 1.1 项目发起人的核心主张

以下内容是用户在整个讨论中反复校正项目方向后留下的高价值共识，优先级高于某个具体模板或工具：

1. **我们训练的是多模态大模型，不是空间程序提取器。**  
   模型实际输入应是 RGB 和自然语言，模型实际输出应是自然、可训练的回答。位姿、深度、mask、
   OBB、实体 UUID、canonical truth 和 operation DAG 属于编译器与验证器的隐藏通道，不能直接喂给
   模型后再宣称获得空间理解。

2. **benchmark 不能退化成“新增视角后不断记忆物体”的单一题型。**  
   渐进式空间记忆是重要能力，但第一版 benchmark 的目标是空间能力的迁移与组合：视觉接地、相机或
   观察者 frame 更新、跨视图关系、度量、心理换位、遮挡与未知判断，需要在同一 episode 中发生读写
   关系，而不是成为互相独立的数据桶。

3. **单条自然语言 CoT 很容易成为新模板。**  
   即使答案前增加“先看哪里、再怎么想”，模型仍可学习固定句式、先猜答案再补解释。模拟器真正不可
   替代的优势是：同一底层场景可以旋转、镜像、移动物体、替换决定帧、改变参照系；语言大体相似，
   但正确答案必须按空间规律变化。这种干预比继续增加 paraphrase 更能抗过拟合。

4. **人类是“RGB 序列 + 更新程序可行”的存在证明。**  
   项目接受 view-based representation、spatial updating 和 mental simulation 的计算直觉：人不需要
   显式打印度量点云才可完成视角更新。因而我们可以让学生模型只看图像经验，同时用模拟器在幕后验证
   它是否遵循正确变换。这里应避免把心理学类比直接当成模型机制证据；它提供设计启发，最终仍需行为
   实验验证。

5. **episode 的意义是让能力发生转化，不只是把多个 QA 装进长上下文。**  
   一次观察应被多个异构问题共同复用：早先 grounding 写入的对象，在后面的 perspective、relation、
   metric、memory 和 unknown 判断中继续被读取。若每一轮都能只靠当前图作答，形式上是多轮，学习上
   仍是 isolated QA。

6. **问题和回答必须自然，但自然语言没有真值权。**  
   先由世界真值和 typed program 决定可答性、证据、变换与答案，再生成自然语言。Claude/Fable/Codex
   一类 LLM 可作为“配音”或表面改写器，但不能创造事实。改写后必须重新解析并通过 certificate。

7. **数值只在它提供训练信号时出现。**  
   距离、角度和大小不能为了显得专业而硬塞；只有当定义明确、视觉上可估、容差可声明并且几何可复算
   时才输出。例如“中心到中心约 5.0 米”需要明确测量定义和容差；边界方向必须使用 margin，不能把
   接近正左的物体武断写成“左前方”。

8. **数据必须能被直接审阅、训练和复现实验。**  
   用户持续要求真实数据样例、轨迹位置、pipeline 位置、网页浏览、全部轨迹分类以及模型实际输出。
   因此每一版数据必须同时提供母表、SFT 导出、oracle、拒绝原因、网页样例和模型原始响应，不能只有
   一张汇总表。

9. **先做可解释的 pilot，再扩量。**  
   QA 数量可以程序化膨胀，稀缺的是场景几何、多样信息结构和完整反事实家族。扩量单位应优先是
   scene/layout/family，而不是同一事实的语言表面数量。

10. **渲染质量服务于可识别性，不与“电影级画质”混为一谈。**  
    HM3D 可作为真实扫描域和 Habitat 生态补充；当前主生产选择 OmniGibson/BEHAVIOR 资产，是因为
    可编辑物体、可控相机、实例级传感器和反事实重放更适合本研究。只要目标物可辨识、关系可判定、
    RGB 不退化，模拟渲染可以可靠提供机制训练；最终必须用跨后端或真实图像评测检验 sim-to-real。

### 1.2 Fable5 对理论与数据结构的关键贡献

Fable5 的高价值意见可压缩为五点：

1. **SenseNova 的问题不只是数据规模，而是目标函数不能识别内部计算。** 大规模 raw QA 可以把每种
   题型的捷径分别磨得更好，却不保证这些能力共享一个可传递中间状态。
2. **文本 CoT 的失败不等于“空间地图”方向失败。** 真正失败的是无验证、无规范 frame、无增量状态、
   只有答案级奖励的长文本链。
3. **空间心智模型应由行为性质定义。** 持久性、路径/顺序相对无关、视角等变、变换组合闭合和知所
   不知，既是评测轴，也可以转化为训练约束。
4. **一次 episode = 一个场景、多个观察写入、多个异构读取，共享一个 state。** PT/相机更新更像写入
   骨干，SR/MM/CR 是在声明 frame 下对共享状态进行读取。
5. **自然推理需要中间“线索路标”。** 早期 perspective 样本的 claim sheet 只存终点事实，导致配音
   LLM 只能把答案扩写成套话。正确结构应是 `cue → transform → conclusion`：哪张图提供了什么线索、
   frame 如何改变、最后得到什么结论。方向边界还必须由 OBB-aware margin 控制，验证器不能硬编码
   `True`。

### 1.3 我们最终采用的修正

- `G/F/B/M/R/P/V` typed operation graph 是**后台中间表示**，不是学生默认输出格式。
- canonical spatial state 只允许作为显式请求的 auxiliary target；普通 QA 不能看到它。
- Grounded-CoT 是一种待验证的监督臂，不预设它一定优于 Answer-only。
- trace 的“格式齐全”不等于 trace 正确，更不等于它被答案因果使用。
- 主论文必须比较相同事实的 episode 与 isolated 两臂，而不是把不同数据量的模型直接比较。
- 所有结论按 scene/family 聚类统计；counterfactual siblings 和语言改写不能被当作独立世界数量。

---

## 2. 第一性原理：为什么 isolated QA 容易学成捷径

### 2.1 答案损失的不可辨识性

对孤立样本 \((O,q,y)\)，我们希望模型学习：

\[
S=\operatorname{Write}(O),\qquad
y=\operatorname{Read}(S,q),
\]

其中 \(S\) 是被多个问题复用的空间状态。但答案级交叉熵同样奖励更便宜的函数：

\[
y=H_q(O,q),
\]

这里 \(H_q\) 可以依赖题型词、选项分布、单帧局部线索和语言先验。只要两条电路输出相同 \(y\)，
标准答案 loss 无法判断哪条电路被使用：

\[
\mathcal L_{\text{ans}}
=-\sum_{t\in y}\log p_\theta(y_t\mid O,q,y_{<t}).
\]

因此“单项 accuracy 提升”不能直接推出共享空间模型出现。

### 2.2 为什么自由文本 CoT 不能自动解决

加入文本轨迹 \(z=(z_1,\ldots,z_m)\) 后：

\[
p_\theta(z,y\mid O,q)
=\left[\prod_{i=1}^{m}p_\theta(z_i\mid O,q,z_{<i})\right]
\cdot p_\theta(y\mid O,q,z).
\]

这引入三个风险：

1. 局部错误进入后续自回归上下文并累积；
2. 大量“像推理”的语言 token 主导梯度，却未必对应空间变换；
3. 模型可以先利用捷径确定答案，再生成与答案一致的事后解释。

因此死掉的不是中间监督，而是**不可执行、不可定位、不可反事实检验的中间监督**。当前 Grounded-CoT
将输出拆为：

\[
z=(z_{\text{cue}},z_{\text{transform}},y),
\]

并暂定：

\[
\mathcal L_{\text{grounded}}
=0.25\mathcal L_{\text{cue}}
+0.50\mathcal L_{\text{transform}}
+0.25\mathcal L_{\text{answer}},
\]

每段先做 token mean，再做 fact/episode mean，避免 transform 仅因更长而获得更大权重。该 loss
仍然需要 transform-shuffled、mismatched-trace 和 conclusion-only 对照，才能证明收益来自正确程序，
而不是额外 token 或提示风格。

### 2.3 episode 如何改变学习经济性

Episode 把一次观察前缀 \(O_{1:M}\) 与 \(K\) 个异构读取绑定：

\[
S_M=\operatorname{Write}(O_{1:M}),\qquad
\hat y_k=\operatorname{Read}(S_M,q_k),\quad k=1,\ldots,K.
\]

如果每个问题都单独走捷径，需要学习 \(K\) 条互不共享的映射；若 grounding、frame 和 belief 能复用，
共享写入开始具有优化上的经济性。但“多问几题”本身仍不足以强迫正确状态，所以还需要：

- **反事实家族**：证据、布局或 frame 改变时，答案必须按解析规律改变；
- **程序签名留出**：训练覆盖全部原子，但测试保留指定组合；
- **unknown sibling**：完整证据可答，删掉决定证据必须弃答；
- **独立执行验证**：从隐藏几何重新计算答案，而不是让语言模型自评。

对于空间变换 \(T\)，理想行为满足：

\[
f_\theta(Tx)=\rho(T)f_\theta(x),
\]

其中 \(\rho(T)\) 是答案空间中可计算的旋转、镜像或 frame 变化。这个等变关系比单题准确率更接近
“模型是否执行空间更新”。

---

## 3. 空间语法与数据通道

### 3.1 Typed Spatial Operation Graph

当前后台 IR 采用七类原子操作：

| 操作 | 含义 | 典型输入 | 典型输出 |
|---|---|---|---|
| `G` Grounding | 在图像中绑定区域、物体或稳定实体 | RGB、类别提示 | entity / evidence |
| `F` Frame | 对齐 ego、allo、相机或物体坐标系 | pose chain、anchor | transform |
| `B` Belief | 累积可观察的场景状态 | entities、frames、history | observable belief |
| `M` Metric | 估距离、尺度、高度或角度 | entity/depth/scale cue | metric value |
| `R` Relation | 判断左右前后、包含、遮挡等 | entity pair、frame | relation |
| `P` Perspective | 预测假想观察者或目标视角 | belief、target view | query coordinates/view |
| `V` Verify | 重放、反事实检查或决定 unknown | claims、certificate | pass/reject/unknown |

类型约束阻止无意义组合。例如 `R` 必须知道比较对象和 frame，`P` 必须有 belief 与 target view，
`M` 必须声明测量定义。程序图本身不含最终答案词，避免“answer leakage through IR”。

### 3.2 三类状态严格分离

1. `world_state`：模拟器完整真值，包含未观察实体；只供 oracle。
2. `observable_state_t`：截至时刻 \(t\) 的输入图像可以支持的事实。
3. `belief_target_t`：模型应安全保留的最小状态，允许
   `visible / seen_not_current / unknown`。

canonical frame 默认可取第一帧相机地面投影为原点，右、前、上分别为 \(+X,+Y,+Z\)。这只是统一
编译和验证的坐标契约；模型面对某个 ego 问题时，必须执行 canonical-to-query-frame 变换，不能把
world X/Y 直接泄漏到问题中。

### 3.3 模型可见与隐藏通道

| 通道 | 内容 | 是否给学生模型 |
|---|---|---:|
| Model view | 有序 RGB、动作说明、自然语言问题、选项 | 是 |
| Assistant target | Answer-only 或自然语言 grounded trace | 训练时是 |
| Compiler IR | typed DAG、entity UUID、pose、OBB、mask、depth | 否 |
| Certificate | 可见性、方向 margin、变换闭合、反事实执行值 | 否 |
| Target render | 假想目标视图或验证帧 | 否，oracle-only |

---

## 4. 从 3D 资产到训练 episode 的自动化 pipeline

```text
OmniGibson / BEHAVIOR 场景与可编辑物体
  → coverage-first 轨迹规划
  → RGB + depth + instance + semantic + camera pose
  → 轨迹/视觉/可指称性质量门
  → observable belief 与 visibility timeline
  → typed program 和 structured answer
  → 独立 geometry replay certificate
  → question realization
  → answer realization / 受约束 LLM 配音
  → episode、isolated、state-aux、RLVR、benchmark 配对导出
  → corpus audit、人工抽检、网页审阅、冻结 checksum
```

关键生产原则：

1. `coverage-first`：轨迹先采，问题后编译，禁止为某道题反向定制视点。
2. `geometry-first, language-last`：几何执行值先于自然语言。
3. `fail-closed`：缺证据、关系 margin 过小、目标不可辨、family 缺成员，整条或整族拒收。
4. `scene/family lock`：同场景和同反事实族不得跨 train/validation/test。
5. `answer-blind question realization`：问题改写器不能读取答案槽。
6. `LLM dubbing`：只改写语言表面；claim sheet 规定允许表达的事实，并要求
   `cue → transform → conclusion` 的角色覆盖。

当前轨迹信息结构包括：

| 轨迹 | 主要信息结构 | 训练/评测能力 |
|---|---|---|
| T1 覆盖漫游 | 平移、重访、闭环 | 建图、长期记忆、出现顺序 |
| T2 稀疏/Among | 宽基线、共同锚点、局部视图 | 跨视图注册、关系组合 |
| T3 原地旋转 | 零平移、已知 yaw 更新 | self-rotation、全景整合 |
| T4 物体环绕 | 物体中心弧线 | 实例恒常、物体朝向 |
| T7 高度/俯仰 | 同站位不同高度和 pitch | 绑定与 frame transfer |
| T8 遮挡揭示 | unknown→known | 物体恒存、认知诚实 |
| T9 干预对 | 镜像、置换、删证据 | 反捷径、反事实一致性 |
| T10 target render | 目标视角 oracle | perspective verifier |

T4、T8 的较低通过率属于真实资产约束，不能通过放宽标签语义掩盖。尤其完整物体环绕要求净空和
非对称物体，天然不适用于所有室内资产。

---

## 5. 当前数据资产快照

### 5.1 主 EpiSpace Pilot

权威目录：`data/epispace_pilot_v1/`。当前 release manifest 标识为
`epispace-pilot-v1.2-2026-07-17`，状态 `pass`。

采集计划共 322 jobs；严格通过 bundle 193 个，分布为：

| 来源 | 通过 | 角色 |
|---|---:|---|
| T1 walkthrough | 28 | write source |
| T3 rotation 两轮 | 83 | write source |
| T4 adaptive arc | 10 | write source |
| T7 elevation | 43 | write source |
| T8 reveal | 8 | write source |
| T10 target view | 21 | oracle verifier |

当前磁盘文件实际行数：

| Artifact | 数量 | 作用 |
|---|---:|---|
| `episodes.ir.jsonl` | 158 | 可观察 episode IR |
| `train.episode_sft.jsonl` | 340 | observe-first/read-many 臂 |
| `train.isolated_sft.jsonl` | 466 | 同事实 isolated 对照臂 |
| `train.state_aux_sft.jsonl` | 102 | 可选 state commit |
| `train.rlvr.jsonl` | 466 | verifier-ready prompt |
| `benchmark.jsonl` | 279 | 全量诊断 |
| `benchmark.family.jsonl` | 110 | family 评测 |
| `benchmark.composition.jsonl` | 58 | 程序签名组合留出 |
| `benchmark.core.jsonl` | 134 | 主评测去重集合 |

注意：`ICLR2027核心主线与数据管线_v1.md` 中存在 651 facts、172 write sources、27 composition 等
另一版统计，而当前 release manifest 和实际文件是 466/158/58。这是明确的**版本一致性待办**：
投稿前必须重新冻结 manifest、论文表格和 compute schedule，禁止混用两个快照。

### 5.2 Scene Dialogue v1

目录：`data/scene_dialogues_v1/`。

- 29 个通过的 scene dialogue；
- 8 个场景因合格轮数不足被拒绝；
- 319 张 RGB；
- `train.dialogue_episode_sft.jsonl`：29 条多轮 episode；
- `train.dialogue_isolated_sft.jsonl`：146 条 isolated 对照；
- 每个场景最多覆盖 inventory、同框关系、last-seen、非同框跨视图关系、object perspective、
  epistemic loop 和 layout summary。

这条语料线验证了“轨迹真值 → 多轮剧本 → claim sheet → 自然回答”的可行性，但它更接近自然多轮
示范，尚未形成 Transform Pilot 那样严格、规模化的反事实变换对照。

### 5.3 Transform Pilot v1

权威目录：`data/transform_pilot_v1/dataset/`。数据集 ID：
`epispace-transform-pilot-v1-2026-07-22`，状态 `data_ready`。

| 项目 | 数量 |
|---|---:|
| teacher records | 2,961 |
| unique geometric facts | 1,803 |
| Self-Rotation records | 2,641 |
| Among-5 records | 320 |
| complete Among families | 16 |
| complete counterfactual query groups | 64 |
| split | train 2,036 / validation 517 / test 408 |
| 平衡去重导出 | train 724 / validation 240 / test 172 |

Among-5 的 `front/back/left/right` 各 80 条，完全平衡；Self-Rotation 仍不平衡，因此主实验应优先使用
平衡去重导出，并把全量集作为语言表面和自然分布诊断。当前自动审计确认：

- Answer-only 与 Grounded-CoT 输入和标签完全一致；
- scene-disjoint split；
- 每个 Among 查询拥有五个完整 sibling：
  `identity / rotate90 / mirror / permute1 / radius185`；
- 旋转、镜像手性、前后保持、置换变化和尺度不变审计均为 100%；
- 每条 Grounded-CoT 均含 grounded `cue → transform → conclusion`。

`data_ready` 只代表 MACHINE READY，不等于 HUMAN REVIEWED 或 TRAINING RELEASE。正式训练冻结前仍需
benchmark 100% 人工复核、训练事实分层抽检至少 10% 且不少于 100 个 facts。

---

## 6. 真实数据样例

### 6.1 多轮 episode 样例：`Rs_int_seed17`

文件：`data/scene_dialogues_v1/scenes/Rs_int_seed17.dialogue.json`  
结构：11 张图，7 轮，图像按真实漫游顺序逐步进入上下文。

| 轮次 | 新图 | 主要读取 | 隐藏程序 |
|---|---|---|---|
| R1 | view-000 | 当前可见物体 inventory | `G→V` |
| R2 | view-001–002 | 同框关系与中心距离 | `G+G→B→R+M→V` |
| R3 | view-003–004 | 冰箱当前不可见、最后出现位置 | `G(sequence)→B_temporal→V` |
| R4 | view-005–007 | 两物体从未同框时的跨视图关系 | `G+G→F*→B_global→R→V` |
| R5 | view-008–009 | 物体锚定的假想观察者朝向 | `G→B→F_query→P→R→V` |
| R6 | view-010 | 未观察类别与闭环持久性 | `G/B→V_unknown/loop` |
| R7 | 无新图 | 可观察布局总结 | 多事实 state commit |

例如 R4 问：

> 到目前为止，公共垃圾桶和门有没有在同一个视角里同时出现过？综合前后所有观察，公共垃圾桶在门的
> 什么方向？

训练回答不是只填“前方”，而是按 claim sheet 组织：

1. `cue`：门只出现在第 6–8 个视角，垃圾桶只出现在第 1–3 个视角，两者从未同框；
2. `transform`：第 6 个视角附近相机掉头约 \(180^\circ\)，把掉头前后证据登记到同一空间；
3. `conclusion`：以出发朝向为参照，公共垃圾桶在门的前方。

这里模型只看到 RGB 和自然语言；“掉头约 \(180^\circ\)”来自位姿链 certificate，不是配音模型猜测。

### 6.2 Self-Rotation 样例

记录：`transform-sft-002caaf8d1584b52cb4a`  
任务标签：`perspective_taking / mental_simulation / view_local`

**模型输入**

> System：这些图按时间顺序拍于同一站位；相邻两图之间，观察者原地向右转 60 度。  
> Question：你现在站在最后一张图的原地朝向。假设接下来向右转 120 度，转身后的画面不会提供。
> 扶手椅会位于你的哪个方向？A.右侧 B.左侧 C.后方 D.前方

**真值与隐藏监督**

- 答案：`B. 左侧`
- operation graph：
  `G → B_view_memory → F_self → R_bearing → V`
- cue：扶手椅在当前图中位于右前方，可见像素 29,571；
- transform：
  \[
  \text{bearing}_{after}
  =\text{bearing}_{before}-\text{observer turn};
  \]
- 答案方位角：\(-81.85^\circ\)；
- OBB-aware effective margin：\(25.93^\circ\)；
- 目标朝向图 `view-004` 被隐藏，防止直接看答案。

**实际零训练输出**

| 模型 | Answer-only | Grounded-CoT prompt |
|---|---|---|
| SenseNova 1.1 | `B`，正确 | `<cue>扶手椅</cue><transform>right</transform><answer>D. 前方</answer>`，错误 |
| SenseNova 1.5 | `B. 左侧`，正确 | `<answer>C</answer>`，错误且未输出 cue/transform |

这个样本直接说明：Answer-only 正确不能证明变换程序存在；强制 CoT 也可能破坏原本正确的答案。

### 6.3 Among-5 跨视图样例

记录：`transform-sft-1f0ac808cc745ebcbd0d`  
场景：`hall_conference_large`  
反事实变体：`radius185`

**输入结构**

- 四张图来自同一布局；
- 相机沿逆时针方向绕中心脚凳移动 \(90^\circ\)，始终面向脚凳；
- 每张图恰好出现“共同脚凳 + 一个外围物体”；
- 盆栽和背包从未同框。

问题：

> 以第 1 张图拍摄者面向脚凳的方向为“前”，盆栽在背包的哪个方向？

答案：`C. 左侧`。

隐藏程序：

```text
G(partial_views, anchor)
→ F_register(anchor, ordered_motion)
→ B_layout5
→ R(query_frame)
→ V_direction
```

certificate 给出查询 frame 为 `camera@view-000`、关系方位角 \(-109.46^\circ\)、OBB-aware effective
margin \(14.40^\circ\)，且 `co_visible_view_ids=[]`。所以模型必须通过共同脚凳和有序相机运动整合至少
两张局部图，无法在单图中直接读取答案。

实际输出：

| 模型 | Answer-only | Grounded-CoT prompt |
|---|---|---|
| SenseNova 1.1 | `A`，错误 | `<cue>Image-1</cue> <transform>Image-2</transform> <answer>C. 左侧</answer>`，答案正确但 trace 内容贫弱 |
| SenseNova 1.5 | `D`，错误 | `<answer>D</answer>`，错误且无 trace |

### 6.4 Teacher record 的核心 schema

```json
{
  "record_id": "stable-id",
  "scene_id": "scene-or-layout",
  "family_id": "counterfactual-family",
  "split": "train|validation|test",
  "model_input": {
    "images": ["ordered-rgb-1", "ordered-rgb-2"],
    "system": "动作与坐标约定",
    "question": "自然语言问题与随机化选项"
  },
  "answer": {"key": "B", "label": "left", "surface": "B. 左侧"},
  "grounded_trace": {
    "cue": {"claims": [], "surface": "证据来自哪里"},
    "transform": {"claims": [], "surface": "如何更新 frame"},
    "conclusion": {"label": "left"}
  },
  "operation_graph": {
    "nodes": ["G", "B", "F", "R", "V"],
    "typed_signature": "answer-free signature"
  },
  "certificate": {
    "measurements": "oracle only",
    "quality_gate": "oracle only"
  }
}
```

---

## 7. 当前 SenseNova 零训练实验

评测范围为 Transform Pilot 全部 2,961 条。四选一随机水平约为 25%，但由于 Self-Rotation 全量标签
不平衡，必须同时参考平衡集和逐任务结果。

### 7.1 全量结果

| 模型 / prompt | Overall | Among-5 | Self-Rotation | trace 完整 | 正确且 trace 完整 |
|---|---:|---:|---:|---:|---:|
| SenseNova 1.1 Answer-only | 24.21% | 9.38% | 26.01% | — | — |
| SenseNova 1.1 Grounded-CoT | 22.05% | 15.00% | 22.91% | 44.85% | 9.32% |
| SenseNova 1.5 Answer-only | 36.24% | 33.75% | 36.54% | — | — |
| SenseNova 1.5 Grounded-CoT | 38.33% | 36.25% | 38.58% | 0% | 0% |

### 7.2 平衡去重集合

| 模型 / prompt | Accuracy |
|---|---:|
| SenseNova 1.1 Answer-only | 28.70% |
| SenseNova 1.1 Grounded-CoT | 25.79% |
| SenseNova 1.5 Answer-only | 33.36% |
| SenseNova 1.5 Grounded-CoT | 35.21% |

### 7.3 可以成立的解释

1. **1.5 的答案能力显著强于 1.1。** 全量 Answer-only 提高 12.03 个百分点；配对 McNemar 检验中，
   1.5-only correct 895 条、1.1-only correct 539 条，差异不是少量样本偶然造成。
2. **1.1 的文本 CoT 总体伤害答案。** Overall 下降 2.16pp；Among 提升 5.63pp，但 Self-Rotation
   下降 3.10pp，说明不同变换任务对 verbalization 的反应不同。
3. **1.5 的 Grounded-CoT prompt 提高 2.09pp，但不能称为 CoT 推理提升。** 它在 2,961 条中没有
   一条完整输出 `cue + transform + answer`；几乎都只输出 `<answer>…</answer>`。更合理的表述是：
   提示语境改变了答案行为，而模型忽略了显式推理合同。
4. **准确率、格式和机制是三个不同变量。** 1.1 能生成格式但 joint 很低；1.5 更准确却完全不遵循
   trace 格式。现有工作若只报告 answer accuracy 或只检查“有没有 CoT”，都会遗漏这个分离。

### 7.4 当前实验不能证明的内容

- 不能证明 Grounded-CoT SFT 有效，因为还没有进行配对训练；
- 不能证明模型内部形成 canonical spatial state；
- 不能证明 episode 优于 isolated QA；
- 不能把全数据 zero-shot 结果当作 held-out 泛化主结果；
- 不能因为 trace 文本与答案一致，就认定 trace 被答案因果使用；
- 不能声称覆盖 SenseNova 全部空间能力，目前 Transform Pilot 只聚焦 Self-Rotation 与 Among-5。

---

## 8. 由现有结果导出的研究点

### R1. Executable spatial updating vs. textual CoT

研究问题：如果 cue 和 transform 由几何程序严格生成，SFT 后是否真正改善未见变换，而不仅是提高
格式服从？  
必要对照：Answer-only、free-form CoT、Grounded-CoT、transform-shuffled、mismatched trace。

### R2. Counterfactual equivariance

研究问题：同一 latent layout 的旋转、镜像、置换和尺度变化，模型答案是否按 \(\rho(T)\) 正确变化？  
核心指标不能只是 sibling 平均 accuracy，还应报告：

\[
\text{FamilyExact}
=\frac{1}{|\mathcal F|}
\sum_{f\in\mathcal F}
\mathbf 1[\text{family }f\text{ 的全部成员均正确}].
\]

尺度 sibling 要求关系不变，旋转/镜像 sibling 要求答案按解析映射变化。

### R3. Persistent episode state

研究问题：同一观察前缀服务多个异构问题时，模型是否形成可复用状态？  
探针：

- delayed query：证据出现很早，问题延后；
- image withdrawal：建态后撤掉原图，仅凭 state 回答；
- order perturbation：语义合法的观察顺序改变后答案保持；
- loop closure：返回起点时实体位置保持；
- read-many amortization：一次观察后连续回答不同能力问题。

### R4. Held-out operation composition

训练可分别见到 `F`、`B`、`R` 等原子，但测试保留完整签名，例如：

```text
G → F_anchor_register → B_layout → R(query_frame) → V
```

如果 episode 训练只提升 IID 题、不提升未见签名，就不能声称组合泛化。

### R5. Knowing what is unknown

通过 `unknown → known → unknown` 的证据三元组检验认知诚实：

- prefix 缺决定证据：应弃答；
- reveal 加入决定证据：应答对；
- delete 再移除决定证据：应恢复弃答。

必须以 reveal 答对为条件评估证据敏感性，防止“始终弃答”获得高一致性。

### R6. State sufficiency 与因果使用

建态完成后比较：

\[
\operatorname{Acc}(\text{state-only})
\quad\text{vs.}\quad
\operatorname{Acc}(\text{full RGB context}).
\]

接近只说明 state 信息充分，还不能单独证明原模型自然使用该 state；进一步需要 state corruption、
entity swap 和 frame perturbation，检查答案是否按 state 干预变化。

### R7. 世界规模而不是 QA 表面规模

扩量应优先增加 scene topology、遮挡模式、资产组合、运动基线和反事实 family。报告时同时列出：

- unique scenes/layouts；
- unique geometric facts；
- program siblings；
- language surfaces；
- image/token exposure。

不能把 2,961 teacher records 写成 2,961 个独立空间事实；当前独立几何事实是 1,803。

### R8. Simulator-to-real / backend transfer

OmniGibson 适合生成可编辑反事实；HM3D 适合提供真实扫描纹理和 Habitat 生态。可采用：

```text
OmniGibson 训练 → HM3D / MindCube / MMSI / VSI 测试
```

或用 HM3D 轨迹作为域外验证。只有跨后端性能仍改善，才能说明方法不是记忆特定资产或渲染风格。

---

## 9. 下一阶段主实验

### 9.1 四个训练臂

在相同初始化、事实、图像和优化步数下比较：

| 臂 | 监督形式 | 回答的问题 |
|---|---|---|
| A | isolated Answer-only SFT | 单纯答案监督能达到什么水平 |
| B | isolated free/templated CoT SFT | 增加文本链是否有效 |
| C | isolated program-grounded trace SFT | 可验证 transform 是否优于普通 CoT |
| D | episode observe-first/read-many + grounded trace | 共享写入是否带来组合泛化 |

为了识别机制，还应增加低成本诊断臂：

- C-shuffle：正确答案配错误 transform；
- C-conclusion：保持相近 token 数但去掉 transform；
- D-question-homogeneous：episode 内全是同类题，检查收益是否来自多轮长度而非跨能力共享；
- D-order-perturbed：合法改变观察顺序。

### 9.2 训练控制

- 只对 assistant tokens 计算 CE；
- episode 内每个事实的 answer/trace span 先独立 mean，再聚合；
- Answer 与 Grounded 两臂共享 `record_id/comparison_id`；
- 同一 family 不跨 split，也尽量不在同 batch 形成答案泄漏；
- 图像出现次数、视觉 token、文本 token、总 token 和优化更新分别报告；
- 通用多模态回放用于防止灾难性遗忘；
- 至少三随机种子，scene/family cluster bootstrap 95% CI。

### 9.3 主指标

1. scene-disjoint atomic accuracy；
2. balanced macro accuracy；
3. held-out signature accuracy；
4. family exact match；
5. frame equivariance exact；
6. accuracy-conditioned evidence sensitivity；
7. executable trace correctness；
8. state sufficiency 与 state corruption sensitivity；
9. no-image、critical-frame ablation、order perturbation；
10. 通用 VQA/对话能力变化。

主论文最干净的成功条件应预注册为：D 相对 A/C 在组合留出上显著提高，同时 family consistency 和
证据敏感性提高，通用能力基本不退化。如果只有普通 accuracy 提升而 family/trace/state 指标不变，
更可能是数据拟合而非空间模型形成。

---

## 10. 当前问题与发布前清单

1. **主训练尚未运行。** 当前最强证据是 zero-shot diagnosis，不是 learning result。
2. **任务覆盖较窄。** Transform Pilot 只有 T3 Self-Rotation 与程序化 Among-5；Metric、遮挡、
   perspective、unknown 等需要并入下一版反事实 family。
3. **Self-Rotation 标签分布不均。** 主结果必须使用平衡集或宏平均。
4. **Trace fidelity 未验证。** 现有 parser 主要检查结构；后续需执行每个 cue/transform claim，并做
   trace corruption 因果测试。
5. **自然语言仍有模板风险。** LLM 配音只能做受保护槽位改写；需要 paraphrase split 和样式鲁棒性。
6. **视觉感知可能是下限瓶颈。** 必须报告 no-image、目标像素、裁切、识别混淆和 critical-frame
   ablation，区分“看不见”与“不会变换”。
7. **人工复核未完成。** `data_ready` 不能被写成正式 training release。
8. **主 EpiSpace 数量存在文档版本差异。** 以当前 manifest/实际文件为准，并在冻结论文数据前统一
   重建所有统计表。
9. **网页目前展示 18 个代表记录，不是全部 2,961 条。** 它适合深入审阅；全量审阅应增加分页索引
   或独立 reviewer packet。
10. **外部迁移尚未测试。** 不应提前声称优于 MindCube/MMSI/VSI 或具备 sim-to-real 泛化。

---

## 11. 文件与网页索引

### 研究与规范

- 本文：`skill/plan/EpiSpace研究脉络与当前数据汇总_v1.md`
- 主研究契约：`skill/plan/ICLR2027核心主线与数据管线_v1.md`
- 组合泛化设计：`skill/plan/空间组合泛化Episode数据设计_v1.md`
- Transform Pilot：`skill/plan/EpiSpace_Transform_Pilot_v1_实施计划.md`
- 轨迹规范：`skill/plan/轨迹采集规范_v1.md`
- 对话生成：`skill/plan/multi-turn-generation.md`
- 训练流程：`skill/plan/训练流程_v1.md`

### 数据

- 主 EpiSpace：`data/epispace_pilot_v1/`
- 多轮 dialogue：`data/scene_dialogues_v1/`
- Transform 母表：`data/transform_pilot_v1/dataset/teacher_records.jsonl`
- Transform 报告：`data/transform_pilot_v1/dataset/corpus_report.json`
- 平衡 Grounded-CoT 训练：
  `data/transform_pilot_v1/dataset/balanced_isolated_grounded_cot_train.jsonl`
- Episode Grounded-CoT：
  `data/transform_pilot_v1/dataset/episode_grounded_cot_train.jsonl`
- 全量评测输入与 oracle：
  `eval_inputs_all.jsonl`、`eval_oracle_all.jsonl`
- 模型原始预测：`data/transform_pilot_v1/dataset/evaluation/`

### Pipeline

- 主编译管线：`episode3d/`
- Transform 数据编译：`episode3d/transform_pilot/`
- Scene dialogue 真值与语言：`episode3d/scene_llm_pipeline/core.py`、
  `episode3d/scene_llm_pipeline/dubbing.py`
- Transform 网页打包：`scripts/build_transform_web.py`

### 网页

```text
http://127.0.0.1:8770/episode3D/web/transform-pilot.html
```

网页当前展示 18 个代表样本、72 条真实模型响应：

\[
18\ \text{records}
\times 2\ \text{models}
\times 2\ \text{prompt modes}
=72\ \text{responses}.
\]

---

## 12. 建议采用的论文叙事

最稳健的论文叙事不是“我们生成了更多空间 QA”，而是：

1. **问题诊断**：答案级监督无法识别共享空间计算；强模型的答案准确率、文本 trace 和真实空间更新
   彼此分离。
2. **数据方法**：用可编辑模拟器生成 coverage-first RGB episode，并将每个 latent fact 扩展为受
   几何 certificate 约束的反事实 sibling family。
3. **学习干预**：在事实和视觉曝光配平下，比较 isolated 与 observe-first/read-many，测试共享写入
   是否成为更经济的解。
4. **机制评测**：不只测 accuracy，还测未见程序组合、family exact、frame equivariance、证据敏感性、
   state sufficiency 与 unknown。
5. **可证伪结论**：只有 episode 臂在这些机制指标和外部域上共同提高，才能主张模型学到了更可复用
   的空间更新；否则应把贡献限定为可控数据引擎与诊断 benchmark。

一句话版本：

> **EpiSpace 不用模拟器替模型做推理，而是用模拟器制造模型无法靠固定答案生存的世界变化，并用
> episode 结构让可复用的空间更新比一题一策更值得学习。**
