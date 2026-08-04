# EpiSpace Transform Pilot v1：可验证视图变换 CoT 数据实施计划

## 1. 研究命题

第一版只检验一个窄而可证伪的命题：**与仅监督答案相比，由模拟器程序生成、逐句受几何证书约束的
`观察 → 变换 → 结论` CoT，能否改善模型在未见空间变换组合上的泛化。** 这里监督的是
`program-grounded textual trace`，不是宣称模型显式预测或内化了隐藏 DAG。隐藏程序的职责是生成真值、
约束语言和执行反事实复核；模型接口仍是有序 RGB、自然语言问题和自然语言 CoT。

零训练实验与训练实验分开陈述：本机冻结 SenseNova-SI 与对应基座，只回答模型是否已具有变换能力、
输出 trace 是否忠实、是否使用了视觉证据；之后在另一台机器进行的配对 SFT 才能回答 Grounded-CoT
是否导致组合泛化提升。

## 2. 两个主任务

### 2.1 Self-Rotation（T3）

T3 在同一站位以 60° 步长覆盖一周。采集约定固定为：声明角度为正表示观察者向右/顺时针转，
对应数学相机 yaw 的负增量。每条基础事实包含：有序模型可见视图、可唯一指称的目标物体、一个由
60° 原子组成的假想转身程序、转身后四方向答案，以及不进入模型输入的目标朝向视图。

两种难度均保留：

1. `view_local`：目标在当前图可见，检验视觉接地后执行转身；
2. `episodic_memory`：目标只在早先视图出现、当前图不可见，检验视图记忆与转身程序的组合。

训练只使用单步原子；验证/测试加入两步、三步组合、逆变换和同净旋转的不同程序表述。基础事实按
`scene × current frame × entity × target frame` 去重；“右转 120°”与“两次右转 60°”是同一几何
事实的程序兄弟，不能重复计入规模。

### 2.2 Among-5（T2 程序化 MindCube 协议）

自然 T4/T1 经严格 preflight 得到 0 个 exact Among，因此不再承担主协议。新协议
`omnigibson_mindcube_among_layout.v2` 在 structure-only 房间中放置一个中心脚凳与四个类别唯一的
外围物体（椅子、背包、垃圾桶、盆栽）。四个相机位于中心与外围环之间，以 90° 步长同向绕行并始终
面向脚凳；每帧必须**恰好**出现“中心脚凳 + 一个外围物体”，每个外围物体恰有一个 witness view，
任何单帧都不能泄露完整布局。这与 MindCube Among 的关键机制一致：中央物体作为跨图锚点，四张
局部证据共同确定外围布局。

几何门检查五物体基数、类别唯一、相机环可行走、物体间隙、四方角误差 ≤5°、半径误差 ≤5 cm；
视觉门进一步要求中心物体四帧完整入镜（≥10k pixels、短边 ≥128 px）且目标外围物体完整入镜
（≥4096 pixels、短边 ≥96 px），任何贴边截断直接拒收。同一合格场景生成五个 family-locked 世界：
`identity / rotate90 / mirror / permute1 / radius185`。前四者改变指定空间因素，`radius185` 只改变尺度，
用于检验答案不变性。训练/评测只接收五个世界全部通过且拥有共同查询菜单的 family。

首版查询只保留 `cross_view_relation`：以第 1 图朝向为公共坐标系，比较两个从未同框的外围物体。
规则四方布局中的 `object_anchored_perspective` 暂不入集，因为通过 OBB-aware 方向 margin 的稳定查询
只剩“站在一个外围物体面向中心时，对面物体仍在前方”，答案分布退化为 100% `front`，模型可凭题型
直接猜中。该诊断被保留在编译器中，但不能计入训练规模；Self-Rotation 的心理换位由 T3 独立承担。

## 3. 数据阶段与产物边界

所有产物写入独立目录 `data/transform_pilot_v1/`，禁止改写 `data/epispace_pilot_v1/`。

1. `preflight/`：无 LLM 的几何 census；分别报告 unique scenes、base geometric facts、program/
   counterfactual siblings、language surfaces 和逐门存活率；
2. `dataset/teacher_records.jsonl`：模型接口、typed program、grounded trace 和隐藏证书的唯一母表；
3. `dataset/isolated_{answer_only,grounded_cot}_{split}.jsonl`：严格共享图像、问题、选项顺序、答案和
   family split 的配对 SFT；
4. `dataset/episode_{answer_only,grounded_cot}_{split}.jsonl`：一次写入四图、多次读取且共享上下文的
   episode 组织臂；`eval_inputs_test.jsonl` 与 `eval_oracle_test.jsonl` 物理分离；
5. `audit/`：自动 replay、人工复核包、双人分歧与裁决；
6. `evaluation/`：冻结模型 prompt、原始输出、解析结果、模型/权重/代码 hash。

训练包永不包含相机位姿、OBB、instance/depth、目标朝向 RGB 或 canonical world truth。网页 Audit View
可按需展开这些信息，但默认 Model View 只能展示真实消息和 assistant target。

## 4. Grounded-CoT 合同

每条 Grounded-CoT 答案恰含三个可解析的有序角色：

```text
<cue>指出哪张已见图像提供了目标/锚点线索。</cue>
<transform>执行转身、视角注册或参照系切换，不暴露坐标和内部 ID。</transform>
<answer>给出唯一选项及四方向结论。</answer>
```

隐藏 claim sheet 记录 cue view、可见像素、观测方位、历史动作、题设动作、查询 frame、方向 margin
和依赖节点。验证器检查拓扑顺序、数值和方向许可、目标朝向隔离、跨图不共视、证据覆盖及结论重放。
第一版文本由确定性语法编译，以消除 LLM 配音引入的事实漂移；LLM 只可在后续作为受保护槽位的
surface editor，不能生成事实或改写方向。

SFT 仅对 assistant token 计算损失。Grounded-CoT 的 span 权重先定为

\[
\mathcal L=0.25\mathcal L_{cue}+0.50\mathcal L_{transform}+0.25\mathcal L_{answer},
\]

先做 span mean，再做 fact mean 与 episode mean，避免长文本自动获得更高权重。Answer-only 与
Grounded-CoT 必须同初始化、同基础事实/图像、匹配 optimizer updates，并使用多个随机种子与
scene-cluster paired bootstrap。后续增加 transform-shuffled/mismatched 和 conclusion-only 诊断臂，
区分正确变换监督与额外 token 的收益。

## 5. 质量门和规模门

自动门先于语言生成：RGB 帧质量、实例可辨识性、类别唯一绑定、角度中心 margin、OBB 角宽后的有效
margin、目标朝向不泄漏、T3 位姿/角度闭合、Among 五物体四扇区、视图图连通和 split 继承。所有失败
保留 reason code。

人工复核：benchmark 100%，训练集分层抽检至少 10% 且不少于 100 facts，其中 50 facts 双人独立
复核；报告 Cohen's kappa、分歧和裁决。`TRAINING READY` 要求 major/unreviewable 为 0、minor 比例
不超过 5%，且两个任务分别达到 scene 与 base-fact 下限；counterfactual sibling 和 paraphrase 不能补足
下限。Among 要求至少 15 个 scene-disjoint、五变体全通过的 complete families；任何缺失变体、单帧
捷径或反事实语义不满足的 family 整组拒收，不得用自然 T1/T4 记录补足。

### 5.1 2026-07-22 首轮 preflight 结论

首轮严格 census 从父 release 绑定了 82 条 T3、10 条 T4 与 25 条 T1。Self-Rotation 在类别唯一、
RGB 可辨识、目标朝向隔离、中心角 margin 和 OBB-aware margin 后，按每 bundle 最多 24 条做平衡选择，
得到 **43 scenes / 1,739 base facts**（其中 episodic-memory 723，view-local 1,016）；这部分已达到
pilot 几何规模门，但尚未完成人工语义审计和语言编译，因此仍不是训练 release。

同一套严格协议在现有自然房间中得到 **0 个 Among-5 layout**。主要淘汰原因是候选五物体无法同时覆盖
首图坐标系的前/右/后/左四扇区，其次是中心锚点跨四图不可唯一绑定或视图注册图不连通。这证明现有
T4“围绕一个自然家具运动”不能直接冒充 MindCube Among。下一采集版本必须新增 `T2A procedural
Among-5`：在资产许可和碰撞约束下，把五个视觉类别唯一、尺度适中的可移动物体程序化放置为中心＋
四方布局；对同一物体集合采样 4 个正交观察点，并对旋转、置换、镜像、距离扰动生成 family-locked
反事实兄弟。每个世界都保存放置前碰撞检查、落地稳定性、四视图可辨识性及程序重放证书。自然 T4
只保留为域迁移测试，不再承担 Among 训练规模。

### 5.2 2026-07-22 程序化 Among 资格筛选

旗舰相机参数经两轮视觉复核后冻结为：外围半径 1.6 m、相机半径 1.0 m、65° FOV、1024² HQ RGB。
首轮 21 个 scene-disjoint identity 场景中 **18 个严格通过、3 个拒收**；拒收原因均为场景结构遮挡使
插入物体缺少可靠实例像素，未通过放宽阈值补救。18 个合格 identity 场景随后采集四个单因素反事实；
其中 2 个场景的 `radius185` 放大半径版本未过视觉门，因此整族拒收，最终得到 **16 个 complete
families（train 8 / validation 4 / test 4）**。数据编译器只在
同一场景五变体全部通过、且四个固定查询在所有变体中均有 ≥8° OBB-aware margin 时接收；因此网页
显示的规模是 complete-family 规模，而不是渲染任务数。

最终母表含 **2,961 records / 1,803 unique geometric facts**：Self-Rotation 2,641 条语言记录，Among
320 条反事实记录。主实验使用方向平衡、每个基础事实只保留一个表面的导出：train 724 / validation
240 / test 172。半径不变、90° 旋转、置换、镜像左右翻转/前后保持五项反事实审计均为 100%；编译时
另删除 32 个不再被合格记录引用的媒体文件。

## 6. 本机零训练评测

默认模型为完整的 `SenseNova-SI-1.1-InternVL3-8B` 与同架构 `InternVL3-8B` 基座。固定两类 prompt：
direct answer 与 visible grounded trace；固定三类证据条件：full RGB、no image、critical-frame ablation。
parser 和阈值仅在 dev 冻结，之后一次性运行 test。

指标包括最终答案、cue/transform/conclusion 分段准确率、三段联合准确率、family exact match、旋转角度/
组合深度曲线、Among 查询类型/视图数曲线、反事实等变一致性、图像增益和关键证据敏感度。零训练结果
不能作为训练方法有效性的证据。

### 6.1 SenseNova-SI-1.1-InternVL3-8B 零训练结果

在 172 条平衡 test 上，Answer-only prompt 为 **18.6%**：Self-Rotation 32.6%，Among 2.5%。强制
`cue → transform → answer` prompt 后答案准确率为 **21.5%**：Self-Rotation 30.4%，Among 11.3%；
方向文本语义准确率 22.1%，三段格式完整率 35.5%，而“答案正确且三段完整”的联合率仅 **8.7%**。
43 条同时输出选项字母和方向词的回答全部字面一致，因此主要失败不是 parser 或字母—文本冲突，而是
跨视图登记与自我朝向更新本身不稳定。该结果只建立 frozen baseline：仅在 prompt 中索要 CoT 不足以
得到可靠空间程序，不能替代后续配对 SFT 因果实验。

## 7. 独立前端

新建短页面 `web/transform-pilot.html`，只服务本 pilot：顶部显示几何事实规模和 readiness gate；中部按
Self-Rotation / Among-5 浏览 family；详情提供 Model View、Audit View、反事实兄弟 diff；底部展示冻结
SenseNova 与基座结果。默认不得加载 oracle target RGB；每条记录都显示“基础几何事实 / 程序兄弟 /
语言表面”三个独立计数，防止把 QA scaling 误写为世界知识 scaling。
