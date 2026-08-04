# Rs_int 几何真值驱动的多角色 LLM Episode 管线（V2）

## 目标

这条管线把 OmniGibson 的 `Rs_int_seed17` 真实轨迹编译成一条可做 Qwen3.5-8B 多轮 SFT 的 11 图、7 轮 episode。模型可见内容始终只有按顺序释放的 RGB、自然中文问题和自然中文回答；空间程序、canonical state、claim sheet、几何值和来源路径全部留在隐藏监督层。

关键不是“让 LLM 自己出题并自己回答”，而是把职责切成四层：

1. **确定性 truth compiler**：读取 `scene_ir.json`、`spatial_episode.json`、`trajectory_plan.json`、`relation_oracle.json`，执行 typed operation graph，生成 node-level claims；
2. **answer-blind question editor**：只看当前已释放视角、能力意图和保护槽，看不到方向、距离、答案或 claim；
3. **claim-grounded answer narrator**：只把 allow-list claims 组织成实例专属的“线索—必要变换—结论”，不重新求解几何；R4/R5 的三个句子角色由本地验证器强制执行；
4. **independent critic + local validators**：逐句检查 claim 绑定、方向、数值、视角编号、unknown 边界、内部术语和 required-claim 覆盖。任一门失败，整条 episode 不导出。

## 冻结的数据源

唯一允许的源 bundle 是：

```text
/data/shichao/data/dataV100/code/OminiGibson/outputs/Rs_int_seed17
```

不能按 `Rs_int` 名称自动寻找，也不能替换成 sweep 目录中的同名 bundle。两者的 episode、位姿、可见性和实体位置不同。产物的 `source.bundle_files_sha256` 和 `manifest.json` 固定源文件哈希，避免 silent drift。当前坐标契约采用 scene-world 平面：`+X` 为右，`+Y` 为前；不能与 `view-000` 锚定的另一套 demo 坐标混用。

## 七轮课程

| 轮 | 新增图 | 能力 | 主要程序 | 学习信号 |
|---|---:|---|---|---|
| 1 | 1 | SR/grounding | `G→V` | 视觉清点，不靠住宅常识补物体 |
| 2 | 2 | SR+MM | `G+G→F→B→M+R→V` | 先确认同框，再给左右和约 1.7 m |
| 3 | 2 | CR/temporal | `G(sequence)→B→V` | 当前不可见与历史见过分离 |
| 4 | 3 | CR+PT+SR+MM | `G+G+G→F*→B→R+M→V` | 电视/冰箱从未同框；用沙发的跨段重现和真实掉头事件连接前后观察 |
| 5 | 2 | PT+SR+CR | `G+G→B→F_query→P→R→V` | 冰箱与转椅/电视从未同框，必须读取此前状态并做物体锚定视角变换 |
| 6 | 1 | CR+PT+校准 | `G→B→V_unknown + F_loop→B→V` | “没看到床”不等于“住宅没有床”；首尾闭环验证冰箱持久性 |
| 7 | 0 | CR+SR+MM | `G*→F*→B_state→R*+M→V` | 撤掉新图，只从共享 state 提交布局总结 |

这条轨迹诚实覆盖 MM、SR、PT、CR，但**不覆盖 SenseNova MR**：资产没有可靠的 object canonical front 或受控旋转干预，不能把普通布局重建冒充 Mental Rotation。`mental_simulation` 在这里是 PT 的语言组织策略，不是 MR 标签。

## V2 的视角变换筛选

R5 不再把物体中心的正负号直接当作四象限标签。程序把目标 OBB 投影到以“转椅→电视”为正前方的查询坐标系，计算中心坐标 `(q_x,q_y)`、投影半长 `(e_x,e_y)` 和到两条象限边界的最小角度 margin：

\[
m_\theta=\min\left(\arctan\frac{|q_x|}{|q_y|},\arctan\frac{|q_y|}{|q_x|}\right).
\]

只有同时满足 `m_theta >= 20°`、`|q_x|-e_x>0`、`|q_y|-e_y>0` 时，才允许硬四象限答案。当前冰箱样本为 `(-3.956, 5.190)m`，最小角度 margin `37.32°`，两轴 clearance 为 `3.405m/4.640m`，因此“左前方”可用。旧沙发样本为 `(-2.344, 0.417)m`，角度 margin 仅 `10.08°`，且 OBB 跨越 front/back 分界，被保存在 `sampling_diagnostics` 中作为拒收边界案例，只允许“大致在左侧、略偏前”的降级语言，不再进入硬标签训练。

R4/R5 的 claim 另带 `reasoning_role={cue,transform,conclusion}`。回答必须严格输出 `evidence→transform→conclusion`，且每句至少引用一个相应角色的 claim；这样 narrator 获得的是可认证中间路标，而不只是终点答案。

## 多角色上下文

回答 narrator 不是统一 prompt。不同轮次注入不同冻结 skill card：`visual_inventory`、`co_visibility_then_metric`、`temporal_recall`、`route_replay`、`mental_simulation`、`calibration`、`landmark_hierarchy`。skill 只控制如何组织已经执行的证据，不授予新增事实的权限。question、answer、critic 是三次隔离调用；每次保存 provider、model、prompt profile、skill、request/response hash 和 cache record。

运行：

```bash
cd /data/shichao/data/dataV100/code/episode3D
make rsint-dialogue
```

把配置中的 `backend.kind` 从 `codex_exec` 改为 `replay`，可在无新模型调用时用内容寻址 cache 重放；任何 prompt、claim、schema、skill 或模型变化都会改变 request hash，旧结果不会误命中。

## 产物

```text
data/rsint_dialogue_pilot_v2/
├── dialogue.compiled.json
├── train.dialogue_episode_sft.jsonl
├── train.dialogue_isolated_sft.jsonl
├── train.dialogue_state_op_aux_sft.jsonl
├── llm_cache/
└── manifest.json

web/data/
├── rsint_dialogue_pilot.v2.json
└── rsint_dialogue_media/
```

- `dialogue.compiled.json`：完整审查 artifact，含 typed DAG、node claims、逐句绑定、三角色 provenance 和验证结果；
- `episode_sft`：一条真正的增量多轮样本，图像按 `1/2/2/3/2/1/0` 进入；
- `isolated_sft`：七条逐轮 prefix-matched 对照，图像证据和答案与 episode 对应轮严格相同，但删除历史 Q/A；
- `state_op_aux_sft`：独立监督可观察 state delta、program、node value 和 verify，不与自然回答混写；
- 网页：在 `#qa-generation` 顶部按真实释放时间线展示“本轮新增图→Q→A→claims/DAG/provenance”。

## SFT loss 契约

episode 作为一条完整 chat sequence 做一次 causal forward。system、user、图像 token 和 role token 的 label 为 `-100`；七个 assistant span 及各自 EOS 被监督。为了避免第7轮长总结压过短记忆回答，先对每轮 token 取均值，再对七轮取均值：

\[
\mathcal{L}_{\mathrm{episode}}
=\frac{1}{7}\sum_{r=1}^{7}
\frac{1}{|A_r|}\sum_{t\in A_r}
-\log p_\theta(y_t\mid y_{<t}, I_{\le r}).
\]

`assistant_span_contract` 对每个 assistant message 保存 `message_index`、`turn_id`、回答 SHA 和 `supervise_eos=true`。`episode3d.qwen_training.supervised_assistant_turns()` 会检查七轮是否被完整、按序覆盖；`assistant_token_groups()` 已能按对话顺序定位所有 assistant token 和每轮 EOS。训练代码不能再假定只有最后一个 assistant target。

## 拒收原则

- 视图前缀不单调、未来图提前进入、11 图未恰好释放一次；
- typed I/O 不合法，claim 的 `node_id` 不存在，几何值不能从 bundle 重算；
- question payload 出现 answer/claim/certificate/oracle，保护槽丢失或问题暗示答案；
- PT 硬四象限的最小角度 margin 小于 20°，或目标 OBB 跨越任一查询轴；
- PT 目标与站位/朝向锚在任一视角共视，导致单图捷径；
- answer 新增实体、房间、方向、数字或把局部二维关系偷换成全局关系；
- 组合题缺少 cue→transform→conclusion，或 sentence role 没有引用对应 reasoning role 的 claim；
- unknown 把“给定视角没看到”写成“场景不存在”；
- required claims 未逐句覆盖，数值/单位不在许可 surface 中；
- critic 不是 `accept`，或报告任一 unsupported sentence；
- episode / isolated 对应轮的 exposure hash 或 answer hash 不一致；
- state-op target 含 UUID、raw world pose、decision margin 或 source oracle path。

本产物仍标为 `development_only`：它证明自动化机制闭环，不代表一条场景已经足够做模型结论。下一步应在更多场景上自动规划同构与反事实 family，并做 episode-vs-isolated、state sufficiency 和 held-out operation composition 实验。
