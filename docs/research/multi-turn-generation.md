# Multi-turn Episodic Dialogue：构造与训练笔记 v1

> 状态：设计讨论总结；配套实现 `scripts/build_training_samples.py`（`episodic_dialogue` 类型），
> 训练配比见 `训练流程_v1.md`，研究契约见 `空间组合泛化Episode数据设计_v1.md`。
> 核心主张：多轮图文交错对话是 episode learning 最 MLLM-native 的形态——
> 场景状态不输出为 JSON，而是隐式存活在对话上下文（KV cache）里。

## 1. 为什么用多轮对话承载 episode

MLLM 没有外挂记忆模块，它维护场景状态的唯一天然机制是自回归上下文。因此
write phase（建态）不必强迫模型输出显式 state token，而是把观察序列变成对话历史；
read phase（读取）变成穿插其间的问题。模型要答对，只能在内部维护一张"平面图"。

"拼到同一张平面图"的含义（跨视图问题的本质）：

- 单张图像的描述天然是相机中心的（"电视在我前方3米"），两个视角各有各的"前方"，
  无法直接比较；非共视实体对（电视/冰箱从未同框）在任何单图内不存在答案。
- 解法分两步，对应契约的两个算子：
  - **F\*（跨视图注册）**：利用连续帧视觉重叠与 ego-motion 链式推出相机位姿链，
    把每帧的相机中心观察翻译进一个全局参照系（俯视平面图，+X 右 / +Y 前）。
  - **B（写入共享 belief）**：把翻译后的实体位置登记进一张持久、增量更新、
    与问题无关的地图（共享 = 服务所有后续 read；写入 = 独立于当前画面持续存在）。
- 之后 R（关系读取）只是两行表项相减。isolated-QA 训练学不到这条链路，
  遇到非共视对只能靠语言先验猜——这正是 SenseNova-SI vision-free 诊断暴露的短板。

## 2. 数据构造：程序选事实，模板做语言，LLM 只做表面（且问答分离）

当前 demo（Rs_int_seed17，41 条中的 3 条 episodic_dialogue）为**纯程序确定性编译**，
运行时无任何 LLM 参与：

1. **剧本由可见性时间线驱动，不是随意编排**：
   - 冰箱最后见于 view-002 → "最后一次见到冰箱"的记忆题安排在 view-003/004 之后；
   - 电视与冰箱共视集合为空（代码断言，前提破坏即拒绝生成）→ 跨视图题必须等
     两段观察都进入上下文后才问；
   - 床存在于 world truth 但从未被观察 → 认知诚实题（必须弃答，抵抗"住宅必有床"先验）。
2. **答案 = 模板 + 几何值填空**：方向/距离/象限全部由 scene_ir OBB 真值计算
   （`relation()/ego_quadrant()/distance()`），margin < 0.4m 不出题。
3. **certificate 与文本同源**：`meta.facts` 记录每句回答背后的可执行事实，
   `tests/test_training_samples.py` 从 bundle 几何独立复算。

规模化时引入 LLM 改写，安全边界必须保持：

- **LLM 只改写问题，永远看不到答案**——改写代理拿到实体名、问题骨架、frame 约定，
  拿不到方向/数值；否则措辞会隐性泄漏答案或让难度与答案相关（最隐蔽的数据污染）。
- 对话级分工：**剧本规划器是程序**（给任意新场景按可见性时间线自动选题：刚出现问
  grounding、离开视野问记忆、非共视对都观察后问跨视图、从未出现问弃答），
  **LLM 只做"配音"**（问题口语化、答案润色），润色后数值/方向逐字段回验，改错即弃。
- 刻意避开"用模型自己直接改写 QA 对"：那等于让空间能力不足的模型给自己出题，
  错误固化进训练集且无 certificate 可验。
- 现有样本标 `paraphrase_pool: true` 作为改写接口；改写后需重新校验答案分布均衡。

## 3. 旗舰样本结构（ep3d-rsint17-dialogue-canonical，11 图 7 轮）

| 轮 | 训练目标 | certificate（测试自动复算） |
|---|---|---|
| 1 | 视觉接地（visible_set 列举） | 与 view-000 可见实体集比对 |
| 2 | 共视关系 | oven left_of fridge，Δ1.724m，margin 0.806m |
| 3 | 记忆回溯（状态在上下文里） | last_seen=view-002，且 view-003/004 确无冰箱 |
| 4 | 跨视图整合（灵魂轮） | co_visible=[]；tv behind fridge，Δ4.658m |
| 5 | 视角采择 | ego_quadrant front-left，query_xy=(-2.344,0.417) |
| 6 | 认知诚实 + 闭环持久 | never_observed(bed)=unknown；fridge@view-010 可见 |
| 7 | 自然语言 state commit（布局总结） | 叠放/左右/居间/距离共 8 条 OBB 事实 |

同 family 两个兄弟变体形成一致性压力（整族锁同一 split）：

- `delayed_kitchen_reveal`：客厅先看 → 跨视图题先答"无法确定"，厨房揭示后修正为同一答案；
- `decisive_views_deleted`：只留 5 个客厅视图 → 同一问题必须永久弃答，
  但客厅内可答题正常回答（防弃答过度泛化）。

答案必须随证据变化，语言捷径无处可藏——这是训练时消灭捷径，
区别于 SenseNova-SI 只在评测时测量捷径（vision-free / circular test）。

## 4. 训练方式：teacher forcing + gpt-turn label mask

多轮 SFT 一律 teacher forcing，无例外；区别只在 loss mask 盖在哪：

- 整段对话用 chat template 拼成**一条序列，一次前向**；causal attention 保证第 k 轮
  回答以数据中的真实历史（前 k-1 轮）为条件，一次前向同时监督所有轮。
- **mask 是 label mask 不是 attention mask**：user 文本、图像 token、role 头 → label=-100；
  每个 assistant span 从首 token 监督到其 `<eos>`（结束符必须监督）。attention 保持完整
  因果下三角——assistant 必须能 attend 到全部历史，否则记忆题无法学。
- 本仓库 JSONL 的 `loss_policy: gpt_turns_only` 即此指令（对应 LLaMA-Factory
  `train_on_prompt=False`，ms-swift / InternVL 官方脚本同理）。
- 变体：全轮监督（默认，我们的每轮答案均为真值生成）vs 只监督末轮
  （`mask_history`，用于中间轮质量可疑的蒸馏数据，代价是前缀重复前向）。
- 细节坑：默认按被监督 token 求平均 → 长回答（布局总结轮）压过短回答（记忆轮）；
  在意则 per-turn 归一化或控制各轮答案长度分布。
- **exposure bias**：teacher forcing 下模型永远以正确历史为条件，推理时却在自己
  生成的历史上滚雪球，多轮空间对话尤其敏感。解法不是改 mask，而是阶段二：
  RLVR/GRPO（certificate 判分，模型首次在自己的 rollout 上被训练）、
  拒绝采样自蒸馏、DPO。对应"SFT 注入行为、RLVR 提纯正确性"的两阶段设计。

## 5. 空间 DAG 的角色：纯后台

- **模型可见文本中无 DAG**。grounded_cot 的推理文本是 DAG 执行顺序的自然语言影子
  （G 定位 → F* 注册 → B 写入 → R 读取），决定"说什么、按什么顺序说"，
  不决定"用什么格式说"。
- **meta 层只存扁平的节点执行值**（relation / ego_quadrant / co_visibility ...），
  未存完整 operation_graph 与 program_signature。
- **已知缺口**：graph-signature holdout（组合泛化主评测轴）需要每条样本的程序签名
  才能做 split；当前训练样本缺该标签。修法：在 meta 加 `program_signature` 字段
  （如跨视图 `G+G->F*->B->R->V`、视角采择 `B->F_query->R->V`），纯元数据不进模型输入。
- 下一步增强（阶段二后）：用 DAG 中间节点值核对模型自由推理中的中间断言，
  做 step-level 过程奖励——彻底关死"文本 CoT 不可验证"的痛点。

## 6. 待办

1. [ ] meta 补 `program_signature` 标签（split 工具按签名分层）。
2. [ ] 剧本规划器通用化：从手工编排 7 轮 → 按可见性时间线对任意场景自动选题。
3. [ ] 问题-答案分离的 LLM 改写管线 + 改写后分布/泄漏回验。
4. [ ] per-turn loss 归一化实验（长总结轮 vs 短记忆轮）。
