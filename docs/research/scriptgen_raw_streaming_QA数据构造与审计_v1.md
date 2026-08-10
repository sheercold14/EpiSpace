# Scriptgen Raw / Streaming QA 数据构造与审计 v1

配套的网页案例索引、测量效度分析与下一轮策略见 [Scriptgen QA 案例评审与生成策略复盘 v1](scriptgen_QA案例评审与生成策略复盘_v1.md)。

## 研究问题与边界

本轮把 `scriptgen_current_2x_v1` 中已经通过编译器校验的题目族，导出为两种可直接用于 VLM 训练和评测的数据：单条独立的 raw QA，以及共享同一轨迹历史、逐步释放图像的 streaming QA。所有金标都来自现有 family episode 或同一 `CapabilityCompiler` 对轨迹前缀的重新编译；SenseNova 只用于冻结诊断，不能生成、修改或否决金标。

输入源位于 OminiGibson 输出目录，生成过程只读。模型可见模态只有 RGB；传感器 NPZ、scene IR、轨迹计划和 certificate 仅用于编译、溯源及完整性检查。床存在性题按本轮范围排除。当前数据标记为 `development_pool`，尚未做正式 train/dev/test 切分，也没有进行 SFT 或 RL 训练，因此模型评测只能说明题目对当前冻结模型的诊断结果，不能证明训练会提升空间泛化。

## 数据规模

源集合包含 51 条已接受轨迹，其中 50 条轨迹产生本轮题目；覆盖 24 个场景、287 个题目族。最终导出：

- raw QA 1,394 条，分为 14 个不超过 100 条的分层交错批次；
- streaming QA 238 条、534 个 turn，分为 3 个批次；
- P1/P2/P3 的 raw 数量分别为 294/1,000/100；
- 生成时记录了 2,572 个依赖文件的大小和 SHA256，总快照约 2.51 GB，快照摘要为 `b93d66e12d9c0e142b0b24e3978cdd72930bfcb1467c81f37a8b87ef9b6be919`。

raw QA 保留题目族已有的五类证据变体：287 条 canonical、287 条 permute、246 条 drop_key、287 条 drop_filler 和 287 条 delay。`drop_key` 少于其他变体，是因为并非每个题目族都存在可删除且会改变充分性的关键帧；生成器没有补造变体。

## 两种 QA 的组织方式

raw QA 的一条记录对应一个 family episode。训练记录包含按顺序排列的 RGB、问题、选项、assistant 金标，以及 `assistant_only` causal loss 约定和 exact-label reward。评测另存 answer-hidden input 与 oracle；input 中没有 answer、certificate 或其他金标字段，二者用同一个 `input_sha256` 对齐。`scene`、`trajectory`、`family` 三层聚类键用于后续无泄漏切分，不能把同一家族的证据变体随机拆到训练集和测试集。

streaming QA 在一条记录内共享轨迹和对话历史，每轮只给出自上一轮以来的新 RGB。训练记录显式列出每个 assistant span，便于只对回答 token 计算 causal loss；RL 可按 turn 使用 exact-label reward。评测同时运行 teacher-forced history（历史中写入 oracle answer）和 free-running history（历史中保留模型自己的 answer），从而把当前轮空间判断错误与历史误差累积区分开。

P1 和 P2/P3 使用不同的流式构造机制：

- P1 以同一 question group 为单位，把首次有效标签以及后续标签变化排成多题对话。前缀至少包含两帧，因为单帧没有可观察的运动转移；即使几何约定能把单帧净转角定义为零，也不把它作为视觉自运动题。
- P2/P3 逐 family 构造 evidence reveal：先取首次可回答之前最后一个 `abstain` 前缀，再取首次 `answerable` 前缀，形成“证据不足→证据充分”的两轮记录。
- `invalid` 表示题目本身不满足几何或协议条件，不能伪装成“无法判断”，因此任何 `invalid` 前缀都不会进入数据。

最终 P1 产生 18 条多题轨迹；P2/P3 产生 220 条 evidence-reveal 记录，分别为 200/20 条。没有 streaming skip。

## 生成策略迭代

候选策略最初允许所有编译器可回答的 P1 前缀。实际检查发现 `path_integration_magnitude` 在 prefix=1 时可以按几何约定得到 `at_most_90`，但模型没有看到任何运动变化，这更像约定记忆而不是视觉空间推理。最终策略要求 P1 的 `prefix_length >= 2`。该调整只删除不合理的提问时机，没有改任何金标。

模型小样本诊断还暴露了答案解析的另一类问题：模型对二选一 closer 题可能输出具体实体名，例如 `fridge`，而协议要求输出 `first` 或 `second`。若问题文本能唯一确定实体对应的选项，这在语义准确率中计为正确，同时在 strict-format 指标中计为不合规。语义正确性与输出格式必须分开报告，否则会把格式错误误判为空间推理错误。该解析改动通过 prompt/version hash 形成 v2 评测，v1 中间结果不进入最终统计。

发布前的 oracle 隔离复核还发现，streaming `input_sha256` 的早期实现会纳入 `status`、`certificate_sha256` 和 assistant span 的答案摘要。answer-hidden 输入虽没有直接暴露这些字段，但输入身份仍会间接依赖金标。最终实现把这些 oracle 派生字段全部排除，并用“只改变答案、status 和 certificate 时输入 hash 必须不变”的回归测试固定该约束。问题、图像、turn 顺序和金标均未因此改变。

## 冻结模型评测协议

诊断模型为本地 `SenseNova-SI-1.5-InternVL3-8B`，使用 greedy decoding。raw QA 对每条记录运行 multimodal 与 vision-free 两个条件；streaming QA 运行“多模态/无视觉 × teacher-forced/free-running”四个条件。vision-free 条件保留问题和答案空间但隐藏图像，用于估计标签、问题措辞或场景先验造成的捷径。报告同时保留语义准确率、语义可解析率和 strict-format 合规率，并按 capability、P1/P2/P3、role、证据变体、family 和 trajectory 聚合。

raw 全量 1,394 条在两个条件下均无运行错误。多模态语义准确率为 26.3%，无视觉为 24.5%；多模态语义可解析率为 96.5%，但 strict-format 合规率只有 57.2%，无视觉分别为 100.0% 和 80.6%。按层级看，多模态/无视觉准确率为：P1 44.6%/34.7%，P2 18.8%/18.0%，P3 47.0%/60.0%。因此，1.7 个百分点的总体视觉增益不能代表各层级：P1 有正向视觉贡献，P2 基本没有，P3 的无视觉结果反而更高，后者需要作为问题文本、答案分布或模型视觉干扰的捷径风险审查，不能解释成跨视图能力已经达到 60%。

按证据变体，多模态 canonical、delay、drop_filler、permute 和 drop_key 的准确率分别为 33.4%、34.8%、33.4%、25.8% 和 0.0%。drop_key 的金标是关键证据删除后的弃答，两个模型条件均为 0%，说明当前模型没有在该干预下学会“证据不足时拒答”。287 个 family 的 all-variant correctness 为 0%，也说明单条题目偶然答对与对证据干预保持一致是两种不同要求。

streaming 的 238 条记录、534 个 turn 在四个条件下同样无运行错误。多模态 teacher-forced 的 turn 语义准确率为 19.5%，最终 turn 为 28.2%，语义可解析率为 97.4%，strict-format 为 69.1%；多模态 free-running 分别为 18.5%、27.3%、95.9% 和 41.2%。无视觉 teacher-forced/free-running 的 turn 准确率为 17.2%/17.6%，最终 turn 均为 28.2%。多模态相对无视觉只有小幅 turn 增益；teacher-forced 到 free-running 的主要退化体现在格式合规率，而不是同等幅度的语义准确率下降。

四个条件的 all-turn correctness 和 P2/P3 evidence-update success 均为 0%。这两个指标要求同一记录的所有轮次都正确，包括先正确弃答、再在证据到达后给出空间答案，因此比单 turn 准确率严格得多。能力分解显示净转向 `path_integration` 的多模态 teacher-forced 准确率为 83.3%，无视觉为 44.4%；历史画面侧 `view_side_check` 为 81.3%/31.3%。相对地，多数带 yaw 的 P2 参照系变换仍接近零或低准确率。该模型能在部分 P1 turn 中使用视觉，但没有完成整条流式证据状态更新，不能用局部高分代替 multi-turn 成功。

最终 case 分类为 106 条 `eligible`、326 条 `shortcut_risk` 和 1,200 条 `valid_hard`；没有 `protocol_failure`，quarantine 为空。238 条 streaming 记录在严格的全 turn 口径下均为 `valid_hard`，这表示当前冻结模型失败，不表示 compiler 证据无效。14 个 raw 批次中有两个因无视觉恢复金标的比例超过 35% 而标记为 `review`，其余批次保持冻结策略。源完整性复核覆盖 2,572 个依赖文件，未发现变化。

## 每百条反思与 case 分类

14 个 raw 批次和 3 个 streaming 批次各自生成结构化 JSON，并汇总为 `research_report.md`。case 分为四类：

- `eligible`：视觉条件正确、无视觉条件不正确，当前冻结模型表现支持该题依赖视觉；
- `valid_hard`：编译器证据有效，但模型答错、输出无法语义解析，或 free-running 历史出现误差累积；这类是有效难例，不因模型失败而改金标；
- `shortcut_risk`：无视觉条件也恢复金标，需要检查标签分布和问题文本捷径，但不等于数据一定无效；
- `protocol_failure`：缺少必需评测条件或发生模型运行错误，进入 quarantine，不能混入能力结论。

每个批次同时记录题型、层级、变体和场景覆盖，判断当前样本是否实际测到声明的空间程序，并演绎其作为训练监督可能支持的泛化机制。这里的“可能支持”只基于数据干预结构：同一几何事实具有完整、删关键证据、删冗余证据、延迟和安全乱序对照，streaming 又提供证据到达前后的弃答/更新监督。没有训练前后及 held-out composition 实验时，不把这种机制判断写成已实现的训练收益。

## 产物与复现

生成入口是 `scripts/build_scriptgen_qa.py`，支持 `generate`、`evaluate`、`report` 和 `all`。默认发布目录为 `outputs/scriptgen_qa_v1/`，主要文件包括：

- `raw_qa.jsonl`、`streaming_qa.jsonl`：训练记录；
- `raw_eval_inputs.jsonl`、`streaming_eval_inputs.jsonl`：不含答案的 benchmark 输入；
- `raw_eval_oracle.jsonl`、`streaming_eval_oracle.jsonl`：独立 oracle 与 certificate；
- `sensenova_predictions.jsonl`、`sensenova_summary.json`：冻结模型逐条输出和汇总；
- `batch_reflections/*.json`、`research_report.md`、`quarantine.jsonl`：逐批审计；
- `review.json`、`index.html`：可筛选查看图像、问题、金标、模型回答和流式时间线的前端。

正式扩量前应先按 `scene/trajectory/family` 聚类确定切分，再冻结切分和生成策略；当前 development pool 不应直接当作无泄漏 benchmark 发布。
