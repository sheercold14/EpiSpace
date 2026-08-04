# Claim-grounded Episode QA 生成与审查管线

## 结论与边界

这条管线解决的不是“让语言模型从程序字段抄答案”，而是把模拟器提供的可执行空间事实，转成大模型真正能消费的“有序 RGB + 自然语言问题 → 简短自然语言回答”。几何程序负责事实正确性，Codex（未来可替换为 API 模型）只负责语言表达；问题生成与答案真值隔离，答案中的每句话又必须能回指几何 claim。

当前产物是一个 **development-only pilot**，不是正式训练数据 release：源语义视觉审计结果为 `fail`、release gate 为 `false`（48 个抽审项中 29 pass、12 minor、6 major、1 unreviewable）。管线已排除已知 major/unreviewable fact 及其污染 family，但这不能替代源数据整改和重新抽审。因此：

- 可以直接用于 dataloader、SFT loss 和推理链路的开发联调；
- 可以做 20-fact 的人工审查、格式 smoke test 和小规模过拟合实验；
- 不应据此宣称数据集已经 release、可用于正式论文主实验，或已经证明 episode learning 有增益；
- 当前 **MR（mental rotation）不支持**。资产没有可靠的 canonical object front，系统选择显式报缺口，而不是伪造物体正面/旋转监督。

权威状态以 [`data/qa_generation_pilot_v1/manifest.json`](../data/qa_generation_pilot_v1/manifest.json) 为准，不能只根据网页观感或 JSONL 文件存在就判断“可发布”。

## 从几何事实到模型样本

一条 QA 的闭环如下：

`Episode IR / geometry certificate → answer-blind blueprint → Codex/API question writer → protected-slot render → claim sheet → answer writer → deterministic validator → independent critic → SFT / heldout / web`

各阶段的权限边界如下。

| 阶段 | 输入与动作 | 关键约束 | 输出去向 |
|---|---|---|---|
| Episode IR / certificate | 从 [`episodes.ir.jsonl`](../data/epispace_pilot_v1/episodes.ir.jsonl) 读取有序视图、可观察状态、typed program 与可重放 certificate | 模拟器/确定性编译器是事实源；pose、depth、mask、entity ID 和 certificate 不进入主模型上下文 | 生成器内部 oracle |
| Answer-blind blueprint | 只保留任务意图、能力标签、允许的提问策略和受保护槽 | 结构检查禁止 `answer`、`claim`、`certificate`、`oracle`、`rationale` 等答案字段进入问题写手请求 | 问题写手输入 |
| Question writer | Codex/API 选择人类式策略并写自然中文模板 | 只能写 `{{subject}}` 等占位符，不能看答案、猜方向或内联受保护值 | 问题模板 |
| Protected-slot render | 本地程序检查每个 required slot 恰好出现一次，再填入实体、视角、关系前提等确定值 | 模型不能改实体绑定；残留槽、重复槽、未知槽或答案暗示直接拒收 | 最终自然语言问题 |
| Claim sheet | 从同一 IR/certificate 独立编译允许陈述的事实、证据视图/实体、source path、方向枚举、数值与容差、epistemic scope、answer key | claim sheet **不暴露给问题写手**；unknown 只能表示“给定观察不足”，不能偷换成“场景中不存在” | 答案写手与验证器输入 |
| Answer writer | 读取最终问题、选定策略和 claim sheet，把程序事实写成 1–3 个短句 | 先给结论；跨视图题可补一句实例相关的观察/变换；每句列出 claim IDs；不生成长文本 CoT | `surface_answer_zh` + 隐藏逐句绑定 |
| Deterministic validator | 本地重放 JSON schema、request ID、策略、answer key、required-claim coverage、方向、数值/单位、弃权范围、内部术语和语料去重 | 任一检查失败即 fail closed，不写入训练导出 | 机器验证报告 |
| Independent critic | 以独立 stage、独立请求逐句核对候选答案与 claim sheet | 当前 pilot 使用同一 Codex provider/model，但调用和上下文彼此独立；它不是“第二种模型”的过度声明，未来可注入独立 provider | critic 通过/拒绝 |
| Export | 按 program disposition 和 episode optimizer contract 分流 | heldout 永不进入 SFT；不足 4–6 个可训练问题的 batch 整批退出优化器 | episode SFT、isolated control、heldout 审查、网页 |

`claim_ids`、逐句绑定和 certificate 是审计信息，不是让模型在 SFT 时输出的隐藏 CoT。主 SFT 的 assistant target 只有编号后的短自然语言 `surface_answer_zh`。

## 20-fact pilot 的组成

配置 [`configs/qa_generation_pilot_v1.json`](../configs/qa_generation_pilot_v1.json) 固定了 4 条 train-split episode、20 个 fact；它是管线证明样例，不是覆盖面充分的 benchmark。按当前 manifest，分流结果为 13 个 train fact、4 个 composition-heldout fact、3 个 T1 batch-shortfall fact。

| 轨迹 | Episode ID | 选入 | Composition heldout | 进入 SFT | 解释 |
|---|---|---:|---:|---:|---|
| T1 | `c58ce8f9-ade1-5cea-8c8b-a750d22e7c8e` | 5 | 2 | 0 | 去掉 counterfactual 与 target-view 两个 heldout 后只剩 3 问，低于每个 episode 4–6 问的 contract；3 个非 heldout fact 标为 `development_only_batch_shortfall`，整批不优化 |
| T3 | `94ae917a-b0ff-51fa-949c-39452aba2248` | 6 | 1 | 5 | 满足问题数、至少三类能力、PT backbone 与 AUX 上限 |
| T4 | `a2f1fe74-da7e-546b-aa9d-b1a2f62f9904` | 5 | 1 | 4 | 满足 optimizer contract |
| T7 | `b1a433ea-e1c0-5284-bbb0-62362974f841` | 4 | 0 | 4 | 满足 optimizer contract |
| 合计 | 4 episodes | 20 | 4 | 13 | 另有 3 个 T1 shortfall；所有产物仍为 `release_eligible=false` |

四个 heldout fact 对应源配置冻结的 `counterfactual_cross_view.v1` 和 `target_view_prediction.v1` 组合签名，只保留在 `qa_records.jsonl`、claim sheet 和网页审查面，不进入两个训练 JSONL。13 个训练 fact 被导出为 3 条多问题 episode records，以及事实一一对应的 13 条 isolated records。

## 如何人工 review

人工审查应按“先看模型证据，再看问题，最后展开 oracle”的顺序，避免先看到答案后产生确认偏差。

1. 先打开 [`manifest.json`](../data/qa_generation_pilot_v1/manifest.json)，确认 `status=development_only`、`release_eligible=false`、源 audit 的 SHA/status/gate、后端 provider/model、request hash 与每个 batch 的 optimizer decision。
2. 在网页选择一条 episode，按顺序浏览全部 RGB；先不展开 claim，判断新问题是否仅依赖这些画面、实体指代是否唯一、视角编号是否对应、问题是否像真实对话而非程序字段朗读。
3. 对照 `SOURCE → NATURALIZED`：旧问法只作诊断参照；新问法不能改变任务、偷带答案、增加不可证实前提，也不能出现 canonical、OBB、claim、certificate 等内部术语。
4. 查看生成答案：简单定位/距离题应短；跨视图、视角转换和时序回溯题可有一句关键实例证据，但不应展开冗长 CoT。unknown 必须限定为“这些画面证据不足”。
5. 最后展开 claim/provenance，逐句核对 claim ID、证据视图、实体、方向、数值/单位和 source path。`question`、`answer`、`critic` 三组 validator 都必须为 pass；模型 critic 通过不能覆盖确定性 validator 失败。
6. 检查 disposition：`train_candidate` 只有在所属 batch 的 `optimizer_eligible=true` 时才可导出；`composition_heldout` 不得混入 SFT；T1 的三个 shortfall 不得被拆散后偷偷训练。
7. 抽查 [`qa_records.jsonl`](../data/qa_generation_pilot_v1/qa_records.jsonl) 与 [`claim_sheets.jsonl`](../data/qa_generation_pilot_v1/claim_sheets.jsonl)，确认网页不是脱离权威产物的手写演示。

网页启动方式：

```bash
python3 -m http.server 8770 --bind 0.0.0.0 \
  --directory /data/shichao/data/dataV100/code
```

打开 `http://SERVER:8770/episode3D/web/#qa-generation`。该导航展示 4 条 episode 的完整有序 RGB、旧/新 QA、能力与 disposition、optimizer 准入、claim/certificate 和生成 provenance。网页是审查投影，不是新的事实源；页面显示与 manifest 冲突时，以 manifest/JSONL 为准并停止使用。

## 构建与离线重放

在项目根目录 `/data/shichao/data/dataV100/code/episode3D` 执行：

```bash
../habitat/.conda/core/bin/python scripts/build_qa_generation_pilot.py \
  --config configs/qa_generation_pilot_v1.json
```

首次命中新的 request hash 时，`codex_exec` 会调用配置中的 Codex CLI；相同 prompt、payload、schema、provider 和 model 会复用内容寻址缓存。要验证“冻结输入 + 冻结模型输出 → 相同导出”，禁止任何在线调用并使用：

```bash
../habitat/.conda/core/bin/python scripts/build_qa_generation_pilot.py \
  --config configs/qa_generation_pilot_v1.json \
  --replay
```

`--replay` 缺缓存、hash 不匹配、缓存 JSON 损坏或 schema 不合法都会硬失败。它重放的是语言模型返回，不会绕过本地几何编译与 validators。

## 路径与产物

| 路径 | 作用 |
|---|---|
| `data/epispace_pilot_v1/episodes.ir.jsonl` | 源 Episode IR、program 与 certificate |
| `data/epispace_pilot_v1/semantic_visual_audit/result.json` | 源语义视觉审计；当前 `fail` |
| `configs/qa_generation_pilot_v1.json` | 20-fact 选择、heldout、后端、缓存和输出位置 |
| `data/qa_generation_cache_v1/` | 以完整请求 hash 寻址的结构化模型响应 |
| `data/qa_generation_pilot_v1/manifest.json` | 状态、源 SHA、统计、batch decisions 与 generation provenance |
| `data/qa_generation_pilot_v1/episode_plans.json` | 每条 episode 的完整曝光视图和选题计划 |
| `data/qa_generation_pilot_v1/qa_records.jsonl` | 20 条旧/新 QA、blueprint、claim、验证结果和 disposition |
| `data/qa_generation_pilot_v1/claim_sheets.jsonl` | 20 份独立可审计 claim sheet |
| `data/qa_generation_pilot_v1/train.episode_sft.jsonl` | 3 条开发态 observe-once/read-many records，共监督 13 facts |
| `data/qa_generation_pilot_v1/train.isolated_sft.jsonl` | 与上述事实匹配的 13 条单问题 control records |
| `web/data/qa_generation_pilot.v1.json` | 网页只读投影 |

## 训练与推理接口

Episode arm 的一条记录先输入一次完整有序 RGB 轨迹，再一次性提出同一场景的 4–5 个问题，assistant 输出编号短答案。Isolated arm 对同一批 13 个 fact 逐条提问；每条仍使用相同的完整轨迹和相同 surface format。两臂通过 `comparison_id`、fact IDs 和监督内容配对，目的不是“多造 13 个不同问题”，而是控制事实后比较共享状态训练与逐题训练。

SFT 只对 assistant tokens 计算交叉熵；system、user、图像占位和 prompt tokens 的 label 必须设为 `-100`：

\[
\mathcal{L}_{\mathrm{SFT}}
=-\frac{1}{|M|}\sum_{t\in M}\log p_\theta(y_t\mid \mathrm{RGB}_{1:m},x,y_{<t}),
\qquad M=\{t:\text{role}(t)=\mathrm{assistant}\}.
\]

不要把 `claim_sheet`、certificate、answer key、validator 报告或 hidden metadata 拼进 model input。Qwen 消息规范化、assistant-only mask 和 exact-fact loss 的实现见 [`qwen3vl_pilot.md`](qwen3vl_pilot.md)。Episode 是 3 次多事实记录、isolated 是 13 次单事实记录；原始 JSONL 只做事实配对，不自动等价于 optimizer-step、token 或 FLOP 匹配。做因果对照前必须使用明确的 group sampler/权重并报告图像与文本 token 预算，不能从这个 20-fact pilot 直接报告 episode-learning 增益。

推理时使用同样的 system + ordered RGB + user questions，但移除 assistant target；模型只需生成编号 surface answers。claim/certificate 可在推理后用于程序化评分和错误归因，不应在主推理上下文中泄漏。

## 接入未来 API 模型

生成器没有把 Codex CLI 写死在几何逻辑中。新后端实现 [`StructuredLLMBackend`](../episode3d/qa_generation/backends.py) 即可：提供 `provider`、`model`，并实现

```python
complete(*, stage, prompt, payload, output_schema) -> BackendResult
```

API 调用必须使用结构化 JSON 输出，随后通过本地 `parse_structured_output`/JSON schema 验证，并返回包含 `request_hash`、provider/model、cache 命中信息的 `BackendResult`。最安全的接法是不修改配置分派，直接注入：

```python
from episode3d.qa_generation.pipeline import build_qa_generation_pilot

manifest = build_qa_generation_pilot(
    "configs/qa_generation_pilot_v1.json",
    backend=my_api_backend,
)
```

API backend 不得改变三条安全边界：question stage 仍须 answer-blind；answer stage 只能看到 claim sheet 允许的事实；critic 必须是独立请求且所有本地 validators 继续执行。若需要更强的审稿隔离，可以让 critic 使用另一 provider/model，但必须把两者的 provenance 都写入 manifest，不能把“同模型独立调用”描述为“独立模型复核”。

## 升级为正式 release 的必要条件

当前文件“可被训练代码读取”不等于“数据已可正式训练发布”。至少还需：修复源 semantic RGB 的 6 major + 1 unreviewable 问题及污染 family；重新编译 IR；重新做独立语义视觉审计并使 gate 通过；扩大到 scene-disjoint 的能力/签名覆盖；为 MR 增加可信的 object-front 定义与验证资产；冻结 episode/isolated 的 compute-matched schedule；最后再重跑 QA 生成、replay、人工抽审和 release verifier。此前所有实验都应标记为 pipeline/development pilot。
