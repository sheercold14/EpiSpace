# EpiSpace 识别门禁前校准失败记录（旧版，非正式审计）

> **归档边界：** 本文件绑定的是已淘汰的旧候选，且当时没有保存可验证的
> human reviewer provenance。文中的“人工”是旧记录用语，不得作为“独立人工审计”
> 或当前 formal release 的证据。当前权威证据只来自正式 release 内
> `semantic_visual_audit/` 的内容寻址 packet、独立 reviewer 文件与裁决结果。

## 结论

这次旧版校准针对 `epispace-pilot-v1-2026-07-16` 的**实际模型输入与 RGB 可解析性**。20 个确定性分层样本中，15 个通过，5 个判为 major issue；该比例只描述定向样本，不能估计总体错误率。5 个 major 均不是隐藏坐标或关系计算错误，而是“实例 mask/几何有效，但 RGB 中自然语言所指物体不能被可靠识别”的视觉接口问题。

因此，原 frozen release 可以作为可复现的旧版研究快照，但不应作为未经隔离的 gold release 直接训练或发布。

机器可读逐条记录见 [`epispace_pilot_v1_stratified_semantic_audit.json`](./epispace_pilot_v1_stratified_semantic_audit.json)。JSON 中保存了全部 20 个 fact、实际 prompt record、episode/scene/source bundle、split、轨迹类别、program、variant、问题类型、人工裁决、major 原因、泄漏裁决、42 张实际查看的 release RGB 路径，以及 4 张 source preview 复核图路径。

## 被审计的冻结版本

| 资产 | SHA-256 |
|---|---|
| `release_manifest.json` | `d18b89cf559a0633b7d13ebdb915dad5138f89ff8cf6d83c5c090c896cc6d4bb` |
| `episodes.ir.jsonl` | `3d0bb50827c7e0bec702318391e829725846852f53b75dc555d0a482a9b2d77a` |
| `train.episode_sft.jsonl` | `365ae60ef54293e2157f428eff018a16928b8240b0ff891cb3bca4759f8729ec` |
| `benchmark.jsonl` | `dc3a6aeb2f96195a61fe9586e986634dd55f0aa717911a37e52ccc40eb2b8e5d` |

审计源目录为：

`/data/shichao/data/dataV100/code/episode3D/data/epispace_pilot_v1`

## 方法

### 1. 确定性分层抽样

先固定 `split × trajectory class × program/family` strata。普通 stratum 选择 `sha256(fact_id)` 最小的样本；一致性 family 先选择 `sha256(consistency_group)` 最小的组，再审计该组全部 siblings。这个规则使样本可以从冻结 release 复现，也避免看到画面后再挑“好例子”或“坏例子”。

覆盖范围：

- 20 个 question，来自 14 个 source episode；
- train / val / test = 8 / 6 / 6；
- T1 / T3 / T4 / T7 / T8 = 5 / 7 / 2 / 2 / 4；
- 13 种 program；
- 8 种 variant 全覆盖：`canonical`、`claim_false`、`claim_true`、`prefix_unknown`、`revealed`、`decisive_deleted`、`frame_a`、`frame_b`。

这是有意覆盖危险面的分层审计，**不是对全体 1021 个可导出 fact 的随机无偏抽样**，所以 75% 不能直接解释为总体准确率。

### 2. 旧版语义裁决（reviewer provenance 未绑定）

每条样本核对以下内容：

1. 模型实际收到的 RGB 和自然语言问题；
2. 问题中的每个 referent 是否能凭 RGB 识别，而不是只能凭 instance ID 知道；
3. frame、左右/前后、跨视图身份绑定是否在人类语义上足够明确；
4. 期望答案是否同时符合 RGB 证据和 certificate；
5. family 干预是否真的改变了决定性视觉证据。

实际查看 42 张 release 中的原始 RGB；对 3 个 major source 又查看了 4 张原始 bundle preview 作路径级复核。`Pass` 表示模型在既定输入接口下有足够视觉证据；`Major` 表示目标监督可能把不可解析的视觉输入当成确定标签，即使隐藏几何本身正确。

### 3. 输入泄漏检查

泄漏检查只序列化真正进入模型的输入 surface：

- train：`train.episode_sft.jsonl` 对应记录的 `messages[0:2]`；
- val/test：`benchmark.jsonl` 对应记录的 `model_input`。

逐条扫描 fact ID、episode ID、scene ID、program ID、evidence entity ID、source bundle 和 certificate 中的绝对 oracle 路径。20/20 均无命中；T10 的 held-out oracle RGB 路径及哈希也未进入输入。

需要保留一个发布边界：SFT 原始 JSON 行顶层含 `hidden_meta`，benchmark 原始 JSON 行顶层含 `program/certificate`。当前 loader 只读取 `messages/model_input`，所以本次没有实际泄漏；第三方若错误地整体序列化 JSON 行则会泄漏。

## 逐条审计账本

训练样本的 `record_id` 指本次实际审计的 episode-SFT prompt；对应 isolated-SFT 导出 ID 保存在 JSON 的 `alternate_export_record_ids`。val/test 的 `record_id` 指 benchmark prompt。

| # | fact / record | source / split / class | program / variant | 裁决 |
|---:|---|---|---|---|
| 1 | `fact-3aa80408612049833aa9`<br>`benchmark-c26e83577ceeeffe1bf4` | `Pomaria_0_int_seed17` / val / T1 | `counterfactual_cross_view.v1` / `claim_false` | **Major**：餐桌只在左缘露出 7×98 px 的窄条，类别不可可靠识别。 |
| 2 | `fact-f4ce008e2ad0ae7e8468`<br>`benchmark-9cfc607a14aa457d77ee` | `Pomaria_0_int_seed17` / val / T1 | `counterfactual_cross_view.v1` / `claim_true` | **Major**：与 #1 共享同一 referent 缺陷。 |
| 3 | `fact-6a946dc342030a898962`<br>`benchmark-7519087b0c25a0886b67` | `Beechwood_0_int_t3_seed23` / test / T3 | `evidence_presence_unknown.v1` / `decisive_deleted` | Pass：两图均无钢琴证据。 |
| 4 | `fact-0cbdcbb51f96b03b5257`<br>`benchmark-69ae9c82331ad30e8ecb` | `Beechwood_0_int_t3_seed23` / test / T3 | `evidence_presence_unknown.v1` / `prefix_unknown` | Pass：两图均无钢琴证据。 |
| 5 | `fact-1ed5f14a49a3c0d79098`<br>`benchmark-6e5f524873146c494ee6` | `Beechwood_0_int_t3_seed23` / test / T3 | `evidence_presence_reveal.v1` / `revealed` | Pass：替换后的第二图中钢琴清晰，干预成立。 |
| 6 | `fact-a9e030ec079543aebcd8`<br>`benchmark-722ff9c8cd2fdee9b4c9` | `Beechwood_1_int_t3_seed23` / val / T3 | `egocentric_relation.v1` / `frame_a` | Pass：镜子、床和第1视角 frame 可解析。 |
| 7 | `fact-46ef91a8046ce9e15d54`<br>`benchmark-73b5cea0db49dfbd5c14` | `Beechwood_1_int_t3_seed23` / val / T3 | `egocentric_relation.v1` / `frame_b` | Pass：同图换成第2视角 frame 后答案正确变换。 |
| 8 | `fact-9670b171d6c1cda473a6`<br>`record-40c3ff50804a23955801` | `grocery_store_cafe_t8_seed17` / train / T8 | `occlusion_unknown.v1` / `decisive_deleted` | Pass：输入中无包装箱证据；但应随失败 sibling 整组隔离。 |
| 9 | `fact-7e6c0cc5204e12f2c78c`<br>`record-98eb52354fdbf3bd9cf4` | `grocery_store_cafe_t8_seed17` / train / T8 | `occlusion_unknown.v1` / `prefix_unknown` | Pass：输入中无包装箱证据；但应随失败 sibling 整组隔离。 |
| 10 | `fact-b28a53b15cd7d588dc61`<br>`record-08f90e8bacf9bb7dfb2b` | `grocery_store_cafe_t8_seed17` / train / T8 | `occlusion_reveal.v1` / `revealed` | **Major**：决定性图中的白色包装箱底部严重裁切，类别不可可靠识别。 |
| 11 | `fact-b3ac72257744da0e4b5d`<br>`record-772cd1e8a4a52711fa8b` | `Benevolence_2_int_seed17` / train / T1 | `metric_distance.v1` / `canonical` | Pass：门、可开启窗清楚，2.1 m 与 certificate 一致。 |
| 12 | `fact-f5cc495385a9a71ea671`<br>`benchmark-a4f8246607d42703f35c` | `Beechwood_0_int_seed17` / test / T1 | `target_view_prediction.v1` / `canonical` | Pass：write views 支持绑定，oracle render 支持答案且未泄漏。 |
| 13 | `fact-6dfb2e454f7b15637196`<br>`record-2689433162b9192cc656` | `Merom_0_int_seed17` / train / T1 | `last_seen_memory.v1` / `canonical` | Pass：垃圾桶最后见于 view-009，即自然语言第10视角。 |
| 14 | `fact-65a398e0cb5c377b853e`<br>`record-3b95cb0a153bd95eb893` | `Benevolence_2_int_t3_seed17` / train / T3 | `rotation_change_detection.v1` / `canonical` | Pass：旋转后床新进入视野。 |
| 15 | `fact-dea777838888764b7877`<br>`benchmark-946339f2050d455d2926` | `hall_conference_large_t3_seed17` / test / T3 | `cross_view_register_relation.v1` / `canonical` | Pass：两根焦点立柱可区分，canonical 关系成立。 |
| 16 | `fact-ae9e47410179f06a107d`<br>`record-d72ff413099e83d14e07` | `Ihlen_1_int_t4_seed17` / train / T4 | `orbit_identity.v1` / `canonical` | Pass：约180°两侧均能绑定同一餐桌。 |
| 17 | `fact-8a2f452337c59c4edb21`<br>`benchmark-3a81b9d7e08b6a666c5e` | `Ihlen_0_int_t4_seed17` / val / T4 | `metric_distance.v1` / `canonical` | Pass：后续干净视角支持盆栽绑定，2.0 m 合理。 |
| 18 | `fact-c0941b8a4b0f236941f7`<br>`record-200d58d12cb3166e899c` | `office_bike_t7_seed17` / train / T7 | `elevation_relation_transfer.v1` / `canonical` | Pass：立柱、自动售货机及跨高度左侧关系可解析。 |
| 19 | `fact-29c2c9d199d6d08dad6e`<br>`benchmark-f923a6d0a3117db074bf` | `restaurant_brunch_t7_seed17` / test / T7 | `egocentric_relation.v1` / `canonical` | **Major**：超低视角遮挡严重，且 right/front dominance 仅 1.162，“右侧”与“右前方”有争议。 |
| 20 | `fact-34210bc36cf23f96196a`<br>`benchmark-57f2dd9a0a7aa6d65b70` | `Wainscott_0_int_t8_seed17` / val / T8 | `cross_view_register_relation.v1` / `canonical` | **Major**：镜子很小且见背面，餐桌锚点裁切；两端自然语言指称均不稳。 |

## 汇总

| 维度 | Pass | Major | 合计 |
|---|---:|---:|---:|
| T1 | 3 | 2 | 5 |
| T3 | 7 | 0 | 7 |
| T4 | 2 | 0 | 2 |
| T7 | 1 | 1 | 2 |
| T8 | 2 | 2 | 4 |
| **总计** | **15** | **5** | **20** |

5 个 major 来自 4 个独立 episode source，覆盖 train、val 和 test，说明问题不是单一训练样本噪声。共同根因是现有门主要证明“目标实例在 mask 中存在、几何关系可计算”，没有证明“模型在 RGB 中能按问题所用类别识别它”。例如，`≥500 visible pixels` 会放过 7×98 px 的桌子窄条，也会放过贴边且白底白物的包装箱。

## 对 release 的处置建议

1. 对 claim、evidence-reveal 等一致性 family，任一决定性 sibling 视觉失效就整组隔离。
2. 增加 referent recognizability gate：短边、边界裁切、最大连通分量、mask 填充率、anchor/best-view 比例，以及类别—外观一致性。
3. gate 必须 task-aware：reveal/decisive 任务要求决定性视图自身合格；普通跨视图绑定任务可允许其他模型可见视图帮助建立身份。
4. 方向标签增加 quadrant/dominant-axis 歧义门；本样本支持把 dominance ratio 低于约 1.25 的关系送人工复核，而不是强制单轴答案。
5. 机器门只能生成 review queue，不能替代人或独立 VLM 的类别可识别性审计。原始单阈值会误杀长而细、空心或可由其他视图恢复身份的合法对象。

## 可复现校验

固化后执行了以下只读校验：

- JSON 解析成功；
- 20 个 audit index 连续，20 个 fact ID 和 20 个 prompt record ID 均唯一；
- 20/20 fact 能在 `episodes.ir.jsonl` 找到；
- episode、scene、source bundle、split、trajectory class、program、variant、question type、问题和答案均与冻结 IR 完全一致；
- 20/20 record 能映射到相应 benchmark 或 episode-SFT prompt；
- 42 张 release RGB 和 4 张 preview 复核图全部存在；
- verdict 汇总为 15 pass / 5 major；
- 输入泄漏命中为 0；
- 四个冻结源文件的 SHA-256 与报告记录完全一致。
