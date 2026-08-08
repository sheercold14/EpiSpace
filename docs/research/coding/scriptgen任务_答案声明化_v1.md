# 任务书:答案声明化 + 首批新题型(answerdecl)v1

> 交给谁:任何接手实现的人或工具(Codex 等)。
> 前置阅读:`docs/research/scriptgen主线上下文交接_v1.md`(仓库规矩)、
> `docs/research/scriptgen能力框架_v1.md`(能力框架与题型表,本任务
> 实现其中题型 1/2/3/4 的编译基座)。
> 分支:`feature/scriptgen-engine-v1` 之上继续。做完不要 push,等人工审。

## 1. 背景与目标

编译器目前把"标准答案怎么算"写死成一套流程:取最后一帧位姿 → 算目标
方位角 → 离散成四扇区。这只认识自运动方向题一种。框架文档规划了十类
题型,每类的答案算法都不同(不同的帧、不同的量、不同的离散)。

本任务把答案算法做成**声明**:编译器建"答案模式登记处"(和谓词库同一
模式),剧本 spec 里声明"用哪个模式、什么参数",编译器照声明执行。
然后用三个新题型验证这个基座,全部骑现有渲染数据(batch_wallfix),
不需要新渲染、不需要 Isaac:

- **净转向题**(能力框架的基元C·路径积分):"这段路你净转向是左还是右?"
- **指回起点题**(组合一的探针,积分+变换、免读出):"出发点现在在你哪个方向?"
- **画面侧题**(操纵检查):"最后看到{target}时,它在你画面的左半边还是右半边?"

完成标准一句话:一条命令对 batch_wallfix/render_0 产出**一组**家族
(方向题 + 上述新题型各一,合格者),每个家族的金标都由声明驱动的同一
编译器产生;自运动方向题的金标值与现状完全一致(left/left/right、
方位角数值不变);全部测试绿。

## 2. 仓库纪律(与上次任务书相同,违反即返工)

1. 新代码只进 `src/spatial_episode/`;`src/episode3d/`、`scripts/pilot/` 一行不碰;
2. 判定走谓词库(判定+见证),阈值进 standards.py 并升版本;
3. 变体金标由重跑同一编译器产生,不许人工指定;不许为过测试手改金标;
4. 每个逻辑单元一个提交,提交信息说清做了什么、为什么;
5. 单元测试用假后端,集成测试吃真渲染数据、数据缺失时干净 skip;
   不写计时断言。

## 3. 预定裁决(必须照办,不要自行发明)

**裁决甲:答案模式登记处,镜像谓词库的模式。** 新模块
`src/spatial_episode/scriptgen/answers.py`:

- `@answer_mode("名字")` 注册函数,签名
  `fn(view, std, **resolved_args) -> AnswerResult`;
- `AnswerResult` 是 frozen dataclass:`label: str`(必须是题面选项之一,
  编译时校验)+ `witness: dict`(方位角/转角/裕度等数值,进证书);
- spec 侧:`ScriptSpec` 增 `answer: AnswerSpec`,其中
  `AnswerSpec = {mode: str, args: dict[str, str|int|float|bool]}`,
  args 里的 `$变量` 与帧范围用 checker 现有的解析函数解析(复用
  `_resolve_clause_args` 的逻辑,必要时把它提为公共函数,不要复制粘贴);
- spec 版本 **v3 → v4**(scriptgen_spec.v4)。

**裁决乙:证书的答案块泛化,证书版本 v1 → v2。**
`AuthoritativeAnswer` 改为:`mode: str`、`label: str`、
`witness: dict[str, float|int|str]`。自运动的旧字段(azimuth_deg、
sector、margin_deg、question_frame)全部进 witness,数值必须与现状
**一字不差**(回归验收见 §5)。变体逻辑里所有 `cert.answer.sector`
改为 `cert.answer.label`。certificate schema_version 升
scriptgen_certificate.v2。

**裁决丙:变体种类由 spec 驱动,算子锚点带回退。**
`VariantBuilder.build_all` 不再无条件造四种,改为只造
`spec.variant_expectations` 里声明的种类(净转向/指回起点没有目标
可删,不声明 drop_key 就不造)。算子锚点泛化:

- permute 的打乱区间:`[frame_vars.get("t_gone", 1), t_q)`——没有
  t_gone 的剧本从第 1 帧起打乱(第 0 帧保留为起点锚,指回起点题需要);
- delay 的停顿帧:上述区间的中点;
- drop_filler 的保护锚:essential 帧 ∪ **全部帧变量的值**(不再写死
  t_seen/t_q 两个名字)。

**裁决丁:新题型 v1 一律复用 plan record 的 binding。** 净转向、指回
起点的答案不引用目标物,但 spec 仍声明 target 槽位、编译时照常传入
binding——避免本轮就开"无目标剧本"的引擎手术。这是记录在案的权宜,
真正的无目标泛化等有需要再做。

**裁决戊:合格性是机会性的,不合格跳过而非报错。** 新题型带各自的
成题条款(见 T4),某条轨迹过不了某题型的条款(例:画面侧题的中线
裕度),该题型的家族**跳过并在组清单里记明原因**,不阻塞其他题型。
已知事实:现有 motif 起手正对目标,画面侧题在现有三条轨迹上可能
全部或部分跳过——这是预期行为,修 motif 是下一个任务的事。

**裁决庚(2026-08-08 补,答复执行方提问):std.v3 的旧 plan record
允许在 std.v4 下编译,以显式声明的版本谱系放行,不放宽阻断本身。**

冲突:wallfix 的 plan record(只读)记录 standard_version=std.v3,
编译器升 std.v4 后,现有 standard_version_drift 审计会阻断 canonical。

裁决依据:该审计的本意是"搜索期的承诺必须在编译期的同一把尺下兑现"。
std.v4 相对 v3 **只新增三个常量、未改动任何既有数值**,因此 v3 时代的
每一条搜索承诺在 v4 下的判定逐项相同——版本字符串变了,尺没变。
按字符串阻断在此处比真实语义更粗,应以显式数据声明兼容谱系:

```python
# standards.py 模块级。谱系准入标准:仅当新版本对旧版本是纯扩展
# (零既有数值变动、只增字段)时才可列入;每项附一行理由。
COMPATIBLE_PLAN_STANDARDS: dict[str, tuple[str, ...]] = {
    # std.v4 只新增 view_side_margin_deg / net_turn_margin_deg /
    # homing_min_distance_m,v3 既有判定逐项不变。
    "std.v4": ("std.v3",),
}
```

执行要求:

1. drift 判定改为:plan 版本 ∉ {当前版本} ∪ 兼容集 才置 mismatch;
   证书照旧同时记录编译版本(v4)与 plan 版本(geometry.standard_version
   =v3),审计链不断;
2. **谱系不传递**:v2 与更早不入列,旧 `batch/` 的 navfix 前数据依然
   被阻断——那次是真语义变更,必须重造,历史不翻案;
3. 配一个钉住"纯扩展"声明的单测:standards.py 里维护
   `STD_V4_ADDED_FIELDS = ("view_side_margin_deg", "net_turn_margin_deg",
   "homing_min_distance_m")`,测试断言 CompileStandard 的字段集合 =
   钉死的 v3 字段清单 ∪ STD_V4_ADDED_FIELDS——将来有人改既有数值或
   偷偷加字段而不更新谱系声明,测试先红;
4. 组打包时 `geometry_plan` **只传给与 plan record capability 相同的
   spec**(即自运动方向题);新题型无搜索期承诺,geometry_plan=None,
   本就不触发 drift 审计。

**裁决己:新阈值三个,standards **std.v3 → std.v4**:**

```python
# --- 答案判界裕度(std.v4)---
view_side_margin_deg: float = 15.0   # 画面侧题:目击帧方位角离视野中线
net_turn_margin_deg: float = 15.0    # 净转向题:|净转角| 离 0° 判界
homing_min_distance_m: float = 1.0   # 指回起点题:终点距起点的最小距离
```

净转角用**带符号累加**(逐帧 wrap 后的带号增量求和),不是现有
cumulative_turn_deg 的绝对值累加——两个概念,别混。

## 4. 任务分解

### T1 答案登记处与 spec 声明(裁决甲/乙,约半天)

answers.py 建登记处;首个模式 `target_sector`(现 derive_answer 的
逻辑原样搬入):args `{obj: "$target", frame: "$t_q"}`,witness 含
azimuth_deg/sector(=label)/margin_deg/question_frame。spec 加 answer
块并升 v4;SELF_MOTION 补声明;compiler.derive_answer 改为登记处分发;
证书答案块按裁决乙泛化,升 v2。

验收:全部现有测试改字段名后绿;三条 wallfix 轨迹重编译,label 与
witness 数值和改前逐项相等(写一个对照测试:硬编码 left/left/right
与三个方位角值)。

### T2 变体机制泛化(裁决丙,约 2 小时)

按裁决丙改 variants.py;family.py 的金标引用同步。
验收:现有 test_variants 全绿(自运动四变体行为不变)。

### T3 新答案模式三个(约半天)

- `view_side`:args `{obj, frame}`;目击帧方位角 >0 → "left_half",
  <0 → "right_half";witness 含方位角与离中线裕度;
- `net_turn`:args `{frames}`(帧范围);带符号净转角 >0 → "left",
  <0 → "right";witness 含 net_turn_deg;
- `start_sector`:args `{frame}`(提问帧);以提问帧位姿算**第 0 帧
  相机位置**的方位角,离散四扇区;witness 含方位角、裕度、起点距离。

配套谓词(判定+见证,单测各一组):

- `view_side_margin_ge(obj, frame)`:|方位角| ≥ view_side_margin_deg;
- `net_turn_margin_ge(frames)`:|带符号净转角| ≥ net_turn_margin_deg;
- `start_far_enough(frame)`:起终点距离 ≥ homing_min_distance_m;
- `start_sector_margin_ge(frame)`:起点方位角的扇区裕度 ≥
  sector_margin_deg(复用现有常量)。

### T4 三个新 spec 进 library(约半天)

| spec | capability | 条款(phase/on_violation) | answer | variant_expectations |
|---|---|---|---|---|
| NET_TURN | path_integration | trackable(search/abstain)、net_turn_margin_ge(search/invalid) | net_turn 0:$t_q | permute→abstain, drop_filler→same, delay→same |
| HOMING | homing_probe | trackable(search/abstain)、start_far_enough(search/invalid)、start_sector_margin_ge(search/invalid) | start_sector $t_q | permute→abstain, drop_filler→same, delay→same |
| VIEW_SIDE | view_side_check | seen_early(compile/abstain)、view_side_margin_ge(search/invalid) | view_side $t_seen | permute→same, drop_key→abstain, drop_filler→same, delay→same |

帧变量:NET_TURN/HOMING 只需 `t_q: last_frame()`;VIEW_SIDE 用自运动
同款三件套。模板选项:净转向 (left, right, 无法判断);指回起点
(front, left, back, right, 无法判断);画面侧 (left_half, right_half,
无法判断)。题面照家族规矩:无帧号、不断言目击、含弃答引导。
abstain_on_unresolvable:VIEW_SIDE 设 ("t_seen",),另两个空。

注意 VIEW_SIDE 的 permute→same 是**关键验证点**:打乱区间从 t_gone
起,目击段不动,答案必须不变——这一格是能力框架签名矩阵里"检查题对
置换免疫"的落地,测试必须覆盖。

### T5 组打包:一条轨迹一组家族(约半天)

family.py 增 `build_question_group(bundle, plan_record, scene_ir, out,
std, scripts, ...)`:对每个 spec 依次编译,合格者各出一个家族目录
(`out/<capability>/family.json + index.html`,媒体共享一份
`out/media/`,相对路径注意加 `../media/` 前缀或每族复制,选前者),
外加 `out/group.json`:trajectory 出处、各题型 家族id/角色
(primary/probe/check)/金标/跳过原因。family schema 升 **v2**:增
`role`、`question_group_id` 字段,金标字段名与证书对齐为 label。
contracts/schema.py 的注册键改 scriptgen_family.v2.schema.json。
family_cli 增 `--group` 模式(默认单剧本行为不变)。

### T6 集成测试与文档(约 2 小时)

- 集成测试:对 wallfix 三条 bundle 跑组打包;断言方向题金标
  left/left/right 不变;净转向/指回起点家族存在且其金标与"从位姿
  手工计算的期望值"一致(测试里用几何独立重算,不许抄编译器输出);
  画面侧家族存在与否按裕度实测,若跳过断言 group.json 记录了原因;
- 假后端单测:qualifying_fake 上四个 spec 各编译一次,验证
  label/witness;VIEW_SIDE 的置换不变性单测;
- 文档:`scriptgen能力框架_v1.md` §8 实现状态打勾;交接文档 §4 补
  answers.py 一行;`docs/scriptgen_engine.md` 补答案声明一节。

## 5. 回归红线(最重要的验收)

1. 自运动方向题在 wallfix 三条轨迹上:label 与 witness 里的
   azimuth/margin 数值和本任务开始前**逐项相等**;
2. 现有全部测试(67 个)改字段名后全绿,不许删测试;
3. 置换变体仍死于 trackable 判弃答(方向题),画面侧题置换后 label
   不变——签名矩阵两个关键格都要有测试钉住;
4. `git diff` 里 standards.py 仅含 std.v4 三个新常量与版本号;
5. episode3d、scripts/pilot 零改动。

## 6. 提交划分建议

1. T1+T2:答案登记处、证书 v2、变体泛化(含回归对照测试);
2. T3+T4:新模式、新谓词、std.v4、三个新 spec(含单测);
3. T5+T6:组打包、family v2、集成测试、文档。

## 7. 明确不做(超出本任务)

- motif 起手改侧面入画、转向偏手性修复、联合搜索(下一个任务:
  画面侧题的产出率靠它拯救);
- 想象视点模式(归参考系变换剧本)、物物关系/比较/先后/计数/存在题型;
- 遮挡式消失子剧本;空缺D;任何渲染。
