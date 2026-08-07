# 任务书:轨迹通行硬验证(navfix)v1

> 交给谁:任何接手实现的人或工具(Codex 等)。
> 前置阅读:`docs/research/scriptgen主线上下文交接_v1.md`(仓库定位与规矩),
> 本文自带做这件事所需的全部细节。
> 分支:`feature/scriptgen-engine-v1` 之上继续。

## 1. 问题与证据

生成器规划轨迹时只保证"不出地板外接框",家具足印从不参与通行判断
(它们只用于视线遮挡)。渲染侧相机悬在 1.5 m,从家具上方滑过画面正常,
质量门查不出来。结果是三条示范轨迹全部物理非法,已逐帧核实:

- plan_0:第 7–9 帧穿过咖啡桌,第 9–15 帧(含提问帧)站在沙发足印内;
- plan_1:起点压在扶手椅足印内,第 4–11 帧横穿床;
- plan_2:起点同上,第 6–9 帧穿床。

配套实测(gates_bedroom,用于估算工作量与效率预算):

- 现状生成吞吐 7690 候选/秒;
- 足印点检+线段检的成本 4 µs/候选(现有六条款全管线约 130 µs/候选);
- 当前 motif 产出的候选 100% 物理非法(0.3 m 膨胀)——
  所以只加验证不改 motif,产出率归零,motif 改造是必做项;
- 该卧室自由面积:0.3 m 膨胀下 32.4%(8.2 m²),0.15 m 下 47.2%。
  直线折线在这种密度下基本必撞,中段需要 A* 布线。

## 2. 完成标准(一句话)

生成器只产出人能实际走出来的轨迹:每帧位姿和相邻帧连线都不进家具的
躯干高度带足印;旧的三条非法轨迹被新谓词拒绝(回归测试为证);
数据重造后全部测试绿、family 一条命令照常出活。

## 3. 必须遵守的仓库纪律(违反即返工)

1. 新代码只进 `src/spatial_episode/`;`src/episode3d/` 和 `scripts/pilot/`
   一行不碰。
2. 一切判定走谓词库(`predicates.py`,判定+见证),一切阈值进
   `standards.py`。本次新增判定阈值,标准版本 **std.v2 → std.v3**。
3. motif 只提议、不判定。motif 内部可以用自己的提议参数
   (现有先例:motif 限转 32°/帧,而标准红线是 40°),但不许 import
   standards 的值来做判定。
4. 变体答案由重跑编译器产生,不许人工指定;不许为让测试通过手改金标。
5. 每个逻辑单元一个提交,提交信息说清做了什么、为什么。
6. 测试基调不变:单元测试用假后端,集成测试吃真渲染数据、
   数据不在时干净 skip;不写计时断言(性能手测,不进 CI)。

## 4. 两个必须预先定死的设计裁决

**裁决甲:通行条款不参与编译期重判。** 编译器目前会把 search 阶段条款
在渲染后端上再判一遍。通行条款不能进这个重判:置换变体的帧序被故意打乱,
相邻位姿跳跃,连线必然穿家具;删无关帧变体合并了线段,可能切家具的角。
这两种情况下轨迹本身(渲染时真实走过的那条)是合法的,非法的只是
"呈现顺序",而呈现顺序的可答性已由 trackable 条款负责。若通行条款
参与重判,置换变体会先死在通行上(判 invalid),而不是死在 trackable 上
(判 abstain),变体预期核对直接崩。

做法:`spec.py` 的 `Clause.phase` 从 `Literal["search","compile"]` 扩为
`Literal["search","compile","search_only"]`,spec 版本升
**scriptgen_spec.v2 → v3**。`generate.py` 调 `check_clauses` 时
phases 传 `("search","search_only","compile")`;`compiler.py` 保持
`("search","compile")` 不变,search_only 条款自然被滤掉,
留一法和变体验证同样不受影响。通行见证仍会写进 plan record
(生成期照常记录),审计链不断。

**裁决乙:高度带数据放 layout,阈值判断放谓词。** `layout_from_scene_ir`
不接触 standards:它把每件非结构物体的旋转足印和 z 跨度原样存进
layout 的新字段 `obstacles`;"哪些障碍落在躯干高度带内"由谓词拿着
std.v3 的阈值现场过滤。这样适配器保持无阈值,换标准不用重建 layout。

## 5. 任务分解

### T1 标准升版(0.5 小时)

`standards.py`:`standard_version` 改 "std.v3",新增

```python
# --- 通行净空(std.v3)---
body_radius_m: float = 0.30        # 人体半径,足印按此膨胀
clearance_z_low_m: float = 0.10    # 躯干高度带下缘(低于此的地毯等不挡路)
clearance_z_high_m: float = 1.70   # 上缘(高于此的吊灯等不挡路)
```

其余数值一个不动。验收:diff 里 standards.py 只有这四行和版本号。

### T2 layout 增 obstacles(1–1.5 小时)

`sceneview.py`:`SceneLayout` 增字段
`obstacles: tuple[Obstacle, ...] = ()`,其中 `Obstacle` 是新的 frozen
dataclass:`label`(raw_label)、`center_xy`、`half_extents_xy`、
`yaw_deg`(足印旋转)、`z_low`、`z_high`。用**旋转矩形**,不用 AABB
——斜摆的床用 AABB 会封掉半个房间,拒绝率虚高。

`behavior.py` 的 `layout_from_scene_ir`:从每个非结构 entity 的 OBB
提取上述字段填入(z 跨度可沿用现有 `_z_span` 的保守算法)。
occluders 逻辑不动——遮挡与通行是两回事,两个列表并存。

验收:gates_bedroom 装载出 22 个 obstacles(高度带过滤前的全量;
过滤是谓词的事),现有测试不红。

### T3 几何与谓词(1–1.5 小时)

`geometry.py` 增两个纯函数:`point_in_rotated_rect`、
`segment_intersects_rotated_rect`(把点/线段变换到矩形自身坐标系,
复用现有 slab test 思路)。

`predicates.py` 增两条,均返回判定+见证:

- `poses_clear(view, std, *, frames)`:每帧相机位姿不落入任何
  "z 跨度与 [clearance_z_low_m, clearance_z_high_m] 相交、
  且按 body_radius_m 膨胀后"的障碍足印。见证:最坏一帧的
  帧号、障碍 label、穿透深度;
- `path_clear(view, std, *, frames)`:相邻帧连线不与上述足印相交。
  见证:最坏一段的帧对、障碍 label。

单元测试:手搭一个含一件障碍的小 layout,四种用例——帧在障碍内、
连线穿障碍、障碍在高度带外(吊灯,应放行)、干净轨迹(应通过)。

### T4 条款接入(0.5 小时)

`spec.py`:按裁决甲扩 `phase` 字面量,spec 版本升 v3。
`library.py`:SELF_MOTION 的 clauses 最前面插两条
(`phase="search_only"`,on_violation 用默认 invalid):

```python
Clause(name="poses_clear", predicate="poses_clear",
       args={"frames": "0:$t_q"}, phase="search_only"),
Clause(name="path_clear", predicate="path_clear",
       args={"frames": "0:$t_q"}, phase="search_only"),
```

放最前是刻意的:4 µs 的检查先毙掉必死候选,省掉后面昂贵的可见性判定。
`generate.py` 的 check_clauses 调用补上 "search_only"。
验收:现有 test_variants 全绿(证明编译期确实没重判通行);
不改 motif 直接生成,gates_bedroom 拒绝直方图里
`clause:poses_clear`/`clause:path_clear` 占绝对多数——这就是第 1 节
"100% 非法"的复现。

### T5 motif 改造:自由空间采样 + A* 布线(0.5–1 天,本任务最大件)

`motifs.py` 重做 walk_and_turn 的路径规划部分,保留其意图结构
(起点看目标 → 走远 → 终点环顾随机方向以平衡答案分布):

1. 模块内建占据栅格:0.05 m 格距,从 layout.obstacles 光栅化,
   膨胀用 motif 自己的提议常数 `PROPOSAL_CLEARANCE_M = 0.35`
   (比标准的 0.30 略紧,与"限转 32° vs 红线 40°"同一模式:
   提议留裕量,判定看标准)。高度带过滤在此处可直接用数值区间常量,
   同样属提议参数。每个 layout 建一次,module 级缓存;
2. 起点:在距目标 2–4 m 的环带里拒绝采样,只收自由格;
3. 中途/终点:自由格里采样,段间用栅格 A* 连接,输出路径点折线;
4. 折线之后沿用现有 `_walk_polyline` 的限速偏航逻辑和终点环顾逻辑,
   答案平衡机制不动。

效率预算(手测,不进 CI):A* 摊到每候选 ≤1 ms,整体吞吐 ≥1000 候选/秒,
gates_bedroom 上 attempts_per_binding=150 内能稳定出计划。

验收:开着 T4 的条款,`generate_plans(layout, SELF_MOTION, std_v3,
seed=17, max_plans=3)` 能出 ≥1 个计划,且其全部位姿与线段通过两条新谓词。

### T6 回归测试(0.5 小时)

新测试文件或并入 test_predicates:读
`OminiGibson/outputs/scripted_demo/batch/plan_0.record.json` 的 poses
(数据不在则 skip),喂给两条新谓词,断言均为 False,且见证里出现
coffee_table 与 sofa。这是本次 bug 的病历,永久留档。

### T7 数据重造(人力 1–2 小时 + GPU 约 15 分钟;渲染步需要 Isaac 环境)

**顺序敏感:std.v3 会让旧 plan record 在编译器的版本核对处被阻断,
这是设计使然。所以 T1–T6 代码全就位后一口气做完本步,中间不停。**

1. 用新引擎重新生成三份计划(seed 17/23/41,场景同前)。注意:
   motif 的随机数用法变了,plan_id 的 attempt 后缀和位姿都会不同,
   这是预期;
2. 渲染进**新目录** `outputs/scripted_demo/batch_navfix/`,旧 batch/
   原样保留(T6 的回归测试和历史证据都指着它);渲染命令见交接文档 §9,
   此步需要 behavior-spatialep 环境和 GPU,若执行方没有,产好
   plan.views.json 和命令行后移交人工执行;
3. `tests/scriptgen/_batchdata.py` 指向新目录;重跑编译,把
   test_compiler/test_variants/test_family 里钉死的期望值
   (扇区、帧变量、essential 帧、延迟旋钮)**从编译器输出重新抄录**,
   不许拍脑袋填;三条轨迹的答案分布若退化(比如全 left),
   换 seed 重生成直到方向至少有两种;
4. 对新 render 目录跑一遍 family_cli,确认一条命令闭环;
5. 更新两份文档里被钉死的数值:`scriptgen调试路线图_v1.md`
   (逐站期望值)与 `scriptgen主线上下文交接_v1.md`(§7 的金标表、
   §4 的 std 版本描述、遗留问题里补一条"旧 batch 为 navfix 前数据,
   仅供回归测试")。`docs/scriptgen_engine.md` 的条款清单补两条新条款。

### 明确不做(超出本任务)

- OminiGibson 侧胶囊碰撞权威复核(排在批量扩容前,另立任务);
- navmesh / traversability 图导出;
- 空缺D(评测指标)。

## 6. 建议的提交划分

1. T1+T2+T3:标准升版、障碍数据、通行谓词(带单测);
2. T4+T6:条款接入与旧轨迹回归测试(此提交后集成测试预期部分红,
   提交信息里写明原因与下一步);
3. T5:motif 自由空间采样与 A* 布线;
4. T7:数据重造与全部钉值/文档更新(此提交后全绿)。

若工具链不便做渲染,提交 3 后停下,把 plan 文件与渲染命令交回。

## 7. 最终验收清单

- [ ] `pytest tests/scriptgen -q` 全绿(新渲染数据在位时);
- [ ] 回归测试证明旧 plan_0 被新谓词拒绝;
- [ ] `git diff` 中 standards.py 仅含 std.v3 四个新常量与版本号;
- [ ] episode3d、scripts/pilot 无任何改动;
- [ ] 变体测试全绿(置换仍是 abstain 且死于 trackable,证明裁决甲落实);
- [ ] family_cli 对 batch_navfix/render_0 一条命令出 family.json + 审核页;
- [ ] 生成吞吐手测 ≥1000 候选/秒,记录在提交信息里。
