# Scriptgen 主线上下文交接 v1

> 用途:新会话/新成员接续开发时的最小充分上下文。截至 2026-08-07,
> 分支 `feature/scriptgen-engine-v1`。空缺 A/B/C 与 navfix 均已完成,
> 空缺 D(评测指标)未动。单步核对流程见
> `docs/research/scriptgen调试路线图_v1.md`。

## 1. 研究定位(为什么做)

评测单位不是单题,而是 episode family:同一场景、同一道题的一组受控变体
(顺序置换 / 删关键帧 / 删无关帧 / 延迟揭示 / 换站位)。评的是行为规律
——该不变时不变(不变性)、该变时按几何规律变(协变性)、证据不足时弃答
(校准)。目的:判别三个竞争假设。

- H0 查表:逐题模式匹配,无内部状态;
- H1 碎片:局部能力孤立,不共享状态;
- H2 心智模型:统一、可更新的空间状态,各能力是对它的读操作。

单题准确率无法区分三者;干预下的行为联动可以。

## 2. 能力基底(测什么)

> 2026-08-08 定稿版见 `docs/research/scriptgen能力框架_v1.md`(含准入
> 标准 C1–C3、题型清单、诊断梯、行为签名矩阵、决策记录)。要点:

| 层级 | 能力 | 一句话 |
|---|---|---|
| 基元 A | 参考系变换 | 站桩想象换视角(角度—误差曲线数据源) |
| 基元 B | 对应锚定 | 同款两实例不认混(需场景摆同资产双胞胎) |
| 基元 C | 路径积分 | 把帧间变化积分成自身运动史(问"你净转了多少") |
| 组合一 | 自运动更新 | 读出+C+A,锚=自身(问看不见的目标现在方位) |
| 组合二 | 跨视图绑定 | 读出+B+A+关系复合,锚=地标(永不同框二物的关系) |
| 元 | 证据校准 | 当且仅当证据不完整时答"无法判断" |

旧表中的"状态持久"已除名:全上下文协议下它等价于多图检索(违准入
标准 C1),其题型(画面侧/先后)降为每家族自带的操纵检查,质检不计分;
真·记忆维持留给流式协议因子。两组合共享变换引擎 A、只换输入通道与锚
→ 可证伪迁移预测:只训组合一,H2 预测组合二中依赖 A 的成分零样本受益,
H0/H1 不预测。同一条自运动轨迹上"检查→积分→积分+变换→全流水线"
四级诊断梯给出失败定位。

## 3. 剧本设计五原则(第一性原理)

- R1 唯一路径:通往答案的推理路径只能是被测能力;其余路径由剧本扣信息堵死;
- R2 路径可走通:唯一路径上每个输入必须能从图像干净取出(公平性);
- R3 其余平凡化:不测的基元被压到平凡,失败可归因;
- R4 裕度可判定:答案距离散边界有余量,证据帧有最小集合;
- R5 旋钮=目标计算规模:延迟长度/想象夹角/锚链长度/累计转向/实例数。

实例:自运动剧本必须有转身限速条款(否则帧间无重叠,自运动不可估计,
题目不公平);持久剧本刻意没有该条款(答案只取决于 t_seen 帧)。条款的
出现与缺席都可从原则推导。

## 4. 引擎现状(已验证)

代码在 `src/spatial_episode/scriptgen/`,67 个测试全绿(集成测试吃三条
已渲染轨迹,机器上没有数据时干净 skip;全程不需要 Isaac Sim)。

- `standards.py`:全部阈值,冻结版本 std.v3(双阈值可见性、15° 扇区裕度、
  每帧转向≤40°/位移≤1.2m,以及人体半径 0.30m、躯干高度带
  0.10–1.70m 的通行约束);改阈值必须再升版;
- `predicates.py`:谓词单点,返回判定+见证;模糊帧不许出题;
  `poses_clear/path_clear` 分别硬验每帧站位和相邻帧线段不进入旋转障碍足印;
  障碍同时包含家具和结构墙,见证显式记录检查的墙数;
- `spec.py`/`library.py`:声明式剧本,schema 为 scriptgen_spec.v3——
  新增三个家族契约字段:`Clause.on_violation`(abstain=证据被毁 /
  invalid=题目出包线)、`abstain_on_unresolvable`(帧变量解析失败的归类)、
  `variant_expectations`(各干预的预期效应,按能力声明),以及条款阶段
  `search_only`(实际采集路径必须通行,但不拿变体的呈现顺序重判通行);
  题面模板为家族安全版(无帧号、不断言目击、全家族逐字共享);
- `checker.py`:$变量/加减法/闭区间帧范围/帧变量解析器(last_visible、
  first_invisible_after 等);对着 SceneView 协议写,换后端即换判定依据;
- `motifs.py`:walk_and_turn 先以 5cm 占据栅格、0.35m 提议净空和 8 邻域
  A* 绕家具/墙,并把自由格预分连通域以避免跨房间搜索不可达终点;
  再限速到 32°/帧,终点随机环顾以平衡答案分布;motif 只提议,
  std.v3 谓词才是最终裁判;
- `generate.py`:主循环,拒绝按条款计数;wall-aware gates_bedroom 手测
  1063 候选/秒(400次生成、2259候选);
- `behavior.py`:三适配器——scene_ir→SceneLayout、trajectory_plan→位姿、
  RenderSceneView(掩码像素权威可见性,与渲染报告 814/814 一致,
  `from_bundle(scene_ir=...)` 支持外置几何真值)、plan_to_agent_views;
  SceneLayout 另存每件非结构物及结构墙的旋转 OBB 和 z 跨度为 obstacles
  (gates_bedroom 为 22 件物体+4 段墙=26),墙不进入可提问 objects;
- `compiler.py`(空缺A):CapabilityCompiler,渲后重解析帧变量、足额裕度
  重判 search/compile 条款(不重判 search_only 通行条款)、渲后位姿推权威答案,
  产出 scriptgen_certificate.v1
  (几何估计降级为对照字段,mismatch 非空即阻断);附留一法 essential 帧集;
- `sceneview.py` 新增 ReindexedSceneView:变体=原始帧的索引序列,
  是干预算子与留一法共用的基座;
- `variants.py`(空缺B):置换/删关键帧/删无关帧/延迟 四算子只产索引序列,
  金标一律由重跑同一编译器产生,与 spec 预期不符抛 FamilyMismatch;
- `family.py`/`family_cli.py`(空缺C):scriptgen_family.v1 单 JSON
  (已注册 contracts/schema.py),打包前审计指称唯一+题面三项泄漏检查,
  任一不过抛 FamilyBlocked;`media.py`:三通道 PNG 导出(自脚本下沉);
- `web/scriptgen_review.html`(单轨迹)+ `web/scriptgen_family_review.html`
  (家族:五条变体序胶片、原始帧号徽标、重复帧标记、金标/状态/预期、
  条款见证、审计灯)。family 页尚未在真浏览器目验(仅 JS 语法检查
  + 数据访问仿真)。

闭环已验证:gates_bedroom 三条 wall-aware std.v3 轨迹,规划 → Isaac 高质量渲染
(1024² RGB/depth/实例标签,约 65–70s/条)→ 掩码复核 → 权威编译 → family。
三条实际路径均以 0.30m 人体圆盘通过 26 个障碍(含4墙)的位姿与线段
硬验证,答案为 left/left/right。render_0 的几何 `t_seen=0` 被掩码
重解析为 1,turned 见证 135.1°→103.1°,扇区仍为 left
(97.1°,裕度 37.9°)。
第一次闭环曾抓出真 bug(质点视锥近似漏掉物体边缘 8k 像素),由此加了
"部分入画一律模糊"规则——渲后复核的价值已实证。

## 5. 三仓库关系与文件契约(流程图)

EpiSpace 零模拟器依赖。它需要的不是模拟器,而是模拟器导出的两种真值文件:
场景几何真值(scene_ir.json,用于免渲染规划)和像素真值(npz 实例掩码,
用于权威编译)。仓库间只以文件交接,没有代码依赖:

```text
┌────────── code/OminiGibson ── 采集库,唯一含模拟器代码 ──────────┐
│ Isaac Sim + CUDA · conda 环境 behavior-spatialep                  │
│ 职责:照计划渲染(scripted_plan 策略)、自采轨迹(旧 T1–T10)     │
└───────────────────────────────────────────────────────────────────┘
      ▲ 契约① 计划文件                  │ 契约② bundle(数据包)
      │ plan.views.json                  │ views/*.sensors.npz(rgb/深度/掩码)
      │ (相机逐帧位姿调度,             │ trajectory_plan/snapshot/report.json
      │  由 plan_to_agent_views 导出)   │ (batch bundle 不含 scene_ir.json,
      │                                  │  需外部指定同场景的一份)
      │                                  ▼
┌────────── code/EpiSpace ── 主库,零模拟器依赖 ────────────────────┐
│                                                                   │
│ scene_ir.json ──► layout(objects+rotated obstacles)               │
│                       └► 剧本→槽位→自由格/A*候选→免渲染硬核验      │
│                              └► plan.record.json(几何临时答案)  │
│                                        [已通,渲染在外部发生]     │
│                                                                   │
│ bundle+scene_ir ──► RenderSceneView(掩码=权威可见性)            │
│        │                                                          │
│        ▼ compiler.py                                   [A 已通]   │
│ 渲后重解析帧变量 → 重判证据/有效性条款 → 权威答案                  │
│        └► certificate(几何only对照;mismatch非空即阻断;         │
│           留一法 essential 帧集)                                  │
│        │                                                          │
│        ▼ variants.py                                   [B 已通]   │
│ 四算子产帧索引序列(置换/删关键帧/删无关帧/延迟)                 │
│   → ReindexedSceneView → 重跑同一编译器 → 各变体金标              │
│        └► 与 spec 声明预期不符 = FamilyMismatch 阻断              │
│        │                                                          │
│        ▼ family.py + family_cli.py(一条命令)         [C 已通]   │
│ 审计(指称唯一/无占位符/无帧号/无答案token)→ 打包                │
│        └► family.json(canonical+4变体+certificates,单文档)     │
│           + media/(相对路径 PNG)+ index.html(family 审核页)   │
│        │                                                          │
│        ▼ 评测:预测记录 + family 标签 → 家族级指标     [空缺D]    │
│                                                                   │
│ web 审核页 · scripts 薄壳 · docs 文档                             │
└───────────────────────────────────────────────────────────────────┘
      ▲ 软链接(临时,待数据迁移归档)
┌────── code/episode3D ── 旧主库,已冻结,只剩 data/ 3.2GB ────────┘
```

状态注记:

- OminiGibson 的 scripted_plan 改动(`omnigibson_episode/scripted.py` 新文件
  + config/acquire 各一处新增 + 演示配方 yaml)**未提交**,与用户既有
  未提交改动并存;演示配方里的 plan_path 是绝对路径,待路径解析器统一;
- EpiSpace 曾内置的 backends/omnigibson 副本已删除(4023a31),
  采集代码只此一份;
- 历史数据经软链接接入,已知 19 个测试因数据版本错位常红
  (期望 586 条、现有 466 条,新版数据下落不明);
- 当前合格数据:`OminiGibson/outputs/scripted_demo/batch_wallfix/` 下
  `render_{0,1,2}` + `plan_{i}.record.json`;seed 为 17/23/4(seed 13
  在 wall-aware 采样后与前两条同为 left,故换 4 保证至少两种方向)。
  `batch_navfix/` 是家具通行已修、墙尚未进采样器的中间批次。旧 `batch/` 三条均会
  穿家具,保持原样且只用于 `test_traversability_regression.py` 病历回归;
  已打包审核站点在同目录 `family_{0,1,2}/index.html`;
- batch bundle 不含 scene_ir.json,编译/打包时用
  `sweeps/t10-target-view-seed17-v1/bundles/gates_bedroom_t10_seed17/scene_ir.json`
  (同场景 63b1adc6,runtime id 与 bundle snapshot 逐一核对一致,
  目标 armchair 72faab69…→988247081)。

## 6. 已拍板的决策

1. certificate 一切以渲染掩码为准,几何估计仅作对照字段;
2. family 存单个 JSON(canonical+全部变体+certificate),帧图相对路径引用;
3. 新代码进 spatial_episode,episode3d 冻结只修错;
4. 采集代码正本是 code/OminiGibson,EpiSpace 内副本已删除(4023a31);
5. scripts/ 只做薄壳,逻辑用到第二次即下沉 src 带测试;保留政策与逐脚本
   判决见 scripts/INDEX.md(16 个活跃,14 个数据来源脚本封存 scripts/pilot/);
6. 改常量必升 standard_version;干预后答案由重跑同一编译器产生,不许人工指定。

## 7. 主线与空缺(按序推进)

```
剧本 → 轨迹计划 → [渲染,外部] → 权威编译 → 变体家族 → 打包 → 打分
  ✅        ✅          ✅          ✅A        ✅B       ✅C    空缺D
                                 cceb13d    9beee33   5a67ecd
```

- A 权威编译(任务#16,已完成):CapabilityCompiler,渲染后端重解析帧变量
  (wallfix render_0 实测 t_seen 由几何 0 重解析为 1,连带 turned 见证
  135.1°→103.1°)、重跑证据/有效性条款、出权威答案与留一法 essential 帧集;
- B 变体家族(#17,已完成):干预算子只产帧索引序列,金标由重跑同一
  编译器产生,预期不符即 FamilyMismatch 阻断;预期效应按能力声明在 spec
  (自运动:置换/删关键帧→弃答,删无关帧/延迟→不变);
- C 打包(#18,已完成):scriptgen_family.v1 进 contracts 版本化;
  指称唯一/占位符/帧号/答案token 四项审计,不过即 FamilyBlocked;
- D 打分(#2,未动):家族级指标(条件一致率/协变率/弃答校准/置换一致),
  输入只有预测记录+family 标签,与模型和生成端解耦。

navfix 后的固定金标如下(全部来自实际掩码重编译,不是手填):

| index | seed/帧数 | render 帧变量 `seen,gone,q` | 权威答案(方位/裕度) | essential | delay 干预 |
|---|---|---|---|---|---|
| 0 | 17 / 12 | 1,2,11 | left (97.1°/37.9°) | 1,2,8 | 9→13 |
| 1 | 23 / 14 | 1,2,13 | left (77.4°/32.4°) | 1,2,3 | 11→15 |
| 2 | 4 / 11 | 1,2,10 | right (-84.5°/39.5°) | 1,6,9 | 8→12 |

一条命令(见 §9)从 `batch_wallfix/render_{0,1,2}` 各产出完整 family
JSON + 审核页,三条家族金标为
canonical/置换/删关/删无关/延迟 = left/弃答/弃答/left/left、
left/弃答/弃答/left/left、right/弃答/弃答/right/right,
延迟旋钮如上表。

## 8. 遗留已知问题(不阻塞,勿丢)

- "back"答案过稀(终点环顾背向目标时转身量小,被 80° 下限拒),需权重校准;
- 大厅类场景 ambiguous_referent 拒绝率高,需带定语指称(区域限定)解;
- 对应锚定剧本需要同资产双实例的场景编辑管线;
- 19 个数据版本错位的常红测试;路径硬编码待 workspace resolver
  (scene_ir 需在命令行显式传路径也属此类);
- family 审核页未在真浏览器目验(仅 node 语法检查 + 数据访问仿真);
- 目前通行权威来自 scene_ir 的静态旋转 OBB;OminiGibson 胶囊碰撞复核与
  navmesh 导出尚未做,属于批量扩容前的下一道安全门;
- 墙 OBB 是保守近似:抽查 46 个既有场景的旧采集轨迹,37 个会触发
  0.30m 墙净空(多数是贴墙,少数大 OBB 可能覆盖门洞/不规则墙内部)。
  `batch_wallfix` 当前场景已零碰撞闭环,但跨场景扩量前必须用 navmesh/
  胶囊碰撞区分"旧采样确实贴墙"与"复合墙 OBB 误杀";
- 两处待复核的自主裁决:①"置换→弃答"的预期是按自运动语义推的,
  改 spec 一行即可换语义,MISMATCH 兜底;② drop_count/delay_extra
  归类为生成参数而不进 standards.py;若改成判定阈值则必须升 std.v4。

## 9. 常用命令

```bash
# 免渲染演示
.venv/bin/python -m spatial_episode.scriptgen.cli --capability self_motion_update --seed 17 --out /tmp/p.json
# 测试
.venv/bin/python -m pytest tests/scriptgen -q
# 渲染(OminiGibson 目录下)
bash scripts/run_in_omnigibson.sh --accept-eula python -m omnigibson_episode.cli acquire \
  --recipe configs/omnigibson_scripted_selfmotion_demo.yaml --output <dir> --gpu-id 3 --headless --overwrite
# 单轨迹审核页
.venv/bin/python scripts/build_scriptgen_review.py --bundle <render_dir> \
  --plan-record <record.json> --scene-ir <scene_ir.json> --out <out_dir>
# family:一条命令,bundle → family.json + media/ + 审核页(交接点命令)
.venv/bin/python -m spatial_episode.scriptgen.family_cli \
  --bundle  <OminiGibson>/outputs/scripted_demo/batch_wallfix/render_0 \
  --plan-record <OminiGibson>/outputs/scripted_demo/batch_wallfix/plan_0.record.json \
  --scene-ir <OminiGibson>/outputs/sweeps/t10-target-view-seed17-v1/bundles/gates_bedroom_t10_seed17/scene_ir.json \
  --out /tmp/family/f0        # 可选 --seed/--drop-count/--delay-extra
# 看审核页
cd /tmp/family/f0 && python -m http.server 8000   # 浏览器开 localhost:8000
```
