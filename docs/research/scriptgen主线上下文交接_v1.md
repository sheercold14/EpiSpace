# Scriptgen 主线上下文交接 v1

> 用途:新会话/新成员接续开发时的最小充分上下文。截至 2026-08-06,
> 分支 `feature/scriptgen-engine-v1`(9 个提交,至 4023a31)。

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

| 层级 | 能力 | 一句话 |
|---|---|---|
| 基元 | 状态持久 | 物体离开视野后仍记得(问"当时在你哪边") |
| 基元 | 对应锚定 | 同款两实例不认混(需场景摆同资产双胞胎) |
| 基元 | 参考系变换 | 站桩想象换视角(角度—错误曲线数据源) |
| 组合 | 跨视图绑定 | A、B 永不同框,经锚接力推关系(锚=地标) |
| 组合 | 自运动更新 | 移动转身后问看不见的目标方位(锚=自身) |
| 元 | 证据校准 | 当且仅当证据不完整时答"无法判断" |

两个组合共享全部基元、只换锚 → 可证伪预测:只训一种组合,H2 应零样本迁移
到另一种,H0/H1 不应。①与⑤构成"问过去(零变换)vs 问现在(变换=转向)"
对照,分离出更新环节的单独贡献。

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

代码在 `src/spatial_episode/scriptgen/`,33 个测试全绿。

- `standards.py`:全部阈值,冻结版本 std.v2(双阈值可见性、15° 扇区裕度、
  每帧转向≤40°/位移≤1.2m 的可追踪约束、搜索期收紧 1.2×);
- `predicates.py`:谓词单点,返回 判定+见证;模糊帧不许出题;
- `spec.py`/`library.py`:声明式剧本;已实现自运动更新(三段式:
  看到→过渡[允许部分可见]→彻底消失→提问);
- `checker.py`:$变量/加减法/闭区间帧范围/帧变量解析器(last_visible、
  first_invisible_after 等);
- `motifs.py`:walk_and_turn(限速 32°/帧,终点随机环顾以平衡答案分布);
- `generate.py`:主循环,拒绝按条款计数;真实场景每秒数千候选;
- `behavior.py`:三适配器——scene_ir→SceneLayout、trajectory_plan→位姿、
  RenderSceneView(掩码像素权威可见性,与渲染报告 814/814 一致)、
  plan_to_agent_views(计划→采集相机调度,往返测试);
- `web/scriptgen_review.html` + `scripts/build_scriptgen_review.py`:
  单轨迹审核页(俯视图/三通道胶片/条款见证/标准表)。

闭环已验证:gates_bedroom 三条轨迹,规划 0.1s → Isaac 渲染约 70s/条 →
掩码复核全过 → 答案重算一致(left/left/right,转向 141–146°)。
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
      │ (相机逐帧位姿调度,             │ scene_ir.json(场景几何真值)
      │  由 plan_to_agent_views 导出)   │ trajectory_plan/snapshot/report.json
      │                                  ▼
┌────────── code/EpiSpace ── 主库,零模拟器依赖 ────────────────────┐
│ scriptgen  剧本→槽位→候选→免渲染核验→轨迹计划        [已通]      │
│ 编译       bundle 掩码→权威答案/证据帧→certificate    [空缺A]    │
│ family     干预算子→重编译→打包                       [空缺B/C]  │
│ 评测       预测记录→家族级指标                        [空缺D]    │
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
- 现成试验数据:`OminiGibson/outputs/scripted_demo/batch/render_{0,1,2}`
  三条已渲染轨迹 + `plan_{i}.record.json`,足够开发整个后半段,无需新渲染。

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
  ✅        ✅          ✅          空缺A      空缺B     空缺C   空缺D
```

- A 权威编译(任务#16):CapabilityCompiler 插件,渲染后端重解析帧变量
  (已知 p0 几何 t_seen=0 而渲染第 1 帧也清晰,以渲染为准)、重跑条款、
  出权威答案与最小证据帧集;
- B 变体家族(#17):干预算子作用于已渲染帧序列,重跑编译器,预期不符
  即 MISMATCH 阻断;
- C 打包(#18):family schema 进 contracts 版本化;题面填空/指称唯一/
  泄漏抽查;
- D 打分(#2):家族级指标(条件一致率/协变率/弃答校准/置换一致),
  输入只有预测记录+family 标签,与模型和生成端解耦。

交接点(用户上调试器的时机):一条命令从 render_0 产出
"canonical + 4 变体"的完整 family JSON + family 审核页。

## 8. 遗留已知问题(不阻塞,勿丢)

- "back"答案过稀(终点环顾背向目标时转身量小,被 80° 下限拒),需权重校准;
- 大厅类场景 ambiguous_referent 拒绝率高,需带定语指称(区域限定)解;
- 对应锚定剧本需要同资产双实例的场景编辑管线;
- t_seen 等帧变量的渲后重解析(归属空缺A);
- 19 个数据版本错位的常红测试;路径硬编码待 workspace resolver。

## 9. 常用命令

```bash
# 免渲染演示
.venv/bin/python -m spatial_episode.scriptgen.cli --capability self_motion_update --seed 17 --out /tmp/p.json
# 测试
.venv/bin/python -m pytest tests/scriptgen -q
# 渲染(OminiGibson 目录下)
bash scripts/run_in_omnigibson.sh --accept-eula python -m omnigibson_episode.cli acquire \
  --recipe configs/omnigibson_scripted_selfmotion_demo.yaml --output <dir> --gpu-id 3 --headless --overwrite
# 审核页
.venv/bin/python scripts/build_scriptgen_review.py --bundle <render_dir> \
  --plan-record <record.json> --scene-ir <scene_ir.json> --out <out_dir>
```
