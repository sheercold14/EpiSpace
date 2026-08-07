# Scriptgen 单步调试路线图(VSCode)

> 两条路线:前半段"生成"(剧本→槽位→候选→免渲染核验→轨迹计划,见文末
> 附录 G0–G6)与后半段"编译打包"(已渲染 bundle → 权威编译 → 四变体 →
> family,见第 0–7 站)。生成段用 seed=17 在 gates_bedroom 真实场景上
> 可逐字节复现 `batch_navfix/plan_0`(已验证),所以每站的期望值都和
> std.v3 的可通行数据对得上。旧 `batch/` 是 navfix 前的反例,仅供回归测试。

## 0. 启动配置

工作区打开 `code/EpiSpace`,`.vscode/launch.json` 加一条(解释器选 `.venv`):

```json
{
  "name": "family: render_0",
  "type": "debugpy",
  "request": "launch",
  "module": "spatial_episode.scriptgen.family_cli",
  "justMyCode": true,
  "args": [
    "--bundle", "/data/shichao/data/dataV100/code/OminiGibson/outputs/scripted_demo/batch_navfix/render_0",
    "--plan-record", "/data/shichao/data/dataV100/code/OminiGibson/outputs/scripted_demo/batch_navfix/plan_0.record.json",
    "--scene-ir", "/data/shichao/data/dataV100/code/OminiGibson/outputs/sweeps/t10-target-view-seed17-v1/bundles/gates_bedroom_t10_seed17/scene_ir.json",
    "--out", "/tmp/family_debug/f0", "--seed", "17"
  ]
}
```

输入固定为 render_0:16 帧,目标 armchair(entity `72faab69-…`,
runtime id 988247081),几何侧临时答案 left。

## 1. 装载:掩码后端就位

**断点 `family.py` → `build_family_site` 中 `doc = build_family_doc(...)` 一行**
(此时 `view` 已构造完成)。

- `view.layout.objects` 里有 armchair,`view.poses` 长度 16;
- `view.entity_runtime_ids["72faab69-77d3-53b7-9c1e-2feeb010c0f2"] == (988247081,)`。

调试控制台逐帧看权威可见性(这就是"掩码为准"的原始数据):

```python
[int(view.visibility("72faab69-77d3-53b7-9c1e-2feeb010c0f2", t).value) for t in range(16)]
```

预期 `[30018, 49160, 0, 0, ..., 0]`。第 1 帧仍高于可见阈值
900(std.v3 `render_min_visible_pixels`),所以权威 `t_seen` 比几何计划晚一帧。

## 2. 帧变量重解析:t_seen 0 → 1

**断点 `checker.py` → `resolve_frame_vars` 的 `return resolved`;
以及 `compiler.py` → `CapabilityCompiler.compile` 中 `report = check_clauses(...)` 一行。**

- 第一次进入(canonical):`resolved == {"t_seen": 1, "t_gone": 2, "t_q": 15}`;
  对照 plan record 里几何侧的 `{"t_seen": 0, ...}`——渲染后端覆盖几何估计,
  正是决策 #1 与"帧变量需渲后重解析"的落点;
- `last_visible` 解析器从帧 15 向前扫,停在第一个 tristate 为 True 的帧(1)。

## 3. 条款重判与权威答案

**断点 `compiler.py` → `derive_answer` 的 return 处。**

- `check_clauses` 以 `phases=("search","compile")、tighten_search=False`
  重判 6 条证据/有效性条款;`poses_clear/path_clear` 是生成期
  `search_only`,不会因置换或删帧变体的呈现顺序而重判;
- `report.passed is True`;witness 里 `turned.cum_turn_deg = 123.0`
  (从渲染重解析的 t_seen=1 起算,计划期从 t_seen=0 起算为 155.0);
- `derive_answer` 返回:`azimuth_deg=107.7, sector="left", margin_deg=27.3`,
  与几何 provisional 一致 → certificate `mismatch is None`。

## 4. 留一法 essential 帧集

**断点 `compiler.py` → `_essential_frames` 的 `essential.append(...)`。**

每次命中即"删掉该帧后编不过或答案变"。render_0 预期恰好命中 4 次,
`essential == (1, 2, 3, 12)`:

- 帧 1、2、3:分别删掉会合并相邻转向,产生 64°/64°/49.6° 跳变,
  超过 std.v3 的 40° 上限,`trackable` 失败;
- 帧 12:删掉后帧 11→13 合成约 41.4° 转向,同理;
- 帧 0 反而可删(目击冗余)——留一法语义:单帧冗余 ≠ 类别冗余,
  所以删关键帧算子删的是"整类"目击帧。

## 5. 四个干预算子(每个 gold 都来自重跑编译器)

**断点 `variants.py` → `VariantBuilder._verified` 中
`cert = self.compiler.compile(...)` 一行(每个变体都会经过这里)。**

按 build_all 顺序观察 `kind / sequence / cert`:

| 变体 | frame_sequence 特征 | 编译器结论 | gold |
|---|---|---|---|
| permute | 目击前缀 0–1 不动,2–14 乱序,15 收尾 | `abstain`,reason=`clause:trackable` | 无法判断 |
| drop_key | `(2,…,15)`,只剩确凿不可见帧 | `abstain`,reason=`frame_var_unresolvable:t_seen` | 无法判断 |
| drop_filler | 删 0、4(逐帧删后重编译验证) | `answerable`,sector 仍 left | left |
| delay | 帧 8 连续出现 5 次(原地停顿) | `answerable`,`knob_levels["delay"]` 13→17 | left |

配套细看点:

- permute 路径可在 `predicates.py` → `step_motion_bounded` 加断点,看
  `worst_step_turn_deg` / `worst_step_translation_m` 如何爆表;
- drop_key 路径在 `checker.py` → `_run_resolver` 的 `return None`(last_visible
  扫完无 True 帧),随后 `compiler.py` 的 `except FrameVarUnresolvable` 分支把
  它按 spec 的 `abstain_on_unresolvable=("t_seen",)` 归类为 abstain;
- 任何"预期 vs 实际"不符都会走到 `variants.py` → `raise FamilyMismatch`
  (正常运行不触发;想演示可临时把 spec 里 permute 的预期改成 "same")。

## 6. 打包审计与落盘

**断点 `family.py` → `_audit` 的 return 处,以及 `build_family_site` 的
`(out / "family.json").write_text(...)`。**

- `checks`:referent_unique=True(场景里 armchair 仅一件)、三项泄漏检查全 False
  (题面无未填占位符、无"第 N 帧"、不含 gold token);
- 落盘后 `/tmp/family_debug/f0/`:`family.json`(单文档,决策 #2)+
  `media/` 48 张 PNG(16 帧 × rgb/depth/instance,相对路径引用)+ `index.html`。

审核页:`cd /tmp/family_debug/f0 && python -m http.server 8000`,浏览器打开
`http://localhost:8000`——五条 filmstrip(变体序,帧下标注原始帧号、重复帧
标记)、金标/状态/预期 chips、可展开的条款见证、审计四绿灯。

## 7. 变奏(可选)

- 换输入:`--bundle render_1 --plan-record plan_1.record.json`(左,t_seen 无漂移)
  或 render_2(右)——三条轨迹金标 left/left/right;
- 阻断演示:把 launch args 的 plan-record 换成手改过 provisional sector 的副本,
  canonical 在 `family.py` 抛 `FamilyBlocked: canonical mismatch`;
- 全程无模拟器:整个调试只读 npz/json,不需要 Isaac Sim。

---

## 附录:生成段单步路线(剧本 → 轨迹计划)

### G0. 入口

生成段有两个入口,推荐第二个(真实场景,可复现 plan_0):

内置演示层(零数据依赖):launch.json 用
`"module": "spatial_episode.scriptgen.cli"`,
args `["--capability","self_motion_update","--seed","17","--out","/tmp/p.json"]`。

真实场景:`cli.py` 目前只认 DEMO_LAYOUT,所以写一个十行的驱动文件
(位置随意,如 `/tmp/gen_debug.py`;断点都打在 src 里,驱动文件在哪不重要):

```python
from spatial_episode.scriptgen.behavior import layout_from_scene_ir
from spatial_episode.scriptgen.generate import generate_plans
from spatial_episode.scriptgen.library import SELF_MOTION
from spatial_episode.scriptgen.standards import STD_V1

SCENE_IR = ("/data/shichao/data/dataV100/code/OminiGibson/outputs/sweeps/"
            "t10-target-view-seed17-v1/bundles/gates_bedroom_t10_seed17/scene_ir.json")
layout = layout_from_scene_ir(SCENE_IR)
report = generate_plans(layout, SELF_MOTION, STD_V1, seed=17,
                        attempts_per_binding=150, max_plans=1)
print(report.plans[0].plan_id, report.rejection_counts)
```

launch 配置用 `"program": "/tmp/gen_debug.py"`,解释器选 `.venv`。
预期输出:`…self_motion_update.b0.s17.a1 {'clause:gap': 1}`,
与 `batch_navfix/plan_0.record.json` 的 plan_id 一致,位姿逐字节相同。

**注意**:生成是共享一个 `random.Random(seed)` 的确定性过程,调试控制台里
不要执行任何会消耗 rng 的表达式(比如手滑调 motif),否则后续 attempt
的随机数错位,就复现不出 a1 了。看值可以,取样不行。

### G1. 场景装载:scene_ir → 可规划布局

**断点 `behavior.py` → `layout_from_scene_ir` 的 return 处。**

gates_bedroom 预期:`objects` 22 件、`obstacles` 22 件(非结构物的原始
旋转 OBB 与 z 跨度)、`occluders` 4 段(跨相机高度的墙体足印)、
`walkable` 由地板足印并集得出,约
(-0.69, -2.42) ~ (3.11, 4.33)。armchair 尺寸 0.96 m,entity id 72faab69 开头。

### G2. 槽位:谁有资格当目标物

**断点 `slotting.py` → `_qualifies` 的每个 return(或 `enumerate_bindings`
的最后一行)。**

22 件物体过筛,预期 8 个候选绑定、14 个拒绝:

- `too_small` × 11:两个茶瓶(0.07)、两个纸杯(0.09)、三个书柜(0.43)、
  台灯、垃圾桶、床头柜、矮的落地灯(0.45)——都不到 SlotSpec 的 0.5 m 下限;
- `ambiguous_referent` × 3:两把 straight_chair 互相取消资格,高的落地灯
  (0.7,过了尺寸关)因为场景里还有另一盏落地灯也被拒——题面要说
  "那盏落地灯"就有歧义,这正是 R1 扣信息原则在槽位层的体现;
- 剩下 8 件(armchair、bed、两张桌、desk、窗、sofa、swivel_chair)成为
  绑定候选,armchair 排第一 → 后面 plan_id 里的 `b0`。

### G3. 主循环:拒绝直到合格

**断点 `generate.py` → `_search_binding` 里 `report = check_clauses(...)`
之后一行;建议加条件断点分两种玩法:`report.passed` 只看成功,
或不加条件逐 attempt 看拒绝。**

seed=17、绑定 armchair 的逐 attempt 实录(帧数是 `rng.randint(10,16)` 抽的):

| attempt | 帧数 | 结局 | 见证 |
|---|---|---|---|
| 0 | 14 | `gap` 拒 | t_seen=9,t_gone=12,t_q=13,消失后只剩 1 帧,不足以测记忆延迟 |
| 1 | 16 | **通过** | t_seen=0,t_gone=2,t_q=15;pose/path 通行见证均 0 碰撞 → plan_0 |

margin_ok 在搜索期带 `tighten=True`,要求 15°×1.2=18° 裕度——所以 -149.4°
(裕度 14.4°)也被拒,这就是"搜索期收紧,渲后复核才有余量"的机制现场。
`rejection_counts` 最终是 `{'clause:gap': 1}`,
拒绝直方图不是装饰,是场景/剧本问题的第一现场。

### G4. motif:候选是怎么长出来的

**断点 `motifs.py` → `walk_and_turn` 的 return 处**(配合 G3 的条件断点
`attempt == 4` 只看成功那次)。

结构:先把 22 件障碍的旋转足印按 0.35 m 膨胀并光栅化到 5 cm 栅格;
起点只从目标 2–4 m 环带的自由格采样,终点从远处自由格采样,直线不通时
用 8 邻域 A* 绕行且禁止斜穿墙角;折线再分配到逐帧位姿,到达后原地环顾
(`look_frames = 16//4 = 4`)。每帧限速 32°(低于标准 40°)。成功起点为
`(1.7125, 0.3321, -137.3°)`,终点为 `(0.8125, 2.9321, 150.5°)`。
motif 的 0.35 m 只是保守提议,最终仍由下一站 std.v3 的 0.30 m 谓词裁决。

### G5. 免渲染核验:几何后端过条款

**断点 `checker.py` → `resolve_frame_vars` 的 return,和
`sceneview.py` → `GeometrySceneView.visibility`。**

成功 attempt 的帧变量:`{'t_seen': 0, 't_gone': 2, 't_q': 15}`。
在 visibility 里能看到几何三态的来历:第 0 帧椅子整体在视锥内、
尺寸/距离比 0.3946 ≥ 0.10 → 可见;第 1 帧比值 0.08 落在几何模糊带,
第 2 帧起不可见。渲染实例掩码则确认第 1 帧仍有 49160 像素,所以渲后
t_seen 从 0 改为 1。最先执行的两条通行条款见证均为
`collision_count=0`;`sector_margin_ge` 见证为 107.7°、left、
裕度 27.3° ≥ 搜索期要求 18°。

### G6. 计划落成

**断点 `generate.py` → `return TrajectoryPlan(...)`。**

核对四件事:`plan_id` 尾缀 `b0.s17.a1`(绑定 0、seed 17、attempt 1);
`provisional_answer` = left / 107.7° / 27.3°,字段名就叫 provisional——
它要等渲染后被编译器重新裁决;`clause_witnesses` 里包含 poses/path 两份
零碰撞见证,turned 是 155.0°(从几何 t_seen=0 起算;渲后 t_seen=1 时为
123.0°);`knob_levels` delay=13。这份对象序列化后就是
`batch_navfix/plan_0.record.json`,也是后半段路线图第 1 站的输入——
两条路线在这里接上。
