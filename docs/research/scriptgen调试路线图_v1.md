# Scriptgen 空缺A/B/C 单步调试路线图(VSCode)

> 目的:人工在调试器里走一遍 "已渲染 bundle → 权威编译 → 四变体 → family 打包"
> 全流程,逐站核对关键值。对应提交 cceb13d(A)/ 9beee33(B)/ 5a67ecd(C)。

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
    "--bundle", "/data/shichao/data/dataV100/code/OminiGibson/outputs/scripted_demo/batch/render_0",
    "--plan-record", "/data/shichao/data/dataV100/code/OminiGibson/outputs/scripted_demo/batch/plan_0.record.json",
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

预期 `[51509, 71712, 11579, 4212, 0, 0, ..., 0]`——注意第 2、3 帧仍高于
可见阈值 900(std.v2 `render_min_visible_pixels`),这是后面 t_seen 漂移的根源。

## 2. 帧变量重解析:t_seen 0 → 3

**断点 `checker.py` → `resolve_frame_vars` 的 `return resolved`;
以及 `compiler.py` → `CapabilityCompiler.compile` 中 `report = check_clauses(...)` 一行。**

- 第一次进入(canonical):`resolved == {"t_seen": 3, "t_gone": 4, "t_q": 15}`;
  对照 plan record 里几何侧的 `{"t_seen": 0, ...}`——渲染后端覆盖几何估计,
  正是决策 #1 与"帧变量需渲后重解析"的落点;
- `last_visible` 解析器从帧 15 向前扫,停在第一个 tristate 为 True 的帧(3)。

## 3. 条款重判与权威答案

**断点 `compiler.py` → `derive_answer` 的 return 处。**

- `check_clauses` 以 `phases=("search","compile")、tighten_search=False`
  重判全部 6 条条款(搜索期的 1.2× 收紧此时关闭,用足额 15° 裕度);
- `report.passed is True`;witness 里 `turned.cum_turn_deg ≈ 102.5`
  (从渲染重解析的 t_seen=3 起算,不再是计划记录里的 145.8);
- `derive_answer` 返回:`azimuth_deg=108.0, sector="left", margin_deg=27.0`,
  与几何 provisional 一致 → certificate `mismatch is None`。

## 4. 留一法 essential 帧集

**断点 `compiler.py` → `_essential_frames` 的 `essential.append(...)`。**

每次命中即"删掉该帧后编不过或答案变"。render_0 预期恰好命中 3 次,
`essential == (1, 12, 13)`:

- 帧 1:删掉后 0→2 帧的偏航跳变 43.3° > 40°(std.v2 `max_step_turn_deg`),
  `trackable` 失败;
- 帧 12、13:两个 32° 转身步合并成 64°,同理;
- 帧 0、2 反而可删(目击冗余)——留一法语义:单帧冗余 ≠ 类别冗余,
  所以删关键帧算子删的是"整类"目击帧。

## 5. 四个干预算子(每个 gold 都来自重跑编译器)

**断点 `variants.py` → `VariantBuilder._verified` 中
`cert = self.compiler.compile(...)` 一行(每个变体都会经过这里)。**

按 build_all 顺序观察 `kind / sequence / cert`:

| 变体 | frame_sequence 特征 | 编译器结论 | gold |
|---|---|---|---|
| permute | 前缀 0–3 不动,4–14 乱序,15 收尾 | `abstain`,reason=`clause:trackable` | 无法判断 |
| drop_key | `(4,…,15)`,只剩确凿不可见帧 | `abstain`,reason=`frame_var_unresolvable:t_seen` | 无法判断 |
| drop_filler | 删 0、2(逐帧删后重编译验证) | `answerable`,sector 仍 left | left |
| delay | 帧 9 连续出现 5 次(原地停顿) | `answerable`,`knob_levels["delay"]` 11→15 | left |

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
