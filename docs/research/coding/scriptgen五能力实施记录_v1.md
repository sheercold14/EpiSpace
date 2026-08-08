# Scriptgen 五能力实施记录 v1

> 对应 `scriptgen五能力实施计划_v1.md`。本文件逐阶段记录实际产出、
> 验证证据和相对计划的偏离；金标始终以编译器重算结果为准。

## 开工裁决（计划 §6）

按计划建议值执行，不另行等待确认：

1. 偏科验收线：left/right 各不低于 25%，back 不低于 15%；
2. 组合二物物关系：采用 Y 的内在朝向；
3. 基元 B：置换导致弃答；
4. 基元 A：深档和浅档都实现；
5. 题型 10：采用统一保守隐藏物尺寸；
6. 净转向：增加大于 90° 的幅度档。

## P0：§3.1 前四项阻塞修复

产出：

- 生成阶段的临时答案改为调用答案模式注册表，与编译器共用唯一分发
  路径；plan 临时答案改为通用的 `mode/label/witness` 结构；
- spec 显式声明 `intervention_window`，置换与延迟算子不再猜测
  `t_gone` 或使用魔法兜底；
- library 构造器为所有带 motif 的 spec 强制附加
  `poses_clear/path_clear` 搜索期通行条款；
- 编译、证书和打包显式记录并使用 `template_index`，不再暗取
  `templates[0]`；
- 对应契约升版：spec v5、plan v2、certificate v3、family v3；
  判定阈值未变化，因此 standard 保持 std.v4。

验证：`tests/scriptgen` 共 84 项全绿；三条 wallfix 轨迹的自运动方向
金标和既有数值钉值保持不变（left / left / right）。

偏离：无。

## P1：基元 C / 组合一扩展

产出：

- `walk_and_turn` 显式抛硬币选择转向符号，并校准终点相对方位；120 条
  确定性样本得到 left 42 / right 50 / back 28，分别为
  35.0% / 41.7% / 23.3%，通过 25% / 25% / 15% 预注册门槛；
- 新增 T2 `stand_and_turn` + `displacement_below` +
  `self_motion_update_pure_rotation`，位移 witness 为 0.0 m；
- 新增 T3 `walk_straight_past` + `turn_below` +
  `self_motion_update_pure_translation`，累计转角 witness 为 0.0°；
- 新增 T4 `walk_multi_turn` + `turn_segments_between` +
  `self_motion_update_multi_turn`，2/3 两档转向段均有确定性样本；
- 新增 T5 `walk_to_occlusion` + `occluded_in_view` +
  `self_motion_update_occluded`；遮挡物由标签白名单改为“顶面达到
  1.5 m 视线高度”，demo 金标为 front；
- 按裁决增加净转角 90° 幅度档：`net_turn_magnitude` 答案模式、
  裕度谓词和 `path_integration_magnitude` spec；
- family 组装自动以 source spec 为主问题，并机会性附挂净转向方向、
  幅度、指回起点和画面侧，仍由各自条款决定成题或跳过；
- 新阈值集中进入 standards，版本升为 std.v5，并显式声明对
  std.v4/std.v3 旧计划为纯扩展兼容。

验证：

- `tests/scriptgen` 共 99 项全绿；四个新子型逐一验证 spec、motif、
  谓词、编译器金标和完整变体矩阵；
- demo 的首条编译器输出依次为 T2 left、T3 back、T4 back、T5 front；
- Beechwood 多房间现有 `scene_ir` 上 T2–T5 均生成计划，
  `poses_clear/path_clear` 碰撞计数均为 0；
- 三条 wallfix 旧轨迹自运动金标及数值钉值仍为 left / left / right，
  新增幅度题均由编译器得到 at_most_90。

偏离：按“不得启动 Isaac Sim”的硬约束，只做了 EpiSpace 侧 OBB 人体
半径通行复核；现有只读 bundle 没有 navmesh/胶囊审计产物，因此计划
§5 所列仓外 navmesh/胶囊二次复核未执行。新子型尚无渲染 bundle，
对应 needs_batch 掩码复核待后续既有渲染管线产生产物后自动启用；本阶段
没有手写或修改任何金标。

## P2：基元 A（参考系变换）

产出：

- 新增 `survey` motif：在同一自由站位以不超过逐帧转向上限的速度完成
  360° 环视，实际位移为 0；
- 新增 `all_landmarks_visible`、`never_all_covisible`、
  `imagined_pose_valid` 和深/浅两档曲线裕度谓词；每个 P/Q/X 至少有
  2 个清晰目击帧，并禁止三者在任何单帧同时清晰可见，显式堵住静态
  单帧旁路；
- 新增 `imagined_sector` 与 `imagined_visibility` 答案模式；前者从
  P 坐标和 P→Q 朝向构造想象位姿，后者经 `GeometrySceneView` 的任意
  构造位姿入口计算视锥与遮挡；
- 深档和浅档各声明 0/45/90/135/180° 五个 spec，共用同一 survey
  录像和绑定。协变不是手改 family 金标，而是每个角度 spec 重新调用
  编译器；帧置换/删目击/删冗余/延迟预期分别为
  same/abstain/same/same，并由 VariantBuilder 复核；
- family 升为 v4，增加全槽位 `referents`，题面按绑定逐一填充 P/Q/X，
  媒体导出覆盖所有绑定实例；问题组登记处将同一 A 轨迹自动编排为
  10 个角度×档位问题；
- 新阈值进入 standards，版本升为 std.v6，并显式保持对
  std.v3–std.v5 的纯扩展兼容。

验证：

- `tests/scriptgen` 共 106 项全绿；合成固定几何上深档五点金标为
  right / right / back / back / left，方位角逐点严格按 -offset 协变；
- Beechwood 真实多房间 `scene_ir` 上生成无碰撞 survey，深档五点为
  front / right / right / back / back，浅档为
  visible / not_visible / not_visible / not_visible / not_visible；上述钉值
  均直接抄录编译器输出；
- 置换保持答案，删尽 X 的目击帧由 `landmark_evidence` 构造性地产生
  弃答；三条 wallfix 的既有 left / left / right 金标和数值钉值不变。

偏离：计划中的浅档“附加想象视图渲染掩码复核”需要仓外渲染器产出
not-in-sequence 视图；受不得启动 Isaac Sim 的硬约束，本阶段只完成
EpiSpace 内几何权威链和任意位姿查询入口，未伪造掩码或手写复核结果。
角度协变采用同一问题组中的声明式 sibling specs，而没有扩张既有四种
帧手术算子；每个角度仍经过同一编译器和 spec 预期核对。
