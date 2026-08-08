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
