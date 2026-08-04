# 真实模型输出解析

`episode3d.model_output_parser` 把模型的自由中文回答转换成
`benchmark_evaluator` 所需的结构化 `answer_value`。它是模型推理与冻结评测器之间的窄接口，
不是另一个 oracle：解析器不读取答案布尔值、空间关系、`answer_status`、参考答案文本或几何
certificate。

## 调用契约

```python
from episode3d.model_output_parser import parse_model_output

prediction = parse_model_output(benchmark_row, model_generated_text)
```

成功时返回的对象可直接作为 prediction JSONL 的一行：

```json
{
  "record_id": "benchmark-...",
  "answer_value": {"claim_correct": false, "relation": "in_front_of"},
  "parse_status": "parsed",
  "parser_schema_version": "epispace.model_output_parser.v1"
}
```

评测器只消费 `record_id` 与 `answer_value`，其余字段用于审计。解析失败不会被丢弃，也不会
猜一个标签，而是显式产生：

```json
{
  "record_id": "benchmark-...",
  "answer_value": {"status": "invalid"},
  "parse_status": "invalid",
  "parse_error": "missing_or_ambiguous_relation",
  "parser_schema_version": "epispace.model_output_parser.v1"
}
```

`{"status":"invalid"}` 不属于 core 任一目标 schema，因此会被现有评测器稳定计为错误。

## Core 五类映射

| task | 模型文本例子 | 结构化输出 |
|---|---|---|
| `counterfactual_verification` | `不对；实际在前方。` | `{"claim_correct":false,"relation":"in_front_of"}` |
| `egocentric_relation` | `餐桌在书架右侧。` | `"right_of"` |
| `evidence_presence_unknown` | `证据不足，还不能确定。` | `null` |
| `evidence_presence_reveal` | `能确定，画面中能看到钢琴。` | `{"status":"present","category":"piano"}` |
| `target_view_prediction` | `看不到。` | `false` |

关系是闭集：`left_of`、`right_of`、`in_front_of`、`behind`。`左前方` 同时命中两个
轴，不会被擅自压成一个标签，而是 `invalid`。claim 任务必须同时从输出中解析出
`claim_correct` 和实际 relation；模型只说“对”时不能从 target 补关系。

unknown 任务要求模型明确表达认识不确定性，例如“无法确定”“证据不足”。“没有看到”或
“不存在”不等同于“依据当前视图无法确定”，不会被解析为 `null`。同理，target-view 中的
“无法确定”不是 `false`，而是 `invalid`。

解析器同时接受简洁自然语言、整个回答为 JSON、Markdown JSON code fence，以及
`最终答案：...` 形式。存在互相冲突的方向或可见性信号时 fail closed。

## 无泄漏边界

允许读取的 row 信息只有：

1. `schema_version`；
2. `target.task_type` 与 `program.program_id`，用于选择输出 grammar 并检查 task/schema
   一致性；
3. 问题给定的 query 常量。

冻结的 `epispace.benchmark.v1` 尚未为 presence-reveal 单独序列化
`query_arguments.category`。因此有且仅有该任务会把
`target.answer_value.category` 作为非决策 schema 常量读取；这个类别已经在自然语言问题
中给出。解析器优先读取未来兼容的 `query_arguments.category`，且绝不读取同一对象中的
`status`。presence 的“看见/未知”仍必须来自模型输出。

其余任务不会访问 `target.answer_value`。测试使用会在读取隐藏 decision 字段时直接抛异常的
Mapping，并通过把 target 布尔值、claim correctness 和 relation 改成相反的 poison 值，验证
解析结果只随模型文本变化。

## 推荐推理落盘流程

对每个 benchmark row 调用模型，保留原始文本，再调用解析器。实验目录建议同时保存：

- `raw_generations.jsonl`：`record_id`、完整 prompt/version、原始输出、采样参数；
- `predictions.jsonl`：本解析器返回的对象；
- `evaluation.json`：冻结 evaluator 的报告。

不要在模型输出后调用任何 certificate、geometry 或 `target.answer_value` 修正预测。解析器版本、
模型 revision、prompt hash 和 benchmark hash 应一起冻结，保证复现实验时能区分模型错误与
parser 错误。

## 验证

```bash
ruff check episode3d/model_output_parser.py tests/test_model_output_parser.py
pytest -q tests/test_model_output_parser.py tests/test_benchmark_evaluator.py
```
