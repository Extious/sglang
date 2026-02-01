# Multi-Agents-Parallel-Orchestration-Dataset 解析说明

来源: [DeepNLP/Multi-Agents-Parallel-Orchestration-Dataset](https://huggingface.co/datasets/DeepNLP/Multi-Agents-Parallel-Orchestration-Dataset)

## 1. 场景与内容

- **场景**：多智能体并行编排（Parallel）与顺序编排（Sequential）。用于 Deep Research、行程规划、多源搜索等任务，同一用户问题由多个 Agent 并行或按计划顺序执行。
- **内容**：每行一个 JSON 对象，表示一次**会话（session）**。会话内可包含多条 **trace**（每个 trace 对应一个 Agent 的一次运行）。Parallel 时多条 trace 对应多个 Agent 并行回答同一问题；Sequential 时由 Planning Agent 生成 plan，再按顺序执行各 Agent。

## 2. 文件格式

- **example_deepnlp_multi_agents_202601.json**：多行，每行一个 JSON。
- **每行结构**：`{ "<session_id>": { "<trace_id_1>": record1, "<trace_id_2>": record2, ... } }`
  - 顶层唯一键：`session_id`（一次多轮/多 Agent 对话的标识）。
  - 内层键：多个 `trace_id`（每个 Agent 单次运行一条 trace；Parallel 时同一 session 下多条 trace）。

## 3. 单条 record 字段（每个 trace_id 对应值）

| 字段 | 类型 | 说明 |
|------|------|------|
| `model` | string | 所用模型，如 `qwen3-vl-8b-instruct`, `qwen3-max-preview`, `qwen3-coder-flash` |
| `session_id` | string | 与顶层键一致 |
| `trace_id` | string | 本 trace 的 ID |
| `function_calls` | array | 该 Agent 运行中的多轮「消息 + 工具调用」列表，见下表 |
| `plan` | array | （仅 Sequential）Planning Agent 输出的任务分解与 Agent 分配列表 |

**function_calls[i]** 为一次「LLM 调用 + 工具调用」的上下文：

| 字段 | 类型 | 说明 |
|------|------|------|
| `messages` | array | 本轮对话消息，元素为 `{"role":"user"|"assistant"|"tool", "content":..., ...}`；assistant 消息可含 `tool_calls` |
| `tools` | array | 本轮可用工具列表（name, description, parameters 等 schema） |
| `tool_calls` | array | （若为 assistant 消息）LLM 输出的工具调用，含 `id`, `type`, `function`（name, arguments） |

**messages 典型顺序**：user（任务/问题）→ assistant（tool_calls）→ tool（各 tool_call_id 的返回）→ assistant（下一轮）→ … 循环直至结束。

## 4. Prompt 结构

- **任务级 prompt**：每条 trace 的第一次 LLM 输入来自 `function_calls[0].messages` 中第一条 `role: user` 的 `content`（如 "Plan a trip from Hangzhou to Beijing...", "Research what's the difference between AI Agent Skills and MCPs?"）。
- **单次 LLM 调用**：由 `function_calls[k].messages` 组成多轮上下文；通常包含 system（若有）、user、assistant、tool 消息；工具调用格式为 OpenAI 风格（tool_calls + tool 消息）。

## 5. 当前 example 文件统计（解析结果）

- **Sessions**: 4
- **Traces**: 9（同一 session 下多 trace = 多 Agent 并行）
- **Models**: qwen3-vl-8b-instruct, qwen3-max-preview, qwen3-coder-flash
- **function_calls 每 trace**：1～3 轮，平均 2
- **plan**：当前 example 中无（仅 Parallel 样例）

## 6. 解析脚本

在本目录（`Multi-Agents-Parallel-Orchestration-Dataset/`）下执行：

```bash
python parse_parallel_orchestration.py
python parse_parallel_orchestration.py /path/to/example_deepnlp_multi_agents_202601.json -v
python parse_parallel_orchestration.py -o summary.json
```

或从 `benchmark/multi_agent/` 运行：`python Multi-Agents-Parallel-Orchestration-Dataset/parse_parallel_orchestration.py`。

输出：控制台打印 sessions/traces/models/function_calls 统计；`-v` 输出每条 record 的简要摘要（含 first_prompt）；`-o` 将完整 summary 写入 JSON。
