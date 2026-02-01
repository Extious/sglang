# Multi-Agent Benchmark Datasets

本目录包含六个多智能体/智能体相关评测数据集：**MAST-Data**、**TRAIL**、**agentcode**、**Multi-Agents-Parallel-Orchestration-Dataset**、**multiagent-router-finetuning**、**system-prompts-multi-agent-systems**。下文分别介绍各数据集的内容、适用场景与 Prompt 结构，并给出下载与工具用法。

---

## 数据集一览

| 数据集 | 来源 | 场景 | 规模/格式 |
|--------|------|------|-----------|
| **MAST-Data** | [mcemri/MAST-Data](https://huggingface.co/datasets/mcemri/MAST-Data) | 多智能体系统（ChatDev、AppWorld、MetaGPT、AG2、HyperAgent 等）执行轨迹，按 MAST 失败分类标注 | JSON 数组；full + human 标注子集 |
| **TRAIL** | [PatronusAI/TRAIL](https://huggingface.co/datasets/PatronusAI/TRAIL) | GAIA 问答、SWE-Bench 修 bug；带错误定位与影响程度标注的智能体 trace | Parquet + GAIA/SWE Bench 下 JSON trace + 标注 JSON |
| **agentcode** | [AlignmentLab-AI/agentcode](https://huggingface.co/datasets/AlignmentLab-AI/agentcode) | 智能体与代码相关指令/对话，用于 CoT、指令遵循等 | JSONL，约 22 万条，~465 MB |
| **Multi-Agents-Parallel-Orchestration** | [DeepNLP/Multi-Agents-Parallel-Orchestration-Dataset](https://huggingface.co/datasets/DeepNLP/Multi-Agents-Parallel-Orchestration-Dataset) | 多智能体并行/顺序编排（Deep Research、行程规划等）；每行 JSON 为 session，内有多条 trace（并行 Agent） | 多行 JSON，每行 `session_id -> trace_id -> record`；record 含 function_calls（messages/tools/tool_calls） |
| **multiagent-router-finetuning** | [bhaiyahnsingh45/multiagent-router-finetuning](https://huggingface.co/datasets/bhaiyahnsingh45/multiagent-router-finetuning) | 多智能体客服路由微调：意图分类、参数抽取、将 query 路由到对应 Agent（technical_support / billing / product_info） | HF Dataset / Parquet；115 条，train 92 / test 23；字段：system_message, agent_name, agent_arguments, 用户问题 |
| **system-prompts-multi-agent-systems** | [kimcomehome/system-prompts-multi-agent-systems](https://huggingface.co/datasets/kimcomehome/system-prompts-multi-agent-systems) | 多智能体系统相关 system prompt 数据 | HF Dataset / Parquet；100 条；具体字段见下载后数据 |

---

## 一、MAST-Data (MAST-Data/)

### 1.1 内容与场景

- **内容**：多智能体系统（MAS）的完整执行轨迹（含任务 prompt、多轮对话、代码执行、评测结果），并按 **MAST (Multi-Agent Systems Failure Taxonomy)** 做失败模式标注。
- **场景**：
  - **ChatDev**：多阶段角色对话（CEO/CPO/CTO/Programmer 等），按阶段完成需求分析 → 语言选择 → 编码 → CodeReview → 文档等；每阶段为多轮 role-playing chat，每轮对应一次 LLM 调用。
  - **AppWorld**：层级式消息传递；Supervisor Agent 编排子 Agent（如 spotify），通过 `send_message` 与代码执行交替，多轮 Supervisor ↔ 子 Agent 对话。
  - **MetaGPT**：线性角色流水线；任务广播后按角色顺序（SimpleCoder → SimpleTester → SimpleReviewer 等）依次产生输出，每角色对应 LLM 调用。
  - **AG2 / HyperAgent**：单次或短链；一条 `problem_statement`（或 SWE-Bench 风格 issue）驱动一次或少数几次 LLM 调用（推理+代码/工具调用）。

### 1.2 Prompt 结构

- **任务级 prompt**：每条 trace 对应一个**任务描述**（如「开发一个跳棋游戏」「在 Spotify 里切歌直到某首」）。在原始 `trace.trajectory` 中通常出现在：
  - **ChatDev**：`**task_prompt**:` 后紧跟的文本；在 `trajectory_expanded.metadata` 中为 `task_prompt` 字段。
  - **AppWorld**：`**** Task N/M (id) ****` 后一行即为 `task`。
  - **AG2/HyperAgent**：`metadata.problem_statement` 或轨迹中的 `problem_statement`。
- **单次 LLM 调用的 prompt 结构**：
  - **ChatDev**：每阶段 = 固定 **role_prompt（user 角色 + assistant 角色）+ phase_prompt**，占位符如 `{task}`, `{codes}`, `{language}` 等由上游阶段填充；即 `background_prompt + user_role_prompt + assistant_role_prompt + phase_prompt`，组成发给 LLM 的 system/user messages。
  - **AppWorld**：Supervisor 与各子 Agent 的输入为「当前任务/上下文 + 收到的 message」，轨迹中以 `message_to_agent` / `response_from_agent` 等块呈现。
  - **MetaGPT**：`[FROM: Human TO: {'<all>'}]` + `CONTENT: task_prompt` 后，各角色按序收到上下文并生成 `NEW MESSAGES`。
- **提取方式**：脚本 `extract_prompt_flow.py` 可从 `MAD_full_dataset.json` / `MAD_human_labelled_dataset.json` 中解析出 `task_prompt`，输出到 `mast_full_prompt_flow.json` / `mast_human_prompt_flow.json`。脚本 `expand_mad_full_trajectory.py` 将 `trace.trajectory` 拆成 `trajectory_expanded`（含 `metadata`、`sections`、按 mas 的 phase/section 结构），便于按段分析 LLM 输入。

### 1.3 文件与字段速查

| 文件 | 说明 |
|------|------|
| `MAD_full_dataset.json` | 每条：`mas_name`, `llm_name`, `benchmark_name`, `trace_id`, `trace`（含 `key`, `index`, `trajectory` 长文本）, `mast_annotation`（1.1～3.3 二值） |
| `MAD_human_labelled_dataset.json` | 每条：`round`, `mas_name`, `benchmark_name`, `trace_id`, `trace` 文本, `annotations`（各 failure mode + annotator_1/2/3 布尔） |

MAST 失败模式编号（1.1～3.3）：1.x 推理/任务理解，2.x 执行/规划，3.x 协作；详见原数据集或 MAST taxonomy。

---

## 二、TRAIL (TRAIL/)

### 2.1 内容与场景

- **内容**：**Trace Reasoning and Agentic Issue Localization**。来自 **GAIA** 与 **SWE-Bench** 的智能体执行 trace（OpenTelemetry 风格 span 树），并配有**错误定位**（category、location=span_id、evidence、impact）与**多维度评分**（reliability、security、instruction_adherence、plan_opt、overall）。
- **场景**：
  - **GAIA**：多步推理/工具调用以回答复杂问题；trace 中常带 `question`、`task_id`、`true_answer` 等；适合评估「任务理解 → 检索/推理 → 答案」的链路及错误发生位置。
  - **SWE-Bench**：根据 issue/problem_statement 在仓库中修 bug；trace 含 `problem_statement`、`instance_id`、`patch`、repo 等；适合评估代码修改与测试通过情况，以及错误所属 span。

### 2.2 Prompt 结构

- **任务级 prompt**：
  - **GAIA**：根任务或主问题多在 `spans[].logs[].body.question` 或 `body.function.arguments` 中的 `question`；也可从根 span 的 `span_attributes` 或 logs 中取。
  - **SWE-Bench**：`problem_statement` 与任务描述常在 `body.function.arguments.item.problem_statement`、`item.question` 等路径。
- **单次 LLM 调用的 prompt**：
  - 最常见位置：**`spans[].span_attributes.input.value`**，多为 JSON 字符串，内含 `messages` 数组（如 `[{"role":"system","content":"..."},{"role":"user","content":"..."}]`）。
  - 其次：**`spans[].logs[].body.function.arguments`**（如 `CodeAgent.run` 等 span 的入参）。
  - 定位技巧：找 `span_attributes` 中 `input.mime_type: application/json` 或 `openinference.span.kind: LLM` 的 span，其 `input.value` 即该次 LLM 输入。
- **提取方式**：`extract_prompt_flow.py` 支持 `--dataset trail_gaia` / `trail_swe`，输出 `trail_gaia_prompt_flow.json` / `trail_swe_bench_prompt_flow.json`（含 trace_id、main_task/question、problem_statement、prompt_flow 等）。

### 2.3 文件与字段速查

| 路径 | 说明 |
|------|------|
| `data/gaia-*.parquet`, `data/swe_bench-*.parquet` | 列 `trace`, `labels`（序列化 trace/标注），GAIA 约 117 条，SWE-Bench 约 31 条 |
| `GAIA/*.json`, `SWE Bench/*.json` | 按 trace_id 命名的原始 trace：`trace_id`, `spans`（OpenTelemetry span 树；span 含 `span_id`, `span_attributes.input.value`, `logs[].body` 等） |
| `processed_annotations_gaia/*.json`, `processed_annotations_swe_bench/*.json` | `trace_id`, `errors[]`（category, location=span_id, evidence, description, impact）, `scores[]`（各维度分数与 overall） |

关联方式：`data/*.parquet` 的 trace_id → `GAIA/*.json` 或 `SWE Bench/*.json` 得 span 树；再用 `processed_annotations_*` 的 `errors[].location`（span_id）对齐到具体 span。

---

## 三、agentcode (agentcode/)

### 3.1 内容与场景

- **内容**：智能体与代码相关的指令/对话数据，属 AlignmentLab-AI 的 **"Double-the-data"** 系列，可与 chain-of-thought、指令遵循等训练或评测配合使用。
- **场景**：通用「智能体 + 代码」类任务（具体任务类型以 HF 或下载后样本为准）；HF 上暂无详细 dataset card，建议下载后查看 `agentcodeclean.jsonl` 前几行确定字段与 prompt 结构。

### 3.2 文件与规模

| 文件 | 说明 |
|------|------|
| `agentcodeclean.jsonl` | 主数据，JSONL，每行一条样本；约 221,095 条，约 465 MB |

### 3.3 下载

在 `benchmark/multi_agent/` 下执行：

```bash
./download.sh agentcode   # 仅 agentcode
./download.sh all         # 全部六个数据集
```

---

## 四、Multi-Agents-Parallel-Orchestration-Dataset

### 4.1 内容与场景

- **内容**：多智能体**并行**（Parallel）与**顺序**（Sequential）编排的 workflow 数据。每行一个 JSON，表示一次会话（session）；会话内可有多条 trace，每条 trace 对应一个 Agent 的一次运行（Parallel 时多 Agent 并行回答同一问题）。
- **场景**：Deep Research（多源搜索）、行程规划（地图/地点 MCP）、多 Agent 顺序执行（Planning Agent 生成 plan 后按序执行各 Agent）。每条 record 含 `function_calls`：messages（user/assistant/tool）+ tools + tool_calls（OpenAI 风格）。

### 4.2 Prompt 结构

- **任务级 prompt**：`function_calls[0].messages` 中第一条 `role: user` 的 `content`（如 "Plan a trip from Hangzhou to Beijing...", "Research what's the difference between AI Agent Skills and MCPs?"）。
- **单次 LLM 调用**：由 `function_calls[k].messages` 组成多轮上下文；assistant 消息可含 `tool_calls`；工具返回以 `role: tool` 消息形式追加。

### 4.3 文件与解析

| 路径 | 说明 |
|------|------|
| `Multi-Agents-Parallel-Orchestration-Dataset/example_deepnlp_multi_agents_202601.json` | 多行 JSON，每行 `{ session_id: { trace_id: record } }`；record 含 `model`, `session_id`, `trace_id`, `function_calls`（及 Sequential 时的 `plan`） |
| `Multi-Agents-Parallel-Orchestration-Dataset/DATA_STRUCTURE.md` | 字段与解析说明 |
| `Multi-Agents-Parallel-Orchestration-Dataset/parse_parallel_orchestration.py` | 解析脚本：统计 sessions/traces/models/function_calls，可选输出每条 record 摘要 |

下载：`./download.sh parallel`。解析：在 `Multi-Agents-Parallel-Orchestration-Dataset/` 下执行 `python parse_parallel_orchestration.py`，或 `python Multi-Agents-Parallel-Orchestration-Dataset/parse_parallel_orchestration.py -v -o summary.json`（从 `benchmark/multi_agent/` 运行）。

---

## 五、multiagent-router-finetuning (multiagent-router-finetuning/)

### 5.1 内容与场景

- **内容**：用于**多智能体客服路由**微调的合成数据。每条样本包含：系统 prompt、用户问题、**目标 Agent 名称**（路由标签）、**Agent 专用参数**（agent_arguments）。适用于训练「意图分类 + 参数抽取 + 路由到对应 Agent」的模型（如 FunctionGemma 等）。
- **场景**：客服场景下将 query 路由到三类 Agent：**technical_support_agent**（技术问题、bug、集成）、**billing_agent**（支付、订阅、发票）、**product_info_agent**（功能、方案、集成与合规）。含少量边缘/模糊 query 以测试鲁棒性。

### 5.2 Prompt 与字段

- **user_content** (string)：用户请求/问题文本。
- **agent_name** (string)：目标 Agent，取值为 `technical_support_agent`、`billing_agent`、`product_info_agent`。
- **agent_arguments** (string)：该 Agent 所需参数的 JSON 字符串（如 `{"issue_type":"crash","priority":"high"}`；billing 含 urgency 等，product_info 含 topic 等）。HF 卡片中另有 system_message，若使用 Parquet 版本可含该系统 prompt 列。

### 5.3 文件与规模

- 下载后位于 `multiagent-router-finetuning/`；主数据为 `dataset.json`（JSON 数组，115 条），亦可使用 HF 自动转换的 Parquet（含 train/test 划分）。
- 规模：115 条，train 92 / test 23；约 47.5 kB。

下载：`./download.sh router`。

---

## 六、system-prompts-multi-agent-systems (system-prompts-multi-agent-systems/)

### 6.1 内容与场景

- **内容**：多智能体系统相关的 **system prompt** 数据；HF 上 README 为空，具体用途与字段需下载后查看数据列。
- **规模**：100 条，约 48.5 kB；格式为 HF Dataset（Parquet 等）。

下载：`./download.sh system_prompts`。下载后查看 `system-prompts-multi-agent-systems/` 内文件或列名以确认字段与 prompt 结构。

---

## 七、工具与脚本

### 7.1 下载

```bash
./download.sh [mast|trail|agentcode|parallel|router|system_prompts|all]   # 默认 all
```

### 7.2 Prompt 流程提取（MAST-Data、TRAIL）

在 `benchmark/multi_agent/` 下：

```bash
python extract_prompt_flow.py                    # 全部
python extract_prompt_flow.py --dataset mast_full
python extract_prompt_flow.py --dataset mast_human
python extract_prompt_flow.py --dataset trail_gaia
python extract_prompt_flow.py --dataset trail_swe
python extract_prompt_flow.py --data-dir /path/to/multi_agent --output-dir /path/to/out
```

输出：`mast_full_prompt_flow.json`, `mast_human_prompt_flow.json`, `trail_gaia_prompt_flow.json`, `trail_swe_bench_prompt_flow.json`（字段见上文各节）。

### 7.3 MAD_full trajectory 展开（MAST-Data）

将 `MAD_full_dataset.json` 的 `trace.trajectory` 解析为结构化 `trajectory_expanded`（按 mas 的 metadata、sections、evaluation 等）：

```bash
python expand_mad_full_trajectory.py
python expand_mad_full_trajectory.py --keep-trajectory
python expand_mad_full_trajectory.py --input /path/to/in.json --output /path/to/out.json
```

按 `mas_name` 的展开结构概览：ChatDev → metadata（task_prompt 等）+ sections（DemandAnalysis/Coding/CodeReview 等）；AppWorld → task + sections（response_from_agent, message_to_agent, code_execution 等）+ evaluation；AG2/HyperAgent → problem_statement + sections；其余 → sections 按日志块划分。

### 4.4 run.sh（多 worker + router）

`run.sh` 会启动 10 个 Qwen3-4B-Thinking worker（每 GPU 一个）并在端口 30000 启动 router 做负载均衡。**前置条件**：需安装 router 包（Python 模块名 `sglang_router`）。

**推荐：直接安装 PyPI 预构建 wheel（无需 Rust 编译）：**

```bash
pip install sglang-router
```

若需从源码构建（需 Rust 环境）：

```bash
cd sgl-model-gateway/bindings/python
pip install maturin
maturin develop --features vendored-openssl
```

安装完成后在 `benchmark/multi_agent/` 下执行 `./run.sh`。

---

## 八、TRAIL 中 LLM Prompt 定位小结

在 GAIA / SWE Bench 的 span 中，输入给 LLM 的 prompt 常见位置：

1. **`spans[].span_attributes.input.value`**：JSON 字符串，通常含 `messages` 数组。
2. **`spans[].logs[].body.function.arguments`**：工具/Agent 调用参数中的 prompt。
3. **GAIA**：`body.question` 或 `body.function.arguments` 中的 question。
4. **SWE-Bench**：`body.function.arguments.item.question`、`item.problem_statement`。

优先查找 `span_attributes` 中 `openinference.span.kind: LLM` 或 `input.mime_type: application/json` 的 span，其 `input.value` 即该次调用的输入 prompt。
