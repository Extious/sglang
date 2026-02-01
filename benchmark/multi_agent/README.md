# Multi-Agent Benchmark Datasets

本目录包含 MAST-Data 与 TRAIL 两个多智能体评测数据集的说明与字段说明。

---

## 一、MAST-Data (MAST-Data/)

来源: [mcemri/MAST-Data](https://huggingface.co/datasets/mcemri/MAST-Data)。多智能体系统执行轨迹，并按 MAST (Multi-Agent Systems Failure Taxonomy) 标注失败类型。

### 1.1 文件列表

| 文件 | 说明 |
|------|------|
| `MAD_full_dataset.json` | 完整数据集，每条为一条 trace + MAST 数值标注 |
| `MAD_human_labelled_dataset.json` | 人工标注子集，含详细 failure mode 与多标注员布尔标注 |
| `README.md` | 数据集卡片 |

### 1.2 MAD_full_dataset.json

**类型**: JSON 数组，每元素为一条记录。

**每条记录顶层字段**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `mas_name` | string | 多智能体系统名称，如 `"ChatDev"`, `"AppWorld"` |
| `llm_name` | string | 使用的 LLM 名称，如 `"GPT-4o"` |
| `benchmark_name` | string | 基准名称，如 `"ProgramDev"`, `"Test-C"` |
| `trace_id` | int | 轨迹编号 |
| `trace` | object | 轨迹元数据与正文，见下表 |
| `mast_annotation` | object | MAST 分类下的二值标注 (0/1)，键为 "1.1"～"3.3" |

**`trace` 对象字段**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `key` | string | 轨迹唯一键，如 `"ChatDev_ProgramDev_GPT4o"` |
| `index` | int | 在该 key 下的索引 |
| `trajectory` | string | 完整执行日志文本（含**<span style="color:red">任务 prompt</span>**、多智能体对话与执行过程） |

**`mast_annotation` 对象**:

键为 MAST 失败模式编号，值为 0（未出现）或 1（出现）。编号与含义对应关系（与 MAD_human_labelled 中的 "failure mode" 一致）:

- **1.x 推理/任务理解**: `1.1` Poor task constraint compliance, `1.2` Inconsistency between reasoning and action, `1.3` Undetected conversation ambiguities and contradictions, `1.4` Fail to elicit clarification (between agents), `1.5` Unaware of stopping conditions  
- **2.x 执行/规划**: `2.1` Unbatched repetitive execution, `2.2` Step repetition, `2.3` Backtracking interruption, `2.4` Conversation reset, `2.5` Derailment from task, `2.6` Disobey role specification  
- **3.x 协作**: `3.1` Disagreement induced inaction, `3.2` Withholding relevant information, `3.3` Ignoring suggestions from agents  

（注：MAD_full_dataset 的 `mast_annotation` 仅包含 1.1～3.3；MAD_human_labelled 的 `annotations` 中还可能包含 3.4、4.1、4.2、4.3 等更多 failure mode。）

### 1.3 MAD_human_labelled_dataset.json

**类型**: JSON 数组，每元素为一条人工标注记录。

**每条记录顶层字段**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `round` | string | 标注轮次，如 `"Round 1"` |
| `mas_name` | string | 多智能体系统名称 |
| `benchmark_name` | string | 基准名称 |
| `trace_id` | int | 轨迹编号 |
| `trace` | string | 完整执行轨迹文本（含**<span style="color:red">任务 prompt</span>**、多轮对话、代码执行输出、以及末尾的 Evaluation 块） |
| `annotations` | array | 对 MAST 各 failure mode 的逐条标注，见下表 |

**`annotations` 数组中每个元素**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `failure mode` | string | MAST 失败模式完整描述（含编号与说明，如 "1.5 Unaware of stopping conditions\n\nLack of recognition..."） |
| `annotator_1` | boolean | 标注员 1 是否判定该条 trace 存在该失败模式 |
| `annotator_2` | boolean | 标注员 2 是否判定存在 |
| `annotator_3` | boolean | 标注员 3 是否判定存在 |

`trace` 末尾的 Evaluation 块为 JSON，包含 `success`, `difficulty`, `num_tests`, `passes`, `failures` 等评测结果。

---

## 二、TRAIL (TRAIL/)

来源: [PatronusAI/TRAIL](https://huggingface.co/datasets/PatronusAI/TRAIL)。Trace Reasoning and Agentic Issue Localization：带错误定位与影响程度标注的智能体执行 trace，来自 GAIA 与 SWE-Bench 任务。

### 2.1 目录与文件概览

| 路径 | 说明 |
|------|------|
| `data/gaia-*.parquet` | GAIA 任务的 HF 表格式：列 `trace`, `labels`（均为 string） |
| `data/swe_bench-*.parquet` | SWE-Bench 任务的 HF 表格式：同上 |
| `GAIA/*.json` | 按 trace_id 的 GAIA 原始 trace（OpenTelemetry spans） |
| `SWE Bench/*.json` | 按 trace_id 的 SWE-Bench 原始 trace，结构同 GAIA |
| `processed_annotations_gaia/*.json` | GAIA 每条 trace 的错误与分数标注 |
| `processed_annotations_swe_bench/*.json` | SWE-Bench 每条 trace 的错误与分数标注 |
| `README.md` | 数据集卡片 |

### 2.2 data/*.parquet

HuggingFace 标准表格式：

| 列名 | 类型 | 说明 |
|------|------|------|
| `trace` | string | 序列化后的整条 trace（可与 GAIA/SWE Bench 下同名 trace_id 的 JSON 对应） |
| `labels` | string | 序列化后的标注（与 processed_annotations_* 中对应 trace_id 的标注一致） |

- `gaia-*.parquet`: 117 条；`swe_bench-*.parquet`: 31 条（以 HF 卡片为准）。

### 2.3 GAIA/*.json 与 SWE Bench/*.json（原始 Trace）

**文件名**: 以 trace_id 命名的 JSON，如 `0adc4f3b99d9564d32811e913cc9d248.json`。

**顶层字段**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `trace_id` | string | 16 进制 trace 标识，与文件名一致 |
| `spans` | array | OpenTelemetry 风格的 span 数组，根 span 的 `parent_span_id` 为 null，子 span 通过 `child_spans` 或 `parent_span_id` 嵌套 |

**每个 span 对象字段**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `timestamp` | string | ISO 时间戳 |
| `trace_id` | string | 所属 trace_id |
| `span_id` | string | 当前 span 的 16 进制 ID（标注中的 `location` 即引用此 id） |
| `parent_span_id` | string \| null | 父 span 的 span_id |
| `trace_state` | string | OpenTelemetry trace state |
| `span_name` | string | 如 `"main"`, `"answer_single_question"`, `"CodeAgent.run"` |
| `span_kind` | string | 如 `"Internal"` |
| `service_name` | string | 服务名，如 `"gaia-annotation-samples/app:GAIA-Samples"` 或 SWE-Bench 对应服务名 |
| `resource_attributes` | object | 资源属性（如 `service.name`, `telemetry.sdk.language` 等） |
| `scope_name` | string | 如 `"patronus.sdk"`, `"openinference.instrumentation.smolagents"` |
| `scope_version` | string | 版本号 |
| `span_attributes` | object | 本 span 属性（如 `pat.app`, `pat.project.id`, **<span style="color:red">`input.value`</span>**（**输入给 LLM 的 Prompt，通常为 JSON 格式的 messages 数组**）, `output.value` 等） |
| `duration` | string | ISO 8601 时长，如 `"PT58.876293S"` |
| `status_code` | string | 如 `"Unset"`, `"Ok"` |
| `status_message` | string | 状态说明 |
| `events` | array | 事件列表 |
| `links` | array | 链接列表 |
| `logs` | array | 本条 span 的 log 列表，见下表 |
| `child_spans` | array | 子 span 数组，结构与顶层 span 相同（递归） |

**`logs` 中每个 log 对象**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `timestamp` | string | 日志时间 |
| `trace_id`, `span_id` | string | 所属 trace/span |
| `trace_flags` | int | 标志位 |
| `severity_text` | string | 如 `"INFO"` |
| `severity_number` | int | 严重程度数值 |
| `service_name` | string | 服务名 |
| `body` | object | 主要内容，常见键：**<span style="color:red">`function.arguments`</span>**（**可能包含输入给 LLM 的 Prompt，如 messages 数组**）, `function.name`, `function.output`（含输入、工具名、输出等）；GAIA 中常有 `question`（**任务 prompt**）, `task_id`, `true_answer`, `Annotator Metadata`；SWE-Bench 中常有 `item`（含 `problem_statement`（**问题描述 prompt**）, `instance_id`, `patch`, `question`（**任务 prompt**） 等） |
| `resource_attributes` | object | 资源属性 |
| `scope_name`, `scope_version` | string | 作用域信息 |
| `log_attributes` | object | 如 `pat.log.id`, `pat.log.type`, `pat.project.name` 等 |
| `evaluations` | array | 评估结果（若有） |
| `annotations` | array | 注解（若有） |

GAIA 的 span 中常包含任务描述 `question`、`task_id`、`true_answer`；SWE-Bench 的 span 中常包含 issue、repo、base_commit、patch、problem_statement、test_patch 等字段（多在 `body.function.arguments.item` 或类似路径）。

### 2.4 processed_annotations_gaia/*.json 与 processed_annotations_swe_bench/*.json

**文件名**: 与对应 trace 的 trace_id 一致，如 `0adc4f3b99d9564d32811e913cc9d248.json`。

**顶层字段**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `trace_id` | string | 与 GAIA/SWE Bench 中同名 JSON 的 trace_id 一致 |
| `errors` | array | 该 trace 中标注出的错误列表，见下表 |
| `scores` | array | 该 trace 的维度评分与总评，见下表 |

**`errors` 中每个元素**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `category` | string | 错误类别，如 `"Formatting Errors"`, `"Incorrect Problem Identification"`, `"Language-only"`, `"Poor Information Retrieval"`, `"Incorrect Memory Usage"`, `"Context Handling Failures"` 等 |
| `location` | string | 对应 span 的 `span_id`（在 GAIA/SWE Bench 的 JSON 中可定位到具体 span） |
| `evidence` | string | 支持该错误判定的证据（日志片段、输出片段等） |
| `description` | string | 错误描述 |
| `impact` | string | 影响程度：`"LOW"`, `"MEDIUM"`, `"HIGH"` |

**`scores` 中每个元素**（通常一条 trace 一个对象）:

| 字段 | 类型 | 说明 |
|------|------|------|
| `reliability_score` | int | 可靠性评分（如 1–5） |
| `reliability_reasoning` | string | 评分理由 |
| `security_score` | int | 安全性评分 |
| `security_reasoning` | string | 评分理由 |
| `instruction_adherence_score` | int | 指令遵循度评分 |
| `instruction_adherence_reasoning` | string | 评分理由 |
| `plan_opt_score` | int | 计划/策略合理性评分 |
| `plan_opt_reasoning` | string | 评分理由 |
| `overall` | float | 综合分数 |

使用方式建议：由 `data/*.parquet` 按 trace_id 关联到 `GAIA/*.json` 或 `SWE Bench/*.json` 得到完整 span 结构，再由 `processed_annotations_gaia/*.json` 或 `processed_annotations_swe_bench/*.json` 中的 `trace_id` 与 `errors[].location`（span_id）对齐到具体 span，用于错误定位与评估。

---

## 三、Prompt 流程提取

脚本 `extract_prompt_flow.py` 可从上述两个数据集中提取 prompt 流程并输出为 JSON。

**用法**（在 `benchmark/multi_agent/` 下执行）:

```bash
# 提取全部（MAST-Data full/human + TRAIL GAIA + TRAIL SWE-Bench）
python extract_prompt_flow.py

# 仅提取指定数据集
python extract_prompt_flow.py --dataset mast_full    # MAD_full_dataset.json
python extract_prompt_flow.py --dataset mast_human   # MAD_human_labelled_dataset.json
python extract_prompt_flow.py --dataset trail_gaia  # TRAIL/GAIA/*.json
python extract_prompt_flow.py --dataset trail_swe    # TRAIL/SWE Bench/*.json

# 指定数据目录与输出目录
python extract_prompt_flow.py --data-dir /path/to/multi_agent --output-dir /path/to/out
```

**输出文件**（默认写入当前目录）:

| 文件 | 来源 | 每条记录主要内容 |
|------|------|------------------|
| `mast_full_prompt_flow.json` | MAD_full_dataset | mas_name, llm_name, benchmark_name, trace_id, task_prompt（来自 trajectory 中的 `**task_prompt**:`） |
| `mast_human_prompt_flow.json` | MAD_human_labelled | round, mas_name, benchmark_name, trace_id, task_prompt（来自 trace 中 `******************** Task N/M (id) ********************` 后一行） |
| `trail_gaia_prompt_flow.json` | TRAIL/GAIA | trace_id, main_task（根任务文本）, prompt_flow（span 内 question/task/user_message 等列表） |
| `trail_swe_bench_prompt_flow.json` | TRAIL/SWE Bench | trace_id, question, problem_statement, prompt_flow |

---

## 四、MAD_full trajectory 详细展开

脚本 `expand_mad_full_trajectory.py` 将 `MAD_full_dataset.json` 中每条记录的 `trace.trajectory` 字符串解析为结构化字段 `trajectory_expanded`，便于按段使用。

**用法**（在 `benchmark/multi_agent/` 下执行）:

```bash
# 默认：读 MAST-Data/MAD_full_dataset.json，写 MAST-Data/MAD_full_dataset_expanded.json（不保留原始 trajectory）
python expand_mad_full_trajectory.py

# 保留原始 trajectory 字段
python expand_mad_full_trajectory.py --keep-trajectory

# 指定输入/输出路径
python expand_mad_full_trajectory.py --input /path/to/MAD_full_dataset.json --output /path/to/expanded.json
```

**按 mas_name 的展开结构**:

| mas_name | trajectory_expanded 主要字段 |
|----------|------------------------------|
| **ChatDev** | `metadata`（**task_prompt**, config_path, project_name, ChatDevConfig, ChatGPTConfig 等）, `task_prompt`, `sections`（phase_or_log：**phase_name** \| DemandAnalysis / Coding / CodeReviewComment 等）, `evaluation` |
| **AppWorld** | `task`（**** Task N/M (id) **** 后一行）, `sections`（response_from_agent, message_to_agent, code_execution, entering_agent_loop, exiting_agent_loop, api_response；每项含 type, header, agent, content）, `evaluation`（JSON：success, difficulty, passes, failures） |
| **AG2 / HyperAgent** | `metadata`（instance_id, problem_statement 等）, `problem_statement`, `sections`（trajectory_content） |
| **Magentic / OpenManus 等** | `sections`（按 RUN.SH STARTING、Processing 等拆分的 block） |

输出文件中每条记录保留原有顶层字段（mas_name, llm_name, benchmark_name, trace_id, trace, mast_annotation），并新增 `trajectory_expanded`；默认会从 `trace` 中移除原始 `trajectory` 以减小体积，可用 `--keep-trajectory` 保留。

### 2.5 如何定位输入给 LLM 的 Prompt

**<span style="color:red">输入给 LLM 的 Prompt 主要位于以下位置：</span>**

1. **`spans[].span_attributes.input.value`**：最常见的位置，通常是一个 JSON 字符串，包含 `messages` 数组，格式如：
   ```json
   {
     "messages": [
       {"role": "system", "content": "..."},
       {"role": "user", "content": "..."}
     ]
   }
   ```

2. **`spans[].logs[].body.function.arguments`**：函数调用参数中可能包含 prompt，特别是 LLM 调用相关的 span（如 `span_name` 为 `"CodeAgent.run"` 或类似名称的 span）。

3. **`spans[].logs[].body.question`**（GAIA）：直接的任务 prompt 文本。

4. **`spans[].logs[].body.function.arguments.item.question`**（SWE-Bench）：任务 prompt 文本。

5. **`spans[].logs[].body.function.arguments.item.problem_statement`**（SWE-Bench）：问题描述 prompt。

**提示**：查找 `span_attributes` 中包含 `"input.mime_type": "application/json"` 或 `"openinference.span.kind": "LLM"` 的 span，这些通常对应 LLM 调用，其 `input.value` 字段即为输入 prompt。
