# Dataset（以 MARBLE 为例）

MAS 实验需要具备 **workflow 图**（agent 与依赖）、**agent 角色/ profile**（对应稳定 system prompt）、以及 **执行 trace**（可抽成 DAG、算 criticality）。本目录以 **MARBLE** 为例说明数据形态，其它具备类似结构的数据集（如 MAST-Data、TRAIL、Multi-Agents-Parallel-Orchestration-Dataset 等）也可复用。

---

## 所需数据形态（抽象）

- **图结构**：节点 = agent step（一次或多次 LLM 调用），边 = 依赖（u→v 表示 v 依赖 u 的输出）。
- **Agent 信息**：agent_id、profile（角色/专长），用于构造 system prompt 并做 agent-aware 路由与 criticality 标注。
- **Trace**：每条样本包含 task、agents、关系与（可选）执行顺序/迭代次数，便于还原 DAG、识别关键路径与 dominator。

---

## MARBLE 作为示例

MARBLE（Multi-Agent Coordination Backbone with LLM Engine）用 YAML 描述任务、Agent 列表、关系与协调模式；Engine 按不同模式调度，对应到发给 LLM 的 prompt 与交互关系。

- **Prompt**：单次调用 = system（agent_id + profile + 角色说明）+ user（任务 + 记忆 + 会话历史 + 可选推理策略）；通信场景会加入 msg_box 与 communicate_to 的 tool 描述。
- **协调模式**：chain（链式）、tree（树形）、graph（图/协作）、star（星形）；关系由 `relationships` 三元组定义，只有存在关系的 agent 对可 communicate_to。
- **共享状态**：msg_box 维护定向会话历史，SharedMemory 做多 agent 共享键值；可用于从 trace 还原执行图与依赖。

本目录下 MARBLE 的 Research / Coding 等场景即「全连接图 + 多 agent profile」或「隐式 DAG（如 creator → adder → optimizer）」的实例，用于验证 agent-sticky 路由、critical-only 备份、criticality 调度等思路。更细的配置层级、Mermaid 图与实现路径见仓库内 MARBLE 代码与配置（如 `marble/configs/`、`marble/agent/`、`marble/graph/`、`marble/engine/`）。
