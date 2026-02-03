# Graph-Aware Multi-Agent LLM Serving（面向 SGLang 的研究整理）

> 背景：当前多智能体（Multi-Agent）应用上层的 agent 编排与下层 LLM serving（SGLang / vLLM 等）基本解耦。上层交互天然可抽象为 graph（DAG/执行图/依赖图），但这些依赖与角色信息并未进入下层的请求调度与 KV cache 管理，造成大量 latency/尾延迟优化机会被浪费。
>
> 本文将你的两个研究切入点系统化：  
> 1) **Fault tolerance：识别并保护 critical agent / critical step（可动态/静态）**，例如对关键节点的 KV cache 做备份与快速恢复；  
> 2) **Agent-aware KV cache management：以 agent 为 locality 单元进行路由与缓存管理**，尽量将同一 agent 的请求调度到同一实例（或同一 cache domain），提高 cache hit，降低 TTFT/端到端延迟。  
> 并给出统一 formulate、实验设计与可复现的初步评测方法。

---

## 1. 关键问题与研究假设

### 1.1 现状缺口

- **上层**：multi-agent workflow 由多个 agent 协作完成任务（planner/manager + specialized agents + aggregator 等），交互过程可以表示为执行图：节点代表一次 agent step（通常对应一次或多次 LLM 调用/工具调用），边代表依赖关系。
- **下层**：serving engine 面向“独立请求”优化（batching、prefix cache、KV 管理、负载均衡），但不理解“这些请求属于同一个 workflow、由哪些 agent 发起、存在何种依赖/关键路径/爆炸半径”。
- **结果**：  
  - 调度无法利用图结构：关键路径优先级、join 阻塞、并行分支的局部性、关键节点的容错需求等；  
  - KV cache 管理与路由策略对 agent 无感：同一 agent 的请求随机分散到多实例导致 L1/L2 cache 命中率下降；  
  - failure 时缺乏“graph-aware recovery”：关键节点失败会导致大规模重算（blast radius 大），但下层无法选择性保护。

### 1.2 两个研究切入点的统一视角

可以将研究目标统一为：

> **Graph-aware policy stack**：为每个 workflow 维护执行图 + 关键性（criticality），并驱动  
> - **调度/路由策略**（agent-aware、dependency-aware、load-aware）  
> - **KV cache 策略**（分层、隔离、迁移、选择性备份/多副本）  
> 从而同时优化 latency（尤其 P95/P99）与 failure 影响半径。

---

## 2. 相关工作与可借鉴机制（重点聚焦“可落在 serving 层”的工作）

### 2.1 KV cache 管理与 prefix caching（serving 核心）

- **vLLM / PagedAttention**：通过分页管理 KV cache，提升吞吐与显存利用，为后续 KV 复用/迁移/共享提供底层思路。
- **SGLang / RadixAttention（prefix cache）**：以 radix tree 管理 prefix KV，适合“多请求共享长 system prompt / 模板前缀”的场景；并提供缓存命中统计与 flush 接口，便于实验。

### 2.2 分层 / 分布式 KV cache：从“仅本 GPU”走向 L1/L2/L3

- **HiCache（SGLang）**：把 KV cache 做成 CPU 式层次结构（L1 GPU / L2 host / L3 distributed），支持 prefetch、写回策略（write-through / write-through-selective / write-back）等。  
  - 这为你的研究提供“机制基础”：你要做的主要是**策略**（哪些 agent/哪些节点需要更积极的写回/多副本/预取）。

### 2.3 Prefill/Decode 解耦（与 multi-agent graph 映射高度相关）

- Prefill 计算密集，Decode 记忆/带宽敏感；PD disaggregation 把两阶段拆开，允许分别优化资源与调度。  
  - 对 multi-agent 来说，许多 agent step 具有不同的输入长度与输出长度分布，PD 能进一步放大“关键路径/关键节点”的差异。

### 2.4 Multi-agent 数据与执行轨迹（用于抽图、做 criticality 分析）

本仓库 `benchmark/multi_agent/` 已包含多个可用于“从 trace 抽 graph、做离线仿真与在线回放”的数据集：

- **MAST-Data**：多智能体系统执行轨迹 + 失败模式标注（Multi-Agent failure taxonomy），适合研究“失败影响/错误传播/关键节点”。
- **TRAIL**：带错误定位与影响程度标注的智能体 trace，适合研究“失败影响半径/恢复策略收益”。
- **Multi-Agents-Parallel-Orchestration-Dataset**：并行/顺序编排的 session trace，适合研究“并行分支、join、图结构与调度”。
- **multiagent-router-finetuning**：带 `agent_name` 的合成客服路由数据，适合快速构造“多 agent 混合请求 + agent locality”微基准（非常适合做你要的初步实验）。

---

## 3. 场景 formulate：把 multi-agent serving 写成可度量的优化问题

### 3.1 Workflow 图（执行依赖）

对一次请求/会话（workflow）定义执行图：

- `G = (V, E)`  
- 节点 `v ∈ V`：一次 agent step（可以是一次 LLM 调用或一组调用）  
- 边 `e = (u → v)`：`v` 依赖 `u` 的输出（message、tool result、plan、代码产物等）

常见图形态：

- **Chain**：A → B → C（典型流水线）
- **Star/Planner-Fanout-Join**：Planner → {Worker_i} → Join（典型“planner + 多专家 + 汇总”）
- **Diamond**：A → (B, C) → D（并行 + join）
- **Loop（展开为 DAG）**：同一 agent 多轮（ReAct/reflect 等）

### 3.2 Serving 资源与 KV 状态

- 有 `K` 个模型实例（你的环境：最多 10×4080，每卡一个实例）。
- 每次 LLM 调用 `c` 有：
  - `T_prefill(c)`：prefill 时间（强依赖 prompt tokens）
  - `T_decode(c)`：decode 时间（依赖输出 tokens、KV 常驻、调度/抢占等）
  - `S_kv(c)`：KV 状态大小（决定备份/迁移成本）
- KV 命中可分层（以 HiCache 为例）：L1(GPU) / L2(host) / L3(shared)。

### 3.3 Failure、重算与“爆炸半径（blast radius）”

定义 instance failure（崩溃、重启、OOM、网络隔离等）在 workflow 图上的影响：

- `BR(v)`：当承载关键节点 `v` 的实例失败导致的“需要重算/等待”的下游节点集合（可近似为 `descendants(v)`，但 join 会放大阻塞效应）。
- 可度量代价：
  - **重算 token 成本**：`Cost_tokens(v) = Σ_{x ∈ ReExec(v)} prompt_tokens(x)`
  - **端到端额外时延**：`ΔLatency(v) = Makespan_with_failure − Makespan_without_failure`
  - **错误传播/失败概率**：失败导致 workflow abort 的概率提升

---

## 4. Critical agent / critical step：定义与识别方法（静态 + 动态）

### 4.1 两类“关键性”

- **Latency-critical**：在关键路径（critical path）上或造成 join 前瓶颈，决定 makespan / P95/P99
- **Reliability-critical**：失败会导致大规模重算或 workflow abort（blast radius 大）

### 4.2 静态识别（只用图结构 + 先验权重）

给每个节点赋权重 `w(v) = E[T_prefill(v) + T_decode(v)]`，可用历史统计或 token 数粗估。

- **关键路径（最长路径）**：DAG 上求最长路径，路径节点为 latency-critical
- **下游覆盖度（影响范围）**：`Σ_{u ∈ descendants(v)} w(u)`（或节点数/重算 tokens）
- **支配点（dominator）**：若从 source 到目标 sink 的所有路径都经过 `v`，则它是强关键点（例如 planner/manager）

可以组合成静态分数：

`C_static(v) = α·I[v ∈ critical_path] + β·Σ_{u ∈ desc(v)} w(u) + γ·join_sensitivity(v)`

### 4.3 动态识别（运行时随输入/负载变化）

动态关键性必须考虑：

- 节点运行时输入长度/输出长度变化（同一 agent 在不同任务上差异很大）
- 当前集群负载与排队延迟
- cache hit（L1/L2/L3）、KV 迁移/写回带宽
- 失败概率（某实例更不稳定/更接近 OOM）

动态分数示例：

`C_dyn(v) = E[ΔLatency(v) | failure] + λ·P(failure_on_instance(v))·BR_cost(v)`

**落地要点**：  
你不必一次做到“完美预测”。初期可用简单 heuristics（关键路径 + fanout）就能验证“critical-only 保护”的收益上界。

---

## 5. Fault tolerance 设计：对 critical 节点做“选择性 KV checkpoint/备份”

### 5.1 保护对象（从便宜到昂贵）

1) **只备份输出（messages/tool results）**：开销低，但恢复仍需重做 prefill（慢）  
2) **备份 prefix KV（system prompt / 固定模板前缀）**：收益/成本性价比最高，且与 radix/prefix cache 机制天然契合  
3) **备份完整会话 KV（长上下文）**：收益最大，成本与带宽压力最大，需要分层/压缩/异步写回

### 5.2 备份落点（用 HiCache 机制承载）

基于 L1/L2/L3：

- **L2 host 备份**：同机恢复快，但机器级故障仍丢失
- **L3 distributed 备份**：跨机恢复，适合抵御 node failure；但要权衡尾延迟与写回带宽

### 5.3 关键研究点（把“机制”变成“策略”）

- **critical-only 写回**：仅对高 `C(v)` 的节点触发写回/多副本（其他节点不付出备份成本）
- **写回时机**：同步（强一致、低风险、高开销） vs 异步（低开销、可能丢最后一段 KV）
- **副本数与放置**：关键节点多副本，非关键节点单副本或不备份

### 5.4 “服务模式”与爆炸半径对比（建议论文里用表呈现）

- M0 无缓存：failure 重发即可，但每次都慢
- M1 本地 cache（无共享）：failure 会丢 KV，关键节点失败 BR 大
- M2 agent-sticky：命中率更高，但同一 agent 状态更集中，failure BR 更集中（需备份对冲）
- M3 shared L3：failure 可从 L3 恢复，BR 小，但引入写回/传输开销与尾延迟风险
- M4 M2 + critical-only 备份：综合性价比往往最好（你的主打方向）

---

## 6. Agent-aware KV cache management：以 agent 为 locality 单元做路由与缓存隔离

### 6.1 核心假设

- 同一 agent 往往复用稳定的 system prompt、工具描述、输出格式约束、memory schema
- 若请求被随机打散到多实例，则 L1/L2 命中下降 → prefill 代价增加 → TTFT/尾延迟上升

### 6.2 路由策略设计空间（从简单到复杂）

1) **agent-sticky hash（baseline）**  
按 `hash(agent_id) → worker` 固定映射；优点简单，缺点不负载均衡

2) **locality-first + load-guard**  
优先发往“agent home worker”，若排队超过阈值则 fallback（兼顾局部性与负载）

3) **graph-aware routing**  
把 workflow 的关键路径/关键节点优先保证局部性；非关键分支更自由地做负载均衡

### 6.3 cache 隔离与配额

- 为不同 agent 设置 cache quota（至少在 L1/L2），防止高流量 agent 挤占其他 agent 的热 KV
- eviction 策略从“纯热度”扩展为“热度 + 关键性（criticality）”

---

## 7. 基于 MARBLE 数据集的实验设计

> 本节基于 `benchmark/multi_agent/MARBLE/` 中的 **Research Collaboration** 和 **Coding Collaboration** trace 设计具体实验，验证前文提出的两个核心假设。

### 7.1 MARBLE 数据集特征分析

#### 7.1.1 Research Collaboration（研究协作场景）

**数据位置**：`MARBLE/multiagentbench/research/research_main.jsonl`（100 条 trace）

**图结构特征**：
- **Agent 数量**：5 个 agent（agent1-agent5）
- **拓扑结构**：全连接图（fully connected），10 条双向协作边
- **关系类型**：`"collaborate with"`（对称协作）
- **迭代次数**：max_iterations = 3

**Agent 角色分布**：
| Agent | 研究方向 | System Prompt 特征 |
|-------|---------|-------------------|
| agent1 | 深度学习可靠性与隐私（联邦学习、差分隐私） | ~1500 tokens |
| agent2 | 生成模型与差分隐私（DP-SAD） | ~1200 tokens |
| agent3 | 计算机视觉与追踪（人脸识别、UAV） | ~1400 tokens |
| agent4 | 图像处理与检索（HDR、去噪） | ~1600 tokens |
| agent5 | 低分辨率识别与知识蒸馏 | ~1400 tokens |

**通信模式**：
```
     agent1
    /  |  \  \
agent2-agent3-agent4-agent5
    \  |  /  /
     (全连接)
```

**关键观察**：
- 每个 agent 有独立且稳定的 system prompt（研究背景描述）
- 同一 agent 在多轮迭代中复用相同 system prompt → **高 prefix 复用潜力**
- 全连接拓扑 → 任一 agent 失败会阻塞所有协作者 → **blast radius 大**

#### 7.1.2 Coding Collaboration（编码协作场景）

**数据位置**：`MARBLE/multiagentbench/coding/coding_main.jsonl`（100 条 trace）

**图结构特征**：
- **Agent 数量**：3 个 agent（agent1-agent3）
- **拓扑结构**：全连接图，6 条双向协作边
- **关系类型**：`"collaborates with"`（对称协作）
- **迭代次数**：max_iterations = 5

**Agent 角色分布（强约束的流水线）**：
| Agent | 角色 | 能力约束 | 依赖关系 |
|-------|------|---------|---------|
| agent1 | Creator（创建者） | 只能 `create_code`，不能修改 | 无前置依赖 |
| agent2 | Functionality Adder（功能补充者） | 只能 `give_advice_and_revise_code` | 依赖 agent1 |
| agent3 | Optimizer（优化者） | 只能 `give_advice_and_revise_code` | 依赖 agent2 |

**隐式执行流（从 profile 约束推断）**：
```
agent1 (create) → agent2 (add features) → agent3 (optimize)
     ↑                                          |
     └──────────── (可能的迭代循环) ─────────────┘
```

**关键观察**：
- 虽然拓扑是全连接，但 **profile 约束形成隐式 DAG**：agent1 → agent2 → agent3
- agent1 是 **dominator 节点**：所有下游都依赖其输出
- 每个 agent 有稳定的角色描述（~300 tokens）+ 共享的任务描述（~800 tokens）
- **关键路径明确**：agent1 的 latency 直接决定整体 makespan

### 7.2 实验一：Agent-Aware 路由策略对 Cache Hit 与 Latency 的影响（验证 H1）

#### 7.2.1 实验目标

验证假设 **H1**：*路由策略越能维持 agent 局部性（或 prompt prefix 局部性），cache hit 越高、TTFT/延迟越低*

#### 7.2.2 实验设计

**Workload 构造**：

| 场景 | 数据源 | 并发模式 | 请求数 | 10 GPU 利用率 |
|------|-------|---------|-------|--------------|
| Research-Seq | research_main.jsonl 前 20 条 | 顺序执行每个 workflow | 20 × 5 × 3 = 300 | 低（验证正确性） |
| Research-Para | research_main.jsonl 前 20 条 | 10 个 workflow 并发 | 同上 | 高（压力测试） |
| Coding-Seq | coding_main.jsonl 前 20 条 | 顺序执行每个 workflow | 20 × 3 × 5 = 300 | 低（验证正确性） |
| Coding-Para | coding_main.jsonl 前 20 条 | 10 个 workflow 并发 | 同上 | 高（压力测试） |
| Mixed-High | 50 coding + 50 research | 20 个 workflow 并发 | ~1500 | 满载（极限测试） |

**请求构造方式**：
```python
# 伪代码：从 MARBLE trace 构造请求
def construct_requests(trace):
    requests = []
    for iteration in range(trace["environment"]["max_iterations"]):
        for agent in trace["agents"]:
            req = {
                "agent_id": agent["agent_id"],
                "workflow_id": trace["task_id"],
                "step_id": f"{iteration}_{agent['agent_id']}",
                "messages": [
                    {"role": "system", "content": agent["profile"]},
                    {"role": "user", "content": trace["task"]["content"]}
                    # + 历史消息（从 shared memory 模拟）
                ]
            }
            requests.append(req)
    return requests
```

**路由策略对照组**：

| 策略 | 实现方式 | 10 GPU 映射示例 | 预期 Cache Hit |
|------|---------|----------------|---------------|
| `round_robin` | 轮询分发到 10 个 worker | 请求分散到所有 GPU | 低（~0%） |
| `random` | 随机分发 | 请求随机分布 | 低（~0%） |
| `agent_sticky_hash` | `hash(agent_id) % 10` | Research: agent1→GPU0,1; agent2→GPU2,3; ... | 高（同 agent 命中固定 GPU） |
| `workflow_sticky_hash` | `hash(workflow_id) % 10` | 同 workflow 的所有 agent 命中同 GPU | 中（同 workflow 命中同实例） |
| `cache_aware` | SGLang 内置 cache-aware 路由 | 基于 prefix 匹配选择 GPU | 中-高（prefix 匹配） |
| `agent_sticky + load_guard` | 优先 agent home GPU，超阈值 fallback | agent1 优先 GPU0,1，负载高时溢出到其他 | 高 + 负载均衡 |

**10 GPU 的 Agent 映射方案**：

```
Research 场景（5 agents × 10 GPUs）:
  agent1 → GPU 0, 5  (primary: 0)
  agent2 → GPU 1, 6  (primary: 1)
  agent3 → GPU 2, 7  (primary: 2)
  agent4 → GPU 3, 8  (primary: 3)
  agent5 → GPU 4, 9  (primary: 4)

Coding 场景（3 agents × 10 GPUs）:
  agent1 (critical) → GPU 0, 1, 2  (3 replicas for fault tolerance)
  agent2            → GPU 3, 4, 5  (3 replicas)
  agent3            → GPU 6, 7, 8  (3 replicas)
  spare             → GPU 9        (load balancing overflow)
```

**集群配置**：
- 10 个 worker（10×4080 GPU），每个运行独立 SGLang 实例
- 1 个 router 进程
- 模型：**Qwen3-4B-Thinking**（单卡 ~8GB，KV cache 空间充裕）

**10 GPU 配置的优势**：
- Research 场景：5 个 agent 可各自独占 2 个 worker（agent-sticky 完美映射）
- Coding 场景：3 个 agent 可各自独占 3 个 worker + 1 个备用
- 更大的 worker 池使 round_robin 的 cache miss 问题更明显，对比效果更显著

#### 7.2.3 测量指标

**主指标**：
```bash
# Cache Hit Ratio（从 /metrics 端点采集）
cache_hit_ratio = Δcached_tokens / (Δcached_tokens + Δprompt_tokens)

# 端到端延迟
latency_p50, latency_p95, latency_p99

# TTFT（Time To First Token）
ttft_p50, ttft_p95, ttft_p99
```

**辅助指标**：
- 每个 worker 的请求分布（验证路由是否生效）
- 每个 agent 的平均 cache hit ratio（分析 agent 间差异）

#### 7.2.4 预期结果

| 策略 | Research Cache Hit | Coding Cache Hit | Latency 改善 |
|------|-------------------|-----------------|-------------|
| round_robin | ~5% | ~5% | baseline |
| agent_sticky_hash | ~60-80% | ~50-70% | -20~40% |
| cache_aware | ~40-60% | ~30-50% | -10~25% |
| agent_sticky + load_guard | ~55-75% | ~45-65% | -15~35% |

**预期洞察**：
- Research 场景因 agent profile 更长（~1500 tokens），cache hit 收益更大
- Coding 场景因任务描述共享度高，workflow_sticky 可能接近 agent_sticky

### 7.3 实验二：Critical-Only KV 备份对 Failure Blast Radius 的影响（验证 H2）

#### 7.3.1 实验目标

验证假设 **H2**：*critical-only KV 备份/恢复能显著降低 instance failure 的 blast radius（ΔLatency 与重算 tokens）*

#### 7.3.2 Criticality 分析（基于 MARBLE 图结构）

**Research Collaboration 的 Criticality**：
```
静态分析：全连接图，所有节点对称
- 无明显 dominator
- 任一节点失败影响 = 4 个协作者（blast radius = 4/5 = 80%）
- 建议策略：均匀保护 或 基于运行时负载动态选择
```

**Coding Collaboration 的 Criticality**：
```
静态分析：隐式 DAG（agent1 → agent2 → agent3）

节点关键性评分：
- agent1: C_static = 1.0 (dominator, 所有路径必经)
  - descendants = {agent2, agent3}
  - blast_radius = 100%

- agent2: C_static = 0.5
  - descendants = {agent3}
  - blast_radius = 50%

- agent3: C_static = 0.0
  - descendants = {}
  - blast_radius = 0%

建议策略：优先保护 agent1 的 KV cache
```

#### 7.3.3 实验设计

**10 GPU 环境下的 Failure 注入方案**：

| 注入点 | 方式 | 时机 | 10 GPU 特殊考虑 |
|-------|------|------|----------------|
| 单 Worker 崩溃 | `kill -9 <worker_pid>` | workflow 执行中（iteration 2） | 杀掉 agent1 的 primary GPU (GPU 0) |
| 多 Worker 崩溃 | 同时 kill 2-3 个 worker | 高并发时 | 测试 agent1 的 3 副本是否足够 |
| Worker 慢响应 | `tc qdisc add dev eth0 root netem delay 500ms` | 持续 | 对特定 GPU 注入延迟 |
| OOM 模拟 | 限制 GPU 显存触发 eviction | 高并发时 | 观察 cache eviction 对 critical agent 的影响 |

**10 GPU 的容错配置**：

```python
# Coding 场景的 GPU 分配与备份策略
gpu_allocation = {
    "agent1": {
        "primary_gpus": [0, 1, 2],      # 3 副本（critical agent）
        "backup_strategy": "L2_write_through",  # 同步写回 host memory
        "criticality": 1.0
    },
    "agent2": {
        "primary_gpus": [3, 4, 5],      # 3 副本
        "backup_strategy": "L2_write_back",     # 异步写回
        "criticality": 0.5
    },
    "agent3": {
        "primary_gpus": [6, 7, 8],      # 3 副本
        "backup_strategy": "none",              # 无备份（非 critical）
        "criticality": 0.0
    },
    "overflow": {
        "gpus": [9],                    # 负载溢出
        "backup_strategy": "none"
    }
}
```

**备份策略对照组**：

| 策略 | 描述 | 备份开销 |
|------|------|---------|
| M0-NoBackup | 无备份，failure 后重算 | 0 |
| M1-AllBackup | 所有节点 KV 写回 L2 | 高 |
| M2-CriticalOnly | 仅 critical 节点（Coding: agent1）写回 L2 | 低 |
| M3-CriticalL3 | Critical 节点写回 L3（跨机） | 中 |

**Coding 场景实验矩阵**：

| 实验 | 备份策略 | 注入失败的 Agent | 预期 ΔLatency |
|------|---------|-----------------|--------------|
| C1 | M0-NoBackup | agent1 | 高（重算全部） |
| C2 | M0-NoBackup | agent3 | 低（无下游） |
| C3 | M2-CriticalOnly | agent1 | 中（从 L2 恢复） |
| C4 | M2-CriticalOnly | agent3 | 低（无备份，无下游） |

**Research 场景实验矩阵**：

| 实验 | 备份策略 | 注入失败的 Agent | 预期 ΔLatency |
|------|---------|-----------------|--------------|
| R1 | M0-NoBackup | agent1 | 高（影响 4 个协作者） |
| R2 | M1-AllBackup | agent1 | 低（全部可恢复） |
| R3 | M2-TopK(2) | agent1 | 中（部分可恢复） |

#### 7.3.4 测量指标

**主指标**：
```python
# Blast Radius（影响范围）
blast_radius = num_affected_requests / total_requests

# 延迟增量
delta_latency = latency_with_failure - latency_without_failure

# 重算 Token 数
recompute_tokens = sum(prompt_tokens for req in recomputed_requests)

# 恢复时间
recovery_time = time_to_first_successful_response_after_failure
```

**辅助指标**：
- 备份写入带宽（MB/s）
- L2/L3 存储占用
- 恢复读取延迟

#### 7.3.5 预期结果

**Coding 场景**：

| 策略 | agent1 失败 ΔLatency | agent3 失败 ΔLatency | 备份开销 |
|------|---------------------|---------------------|---------|
| M0-NoBackup | +200~300% | +0~10% | 0 |
| M2-CriticalOnly | +30~50% | +0~10% | 低（仅 1/3 节点） |
| M1-AllBackup | +20~40% | +20~40% | 高（3/3 节点） |

**关键洞察**：M2-CriticalOnly 在 Coding 场景性价比最高，因为 agent1 是唯一的 dominator。

**Research 场景**：

| 策略 | 任一 agent 失败 ΔLatency | 备份开销 |
|------|------------------------|---------|
| M0-NoBackup | +150~250% | 0 |
| M2-TopK(2) | +80~120% | 中（2/5 节点） |
| M1-AllBackup | +30~50% | 高（5/5 节点） |

**关键洞察**：Research 场景因全连接拓扑，需要更多备份才能有效降低 blast radius。

### 7.4 实验三：Graph-Aware 调度对关键路径延迟的影响

#### 7.4.1 实验目标

验证：*利用图结构信息进行优先级调度，能否降低关键路径延迟（尤其 P95/P99）*

#### 7.4.2 实验设计

**Coding 场景的优先级调度**：
```python
# 基于 criticality 的优先级
priority_map = {
    "agent1": 3,  # 最高优先级（dominator）
    "agent2": 2,  # 中等优先级
    "agent3": 1,  # 最低优先级
}

# 调度策略
def schedule_request(req, queues):
    priority = priority_map[req["agent_id"]]
    queues[priority].append(req)
```

**对照组**：

| 策略 | 描述 |
|------|------|
| FIFO | 先进先出，无优先级 |
| Criticality-Priority | 按 criticality 分数排序 |
| Shortest-Job-First | 按预估 token 数排序 |
| Criticality + SJF | 组合策略 |

**负载条件（10 GPU 环境）**：
- 低负载：1-2 个 workflow 并发（每 GPU < 1 req/s）
- 中负载：5-10 个 workflow 并发（每 GPU ~2-3 req/s）
- 高负载：20 个 workflow 并发（每 GPU ~5+ req/s，触发排队）
- 极限负载：50 个 workflow 并发（测试调度策略上限）

#### 7.4.3 测量指标

```python
# 关键路径完成时间
critical_path_latency = max(latency[agent1], latency[agent2], latency[agent3])

# Makespan（整体完成时间）
makespan = time_last_response - time_first_request

# 尾延迟
tail_latency_p95, tail_latency_p99
```

#### 7.4.4 预期结果

| 策略 | 低负载 Makespan | 高负载 Makespan | P99 改善 |
|------|----------------|----------------|---------|
| FIFO | baseline | baseline | - |
| Criticality-Priority | ~同 | -10~20% | -15~30% |
| Criticality + SJF | ~同 | -15~25% | -20~35% |

### 7.5 实验实施脚本框架

#### 7.5.0 10×4080 GPU 环境配置

**硬件配置**：
```
GPU: 10 × NVIDIA GeForce RTX 4080 (16GB VRAM each)
Total VRAM: 160GB
Interconnect: PCIe (no NVLink)
```

**模型选择建议**：
| 模型 | 参数量 | 单卡显存占用 | 适用性 |
|------|-------|-------------|-------|
| **Qwen3-4B-Thinking** | 4B | ~8GB (FP16) | **本实验使用**，支持思维链推理，单卡充裕 |
| Qwen2.5-7B-Instruct | 7B | ~14GB (FP16) | 备选，单卡可运行 |
| Llama-3.1-8B-Instruct | 8B | ~16GB (FP16) | 备选，显存较紧 |

**Qwen3-4B-Thinking 特点**：
- 支持 thinking 模式，适合复杂推理任务（Research/Coding 场景）
- 4B 参数量在 16GB 4080 上运行充裕，留有足够 KV cache 空间
- 可通过 `enable_thinking=True` 开启思维链输出

**环境准备脚本**：
```bash
#!/bin/bash
# benchmark/multi_agent/setup_10gpu_env.sh

# 检查 GPU 状态
echo "Checking GPU status..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

# 创建日志目录
mkdir -p logs results

# 验证 10 张 GPU 可用
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$GPU_COUNT" -lt 10 ]; then
    echo "Error: Expected 10 GPUs, found $GPU_COUNT"
    exit 1
fi
echo "Found $GPU_COUNT GPUs, ready to proceed."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
export SGLANG_ALLOW_OVERWRITE=1
```

#### 7.5.1 Workload 生成器

```python
# benchmark/multi_agent/marble_workload_generator.py

import json
from pathlib import Path

def load_marble_traces(scenario: str, num_traces: int = 20):
    """加载 MARBLE trace 数据"""
    path = Path(f"MARBLE/multiagentbench/{scenario}/{scenario}_main.jsonl")
    traces = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= num_traces:
                break
            traces.append(json.loads(line))
    return traces

def construct_requests(trace: dict, iteration: int = 0):
    """从单个 trace 构造 LLM 请求"""
    requests = []
    for agent in trace["agents"]:
        req = {
            "agent_id": agent["agent_id"],
            "workflow_id": str(trace["task_id"]),
            "step_id": f"iter{iteration}_{agent['agent_id']}",
            "messages": [
                {"role": "system", "content": agent["profile"]},
                {"role": "user", "content": trace["task"]["content"]}
            ],
            "metadata": {
                "scenario": trace["scenario"],
                "relationships": trace["relationships"]
            }
        }
        requests.append(req)
    return requests

def compute_criticality(trace: dict) -> dict:
    """计算每个 agent 的 criticality 分数"""
    scenario = trace["scenario"]
    agents = [a["agent_id"] for a in trace["agents"]]

    if scenario == "coding":
        # Coding: 基于隐式 DAG 的 criticality
        return {
            "agent1": 1.0,  # dominator
            "agent2": 0.5,
            "agent3": 0.0
        }
    elif scenario == "research":
        # Research: 全连接，均匀 criticality
        n = len(agents)
        return {a: 1.0 / n for a in agents}
    else:
        return {a: 0.5 for a in agents}

def get_agent_gpu_mapping(scenario: str, num_gpus: int = 10) -> dict:
    """获取 agent 到 GPU 的映射（10 GPU 环境）"""
    if scenario == "coding":
        # 3 agents, 每个 agent 分配 3 个 GPU，1 个备用
        return {
            "agent1": [0, 1, 2],   # critical agent, 3 replicas
            "agent2": [3, 4, 5],
            "agent3": [6, 7, 8],
            "_overflow": [9]
        }
    elif scenario == "research":
        # 5 agents, 每个 agent 分配 2 个 GPU
        return {
            "agent1": [0, 5],
            "agent2": [1, 6],
            "agent3": [2, 7],
            "agent4": [3, 8],
            "agent5": [4, 9]
        }
    else:
        return {}

class AgentStickyRouter:
    """Agent-Sticky 路由器实现（10 GPU 版本）"""

    def __init__(self, scenario: str, num_gpus: int = 10, base_port: int = 30000):
        self.scenario = scenario
        self.num_gpus = num_gpus
        self.base_port = base_port
        self.gpu_mapping = get_agent_gpu_mapping(scenario, num_gpus)
        self.request_counts = {i: 0 for i in range(num_gpus)}

    def route(self, agent_id: str, load_threshold: int = 10) -> int:
        """路由请求到合适的 GPU"""
        if agent_id not in self.gpu_mapping:
            # 未知 agent，使用 overflow 或 round-robin
            overflow = self.gpu_mapping.get("_overflow", list(range(self.num_gpus)))
            return min(overflow, key=lambda g: self.request_counts[g])

        candidate_gpus = self.gpu_mapping[agent_id]

        # 优先选择负载最低的 GPU
        best_gpu = min(candidate_gpus, key=lambda g: self.request_counts[g])

        # 如果负载超过阈值，考虑 overflow
        if self.request_counts[best_gpu] > load_threshold:
            overflow = self.gpu_mapping.get("_overflow", [])
            if overflow:
                overflow_gpu = min(overflow, key=lambda g: self.request_counts[g])
                if self.request_counts[overflow_gpu] < self.request_counts[best_gpu]:
                    best_gpu = overflow_gpu

        self.request_counts[best_gpu] += 1
        return best_gpu

    def get_url(self, gpu_id: int) -> str:
        return f"http://localhost:{self.base_port + gpu_id}"
```

#### 7.5.2 实验运行脚本

```bash
#!/bin/bash
# benchmark/multi_agent/run_marble_experiments.sh

# 实验参数
SCENARIOS=("coding" "research")
ROUTING_POLICIES=("round_robin" "agent_sticky_hash" "cache_aware")
NUM_WORKERS=10
NUM_TRACES=20
BASE_PORT=30000
MODEL_PATH="Qwen/Qwen3-4B-Thinking-2507"

# 启动 10 个 workers（每张 4080 一个）
echo "Starting $NUM_WORKERS workers..."
for i in $(seq 0 $((NUM_WORKERS-1))); do
    CUDA_VISIBLE_DEVICES=$i python -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --port $((BASE_PORT + i)) \
        --host 0.0.0.0 \
        --dp 1 \
        --enable-metrics \
        --reasoning-parser qwen3 \
        --log-level warning \
        > logs/worker_$i.log 2>&1 &
    echo "  Worker $i started on GPU $i, port $((BASE_PORT + i))"
done

# 等待所有 worker 启动完成
echo "Waiting for workers to be ready..."
for i in $(seq 0 $((NUM_WORKERS-1))); do
    while ! curl -s http://localhost:$((BASE_PORT + i))/health > /dev/null 2>&1; do
        sleep 2
    done
    echo "  Worker $i ready"
done
echo "All workers ready!"

# 构建 worker URL 列表
WORKER_URLS=""
for i in $(seq 0 $((NUM_WORKERS-1))); do
    WORKER_URLS="$WORKER_URLS http://localhost:$((BASE_PORT + i))"
done

# 运行实验
for scenario in "${SCENARIOS[@]}"; do
    for policy in "${ROUTING_POLICIES[@]}"; do
        echo "========================================"
        echo "Running: scenario=$scenario, policy=$policy"
        echo "========================================"

        # 清空所有 worker 的 cache
        for i in $(seq 0 $((NUM_WORKERS-1))); do
            curl -X POST http://localhost:$((BASE_PORT + i))/flush_cache
        done

        python marble_benchmark.py \
            --scenario $scenario \
            --routing-policy $policy \
            --num-traces $NUM_TRACES \
            --num-workers $NUM_WORKERS \
            --base-port $BASE_PORT \
            --output-dir results/${scenario}_${policy}

        echo "Completed: scenario=$scenario, policy=$policy"
    done
done

# 清理
echo "Stopping all workers..."
pkill -f "sglang.launch_server"
echo "Done!"
```

#### 7.5.3 指标采集脚本

```python
# benchmark/multi_agent/collect_metrics.py

import requests
import time
from typing import Dict, List

NUM_WORKERS = 10
BASE_PORT = 30000

def get_worker_urls(num_workers: int = NUM_WORKERS, base_port: int = BASE_PORT) -> List[str]:
    """获取所有 worker 的 URL"""
    return [f"http://localhost:{base_port + i}" for i in range(num_workers)]

def collect_worker_metrics(worker_urls: List[str] = None) -> Dict:
    """从所有 10 个 worker 采集 Prometheus 指标"""
    if worker_urls is None:
        worker_urls = get_worker_urls()

    metrics = {}
    for url in worker_urls:
        try:
            resp = requests.get(f"{url}/metrics", timeout=5)
            metrics[url] = parse_prometheus_metrics(resp.text)
        except Exception as e:
            print(f"Warning: Failed to collect metrics from {url}: {e}")
            metrics[url] = {}
    return metrics

def compute_cache_hit_ratio(before: Dict, after: Dict) -> float:
    """计算实验前后的 cache hit ratio（汇总 10 个 worker）"""
    delta_cached = sum(
        after[url].get("sglang:cached_tokens_total", 0) - before[url].get("sglang:cached_tokens_total", 0)
        for url in before
    )
    delta_prompt = sum(
        after[url].get("sglang:prompt_tokens_total", 0) - before[url].get("sglang:prompt_tokens_total", 0)
        for url in before
    )
    return delta_cached / (delta_cached + delta_prompt) if (delta_cached + delta_prompt) > 0 else 0

def compute_per_gpu_stats(before: Dict, after: Dict) -> Dict:
    """计算每个 GPU 的统计信息"""
    stats = {}
    for i, url in enumerate(get_worker_urls()):
        if url in before and url in after:
            delta_cached = after[url].get("sglang:cached_tokens_total", 0) - before[url].get("sglang:cached_tokens_total", 0)
            delta_prompt = after[url].get("sglang:prompt_tokens_total", 0) - before[url].get("sglang:prompt_tokens_total", 0)
            delta_requests = after[url].get("sglang:requests_total", 0) - before[url].get("sglang:requests_total", 0)

            stats[f"GPU_{i}"] = {
                "requests": delta_requests,
                "prompt_tokens": delta_prompt,
                "cached_tokens": delta_cached,
                "cache_hit_ratio": delta_cached / (delta_cached + delta_prompt) if (delta_cached + delta_prompt) > 0 else 0
            }
    return stats

def flush_all_caches(worker_urls: List[str] = None):
    """清空所有 10 个 worker 的 cache"""
    if worker_urls is None:
        worker_urls = get_worker_urls()

    for url in worker_urls:
        try:
            requests.post(f"{url}/flush_cache", timeout=5)
        except Exception as e:
            print(f"Warning: Failed to flush cache on {url}: {e}")
```

#### 7.5.4 完整 Benchmark 脚本（10 GPU 版本）

```python
# benchmark/multi_agent/marble_benchmark.py

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict
import aiohttp
import numpy as np

NUM_GPUS = 10
BASE_PORT = 30000

async def send_request(session: aiohttp.ClientSession, url: str, messages: List[Dict],
                       agent_id: str, workflow_id: str, step_id: str) -> Dict:
    """发送单个请求并记录延迟（Qwen3-4B-Thinking）"""
    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": 1024,  # Thinking 模型可能输出更长
        "temperature": 0.7,
        # Qwen3 thinking 模式会自动启用
    }

    start_time = time.perf_counter()
    ttft = None

    try:
        async with session.post(f"{url}/v1/chat/completions", json=payload) as resp:
            # 流式响应获取 TTFT
            async for chunk in resp.content.iter_any():
                if ttft is None:
                    ttft = time.perf_counter() - start_time
                break

            result = await resp.json()
            end_time = time.perf_counter()

            return {
                "success": True,
                "agent_id": agent_id,
                "workflow_id": workflow_id,
                "step_id": step_id,
                "latency": end_time - start_time,
                "ttft": ttft or (end_time - start_time),
                "prompt_tokens": result.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": result.get("usage", {}).get("completion_tokens", 0),
                "gpu_url": url
            }
    except Exception as e:
        return {
            "success": False,
            "agent_id": agent_id,
            "workflow_id": workflow_id,
            "step_id": step_id,
            "error": str(e),
            "gpu_url": url
        }

class Router:
    """路由器基类"""
    def __init__(self, num_gpus: int = NUM_GPUS, base_port: int = BASE_PORT):
        self.num_gpus = num_gpus
        self.base_port = base_port
        self.counter = 0

    def get_url(self, gpu_id: int) -> str:
        return f"http://localhost:{self.base_port + gpu_id}"

    def route(self, agent_id: str, workflow_id: str) -> str:
        raise NotImplementedError

class RoundRobinRouter(Router):
    def route(self, agent_id: str, workflow_id: str) -> str:
        gpu_id = self.counter % self.num_gpus
        self.counter += 1
        return self.get_url(gpu_id)

class AgentStickyHashRouter(Router):
    def route(self, agent_id: str, workflow_id: str) -> str:
        gpu_id = hash(agent_id) % self.num_gpus
        return self.get_url(gpu_id)

class WorkflowStickyHashRouter(Router):
    def route(self, agent_id: str, workflow_id: str) -> str:
        gpu_id = hash(workflow_id) % self.num_gpus
        return self.get_url(gpu_id)

def create_router(policy: str) -> Router:
    """创建路由器"""
    routers = {
        "round_robin": RoundRobinRouter,
        "agent_sticky_hash": AgentStickyHashRouter,
        "workflow_sticky_hash": WorkflowStickyHashRouter,
    }
    return routers.get(policy, RoundRobinRouter)()

async def run_benchmark(scenario: str, routing_policy: str, num_traces: int,
                        concurrency: int, output_dir: str):
    """运行 benchmark"""
    # 加载 traces
    trace_path = Path(f"MARBLE/multiagentbench/{scenario}/{scenario}_main.jsonl")
    traces = []
    with open(trace_path) as f:
        for i, line in enumerate(f):
            if i >= num_traces:
                break
            traces.append(json.loads(line))

    # 创建路由器
    router = create_router(routing_policy)

    # 构造请求
    all_requests = []
    for trace in traces:
        max_iter = 3 if scenario == "research" else 5
        for iteration in range(max_iter):
            for agent in trace["agents"]:
                all_requests.append({
                    "agent_id": agent["agent_id"],
                    "workflow_id": str(trace["task_id"]),
                    "step_id": f"iter{iteration}_{agent['agent_id']}",
                    "messages": [
                        {"role": "system", "content": agent["profile"]},
                        {"role": "user", "content": trace["task"]["content"]}
                    ]
                })

    print(f"Total requests: {len(all_requests)}")
    print(f"Routing policy: {routing_policy}")
    print(f"Concurrency: {concurrency}")

    # 运行请求
    results = []
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(session, req):
        async with semaphore:
            url = router.route(req["agent_id"], req["workflow_id"])
            return await send_request(
                session, url, req["messages"],
                req["agent_id"], req["workflow_id"], req["step_id"]
            )

    async with aiohttp.ClientSession() as session:
        start_time = time.perf_counter()
        tasks = [bounded_request(session, req) for req in all_requests]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time

    # 分析结果
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    latencies = [r["latency"] for r in successful]
    ttfts = [r["ttft"] for r in successful]

    stats = {
        "scenario": scenario,
        "routing_policy": routing_policy,
        "num_traces": num_traces,
        "total_requests": len(all_requests),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "total_time": total_time,
        "throughput": len(successful) / total_time,
        "latency_p50": np.percentile(latencies, 50) if latencies else 0,
        "latency_p95": np.percentile(latencies, 95) if latencies else 0,
        "latency_p99": np.percentile(latencies, 99) if latencies else 0,
        "ttft_p50": np.percentile(ttfts, 50) if ttfts else 0,
        "ttft_p95": np.percentile(ttfts, 95) if ttfts else 0,
        "ttft_p99": np.percentile(ttfts, 99) if ttfts else 0,
    }

    # 按 GPU 统计
    gpu_stats = {}
    for r in successful:
        gpu = r["gpu_url"]
        if gpu not in gpu_stats:
            gpu_stats[gpu] = {"count": 0, "latencies": []}
        gpu_stats[gpu]["count"] += 1
        gpu_stats[gpu]["latencies"].append(r["latency"])

    stats["per_gpu_distribution"] = {
        gpu: {"count": s["count"], "avg_latency": np.mean(s["latencies"])}
        for gpu, s in gpu_stats.items()
    }

    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    with open(output_path / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nResults saved to {output_dir}")
    print(f"Latency P50/P95/P99: {stats['latency_p50']:.3f}s / {stats['latency_p95']:.3f}s / {stats['latency_p99']:.3f}s")
    print(f"TTFT P50/P95/P99: {stats['ttft_p50']:.3f}s / {stats['ttft_p95']:.3f}s / {stats['ttft_p99']:.3f}s")
    print(f"Throughput: {stats['throughput']:.2f} req/s")

    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["coding", "research"], required=True)
    parser.add_argument("--routing-policy", default="round_robin")
    parser.add_argument("--num-traces", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        args.scenario, args.routing_policy, args.num_traces,
        args.concurrency, args.output_dir
    ))
```

#### 7.5.5 Failure 注入脚本（10 GPU 版本）

```bash
#!/bin/bash
# benchmark/multi_agent/inject_failure.sh

# 用法: ./inject_failure.sh <gpu_id> <failure_type> [duration_seconds]

GPU_ID=${1:-0}
FAILURE_TYPE=${2:-"kill"}
DURATION=${3:-30}
BASE_PORT=30000

WORKER_PORT=$((BASE_PORT + GPU_ID))
WORKER_PID=$(lsof -t -i:$WORKER_PORT)

case $FAILURE_TYPE in
    "kill")
        echo "Killing worker on GPU $GPU_ID (port $WORKER_PORT, PID $WORKER_PID)"
        kill -9 $WORKER_PID
        ;;
    "slow")
        echo "Injecting 500ms delay to GPU $GPU_ID for $DURATION seconds"
        # 需要 root 权限
        sudo tc qdisc add dev lo root netem delay 500ms
        sleep $DURATION
        sudo tc qdisc del dev lo root netem
        echo "Delay removed"
        ;;
    "restart")
        echo "Restarting worker on GPU $GPU_ID"
        kill -9 $WORKER_PID
        sleep 2
        CUDA_VISIBLE_DEVICES=$GPU_ID python -m sglang.launch_server \
            --model-path Qwen/Qwen3-4B-Thinking-2507 \
            --port $WORKER_PORT \
            --host 0.0.0.0 \
            --dp 1 \
            --enable-metrics \
            --reasoning-parser qwen3 \
            > logs/worker_${GPU_ID}.log 2>&1 &
        echo "Worker restarted"
        ;;
    *)
        echo "Unknown failure type: $FAILURE_TYPE"
        echo "Usage: $0 <gpu_id> <kill|slow|restart> [duration]"
        exit 1
        ;;
esac
```

### 7.6 实验结果记录模板（10 GPU 版本）

#### 7.6.1 实验一结果（待填充）

**整体指标**：

| 场景 | 路由策略 | Cache Hit Ratio | TTFT P50 (ms) | TTFT P95 (ms) | Latency P50 (ms) | Latency P95 (ms) | Throughput (req/s) |
|------|---------|-----------------|---------------|---------------|------------------|------------------|-------------------|
| Research | round_robin | - | - | - | - | - | - |
| Research | agent_sticky_hash | - | - | - | - | - | - |
| Research | workflow_sticky_hash | - | - | - | - | - | - |
| Research | cache_aware | - | - | - | - | - | - |
| Coding | round_robin | - | - | - | - | - | - |
| Coding | agent_sticky_hash | - | - | - | - | - | - |
| Coding | workflow_sticky_hash | - | - | - | - | - | - |
| Coding | cache_aware | - | - | - | - | - | - |

**Per-GPU 请求分布（验证路由策略生效）**：

| 路由策略 | GPU0 | GPU1 | GPU2 | GPU3 | GPU4 | GPU5 | GPU6 | GPU7 | GPU8 | GPU9 | 分布均匀度 |
|---------|------|------|------|------|------|------|------|------|------|------|-----------|
| round_robin | - | - | - | - | - | - | - | - | - | - | 高 |
| agent_sticky (coding) | agent1 | agent1 | agent1 | agent2 | agent2 | agent2 | agent3 | agent3 | agent3 | overflow | 按 agent 聚集 |
| agent_sticky (research) | a1 | a2 | a3 | a4 | a5 | a1 | a2 | a3 | a4 | a5 | 按 agent 聚集 |

#### 7.6.2 实验二结果（待填充）

**Coding 场景 Failure 注入**：

| 备份策略 | 失败 GPU | 失败 Agent | ΔLatency (%) | Recompute Tokens | Recovery Time (ms) | 受影响请求数 |
|---------|---------|-----------|--------------|------------------|-------------------|-------------|
| M0-NoBackup | GPU0 | agent1 | - | - | - | - |
| M0-NoBackup | GPU6 | agent3 | - | - | - | - |
| M2-CriticalOnly | GPU0 | agent1 | - | - | - | - |
| M2-CriticalOnly | GPU6 | agent3 | - | - | - | - |

**Research 场景 Failure 注入**：

| 备份策略 | 失败 GPU | 失败 Agent | ΔLatency (%) | Recompute Tokens | Recovery Time (ms) | 受影响请求数 |
|---------|---------|-----------|--------------|------------------|-------------------|-------------|
| M0-NoBackup | GPU0 | agent1 | - | - | - | - |
| M1-AllBackup | GPU0 | agent1 | - | - | - | - |
| M2-TopK(2) | GPU0 | agent1 | - | - | - | - |

**多 GPU 同时失败测试**：

| 场景 | 失败 GPU 数 | 失败的 Agent | 备份策略 | 系统是否可用 | ΔLatency (%) |
|------|-----------|-------------|---------|-------------|--------------|
| Coding | 1 (GPU0) | agent1 部分 | M2-CriticalOnly | 是（有副本） | - |
| Coding | 3 (GPU0,1,2) | agent1 全部 | M2-CriticalOnly | 否 | N/A |
| Coding | 2 (GPU0,3) | agent1+agent2 部分 | M2-CriticalOnly | 是 | - |

#### 7.6.3 实验三结果（待填充）

**Graph-Aware 调度（Coding 场景）**：

| 调度策略 | 并发数 | Makespan (s) | Critical Path Latency (s) | P95 (ms) | P99 (ms) |
|---------|-------|-------------|--------------------------|----------|----------|
| FIFO | 10 | - | - | - | - |
| FIFO | 20 | - | - | - | - |
| FIFO | 50 | - | - | - | - |
| Criticality-Priority | 10 | - | - | - | - |
| Criticality-Priority | 20 | - | - | - | - |
| Criticality-Priority | 50 | - | - | - | - |

### 7.7 预期论文贡献点

基于上述实验，预期可形成以下贡献：

1. **Characterization**：首次系统分析 multi-agent workflow 的图结构特征与 LLM serving 的交互（prefix 复用、关键路径、blast radius）

2. **Agent-Aware Routing**：提出并验证 agent-sticky 路由策略，在 MARBLE 数据集上实现 X% cache hit 提升和 Y% 延迟降低

3. **Critical-Only KV Backup**：提出基于 criticality 的选择性 KV 备份策略，在 Coding 场景实现 Z% 的 failure recovery 加速，同时备份开销仅为全量备份的 1/3

4. **Graph-Aware Scheduling**：验证 criticality-based 优先级调度在高负载下对尾延迟的改善效果

