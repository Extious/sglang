# Multi-Agent Serving (MAS) Report

> 背景：上层 multi-agent 编排与下层 LLM serving 解耦，workflow 的图结构（DAG/依赖）未进入调度与 KV 管理，导致 latency 与 failure 影响半径优化不足。
>
> 研究切入点：1) **Fault tolerance**——识别并保护 critical agent/step，对关键节点 KV 做选择性备份与恢复；2) **Agent-aware KV cache**——以 agent 为 locality 单元做路由与缓存，提高 cache hit、降低 TTFT/尾延迟。

---

## 1. 问题与统一视角

**现状缺口**：上层 workflow 可表示为执行图（节点=agent step，边=依赖），下层 serving 按「独立请求」优化，不理解 workflow/agent/关键路径；调度无法利用图结构，KV 对 agent 无感，failure 时缺乏 graph-aware recovery。

**统一视角**：为每个 workflow 维护执行图 + 关键性（criticality），驱动**调度/路由策略**（agent-aware、dependency-aware、load-aware）与 **KV 策略**（分层、隔离、选择性备份/多副本），同时优化 latency（P95/P99）与 failure 影响半径。

---

## 2. 相关工作（serving 层）

- **KV / prefix cache**：vLLM PagedAttention、SGLang RadixAttention，为 KV 复用/迁移提供基础；Radix 适合多请求共享长 system prompt。
- **分层 KV（HiCache）**：L1 GPU / L2 host / L3 distributed，支持 prefetch、写回策略；研究重点是**策略**（哪些 agent/节点需更积极写回/多副本/预取）。
- **Prefill/Decode 解耦**：与 multi-agent 关键路径/关键节点差异高度相关。
- **数据**：MAST-Data、TRAIL、Multi-Agents-Parallel-Orchestration-Dataset、multiagent-router-finetuning 等可用于从 trace 抽图、做 criticality 分析与实验。

---

## 3. 场景建模

**Workflow 图**：`G=(V,E)`，节点 v = 一次 agent step，边 (u→v) = v 依赖 u 的输出。常见形态：Chain、Star/Planner-Fanout-Join、Diamond、Loop（展开为 DAG）。

**Serving 资源**：K 个实例；每次调用有 T_prefill、T_decode、S_kv；KV 命中可分层 L1/L2/L3。

**Failure 与爆炸半径**：`BR(v)` = 节点 v 所在实例失败时需重算/等待的下游集合。可度量：重算 token 成本、ΔLatency、abort 概率。

---

## 4. Critical agent / step：定义与识别

- **Latency-critical**：在关键路径上或造成 join 前瓶颈。
- **Reliability-critical**：失败导致大规模重算或 workflow abort（blast radius 大）。

**静态**：权重 w(v)、关键路径（最长路径）、下游覆盖度、支配点（dominator）；可组合为 `C_static(v)`。**动态**：结合运行时输入/输出长度、负载、cache hit、失败概率等，如 `C_dyn(v)`。初期可用关键路径 + fanout 等 heuristics 验证 critical-only 保护收益。

---

## 5. Fault tolerance：选择性 KV 备份

**保护对象**：只备份输出 → 备份 prefix KV（性价比高，与 radix 契合）→ 完整会话 KV（收益大、成本高）。

**落点**：L2 host（同机恢复）、L3 distributed（跨机、抵御 node failure）。

**策略**：critical-only 写回；写回时机（同步/异步）；关键节点多副本、非关键单副本或不备份。

**服务模式**：M0 无缓存；M1 本地无共享；M2 agent-sticky（需备份对冲）；M3 shared L3；**M4 M2 + critical-only 备份**（主打方向）。

---

## 6. Agent-aware KV cache：路由与隔离

**假设**：同一 agent 复用稳定 system prompt/工具描述；请求随机打散导致 L1/L2 命中下降、TTFT/尾延迟上升。

**路由**：agent-sticky hash → locality-first + load-guard → graph-aware（关键路径/节点优先局部性，非关键分支负载均衡）。

**隔离**：按 agent 设 cache quota；eviction 扩展为「热度 + criticality」。
