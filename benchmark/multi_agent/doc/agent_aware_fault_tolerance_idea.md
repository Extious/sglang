# Agent-Aware 容错路由思路（论文草案）

## 1. 问题背景
多 Agent 应用与模型推理层解耦后，任务时延不仅取决于模型算力，也受到路由策略与 KV Cache 复用效率影响。

在典型 Heavy-Swarm 工作流中，关键路径可抽象为：
- A: Question Agent
- C: Analysis Agent
- F: Synthesis Agent

若关键路径上某个 Agent 请求集中在单个 worker，而该 worker 故障，则重试请求会被路由到其它 worker。由于目标 worker 缺少对应前缀的 KV Cache，请求需要重新 prefill，导致恢复阶段时延显著上升。

## 2. 核心想法
提出 **Agent-Aware Router**：在保持负载可控的前提下，优先保障关键路径 Agent 的跨 worker Cache 覆盖。

与传统 Cache-Aware（按当前命中/负载贪心）不同，Agent-Aware 在长期分布上显式约束“哪些 Agent 应该出现在哪些 worker”，从而在故障后仍可复用 system prompt 等公共前缀缓存。

## 3. 目标分布设计
设 A-F 分别代表：Question / Research / Analysis / Alternatives / Verification / Synthesis。

给定 6 个 worker，设计分布为：
- worker1,2: A B C F
- worker3,4: A C D F
- worker5,6: A C E F

设计动机：
1. A/C/F 关键路径全覆盖到全部 worker，提升 failover 后缓存复用概率。
2. B/D/E 非关键路径分组分摊，避免所有 Agent 全量复制导致缓存碎片化。
3. 在容错与容量之间取得折中：关键路径高冗余，非关键路径低冗余。

## 4. 故障场景下的预期收益
当任一 worker 故障时：
1. 原 worker 上未完成请求触发重试。
2. 重试路由到存活 worker。
3. 对 A/C/F 请求，目标 worker 大概率已有对应前缀缓存，prefill 成本下降。
4. 端到端表现为：
   - 关键路径时延（尤其 P95）下降
   - 故障后恢复曲线更平滑
   - 总任务完成时间抖动减小

## 5. 与基线方法的对比定义
- **Baseline**: Official Cache-Aware Router
- **Proposed**: Agent-Aware Router（显式 Agent-Worker 覆盖约束）

公平性原则：
1. 相同 worker 部署与模型配置。
2. 每轮实验前统一 flush KV Cache。
3. 相同任务集与并发设置。
4. 相同故障注入流程（仅在 router 层注入）。

## 6. 失效与恢复模型（用于实验）
- 故障注入：router 暂时不可达某个 worker（逻辑 down）。
- 注入时机：执行到指定任务进度（例如完成 40 或 50 个 task）后触发。
- 恢复时机：故障后固定时间恢复（例如 30s）。

该模型模拟线上常见场景：单机短暂失效、网络隔离、节点重启。

## 7. 关键评估指标
建议至少报告以下指标：
1. Task 完成时延：mean / P50 / P95
2. 关键路径时延：A+C+F 聚合时延（尤其 P95）
3. 故障后窗口时延：post-failure P95
4. 请求级缓存复用：
   - 每 task-每 agent 的 cache hit rate
   - 全局 prefill cached_tokens / prompt_tokens
5. 路由容错行为：failover 请求数、失败重试成功率
6. worker 负载分布：按 worker 请求量与 prefill token 分布

## 8. 论文主张（可检验）
本文主张：
> 面向多 Agent 工作流的路由策略应显式建模关键路径的可恢复性，而不仅是即时缓存命中率；通过关键路径 Agent 的跨 worker 覆盖，可在故障场景下显著降低尾时延。

可检验假设：
- H1: 在 worker failure 场景下，Agent-Aware 的关键路径 P95 低于 Cache-Aware。
- H2: Agent-Aware 的 post-failure 恢复段时延斜率更小（恢复更快）。
- H3: Agent-Aware 在关键路径 Agent 上的 cache hit rate 更高或下降更小。

## 9. 方法边界与风险
1. 若总内存/缓存预算过小，关键路径冗余可能挤占非关键路径缓存。
2. 若任务分布与关键路径假设不一致（例如 B/D/E 成为瓶颈），收益会减弱。
3. 若故障持续时间远大于缓存有效时间，冗余缓存优势会下降。

## 10. 后续可扩展方向
1. 动态关键路径识别（随 workload 变化在线更新 A/C/F 集合）。
2. 多目标路由优化（时延、命中率、负载均衡、显存占用联合优化）。
3. 从静态覆盖扩展到概率覆盖（基于故障风险与流量预测自适应分配）。
