# Experiment 1：Agent-Aware 路由（验证 H1）

## 思路

**假设 H1**：路由策略越能维持 **agent 局部性**（同一 agent 的请求尽量落到同一实例或同一 cache domain），prefix/KV cache 命中率越高，TTFT 与端到端延迟越低。

**做法**：在相同 workload 下对比多种路由策略——例如 round-robin / random（baseline，无局部性）、agent-sticky（按 agent_id 固定或哈希到 worker）、workflow-sticky（同 workflow 同实例）、以及 locality-first + load-guard（优先 agent home worker，超阈值再 fallback）。观察 cache hit ratio、latency/TTFT 的 P50/P95/P99 是否随 agent 局部性增强而改善。

**核心指标**：cache hit ratio（或 cached tokens / prompt tokens 相关度量）、TTFT、端到端 latency。可选：按 agent 的命中率分布，验证路由是否按预期将同 agent 请求聚集。

**不涉及**：具体 trace 条数、GPU 数量与映射、脚本名等实现细节；仅描述实验目的、对照策略与指标。
