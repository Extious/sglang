# Experiment 3：MAS 调度（基于 criticality 优先级）（验证 H3）

## 思路

**假设 H3**：利用 workflow 图与 **criticality** 做请求/step **优先级调度**（例如关键路径上的 agent 或 dominator 优先），在高负载、排队明显时能缩短关键路径完成时间与 makespan，改善 P95/P99 尾延迟。

**做法**：在相同 workload 与并发下，对比 FIFO、按 criticality 排序、SJF（按预估 token/时长）、以及 criticality + SJF 等策略。优先调度高 criticality 的 step，使关键路径少受排队影响。观察整体 makespan、关键路径延迟、以及 P95/P99 是否随 criticality-aware 调度而改善。

**核心指标**：makespan、critical path latency（或等价定义）、latency P95/P99。预期在高负载下，criticality-based 或 criticality+SJF 相比 FIFO 有可见的尾延迟下降。

**不涉及**：具体并发数、设备规模、队列实现等；仅描述实验目的、调度策略对照与指标。
