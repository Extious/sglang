# Experiment 2：Critical-Only KV 备份（验证 H2）

## 思路

**假设 H2**：仅对 **critical agent/step** 做 KV 备份（或写回 L2/L3），能在 instance failure 时显著缩小 **blast radius**（需重算的节点/请求数、ΔLatency、recompute tokens），同时控制备份开销，优于「无备份」或「全量备份」。

**做法**：基于 workflow 图做 criticality 分析（关键路径、dominator、下游覆盖度等），标出高 criticality 节点；对比策略：M0 无备份、M1 全量写回、M2 仅 critical 节点写回、M3 critical 写回 L3 等。在受控条件下注入 instance failure（如单/多 worker 失效），测量恢复前后延迟增量、重算 token 数、恢复时间等。

**核心指标**：blast radius（受影响请求/节点比例）、ΔLatency、recompute tokens、recovery time；以及备份侧开销（写回带宽、存储占用）。预期 M2 critical-only 在 dominator 明显的场景（如链式/星形）性价比最高。

**不涉及**：具体设备型号、GPU 映射、注入脚本名等；仅描述实验目的、criticality 设定、备份策略对照与指标。
