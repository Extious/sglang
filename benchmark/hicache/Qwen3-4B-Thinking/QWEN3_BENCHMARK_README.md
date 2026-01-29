# Qwen3-4B-Thinking-2507 基准测试（HiCache）说明

本文档说明如何在 SGLang 上对 `Qwen/Qwen3-4B-Thinking-2507` 进行服务侧基准测试，并基于 `plots/` 中的图像对结果做解读与建议。

## 目录结构

- `run_qwen3_benchmark.slurm`：一键跑基准测试（SLURM）
- `results/`：原始结果（JSONL，每行对应一个 request_rate）
- `plot_results.py`：把 `results/*.jsonl` 画成曲线图
- `plots/`：绘图输出（PNG）
- `logs/`：服务端与 benchmark 日志

## 测试配置（Configurations）

本目录下的结果/图像包含以下配置（legend 名称与 `results/` 文件名一致）：

1. `disable-radix-cache`：关闭前缀缓存（`--disable-radix-cache`），命中率应接近 0
2. `schedule-policy-fcfs`：启用前缀缓存，但调度策略为 FCFS（`--schedule-policy fcfs`）
3. `long-prefix-match`：启用前缀缓存，调度策略为 LPM（`--schedule-policy lpm`，Long Prefix Match）
4. `enable-hierarchical-cache`：启用分层缓存（`--enable-hierarchical-cache`）

说明：
- `plot_results.py` 会把 `results/` 里“存在的所有配置文件”都画出来；如果你重新跑 benchmark 但只生成了 3 个配置文件，那么图里就只会出现 3 条曲线。
- 目前 `run_qwen3_benchmark.slurm` 默认包含的配置以脚本里 `CONFIGS` 为准；如需完整复现本 README 的 4 组对比，可在脚本中补齐 `schedule-policy-fcfs`，并为 `long-prefix-match` 显式设置 `--schedule-policy lpm`。

## 数据集

当前结果来自 ShareGPT 多轮对话数据集（`sharegpt`）。从已有结果统计看：
- 平均输入长度约 `1660` tokens
- 平均输出长度 `64` tokens
- 每个 request_rate 测试 `total_requests=1280`

## 前置条件

1. 下载数据集：
   ```bash
   cd /home/comp/zhanzhao/sglang/benchmark/hicache
   ./download.sh sharegpt
   ```

2. 需要可用的 SLURM 计算资源（以 `run_qwen3_benchmark.slurm` 为准），当前脚本默认：
   - `--partition=short`
   - `--gres=gpu:a100:1`
   - `--cpus-per-task=16`
   - `--mem=150G`
   - `--time=24:00:00`
   - `--exclusive`

3. 绘图依赖 `matplotlib`（用于运行 `plot_results.py`）。

## 运行方法（SLURM）

在本目录提交任务：

```bash
cd /home/comp/zhanzhao/sglang/benchmark/hicache/Qwen3-4B-Thinking
sbatch run_qwen3_benchmark.slurm
```

可选：覆盖脚本中的环境变量（示例）：

```bash
sbatch --export=ALL,MODEL_PATH=Qwen/Qwen3-4B-Thinking-2507,MEM_FRACTION_STATIC=0.85 run_qwen3_benchmark.slurm
```

## 监控与日志

```bash
squeue -u "$USER"

# SLURM stdout/stderr（JOB_ID 替换为实际值）
tail -f logs/qwen3_benchmark_JOB_ID.out
tail -f logs/qwen3_benchmark_JOB_ID.err

# 每个配置的服务端与 benchmark 日志
tail -f logs/server_disable-radix-cache.log
tail -f logs/bench_disable-radix-cache.log
```

## 结果文件

结果保存在 `results/`，命名格式：

```
results/qwen3_<config>_<YYYYmmdd_HHMMSS>.jsonl
```

每行是一个 `summary`（对应一个 `request_rate`），常用字段含义：
- `request_rate`：压测目标请求速率（req/s）
- `throughput`：实际完成吞吐（req/s，饱和后会低于 `request_rate`）
- `average_ttft`：平均首 token 延迟（s）
- `average_latency`：平均端到端延迟（s）
- `p90_latency`：P90 端到端延迟（s）
- `output_token_throughput`：输出 token 吞吐（tokens/s）
- `cache_hit_rate`：前缀缓存命中率（0~1）

## 绘图

在本目录执行：

```bash
python3 plot_results.py
```

生成的图像在 `plots/` 下：
- `plots/throughput.png`
- `plots/output_token_throughput.png`
- `plots/average_ttft.png`
- `plots/average_latency.png`
- `plots/p90_latency.png`
- `plots/cache_hit_rate.png`

## 基于图像的结果解读（plots/）

### 总览结论（以 `request_rate=16` 为例）

下表摘取了 `request_rate=16` 时的关键指标（来自 `results/*.jsonl`，与图中对应点一致）：

| 配置 | throughput (req/s) | avg TTFT (s) | avg latency (s) | p90 latency (s) | out tok tput (tok/s) | hit rate |
| --- | ---:| ---:| ---:| ---:| ---:| ---:|
| disable-radix-cache | 5.527 | 1.522 | 16.171 | 39.838 | 353.753 | 0.000 |
| enable-hierarchical-cache | 5.383 | 2.329 | 16.307 | 37.708 | 344.513 | 0.663 |
| long-prefix-match | 7.606 | 0.214 | 5.802 | 29.241 | 486.795 | 0.606 |
| schedule-policy-fcfs | 7.113 | 1.105 | 4.214 | 11.030 | 455.207 | 0.596 |

可见：
- **吞吐/输出 token 吞吐**：`long-prefix-match` 最高，其次 `schedule-policy-fcfs`；关闭前缀缓存或启用分层缓存都明显更低。
- **TTFT**：`long-prefix-match` 在高压下依然很低（优势非常明显）。
- **尾延迟**：在极高压点（16 req/s）下，`schedule-policy-fcfs` 的 `p90_latency` 明显更好；`long-prefix-match` 更容易出现长尾。
- **命中率**：`enable-hierarchical-cache` 最高，但并未转化为吞吐/延迟优势（需要进一步排查/调参）。

### 吞吐（Throughput）

![Throughput vs Request Rate](plots/throughput.png)

- 随着 `request_rate` 增加，各配置的实际 `throughput` 会逐步饱和。
- `long-prefix-match`/`schedule-policy-fcfs` 的饱和吞吐更高（约 7 req/s 量级），而 `disable-radix-cache`/`enable-hierarchical-cache` 更早到达瓶颈（约 5~5.5 req/s）。

### 输出 Token 吞吐（Output Token Throughput）

![Output Token Throughput vs Request Rate](plots/output_token_throughput.png)

- 趋势与 `throughput` 一致：`long-prefix-match` 最高、`schedule-policy-fcfs` 次之。
- 说明提升主要来自“更快完成请求/更高的解码产出”，而不是仅仅提高缓存命中率指标本身。

### 首 Token 延迟（TTFT）

![Average TTFT vs Request Rate](plots/average_ttft.png)

- `long-prefix-match` 在高压下 TTFT 依然保持较低，适合对交互体验敏感的场景。
- `disable-radix-cache` 与 `enable-hierarchical-cache` 在高压下 TTFT 增长明显。

### 平均端到端延迟（Average Latency）

![Average Latency vs Request Rate](plots/average_latency.png)

- 高压下（例如 16 req/s），关闭前缀缓存/启用分层缓存会出现显著的平均延迟上升。
- `schedule-policy-fcfs` 在极高压点的平均延迟更低，但需要结合 TTFT 与尾延迟一起看取舍。

### P90 端到端延迟（P90 Latency）

![P90 Latency vs Request Rate](plots/p90_latency.png)

- `schedule-policy-fcfs` 在高压点的尾延迟更稳；`long-prefix-match` 可能为了更好的前缀复用而牺牲公平性，从而带来更长的 tail。
- 如果线上更关注“高分位延迟 SLA”，建议重点对比这一张图。

### 缓存命中率（Cache Hit Rate）

![Cache Hit Rate vs Request Rate](plots/cache_hit_rate.png)

- `disable-radix-cache` 命中率为 0，符合预期。
- `long-prefix-match` 与 `schedule-policy-fcfs` 命中率在 ~0.6 附近，说明 ShareGPT 场景存在较强的前缀可复用性。
- `enable-hierarchical-cache` 命中率更高（~0.66），但吞吐与延迟并未改善：可能存在额外开销（例如数据搬运/管理成本）抵消收益，建议结合 profiler/日志进一步定位。

## 建议（如何选配置）

- 追求更好的交互体验（低 TTFT）与更高吞吐：优先考虑 `long-prefix-match`（`--schedule-policy lpm`）。
- 更关注尾延迟稳定性：对比 `schedule-policy-fcfs`（`--schedule-policy fcfs`），尤其在高压区间。
- `enable-hierarchical-cache`：当前结果仅体现“命中率更高”，但未体现吞吐/延迟收益；建议在明确瓶颈与开销来源后再作为默认选项。

## 常见问题排查

### 服务启动失败
- 查看 `logs/server_<config>.log`
- 检查 `MODEL_PATH` 是否可用、GPU 显存是否足够
- 检查端口占用（脚本默认 `PORT=30000`）

### benchmark 失败或结果为空
- 查看 `logs/bench_<config>.log`
- 确认 ShareGPT 数据已下载完成（见“前置条件”）

### 图像生成失败
- 确认本机/节点安装了 `matplotlib`
- 确认 `results/` 下存在至少一个 `qwen3_*.jsonl`
