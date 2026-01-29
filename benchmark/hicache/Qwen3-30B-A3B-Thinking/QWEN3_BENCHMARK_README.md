# Qwen3-30B-A3B-Thinking-2507 基准测试（HiCache）说明

本文档说明如何在 SGLang 上对 `Qwen/Qwen3-30B-A3B-Thinking-2507` 进行服务侧基准测试，并基于 `plots/` 中的图像对结果做解读与建议。

## 目录结构

- `run_qwen3_benchmark.slurm`：一键跑基准测试（SLURM）
- `results/`：原始结果（JSONL，每行对应一个 request_rate）
- `plot_results.py`：把 `results/*.jsonl` 画成曲线图
- `plots/`：绘图输出（PNG）
- `logs/`：服务端与 benchmark 日志

## 测试配置（Configurations）

本目录下的结果/图像包含以下配置（legend 名称与 `results/` 文件名一致）：

1. `disable-radix-cache`：关闭前缀缓存（`--disable-radix-cache`），命中率应接近 0
2. `schedule-policy-fcfs`：启用前缀缓存，调度策略为 FCFS（`--schedule-policy fcfs`，First Come First Served）
3. `long-prefix-match`：启用前缀缓存，调度策略为 LPM（`--schedule-policy lpm`，Long Prefix Match）
4. `enable-hierarchical-cache`：启用分层缓存（`--enable-hierarchical-cache`）

说明：
- `plot_results.py` 会把 `results/` 里“存在的所有配置文件”都画出来；如果你重新跑 benchmark 但只生成了部分配置文件，那么图里就只会出现对应的曲线。

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
   - `--gres=gpu:a100:2`（使用 2 张 A100 GPU）
   - `--cpus-per-task=16`
   - `--mem=320G`
   - `--time=24:00:00`
   - `--exclusive`
   
   注意：由于 Qwen3-30B-A3B-Thinking 模型参数较大，脚本默认使用 2 张 GPU 进行并行推理。支持两种并行模式：
   - **Pipeline Parallelism (PP)**：默认模式，使用 `--pp-size 2` 将模型按层切分到 2 张 GPU（适合 PCIe 连接的 GPU）
   - **Tensor Parallelism (TP)**：使用 `--tp 2` 将模型张量切分到 2 张 GPU（需要高带宽互连如 NVLink）

3. 绘图依赖 `matplotlib`（用于运行 `plot_results.py`）。

## 运行方法（SLURM）

在本目录提交任务：

```bash
cd /home/comp/zhanzhao/sglang/benchmark/hicache/Qwen3-30B-A3B-Thinking
sbatch run_qwen3_benchmark.slurm
```

可选：覆盖脚本中的环境变量（示例）：

```bash
# 使用默认的 Pipeline Parallelism (PP=2)
sbatch --export=ALL,MODEL_PATH=Qwen/Qwen3-30B-A3B-Thinking-2507,MEM_FRACTION_STATIC=0.85 run_qwen3_benchmark.slurm

# 使用 Tensor Parallelism (TP=2)
sbatch --export=ALL,MODEL_PATH=Qwen/Qwen3-30B-A3B-Thinking-2507,PARALLEL_MODE=tp,MEM_FRACTION_STATIC=0.85 run_qwen3_benchmark.slurm
```

环境变量说明：
- `PARALLEL_MODE`：并行模式，可选 `pp`（默认）或 `tp`
- `MODEL_PATH`：模型路径，默认 `Qwen/Qwen3-30B-A3B-Thinking-2507`
- `MEM_FRACTION_STATIC`：GPU 内存使用比例，默认 `0.85`

## 监控与日志

```bash
squeue -u "$USER"

# SLURM stdout/stderr（JOB_ID 替换为实际值）
tail -f logs/qwen3_30b_a3b_benchmark_JOB_ID.out
tail -f logs/qwen3_30b_a3b_benchmark_JOB_ID.err

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
