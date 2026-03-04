# Fault Tolerance Testing - Optimized Configuration

## 改进总结

### 问题
原始配置下，worker 故障时延迟增加到 100+ 秒，远超预期的 2 倍正常延迟。

### 根本原因
1. **HTTP 超时过长**：默认 30 秒 worker 超时
2. **健康检查慢**：60 秒检查一次，3 次失败才标记 unhealthy（需要 3 分钟）
3. **无流式传输**：无法快速检测连接中断

### 解决方案

#### 1. 路由器优化配置
修改了 `run_router.sh`，添加以下参数：

```bash
--request-timeout-secs 5          # 5 秒超时（vs 默认 1800s）
--health-check-interval-secs 5    # 5 秒检查一次（vs 默认 60s）
--health-failure-threshold 2      # 2 次失败标记 unhealthy（vs 默认 3）
--health-success-threshold 2      # 2 次成功恢复 healthy（vs 默认 2）
--retry-max-retries 2             # 最多重试 2 次（vs 默认 5）
```

**预期效果**：
- 单次请求超时：5 秒
- Worker 标记为 unhealthy：10 秒（2 次失败 × 5 秒）
- 重试延迟：5 + 正常处理时间
- **总延迟：15-20 秒**（接近 2 倍正常延迟）

#### 2. 启用流式传输
修改了 `heavy_swarm.py`，在 HeavySwarm 初始化时添加：

```python
streaming_on=True  # 启用流式传输
```

**好处**：
- 更快检测连接中断
- 减少等待完整响应的时间
- 改善用户体验

## 使用方法

### 方式 1：完整自动化测试（推荐）

```bash
cd /home/comp/25482351/sglang/benchmark/multi_agent/script

# 运行完整测试（自动重启路由器 + 故障注入）
./run_fault_tolerance_test.sh
```

这个脚本会：
1. 停止现有路由器
2. 启动优化配置的路由器
3. 运行故障注入测试

### 方式 2：手动步骤

#### 步骤 1：重启路由器
```bash
# 停止现有路由器
pkill -f "sglang::router"

# 启动优化配置的路由器
cd /home/comp/25482351/sglang/benchmark/multi_agent/script
./run_router.sh --worker-urls-file ../logs/worker_urls.txt
```

#### 步骤 2：运行故障注入测试
```bash
# 获取 SLURM Job ID
SLURM_JOB_ID=$(squeue -u $USER -h -o "%i" | head -1)

# 运行测试
./run_heavy_swarm_with_fault_injection.sh \
  --failed-worker-url http://hkbugpusrv10:8000 \
  --slurm-job-id $SLURM_JOB_ID \
  --inject-after 300 \
  --recover-after 30
```

## 预期结果对比

### 优化前
- 正常任务延迟：~30 秒
- 故障期间延迟：**100+ 秒**
- 倍数：**3-4x**

### 优化后
- 正常任务延迟：~30 秒
- 故障期间延迟：**~50-60 秒**
- 倍数：**~2x**（符合预期）

## 时间线分析

### 故障注入时刻（T=0）
```
T+0s:   Worker 被 kill -STOP 挂起
T+5s:   第 1 次健康检查失败
T+10s:  第 2 次健康检查失败 → Worker 标记为 unhealthy
T+10s+: 新请求不再路由到故障 worker
```

### 正在进行的请求
```
请求发送到故障 worker
    ↓
等待 5 秒超时
    ↓
重试到健康 worker（立即，因为故障 worker 已标记 unhealthy）
    ↓
正常处理（~10-15 秒）
    ↓
总延迟：5 + 10-15 = 15-20 秒
```

### 故障恢复（T+30s）
```
T+30s:  Worker 被 kill -CONT 恢复
T+35s:  第 1 次健康检查成功
T+40s:  第 2 次健康检查成功 → Worker 标记为 healthy
T+40s+: Worker 重新接收请求
```

## 监控和日志

### 路由器日志
```bash
tail -f /home/comp/25482351/sglang/benchmark/multi_agent/logs/router_optimized.log
```

### 故障注入日志
```bash
tail -f /home/comp/25482351/sglang/benchmark/multi_agent/logs/fault_injection_optimized_*.log
```

### 性能报告
测试完成后查看：
```bash
ls -lt /home/comp/25482351/sglang/benchmark/multi_agent/src/application/swarms/agent_workspace/timing_reports/
```

## 故障排查

### 如果延迟仍然很高

1. **检查路由器配置**
```bash
ps aux | grep "sglang::router"
# 确认路由器使用了优化参数
```

2. **检查网络延迟**
```bash
# 测试到 worker 的延迟
time curl --noproxy '*' http://hkbugpusrv10:8000/health
```

3. **检查并发请求数**
```bash
# 减少并发进程
export HEAVY_SWARM_TASK_PROCESSES=3  # 从 6 减少到 3
```

### 如果流式传输不工作

检查 HeavySwarm 是否支持 `streaming_on` 参数：
```bash
grep -A5 "streaming_on" /home/comp/25482351/sglang/benchmark/multi_agent/src/application/swarms/swarms/swarms/structs/heavy_swarm.py
```

如果不支持，可以移除该参数，优化的超时配置仍然有效。

## 文件修改清单

1. ✅ `run_router.sh` - 添加优化的超时参数
2. ✅ `heavy_swarm.py` - 启用流式传输
3. ✅ `run_fault_tolerance_test.sh` - 新建完整测试脚本
4. ✅ `run_heavy_swarm_with_fault_injection.sh` - 已修复（之前的工作）

## 回滚方法

如果需要恢复原始配置：

```bash
cd /home/comp/25482351/sglang/benchmark/multi_agent/script
git diff run_router.sh
git checkout run_router.sh

cd ../src/application/swarms
git diff heavy_swarm.py
git checkout heavy_swarm.py
```
