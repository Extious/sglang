# Experiment 1: Agent-Aware Routing

目标：使用 MARBLE 数据集，用 LangChain 构建 multi-agent 上层应用；底层用 SGLang workers + router，验证不同路由策略对 cache hit / latency 的影响。

## 集群与资源建议

- Server 部署：使用 **long 分区 + rtx4090**（例如 gpu10-19），**排除 debug 节点**（gpu10-13）。
- 默认配置：**3 个节点，每节点 2 张 GPU，总共 5 个 workers**（节点 0-1 各 2 个，节点 2 有 1 个）。
- 每个 rtx4090 节点通常 2 卡；可通过 `NUM_WORKERS` 环境变量调整 worker 数量（默认 5）。

## 架构与内网地址

- Server：启动多个 SGLang workers，绑定 `0.0.0.0`，将可被其他节点访问的地址写入 `logs/worker_urls.txt`（如 `http://gpu10:8000`）。
- Router：读取 `logs/worker_urls.txt`，启动 router，绑定 `0.0.0.0`，写入 `logs/router_url.txt`（如 `http://gpu20:30000`）。
- Client：LangChain 应用只连接 router（读 `logs/router_url.txt` 或传 `--router-url`）。

注意：跨节点场景必须使用**内网主机名或内网 IP**，不要用 `127.0.0.1`。

## 文件说明

| 文件 | 说明 |
|------|------|
| `run_experiment1.sh` | **一键运行脚本**：提交 server job，监控部署，启动 router，运行 client |
| `deploy_server.py` | 在当前节点启动 workers，写 `logs/worker_urls.txt`、`logs/server_node.txt` |
| `deploy_router.py` | 在当前机器启动 router，读 `logs/worker_urls.txt`，写 `logs/router_url.txt`、`logs/router_node.txt` |
| `run_langchain_app.py` | LangChain 客户端，读 `logs/router_url.txt` 或参数，写 `results/*.jsonl` |
| `run_server.slurm` | 仅用于部署 server（long + rtx4090） |

## 使用方法

### 方式一：一键部署（推荐）

```bash
cd benchmark/multi_agent/experiment1
mkdir -p logs results

# 设置环境变量（可选）
export ROUTER_POLICY=round_robin

# 一键部署：提交 server job，监控部署，启动 router
./run_experiment1.sh
```

脚本会自动：
1. 提交 `run_server.slurm` job（3 个节点，5 张 4090，排除 debug 节点）
2. 等待 workers 部署完成（检查 `logs/worker_urls.txt`）
3. 启动 router（后台运行）
4. 等待 router 就绪（检查 `logs/router_url.txt`）

部署完成后，**手动运行 LangChain client**：
```bash
python run_langchain_app.py --router-url-file logs/router_url.txt --num-traces 20
```

按 `Ctrl+C` 会自动清理 router 进程。

### 方式二：分步运行

#### 1. 部署 Server（Slurm）

```bash
cd benchmark/multi_agent/experiment1
mkdir -p logs results
sbatch run_server.slurm
```

Server 启动后会在 `logs/worker_urls.txt` 写入 worker 列表。

#### 2. 启动 Router（本地运行，无需 Slurm）

在任意一台能访问 workers 内网地址的机器上运行（可与 server 不同）：

```bash
cd benchmark/multi_agent/experiment1
export ROUTER_POLICY=round_robin
python deploy_router.py --worker-urls-file logs/worker_urls.txt --output-dir logs
```

启动后会写 `logs/router_url.txt`，client 将使用该地址。

#### 3. 运行 LangChain Client（本地运行，无需 Slurm）

```bash
cd benchmark/multi_agent/experiment1
python run_langchain_app.py --router-url-file logs/router_url.txt --num-traces 20 --max-iterations 1 --concurrency 10
```

输出默认写到 `results/*_langchain.jsonl`（或用 `--output` 指定）。

### 对比不同路由策略

使用一键脚本时，设置不同的 `ROUTER_POLICY` 后重新运行：

```bash
export ROUTER_POLICY=agent_sticky_hash
./run_experiment1.sh
```

或分步运行时，依次停止并重启 router（仅改 `ROUTER_POLICY`），然后重复运行 client。

## 常用环境变量

- `MODEL_PATH`: model path (default: Qwen/Qwen3-4B-Thinking-2507)
- `NUM_WORKERS`: number of workers (default: 5)
- `WORKER_BASE_PORT`: base port (default: 8000)
- `ROUTER_PORT`: router port (default: 30000)
- `ROUTER_POLICY`: routing policy (default: round_robin)
- `MARBLE_DIR`: MARBLE dataset root (default: ../MARBLE)
- `NUM_TRACES`: number of traces to load (default: 20)
- `MAX_ITERATIONS`: max iterations per trace (default: 1)
- `CONCURRENCY`: concurrent requests (default: 10)

## 输出文件

- `logs/worker_urls.txt`: Worker URLs（内网地址）
- `logs/server_node.txt`: Server 节点名
- `logs/router_url.txt`: Router URL（内网地址）
- `logs/router_node.txt`: Router 节点名
- `logs/router.pid`: Router 进程 ID（用于清理）
- `logs/router.log`: Router 运行日志
- `results/<timestamp>_<policy>_langchain.jsonl`: Client 运行结果

## 注意事项

1. Server 使用 **long 分区 + rtx4090**，不使用 debug 节点
2. 所有脚本使用**内网 IP/主机名**（通过 `hostname -s` 获取），不使用 127.0.0.1
3. Server 和 Router 可以部署在不同节点
4. 确保 MARBLE 数据集已下载到 `../MARBLE` 目录
5. 使用一键脚本时，按 `Ctrl+C` 会自动清理 router 进程

