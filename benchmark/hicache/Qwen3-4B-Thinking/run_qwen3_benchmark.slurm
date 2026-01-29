#!/bin/bash
#SBATCH --job-name=qwen3_benchmark
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --exclusive
# To specify a fixed node, uncomment and modify the line below:
#SBATCH --output=/home/comp/zhanzhao/sglang/benchmark/hicache/Qwen3-4B-Thinking/logs/qwen3_benchmark_%j.out
#SBATCH --error=/home/comp/zhanzhao/sglang/benchmark/hicache/Qwen3-4B-Thinking/logs/qwen3_benchmark_%j.err

set -euo pipefail

BASE_DIR="/home/comp/zhanzhao/sglang/benchmark/hicache/Qwen3-4B-Thinking"
LOG_DIR="${BASE_DIR}/logs"
RESULT_DIR="${BASE_DIR}/results"

# Create directories (should already exist, but ensure they do)
mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

# Change to BASE_DIR
cd "${BASE_DIR}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Thinking-2507}"
PORT="${PORT:-30000}"
REPO_ROOT="/home/comp/zhanzhao/sglang"
# Limit GPU memory usage (0.0-1.0, default 0.85 means 85% of GPU memory)
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Disable proxy for localhost connections to avoid warmup failures
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"

declare -A CONFIGS=(
    ["disable-radix-cache"]="--disable-radix-cache"
    ["long-prefix-match"]=""
    ["enable-hierarchical-cache"]="--enable-hierarchical-cache"
)

wait_for_server() {
    local max_attempts=120
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if (netstat -tuln 2>/dev/null | grep -q ":${PORT} ") || \
           (ss -tuln 2>/dev/null | grep -q ":${PORT} "); then
            if curl -s http://localhost:${PORT}/v1/models > /dev/null 2>&1; then
                echo "Server is ready"
                sleep 5
                return 0
            fi
        fi
        echo "Waiting for server... (attempt $((attempt+1))/$max_attempts)"
        sleep 5
        attempt=$((attempt+1))
    done
    
    echo "Server failed to start"
    return 1
}

run_benchmark() {
    local config_name=$1
    local server_args=$2
    
    echo "=========================================="
    echo "Running benchmark: ${config_name}"
    echo "=========================================="
    
    local result_file="${RESULT_DIR}/qwen3_${config_name}_$(date +%Y%m%d_%H%M%S).jsonl"
    
    echo "Starting server with args: ${server_args}"
    cd "${BASE_DIR}"
    
    local cmd=(
        python3 -m sglang.launch_server
        --model-path "${MODEL_PATH}"
        --port "${PORT}"
        --host 0.0.0.0
        --log-level info
        --mem-fraction-static "${MEM_FRACTION_STATIC:-0.85}"
    )
    
    if [ -n "${server_args}" ]; then
        read -a args_array <<< "${server_args}"
        cmd+=("${args_array[@]}")
    fi
    
    "${cmd[@]}" > "${LOG_DIR}/server_${config_name}.log" 2>&1 &
    local server_pid=$!
    echo "Server PID: ${server_pid}"
    
    if ! wait_for_server; then
        echo "Failed to start server, skipping..."
        kill $server_pid 2>/dev/null || true
        return 1
    fi
    
    echo "Running benchmark with bench_multiturn.py"
    python3 "${BASE_DIR}/../bench_multiturn.py" \
        --model-path "${MODEL_PATH}" \
        --port "${PORT}" \
        --host localhost \
        --log-file "${result_file}" \
        > "${LOG_DIR}/bench_${config_name}.log" 2>&1
    
    local bench_exit_code=$?
    
    echo "Stopping server..."
    kill $server_pid 2>/dev/null || true
    sleep 10
    pkill -9 -f "sglang.launch_server.*port.*${PORT}" 2>/dev/null || true
    sleep 5
    
    if [ $bench_exit_code -eq 0 ]; then
        echo "Benchmark completed successfully: ${config_name}"
    else
        echo "Benchmark failed: ${config_name}"
    fi
    
    return $bench_exit_code
}

echo "Starting Qwen3 benchmark suite"
echo "Model: ${MODEL_PATH}"
echo "Running on node: ${SLURM_NODELIST:-unknown}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"

for config_name in "${!CONFIGS[@]}"; do
    server_args="${CONFIGS[$config_name]}"
    run_benchmark "${config_name}" "${server_args}"
    sleep 10
done

echo "All benchmarks completed"
echo "Results: ${RESULT_DIR}"
echo "Logs: ${LOG_DIR}"
