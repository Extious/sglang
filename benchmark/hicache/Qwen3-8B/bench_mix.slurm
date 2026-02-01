#!/bin/bash
#SBATCH --job-name=qwen3_8b_bench_mix
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=/home/comp/zhanzhao/sglang/benchmark/hicache/Qwen3-8B/bench_mix_%j.out
#SBATCH --error=/home/comp/zhanzhao/sglang/benchmark/hicache/Qwen3-8B/bench_mix_%j.err

set -euo pipefail

BASE_DIR="/home/comp/zhanzhao/sglang/benchmark/hicache/Qwen3-8B"
LOG_DIR="${BASE_DIR}/logs"
RESULT_DIR="${BASE_DIR}/results"
REPO_ROOT="/home/comp/zhanzhao/sglang"

cd "${BASE_DIR}"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

echo "SLURM stdout/stderr: ${BASE_DIR}/bench_mix_${SLURM_JOB_ID:-$$}.out / .err"
echo "App logs: ${LOG_DIR}/server_bench_mix.log, ${LOG_DIR}/bench_mix_result.log"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
DATASET_PATH="${DATASET_PATH:-/home/comp/zhanzhao/sglang/benchmark/hicache/ShareGPT_V3_unfiltered_cleaned_split.json}"
PORT="${PORT:-30000}"
MEM_FRACTION="${MEM_FRACTION:-0.7}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Disable proxy for localhost connections
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"

echo "Starting Qwen3-8B bench_mix benchmark"
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Running on node: ${SLURM_NODELIST:-unknown}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "=== GPU status before starting (check if other processes use memory) ==="
nvidia-smi || true
echo "=== End nvidia-smi ==="

# Start server with hierarchical cache
echo "Starting server..."
python3 -m sglang.launch_server \
    --attention-backend triton \
    --model-path "${MODEL_PATH}" \
    --log-level info \
    --pp-size 1 \
    --mem-fraction-static "${MEM_FRACTION}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --enable-metrics \
    --enable-cache-report \
    --page-size 64 \
    --enable-hierarchical-cache \
    --hicache-ratio 2.5 \
    --hicache-size 0 \
    --hicache-io-backend kernel \
    --hicache-mem-layout layer_first \
    --hicache-write-policy write_through \
    > "${LOG_DIR}/server_bench_mix.log" 2>&1 &

SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

# Wait for server to be ready
wait_for_server() {
    local max_attempts=120
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:${PORT}/v1/models > /dev/null 2>&1; then
            echo "Server is ready"
            sleep 5
            return 0
        fi
        echo "Waiting for server... (attempt $((attempt+1))/$max_attempts)"
        sleep 5
        attempt=$((attempt+1))
    done

    echo "Server failed to start"
    return 1
}

if ! wait_for_server; then
    echo "Failed to start server"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Create benchmark config
CONFIG_PATH="/tmp/bench_mix_config_${SLURM_JOB_ID:-$$}.json"
cat > "${CONFIG_PATH}" << 'EOF'
{
  "num_rounds": 10,
  "num_clients": 60,
  "round_ratios": [50, 25, 15, 15, 10, 10, 9, 8, 7, 6],
  "mean_new_tokens_per_round": [1000, 400, 350, 300, 280, 260, 240, 220, 210, 200],
  "mean_return_tokens_per_round": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
  "mean_inter_round_interval": [30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
}
EOF
export CONFIG_PATH

echo "Running bench_mix.py..."
python3 "${REPO_ROOT}/benchmark/hicache/bench_mix.py" \
    --model-path "${MODEL_PATH}" \
    --dataset-path "${DATASET_PATH}" \
    --port "${PORT}" \
    --duration 600 \
    --result-file "${RESULT_DIR}/bench_mix_${SLURM_JOB_ID:-$$}.json" \
    > "${LOG_DIR}/bench_mix_result.log" 2>&1

BENCH_EXIT_CODE=$?

# Cleanup
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null || true
sleep 5
pkill -9 -f "sglang.launch_server.*port.*${PORT}" 2>/dev/null || true

# Cleanup config
rm -f "${CONFIG_PATH}"

if [ $BENCH_EXIT_CODE -eq 0 ]; then
    echo "Benchmark completed successfully"
else
    echo "Benchmark failed with exit code: ${BENCH_EXIT_CODE}"
fi

echo "Results: ${LOG_DIR}/bench_mix_result.log"
echo "Server log: ${LOG_DIR}/server_bench_mix.log"
